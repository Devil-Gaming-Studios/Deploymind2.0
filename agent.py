from typing import Optional,Dict
import os
from dotenv import load_dotenv
import asyncio
import time

from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.context import Context
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk import Workflow
from google.adk.workflow import node
from pydantic import BaseModel, Field

from tools.github_agent import (
    create_github_analysis_agent,
    RepoInput,
    RepoOutput,
    create_retrive_file_agent,
    file_retrival_output,
    file_retrival_input,
    create_github_fix_agent,
    FixApplyInput,
    FixApplyOutput,
)
from log_analysis_agent import (
    create_log_analysis_agent,
    LogInput,
    LogOutput,
)
from IssueClassificationAgent import(
    issue_classification_agent,
    ClassificationInput,
    ClassificationOutput,
)
from tools.vercel_agent import (
    get_deployment_logs,
    get_latest_failed_deployment,
    trigger_redeploy,
)
from fix_generator_agent import (
    create_fix_generator_agent,
    fix_generator_input,
    fix_generator_output
)

from validation_agent import (
    FixValidationInput, 
    FixValidationOutput, 
    validation_agent)

from memory_store import compute_signature, load_memory, save_memory
from tools.github_rest import get_latest_commit_sha


MAX_LOOPING1 = 3
APP_NAME = "deployMind"
MAX_LOOPING0 = 3


load_dotenv(".env")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
VERCEL_TOKEN = os.getenv("VERCEL_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")
BASE_BRANCH = os.getenv("BASE_BRANCH", "main")
DEPLOYMENT_ID_FALLBACK = os.getenv("DEPLOYMENT_ID")  
REQUIRE_APPROVAL = os.getenv("REQUIRE_APPROVAL", "false").lower() in ("1", "true", "yes")
VERCEL_PROJECT = os.getenv("VERCEL_PROJECT") or GITHUB_REPO
TAVELY_TOKEN = os.getenv("TAVELY_TOKEN")

# State keys that get persisted to disk under the failure signature after
# every node, so a resumed run can skip straight past whatever already
# completed successfully.
CHECKPOINT_KEYS = [
    "deployment_id",
    "github_owner",
    "repository_name",
    "base_branch",
    "commit_sha",
    "repo_analysis_output",
    "log_analysis",
    "root_cause",
    "retrieved_files",
    "issue_classification",
    "proposed_fix",
    "validation",
    "apply_result",
]


def _checkpoint(ctx: Context) -> None:
    """Persist current progress to disk under this run's failure signature."""
    signature = ctx.state.get("memory_signature")
    if not signature:
        return
    snapshot = {k: ctx.state.get(k) for k in CHECKPOINT_KEYS}
    save_memory(signature, snapshot)


def as_model(value,model):
    if isinstance(value,model):
        return value
    return model.model_validate(value)

async def resolve_deployment_id() -> str:
    import sys

    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    if DEPLOYMENT_ID_FALLBACK:
        return DEPLOYMENT_ID_FALLBACK

    if VERCEL_PROJECT and VERCEL_TOKEN:
        print(f"Looking up the latest FAILED deployment for '{VERCEL_PROJECT}'...")
        found = await get_latest_failed_deployment(VERCEL_PROJECT, VERCEL_TOKEN)
        if found:
            print(f"Found failed deployment: {found}")
            return found
        print("No failed deployment found via the Vercel API.")

    return input("Enter the failed Vercel deployment ID: ").strip()

class DeploymentContext(BaseModel):
    deployment_id: str = Field(description="Vercel deployment ID to investigate")
    github_owner: str = Field(description="GitHub username or organisation")
    github_repo: str = Field(description="GitHub repository name")
    base_branch: str = Field(default="main", description="Production/base branch")
    framework: Optional[str] = Field(default=None, description="Primary language/framework")
    additional_context: Optional[str] = Field(
        default=None, description="Any project-specific instruction"
    )


@node(rerun_on_resume=True)
async def fetch_logs_and_check_memory_node(ctx: Context) -> None:
    """Always-first node: fetch deployment logs, compute this failure's
    signature, and if we've seen it before, hydrate ctx.state with whatever
    progress was already made so downstream nodes can skip redundant work.
    """
    deploy_context = DeploymentContext.model_validate(ctx.state["deployment_context"])

    logs = await get_deployment_logs(deploy_context.deployment_id, ctx.state["vercel_token"])
    ctx.state["deployment_logs"] = logs
    ctx.state["deployment_id"] = deploy_context.deployment_id
    ctx.state["github_owner"] = deploy_context.github_owner
    ctx.state["repository_name"] = deploy_context.github_repo
    ctx.state["base_branch"] = deploy_context.base_branch

    # Cheap, non-agentic lookup — anchors the signature to the actual commit
    # being deployed, since repo_analysis_agent's latest_deployment field
    # (and log_analysis_agent's diagnosis, which consumes it) both reason
    # over this commit's real file contents. If it fails, commit_sha is None
    # and compute_signature() degrades to log-text-only matching.
    commit_sha = await get_latest_commit_sha(
        deploy_context.github_owner,
        deploy_context.github_repo,
        deploy_context.base_branch,
        ctx.state["github_token"],
    )
    ctx.state["commit_sha"] = commit_sha

    signature = compute_signature(
        deploy_context.github_owner, deploy_context.github_repo, logs, commit_sha
    )
    ctx.state["memory_signature"] = signature

    cached = load_memory(signature)
    if cached:
        completed = [k for k in CHECKPOINT_KEYS if cached.get(k) is not None]
        print(
            f"[memory] Recognized this failure before (signature={signature}). "
            f"Resuming — already have: {', '.join(completed) or 'nothing yet'}."
        )
        for key, value in cached.items():
            if value is not None:
                ctx.state[key] = value
    else:
        print(f"[memory] New failure signature ({signature}) — starting fresh.")


@node(rerun_on_resume=True)
async def repo_analysis_node(ctx:Context) -> RepoOutput:
    if ctx.state.get("repo_analysis_output"):
        print("[memory] Reusing cached repo analysis — skipping GitHub MCP calls.")
        return RepoOutput.model_validate(ctx.state["repo_analysis_output"])

    toolset,github_analysis_model = await create_github_analysis_agent(ctx.state["github_token"])
    deploy_context = DeploymentContext.model_validate(ctx.state["deployment_context"])
    repo_input=RepoInput(
        github_owner = deploy_context.github_owner,
        repo_name = deploy_context.github_repo,
    )
    result = as_model(
        await ctx.run_node(
            github_analysis_model,node_input = repo_input),RepoOutput)

    ctx.state.update(
        {
            "deployment_id": deploy_context.deployment_id,
            "github_owner": deploy_context.github_owner,
            "repository_name": deploy_context.github_repo,
            "base_branch": deploy_context.base_branch,
            "repo_analysis_output": result.model_dump(),
        }
    )

    await toolset.close()
    print(result)
    _checkpoint(ctx)
    return result

#@node(rerun_on_resume=True)
async def log_analysis_node(ctx:Context)->LogOutput:
    if ctx.state.get("log_analysis") and ctx.state.get("iteration_no", 1) == 1:
        print("[memory] Reusing cached log analysis — skipping log-analysis LLM calls.")
        return LogOutput.model_validate(ctx.state["log_analysis"])

    tavely_toolset,log_analysis_agent = await create_log_analysis_agent(ctx.state["tavely_token"])
    repo_analysis_output = as_model(ctx.state["repo_analysis_output"],RepoOutput)

    retrived_files : Dict[str,str] = ctx.state.get("retrieved_files") or {}
    analysis_result: Optional[LogOutput] = None

    # Logs were already fetched by fetch_logs_and_check_memory_node.
    logs = ctx.state["deployment_logs"]

    for round_num in range(MAX_LOOPING1):
        log_input = LogInput(
            deployment_logs = logs,
            repo_output = repo_analysis_output,
            retrived_files = retrived_files
        )

        analysis_result = as_model(
            await ctx.run_node(
                log_analysis_agent, node_input = log_input), LogOutput
        )

        if not analysis_result.needs_extra_files or not analysis_result.required_files:
            break

        file_retriver_toolset, file_retrival_agent = await create_retrive_file_agent(ctx.state["github_token"])

        file_agent_input = file_retrival_input(
            github_owner = ctx.state["github_owner"],
            repo_name = ctx.state["repository_name"],
            files_required = analysis_result.required_files
        )

        file_retrived_raw = await ctx.run_node(file_retrival_agent, node_input=file_agent_input)
        await file_retriver_toolset.close()

        if file_retrived_raw is None:
            print(
                "File retrieval agent returned no structured result this round "
                "(likely a dropped/incomplete turn) — continuing without new files."
            )
        else:
            file_retrived = as_model(file_retrived_raw, file_retrival_output)
            retrived_files.update(file_retrived.files)
            ctx.state["retrieved_files"] = retrived_files
            print(f'iteration {round_num} :\nroot cause : {analysis_result.root_cause}\nfiles retrived : {retrived_files}\n')

        await file_retriver_toolset.close()

        retrived_files.update(file_retrived.files)

        
        print(f'iteration {round_num} :\nroot cause : {analysis_result.root_cause}\nfiles retrived : {retrived_files}\n')

    ctx.state["retrieved_files"] = retrived_files
    ctx.state["root_cause"] = analysis_result.root_cause if analysis_result else ""
    ctx.state["log_analysis"] = analysis_result.model_dump() if analysis_result else {}

            
    await tavely_toolset.close()

    print(analysis_result)
    _checkpoint(ctx)
    return analysis_result

#@node(rerun_on_resume=True) 
async def issue_classification_agent_node(ctx:Context) -> ClassificationOutput:
    if ctx.state.get("issue_classification") and ctx.state.get("iteration_no", 1) == 1:
        print("[memory] Reusing cached issue classification — skipping classification LLM call.")
        return ClassificationOutput.model_validate(ctx.state["issue_classification"])

    repo_analysis_output = as_model(ctx.state["repo_analysis_output"],RepoOutput)
    log_analysis_output = as_model(ctx.state["log_analysis"],LogOutput)

    issue_classification_input = ClassificationInput(
        repo_output = repo_analysis_output,
        log_output = log_analysis_output,
    )

    result = as_model(await ctx.run_node(issue_classification_agent,node_input = issue_classification_input),ClassificationOutput)

    print(result.category)

    ctx.state["issue_classification"] = result.model_dump()
    _checkpoint(ctx)

    return result

#@node(rerun_on_resume=True)
async def fix_node(ctx:Context) -> fix_generator_output:
    if ctx.state.get("proposed_fix") and ctx.state.get("iteration_no", 1) == 1:
        print("[memory] Reusing cached proposed fix — skipping fix-generation LLM call.")
        return fix_generator_output.model_validate(ctx.state["proposed_fix"])

    toolset,fix_generator_agent = await create_fix_generator_agent(ctx.state["tavely_token"])
    repo_analysis_output = as_model(ctx.state["repo_analysis_output"],RepoOutput)
    log_analysis_output = as_model(ctx.state["log_analysis"],LogOutput)
    issue_classification_output = as_model(ctx.state["issue_classification"],ClassificationOutput)

    fix_agent_input = fix_generator_input(
        deployment_logs = ctx.state["deployment_logs"],
        root_cause = log_analysis_output.root_cause,
        classification = issue_classification_output,
        log_snippet = log_analysis_output.useful_snippet,
        repo_analysis_output = repo_analysis_output,
        relavent_files = ctx.state["retrieved_files"],
    )

    result = as_model(
        await ctx.run_node(fix_generator_agent,node_input = fix_agent_input),fix_generator_output
    )
    ctx.state["proposed_fix"] = result.model_dump()

    await toolset.close()

    for change in result.file_changes:
            print(f"  - {change.file_path}: {change.change_summary}")

    _checkpoint(ctx)
    return result

#@node(rerun_on_resume=True)
async def validation_node(ctx: Context) -> FixValidationOutput:
    if ctx.state.get("validation") and ctx.state.get("iteration_no", 1) == 1:
        print("[memory] Reusing cached validation result — skipping validation LLM call.")
        return FixValidationOutput.model_validate(ctx.state["validation"])

    fix = as_model(ctx.state["proposed_fix"],fix_generator_output)
    repo_analysis = as_model(ctx.state["repo_analysis_output"],RepoOutput)
    validation = as_model(
        await ctx.run_node(
            validation_agent,
            node_input=FixValidationInput(
                root_cause=ctx.state.get("root_cause", ""),
                deployment_logs=ctx.state.get("deployment_logs", ""),
                proposed_fix=fix,
                repo_analysis_output=repo_analysis,
            ),
        ),
        FixValidationOutput,
    )
    ctx.state["validation"] = validation.model_dump()
    print(
        "Validation:",
        f"valid={validation.is_valid}",
        f"needs_human_approval={validation.needs_human_approval}",
    )
    _checkpoint(ctx)
    return validation

@node(rerun_on_resume = True)
async def log_analysis_to_validation_node(ctx:Context):
    for _ in range(MAX_LOOPING0):
        log_analysis_output = await log_analysis_node(ctx)
        issue_classification_output = await issue_classification_agent_node(ctx)
        fix_agent_output = await fix_node(ctx)
        validation_output = await validation_node(ctx)

        ctx.state["iteration_no"] = ctx.state.get("iteration_no",1) + 1
        if validation_output.is_valid or not validation_output.issues_found :
            break

    return validation_output

async def request_human_approval(
    fix: fix_generator_output, validation: FixValidationOutput
) -> bool:
    print("\n" + "=" * 70)
    print("HUMAN APPROVAL REQUIRED")
    print("=" * 70)
    print(f"Fix:        {fix.fix_summary}")
    print(f"Risk:       {fix.risk_level}")
    print(f"Reasoning:  {fix.reasoning}")
    if validation.issues_found:
        print("Issues flagged by validation agent:")
        for issue in validation.issues_found:
            print(f"  - {issue}")
    print("Files to change:")
    for change in fix.file_changes:
        print(f"  - {change.file_path}: {change.change_summary}")
    print("=" * 70)

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, input, "Approve and redeploy this fix? [y/N]: ")
    return answer.strip().lower() in ("y", "yes")

@node(rerun_on_resume=True)
async def apply_fix_node(ctx: Context):
    if ctx.state.get("apply_result"):
        print("[memory] This exact fix was already applied previously — skipping re-apply.")
        return ctx.state["apply_result"]

    validation = FixValidationOutput.model_validate(ctx.state["validation"])
    fix = fix_generator_output.model_validate(ctx.state["proposed_fix"])

    if not fix.can_fix or not fix.file_changes:
        print("No applicable fix was generated. Stopping without changes.")
        return {"status": "no_fix"}

    if not validation.is_valid:
        print("Validation rejected the fix. Stopping without changes.")
        return {"status": "invalid_fix", "issues": validation.issues_found}

    # Diagram branch: validation -> "if yes" -> human gate ; "if no" -> push directly.
    # REQUIRE_APPROVAL forces the gate on for every fix (demo-friendly override).
    approved = True
    if validation.needs_human_approval or REQUIRE_APPROVAL:
        approved = await request_human_approval(fix, validation)

    if not approved:
        print("Fix rejected by human reviewer. Aborting redeploy.")
        return {"status": "rejected_by_human"}

    # Push the fix via the write-enabled GitHub agent.
    # Timestamp suffix keeps the branch unique so demos can be re-run without
    # hitting "reference already exists".
    new_branch = f"deploymind/fix-{ctx.state['deployment_id'][:8]}-{int(time.time())}"
    toolset, github_fix_agent = await create_github_fix_agent(ctx.state["github_token"])
    raw_result = await ctx.run_node(
        github_fix_agent,
        node_input=FixApplyInput(
            github_owner=ctx.state["github_owner"],
            repository_name=ctx.state["repository_name"],
            base_branch=ctx.state.get("base_branch", "main"),
            new_branch=new_branch,
            commit_message=f"fix: {fix.fix_summary}",
            pr_title=f"[DeployMind] {fix.fix_summary}",
            pr_body=fix.reasoning,
            file_changes=fix.file_changes,
        ),
    )
    await toolset.close()

    # The write agent is non-deterministic: it can finish without returning a
    # structured result (e.g. a transient MCP error swallowed by graceful
    # handling). Degrade gracefully instead of crashing the whole pipeline.
    if raw_result is None:
        print(
            "GitHub fix agent returned no result. It may have partially applied "
            f"the fix on branch '{new_branch}'. Check the repo and re-run if needed."
        )
        return {"status": "apply_incomplete", "branch": new_branch}

    apply_result = as_model(raw_result, FixApplyOutput)
    print("Apply result:", apply_result)
    ctx.state["apply_result"] = apply_result.model_dump()
    _checkpoint(ctx)

    if not apply_result.success:
        print("GitHub fix agent reported failure; skipping redeploy.")
        return apply_result

    # Trigger redeploy (pushing a connected repo usually auto-deploys; this is the explicit path).
    try:
        redeploy = await trigger_redeploy(
            project_name=ctx.state["repository_name"],
            github_owner=ctx.state["github_owner"],
            github_repo=ctx.state["repository_name"],
            vercel_token=ctx.state["vercel_token"],
            git_ref=new_branch,
        )
        print("Redeploy triggered:", redeploy.get("url") or redeploy)
    except Exception as exc:  # noqa: BLE001 - non-fatal; auto-deploy may already cover it
        print(f"Redeploy note (auto-deploy may handle this): {exc}")

    return apply_result

root_agent = Workflow(
    name="root_agent",
    edges=[
        ("START", fetch_logs_and_check_memory_node),
        (fetch_logs_and_check_memory_node, repo_analysis_node),
        (repo_analysis_node,log_analysis_to_validation_node),
        #(repo_analysis_node, log_analysis_node),
        # (log_analysis_node, issue_classification_agent_node),
        # (issue_classification_agent_node, fix_node),
        # (fix_node, validation_node),
        #(validation_node, apply_fix_node)
        (log_analysis_to_validation_node,apply_fix_node)
    ],
)

async def async_main():
    iteration_no = 1
    session_service = InMemorySessionService()
    missing = [
        name
        for name, value in {
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required .env values: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
    deployment_id = await resolve_deployment_id()
    if not deployment_id:
        raise SystemExit("No deployment ID provided.")
    query = DeploymentContext(
        deployment_id=deployment_id,
        github_owner=GITHUB_OWNER,
        github_repo=GITHUB_REPO,
        base_branch=BASE_BRANCH,
    )
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id="user123",
        state={
            "github_token": GITHUB_TOKEN,
            "vercel_token": VERCEL_TOKEN,
            "tavely_token": TAVELY_TOKEN,
            "deployment_context": query.model_dump(),
        },
    )
    print(f"User Query: '{query}'")
    content = types.Content(
        role="user", parts=[types.Part(text="Start deployment investigation")]
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    print("Running agent...")
    events_async = runner.run_async(
        session_id=session.id, user_id=session.user_id, new_message=content
    )
    async for event in events_async:
        print(f"Event received: {event.content}")
        print(f"Event author: {event.author}")


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except Exception as e:
        print(f"An error occurred: {e}")