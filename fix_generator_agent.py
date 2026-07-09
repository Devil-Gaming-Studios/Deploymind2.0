from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from pydantic import Field,BaseModel
from typing import List,Optional,Dict
from config import FAST_MODEL,SMART_MODEL
from tools.github_agent import RepoOutput,FileChange

class fix_generator_input(BaseModel):
    deployment_logs : str = Field(description="deployment logs provided")
    root_cause : str = Field(description="root cause of the issue")
    log_snippet : str = Field(description="the snippet of the log that might be pointing to the issue")
    repo_analysis_output : RepoOutput = Field(description="the repository analysis output")
    relavent_files : Dict[str,str] = Field(
        default_factory = dict,
        description="Source files retrieved during investigation (path -> contents)",
    )

class fix_generator_output(BaseModel):
    can_fix: bool = Field(
        description="Whether a confident, concrete fix could be produced"
    )
    fix_summary: str = Field(description="Short one-line description of the fix")
    file_changes: List[FileChange] = Field(
        default_factory=list,
        description="Files to change, each with its FULL new contents",
    )
    risk_level: str = Field(description="Low, Medium, or High")
    requires_human_review: bool = Field(
        description="Whether a human should review before applying"
    )
    reasoning: str = Field(
        description="Why this fix addresses the root cause and what it touches"
    )

async def create_fix_generator_agent(tavely_key : str):

    toolset = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="npx",
                        args=["-y", "tavily-mcp"],
                        env={"TAVILY_API_KEY": tavely_key},
                    ),
                    timeout=60,  # increase from default if npx is slow to install/start
                )
            )

    fix_agent = LlmAgent(
    model=SMART_MODEL,
    name="fix_generation_agent",
    instruction="""
        You are a Deployment Fix Generation Agent.

        Your responsibility is to propose a SAFE, MINIMAL fix for a deployment
        failure, given a confirmed root cause and the relevant source files.

        Inputs available:
        1. root_cause      - the confirmed reason the deployment failed
        2. log_snippet        - log lines and repo findings supporting the diagnosis
        3. repo_analysis_output   - framework, language, package manager, structure
        4. relevant_files  - the actual contents of the files involved
        5. deployment_logs - the raw build/runtime logs

        Tools available:
        1. toolset - tavely toolset for google searching if required

        Tasks:
        1. Decide whether you can produce a confident, concrete fix.
        2. If yes, produce the MINIMAL set of file changes that fixes the root
           cause and nothing else.
        3. For every changed file, return its FULL new contents in new_content
           (not a diff, not a snippet) so it can be committed as-is.
        4. Assess the risk level and whether a human must review the change.
        5. Use the tavely toolset in hand when require for generating fixes.

        TypeScript guidance (important):
        - If the project language is TypeScript (check repo_analysis) and you add
          a JavaScript library that does NOT ship its own type declarations
          (e.g. lodash, express), you MUST also add the matching @types/*
          package to devDependencies (e.g. add both "lodash" to dependencies
          AND "@types/lodash" to devDependencies). A strict TypeScript build
          ("strict": true) fails with "Could not find a declaration file for
          module 'X'" otherwise. Libraries that bundle their own types (e.g.
          axios, react) do not need an @types/* package.

        Hard safety rules:
        - Only change files when you have their current contents in relevant_files.
          If you need a file you do not have, set can_fix=false and explain.
        - Make the smallest change that fixes the problem. Do NOT refactor,
          rename, reformat, or "improve" unrelated code.
        - NEVER touch secrets, credentials, lockfile hashes you cannot compute,
          or delete files.
        - Do NOT invent file paths. Use paths confirmed by the repo analysis or
          the retrieved files.
        - If the fix is risky, ambiguous, or you are not confident, set
          requires_human_review=true (and prefer risk_level Medium or High).

        Risk guidance:
        - Low:    isolated config/typo/dependency-version change, well-understood.
        - Medium: source code logic change, or touches build configuration.
        - High:   touches multiple files, security-sensitive, or uncertain.

        Output requirements:
        - can_fix: true only when file_changes fully address the root cause.
        - fix_summary: concise, e.g. "Add missing 'sharp' dependency".
        - file_changes: full new contents for each file.
        - risk_level: Low | Medium | High.
        - requires_human_review: true unless the fix is clearly Low risk.
        - reasoning: explain why this fixes the root cause and what it changes.

        Return only data matching the FixGenerationOutput schema.
        """,
    input_schema=fix_generator_input,
    output_schema=fix_generator_output,
    tools = [toolset]
    )

    return toolset,fix_agent