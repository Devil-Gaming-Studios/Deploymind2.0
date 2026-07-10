from pydantic import BaseModel, Field
from typing import List, Literal

from tools.github_agent import RepoOutput
from log_analysis_agent import LogOutput
from google.adk.agents.llm_agent import LlmAgent

from config import FAST_MODEL


class ClassificationInput(BaseModel):
    repo_output: RepoOutput
    log_output: LogOutput


class ClassificationOutput(BaseModel):
    category: Literal[
        "dependency_issue",
        "configuration_issue",
        "environment_variable_issue",
        "build_tool_issue",
        "framework_version_mismatch",
        "code_bug",
        "infrastructure_platform_issue",
        "unknown",
    ] = Field(description="Primary category best describing the deployment failure")

    severity: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="How severe the failure is for the deployment's ability to run at all"
    )



issue_classification_agent = LlmAgent(
    model=FAST_MODEL,
    name="issue_classification_agent",
    instruction="""
        You are a Deployment Issue Classification Agent.

        Your responsibility is to take the repository analysis and the log
        analysis root-cause diagnosis and classify the deployment failure into
        a single, well-defined category, so downstream agents can decide
        whether an automated fix is appropriate or whether a human should be
        looped in immediately.

        Inputs available:

        1. Repository analysis output (summary, structure, languages,
           framework, deployment files, important config files).
        2. Log analysis output (analysis, root cause, confidence,
           useful log snippet, additional information).

        Tasks:

        1. Read the root cause and analysis carefully.
        2. Assign exactly one category from the allowed set:
           - dependency_issue: missing/broken/incompatible packages,
             empty or malformed package.json, lockfile mismatches.
           - configuration_issue: wrong or missing config file values
             (vercel.json, next.config.*, tsconfig.json, etc.).
           - environment_variable_issue: missing or incorrect env vars/secrets
             required at build or runtime.
           - build_tool_issue: failures in the build step itself (bundler,
             compiler, transpiler errors) not caused by dependencies or config.
           - framework_version_mismatch: incompatible versions between
             the framework and Node/runtime/other tooling.
           - code_bug: a genuine logic or syntax error in application source
             code that would fail regardless of environment.
           - infrastructure_platform_issue: platform-level problems (e.g.
             the hosting provider's build environment, quota, networking)
             not resolvable by changing files in the repository.
           - unknown: root cause is too ambiguous or logs are insufficient
             to confidently pick another category.

        3. Assign severity:
           - Critical: deployment cannot succeed at all until resolved.
           - High: deployment likely fails or is unstable.
           - Medium: deployment may partially work but with real risk.
           - Low: cosmetic or non-blocking issue.


        Rules:

        - Do NOT generate code fixes.
        - Do NOT propose file changes.
        - Do NOT invent details not supported by the inputs.
        - If evidence conflicts or is thin, prefer lower confidence and the
          "unknown" category over a confident but unsupported guess.

        Return only data matching the ClassificationOutput schema.
        """,
    input_schema=ClassificationInput,
    output_schema=ClassificationOutput,
)