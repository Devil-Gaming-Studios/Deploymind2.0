from pydantic import BaseModel,Field
from typing import List,Dict

from tools.github_agent import RepoOutput
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from config import SMART_MODEL

class LogInput(BaseModel):
    deployment_logs : str= Field(description="Deployment logs got from the vercel")
    repo_output : RepoOutput
    retrived_files : Dict[str,str] = Field(
        default_factory=dict
    )

class LogOutput(BaseModel):
    analysis : str = Field(
        description="Anylisis of the deployment logs"
    )
    root_cause : str = Field(
        description="root cause to the deployment error by reasoning and search from the web"
    )
    required_files : List[str] = Field(description="list of files that are to be involved in the fixing process")
    needs_extra_files : bool = Field(description="Do the fix requires extra files?")
    confidence : str = Field(description="what is the confidence on the detected cause of the problem? [High, medium, low]")
    useful_snippet : str= Field(description="which part of the deployment logs actually corresponds to the actual issue")
    additional_information : str = Field(description="addtional information about the error logs")

async def create_log_analysis_agent(tavely_key:str):

    toolset = MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "tavily-mcp"],
                    env={"TAVILY_API_KEY": tavely_key},
                ),
                timeout=60,  # increase from default if npx is slow to install/start
            )
        )

    log_analysis_agent = LlmAgent(
        model = SMART_MODEL,
        name = 'log_analysis_agent',
        instruction = """
            You are a Deployment Log Analysis Agent.

            Your responsibility is to determine the root cause of deployment failures
            using deployment logs and repository analysis information.

            Inputs available:

            1. Deployment logs
            2. Repository summary
            3. Project structure
            4. Framework information
            5. Language information
            6. Architecture summary
            7. Important configuration files
            8. Deployment-related files
            9. Potential debugging targets
            10. Latest deployment files

            Tools available:
            1. Tavely search Tool

            Tasks:

            1. Analyze the deployment logs carefully.
            2. Identify the exact failure point.
            3. Correlate log failures with repository structure.
            4. Determine the most likely root cause.
            5. Collect evidence supporting the diagnosis.
            6. Determine whether additional source code inspection is required.
            7. Identify the minimum set of files required for further investigation.
            8. Assess confidence in the diagnosis.
            9. Read the latest commited file content and if you find anything that might be causing error.

            Rules:

            - Do NOT generate code fixes.
            - Do NOT propose implementation changes.
            - Do NOT suggest pull requests.
            - Do NOT rewrite source code.
            - Focus only on diagnosis and investigation.
            - Prefer precise root causes over generic explanations.
            - Use repository context when interpreting logs.
            - If confidence is low, request additional files rather than guessing.

            Guidance for required_files:

            Include only files that are necessary to validate or further investigate
            the root cause.

            Examples:

            Example 1:
            Log:
                Cannot resolve '@/lib/auth'

            Required files:
                - tsconfig.json
                - src/lib/auth.ts

            Example 2:
            Log:
                Prisma schema not found

            Required files:
                - prisma/schema.prisma
                - package.json

            Example 3:
            Log:
                Type error in src/app/page.tsx

            Required files:
                - src/app/page.tsx
                - tsconfig.json

            Output Requirements:

            analysis:
            - Detailed investigation summary.
            - Explain how the logs were interpreted.
            - Explain which repository information was used.
            - Explain why the identified root cause is likely.

            root_cause:
            - Single concise statement describing the failure.
            - Use the lastest commited file content to detect the root cause.
            - Use tavely search tool for google searches of the issue in hand if neccessary.


            confidence:
            - Must be one of:
            - High
            - Medium
            - Low

            useful_snippet:
            - Specific log entries and repository findings supporting the diagnosis.

            required_files:
            - Only files necessary for further investigation.

            need_extra_files:
            - true if source code inspection is required.
            - false if the root cause is already confirmed.

            additional_information:
            - Explain what information is missing and why.
            - Empty string if no additional information is needed.

            Return only data matching the LogAnalysisOutput schema.
            """,
        input_schema = LogInput,
        output_schema = LogOutput,
        tools = [toolset],
    )

    return toolset,log_analysis_agent


