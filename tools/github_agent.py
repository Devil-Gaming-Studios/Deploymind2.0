import os

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from pydantic import Field,BaseModel
from typing import List,Optional,Dict
from config import FAST_MODEL,SMART_MODEL

class RepoInput(BaseModel):
    github_owner : str = Field(description = "GutHub username or organization")
    repo_name : str = Field(description = "Repository name")

class RepoOutput(BaseModel):
    summary : str = Field(description = "Repository summary in full detail")
    structure : str = Field(description = "Repository structure in proper tree formating and new line characters")
    languages : List[str] = Field(description = "Most common languages used in the project")
    framework : Optional[str] = Field(
        default = None,
        description = "Detected Framework"
    )
    deployment_files: List[str] = Field(
        description="Files affecting deployment and CI/CD"
    )
    latest_deployment : Dict[str,str] = Field(
        description = "The list of the files which were changed in the latest commit (file name -> file content)"
    )
    suspected_issue : Optional[str] = Field(
        default=None,
        description="description of any suspected issue with the file name.")
    important_config_files: List[str] = Field(
        description="Configuration files critical for builds and deployment"
    )

async def create_github_analysis_agent(github_token:str):
    toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=r"C:\tools\github-mcp-server\github-mcp-server.exe",
                    args=["stdio"],
                    env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
                ),
            ),
        )

    repo_agent = LlmAgent(
        model = FAST_MODEL,
        name = 'repo_analysis_agent',
        instruction="""
            You are a Repository Analysis Agent.
            Use GitHub MCP tools to:

                1. Retrieve repository structure.
                2. Read important configuration files.
                3. Inspect deployment-related files.
                4. Gather information needed for repository analysis.

                Allowed operations:
                - Read repository structure
                - Read file contents
                - Search repository code
                - Read most recent deployment

                Forbidden operations:
                - Create pull requests
                - Create branches
                - Create commits
                - Create issues
                - Modify files
                - Delete files
                        
            Only analyze the repository and return RepoAnalysisOutput.
            Your responsibility is to understand the repository before
            other debugging agents begin investigation.

            Tasks:

            1. Inspect repository structure.
            2. Identify framework(s) used.
            3. Identify programming language(s).
            4. Detect deployment platform configuration.
            5. Detect CI/CD configuration.
            6. Detect environment variable usage.
            7. Identify important configuration files.
            8. Highlight files commonly involved in deployment failures.
            9. If no issue is suspected, set suspected_issue to null
            
            Focus especially on:

            - package.json
            - pnpm-lock.yaml
            - yarn.lock
            - package-lock.json
            - next.config.*
            - vite.config.*
            - tsconfig.json
            - vercel.json
            - Dockerfile
            - docker-compose.yml
            - .github/workflows/*
            - prisma/schema.prisma
            - requirements.txt
            - pyproject.toml
            - Cargo.toml
            - go.mod
            - server.js
            
            

            Do NOT generate fixes.

            Return only information matching RepoOutput.
    """,
        input_schema = RepoInput,
        output_schema = RepoOutput,
        tools = [toolset],
    )
    return toolset,repo_agent

class file_retrival_input(BaseModel):
    github_owner : str = Field(description = "github owner name")
    repo_name : str = Field(description = "Repository name")
    files_required : List[str] = Field("paths of the files that are required")

class file_retrival_output(BaseModel):
    files : Dict[str,str] = Field(description = "dictonary of Files : {filename,content}")

async def create_retrive_file_agent(github_token:str):
    toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=r"C:\tools\github-mcp-server\github-mcp-server.exe",
                    args=["stdio"],
                    env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
                ),
            ),
        )

    file_retrival_agent = LlmAgent(
        model=FAST_MODEL,
        name="github_file_retrieval_agent",
        instruction="""
            You are a GitHub File Retrieval Agent.
            Your responsibility is to fetch source files from a repository.
            Input contains:
            - github_owner
            - repository_name
            - files
            Tasks:
            1. Locate each requested file.
            2. Retrieve the file contents.
            3. Return contents exactly as stored.
            Rules:
            - Do not analyze code.
            - Do not summarize code.
            - Do not diagnose issues.
            - Do not generate fixes.
            - Do not retrieve unrelated files.
            - If a file cannot be found, omit it from the response.

            Return only FileRetrievalOutput.
            """,
        input_schema=file_retrival_input,
        output_schema=file_retrival_output,
        tools=[toolset],
    )

    return toolset,file_retrival_agent

class FileChange(BaseModel):
    file_path: str = Field(description="Repository-relative path of the file to change")
    new_content: str = Field(
        description="Full new content of the file after the fix is applied"
    )
    change_summary: str = Field(
        description="One-line explanation of what changed in this file and why"
    )

class FixApplyInput(BaseModel):
    github_owner: str = Field(description="GitHub username or organization")
    repository_name: str = Field(description="Repository name")
    base_branch: str = Field(default="main", description="Branch to branch off from")
    new_branch: str = Field(description="Name of the new branch to create for the fix")
    commit_message: str = Field(description="Commit message for the fix")
    pr_title: str = Field(description="Title for the pull request")
    pr_body: str = Field(description="Body/description for the pull request")
    file_changes: List[FileChange] = Field(
        description="Files to create or update with their full new contents"
    )


class FixApplyOutput(BaseModel):
    success: bool = Field(description="Whether the fix was pushed successfully")
    branch: Optional[str] = Field(default=None, description="Branch the fix was pushed to")
    pull_request_url: Optional[str] = Field(
        default=None, description="URL of the opened pull request, if any"
    )
    applied_files: List[str] = Field(
        default_factory=list, description="File paths that were created or updated"
    )
    message: str = Field(description="Human-readable summary of the result")


async def create_github_fix_agent(github_token: str):
    """Creates a WRITE-ENABLED GitHub agent that pushes an approved fix.

    Unlike the read-only repo/file agents, this one is permitted to create a
    branch, commit the changed files, and open a pull request. It must NEVER
    push directly to the base branch and must NEVER delete files.
    """
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=r"C:\tools\github-mcp-server\github-mcp-server.exe",
                args=["stdio"],
                env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
            ),
        ),
    )
    fix_apply_agent = LlmAgent(
        model=SMART_MODEL,
        name="github_fix_apply_agent",
        instruction="""
            You are a GitHub Fix Apply Agent.
            Your responsibility is to safely push an APPROVED fix to a repository.

            Input contains:
            - github_owner, repository_name
            - base_branch (the branch to start from)
            - new_branch (the branch you must create for the fix)
            - commit_message, pr_title, pr_body
            - file_changes: a list of files with their FULL new contents

            Tasks (in this exact order):
            1. Create the new branch `new_branch` from `base_branch`.
            2. For each entry in file_changes, create or update that file on the
               new branch with the EXACT contents provided in new_content.
            3. Open a pull request from `new_branch` into `base_branch` using
               pr_title and pr_body.

            Hard safety rules:
            - NEVER commit directly to the base branch.
            - NEVER delete files.
            - NEVER modify files that are not listed in file_changes.
            - Write file contents EXACTLY as provided. Do not reformat or "improve".
            - If any step fails, set success=false and explain in `message`.

            Return only data matching FixApplyOutput, including the pull request
            URL when one is created.
            """,
        input_schema=FixApplyInput,
        output_schema=FixApplyOutput,
        tools=[toolset],
    )
    return toolset, fix_apply_agent