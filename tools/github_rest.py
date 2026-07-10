"""
Lightweight, non-agentic GitHub REST helpers.

These bypass the GitHub MCP server / LLM entirely. They exist for cheap
metadata lookups — like "what's the latest commit SHA on this branch?" —
that don't need an LLM to interpret and shouldn't cost an agent call just to
answer. Used to anchor DeployMind's memory signature to the actual commit
being deployed, not just the deployment logs' text.
"""

import httpx
from typing import Optional


async def get_latest_commit_sha(
    owner: str, repo: str, branch: str, github_token: str
) -> Optional[str]:
    """Return the latest commit SHA on `branch`, or None if it can't be
    determined (bad token, rate limited, branch not found, etc.). Callers
    should treat None as "unknown" and degrade gracefully rather than fail.
    """
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}",
                headers=headers,
            )
            if response.status_code != 200:
                return None
            data = response.json()
            return data.get("sha")
    except httpx.HTTPError:
        return None