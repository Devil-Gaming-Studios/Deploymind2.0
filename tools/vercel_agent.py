import json
import httpx


async def get_deployment_logs(deployment_id: str, vercel_token: str) -> str:
    """Fetch build/runtime events for a Vercel deployment and return them as
    a single plain-text log string (which is what the Log Analysis Agent expects).

    The Vercel events endpoint returns a JSON array of event objects. Each event
    usually carries its line of text in either `text` or `payload.text`.
    """
    headers = {
        "Authorization": f"Bearer {vercel_token}",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"https://api.vercel.com/v3/deployments/{deployment_id}/events",
            headers=headers,
            params={
                "limit": -1,
                "builds": 1,
            },
        )
        response.raise_for_status()
        events = response.json()

    lines: list[str] = []
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            text = event.get("text")
            if not text:
                payload = event.get("payload") or {}
                if isinstance(payload, dict):
                    text = payload.get("text")
            if text:
                lines.append(str(text))

    if lines:
        return "\n".join(lines)

    # Fallback: return a truncated JSON dump so the agent still has something to read.
    return json.dumps(events)[:20000]


async def get_latest_failed_deployment(
    project_name: str, vercel_token: str, team_id: str | None = None
) -> str | None:
    """Return the deployment ID of the most recent FAILED deployment for a
    Vercel project, or None if there are no failed deployments.

    Lets DeployMind find what to debug on its own instead of a human pasting
    the deployment ID.
    """
    headers = {"Authorization": f"Bearer {vercel_token}"}
    params: dict = {"app": project_name, "limit": 20}
    if team_id:
        params["teamId"] = team_id

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            "https://api.vercel.com/v6/deployments",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    deployments = data.get("deployments", []) if isinstance(data, dict) else []
    failed = [
        d
        for d in deployments
        if (d.get("state") or d.get("readyState")) == "ERROR"
    ]
    if not failed:
        return None

    failed.sort(key=lambda d: d.get("created", 0), reverse=True)
    return failed[0].get("uid") or failed[0].get("id")


async def trigger_redeploy(
    project_name: str,
    github_owner: str,
    github_repo: str,
    vercel_token: str,
    git_ref: str = "main",
    team_id: str | None = None,
) -> dict:
    """Trigger a fresh Vercel deployment from a git ref.

    NOTE: If the Vercel project is connected to GitHub, simply pushing the fix
    branch (done by the GitHub fix agent) already triggers an automatic deploy.
    This helper is the explicit fallback. Some Vercel projects require
    `gitSource.repoId` instead of org/repo — adjust for your project if the API
    rejects this payload.
    """
    headers = {
        "Authorization": f"Bearer {vercel_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "name": project_name,
        "gitSource": {
            "type": "github",
            "org": github_owner,
            "repo": github_repo,
            "ref": git_ref,
        },
    }
    params = {"forceNew": "1"}
    if team_id:
        params["teamId"] = team_id

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.vercel.com/v13/deployments",
            headers=headers,
            params=params,
            json=payload,
        )
        response.raise_for_status()
        return response.json()
