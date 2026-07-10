"""
Persistent memory for DeployMind.

Two jobs:

1. Recognize when the *same underlying failure* has been seen before (even
   across different deployment IDs / re-runs) by hashing a normalized version
   of the deployment logs — with volatile bits like timestamps, commit
   hashes, and deployment IDs stripped out first.

2. Checkpoint pipeline progress under that signature after every node, so if
   the process crashes (429s, network errors, etc.) or you re-run the exact
   same failing deployment, DeployMind can resume from the last completed
   node instead of starting the whole investigation over.

Storage is a single local JSON file. This is intentionally simple — fine for
a single-user CLI tool. If you later run this as a shared service, swap
_load_all/_save_all for a real DB (sqlite/Redis) without touching callers.
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

MEMORY_FILE = Path(__file__).parent / "deploymind_memory.json"


def _load_all() -> Dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {}
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: Dict[str, Any]) -> None:
    MEMORY_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def normalize_logs(logs: str) -> str:
    """Strip tokens that differ between deployments of the *same* underlying
    error (timestamps, commit/build hashes, deployment IDs, durations,
    arbitrary numbers) so the normalized text hashes identically across runs.
    """
    text = logs
    text = re.sub(r"\bdpl_[A-Za-z0-9]+\b", "<deployment_id>", text)
    text = re.sub(r"\b[0-9a-f]{7,40}\b", "<hash>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?Z?\b", "<timestamp>", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(ms|s|sec|seconds|m|min|minutes)\b", "<duration>", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "<num>", text)
    return text.strip()


def compute_signature(
    owner: str, repo: str, logs: str, commit_sha: Optional[str] = None
) -> str:
    """Signature = normalized log text + the commit it happened on.

    Anchoring to commit_sha matters because repo_analysis_agent's
    `latest_deployment` field (and therefore log_analysis_agent's diagnosis,
    which consumes it) reasons over the *actual contents* of the latest
    commit. Without the commit in the signature, two different code states
    that happen to produce similar-looking log text would incorrectly share
    a cache entry — reusing a stale fix, or masking a genuinely new bug as
    "already seen". If commit_sha can't be determined (REST lookup failed),
    we degrade to log-text-only matching rather than erroring out.
    """
    normalized = normalize_logs(logs)
    basis = f"{normalized}|commit:{commit_sha or 'unknown'}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return f"{owner}/{repo}:{digest}"


def load_memory(signature: str) -> Optional[Dict[str, Any]]:
    return _load_all().get(signature)


def save_memory(signature: str, checkpoint: Dict[str, Any]) -> None:
    data = _load_all()
    existing = data.get(signature, {})
    existing.update({k: v for k, v in checkpoint.items() if v is not None})
    data[signature] = existing
    _save_all(data)


def clear_memory(signature: str) -> None:
    data = _load_all()
    if signature in data:
        del data[signature]
        _save_all(data)