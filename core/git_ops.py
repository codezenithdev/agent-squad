"""Git integration (v2.5).

The aggregator turns each run's generated workspace into a real git branch: init
the repo (if needed), commit all files, and the 10-section document doubles as a
PR description. Local-only by default — no push (per the project's chosen scope).

The workspace lives under ``workspaces/`` which the *outer* repo git-ignores, so
this nested repo is independent and never tracked by the parent.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from config import get_settings


def git_available() -> bool:
    try:
        return (
            subprocess.run(
                ["git", "--version"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=60
    )


def commit_workspace(workspace_dir: str, branch_name: str, message: str) -> dict:
    """Init (if needed) a git repo in the workspace and commit all files to
    ``branch_name``. Returns a status dict; never raises."""
    result = {"committed": False, "branch": branch_name, "commit": "", "reason": ""}
    root = Path(workspace_dir)
    if not root.is_dir():
        result["reason"] = "no workspace"
        return result
    if not git_available():
        result["reason"] = "git unavailable"
        return result

    s = get_settings()
    identity = [
        "-c", f"user.name={s.git_author_name}",
        "-c", f"user.email={s.git_author_email}",
    ]
    try:
        if not (root / ".git").exists():
            _git(["init", "-q", "-b", branch_name], root)
        else:
            _git(["checkout", "-q", "-B", branch_name], root)
        _git(["add", "-A"], root)
        commit = _git(identity + ["commit", "-q", "-m", message], root)
        if commit.returncode != 0:
            result["reason"] = (
                commit.stderr or commit.stdout or "nothing to commit"
            ).strip()[:200]
            return result
        sha = _git(["rev-parse", "--short", "HEAD"], root)
        result["committed"] = True
        result["commit"] = sha.stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        result["reason"] = str(e)[:200]
    return result
