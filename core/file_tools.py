"""Workspace-scoped file tools for the code-writing agent (v2.0).

The coder is given LangChain ``@tool``s to write/read/list files. Every path is
resolved *inside* a single run's workspace directory and checked so the agent
can never escape it (no ``..`` traversal, no absolute paths). All generated code
lives under ``workspaces/{thread_id}/`` (git-ignored) and is only ever executed
inside a sandbox later (v2.1) — never on the host.

``make_file_tools(workspace_dir)`` returns tools bound to one workspace via a
closure, so the model never sees or controls the root path.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

# Repo-root/workspaces — one subdir per run.
WORKSPACES_ROOT = Path(__file__).resolve().parent.parent / "workspaces"

# Cap a single write so a runaway model can't fill the disk.
_MAX_FILE_BYTES = 512_000


def workspace_path(thread_id: str) -> Path:
    """Absolute path to a run's workspace (created if missing)."""
    safe = "".join(c for c in thread_id if c.isalnum() or c in ("-", "_")) or "run"
    path = WORKSPACES_ROOT / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_inside(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``, refusing anything that escapes it."""
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path '{rel}' escapes the workspace")
    return target


def list_workspace_files(workspace_dir: str) -> list[str]:
    """Sorted relative paths of all files in a workspace (manifest builder)."""
    root = Path(workspace_dir).resolve()
    if not root.exists():
        return []
    return sorted(
        str(p.relative_to(root)).replace("\\", "/")
        for p in root.rglob("*")
        if p.is_file()
    )


def read_workspace_digest(workspace_dir: str, max_bytes: int = 60_000) -> str:
    """Concatenate all workspace files (with path headers) into one text blob for
    the LLM agents that review code. Truncated at ``max_bytes`` to bound tokens."""
    if not workspace_dir:
        return ""
    root = Path(workspace_dir).resolve()
    if not root.exists():
        return ""
    parts: list[str] = []
    total = 0
    for rel in list_workspace_files(workspace_dir):
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        block = f"\n----- {rel} -----\n{text}\n"
        if total + len(block) > max_bytes:
            parts.append(f"\n----- {rel} -----\n[truncated: digest size limit reached]\n")
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def make_file_tools(workspace_dir: str) -> list:
    """Return [write_file, read_file, list_files] bound to one workspace."""
    root = Path(workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a text file at the given workspace-relative path (creating
        parent directories). Overwrites if it exists. Returns a confirmation."""
        if len(content.encode("utf-8")) > _MAX_FILE_BYTES:
            return f"ERROR: refused — '{path}' exceeds {_MAX_FILE_BYTES} bytes."
        try:
            target = _resolve_inside(root, path)
        except ValueError as e:
            return f"ERROR: {e}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {path} ({len(content)} chars)"

    @tool
    def read_file(path: str) -> str:
        """Read and return the contents of a workspace-relative file."""
        try:
            target = _resolve_inside(root, path)
        except ValueError as e:
            return f"ERROR: {e}"
        if not target.is_file():
            return f"ERROR: '{path}' does not exist"
        return target.read_text(encoding="utf-8", errors="replace")

    @tool
    def list_files() -> str:
        """List every file written so far (workspace-relative paths)."""
        files = list_workspace_files(str(root))
        return "\n".join(files) if files else "(empty)"

    return [write_file, read_file, list_files]
