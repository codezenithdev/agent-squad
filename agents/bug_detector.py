"""The bug detector — runs real static analysis on the generated code (v2.1).

Modes:
  * **mock**: scripted BUGS_FOUND-then-CLEAN (free, deterministic).
  * **real + Docker**: run bandit (security) + ruff (lint) against the backend
    inside a container; their exit codes decide BUGS_FOUND vs CLEAN, and the
    real findings are embedded.
  * **real, no Docker**: fall back to an LLM audit of the file digest.

Output contract unchanged: starts with 'BUGS_FOUND:' or 'CLEAN:'.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from langchain_core.messages import AIMessage

from config import get_settings
from core.file_tools import read_workspace_digest
from core.llm import complete
from core.sandbox import DockerSandbox, docker_available
from core.state import AgentState
from core.tools import non_empty

BUG_SYSTEM = (
    "You are a security and reliability auditor. Scan the provided code for "
    "security, reliability, performance, and framework-specific anti-patterns. "
    "If you find issues, respond starting with 'BUGS_FOUND:' followed by a "
    "numbered list, each item tagged [HIGH], [MED], or [LOW] with a concrete "
    "fix. If the code is clean, respond with exactly 'CLEAN: no issues found'."
)


def _run_static_analysis(settings, workspace: str) -> str:
    """Run bandit + ruff in Docker; return a BUGS_FOUND:/CLEAN: string."""
    sub = "backend" if (Path(workspace) / "backend").is_dir() else "."
    command = (
        f"cd {sub} && pip install -q bandit ruff 2>&1 ; "
        "echo '=== bandit ===' ; bandit -r . -ll -q ; B=$? ; "
        "echo '=== ruff ===' ; ruff check . ; R=$? ; "
        'echo "EXITCODES bandit=$B ruff=$R"'
    )
    sandbox = DockerSandbox(
        settings.sandbox_python_image,
        memory=settings.sandbox_memory,
        cpus=settings.sandbox_cpus,
        timeout=settings.sandbox_timeout,
    )
    res = sandbox.run(workspace, command)
    output = res.tail(3500)
    match = re.search(r"EXITCODES bandit=(\d+) ruff=(\d+)", res.stdout + res.stderr)

    if match:
        issues = match.group(1) != "0" or match.group(2) != "0"
    else:
        # Analysis didn't complete cleanly -> surface it as something to fix.
        issues = True

    if issues:
        return f"BUGS_FOUND:\n[static analysis in Docker]\n{output}"
    return "CLEAN: no issues found by static analysis"


async def bug_detector(state: AgentState, config=None) -> dict:
    settings = get_settings()
    workspace = state.get("workspace_dir", "")

    if settings.use_mock_llm:
        report = await complete(
            "bug_detector",
            BUG_SYSTEM,
            f"Code to audit (files):\n{read_workspace_digest(workspace)}",
        )
    elif workspace and docker_available():
        report = await asyncio.to_thread(_run_static_analysis, settings, workspace)
    else:
        raw = await complete(
            "bug_detector",
            BUG_SYSTEM,
            f"Code to audit (files):\n{read_workspace_digest(workspace)}",
        )
        report = f"{raw}\n\n[note: Docker sandbox unavailable; audited by LLM]"

    non_empty(report, "bug_report")
    return {
        "bug_report": report,
        "messages": [
            AIMessage(content=f"[bug_detector] {report.splitlines()[0]}",
                      name="bug_detector")
        ],
    }
