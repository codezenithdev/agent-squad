"""The tester — runs the generated project's tests for real (v2.1).

Modes:
  * **mock**: keep the scripted FAIL-then-PASS behavior (free, deterministic,
    drives the suite + offline demos).
  * **real + Docker available**: actually run ``pytest`` against the generated
    backend inside a throwaway container and report the real result.
  * **real, no Docker**: fall back to an LLM assessment of the file digest (so
    the pipeline still works), clearly marked.

Output contract is unchanged: ``test_results`` starts with ``PASS:`` or ``FAIL:``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.messages import AIMessage

from config import get_settings
from core.file_tools import read_workspace_digest
from core.llm import complete
from core.sandbox import DockerSandbox, docker_available
from core.state import AgentState
from core.tools import non_empty

TESTER_SYSTEM = (
    "You are a QA engineer. Given the implementation, design unit tests, "
    "integration test stubs, and run conceptual static checks. If anything is "
    "wrong or untested, respond starting with 'FAIL:' and list the specific "
    "failures. If everything passes, respond starting with 'PASS:' and give a "
    "short summary."
)


def _run_pytest(settings, workspace: str) -> str:
    """Run the backend test suite inside Docker; return a PASS:/FAIL: string."""
    sub = "backend" if (Path(workspace) / "backend").is_dir() else "."
    command = (
        f"cd {sub} && pip install -q -r requirements.txt pytest 2>&1 "
        f"&& python -m pytest -q 2>&1"
    )
    sandbox = DockerSandbox(
        settings.sandbox_python_image,
        memory=settings.sandbox_memory,
        cpus=settings.sandbox_cpus,
        timeout=settings.sandbox_timeout,
    )
    res = sandbox.run(workspace, command)
    prefix = "PASS:" if res.ok else "FAIL:"
    return f"{prefix}\n[ran pytest in Docker, exit={res.exit_code}]\n{res.tail(3000)}"


async def tester(state: AgentState, config=None) -> dict:
    settings = get_settings()
    workspace = state.get("workspace_dir", "")

    if settings.use_mock_llm:
        results = await complete(
            "tester",
            TESTER_SYSTEM,
            f"Code under test (files):\n{read_workspace_digest(workspace)}",
        )
    elif workspace and docker_available():
        # Real execution off the event loop (subprocess blocks).
        results = await asyncio.to_thread(_run_pytest, settings, workspace)
    else:
        # No sandbox available -> LLM assessment, clearly marked.
        raw = await complete(
            "tester",
            TESTER_SYSTEM,
            f"Code under test (files):\n{read_workspace_digest(workspace)}",
        )
        results = f"{raw}\n\n[note: Docker sandbox unavailable; assessed by LLM]"

    non_empty(results, "test_results")
    return {
        "test_results": results,
        "messages": [
            AIMessage(content=f"[tester] {results.splitlines()[0]}", name="tester")
        ],
    }
