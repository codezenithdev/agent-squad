"""Smoke test — runs the full pipeline end to end and writes output.md.

This is the spec's official example. It:
  1. runs the whole graph on the job-board requirement,
  2. prints each agent step as it completes (live trace),
  3. writes the final document to ``output.md``.

Works in MOCK mode out of the box (free, no API key). To run it against real
GPT-4o / gpt-4o-mini, see the "GOING REAL" notes at the bottom of this file.

    python test_run.py
"""
from __future__ import annotations

import asyncio

from config import get_settings
from core.graph import graph
from core.llm import reset_mock
from core.state import initial_state

REQUIREMENT = (
    "Design a scalable job board with Next.js frontend, FastAPI backend, "
    "PostgreSQL, user auth, job listings with search, and an employer dashboard"
)

OUTPUT_PATH = "output.md"


def _summary(node: str, update: dict) -> str:
    """One-line description of what a node just did, for the live trace."""
    if node == "supervisor":
        return f"supervisor routes to -> {update.get('next')}"
    if node == "bug_detector":
        return f"bug_detector: {update.get('bug_report', '').splitlines()[0]}"
    if node == "tester":
        return f"tester: {update.get('test_results', '').splitlines()[0]}"
    if node == "reviewer":
        return f"reviewer: {update.get('review_decision')}"
    if node == "coder":
        rerun = "review_decision" in update  # cleared keys appear only on re-run
        return "coder: " + ("revised code (re-verify)" if rerun else "initial code")
    if node == "aggregator":
        return "aggregator: compiled final document"
    return f"{node}: produced {[k for k in update if k != 'messages']}"


async def main() -> None:
    settings = get_settings()
    mode = "MOCK (free, offline)" if settings.use_mock_llm else "REAL OpenAI"
    print(f"Running multi-agent pipeline in {mode} mode.\n")
    print(f"Requirement:\n  {REQUIREMENT}\n")

    reset_mock()  # no-op in real mode
    config = {"configurable": {"thread_id": "smoke-test"}, "recursion_limit": 50}

    print("Agent steps (as they complete):")
    step = 0
    async for chunk in graph.astream(
        initial_state(REQUIREMENT), config, stream_mode="updates"
    ):
        for node, update in chunk.items():
            step += 1
            print(f"  {step:>2}. {_summary(node, update)}")

    snapshot = await graph.aget_state(config)
    state = snapshot.values
    document = state.get("final_document", "")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(document)

    print("\n--- Result ---")
    print(f"  frameworks detected : {state.get('detected_frontend_framework')} / "
          f"{state.get('detected_backend_framework')}")
    print(f"  bug-fix iterations  : {state.get('bug_iteration_count')}")
    print(f"  test-fix iterations : {state.get('iteration_count')}")
    print(f"  review decision     : {state.get('review_decision')}")
    print(f"  total steps         : {step}")
    print(f"  document written    : {OUTPUT_PATH} ({len(document)} chars)")


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# GOING REAL (switch from mock to real OpenAI)
# ---------------------------------------------------------------------------
# 1. Copy .env.example to .env and set:
#       USE_MOCK_LLM=false
#       OPENAI_API_KEY=sk-...your key...
# 2. (Optional) Enable LangSmith tracing to inspect every agent call:
#       LANGCHAIN_TRACING_V2=true
#       LANGCHAIN_API_KEY=ls-...your key...
#       LANGCHAIN_PROJECT=multi-agent-architect
# 3. Run:  python test_run.py
#
# COST AWARENESS: a full run is ~20-30 LLM calls. The supervisor and reviewer
# use GPT-4o; the other eight workers use gpt-4o-mini. The verify/fix loops can
# add a few more coder/bug_detector/tester calls. With real models the bug
# detector and tester won't follow the scripted "fail then pass" pattern, so the
# number of loop iterations will vary per run (bounded by max_iterations=3).
