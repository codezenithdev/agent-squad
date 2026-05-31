"""End-to-end integration test: the full graph through the supervisor loop.

Runs entirely in mock mode (free, deterministic). Proves the 11 agents wire
together, the fix-loops resolve, and the aggregator emits a complete document.
"""
import uuid

import pytest

import core.llm as llm_module
from config import Settings
from core.graph import graph
from core.llm import reset_mock
from core.state import initial_state


@pytest.fixture(autouse=True)
def force_mock(monkeypatch):
    fake = lambda: Settings(_env_file=None, use_mock_llm=True, max_iterations=3)
    monkeypatch.setattr(llm_module, "get_settings", fake)
    # The coder's file-agent loop resolves settings independently.
    import core.agent_loop as al

    monkeypatch.setattr(al, "get_settings", fake)


async def test_full_pipeline_mock():
    reset_mock()
    config = {
        "configurable": {"thread_id": f"test-{uuid.uuid4()}"},
        "recursion_limit": 50,
    }
    requirement = "Design a job board with a Next.js frontend and FastAPI backend"

    final = await graph.ainvoke(initial_state(requirement), config)

    # Framework detection flowed through.
    assert final["detected_frontend_framework"] == "nextjs"
    assert final["detected_backend_framework"] == "fastapi"

    # The scripted mock resolves exactly one bug-fix and one test-fix cycle.
    assert final["bug_iteration_count"] == 1
    assert final["iteration_count"] == 1
    assert final["review_decision"] == "APPROVE"

    # The aggregator produced all ten document sections.
    doc = final["final_document"]
    for n in range(1, 11):
        assert f"## {n}." in doc
