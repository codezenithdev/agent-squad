"""The SSE streaming endpoint emits live step events then a final document."""
import json

import httpx
import pytest

from api.main import app

REQUIREMENT = "Design a job board with a Next.js frontend and FastAPI backend"


@pytest.fixture(autouse=True)
def force_mock(monkeypatch):
    # get_settings() is cache-cleared per test by conftest; forcing mock here
    # keeps the streamed pipeline offline/free regardless of .env.
    monkeypatch.setenv("USE_MOCK_LLM", "true")


async def _collect_events(thread_id: str) -> list[dict]:
    transport = httpx.ASGITransport(app=app)
    events: list[dict] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        async with c.stream(
            "POST", "/run/stream",
            json={"requirement": REQUIREMENT, "thread_id": thread_id},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: "):]))
    return events


async def test_stream_emits_steps_then_done():
    events = await _collect_events("t-stream-1")

    steps = [e for e in events if e.get("type") == "step"]
    done = [e for e in events if e.get("type") == "done"]

    assert len(steps) > 5  # planner..aggregator + supervisor hops
    assert any(e["node"] == "coder" for e in steps)
    assert len(done) == 1

    final = done[0]
    assert final["final_document"].count("## ") >= 10  # the 10 sections
    assert final["frameworks_detected"] == {"frontend": "nextjs", "backend": "fastapi"}
