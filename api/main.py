"""FastAPI wrapper around the compiled graph.

Two endpoints:
  * POST /run            -> run the full pipeline for a thread_id and return the
                            document, how many steps it took, and the detected
                            stack.
  * GET  /status/{id}    -> read the persisted state for a thread_id (this is
                            why we compiled with a checkpointer in Phase 6).

The graph is imported once at module load (``from core.graph import graph``) so
its in-memory checkpointer is shared across requests — that's what lets /status
see what /run produced.

Run it with:   uvicorn api.main:app --reload --port 8000
Then open the interactive docs at http://localhost:8000/docs
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import get_settings
from core.graph import graph
from core.llm import reset_mock
from core.state import initial_state

app = FastAPI(title="Multi-Agent Architect", version="1.0.0")

# CORS for local frontends (any localhost / 127.0.0.1 port).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    requirement: str = Field(..., description="The software requirement to design.")
    thread_id: str = Field(..., description="Unique id for this run (for /status).")


class RunResponse(BaseModel):
    final_document: str
    steps_taken: int
    frameworks_detected: dict


def _frameworks(state: dict) -> dict:
    return {
        "frontend": state.get("detected_frontend_framework", ""),
        "backend": state.get("detected_backend_framework", ""),
    }


def _describe(node: str, update: dict) -> str:
    """One-line summary of what a node just did (for the live stream)."""
    if node == "supervisor":
        return f"routing to {update.get('next')}"
    if node == "bug_detector":
        line = (update.get("bug_report") or "").splitlines()
        return line[0] if line else "scanned"
    if node == "tester":
        line = (update.get("test_results") or "").splitlines()
        return line[0] if line else "tested"
    if node == "reviewer":
        return f"verdict: {update.get('review_decision')}"
    if node == "coder":
        return (
            f"wrote {len(update.get('files', []))} files"
            if update.get("files")
            else "updated code"
        )
    if node == "aggregator":
        return "compiled final document"
    return "produced " + ", ".join(k for k in update if k != "messages")


async def _sse_stream(requirement: str, thread_id: str):
    """Yield Server-Sent Events: one per agent step, then a final 'done' event."""
    reset_mock()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    step = 0
    async for chunk in graph.astream(
        initial_state(requirement), config, stream_mode="updates"
    ):
        for node, update in chunk.items():
            step += 1
            payload = {
                "type": "step",
                "step": step,
                "node": node,
                "summary": _describe(node, update),
            }
            yield f"data: {json.dumps(payload)}\n\n"

    state = (await graph.aget_state(config)).values
    done = {
        "type": "done",
        "steps": step,
        "frameworks_detected": _frameworks(state),
        "review_decision": state.get("review_decision"),
        "final_document": state.get("final_document", ""),
    }
    yield f"data: {json.dumps(done)}\n\n"


@app.post("/run/stream")
async def run_stream(req: RunRequest):
    """Stream each agent step live as Server-Sent Events (text/event-stream)."""
    return StreamingResponse(
        _sse_stream(req.requirement, req.thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/")
async def root():
    """Health check — also tells you whether you're in mock or real mode."""
    return {"status": "ok", "mock_mode": get_settings().use_mock_llm}


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    """Run the whole pipeline to completion and return the result.

    Note: this awaits the entire run, so for real (non-mock) models the request
    can take a while. A production variant would launch the run as a background
    task and let the client poll /status — which already works, since the graph
    persists state per thread_id.
    """
    reset_mock()  # fresh scripted mock sequence per run (no-op in real mode)
    config = {
        "configurable": {"thread_id": req.thread_id},
        "recursion_limit": 50,
    }

    steps = 0
    async for chunk in graph.astream(
        initial_state(req.requirement), config, stream_mode="updates"
    ):
        steps += len(chunk)  # one node execution per key in the chunk

    snapshot = await graph.aget_state(config)
    state = snapshot.values
    return RunResponse(
        final_document=state.get("final_document", ""),
        steps_taken=steps,
        frameworks_detected=_frameworks(state),
    )


@app.get("/status/{thread_id}")
async def status(thread_id: str) -> dict:
    """Return the persisted state snapshot for a thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)

    if not snapshot.values:
        raise HTTPException(
            status_code=404, detail=f"No run found for thread_id '{thread_id}'"
        )

    state = snapshot.values
    return {
        "thread_id": thread_id,
        "finished": snapshot.next == (),       # empty == run complete
        "pending_next": list(snapshot.next),   # nodes still to run, if any
        "frameworks_detected": _frameworks(state),
        "review_decision": state.get("review_decision"),
        "iteration_count": state.get("iteration_count", 0),
        "bug_iteration_count": state.get("bug_iteration_count", 0),
        "has_final_document": bool(state.get("final_document")),
    }


if __name__ == "__main__":
    # In-process exercise of the API (no network/port needed) using httpx's
    # ASGI transport. Demonstrates the same calls a real client would make.
    import asyncio

    import httpx

    REQUIREMENT = (
        "Design a scalable job board with Next.js frontend, FastAPI backend, "
        "PostgreSQL, user auth, job listings with search, and an employer dashboard"
    )

    async def demo() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            print("GET / :", (await c.get("/")).json())

            print("\nGET /status/job-1 (before any run):")
            r = await c.get("/status/job-1")
            print(f"  {r.status_code} -> {r.json()}")

            print("\nPOST /run (thread_id='job-1') ...")
            r = await c.post("/run", json={"requirement": REQUIREMENT, "thread_id": "job-1"})
            data = r.json()
            print(f"  {r.status_code}")
            print(f"  steps_taken         : {data['steps_taken']}")
            print(f"  frameworks_detected : {data['frameworks_detected']}")
            print(f"  final_document      : {len(data['final_document'])} chars")

            print("\nGET /status/job-1 (after run):")
            print(f"  {(await c.get('/status/job-1')).json()}")

    asyncio.run(demo())
