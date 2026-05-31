"""The StateGraph — how nodes are wired together (now complete: all 11 nodes).

A LangGraph graph is: add nodes, set an entry point, add edges. The supervisor
pattern uses *conditional* edges out of the supervisor (the destination depends
on ``state['next']``) and plain edges from every worker back to the supervisor.

The wiring is data-driven: list a worker in ``BUILT_WORKERS`` and it becomes a
real node with an edge back to the supervisor. Any route in ``VALID_ROUTES``
that isn't built (only ``FINISH`` now) maps to END.

We compile with a ``MemorySaver`` checkpointer. That gives the graph *memory*:
each run is keyed by a ``thread_id``, its state is persisted step-by-step, and
you can read it back later with ``aget_state(config)`` — which is exactly what
the ``/status`` API endpoint will use in Phase 7.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.aggregator import aggregator
from agents.architect import architect
from agents.backend import backend
from agents.bug_detector import bug_detector
from agents.coder import coder
from agents.database import database
from agents.frontend import frontend
from agents.planner import planner
from agents.reviewer import reviewer
from agents.supervisor import supervisor
from agents.tester import tester
from core.state import VALID_ROUTES, AgentState

# All 11 workers are now implemented.
BUILT_WORKERS = {
    "planner": planner,
    "architect": architect,
    "frontend": frontend,
    "backend": backend,
    "database": database,
    "coder": coder,
    "bug_detector": bug_detector,
    "tester": tester,
    "reviewer": reviewer,
    "aggregator": aggregator,
}


def route_from_supervisor(state: AgentState) -> str:
    """Conditional-edge function: route to whatever the supervisor wrote."""
    return state["next"]


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    # 1) Nodes: the supervisor plus every built worker.
    builder.add_node("supervisor", supervisor)
    for name, fn in BUILT_WORKERS.items():
        builder.add_node(name, fn)

    # 2) Entry point.
    builder.set_entry_point("supervisor")

    # 3) Conditional edges. Built workers point to themselves; FINISH -> END.
    route_map = {
        route: (route if route in BUILT_WORKERS else END) for route in VALID_ROUTES
    }
    builder.add_conditional_edges("supervisor", route_from_supervisor, route_map)

    # 4) Each worker loops back to the supervisor — the hub of the pattern.
    for name in BUILT_WORKERS:
        builder.add_edge(name, "supervisor")

    # 5) Compile with a checkpointer so runs are persisted per thread_id.
    return builder.compile(checkpointer=checkpointer or MemorySaver())


graph = build_graph()


if __name__ == "__main__":
    import asyncio

    from core.llm import reset_mock
    from core.state import initial_state

    REQUIREMENT = (
        "Design a scalable job board with Next.js frontend, FastAPI backend, "
        "PostgreSQL, user auth, job listings with search, and an employer dashboard"
    )

    def _describe(node: str, update: dict) -> str:
        if node == "supervisor":
            nxt = update.get("next")
            counters = {
                k: update[k]
                for k in ("iteration_count", "bug_iteration_count")
                if k in update
            }
            return f"supervisor -> '{nxt}'" + (f"   {counters}" if counters else "")
        if node == "bug_detector":
            return f"bug_detector -> {update.get('bug_report', '').splitlines()[0]}"
        if node == "tester":
            return f"tester       -> {update.get('test_results', '').splitlines()[0]}"
        if node == "reviewer":
            return f"reviewer     -> {update.get('review_decision')}"
        if node == "coder":
            rerun = "review_decision" in update
            tag = "re-run (cleared stale analyses)" if rerun else "first implementation"
            return f"coder        -> code updated [{tag}]"
        if node == "aggregator":
            return f"aggregator   -> final_document ({len(update.get('final_document',''))} chars)"
        return f"{node:9} produced {[k for k in update if k != 'messages']}"

    async def demo() -> None:
        # With a checkpointer, every run needs a thread_id.
        config = {"configurable": {"thread_id": "demo-1"}, "recursion_limit": 50}

        print("Live trace (stream_mode='updates'):\n")
        reset_mock()
        step = 0
        async for chunk in graph.astream(
            initial_state(REQUIREMENT), config, stream_mode="updates"
        ):
            for node, update in chunk.items():
                step += 1
                print(f"  step {step:>2}: {_describe(node, update)}")

        # Read the final state back FROM THE CHECKPOINTER (not from the stream).
        snapshot = await graph.aget_state(config)
        doc = snapshot.values["final_document"]

        print("\nfinal_document compiled:", len(doc), "chars")
        print("sections:")
        for line in doc.splitlines():
            if line.startswith("## "):
                print("   ", line[3:])

        print("\nCheckpointer proof (thread 'demo-1'):")
        print("   pending next nodes:", snapshot.next, "(empty () == run finished)")
        print("   stack recorded:",
              snapshot.values["detected_frontend_framework"], "/",
              snapshot.values["detected_backend_framework"])

    asyncio.run(demo())
