"""The StateGraph — how nodes are wired together.

A LangGraph graph is: add nodes, set an entry point, add edges. The supervisor
pattern uses *conditional* edges out of the supervisor (the destination depends
on ``state['next']``) and plain edges from every worker back to the supervisor.

This file grows across phases, but the wiring is now data-driven: list a worker
in ``BUILT_WORKERS`` and it becomes a real node with an edge back to the
supervisor. Any route in ``VALID_ROUTES`` that isn't built yet (and ``FINISH``)
is automatically wired to END, so the graph always runs and terminates. The
supervisor itself never changes.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.architect import architect
from agents.backend import backend
from agents.database import database
from agents.frontend import frontend
from agents.planner import planner
from agents.supervisor import supervisor
from core.state import VALID_ROUTES, AgentState

# Workers implemented so far. Add to this dict as each phase lands.
BUILT_WORKERS = {
    "planner": planner,
    "architect": architect,
    "frontend": frontend,
    "backend": backend,
    "database": database,
}


def route_from_supervisor(state: AgentState) -> str:
    """Conditional-edge function: route to whatever the supervisor wrote."""
    return state["next"]


def build_graph():
    builder = StateGraph(AgentState)

    # 1) Nodes: the supervisor plus every built worker.
    builder.add_node("supervisor", supervisor)
    for name, fn in BUILT_WORKERS.items():
        builder.add_node(name, fn)

    # 2) Entry point.
    builder.set_entry_point("supervisor")

    # 3) Conditional edges. Every legal route needs a destination: built workers
    #    point to themselves; everything else (incl. FINISH) points to END.
    route_map = {
        route: (route if route in BUILT_WORKERS else END) for route in VALID_ROUTES
    }
    builder.add_conditional_edges("supervisor", route_from_supervisor, route_map)

    # 4) Each worker loops back to the supervisor — the hub of the pattern.
    for name in BUILT_WORKERS:
        builder.add_edge(name, "supervisor")

    return builder.compile()


graph = build_graph()


if __name__ == "__main__":
    import asyncio

    from core.llm import reset_mock
    from core.state import initial_state

    REQUIREMENT = (
        "Design a scalable job board with Next.js frontend, FastAPI backend, "
        "PostgreSQL, user auth, job listings with search, and an employer dashboard"
    )

    async def demo() -> None:
        config = {"recursion_limit": 50}

        print("Live trace (stream_mode='updates'):\n")
        reset_mock()
        step = 0
        async for chunk in graph.astream(
            initial_state(REQUIREMENT), config, stream_mode="updates"
        ):
            for node, update in chunk.items():
                step += 1
                if node == "supervisor":
                    nxt = update.get("next")
                    built = nxt in BUILT_WORKERS or nxt == "FINISH"
                    note = "" if built else "  (not built yet -> END)"
                    print(f"  step {step:>2}: supervisor -> '{nxt}'{note}")
                else:
                    fields = [k for k in update if k != "messages"]
                    print(f"  step {step:>2}: {node:9} produced {fields}")

        print("\nFinal state summary:")
        reset_mock()
        final = await graph.ainvoke(initial_state(REQUIREMENT), config)
        print(
            f"  detected_frontend_framework = {final['detected_frontend_framework']!r}"
        )
        print(
            f"  detected_backend_framework  = {final['detected_backend_framework']!r}"
        )
        print(f"  task_graph     : {len(final['task_graph'])} tasks")
        print(f"  system_design  : {len(final['system_design'])} chars")
        print(f"  frontend_spec  : {len(final['frontend_spec'])} chars")
        print(f"  backend_spec   : {len(final['backend_spec'])} chars")
        print(f"  db_schema      : {len(final['db_schema'])} chars")

    asyncio.run(demo())
