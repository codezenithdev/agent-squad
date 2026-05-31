"""The StateGraph — how nodes are wired together.

A LangGraph graph is: add nodes, set an entry point, add edges. The supervisor
pattern uses *conditional* edges out of the supervisor (the destination depends
on ``state['next']``) and plain edges from every worker back to the supervisor.

This file grows across phases. Right now (Phase 3) only the supervisor and
planner are real nodes. Every other route the supervisor might choose is
temporarily wired to END, so the graph runs and terminates even though the other
agents don't exist yet. As we build each agent, we replace its ``END`` with the
real node — the supervisor itself never changes.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.planner import planner
from agents.supervisor import supervisor
from core.state import AgentState


def route_from_supervisor(state: AgentState) -> str:
    """The conditional-edge function: tell LangGraph where to go next by reading
    the destination the supervisor just wrote into the state."""
    return state["next"]


# Targets not yet implemented -> END for now (replaced as agents are built).
_NOT_BUILT_YET = [
    "architect",
    "frontend",
    "backend",
    "database",
    "coder",
    "bug_detector",
    "tester",
    "reviewer",
    "aggregator",
]


def build_graph():
    builder = StateGraph(AgentState)

    # 1) Register the nodes that exist.
    builder.add_node("supervisor", supervisor)
    builder.add_node("planner", planner)

    # 2) Start at the supervisor.
    builder.set_entry_point("supervisor")

    # 3) Conditional edges: supervisor -> (chosen node) based on state['next'].
    route_map = {"planner": "planner", "FINISH": END}
    for name in _NOT_BUILT_YET:
        route_map[name] = END  # temporary terminal shortcut
    builder.add_conditional_edges("supervisor", route_from_supervisor, route_map)

    # 4) Workers always return to the supervisor (the hub of the pattern).
    builder.add_edge("planner", "supervisor")

    return builder.compile()


graph = build_graph()


if __name__ == "__main__":
    import asyncio

    from core.llm import reset_mock
    from core.state import initial_state

    REQUIREMENT = "Design a job board with a Next.js frontend and FastAPI backend"
    BUILT = {"supervisor", "planner"}

    async def demo() -> None:
        reset_mock()
        config = {"recursion_limit": 50}

        print("Live trace (stream_mode='updates'):\n")
        step = 0
        async for chunk in graph.astream(
            initial_state(REQUIREMENT), config, stream_mode="updates"
        ):
            for node, update in chunk.items():
                step += 1
                if node == "supervisor":
                    nxt = update.get("next")
                    note = (
                        ""
                        if nxt in BUILT or nxt == "FINISH"
                        else "  (not built yet -> END)"
                    )
                    print(f"  step {step}: supervisor decided next = '{nxt}'{note}")
                else:
                    n = len(update.get("task_graph", []))
                    print(f"  step {step}: {node} ran -> task_graph ({n} tasks)")

        print("\nResulting task_graph:")
        reset_mock()
        final = await graph.ainvoke(initial_state(REQUIREMENT), config)
        for i, task in enumerate(final["task_graph"], 1):
            print(f"  {i}. {task}")

    asyncio.run(demo())
