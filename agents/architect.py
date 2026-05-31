"""The architect — turns the requirement + task list into a system design.

Same three-step shape as the planner. It reads ``input`` and ``task_graph`` and
produces the high-level ``system_design`` that the per-layer agents build on.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from core.llm import complete
from core.state import AgentState
from core.tools import non_empty

ARCHITECT_SYSTEM = (
    "You are a principal software architect. Given a requirement and a task "
    "list, produce a concise high-level system design: major components, how "
    "they communicate, data flow, scaling strategy, and key cross-cutting "
    "concerns (auth, logging, observability). Do not write code."
)


async def architect(state: AgentState) -> dict:
    tasks = "\n".join(f"- {t}" for t in state.get("task_graph", []))
    user = (
        f"Requirement:\n{state['input']}\n\n"
        f"Planned tasks:\n{tasks}\n\n"
        "Produce the high-level system design."
    )
    design = await complete("architect", ARCHITECT_SYSTEM, user)
    non_empty(design, "system_design")

    return {
        "system_design": design,
        "messages": [
            AIMessage(content="[architect] system design produced", name="architect")
        ],
    }
