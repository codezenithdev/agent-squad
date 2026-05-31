"""The planner — our first worker agent.

Every worker follows the same three-step shape, so understanding this one means
understanding all ten:

  1. Read what it needs from ``state``.
  2. Call ``complete(role, system, user)`` (mock or real, it doesn't care).
  3. Return a dict of **partial** state updates (never the whole state),
     validating that the output isn't empty first.

The planner turns the raw requirement into an ordered ``task_graph``.
"""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from core.llm import complete
from core.state import AgentState
from core.tools import non_empty

PLANNER_SYSTEM = (
    "You are a senior technical project planner. Given a software requirement, "
    "produce an ordered list of concrete, buildable engineering tasks. One task "
    "per line, numbered. Be specific and concise."
)


def _parse_tasks(raw: str) -> list[str]:
    """Turn the model's numbered/bulleted text into a clean list of strings."""
    tasks: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip a leading "1.", "2)", "-", or "*" bullet/number.
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line)
        if line:
            tasks.append(line)
    return tasks


async def planner(state: AgentState) -> dict:
    user = (
        "Break the following software requirement into an ordered list of "
        "concrete build tasks (one per line, numbered).\n\n"
        f"Requirement:\n{state['input']}"
    )
    raw = await complete("planner", PLANNER_SYSTEM, user)
    tasks = _parse_tasks(raw)

    non_empty(tasks, "task_graph")  # enforce the "never return empty" rule

    return {
        "task_graph": tasks,
        "messages": [
            AIMessage(content=f"[planner] produced {len(tasks)} tasks", name="planner")
        ],
    }
