"""The tester — writes tests and runs conceptual static checks on the code.

Output contract (the supervisor keys off this):
  * problems found -> starts with 'FAIL:' then a list of specific failures.
  * all good       -> starts with 'PASS:' then a short summary.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from core.file_tools import read_workspace_digest
from core.llm import complete
from core.state import AgentState
from core.tools import non_empty

TESTER_SYSTEM = (
    "You are a QA engineer. Given the implementation, design unit tests, "
    "integration test stubs, and run conceptual static checks. If anything is "
    "wrong or untested, respond starting with 'FAIL:' and list the specific "
    "failures. If everything passes, respond starting with 'PASS:' and give a "
    "short summary."
)


async def tester(state: AgentState) -> dict:
    user = (
        f"Backend framework: {state.get('detected_backend_framework', 'unknown')}\n\n"
        f"Code under test (files):\n{read_workspace_digest(state.get('workspace_dir', ''))}"
    )
    results = await complete("tester", TESTER_SYSTEM, user)
    non_empty(results, "test_results")

    return {
        "test_results": results,
        "messages": [
            AIMessage(content=f"[tester] {results.splitlines()[0]}", name="tester")
        ],
    }
