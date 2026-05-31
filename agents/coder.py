"""The coder — generates and revises the implementation.

This agent runs in two modes:
  * **First run** (``code`` is empty): generate the full implementation from the
    design + specs + schema, using the detected frameworks for idiomatic output.
  * **Re-run** (``code`` already exists): the supervisor sent us back because a
    bug was found, a test failed, or the reviewer rejected. Incorporate that
    feedback into the existing code.

THE CRITICAL PART — stale-analysis reset:
On a re-run the code *changes*, which means the previous ``bug_report``,
``test_results``, and the reviewer's ``review_decision``/``review_notes`` now
describe code that no longer exists. If we left them in place, the supervisor's
rules would keep reacting to stale verdicts and either loop forever or skip
re-verification. So a re-run clears all of them, forcing a fresh
bug-scan -> test -> review pass over the new code. This is what makes the
coder<->bug_detector and coder<->tester cycles actually work.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from core.llm import complete
from core.state import AgentState
from core.tools import non_empty

CODER_SYSTEM = (
    "You are a senior full-stack engineer. Produce idiomatic, secure "
    "implementation code for the specified stack: parameterized queries, proper "
    "authentication/authorization checks, input validation, and error handling. "
    "On a revision, address every piece of feedback precisely and return the "
    "full updated implementation."
)


def _gather_feedback(state: AgentState) -> str:
    """Collect any outstanding feedback to fold into a revision."""
    parts: list[str] = []
    bug_report = state.get("bug_report", "")
    if bug_report and "BUGS_FOUND" in bug_report:
        parts.append(f"Bug report to fix:\n{bug_report}")
    test_results = state.get("test_results", "")
    if test_results.startswith("FAIL"):
        parts.append(f"Failing tests to fix:\n{test_results}")
    if state.get("review_decision") == "REJECT" and state.get("review_notes"):
        parts.append(f"Reviewer rejection notes to address:\n{state['review_notes']}")
    return "\n\n".join(parts)


async def coder(state: AgentState) -> dict:
    is_rerun = bool(state.get("code"))

    base = (
        f"Frontend framework: {state.get('detected_frontend_framework', 'unknown')}\n"
        f"Backend framework: {state.get('detected_backend_framework', 'unknown')}\n\n"
        f"System design:\n{state.get('system_design', '')}\n\n"
        f"Frontend spec:\n{state.get('frontend_spec', '')}\n\n"
        f"Backend spec:\n{state.get('backend_spec', '')}\n\n"
        f"Database schema:\n{state.get('db_schema', '')}\n"
    )
    if is_rerun:
        user = (
            base
            + "\n\nThis is a revision. Address the following feedback and return "
            "the full updated implementation:\n\n"
            + _gather_feedback(state)
        )
    else:
        user = base + "\n\nGenerate the full initial implementation."

    code = await complete("coder", CODER_SYSTEM, user)
    non_empty(code, "code")

    updates: dict = {
        "code": code,
        "messages": [
            AIMessage(
                content=f"[coder] {'revised' if is_rerun else 'initial'} implementation",
                name="coder",
            )
        ],
    }

    if is_rerun:
        # The code changed -> prior analyses & the prior verdict are now stale.
        # Clear them so the supervisor re-verifies the new code from scratch.
        updates["bug_report"] = ""
        updates["test_results"] = ""
        updates["review_decision"] = None
        updates["review_notes"] = ""

    return updates
