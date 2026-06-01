"""The reviewer — the final holistic judgment (uses GPT-4o via model tiering).

Reviews everything together: architecture coherence, code quality, security
(cross-checked against the bug_report), test coverage, spec alignment, and
framework best practices. It must end with a clear verdict.

Output contract: the first line is the decision (APPROVE or REJECT); the rest is
the notes. We parse that into:
  * review_decision -> exactly 'APPROVE' or 'REJECT'
  * review_notes    -> actionable feedback (especially important on REJECT, since
    the coder reads it on the next revision).
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from core.file_tools import read_workspace_digest
from core.llm import complete
from core.state import AgentState
from core.tools import non_empty

REVIEWER_SYSTEM = (
    "You are a staff engineer doing a final review. Judge architecture "
    "coherence, code quality, security (cross-check the bug report), test "
    "coverage, spec alignment, and framework best practices. Respond with the "
    "verdict on the FIRST line: 'APPROVE' or 'REJECT'. On the following lines "
    "give concise, actionable notes (required if you REJECT)."
)


def _parse_review(raw: str) -> tuple[str, str]:
    """First line -> decision; remaining lines -> notes. Robust to extra text."""
    lines = raw.splitlines()
    first = lines[0].strip().upper() if lines else ""
    if first.startswith("APPROVE"):
        decision = "APPROVE"
    elif first.startswith("REJECT"):
        decision = "REJECT"
    else:
        # Fall back to scanning the whole response; default to REJECT if unclear
        # (safer to ask for another pass than to wrongly approve).
        decision = "APPROVE" if "APPROVE" in raw.upper() else "REJECT"
    notes = "\n".join(lines[1:]).strip()
    return decision, notes


async def reviewer(state: AgentState) -> dict:
    # Stable, large context -> cacheable prefix; the variable verdict inputs
    # (bug report, test results) + instruction stay in the per-call message.
    cache_prefix = (
        f"Requirement:\n{state['input']}\n\n"
        f"System design:\n{state.get('system_design', '')}\n\n"
        f"Code (files):\n{read_workspace_digest(state.get('workspace_dir', ''))}"
    )
    user = (
        f"Bug report:\n{state.get('bug_report', '')}\n\n"
        f"Test results:\n{state.get('test_results', '')}\n\n"
        "Give your verdict (APPROVE/REJECT on the first line) and notes."
    )
    raw = await complete(
        "reviewer", REVIEWER_SYSTEM, user, temperature=0.1, cache_prefix=cache_prefix
    )
    decision, notes = _parse_review(raw)
    non_empty(decision, "review_decision")

    return {
        "review_decision": decision,
        "review_notes": notes,
        "messages": [AIMessage(content=f"[reviewer] {decision}", name="reviewer")],
    }
