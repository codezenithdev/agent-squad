"""The bug detector — scans the code after the coder runs.

Looks for security issues (SQL injection, XSS, IDOR, missing auth, hardcoded
secrets, insecure deserialization), reliability issues (unhandled promises,
missing error boundaries, race conditions, N+1 queries), performance issues
(missing indexes, unbounded queries, blocking calls in async code), and
framework-specific anti-patterns for the detected stack.

Output contract (the supervisor keys off this):
  * issues found  -> starts with 'BUGS_FOUND:' then a numbered list with
    severity (HIGH/MED/LOW) and a fix suggestion.
  * clean         -> exactly 'CLEAN: no issues found'.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from core.file_tools import read_workspace_digest
from core.llm import complete
from core.state import AgentState
from core.tools import non_empty

BUG_SYSTEM = (
    "You are a security and reliability auditor. Scan the provided code for "
    "security, reliability, performance, and framework-specific anti-patterns. "
    "If you find issues, respond starting with 'BUGS_FOUND:' followed by a "
    "numbered list, each item tagged [HIGH], [MED], or [LOW] with a concrete "
    "fix. If the code is clean, respond with exactly 'CLEAN: no issues found'."
)


async def bug_detector(state: AgentState) -> dict:
    user = (
        f"Frontend framework: {state.get('detected_frontend_framework', 'unknown')}\n"
        f"Backend framework: {state.get('detected_backend_framework', 'unknown')}\n\n"
        f"Code to audit (files):\n{read_workspace_digest(state.get('workspace_dir', ''))}"
    )
    report = await complete("bug_detector", BUG_SYSTEM, user)
    non_empty(report, "bug_report")

    return {
        "bug_report": report,
        "messages": [
            AIMessage(content=f"[bug_detector] {report.splitlines()[0]}",
                      name="bug_detector")
        ],
    }
