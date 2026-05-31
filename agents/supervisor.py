"""The supervisor — the router at the center of the pattern.

After every worker finishes, control returns here. The supervisor looks at the
current state and decides who runs next. It does NOT ask an LLM to make that
decision: the spec's 13 rules are all deterministic predicates on state
(`if system_design is empty -> architect`), so we evaluate them in plain Python.
Paying GPT-4o to compute `if x == ""` would be slower, costlier, and less
reliable.

We still honor the spec's hard rule — *structured output, never free-text
parsing* — by returning the choice wrapped in a validated ``RouteDecision``
(see core/state.py). A routing typo therefore raises immediately instead of
silently sending the graph somewhere that doesn't exist.

The routing is split into a pure function ``decide_route(state)`` (trivial to
read and unit-test) and a thin async node ``supervisor(state)`` that wraps it.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from config import get_settings
from core.state import AgentState, RouteDecision


def decide_route(state: AgentState) -> tuple[str, dict]:
    """Pure routing logic. Returns ``(next_node, extra_state_updates)``.

    ``extra_state_updates`` carries loop-counter increments so that the act of
    *deciding* to re-run the coder is what advances the circuit-breaker counter.
    """
    max_i = get_settings().max_iterations

    task_graph = state.get("task_graph") or []
    system_design = state.get("system_design", "")
    frontend_spec = state.get("frontend_spec", "")
    backend_spec = state.get("backend_spec", "")
    db_schema = state.get("db_schema", "")
    code = state.get("code", "")
    bug_report = state.get("bug_report", "")
    test_results = state.get("test_results", "")
    review_decision = state.get("review_decision")
    review_notes_done = review_decision is not None
    final_document = state.get("final_document", "")
    iteration = state.get("iteration_count", 0)
    bug_iteration = state.get("bug_iteration_count", 0)

    # Circuit breaker: once either fix-budget is spent, stop looping and let the
    # pipeline move forward to judgment/compilation regardless of open issues.
    breaker = iteration >= max_i or bug_iteration >= max_i

    # --- Rules 1-5: the linear design phase -------------------------------
    if not task_graph:
        return "planner", {}
    if not system_design:
        return "architect", {}
    if not frontend_spec:
        return "frontend", {}
    if not backend_spec:
        return "backend", {}
    if not db_schema:
        return "database", {}

    # --- Rule 6: first implementation, or re-code after a REJECT ----------
    # The REJECT branch increments iteration_count so a stubborn reviewer can't
    # loop forever (hardening; see notes in coder.py about clearing the verdict).
    if not code:
        return "coder", {}
    if review_decision == "REJECT" and iteration < max_i:
        return "coder", {"iteration_count": iteration + 1}

    # --- Rules 7-10: the verify/fix loop (skipped once breaker trips) ------
    if not breaker:
        # 7: scan the code if we haven't yet
        if not bug_report:
            return "bug_detector", {}
        # 8: bugs found -> fix, counting the cycle
        if "BUGS_FOUND" in bug_report and bug_iteration < max_i:
            return "coder", {"bug_iteration_count": bug_iteration + 1}
        # 9: run tests if we haven't yet
        if not test_results:
            return "tester", {}
        # 10: tests failed -> fix, counting the cycle
        if "FAIL" in test_results and iteration < max_i:
            return "coder", {"iteration_count": iteration + 1}

    # --- Rule 11: holistic review -----------------------------------------
    if not review_notes_done:
        return "reviewer", {}

    # --- Rule 12 (+ terminal hardening): compile the deliverable ----------
    # Approve -> aggregate. Also aggregate when the breaker is tripped even on a
    # REJECT, so the run always reaches a document instead of dead-ending.
    if not final_document and (review_decision == "APPROVE" or breaker):
        return "aggregator", {}

    # --- Rule 13 + absolute fallback: we're done --------------------------
    return "FINISH", {}


async def supervisor(state: AgentState) -> dict:
    """Async node wrapper: decide the route, validate it, return partial state."""
    route, extra = decide_route(state)

    # Structured output: validated, never free-text-parsed.
    decision = RouteDecision(next=route)

    return {
        "next": decision.next,
        "messages": [AIMessage(content=f"route -> {decision.next}", name="supervisor")],
        **extra,
    }
