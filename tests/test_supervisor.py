"""The deterministic router (agents/supervisor.py::decide_route).

These tests are the heart of the safety net: they pin every routing rule, the
counter increments, the circuit breaker, and the guaranteed terminal path —
all without a single LLM call.
"""
import pytest

import agents.supervisor as sup
from config import Settings
from core.state import initial_state


@pytest.fixture(autouse=True)
def fixed_max_iterations(monkeypatch):
    # Pin max_iterations=3 regardless of the local .env.
    monkeypatch.setattr(
        sup, "get_settings", lambda: Settings(_env_file=None, max_iterations=3)
    )


def _designed():
    """A state where the linear design phase + first code are already done."""
    s = initial_state("x")
    s.update(
        task_graph=["t"],
        system_design="d",
        frontend_spec="f",
        backend_spec="b",
        db_schema="sch",
        code="c",
    )
    return s


def _route(**over):
    s = _designed()
    s.update(over)
    return sup.decide_route(s)


# --- design phase ----------------------------------------------------------

def test_empty_state_routes_to_planner():
    assert sup.decide_route(initial_state("x"))[0] == "planner"


def test_design_phase_in_order():
    s = initial_state("x")
    s["task_graph"] = ["t"]
    assert sup.decide_route(s)[0] == "architect"
    s["system_design"] = "d"
    assert sup.decide_route(s)[0] == "frontend"
    s["frontend_spec"] = "f"
    assert sup.decide_route(s)[0] == "backend"
    s["backend_spec"] = "b"
    assert sup.decide_route(s)[0] == "database"
    s["db_schema"] = "sch"
    assert sup.decide_route(s)[0] == "coder"  # code still empty


# --- verify/fix loop -------------------------------------------------------

def test_code_present_routes_to_bug_detector():
    assert _route()[0] == "bug_detector"


def test_bugs_found_routes_to_coder_and_counts():
    nxt, extra = _route(bug_report="BUGS_FOUND: x")
    assert nxt == "coder"
    assert extra == {"bug_iteration_count": 1}


def test_clean_routes_to_tester():
    assert _route(bug_report="CLEAN: ok")[0] == "tester"


def test_fail_routes_to_coder_and_counts():
    nxt, extra = _route(bug_report="CLEAN", test_results="FAIL: x")
    assert nxt == "coder"
    assert extra == {"iteration_count": 1}


def test_pass_routes_to_reviewer():
    assert _route(bug_report="CLEAN", test_results="PASS: ok")[0] == "reviewer"


# --- review + completion ---------------------------------------------------

def test_reject_with_budget_routes_to_coder():
    nxt, extra = _route(bug_report="CLEAN", test_results="PASS", review_decision="REJECT")
    assert nxt == "coder"
    assert extra == {"iteration_count": 1}


def test_approve_routes_to_aggregator():
    assert _route(
        bug_report="CLEAN", test_results="PASS", review_decision="APPROVE"
    )[0] == "aggregator"


def test_final_document_routes_to_finish():
    assert _route(
        bug_report="CLEAN",
        test_results="PASS",
        review_decision="APPROVE",
        final_document="DOC",
    )[0] == "FINISH"


# --- circuit breaker + terminal hardening ----------------------------------

def test_breaker_still_runs_tester_once():
    # Bug-fix budget spent before the tester ran -> the safety net gives it one
    # turn so we never review without any test signal.
    assert _route(bug_report="BUGS_FOUND: x", bug_iteration_count=3)[0] == "tester"


def test_breaker_after_test_signal_goes_to_reviewer():
    # Once a test result exists, the breaker routes on to review (no loop).
    assert _route(
        bug_report="BUGS_FOUND: x", bug_iteration_count=3, test_results="PASS: ok"
    )[0] == "reviewer"


def test_breaker_reject_goes_to_aggregator_not_dead_end():
    # Reject after budget exhausted -> aggregator (never dead-ends).
    nxt, _ = _route(
        bug_report="CLEAN",
        test_results="PASS",
        review_decision="REJECT",
        iteration_count=3,
    )
    assert nxt == "aggregator"
