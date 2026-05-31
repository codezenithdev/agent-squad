"""Output parsers: planner task list + reviewer verdict."""
from agents.planner import _parse_tasks
from agents.reviewer import _parse_review


def test_parse_tasks_numbered():
    assert _parse_tasks("1. First\n2. Second\n3) Third") == ["First", "Second", "Third"]


def test_parse_tasks_bullets_and_blanks():
    raw = "- alpha\n\n* beta\n   \n1. gamma"
    assert _parse_tasks(raw) == ["alpha", "beta", "gamma"]


def test_review_approve():
    decision, notes = _parse_review("APPROVE\nLooks good.\nShip it.")
    assert decision == "APPROVE"
    assert "Looks good" in notes


def test_review_reject():
    decision, notes = _parse_review("REJECT\nFix the SQL injection.")
    assert decision == "REJECT"
    assert "SQL injection" in notes


def test_review_fallback_scans_body():
    # Verdict not on the first line -> scan the whole response.
    decision, _ = _parse_review("Here is my review.\nFinal verdict: APPROVE")
    assert decision == "APPROVE"


def test_review_defaults_to_reject_when_unclear():
    decision, _ = _parse_review("ambiguous text with no clear verdict")
    assert decision == "REJECT"
