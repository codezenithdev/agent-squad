"""The aggregator's empty-section handling (the clarity fix)."""
import pytest

import core.llm as llm_module
from agents.aggregator import aggregator
from config import Settings
from core.state import initial_state


@pytest.fixture(autouse=True)
def force_mock(monkeypatch):
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings(_env_file=None, use_mock_llm=True)
    )


def _ready_state(**over):
    s = initial_state("Design a job board with Next.js and FastAPI")
    s.update(
        system_design="d",
        frontend_spec="f",
        backend_spec="b",
        db_schema="sch",
        code="c",
        detected_frontend_framework="nextjs",
        detected_backend_framework="fastapi",
        review_decision="APPROVE",
        review_notes="ok",
    )
    s.update(over)
    return s


async def test_empty_scan_and_tests_show_a_note_not_blank():
    s = _ready_state(bug_report="", test_results="", bug_iteration_count=3, iteration_count=3)
    doc = (await aggregator(s))["final_document"]
    assert "no final scan recorded" in doc
    assert "no test run recorded" in doc
    # No dangling "Final scan:" / "Final result:" with nothing after the colon.
    assert "- Final scan: \n" not in doc
    assert "- Final result: \n" not in doc


async def test_real_scan_and_tests_are_shown_verbatim():
    s = _ready_state(bug_report="CLEAN: no issues found", test_results="PASS: 24 tests")
    doc = (await aggregator(s))["final_document"]
    assert "CLEAN: no issues found" in doc
    assert "PASS: 24 tests" in doc
