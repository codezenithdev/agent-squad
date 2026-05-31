"""The coder's stale-analysis reset — the bit that makes the fix-loops converge."""
import pytest

import core.llm as llm_module
from agents.coder import coder
from config import Settings
from core.state import initial_state


@pytest.fixture(autouse=True)
def force_mock(monkeypatch):
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings(_env_file=None, use_mock_llm=True)
    )


async def test_first_run_sets_code_only():
    out = await coder(initial_state("x"))  # code empty -> first run
    assert out["code"].strip()
    # No stale-clear on the first run.
    assert "bug_report" not in out
    assert "review_decision" not in out


async def test_rerun_clears_stale_analyses():
    s = initial_state("x")
    s.update(
        code="old code",
        bug_report="BUGS_FOUND: sqli",
        test_results="FAIL: t",
        review_decision="REJECT",
        review_notes="please fix",
    )
    out = await coder(s)
    assert out["code"].strip()
    assert out["bug_report"] == ""
    assert out["test_results"] == ""
    assert out["review_decision"] is None
    assert out["review_notes"] == ""
