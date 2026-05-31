"""The coder writes real files (v2.0) and resets stale analyses on a re-run."""
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
    # agent_loop reads get_settings via config.get_settings too; patch there as well.
    import core.agent_loop as al

    monkeypatch.setattr(
        al, "get_settings", lambda: Settings(_env_file=None, use_mock_llm=True)
    )


async def test_first_run_writes_files():
    out = await coder(initial_state("x"))  # no files yet -> first run
    assert out["files"], "coder should write a non-empty file manifest"
    assert out["workspace_dir"]
    # No stale-clear on the first run.
    assert "bug_report" not in out
    assert "review_decision" not in out


async def test_rerun_clears_stale_analyses():
    s = initial_state("x")
    s.update(
        files=["backend/main.py"],  # files present -> this is a re-run
        workspace_dir="",
        bug_report="BUGS_FOUND: sqli",
        test_results="FAIL: t",
        review_decision="REJECT",
        review_notes="please fix",
    )
    out = await coder(s)
    assert out["files"]
    assert out["bug_report"] == ""
    assert out["test_results"] == ""
    assert out["review_decision"] is None
    assert out["review_notes"] == ""
