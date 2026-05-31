"""The mock LLM path: returns text and follows the scripted fix-loop sequence."""
import pytest

import core.llm as llm_module
from config import Settings
from core.llm import complete, reset_mock


@pytest.fixture(autouse=True)
def force_mock(monkeypatch):
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings(_env_file=None, use_mock_llm=True)
    )


async def test_complete_returns_nonempty_text():
    out = await complete("planner", "system", "Build X")
    assert isinstance(out, str) and out.strip()


async def test_bug_detector_finds_then_clean():
    reset_mock()
    first = await complete("bug_detector", "s", "code")
    second = await complete("bug_detector", "s", "code")
    assert first.startswith("BUGS_FOUND")
    assert second.startswith("CLEAN")


async def test_tester_fails_then_passes():
    reset_mock()
    assert (await complete("tester", "s", "code")).startswith("FAIL")
    assert (await complete("tester", "s", "code")).startswith("PASS")
