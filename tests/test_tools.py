"""Framework detection + small helpers (core/tools.py)."""
import pytest

from core.tools import (
    BACKEND_KEYWORDS,
    FRONTEND_KEYWORDS,
    detect_framework,
    non_empty,
    truncate,
)


def test_detect_nextjs():
    assert detect_framework("Build with Next.js", FRONTEND_KEYWORDS) == "nextjs"


def test_nextjs_wins_over_react():
    # Next.js is React-based; we want the specific framework, not 'react'.
    assert detect_framework("a Next.js app using React", FRONTEND_KEYWORDS) == "nextjs"


def test_sveltekit_wins_over_svelte():
    assert detect_framework("a SvelteKit project", FRONTEND_KEYWORDS) == "sveltekit"


def test_detect_fastapi():
    assert detect_framework("FastAPI backend", BACKEND_KEYWORDS) == "fastapi"


def test_gin_word_boundary():
    # 'gin' must not match inside 'imagine'.
    assert detect_framework("imagine a service", BACKEND_KEYWORDS) == "unknown"
    assert detect_framework("built with Gin", BACKEND_KEYWORDS) == "gin"


def test_unknown_default():
    assert detect_framework("a plain CLI tool", FRONTEND_KEYWORDS) == "unknown"


def test_non_empty_returns_value():
    assert non_empty("hello", "f") == "hello"
    assert non_empty(["a"], "f") == ["a"]


@pytest.mark.parametrize("bad", ["", "   ", None, [], {}])
def test_non_empty_raises_on_empty(bad):
    with pytest.raises(ValueError):
        non_empty(bad, "field")


def test_truncate():
    assert truncate("abcdef", 3).startswith("abc")
    assert truncate("short", 100) == "short"
