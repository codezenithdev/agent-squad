"""Shared pytest fixtures.

Each test starts from a clean slate: mock call-counters reset and the cached
Settings cleared, so tests don't leak state into each other.
"""
import pytest

from config import get_settings
from core.llm import reset_mock, reset_usage


@pytest.fixture(autouse=True)
def _isolate():
    reset_mock()
    reset_usage()
    get_settings.cache_clear()
    yield
    reset_mock()
    reset_usage()
    get_settings.cache_clear()
