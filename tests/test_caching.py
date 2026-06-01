"""Prompt-caching message assembly + usage accounting (v2.2)."""
from core.llm import _build_messages, record_usage, reset_usage, usage_summary


def test_anthropic_marks_prefix_with_cache_control():
    sys, usr = _build_messages("anthropic", "SYS", "USR", "BIGCONTEXT")
    assert isinstance(sys.content, list)
    assert sys.content[0]["text"] == "SYS"
    assert sys.content[-1]["text"] == "BIGCONTEXT"
    assert sys.content[-1]["cache_control"] == {"type": "ephemeral"}
    assert usr.content == "USR"


def test_anthropic_no_prefix_still_caches_system():
    sys, _ = _build_messages("anthropic", "SYS", "USR", "")
    assert sys.content[-1]["text"] == "SYS"
    assert sys.content[-1]["cache_control"] == {"type": "ephemeral"}


def test_openai_is_plain_with_prefix_in_front():
    sys, usr = _build_messages("openai", "SYS", "USR", "CTX")
    assert sys.content == "SYS"  # plain string, no blocks
    assert usr.content.startswith("CTX") and usr.content.endswith("USR")


def test_usage_recording_and_reset():
    reset_usage()

    class R:
        usage_metadata = {
            "input_tokens": 1000,
            "output_tokens": 50,
            "input_token_details": {"cache_read": 800, "cache_creation": 200},
        }

    record_usage(R())
    s = usage_summary()
    assert s == {
        "calls": 1,
        "input": 1000,
        "output": 50,
        "cache_read": 800,
        "cache_creation": 200,
    }
    reset_usage()
    assert usage_summary()["calls"] == 0


def test_usage_tolerates_missing_metadata():
    reset_usage()
    record_usage(object())  # no usage_metadata -> must not crash
    assert usage_summary()["calls"] == 1
