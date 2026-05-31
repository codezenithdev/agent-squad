"""Provider resolution + model tiering (Phase 10 logic)."""
import core.llm as llm_module
from config import Settings


def _settings(**kw):
    # _env_file=None makes Settings hermetic (ignores the local .env).
    return Settings(_env_file=None, **kw)


# --- resolve_provider ------------------------------------------------------

def test_auto_prefers_openai_when_both_keys():
    s = _settings(llm_provider="auto", openai_api_key="o", anthropic_api_key="a")
    assert s.resolve_provider() == "openai"


def test_auto_falls_back_to_anthropic():
    s = _settings(llm_provider="auto", openai_api_key=None, anthropic_api_key="a")
    assert s.resolve_provider() == "anthropic"


def test_auto_defaults_openai_when_no_keys():
    s = _settings(llm_provider="auto", openai_api_key=None, anthropic_api_key=None)
    assert s.resolve_provider() == "openai"


def test_explicit_provider_overrides_keys():
    s = _settings(llm_provider="anthropic", openai_api_key="o", anthropic_api_key=None)
    assert s.resolve_provider() == "anthropic"


# --- tier_models / key_for -------------------------------------------------

def test_tier_models_openai():
    assert _settings().tier_models("openai") == ("gpt-4o", "gpt-4o-mini")


def test_tier_models_anthropic():
    assert _settings().tier_models("anthropic") == (
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    )


def test_key_for():
    s = _settings(openai_api_key="o", anthropic_api_key="a")
    assert s.key_for("openai") == "o"
    assert s.key_for("anthropic") == "a"


# --- model_for_role (tier mapping) -----------------------------------------

def test_model_for_role_strong_vs_worker(monkeypatch):
    s = _settings(llm_provider="anthropic", anthropic_api_key="a")
    monkeypatch.setattr(llm_module, "get_settings", lambda: s)
    assert llm_module.model_for_role("reviewer") == ("anthropic", "claude-sonnet-4-5")
    assert llm_module.model_for_role("supervisor") == ("anthropic", "claude-sonnet-4-5")
    assert llm_module.model_for_role("planner") == ("anthropic", "claude-haiku-4-5")
    assert llm_module.model_for_role("coder") == ("anthropic", "claude-haiku-4-5")
