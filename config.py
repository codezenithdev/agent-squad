"""Application configuration.

Everything tunable lives here, loaded once from environment variables / a local
``.env`` file via *pydantic-settings*. Declaring config as a typed model means
it is validated at startup and available as an editor-autocompletable object
instead of scattered ``os.getenv`` calls.

Two switches matter most:
  * ``use_mock_llm``  : True  -> free, offline, deterministic mock responses.
                        False -> real LLM calls.
  * ``llm_provider``  : 'auto' (prefer OpenAI when its key is set, else
                        Anthropic), or force 'openai' / 'anthropic'.

The system is provider-agnostic: each role maps to a "strong" or "worker" model
tier, and each provider supplies concrete model names for those tiers.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM mode -----------------------------------------------------------
    use_mock_llm: bool = True

    # --- Provider selection -------------------------------------------------
    # 'auto' -> prefer OpenAI when OPENAI_API_KEY is set, else Anthropic.
    llm_provider: str = "auto"  # auto | openai | anthropic

    # --- API keys (only the chosen provider's key is required) -------------
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # --- LangSmith tracing (optional observability) ------------------------
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "multi-agent-architect"

    # --- Model tiering, per provider ---------------------------------------
    # The reviewer does the "hardest thinking" -> strong tier.
    # Every other worker uses the cheaper/faster tier. (The supervisor routes in
    # pure Python and never calls an LLM.)
    openai_strong_model: str = "gpt-4o"
    openai_worker_model: str = "gpt-4o-mini"
    anthropic_strong_model: str = "claude-sonnet-4-5"
    anthropic_worker_model: str = "claude-haiku-4-5"

    # --- Iteration / circuit-breaker limit ---------------------------------
    max_iterations: int = 3

    # --- Execution sandbox (v2.1) ------------------------------------------
    # Generated code is run ONLY inside throwaway Docker containers, never on the
    # host. These bound each container run.
    sandbox_python_image: str = "python:3.11-slim"
    sandbox_node_image: str = "node:20-slim"
    sandbox_timeout: int = 300  # seconds per sandbox command
    sandbox_memory: str = "1g"
    sandbox_cpus: str = "2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # OPENAI_API_KEY <-> openai_api_key
        extra="ignore",        # ignore unrelated vars in the environment
    )

    # --- Provider helpers ---------------------------------------------------

    def resolve_provider(self) -> str:
        """Decide which provider to use. Explicit setting wins; otherwise 'auto'
        prefers OpenAI when its key is present, then Anthropic. Falls back to
        'openai' so tier lookups still work in mock mode with no keys."""
        if self.llm_provider in ("openai", "anthropic"):
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        return "openai"

    def tier_models(self, provider: str) -> tuple[str, str]:
        """Return ``(strong_model, worker_model)`` for the given provider."""
        if provider == "anthropic":
            return self.anthropic_strong_model, self.anthropic_worker_model
        return self.openai_strong_model, self.openai_worker_model

    def key_for(self, provider: str) -> Optional[str]:
        return self.anthropic_api_key if provider == "anthropic" else self.openai_api_key

    def apply_provider_env(self) -> None:
        """Push provider + tracing vars into ``os.environ`` so LangChain /
        LangSmith / the provider SDKs (which read the process environment) pick
        them up. Safe to call unconditionally."""
        if self.langchain_tracing_v2 and self.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
        # Overwrite (not setdefault): an ambient EMPTY key var must not win.
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

    def recover_keys_from_dotenv(self) -> None:
        """Recover API keys from the .env file when an ambient EMPTY env var
        (e.g. a shell with ``ANTHROPIC_API_KEY=``) shadows the file value in
        pydantic's precedence order. Only fills keys that came back empty.

        Called by ``get_settings()`` (the app path); tests that construct
        ``Settings(_env_file=None, ...)`` directly skip it and stay hermetic."""
        if self.openai_api_key and self.anthropic_api_key:
            return
        try:
            from dotenv import dotenv_values

            vals = dotenv_values(self.model_config.get("env_file", ".env"))
        except Exception:
            vals = {}
        if not self.openai_api_key and vals.get("OPENAI_API_KEY"):
            self.openai_api_key = vals["OPENAI_API_KEY"]
        if not self.anthropic_api_key and vals.get("ANTHROPIC_API_KEY"):
            self.anthropic_api_key = vals["ANTHROPIC_API_KEY"]


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the environment is parsed exactly once per process."""
    settings = Settings()
    settings.recover_keys_from_dotenv()  # un-shadow keys hidden by empty env vars
    return settings


if __name__ == "__main__":
    # Quick sanity check: `python config.py` prints the resolved settings.
    s = get_settings()
    provider = s.resolve_provider()
    strong, worker = s.tier_models(provider)
    print("Resolved settings:")
    print(f"  use_mock_llm        = {s.use_mock_llm}")
    print(f"  llm_provider        = {s.llm_provider}  -> resolved: {provider}")
    print(f"  strong-tier model   = {strong}")
    print(f"  worker-tier model   = {worker}")
    print(f"  max_iterations      = {s.max_iterations}")
    print(f"  tracing_enabled     = {s.langchain_tracing_v2}")
    print(f"  openai_key_set      = {bool(s.openai_api_key)}")
    print(f"  anthropic_key_set   = {bool(s.anthropic_api_key)}")
