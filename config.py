"""Application configuration.

Everything tunable lives here, loaded once from environment variables / a local
``.env`` file via *pydantic-settings*. Declaring config as a typed model means
it is validated at startup and available as an editor-autocompletable object
instead of scattered ``os.getenv`` calls.

The single most important switch is ``use_mock_llm``:
    True  -> deterministic, free, offline mock responses (learn the orchestration)
    False -> real OpenAI calls (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM mode -----------------------------------------------------------
    use_mock_llm: bool = True

    # --- OpenAI (only required when use_mock_llm is False) ------------------
    openai_api_key: Optional[str] = None

    # --- LangSmith tracing (optional observability) ------------------------
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "multi-agent-architect"

    # --- Model tiering ------------------------------------------------------
    # The supervisor and reviewer do the "hardest thinking" -> GPT-4o.
    # Every other worker uses the cheaper/faster mini model.
    supervisor_model: str = "gpt-4o"
    reviewer_model: str = "gpt-4o"
    worker_model: str = "gpt-4o-mini"

    # --- Iteration / circuit-breaker limit ---------------------------------
    # Max Coder<->Tester and Coder<->BugDetector fix-cycles before we give up
    # and force the pipeline forward (prevents infinite loops).
    max_iterations: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # OPENAI_API_KEY <-> openai_api_key
        extra="ignore",        # ignore unrelated vars in the environment
    )

    def apply_tracing_env(self) -> None:
        """Push tracing/OpenAI vars into ``os.environ`` so LangChain, LangSmith,
        and the OpenAI SDK (which read the process environment directly) pick
        them up. Safe to call unconditionally."""
        if self.langchain_tracing_v2 and self.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
        if self.openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the environment is parsed exactly once per process."""
    return Settings()


if __name__ == "__main__":
    # Quick sanity check: `python config.py` prints the resolved settings.
    s = get_settings()
    print("Resolved settings:")
    print(f"  use_mock_llm      = {s.use_mock_llm}")
    print(f"  supervisor_model  = {s.supervisor_model}")
    print(f"  reviewer_model    = {s.reviewer_model}")
    print(f"  worker_model      = {s.worker_model}")
    print(f"  max_iterations    = {s.max_iterations}")
    print(f"  tracing_enabled   = {s.langchain_tracing_v2}")
    print(f"  openai_key_set    = {bool(s.openai_api_key)}")
