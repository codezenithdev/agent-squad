"""The LLM layer — the single spine every agent calls.

One async function, ``complete(role, system, user)``, hides four concerns from
the agents:

  1. **Mock vs real.** When ``settings.use_mock_llm`` is True it returns
     deterministic, offline, free text. When False it calls a real provider.
  2. **Provider.** OpenAI *or* Anthropic (Claude), chosen by
     ``settings.resolve_provider()``. We build the client with LangChain's
     ``init_chat_model`` so there is no separate per-provider code path.
  3. **Model tiering.** Each role maps to a "strong" tier (the reviewer) or a
     "worker" tier (everyone else); each provider supplies the concrete names.
  4. **Reliability.** Real calls are wrapped in tenacity retry with exponential
     backoff.

Because every agent goes through here, flipping config switches the whole system
between mock/real and OpenAI/Anthropic — the 11 agents never change.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.tools import truncate


# ---------------------------------------------------------------------------
# Provider + model tiering
# ---------------------------------------------------------------------------

def model_for_role(role: str) -> tuple[str, str]:
    """Return ``(provider, model)`` for a role, honoring provider resolution and
    the strong/worker tier split. The reviewer (and the never-LLM-calling
    supervisor) get the strong tier; everyone else gets the worker tier."""
    s = get_settings()
    provider = s.resolve_provider()
    strong, worker = s.tier_models(provider)
    model = strong if role in ("reviewer", "supervisor") else worker
    return provider, model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def complete(
    role: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    cache_prefix: str = "",
    web_search: bool = False,
) -> str:
    """Return the model's text for a prompt.

    ``cache_prefix`` is large, stable context (specs, design, a code digest) that
    repeats across calls. It's placed in a cacheable prefix so repeated calls are
    cheap (Anthropic prompt caching; OpenAI auto-caches stable prefixes).

    ``web_search`` lets the model fetch current docs via Claude's server-side
    web search tool. It only applies on the Anthropic provider (no-op otherwise).

    In mock mode this is instant and free.
    """
    settings = get_settings()
    if settings.use_mock_llm:
        return _mock_complete(role, user)

    provider = settings.resolve_provider()
    if not settings.key_for(provider):
        raise RuntimeError(
            f"use_mock_llm is False and provider '{provider}' has no API key set. "
            f"Set the matching key in .env (OPENAI_API_KEY / ANTHROPIC_API_KEY), "
            f"choose a different LLM_PROVIDER, or set USE_MOCK_LLM=true."
        )
    settings.apply_provider_env()
    return await _complete_real(role, system, user, temperature, cache_prefix, web_search)


# ---------------------------------------------------------------------------
# Real provider path (with retry/backoff)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _get_chat_model(provider: str, model: str, temperature: float):
    """Cache one chat client per (provider, model, temperature). Imported lazily
    via init_chat_model, which picks ChatOpenAI / ChatAnthropic for us."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(model, model_provider=provider, temperature=temperature)


def web_search_tool() -> dict:
    """Anthropic server-side web search tool definition. Bind it and Claude runs
    the searches itself (no client-side execution), returning a cited answer."""
    s = get_settings()
    return {
        "type": s.web_search_tool_type,
        "name": "web_search",
        "max_uses": s.web_search_max_uses,
    }


def _as_text(content) -> str:
    """Normalize a message's content to a plain string. Anthropic can return a
    list of content blocks; OpenAI returns a string. Handle both."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", str(block)))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Prompt assembly + caching (v2.2)
# ---------------------------------------------------------------------------
# Anthropic prompt caching: mark a stable prefix block with
# cache_control={"type": "ephemeral"} and Anthropic caches everything up to it,
# so repeated calls sharing that prefix pay only ~10% for the cached tokens.
# OpenAI auto-caches identical prefixes, so we just keep stable content at the
# front. Caching only kicks in above the provider's minimum prefix size
# (~1024 tokens), so it helps the big-context calls, not the tiny prompts.

def _build_messages(provider: str, system: str, user: str, cache_prefix: str):
    from langchain_core.messages import HumanMessage, SystemMessage

    if provider == "anthropic":
        blocks = [{"type": "text", "text": system}]
        if cache_prefix:
            blocks.append({"type": "text", "text": cache_prefix})
        # Cache everything up to & including the last system block.
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
        return [SystemMessage(content=blocks), HumanMessage(content=user)]

    # OpenAI / others: keep the stable context at the front (auto-cached).
    user_text = f"{cache_prefix}\n\n{user}" if cache_prefix else user
    return [SystemMessage(content=system), HumanMessage(content=user_text)]


# Running token tally so caching savings are observable.
_usage = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}


def reset_usage() -> None:
    for key in _usage:
        _usage[key] = 0


def usage_summary() -> dict:
    return dict(_usage)


def record_usage(response) -> None:
    meta = getattr(response, "usage_metadata", None) or {}
    _usage["calls"] += 1
    _usage["input"] += meta.get("input_tokens", 0) or 0
    _usage["output"] += meta.get("output_tokens", 0) or 0
    details = meta.get("input_token_details", {}) or {}
    _usage["cache_read"] += details.get("cache_read", 0) or 0
    _usage["cache_creation"] += details.get("cache_creation", 0) or 0


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
async def _complete_real(
    role: str,
    system: str,
    user: str,
    temperature: float,
    cache_prefix: str = "",
    web_search: bool = False,
) -> str:
    """Call the resolved provider asynchronously. tenacity retries this up to 4
    times with exponential backoff (1s, 2s, 4s, ... capped at 30s)."""
    provider, model = model_for_role(role)
    chat = _get_chat_model(provider, model, temperature)
    # Web search is an Anthropic server-side tool; bind it only there.
    if web_search and provider == "anthropic" and get_settings().enable_web_search:
        chat = chat.bind_tools([web_search_tool()])
    response = await chat.ainvoke(_build_messages(provider, system, user, cache_prefix))
    record_usage(response)
    return _as_text(response.content)


# ---------------------------------------------------------------------------
# Mock path (deterministic, scripted)
# ---------------------------------------------------------------------------
# Some roles vary their answer by call number so we can demonstrate the fix
# loops: the bug detector finds bugs the *first* time then reports clean; the
# tester fails the *first* time then passes. A per-role counter drives that.
# Call reset_mock() between independent runs.

_mock_calls: dict[str, int] = defaultdict(int)


def reset_mock() -> None:
    """Reset the per-role mock call counters (call before each fresh run)."""
    _mock_calls.clear()


def _mock_complete(role: str, user: str) -> str:
    idx = _mock_calls[role]
    _mock_calls[role] += 1
    return _mock_response(role, idx, user)


def _mock_response(role: str, idx: int, user: str) -> str:
    ctx = truncate(user, 200)

    if role == "planner":
        return (
            "1. Set up project scaffolding and configuration\n"
            "2. Design and create the database schema\n"
            "3. Implement authentication and authorization\n"
            "4. Build the core backend API endpoints\n"
            "5. Build the frontend pages, components, and routing\n"
            "6. Implement search and the primary dashboard views\n"
            "7. Write automated tests and wire up CI"
        )

    if role == "architect":
        return (
            "[mock] High-level architecture: a 3-tier system (client -> API -> "
            "database) with stateless API servers behind a load balancer, a "
            "relational primary store, and a cache for hot reads. Cross-cutting: "
            "auth middleware, structured logging, health checks, and CI/CD.\n"
            f"Derived from requirement: {ctx}"
        )

    if role == "frontend":
        return (
            "[mock] Frontend spec:\n"
            "- Component tree: Layout > (Navbar, ContentOutlet, Footer); feature "
            "components per route.\n"
            "- Routing: convention/file-based routes idiomatic to the requested "
            "framework.\n"
            "- State: server state via a data-fetching layer; local UI state in "
            "components.\n"
            f"Context: {ctx}"
        )

    if role == "backend":
        return (
            "[mock] Backend spec:\n"
            "- API contracts: versioned REST resources with typed request/"
            "response schemas.\n"
            "- Load balancing: stateless instances behind an L7 balancer with "
            "horizontal autoscaling.\n"
            "- Infra: containerized services, DB migrations on deploy, secrets "
            "via environment.\n"
            f"Context: {ctx}"
        )

    if role == "database":
        return (
            "[mock] Database schema:\n"
            "- users(id PK, email UNIQUE, password_hash, role, created_at)\n"
            "- core_entity(id PK, owner_id FK->users, title, body, status, "
            "created_at)\n"
            "- indexes: btree(owner_id), btree(status), full-text index on "
            "(title, body)."
        )

    if role == "coder":
        return (
            "[mock] # Implementation (illustrative, not exhaustive)\n"
            "# Backend: typed request handlers, input validation, parameterized "
            "queries (no string-built SQL).\n"
            "# Frontend: components wired to the API with loading/error states.\n"
            "# Auth: hashed passwords, signed sessions, ownership checks on every "
            "protected route.\n"
            f"# Built against specs derived from: {ctx}"
        )

    if role == "bug_detector":
        if idx == 0:
            return (
                "BUGS_FOUND:\n"
                "1. [HIGH] Possible SQL injection in the search handler — switch "
                "to parameterized queries. (fix: bind params)\n"
                "2. [MED] Missing authorization check on the dashboard endpoint "
                "— verify resource ownership. (fix: add ownership guard)"
            )
        return "CLEAN: no issues found"

    if role == "tester":
        if idx == 0:
            return (
                "FAIL:\n"
                "- test_search_returns_results: expected non-empty result set "
                "for seeded data, got empty.\n"
                "- test_auth_required: a protected route returned 200 with no "
                "session."
            )
        return (
            "PASS:\n"
            "- unit tests: 24 passed\n"
            "- integration stubs: 6 passed\n"
            "- static checks: no blocking issues"
        )

    if role == "reviewer":
        # First line = decision; the rest = notes. The mock approves on the
        # happy path (the loops above already resolved the issues).
        return (
            "APPROVE\n"
            "Architecture is coherent, the previously flagged security issues "
            "were addressed, and tests pass. Minor follow-up: add rate limiting "
            "and request logging before production."
        )

    if role == "aggregator":
        return f"[mock] Compiled deliverable derived from: {ctx}"

    return f"[mock:{role}] {ctx}"


if __name__ == "__main__":
    import asyncio
    import textwrap

    async def demo() -> None:
        s = get_settings()
        provider = s.resolve_provider()
        print(f"use_mock_llm = {s.use_mock_llm} | llm_provider = {s.llm_provider} "
              f"-> resolved: {provider}\n")

        print("Provider + model tiering (role -> provider:model):")
        for role in ["reviewer", "planner", "coder", "frontend"]:
            p, m = model_for_role(role)
            print(f"  {role:10} -> {p}:{m}")

        print("\nplanner mock output:")
        out = await complete("planner", "You are a planner.", "Build a job board")
        print(textwrap.indent(out, "    "))

        print("\nScripted loop behavior (same role, successive calls):")
        reset_mock()
        for i in range(3):
            r = await complete("bug_detector", "scan", "code")
            print(f"  bug_detector call {i} -> {r.splitlines()[0]}")
        for i in range(2):
            r = await complete("tester", "test", "code")
            print(f"  tester       call {i} -> {r.splitlines()[0]}")

    asyncio.run(demo())
