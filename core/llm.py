"""The LLM layer — the single spine every agent calls.

One async function, ``complete(role, system, user)``, hides three concerns from
the agents:

  1. **Mock vs real.** When ``settings.use_mock_llm`` is True it returns
     deterministic, offline, free text. When False it calls real OpenAI.
  2. **Model tiering.** The role decides the model: supervisor/reviewer get the
     strong model, everyone else gets the cheap/fast one.
  3. **Reliability.** Real calls are wrapped in tenacity retry with exponential
     backoff, satisfying the spec's "retry on all OpenAI calls" constraint.

Because every agent goes through here, flipping one config flag switches the
whole system between learning-mode and production-mode.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.tools import truncate


# ---------------------------------------------------------------------------
# Model tiering
# ---------------------------------------------------------------------------

def model_for_role(role: str) -> str:
    """Map an agent role to its model tier (your spec's GPT-4o / mini split)."""
    s = get_settings()
    if role == "supervisor":
        return s.supervisor_model
    if role == "reviewer":
        return s.reviewer_model
    return s.worker_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def complete(role: str, system: str, user: str, temperature: float = 0.3) -> str:
    """Return the model's text for a (system, user) prompt pair.

    In mock mode this is instant and free; in real mode it calls OpenAI through
    the retrying helper below.
    """
    settings = get_settings()
    if settings.use_mock_llm:
        return _mock_complete(role, user)

    if not settings.openai_api_key:
        raise RuntimeError(
            "use_mock_llm is False but OPENAI_API_KEY is not set. "
            "Add it to .env or set use_mock_llm=True for offline mode."
        )
    settings.apply_tracing_env()
    return await _complete_real(role, system, user, temperature)


# ---------------------------------------------------------------------------
# Real OpenAI path (with retry/backoff)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _get_chat_model(model: str, temperature: float):
    """Cache one ChatOpenAI client per (model, temperature). Imported lazily so
    mock mode never needs langchain_openai or an API key at import time."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
async def _complete_real(role: str, system: str, user: str, temperature: float) -> str:
    """Call OpenAI asynchronously. tenacity retries this up to 4 times with
    exponential backoff (1s, 2s, 4s, ... capped at 30s) on transient errors."""
    from langchain_core.messages import HumanMessage, SystemMessage

    chat = _get_chat_model(model_for_role(role), temperature)
    response = await chat.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content


# ---------------------------------------------------------------------------
# Mock path (deterministic, scripted)
# ---------------------------------------------------------------------------
# Some roles vary their answer by call number so we can demonstrate the fix
# loops in Phase 5: the bug detector finds bugs the *first* time then reports
# clean; the tester fails the *first* time then passes. A per-role counter
# drives that. Call reset_mock() between independent runs.

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
        print(f"use_mock_llm = {s.use_mock_llm}\n")

        print("Model tiering (role -> model):")
        for role in ["supervisor", "reviewer", "planner", "coder", "frontend"]:
            print(f"  {role:12} -> {model_for_role(role)}")

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
