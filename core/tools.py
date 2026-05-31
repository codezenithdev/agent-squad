"""Shared, dependency-free helpers used across agents.

The spec lists a ``tools.py`` module. None of the agents in this system actually
*call* an external tool (no web search, no code execution), so rather than
inventing LangChain ``@tool`` objects that nobody invokes, this module holds the
small deterministic utilities the agents genuinely share:

  * framework detection (pure keyword matching — no LLM needed),
  * an output validator that enforces the "never return empty" constraint,
  * a text truncation helper for echoing context.

This is also where you would register real LangChain tools later if, say, the
coder gained the ability to actually run code.
"""
from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------
# Each entry is (canonical_name, [search phrases]). ORDER MATTERS: more specific
# frameworks are listed before the generic base they build on, so "Next.js"
# resolves to 'nextjs' rather than 'react', and "SvelteKit" beats "svelte".

FRONTEND_KEYWORDS: list[tuple[str, list[str]]] = [
    ("nextjs", ["next.js", "nextjs", "next js"]),
    ("nuxt", ["nuxt"]),                       # Nuxt is Vue-based -> before 'vue'
    ("remix", ["remix"]),                     # Remix is React-based -> before 'react'
    ("astro", ["astro"]),
    ("sveltekit", ["sveltekit", "svelte kit"]),  # before 'svelte'
    ("svelte", ["svelte"]),
    ("angular", ["angular"]),
    ("vue", ["vue.js", "vuejs", "vue"]),
    ("react", ["react.js", "reactjs", "react"]),
]

BACKEND_KEYWORDS: list[tuple[str, list[str]]] = [
    ("fastapi", ["fastapi", "fast api"]),
    ("nestjs", ["nestjs", "nest.js", "nest js"]),  # NestJS uses Express -> before 'express'
    ("express", ["express.js", "expressjs", "express"]),
    ("django", ["django"]),
    ("flask", ["flask"]),
    ("rails", ["ruby on rails", "rails"]),
    ("spring", ["spring boot", "spring"]),
    ("gin", ["gin"]),
    ("fiber", ["fiber"]),
    ("hono", ["hono"]),
    ("laravel", ["laravel"]),
]


def detect_framework(
    text: str,
    keywords: list[tuple[str, list[str]]],
    default: str = "unknown",
) -> str:
    """Return the canonical name of the first framework whose keyword appears in
    ``text`` (whole-word, case-insensitive), else ``default``.

    Whole-word matching (via ``\\b`` boundaries) matters: a naive substring
    check would match 'gin' inside 'imagine'. We escape each phrase so dots in
    'next.js' are treated literally.
    """
    for canonical, phrases in keywords:
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE):
                return canonical
    return default


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def non_empty(value, field_name: str):
    """Enforce the spec constraint that an agent must never return empty output.

    Raises ``ValueError`` if ``value`` is None, blank string, or an empty
    collection. Returns ``value`` unchanged otherwise, so it can be used inline:
        return {"code": non_empty(code, "code")}
    """
    is_blank_str = isinstance(value, str) and not value.strip()
    is_empty_collection = isinstance(value, (list, dict, tuple, set)) and len(value) == 0
    if value is None or is_blank_str or is_empty_collection:
        raise ValueError(f"Agent produced empty output for '{field_name}'")
    return value


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def truncate(text: str, limit: int = 240) -> str:
    """Trim text to ``limit`` chars with an ellipsis — used when echoing the
    requirement back into a generated spec so output stays readable."""
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def join_lines(items: Iterable[str]) -> str:
    """Join non-empty, stripped lines with newlines."""
    return "\n".join(line.strip() for line in items if line and line.strip())


if __name__ == "__main__":
    sample = (
        "Design a scalable job board with Next.js frontend, FastAPI backend, "
        "PostgreSQL, user auth, job listings with search, and an employer "
        "dashboard"
    )
    print("Input:", sample, "\n")
    print("frontend ->", detect_framework(sample, FRONTEND_KEYWORDS))
    print("backend  ->", detect_framework(sample, BACKEND_KEYWORDS))
    print("unknown  ->", detect_framework("a plain CLI tool", FRONTEND_KEYWORDS))

    print("\nnon_empty:")
    print("  ok   ->", non_empty("some output", "code"))
    try:
        non_empty("   ", "code")
    except ValueError as e:
        print("  blank ->", e)
