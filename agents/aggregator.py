"""The aggregator — the final agent. Compiles the whole run into one document.

Unlike the other workers, most of this agent is *deterministic assembly*: it
stitches the state fields that previous agents produced into a structured
10-section markdown deliverable. It makes a single LLM call only for the
executive summary (nice prose), then templates the rest. Producing the final
artifact is mostly a formatting job, not a reasoning job, so we don't pay for
what we don't need.

Sections (per spec): 1 Executive summary, 2 Detected stack, 3 Architecture,
4 Frontend spec, 5 Backend spec, 6 Database schema, 7 Implementation,
8 Security audit, 9 Test plan, 10 Deployment notes.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from core.llm import complete
from core.state import AgentState
from core.tools import non_empty

AGGREGATOR_SYSTEM = (
    "You are a technical writer. Write a concise executive summary (3-5 "
    "sentences) describing the delivered system and its stack."
)

# Framework-aware deployment hints for section 10.
_FE_DEPLOY = {
    "nextjs": "Deploy to Vercel or a Node container; use SSR/ISR where useful.",
    "nuxt": "Deploy via Nitro to a Node/edge host or container.",
    "sveltekit": "Deploy with the appropriate adapter (Node/edge/static).",
    "remix": "Deploy to a Node server or edge runtime.",
    "astro": "Build static/SSR output and serve behind a CDN.",
    "angular": "Build to static assets and serve via CDN; SSR via Angular Universal if needed.",
    "vue": "Build to static assets and serve behind a CDN.",
    "react": "Build to static assets and serve behind a CDN.",
}
_BE_DEPLOY = {
    "fastapi": "Serve with uvicorn/gunicorn workers behind a reverse proxy; containerize and autoscale.",
    "express": "Run under a process manager or container behind a load balancer.",
    "nestjs": "Containerize and run behind an L7 load balancer.",
    "django": "Serve with gunicorn/uWSGI behind nginx; run migrations on deploy.",
    "flask": "Serve with gunicorn behind a reverse proxy.",
    "rails": "Serve with Puma behind a reverse proxy; run migrations on deploy.",
    "spring": "Package as a container/JAR and run behind a load balancer.",
    "gin": "Build a static binary, containerize, and run behind a load balancer.",
}


def _deployment_notes(fe: str, be: str) -> str:
    return (
        f"- **Frontend ({fe}):** "
        f"{_FE_DEPLOY.get(fe, 'Build artifacts and serve behind a CDN.')}\n"
        f"- **Backend ({be}):** "
        f"{_BE_DEPLOY.get(be, 'Containerize and run behind an L7 load balancer with autoscaling.')}\n"
        "- Run database migrations on deploy; manage secrets via environment variables."
    )


def _or_note(value: str, note: str) -> str:
    """Return the stripped value, or an explanatory note if it's empty.

    The verify/fix loop can end with ``bug_report``/``test_results`` cleared (the
    coder wipes them on its last revision, then the circuit breaker routes past
    the verifiers). Rather than print a blank line, we explain why.
    """
    value = (value or "").strip()
    return value if value else f"_{note}_"


def _assemble(state: AgentState, summary: str) -> str:
    fe = state.get("detected_frontend_framework", "unknown")
    be = state.get("detected_backend_framework", "unknown")
    tasks = state.get("task_graph", [])
    task_lines = "\n".join(f"{i}. {t}" for i, t in enumerate(tasks, 1))

    return "\n".join(
        [
            "# Project Design & Implementation Document\n",
            "## 1. Executive Summary\n\n" + summary.strip() + "\n",
            (
                "## 2. Detected Stack\n\n"
                f"- **Frontend framework:** {fe}\n"
                f"- **Backend framework:** {be}\n"
            ),
            (
                "## 3. Architecture Overview\n\n"
                + state.get("system_design", "").strip()
                + ("\n\n**Planned work:**\n\n" + task_lines if tasks else "")
                + "\n"
            ),
            "## 4. Frontend Specification\n\n" + state.get("frontend_spec", "").strip() + "\n",
            "## 5. Backend Specification\n\n" + state.get("backend_spec", "").strip() + "\n",
            "## 6. Database Schema\n\n" + state.get("db_schema", "").strip() + "\n",
            "## 7. Implementation\n\n```\n" + state.get("code", "").strip() + "\n```\n",
            (
                "## 8. Security Audit Summary\n\n"
                f"- Bug-fix iterations run: {state.get('bug_iteration_count', 0)}\n"
                "- Final scan: "
                + _or_note(
                    state.get("bug_report", ""),
                    "no final scan recorded — code was revised to the "
                    "circuit-breaker limit; see review notes",
                )
                + "\n"
            ),
            (
                "## 9. Test Plan\n\n"
                f"- Test-fix iterations run: {state.get('iteration_count', 0)}\n"
                "- Final result: "
                + _or_note(
                    state.get("test_results", ""),
                    "no test run recorded — code was revised to the "
                    "circuit-breaker limit; see review notes",
                )
                + "\n"
            ),
            "## 10. Deployment Notes\n\n" + _deployment_notes(fe, be) + "\n",
            (
                "---\n\n"
                f"_Final review verdict: **{state.get('review_decision')}**_\n\n"
                f"{state.get('review_notes', '').strip()}\n"
            ),
        ]
    )


async def aggregator(state: AgentState) -> dict:
    summary = await complete(
        "aggregator",
        AGGREGATOR_SYSTEM,
        f"Requirement:\n{state['input']}\n\n"
        f"System design:\n{state.get('system_design', '')}\n\n"
        "Write the executive summary.",
    )
    document = _assemble(state, summary)
    non_empty(document, "final_document")

    return {
        "final_document": document,
        "messages": [
            AIMessage(content="[aggregator] final document compiled", name="aggregator")
        ],
    }
