"""The backend agent — FRAMEWORK-AWARE.

Mirror of the frontend agent:
  1. **Detect** the backend framework from the requirement. Per the spec, when
     nothing matches we default to FastAPI and disclose the assumption.
  2. **Generate** a spec (API contracts, load balancing, infra) idiomatic to
     that framework.

Returns ``detected_backend_framework`` + ``backend_spec``.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from core.llm import complete
from core.state import AgentState
from core.tools import BACKEND_KEYWORDS, detect_framework, non_empty

BACKEND_SYSTEM = (
    "You are a senior backend architect. Produce a backend specification that "
    "is SPECIFIC to the named framework — use its real idioms, never generic "
    "advice. Cover exactly three sections: (1) API contracts/routes, (2) load "
    "balancing & scaling, (3) infrastructure & deployment patterns."
)

# Per spec: unknown backend defaults to FastAPI.
_DEFAULT_BACKEND = "fastapi"


async def backend(state: AgentState) -> dict:
    detected = detect_framework(state["input"], BACKEND_KEYWORDS)

    assumption = ""
    framework = detected
    if detected == "unknown":
        framework = _DEFAULT_BACKEND
        assumption = (
            f"Assumption: no backend framework was named in the requirement, "
            f"so this spec defaults to {framework}.\n\n"
        )

    user = (
        f"Target framework: {framework}\n\n"
        f"Requirement:\n{state['input']}\n\n"
        f"System design:\n{state.get('system_design', '')}\n\n"
        f"Produce the {framework} backend spec (API contracts, load balancing, infra)."
    )
    spec = await complete("backend", BACKEND_SYSTEM, user)
    spec = assumption + spec
    non_empty(spec, "backend_spec")

    return {
        "detected_backend_framework": framework,
        "backend_spec": spec,
        "messages": [
            AIMessage(content=f"[backend] framework={framework}", name="backend")
        ],
    }
