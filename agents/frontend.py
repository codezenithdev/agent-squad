"""The frontend agent — FRAMEWORK-AWARE.

Two steps:
  1. **Detect** the frontend framework from the requirement text using the
     deterministic keyword matcher (no LLM needed). If nothing matches we assume
     React and say so explicitly, so downstream agents still have a concrete
     target instead of "unknown".
  2. **Generate** a spec (component tree, routing, state management) specific to
     that framework — the prompt names the framework so output is never generic.

Returns ``detected_frontend_framework`` + ``frontend_spec``.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from core.llm import complete
from core.state import AgentState
from core.tools import FRONTEND_KEYWORDS, detect_framework, non_empty

FRONTEND_SYSTEM = (
    "You are a senior frontend architect. Produce a frontend specification that "
    "is SPECIFIC to the named framework — use its real conventions, never "
    "generic advice. Cover exactly three sections: (1) component tree, "
    "(2) routing strategy, (3) state management approach."
)

# When detection fails, this is the framework we assume (and disclose).
_DEFAULT_FRONTEND = "react"


async def frontend(state: AgentState) -> dict:
    detected = detect_framework(state["input"], FRONTEND_KEYWORDS)

    assumption = ""
    framework = detected
    if detected == "unknown":
        framework = _DEFAULT_FRONTEND
        assumption = (
            f"Assumption: no frontend framework was named in the requirement, "
            f"so this spec assumes {framework}.\n\n"
        )

    user = (
        f"Target framework: {framework}\n\n"
        f"Requirement:\n{state['input']}\n\n"
        f"System design:\n{state.get('system_design', '')}\n\n"
        f"Produce the {framework} frontend spec (component tree, routing, state)."
    )
    spec = await complete("frontend", FRONTEND_SYSTEM, user)
    spec = assumption + spec
    non_empty(spec, "frontend_spec")

    return {
        "detected_frontend_framework": framework,
        "frontend_spec": spec,
        "messages": [
            AIMessage(content=f"[frontend] framework={framework}", name="frontend")
        ],
    }
