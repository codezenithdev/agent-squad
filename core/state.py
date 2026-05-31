"""The shared State — the single source of truth every agent reads and writes.

In LangGraph the *State* is the data that flows through the graph. Each node
(agent) receives the whole state, does its work, and returns a dict of **partial
updates**. LangGraph merges those updates into the state using a per-field rule
called a *reducer*.

There are two kinds of fields, and understanding the difference is the single
most important LangGraph concept:

  * **Plain fields** (str, list, int, ...) -> *last write wins* (overwrite).
    If the planner sets ``task_graph`` and later the coder returns a new
    ``task_graph``, the new value replaces the old one.

  * **Annotated[list, add_messages]** -> the reducer **APPENDS** instead of
    overwriting, so the message log grows over the whole run instead of each
    agent clobbering the previous agent's messages.

We demonstrate exactly this behavior in the ``__main__`` block below
(``python core/state.py``).
"""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, field_validator


# Every place the supervisor is allowed to send control. 'FINISH' is the
# sentinel that ends the run (it maps to LangGraph's END in the graph).
VALID_ROUTES: tuple[str, ...] = (
    "planner",
    "architect",
    "frontend",
    "backend",
    "database",
    "coder",
    "bug_detector",
    "tester",
    "reviewer",
    "aggregator",
    "FINISH",
)


class AgentState(TypedDict, total=False):
    """The shared whiteboard. Every field starts empty and gets filled in by the
    agent responsible for it as the pipeline progresses.

    ``total=False`` means a dict doesn't have to contain every key to satisfy
    the type — which matches how nodes return *partial* updates.
    """

    # The original user requirement.
    input: str

    # Planner output: an ordered list of build tasks.
    task_graph: list[str]

    # Architect output: the high-level system design.
    system_design: str

    # Detected tech stack (lowercase, no spaces, e.g. 'nextjs', 'fastapi').
    detected_frontend_framework: str
    detected_backend_framework: str

    # Per-layer specs produced by the framework-aware agents.
    frontend_spec: str
    backend_spec: str
    db_schema: str

    # Coder output.
    # v2: the coder now writes a real multi-file project to disk. `workspace_dir`
    # is the absolute path to that project, `files` is the relative-path manifest.
    # `code` is kept as an optional short human summary (no longer the source of
    # truth — the files on disk are).
    workspace_dir: str
    files: list[str]
    code: str

    # Bug detector output: starts with 'BUGS_FOUND:' or 'CLEAN:'.
    bug_report: str

    # Tester output: starts with 'FAIL:' or 'PASS:'.
    test_results: str

    # Reviewer output.
    review_decision: Optional[str]  # 'APPROVE' | 'REJECT' | None (not run yet)
    review_notes: str

    # Aggregator output: the final markdown deliverable.
    final_document: str

    # --- Loop counters (circuit breakers) ---
    iteration_count: int       # Tester  -> Coder cycles (max == max_iterations)
    bug_iteration_count: int   # BugDetector -> Coder cycles (max == max_iterations)

    # --- Routing ---
    next: str                  # the supervisor's chosen next node

    # --- Conversation log: APPENDS (does not overwrite) ---
    messages: Annotated[list, add_messages]


class RouteDecision(BaseModel):
    """Structured output for the supervisor.

    The supervisor never returns free text that we then parse — it returns this
    validated model. The validator guarantees ``next`` is always a real,
    routable target, so a typo can never silently break routing.
    """

    next: str

    @field_validator("next")
    @classmethod
    def must_be_valid_route(cls, v: str) -> str:
        if v not in VALID_ROUTES:
            raise ValueError(
                f"'{v}' is not a valid route. Must be one of: {', '.join(VALID_ROUTES)}"
            )
        return v


def initial_state(user_input: str) -> AgentState:
    """Build a fully-initialized state so no field is ever missing at runtime.

    Starting every field at its empty value is what lets the supervisor use
    simple ``if x == ''`` checks to decide what still needs doing.
    """
    return AgentState(
        input=user_input,
        task_graph=[],
        system_design="",
        detected_frontend_framework="",
        detected_backend_framework="",
        frontend_spec="",
        backend_spec="",
        db_schema="",
        workspace_dir="",
        files=[],
        code="",
        bug_report="",
        test_results="",
        review_decision=None,
        review_notes="",
        final_document="",
        iteration_count=0,
        bug_iteration_count=0,
        next="",
        messages=[],
    )


if __name__ == "__main__":
    # `python core/state.py` walks through the three core ideas of this phase.
    from langchain_core.messages import AIMessage, HumanMessage

    print("1) A fresh initial state (every field starts empty):")
    s = initial_state("Build a job board")
    for k, v in s.items():
        print(f"   {k:28} = {v!r}")

    print("\n2) The add_messages reducer — APPEND vs overwrite:")
    existing = [HumanMessage(content="planner ran")]
    incoming = [AIMessage(content="architect ran")]
    merged = add_messages(existing, incoming)
    print(f"   existing : {[m.content for m in existing]}")
    print(f"   incoming : {[m.content for m in incoming]}")
    print(f"   merged   : {[m.content for m in merged]}   <- appended, not replaced")

    print("\n3) RouteDecision validation (structured output, not free text):")
    print(f"   valid   -> {RouteDecision(next='planner')}")
    try:
        RouteDecision(next="not_a_real_node")
    except Exception as e:
        print(f"   invalid -> rejected with {type(e).__name__} (typo can't slip through)")
