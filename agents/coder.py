"""The coder — writes a real, multi-file project to disk (v2.0).

Before v2 the coder returned a single ``code`` string. Now it drives a
tool-using agent (``core/agent_loop.run_file_agent``) that writes actual files
into the run's workspace (``workspaces/{thread_id}/``) using write_file /
read_file / list_files. Downstream agents read those files; the sandbox (v2.1)
runs them.

Two modes (unchanged in spirit):
  * **First run** (no files yet): generate the full project from the specs.
  * **Re-run**: read the relevant files, apply the bug/test/review feedback, and
    write them back. As before, a re-run clears the stale ``bug_report`` /
    ``test_results`` / ``review_decision`` so the verifiers re-check fresh code.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from core.agent_loop import run_file_agent
from core.file_tools import workspace_path
from core.state import AgentState
from core.tools import non_empty

CODER_SYSTEM = (
    "You are a senior full-stack engineer. Write a complete, runnable project to "
    "disk using the file tools (write_file, read_file, list_files). Produce "
    "idiomatic, secure code for the specified stack: parameterized queries, auth "
    "checks, input validation, error handling. Include a backend, a frontend, "
    "tests, Dockerfiles, and a docker-compose.yml so the project can be built and "
    "run. When revising, read the relevant files, fix every issue in the "
    "feedback, and write the files back. Stop when the project is complete."
)


def _gather_feedback(state: AgentState) -> str:
    parts: list[str] = []
    bug_report = state.get("bug_report", "")
    if bug_report and "BUGS_FOUND" in bug_report:
        parts.append(f"Bug report to fix:\n{bug_report}")
    test_results = state.get("test_results", "")
    if test_results.startswith("FAIL"):
        parts.append(f"Failing tests to fix:\n{test_results}")
    if state.get("review_decision") == "REJECT" and state.get("review_notes"):
        parts.append(f"Reviewer rejection notes to address:\n{state['review_notes']}")
    return "\n\n".join(parts)


def _workspace_for(state: AgentState, config) -> str:
    """Reuse the workspace already on state, else derive it from the thread_id."""
    if state.get("workspace_dir"):
        return state["workspace_dir"]
    thread_id = "run"
    if isinstance(config, dict):
        thread_id = (config.get("configurable") or {}).get("thread_id") or "run"
    return str(workspace_path(thread_id))


async def coder(state: AgentState, config=None) -> dict:
    workspace_dir = _workspace_for(state, config)
    is_rerun = bool(state.get("files"))

    base = (
        f"Frontend framework: {state.get('detected_frontend_framework', 'unknown')}\n"
        f"Backend framework: {state.get('detected_backend_framework', 'unknown')}\n\n"
        f"System design:\n{state.get('system_design', '')}\n\n"
        f"Frontend spec:\n{state.get('frontend_spec', '')}\n\n"
        f"Backend spec:\n{state.get('backend_spec', '')}\n\n"
        f"Database schema:\n{state.get('db_schema', '')}\n"
    )
    if is_rerun:
        user = (
            base
            + "\n\nThis is a revision. The current files are listed via list_files. "
            "Read what you need, address the following feedback, and write the "
            "updated files back:\n\n"
            + _gather_feedback(state)
        )
    else:
        user = base + "\n\nWrite the full initial project to disk now."

    files = await run_file_agent("coder", CODER_SYSTEM, user, workspace_dir)
    non_empty(files, "files")

    updates: dict = {
        "workspace_dir": workspace_dir,
        "files": files,
        "code": f"{len(files)} files written to {workspace_dir}",
        "messages": [
            AIMessage(
                content=f"[coder] {'revised' if is_rerun else 'wrote'} {len(files)} files",
                name="coder",
            )
        ],
    }

    if is_rerun:
        # The code changed -> prior analyses & the prior verdict are now stale.
        updates["bug_report"] = ""
        updates["test_results"] = ""
        updates["review_decision"] = None
        updates["review_notes"] = ""

    return updates
