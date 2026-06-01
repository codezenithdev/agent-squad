"""Tool-calling agent loop (v2.0).

The text-only ``complete()`` in core/llm.py can't use tools. This module adds a
small, version-stable tool loop: bind the workspace file tools to the model, let
it call them until it stops, and return the manifest of files it wrote.

We use a manual bind_tools loop (rather than a prebuilt agent) so the control
flow is explicit and identical across OpenAI and Anthropic.

Mock mode writes a small deterministic stub project so the whole pipeline runs
offline and free — no tool-calling model required.
"""
from __future__ import annotations

from config import get_settings
from core.file_tools import list_workspace_files, make_file_tools
from core.llm import _get_chat_model, model_for_role, record_usage, web_search_tool


async def run_file_agent(
    role: str,
    system: str,
    user: str,
    workspace_dir: str,
    max_steps: int = 40,
) -> list[str]:
    """Drive a tool-using agent that writes files into ``workspace_dir``.
    Returns the relative-path manifest of everything written."""
    settings = get_settings()

    if settings.use_mock_llm:
        _write_mock_project(workspace_dir)
        return list_workspace_files(workspace_dir)

    settings.apply_provider_env()
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    tools = make_file_tools(workspace_dir)
    tools_by_name = {t.name: t for t in tools}
    provider, model = model_for_role(role)
    # File tools are client-side (we execute them in the loop). On Anthropic we
    # also bind the server-side web search tool so the coder can look up current
    # framework APIs; Anthropic resolves those searches itself (no loop entry).
    bind_list = list(tools)
    if provider == "anthropic" and settings.enable_web_search:
        bind_list.append(web_search_tool())
    chat = _get_chat_model(provider, model, 0.2).bind_tools(bind_list)

    # Anthropic: cache the system + initial instruction so each subsequent loop
    # step (which re-sends the whole growing conversation) only pays for the
    # cached prefix. This is the biggest caching win in the system.
    if provider == "anthropic":
        ephemeral = {"cache_control": {"type": "ephemeral"}}
        messages = [
            SystemMessage(content=[{"type": "text", "text": system, **ephemeral}]),
            HumanMessage(content=[{"type": "text", "text": user, **ephemeral}]),
        ]
    else:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]

    for _ in range(max_steps):
        ai = await chat.ainvoke(messages)
        record_usage(ai)
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            target = tools_by_name.get(tc["name"])
            try:
                result = (
                    target.invoke(tc["args"])
                    if target
                    else f"ERROR: unknown tool '{tc['name']}'"
                )
            except Exception as e:  # surface tool errors back to the model
                result = f"ERROR: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return list_workspace_files(workspace_dir)


def _write_mock_project(workspace_dir: str) -> None:
    """Write a tiny but believable multi-file FastAPI + Next.js project so the
    mock pipeline produces a real on-disk workspace (free, deterministic)."""
    from core.file_tools import make_file_tools

    write = {t.name: t for t in make_file_tools(workspace_dir)}["write_file"]

    files = {
        "README.md": "# Generated project (mock)\n\nFastAPI backend + Next.js frontend.\n",
        "docker-compose.yml": (
            "services:\n"
            "  backend:\n    build: ./backend\n    ports: ['8000:8000']\n"
            "  frontend:\n    build: ./frontend\n    ports: ['3000:3000']\n"
            "  db:\n    image: postgres:16\n    environment:\n      POSTGRES_PASSWORD: dev\n"
        ),
        "backend/requirements.txt": "fastapi\nuvicorn[standard]\npytest\nhttpx\n",
        "backend/main.py": (
            "from fastapi import FastAPI\n\n"
            "app = FastAPI()\n\n"
            "@app.get('/health')\n"
            "def health():\n    return {'status': 'ok'}\n"
        ),
        "backend/tests/test_health.py": (
            "from fastapi.testclient import TestClient\n"
            "from main import app\n\n"
            "def test_health():\n"
            "    assert TestClient(app).get('/health').json() == {'status': 'ok'}\n"
        ),
        "frontend/package.json": (
            '{\n  "name": "frontend",\n  "scripts": {"build": "next build"},\n'
            '  "dependencies": {"next": "latest", "react": "latest"}\n}\n'
        ),
        "frontend/app/page.tsx": (
            "export default function Home() {\n"
            "  return <main>Job Board</main>;\n}\n"
        ),
    }
    for path, content in files.items():
        write.invoke({"path": path, "content": content})
