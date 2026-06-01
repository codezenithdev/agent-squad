# Multi-Agent Architect

A **multi-agent code-generation platform** built with
[LangGraph](https://langchain-ai.github.io/langgraph/) using the **supervisor
pattern**. Give it a software requirement in plain English and eleven
specialized agents collaborate to plan, design, **write a real multi-file
project to disk, run it in a sandbox, self-heal against real failures**, review
it, commit it to a git branch, and compile a full design document.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.6%2B-green)
![Providers](https://img.shields.io/badge/LLM-OpenAI%20or%20Claude-orange)

> **Runs free, offline, out of the box.** A built-in mock mode returns
> deterministic responses, so the whole pipeline runs with **no API key and zero
> cost**. Flip one flag for real models (OpenAI **or** Anthropic/Claude).

---

## What it does

- **Writes real code.** The coder uses file tools to write an actual multi-file
  project (backend, frontend, tests, Dockerfiles, compose) into
  `workspaces/{thread_id}/` — not a string, real files on disk.
- **Runs it for real.** A Docker sandbox executes the generated project so the
  tester runs real `pytest` and the bug detector runs real `bandit`/`ruff` —
  the self-healing loop reacts to **actual** failures, not guesses.
- **Two providers, with caching.** Works with OpenAI or Anthropic/Claude
  (`LLM_PROVIDER`); on Claude it uses **prompt caching** (~90% off repeated
  prefixes) and **live web search** to pull current framework docs.
- **Streams live.** `POST /run/stream` (SSE) and a **Streamlit UI** show each
  agent's step as it happens.
- **Commits the result.** The generated workspace becomes a git branch with a
  generated PR description.
- **Measures itself.** An eval harness runs requirements N times and reports
  approval rate, first-pass rate, fix-loop iterations, time, and tokens.

---

## Architecture

One **supervisor** sits at the center. After every agent finishes, control
returns to the supervisor, which inspects the shared state and decides who runs
next — until the work is done. Workers always edge back to the supervisor; the
**fix-loops** are the supervisor re-routing to the `coder`:

- `bug_detector` reports `BUGS_FOUND` → back to `coder`
- `tester` reports `FAIL` → back to `coder`
- `reviewer` returns `REJECT` → back to `coder`

Each loop is bounded by a **circuit breaker** (`MAX_ITERATIONS`, default 3), and
a guaranteed terminal path ensures the run always ends with a document — never a
dead-end.

### The 11 agents

| Agent | Tier | Responsibility |
|-------|------|----------------|
| **supervisor** | — (pure Python) | Deterministic router; returns a validated `RouteDecision` |
| **planner** | worker | Breaks the requirement into an ordered task list |
| **architect** | worker | High-level system design |
| **frontend** | worker | Detects the frontend framework; writes a framework-specific spec (web search on Claude) |
| **backend** | worker | Detects the backend framework; writes a framework-specific spec (web search on Claude) |
| **database** | worker | Relational schema (tables, keys, indexes) |
| **coder** | worker | **Writes the real multi-file project** via file tools |
| **bug_detector** | worker | Runs **bandit + ruff** in Docker → `BUGS_FOUND`/`CLEAN` (LLM fallback if no Docker) |
| **tester** | worker | Runs real **pytest** in Docker → `FAIL`/`PASS` (LLM fallback if no Docker) |
| **reviewer** | **strong** | Holistic review → `APPROVE`/`REJECT` |
| **aggregator** | worker | Compiles the 10-section document + commits the workspace to a git branch |

Model **tiers** map per provider: OpenAI → `gpt-4o` (strong) / `gpt-4o-mini`
(worker); Anthropic → `claude-sonnet-4-5` (strong) / `claude-haiku-4-5`
(worker). All overridable via env.

---

## Project structure

```
multi-agent-architect/
├── agents/            # the 11 agents (supervisor + 10 workers)
├── core/
│   ├── state.py       # shared AgentState (TypedDict) + RouteDecision
│   ├── llm.py         # complete(): provider-agnostic, prompt caching, retry, usage
│   ├── tools.py       # framework detection + helpers (no LLM)
│   ├── file_tools.py  # workspace-scoped write/read/list tools for the coder
│   ├── agent_loop.py  # tool-calling loop that writes the project to disk
│   ├── sandbox.py     # DockerSandbox: run generated code in throwaway containers
│   ├── git_ops.py     # commit the generated workspace to a branch
│   └── graph.py       # the StateGraph wiring + MemorySaver checkpointer
├── api/main.py        # FastAPI: POST /run, POST /run/stream (SSE), GET /status
├── ui/app.py          # Streamlit live dashboard
├── evals/             # eval harness: run_eval.py, report.py, requirements.txt
├── workspaces/        # generated projects (git-ignored)
├── config.py          # pydantic-settings config (reads .env)
├── test_run.py        # smoke test -> writes output.md
├── requirements.txt / requirements-dev.txt
└── .env.example
```

---

## Quick start (mock — free, no key)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate   |   macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python test_run.py        # full pipeline in mock mode -> writes output.md
```

`test_run.py` prints each agent step as it runs. In mock mode the coder writes a
small stub project to `workspaces/smoke-test/`.

### Run the API

```bash
uvicorn api.main:app --reload --port 8000      # docs at /docs
```

- `POST /run` — `{ "requirement": "...", "thread_id": "job-1" }` → final document + detected stack
- `POST /run/stream` — same body, **Server-Sent Events**: one event per agent step, then a `done` event
- `GET /status/{thread_id}` — persisted state snapshot for a run

### Run the live UI

```bash
uvicorn api.main:app --port 8000     # terminal 1
streamlit run ui/app.py               # terminal 2 -> opens in your browser
```

### Run the eval harness

```bash
python -m evals.run_eval --runs 3 --limit 2   # writes evals/report.md + report.json
```

---

## Going real (OpenAI or Claude)

```bash
cp .env.example .env     # then edit:
#   USE_MOCK_LLM=false
#   LLM_PROVIDER=auto            # auto (prefer OpenAI) | openai | anthropic
#   OPENAI_API_KEY=sk-...        # and/or ANTHROPIC_API_KEY=sk-ant-...
python test_run.py
```

**For real execution (the sandbox), start Docker Desktop.** The tester/bug
detector then run real `pytest`/`bandit` in containers; without Docker they fall
back to an LLM assessment.

> ⏱️ **Real runs are slow and cost money — by nature.** A single real run writes
> a full multi-file project via many sequential tool calls and runs the fix-loop
> up to 3×; with the Docker sandbox doing `pip install` + tests each pass, **one
> run can take ~8 minutes** and the reviewer often `REJECT`s a large app within
> the iteration budget. That's expected. Use **mock** for fast iteration, run
> **small real batches** (`--runs 1 --limit 1`) to spot-check, and lower
> `MAX_ITERATIONS` if you want faster (less thorough) runs. On Claude, prompt
> caching meaningfully cuts the cost of repeated context.

---

## Configuration

All settings load from environment / `.env` via `config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `USE_MOCK_LLM` | `true` | `true` = free offline mocks; `false` = real models |
| `LLM_PROVIDER` | `auto` | `auto` (prefer OpenAI) · `openai` · `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | the chosen provider's key |
| `OPENAI_STRONG_MODEL` / `OPENAI_WORKER_MODEL` | `gpt-4o` / `gpt-4o-mini` | OpenAI tiers |
| `ANTHROPIC_STRONG_MODEL` / `ANTHROPIC_WORKER_MODEL` | `claude-sonnet-4-5` / `claude-haiku-4-5` | Anthropic tiers |
| `MAX_ITERATIONS` | `3` | circuit-breaker limit for fix-loops |
| `ENABLE_WEB_SEARCH` | `true` | Claude web search (Anthropic only; no-op on OpenAI) |
| `ENABLE_GIT` | `true` | commit the generated workspace to a branch |
| `SANDBOX_PYTHON_IMAGE` / `SANDBOX_NODE_IMAGE` | `python:3.11-slim` / `node:20-slim` | sandbox images |
| `SANDBOX_TIMEOUT` / `SANDBOX_MEMORY` / `SANDBOX_CPUS` | `300` / `1g` / `2` | per-container limits |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` | `false` / — | LangSmith tracing |

---

## How it works (the concepts)

A LangGraph app is a **State** (shared data) → **Nodes** (functions that update
state) → **Edges** (who runs next) → compile → invoke.

- **Shared state** (`core/state.py`): a `TypedDict` every agent reads and writes;
  each returns only a *partial* update. `messages` uses the `add_messages`
  reducer (appends); everything else is last-write-wins.
- **Deterministic supervisor** (`agents/supervisor.py`): the routing rules are
  predicates on state (`if system_design == "" → architect`), evaluated in plain
  Python and returned as a validated Pydantic `RouteDecision` — no LLM, no
  free-text parsing.
- **Tool-using coder** (`core/agent_loop.py` + `core/file_tools.py`): a
  `bind_tools` loop lets the model call `write_file`/`read_file`/`list_files`
  (scoped to the run's workspace) until the project is written.
- **Real execution** (`core/sandbox.py`): generated code runs **only** inside
  throwaway Docker containers (memory/cpu/timeout limits), so the tester/bug
  detector report real results.
- **Self-healing loops**: on a re-run the coder clears the stale
  `bug_report`/`test_results`/`review_decision`, forcing a fresh
  audit → test → review pass so loops converge instead of spinning on stale data.
- **Provider-agnostic LLM** (`core/llm.py`): one `complete()` helper hides the
  mock/real switch, OpenAI-vs-Anthropic selection (`init_chat_model`), model
  tiering, `tenacity` retry, **Anthropic prompt caching** (`cache_control`), and
  optional **Claude web search**.
- **Git + eval**: the aggregator commits the workspace to a branch with a
  generated PR description; the eval harness aggregates metrics over many runs.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q        # fast, free, offline; Docker/real tests are gated/skipped
```

## Tech stack

LangGraph · LangChain · langchain-openai · langchain-anthropic ·
Pydantic / pydantic-settings · FastAPI · Uvicorn · Streamlit · Tenacity ·
Docker · LangSmith (optional).

## License

**TBD** — no license has been chosen yet; a public repo without one is not
legally reusable by others. Adding one (e.g. MIT) is a one-file drop-in.
