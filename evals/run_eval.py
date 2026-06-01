"""Evaluation harness (v2.6).

Runs each requirement N times through the full pipeline and aggregates metrics
(approval rate, first-pass rate, fix-loop iterations, wall time, tokens), so you
can make measured claims like "across 30 runs, first-pass approval was 73%".

    python -m evals.run_eval --runs 3 --limit 2

Works in mock mode (free, deterministic — good for a smoke test) and in real
mode (set USE_MOCK_LLM=false; metrics become meaningful but cost money + time).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

from core.graph import graph
from core.llm import reset_mock, reset_usage, usage_summary
from core.state import initial_state
from evals.report import render_markdown

HERE = Path(__file__).resolve().parent
REQ_PATH = HERE / "requirements.txt"
REPORT_MD = HERE / "report.md"
REPORT_JSON = HERE / "report.json"


def load_requirements(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


async def run_once(requirement: str, thread_id: str) -> dict:
    reset_mock()
    reset_usage()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    start = time.perf_counter()
    steps = 0
    async for chunk in graph.astream(
        initial_state(requirement), config, stream_mode="updates"
    ):
        steps += len(chunk)
    seconds = time.perf_counter() - start
    state = (await graph.aget_state(config)).values
    usage = usage_summary()
    return {
        "requirement": requirement,
        "decision": state.get("review_decision"),
        "iterations": state.get("iteration_count", 0),
        "bug_iterations": state.get("bug_iteration_count", 0),
        "steps": steps,
        "seconds": round(seconds, 2),
        "input_tokens": usage["input"],
        "cache_read": usage["cache_read"],
        "doc_chars": len(state.get("final_document", "")),
        "files": len(state.get("files", [])),
    }


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"runs": 0}

    def avg(key: str) -> float:
        return round(sum(r[key] for r in records) / n, 2)

    approve = sum(1 for r in records if r["decision"] == "APPROVE")
    first_pass = sum(
        1
        for r in records
        if r["decision"] == "APPROVE" and r["iterations"] == 0 and r["bug_iterations"] == 0
    )
    return {
        "runs": n,
        "approval_rate": round(approve / n, 3),
        "first_pass_rate": round(first_pass / n, 3),
        "avg_iterations": avg("iterations"),
        "avg_bug_iterations": avg("bug_iterations"),
        "avg_steps": avg("steps"),
        "avg_seconds": avg("seconds"),
        "avg_input_tokens": avg("input_tokens"),
        "avg_doc_chars": avg("doc_chars"),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent eval harness.")
    parser.add_argument("--runs", type=int, default=3, help="runs per requirement")
    parser.add_argument("--limit", type=int, default=0, help="max requirements (0=all)")
    args = parser.parse_args()

    requirements = load_requirements(REQ_PATH)
    if args.limit:
        requirements = requirements[: args.limit]

    print(f"Evaluating {len(requirements)} requirement(s) x {args.runs} run(s)\n")
    records: list[dict] = []
    for i, req in enumerate(requirements):
        for run in range(args.runs):
            tid = f"eval-{i}-{run}-{uuid4().hex[:6]}"
            rec = await run_once(req, tid)
            records.append(rec)
            print(
                f"  [{i}.{run}] {rec['decision']:>7} | "
                f"iters={rec['iterations']}+{rec['bug_iterations']} | "
                f"{rec['seconds']}s | {rec['files']} files"
            )

    overall = aggregate(records)
    per_req = {req: aggregate([r for r in records if r["requirement"] == req])
               for req in requirements}

    REPORT_JSON.write_text(
        json.dumps({"overall": overall, "per_requirement": per_req, "records": records},
                   indent=2),
        encoding="utf-8",
    )
    REPORT_MD.write_text(render_markdown(requirements, overall, per_req), encoding="utf-8")

    print("\n=== overall ===")
    for key, value in overall.items():
        print(f"  {key}: {value}")
    print(f"\nWrote {REPORT_MD.name} and {REPORT_JSON.name}")


if __name__ == "__main__":
    asyncio.run(main())
