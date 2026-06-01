"""Eval harness aggregation + requirements loading (v2.6)."""
from evals.run_eval import REQ_PATH, aggregate, load_requirements


def test_aggregate_metrics():
    records = [
        {"decision": "APPROVE", "iterations": 0, "bug_iterations": 0,
         "steps": 30, "seconds": 1.0, "input_tokens": 0, "doc_chars": 100},
        {"decision": "REJECT", "iterations": 3, "bug_iterations": 3,
         "steps": 41, "seconds": 2.0, "input_tokens": 0, "doc_chars": 200},
    ]
    s = aggregate(records)
    assert s["runs"] == 2
    assert s["approval_rate"] == 0.5
    assert s["first_pass_rate"] == 0.5  # only the APPROVE+0-iters run counts
    assert s["avg_iterations"] == 1.5
    assert s["avg_doc_chars"] == 150.0


def test_aggregate_empty():
    assert aggregate([]) == {"runs": 0}


def test_requirements_file_parses():
    reqs = load_requirements(REQ_PATH)
    assert len(reqs) >= 5
    assert all(not r.startswith("#") for r in reqs)
    assert any("Next.js" in r for r in reqs)
