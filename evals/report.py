"""Render eval metrics as markdown."""
from __future__ import annotations


def render_markdown(requirements: list[str], overall: dict, per_requirement: dict) -> str:
    lines = ["# Evaluation report", "", "## Overall", ""]
    for key, value in overall.items():
        lines.append(f"- **{key}**: {value}")
    lines += ["", "## Per requirement", ""]
    lines.append("| requirement | runs | approval | first-pass | avg iters (test+bug) | avg s |")
    lines.append("|---|---|---|---|---|---|")
    for req in requirements:
        m = per_requirement.get(req, {})
        short = (req[:58] + "…") if len(req) > 58 else req
        lines.append(
            f"| {short} | {m.get('runs', 0)} | {m.get('approval_rate', 0)} | "
            f"{m.get('first_pass_rate', 0)} | "
            f"{m.get('avg_iterations', 0)}+{m.get('avg_bug_iterations', 0)} | "
            f"{m.get('avg_seconds', 0)} |"
        )
    return "\n".join(lines) + "\n"
