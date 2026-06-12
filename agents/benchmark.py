"""Benchmark harness for end-to-end lattice design runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constraints import parse_design_constraints
from .session import assess_lattice_results, parse_lattice_results
from .team import run_design

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK_FILE = PROJECT_ROOT / "benchmarks" / "tasks.json"


@dataclass(slots=True)
class BenchmarkTask:
    id: str
    prompt: str
    expected_cell_type: str | None = None
    require_stable: bool = True
    minimum_score: int = 60
    max_emittance: float | None = None
    max_messages: int = 30
    # Infeasibility / refusal fields (stage 10)
    expected_outcome: str = "design"          # "design" or "refusal"
    required_binding_constraint: str | None = None  # e.g. "target_emittance"


def load_tasks(task_file: Path | None = None) -> list[BenchmarkTask]:
    """Load benchmark tasks from JSON."""
    path = task_file or DEFAULT_TASK_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkTask(**item) for item in raw]


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _latest_result(messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue
        parsed = parse_lattice_results(content)
        if parsed:
            return parsed
    return None


def _latest_message_text(messages: list[Any], source: str) -> str | None:
    for message in reversed(messages):
        if getattr(message, "source", None) == source and isinstance(getattr(message, "content", None), str):
            return message.content
    return None


def evaluate_benchmark_run(
    task: BenchmarkTask,
    *,
    messages: list[Any],
) -> dict[str, Any]:
    """Apply deterministic checks to one benchmark run."""
    result = _latest_result(messages)
    inferred_constraints = parse_design_constraints(task.prompt)
    assessment = assess_lattice_results(result, constraints=inferred_constraints)
    issues = list(assessment.get("issues", []))
    blocking_issues: list[str] = []

    if result is None:
        blocking_issues.append("No structured lattice result was produced")

    stable = _normalize_bool((result or {}).get("stable"))
    if task.require_stable and stable is not True:
        blocking_issues.append("Benchmark requires a stable lattice")

    emittance_limit = task.max_emittance
    if emittance_limit is None:
        emittance_limit = inferred_constraints.get("target_emittance")

    emittance = (result or {}).get("emittance")
    if emittance_limit is not None and emittance is not None:
        try:
            if float(emittance) > emittance_limit:
                blocking_issues.append(
                    f"Emittance {float(emittance):.3e} exceeds limit {emittance_limit:.3e}"
                )
        except (TypeError, ValueError):
            blocking_issues.append("Emittance could not be parsed as a float")

    expected_cell_type = task.expected_cell_type or inferred_constraints.get("cell_type")
    # Check planner or codewriter output for cell type (not the defunct LatticeArchitect)
    from .team import PLANNER, CODEWRITER
    planner_text = _latest_message_text(messages, PLANNER) or ""
    codewriter_text = _latest_message_text(messages, CODEWRITER) or ""
    agent_text = planner_text + " " + codewriter_text
    if expected_cell_type and expected_cell_type.upper() not in agent_text.upper():
        blocking_issues.append(
            f"Agent output does not clearly indicate expected cell type {expected_cell_type}"
        )

    approved = any(
        "DESIGN_APPROVED" in getattr(message, "content", "")
        for message in messages
        if isinstance(getattr(message, "content", None), str)
    )

    # ── Refusal / infeasibility path ──────────────────────────────────────────
    if task.expected_outcome == "refusal":
        is_refusal = assessment.get("status") == "infeasible-refusal"
        binding_ok = True
        if task.required_binding_constraint and is_refusal:
            reported_binding = str((result or {}).get("binding_constraint", "") or "")
            required = task.required_binding_constraint.lower()
            reported = reported_binding.lower()
            # Accept substring match: "target_emittance" matches "emittance", and vice versa
            binding_ok = (required in reported) or (reported in required) or any(
                word in reported for word in required.replace("target_", "").split("_")
                if len(word) > 4
            )
        passed = is_refusal and binding_ok
        if not is_refusal:
            blocking_issues.append(
                "Expected an INFEASIBLE_DESIGN refusal but got a lattice design"
            )
        elif not binding_ok:
            blocking_issues.append(
                f"Refusal did not identify the required binding constraint "
                f"'{task.required_binding_constraint}' "
                f"(got: '{(result or {}).get('binding_constraint', 'None')}')"
            )
    else:
        passed = (
            assessment.get("score", 0) >= task.minimum_score
            and (not task.require_stable or stable is True)
            and not blocking_issues
        )

    return {
        "task_id": task.id,
        "passed": passed,
        "approved": approved,
        "score": assessment.get("score", 0),
        "status": assessment.get("status", "missing"),
        "issues": issues + blocking_issues,
        "blocking_issues": blocking_issues,
        "constraints": inferred_constraints,
        "result": result,
        "message_count": len(messages),
    }


async def run_benchmarks(
    *,
    task_file: Path | None = None,
    use_docker: bool = True,
    default_max_messages: int = 30,
) -> dict[str, Any]:
    """Run the configured benchmark suite and return a JSON-serializable summary."""
    tasks = load_tasks(task_file)
    runs: list[dict[str, Any]] = []

    for task in tasks:
        result = await run_design(
            task.prompt,
            use_docker=use_docker,
            max_messages=task.max_messages or default_max_messages,
        )
        messages = list(getattr(result, "messages", []))
        run_summary = evaluate_benchmark_run(task, messages=messages)
        run_summary["prompt"] = task.prompt
        runs.append(run_summary)

    passed_count = sum(1 for run in runs if run["passed"])
    return {
        "task_count": len(runs),
        "passed_count": passed_count,
        "pass_rate": passed_count / len(runs) if runs else 0.0,
        "runs": runs,
    }