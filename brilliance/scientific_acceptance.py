"""Evidence requirements for scientific claims, separate from execution success.

The core transports and checks evidence. Installed expert modules own request
recognition, domain instructions, preflight advice and numerical validators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def load_expert_registry(project_directory: Path | str | None = None) -> Any:
    """Use configured expertise and this project, never an implicit topic import."""
    from .experts import ExpertRegistry
    return ExpertRegistry.current(project_directory=project_directory)


def request_acceptance_requirements(request: str, *, registry: Any = None) -> tuple[str, ...]:
    """Collect applicable evidence requirements from enabled expert modules."""
    registry = registry if registry is not None else load_expert_registry()
    return tuple(check.id for check in registry.acceptance_checks if check.applies_to(str(request)))










def assess_scientific_acceptance(
    reports: Mapping[str, Any] | None,
    *,
    required: Sequence[str] = (),
    lattice: Any = None,
    registry: Any = None,
    evidence_directory: Path | str | None = None,
) -> dict[str, Any]:
    """Validate declared reports without changing the calculation-validity flag."""

    if reports is not None and not isinstance(reports, Mapping):
        raise TypeError("acceptance must be a mapping of acceptance names to evidence reports")
    if not isinstance(required, (list, tuple)) or any(not isinstance(name, str) or not name.strip() for name in required):
        raise TypeError("required scientific acceptance names must be a list of nonempty strings")
    reports = dict(reports or {})
    if any(not isinstance(name, str) or not name.strip() for name in reports):
        raise TypeError("scientific acceptance names must be nonempty strings")
    names = list(dict.fromkeys([*required, *reports]))
    if names and registry is None:
        registry = load_expert_registry()
    checks: dict[str, dict[str, Any]] = {}
    for name in names:
        report = reports.get(name)
        issues: list[str] = []
        try:
            check = registry.acceptance_check(name)
        except LookupError:
            check = None
            issues.append("the expert acceptance validator is not installed or enabled")
        if not isinstance(report, Mapping):
            issues.append("the required numerical acceptance report is missing")
        elif check is not None:
            if check.requires_lattice and lattice is None:
                issues.append("this expert acceptance requires the exact archived candidate_lattice")
            try:
                from .candidate_evidence import numerical_evidence_directory
                with numerical_evidence_directory(evidence_directory):
                    findings = check.validator(report, lattice)
                if not isinstance(findings, (list, tuple)) or any(not isinstance(item, str) for item in findings):
                    raise TypeError("validator must return a list of issue strings, empty only when valid")
                issues.extend(findings)
            except Exception as exc:
                # A broken extension is not acceptance. Preserve its diagnostic
                # without allowing an independently valid calculation to vanish.
                issues.append(f"expert validator failed: {type(exc).__name__}: {exc}")
        # An absent report is unfinished work, not a failed numerical test.
        # It is still NOT a pass: completion and candidate promotion continue
        # to require independently validated evidence. A supplied invalid
        # report or an unavailable validator is a genuine failure, not pending.
        status = ("pending" if name not in reports and check is not None
                  else "failed" if issues else "passed")
        checks[name] = {"passed": not issues, "status": status, "issues": issues}
        if check is not None:
            checks[name]["module_id"] = check.module_id
            checks[name]["module_version"] = registry.module(check.module_id).version
    status = ("failed" if any(check["status"] == "failed" for check in checks.values())
              else "pending" if any(check["status"] == "pending" for check in checks.values())
              else "passed")
    return {
        "schema_version": "1.0",
        "required": list(dict.fromkeys(required)),
        "reports": reports,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
        "status": status,
        "issues": [f"{name}: {issue}" for name, check in checks.items() for issue in check["issues"]],
    }


def compact_scientific_acceptance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep verdicts and exact evidence identities, not the dense optics profile."""

    result = {key: value[key] for key in ("schema_version", "required", "checks", "passed", "status", "issues") if key in value}
    reports = value.get("reports", {})
    result["reports"] = {
        str(name): {key: report[key] for key in ("schema", "lattice_id", "cell_fingerprint", "thresholds", "metrics", "passed") if key in report}
        for name, report in reports.items() if isinstance(report, Mapping)
    } if isinstance(reports, Mapping) else {}
    return result
