"""Evidence requirements for scientific claims, separate from execution success.

The core transports and checks evidence. Installed expert modules own request
recognition, domain instructions, preflight advice and numerical validators.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

ACCEPTANCE_ENVIRONMENT = "BRILLIANCE_REQUEST_ACCEPTANCE"


def load_expert_registry(project_directory: Path | str | None = None) -> Any:
    """Use configured expertise and this project, never an implicit topic import."""
    from .experts import ExpertRegistry
    return ExpertRegistry.current(project_directory=project_directory)


def request_acceptance_requirements(request: str, *, registry: Any = None) -> tuple[str, ...]:
    """Collect applicable evidence requirements from enabled expert modules."""
    registry = registry if registry is not None else load_expert_registry()
    return tuple(check.id for check in registry.acceptance_checks if check.applies_to(str(request)))


def acceptance_instructions(required: Sequence[str], *, registry: Any = None) -> str:
    """Render module-owned guidance only for the current evidence requirements."""
    if not required:
        return ""
    registry = registry if registry is not None else load_expert_registry()
    sections = []
    for name in required:
        try:
            check = registry.acceptance_check(name)
        except LookupError:
            sections.append(f"Required evidence {name}: its expert validator is not installed/enabled. "
                            "Preserve existing evidence and report the missing capability; do not claim success.")
        else:
            sections.append(f"Expert requirement {check.id} ({check.module_id}):\n{check.instructions}")
    return "\n\n".join(sections)


def _is_followup_request(text: str) -> bool:
    """Recognize common replies/amendments, not arbitrary semantic topic links.

    A pronoun anywhere in a question is insufficient: "What is this?" and
    "Explain the previous result" are inspections, not new design campaigns.
    Requirements are inherited only when a pending request exists in history.
    """
    rendered = str(text).strip()
    if not rendered:
        return False
    choice = r"(?:[a-z]|[0-9]{1,2})"
    if re.fullmatch(
        rf"(?:option\s+)?(?:{choice}|\({choice}\)|\[{choice}\])[.!。！]?",
        rendered, re.IGNORECASE,
    ):
        return True
    if re.fullmatch(
        r"(?:yes|no|ok(?:ay)?|sure|do\s+that|use\s+that|try\s+that)[.!。！]?",
        rendered, re.IGNORECASE,
    ):
        return True
    # An acknowledgement can prefix a real amendment; it must not turn
    # "No, cancel this task" into permission to continue the pending campaign.
    rendered = re.sub(r"^(?:yes|no|ok(?:ay)?|sure)\s*,\s*", "", rendered,
                      flags=re.IGNORECASE)
    if re.match(
        r"^(?:please\s+)?(?:continue|resume|go\s+on|go\s+ahead|carry\s+on|"
        r"keep\s+going|proceed)\b",
        rendered, re.IGNORECASE,
    ):
        return True
    if re.fullmatch(r"(?:继续|继续吧|接着)[.!。！]?", rendered):
        return True
    return bool(re.match(
        r"^(?:please\s+)?(?:choose|select|keep|change|set|make|increase|decrease|"
        r"raise|lower|reduce|relax|release|allow|preserve|retain|shorten|lengthen|"
        r"expand|enlarge|tighten|loosen)\s+"
        r"(?:the|this|that|these|those|it|its|both|same|current|previous|option)\b",
        rendered, re.IGNORECASE,
    ))


def continued_request_acceptance(
    request: str, history: Sequence[Mapping[str, Any]] = (), *, registry: Any = None,
) -> tuple[str, ...]:
    """Carry evidence requirements across replies to a pending request.

    Explicit expert requests own their scope. Common option replies and parameter
    amendments inherit through other such replies until a saved completion or a
    substantive new request. No history is edited. This bounded-history fallback
    is not a general intent classifier or a persistent multi-task request store.
    """
    direct = request_acceptance_requirements(request, registry=registry)
    if direct:
        return direct
    if not _is_followup_request(request):
        return ()
    for row in reversed(history):
        if row.get("role") == "assistant":
            meta = row.get("meta", {})
            if isinstance(meta, Mapping) and (
                isinstance(meta.get("completion_evidence"), Mapping)
                or (meta.get("runtime") == "native_research_agent" and meta.get("kind") == "result"
                    and meta.get("request_satisfied") is True)
            ):
                return ()
        if row.get("role") == "user":
            content = str(row.get("content", ""))
            if not content.strip():
                continue
            required = request_acceptance_requirements(content, registry=registry)
            if required:
                return required
            if not _is_followup_request(content):
                return ()
    return ()


def environment_acceptance_requirements() -> tuple[str, ...]:
    raw = os.environ.get(ACCEPTANCE_ENVIRONMENT, "")
    if not raw.strip():
        return ()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("managed scientific acceptance requirements are not valid JSON") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("managed scientific acceptance requirements must be a list of names")
    return tuple(dict.fromkeys(value))


def optimization_acceptance_warnings(source: str, required: Sequence[str], *, registry: Any = None) -> list[str]:
    """Collect advisory checks supplied by the selected expert modules."""
    if not required:
        return []
    registry = registry if registry is not None else load_expert_registry()
    warnings = []
    for name in required:
        try:
            check = registry.acceptance_check(name)
        except LookupError:
            warnings.append(f"Required expert validator is not installed/enabled: {name}")
            continue
        if check.preflight is not None:
            advice = check.preflight(source)
            if not isinstance(advice, (list, tuple)) or any(not isinstance(item, str) for item in advice):
                raise TypeError(f"expert preflight {name!r} must return a list of advisory strings")
            warnings.extend(advice)
    return warnings


def assess_scientific_acceptance(
    reports: Mapping[str, Any] | None,
    *,
    required: Sequence[str] = (),
    lattice: Any = None,
    registry: Any = None,
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


def receipt_acceptance_issues(receipt: Mapping[str, Any], required: Sequence[str]) -> list[str]:
    """Check the current request against independent saved validation, not prose."""

    if not required:
        return []
    validation = receipt.get("validation")
    checked = validation.get("scientific_acceptance") if isinstance(validation, Mapping) else None
    if not isinstance(checked, Mapping):
        return [f"{name}: no independently validated numerical acceptance report exists in this receipt" for name in required]
    checks = checked.get("checks", {})
    issues = []
    for name in required:
        check = checks.get(name) if isinstance(checks, Mapping) else None
        if not isinstance(check, Mapping) or check.get("passed") is not True:
            detail = "; ".join(str(item) for item in check.get("issues", [])) if isinstance(check, Mapping) else "missing numerical acceptance evidence"
            issues.append(f"{name}: {detail or 'acceptance did not pass'}")
    return issues
