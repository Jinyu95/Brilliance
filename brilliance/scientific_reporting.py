"""Presentation metadata, not a scientific controller."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_scientific_outcome(result: Mapping[str, Any]) -> dict[str, Any]:
    """Describe explicit reports without rewriting the author's result or plan.

    A successful process, an author-declared scientific check and an independently
    bound evaluation are different things. No implicit thresholds are applied.
    """
    result = dict(result)
    data = result.get("data") or {}
    acceptance = result.get("scientific_acceptance") or {}
    failed = []
    for source, checks in (("checks", data.get("checks")), ("gates", data.get("gates")),
                           ("acceptance", acceptance.get("checks"))):
        if not isinstance(checks, Mapping):
            continue
        for name, check in checks.items():
            report = check if isinstance(check, Mapping) else {"passed": check}
            if report.get("passed") is False or report.get("status") == "failed":
                failed.append({"check": str(name), "source": source, "issues": report.get("issues", [])})
    declared = result.get("scientific_passed")
    calculation = "reported_failed" if declared is False or failed else (
        "reported_passed" if declared is True else "not_assessed")
    request = result.get("request_satisfied")
    result["outcome_presentation"] = {
        "heading": "Research result",
        "calculation_status": calculation,
        "evidence_status": ("reporting_failed" if data.get("reporting_failure")
                            else "reported_checks_failed" if declared is False or failed
                            else "reported_result"),
        "request_acceptance_status": ("reported_satisfied" if request is True
                                      else "reported_incomplete" if request is False else "not_assessed"),
        "failed_checks": failed,
        "authority": "Author declarations and explicitly supplied reports; no automatic design certification.",
    }
    return result
