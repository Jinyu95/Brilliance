"""Presentation metadata, not a scientific controller."""

from __future__ import annotations

import re
from typing import Any, Mapping


def is_artifact_reference(value: Any, name: str = "") -> bool:
    """Recognize file references for display, without opening or trusting them."""
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        return False
    text = value.strip()
    return bool(
        name.lower() in {"path", "file", "filename", "directory", "output_directory", "output_dir"}
        or re.match(r"^(?:[A-Za-z]:[\\/]|[/\\]|\.?\.?[/\\]|workspace[/\\]|artifacts[/\\])", text)
        or re.fullmatch(r"[^<>\n]+\.(?:json|csv|npz|npy|png|svg|pdf|jl|py|toml|txt|h5|hdf5|parquet|zip|mat)", text, re.I)
    )


def artifact_only_table(table: Mapping[str, Any]) -> bool:
    """A list of file locations is provenance, not a measured-values table."""
    rows = table.get("rows")
    return bool(isinstance(rows, list) and rows and all(
        isinstance(row, Mapping)
        and any(is_artifact_reference(value, str(key)) for key, value in row.items())
        and all(value is None or is_artifact_reference(value, str(key))
                or isinstance(value, str) and str(key).lower() in {"quantity", "name", "title", "label"}
                for key, value in row.items())
        for row in rows))


def artifact_references(result: Mapping[str, Any]) -> list[dict[str, str]]:
    data = result.get("data") or {}
    references = [{"name": str(key), "path": value} for key, value in data.items()
                  if is_artifact_reference(value, str(key))] if isinstance(data, Mapping) else []
    for table in result.get("tables", []):
        if isinstance(table, Mapping) and artifact_only_table(table):
            for row in table["rows"]:
                references.extend({"name": str(row.get("quantity") or key), "path": value}
                                  for key, value in row.items() if is_artifact_reference(value, str(key)))
    return [dict(item) for item in {tuple(item.items()): item for item in references}.values()]


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
    references = artifact_references(result)
    artifacts_only = bool(references and not any(
        key not in {"stdout", "stderr", "returncode", "publication"}
        and not is_artifact_reference(value, str(key)) and value is not None
        for key, value in data.items())
        and not any(isinstance(table, Mapping) and table.get("rows") and not artifact_only_table(table)
                    for table in result.get("tables", []))
        and not result.get("plots") and not result.get("candidate_evaluation"))
    result["outcome_presentation"] = {
        "heading": "Artifacts saved" if artifacts_only else "Research result",
        "publication_status": "artifacts_only" if artifacts_only else "reported_content",
        "needs_scientific_interpretation": artifacts_only,
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
