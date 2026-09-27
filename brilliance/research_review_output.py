"""Render an aggregate research-process review without reading project records."""
from __future__ import annotations

import csv
import html
import io
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_text


_MISSING = "not recorded"
_CSV_COLUMNS = {
    "metrics.csv": ("metrics", (
        "id", "label", "value", "unit", "numerator", "denominator", "definition", "coverage", "group",
    )),
    "tools.csv": ("tools", (
        "name", "category", "runtime", "calls", "call_share_percent", "succeeded", "failed",
        "unknown", "skipped", "success_rate_percent", "outcome_coverage_percent",
    )),
    "tool_calls.csv": ("tool_calls", (
        "id", "runtime", "tool", "outcome", "source", "event_index", "result_event_index",
        "argument_keys", "program", "run_id", "job_id", "error_class", "error_message",
    )),
    "web_calls.csv": ("online_research.calls", (
        "id", "runtime", "tool", "kind", "source", "event_index", "result_event_index",
        "query", "queries", "provider", "scope", "status", "result_count", "nonempty_results", "results",
        "requested_url", "requested_urls", "final_url", "domain", "title", "content_type", "html_content_scope",
        "returned_characters", "content_returned", "requested_offset", "requested_limit",
        "offset", "next_offset", "total_characters",
        "source_truncated", "fallback_from", "error_class", "error", "warnings",
        "request_detail_known", "retrieved_at",
    )),
    "web_providers.csv": ("online_research.providers", (
        "provider", "calls", "succeeded", "failed", "unknown", "skipped", "completed",
        "success_rate_percent", "result_count", "nonempty_results", "results_known",
        "nonempty_result_rate_percent", "warning_mentions", "unconfigured_warnings",
        "suppressed_warnings", "http_attempts",
    )),
    "web_sources.csv": ("online_research.sources", (
        "url", "domain", "title", "requested_urls", "calls", "succeeded", "failed", "unknown",
        "skipped", "completed", "success_rate_percent", "content_calls", "content_known",
        "returned_characters", "unique_returned_characters", "recorded_total_characters",
        "extracted_text_coverage_percent", "html_content_scopes", "content_types",
        "content_rate_percent", "interval_calls", "source_truncated", "error_classes", "fallback_calls", "coverage",
    )),
    "executions.csv": ("executions", (
        "run_id", "program", "status", "elapsed_s", "source_sha256", "source",
    )),
    "repairs.csv": ("repair_episodes", (
        "program", "failure_run_ids", "success_run_id", "edit_operations", "status",
        "source_changed", "failure_source_sha256",
    )),
}


def _text(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return _MISSING
    if isinstance(value, float) and not math.isfinite(value):
        return _MISSING
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _md(value: Any) -> str:
    text = html.escape(_text(value), quote=False)
    return (text.replace("\\", "\\\\").replace("|", "\\|")
            .replace("[", "\\[").replace("]", "\\]").replace("`", "\\`")
            .replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>"))


def _rows(report: dict, key: str) -> list[dict]:
    value: Any = report
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _number(value: Any) -> str:
    """Use readable numbers in prose while leaving machine-readable exports exact."""
    if isinstance(value, bool):
        return _text(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return _text(value)


def _percent(value: Any) -> str:
    text = _number(value)
    return text if text == _MISSING else text + "%"


def _short_hash(value: Any) -> Any:
    if (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdefABCDEF" for char in value)):
        return value[:12] + "…"
    return value


def _table(headers: tuple[str, ...], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No records available.", ""]
    return [
        "| " + " | ".join(_md(cell) for cell in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_md(cell) for cell in row) + " |" for row in rows),
        "",
    ]


def _executive_summary(project: dict, metrics: list[dict]) -> str:
    preferred = ("human_interventions", "agent_turns", "native_model_responses", "tool_calls",
                 "program_success_rate", "total_tokens")
    by_id = {metric.get("id"): metric for metric in metrics}
    candidates = [by_id[key] for key in preferred if key in by_id]
    # Older schemas can still provide one available measure per defined group.
    if not candidates:
        seen = set()
        for metric in metrics:
            group = _text(metric.get("group"))
            if group not in seen and metric.get("value") is not None:
                candidates.append(metric)
                seen.add(group)
            if len(candidates) == 6:
                break
    selected = []
    for metric in candidates:
        value = _number(metric.get("value"))
        unit = metric.get("unit")
        if value != _MISSING:
            if unit == "percent":
                value += "%"
            elif unit and unit != "count":
                value += " " + _text(unit)
        selected.append(f"{_md(metric.get('label') or metric.get('id'))}: {_md(value)}")
    title = _md(project.get("title") or project.get("id"))
    overview = (
        f"This review summarizes the saved research-process records for {title}. "
        "It describes recorded human participation, model activity, tool use, program "
        "execution and repair, and token usage. "
    )
    if selected:
        overview += "Selected recorded measures are " + "; ".join(selected) + ". "
    return overview + "Each measure uses the evidence scope and definition reported below."


def _online_research(report: dict) -> list[str]:
    online = report.get("online_research")
    if not isinstance(online, dict):
        return []
    lines = ["## Online search and source reading", "",
             "Request success, a nonempty search-result list, and returned source text are "
             "different measures. None establishes that a source is relevant, correct, or used "
             "in the final research. Missing outcomes and missing content counts remain "
             "**not recorded**; they are not treated as empty results or failures. Explicit "
             "request failures count as no results or no content for the corresponding yield rate.", ""]
    metrics = {row.get("id"): row for row in _rows(online, "metrics")}
    highlights = []
    for identifier in ("web_search_success_rate", "web_search_nonempty_result_rate",
                       "web_read_success_rate", "web_read_content_rate"):
        row = metrics.get(identifier)
        if row:
            numerator, denominator = row.get("numerator"), row.get("denominator")
            fraction = (f"{_number(numerator)} / {_number(denominator)}"
                        if numerator is not None and denominator is not None else None)
            highlights.append([row.get("label") or identifier, _percent(row.get("value")),
                               fraction, row.get("definition"), row.get("coverage")])
    lines += _table(("Measure", "Rate", "Numerator / denominator", "Definition", "Coverage"),
                    highlights)
    lines += ["### Search providers", "",
              "Provider success = succeeded / completed requests, where completed means "
              "succeeded + failed. Nonempty-result rate uses calls with a recorded result count. "
              "Warnings about an unavailable or suppressed provider are reported separately; "
              "they do not establish an HTTP attempt. HTTP attempt totals are **not recorded** "
              "unless saved explicitly.", ""]
    lines += _table(("Provider", "Calls", "Succeeded / failed / unknown / skipped",
                     "Success rate", "Results returned", "Nonempty / result counts known",
                     "Nonempty-result rate", "Warnings / unconfigured / suppressed"), [
        [row.get("provider"), row.get("calls"),
         " / ".join(_number(row.get(key)) for key in ("succeeded", "failed", "unknown", "skipped")),
         _percent(row.get("success_rate_percent")),
         row.get("result_count"),
         f"{_number(row.get('nonempty_results'))} / {_number(row.get('results_known'))}",
         _percent(row.get("nonempty_result_rate_percent")),
         " / ".join(_number(row.get(key)) for key in
                    ("warning_mentions", "unconfigured_warnings", "suppressed_warnings"))]
        for row in _rows(online, "providers")
    ])
    calls = _rows(online, "calls")
    lines += ["### Search query audit", "",
              "Each row represents a recorded search call. Result counts include repeated "
              "URLs across calls; result titles and URLs, when saved, are retained in both the JSON "
              "call records and the `results` column of `web_calls.csv`. That CSV also contains "
              "event references and complete selected metadata.", ""]
    lines += _table(("Call", "Tool", "Provider / scope", "Query", "Outcome", "Results",
                     "Error / warnings"), [
        [row.get("id"), row.get("tool"),
         " / ".join(_text(row.get(key)) for key in ("provider", "scope")), row.get("query"),
         row.get("status"), row.get("result_count"), _web_issue(row)]
        for row in calls if row.get("kind") == "search"
    ])
    lines += ["### Source-reading audit", "",
              "Offsets and next offsets describe pagination through extracted text. Returned "
              "characters measure the saved response text, not bytes downloaded or words read. "
              "A fallback can return an abstract or a different URL. A successful read therefore "
              "does not establish access to a full paper.", ""]
    lines += _table(("Call", "Requested URL", "Returned URL", "Outcome", "Characters",
                     "Offset / next / total", "Scope / content type", "Fallback / error"), [
        [row.get("id"), row.get("requested_url") or row.get("requested_urls"),
         row.get("final_url"), row.get("status"),
         row.get("returned_characters"),
         " / ".join(_number(row.get(key)) for key in ("offset", "next_offset", "total_characters")),
         " / ".join(_text(row.get(key)) for key in ("html_content_scope", "content_type")),
         row.get("fallback_from") or _web_issue(row)]
        for row in calls if row.get("kind") == "read"
    ])
    other = [row for row in calls if row.get("kind") not in ("search", "read")]
    if other:
        lines += ["### Other online calls", ""]
        lines += _table(("Call", "Tool", "Outcome", "Error / warnings"), [
            [row.get("id"), row.get("tool"), row.get("status"), _web_issue(row)] for row in other
        ])
    lines += ["### Source coverage", "",
              "Extracted-text coverage combines the union of recorded character intervals for "
              "each source, so overlapping or repeated pages are not double-counted. The denominator "
              "is the recorded extracted-text length. This is not full-document coverage, "
              "citation precision, source relevance, or proof that a result used the source.", ""]
    lines += _table(("Source URL", "Calls", "Success rate",
                     "Content / content counts known", "Returned characters",
                     "Unique / recorded total characters", "Extracted-text coverage",
                     "Scope", "Fallback calls", "Coverage notes"), [
        [row.get("url"), row.get("calls"),
         _percent(row.get("success_rate_percent")),
         f"{_number(row.get('content_calls'))} / {_number(row.get('content_known'))}",
         row.get("returned_characters"),
         f"{_number(row.get('unique_returned_characters'))} / {_number(row.get('recorded_total_characters'))}",
         _percent(row.get("extracted_text_coverage_percent")),
         row.get("html_content_scopes"), row.get("fallback_calls"), row.get("coverage")]
        for row in _rows(online, "sources")
    ])
    warnings = online.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines += ["### Online-research limitations", ""]
        lines += ["- " + _md(warning) for warning in warnings]
        lines.append("")
    return lines


def _web_issue(row: dict) -> Any:
    parts = []
    if row.get("error"):
        parts.append(_text(row["error"]))
    warnings = row.get("warnings")
    if isinstance(warnings, list):
        parts.extend(_text(warning) for warning in warnings)
    return "; ".join(parts) or None


def render_markdown(report: dict) -> str:
    """Render collector-provided aggregates; never infer missing counts as zero."""
    project = report.get("project")
    project = project if isinstance(project, dict) else {}
    metrics = _rows(report, "metrics")
    lines = ["# Research process review", "", _executive_summary(project, metrics), ""]
    lines += _table(("Project field", "Recorded value"), [
        [label, project.get(key)] for key, label in (
            ("id", "Project ID"), ("title", "Title"), ("path", "Source project"),
            ("created_at", "Created at"), ("reviewed_at", "Reviewed at"),
            ("status", "Project status"), ("branch_origin", "Branch origin"),
        )
    ])

    backend = project.get("backend")
    backend = backend if isinstance(backend, dict) else {}
    environment = report.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    compute = environment.get("compute")
    compute = compute if isinstance(compute, dict) else {}
    lines += ["## Model and execution setup", ""]
    lines += _table(("Setting", "Recorded value"), [
        ["Model backend", backend.get("name")],
        ["Configured model", backend.get("model")],
        ["Model wire API", backend.get("wire_api")],
        ["Compute backend", compute.get("backend")],
        ["Local worker processes", compute.get("local_worker_processes")],
        ["Threads per worker", compute.get("threads_per_worker")],
        ["Internet access", environment.get("internet_access")],
        ["Automatic continuation", environment.get("auto_continue")],
        ["Maximum autonomous increments", environment.get("max_autonomous_increments")],
    ])

    lines += ["## Metrics", ""]
    groups: dict[str, list[dict]] = OrderedDict()
    for metric in metrics:
        groups.setdefault(_text(metric.get("group")), []).append(metric)
    if not groups:
        lines += ["No metric records available.", ""]
    for group, grouped in groups.items():
        lines += ["### " + _md(group), ""]
        lines += _table(("Metric", "Value", "Unit", "Coverage", "Definition"), [
            [metric.get("label") or metric.get("id"), _number(metric.get("value")),
             metric.get("unit"), metric.get("coverage"), metric.get("definition")]
            for metric in grouped
        ])

    lines += ["## Tool usage", "",
              "Success rate = succeeded / (succeeded + failed). Unknown and skipped calls are "
              "excluded from this denominator. Outcome coverage = (succeeded + failed) / all calls; "
              "call share = this tool's calls / all tool calls. A zero denominator is **not recorded**. "
              "Technical success does not establish scientific usefulness.", "",
              "`tool_calls.csv` records individual call outcomes, available error details, "
              "program/run associations, and source-event references.", ""]
    tools = _rows(report, "tools")
    lines += _table(("Tool", "Category", "Runtime", "Calls", "Call share", "Succeeded", "Failed",
                     "Unknown", "Skipped", "Success rate", "Outcome coverage"), [
        [row.get("name"), row.get("category"), row.get("runtime"), row.get("calls"),
         _percent(row.get("call_share_percent")), row.get("succeeded"), row.get("failed"),
         row.get("unknown"), row.get("skipped"), _percent(row.get("success_rate_percent")),
         _percent(row.get("outcome_coverage_percent"))]
        for row in tools
    ])
    lines += _online_research(report)

    lines += ["## Program executions", "",
              "Recorded technical success does not establish scientific validity, satisfaction "
              "of research requirements, or acceptance of a result.", ""]
    lines += _table(("Run ID", "Program", "Technical status", "Elapsed (s)",
                     "Source SHA-256 prefix", "Evidence source"), [
        [row.get("run_id"), row.get("program"), row.get("status"),
         _number(row.get("elapsed_s")), _short_hash(row.get("source_sha256")), row.get("source")]
        for row in _rows(report, "executions")
    ])

    lines += ["## Repair episodes", "",
              "Edit operations are **not recorded** when individual operations cannot be "
              "attributed to an episode. Source changes compare the saved failure and recovery "
              "program hashes; recovery can also follow input or environment changes.", ""]
    lines += _table(("Program", "Failed run IDs", "Recovery run ID", "Edit operations", "Status",
                     "Source changed", "Failure SHA-256 prefix"), [
        [row.get("program"), row.get("failure_run_ids"), row.get("success_run_id"),
         row.get("edit_operations"), row.get("status"), row.get("source_changed"),
         _short_hash(row.get("failure_source_sha256"))]
        for row in _rows(report, "repair_episodes")
    ])

    lines += ["## Program files", ""]
    lines += _table(("Path", "Lines", "SHA-256 prefix"), [
        [row.get("path"), row.get("lines"), _short_hash(row.get("sha256"))]
        for row in _rows(report, "program_files")
    ])

    # These are allowlisted summaries, not raw conversation or model-event dumps.
    for key, heading, columns in (
        ("models", "Model summaries", ("model", "returned_models", "reasoning_efforts", "runtime", "provider", "sessions", "responses", "turns",
                                        "input_tokens", "output_tokens", "total_tokens", "coverage")),
        ("sessions", "Session summaries", ("source", "runtime", "model",
                                            "accepted_model_responses", "rejected_model_responses",
                                            "events", "session_id", "started_at", "ended_at", "status")),
    ):
        rows = _rows(report, key)
        present = tuple(column for column in columns if any(column in row for row in rows))
        if rows and present:
            lines += ["## " + heading, ""]
            lines += _table(tuple(column.replace("_", " ").title() for column in present),
                            [[row.get(column) for column in present] for row in rows])

    lines += ["## Appendix: methodology, provenance, and limitations", "",
              "- Metric definitions and coverage are supplied by the collector in the tables above. "
              "Counts describe saved events within that scope.",
              "- Missing values are shown as **not recorded**. An explicit numeric zero remains zero.",
              "- Human-message and iteration measures describe recorded activity; they cannot "
              "recover unlogged human effort or model reasoning.",
              "- Repair associations follow the collector's definition. An association between "
              "edits and later success does not establish causation.",
              "- Token figures are observed subtotals for sources with recorded usage, not a complete "
              "billing statement. Missing usage must not be interpreted as zero.",
              "- Complete SHA-256 hashes are retained in JSON and applicable CSV files; Markdown "
              "tables show their first 12 characters. Numeric prose and tables use at most two "
              "decimal places; machine-readable exports retain recorded precision.",
              "- This report contains aggregates and evidence references, plus selected online "
              "search queries, requested URLs, source metadata, and errors for auditing. It does "
              "not reproduce full user prompts, program bodies, retrieved article contents, "
              "or model reasoning.", "",
              "Schema version: " + _md(report.get("schema_version")) + ".", "",
              "### Evidence sources", ""]
    sources = report.get("sources")
    if isinstance(sources, list) and sources:
        lines += ["- " + _md(source) for source in sources]
        lines.append("")
    else:
        lines += ["No evidence sources recorded.", ""]
    lines += ["### Collection warnings", ""]
    warnings = report.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines += ["- " + _md(warning) for warning in warnings]
        lines.append("")
    else:
        lines += ["No collection warnings recorded; this does not establish evidence completeness.", ""]
    return "\n".join(lines)


def _csv_cell(value: Any) -> str:
    text = _text(value)
    # Quoting by csv.writer alone does not stop spreadsheet formula evaluation.
    if isinstance(value, str) and text.lstrip().lstrip("\ufeff").lstrip().startswith(("=", "+", "@", "-")):
        return "'" + text
    return text


def write_review(report: dict, output_dir: Path) -> dict[str, Path]:
    """Write a review to the chosen destination without modifying the input report.

    JSON preserves the full aggregate schema and nulls. CSV uses UTF-8 with a BOM
    for Excel, explicit missing-value text, and protection against string formulas.
    The collector excludes full prompts, retrieved contents, program bodies and reasoning;
    selected search queries, URLs and error metadata are deliberately included for auditing.
    """
    output_dir = Path(output_dir)
    documents = {
        "report.md": render_markdown(report),
        "report.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    }
    for filename, (key, columns) in _CSV_COLUMNS.items():
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in _rows(report, key):
            writer.writerow([_csv_cell(row.get(column)) for column in columns])
        documents[filename] = "\ufeff" + stream.getvalue()
    paths = {}
    for filename, content in documents.items():
        path = output_dir / filename
        atomic_write_text(path, content)
        paths[filename] = path
    return paths
