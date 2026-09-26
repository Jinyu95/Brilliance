"""Evidence-grounded conversation and reporting for research projects.

This module never performs accelerator calculations.  It translates accepted
JuTrack evidence into bounded human-facing messages and keeps ordinary chat
separate from the immutable scientific project revisions.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import content_hash, json_safe


@dataclass(frozen=True, slots=True)
class StageMessage:
    task_id: str
    title: str
    status: str
    text: str
    evidence: Mapping[str, Any] | None = None


def compact_operational_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse platform chatter in the view; never edit history or scientific prose."""
    output: list[dict[str, Any]] = []
    for message in messages:
        content = str(message.get("content", "")).strip()
        meta = message.get("meta") or {}
        operational = (
            content.startswith("Preflight advisory:")
            or meta.get("kind") == "heartbeat"
            or meta.get("scientific_role") == "activity"
            or content.startswith("**Execution ") and "submitted as a durable JuTrack calculation" in content
            or content.startswith("The long calculation was submitted as")
            or content.startswith("The durable calculation `")
            or content.startswith("Model: **") and "saved research evidence" in content
        )
        failure = content.startswith((
            "This calculation could not complete.",
            "This execution stopped before the stage finished.",
            "The iterative research loop stopped before a defensible result was ready.",
            "The research turn stopped at its operational boundary:",
            "The background research worker was interrupted while handling",
        ))
        if message.get("role") != "assistant" or not content or not (operational or failure):
            output.append(message)
            continue
        # Scientific attachments must stay visible even on an operational message.
        attachments = list(message.get("attachments") or [])
        row = {"id": message["id"], "timestamp": message.get("timestamp"), "text": content}
        headline = ""
        if failure:
            headline = ("An execution issue occurred; saved results were preserved. "
                        "See technical details for the error and recovery status.")
        entry = {**message, "content": headline,
                 "meta": {**meta, "presentation_role": "technical_details"},
                 "attachments": [*attachments, {"type": "technical_details", "title": "Technical details",
                                                 "rows": [row]}]}
        if (not failure and not attachments and output
                and output[-1].get("meta", {}).get("presentation_role") == "technical_details"
                and not output[-1].get("content")
                and len(output[-1].get("attachments", [])) == 1):
            output[-1]["attachments"][0]["rows"].append(row)
        else:
            output.append(entry)
    return output


def deduplicate_chat_attachments(messages: list[dict[str, Any]]) -> None:
    """Reference repeated evidence without hiding new candidate results.

    De-duplication uses exact content and available provenance, never a plot
    filename category or its byte size. Reset at each scientist message so an
    explicit follow-up can show a previous figure again. Saved history is not
    edited; only the rendered snapshot is compacted. Identical result-card
    content can share a single presentation across repeated calculations, but
    every distinct calculation receipt remains attached to that presentation.
    """
    seen: dict[str, str] = {}
    image_versions: dict[str, str] = {}
    summaries_by_content: dict[str, dict[str, Any]] = {}
    selected_presentations: set[str] = set()
    for message in messages:
        if message.get("role") == "user":
            seen.clear()
            image_versions.clear()
            summaries_by_content.clear()
            selected_presentations.clear()
        if message.get("meta", {}).get("runtime") == "managed_compute":
            # Historical managed completions embedded the same table twice:
            # generated Markdown plus a structured attachment. Remove only the
            # exact platform-generated block, leaving scientist-facing prose
            # and every independently-valued table intact.
            from .jobs import _markdown_rows
            summaries = [item for item in message.get("attachments", [])
                         if isinstance(item, Mapping) and item.get("type") == "evidence_summary"]
            from .output_tables import output_tables
            for item in summaries:
                if item.get("program_stdout") and not item.get("tables"):
                    item["tables"] = output_tables(item["program_stdout"])
            # Read-only migration of the exact old platform log-as-table block.
            content = str(message.get("content", ""))
            if any(item.get("program_stdout") for item in summaries):
                message["content"] = re.sub(
                    r"\| Quantity \| Value \|\n\|---\|---:\|\n(?:\|[^\n]*\|\n?)+",
                    "Raw execution logs are available under Program output.\n", content, count=1)
            if any(table for item in summaries for table in item.get("tables", [])
                   if isinstance(table, Mapping) and not table.get("generated_from_evidence")):
                for item in summaries:
                    prefix = "**Research result**\n\n" + str(item.get("summary", "")) + "\n\n"
                    header = "| Quantity | Value |\n|---|---:|\n"
                    content = str(message.get("content", ""))
                    if content.startswith(prefix + header):
                        tail = content[len(prefix):]
                        block, separator, remainder = tail.partition("\n\n")
                        if all(row.startswith("|") and row.endswith("|") for row in block.splitlines()):
                            message["content"] = prefix.rstrip() + ("\n\n" + remainder if separator else "")
            for attachment in message.get("attachments", []):
                if not isinstance(attachment, Mapping) or attachment.get("type") != "evidence_summary":
                    continue
                for table in attachment.get("tables", []):
                    if not isinstance(table, Mapping) or not isinstance(table.get("rows"), list):
                        continue
                    title = str(table.get("title", "Results")).strip() or "Results"
                    block = f"**{title}**\n\n" + "\n".join(_markdown_rows(table["rows"]))
                    message["content"] = str(message.get("content", "")).replace(block, "", 1)
            message["content"] = re.sub(r"\n{3,}", "\n\n", str(message.get("content", ""))).strip()
        attachments = []
        references: set[tuple[str, str]] = set()

        def duplicate(key: str, title: str, *, aliases: tuple[str, ...] = ()) -> bool:
            identities = (key, *aliases)
            previous = next((seen[item] for item in identities if item in seen), None)
            if previous is None:
                for item in identities:
                    seen[item] = str(message["id"])
                return False
            for item in identities:
                seen[item] = previous
            # Same-message copies need no reference to themselves.
            if previous != str(message["id"]) and (previous, title) not in references:
                references.add((previous, title))
                attachments.append({
                    "type": "evidence_reference",
                    "title": title,
                    "message_id": previous,
                })
            return True

        for attachment in message.get("attachments", []):
            if not isinstance(attachment, Mapping):
                continue
            row = dict(attachment)
            kind = row.get("type")
            if row.get("presentation_role") == "selected_evidence":
                # The final selected result is a coherent scientific handoff,
                # not a pointer back to scattered intermediate messages.
                # Render its own exact figures/tables once, even if progress
                # already showed them. Repeated final copies remain compact.
                identity = content_hash(row)
                if identity in selected_presentations:
                    continue
                selected_presentations.add(identity)
                attachments.append(row)
                continue
            if kind == "evidence_reference":
                reference = (str(row.get("message_id", "")), str(row.get("title", "")))
                if reference in references:
                    continue
                references.add(reference)
            elif kind == "image":
                digest = str(row.get("content_sha256") or "")
                locations = [f"image:{row[key]}" for key in ("url", "path") if row.get(key)]
                # An overwritten path with new known bytes is a new figure.
                # A legacy progress attachment without a hash can still match
                # the later, hashed final attachment at the same location.
                identities = [f"image:{digest}"] if digest else []
                identities.extend(location for location in locations if not digest or
                                  image_versions.get(location, digest) == digest)
                repeated = identities and duplicate(identities[0], str(row.get("title") or "Figure"), aliases=tuple(identities[1:]))
                if digest:
                    for location in locations:
                        image_versions[location] = digest
                        seen[location] = seen[f"image:{digest}"]
                if repeated:
                    continue
            elif kind == "evidence_summary":
                # A source/result/lattice id distinguishes equally-valued but
                # independently evaluated candidates. Legacy untagged tables
                # can still be compacted by exact contents.
                provenance = next(({key: row[key]} for key in ("lattice_id", "run_id", "result_path") if row.get(key)), {})
                # A retry may produce a new receipt with exactly the same
                # public result. Keep that result once (including its methods,
                # validation status and limitations), preserving all receipts.
                # Candidate identity and all scientific/display values stay in
                # the fingerprint: equal headline values alone are not enough.
                whole_key = "summary:" + content_hash({
                    key: value for key, value in row.items()
                    if key not in {"run_id", "result_path", "evidence_sources"}
                })
                prior_summary = summaries_by_content.get(whole_key)
                if prior_summary is not None:
                    sources = _summary_evidence_sources(prior_summary, row)
                    if len(sources) > 1:
                        prior_summary["evidence_sources"] = sources
                if duplicate(whole_key, str(row.get("title") or "Scientific result")):
                    continue
                summaries_by_content[whole_key] = row
                tables = []
                for table in row.get("tables", []):
                    if not isinstance(table, Mapping):
                        continue
                    key = "table:" + content_hash({"table": table, "source": provenance})
                    if not duplicate(key, str(table.get("title") or "Evidence table")):
                        tables.append(dict(table))
                row["tables"] = tables
                if not tables and not any(row.get(key) for key in ("summary", "steps", "limitations")):
                    continue
            attachments.append(row)
        message["attachments"] = attachments


def _summary_evidence_sources(*summaries: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain distinct receipts when identical display content is consolidated."""
    sources: list[dict[str, Any]] = []
    identities: set[str] = set()
    for summary in summaries:
        inherited = summary.get("evidence_sources", [])
        rows = list(inherited) if isinstance(inherited, (list, tuple)) else []
        rows.append({key: summary[key] for key in ("run_id", "result_path", "lattice_id")
                     if summary.get(key)})
        for row in rows:
            if not isinstance(row, Mapping) or not row:
                continue
            identity = content_hash(row)
            if identity not in identities:
                identities.add(identity)
                sources.append(dict(row))
    return sources


def evidence_by_task(project: Any) -> dict[str, Mapping[str, Any]]:
    """Return the newest evidence record for every task."""
    records: dict[str, Mapping[str, Any]] = {}
    for evidence in getattr(project, "evidence", ()):
        if isinstance(evidence, Mapping) and evidence.get("task_id"):
            records[str(evidence["task_id"])] = evidence
    return records


def stage_messages(project: Any) -> list[StageMessage]:
    """Build a compact chronological research transcript from project state."""
    evidence = evidence_by_task(project)
    messages: list[StageMessage] = []
    for task in getattr(project, "tasks", ()):
        record = evidence.get(str(task.id))
        if task.status == "accepted" and record is not None:
            messages.append(StageMessage(
                task_id=str(task.id),
                title=str(task.title),
                status="accepted",
                text=summarize_stage(task, record),
                evidence=record,
            ))
        elif str(task.id) == getattr(project, "current_task_id", None):
            state = str(task.status).replace("_", " ")
            messages.append(StageMessage(
                task_id=str(task.id),
                title=str(task.title),
                status=str(task.status),
                text=(
                    f"I am at **{task.title}** ({state}). {task.objective} "
                    f"The result will not be accepted until its deterministic evidence gate passes."
                ),
                evidence=record,
            ))
    return messages


def summarize_stage(task: Any, evidence: Mapping[str, Any]) -> str:
    """Explain one result using its stored values and capability contract."""
    result = evidence.get("structured_result")
    result = result if isinstance(result, Mapping) else {}
    capability = str(getattr(task, "capability", ""))
    validator = _validator(evidence)
    accepted = f" **Accepted by {validator}.**" if validator else " **Evidence accepted.**"

    if capability in {"knowledge_synthesis", "reference_comparison"}:
        sources = _mapping_list(result.get("sources"))
        findings = [str(item) for item in result.get("findings", []) if str(item).strip()]
        lead = f"I retrieved {len(sources)} traceable source records"
        if findings:
            lead += f". The main methodological finding is: {findings[0]}"
        return lead + accepted

    if capability in {"validated_mba_cell", "validated_periodic_cell"}:
        best = result.get("best_run")
        metrics = best.get("metrics", {}) if isinstance(best, Mapping) else {}
        campaign = result.get("campaign", {})
        stages = campaign.get("stages", []) if isinstance(campaign, Mapping) else []
        screened = _first_nested(stages, "screened_candidates")
        study = result.get("study_spec", {})
        metadata = study.get("metadata", {}) if isinstance(study, Mapping) else {}
        lead_parameters = (
            metadata.get("research_lead_parameters", {})
            if isinstance(metadata, Mapping) else {}
        )
        search_clause = (
            " The Research Lead explicitly declared the archived search envelope."
            if isinstance(lead_parameters, Mapping) and lead_parameters else ""
        )
        return (
            f"The JuTrack seed campaign screened {_number(screened, 0)} candidates and selected a "
            f"stable {_number(metrics.get('energy_gev'), 3)} GeV solution: "
            f"**{_number(metrics.get('emittance_pm'), 3)} pm**, "
            f"**{_number(metrics.get('circumference_m'), 3)} m**, with maximum "
            f"|K1| = {_number(metrics.get('max_abs_k1'), 3)} m⁻² and phase error "
            f"{_number(metrics.get('phase_error_deg'), 3)}°."
            + search_clause + accepted
        )

    if capability == "lattice_construction":
        elements = _lattice_elements(result)
        proxy = _is_proxy_ring(result)
        return (
            f"JuTrack assembled the accepted {'homogeneous recurring-cell proxy' if proxy else 'design'} "
            f"into **{len(elements):,} elements** over "
            f"**{_number(result.get('circumference_m', result.get('length_m')), 3)} m**. "
            f"The total bend is {_number(result.get('bend_sum_rad'), 7)} rad and the geometry "
            f"closure error is {_scientific(result.get('geometry_closure_error_rad'))} rad."
            + (
                " This does **not** establish dispersion suppressors, insertion straights, or a "
                "practical full MBA achromat."
                if proxy else ""
            )
            + accepted
        )

    if capability == "linear_matching":
        profile = _mapping_list(result.get("twiss_profile"))
        max_bx = _max_abs(profile, "beta_x")
        max_by = _max_abs(profile, "beta_y")
        max_dx = _max_abs(profile, "Dx")
        return (
            f"The closed ring is linearly stable. JuTrack gives fractional tunes "
            f"**({_number(result.get('tune_x'), 5)}, {_number(result.get('tune_y'), 5)})**, "
            f"maximum βx/βy of {_number(max_bx, 3)}/{_number(max_by, 3)} m, maximum "
            f"|Dx| of {_number(max_dx, 4)} m, and periodic closure "
            f"{_scientific(result.get('periodic_closure_max'))}.{accepted}"
        )

    if capability == "radiation_analysis":
        ring = result.get("jutrack_ring_parameters", {})
        alpha_c = ring.get("alphac") if isinstance(ring, Mapping) else None
        return (
            f"JuTrack radiation integrals predict **ε₀ = {_number(result.get('emittance_pm'), 3)} pm rad** "
            f"with rms energy spread {_percent(result.get('energy_spread'), 4)} and momentum "
            f"compaction {_scientific(alpha_c)}. The damping partitions "
            f"(Jx, Jy, Jz) = ({_number(result.get('Jx'), 4)}, "
            f"{_number(result.get('Jy'), 4)}, {_number(result.get('Jz'), 4)}) sum to "
            f"{_number(result.get('robinson_sum'), 5)}.{accepted}"
        )

    if capability == "chromatic_correction":
        natural = _pair(result.get("natural_chromaticities"))
        target = _pair(result.get("target_chromaticities"))
        corrected = _pair(result.get("corrected_chromaticities"))
        strengths = result.get("k2_values", {})
        return (
            f"The two-family JuTrack response moved chromaticity from "
            f"**({natural[0]}, {natural[1]})** to **({corrected[0]}, {corrected[1]})** "
            f"for target ({target[0]}, {target[1]}). The fitted strengths are "
            f"SD = {_number(strengths.get('SD') if isinstance(strengths, Mapping) else None, 4)} "
            f"and SF = {_number(strengths.get('SF') if isinstance(strengths, Mapping) else None, 4)} "
            f"m⁻³; response condition number = {_number(result.get('response_condition'), 4)}."
            + accepted
        )

    if capability == "multiobjective_optimization":
        best = result.get("best_candidate", {})
        return (
            f"The bounded JuTrack search evaluated **{_number(result.get('evaluated_candidates'), 0)} "
            f"candidates**. The selected feasible point has chromaticity norm "
            f"{_scientific(best.get('chromatic_norm') if isinstance(best, Mapping) else None)} "
            f"and sextupole-strength norm "
            f"{_number(best.get('strength_norm') if isinstance(best, Mapping) else None, 3)}."
            + accepted
        )

    if capability == "nonlinear_tracking":
        scans = _mapping_list(result.get("scans"))
        offsets = ", ".join(f"{100 * float(scan.get('dp', 0)):+.1f}%" for scan in scans)
        areas = ", ".join(_number(scan.get("da_area_mm2"), 3) for scan in scans)
        return (
            f"JuTrack ringpass completed **{_number(result.get('nturns'), 0)} turns** at momentum "
            f"offsets {offsets}. The corresponding reported dynamic-aperture areas are "
            f"{areas} mm². This is magnetic-lattice tracking; RF and collective effects remain "
            f"outside this gate.{accepted}"
        )

    if capability == "error_correction":
        stats = result.get("ensemble_statistics", {})
        return (
            f"All {_number(stats.get('seed_count') if isinstance(stats, Mapping) else None, 0)} "
            f"JuTrack error seeds passed correction. Median orbit RMS improved from "
            f"{_micrometres(stats.get('median_rms_before_m') if isinstance(stats, Mapping) else None)} "
            f"to {_micrometres(stats.get('median_rms_after_m') if isinstance(stats, Mapping) else None)}, "
            f"an improvement factor of "
            f"{_number(stats.get('improvement_factor') if isinstance(stats, Mapping) else None, 3)}."
            + accepted
        )

    if capability == "cross_validation":
        return (
            f"The explicit full-ring map and independent repeated-block composition agree: maximum "
            f"relative difference **{_scientific(result.get('maximum_relative_difference'))}**, "
            f"below the {_scientific(result.get('tolerance'))} tolerance. This checks assembly and "
            f"map composition within JuTrack; it is not a second-code benchmark.{accepted}"
        )

    if capability == "publication_synthesis":
        findings = [str(item) for item in result.get("findings", []) if str(item).strip()]
        statement = findings[0] if findings else "All planned evidence gates have been synthesized."
        return statement + accepted

    return f"{task.title} produced accepted, archived evidence from {evidence.get('source', 'the registered service')}." + accepted


def summarize_rejected_stage(task: Any, evidence: Mapping[str, Any]) -> str:
    """Describe a rejected attempt as an observation, not a project failure."""
    result = evidence.get("structured_result")
    result = result if isinstance(result, Mapping) else {}
    campaign = result.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    stages = campaign.get("stages")
    stages = stages if isinstance(stages, (list, tuple)) else ()
    linear = next(
        (
            item for item in stages
            if isinstance(item, Mapping) and item.get("name") == "linear_scan"
        ),
        {},
    )
    metrics = linear.get("metrics") if isinstance(linear, Mapping) else {}
    metrics = metrics if isinstance(metrics, Mapping) else {}
    try:
        screened = int(metrics.get("screened_candidates", 0))
        planned = int(metrics.get("planned_experiments", 0))
        completed = int(metrics.get("completed_experiments", 0))
        calls = int(metrics.get("simulator_calls", 0))
    except (TypeError, ValueError):
        screened = planned = completed = calls = 0
    if planned == 0 or completed == 0 or calls == 0:
        return (
            f"The search screened **{screened} candidates**, but the selection layer sent "
            "no executable finalist to JuTrack. This attempt is inconclusive: it identifies "
            "a campaign/service problem, not an infeasible accelerator design. The supervisor "
            "kept the result for diagnosis and did not accept a numerical claim."
        )
    best = metrics.get("best_observed")
    best = best if isinstance(best, Mapping) else {}
    reasons = metrics.get("rejection_reasons")
    reasons = reasons if isinstance(reasons, Mapping) else {}
    ranked_reasons = sorted(
        reasons.items(), key=lambda item: int(item[1]), reverse=True
    )[:3]
    reason_text = ", ".join(f"{name} ({count})" for name, count in ranked_reasons)
    observation = (
        f"JuTrack evaluated **{completed} of {planned} finalists**, but none passed every "
        "declared gate."
    )
    if best:
        observation += (
            " The nearest archived result had "
            f"{_number(best.get('emittance_pm'), 3)} pm emittance, "
            f"{_number(best.get('circumference_m'), 3)} m circumference, and "
            f"{_number(best.get('phase_error_deg'), 3)} deg phase error."
        )
    if reason_text:
        observation += f" The most frequent failed checks were {reason_text}."
    return observation + " The result remains a rejected experiment, not a project crash."


def project_key_metrics(project: Any) -> list[dict[str, str]]:
    """Collect headline design values from the newest accepted stage evidence."""
    results = _results_by_capability(project)
    seed = results.get("validated_mba_cell", results.get("validated_periodic_cell", {}))
    best = seed.get("best_run", {}) if isinstance(seed, Mapping) else {}
    seed_metrics = best.get("metrics", {}) if isinstance(best, Mapping) else {}
    construction = results.get("lattice_construction", {})
    optics = results.get("linear_matching", {})
    radiation = results.get("radiation_analysis", {})
    tracking = results.get("nonlinear_tracking", {})
    robustness = results.get("error_correction", {})
    stats = robustness.get("ensemble_statistics", {}) if isinstance(robustness, Mapping) else {}
    scans = _mapping_list(tracking.get("scans") if isinstance(tracking, Mapping) else None)

    rows: list[dict[str, str]] = []
    _metric(rows, "Natural emittance", radiation.get("emittance_pm", seed_metrics.get("emittance_pm")), "pm rad", 3)
    _metric(rows, "Circumference", construction.get("circumference_m", seed_metrics.get("circumference_m")), "m", 3)
    if optics:
        rows.append({
            "quantity": "Fractional tunes",
            "value": f"({_number(optics.get('tune_x'), 5)}, {_number(optics.get('tune_y'), 5)})",
            "unit": "",
        })
    if _finite(radiation.get("energy_spread")):
        _metric(rows, "Energy spread", 100.0 * float(radiation["energy_spread"]), "%", 4)
    if scans:
        areas = [float(item["da_area_mm2"]) for item in scans if _finite(item.get("da_area_mm2"))]
        if areas:
            _metric(rows, "Minimum tracked DA area", min(areas), "mm²", 3)
    if isinstance(stats, Mapping) and _finite(stats.get("success_fraction")):
        _metric(rows, "Correction success", 100.0 * float(stats["success_fraction"]), "%", 1)
    return rows


def final_result_text(project: Any) -> str:
    """Return a concise final answer suitable for the last assistant message."""
    metrics = {row["quantity"]: row for row in project_key_metrics(project)}
    emittance = metrics.get("Natural emittance", {}).get("value", "not available")
    circumference = metrics.get("Circumference", {}).get("value", "not available")
    complete = getattr(project, "status", "") == "complete"
    accepted = sum(getattr(item, "status", "") == "accepted" for item in getattr(project, "tasks", ()))
    total = len(getattr(project, "tasks", ()))
    if complete:
        construction = _results_by_capability(project).get("lattice_construction", {})
        if _is_proxy_ring(construction):
            return (
                f"The JuTrack computation completed **{accepted}/{total} evidence gates** for a "
                f"homogeneous recurring-cell proxy: **{emittance} pm rad** at **{circumference} m**. "
                "This is not yet a practical full MBA design: dispersion suppressors, insertion "
                "straights, and their rematching were not constructed. The archived plots and "
                "numbers must be interpreted inside that boundary."
            )
        return (
            f"The JuTrack research chain is complete: **{accepted}/{total} evidence gates passed**. "
            f"The selected lattice reaches **{emittance} pm rad** natural emittance at "
            f"**{circumference} m**, satisfying the stated design targets. The plots below are "
            f"generated directly from the accepted lattice and physics artifacts."
        )
    return (
        f"The project currently has **{accepted}/{total} accepted stages**. I will update this "
        f"conversation automatically as each JuTrack result passes or fails its gate."
    )






def project_limitations(project: Any) -> list[str]:
    limitations: list[str] = []
    for result in _results_by_capability(project).values():
        for key in ("model_limits", "uncertainties"):
            values = result.get(key)
            if isinstance(values, (list, tuple)):
                limitations.extend(str(item) for item in values if str(item).strip())
    limitations.extend(str(item) for item in getattr(project, "assumptions", ()) if str(item).strip())
    return list(dict.fromkeys(limitations))


def markdown_report(project: Any) -> str:
    """Create a bounded, portable report without embedding raw numerical arrays."""
    lines = [
        f"# {project.title}",
        "",
        f"**Research question:** {project.goal}",
        "",
        final_result_text(project),
        "",
        "## Headline results",
        "",
        "| Quantity | Value | Unit |",
        "|---|---:|---|",
    ]
    for row in project_key_metrics(project):
        lines.append(f"| {row['quantity']} | {row['value']} | {row['unit']} |")
    lines.extend(["", "## Evidence chain", ""])
    for index, message in enumerate(stage_messages(project), start=1):
        lines.extend([f"### {index}. {message.title}", "", message.text, ""])
    limitations = project_limitations(project)
    if limitations:
        lines.extend(["## Assumptions and limits", ""])
        lines.extend(f"- {item}" for item in limitations)
        lines.append("")
    lines.extend([
        "## Provenance",
        "",
        f"Project `{project.id}`, revision {project.revision}. Numerical authority: JuTrack. "
        "Acceptance authority: registered deterministic validators.",
        "",
    ])
    return "\n".join(lines)


class ResearchConversationStore:
    """Append-only result Q&A that cannot race or mutate physics revisions."""

    def __init__(self, project_root: Path | str):
        self.root = Path(project_root).resolve()

    def append_exchange(self, project_id: str, question: str, answer: str) -> Path:
        return self.append_messages(project_id, (
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ))

    def append_message(
        self,
        project_id: str,
        role: str,
        content: str,
        *,
        attachments: Iterable[Mapping[str, Any]] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Path:
        message: dict[str, Any] = {"role": role, "content": content}
        if attachments is not None:
            message["attachments"] = list(attachments)
        if meta is not None:
            message["meta"] = dict(meta)
        return self.append_messages(project_id, (message,))

    def append_messages(
        self,
        project_id: str,
        messages: Iterable[Mapping[str, Any]],
    ) -> Path:
        timestamp = datetime.now(timezone.utc).isoformat()
        normalized = [
            {
                "role": "user" if str(message.get("role")) == "user" else "assistant",
                "content": str(message.get("content", "")).strip(),
                **(
                    {"attachments": json_safe(message.get("attachments"))}
                    if isinstance(message.get("attachments"), (list, tuple))
                    else {}
                ),
                **(
                    {"meta": json_safe(message.get("meta"))}
                    if isinstance(message.get("meta"), Mapping)
                    else {}
                ),
            }
            for message in messages
            if str(message.get("content", "")).strip()
        ]
        if not normalized:
            raise ValueError("At least one non-empty conversation message is required")
        payload = {
            "schema_version": "1.0",
            "project_id": project_id,
            "timestamp": timestamp,
            "messages": normalized,
        }
        digest = content_hash(payload, length=20)
        compact_time = re.sub(r"[^0-9]", "", timestamp)[:20]
        path = self.root / project_id / "conversation" / f"{compact_time}-{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
        except FileExistsError:
            pass
        from .conversation_checkpoints import ConversationCheckpoints
        checkpoints = ConversationCheckpoints(self.root)
        try:
            checkpoints.capture(project_id, path)
        except Exception as exc:
            # Optional archival must never lose a scientific chat or fail a job.
            try:
                checkpoints.record_failure(project_id, path, exc)
            except OSError:
                pass
        return path

    def messages(self, project_id: str) -> list[Mapping[str, Any]]:
        messages: list[Mapping[str, Any]] = []
        directory = self.root / project_id / "conversation"
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            timestamp = payload.get("timestamp", "")
            payload_messages = payload.get("messages", [])
            for index, message in enumerate(payload_messages):
                if isinstance(message, Mapping):
                    checkpoint = self.root / project_id / "chat_checkpoints" / path.name
                    meta = dict(message.get("meta") or {})
                    if index == len(payload_messages) - 1 and checkpoint.is_file():
                        meta["conversation_checkpoint"] = path.stem
                        meta["checkpoint_project_id"] = project_id
                    messages.append({**dict(message), "timestamp": timestamp, "meta": meta})
        return messages


@dataclass(frozen=True, slots=True)
class _TaskProxy:
    capability: str
    title: str


def _results_by_capability(project: Any) -> dict[str, Mapping[str, Any]]:
    task_records = evidence_by_task(project)
    results: dict[str, Mapping[str, Any]] = {}
    for task in getattr(project, "tasks", ()):
        record = task_records.get(str(task.id))
        result = record.get("structured_result") if isinstance(record, Mapping) else None
        if isinstance(result, Mapping):
            results[str(task.capability)] = result
    return results


def _lattice_elements(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    elements = _mapping_list(result.get("lattice_elements"))
    if elements:
        return elements
    artifact = result.get("lattice_artifact")
    return _mapping_list(artifact.get("elements")) if isinstance(artifact, Mapping) else []


def _is_proxy_ring(result: Mapping[str, Any]) -> bool:
    if str(result.get("design_scope", "")) == "homogeneous_recurring_cell_proxy":
        return True
    artifact = result.get("lattice_artifact")
    if not isinstance(artifact, Mapping):
        return False
    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    if str(metadata.get("design_scope", "")) == "homogeneous_recurring_cell_proxy":
        return True
    try:
        return (
            str(metadata.get("construction", "")) == "exact periodic repetition"
            and int(metadata.get("repetitions", 1))
            > int(metadata.get("source_periodicity", metadata.get("repetitions", 1)))
        )
    except (TypeError, ValueError):
        return False


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _validator(evidence: Mapping[str, Any]) -> str:
    validation = evidence.get("deterministic_validation")
    if isinstance(validation, Mapping) and validation.get("passed") is True:
        return str(validation.get("validator", "the deterministic gate"))
    return ""


def _first_nested(stages: Any, key: str) -> Any:
    if not isinstance(stages, (list, tuple)):
        return None
    for stage in stages:
        metrics = stage.get("metrics") if isinstance(stage, Mapping) else None
        if isinstance(metrics, Mapping) and key in metrics:
            return metrics[key]
    return None


def _max_abs(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [abs(float(row[key])) for row in rows if _finite(row.get(key))]
    return max(values) if values else None


def _pair(value: Any) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)):
        return "n/a", "n/a"
    return (
        _number(value[0] if len(value) else None, 5),
        _number(value[1] if len(value) > 1 else None, 5),
    )


def _metric(rows: list[dict[str, str]], name: str, value: Any, unit: str, digits: int) -> None:
    if _finite(value):
        rows.append({"quantity": name, "value": _number(value, digits), "unit": unit})


def _number(value: Any, digits: int = 3) -> str:
    if not _finite(value):
        return "n/a"
    number = float(value)
    if digits == 0:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def _scientific(value: Any) -> str:
    return f"{float(value):.3e}" if _finite(value) else "n/a"


def _percent(value: Any, digits: int) -> str:
    return f"{100.0 * float(value):.{digits}f}%" if _finite(value) else "n/a"


def _micrometres(value: Any) -> str:
    return f"{1.0e6 * float(value):,.3f} µm" if _finite(value) else "n/a"




def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
