"""Bounded handoff of background results and coding failures to the research agent."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_hash
from .failure_diagnostics import execution_failure_summary
from .presentation import ResearchConversationStore
from .research_environment import ResearchEnvironment
from .workspace_inspection import saved_recovery_context

_LEGACY_LATTICE_SCOPE_ERRORS = (
    "ValueError: lattice does not declare an integer periodic-cell partition",
    "ValueError: invalid cell_element_count in the lattice archive",
)

# Saved programs may have failed before progress payloads had a dedicated type.
# Match only reporting contracts, never general ValueError or filesystem errors.
_LEGACY_REPORTING_INPUT_ERRORS = (
    "ValueError: progress kind must be plan, progress, finding, decision, repair, or next",
    "ValueError: scientific_role must be milestone or activity",
    "ValueError: each progress table must be a mapping with a rows list",
    "ValueError: progress table must contain a rows list",
    "ValueError: every progress table row must be a mapping of column names",
    "ValueError: each progress plot must be a path or a mapping with a path",
    "ValueError: progress plot must be stored below workspace/artifacts",
    "ValueError: progress event_key must not be empty",
    "ValueError: status must be completed, infeasible, or partial",
)


def _reporting_input_failure(headline: str) -> bool:
    return bool(
        re.match(r"^(?:\w+\.)*ProgressInputError:", headline)
        or headline in _LEGACY_REPORTING_INPUT_ERRORS
        or re.fullmatch(r"ValueError: progress plot does not exist: [^\r\n]+", headline)
    )


def scientist_request(root: Path, project_id: str) -> dict[str, str]:
    rows = ResearchConversationStore(root).messages(project_id)
    row = next((item for item in reversed(rows) if item.get("role") == "user"), None)
    if row is None:
        return {}
    return {"id": content_hash({"timestamp": row.get("timestamp"), "content": row["content"]}),
            "content": str(row["content"])}


def coding_failure_summary(error: str) -> str | None:
    # Compatibility hints for familiar API mistakes, NOT an admission gate.
    # Every failed saved program can receive diagnosis; the agent must establish
    # the actual cause before deciding whether a code repair is appropriate.
    matches = re.findall(r"(?m)^\s*((?:TypeError|AttributeError|KeyError|NameError|UnboundLocalError|SyntaxError|IndentationError|(?:\w+\.)*(?:OptimizerInputError|LatticeInputError|LatticeTopologyError|LatticeScopeError|ProgressInputError)):[^\r\n]+)", error)
    if matches:
        return matches[-1][:800]
    progress = [line.strip() for line in error.splitlines()
                if _reporting_input_failure(line.strip())]
    if progress:
        return progress[-1][:800]
    # Archived programs predate the typed cell/full-ring scope contract. Only
    # these exact metadata errors are safe to route to a coding repair; no
    # physical periodic-solution or bend-closure failures belong in this branch.
    scopes = re.findall(
        r"(?m)^\s*(" + "|".join(re.escape(item) for item in _LEGACY_LATTICE_SCOPE_ERRORS)
        + r")\s*$", error,
    )
    if scopes:
        return scopes[-1]
    # These exact pre-typed editor errors also appear in saved failed programs.
    # Do not classify general topology/solver ValueErrors as programming errors.
    topology = re.findall(r"(?m)^\s*(ValueError: (?:JuTrack row synchronization cannot (?:change topology|reorder elements)|Dense optics proposals cannot change element order, names, or kinds))\s*$", error)
    if topology:
        return topology[-1]
    # These are optimizer input-contract errors, not failure to find a physical
    # solution. Other ValueError failures are deliberately not classified here.
    contracts = re.findall(r"(?m)^\s*(ValueError: (?:named bounds must|bounds must|initial_position must|named initial_position must|variable_names must|objective_names must|optimization checkpoint does not match|(?:population_size|generations)=\d+ is below the declared production minimum)[^\r\n]+)", error)
    if not contracts:
        # Compatibility for archived pre-typed constructor failures only; do
        # not classify ordinary solver ValueErrors as coding mistakes.
        contracts = re.findall(r"(?m)^\s*(ValueError: [^\r\n]+ length_m must be non-negative)\s*$", error)
    return contracts[-1][:800] if contracts else None


def program_failure_guidance(error: str) -> dict[str, str]:
    """Plain-language explanation and a narrow, non-physics repair instruction."""
    headline = execution_failure_summary(error)
    if headline in {"KeyError: 'boundary_optics'", 'KeyError: "boundary_optics"'}:
        return {
            "explanation": "The program tried to report boundary optics that this evaluation did not provide. "
                "A rejected or unstable candidate may legitimately have no periodic optics.",
            "repair_hint": "Inspect the exact evaluation's stable/passed flags and rejection reason first. "
                "Use api.evaluation_snapshot(evaluation) for reporting, including editor.evaluate_candidate results; "
                "unavailable boundary values are None, never zero. Check boundary_optics_available. "
                "Save selected editor evaluations with output_path before optional formatting. "
                "Do not report an optimizer finalist as a matched lattice unless its measured checks pass. "
                "Recover saved outputs before deciding whether another numerical experiment is necessary.",
        }
    if _reporting_input_failure(headline):
        return {
            "explanation": "The generated program could not publish a progress or result update because "
                "its reporting arguments were invalid. This is a reporting API error, not evidence "
                "that the accelerator calculation or lattice failed.",
            "repair_hint": "Inspect the exact saved program and diagnostic in this project. "
                "Correct only the reporting call: run.progress uses kind for the message category "
                "and scientific_role for milestone/activity; tables need a rows list of mappings, "
                "plots must reference real files below workspace/artifacts, and event_key must not "
                "be empty. run.finish status is completed, infeasible, or partial; an intermediate "
                "stage remains partial, not a completed request. Do not invent output files or "
                "suppress validation. Preserve the saved "
                "TME or other candidate, its full-precision fields, calculation outputs and optimizer "
                "checkpoints. Recover the exact saved stage and publish its existing results before "
                "continuing the original request. Do not regenerate a seed, repeat a completed search, "
                "change physics requirements, or claim new numerical evidence from a reporting repair. "
                "If needed results were never saved, disclose that limitation before planning a "
                "bounded recovery; do not claim the checkpoint contains them.",
        }
    if "LatticeScopeError:" in headline or headline in _LEGACY_LATTICE_SCOPE_ERRORS:
        return {
            "explanation": "The generated program's lattice metadata does not consistently say "
                "whether the saved rows are one periodic cell or an explicitly repeated full ring. "
                "Brilliance could not safely select one cell. This is a data/API setup error, "
                "not evidence that the accelerator lattice is unstable or infeasible.",
            "repair_hint": "Inspect the exact archived candidate, its construction source and the "
                "machine contract in this project. Establish whether those physical rows are one "
                "cell or a full ring before correcting metadata. For a confirmed single cell use "
                "api.periodic_cell(..., intended_repetitions=P); for a full ring built from a "
                "confirmed cell use api.cell_ring(..., periods=P). Preserve every physical element "
                "and its fields, original machine requirements, accepted parent and checkpoints. "
                "Do not infer scope from bend angle alone, borrow another project's lattice, or "
                "pad, truncate, slice or repeat rows arbitrarily to make a partition pass. If scope "
                "cannot be established from saved provenance, report the missing evidence instead "
                "of guessing. If only setup or reporting failed, correct that step and recover "
                "the saved numerical results; do not regenerate a seed or restart optimization.",
        }
    if "below the declared production minimum" in headline:
        return {
            "explanation": "The generated optimizer setup does not meet its declared population or "
                "generation budget. This is a setup error, not evidence that the lattice is infeasible.",
            "repair_hint": "Keep the saved matched candidate and original machine limits. Before "
                "loading JuTrack, inspect design.as_dict() and call api.plan_lattice_optimization "
                "with the compiled outer and matching parameterizations. Show the actual knob table. "
                "Check which element fields were intended: fields=('length_m', 'k1_m2') frees BOTH "
                "fields on eligible quadrupoles. Use included_element_names_by_field for explicit "
                "field ownership. Do not silently add knobs, lower minimum_population_per_knob, "
                "relax physics limits, regenerate the unit, or restart completed optimization. "
                "Size the requested campaign from the verified plan and report the corrected budget.",
        }
    topology = ("LatticeTopologyError:" in headline or headline.startswith((
        "ValueError: JuTrack row synchronization cannot",
        "ValueError: Dense optics proposals cannot change element order")))
    if topology:
        return {
            "explanation": "The generated code changed the element list but reused a simulation object "
                "built for the original layout. It cannot update that different layout in place. "
                "This is a coding/API mismatch, not an unstable lattice or a rejected design.",
            "repair_hint": "Keep the physical candidate and its completed optimization checkpoint. "
                "For bend-centre/drift-interior diagnostics, use api.build_jutrack_profile_sampler "
                "on the UNSPLIT selected candidate and call sampler.linear_metrics(sample_fractions="
                "{physical_element_name: [0.5]}). This returns requested_sample/source_element/"
                "sample_fraction in twiss_profile. Plot that profile with the unsplit physical rows. "
                "For a genuine insertion/removal/rename/type change, build a separate new editor from "
                "the changed candidate instead of reusing the old one. Never remove the topology guard. "
                "If only postprocessing failed, read api.read_optimization_checkpoint(path), reconstruct "
                "the selected rows using the saved program and best_candidate['variables'], replay that "
                "one candidate and finish reporting; do not repeat the population search.",
        }
    if coding_failure_summary(headline):
        return {"explanation": "The generated research program used an invalid argument, variable or API call. "
                "This is a coding error, not a physics conclusion.", "repair_hint": ""}
    return {"explanation": "", "repair_hint": ""}


def repair_pause_explanation(recovery: Mapping[str, Any]) -> str:
    """Never leave the user with an unexplained 'repair was not started'."""
    reason = recovery.get("reason")
    if reason in {"request_budget_exhausted", "auto_continue_disabled"}:
        from .research_continuation import pause_explanation
        return pause_explanation(reason)
    if reason == "repair_handoff_failed":
        reason = "handoff_failed"
    reasons = {
        "repair_disabled": "Automatic diagnosis and repair is disabled by recovery.max_attempts_per_request=0 in the runtime TOML. "
            "Enable it to permit bounded recovery, or ask the agent to inspect the saved program.",
        "repair_budget_exhausted": "Automatic diagnosis and repair reached its configured limit for this request. "
            "The saved diagnostic needs review before another attempt; completed calculations were not reset.",
        "same_error_budget_exhausted": "Edited attempts keep encountering the same error. "
            "Automatic repair paused at recovery.max_attempts_per_error; inspect the saved diagnosis before retrying.",
        "same_source_failed_again": "The same unchanged program and inputs already failed with this error. "
            "Another automatic handoff was not started. Review the diagnosis and change the faulty code "
            "or inputs before retrying; do not repeat a completed calculation.",
        "scientist_paused_or_cancelled": "Automatic repair was not started because you paused or stopped "
            "this research. Send a new instruction when you want it to resume.",
        "request_changed_or_not_recorded": "Automatic repair was not started because the original request "
            "was superseded or could not be confirmed. Your latest instruction takes precedence.",
        "handoff_failed": "Brilliance could not start the repair handoff. The handoff diagnostic was saved; "
            "ask the agent to inspect this failed job before retrying.",
        "not_a_saved_program": "This failed operation is not a saved research program. "
            "Its owning service needs diagnosis; no unrelated program was restarted.",
        "program_outside_workspace": "The failed program path is outside this project's workspace. "
            "Correct its project ownership before automatic recovery can proceed.",
        "not_a_repairable_program_error": "This historical failure was stopped by the previous "
            "exception-classification policy. Reopen the project with the updated application and "
            "request diagnosis from its saved checkpoint; no numerical result has been discarded.",
    }
    return reasons.get(reason, "Automatic repair could not be scheduled. Ask the agent to inspect the "
                       "saved diagnostic before retrying; no further calculation was started.")


def _committed_saved_program(path: Path, directory: Path) -> dict[str, Any] | None:
    """Read only a validated, archived saved-program execution in this project."""
    try:
        resolved = path.resolve()
        resolved.relative_to((directory / "workspace" / "artifacts").resolve())
        if not resolved.name.startswith("expert-execution-") or resolved.suffix != ".json":
            return None
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if value.get("tool_id") not in {
            "accelerator.run_research_program", "accelerator.run_long_research_program"
        } or value.get("validation", {}).get("passed") is not True:
            return None
        if value.get("project_id", directory.name) != directory.name:
            return None
        program = value.get("result", {}).get("program", {})
        snapshot = Path(str(program.get("snapshot", "")))
        if not snapshot.is_absolute():
            snapshot = directory / snapshot
        snapshot.resolve().relative_to((directory / "workspace").resolve())
        if not program.get("sha256") or not snapshot.is_file():
            return None
        return value
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def latest_saved_program_evidence(directory: Path) -> tuple[Path, dict[str, Any]] | None:
    """Locate the newest committed program for an internal result-review handoff."""
    paths = sorted((directory / "workspace" / "artifacts").glob("expert-execution-*.json"),
                   key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in paths:
        execution = _committed_saved_program(path, directory)
        if execution is not None:
            return path, execution
    return None


def schedule_result_review(root: Path, jobs: Any, job: Any, project: Any,
                           artifact: Path, execution: Mapping[str, Any], *,
                           scientist_origin: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Bring completed durable evidence back to the agent, without choosing physics."""
    from .codex_runtime import CodexRuntimeStore
    from .atomic_io import atomic_write_text
    from .research_continuation import handoff_payload, turn_allowance
    from .research_session import SessionWorkspace

    native_boundary = job.action == "codex_turn" and scientist_origin is not None
    if not native_boundary and job.payload.get("tool_id") != "accelerator.run_long_research_program":
        return {"scheduled": False, "reason": "not_a_saved_program"}
    root = Path(root).resolve()
    directory = root / project.id
    origin = dict(scientist_origin) if native_boundary else job.payload.get("scientist_request", {})
    if not origin or origin.get("id") != scientist_request(root, project.id).get("id"):
        return {"scheduled": False, "reason": "request_changed_or_not_recorded"}
    if SessionWorkspace(directory).pause_file.exists() or jobs.cancellation_requested(project.id, job.job_id):
        return {"scheduled": False, "reason": "scientist_paused_or_cancelled"}
    records = directory / "workspace" / "result_reviews"
    marker = records / f"{job.job_id}.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    active = jobs.active(project.id, action="codex_turn")
    allowance = turn_allowance(root, project.id, origin) if active is None else None
    if allowance is not None and not allowance["allowed"]:
        return {"scheduled": False, "reason": allowance["reason"]}
    previous = [json.loads(path.read_text(encoding="utf-8")) for path in records.glob("*.json")]
    used = sum(item.get("scientist_request_id") == origin["id"] for item in previous)
    # Honor an explicitly configured legacy cap, but do not impose a second
    # hidden eight-review budget on top of the shared request allowance.
    legacy_limit = os.environ.get("BRILLIANCE_RESULT_REVIEW_LIMIT")
    try:
        legacy_limit = max(0, int(legacy_limit)) if legacy_limit is not None else None
    except ValueError:
        legacy_limit = None
    if legacy_limit is not None and used >= legacy_limit:
        return {"scheduled": False, "reason": "result_review_limit"}
    if native_boundary:
        committed = _committed_saved_program(artifact, directory)
        if committed is None:
            return {"scheduled": False, "reason": "no_validated_saved_program"}
        # Use committed evidence, never a caller-supplied substitute result.
        execution = committed
    result = execution.get("result", {})
    report = result.get("result", result)
    introduction = (
        "The internal model/tool-step slice ended with committed numerical evidence and unfinished work. "
        "This is an internal result handoff at a controller boundary, not a scientist decision or "
        "a new user instruction; newer scientist instructions take precedence.\n"
    ) if native_boundary else (
        "A background research program has completed. This is an internal result handoff, "
        "not a new user instruction; newer scientist instructions take precedence.\n"
    )
    prompt = (
        introduction + f"Original request: {origin['content']}\n"
        f"Evidence: {artifact.relative_to(directory).as_posix()}\n"
        f"Summary: {report.get('summary', '')}\n"
        f"Program declared request_satisfied={report.get('request_satisfied')}; "
        f"validation={execution.get('validation', {})}\n"
        "Use the saved project evidence and existing artifacts. Explain the actual numerical "
        "result, its meaning, and limitations in chat. Do not rerun a completed calculation just "
        "to read its result. Check the scope: a component or seed is not the requested complete "
        "system. If work within the original request remains, perform the next justified step; "
        "if the request is answered, give the final answer. If a scientific decision is needed, "
        "ask a specific question. A negative matching/search outcome is not a coding error: "
        "inspect its residuals, bounds and knob ownership; do not rerun the unchanged failed attempt "
        "or relax user requirements. Do not choose a new architecture merely because a short search "
        "missed an aspiration. Do not install packages or change the configured runtime."
    )
    record = {"scheduled": False, "reason": "handoff_pending", "scientist_request_id": origin["id"],
              "review": used + 1, "artifact": str(artifact),
              "trigger": "native_step_boundary" if native_boundary else "durable_program_result"}
    records.mkdir(parents=True, exist_ok=True)
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(record, handle)
    except FileExistsError:
        return json.loads(marker.read_text(encoding="utf-8"))
    try:
        if active is not None:
            CodexRuntimeStore(root, project.id).queue_input(prompt)
        else:
            dispatched = jobs.submit(project.id, project.revision, action="codex_turn", payload={
                "message": prompt, "initial": False, **handoff_payload(allowance),
                "research_mode": "result_review", "review_of": job.job_id, "repair_request_id": origin["id"],
            })
            if dispatched.payload.get("review_of") != job.job_id:
                CodexRuntimeStore(root, project.id).queue_input(prompt)
        record.update(scheduled=True, reason="")
        atomic_write_text(marker, json.dumps(record))
    except Exception:
        marker.unlink(missing_ok=True)
        raise
    return record


def schedule_coding_repair(root: Path, jobs: Any, job: Any, project: Any,
                           failure_path: Path, error: str) -> dict[str, Any]:
    """Return any failed saved execution for bounded diagnosis and possible repair.

    Exception names supply hints, never permission to retry, change requirements
    or repair the platform. Cancellation, scope and loop guards own admission.
    """
    from .codex_runtime import CodexRuntimeStore
    from .atomic_io import atomic_write_text
    from .research_continuation import handoff_payload, turn_allowance
    from .research_session import SessionWorkspace

    if str(job.payload.get("tool_id")) not in {
        "accelerator.run_research_program", "accelerator.run_long_research_program"
    }:
        return {"scheduled": False, "reason": "not_a_saved_program"}
    headline = execution_failure_summary(error)
    root = Path(root).resolve()
    origin = job.payload.get("scientist_request", {})
    latest = scientist_request(root, project.id)
    if not origin or origin.get("id") != latest.get("id"):
        return {"scheduled": False, "reason": "request_changed_or_not_recorded"}
    directory = root / project.id
    if SessionWorkspace(directory).pause_file.exists() or jobs.cancellation_requested(project.id, job.job_id):
        return {"scheduled": False, "reason": "scientist_paused_or_cancelled"}
    policy = ResearchEnvironment.load()
    budget = policy.repair_max_attempts
    if budget == 0:
        return {"scheduled": False, "reason": "repair_disabled"}
    records = directory / "workspace" / "runtime_failures"
    marker = records / f"{job.job_id}.recovery.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    active = jobs.active(project.id, action="codex_turn")
    allowance = turn_allowance(root, project.id, origin) if active is None else None
    if allowance is not None and not allowance["allowed"]:
        return {"scheduled": False, "reason": allowance["reason"]}
    previous = [json.loads(path.read_text(encoding="utf-8")) for path in records.glob("*.recovery.json")]
    attempts = [item for item in previous if item.get("scientist_request_id") == origin["id"]]
    arguments = job.payload.get("arguments", {})
    script = str(arguments.get("script", ""))
    workspace = directory / "workspace"
    relative = Path(script)
    if not relative.is_absolute() and relative.parts and relative.parts[0].casefold() == "workspace":
        relative = Path(*relative.parts[1:])
    source = (workspace / relative).resolve()
    if not source.is_relative_to(workspace):
        return {"scheduled": False, "reason": "program_outside_workspace"}
    try:
        source_hash = content_hash(source.read_text(encoding="utf-8")) if source.is_file() else "unavailable"
    except OSError:
        source_hash = "unavailable"  # A source access error still warrants read-only diagnosis.
    arguments_hash = content_hash(arguments.get("arguments", {}))
    failure_key = content_hash({"source": source_hash, "script": source.relative_to(workspace).as_posix(),
                                "arguments": arguments_hash, "error": headline})
    if any(item.get("failure_key") == failure_key for item in attempts):
        return {"scheduled": False, "reason": "same_source_failed_again"}
    # Moving to a distinct exception is observable debugging progress, not proof
    # of physics progress. It receives its own allowance within the total cap.
    error_key = content_hash(re.sub(r"0x[0-9a-fA-F]+", "<address>", headline))
    repeated = sum(item.get("error_key", content_hash(item.get("summary", ""))) == error_key
                   for item in attempts)
    if repeated >= policy.repair_max_same_error:
        return {"scheduled": False, "reason": "same_error_budget_exhausted"}
    if len(attempts) >= budget:
        return {"scheduled": False, "reason": "repair_budget_exhausted"}
    prompt = (
        "Automatic diagnosis and possible repair of a failed research execution, not a new scientist request. "
        "Any newer scientist instruction takes precedence over this repair notice.\n"
        f"Original request: {origin['content']}\n"
        f"Program: {script}\nObserved exception: {headline}\n"
        f"Diagnostic: {failure_path.relative_to(directory).as_posix()}\n"
        f"Recovery attempt {len(attempts) + 1}/{budget}. Inspect the traceback, exact saved source, "
        "arguments, numerical outcomes and checkpoints BEFORE choosing a change. An exception name "
        "does not establish the cause. Explain what failed, which results survive, and the smallest "
        "justified next action in chat.\n"
        "For a demonstrated coding/API/reporting error, repair project-local code or inputs and verify "
        "the smallest affected step with the configured runtime. Do not rerun an unchanged failed attempt. "
        "Preserve successful evidence and optimizer checkpoints; if only postprocessing failed, recover "
        "the saved result without repeating optimization. Never replace a missing numerical value with "
        "a fabricated result or suppress an exception as a physics success.\n"
        "For an unsuccessful numerical search, inspect actual feasibility and residuals before "
        "postprocessing a selected candidate. An infeasible archive is not a valid design. A missing "
        "feasible solution is evidence to diagnose and report, not permission to remove a physics gate. "
        "Distinguish original hard requirements from model-invented conditions: a reference scale or "
        "aspiration must not silently become a validity gate. Correct a mistaken objective or gate only "
        "with evidence from the original requirements and applicable physics knowledge; a changed "
        "numerical problem needs a new campaign identity and freshly evaluated objective values. "
        "A justified next scientific experiment must remain within the original request and limits; "
        "do not relaunch a large campaign solely to repair its reporting.\n"
        "For runtime/access/credential/resource problems, perform read-only diagnosis first. Do not "
        "install packages, change interpreters or Julia configuration, permissions, credentials, "
        "internet policy, or physical requirements. If recovery needs new authority or a scientist "
        "decision, preserve the checkpoint and ask one specific question. Internet research follows "
        "the configured network policy, including during recovery. Do not substitute a stored template. "
        "After a repair, resume the original authorized research; recovery alone does not fulfill it. "
        "Unknown errors are to be investigated, not automatically "
        "declared unrepairable."
        "\n" + program_failure_guidance(error)["repair_hint"] +
        "\n" + saved_recovery_context(directory / "workspace")
    )
    record = {"scheduled": True, "scientist_request_id": origin["id"], "failure_key": failure_key,
              "attempt": len(attempts) + 1, "source_sha": source_hash, "summary": headline,
              "arguments_sha": arguments_hash, "mode": "diagnose_and_repair",
              "error_key": error_key, "error_attempt": repeated + 1,
              "policy": {"source": str(policy.config_path), "max_attempts_per_request": budget,
                         "max_attempts_per_error": policy.repair_max_same_error}}
    records.mkdir(parents=True, exist_ok=True)
    # Claim the handoff once even if the completion notification is delivered twice.
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
    except FileExistsError:
        return json.loads(marker.read_text(encoding="utf-8"))
    try:
        if active is not None:
            CodexRuntimeStore(root, project.id).queue_input(prompt)
        else:
            dispatched = jobs.submit(project.id, project.revision, action="codex_turn", payload={
                "message": prompt, "initial": False, **handoff_payload(allowance),
                "research_mode": "research", "recovery_mode": "diagnose_and_repair", "repair_of": job.job_id,
                "repair_request_id": origin["id"],
            })
            if dispatched.payload.get("repair_of") != job.job_id:
                CodexRuntimeStore(root, project.id).queue_input(prompt)
        atomic_write_text(marker, json.dumps(record))
    except Exception:
        marker.unlink(missing_ok=True)
        raise
    return record
