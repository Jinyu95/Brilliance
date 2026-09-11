"""Bounded handoff of background results and coding failures to the research agent."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_hash
from .presentation import ResearchConversationStore
from .workspace_inspection import saved_recovery_context


def scientist_request(root: Path, project_id: str) -> dict[str, str]:
    rows = ResearchConversationStore(root).messages(project_id)
    row = next((item for item in reversed(rows) if item.get("role") == "user"), None)
    if row is None:
        return {}
    return {"id": content_hash({"timestamp": row.get("timestamp"), "content": row["content"]}),
            "content": str(row["content"])}


def coding_failure_summary(error: str) -> str | None:
    # Do not send infrastructure, cancellation, or a physical rejection through
    # a code-repair loop. Keep the complete traceback in the diagnostic artifact.
    matches = re.findall(r"(?m)^\s*((?:TypeError|AttributeError|KeyError|NameError|UnboundLocalError|SyntaxError|IndentationError|(?:\w+\.)*(?:OptimizerInputError|LatticeInputError|LatticeTopologyError)):[^\r\n]+)", error)
    if matches:
        return matches[-1][:800]
    # These exact pre-typed editor errors also appear in saved failed programs.
    # Do not classify general topology/solver ValueErrors as programming errors.
    topology = re.findall(r"(?m)^\s*(ValueError: (?:JuTrack row synchronization cannot (?:change topology|reorder elements)|Dense optics proposals cannot change element order, names, or kinds))\s*$", error)
    if topology:
        return topology[-1]
    # These are optimizer input-contract errors, not failure to find a physical
    # solution. Other ValueError failures are deliberately not classified here.
    contracts = re.findall(r"(?m)^\s*(ValueError: (?:named bounds must|bounds must|initial_position must|named initial_position must|variable_names must|objective_names must|optimization checkpoint does not match)[^\r\n]+)", error)
    if not contracts:
        # Compatibility for archived pre-typed constructor failures only; do
        # not classify ordinary solver ValueErrors as coding mistakes.
        contracts = re.findall(r"(?m)^\s*(ValueError: [^\r\n]+ length_m must be non-negative)\s*$", error)
    return contracts[-1][:800] if contracts else None


def execution_failure_summary(error: str) -> str:
    """Keep the actual non-repairable error visible without dumping its traceback."""
    matches = re.findall(r"(?m)^\s*([A-Za-z_][\w.]*(?:Error|Exception):[^\r\n]+)", error)
    return matches[-1][:800] if matches else error.splitlines()[0][:800] if error.strip() else "Calculation could not complete"


def program_failure_guidance(error: str) -> dict[str, str]:
    """Plain-language explanation and a narrow, non-physics repair instruction."""
    headline = coding_failure_summary(error) or execution_failure_summary(error)
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
    if coding_failure_summary(error):
        return {"explanation": "The generated research program used an invalid argument, variable or API call. "
                "This is a coding error, not a physics conclusion.", "repair_hint": ""}
    return {"explanation": "", "repair_hint": ""}


def repair_pause_explanation(recovery: Mapping[str, Any]) -> str:
    """Never leave the user with an unexplained 'repair was not started'."""
    reason = recovery.get("reason")
    if reason == "repair_handoff_failed":
        reason = "handoff_failed"
    reasons = {
        "repair_disabled": "Automatic code repair is disabled by BRILLIANCE_AUTO_REPAIR_ATTEMPTS=0. "
            "Enable it to permit bounded coding repairs, or ask the agent to inspect the saved program.",
        "repair_budget_exhausted": "Automatic code repair reached its safe limit for this request. "
            "The diagnostic needs review before another attempt; the same calculation was not restarted.",
        "same_source_failed_again": "The same unchanged program already failed with this error. "
            "It was not rerun again. Ask the agent to inspect and correct the saved source before retrying.",
        "scientist_paused_or_cancelled": "Automatic repair was not started because you paused or stopped "
            "this research. Send a new instruction when you want it to resume.",
        "request_changed_or_not_recorded": "Automatic repair was not started because the original request "
            "was superseded or could not be confirmed. Your latest instruction takes precedence.",
        "handoff_failed": "Brilliance could not start the repair handoff. The handoff diagnostic was saved; "
            "ask the agent to inspect this failed job before retrying.",
        "not_a_repairable_program_error": "Automatic repair was not started because this failure has not "
            "been classified as a safely repairable program error. Ask the agent to inspect the saved "
            "diagnostic; runtime/access problems require attention rather than repeating the calculation.",
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
    from .jobs import launch_research_job
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
    try:
        limit = max(0, min(32, int(os.environ.get("BRILLIANCE_RESULT_REVIEW_LIMIT", "8"))))
    except ValueError:
        limit = 8
    previous = [json.loads(path.read_text(encoding="utf-8")) for path in records.glob("*.json")]
    used = sum(item.get("scientist_request_id") == origin["id"] for item in previous)
    if used >= limit:
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
    record = {"scheduled": True, "scientist_request_id": origin["id"],
              "review": used + 1, "artifact": str(artifact),
              "trigger": "native_step_boundary" if native_boundary else "durable_program_result"}
    records.mkdir(parents=True, exist_ok=True)
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(record, handle)
    except FileExistsError:
        return json.loads(marker.read_text(encoding="utf-8"))
    if jobs.active(project.id, action="codex_turn"):
        CodexRuntimeStore(root, project.id).queue_input(prompt)
    else:
        review = jobs.submit(project.id, project.revision, action="codex_turn", payload={
            "message": prompt, "initial": False, "increment": 1, "max_increments": 1,
            "research_mode": "result_review", "review_of": job.job_id, "repair_request_id": origin["id"],
        })
        launch_research_job(root, review)
    return record


def schedule_coding_repair(root: Path, jobs: Any, job: Any, project: Any,
                           failure_path: Path, error: str) -> dict[str, Any]:
    from .codex_runtime import CodexRuntimeStore
    from .jobs import launch_research_job
    from .research_session import SessionWorkspace

    headline = coding_failure_summary(error)
    if not headline or str(job.payload.get("tool_id")) not in {
        "accelerator.run_research_program", "accelerator.run_long_research_program"
    }:
        return {"scheduled": False, "reason": "not_a_repairable_program_error"}
    origin = job.payload.get("scientist_request", {})
    latest = scientist_request(root, project.id)
    if not origin or origin.get("id") != latest.get("id"):
        return {"scheduled": False, "reason": "request_changed_or_not_recorded"}
    directory = root / project.id
    if SessionWorkspace(directory).pause_file.exists() or jobs.cancellation_requested(project.id, job.job_id):
        return {"scheduled": False, "reason": "scientist_paused_or_cancelled"}
    try:
        budget = min(3, max(0, int(os.environ.get("BRILLIANCE_AUTO_REPAIR_ATTEMPTS", "2"))))
    except ValueError:
        budget = 2
    if budget == 0:
        return {"scheduled": False, "reason": "repair_disabled"}
    records = directory / "workspace" / "runtime_failures"
    marker = records / f"{job.job_id}.recovery.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    previous = [json.loads(path.read_text(encoding="utf-8")) for path in records.glob("*.recovery.json")]
    attempts = [item for item in previous if item.get("scientist_request_id") == origin["id"]]
    if len(attempts) >= budget:
        return {"scheduled": False, "reason": "repair_budget_exhausted"}
    arguments = job.payload.get("arguments", {})
    script = str(arguments.get("script", ""))
    source = (directory / "workspace" / script).resolve()
    source_hash = content_hash(source.read_text(encoding="utf-8")) if source.is_relative_to(directory / "workspace") and source.is_file() else "unavailable"
    failure_key = content_hash({"source": source_hash, "error": headline})
    if any(item.get("failure_key") == failure_key for item in attempts):
        return {"scheduled": False, "reason": "same_source_failed_again"}
    prompt = (
        "Automatic recovery of a background PROGRAMMING error, not a new scientist request. "
        "Any newer scientist instruction takes precedence over this repair notice.\n"
        f"Original request: {origin['content']}\n"
        f"Program: {script}\nExact error: {headline}\n"
        f"Diagnostic: {failure_path.relative_to(directory).as_posix()}\n"
        f"Repair attempt {len(attempts) + 1}/{budget}. Read the saved source, patch the coding error, "
        "and verify it with the configured runtime. Do not rerun unchanged source. Preserve successful "
        "evidence and optimizer checkpoints; if only postprocessing failed, recover the saved result "
        "without repeating optimization. Do not install packages, change interpreters, relax physical "
        "requirements, substitute templates, or search the internet. Show a concise repair update "
        "and then useful numerical results. If a repair needs a scientist decision, explain it and stop."
        "\n" + program_failure_guidance(error)["repair_hint"] +
        "\n" + saved_recovery_context(directory / "workspace")
    )
    record = {"scheduled": True, "scientist_request_id": origin["id"], "failure_key": failure_key,
              "attempt": len(attempts) + 1, "source_sha": source_hash, "summary": headline}
    records.mkdir(parents=True, exist_ok=True)
    # Claim the handoff once even if the completion notification is delivered twice.
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
    except FileExistsError:
        return json.loads(marker.read_text(encoding="utf-8"))
    active = jobs.active(project.id, action="codex_turn")
    if active is not None:
        CodexRuntimeStore(root, project.id).queue_input(prompt)
    else:
        recovery = jobs.submit(project.id, project.revision, action="codex_turn", payload={
            "message": prompt, "initial": False, "increment": 1, "max_increments": 1,
            "research_mode": "research", "repair_of": job.job_id,
            "repair_request_id": origin["id"],
        })
        launch_research_job(root, recovery)
    return record
