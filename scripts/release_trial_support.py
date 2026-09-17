"""Read-only provenance checks for fresh, unattended application release trials.

These are release-test assertions, not physics requirements sent to the model.
The manifest is not a security boundary against a developer who edits all logs.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from brilliance.codex_backends import describe_codex_backend
from brilliance.jobs import (
    WORKER_CLAIM_TIMEOUT_SECONDS,
    WORKER_START_GRACE_SECONDS,
    ResearchJobStore,
    _process_alive,
)

SCHEMA = "brilliance_fresh_release_trial/v1"
MANIFEST = "release_trial.json"
RENDER_CHECKS = {"exact_selected_result", "images_loaded", "real_tables_rendered",
                 "duplicate_images_absent", "browser_errors_absent", "external_requests_absent"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_fingerprints(environment) -> dict[str, str]:
    """Freeze application, expert knowledge, validation code and native source.

    No history, output, package caches, .env contents or credentials are copied.
    """
    source = Path(environment.source_root)
    files: set[Path] = set()
    for root in (source / "brilliance", source / "scripts", *environment.expert_paths,
                 Path(environment.jutrack_root) / "src",
                 Path(environment.jutrack_root) / "python_integration"):
        if root.is_dir():
            files.update(path.resolve() for path in root.rglob("*")
                         if path.is_file() and path.suffix in {".py", ".jl", ".md", ".toml", ".json", ".js", ".css", ".html"}
                         and not any(part in {"__pycache__", ".git", "node_modules", ".venv"} for part in path.parts))
    files.update(path.resolve() for path in (
        source / "pyproject.toml", source / "tests/scientific_history_ui.cjs",
        Path(environment.config_path), Path(environment.pyjutrack_module),
        Path(environment.julia_project) / "Project.toml", Path(environment.julia_project) / "Manifest.toml",
        Path(environment.python_julia_environment) / "Project.toml",
        Path(environment.python_julia_environment) / "Manifest.toml",
    ) if path.is_file())
    return {str(path): digest(path) for path in sorted(files)}


def configuration(environment) -> dict[str, Any]:
    """Persist public identity only; hash effective overrides without their values."""
    backend = describe_codex_backend()
    public = {key: backend.get(key) for key in (
        "key", "name", "provider_id", "model", "mode", "wire_api", "fingerprint", "configured")}
    overrides = {key: value for key, value in os.environ.items()
                 if key.startswith("BRILLIANCE_") and not key.startswith("BRILLIANCE_RUN_")
                 and not any(secret in key.upper() for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    return {"backend": public,
            "overrides_sha256": hashlib.sha256(json.dumps(overrides, sort_keys=True).encode()).hexdigest(),
            "runtime_config": str(environment.config_path),
            "python": str(environment.python_executable), "julia": str(environment.julia_executable),
            "jutrack": str(environment.jutrack_root), "internet_access": environment.internet_access,
            "remote_mcp_enabled": environment.remote_mcp_enabled,
            "local_mcp_enabled": environment.local_mcp_enabled}


def begin_manifest(root: Path, prompt: str, requirements: Mapping, environment) -> dict:
    """Require a new empty test root before creating the ordinary chat request."""
    root = root.resolve()
    source = Path(environment.source_root).resolve()
    if root == source or root == Path(environment.project_store).resolve() or root.is_relative_to(source / "research_projects"):
        raise ValueError("Use a new dedicated validation root, never user history or the repository root")
    if root.exists() and any(root.iterdir()):
        raise ValueError("A fresh release trial requires an empty root; do not reuse saved seeds or projects")
    if environment.internet_access or environment.remote_mcp_enabled:
        raise ValueError("Research internet/search and remote MCP must be disabled for this offline trial; model inference is allowed")
    root.mkdir(parents=True, exist_ok=True)
    value = {"schema": SCHEMA, "started_at": now(), "project_root": str(root),
             "prompt": prompt, "requirements": dict(requirements), "empty_root_at_start": True,
             "configuration": configuration(environment), "sources": source_fingerprints(environment),
             "operator_policy": "No developer steering, copied candidates, edited programs or model changes. Observe only; automatic application recovery is permitted.",
             "scope": "Fresh default/chat request to a first computed full MBA candidate; not optimization optimality or nonlinear qualification."}
    # Exclusive creation: never overwrite an earlier trial's provenance.
    with (root / MANIFEST).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
    return value


def fresh_checks(manifest: Mapping, directory: Path, environment) -> dict[str, bool]:
    root = directory.parent.resolve()
    return {"fresh_trial_manifest": manifest.get("schema") == SCHEMA,
            "empty_root_at_start": manifest.get("empty_root_at_start") is True,
            "one_new_project": [path.name for path in root.glob("project-*") if path.is_dir()] == [directory.name],
            "trial_root_bound": manifest.get("project_root") == str(root),
            "sources_frozen": manifest.get("sources") == source_fingerprints(environment),
            "configuration_frozen": manifest.get("configuration") == configuration(environment),
            "research_network_disabled": not environment.internet_access and not environment.remote_mcp_enabled}


def observe_job_liveness(directory: Path, jobs) -> dict:
    """Observe persisted worker PIDs without active()/recover() state mutation.

    An unlaunched queued job is not a dead worker. A claimed running job whose
    PID is absent/unobservable cannot be reported as ongoing numerical work.
    The existing platform-safe probe never sends a signal on Windows. It tests
    PID liveness, not process identity; a failed probe may also mean no access.
    """
    records, active, stale, unknown = [], [], [], []
    store = ResearchJobStore(directory.parent)
    for job in jobs:
        state = job.state.value
        if state not in {"running", "queued", "cancel_requested"}:
            continue
        row = {"job_id": job.job_id, "stored_state": state,
               "worker_log": f"jobs/{job.job_id}/worker.log"}
        pid = getattr(job, "worker_pid", None)
        if state == "queued":
            launches = store.launch_history(directory.name, job.job_id)
            launch = launches[-1] if launches else {}
            pid = launch.get("pid", pid)
            row["launch_recorded"] = bool(launch)
            if not launch:
                row.update(observation="queued_awaiting_launch", process_alive=None)
                active.append(job.job_id)
                records.append(row)
                continue
            try:
                started = datetime.fromisoformat(str(launch["started_at"]).replace("Z", "+00:00"))
                age = max(0.0, (datetime.now(timezone.utc) - started.replace(tzinfo=started.tzinfo or timezone.utc)).total_seconds())
            except (KeyError, TypeError, ValueError):
                age = None
            row["launch_age_s"] = age
        valid_pid = isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
        alive = _process_alive(pid) if valid_pid else None
        row.update(worker_pid=pid, process_alive=alive)
        if state in {"running", "cancel_requested"}:
            if alive is True:
                row["observation"] = "worker_observed_alive"
                active.append(job.job_id)
            elif alive is False:
                row["observation"] = "stale_worker_not_observed_alive"
                stale.append(job.job_id)
            else:
                row["observation"] = "running_worker_pid_missing"
                unknown.append(job.job_id)
        elif (alive is True and age is not None and age >= WORKER_CLAIM_TIMEOUT_SECONDS
              and not (directory / "jobs" / job.job_id / "claim.json").exists()):
            row["observation"] = "stale_worker_never_claimed_job"
            stale.append(job.job_id)
        elif alive is not True and age is not None and age >= WORKER_START_GRACE_SECONDS:
            row["observation"] = "stale_launch_no_live_worker"
            stale.append(job.job_id)
        else:
            row["observation"] = "queued_launch_pending"
            active.append(job.job_id)
        records.append(row)
    return {"observations": records, "active_job_ids": active,
            "stale_job_ids": stale, "unverifiable_job_ids": unknown,
            "read_only": True,
            "scope": "PID observation only; job revisions, launch/claim records and worker processes were not changed."}


def _program_attempt_counts(sessions: list[list[dict]]) -> dict[str, int]:
    """Count recorded outcomes, not job states or guesses from assistant prose."""
    counts = Counter(dict.fromkeys(("run_program_requests", "cached_results", "preflight_rejections",
        "api_rejections", "synchronous_execution_failures", "synchronous_execution_completions",
        "managed_submissions", "managed_submission_failures", "partial_scientific_results",
        "result_validation_failures", "unclassified_run_errors", "in_turn_execution_recoveries"), 0))
    for events in sessions:
        arguments: dict[str, dict] = {}
        failed: set[str] = set()
        patched: set[str] = set()
        for event in events:
            name = event.get("name")
            if event.get("kind") == "tool_call":
                arguments[name] = event.get("arguments", {})
                counts["run_program_requests"] += name == "run_program"
                continue
            if event.get("kind") != "tool_result":
                continue
            result = event.get("result", {})
            call = arguments.get(name, {})
            program_record = result.get("program")
            program = str(call.get("name") or (program_record.get("source", "") if isinstance(program_record, dict) else ""))
            program = program.replace("\\", "/").removeprefix("programs/")
            if name in {"write_program", "patch_program"} and result.get("ok") is True:
                if program in failed:
                    patched.add(program)
                continue
            if name != "run_program":
                continue
            if result.get("cached"):
                counts["cached_results"] += 1
                continue
            preflight = result.get("preflight") or {}
            if result.get("ok") is False:
                if preflight.get("passed") is False:
                    counts["preflight_rejections"] += 1
                elif result.get("kind") == "completed_execution":
                    counts["result_validation_failures"] += 1
                    failed.add(program)
                elif preflight.get("passed") is True:
                    # This return shape is emitted after a child-process error.
                    # A failed submit command is not proof its durable job ran.
                    background = call.get("background") or preflight.get("durable_required")
                    counts["managed_submission_failures" if background else "synchronous_execution_failures"] += 1
                    failed.add(program)
                elif str(result.get("error", "")).startswith(("arguments must be one object", "program does not exist:")):
                    counts["api_rejections"] += 1
                else:
                    counts["unclassified_run_errors"] += 1
            elif result.get("kind") == "managed_calculation":
                counts["managed_submissions"] += 1
            elif result.get("kind") == "completed_execution" and result.get("execution_valid") is True:
                counts["synchronous_execution_completions"] += 1
                counts["partial_scientific_results"] += result.get("scientific_passed") is not True
                if program in failed and program in patched:
                    counts["in_turn_execution_recoveries"] += 1
                    failed.discard(program)
                    patched.discard(program)
    return dict(counts)


def _optimizer_campaigns(directory: Path) -> list[dict]:
    """Recognize native checkpoint contents in both supported storage layouts.

    Do not recurse through per-evaluation caches or count .geometry sidecars,
    which may repeat the optimizer result without being another PSO campaign.
    """
    root = directory / "workspace/optimizations"
    campaigns = []
    for path in sorted(set(root.glob("*.json")) | set(root.glob("*/checkpoint.json"))):
        checkpoint = read_json(path)
        if (checkpoint.get("mode") not in {"single_objective", "multi_objective"}
                or checkpoint.get("contract", {}).get("optimizer") != "particle_swarm_optimize"
                or not isinstance(checkpoint.get("evaluation_count"), int)
                or not isinstance(checkpoint.get("completed_generation"), int)):
            continue
        history = checkpoint.get("history") or []
        campaigns.append({"campaign": path.stem if path.parent == root else path.parent.name,
                          "path": path.relative_to(directory).as_posix(),
                          "contract_fingerprint": checkpoint.get("contract_fingerprint"),
                          "evaluation_count": checkpoint["evaluation_count"],
                          "population_size": checkpoint["contract"].get("population_size"),
                          "completed_generation": checkpoint["completed_generation"],
                          "complete": checkpoint.get("complete"),
                          "initial_best": history[0].get("best") if history else None,
                          "final_best": history[-1].get("best") if history else None})
    return campaigns


def trace_audit(directory: Path, jobs, initial_request: str, executions: list[dict]) -> dict:
    """Count actual work and require saved program authorship from model traces.

    Missing authorship cannot be inferred from a plausible file. Other harness
    trace formats need a reader before earning the unattended-release claim.
    """
    sessions = [read_json(path).get("events", [])
                for path in sorted((directory / "workspace/agent_sessions").glob("*.json"))]
    events = [event for session in sessions for event in session]
    calls = [event for event in events if event.get("kind") == "tool_call"]
    # The preflight hash reflects normalized saved bytes (write_program adds a
    # final newline) and also accounts for successful model patch_program calls.
    authored = {event.get("result", {}).get("preflight", {}).get("sha256") for event in events
                if event.get("kind") == "tool_result" and event.get("name") in {"write_program", "patch_program"}
                and event.get("result", {}).get("ok") is True}
    programs = [execution["result"]["program"] for execution in executions
                if isinstance(execution.get("result", {}).get("program"), dict)]
    source_checks = []
    for program in programs:
        path = Path(str(program.get("snapshot", ""))).resolve()
        source_checks.append(path.is_relative_to(directory.resolve()) and path.is_file()
                             and digest(path) == program.get("sha256") and program.get("sha256") in authored)
    initial = [job for job in jobs if job.action == "codex_turn" and job.payload.get("initial") is True]
    chain = len(initial) == 1 and initial[0].payload.get("message") == initial_request
    job_ids = {job.job_id for job in jobs}
    for job in jobs:
        if job.action != "codex_turn" or job in initial:
            continue
        payload = job.payload
        derived = payload.get("review_of") or payload.get("repair_of")
        continuation = payload.get("increment", 1) > 1
        chain = chain and (derived in job_ids or continuation)
    counts = Counter(event.get("name") for event in calls)
    campaigns = _optimizer_campaigns(directory)
    reporting = {str(item.get("result", {}).get("run_id")): item["result"]["result"] for item in executions
                 if isinstance(item.get("result", {}).get("result", {}).get("data", {}).get("reporting_failure"), dict)}
    rejected = Counter(event.get("reason", "unspecified") for event in events if event.get("kind") == "model_output_rejected")
    return {"checks": {"no_unattributed_agent_turns": bool(chain),
                       "programs_authored_in_model_trace": bool(source_checks) and all(source_checks)},
            "job_counts": dict(Counter(job.state.value for job in jobs)),
            "agent_turns": sum(job.action == "codex_turn" for job in jobs),
            "managed_calculations": sum(job.action == "expert_tool" for job in jobs),
            "automatic_repairs": sum(bool(job.payload.get("repair_of")) for job in jobs),
            "automatic_result_reviews": sum(bool(job.payload.get("review_of")) for job in jobs),
            "program_attempts": _program_attempt_counts(sessions),
            "model_output_rejections": dict(rejected),
            "optional_reporting": {
                "failure_results": len(reporting),
                "checkpoint_preserving_recoveries": sum(bool(value.get("data", {}).get("numerical_checkpoint")) for value in reporting.values()),
                "run_ids": list(reporting),
                "scope": "Recovery means the existing numerical checkpoint survived optional reporting failure, not that presentation was repaired or physics accepted."},
            "model_responses": sum(event.get("kind") == "model" for event in events),
            "tool_calls": dict(counts), "committed_program_executions": len(programs),
            "committed_calculation_seconds": sum(float(item.get("result", {}).get("duration_s") or 0) for item in executions),
            "runtime_failure_records": len(list((directory / "workspace/runtime_failures").glob("*.json"))),
            "optimizer_campaigns": campaigns,
            "recorded_optimizer_evaluations": sum(item["evaluation_count"] for item in campaigns) if campaigns else None,
            "evaluation_count_note": "Counts from recognized PSO checkpoints only (null means none found); initialization screening, finalist refinement and other native calls are not included. No elapsed-time/latency claim is inferred.",
            "authorship_scope": "Exact immutable executed-program hashes must occur in saved model write_program calls; unrecorded operator assistance is not asserted absent by text length."}


def selected_presentation(messages: list[dict], proof: Mapping, candidate_id: str) -> dict:
    """An image/table from another turn or candidate cannot satisfy this answer."""
    selected = [row for row in messages if (row.get("meta", {}).get("completion_evidence") or {}).get("evidence_run_id")
                == proof.get("evidence_run_id")]
    if not selected:
        return {"selected_summary_in_chat": False, "selected_image_in_chat": False, "selected_table_in_chat": False}
    last = selected[-1]
    attachments = [item for item in last.get("attachments", []) if item.get("presentation_role") == "selected_evidence"
                   and item.get("lattice_id") == candidate_id]
    return {"selected_summary_in_chat": bool(last.get("content", "").strip()),
            "selected_image_in_chat": any(item.get("type") == "image" for item in attachments),
            "selected_table_in_chat": any(item.get("type") == "evidence_summary" and any(
                table.get("rows") for table in item.get("tables", [])) for item in attachments)}


def browser_checks(path: Path | None, *, directory: Path, candidate_id: str, run_id: str,
                   result_path: Path, source_root: Path) -> dict[str, bool]:
    checks = {"browser_selected_result_rendered": False}
    if path is None:
        return checks
    value = read_json(path)
    screenshot = Path(str(value.get("screenshot", ""))).resolve()
    expected_assets = {name: digest(source_root / "brilliance/web/static" / name)
                       for name in ("app.js", "styles.css", "workbench.css", "index.html")}
    checks["browser_selected_result_rendered"] = (
        value.get("schema") == "brilliance_rendered_selection/v1"
        and value.get("project_directory") == str(directory.resolve())
        and value.get("candidate_id") == candidate_id and value.get("run_id") == run_id
        and value.get("result_sha256") == digest(result_path)
        and value.get("assets") == expected_assets and RENDER_CHECKS <= value.get("checks", {}).keys()
        and all(flag is True for flag in value["checks"].values())
        and screenshot.is_file() and value.get("screenshot_sha256") == digest(screenshot))
    return checks
