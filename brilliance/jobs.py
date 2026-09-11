"""Durable background jobs for long-running JuTrack research stages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .artifacts import canonical_json, content_hash, json_safe
from .projects import ProjectStore

JOB_SCHEMA_VERSION = "1.1"
WORKER_START_GRACE_SECONDS = 30.0
WORKER_CLAIM_TIMEOUT_SECONDS = 120.0
WORKER_MAX_LAUNCH_ATTEMPTS = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ResearchJob:
    schema_version: str
    job_id: str
    revision: int
    project_id: str
    project_revision: int
    action: str
    state: JobState
    created_at: str
    updated_at: str
    worker_pid: int | None = None
    controller_status: str | None = None
    detail: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchJob":
        return cls(**{
            **dict(data),
            "state": JobState(data["state"]),
            "payload": dict(data.get("payload", {})),
        })


class ResearchJobStore:
    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()

    def _jobs_dir(self, project_id: str) -> Path:
        return self.project_root / project_id / "jobs"

    def submit(
        self,
        project_id: str,
        project_revision: int,
        *,
        action: str = "continue",
        payload: Mapping[str, Any] | None = None,
    ) -> ResearchJob:
        if action not in {"codex_turn", "expert_tool"}:
            raise ValueError(f"Unknown research job action: {action!r}")
        active = self.active(project_id, action=action)
        if active is not None:
            if action == "expert_tool" and canonical_json(active.payload) != canonical_json(
                payload or {}
            ):
                raise RuntimeError(
                    "this project already has an active managed expert calculation; "
                    "let it finish or explicitly cancel it before submitting another"
                )
            return active
        created = _now()
        job_id = "job-" + content_hash(
            {
                "project_id": project_id,
                "project_revision": project_revision,
                "action": action,
                "payload": payload or {},
                "created_at": created,
            },
            length=20,
        )
        job = ResearchJob(
            schema_version=JOB_SCHEMA_VERSION,
            job_id=job_id,
            revision=1,
            project_id=project_id,
            project_revision=project_revision,
            action=action,
            state=JobState.QUEUED,
            created_at=created,
            updated_at=created,
            payload=dict(payload or {}),
        )
        self.save(job)
        return job

    def save(self, job: ResearchJob) -> Path:
        directory = self._jobs_dir(job.project_id) / job.job_id / "revisions"
        payload = json_safe(asdict(job))
        digest = content_hash(payload, length=12)
        path = directory / f"{job.revision:04d}-{digest}.json"
        _write_once(path, payload)
        return path

    def load(self, project_id: str, job_id: str) -> ResearchJob | None:
        revisions = sorted(
            (self._jobs_dir(project_id) / job_id / "revisions").glob("*.json")
        )
        if not revisions:
            return None
        return ResearchJob.from_dict(json.loads(revisions[-1].read_text(encoding="utf-8")))

    def list(self, project_id: str) -> tuple[ResearchJob, ...]:
        jobs: list[ResearchJob] = []
        for directory in self._jobs_dir(project_id).glob("job-*"):
            job = self.load(project_id, directory.name)
            if job is not None:
                jobs.append(job)
        return tuple(sorted(jobs, key=lambda item: item.created_at))

    def active(self, project_id: str, *, action: str | None = None) -> ResearchJob | None:
        action_group = {action} if action else None
        for item in reversed(self.list(project_id)):
            if item.state not in {
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.CANCEL_REQUESTED,
            }:
                continue
            if action_group is not None and item.action not in action_group:
                continue
            if item.state in {JobState.RUNNING, JobState.CANCEL_REQUESTED} and item.worker_pid is not None:
                if not _process_alive(item.worker_pid):
                    self.finish(
                        item,
                        state=(
                            JobState.CANCELLED
                            if item.state is JobState.CANCEL_REQUESTED
                            else JobState.PAUSED
                        ),
                        controller_status=(
                            "cancelled"
                            if item.state is JobState.CANCEL_REQUESTED
                            else "worker_process_lost"
                        ),
                        detail=(
                            "The managed calculation stopped after the scientist requested cancellation."
                            if item.state is JobState.CANCEL_REQUESTED
                            else "The detached worker exited before committing a terminal state. "
                            "Project files and the Codex thread are preserved; the next message "
                            "can resume the work."
                        ),
                    )
                    continue
            if item.state is JobState.QUEUED:
                claim = self._jobs_dir(project_id) / item.job_id / "claim.json"
                attempts = self.launch_history(project_id, item.job_id)
                latest_launch = attempts[-1] if attempts else None
                if (
                    not claim.exists()
                    and latest_launch is not None
                    and isinstance(latest_launch.get("pid"), int)
                    and _process_alive(int(latest_launch["pid"]))
                    and _launch_age_seconds(latest_launch)
                    >= WORKER_CLAIM_TIMEOUT_SECONDS
                ):
                    self.finish(
                        item,
                        state=JobState.PAUSED,
                        controller_status="worker_claim_timeout",
                        detail=(
                            "The configured worker process started but did not claim the job "
                            f"within {WORKER_CLAIM_TIMEOUT_SECONDS:.0f} s. Its PID, exact command, "
                            "worker log, environment contract, and job payload are preserved. "
                            "No scientific result was changed."
                        ),
                    )
                    continue
                if not claim.exists() and self._launch_recovery_due(item):
                    if len(attempts) < WORKER_MAX_LAUNCH_ATTEMPTS:
                        try:
                            launch_research_job(self.project_root, item)
                        except Exception as exc:
                            if len(attempts) + 1 >= WORKER_MAX_LAUNCH_ATTEMPTS:
                                self.finish(
                                    item,
                                    state=JobState.PAUSED,
                                    controller_status="worker_launch_failed",
                                    detail=(
                                        "The managed worker could not be started after "
                                        f"{len(attempts) + 1} attempts: {type(exc).__name__}: {exc}. "
                                        "The exact launch history and job payload are preserved."
                                    ),
                                )
                                continue
                        return self.load(project_id, item.job_id) or item
                    self.finish(
                        item,
                        state=JobState.PAUSED,
                        controller_status="worker_launch_failed",
                        detail=(
                            "The managed worker exited before claiming the calculation after "
                            f"{len(attempts)} launch attempts. The exact launch PIDs, command, "
                            "worker log, and job payload are preserved for diagnosis."
                        ),
                    )
                    continue
            return item
        return None

    def recover(self, project_id: str) -> tuple[ResearchJob, ...]:
        """Refresh process state and safely relaunch only unclaimed queued jobs."""

        actions = ("codex_turn", "expert_tool")
        for action in actions:
            self.active(project_id, action=action)
        return self.list(project_id)

    def launch_history(self, project_id: str, job_id: str) -> tuple[dict[str, Any], ...]:
        directory = self._jobs_dir(project_id) / job_id / "launches"
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*-result.json")) if directory.is_dir() else ():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, Mapping):
                records.append(dict(value))
        return tuple(records)

    def record_launch(
        self,
        job: ResearchJob,
        *,
        command: list[str],
        pid: int | None,
        error: str | None = None,
    ) -> Path:
        history = self.launch_history(job.project_id, job.job_id)
        attempt = len(history) + 1
        payload = {
            "schema_version": "1.0",
            "job_id": job.job_id,
            "attempt": attempt,
            "started_at": _now(),
            "pid": pid,
            "alive_after_spawn": bool(pid is not None and _process_alive(pid)),
            "command": list(command),
            "error": error,
        }
        path = (
            self._jobs_dir(job.project_id)
            / job.job_id
            / "launches"
            / f"{attempt:04d}-result.json"
        )
        _write_once(path, payload)
        return path

    def _launch_recovery_due(self, job: ResearchJob) -> bool:
        history = self.launch_history(job.project_id, job.job_id)
        if not history:
            return _job_age_seconds(job) >= WORKER_START_GRACE_SECONDS
        latest = history[-1]
        pid = latest.get("pid")
        if isinstance(pid, int) and _process_alive(pid):
            return False
        return _launch_age_seconds(latest) >= WORKER_START_GRACE_SECONDS

    def claim(self, project_id: str, job_id: str) -> ResearchJob:
        job = self.load(project_id, job_id)
        if job is None:
            raise LookupError(f"Research job not found: {job_id}")
        if job.state is not JobState.QUEUED:
            raise RuntimeError(f"Research job is not queued: {job.state.value}")
        claim = self._jobs_dir(project_id) / job_id / "claim.json"
        _write_once(claim, {"job_id": job_id, "pid": os.getpid(), "claimed_at": _now()})
        running = replace(
            job,
            revision=job.revision + 1,
            state=JobState.RUNNING,
            updated_at=_now(),
            worker_pid=os.getpid(),
        )
        self.save(running)
        return running

    def finish(
        self,
        job: ResearchJob,
        *,
        state: JobState,
        controller_status: str,
        detail: str,
    ) -> ResearchJob:
        current = self.load(job.project_id, job.job_id) or job
        finished = replace(
            current,
            revision=max(job.revision, current.revision) + 1,
            state=state,
            updated_at=_now(),
            controller_status=controller_status,
            detail=detail,
        )
        self.save(finished)
        return finished

    def cancellation_requested(self, project_id: str, job_id: str) -> bool:
        return (self._jobs_dir(project_id) / job_id / "cancel.request.json").is_file()

    def request_cancel(self, project_id: str, job_id: str) -> ResearchJob:
        """Request a cooperative stop; queued jobs are cancelled immediately."""

        job = self.load(project_id, job_id)
        if job is None:
            raise LookupError(f"Research job not found: {job_id}")
        if job.action != "expert_tool":
            raise ValueError("only managed expert calculations expose scientist cancellation")
        if job.state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}:
            return job
        marker = self._jobs_dir(project_id) / job_id / "cancel.request.json"
        _write_once(marker, {"job_id": job_id, "requested_at": _now()})
        if job.state is JobState.QUEUED:
            return self.finish(
                job,
                state=JobState.CANCELLED,
                controller_status="cancelled_before_start",
                detail="The queued calculation was cancelled before it started.",
            )
        if job.state is JobState.CANCEL_REQUESTED:
            return job
        updated = replace(
            job,
            revision=job.revision + 1,
            state=JobState.CANCEL_REQUESTED,
            updated_at=_now(),
            controller_status="cancellation_requested",
            detail="Cancellation requested; stopping at the next safe evaluation boundary.",
        )
        self.save(updated)
        return updated


def run_research_job(
    root: Path | str,
    project_id: str,
    job_id: str,
    *,
    grid_size: int = 11,
    max_transitions: int = 64,
) -> ResearchJob:
    # Check before claiming the job. A manually invoked worker under an
    # unrelated Python must leave the queued job resumable by the configured
    # worker rather than mutating project state.
    from .research_environment import ResearchEnvironment

    ResearchEnvironment.load().require_active_python(operation="Research worker")
    project_store = ProjectStore(root)
    jobs = ResearchJobStore(root)
    job = jobs.claim(project_id, job_id)
    project = project_store.load(project_id)
    if project is None:
        return jobs.finish(
            job, state=JobState.FAILED, controller_status="not_found",
            detail=f"Project {project_id!r} no longer exists.",
        )
    if job.action == "expert_tool":
        return _run_managed_expert_job(root, project_store, jobs, job, project)
    if job.action == "codex_turn":
        from .research_recovery import scientist_request

        origin_start = scientist_request(Path(root), project.id)
        if job.payload.get("repair_of") or job.payload.get("review_of"):
            from .research_session import SessionWorkspace

            if (job.payload.get("repair_request_id") != scientist_request(Path(root), project.id).get("id")
                    or SessionWorkspace(Path(root) / project.id).pause_file.exists()):
                return jobs.finish(job, state=JobState.CANCELLED,
                    controller_status="repair_superseded", detail="A newer scientist request or pause superseded automatic repair.")
        from .codex_runtime import (
            CodexRuntimeStore,
            configured_auto_continue,
            configured_max_increments,
            run_codex_turn,
        )

        message = str(job.payload.get("message", "")).strip()
        if not message:
            return jobs.finish(
                job,
                state=JobState.PAUSED,
                controller_status="codex_invalid_input",
                detail="The research increment has no scientist message; project state is preserved.",
            )
        try:
            increment = max(1, int(job.payload.get("increment", 1)))
        except (TypeError, ValueError):
            increment = 1
        try:
            max_increments = max(
                increment,
                int(job.payload.get("max_increments", configured_max_increments())),
            )
        except (TypeError, ValueError):
            max_increments = max(increment, configured_max_increments())
        outcome = run_codex_turn(
            root,
            project,
            message,
            initial=bool(job.payload.get("initial", False)),
            increment=increment,
            max_increments=max_increments,
            research_mode=str(job.payload.get("research_mode", "research")),
        )
        finished = jobs.finish(
            job,
            state=(
                JobState.COMPLETE
                if outcome.status in {"complete", "checkpoint"}
                else JobState.PAUSED
            ),
            controller_status=f"codex_{outcome.status}",
            detail=outcome.detail,
        )
        pending = CodexRuntimeStore(root, project.id).consume_pending()
        next_payload: dict[str, Any] | None = None
        if pending:
            from .research_admission import assess_research_admission

            latest = project_store.load(project.id) or project
            pending_message = (
                "The scientist sent the following follow-up message(s) while the previous "
                "increment was running. Address all of them in order:\n\n"
                + "\n\n".join(pending)
            )
            admission = assess_research_admission(latest, root, pending_message)
            next_payload = {
                "message": pending_message,
                "initial": False,
                "increment": 1,
                "max_increments": configured_max_increments(),
                "research_mode": admission.mode,
            }
        elif outcome.disposition == "native_repair" and outcome.recovery:
            from .presentation import ResearchConversationStore
            from .research_recovery import program_failure_guidance, repair_pause_explanation, schedule_coding_repair

            # Normalize the inline calculation into the existing bounded repair
            # contract without changing this job's saved payload or history.
            repair_job = replace(finished, payload={**finished.payload,
                "tool_id": "accelerator.run_research_program",
                "arguments": {"script": outcome.recovery["script"]},
                "scientist_request": origin_start})
            try:
                repair = schedule_coding_repair(Path(root).resolve(), jobs, repair_job,
                    project_store.load(project.id) or project, Path(outcome.recovery["diagnostic"]),
                    outcome.recovery["error"])
            except Exception as repair_error:
                repair = {"scheduled": False, "reason": "repair_handoff_failed", "error": str(repair_error)}
            if not repair.get("scheduled"):
                message = (program_failure_guidance(outcome.recovery["error"])["explanation"] + "\n\n" +
                           repair_pause_explanation(repair) +
                           " Saved calculations are preserved; the research request is incomplete.")
                detail = message + " (" + str(repair.get("reason")) + ")"
                ResearchConversationStore(root).append_message(project.id, "assistant", message,
                    meta={"runtime": "native_controller", "kind": "checkpoint", "recovery": repair})
                finished = jobs.finish(finished, state=JobState.PAUSED,
                    controller_status="codex_repair_boundary", detail=detail)
        elif outcome.disposition == "native_continue":
            from .presentation import ResearchConversationStore
            from .research_recovery import latest_saved_program_evidence, schedule_result_review

            latest = project_store.load(project.id) or project
            review: dict[str, Any]
            try:
                saved = latest_saved_program_evidence(Path(root).resolve() / project.id)
                review = (
                    schedule_result_review(Path(root), jobs, finished, latest, saved[0], saved[1],
                                           scientist_origin=origin_start)
                    if saved is not None else
                    {"scheduled": False, "reason": "no_validated_saved_program"}
                )
            except Exception as review_error:
                review = {"scheduled": False, "reason": "result_review_failed", "error": str(review_error)}
                _write_once(Path(root) / project.id / "workspace" / "result_reviews" /
                            f"{job.job_id}.failure.json", {
                                "error": str(review_error), "traceback": traceback.format_exc(),
                            })
            if not review.get("scheduled"):
                reason = str(review.get("reason", "result_review_failed"))
                explanation = {
                    "result_review_limit": "The bounded automatic result-review budget has been reached.",
                    "no_validated_saved_program": "No validated saved-program artifact is available for automatic continuation.",
                    "request_changed_or_not_recorded": "A newer scientist request superseded this automatic continuation.",
                    "scientist_paused_or_cancelled": "The scientist paused or cancelled automatic continuation.",
                }.get(reason, "Automatic result review could not be started.")
                detail = explanation + " Saved results remain intact; the original research request is not marked complete."
                ResearchConversationStore(root).append_message(
                    project.id, "assistant", detail,
                    meta={"runtime": "native_controller", "kind": "checkpoint", "review": review},
                )
                finished = jobs.finish(finished, state=JobState.PAUSED,
                    controller_status=f"codex_{reason}", detail=detail)
        elif (
            configured_auto_continue()
            and outcome.disposition == "continue"
            and increment < max_increments
        ):
            next_payload = {
                "message": (
                    "Continue autonomously from the preserved checkpoint. Do not repeat completed "
                    "work. Inspect the canonical state and saved artifacts, then complete the next "
                    "highest-value accelerator-physics milestone with JuTrack."
                ),
                "initial": False,
                "increment": increment + 1,
                "max_increments": max_increments,
                "research_mode": str(job.payload.get("research_mode", "research")),
            }
        if next_payload is not None:
            latest = project_store.load(project.id) or project
            queued = jobs.submit(
                latest.id,
                latest.revision,
                action="codex_turn",
                payload=next_payload,
            )
            if queued.state is JobState.QUEUED:
                launch_research_job(root, queued)
        return finished
    return jobs.finish(
        job,
        state=JobState.FAILED,
        controller_status="retired_job_action",
        detail=f"The saved job action {job.action!r} belongs to the retired workflow engine.",
    )


def launch_research_job(root: Path | str, job: ResearchJob) -> int:
    """Launch a detached worker with the configured Python executable only."""

    from .research_environment import ResearchEnvironment

    environment = ResearchEnvironment.load()
    root_path = Path(root).resolve()
    project_directory = root_path / job.project_id
    worker_environment = environment.project_process_environment(
        project_directory / "workspace"
    )
    # A queued calculation may start after the originating model process exits.
    # Reconstruct claim requirements from its durable original scientist request,
    # never from whichever unrelated request is currently open in the UI.
    if job.action == "expert_tool":
        from .experts import ExpertRegistry
        from .scientific_acceptance import ACCEPTANCE_ENVIRONMENT, request_acceptance_requirements

        origin = job.payload.get("scientist_request", {})
        original_request = str(origin.get("content", "")) if isinstance(origin, Mapping) else ""
        recorded = job.payload.get("scientific_acceptance_requirements")
        required = (recorded if isinstance(recorded, list) and all(isinstance(name, str) for name in recorded)
                    else list(request_acceptance_requirements(original_request,
                        registry=ExpertRegistry.from_environment(environment, project_directory=project_directory))))
        worker_environment[ACCEPTANCE_ENVIRONMENT] = json.dumps(
            required
        )
    log_path = project_directory / "jobs" / job.job_id / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(environment.python_executable),
        "-m",
        "brilliance",
        "project-worker",
        job.project_id,
        job.job_id,
        "--root",
        str(root_path),
    ]
    jobs = ResearchJobStore(root_path)
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            kwargs: dict[str, Any] = {
                "cwd": str(environment.source_root),
                "env": worker_environment,
                "stdin": subprocess.DEVNULL,
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
                "close_fds": True,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                )
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(command, **kwargs)
        pid = int(process.pid)
        jobs.record_launch(job, command=command, pid=pid)
        return pid
    except Exception as exc:
        jobs.record_launch(
            job,
            command=command,
            pid=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _managed_expert_completion_message(
    tool_id: str,
    result: Mapping[str, Any],
    *,
    validated: bool,
    scientific_passed: bool | None,
    attachments: list[Mapping[str, Any]] | None = None,
) -> str:
    """Render any saved-program or small expert-tool result without domain templates."""

    nested = result.get("result")
    report = nested if isinstance(nested, Mapping) else result
    summary = str(report.get("summary", "")).strip() or f"{tool_id} completed."
    lines = ["**Research result**", "", summary]
    attached_tables = [
        table for attachment in attachments or []
        if attachment.get("type") == "evidence_summary"
        for table in attachment.get("tables", []) if isinstance(table, Mapping)
    ]
    attached_scalar_rows = [row for table in attached_tables for row in table.get("rows", [])
                            if isinstance(row, Mapping) and "quantity" in row and "value" in row]
    curated_tables = [table for table in report.get("tables", [])
                      if isinstance(table, Mapping) and not table.get("generated_from_evidence")]
    data = report.get("data", {})
    if isinstance(data, Mapping) and data and not curated_tables:
        scalar_rows = [(key, value) for key, value in list(data.items())[:24]
                       if not isinstance(value, (Mapping, list, tuple))
                       and {"quantity": str(key), "value": value} not in attached_scalar_rows]
        if scalar_rows:
            lines.extend(["", "| Quantity | Value |", "|---|---:|"])
            for key, value in scalar_rows:
                lines.append(f"| {_markdown_cell(key)} | {_markdown_cell(value)} |")
    tables = report.get("tables", [])
    for table in tables if isinstance(tables, list) else []:
        if not isinstance(table, Mapping):
            continue
        if table in attached_tables:
            # Typed evidence is rendered as a real table by the chat client.
            # Do not expand the same rows into a second Markdown table.
            continue
        rows = table.get("rows", [])
        if isinstance(rows, list) and rows:
            title = str(table.get("title", "Results")).strip() or "Results"
            lines.extend(["", f"**{title}**", "", *_markdown_rows(rows)])
    limitations = report.get("limitations", [])
    if isinstance(limitations, list) and limitations:
        lines.extend(["", "**Limitations**", ""])
        lines.extend(f"- {str(item)}" for item in limitations[:12])
    if not validated:
        lines.extend(["", "The output contract failed validation; all files were preserved."])
    elif scientific_passed is False:
        lines.extend(["", "This is valid negative evidence: the declared scientific checks did not pass."])
    elif scientific_passed is None:
        lines.extend(["", "No scientific pass was declared; treat this result as provisional evidence."])
    candidate = result.get("candidate")
    if isinstance(candidate, Mapping) and candidate.get("lattice_id"):
        action = str(result.get("working_lattice_action", "branch"))
        lines.extend(["", f"Candidate lattice: `{candidate['lattice_id']}` ({action})."])
    return "\n".join(lines)

def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_rows(rows: list[Any]) -> list[str]:
    mapped = [row for row in rows if isinstance(row, Mapping)]
    if not mapped:
        return ["No tabular rows were recorded."]
    columns: list[str] = []
    for row in mapped[:50]:
        for key in row:
            name = str(key)
            if name not in columns:
                columns.append(name)
            if len(columns) >= 16:
                break
        if len(columns) >= 16:
            break
    output = [
        "| " + " | ".join(_markdown_cell(item) for item in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in mapped[:30]:
        output.append(
            "| " + " | ".join(_markdown_cell(row.get(item, "")) for item in columns) + " |"
        )
    if len(mapped) > 30:
        output.append(f"\n{len(mapped) - 30} additional rows are preserved in the run artifact.")
    return output


def _run_managed_expert_job(
    root: Path | str,
    project_store: ProjectStore,
    jobs: ResearchJobStore,
    job: ResearchJob,
    project: Any,
) -> ResearchJob:
    """Execute one long expert calculation independently of an LLM turn."""

    from .codex_runtime import _turn_attachments
    from .codex_tools import _compact_expert_execution, _record_working_lattice
    from .experts import (
        ExpertRegistry,
        ExpertToolCancelled,
        ExpertToolContext,
        execute_expert_tool,
    )
    from .presentation import ResearchConversationStore
    from .research_environment import ResearchEnvironment

    started_at = datetime.now(timezone.utc).timestamp()
    tool_id = str(job.payload.get("tool_id", "")).strip()
    arguments = job.payload.get("arguments", {})
    if not tool_id or not isinstance(arguments, Mapping):
        return jobs.finish(
            job,
            state=JobState.FAILED,
            controller_status="invalid_expert_job",
            detail="Managed expert job requires a tool_id and an argument object.",
        )
    if jobs.cancellation_requested(job.project_id, job.job_id):
        return jobs.finish(
            job,
            state=JobState.CANCELLED,
            controller_status="cancelled_before_execution",
            detail="The calculation was cancelled before numerical execution began.",
        )
    project_directory = Path(root).resolve() / project.id
    conversations = ResearchConversationStore(root)
    environment: Any | None = None
    try:
        environment = ResearchEnvironment.load()
        os.environ.update(environment.project_process_environment(
            project_directory / "workspace", existing=os.environ
        ))
        registry = ExpertRegistry.from_environment(
            environment, project_directory=project_directory
        )
        context = ExpertToolContext(
            project_root=Path(root).resolve(),
            project_directory=project_directory,
            workspace=project_directory / "workspace",
            project=project,
            environment=environment,
            compute_job_id=job.job_id,
            cancellation_requested=lambda: jobs.cancellation_requested(
                job.project_id, job.job_id
            ),
        )
        execution = execute_expert_tool(
            registry,
            tool_id,
            dict(arguments),
            context=context,
            supplied_arguments_override=(
                dict(job.payload.get("supplied_arguments", {}))
                if isinstance(job.payload.get("supplied_arguments"), Mapping)
                else None
            ),
            argument_resolution_override=(
                {
                    str(name): str(source)
                    for name, source in job.payload.get("argument_resolution", {}).items()
                }
                if isinstance(job.payload.get("argument_resolution"), Mapping)
                and job.payload.get("argument_resolution")
                else None
            ),
        )
        artifact = context.workspace / "artifacts" / f"{execution['execution_id']}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(
            json_safe(execution), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        if artifact.exists():
            if artifact.read_text(encoding="utf-8") != rendered:
                raise RuntimeError(f"expert execution artifact conflict: {artifact}")
        else:
            artifact.write_text(rendered, encoding="utf-8", newline="\n")
        from .research_state import ResearchStateStore

        ResearchStateStore(project_directory).record_execution(execution, artifact=artifact)
        _record_working_lattice(project_directory, execution)
        project_store.snapshot_manifest(project.id)
        compact = _compact_expert_execution({**execution, "artifact": str(artifact)})
        validated = execution.get("validation", {}).get("passed") is True
        result = execution.get("result", {})
        verdict = execution.get("validation", {})
        scientific_passed = (
            verdict.get("scientific_passed")
            if isinstance(verdict, Mapping)
            and isinstance(verdict.get("scientific_passed"), bool)
            else result.get("passed")
            if isinstance(result, Mapping) and isinstance(result.get("passed"), bool)
            else None
        )
        attachments = _turn_attachments(Path(root).resolve(), project.id, started_at)
        conversations.append_message(
            project.id,
            "assistant",
            _managed_expert_completion_message(
                tool_id,
                result if isinstance(result, Mapping) else {},
                validated=validated,
                scientific_passed=scientific_passed,
                attachments=attachments,
            ),
            attachments=attachments,
            meta={
                "runtime": "managed_compute",
                "compute_job_id": job.job_id,
                "tool_id": tool_id,
                "result": compact,
            },
        )
        finished = jobs.finish(
            job,
            state=JobState.COMPLETE,
            controller_status=(
                "expert_result_feasible"
                if validated and scientific_passed is True
                else "expert_result_infeasible"
                if validated and scientific_passed is False
                else "expert_validation_passed"
                if validated
                else "expert_validation_failed"
            ),
            detail=f"Managed expert calculation completed; artifact: {artifact}",
        )
        from .research_recovery import schedule_result_review
        try:
            review = schedule_result_review(Path(root).resolve(), jobs, finished, project, artifact, execution)
            if review.get("reason") == "result_review_limit":
                conversations.append_message(project.id, "assistant",
                    "The automatic result-review budget has been reached. The calculation and its "
                    "artifacts are preserved, but the original request may still be incomplete. "
                    "Review the current result or send a specific next instruction.",
                    meta={"runtime": "managed_compute", "kind": "checkpoint"})
        except Exception as review_error:
            _write_once(artifact.with_suffix(".review_failure.json"), {
                "error": str(review_error), "traceback": traceback.format_exc(),
            })
            conversations.append_message(project.id, "assistant",
                "The calculation completed and its results are saved, but automatic result review "
                "could not start. No numerical work was lost.",
                meta={"runtime": "managed_compute", "kind": "review_failure"})
        return finished
    except ExpertToolCancelled:
        conversations.append_message(
            project.id,
            "assistant",
            f"Managed calculation **{tool_id}** stopped at a safe boundary at your request.",
            meta={
                "runtime": "managed_compute",
                "compute_job_id": job.job_id,
                "tool_id": tool_id,
            },
        )
        return jobs.finish(
            job,
            state=JobState.CANCELLED,
            controller_status="cancelled_by_scientist",
            detail="The calculation stopped at a safe evaluation boundary.",
        )
    except Exception as exc:
        failure_path = (
            project_directory / "workspace" / "runtime_failures" / f"{job.job_id}.json"
        )
        failure_record = {
            "schema_version": "1.0",
            "job_id": job.job_id,
            "project_id": project.id,
            "tool_id": tool_id,
            "failed_at": _now(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "python_executable": sys.executable,
            "configured_python": (
                str(environment.python_executable) if environment is not None else None
            ),
            "runtime_config": (
                str(environment.config_path) if environment is not None else None
            ),
        }
        _write_once(failure_path, failure_record)
        from .research_recovery import (
            coding_failure_summary,
            execution_failure_summary,
            program_failure_guidance,
            repair_pause_explanation,
            schedule_coding_repair,
        )

        headline = coding_failure_summary(str(exc)) or execution_failure_summary(str(exc))
        explanation = program_failure_guidance(str(exc))["explanation"]
        failed = jobs.finish(
            job, state=JobState.FAILED, controller_status="expert_execution_failed",
            detail=f"{headline}; diagnostic: {failure_path}",
        )
        try:
            recovery = schedule_coding_repair(Path(root).resolve(), jobs, failed, project,
                                              failure_path, str(exc))
        except Exception as recovery_error:
            recovery = {"scheduled": False, "reason": "handoff_failed"}
            _write_once(failure_path.with_suffix(".handoff.json"), {
                "error": str(recovery_error), "traceback": traceback.format_exc(),
            })
        if recovery.get("scheduled"):
            message = (
                "The calculation hit a coding error before this stage could finish. "
                f"I’m returning it to the research agent for correction (attempt {recovery['attempt']}). "
                "Existing results and checkpoints are preserved; this is not a physics rejection.\n\n"
                f"{explanation}\n\n"
                f"Issue: `{headline}`\n\nDetails: `{failure_path.relative_to(project_directory).as_posix()}`"
            )
        else:
            message = (
                "This calculation could not complete. Existing results and checkpoints are preserved; "
                "no new physics result was accepted.\n\n"
                f"{explanation}\n\n"
                f"Issue: `{headline}`\n\n"
                + repair_pause_explanation(recovery)
                + f"\n\nDetails: `{failure_path.relative_to(project_directory).as_posix()}`"
            )
        conversations.append_message(
            project.id,
            "assistant",
            message,
            meta={
                "runtime": "managed_compute",
                "compute_job_id": job.job_id,
                "tool_id": tool_id,
                "recovery": recovery,
                "diagnostic": str(failure_path),
            },
        )
        return failed


def _write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError(f"immutable research-job conflict: {path}")


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # Unlike POSIX, os.kill(pid, 0) is implemented with TerminateProcess on
        # Windows.  A status refresh must never kill the worker it is probing.
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                handle, ctypes.byref(exit_code)
            ):
                return False
            return int(exit_code.value) == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _job_age_seconds(job: ResearchJob) -> float:
    try:
        created = datetime.fromisoformat(job.created_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - created).total_seconds())


def _launch_age_seconds(launch: Mapping[str, Any]) -> float:
    try:
        started = datetime.fromisoformat(
            str(launch.get("started_at", "")).replace("Z", "+00:00")
        )
    except ValueError:
        return WORKER_START_GRACE_SECONDS
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
