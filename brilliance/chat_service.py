"""Chat-native application service over durable JuTrack research projects."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote
from uuid import uuid4

from .agent_tools import AgentEvidenceStore, render_agent_artifact
from .artifacts import content_hash
from .chat_artifacts import present_local_artifacts
from .codex_backends import describe_codex_backend
from .codex_runtime import (
    CodexRuntimeStore,
    _resolve_completion_artifact,
    _turn_attachments,
    configured_max_increments,
    research_progress,
    resolve_workspace_artifact,
    workspace_research_updates,
)
from .jobs import JobState, ResearchJob, ResearchJobStore, launch_research_job
from .lattice import LatticeIR
from .optimization_plotting import (
    OptimizationPlotUnavailable,
    render_latest_optimization_plot,
)
from .plotting import (
    chromaticity_plot_rows,
    damping_partition_rows,
    dynamic_aperture_plot_rows,
    element_span_m,
    lattice_plot_rows,
    optics_lattice_plot_data,
    profile_through,
    radiation_integral_rows,
    ring_plot_rows,
    robustness_plot_rows,
)
from .presentation import (
    ResearchConversationStore,
    compact_operational_messages,
    deduplicate_chat_attachments,
    final_result_text,
    project_key_metrics,
    summarize_rejected_stage,
    summarize_stage,
)
from .projects import (
    GENERAL_CAPABILITIES,
    ProjectStore,
    ResearchProject,
    create_planning_project,
)
from .research_admission import assess_research_admission
from .research_environment import ResearchEnvironment, ResearchEnvironmentError

_SNAPSHOT_IMAGE_HASH_LIMIT = 64 * 1024 * 1024


def _snapshot_image_digest(path: Path) -> str:
    """Read-only exact image identity, bounded and cached for polling snapshots."""
    try:
        stat = path.stat()
        if stat.st_size > _SNAPSHOT_IMAGE_HASH_LIMIT:
            return ""
        return _cached_snapshot_image_digest(
            path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        )
    except OSError:
        return ""


@lru_cache(maxsize=128)
def _cached_snapshot_image_digest(path: Path, size: int, modified: int, changed: int) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            remaining = size
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    return ""  # An image is still being written/replaced.
                digest.update(block)
                remaining -= len(block)
            if handle.read(1):
                return ""
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (size, modified, changed):
            return ""
    except OSError:
        return ""
    return digest.hexdigest()


class ResearchChatService:
    """Own web-chat commands without owning scientific acceptance decisions."""

    def __init__(self, project_root: Path | str):
        self.store = ProjectStore(project_root)
        self.jobs = ResearchJobStore(self.store.root)
        self.conversations = ResearchConversationStore(self.store.root)
        self.agent_evidence = AgentEvidenceStore(self.store.root)
        self._summary_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._command_lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self.store.root

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        for project_id in self.store.list_project_ids():
            signature = self._project_file_version(project_id)
            cached = self._summary_cache.get(project_id)
            if cached is not None and cached[0] == signature:
                projects.append(dict(cached[1]))
                continue
            project = self.store.load(project_id)
            if project is None:
                continue
            accepted = sum(task.status == "accepted" for task in project.tasks)
            summary = {
                "id": project.id,
                "title": project.title,
                "goal": project.goal,
                "status": project.status,
                "updated_at": project.updated_at,
                "progress": {"accepted": accepted, "total": len(project.tasks)},
                "backend": self._recorded_project_backend(project.id),
                "active": (
                    self._active_agent_job(project.id) is not None
                    or self._active_compute_job(project.id) is not None
                ),
            }
            self._summary_cache[project_id] = (signature, summary)
            projects.append(dict(summary))
        live_ids = {item["id"] for item in projects}
        self._summary_cache = {
            key: value for key, value in self._summary_cache.items() if key in live_ids
        }
        return projects

    def recover_jobs(self) -> list[dict[str, str]]:
        """Supervise durable workers independently of any open chat request."""

        errors: list[dict[str, str]] = []
        for project_id in self.store.list_project_ids():
            try:
                self.jobs.recover(project_id)
                # A message can arrive after the model's last input check but
                # before its job commits completion. Start a follow-up turn for
                # that durable input; it must not silently remain in the inbox.
                with self._command_lock:
                    runtime = CodexRuntimeStore(self.root, project_id)
                    if self._active_agent_job(project_id) is None and not self.session_workspace(project_id).pause_file.exists():
                        pending = runtime.consume_pending()
                        if pending:
                            try:
                                self._send_message(project_id, "\n\n".join(pending), record=False)
                            except Exception:
                                for message in pending:
                                    runtime.queue_input(message)
                                raise
            except Exception as exc:
                # One damaged historical project must not stop supervision for
                # every other active calculation. The next cycle retries it.
                errors.append(
                    {
                        "project_id": project_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return errors

    def create(self, goal: str, context: str = "") -> Mapping[str, Any]:
        project = replace(
            create_planning_project(goal, context),
            status="researching",
            model="codex",
            next_action="The configured model is beginning the requested accelerator-physics research.",
        )
        admission = assess_research_admission(project, self.root, goal)
        # The first request is real conversation history, not just a UI-synthesized
        # row. Durable workers need its identity for recovery and supersession.
        self.store.save(project)
        self.conversations.append_message(project.id, "user", goal)
        if not admission.allowed:
            project = replace(
                project,
                status="needs_input",
                next_action=admission.detail,
            )
            self.store.save(project)
            self.conversations.append_message(
                project.id,
                "assistant",
                admission.detail,
                meta={"admission_code": admission.code, "runtime": "design_kernel"},
            )
            self.store.snapshot_manifest(project.id)
            return {
                "action": "needs_input",
                "project": self._project_meta(project),
                "job": None,
                "admission": {
                    "code": admission.code,
                },
            }
        if admission.detail:
            project = replace(project, next_action=admission.detail)
        self.store.save(project)
        job = self._queue(
            project,
            action="codex_turn",
            payload={
                "message": goal,
                "initial": True,
                "explicit_request": True,
                "increment": 1,
                "max_increments": configured_max_increments(),
                "research_mode": admission.mode,
            },
        )
        return {
            "action": "codex_turn",
            "project": self._project_meta(project),
            "job": _job_meta(job),
            "admission": {
                "code": admission.code,
            },
        }

    def branch_checkpoint(self, project_id: str, checkpoint_id: str) -> Mapping[str, Any]:
        from .conversation_checkpoints import ConversationCheckpoints
        with self._command_lock:
            self._load(project_id)
            project = ConversationCheckpoints(self.root).fork(project_id, checkpoint_id)
            self.conversations.append_message(project.id, "assistant",
                "Restored the saved conversation checkpoint into a new project. "
                "The original project and its running jobs are unchanged. No calculation was restarted. "
                "Send your next instruction here; numerical continuation uses saved campaign checkpoints, not process memory.",
                meta={"checkpoint_branch": True})
            return {"project": self._project_meta(project), "action": "checkpoint_branch"}

    def continue_checkpoint(self, project_id: str, checkpoint_id: str, *,
                            request_id: str) -> Mapping[str, Any]:
        """Fork the selected saved state and queue its next turn exactly once.

        The click is an operational instruction. It does not manufacture a new
        user message or substitute the source project's later failed state.
        """
        from .atomic_io import atomic_write_text
        from .conversation_checkpoints import ConversationCheckpoints
        from .research_recovery import scientist_request

        if not re.fullmatch(r"project-[A-Za-z0-9_-]+", project_id):
            raise ValueError("Invalid project identifier")
        if not re.fullmatch(r"[0-9]+-[0-9a-f]+", checkpoint_id):
            raise ValueError("Invalid conversation checkpoint identifier")
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", request_id):
            raise ValueError("A valid continuation request identifier is required")
        with self._command_lock, self._protocol_action_lock(project_id):
            receipt_path = (self.root / project_id / "codex" / "checkpoint_actions" /
                            (content_hash(request_id) + ".json"))
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (receipt.get("source_project") != project_id or
                        receipt.get("source_checkpoint") != checkpoint_id):
                    raise ValueError("This continuation request belongs to a different checkpoint")
            else:
                receipt = {"source_project": project_id, "source_checkpoint": checkpoint_id,
                           "branch_project": "project-" + uuid4().hex[:14]}
                # Reserve the destination before copying so a lost HTTP response
                # or process restart cannot create another branch for this click.
                atomic_write_text(receipt_path, json.dumps(receipt, indent=2))
            branch_id = receipt.get("branch_project", "")
            if not isinstance(branch_id, str) or not re.fullmatch(r"project-[A-Za-z0-9_-]+", branch_id):
                raise ValueError("Invalid saved continuation destination")
            project = self.store.load(branch_id)
            if project is None:
                project = ConversationCheckpoints(self.root).fork(
                    project_id, checkpoint_id, continuation=True, new_id=branch_id)
            prior = next((job for job in self.jobs.list(branch_id)
                          if job.payload.get("checkpoint_continuation_id") == request_id), None)
            if prior is not None:
                # A job may have committed just before the response was lost.
                # Recover an unclaimed launch, never submit a second model turn.
                self.jobs.recover(branch_id)
                job = self.jobs.load(branch_id, prior.job_id) or prior
            else:
                origin = scientist_request(self.root, branch_id)
                if not origin:
                    raise ValueError("This checkpoint has no saved scientist request to continue")
                prompt = (
                    "Continue the scientist's research from this selected conversation checkpoint. "
                    "This is an internal continuation handoff authorized by the scientist's click, "
                    "not a new scientific request.\n"
                    f"Latest scientist request: {origin['content']}\n"
                    "The conversation and scientific files are restored in the current workspace. "
                    "Read saved results and optimizer checkpoints to choose the next justified step; "
                    "do not repeat completed calculations just to recover context. Preserve the "
                    "original scientific requirements and subsequent instructions. Running processes "
                    "and provider sessions were not copied: do not claim that pending computations "
                    "finished, replay archived job commands, or control jobs in the original project. "
                    "Use the current project workspace for all new work; adapt any old absolute paths "
                    "before executing saved programs. Continue with substantive scientific progress "
                    "without a restart announcement or repeated introduction. If the saved evidence "
                    "already fulfills the request, provide its readable final report."
                )
                project = replace(project, revision=project.revision + 1, status="researching",
                                  next_action="Continuing research from the selected saved results.")
                self.store.save(project)
                job = self._queue(project, action="codex_turn", payload={
                    "message": prompt, "initial": False, "explicit_request": True,
                    "scientist_request": origin, "increment": 1,
                    "max_increments": configured_max_increments(),
                    "research_mode": "result_review", "checkpoint_continuation": True,
                    "checkpoint_continuation_id": request_id,
                    "source_checkpoint": checkpoint_id,
                })
            atomic_write_text(receipt_path, json.dumps({**receipt, "job_id": job.job_id}, indent=2))
            return {"action": "checkpoint_continue", "project": self._project_meta(self._load(branch_id)),
                    "job": _job_meta(job), "already_started": prior is not None,
                    "continuation": {"source_project": project_id, "source_checkpoint": checkpoint_id}}

    def snapshot(self, project_id: str) -> Mapping[str, Any]:
        project = self._load(project_id)
        job = self._latest_research_job(project.id)
        agent_job = self._active_agent_job(project.id)
        computations = self._active_compute_jobs(project.id)
        return {
            "project": self._project_meta(project),
            "messages": self._messages(project, job),
            "live": self._live_state(project, agent_job or job),
            "computations": [self._computation_meta(item) for item in computations],
            "version": self.version(project.id),
            "session": self.session_state(project.id),
        }

    def session_workspace(self, project_id: str):
        from .research_session import SessionWorkspace

        project = self._load(project_id)
        return SessionWorkspace(self.root / project.id)

    def session_state(self, project_id: str) -> Mapping[str, Any]:
        from .codex_backends import resolve_codex_backend

        workspace = self.session_workspace(project_id)
        runtime = CodexRuntimeStore(self.root, project_id).load()
        try:
            resolve_codex_backend()
            can_pause = True
        except (ValueError, RuntimeError):
            can_pause = False
        return {
            "pause_requested": workspace.pause_file.exists(),
            "can_pause": can_pause,
            "activities": runtime.get("activities", []),
            "source_activities": runtime.get("source_activities", []),
            "programs": workspace.list_programs(),
            "protocol_recovery": self._protocol_recovery(project_id, runtime),
        }

    def _protocol_recovery(self, project_id: str, runtime: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Expose an actionable pause, never the private rejected model response."""
        from .research_recovery import latest_saved_program_evidence, scientist_request

        job = self._latest_agent_job(project_id)
        if job is None or self._active_agent_job(project_id) is not None:
            return None
        rows = self.conversations.messages(project_id)
        user = next((row for row in reversed(rows) if row.get("role") == "user"), None)
        if user is None or str(user.get("timestamp", "")) > job.created_at:
            return None  # A newer request superseded the failed turn.
        failure = runtime.get("protocol_failure")
        if not failure and runtime.get("completed_turn", {}).get("status") == "failed":
            # Older versions accidentally labelled failed provider turns complete.
            # Recover the matching public terminal error without modifying history.
            from .codex_runtime import CodexProviderTurnError, _check_completed_turn
            terminal_id = runtime["completed_turn"].get("id")
            directory = self.root / project_id / "codex" / "events"
            for path in sorted(directory.glob("*.json"), reverse=True)[:30]:
                try:
                    event = json.loads(path.read_text(encoding="utf-8"))
                    turn = event.get("payload", {}).get("turn", {})
                    if event.get("method") == "turn/completed" and turn.get("id") == terminal_id:
                        _check_completed_turn(turn)
                        break
                except CodexProviderTurnError as exc:
                    failure = {"type": "provider_turn_failure", "reason": exc.code,
                               "message": str(exc), "can_retry": True}
                    break
                except (OSError, ValueError, AttributeError):
                    continue
        if not isinstance(failure, Mapping) or not failure:
            # Read-only compatibility for the old terminal message. Do not edit
            # archived chat or guess a physics failure from a generic pause.
            last = next((row for row in reversed(rows) if row.get("role") == "assistant"), {})
            if "could not encode its next tool request correctly" not in str(last.get("content", "")):
                return None
            failure = {"reason": "legacy_model_output_rejected", "can_retry": True}
            sessions = sorted((self.root / project_id / "workspace" / "agent_sessions").glob("native-*.json"),
                              key=lambda path: path.stat().st_mtime_ns, reverse=True)
            if sessions:
                try:
                    if sessions[0].stat().st_size <= 8 * 1024 * 1024:
                        events = json.loads(sessions[0].read_text(encoding="utf-8")).get("events", [])
                        rejected = next((event for event in reversed(events) if event.get("kind") == "model_output_rejected"), {})
                        failure = {**failure, "reason": rejected.get("reason", failure["reason"])}
                except (OSError, ValueError, AttributeError, TypeError):
                    pass
        origin = scientist_request(self.root, project_id)
        failure_id = content_hash({"job": job.job_id, "request": origin.get("id"),
                                   "reason": failure.get("reason"), "diagnostic": failure.get("diagnostic_path")})
        if runtime.get("protocol_recovery_dismissed") == failure_id:
            return None
        reason = str(failure.get("reason", "model_output_rejected"))
        message = (
            "The model repeatedly exceeded the public-response length limit. Its next research step was not executed."
            if reason == "public_prose_too_long" else
            "The provider hit its output-token limit before completing the next response. "
            "Automatic budget/shorter-step recovery did not complete it. No partial tool request was executed. "
            "Check this provider's configured output budget and ceiling before retrying."
            if reason == "output_token_limit" else
            "The provider returned incomplete responses. This is not a rejected physics result. "
            "Check the model connection and output budget; its next research step was not executed."
            if reason == "incomplete_provider_response" else
            "The model could not produce a valid next tool request after bounded repair. Its next research step was not executed."
        )
        if failure.get("type") == "provider_turn_failure":
            message = (str(failure.get("message") or "The model request failed.")
                       + " Retry when service is available, or select another model. Saved numerical outputs remain intact; this is not a physics failure.")
        if reason == "output_token_limit" and type(failure.get("requested_max_tokens")) is int:
            message += f" Last requested output allowance: {failure['requested_max_tokens']} tokens."
        try:
            attempts = max(0, int(failure.get("attempts", 0)))
        except (TypeError, ValueError):
            attempts = 0
        return {"failure_id": failure_id, "request_id": origin["id"], "reason": reason,
                "message": message, "attempts": attempts, "can_retry": failure.get("can_retry", True) is True,
                "checkpoint_available": latest_saved_program_evidence(self.root / project_id) is not None}

    def recover_protocol(self, project_id: str, *, action: str, failure_id: str,
                         request_id: str) -> Mapping[str, Any]:
        """Explicit retry/dismissal without adding a new scientific user request."""
        from .research_recovery import latest_saved_program_evidence, scientist_request

        if action not in {"retry", "dismiss"}:
            raise ValueError("Recovery action must be retry or dismiss")
        if not failure_id or not request_id:
            raise ValueError("The displayed failure and scientist request identifiers are required")
        with self._command_lock, self._protocol_action_lock(project_id):
            project = self._load(project_id)
            origin = scientist_request(self.root, project_id)
            if origin.get("id") != request_id:
                raise ValueError("The scientist request changed. Refresh the conversation before retrying.")
            prior = next((job for job in reversed(self.jobs.list(project_id))
                          if job.payload.get("protocol_retry_of") == failure_id), None)
            if prior is not None and action == "retry":
                return {"action": "already_retried", "job": _job_meta(prior),
                        "detail": "This checkpoint retry was already submitted; no duplicate turn was started."}
            if self._active_agent_job(project_id) is not None:
                raise ValueError("A research turn is already active. Send a message to steer it or pause it.")
            runtime = CodexRuntimeStore(self.root, project_id)
            recovery = self._protocol_recovery(project_id, runtime.load())
            if recovery is None or recovery["failure_id"] != failure_id:
                raise ValueError("This recovery action is stale. Refresh the conversation.")
            workspace = self.session_workspace(project_id)
            if action == "dismiss":
                workspace.pause_file.touch()
                runtime.update(protocol_recovery_dismissed=failure_id, status="paused",
                               label="Paused by the scientist; saved evidence is preserved")
                self.store.snapshot_manifest(project_id)
                return {"action": "dismissed", "detail": "Research remains paused. Saved evidence and independent numerical jobs are unchanged."}
            if not recovery["can_retry"]:
                raise ValueError("This failure is not eligible for a checkpoint retry")
            saved = latest_saved_program_evidence(self.root / project_id)
            evidence = (
                f"Validated evidence: {saved[0].relative_to(self.root / project_id).as_posix()}.\n"
                if saved is not None else "No validated calculation receipt is available. Inspect saved context before executing anything.\n"
            )
            prompt = (
                "The scientist explicitly requested a retry from a model-response protocol pause. "
                "This is an internal recovery handoff, not a new scientific request.\n"
                f"Latest scientist request: {origin['content']}\n"
                f"Operational issue: {recovery['message']}\n" + evidence +
                "Resume that request from the preserved workspace and committed evidence. Do not repeat "
                "the baseline calculation or its tables just to recover context. Use read_artifact for "
                "saved results. State the next justified step briefly, then use correctly encoded tool "
                "arguments to perform it. Retain the configured interpreter, model, network policy and "
                "physics requirements. Do not cancel independent numerical jobs. If a genuine scientist "
                "decision or unrecoverable blocker remains, ask one specific actionable question."
            )
            workspace.pause_file.unlink(missing_ok=True)
            from .research_continuation import handoff_payload, pause_explanation, turn_allowance
            allowance = turn_allowance(self.root, project_id, origin, automatic=False)
            if not allowance["allowed"]:
                return {"action": "needs_input", "detail": pause_explanation(allowance["reason"])}
            project = replace(project, status="researching", next_action="Resuming the latest request from saved evidence.")
            self.store.save(project)
            job = self._queue(project, action="codex_turn", payload={
                "message": prompt, "initial": False, "explicit_request": True, **handoff_payload(allowance),
                "research_mode": "result_review", "protocol_retry_of": failure_id,
                "repair_request_id": origin["id"], "scientist_request": origin,
            })
            self.store.snapshot_manifest(project_id)
            return {"action": "retry_started", "job": _job_meta(job),
                    "detail": "Resuming the latest scientist request from saved evidence. Completed calculations are preserved."}

    @contextmanager
    def _protocol_action_lock(self, project_id: str):
        """Cross-process, crash-released lock for idempotent checkpoint actions."""
        project = self._load(project_id)
        directory = self.root / project.id / "codex"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "protocol-action.lock").open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("Another recovery action is in progress. Refresh the conversation.") from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def pause_agent(self, project_id: str) -> Mapping[str, Any]:
        workspace = self.session_workspace(project_id)
        if not self.session_state(project_id)["can_pause"]:
            raise ValueError("This provider runtime does not support cooperative pause")
        workspace.pause_file.touch()
        if self._active_agent_job(project_id) is None:
            return {"paused": True, "detail": "Automatic continuation is paused. Independent numerical jobs will keep running."}
        CodexRuntimeStore(self.root, project_id).update(label="Pause requested; finishing the current operation")
        return {"paused": False, "detail": "Pause requested. Independent numerical jobs will keep running."}

    def cancel_computation(self, project_id: str, job_id: str) -> Mapping[str, Any]:
        """Request a user-authorized cooperative stop of a managed calculation."""

        self._load(project_id)
        job = self.jobs.request_cancel(project_id, job_id)
        self.store.snapshot_manifest(project_id)
        return self._computation_meta(job)

    def send_message(self, project_id: str, text: str) -> Mapping[str, Any]:
        with self._command_lock:
            return self._send_message(project_id, text)

    def _send_message(self, project_id: str, text: str, *, record: bool = True) -> Mapping[str, Any]:
        """Answer typed presentation requests or queue scientific reasoning to Codex."""
        message = str(text).strip()
        if not message:
            raise ValueError("Message cannot be empty")
        project = self._load(project_id)
        if record:
            self.conversations.append_message(project.id, "user", message)
            self.session_workspace(project.id).pause_file.unlink(missing_ok=True)
        active = self._active_codex_job(project.id)
        if active is not None:
            CodexRuntimeStore(self.root, project.id).queue_input(message)
            self.store.snapshot_manifest(project.id)
            return {
                "action": "queued_for_codex",
                "job": _job_meta(active),
                "answer": (
                    "Your message is steering the active research turn. The model will incorporate "
                    "it at the next safe command boundary without losing workspace or history."
                ),
            }
        admission = assess_research_admission(project, self.root, message)
        if not admission.allowed:
            self.conversations.append_message(
                project.id,
                "assistant",
                admission.detail,
                meta={"admission_code": admission.code, "runtime": "design_kernel"},
            )
            self.store.snapshot_manifest(project.id)
            return {
                "action": "needs_input",
                "answer": admission.detail,
                "job": None,
            }
        project = replace(
            project,
            status="researching",
            next_action=(
                admission.detail
                or "The configured model is beginning the requested accelerator-physics research."
            ),
        )
        self.store.save(project)
        # Older inbox messages are already part of public conversation history.
        # Do not replay them after this newer user request as if they came later.
        CodexRuntimeStore(self.root, project.id).consume_pending()
        from .research_continuation import request_started
        from .research_recovery import scientist_request
        origin = scientist_request(self.root, project.id)
        job = self._queue(
            project,
            action="codex_turn",
            payload={
                "message": message,
                "initial": False,
                "explicit_request": record or not request_started(self.root, project.id, origin),
                "scientist_request": origin,
                "increment": 1,
                "max_increments": configured_max_increments(),
                "research_mode": admission.mode,
            },
        )
        self.store.snapshot_manifest(project.id)
        return {
            "action": "codex_turn",
            "job": _job_meta(job),
            "admission": {
                "code": admission.code,
            },
        }

    def artifact(self, project_id: str, artifact_id: str) -> Mapping[str, Any]:
        project = self._load(project_id)
        if artifact_id.startswith("agent-"):
            return render_agent_artifact(self.agent_evidence.load(project.id, artifact_id))
        result: Mapping[str, Any]
        if artifact_id == "lattice":
            result = self._capability_result(project, "lattice_construction")
            elements = _lattice_elements(result)
            proxy = _is_proxy_ring_result(result)
            return {
                "kind": "lattice",
                "title": "Repeated-cell proxy survey" if proxy else "Accepted ring floor plan",
                "rows": ring_plot_rows(elements),
                "meta": {
                    "element_count": len(elements),
                    "circumference_m": result.get("circumference_m", result.get("length_m")),
                    "closure_error_rad": result.get("geometry_closure_error_rad"),
                    "design_scope": (
                        "homogeneous recurring-cell proxy" if proxy else "accepted ring"
                    ),
                    "physical_fingerprint": _lattice_physical_fingerprint(result),
                },
            }
        if artifact_id == "lattice-block":
            result = self._capability_result(project, "lattice_construction")
            elements = _lattice_elements(result)
            superperiods = _superperiod_count(project, result, len(elements))
            block_size = max(1, len(elements) // superperiods) if elements else 0
            cell_elements = elements[:block_size]
            bend_inventory = _bend_inventory(cell_elements)
            proxy = _is_proxy_ring_result(result)
            family = _mba_family_label(project, result)
            return {
                "kind": "lattice_block",
                "title": (
                    f"One {family} recurring-cell proxy"
                    if proxy
                    else f"One periodic {family} superperiod"
                ),
                "rows": lattice_plot_rows(cell_elements),
                "meta": {
                    "cell_elements": block_size,
                    "superperiods_in_ring": superperiods,
                    "cell_length_m": element_span_m(cell_elements),
                    **bend_inventory,
                    "bend_definition": (
                        "primary bends have positive angle; physical bending magnets also "
                        "include auxiliary negative-angle reverse bends"
                    ),
                    "construction_source": "brilliance.physics_services.construct_full_ring",
                    "design_scope": (
                        "homogeneous recurring-cell proxy" if proxy else "accepted superperiod"
                    ),
                },
            }
        if artifact_id == "optics":
            lattice = self._capability_result(project, "lattice_construction")
            elements = _lattice_elements(lattice)
            superperiods = _superperiod_count(project, lattice, len(elements))
            cell_size = max(1, len(elements) // superperiods) if elements else 0
            cell_elements = elements[:cell_size]
            cell_length = element_span_m(cell_elements)
            result = self._capability_result(project, "linear_matching")
            profile = _mapping_rows(result.get("twiss_profile"))
            cell_profile = profile_through(profile, cell_length)
            family = _mba_family_label(project, lattice)
            return {
                "kind": "optics",
                "title": f"Matched optics through one periodic {family} superperiod",
                **optics_lattice_plot_data(cell_profile, cell_elements),
                "meta": {
                    "view": f"one periodic {family} superperiod",
                    "cell_length_m": cell_length,
                    "cell_points": len(cell_profile),
                    "superperiods_in_ring": superperiods,
                    **_bend_inventory(cell_elements),
                    "bend_definition": (
                        "primary bends have positive angle; physical bending magnets also "
                        "include auxiliary negative-angle reverse bends"
                    ),
                    "construction_source": "brilliance.physics_services.construct_full_ring",
                    "tune_x": result.get("tune_x"),
                    "tune_y": result.get("tune_y"),
                    "stable": result.get("stable"),
                    "closure": result.get("periodic_closure_max"),
                },
            }
        if artifact_id == "ring-optics":
            lattice = self._capability_result(project, "lattice_construction")
            elements = _lattice_elements(lattice)
            result = self._capability_result(project, "linear_matching")
            profile = _mapping_rows(result.get("twiss_profile"))
            return {
                "kind": "optics",
                "title": "Matched optics around the full ring",
                **optics_lattice_plot_data(profile, elements),
                "meta": {
                    "view": "full ring",
                    "ring_points": len(profile),
                    "tune_x": result.get("tune_x"),
                    "tune_y": result.get("tune_y"),
                    "stable": result.get("stable"),
                    "closure": result.get("periodic_closure_max"),
                },
            }
        if artifact_id == "radiation":
            result = self._capability_result(project, "radiation_analysis")
            ring = result.get("jutrack_ring_parameters", {})
            return {
                "kind": "radiation",
                "title": "Radiation equilibrium",
                "integrals": radiation_integral_rows(result),
                "partitions": damping_partition_rows(result),
                "meta": {
                    "emittance_pm": result.get("emittance_pm"),
                    "energy_spread": result.get("energy_spread"),
                    "momentum_compaction": ring.get("alphac")
                    if isinstance(ring, Mapping)
                    else None,
                    "robinson_sum": result.get("robinson_sum"),
                },
            }
        if artifact_id == "chromaticity":
            result = self._capability_result(project, "chromatic_correction")
            return {
                "kind": "chromaticity",
                "title": "Chromaticity correction",
                "rows": chromaticity_plot_rows(result),
                "meta": {
                    "k2_values": result.get("k2_values", {}),
                    "response_condition": result.get("response_condition"),
                },
            }
        if artifact_id == "dynamic-aperture":
            result = self._capability_result(project, "nonlinear_tracking")
            return {
                "kind": "dynamic_aperture",
                "title": "Off-momentum dynamic aperture",
                "rows": dynamic_aperture_plot_rows(_mapping_rows(result.get("scans"))),
                "meta": {"nturns": result.get("nturns"), "model": result.get("tracking_model")},
            }
        if artifact_id == "robustness":
            result = self._capability_result(project, "error_correction")
            return {
                "kind": "robustness",
                "title": "Error correction ensemble",
                "rows": robustness_plot_rows(_mapping_rows(result.get("seeds"))),
                "meta": result.get("ensemble_statistics", {}),
            }
        if artifact_id == "cross-validation":
            result = self._capability_result(project, "cross_validation")
            return {
                "kind": "comparison",
                "title": "Independent numerical composition check",
                "rows": _mapping_rows(result.get("comparison_table")),
                "meta": {
                    "passed": result.get("passed"),
                    "maximum_relative_difference": result.get("maximum_relative_difference"),
                    "tolerance": result.get("tolerance"),
                },
            }
        if artifact_id == "research-plots":
            return self._agent_plot_artifact(project)
        raise LookupError(f"Unknown project artifact view: {artifact_id}")

    def lattice_file(self, project_id: str, lattice_id: str, artifact: str) -> Path:
        """Resolve a canonical lattice definition or its reproduction companion."""
        self._load(project_id)
        return self.store.lattice_package_file(project_id, lattice_id, artifact)

    def workspace_artifact_file(self, project_id: str, relative_path: str) -> Path:
        """Resolve a Codex-created plot, table, or data file inside one project."""
        self._load(project_id)
        return resolve_workspace_artifact(self.root, project_id, relative_path)

    def report(self, project_id: str) -> str:
        from .research_reporting import saved_report
        self._load(project_id)
        report = saved_report(self.root, project_id)
        if not report:
            raise LookupError("The research model has not completed a scientific report for this request.")
        return report

    def version(self, project_id: str) -> str:
        project_dir = self.root / project_id
        if not project_dir.is_dir():
            raise LookupError(f"Research project not found: {project_id}")
        return self._project_file_version(project_id)

    def _project_file_version(self, project_id: str) -> str:
        """Hash small filesystem identities without parsing multi-megabyte evidence."""
        project_dir = self.root / project_id
        revisions = sorted((project_dir / "revisions").glob("*.json"))
        latest_revision = revisions[-1] if revisions else None
        job_revisions = [
            paths[-1]
            for directory in (project_dir / "jobs").glob("job-*")
            if (paths := sorted((directory / "revisions").glob("*.json")))
        ]
        conversations = sorted((project_dir / "conversation").glob("*.json"))
        codex_state = project_dir / "codex" / "state.json"
        workspace_artifacts = (
            [
                path
                for path in (project_dir / "workspace" / "artifacts").rglob("*")
                if path.is_file()
            ]
            if (project_dir / "workspace" / "artifacts").is_dir()
            else []
        )
        workspace_progress = (
            [
                path
                for path in (project_dir / "workspace" / "progress").rglob("*.json")
                if path.is_file()
            ]
            if (project_dir / "workspace" / "progress").is_dir()
            else []
        )
        tracked = [
            path
            for path in [
                latest_revision,
                *job_revisions,
                *conversations,
                codex_state if codex_state.is_file() else None,
                *workspace_artifacts,
                *workspace_progress,
                *(project_dir / "workspace" / "programs").glob("*.py"),
            ]
            if path
        ]
        identities = [
            (str(path.relative_to(project_dir)), path.stat().st_mtime_ns, path.stat().st_size)
            for path in tracked
        ]
        # Keep the elapsed-time display alive while one native JuTrack batch is
        # still running and therefore has not written another progress file.
        if self._active_compute_jobs(project_id):
            identities.append(("managed-compute-heartbeat", int(time.time() // 10), 0))
        return content_hash(identities, length=20)

    def _messages(
        self,
        project: ResearchProject,
        job: ResearchJob | None,
    ) -> list[Mapping[str, Any]]:
        messages: list[dict[str, Any]] = []
        conversation = self.conversations.messages(project.id)
        if not any(item.get("role") == "user" and item.get("content") == project.goal for item in conversation):
            messages.append(
                self._message(
                    project.id + ":goal",
                    "user",
                    "goal",
                    project.created_at,
                    project.goal,
                )
            )
        if project.model != "codex":
            contract = project.goal_interpretation
            if project.assumptions:
                contract += "\n\nAssumptions used to start autonomously:\n" + "\n".join(
                    f"- {item}" for item in project.assumptions
                )
            messages.append(
                self._message(
                    project.id + ":contract",
                    "assistant",
                    "contract",
                    project.created_at,
                    contract,
                    meta={"machine": project.machine_summary, "authority": "JuTrack"},
                )
            )

        task_by_id = {task.id: task for task in project.tasks}
        for evidence in project.evidence:
            task_id = str(evidence.get("task_id", ""))
            task = task_by_id.get(task_id)
            if task is None:
                continue
            capability = str(task.capability)
            validation = evidence.get("deterministic_validation", {})
            validator = validation.get("validator") if isinstance(validation, Mapping) else None
            accepted = bool(
                isinstance(validation, Mapping)
                and validation.get("passed") is True
                and evidence.get("status") != "failed"
            )
            tool = {
                "type": "tool",
                "title": GENERAL_CAPABILITIES.get(capability, {}).get("title", task.title),
                "status": "accepted" if accepted else "rejected",
                "source": evidence.get("source", "registered service"),
                "validator": validator,
                "evidence_id": evidence.get("evidence_id"),
            }
            plot = _capability_plot(capability, project.id, project.revision) if accepted else None
            attachments = [tool] + ([plot] if plot else [])
            if accepted:
                for lattice_id in _evidence_lattice_ids(evidence):
                    attachments.extend(_lattice_download_attachments(project.id, lattice_id))
            messages.append(
                self._message(
                    str(evidence.get("evidence_id", task_id)),
                    "assistant",
                    "stage" if accepted else "observation",
                    str(evidence.get("timestamp", project.updated_at)),
                    (
                        summarize_stage(task, evidence)
                        if accepted
                        else summarize_rejected_stage(task, evidence)
                    ),
                    meta={"task_id": task_id, "title": task.title, "accepted": accepted},
                    attachments=attachments,
                )
            )

        for decision in project.dialogue:
            role = str(decision.get("role", ""))
            if role == "scientist":
                messages.append(
                    self._message(
                        content_hash(decision, length=20),
                        "user",
                        "decision",
                        str(decision.get("timestamp", project.updated_at)),
                        str(decision.get("content", "")),
                    )
                )
                continue
            if role == "research_supervisor" and decision.get("verdict") == "revise":
                old = decision.get("old_parameters")
                new = decision.get("new_parameters")
                old = old if isinstance(old, Mapping) else {}
                new = new if isinstance(new, Mapping) else {}
                changed = [key for key, value in new.items() if old.get(key) != value]
                task = task_by_id.get(str(decision.get("task_id", "")))
                title = task.title if task is not None else "the rejected stage"
                rationale = str(decision.get("content", "")).strip().rstrip(".")
                messages.append(
                    self._message(
                        content_hash(decision, length=20),
                        "assistant",
                        "supervision",
                        str(decision.get("timestamp", project.updated_at)),
                        (
                            f"The previous **{title}** experiment did not pass its physics gate. "
                            f"I changed {', '.join(changed) or 'the bounded search envelope'} because "
                            f"{rationale}. The rejected evidence remains archived, and JuTrack is "
                            "testing the revised hypothesis."
                        ),
                        meta={
                            "agent": "Research Supervisor",
                            "task_id": decision.get("task_id"),
                            "changed_parameters": changed,
                        },
                    )
                )

        for index, message in enumerate(conversation):
            message = _rehydrate_completion_message(self.root, project.id, message)
            messages.append(
                self._message(
                    content_hash({"index": index, **dict(message)}, length=20),
                    "user" if message.get("role") == "user" else "assistant",
                    "chat",
                    str(message.get("timestamp", project.updated_at)),
                    str(message.get("content", "")),
                    meta=message.get("meta") if isinstance(message.get("meta"), Mapping) else None,
                    attachments=(
                        [
                            dict(item)
                            for item in message.get("attachments", [])
                            if isinstance(item, Mapping)
                        ]
                        if isinstance(message.get("attachments"), (list, tuple))
                        else None
                    ),
                )
            )

        update_labels = {
            "plan": "Plan",
            "progress": "Progress",
            "finding": "Finding",
            "decision": "Decision",
            "repair": "Repair",
            "next": "Next",
        }
        # Retain the scientific story from its contract onwards. Operational
        # counters belong in one expandable log, not a rolling window which
        # eventually removes the opening hypothesis and reference evidence.
        updates, activity = _separate_research_updates(
            workspace_research_updates(self.root, project.id, limit=None)
        )
        for update in updates:
            kind = str(update.get("kind", "finding"))
            label = update_labels.get(kind, "Research update")
            stage = str(update.get("stage", "Research update"))
            detail = str(update.get("detail", "")).strip()
            attachments: list[Mapping[str, Any]] = []
            metrics = update.get("metrics", {})
            if isinstance(metrics, Mapping) and metrics:
                metric_rows = [
                    {"quantity": str(key), "value": value, "unit": ""}
                    for key, value in metrics.items()
                    if isinstance(value, (str, int, float, bool))
                ][:16]
                if metric_rows:
                    attachments.append(
                        {"type": "metrics", "title": "Current numerical values", "rows": metric_rows}
                    )
            tables = update.get("tables", [])
            if isinstance(tables, list) and tables:
                attachments.append(
                    {
                        "type": "evidence_summary",
                        "title": stage,
                        "summary": detail,
                        "tables": [dict(item) for item in tables if isinstance(item, Mapping)],
                        "run_id": update.get("run_id"),
                        "lattice_id": update.get("candidate_id"),
                        "scientific_stage": {key: update[key] for key in (
                            "stage_id", "hypothesis_id", "candidate_id", "parent_candidate_id",
                            "outcome", "interpretation", "next_decision",
                        ) if update.get(key) is not None},
                    }
                )
            for plot in update.get("plots", []) if isinstance(update.get("plots"), list) else []:
                row = {"path": str(plot)} if isinstance(plot, str) else dict(plot) if isinstance(plot, Mapping) else {}
                relative = str(row.get("path", "")).strip()
                if not relative:
                    continue
                try:
                    resolved = resolve_workspace_artifact(self.root, project.id, relative)
                except LookupError:
                    continue
                attachments.append(
                    {
                        "type": "image",
                        "title": str(row.get("title") or Path(relative).stem.replace("_", " ")),
                        "url": (
                            f"/api/projects/{quote(project.id, safe='')}/workspace-artifacts/"
                            + quote(relative, safe="/")
                        ),
                        "media_type": (
                            "image/svg+xml" if resolved.suffix.casefold() == ".svg" else "image/png"
                        ),
                        "path": relative,
                        "content_sha256": _snapshot_image_digest(resolved),
                    }
                )
            messages.append(
                self._message(
                    "research-update:" + str(update.get("id", content_hash(update, length=16))),
                    "assistant",
                    "progress",
                    str(update.get("timestamp", project.updated_at)),
                    f"**{label} — {stage}**" + (f"\n\n{detail}" if detail else ""),
                    meta={"runtime": "codex", "progress_kind": kind,
                          **{key: update[key] for key in (
                              "stage_id", "hypothesis_id", "candidate_id", "parent_candidate_id",
                              "outcome", "scientific_role", "interpretation", "next_decision",
                          ) if update.get(key) is not None}},
                    attachments=attachments,
                )
            )
        if activity:
            messages.append(self._message(
                project.id + ":calculation-activity", "assistant", "activity",
                str(activity[-1].get("timestamp", project.updated_at)), "",
                meta={"runtime": "codex", "scientific_role": "activity"},
                attachments=[{"type": "activity", "title": "Calculation activity and runtime details",
                              "rows": [{"title": item.get("stage"), "summary": item.get("detail"),
                                        "status": item.get("outcome") or "recorded",
                                        "run_id": item.get("run_id"),
                                        "timestamp": item.get("timestamp")}
                                       for item in activity]}],
            ))

        if project.status == "complete":
            attachments: list[Mapping[str, Any]] = [
                {
                    "type": "metrics",
                    "title": "Headline design results",
                    "rows": project_key_metrics(project),
                }
            ]
            for artifact_id, title, expanded in (
                ("lattice", "Full-ring survey", False),
                ("optics", "One-cell lattice and matched optics", True),
                ("radiation", "Radiation equilibrium", False),
                ("chromaticity", "Chromaticity correction", False),
                ("dynamic-aperture", "Off-momentum dynamic aperture", False),
                ("robustness", "Error correction ensemble", False),
                ("cross-validation", "Independent numerical check", False),
            ):
                if self._has_artifact(project, artifact_id):
                    attachments.append(
                        _artifact_attachment(
                            project.id, project.revision, artifact_id, title, expanded
                        )
                    )
            if self._has_agent_plots(project):
                attachments.append(
                    _artifact_attachment(
                        project.id,
                        project.revision,
                        "research-plots",
                        "Additional research plots",
                        False,
                    )
                )
            from .research_reporting import saved_report
            if saved_report(self.root, project.id):
                attachments.append(
                    {
                        "type": "download",
                        "title": "Download scientific report",
                        "url": f"/api/projects/{project.id}/report",
                    }
                )
            if project.active_lattice_id:
                attachments.extend(
                    _lattice_download_attachments(project.id, project.active_lattice_id)
                )
            messages.append(
                self._message(
                    project.id + f":final:{project.revision}",
                    "assistant",
                    "final",
                    project.updated_at,
                    final_result_text(project),
                    attachments=attachments,
                )
            )

        if job is not None and job.state is JobState.FAILED:
            current = next(
                (task for task in project.tasks if task.id == project.current_task_id), None
            )
            stage = current.title if current is not None else "the current research stage"
            detail = job.detail.strip() or "The background research service was interrupted."
            if job.controller_status and job.controller_status.startswith("codex_"):
                content = (
                    "The research turn paused, but its thread, workspace, generated code, "
                    f"and JuTrack artifacts were preserved. {detail} Send another message to "
                    "resume the same session."
                )
            elif job.controller_status == "agent_plan_failed":
                content = (
                    "I could not start the research graph because the reasoning service did not "
                    f"return a usable plan. {detail} Your goal is preserved; **retry** resumes "
                    "planning without launching a template workflow."
                )
            elif job.controller_status == "agent_turn_failed":
                content = (
                    "I could not complete that chat action because its reasoning or tool service "
                    f"was interrupted. {detail} No accepted result or lattice state was changed."
                )
            else:
                content = (
                    f"The background research worker was interrupted while handling **{stage}**. "
                    f"{detail} This is not a rejected physics result. Accepted evidence and the "
                    "current stage are preserved, so **retry** can resume safely."
                )
            messages.append(
                self._message(
                    job.job_id + f":failed:{job.revision}",
                    "assistant",
                    "error",
                    job.updated_at,
                    content,
                    meta={"job_id": job.job_id, "controller_status": job.controller_status},
                )
            )

        job_is_active = job is not None and job.state in {JobState.QUEUED, JobState.RUNNING}
        if project.status != "complete" and not job_is_active:
            current = next(
                (task for task in project.tasks if task.id == project.current_task_id), None
            )
            if current is not None and (
                current.status == "needs_input" or current.capability == "scientist_dialogue"
            ):
                content = current.scientist_question or project.next_action
                messages.append(
                    self._message(
                        project.id + f":question:{project.revision}",
                        "assistant",
                        "question",
                        project.updated_at,
                        content,
                    )
                )

        messages.sort(key=lambda item: (str(item["timestamp"]), _message_priority(item["kind"])))
        messages = [present_local_artifacts(self.root, project.id, item) for item in messages]
        deduplicate_chat_attachments(messages)
        messages = compact_operational_messages(messages)
        return messages

    def _message(
        self,
        message_id: str,
        role: str,
        kind: str,
        timestamp: str,
        content: str,
        *,
        meta: Mapping[str, Any] | None = None,
        attachments: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": message_id,
            "role": role,
            "kind": kind,
            "timestamp": timestamp,
            "content": content,
            "meta": dict(meta or {}),
            "attachments": list(attachments or []),
        }

    def _recorded_project_backend(self, project_id: str) -> Mapping[str, Any]:
        """Last recorded research model, independent of the next-turn selection."""
        backend = CodexRuntimeStore(self.root, project_id).load().get("backend")
        if isinstance(backend, Mapping) and backend.get("model"):
            return dict(backend)
        return {"model": "", "name": "Model not recorded"}

    def _project_meta(self, project: ResearchProject) -> Mapping[str, Any]:
        accepted = sum(task.status == "accepted" for task in project.tasks)
        current = next((task for task in project.tasks if task.id == project.current_task_id), None)
        meta = {
            "id": project.id,
            "title": project.title,
            "goal": project.goal,
            "status": project.status,
            "revision": project.revision,
            "updated_at": project.updated_at,
            "active_lattice_id": project.active_lattice_id,
            "runtime": "codex",
            "backend": self._recorded_project_backend(project.id),
            "research_network": self._research_network_meta(),
            "progress": {"accepted": accepted, "total": len(project.tasks)},
            "current_task": (
                {
                    "id": current.id,
                    "title": current.title,
                    "status": current.status,
                    "capability": current.capability,
                }
                if current
                else None
            ),
        }
        origin_path = self.root / project.id / "workspace" / "RESTORED_CHECKPOINT.json"
        try:
            if origin_path.stat().st_size <= 4 * 1024 * 1024:
                origin = json.loads(origin_path.read_text(encoding="utf-8"))
                if (isinstance(origin, dict) and
                        re.fullmatch(r"project-[A-Za-z0-9_-]+", str(origin.get("source_project", ""))) and
                        re.fullmatch(r"[0-9]+-[0-9a-f]+", str(origin.get("source_checkpoint", "")))):
                    meta["checkpoint_origin"] = {key: origin[key]
                                                 for key in ("source_project", "source_checkpoint")}
        except (OSError, ValueError):
            pass
        return meta

    @staticmethod
    def _research_network_meta() -> Mapping[str, Any]:
        try:
            environment = ResearchEnvironment.load()
            enabled = environment.internet_access
            local_mcp_enabled = environment.local_mcp_enabled
            remote_mcp_enabled = environment.remote_mcp_enabled
        except ResearchEnvironmentError:
            # Fail closed: an invalid runtime must never look online in the UI.
            enabled = False
            local_mcp_enabled = False
            remote_mcp_enabled = False
        return {
            "enabled": enabled,
            "label": "Web online" if enabled else "Web offline",
            "local_mcp_enabled": local_mcp_enabled,
            "remote_mcp_enabled": remote_mcp_enabled,
        }

    def _live_state(
        self, project: ResearchProject, job: ResearchJob | None
    ) -> Mapping[str, Any] | None:
        codex = CodexRuntimeStore(self.root, project.id).load()
        if str(codex.get("status")) in {"starting", "running"} or (
            job is not None
            and job.action == "codex_turn"
            and job.state in {JobState.QUEUED, JobState.RUNNING}
        ):
            checkpoint_status = str(codex.get("status", "idle"))
            job_owns_live_state = (
                job is not None
                and job.action == "codex_turn"
                and checkpoint_status not in {"starting", "running"}
            )
            label = (
                "Research increment is queued"
                if job_owns_live_state and job.state is JobState.QUEUED
                else (
                    "Finalizing the saved research checkpoint"
                    if job_owns_live_state
                    else str(codex.get("label") or "Working with JuTrack")
                )
            )
            progress = research_progress(codex)
            updates = workspace_research_updates(self.root, project.id)
            progress["updates"] = updates
            if updates and not str(progress.get("narrative", "")).strip():
                progress["narrative"] = str(
                    updates[-1].get("detail") or updates[-1].get("stage") or ""
                )[:1_000]
            if updates and not progress.get("running"):
                progress["title"] = updates[-1]["stage"]
            if job_owns_live_state:
                progress = {
                    **progress,
                    "title": label,
                    "increment": job.payload.get("increment", 1),
                    "max_increments": job.payload.get("max_increments", 1),
                    "elapsed_seconds": 0,
                    "running": 0,
                    "milestones": [],
                }
            return {
                "job_id": job.job_id if job is not None else None,
                "state": job.state.value if job_owns_live_state else checkpoint_status,
                "label": label,
                "task_id": None,
                "runtime": "codex",
                "backend": codex.get("backend", describe_codex_backend()),
                "research_network": self._research_network_meta(),
                "durable": True,
                "version": codex.get("version"),
                "thread_id": codex.get("thread_id"),
                "turn_id": codex.get("turn_id"),
                "activities": codex.get("activities", []),
                "partial_response": codex.get("partial_response", ""),
                "progress": progress,
                "token_usage": codex.get("token_usage", {}),
                "pending_count": len(
                    list(CodexRuntimeStore(self.root, project.id).pending_dir.glob("*.json"))
                ),
            }
        if job is None or job.state not in {JobState.QUEUED, JobState.RUNNING}:
            return None
        if job.action == "agent_plan":
            return {
                "job_id": job.job_id,
                "state": job.state.value,
                "label": "The LLM Research Lead is designing a goal-specific tool graph",
                "task_id": None,
                "durable": True,
            }
        if job.action == "agent_turn":
            return {
                "job_id": job.job_id,
                "state": job.state.value,
                "label": "The LLM Research Lead is selecting and running the next tool",
                "task_id": None,
                "durable": True,
            }
        current = next((task for task in project.tasks if task.id == project.current_task_id), None)
        return {
            "job_id": job.job_id,
            "state": job.state.value,
            "label": (
                f"JuTrack is working on {current.title}"
                if current is not None
                else "Brilliance research is running"
            ),
            "task_id": current.id if current else None,
            "durable": True,
        }

    def _load(self, project_id: str) -> ResearchProject:
        project = self.store.load(project_id)
        if project is None:
            raise LookupError(f"Research project not found: {project_id}")
        return project

    def _queue(
        self,
        project: ResearchProject,
        *,
        action: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ResearchJob:
        job = self.jobs.submit(project.id, project.revision, action=action, payload=payload)
        if job.state is JobState.QUEUED:
            launch_research_job(self.root, job)
        return self.jobs.load(project.id, job.job_id) or job

    def _active_agent_job(self, project_id: str) -> ResearchJob | None:
        return next(
            (
                job
                for job in reversed(self.jobs.list(project_id))
                if job.action == "codex_turn"
                and job.state in {JobState.QUEUED, JobState.RUNNING}
            ),
            None,
        )

    def _latest_agent_job(self, project_id: str) -> ResearchJob | None:
        return next(
            (
                job
                for job in reversed(self.jobs.list(project_id))
                if job.action == "codex_turn"
            ),
            None,
        )

    def _latest_research_job(self, project_id: str) -> ResearchJob | None:
        jobs = [
            job
            for job in self.jobs.list(project_id)
            if job.action == "codex_turn"
        ]
        return jobs[-1] if jobs else None

    def _active_codex_job(self, project_id: str) -> ResearchJob | None:
        return self.jobs.active(project_id, action="codex_turn")

    def _active_compute_job(self, project_id: str) -> ResearchJob | None:
        return self.jobs.active(project_id, action="expert_tool")

    def _active_compute_jobs(self, project_id: str) -> list[ResearchJob]:
        return [
            item
            for item in self.jobs.list(project_id)
            if item.action == "expert_tool"
            and item.state
            in {
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.CANCEL_REQUESTED,
            }
        ]

    def _computation_meta(self, job: ResearchJob) -> Mapping[str, Any]:
        progress = None
        progress_path = None
        directory = self.root / job.project_id / "workspace" / "progress"
        paths = (
            sorted(
                directory.glob("*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if directory.is_dir()
            else []
        )
        for path in paths:
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(candidate, Mapping) and candidate.get("compute_job_id") == job.job_id:
                progress = dict(candidate)
                progress_path = path
                break
        progress = dict(progress or {})
        try:
            elapsed_s = max(
                0.0,
                datetime.now().astimezone().timestamp()
                - datetime.fromisoformat(job.created_at).timestamp(),
            )
        except ValueError:
            elapsed_s = float(progress.get("elapsed_s", 0.0))
        progress["elapsed_s"] = max(elapsed_s, float(progress.get("elapsed_s", 0.0)))
        from .compute_monitor import observe_calculation
        telemetry = observe_calculation(self.root / job.project_id / "workspace", progress_path, progress)
        return {
            "id": job.job_id,
            "revision": job.revision,
            "state": job.state.value,
            "tool_id": job.payload.get("tool_id"),
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "detail": job.detail,
            "progress": progress,
            "telemetry": telemetry,
            "can_cancel": job.state
            in {
                JobState.QUEUED,
                JobState.RUNNING,
            },
            "cancel_requested": job.state is JobState.CANCEL_REQUESTED,
            "durable": True,
        }

    def _capability_result(self, project: ResearchProject, capability: str) -> Mapping[str, Any]:
        task_ids = {task.id for task in project.tasks if task.capability == capability}
        for evidence in reversed(project.evidence):
            if evidence.get("task_id") in task_ids:
                result = evidence.get("structured_result")
                return result if isinstance(result, Mapping) else {}
        return {}

    def _has_artifact(self, project: ResearchProject, artifact_id: str) -> bool:
        capability = {
            "lattice": "lattice_construction",
            "lattice-block": "lattice_construction",
            "optics": "linear_matching",
            "ring-optics": "linear_matching",
            "radiation": "radiation_analysis",
            "chromaticity": "chromatic_correction",
            "dynamic-aperture": "nonlinear_tracking",
            "robustness": "error_correction",
            "cross-validation": "cross_validation",
        }.get(artifact_id)
        return bool(capability and self._capability_result(project, capability))

    def _has_agent_plots(self, project: ResearchProject) -> bool:
        return any(
            isinstance(evidence.get("structured_result"), Mapping)
            and bool(evidence["structured_result"].get("plot_data"))
            for evidence in project.evidence
        )

    def _agent_plot_artifact(self, project: ResearchProject) -> Mapping[str, Any]:
        task_titles = {task.id: task.title for task in project.tasks}
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for evidence in project.evidence:
            result = evidence.get("structured_result")
            if not isinstance(result, Mapping):
                continue
            for row in _mapping_rows(result.get("plot_data"))[:5000]:
                try:
                    point = {
                        "series": str(row.get("series", "result")),
                        "x": float(row.get("x")),
                        "y": float(row.get("y")),
                        "provenance": str(row.get("provenance", "")),
                    }
                except (TypeError, ValueError):
                    continue
                key = (
                    str(task_titles.get(evidence.get("task_id"), "Research result")),
                    str(row.get("x_label", "x")),
                    str(row.get("y_label", "y")),
                )
                groups.setdefault(key, []).append(point)
        return {
            "kind": "research_plots",
            "title": "Additional research plots",
            "groups": [
                {"title": key[0], "x_label": key[1], "y_label": key[2], "rows": rows}
                for key, rows in groups.items()
            ],
        }

    def _plot_reply(
        self, project: ResearchProject, message: str
    ) -> tuple[str, list[Mapping[str, Any]]] | None:
        request = message.lower()
        if not _looks_like_plot_request(request):
            return None
        inferred = request
        # Only inherit a previous subject for genuinely anaphoric follow-ups.
        # An explicit but currently unregistered subject (for example x-px
        # phase space) must reach the Research Lead instead of walking farther
        # back through history and attaching an unrelated legacy plot.
        if not _plot_subjects(inferred) and _is_vague_plot_followup(request):
            for previous in reversed(self.conversations.messages(project.id)):
                if previous.get("role") != "user":
                    continue
                candidate = str(previous.get("content", "")).lower()
                if _plot_subjects(candidate):
                    inferred = candidate + " " + request
                    break

        subjects = _plot_subjects(inferred)
        if not subjects:
            return None

        full_ring = any(
            term in inferred
            for term in (
                "full ring",
                "whole ring",
                "entire ring",
                "ring survey",
                "floor plan",
                "survey",
            )
        )
        lattice_result = self._capability_result(project, "lattice_construction")
        proxy = _is_proxy_ring_result(lattice_result)
        family = _mba_family_label(project, lattice_result)
        attachments: list[Mapping[str, Any]] = []
        labels: list[str] = []
        if "lattice" in subjects:
            # A cell-level optics artifact already carries its lattice strip on
            # the same longitudinal coordinate.  Keep the separate artifact
            # only for lattice-only requests or a physical full-ring survey.
            if full_ring or "optics" not in subjects:
                artifact_id = "lattice" if full_ring else "lattice-block"
                title = (
                    (
                        "Repeated-cell proxy survey"
                        if full_ring
                        else f"One {family} recurring-cell proxy"
                    )
                    if proxy
                    else ("Full-ring survey" if full_ring else f"One periodic {family} superperiod")
                )
                attachments.append(
                    _artifact_attachment(project.id, project.revision, artifact_id, title, True)
                )
                labels.append(title.lower())
        if "optics" in subjects:
            artifact_id = "ring-optics" if full_ring else "optics"
            title = (
                (
                    "Repeated-cell proxy lattice and optics"
                    if full_ring
                    else "One proxy-superperiod lattice and optics"
                )
                if proxy
                else (
                    "Full-ring lattice and optics"
                    if full_ring
                    else "One-cell lattice and matched optics"
                )
            )
            attachments.append(
                _artifact_attachment(project.id, project.revision, artifact_id, title, True)
            )
            labels.append(title.lower())
        remaining = {
            "radiation": ("radiation", "Radiation equilibrium"),
            "chromaticity": ("chromaticity", "Chromaticity correction"),
            "dynamic_aperture": ("dynamic-aperture", "Off-momentum dynamic aperture"),
            "robustness": ("robustness", "Error correction ensemble"),
            "validation": ("cross-validation", "Independent numerical check"),
        }
        for subject, (artifact_id, title) in remaining.items():
            if subject in subjects:
                attachments.append(
                    _artifact_attachment(project.id, project.revision, artifact_id, title, True)
                )
                labels.append(title.lower())
        available = [
            item for item in attachments if self._has_artifact(project, str(item["artifact_id"]))
        ]
        if not available:
            return (
                "That plot is not available because its JuTrack stage has not produced accepted "
                "arrays yet. I will not invent a figure from prose.",
                [],
            )
        if proxy and ({"lattice", "optics"} & subjects):
            answer = (
                "Here is the accepted **homogeneous recurring-cell proxy** view. It is derived "
                f"from JuTrack arrays, but it is not a completed practical {family} achromat: no "
                "matched dispersion suppressors or insertion straights were constructed."
            )
        elif not full_ring and ({"lattice", "optics"} & subjects):
            answer = (
                f"Here is **one periodic {family} superperiod**, extracted from the accepted full-ring "
                "JuTrack lattice and matched Twiss arrays. The longitudinal axis is limited to "
                "one of the 24 identical superperiods so the magnet sequence and optical "
                "functions remain readable. Ask for the **full-ring survey** explicitly if needed."
            )
        else:
            answer = "Here is the accepted JuTrack " + " and ".join(labels) + "."
        return answer, available

    def _optimization_plot_reply(
        self, project: ResearchProject, message: str
    ) -> tuple[str, list[Mapping[str, Any]]] | None:
        """Render an existing optimizer trace directly instead of redoing research."""

        if not _looks_like_optimization_plot_request(message):
            return None
        project_directory = self.root / project.id
        try:
            rendered = render_latest_optimization_plot(project_directory)
        except OptimizationPlotUnavailable:
            return None

        artifact_root = (project_directory / "workspace" / "artifacts").resolve()

        def attachment(path_key: str, kind: str, title: str) -> dict[str, Any]:
            path = Path(str(rendered[path_key])).resolve()
            relative = path.relative_to(artifact_root).as_posix()
            return {
                "type": kind,
                "title": title,
                "url": (
                    f"/api/projects/{quote(project.id, safe='')}/workspace-artifacts/"
                    + quote(relative, safe="/")
                ),
                "artifact_path": relative,
                "media_type": "image/png" if kind == "image" else "text/csv",
                "size": path.stat().st_size,
            }

        baseline = rendered.get("baseline_emittance_pm")
        current = rendered.get("selected_emittance_pm")
        improvement = rendered.get("relative_improvement")
        selected_lattice = str(rendered.get("selected_lattice_id", ""))
        is_working_lattice = bool(
            selected_lattice and selected_lattice == str(project.active_lattice_id or "")
        )
        status_label = (
            "Current working lattice" if is_working_lattice else "Latest optimized branch"
        )
        baseline_text = f"{float(baseline):.3f}" if baseline is not None else "—"
        current_text = f"{float(current):.3f}" if current is not None else "—"
        improvement_text = f"{100.0 * float(improvement):.1f}%" if improvement is not None else "—"
        generation_range = f"{rendered['generation_start']}–{rendered['generation_end']}"
        population = rendered.get("population")
        evaluations = rendered.get("evaluated_candidates")
        evaluation_text = f"{int(evaluations):,}" if evaluations is not None else "—"
        answer = (
            "Here is the optimization convergence plotted directly from the archived numeric "
            "history. JuTrack and the optimizer were not rerun.\n\n"
            "| Quantity | Value |\n"
            "|---|---:|\n"
            f"| Baseline natural emittance | {baseline_text} pm rad |\n"
            f"| {status_label} natural emittance | {current_text} pm rad |\n"
            f"| Reduction from baseline | {improvement_text} |\n"
            f"| Generations shown | {generation_range} |\n"
            f"| Population | {population if population is not None else '—'} |\n"
            f"| Candidate evaluations | {evaluation_text} |"
        )
        if not is_working_lattice and selected_lattice:
            answer += (
                "\n\nThe plotted endpoint is the latest validated optimization branch "
                f"`{selected_lattice}`; it has not been promoted to the canonical working lattice."
            )
        attachments = [
            attachment("png", "image", "MBA emittance optimization convergence"),
            attachment("csv", "download", "Optimization history data"),
        ]
        return answer, attachments


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _lattice_elements(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    elements = _mapping_rows(result.get("lattice_elements"))
    if elements:
        return elements
    artifact = result.get("lattice_artifact")
    return _mapping_rows(artifact.get("elements")) if isinstance(artifact, Mapping) else []


def _bend_inventory(elements: list[Mapping[str, Any]]) -> Mapping[str, int]:
    angles = []
    for item in elements:
        if str(item.get("type", item.get("kind", ""))).upper() not in {"SBEND", "RBEND"}:
            continue
        try:
            angle = float(item.get("angle_rad", item.get("angle", 0.0)))
        except (TypeError, ValueError):
            continue
        if abs(angle) > 0.0:
            angles.append(angle)
    return {
        "primary_bends": sum(angle > 0.0 for angle in angles),
        "reverse_bends": sum(angle < 0.0 for angle in angles),
        "physical_bending_magnets": len(angles),
    }


def _is_proxy_ring_result(result: Mapping[str, Any]) -> bool:
    if str(result.get("design_scope", "")) == "homogeneous_recurring_cell_proxy":
        return True
    artifact = result.get("lattice_artifact")
    metadata = artifact.get("metadata") if isinstance(artifact, Mapping) else None
    if not isinstance(metadata, Mapping):
        return False
    if str(metadata.get("design_scope", "")) == "homogeneous_recurring_cell_proxy":
        return True
    try:
        return str(metadata.get("construction", "")) == "exact periodic repetition" and int(
            metadata.get("repetitions", 1)
        ) > int(metadata.get("source_periodicity", metadata.get("repetitions", 1)))
    except (TypeError, ValueError):
        return False


def _lattice_physical_fingerprint(result: Mapping[str, Any]) -> str | None:
    value = result.get("physical_fingerprint")
    if value:
        return str(value)
    artifact = result.get("lattice_artifact")
    if not isinstance(artifact, Mapping):
        return None
    try:
        return LatticeIR.from_dict(artifact).physical_fingerprint
    except (KeyError, TypeError, ValueError):
        return None


def _mba_family_label(
    project: ResearchProject,
    result: Mapping[str, Any],
) -> str:
    """Resolve the requested MBA family without assuming seven bends."""

    candidates: list[Any] = [result.get("lattice_family")]
    artifact = result.get("lattice_artifact")
    if isinstance(artifact, Mapping):
        candidates.append(artifact.get("lattice_family"))
        metadata = artifact.get("metadata")
        if isinstance(metadata, Mapping):
            candidates.extend((metadata.get("lattice_family"), metadata.get("n_cell_equivalents")))
    candidates.extend((project.goal, project.machine_summary))
    for candidate in candidates:
        text = str(candidate or "").strip()
        match = re.search(r"\b(?P<h>h[- ]?)?(?P<n>\d+)\s*ba\b", text, re.IGNORECASE)
        if match:
            return f"{'H' if match.group('h') else ''}{int(match.group('n'))}BA"
        try:
            bend_count = int(candidate)
        except (TypeError, ValueError):
            continue
        if bend_count >= 2:
            return f"{bend_count}BA"
    return "MBA"


def _superperiod_count(
    project: ResearchProject,
    lattice_result: Mapping[str, Any],
    element_count: int,
) -> int:
    """Resolve the physical achromat count, not the homogeneous bend-cell count."""
    candidates: list[Any] = []
    artifact = lattice_result.get("lattice_artifact")
    if isinstance(artifact, Mapping):
        metadata = artifact.get("metadata")
        if isinstance(metadata, Mapping):
            candidates.append(metadata.get("source_periodicity"))
        candidates.append(artifact.get("source_periodicity"))
    for known in project.knowns:
        match = re.search(r"\bperiodicity\s*=\s*(\d+)", str(known), re.IGNORECASE)
        if match:
            candidates.append(match.group(1))
    match = re.search(r"\b(\d+)\s+(?:\w+\s+)?superperiods?\b", project.machine_summary)
    if match:
        candidates.append(match.group(1))
    candidates.append(lattice_result.get("repetitions"))
    for candidate in candidates:
        try:
            count = int(candidate)
        except (TypeError, ValueError):
            continue
        if count > 0 and (
            element_count == 0 or (count <= element_count and element_count % count == 0)
        ):
            return count
    return 1


def _looks_like_plot_request(text: str) -> bool:
    plot_terms = ("plot", "figure", "chart", "graph", "visualiz", "draw")
    return any(term in text for term in plot_terms) or (
        any(term in text for term in ("show", "display", "give me", "i need"))
        and bool(_plot_subjects(text))
    )


def _looks_like_optimization_plot_request(text: str) -> bool:
    """Recognize only explicit requests to visualize an existing optimizer trace."""

    normalized = text.lower()
    plot_terms = ("plot", "figure", "chart", "graph", "visualiz", "show", "display")
    optimization_terms = (
        "optimization",
        "optimisation",
        "optimizer",
        "optimiser",
        "convergence",
        "mopso",
        "pso progress",
        "generation history",
    )
    return any(term in normalized for term in plot_terms) and any(
        term in normalized for term in optimization_terms
    )


def _is_vague_plot_followup(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return normalized in {
        "plot it",
        "show it",
        "display it",
        "draw it",
        "show the plot",
        "display the plot",
        "give me the plot",
        "i need the plot",
        "show the figure",
        "display the figure",
        "give me the figure",
        "i need the figure",
        "where is the plot",
        "where is the figure",
    }


def _plot_subjects(text: str) -> set[str]:
    subjects: set[str] = set()
    if any(
        term in text
        for term in (
            "lattice",
            "period cell",
            "periodic cell",
            "superperiod",
            "magnet layout",
            "cell layout",
        )
    ):
        subjects.add("lattice")
    if any(term in text for term in ("optic", "twiss", "beta", "dispersion", "betatron function")):
        subjects.add("optics")
    if any(
        term in text
        for term in (
            "radiation",
            "emittance",
            "damping partition",
            "radiation integral",
            "energy spread",
        )
    ):
        subjects.add("radiation")
    if "chromatic" in text or "sextupole response" in text:
        subjects.add("chromaticity")
    if any(
        term in text
        for term in ("dynamic aperture", "momentum aperture", "acceptance", "tracking boundary")
    ):
        subjects.add("dynamic_aperture")
    if any(
        term in text for term in ("robust", "error seed", "orbit correction", "correction ensemble")
    ):
        subjects.add("robustness")
    if any(
        term in text
        for term in ("cross-validation", "cross validation", "comparison check", "validation table")
    ):
        subjects.add("validation")
    return subjects


def _artifact_attachment(
    project_id: str,
    revision: int,
    artifact_id: str,
    title: str,
    expanded: bool,
) -> Mapping[str, Any]:
    return {
        "type": "artifact",
        "artifact_id": artifact_id,
        "title": title,
        "expanded": expanded,
        "url": f"/api/projects/{project_id}/artifacts/{artifact_id}?revision={revision}",
    }


def _lattice_download_attachments(project_id: str, lattice_id: str) -> list[Mapping[str, Any]]:
    base = f"/api/projects/{project_id}/lattices/{lattice_id}"
    return [
        {
            "type": "download",
            "title": f"Download {lattice_id} canonical JSON",
            "url": f"{base}/definition",
        },
        {
            "type": "download",
            "title": f"Download {lattice_id} JuTrack reproduction code",
            "url": f"{base}/reproduction",
        },
        {
            "type": "download",
            "title": f"Download {lattice_id} provenance",
            "url": f"{base}/provenance",
        },
    ]


def _evidence_lattice_ids(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    result = evidence.get("structured_result")
    if not isinstance(result, Mapping):
        return ()
    candidates: list[Any] = [result.get("lattice_artifact")]
    best_run = result.get("best_run")
    if isinstance(best_run, Mapping):
        raw = best_run.get("raw_result")
        if isinstance(raw, Mapping):
            candidates.append(raw.get("lattice_artifact"))
    return tuple(
        dict.fromkeys(
            str(item.get("lattice_id"))
            for item in candidates
            if isinstance(item, Mapping) and item.get("lattice_id")
        )
    )


def _capability_plot(capability: str, project_id: str, revision: int) -> Mapping[str, Any] | None:
    mapping = {
        "lattice_construction": ("lattice", "View lattice", False),
        "linear_matching": ("optics", "View optics", False),
        "radiation_analysis": ("radiation", "View radiation result", False),
        "chromatic_correction": ("chromaticity", "View chromaticity", False),
        "nonlinear_tracking": ("dynamic-aperture", "View dynamic aperture", False),
        "error_correction": ("robustness", "View correction ensemble", False),
        "cross_validation": ("cross-validation", "View numerical comparison", False),
    }
    selected = mapping.get(capability)
    return (
        _artifact_attachment(project_id, revision, selected[0], selected[1], selected[2])
        if selected
        else None
    )


def _rehydrate_completion_message(
    project_root: Path, project_id: str, message: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Repair the view of legacy final attachments, never saved chat or evidence.

    The existing immutable completion proof is the sole source of authority;
    neither file recency nor another project's candidate can select evidence.
    """
    meta = message.get("meta", {})
    proof = meta.get("completion_evidence") if isinstance(meta, Mapping) else None
    if (message.get("role") != "assistant" or not isinstance(proof, Mapping)
            or not proof.get("evidence_run_id") or proof.get("execution_valid") is not True):
        return message
    old_attachments = message.get("attachments", [])
    if any(isinstance(item, Mapping) and item.get("presentation_role") == "selected_evidence"
           for item in old_attachments):
        return message
    artifact_root = (Path(project_root).resolve() / project_id / "workspace" / "artifacts").resolve()
    signature: tuple[Any, ...] = ()
    try:
        source = _resolve_completion_artifact(artifact_root, str(proof.get("evidence_full_result") or ""))
        if source.name == "result.json" and source.is_file():
            # Stat only the one cited receipt directory; hashing and decoding
            # are cached. Long project histories do not cause quadratic scans.
            signatures = []
            for path in sorted(source.parent.rglob("*")):
                path.resolve().relative_to(artifact_root)
                if path.is_file():
                    stat = path.stat()
                    signatures.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            signature = tuple(signatures)
    except (OSError, ValueError):
        pass
    attachments = json.loads(_cached_completion_attachments(
        str(Path(project_root).resolve()), project_id,
        json.dumps(dict(proof), sort_keys=True, ensure_ascii=False), signature,
    ))
    alternatives = []
    for item in old_attachments:
        if (isinstance(item, Mapping) and item.get("type") == "evidence_summary"
                and item.get("run_id") != proof["evidence_run_id"]):
            alternatives.append({"run_id": item.get("run_id"), "summary": item.get("summary", ""),
                                 "status": item.get("status"), "disposition": "not_selected_for_this_answer"})
    figures = sum(item.get("type") == "image" for item in attachments)
    for item in attachments:
        if item.get("presentation_role") == "selected_evidence" and item.get("type") == "evidence_summary":
            item["history_rehydrated"] = (
                f"Showing the exact calculation cited by this saved answer ({figures} figures). "
                "The original message and research files have not been changed."
            )
    if alternatives:
        attachments.append({"type": "alternative_results",
                            "title": "Earlier attached calculations — not selected for this answer",
                            "rows": alternatives})
    return {**dict(message), "attachments": attachments,
            "meta": {**dict(meta), "evidence_rehydrated": True}}


@lru_cache(maxsize=128)
def _cached_completion_attachments(
    project_root: str, project_id: str, proof_json: str, signature: tuple[Any, ...],
) -> str:
    del signature  # Cache invalidates when any file in the cited receipt changes.
    attachments = _turn_attachments(
        Path(project_root), project_id, 0, completion_evidence=json.loads(proof_json),
        include_recent_results=False,
    )
    return json.dumps(attachments, ensure_ascii=False)


def _separate_research_updates(
    updates: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Separate scientific evidence from operational updates without erasing either.

    Topic modules supply explicit roles. Legacy findings, decisions, repairs,
    plans and progress containing scientific artifacts remain in the chat;
    runtime starts and plain counters are available in the activity log.
    """
    milestones: list[Mapping[str, Any]] = []
    activity: list[Mapping[str, Any]] = []
    for update in updates:
        role = update.get("scientific_role")
        operational = role == "activity" or (
            role != "milestone" and update.get("kind", "progress") == "progress"
            and not update.get("tables") and not update.get("plots")
        )
        (activity if operational else milestones).append(update)
    return milestones, activity


def _message_priority(kind: str) -> int:
    return {
        "goal": 0,
        "contract": 1,
        "decision": 2,
        "stage": 3,
        "error": 4,
        "question": 5,
        "chat": 6,
        "progress": 7,
        "final": 8,
    }.get(kind, 9)


def _job_meta(job: ResearchJob | None) -> Mapping[str, Any] | None:
    if job is None:
        return None
    return {
        "id": job.job_id,
        "revision": job.revision,
        "state": job.state.value,
        "action": job.action,
        "detail": job.detail,
    }
