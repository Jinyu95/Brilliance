"""Thin Codex session adapter for the Brilliance research chat.

Codex owns conversation, reasoning, coding, and shell/tool iteration. Brilliance
adds only the configured accelerator runtime, durable state, live presentation,
and scientific artifact boundaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from .artifacts import content_hash, json_safe
from .codex_backends import CodexBackend, CodexBackendError, resolve_codex_backend
from .presentation import ResearchConversationStore
from .projects import ResearchProject
from .research_environment import ResearchEnvironment, ResearchEnvironmentError

CODEX_RUNTIME_SCHEMA_VERSION = "2.0"
ACCELERATOR_AGENT_PROFILE_VERSION = "11.0"
TURN_SHUTDOWN_RESERVE_SECONDS = 10.0
RESEARCH_STATUS_PATTERN = re.compile(
    r"\s*<!--\s*brilliance-research-status:\s*"
    r"(continue|complete|needs_input)\s*-->\s*",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def configured_max_increments() -> int:
    try:
        default = ResearchEnvironment.load().max_autonomous_increments
    except ResearchEnvironmentError:
        default = 1
    return max(
        1,
        min(20, _safe_int(os.environ.get("BRILLIANCE_RESEARCH_MAX_INCREMENTS"), default)),
    )


def configured_turn_timeout(environment: ResearchEnvironment) -> int:
    legacy = max(
        30,
        _safe_int(os.environ.get("BRILLIANCE_CODEX_TURN_TIMEOUT"), environment.turn_wall_time_s),
    )
    return max(30, _safe_int(os.environ.get("BRILLIANCE_CODEX_INCREMENT_TIMEOUT"), legacy))


def configured_auto_continue() -> bool:
    configured = os.environ.get("BRILLIANCE_RESEARCH_AUTO_CONTINUE")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return ResearchEnvironment.load().auto_continue
    except ResearchEnvironmentError:
        return False


def configured_max_consecutive_timeouts() -> int:
    return max(
        1,
        min(4, _safe_int(os.environ.get("BRILLIANCE_MAX_CONSECUTIVE_TIMEOUTS", "2"), 2)),
    )


def configured_first_execution_nudge() -> int:
    """Seconds of model-only work allowed before requesting an executable milestone."""

    return max(
        60,
        min(900, _safe_int(os.environ.get("BRILLIANCE_FIRST_EXECUTION_NUDGE_S", "120"), 120)),
    )


def configured_second_execution_nudge() -> int:
    """Absolute turn age for the stronger no-result execution reminder."""

    first = configured_first_execution_nudge()
    return max(
        first + 60,
        min(1_200, _safe_int(os.environ.get("BRILLIANCE_SECOND_EXECUTION_NUDGE_S", "300"), 300)),
    )


def configured_max_pre_execution() -> int:
    """Maximum model-owned preparation time before a numerical program starts."""

    second = configured_second_execution_nudge()
    return max(
        second + 60,
        min(1_200, _safe_int(os.environ.get("BRILLIANCE_MAX_PRE_EXECUTION_S", "180"), 180)),
    )


def configured_chat_update_interval() -> int:
    """Maximum quiet period before Brilliance publishes a durable chat status."""

    return max(
        30,
        min(300, _safe_int(os.environ.get("BRILLIANCE_CHAT_UPDATE_INTERVAL_S", "60"), 60)),
    )


def research_progress(state: Mapping[str, Any]) -> dict[str, Any]:
    activities = [dict(item) for item in state.get("activities", []) if isinstance(item, Mapping)]
    completed_states = {"completed", "complete", "succeeded", "success"}
    failed_states = {"failed", "error", "paused", "cancelled", "canceled"}
    running_states = {"running", "inprogress", "in_progress", "retrying"}

    def status(item: Mapping[str, Any]) -> str:
        return str(item.get("status", "")).replace(" ", "").lower()

    current = next(
        (item for item in reversed(activities) if status(item) in running_states),
        activities[-1] if activities else None,
    )
    milestones: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in reversed(activities):
        title = str(item.get("title", "")).strip()
        if not title or title in seen:
            continue
        seen.add(title)
        milestones.append(
            {
                "id": str(item.get("id", "")),
                "title": title,
                "status": str(item.get("status", "running")),
                "kind": str(item.get("kind", "activity")),
                "timestamp": str(item.get("timestamp", "")),
                "summary": str(item.get("summary", "")),
            }
        )
        if len(milestones) >= 6:
            break
    milestones.reverse()
    elapsed = 0
    if state.get("turn_started_at"):
        try:
            started = datetime.fromisoformat(str(state["turn_started_at"]).replace("Z", "+00:00"))
            elapsed = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
        except ValueError:
            pass
    findings = [
        {
            "title": str(item.get("title", "Research result")),
            "summary": str(item.get("summary", "")),
            "status": str(item.get("status", "completed")),
            "timestamp": str(item.get("timestamp", "")),
        }
        for item in activities
        if str(item.get("summary", "")).strip() and status(item) in completed_states
    ][-4:]
    errors = [
        {
            "title": str(item.get("title", "Calculation issue")),
            "summary": str(item.get("summary", ""))
            or _failure_summary(str(item.get("detail", ""))),
            "timestamp": str(item.get("timestamp", "")),
        }
        for item in activities
        if status(item) in failed_states
    ][-3:]
    return {
        "title": str(
            (current or {}).get("title") or state.get("label") or "Preparing research"
        ),
        "increment": max(1, _safe_int(state.get("increment"), 1)),
        "max_increments": max(1, _safe_int(state.get("max_increments"), 1)),
        "elapsed_seconds": elapsed,
        "completed": sum(status(item) in completed_states for item in activities),
        "running": sum(status(item) in running_states for item in activities),
        "failed": sum(status(item) in failed_states for item in activities),
        "milestones": milestones,
        "latest_results": findings,
        "latest_errors": errors,
        "narrative": str(
            state.get("live_narrative") or state.get("partial_response") or ""
        )[-1_000:],
    }


@dataclass(frozen=True, slots=True)
class CodexTurnOutcome:
    status: str
    detail: str
    thread_id: str | None = None
    turn_id: str | None = None
    final_response: str = ""
    disposition: str = "complete"
    recovery: Mapping[str, str] | None = None
    completion_evidence: Mapping[str, Any] | None = None


class CodexRuntimeStore:
    """Bounded mutable state for one live Codex thread."""

    EVENT_RETENTION = 256
    CONSUMED_INPUT_RETENTION = 32

    def __init__(self, project_root: Path | str, project_id: str):
        self.project_root = Path(project_root).resolve()
        self.project_id = project_id
        self.directory = self.project_root / project_id / "codex"
        self.state_path = self.directory / "state.json"
        self.events_dir = self.directory / "events"
        self.pending_dir = self.directory / "pending"

    def load(self) -> dict[str, Any]:
        if self.state_path.is_file():
            try:
                value = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(value, Mapping):
                    return {**self._initial_state(), **dict(value)}
            except (OSError, json.JSONDecodeError):
                pass
        return self._initial_state()

    def update(self, **changes: Any) -> dict[str, Any]:
        state = self.load()
        state.update(json_safe(changes))
        state.update(
            {
                "schema_version": CODEX_RUNTIME_SCHEMA_VERSION,
                "project_id": self.project_id,
                "version": _safe_int(state.get("version"), 0) + 1,
                "updated_at": _now(),
            }
        )
        self._write_state(state)
        return state

    def start_turn(self, *, label: str, increment: int = 1, max_increments: int = 1) -> dict[str, Any]:
        previous = self.load()
        return self.update(
            status="starting",
            label=label,
            turn_started_at=_now(),
            increment=increment,
            max_increments=max_increments,
            turn_id=None,
            partial_response="",
            live_narrative="",
            error="",
            activities=[],
            token_usage={},
            research_program_started_at="",
            research_program_command="",
            protocol_failure=None,
            consecutive_timeouts=(
                0 if increment <= 1 else _safe_int(previous.get("consecutive_timeouts"), 0)
            ),
        )

    def append_activity(self, activity: Mapping[str, Any]) -> dict[str, Any]:
        state = self.load()
        activities = [dict(item) for item in state.get("activities", []) if isinstance(item, Mapping)]
        normalized = {
            "id": str(activity.get("id") or content_hash(activity, length=16)),
            "kind": str(activity.get("kind", "activity")),
            "title": str(activity.get("title", "Research activity"))[:180],
            "status": str(activity.get("status", "running")),
            "detail": str(activity.get("detail", ""))[:2_000],
            "summary": str(activity.get("summary", ""))[:500],
            "timestamp": str(activity.get("timestamp", _now())),
        }
        index = next(
            (i for i, item in enumerate(activities) if item.get("id") == normalized["id"]),
            None,
        )
        if index is None:
            activities.append(normalized)
        else:
            activities[index] = normalized
        return self.update(activities=activities[-40:], label=_stage_title(normalized))

    def append_event(self, event: Mapping[str, Any]) -> Path:
        payload = {
            "schema_version": CODEX_RUNTIME_SCHEMA_VERSION,
            "project_id": self.project_id,
            "timestamp": _now(),
            **json_safe(event),
        }
        stamp = re.sub(r"[^0-9]", "", str(payload["timestamp"]))[:20]
        path = self.events_dir / f"{stamp}-{content_hash(payload, length=20)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        except FileExistsError:
            pass
        self._prune_files(self.events_dir, keep=self.EVENT_RETENTION)
        return path

    def queue_input(self, message: str) -> Path:
        payload = {
            "schema_version": CODEX_RUNTIME_SCHEMA_VERSION,
            "project_id": self.project_id,
            "timestamp": _now(),
            "message": str(message).strip(),
        }
        stamp = re.sub(r"[^0-9]", "", payload["timestamp"])[:20]
        path = self.pending_dir / f"{stamp}-{content_hash(payload, length=20)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        except FileExistsError:
            pass
        return path

    def consume_pending(self) -> list[str]:
        messages: list[str] = []
        consumed = self.directory / "consumed"
        consumed.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.pending_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                message = str(payload.get("message", "")).strip()
                if message:
                    messages.append(message)
                target = consumed / path.name
                path.replace(target) if not target.exists() else path.unlink(missing_ok=True)
            except (OSError, json.JSONDecodeError):
                continue
        self._prune_files(consumed, keep=self.CONSUMED_INPUT_RETENTION)
        return messages

    @staticmethod
    def _prune_files(directory: Path, *, keep: int) -> None:
        if keep < 0 or not directory.is_dir():
            return
        paths = sorted(path for path in directory.glob("*.json") if path.is_file())
        for path in paths[:-keep] if keep else paths:
            with suppress(OSError):
                path.unlink()

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema_version": CODEX_RUNTIME_SCHEMA_VERSION,
            "project_id": self.project_id,
            "version": 0,
            "status": "idle",
            "label": "Research session is ready",
            "turn_started_at": None,
            "increment": 1,
            "max_increments": 1,
            "thread_id": None,
            "turn_id": None,
            "partial_response": "",
            "live_narrative": "",
            "activities": [],
            "token_usage": {},
            "backend": {},
            "agent_profile": {},
            "error": "",
            "consecutive_timeouts": 0,
            "updated_at": _now(),
        }

    def _write_state(self, state: Mapping[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / f".state-{os.getpid()}-{time.time_ns()}.tmp"
        temporary.write_text(
            json.dumps(json_safe(state), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _replace_with_retry(temporary, self.state_path)


def ensure_codex_workspace(
    project_root: Path | str,
    project: ResearchProject,
    environment: ResearchEnvironment | None = None,
    *,
    research_query: str = "",
) -> Path:
    from .agent_tools import agent_tool_catalog
    from .experts import ExpertRegistry

    root = Path(project_root).resolve()
    project_directory = root / project.id
    workspace = project_directory / "workspace"
    (workspace / "artifacts").mkdir(parents=True, exist_ok=True)
    runtime = environment or ResearchEnvironment.load()
    registry = ExpertRegistry.from_environment(runtime, project_directory=project_directory)
    runtime.prepare_project_julia_environment(workspace)
    _write_managed_agents(
        project_directory / "AGENTS.md",
        _workspace_agents_md(project, root, runtime),
    )
    _write_generated_text(
        workspace / "PROJECT_CONTEXT.md",
        _project_context(project, project_directory, runtime),
    )
    _write_generated_text(
        workspace / "RESEARCH_ENVIRONMENT.json",
        runtime.render_manifest(project_root=root),
    )
    _write_generated_text(
        workspace / "TOOL_CATALOG.json",
        json.dumps(
            {
                "schema_version": "2.0",
                "project_id": project.id,
                "evidence_tools": agent_tool_catalog(project),
                "program_tools": registry.tool_catalog(),
                "public_api": "brilliance.accelerator_api",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )
    _write_generated_text(
        workspace / "ACCELERATOR_RESEARCH_HANDBOOK.md",
        runtime.handbook.read_text(encoding="utf-8"),
    )
    api_reference = runtime.accelerator_api.parent / "resources" / "accelerator_api_reference.md"
    _write_generated_text(
        workspace / "ACCELERATOR_API_REFERENCE.md",
        api_reference.read_text(encoding="utf-8"),
    )
    from .local_jutrack_help import render_reference
    try:
        native_reference = render_reference(runtime.pyjutrack_module)
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        native_reference = (
            "# Configured native JuTrack reference unavailable\n\n"
            f"Source parsing failed: {type(exc).__name__}. The application must repair "
            "its configured reference. Do not discover another runtime or guess schemas.\n"
        )
    _write_generated_text(workspace / "JUTRACK_API_REFERENCE.md", native_reference)
    query = "\n".join(item for item in (project.goal.strip(), research_query.strip()) if item)
    expert_assets = _stage_expert_reference_assets(registry, workspace, query=query)
    _write_generated_text(
        workspace / "EXPERT_ASSETS.json",
        json.dumps(expert_assets, ensure_ascii=False, indent=2, sort_keys=True),
    )
    _write_generated_text(
        workspace / "EXPERT_CATALOG.json",
        json.dumps(registry.catalog(goal=query), ensure_ascii=False, indent=2, sort_keys=True),
    )
    expert_context = registry.render_context(
        query,
        max_chars=runtime.expert_max_context_chars,
        max_chunks=10,
    )
    if expert_assets["assets"]:
        expert_context += _expert_asset_context(expert_assets)
    _write_generated_text(workspace / "EXPERT_CONTEXT.md", expert_context)
    return project_directory


def _stage_expert_reference_assets(
    registry: Any,
    workspace: Path,
    *,
    query: str,
) -> dict[str, Any]:
    """Copy relevant expert reference data into the isolated research workspace."""

    chunks = registry.retrieve(query, max_chars=16_000, max_chunks=10)
    selected_ids = {item.module_id for item in chunks}
    root = (workspace / "expert_assets").resolve()
    assets: list[dict[str, Any]] = []
    total_bytes = 0
    for module in registry.modules:
        if module.id not in selected_ids:
            continue
        safe_module = re.sub(r"[^A-Za-z0-9_.-]+", "-", module.id).strip("-.")
        for source in registry.reference_assets(module.id):
            relative = source.relative_to(module.root)
            size = source.stat().st_size
            if size > 4_000_000 or total_bytes + size > 20_000_000:
                continue
            destination = (root / safe_module / relative).resolve()
            destination.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}-{os.getpid()}-{time.time_ns()}.tmp"
            )
            temporary.write_bytes(source.read_bytes())
            _replace_with_retry(temporary, destination)
            workspace_relative = destination.relative_to(workspace).as_posix()
            assets.append({
                "module_id": module.id,
                "relative": relative.as_posix(),
                "workspace_path": workspace_relative,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "size": size,
            })
            total_bytes += size
    return {
        "schema_version": "1.0",
        "registry_fingerprint": registry.fingerprint,
        "assets": assets,
    }


def _expert_asset_context(manifest: Mapping[str, Any]) -> str:
    lines = [
        "\n## Installed reference assets",
        "",
        "Resolve these immutable workspace copies with "
        "`run.expert_asset(module_id, relative)`; do not search the repository.",
        "",
    ]
    for item in manifest.get("assets", []):
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"- `{item.get('module_id')}` / `{item.get('relative')}` "
            f"({item.get('size')} bytes)"
        )
    return "\n".join(lines).rstrip() + "\n"


def workspace_artifacts(
    project_root: Path | str,
    project_id: str,
    *,
    modified_after: float = 0.0,
) -> list[dict[str, Any]]:
    directory = Path(project_root).resolve() / project_id / "workspace" / "artifacts"
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.stat().st_mtime < modified_after:
            continue
        records.append(_workspace_artifact_record(path, directory, project_id))
    return records


def _workspace_artifact_record(path: Path, directory: Path, project_id: str) -> dict[str, Any]:
    relative = path.relative_to(directory).as_posix()
    return {
        "path": relative,
        "title": _workspace_artifact_title(relative),
        "media_type": _workspace_artifact_media_type(path),
        "size": path.stat().st_size,
        "content_sha256": _workspace_artifact_digest(path),
        "modified_at": path.stat().st_mtime,
        "url": (
            f"/api/projects/{quote(project_id, safe='')}/workspace-artifacts/"
            + quote(relative, safe="/")
        ),
    }


def resolve_workspace_artifact(
    project_root: Path | str, project_id: str, relative_path: str
) -> Path:
    root = (Path(project_root).resolve() / project_id / "workspace" / "artifacts").resolve()
    path = (root / str(relative_path)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LookupError("workspace artifact path escapes its project") from exc
    if not path.is_file():
        raise LookupError(f"workspace artifact not found: {relative_path}")
    return path


def workspace_research_updates(
    project_root: Path | str,
    project_id: str,
    *,
    limit: int | None = 12,
    modified_after: float = 0.0,
) -> list[dict[str, Any]]:
    directory = Path(project_root).resolve() / project_id / "workspace" / "progress"
    by_id: dict[str, dict[str, Any]] = {}
    paths = (
        sorted(path for path in directory.rglob("*.json") if path.is_file())
        if directory.is_dir()
        else ()
    )
    for path in paths:
        if path.stat().st_mtime < modified_after:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        event_id = str(value.get("id") or "research-update:" + content_hash(value, length=16))
        # Old optimizer callbacks used the generation in the event identity,
        # producing dozens of near-identical chat cards and pushing actual
        # findings outside the presentation window. Fold only these known
        # machine-generated counters; preserve arbitrary scientific milestones.
        legacy_campaign = re.fullmatch(
            r"(?:Evaluating generation \d+|Generation \d+ complete|Preparing feasible population|"
            r"Resuming optimization|Optimization): (.+)", str(value.get("stage", ""))
        )
        if legacy_campaign and str(value.get("kind", "progress")) == "progress":
            scope = str(value.get("compute_job_id") or (
                path.parent.name if path.parent.parent.name == "events" else path.stem
            ))
            event_id = "optimization-activity:" + content_hash(
                {"scope": scope, "campaign": legacy_campaign.group(1)}, length=20
            )
        update = {
            "id": event_id,
            "kind": str(value.get("kind", "finding")),
            "stage": str(value.get("stage") or value.get("title") or "Research progress"),
            "detail": str(value.get("detail", "")),
            "timestamp": str(value.get("timestamp") or value.get("updated_at") or ""),
            "completed": value.get("completed"),
            "total": value.get("total"),
            "metrics": value.get("metrics", {}),
            "tables": value.get("tables", []),
            "plots": value.get("plots", []),
            "compute_job_id": value.get("compute_job_id"),
            "run_id": path.parent.name if path.parent.parent.name == "events" else path.stem,
            **{key: value[key] for key in (
                "stage_id", "hypothesis_id", "candidate_id", "parent_candidate_id",
                "outcome", "scientific_role", "interpretation", "next_decision",
            ) if value.get(key) is not None},
        }
        previous = by_id.get(event_id)
        if previous is None or update["timestamp"] >= previous["timestamp"]:
            by_id[event_id] = update
    updates = sorted(by_id.values(), key=lambda item: (item["timestamp"], item["id"]))
    return updates if limit is None else updates[-max(1, int(limit)) :]


def run_codex_turn(
    project_root: Path | str,
    project: ResearchProject,
    message: str,
    *,
    initial: bool = False,
    increment: int = 1,
    max_increments: int | None = None,
    research_mode: str = "research",
) -> CodexTurnOutcome:
    return asyncio.run(
        _run_codex_turn(
            project_root,
            project,
            message,
            initial=initial,
            increment=max(1, increment),
            max_increments=max_increments or configured_max_increments(),
            research_mode=research_mode,
        )
    )


_PROGRAM_ACTION = re.compile(
    r"\b(design|build|construct|create|calculate|compute|evaluate|simulate|track|"
    r"optimi[sz]e|match|plot|insert|remove|modify|change|correct)\w*\b",
    re.IGNORECASE,
)
_PROGRAM_CONTINUATION = re.compile(
    r"^\s*(continue|resume|proceed|keep going|go on)(?:\s+(?:the )?(?:research|work|calculation|design))?[.!]?\s*$",
    re.IGNORECASE,
)
_NUMERICAL_EVIDENCE_REQUEST = re.compile(
    r"\b(current|show|give|list|report|value|values|plot|table|calculate|compute)\b.*"
    r"\b(lattice|elements?|optics?|emittance|chromaticity|tune|trajectory|phase[ -]?space|"
    r"dynamic aperture|momentum aperture|radiation|tracking|twiss|dispersion|beta)\b",
    re.IGNORECASE,
)


def _native_research_stage_requested(
    message: str, *, initial: bool, research_mode: str
) -> bool:
    """Keep every supported research chat in one coherent native agent."""

    del message, initial
    return research_mode in {"research", "result_review"}


def _native_execution_required(message: str, *, initial: bool) -> bool:
    """Require new evidence for calculations while allowing conceptual discussion."""

    return bool(
        _PROGRAM_ACTION.search(message)
        or _PROGRAM_CONTINUATION.match(message)
        or _NUMERICAL_EVIDENCE_REQUEST.search(message)
        or _requires_jutrack_execution(
            message, initial=initial, research_mode="research"
        )
    )


def _saved_project_state_context(project_directory: Path) -> str:
    """Return a bounded handoff for a model-authored follow-up program."""

    from .workspace_inspection import saved_lattice_context, saved_recovery_context

    lines: list[str] = [saved_recovery_context(project_directory / "workspace"),
                        saved_lattice_context(project_directory / "workspace")]
    pointer = project_directory / "workspace" / "WORKING_LATTICE.json"
    if pointer.is_file():
        lines.append(
            "A canonical working lattice is supplied to the program as "
            "run.working_lattice_file; load that exact file with api.load_lattice."
        )
        with suppress(OSError):
            lines.append(pointer.read_text(encoding="utf-8")[:2_000])
    results = sorted(
        (project_directory / "workspace" / "artifacts" / "programs").glob(
            "*/result.json"
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if results:
        with suppress(OSError):
            lines.extend(
                [
                    "Latest saved program result:",
                    results[0].read_text(encoding="utf-8")[:10_000],
                ]
            )
        for result_path in results:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                relative = str(result.get("candidate_lattice") or "").strip()
                candidate = (result_path.parent / relative).resolve()
                candidate.relative_to(result_path.parent.resolve())
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if relative and candidate.is_file():
                lines.append(
                    "The newest archived candidate, even if rejected by a physics gate, is "
                    "supplied to the next program as run.latest_candidate_lattice_file. Load it "
                    "with api.load_lattice when continuing that hypothesis."
                )
                break
    return "\n".join(lines)


def _latest_committed_program_receipt(project_directory: Path) -> dict[str, Any]:
    """Carry completion semantics across durable job boundaries, not just prose."""
    from .native_research_agent import _compact_execution

    paths = sorted((project_directory / "workspace" / "artifacts").glob("expert-execution-*.json"),
                   key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("tool_id") not in {"accelerator.run_research_program", "accelerator.run_long_research_program"}:
                continue
            validation = value.get("validation", {})
            if validation.get("passed") is not True:
                continue
            return _compact_execution({**value, "execution_valid": True,
                                       "scientific_passed": validation.get("scientific_passed")})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return {}


def _lookup_committed_program_receipt(project_directory: Path, run_id: str) -> dict[str, Any]:
    """Resolve exact older evidence, never a loose model-authored result JSON."""
    from .codex_tools import _compact_expert_execution
    from .native_research_agent import _compact_execution
    from .research_recovery import _committed_saved_program

    directory = project_directory.resolve()
    eligible: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    paths = sorted((directory / "workspace" / "artifacts").glob("expert-execution-*.json"),
                   key=lambda path: path.stat().st_mtime_ns, reverse=True)
    seen: set[str] = set()
    for path in paths:
        execution = _committed_saved_program(path, directory)
        if execution is None:
            continue
        try:
            program = execution["result"]["program"]
            snapshot = Path(str(program["snapshot"]))
            if not snapshot.is_absolute():
                snapshot = directory / snapshot
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != program["sha256"]:
                continue
            receipt = _compact_execution(_compact_expert_execution(execution))
            identity = str(receipt.get("run_id") or "")
            if (not identity or identity in seen or receipt.get("execution_valid") is not True
                    or receipt.get("scientific_passed") is not True or receipt.get("status") != "completed"):
                continue
            seen.add(identity)
            receipt["execution_artifact"] = str(path)
            if identity == run_id:
                selected = receipt
            if len(eligible) < 12:
                eligible.append({"run_id": identity, "summary": str(receipt.get("summary") or "")[:320],
                    "full_result": receipt.get("full_result"), "candidate_lattice": receipt.get("candidate_lattice")})
            if (not run_id or selected is not None) and len(eligible) >= 12:
                break
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            continue
    return {"receipt": selected, "eligible_runs": eligible,
            "error": "" if selected or not run_id else "No committed successful execution with a verified source matches that run_id in this project."}


async def _run_provider_native_stage(
    root: Path,
    project: ResearchProject,
    message: str,
    *,
    initial: bool,
    research_mode: str,
    environment: ResearchEnvironment,
    backend: CodexBackend,
    project_directory: Path,
    store: CodexRuntimeStore,
    started_at: float,
) -> CodexTurnOutcome | None:
    """Run the provider-native write/run/inspect loop for computational requests."""

    from .experts import ExpertRegistry
    from .native_research_agent import (
        NativeResearchAgentError,
        run_native_research_agent,
        supports_native_research_agent,
    )

    enabled = os.environ.get("BRILLIANCE_NATIVE_RESEARCH_AGENT", "true")
    if enabled.strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None
    if not _native_research_stage_requested(
        message, initial=initial, research_mode=research_mode
    ) or not supports_native_research_agent(backend):
        return None

    registry = ExpertRegistry.from_environment(
        environment, project_directory=project_directory
    )
    request = "\n".join(item for item in (project.goal.strip(), message.strip()) if item)
    expert_context = registry.render_context(
        request,
        max_chars=environment.expert_max_context_chars,
        max_chunks=6,
    )
    asset_manifest_path = project_directory / "workspace" / "EXPERT_ASSETS.json"
    if asset_manifest_path.is_file():
        with suppress(OSError, TypeError, json.JSONDecodeError):
            asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
            if isinstance(asset_manifest, Mapping) and asset_manifest.get("assets"):
                expert_context += _expert_asset_context(asset_manifest)
    api_reference = (
        environment.accelerator_api.parent / "resources" / "accelerator_api_reference.md"
    ).read_text(encoding="utf-8")
    state_context = _saved_project_state_context(project_directory)
    conversation = ResearchConversationStore(root)
    from .research_recovery import scientist_request
    from .research_session import SessionWorkspace

    session_workspace = SessionWorkspace(project_directory)
    completion_origin = scientist_request(root, project.id)
    history = conversation.messages(project.id)
    if history and history[-1].get("role") == "user" and history[-1].get("content") == message:
        history = history[:-1]
    conversation.append_message(
        project.id,
        "assistant",
        f"Model: **{backend.model}**. I am reviewing your request and the saved research evidence.",
        meta={"runtime": "native_research_agent", "kind": "plan"},
    )
    store.update(
        status="running",
        label="Preparing the next executable calculation",
        live_narrative="The configured model is writing a short saved research program.",
    )

    async def publish(event: Mapping[str, Any]) -> None:
        nonlocal completion_origin
        kind = str(event.get("kind", ""))
        if kind == "model_started":
            store.update(label=f"{backend.model} is preparing the next response", live_narrative="", partial_response="")
        elif kind == "assistant_text":
            store.update(partial_response=str(event.get("text", "")), live_narrative=str(event.get("text", ""))[-12000:])
        elif kind == "input_received":
            completion_origin = scientist_request(root, project.id)
            store.update(label="Incorporating your latest message", live_narrative="The new request has reached the active model loop.")
        elif kind in {"tool_started", "tool_finished"}:
            store.append_activity({
                "id": event.get("id"), "kind": "research_tool",
                "title": str(event.get("name", "tool")),
                "status": "running" if kind == "tool_started" else "completed" if event.get("ok") else "failed",
                "detail": str(event.get("error", "")), "summary": str(event.get("error", "")),
            })
        elif kind == "user_update":
            update_kind = str(event.get("update_kind", "progress"))
            conversation.append_message(
                project.id,
                "assistant",
                str(event.get("message", "")),
                meta={"runtime": "native_research_agent", "kind": update_kind},
            )
            store.update(
                label=str(event.get("message", ""))[:120] or "Research update",
                live_narrative=str(event.get("message", "")),
            )
        elif kind in {"program_written", "program_patched"}:
            passed = event.get("passed") is True
            store.update(
                research_program_authored_at=_now(),
                label=(
                    "Saved program passed preflight"
                    if passed
                    else "Correcting saved program preflight"
                ),
                live_narrative=(
                    f"{event.get('program')} — {event.get('lines', '?')} lines"
                    if passed
                    else "; ".join(str(item) for item in event.get("issues", []))[:1000]
                ),
            )
        elif kind == "run_started":
            text = (
                f"**Execution {event.get('attempt')}** — "
                + (
                    "submitted as a durable JuTrack calculation: "
                    if event.get("background")
                    else "running with JuTrack: "
                )
                + f"`{event.get('program')}`."
            )
            if event.get("background"):
                conversation.append_message(
                    project.id,
                    "assistant",
                    text,
                    meta={"runtime": "native_research_agent", "kind": "progress"},
                )
            store.update(
                research_program_started_at=_now(),
                label="JuTrack calculation is running",
                live_narrative=text,
            )
        elif kind == "run_failed":
            error = str(event.get("error", ""))
            store.update(label="Revising from the execution error", live_narrative=error[:1000])
        elif kind == "run_completed":
            receipt = event.get("receipt", {})
            receipt = receipt if isinstance(receipt, Mapping) else {}
            if receipt.get("kind") == "managed_calculation":
                label = "Long calculation is running independently"
            else:
                label = "Interpreting the JuTrack result"
            store.update(label=label, live_narrative=str(receipt.get("summary", ""))[:1000])

    try:
        result = await run_native_research_agent(
            backend,
            project_root=root,
            project_id=project.id,
            workspace=project_directory / "workspace",
            python_executable=environment.python_executable,
            process_environment=environment.process_environment(),
            turn_wall_time_s=environment.turn_wall_time_s,
            project_goal=project.goal,
            scientist_message=message,
            expert_context=expert_context,
            project_state_context=state_context,
            api_reference=api_reference,
            knowledge_lookup=registry.read_knowledge,
            expert_registry=registry,
            prior_receipt=(_latest_committed_program_receipt(project_directory)
                           if research_mode == "result_review" else None),
            receipt_lookup=lambda run_id: _lookup_committed_program_receipt(project_directory, run_id),
            require_execution=(research_mode != "result_review" and _native_execution_required(message, initial=initial)),
            on_event=publish,
            conversation_history=history,
            consume_input=store.consume_pending,
            pause_requested=session_workspace.pause_file.exists,
        )
    except (NativeResearchAgentError, OSError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        response = (
            "The iterative research loop stopped before a defensible result was ready. "
            f"Exact boundary: `{error}`. Saved programs and every JuTrack artifact were "
            "preserved; no numerical claim was accepted."
        )
        attachments = _turn_attachments(root, project.id, started_at)
        conversation.append_message(project.id, "assistant", response, attachments=attachments)
        store.update(status="paused", label="Research loop needs attention", error=error)
        return CodexTurnOutcome(
            status="paused",
            detail=response,
            thread_id=None,
            turn_id=None,
            final_response=response,
            disposition="needs_input",
        )

    attachments = _turn_attachments(
        root, project.id, started_at, completion_evidence=result.completion_evidence,
    )
    saved_result = _presentable_result_narrative(attachments)
    response = result.response.strip() or saved_result
    if not response:
        response = (
            "The calculation did not publish a scientist-facing result. No additional "
            "numerical claim was accepted."
        )
    response = _with_artifact_notice(response, attachments)
    completion_evidence = (
        {**dict(result.completion_evidence), "scientist_request": completion_origin,
         "completed_at": _now()} if result.completion_evidence else None)
    protocol_failure = dict(result.protocol_failure) if result.protocol_failure else None
    conversation.append_message(project.id, "assistant", response, attachments=attachments,
        meta={"runtime": "native_research_agent", "kind": "result",
              "request_satisfied": result.request_satisfied,
              "scientist_request": completion_origin,
              "completion_evidence": completion_evidence,
              "protocol_failure": ({key: protocol_failure[key] for key in
                  ("type", "reason", "attempts", "can_retry") if key in protocol_failure}
                  if protocol_failure else None)})
    recovery = None
    if result.failed_program and result.last_error:
        failure = project_directory / "workspace" / "runtime_failures" / (
            "job-native-" + content_hash((started_at, result.failed_program, result.last_error)) + ".json")
        recovery = {"script": result.failed_program, "error": result.last_error,
                    "diagnostic": str(failure), "mode": "diagnose_and_repair"}
        _write_generated_text(failure, json.dumps({"kind": "native_step_boundary", **recovery}, ensure_ascii=False))
    status = ("complete" if result.request_satisfied or result.managed_job_id
              else "checkpoint" if result.continuation_needed or recovery else "paused")
    store.update(
        status=status,
        completion_evidence=completion_evidence,
        protocol_failure=protocol_failure,
        label=(
            "Long calculation is running independently"
            if result.managed_job_id
            else "JuTrack result is ready"
            if result.request_satisfied
            else "Research context checkpoint saved"
            if result.continuation_needed
            else "Model response needs attention; research is unfinished"
            if protocol_failure
            else "Validated checkpoint saved; research is unfinished"
        ),
        live_narrative="",
        error=(
            str(protocol_failure["message"])
            if protocol_failure
            else result.last_error
            if status == "paused" and result.last_error
            else "The current scientist request is not yet satisfied."
            if status == "paused"
            else ""
        ),
        consecutive_timeouts=0,
    )
    from .projects import ProjectStore

    ProjectStore(root).snapshot_manifest(project.id)
    completed_at = _now()
    return CodexTurnOutcome(
        status=status,
        detail=(
            f"Native research loop completed {result.completed_executions} execution(s) "
            f"in {result.execution_attempts} attempt(s)."
        ),
        thread_id=None,
        turn_id=f"native-agent-{content_hash((project.id, completed_at), length=16)}",
        final_response=response,
        disposition="complete" if completion_evidence else "native_repair" if recovery else "native_continue" if result.continuation_needed else "needs_input",
        recovery=recovery,
        completion_evidence=completion_evidence,
    )


async def _run_codex_turn(
    project_root: Path | str,
    project: ResearchProject,
    message: str,
    *,
    initial: bool,
    increment: int,
    max_increments: int,
    research_mode: str,
) -> CodexTurnOutcome:
    from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
    from openai_codex.types import ReasoningEffort

    root = Path(project_root).resolve()
    store = CodexRuntimeStore(root, project.id)
    started_at = time.time()
    store.start_turn(
        label="Preparing the configured research model",
        increment=increment,
        max_increments=max_increments,
    )
    try:
        environment = ResearchEnvironment.load()
        environment.require_active_python(operation="Codex research worker")
        project_directory = ensure_codex_workspace(
            root, project, environment, research_query=message
        )
        backend = resolve_codex_backend()
    except (ResearchEnvironmentError, CodexBackendError, OSError, ValueError) as exc:
        return _pause_turn(root, project, str(exc), exc)

    previous = store.load()
    previous_backend = str(previous.get("backend", {}).get("fingerprint", ""))
    previous_profile = str(previous.get("agent_profile", {}).get("fingerprint", ""))
    profile = _agent_profile(environment, project_directory=project_directory)
    if previous_profile != profile["fingerprint"]:
        _archive_stale_handoff(project_directory, previous_profile_fingerprint=previous_profile)
    store.update(backend=backend.public(), agent_profile=profile)

    authored_outcome = await _run_provider_native_stage(
        root,
        project,
        message,
        initial=initial,
        research_mode=research_mode,
        environment=environment,
        backend=backend,
        project_directory=project_directory,
        store=store,
        started_at=started_at,
    )
    if authored_outcome is not None:
        return authored_outcome

    default_effort = "high" if backend.supports_reasoning_effort else "low"
    effort_name = os.environ.get("BRILLIANCE_CODEX_EFFORT", default_effort).strip().lower()
    try:
        effort = ReasoningEffort(effort_name)
    except ValueError:
        effort = ReasoningEffort.high
    # OpenAI-backed turns can use Codex automatic review.  Compatible third-party
    # gateways do not necessarily host the reviewer model, so keep their ordinary
    # command policy non-interactive and pre-approve only Brilliance's confined
    # workspace writer in its MCP configuration below.
    approval = ApprovalMode.auto_review if backend.key == "openai" else ApprovalMode.deny_all
    sandbox = Sandbox.full_access if environment.codex_sandbox == "full-access" else Sandbox.workspace_write
    aliases = environment.prepare_sandbox_runtime_aliases(project_directory / "workspace")
    timeout = configured_turn_timeout(environment)
    process_environment = environment.process_environment()
    from .experts import ExpertRegistry
    from .scientific_acceptance import ACCEPTANCE_ENVIRONMENT, continued_request_acceptance

    process_environment[ACCEPTANCE_ENVIRONMENT] = json.dumps(list(continued_request_acceptance(
        message, ResearchConversationStore(root).messages(project.id),
        registry=ExpertRegistry.from_environment(environment, project_directory=project_directory))))
    matplotlib = project_directory / "workspace" / "runtime" / "matplotlib"
    matplotlib.mkdir(parents=True, exist_ok=True)
    julia_executable = aliases.get("julia_executable", environment.julia_executable)
    process_environment.update(
        {
            "BRILLIANCE_PROJECT_ROOT": str(root),
            "BRILLIANCE_PROJECT_ID": project.id,
            "JULIA_DEPOT_PATH": str(environment.julia_depot),
            "JUTRACK_ROOT": str(aliases["jutrack_root"]),
            "JUTRACK_PYTHON": str(aliases["pyjutrack_module"]),
            "JUTRACK_JL_PROJECT": str(aliases["jutrack_root"]),
            "JULIA_PROJECT": str(aliases["jutrack_root"]),
            "PYTHON_JULIACALL_EXE": str(julia_executable),
            "PYTHON_JULIACALL_BINDIR": str(Path(julia_executable).parent),
            "MPLCONFIGDIR": str(matplotlib),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "BRILLIANCE_MODEL_TURN_DEADLINE_EPOCH": str(
                started_at + timeout - TURN_SHUTDOWN_RESERVE_SECONDS
            ),
        }
    )
    if environment.codex_sandbox == "workspace-write":
        process_environment["PYTHON_JULIAPKG_PROJECT"] = str(aliases["python_julia_environment"])
        process_environment["PYTHON_JULIACALL_PROJECT"] = str(aliases["python_julia_environment"])

    config = CodexConfig(
        config_overrides=(
            *backend.config_overrides,
            *_codex_research_config_overrides(environment, str(environment.julia_depot)),
            *_codex_workspace_mcp_overrides(
                environment, project_root=root, project_id=project.id
            ),
        ),
        cwd=str(project_directory),
        client_name="brilliance",
        client_title="Brilliance Research",
        env=process_environment,
    )
    codex = AsyncCodex(config)
    closed = False

    async def close_codex() -> None:
        nonlocal closed
        if not closed:
            closed = True
            await codex.close()

    thread = None
    handle = None
    try:
        await codex.__aenter__()
        thread_id = _compatible_thread_id(
            previous,
            previous_backend_fingerprint=previous_backend,
            current_backend_fingerprint=backend.fingerprint,
            previous_profile_fingerprint=previous_profile,
            current_profile_fingerprint=profile["fingerprint"],
        )
        instructions = _developer_instructions(
            project, root, environment, research_mode=research_mode, query=message
        )
        thread_kwargs = {
            "approval_mode": approval,
            "cwd": str(project_directory),
            "developer_instructions": instructions,
            "sandbox": sandbox,
        }
        if thread_id and not _context_rollover_required(previous):
            try:
                thread = await codex.thread_resume(thread_id, **thread_kwargs)
            except Exception:
                thread = await codex.thread_start(**thread_kwargs)
        else:
            thread = await codex.thread_start(**thread_kwargs)
        if not thread_id:
            with suppress(Exception):
                await thread.set_name(project.title[:120])
        prompt = _turn_prompt(project, message, initial=initial, research_mode=research_mode)
        handle = await thread.turn(prompt, effort=effort)
        store.update(status="running", thread_id=thread.id, turn_id=handle.id)
        completed, streamed, control = await _consume_turn_stream(
            handle,
            store,
            timeout_seconds=max(1.0, timeout - TURN_SHUTDOWN_RESERVE_SECONDS),
            close_codex=close_codex,
        )
        if control is not None:
            if control.get("kind") == "managed_expert_calculation":
                response = _managed_job_checkpoint_message(control)
                disposition = "complete"
            else:
                response = _operational_abort_message(control)
                disposition = "needs_input"
            attachments = _turn_attachments(root, project.id, started_at)
            ResearchConversationStore(root).append_message(
                project.id, "assistant", response, attachments=attachments
            )
            store.update(
                status="complete",
                partial_response="",
                live_narrative="",
                error="" if control.get("kind") == "managed_expert_calculation" else control.get("detail", ""),
            )
            from .projects import ProjectStore

            ProjectStore(root).snapshot_manifest(project.id)
            return CodexTurnOutcome(
                status="complete" if disposition == "complete" else "paused",
                detail=response,
                thread_id=thread.id,
                turn_id=getattr(handle, "id", None),
                final_response=response,
                disposition=disposition,
            )
        if completed is None:
            raise RuntimeError("Codex stream ended without a completed turn")
        response = _final_response(completed) or "".join(streamed).strip()
        response, disposition = _response_disposition(response)
        if (
            _requires_jutrack_execution(
                message, initial=initial, research_mode=research_mode
            )
            and not store.load().get("research_program_started_at")
        ):
            response = (
                "This design turn ended before any saved JuTrack program was executed. "
                "No lattice, optics, or emittance claim was accepted. The controller treats "
                "this as an incomplete model turn, not as a design result."
            )
            disposition = "needs_input"
        if disposition == "continue" and not configured_auto_continue():
            disposition = "needs_input"
            response = (
                response.rstrip()
                + "\n\nI paused at this scientific checkpoint so you can inspect the result or steer "
                "the next calculation."
            )
        attachments = _turn_attachments(root, project.id, started_at)
        if not response.strip():
            response = _presentable_result_narrative(attachments) or (
                "The model turn ended without a user-facing scientific explanation. No result "
                "has been accepted; inspect the live checkpoint before continuing."
            )
        response = _with_artifact_notice(response, attachments)
        ResearchConversationStore(root).append_message(
            project.id, "assistant", response, attachments=attachments
        )
        store.update(
            status="complete",
            thread_id=thread.id,
            turn_id=getattr(handle, "id", None),
            partial_response="",
            live_narrative="",
            error="",
            consecutive_timeouts=0,
            completed_turn=_compact_turn_metadata(completed),
        )
        from .projects import ProjectStore

        ProjectStore(root).snapshot_manifest(project.id)
        return CodexTurnOutcome(
            status="complete",
            detail="Codex research increment completed.",
            thread_id=thread.id,
            turn_id=getattr(handle, "id", None),
            final_response=response,
            disposition=disposition,
        )
    except TimeoutError as exc:
        return _checkpoint_turn(
            root,
            project,
            timeout_seconds=timeout,
            increment=increment,
            max_increments=max_increments,
            backend=backend.public(),
            started_at=started_at,
            exc=exc,
            thread=thread,
            handle=handle,
        )
    except Exception as exc:
        return _pause_turn(root, project, f"{type(exc).__name__}: {exc}", exc, thread=thread, handle=handle)
    finally:
        with suppress(Exception):
            await close_codex()


async def _consume_turn_stream(
    handle: Any,
    store: CodexRuntimeStore,
    *,
    timeout_seconds: float,
    close_codex: Any,
) -> tuple[Mapping[str, Any] | None, list[str], Mapping[str, Any] | None]:
    completed: Mapping[str, Any] | None = None
    control: Mapping[str, Any] | None = None
    streamed: list[str] = []
    stream_state: dict[str, Any] = {
        "narrative_buffer": "",
        "program_failures": 0,
        "source_transport_failures": 0,
        "last_visible_update": time.monotonic(),
        "heartbeat_published": False,
        "pre_execution_commentary_published": False,
    }

    async def consume() -> None:
        nonlocal completed, control
        async for notification in handle.stream():
            terminal = _record_notification(
                store, notification, streamed_response=streamed, stream_state=stream_state
            )
            if terminal is not None:
                completed = terminal
            repair = stream_state.pop("repair_directive", None)
            if isinstance(repair, Mapping):
                message = str(repair.get("message", "")).strip()
                if message:
                    ResearchConversationStore(store.project_root).append_message(
                        store.project_id,
                        "assistant",
                        message,
                        meta={"runtime": "research_controller", "kind": "repair"},
                    )
                stream_state["last_visible_update"] = time.monotonic()
                stream_state["heartbeat_published"] = False
                steer_call = getattr(handle, "steer", None)
                if callable(steer_call):
                    with suppress(Exception):
                        await steer_call(str(repair.get("instruction", "")))
            receipt = stream_state.get("managed_job_receipt")
            abort = stream_state.get("operational_abort")
            if isinstance(receipt, Mapping) or isinstance(abort, Mapping):
                control = dict(receipt if isinstance(receipt, Mapping) else abort)
                store.update(
                    label=(
                        "Managed calculation is running independently"
                        if isinstance(receipt, Mapping)
                        else "Stopped a wasteful retry loop"
                    ),
                    live_narrative=(
                        "The durable worker owns this calculation; the model is no longer polling it."
                        if isinstance(receipt, Mapping)
                        else "The operational error budget was exhausted; files were preserved."
                    ),
                )
                interrupt = getattr(handle, "interrupt", None)
                if callable(interrupt):
                    with suppress(Exception):
                        await interrupt()
                break

    async def heartbeat() -> None:
        interval = float(configured_chat_update_interval())
        progress_directory = store.project_root / store.project_id / "workspace" / "progress"
        latest_progress_mtime = max(
            (path.stat().st_mtime_ns for path in progress_directory.rglob("*.json")),
            default=0,
        ) if progress_directory.is_dir() else 0
        while True:
            await asyncio.sleep(min(10.0, max(0.05, interval / 4.0)))
            current_state = store.load()
            if current_state.get("status") not in {"starting", "running"}:
                return
            store.update(heartbeat_at=_now())
            current_progress_mtime = max(
                (path.stat().st_mtime_ns for path in progress_directory.rglob("*.json")),
                default=0,
            ) if progress_directory.is_dir() else 0
            if current_progress_mtime > latest_progress_mtime:
                latest_progress_mtime = current_progress_mtime
                stream_state["last_visible_update"] = time.monotonic()
                stream_state["heartbeat_published"] = False
                continue
            now = time.monotonic()
            if now - float(stream_state.get("last_visible_update", now)) < interval:
                continue
            if stream_state.get("heartbeat_published"):
                continue
            progress = research_progress(current_state)
            title = str(progress.get("title") or "Research is still running")
            elapsed = int(progress.get("elapsed_seconds") or 0)
            if current_state.get("research_program_started_at"):
                detail = (
                    "The saved numerical program is still running. It has not published a new "
                    "numerical milestone during the last update interval."
                )
            elif current_state.get("research_program_authored_at"):
                detail = (
                    "The research program has been saved and is being checked before its first "
                    "JuTrack execution. No numerical result exists yet."
                )
            else:
                detail = (
                    "No JuTrack research program has started yet; the model is still preparing "
                    "the executable method."
                )
            errors = progress.get("latest_errors", [])
            if isinstance(errors, list) and errors:
                latest_error = errors[-1]
                if isinstance(latest_error, Mapping) and latest_error.get("summary"):
                    detail += " Latest issue: " + str(latest_error["summary"])
            ResearchConversationStore(store.project_root).append_message(
                store.project_id,
                "assistant",
                f"**Still working — {title}**\n\n{detail} Elapsed time: {elapsed} s.",
                meta={
                    "runtime": "research_controller",
                    "kind": "heartbeat",
                    "turn_id": current_state.get("turn_id"),
                },
            )
            stream_state["last_visible_update"] = now
            stream_state["heartbeat_published"] = True

    async def steer() -> None:
        steer_call = getattr(handle, "steer", None)
        if not callable(steer_call):
            return
        while True:
            await asyncio.sleep(0.25)
            messages = store.consume_pending()
            for index, message in enumerate(messages):
                try:
                    await steer_call(
                        "The scientist is steering this active research turn. Incorporate this "
                        "direction at the next safe boundary and acknowledge the change:\n\n"
                        + message
                    )
                except Exception:
                    for pending in messages[index:]:
                        store.queue_input(pending)
                    return
                store.append_activity(
                    {
                        "id": "scientist-steer-" + content_hash(message, length=12),
                        "kind": "decision",
                        "title": "Scientist changed the active research direction",
                        "status": "completed",
                        "summary": message[:500],
                    }
                )

    async def execution_watchdog() -> None:
        """Redirect model-only wandering while leaving submitted calculations alone."""

        nonlocal control

        steer_call = getattr(handle, "steer", None)
        # Tiny unit-test and interactive turns do not need a research watchdog.
        if not callable(steer_call) or timeout_seconds < 120:
            return
        schedule = (
            (
                float(configured_first_execution_nudge()),
                "Execution milestone: stop expanding the plan and do not reread documentation or "
                "inspect another project. Save and execute the smallest complete implementation "
                "of the already-selected expert route. 'Smallest' never permits dropping a "
                "required seed hypothesis, comparison, gate, or diagnostic from that route. It "
                "may publish a feasible seed or valid negative evidence; it does not need to "
                "solve the entire research problem in one run.",
                "Refocusing on the first executable scientific result",
            ),
            (
                float(configured_second_execution_nudge()),
                "No Brilliance research program has been executed yet. Stop adding architecture "
                "and avoid copying historical code. Complete and run the current expert route's "
                "first valid milestone now, including run.finish with a clear result or failure "
                "diagnosis. Preserve every mandatory route comparison and physics gate; defer "
                "only work that the selected method explicitly labels optional.",
                "Requiring a runnable result before more model planning",
            ),
        )
        elapsed = 0.0
        for due, message, title in schedule:
            wait_for = due - elapsed
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            elapsed = due
            if stream_state.get("research_program_started"):
                return
            if store.load().get("status") not in {"starting", "running"}:
                return
            try:
                await steer_call(message)
            except Exception:
                return
            store.append_activity(
                {
                    "id": "execution-watchdog-" + content_hash((due, message), length=12),
                    "kind": "decision",
                    "title": title,
                    "status": "completed",
                    "summary": message,
                }
            )
            store.update(
                label=title,
                live_narrative=(
                    "The research controller redirected model-only work toward a saved, "
                    "executable numerical milestone."
                ),
            )
        deadline = float(configured_max_pre_execution())
        remaining = deadline - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)
        if stream_state.get("research_program_started"):
            return
        if store.load().get("status") not in {"starting", "running"}:
            return
        record = {
            "kind": "pre_execution_stall",
            "fatal": True,
            "repairable": False,
            "detail": (
                f"The model used {int(deadline)} seconds without starting a saved JuTrack "
                "research program. The controller stopped model-only preparation; no numerical "
                "claim or lattice state changed."
            ),
        }
        control = record
        stream_state["operational_abort"] = record
        store.append_activity(
            {
                "id": "pre-execution-stall-" + content_hash(deadline, length=12),
                "kind": "error",
                "title": "Stopped model-only preparation before it consumed the full turn",
                "status": "failed",
                "summary": record["detail"],
            }
        )
        interrupt = getattr(handle, "interrupt", None)
        if callable(interrupt):
            with suppress(Exception):
                await interrupt()
        # Some third-party providers do not acknowledge an interrupt until a
        # long reasoning response ends. Before a program starts there is no
        # JuTrack child to preserve, so close only the model transport instead
        # of allowing hidden preparation to consume the whole turn budget.
        await asyncio.sleep(2.0)
        if not stream_state.get("research_program_started"):
            with suppress(Exception):
                await close_codex()

    stream_task = asyncio.create_task(consume())
    heartbeat_task = asyncio.create_task(heartbeat())
    steering_task = asyncio.create_task(steer())
    watchdog_task = asyncio.create_task(execution_watchdog())
    try:
        done, _ = await asyncio.wait({stream_task}, timeout=timeout_seconds)
        if stream_task not in done:
            interrupt = getattr(handle, "interrupt", None)
            if callable(interrupt):
                with suppress(Exception):
                    await asyncio.wait_for(interrupt(), timeout=2.0)
            await close_codex()
            raise TimeoutError("Codex research turn reached its configured time slice")
        await stream_task
        return completed, streamed, control
    finally:
        tasks = (stream_task, heartbeat_task, steering_task, watchdog_task)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _record_notification(
    store: CodexRuntimeStore,
    notification: Any,
    *,
    streamed_response: list[str] | None = None,
    stream_state: dict[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    method = str(getattr(notification, "method", ""))
    payload = _model_dict(getattr(notification, "payload", {}))
    # Token/reasoning/message deltas can arrive hundreds of times per second.
    # Their useful state is already compacted below; persisting every token
    # creates a misleading flood of tiny files and can evict the actual failed
    # command from the bounded event history.
    if method not in {
        "item/reasoning/textDelta",
        "item/agentMessage/delta",
        "turn/tokenUsage/updated",
        "thread/tokenUsage/updated",
    }:
        store.append_event({"method": method, "payload": payload})
    state = stream_state if stream_state is not None else {}
    if method == "item/agentMessage/delta":
        delta = str(payload.get("delta", ""))
        if delta and streamed_response is not None:
            streamed_response.append(delta)
            item_id = str(payload.get("itemId", ""))
            if item_id != str(state.get("agent_message_id", "")):
                state["agent_message_id"] = item_id
                state["agent_message_buffer"] = ""
            state["agent_message_buffer"] = (
                str(state.get("agent_message_buffer", "")) + delta
            )[-2_000:]
            narrative = RESEARCH_STATUS_PATTERN.sub(
                "", str(state["agent_message_buffer"])
            ).strip()
            now = time.monotonic()
            if now - float(state.get("last_delta_write", 0.0)) >= 0.25:
                store.update(partial_response=narrative[-8_000:], live_narrative=narrative[-1_000:])
                state["last_delta_write"] = now
        return None
    if method in {"item/started", "item/updated", "item/completed"}:
        item = payload.get("item", payload)
        item = item if isinstance(item, Mapping) else {}
        if _is_research_program_execution(item) and not state.get("research_program_started"):
            state["research_program_started"] = True
            store.update(
                research_program_started_at=_now(),
                research_program_command=_short_command(str(item.get("command", ""))),
            )
        status = "completed" if method == "item/completed" else "running"
        activity = _activity_from_item(item, status=status)
        if activity is not None:
            if method == "item/completed" and str(item.get("type", "")) == "commandExecution":
                output = _command_output(item)
                if _completed_command_succeeded(item):
                    activity["summary"] = _result_summary(output)
                else:
                    activity["status"] = "failed"
                    activity["summary"] = _failure_summary(output)
            elif method == "item/completed" and str(item.get("type", "")) in {
                "mcpToolCall",
                "dynamicToolCall",
            } and not _completed_command_succeeded(item):
                activity["status"] = "failed"
                activity["summary"] = _mcp_failure_summary(item)
            store.append_activity(activity)
        if method == "item/completed":
            if (
                str(item.get("type", "")) in {"mcpToolCall", "dynamicToolCall"}
                and _completed_command_succeeded(item)
                and str(item.get("tool", item.get("name", ""))).casefold()
                == "write_research_program"
            ):
                state["research_program_authored"] = True
                store.update(research_program_authored_at=_now())
            if str(item.get("type", "")) == "agentMessage":
                text = RESEARCH_STATUS_PATTERN.sub("", str(item.get("text", ""))).strip()
                if text:
                    phase = str(item.get("phase", "")).strip().lower()
                    if phase == "commentary":
                        publish = _publish_agent_commentary(text, state)
                        if publish:
                            ResearchConversationStore(store.project_root).append_message(
                                store.project_id,
                                "assistant",
                                text,
                                meta={
                                    "runtime": "codex",
                                    "kind": "commentary",
                                    "stream_item_id": item.get("id"),
                                },
                            )
                            state["last_visible_update"] = time.monotonic()
                            state["heartbeat_published"] = False
                        store.update(partial_response="", live_narrative=text[-1_000:])
                    else:
                        store.update(
                            partial_response=text[-8_000:],
                            live_narrative=text[-1_000:],
                        )
            receipt = _managed_job_receipt_from_item(item)
            if receipt is not None:
                state["managed_job_receipt"] = receipt
            failure = _command_operational_failure(item) or _mcp_operational_failure(item)
            if failure is not None:
                if failure.get("kind") == "source_transport_failure":
                    attempts = int(state.get("source_transport_failures", 0)) + 1
                    state["source_transport_failures"] = attempts
                    if attempts == 1:
                        state["repair_directive"] = {
                            "message": (
                                "**Repair — program source transport**\n\nWindows rejected an "
                                "oversized shell command before the research program was created. "
                                "I redirected program and JSON creation to Brilliance's atomic "
                                "local workspace tool; JuTrack has not been run yet."
                            ),
                            "instruction": (
                                "The source-writing command failed at the Windows command-line "
                                "boundary. Do not retry, shorten, encode, or chunk that shell "
                                "command, and do not reread documentation. Split the route into "
                                "one <=250-line scientific stage now. Use native apply_patch or "
                                "the local Brilliance workspace writer for the program and run "
                                "JSON, run check-program, then execute that stage."
                            ),
                        }
                        failure = None
                    else:
                        failure = {**failure, "fatal": True, "repairable": False}
                elif failure.get("kind") == "workspace_writer_rejected":
                    attempts = int(state.get("workspace_writer_failures", 0)) + 1
                    state["workspace_writer_failures"] = attempts
                    if attempts == 1:
                        state["repair_directive"] = {
                            "message": (
                                "**Repair — local program writer**\n\nThe local writer was "
                                "rejected by the non-interactive approval policy; this was an "
                                "internal integration failure, not a user rejection and not a "
                                "physics result. I redirected source creation to Codex's native "
                                "workspace patch operation."
                            ),
                            "instruction": (
                                "The local MCP writer was rejected by the runtime approval layer. "
                                "Do not repeat that rejected transport operation. Use the "
                                "native apply_patch file-edit operation to create the already "
                                "drafted program and run JSON under workspace/programs, then run "
                                "check-program and execute the stage."
                            ),
                        }
                        failure = None
                    else:
                        failure = {**failure, "fatal": True, "repairable": False}
                elif failure.get("repairable"):
                    state["program_failures"] = int(state.get("program_failures", 0)) + 1
                    if int(state["program_failures"]) > 2:
                        failure = {**failure, "fatal": True, "repairable": False}
                    else:
                        failure = None
                if failure is not None and failure.get("fatal"):
                    state["operational_abort"] = failure
        return None
    if method in {"turn/tokenUsage/updated", "thread/tokenUsage/updated"}:
        store.update(token_usage=payload)
        return None
    if method == "turn/completed":
        return payload
    if method in {"turn/started", "thread/started"}:
        store.update(status="running")
    return None


def _publish_agent_commentary(text: str, state: Mapping[str, Any]) -> bool:
    """Keep the chat scientific instead of logging every coding thought.

    Before JuTrack starts, one action/expectation message is useful.  Repeated
    "let me inspect another signature" messages belong in the activity stream,
    not in the scientist conversation.  Failures and decisions always pass.
    """

    normalized = " ".join(str(text).split()).casefold()
    urgent = any(
        token in normalized
        for token in (
            " failed",
            " failure",
            " error",
            " rejected",
            " blocked",
            " repair",
        )
    )
    if urgent:
        return True
    if not state.get("research_program_started"):
        if state.get("pre_execution_commentary_published"):
            return False
        if isinstance(state, dict):
            state["pre_execution_commentary_published"] = True
        return True

    # Once execution starts, run.progress owns routine status. Preserve only a
    # scientific finding/decision from model commentary; suppress coding and
    # polling narration that otherwise floods the chat beside real evidence.
    material = any(
        token in normalized
        for token in (
            " result",
            " evidence",
            " found",
            " measured",
            " achieved",
            " passed",
            " selected",
            " rejected",
            " comparison",
            " limitation",
        )
    )
    if not material:
        return False
    fingerprint = content_hash(normalized, length=16)
    if fingerprint == state.get("last_published_commentary"):
        return False
    if isinstance(state, dict):
        state["last_published_commentary"] = fingerprint
    return True


def _is_research_program_execution(item: Mapping[str, Any]) -> bool:
    """Return true only when a registered saved-program runner was invoked."""

    kind = str(item.get("type", ""))
    if kind == "commandExecution":
        command = str(item.get("command", "")).lower()
        return any(
            tool_id in command
            for tool_id in (
                "accelerator.run_research_program",
                "accelerator.run_long_research_program",
            )
        )
    if kind in {"mcpToolCall", "dynamicToolCall"}:
        tool = str(item.get("tool", item.get("name", ""))).lower()
        return tool in {
            "accelerator.run_research_program",
            "accelerator.run_long_research_program",
        }
    return False


def _activity_from_item(item: Mapping[str, Any], *, status: str) -> dict[str, Any] | None:
    kind = str(item.get("type", ""))
    item_id = str(item.get("id") or content_hash(item, length=12))
    if kind == "commandExecution":
        command = str(item.get("command", ""))
        return {
            "id": item_id,
            "kind": "calculation",
            "title": _command_activity_title(command, output=_command_output(item)),
            "status": status,
            "detail": _short_command(command),
        }
    if kind in {"mcpToolCall", "dynamicToolCall"}:
        return {
            "id": item_id,
            "kind": "tool",
            "title": "Using " + str(item.get("tool", item.get("name", "a research tool"))),
            "status": status,
        }
    if kind == "fileChange":
        return {"id": item_id, "kind": "code", "title": "Saving research code", "status": status}
    if kind == "webSearch":
        return {
            "id": item_id,
            "kind": "search",
            "title": "Searching: " + str(item.get("query", ""))[:180],
            "status": status,
        }
    if kind == "contextCompaction":
        return {"id": item_id, "kind": "context", "title": "Compacting research context", "status": status}
    if kind == "agentMessage":
        return {"id": item_id, "kind": "response", "title": "Preparing the research result", "status": status}
    if kind == "reasoning":
        return {"id": item_id, "kind": "reasoning", "title": "Analyzing the next scientific action", "status": status}
    return None


def _command_activity_title(command: str, *, output: str = "") -> str:
    text = " ".join(str(command).lower().split())
    result = " ".join(str(output).lower().split())
    if "run_research_program" in text:
        return "Running the saved accelerator-research program"
    if "run_long_research_program" in text or "submit-expert-job" in text:
        return "Submitting a durable accelerator calculation"
    if "expert-knowledge" in text:
        return "Retrieving relevant accelerator expertise"
    if "query_evidence" in text or "inspect_evidence" in text:
        return "Inspecting archived research evidence"
    if "publish-lattice" in text:
        return "Validating and publishing the reproducible lattice"
    if "evaluate-lattice" in text:
        return "Evaluating a lattice candidate with JuTrack"
    if "progress" in text:
        return "Publishing a scientific progress update"
    if any(token in text for token in ("dynamic_aperture", "acceptance", "ringpass")):
        return "Computing nonlinear particle survival with JuTrack"
    if any(token in text for token in ("radiation", "emittance", "ringpara")):
        return "Computing radiation and equilibrium quantities with JuTrack"
    if any(token in text for token in ("optics", "twiss", "chromatic")):
        return "Computing lattice optics with JuTrack"
    if any(token in text + " " + result for token in ("pso", "optim", "pareto")):
        return "Running a bounded numerical optimization"
    if any(token in text for token in ("matplotlib", "plot", ".svg", ".png")):
        return "Generating a scientific figure"
    return "Running a bounded accelerator-physics calculation"


def _completed_command_succeeded(item: Mapping[str, Any]) -> bool:
    status = str(item.get("status", "")).lower()
    return status in {"completed", "complete", "succeeded", "success"}


def _command_output(item: Mapping[str, Any]) -> str:
    for key in ("aggregatedOutput", "aggregated_output", "output", "stderr"):
        if item.get(key) is not None:
            return str(item[key])
    return ""


def _mcp_failure_summary(item: Mapping[str, Any]) -> str:
    error = item.get("error")
    if isinstance(error, Mapping):
        message = str(error.get("message", "")).strip()
    else:
        message = str(error or "").strip()
    return _concise_narrative(message, limit=360) if message else "The local tool call failed."


def _mcp_operational_failure(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recognize failures at the local program-authoring boundary.

    Codex reports an approval-policy denial as if the user rejected the call.
    Brilliance has no interactive approval channel, so retrying that same MCP
    call can never help and must be redirected immediately.
    """

    if str(item.get("type", "")) not in {"mcpToolCall", "dynamicToolCall"}:
        return None
    if _completed_command_succeeded(item):
        return None
    tool = str(item.get("tool", item.get("name", ""))).casefold()
    if tool not in {
        "write_research_program",
        "replace_research_program_text",
        "write_research_arguments",
    }:
        return None
    summary = _mcp_failure_summary(item)
    folded = summary.casefold()
    rejected = "reject" in folded or "approval" in folded or "permission" in folded
    return {
        "kind": "workspace_writer_rejected" if rejected else "operational_abort",
        "fatal": not rejected,
        "repairable": rejected,
        "detail": (
            "The non-interactive Codex approval layer rejected Brilliance's local workspace "
            "writer. This was an internal authoring failure, not a scientist decision."
            if rejected
            else summary
        ),
        "tool": tool,
    }


def _command_operational_failure(item: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(item.get("type", "")) != "commandExecution" or _completed_command_succeeded(item):
        return None
    command = str(item.get("command", ""))
    output = _command_output(item)
    combined = (command + "\n" + output).casefold()
    if "createprocessasuserw failed: 206" in combined or (
        "windows sandbox" in combined and "failed: 206" in combined
    ):
        return {
            "kind": "source_transport_failure",
            "fatal": False,
            "repairable": True,
            "detail": (
                "Windows rejected an oversized command line. Research program source must be "
                "written with the Brilliance workspace MCP tool, never embedded in PowerShell."
            ),
            "command": _short_command(command),
        }
    if "run-expert-tool" not in command:
        return None
    repairable = any(name in command for name in (
        "accelerator.run_research_program", "accelerator.run_long_research_program"))
    return {
        "kind": "operational_abort",
        "fatal": not repairable,
        "repairable": repairable,
        "detail": _failure_summary(output),
        "command": _short_command(command),
    }


def _managed_job_receipt_from_item(item: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(item.get("type", "")) != "commandExecution" or not _completed_command_succeeded(item):
        return None
    output = _command_output(item)
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and value.get("kind") == "managed_expert_calculation":
            return dict(value)
    return None


def _failure_summary(output: str) -> str:
    structured = _structured_command_summary(output)
    if structured:
        return structured
    from .failure_diagnostics import execution_failure_summary
    return _concise_narrative(execution_failure_summary(output), limit=360)


def _result_summary(output: str) -> str:
    structured = _structured_command_summary(output)
    if structured:
        return structured
    lines = [" ".join(line.split()) for line in str(output).splitlines() if line.strip()]
    signal = re.compile(
        r"(?:completed|passed|failed|stable|unstable|emittance|tune|chromatic|dispersion|"
        r"converged|candidate|saved|dynamic aperture|momentum acceptance)",
        re.IGNORECASE,
    )
    selected = [line for line in lines if signal.search(line) and len(line) <= 420]
    return _concise_narrative("; ".join(selected[-3:]), limit=500) if selected else ""


def _structured_command_summary(output: str) -> str:
    decoder = json.JSONDecoder()
    for index, character in enumerate(str(output)):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(str(output)[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        classification = str(value.get("classification", "")).strip()
        if classification:
            checks = []
            for group in ("topology_checks", "numerical_checks", "checks"):
                rows = value.get(group, [])
                if isinstance(rows, list):
                    checks.extend(
                        str(row.get("id"))
                        for row in rows
                        if isinstance(row, Mapping) and row.get("passed") is False
                    )
            return f"Design gate {classification}" + (
                "; failed checks: " + ", ".join(checks) if checks else ""
            )
        result = value.get("result")
        if isinstance(result, Mapping) and result.get("summary"):
            return _concise_narrative(str(result["summary"]), limit=500)
    return ""


def _concise_narrative(value: str, *, limit: int) -> str:
    text = " ".join(re.sub(r"[`#*>]", "", str(value)).split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _stage_title(activity: Mapping[str, Any]) -> str:
    return str(activity.get("title") or "Research in progress")


def _model_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        rendered = dump(mode="json", by_alias=True)
        return dict(rendered) if isinstance(rendered, Mapping) else {}
    return {}


def _final_response(completed: Mapping[str, Any]) -> str:
    turn = completed.get("turn", completed)
    if not isinstance(turn, Mapping):
        return ""
    messages = []
    for item in turn.get("items", []) if isinstance(turn.get("items"), list) else []:
        if isinstance(item, Mapping) and item.get("type") == "agentMessage":
            text = str(item.get("text", "")).strip()
            if text:
                messages.append(text)
    return "\n\n".join(messages)


def _response_disposition(response: str) -> tuple[str, str]:
    matches = list(RESEARCH_STATUS_PATTERN.finditer(str(response)))
    disposition = matches[-1].group(1).lower() if matches else "complete"
    cleaned = RESEARCH_STATUS_PATTERN.sub("\n", str(response)).strip()
    return cleaned, disposition


def _compact_turn_metadata(completed: Mapping[str, Any]) -> dict[str, Any]:
    turn = completed.get("turn", completed)
    turn = turn if isinstance(turn, Mapping) else {}
    return {
        "id": turn.get("id"),
        "status": turn.get("status"),
        "item_count": len(turn.get("items", [])) if isinstance(turn.get("items"), list) else 0,
    }


def _compatible_thread_id(
    state: Mapping[str, Any],
    *,
    previous_backend_fingerprint: str,
    current_backend_fingerprint: str,
    previous_profile_fingerprint: str,
    current_profile_fingerprint: str,
) -> str:
    if not previous_backend_fingerprint or not previous_profile_fingerprint:
        return ""
    if previous_backend_fingerprint != current_backend_fingerprint:
        return ""
    if previous_profile_fingerprint != current_profile_fingerprint:
        return ""
    return str(state.get("thread_id") or "")


def _context_rollover_required(state: Mapping[str, Any]) -> bool:
    usage = state.get("token_usage", {})
    usage = usage if isinstance(usage, Mapping) else {}
    last = usage.get("last", usage)
    last = last if isinstance(last, Mapping) else {}
    total = _safe_int(last.get("totalTokens", last.get("total_tokens")), 0)
    window = _safe_int(
        usage.get("modelContextWindow", usage.get("model_context_window")), 0
    )
    return bool(window > 0 and total >= int(0.90 * window))


def _agent_profile(
    environment: ResearchEnvironment, *, project_directory: Path | str | None = None
) -> dict[str, str]:
    from .experts import EXPERT_TOOL_API_VERSION, ExpertRegistry

    registry = ExpertRegistry.from_environment(
        environment,
        project_directory=Path(project_directory) if project_directory is not None else None,
    )
    identity = {
        "version": ACCELERATOR_AGENT_PROFILE_VERSION,
        "expert_tool_api": EXPERT_TOOL_API_VERSION,
        "expert_registry": [
            {"id": item.id, "version": item.version, "status": item.status}
            for item in registry.modules
        ],
        "public_api": str(environment.accelerator_api),
    }
    return {
        "version": ACCELERATOR_AGENT_PROFILE_VERSION,
        "expert_tool_api": str(EXPERT_TOOL_API_VERSION),
        "expert_registry": content_hash(identity["expert_registry"], length=20),
        "fingerprint": content_hash(identity, length=20),
    }


def _archive_stale_handoff(
    project_directory: Path, *, previous_profile_fingerprint: str
) -> Path | None:
    path = Path(project_directory) / "workspace" / "RESEARCH_HANDOFF.md"
    if not path.is_file():
        return None
    archive = path.parent / "handoffs"
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / (
        "obsolete-" + (previous_profile_fingerprint or "unknown") + f"-{time.time_ns()}.md"
    )
    path.replace(target)
    return target


def _checkpoint_turn(
    project_root: Path | str,
    project: ResearchProject,
    *,
    timeout_seconds: int,
    increment: int,
    max_increments: int,
    backend: Mapping[str, Any],
    started_at: float,
    exc: BaseException,
    thread: Any = None,
    handle: Any = None,
) -> CodexTurnOutcome:
    root = Path(project_root).resolve()
    store = CodexRuntimeStore(root, project.id)
    checkpoint_state = store.load()
    timeouts = _safe_int(checkpoint_state.get("consecutive_timeouts"), 0) + 1
    _mark_optimizer_progress_interrupted(root, project.id, modified_after=started_at - 1.0)
    may_continue = (
        configured_auto_continue()
        and increment < max_increments
        and timeouts < configured_max_consecutive_timeouts()
    )
    attachments = _turn_attachments(root, project.id, started_at)
    partial = RESEARCH_STATUS_PATTERN.sub(
        "", str(checkpoint_state.get("partial_response", ""))
    ).strip()
    saved_result = _presentable_result_narrative(attachments)
    draft_status = "" if saved_result else _program_checkpoint_narrative(
        root, project.id, modified_after=started_at - 1.0, runtime_state=checkpoint_state
    )
    visible_checkpoint = "\n\n".join(
        item
        for item in (partial[-3_000:], saved_result, draft_status)
        if item and item not in partial
    )
    if may_continue:
        disposition = "continue"
        boundary = (
            f"The {timeout_seconds}-second model time slice ended. The Codex thread, saved "
            "programs, JuTrack outputs, and project state were preserved. A next increment may "
            "resume from this checkpoint; managed calculations continue independently."
        )
    else:
        disposition = "needs_input"
        reason = (
            f" after {timeouts} consecutive full time slices" if timeouts >= configured_max_consecutive_timeouts() else ""
        )
        boundary = (
            f"The {timeout_seconds}-second model time slice ended{reason}. Saved code, JuTrack "
            "outputs, and the thread were preserved. No managed calculation was stopped. Send a "
            "follow-up to resume or change direction."
        )
    message = "**Research checkpoint**"
    if visible_checkpoint:
        message += "\n\n" + visible_checkpoint
    else:
        progress = research_progress(checkpoint_state)
        message += (
            "\n\nNo user-facing scientific result was produced in this time slice. The last "
            f"recorded activity was **{progress.get('title', 'research reasoning')}**."
        )
    message += "\n\n" + boundary
    ResearchConversationStore(root).append_message(
        project.id, "assistant", message, attachments=attachments
    )
    thread_id = getattr(thread, "id", None) or store.load().get("thread_id")
    turn_id = getattr(handle, "id", None) or store.load().get("turn_id")
    store.update(
        status="paused",
        backend=dict(backend),
        thread_id=thread_id,
        turn_id=turn_id,
        consecutive_timeouts=timeouts,
        partial_response="",
        live_narrative="",
        error=f"{type(exc).__name__}: {exc}",
    )
    return CodexTurnOutcome(
        status="paused",
        detail=message,
        thread_id=thread_id,
        turn_id=turn_id,
        final_response=message,
        disposition=disposition,
    )


def _program_checkpoint_narrative(
    project_root: Path,
    project_id: str,
    *,
    modified_after: float,
    runtime_state: Mapping[str, Any],
) -> str:
    """Explain exactly why a timeout has no typed scientific result."""

    directory = Path(project_root) / project_id / "workspace" / "programs"
    candidates: list[Path] = []
    if directory.is_dir():
        candidates = [
            path
            for path in directory.glob("*.py")
            if path.is_file() and path.stat().st_mtime >= modified_after
        ]
    started = bool(runtime_state.get("research_program_started_at"))
    if not candidates:
        if started:
            return (
                "A registered research-program execution was started, but it did not publish a "
                "typed result before the boundary. No numerical claim was accepted."
            )
        return (
            "No user-facing scientific result was produced. No saved research program was created "
            "or executed in this time slice, so there is no JuTrack result, table, or plot to "
            "present."
        )

    program = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        source = program.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        source = ""
    lines = len(source.splitlines())
    complete = bool(re.search(r"\brun\.finish\s*\(", source))
    description = f"`workspace/programs/{program.name}` ({lines} lines)"
    if not complete:
        return (
            f"The model saved only an unfinished program draft, {description}; it has no "
            "`run.finish(...)` result publication and was not accepted as scientific evidence. "
            "No JuTrack result, table, or plot exists for this turn."
        )
    if not started:
        return (
            f"The model saved {description}, but never invoked the registered research-program "
            "runner. Therefore the source was not evaluated and no numerical result was accepted."
        )
    return (
        f"The saved program {description} was launched but did not publish a typed result before "
        "the boundary. Its partial files are preserved, but no numerical claim was accepted."
    )


def _pause_turn(
    project_root: Path | str,
    project: ResearchProject,
    detail: str,
    exc: BaseException,
    *,
    thread: Any = None,
    handle: Any = None,
) -> CodexTurnOutcome:
    root = Path(project_root).resolve()
    store = CodexRuntimeStore(root, project.id)
    state = store.load()
    concise_detail = _pause_failure_detail(detail, exc, state)
    message = (
        "The research turn stopped at its operational boundary: "
        + concise_detail
        + " Project files and prior scientific evidence were preserved."
    )
    ResearchConversationStore(root).append_message(project.id, "assistant", message)
    store.update(
        status="paused",
        thread_id=getattr(thread, "id", None) or store.load().get("thread_id"),
        turn_id=getattr(handle, "id", None) or store.load().get("turn_id"),
        partial_response="",
        live_narrative="",
        error=f"{type(exc).__name__}: {concise_detail}",
    )
    return CodexTurnOutcome(status="paused", detail=message, final_response=message, disposition="needs_input")


def _pause_failure_detail(
    detail: str,
    exc: BaseException,
    state: Mapping[str, Any],
) -> str:
    """Remove source/code tails from transport failures and retain the useful cause."""

    if type(exc).__name__ != "TransportClosedError" and "process closed stdout" not in detail:
        return _concise_narrative(detail, limit=700)
    activities = state.get("activities", [])
    failures = [
        item
        for item in activities
        if isinstance(item, Mapping) and str(item.get("status", "")) == "failed"
    ] if isinstance(activities, list) else []
    if failures:
        summary = str(failures[-1].get("summary", "")).strip()
        if summary:
            return (
                "the model process was stopped or exited before completing its current action. "
                "The last failed operation was: "
                + _concise_narrative(summary, limit=420)
            )
    return (
        "the model process was stopped or exited before completing its current action. "
        "No additional numerical failure was recorded"
    )


def _mark_optimizer_progress_interrupted(
    project_root: Path, project_id: str, *, modified_after: float
) -> int:
    directory = Path(project_root) / project_id / "workspace" / "progress"
    changed = 0
    for path in directory.glob("*.json") if directory.is_dir() else ():
        if path.stat().st_mtime < modified_after:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping) or value.get("compute_job_id"):
            continue
        if str(value.get("run_state", "")).lower() != "running":
            continue
        updated = dict(value)
        updated["run_state"] = "interrupted"
        updated["detail"] = (
            "The model-owned process was interrupted at the turn boundary and is not continuing "
            "in the background. Completed evidence was preserved."
        )
        _write_generated_text(path, json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True))
        changed += 1
    return changed


def _managed_job_checkpoint_message(receipt: Mapping[str, Any]) -> str:
    model = receipt.get("tool_id", "long research program")
    job_id = receipt.get("job_id", "unknown")
    return (
        f"The durable calculation `{model}` is running as job `{job_id}`. The model turn has "
        "ended instead of polling it. Progress and the final tables/plots will appear in this "
        "chat automatically; you may send new messages or explicitly stop the calculation."
    )


def _operational_abort_message(record: Mapping[str, Any]) -> str:
    if record.get("kind") == "pre_execution_stall":
        return (
            str(record.get("detail", "Model-only preparation exceeded its limit."))
            + " This limit applies only before simulation starts; submitted JuTrack and "
            "optimization jobs are not interrupted."
        )
    return (
        "The saved calculation reached its bounded repair limit: "
        + str(record.get("detail", "unknown operational error"))
        + " Source, arguments, and partial outputs were preserved; no scientific state changed."
    )


def _workspace_artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def _workspace_artifact_media_type(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    try:
        prefix = path.read_bytes()[:256].lstrip()
    except OSError:
        return guessed
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith((b"<?xml", b"<svg")) and b"<svg" in prefix:
        return "image/svg+xml"
    return guessed


def _workspace_artifact_title(relative_path: str) -> str:
    stem = Path(relative_path).stem.replace("_", " ").replace("-", " ")
    return " ".join(stem.split()).title() or "Research artifact"


def _resolve_completion_artifact(artifact_root: Path, reference: str) -> Path:
    """Resolve typed receipt paths without borrowing another project's files."""
    if not reference:
        raise ValueError("completion receipt has no artifact path")
    path = Path(reference)
    if not path.is_absolute():
        first = path.parts[0] if path.parts else ""
        if first == "workspace":
            path = artifact_root.parent.parent / path
        elif first == "artifacts":
            path = artifact_root.parent / path
        else:
            path = artifact_root / path
    path = path.resolve()
    path.relative_to(artifact_root.resolve())
    return path


def _turn_attachments(
    project_root: Path, project_id: str, started_at: float, *,
    completion_evidence: Mapping[str, Any] | None = None,
    include_recent_results: bool = True,
) -> list[dict[str, Any]]:
    records = (workspace_artifacts(project_root, project_id, modified_after=started_at - 1.0)
               if include_recent_results else [])
    alternatives: list[dict[str, Any]] = []
    selected_result: str | None = None
    if completion_evidence:
        # Completion can deliberately cite an older, better candidate. File
        # recency must not substitute a later rejected branch for that receipt.
        artifact_root = (project_root / project_id / "workspace" / "artifacts").resolve()
        try:
            raw = str(completion_evidence.get("evidence_full_result") or "")
            evidence_path = _resolve_completion_artifact(artifact_root, raw)
            selected_result = evidence_path.relative_to(artifact_root).as_posix()
            if (evidence_path.name != "result.json" or not evidence_path.is_file()
                    or evidence_path.parent.name != completion_evidence.get("evidence_run_id")):
                raise ValueError("completion result and run identity disagree")
            selected_value = json.loads(evidence_path.read_text(encoding="utf-8"))
            selected_candidate = selected_value.get("candidate_lattice")
            proof_candidate = completion_evidence.get("evidence_candidate_lattice")
            if proof_candidate:
                candidate_path = (evidence_path.parent / str(selected_candidate or "")).resolve()
                proof_path = _resolve_completion_artifact(artifact_root, str(proof_candidate))
                candidate_path.relative_to(evidence_path.parent)
                if candidate_path != proof_path.resolve() or not candidate_path.is_file():
                    raise ValueError("completion candidate does not match its result")
            for record in records:
                if Path(record["path"]).name != "result.json" or record["path"] == selected_result:
                    continue
                try:
                    branch = json.loads(resolve_workspace_artifact(
                        project_root, project_id, record["path"]).read_text(encoding="utf-8"))
                except (OSError, ValueError, LookupError):
                    continue
                if isinstance(branch, Mapping) and branch.get("summary"):
                    alternatives.append({
                        "run_id": Path(record["path"]).parent.name,
                        "summary": str(branch["summary"]),
                        "status": branch.get("status"),
                        "disposition": "not_selected_for_this_answer",
                    })
            records = [_workspace_artifact_record(path, artifact_root, project_id)
                       for path in sorted(evidence_path.parent.rglob("*")) if path.is_file()]
        except (OSError, ValueError, TypeError, AttributeError):
            # Never disguise a broken evidence link with an unrelated result.
            return [{"type": "evidence_summary", "title": "Selected evidence unavailable",
                     "summary": "The answer's exact calculation receipt could not be resolved. "
                                "No newer candidate was substituted; saved research is unchanged.",
                     "run_id": completion_evidence.get("evidence_run_id"),
                     "presentation_role": "selected_evidence_unavailable", "tables": []}]
    attachments: list[dict[str, Any]] = []
    declared: dict[str, dict[str, Any]] = {}
    seen_hashes: set[str] = set()
    presentation_lattice_id: str | None = None

    def declare(base: Path, item: Any, *, kind: str, default_title: str,
                artifact_role: str | None = None) -> None:
        if isinstance(item, (str, Path)):
            row = {"path": str(item)}
        elif isinstance(item, Mapping):
            row = dict(item)
        else:
            return
        raw_path = str(row.get("path", "")).strip()
        path = Path(raw_path)
        if not raw_path or path.is_absolute() or ".." in path.parts:
            return
        relative = (base / path).as_posix()
        declared[relative] = {
            "type": kind,
            "title": str(row.get("title") or default_title),
            "lattice_id": row.get("lattice_id") or presentation_lattice_id,
            "presentation_role": "selected_evidence" if selected_result else "calculation_result",
            "artifact_role": artifact_role,
        }

    # A typed result is the presentation contract. Tables are rendered directly;
    # only plots and downloads named by that result are exposed as attachments.
    for record in records:
        if Path(record["path"]).name != "result.json":
            continue
        if selected_result and record["path"] != selected_result:
            continue
        try:
            value = json.loads(
                resolve_workspace_artifact(project_root, project_id, record["path"]).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError, LookupError):
            continue
        if isinstance(value, Mapping) and (value.get("tables") or value.get("summary")):
            tables = value.get("tables", []) if isinstance(value.get("tables"), list) else []
            data = value.get("data", {}) if isinstance(value.get("data"), Mapping) else {}
            lattice_id = None if value.get("candidate_lattice") else data.get("lattice_id")
            if value.get("candidate_lattice"):
                try:
                    lattice_record = json.loads(resolve_workspace_artifact(
                        project_root, project_id,
                        (Path(record["path"]).parent / value["candidate_lattice"]).as_posix(),
                    ).read_text(encoding="utf-8"))
                    lattice_id = lattice_record.get("lattice_id")
                except (OSError, ValueError, LookupError, AttributeError):
                    pass
            presentation_lattice_id = lattice_id
            verified_metrics = (data.get("verified_candidate_metrics", {})
                                if any(isinstance(table, Mapping) and table.get("kind") == "candidate_evaluation"
                                       for table in tables) else {})
            if not isinstance(verified_metrics, Mapping):
                verified_metrics = {}
            scalar_rows = [
                {"quantity": str(key), "value": item}
                for key, item in data.items()
                if isinstance(item, (int, float)) and not isinstance(item, bool)
                and key not in verified_metrics and key not in {
                    "declared_mirror_mismatch_count", "execution_count", "returncode",
                }
            ]
            # Curated tables (e.g. bend inventory) do not replace independent
            # scalar findings (e.g. chromaticities). Omit only exact key/value
            # observations already represented in those tables.
            def already_tabulated(row: Mapping[str, Any]) -> bool:
                return any(
                    isinstance(entry, Mapping) and (
                        (row["quantity"] in entry and entry[row["quantity"]] == row["value"])
                        or (entry.get("quantity") == row["quantity"] and entry.get("value") == row["value"])
                    )
                    for table in tables if isinstance(table, Mapping)
                    for entry in table.get("rows", []) if isinstance(table.get("rows"), list)
                )

            remaining = [row for row in scalar_rows if not already_tabulated(row)]
            rendered_tables = ([{"title": "Computed quantities", "kind": "automatic_quantities", "rows": remaining[:30]}]
                               if remaining else []) + tables
            attachments.append(
                {
                    "type": "evidence_summary",
                    "title": str(value.get("title") or "Scientific result"),
                    "summary": value.get("summary", ""),
                    "steps": value.get("methods", []),
                    "tables": rendered_tables,
                    "limitations": value.get("limitations", []),
                    "pending_acceptance": value.get("pending_acceptance", []),
                    "status": value.get("status"),
                    "scientific_passed": value.get("scientific_passed"),
                    "request_satisfied": value.get("request_satisfied"),
                    "candidate_lattice": value.get("candidate_lattice"),
                    "lattice_id": lattice_id,
                    "candidate_handoff": value.get("candidate_handoff", {}),
                    "working_lattice_action": (completion_evidence.get("working_lattice_action")
                                               if selected_result and completion_evidence else None),
                    "scientific_stage": value.get("scientific_stage", {}),
                    "outcome_presentation": value.get("outcome_presentation", {}),
                    "presentation_role": "selected_evidence" if selected_result else "calculation_result",
                    "run_id": Path(str(record["path"])).parent.name,
                    "result_path": str(record["path"]),
                }
            )
            base = Path(str(record["path"])).parent
            for plot in value.get("plots", []) if isinstance(value.get("plots"), list) else []:
                declare(base, plot, kind="image", default_title="Scientific figure")
            if value.get("candidate_lattice"):
                declare(
                    base,
                    value["candidate_lattice"],
                    kind="download",
                    default_title="Candidate lattice definition",
                    artifact_role="candidate_definition",
                )
            for download in (
                value.get("downloads", []) if isinstance(value.get("downloads"), list) else []
            ):
                declare(base, download, kind="download", default_title="Research data")

    undeclared_images: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: float(row.get("modified_at", 0.0))):
        declaration = declared.get(str(record.get("path", "")))
        media_type = str(record.get("media_type", ""))
        if declaration is None:
            if media_type.startswith("image/"):
                undeclared_images.append({"type": "image", **record})
            continue
        digest = str(record.get("content_sha256", ""))
        if digest and digest in seen_hashes:
            continue
        if digest:
            seen_hashes.add(digest)
        attachments.append({**record, **declaration})

    # A program interrupted before finish may still have produced a useful
    # figure. Show a small number inline, but never expose raw stdout, stderr,
    # source, arguments, or runner JSON as the primary answer.
    if not any(item.get("type") == "evidence_summary" for item in attachments):
        for record in undeclared_images[-4:]:
            digest = str(record.get("content_sha256", ""))
            if not digest or digest not in seen_hashes:
                attachments.append({**record, "presentation_role": "unverified_artifact",
                                    "title": str(record.get("title", "Figure")) + " — unverified output"})
                if digest:
                    seen_hashes.add(digest)

    images = [item for item in attachments if item.get("type") == "image"]
    summaries = [item for item in attachments if item.get("type") == "evidence_summary"]
    downloads = [item for item in attachments if item.get("type") == "download"]
    if not selected_result:
        images, summaries, downloads = images[-8:], summaries[-4:], downloads[-4:]
    downloads.sort(key=lambda item: item.get("artifact_role") != "candidate_definition")
    comparison = [{"type": "alternative_results", "title": "Other calculations — not selected for this answer",
                   "rows": alternatives}] if alternatives else []
    return [*summaries, *images, *downloads, *comparison]


def _presentable_result_narrative(attachments: list[Mapping[str, Any]]) -> str:
    """Create a compact chat answer from the newest typed program result."""

    summaries = [item for item in attachments if item.get("type") == "evidence_summary"]
    if not summaries:
        return ""
    result = summaries[-1]
    summary = str(result.get("summary", "")).strip()
    if not summary:
        return ""
    lines = ["**Latest saved scientific result**", summary]
    tables = [item for item in result.get("tables", []) if isinstance(item, Mapping)]
    if tables:
        labels = [str(item.get("title") or "evidence table") for item in tables]
        lines.append("Inline evidence: " + ", ".join(labels) + ".")
    limitations = [str(item).strip() for item in result.get("limitations", []) if str(item).strip()]
    if limitations:
        lines.append("Limitations: " + "; ".join(limitations[:3]))
    pending = [str(item).strip() for item in result.get("pending_acceptance", []) if str(item).strip()]
    if pending:
        lines.append("Pending final acceptance: " + "; ".join(pending[:3]))
    return "\n\n".join(lines)


def _with_artifact_notice(response: str, attachments: list[Mapping[str, Any]]) -> str:
    if not attachments:
        return response
    images = sum(item.get("type") == "image" for item in attachments)
    downloads = sum(item.get("type") == "download" for item in attachments)
    if not images and not downloads:
        return response
    return response.rstrip() + f"\n\nAttached: {images} figure(s) and {downloads} reproducible file(s)."


def _codex_research_config_overrides(
    environment: ResearchEnvironment, installed_depot: str
) -> tuple[str, ...]:
    base = (
        f"sandbox_workspace_write.writable_roots=[{json.dumps(installed_depot)}]",
        "sandbox_workspace_write.network_access="
        + ("true" if environment.internet_access else "false"),
        *_codex_mcp_policy_overrides(environment),
    )
    if environment.internet_access:
        return (*base, 'web_search="live"', "features.apps=true")
    return (
        *base,
        'web_search="disabled"',
        "features.apps=false",
        "features.browser_use=false",
        "features.browser_use_external=false",
        "features.in_app_browser=false",
        "features.computer_use=false",
        "features.plugins=false",
        "features.remote_plugin=false",
    )


def _codex_mcp_policy_overrides(
    environment: ResearchEnvironment, *, config_path: Path | None = None
) -> tuple[str, ...]:
    path = config_path or Path.home() / ".codex" / "config.toml"
    if not path.is_file():
        return ()
    try:
        servers = tomllib.loads(path.read_text(encoding="utf-8")).get("mcp_servers", {})
    except (OSError, tomllib.TOMLDecodeError):
        return ("mcp_servers={}",)
    if not isinstance(servers, Mapping):
        return ()
    overrides: list[str] = []
    for name, value in servers.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(name)):
            return ("mcp_servers={}",)
        settings = value if isinstance(value, Mapping) else {}
        remote = bool(settings.get("url") or settings.get("http_url"))
        disabled = (remote and not environment.remote_mcp_enabled) or (
            not remote and not environment.local_mcp_enabled
        )
        if disabled:
            overrides.append(f"mcp_servers.{name}.enabled=false")
    return tuple(overrides)


def _codex_workspace_mcp_overrides(
    environment: ResearchEnvironment,
    *,
    project_root: Path | None = None,
    project_id: str | None = None,
) -> tuple[str, ...]:
    """Install the always-local atomic program writer for this Codex process."""

    if not environment.local_mcp_enabled:
        return ()
    python = json.dumps(str(environment.python_executable))
    overrides = [
        f"mcp_servers.brilliance_workspace.command={python}",
        'mcp_servers.brilliance_workspace.args=["-m","brilliance.workspace_mcp"]',
        "mcp_servers.brilliance_workspace.enabled=true",
        'mcp_servers.brilliance_workspace.default_tools_approval_mode="approve"',
        "mcp_servers.brilliance_workspace.startup_timeout_sec=15",
        "mcp_servers.brilliance_workspace.tool_timeout_sec=30",
    ]
    if project_root is not None and project_id:
        root_value = json.dumps(str(Path(project_root).resolve()))
        project_value = json.dumps(str(project_id))
        overrides.append(
            "mcp_servers.brilliance_workspace.env="
            "{BRILLIANCE_PROJECT_ROOT="
            f"{root_value},BRILLIANCE_PROJECT_ID={project_value}}}"
        )
    return tuple(overrides)


def _developer_instructions(
    project: ResearchProject,
    project_root: Path,
    environment: ResearchEnvironment,
    *,
    research_mode: str = "research",
    query: str = "",
) -> str:
    from .coding_policy import SCIENTIFIC_CODING_POLICY
    from .experts import ExpertRegistry

    command = _project_command(project, project_root, environment)
    registry = ExpertRegistry.from_environment(
        environment, project_directory=project_root / project.id
    )
    request = "\n".join(item for item in (project.goal.strip(), query.strip()) if item)
    knowledge = registry.render_context(
        request,
        max_chars=environment.expert_max_context_chars,
        max_chunks=4,
    )
    from .scientific_acceptance import acceptance_instructions, continued_request_acceptance

    required = continued_request_acceptance(
        query, ResearchConversationStore(project_root).messages(project.id), registry=registry)
    acceptance = acceptance_instructions(required, registry=registry)
    network = (
        "Internet research is enabled; use it only when the request needs external sources."
        if environment.internet_access
        else "Internet research is disabled. Do not search, browse, download, or call remote MCP."
    )
    return SCIENTIFIC_CODING_POLICY + f"""
You are Brilliance, a general accelerator-physics coding agent. The newest scientist
message is the immediate task. JuTrack is the sole numerical beam-dynamics authority.

Configured runtime (already verified): Python `{environment.python_executable}`; JuTrack
`{environment.jutrack_root}`; public API `brilliance.accelerator_api`. Never search for runtimes,
install packages, browse sibling projects, or replace JuTrack with
hand calculations.

Inspect the exact saved inputs and focused contracts needed for the requested calculation.
`workspace/ACCELERATOR_API_REFERENCE.md` documents public helpers;
`workspace/JUTRACK_API_REFERENCE.md` contains read-only signatures, documentation and static
return information from the configured Python wrapper, without loading Julia. Read relevant
sections before writing code; no environment rediscovery is needed.
Use `brilliance_workspace.write_research_program` to save a short Python experiment under
`workspace/programs/`, save arguments with `write_research_arguments`, run `{command} check-program
programs/NAME.py`, then execute the actual small calculation with `accelerator.run_research_program`. Use
`accelerator.run_long_research_program` only for an hours-or-days job and never poll it. Source must
use `brilliance.research_api.open_research_run` and `brilliance.accelerator_api`. A first program is
normally under 250 lines and produces one meaningful JuTrack result, not a schema or environment
probe. A representative evaluation of the real custom residual/objective is useful evidence.
If repair is needed, inspect the relevant saved source/API section and repair that layer;
do not repeat an unchanged failed attempt. Never transport code through PowerShell, stdin,
base64, or redirection.

The saved program owns task-specific data, matching, optimization, plots, and limitations. Use
exact conserved length/angle constraints; evaluate one cell in inner loops; never match an
unspecified tune. A design result must distinguish `scientific_passed`, `design_status`, and
`aspiration_met`. A feasible but poor seed is not a successful design, and a route is not saturated
without convergence evidence. For lattice design use `api.save_optics_lattice_plot` for one-cell
beta/dispersion/lattice evidence. Radiation interpretation requires JuTrack radiation and the
per-bend consistency gate.

Chat is the research interface. `run.progress` must publish each material numerical milestone with
clear text and current tables/plots; `run.finish` must explain Result, Evidence, Limitations, and
Next. Files are secondary reproducibility material. Missing a target is valid negative evidence;
an API exception is not a scientific result.

Select a fitting installed capability or author a justified method from the physics requirements.
Treat draft expertise as provisional, preserve pending comparisons and use numerical evidence
to choose the next change. Historical results from other projects are not permitted design seeds.
End with exactly one private
`brilliance-research-status` marker: `complete`, `continue`, or `needs_input`.

Mode: {research_mode}. {network}

Relevant installed knowledge:

{knowledge}

{acceptance}
"""


def _turn_prompt(
    project: ResearchProject,
    message: str,
    *,
    initial: bool,
    research_mode: str = "research",
) -> str:
    prefix = (
        "Inspect the necessary current-project inputs and focused API/method contract, then "
        "save a short scientific program with brilliance_workspace.write_research_program. "
        "Use write_research_arguments, check-program and execution to test the actual calculation "
        "or callback before a larger run. Reuse an installed method when it fits; otherwise "
        "write task-specific scientific Python. Do not bootstrap or discover a runtime."
        if initial
        else "Continue from saved code, evidence, and contracts without replaying completed work."
    )
    return (
        f"{prefix}\n\nProject goal:\n{project.goal.strip()}"
        f"\n\nScientist message:\n{message.strip()}"
        "\n\nThis is one bounded research increment. Preserve outputs and end with the required "
        "brilliance-research-status marker."
    )


def _requires_jutrack_execution(
    message: str, *, initial: bool, research_mode: str
) -> bool:
    """Gate computational greenfield designs against a text-only completion."""

    if not initial or research_mode != "research":
        return False
    from .research_admission import complex_greenfield_request

    return complex_greenfield_request(message)


def _workspace_agents_md(
    project: ResearchProject, project_root: Path, environment: ResearchEnvironment
) -> str:
    from .coding_policy import SCIENTIFIC_CODING_POLICY
    command = _project_command(project, project_root, environment)
    return f"""<!-- brilliance-managed: accelerator-agent-v{ACCELERATOR_AGENT_PROFILE_VERSION} -->
# Brilliance accelerator-research workspace

Project: `{project.id}`

> {project.goal}

Configured Python: `{environment.python_executable}`  
Configured JuTrack: `{environment.jutrack_root}`  
Public API: `brilliance.accelerator_api`  
Project command: `{command}`

{SCIENTIFIC_CODING_POLICY}

The runtime is prepared. Read the required local API/method reference and exact current-project
inputs, then save a bounded program with `brilliance_workspace.write_research_program` and JSON
arguments with `write_research_arguments`. The generated `workspace/JUTRACK_API_REFERENCE.md`
describes the configured native wrapper without importing Julia.

```powershell
{command} check-program programs/NAME.py
{command} run-expert-tool accelerator.run_research_program --arguments-file workspace/programs/run.json
```

Programs use `open_research_run()` plus `brilliance.accelerator_api`, publish useful
`run.progress(...)` messages with tables/plots, and finish with a scientist-readable result. Use
the long-program tool for durable optimization. Do not put source in shell commands or treat
files as the user-facing answer. Exact public API signatures are in
`workspace/ACCELERATOR_API_REFERENCE.md`; reading them before a new calculation is encouraged.

"""


def _project_context(
    project: ResearchProject, project_directory: Path, environment: ResearchEnvironment
) -> str:
    lines = [
        f"# Research project: {project.title}",
        "",
        f"Goal: {project.goal}",
        f"Project ID: `{project.id}`",
        f"Archive: `{project_directory}`",
        f"Configured Python: `{environment.python_executable}`",
        f"Configured JuTrack: `{environment.jutrack_root}`",
        "",
    ]
    if project.context:
        lines.extend(["## Scientist context", "", project.context[:20_000], ""])
    if project.active_lattice_id:
        lines.extend(["## Canonical lattice", "", f"Active ID: `{project.active_lattice_id}`", ""])
    else:
        lines.extend(
            [
                "## Current subject",
                "",
                "No canonical lattice is active. This is not a blocker for non-lattice research.",
                "",
            ]
        )
    return "\n".join(lines)


def _project_command(
    project: ResearchProject, project_root: Path, environment: ResearchEnvironment
) -> str:
    python = environment.manifest(project_root=project_root)["python"]["command"]
    return (
        f"{python} -m brilliance.codex_tools --project-root {str(project_root)!r} "
        f"--project-id {project.id!r}"
    )


def _write_generated_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{os.getpid()}-{time.time_ns()}.tmp"
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                source.unlink(missing_ok=True)
                raise
            time.sleep(0.025 * (attempt + 1))


def _write_managed_agents(path: Path, content: str) -> None:
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        generated = "<!-- brilliance-managed:" in existing or existing.startswith(
            "# Brilliance Codex research workspace"
        )
        if not generated:
            _write_generated_text(
                path.parent / "workspace" / "BRILLIANCE_AGENT_INSTRUCTIONS.md", content
            )
            return
    _write_generated_text(path, content)


def _short_command(command: str) -> str:
    compact = " ".join(str(command).split())
    return compact[:177] + "..." if len(compact) > 180 else compact or "a research command"
