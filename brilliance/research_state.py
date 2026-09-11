"""Domain-neutral, append-only state for interactive scientific research.

The project revision records the broad research programme.  This module records
the much smaller *current experimental contract*: the subject being studied,
the constraints that must survive follow-up requests, and the exact arguments
that produced the current result.  Expert modules may opt into this state with
one ``state_scope`` declaration; the kernel does not need to understand the
physics represented by the values.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_hash, json_safe

RESEARCH_STATE_SCHEMA_VERSION = "1.0"
_SCOPE = re.compile(r"^[a-z][a-z0-9_.-]{2,127}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"immutable research-state conflict: {path}")


class ResearchStateStore:
    """Store and retrieve current contracts without assuming a research domain."""

    def __init__(self, project_directory: Path | str):
        self.project_directory = Path(project_directory).resolve()
        self.workspace = self.project_directory / "workspace"
        self.records = self.workspace / "research_state" / "records"

    def record_execution(
        self,
        execution: Mapping[str, Any],
        *,
        artifact: Path | str | None = None,
    ) -> Path | None:
        contract = _execution_contract(execution)
        if contract is None:
            return None
        scope = str(contract.get("scope", "")).strip().lower()
        if not _SCOPE.fullmatch(scope):
            raise ValueError(f"invalid research-state scope: {scope!r}")
        validation = execution.get("validation", {})
        result = execution.get("result", {})
        validation = validation if isinstance(validation, Mapping) else {}
        result = result if isinstance(result, Mapping) else {}
        execution_valid = validation.get("passed") is True
        scientific_verdict = validation.get("scientific_passed")
        if not isinstance(scientific_verdict, bool):
            scientific_verdict = (
                result.get("passed") if isinstance(result.get("passed"), bool) else None
            )
        scientific = (
            "satisfied"
            if scientific_verdict is True
            else "not_satisfied"
            if scientific_verdict is False
            else "not_applicable"
        )
        active = bool(
            execution_valid
            and scientific != "not_satisfied"
            and contract.get("advance", True) is not False
        )
        subject = contract.get("subject", {})
        if not isinstance(subject, Mapping):
            subject = {}
        record = {
            "schema_version": RESEARCH_STATE_SCHEMA_VERSION,
            "recorded_at": _now(),
            "project_id": execution.get("project_id"),
            "scope": scope,
            "subject": dict(subject),
            "parent_subject": (
                dict(contract.get("parent_subject", {}))
                if isinstance(contract.get("parent_subject"), Mapping)
                else {}
            ),
            "arguments": (
                dict(contract.get("arguments", {}))
                if isinstance(contract.get("arguments"), Mapping)
                else {}
            ),
            "constraints": json_safe(contract.get("constraints", [])),
            "decisions": json_safe(contract.get("decisions", [])),
            "execution_id": execution.get("execution_id"),
            "tool_id": execution.get("tool_id"),
            "module_id": execution.get("module_id"),
            "module_version": execution.get("module_version"),
            "tool_implementation_hash": execution.get("tool_implementation_hash"),
            "execution_valid": execution_valid,
            "scientific_status": scientific,
            "active": active,
            "artifact": str(artifact) if artifact else None,
        }
        identity = {
            key: value
            for key, value in record.items()
            if key not in {"recorded_at", "artifact"}
        }
        digest = content_hash(identity, length=24)
        path = self.records / f"{digest}.json"
        # Preserve the first archival time/path for idempotent re-indexing.
        if path.is_file():
            return path
        _write_once(path, record)
        return path

    def latest(
        self,
        scope: str,
        *,
        subject_id: str | None = None,
        include_inactive: bool = False,
    ) -> dict[str, Any] | None:
        normalized = str(scope).strip().lower()
        if not _SCOPE.fullmatch(normalized):
            return None
        self._bootstrap_execution_artifacts()
        candidates: list[tuple[str, int, dict[str, Any]]] = []
        for path in self.records.glob("*.json") if self.records.is_dir() else ():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(value, Mapping) or value.get("scope") != normalized:
                continue
            if not include_inactive and value.get("active") is not True:
                continue
            subject = value.get("subject", {})
            actual_id = str(subject.get("id", "")) if isinstance(subject, Mapping) else ""
            parent = value.get("parent_subject", {})
            parent_id = str(parent.get("id", "")) if isinstance(parent, Mapping) else ""
            if subject_id and str(subject_id) not in {actual_id, parent_id}:
                continue
            candidates.append(
                (str(value.get("recorded_at", "")), path.stat().st_mtime_ns, dict(value))
            )
        return max(candidates, default=("", 0, None), key=lambda item: (item[0], item[1]))[2]

    def find(
        self,
        scope: str,
        *,
        subject_id: str | None = None,
        include_inactive: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Return all matching contracts in archival order.

        ``latest`` is appropriate for a single current experimental contract.
        Population campaigns and hypothesis banks instead have several sibling
        executions that must remain independently addressable.  This query keeps
        that distinction domain-neutral and never rewrites the append-only store.
        """

        normalized = str(scope).strip().lower()
        if not _SCOPE.fullmatch(normalized):
            return ()
        self._bootstrap_execution_artifacts()
        candidates: list[tuple[str, int, dict[str, Any]]] = []
        for path in self.records.glob("*.json") if self.records.is_dir() else ():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(value, Mapping) or value.get("scope") != normalized:
                continue
            if not include_inactive and value.get("active") is not True:
                continue
            subject = value.get("subject", {})
            actual_id = str(subject.get("id", "")) if isinstance(subject, Mapping) else ""
            parent = value.get("parent_subject", {})
            parent_id = str(parent.get("id", "")) if isinstance(parent, Mapping) else ""
            if subject_id and str(subject_id) not in {actual_id, parent_id}:
                continue
            candidates.append(
                (str(value.get("recorded_at", "")), path.stat().st_mtime_ns, dict(value))
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in candidates)

    def summary(self) -> dict[str, Any]:
        self._bootstrap_execution_artifacts()
        current: dict[str, dict[str, Any]] = {}
        for path in self.records.glob("*.json") if self.records.is_dir() else ():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(value, Mapping) or value.get("active") is not True:
                continue
            scope = str(value.get("scope", ""))
            previous = current.get(scope)
            if previous is None or str(value.get("recorded_at", "")) >= str(
                previous.get("recorded_at", "")
            ):
                current[scope] = dict(value)
        return {
            "schema_version": RESEARCH_STATE_SCHEMA_VERSION,
            "contracts": [
                {
                    "scope": scope,
                    "subject": value.get("subject", {}),
                    "arguments": value.get("arguments", {}),
                    "constraints": value.get("constraints", []),
                    "execution_id": value.get("execution_id"),
                    "tool_id": value.get("tool_id"),
                }
                for scope, value in sorted(current.items())
            ],
        }

    def _bootstrap_execution_artifacts(self) -> None:
        """Index old execution files lazily without rewriting those immutable files."""

        marker = self.workspace / "research_state" / "bootstrap-v1.json"
        artifact_root = self.workspace / "artifacts"
        latest_mtime = max(
            (path.stat().st_mtime_ns for path in artifact_root.glob("expert-execution-*.json")),
            default=0,
        )
        if marker.is_file():
            try:
                previous = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
            if int(previous.get("latest_execution_mtime_ns", -1)) == latest_mtime:
                return
        indexed = 0
        for path in artifact_root.glob("expert-execution-*.json") if artifact_root.is_dir() else ():
            try:
                execution = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(execution, Mapping) and self.record_execution(
                    execution, artifact=path
                ) is not None:
                    indexed += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_name(
            f".{marker.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": RESEARCH_STATE_SCHEMA_VERSION,
                    "latest_execution_mtime_ns": latest_mtime,
                    "indexed_contracts": indexed,
                    "updated_at": _now(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(marker)


def _execution_contract(execution: Mapping[str, Any]) -> dict[str, Any] | None:
    result = execution.get("result", {})
    result = result if isinstance(result, Mapping) else {}
    declared = result.get("research_contract")
    if isinstance(declared, Mapping):
        return dict(declared)
    state = execution.get("state_contract")
    if isinstance(state, Mapping) and str(state.get("scope", "")).strip():
        return dict(state)

    # Backward-compatible convention for existing accelerator artifacts.  The
    # kernel treats these as opaque internal-optics constraints; it does not
    # interpret MBA topology or any particular physics target.
    targets = result.get("internal_optics_targets")
    candidate = result.get("candidate", {})
    validation = execution.get("validation", {})
    if (
        isinstance(targets, list)
        and targets
        and isinstance(candidate, Mapping)
        and str(candidate.get("lattice_id", ""))
        and isinstance(validation, Mapping)
        and validation.get("passed") is True
        and result.get("passed") is True
    ):
        return {
            "scope": "accelerator.lattice.internal_optics",
            "subject": {"kind": "lattice", "id": candidate.get("lattice_id")},
            "parent_subject": dict(result.get("source_parent", {}))
            if isinstance(result.get("source_parent"), Mapping)
            else {},
            "arguments": {"internal_optics_targets": list(targets)},
            "constraints": list(targets),
        }
    return None


def infer_subject_id(context: Any, arguments: Mapping[str, Any]) -> str | None:
    """Infer a state subject without embedding any domain-specific field values."""

    raw = str(arguments.get("lattice_artifact", "") or "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path(context.workspace) / candidate
        try:
            candidate = candidate.resolve()
            candidate.relative_to(Path(context.workspace).resolve())
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(value, Mapping) and str(value.get("lattice_id", "")):
                return str(value["lattice_id"])
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    pointer = Path(context.workspace) / "WORKING_LATTICE.json"
    if pointer.is_file():
        try:
            value = json.loads(pointer.read_text(encoding="utf-8"))
            if isinstance(value, Mapping) and str(value.get("lattice_id", "")):
                return str(value["lattice_id"])
        except (OSError, json.JSONDecodeError):
            pass
    active = context.active_lattice()
    return str(active.lattice_id) if active is not None else None


__all__ = [
    "RESEARCH_STATE_SCHEMA_VERSION",
    "ResearchStateStore",
    "infer_subject_id",
]
