"""Durable project records for conversational, program-driven research.

Projects intentionally do not contain a workflow graph.  They retain the
scientist's request, conversation-compatible legacy fields, accepted lattice
history, and immutable evidence references.  Codex decides the next experiment
from the conversation and saved workspace programs.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifacts import canonical_json, content_hash, json_safe

PROJECT_SCHEMA_VERSION = "3.0"
MANIFEST_LIVE_ROOTS = {"codex", "conversation", "jobs", "workspace", "chat_checkpoints"}

# Labels are retained only so historical evidence remains readable in the UI.
# They are not a capability menu and do not select research workflows.
GENERAL_CAPABILITIES: dict[str, dict[str, str]] = {
    key: {"title": title}
    for key, title in {
        "scientist_dialogue": "Scientist decision",
        "knowledge_synthesis": "Physics synthesis",
        "reference_comparison": "Reference comparison",
        "validated_periodic_cell": "Periodic-cell study",
        "validated_mba_cell": "MBA-cell study",
        "jutrack_script": "JuTrack research program",
        "lattice_construction": "Lattice construction",
        "linear_matching": "Linear-optics matching",
        "radiation_analysis": "Radiation analysis",
        "chromatic_correction": "Chromatic correction",
        "multiobjective_optimization": "Multi-objective optimization",
        "nonlinear_tracking": "Nonlinear tracking",
        "error_correction": "Errors and correction",
        "cross_validation": "Cross-validation",
        "publication_synthesis": "Scientific synthesis",
    }.items()
}


@dataclass(frozen=True, slots=True)
class ProjectTask:
    """Legacy task record kept for lossless loading of saved projects."""

    id: str
    title: str
    objective: str
    capability: str
    dependencies: tuple[str, ...] = ()
    status: str = "proposed"
    rationale: str = ""
    acceptance_criteria: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    deliverables: tuple[str, ...] = ()
    suggested_jutrack_calls: tuple[str, ...] = ()
    scientist_question: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectTask":
        sequence_fields = {
            "dependencies",
            "acceptance_criteria",
            "required_inputs",
            "deliverables",
            "suggested_jutrack_calls",
        }
        allowed = set(cls.__dataclass_fields__)
        values = {key: value for key, value in data.items() if key in allowed}
        for key in sequence_fields:
            values[key] = tuple(values.get(key, ()))
        values["parameters"] = dict(values.get("parameters", {}))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ResearchProject:
    schema_version: str
    id: str
    revision: int
    created_at: str
    updated_at: str
    goal: str
    context: str
    title: str
    goal_interpretation: str
    machine_summary: str
    success_definition: tuple[str, ...]
    knowns: tuple[str, ...]
    unknowns: tuple[str, ...]
    assumptions: tuple[str, ...]
    hypotheses: tuple[str, ...]
    tasks: tuple[ProjectTask, ...]
    current_task_id: str | None
    status: str
    next_action: str
    claim_boundary: str
    model: str
    dialogue: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    lattice_history: tuple[Mapping[str, Any], ...] = ()
    active_lattice_id: str | None = None
    goal_contract: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchProject":
        sequence_fields = {
            "success_definition",
            "knowns",
            "unknowns",
            "assumptions",
            "hypotheses",
            "dialogue",
            "evidence",
            "lattice_history",
            "goal_contract",
        }
        defaults: dict[str, Any] = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "revision": 1,
            "created_at": _now(),
            "updated_at": _now(),
            "context": "",
            "title": "Accelerator research",
            "goal_interpretation": "",
            "machine_summary": "",
            "success_definition": (),
            "knowns": (),
            "unknowns": (),
            "assumptions": (),
            "hypotheses": (),
            "current_task_id": None,
            "status": "researching",
            "next_action": "Continue from the saved conversation and workspace.",
            "claim_boundary": "Numerical claims require saved JuTrack evidence.",
            "model": "codex",
            "dialogue": (),
            "evidence": (),
            "lattice_history": (),
            "active_lattice_id": None,
            "goal_contract": (),
        }
        allowed = set(cls.__dataclass_fields__) - {"tasks"}
        values = {key: data.get(key, default) for key, default in defaults.items() if key in allowed}
        for key in allowed - values.keys():
            if key in data:
                values[key] = data[key]
        for key in sequence_fields:
            values[key] = tuple(values.get(key, ()))
        values["tasks"] = tuple(
            ProjectTask.from_dict(item)
            for item in data.get("tasks", ())
            if isinstance(item, Mapping)
        )
        return cls(**values)


class ProjectStore:
    """Append-only project revisions with content-addressed large state."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        return self.root / project_id

    def save(self, project: ResearchProject) -> Path:
        directory = self.project_dir(project.id) / "revisions"
        directory.mkdir(parents=True, exist_ok=True)
        payload = self._externalize_project_state(project.id, json_safe(project))
        path = directory / f"{project.revision:04d}-{content_hash(payload, length=12)}.json"
        _write_once(path, payload)
        self.snapshot_manifest(project.id)
        return path

    def list_project_ids(self) -> list[str]:
        projects: list[tuple[int, str]] = []
        for directory in self.root.glob("project-*"):
            revisions = list((directory / "revisions").glob("*.json"))
            if revisions:
                projects.append((max(path.stat().st_mtime_ns for path in revisions), directory.name))
        return [project_id for _, project_id in sorted(projects, reverse=True)]

    def load(self, project_id: str) -> ResearchProject | None:
        revisions = list((self.project_dir(project_id) / "revisions").glob("*.json"))
        if not revisions:
            return None

        def identity(path: Path) -> tuple[int, int]:
            try:
                revision = int(path.name.split("-", 1)[0])
            except ValueError:
                revision = -1
            return revision, path.stat().st_mtime_ns

        payload = json.loads(max(revisions, key=identity).read_text(encoding="utf-8"))
        return ResearchProject.from_dict(self._hydrate_project_state(project_id, payload))

    def _externalize_project_state(
        self, project_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        rendered = dict(payload)
        project_directory = self.project_dir(project_id)
        state_directory = project_directory / "state"

        def externalize(items: Any, collection: str) -> list[Any]:
            if not isinstance(items, (list, tuple)):
                return []
            references: list[Any] = []
            for item in items:
                if not isinstance(item, Mapping):
                    references.append(item)
                    continue
                package = None
                if collection == "lattices":
                    try:
                        from .lattice_archive import archive_lattice_package

                        package = archive_lattice_package(project_directory, project_id, item)
                    except (KeyError, TypeError, ValueError):
                        package = None
                digest = content_hash(item, length=24)
                path = (
                    package.definition_path
                    if package is not None
                    else state_directory / collection / f"{digest}.json"
                )
                if package is None:
                    _write_once(path, item)
                reference: dict[str, Any] = {
                    "$ref": path.relative_to(project_directory).as_posix(),
                    "content_hash": digest,
                }
                for key in (
                    "evidence_id",
                    "task_id",
                    "status",
                    "timestamp",
                    "source",
                    "lattice_id",
                    "parent_lattice_id",
                    "title",
                ):
                    if key in item:
                        reference[key] = json_safe(item.get(key))
                references.append(reference)
            return references

        rendered["evidence"] = externalize(rendered.get("evidence", ()), "evidence")
        rendered["lattice_history"] = externalize(
            rendered.get("lattice_history", ()), "lattices"
        )
        rendered["state_storage"] = {"version": "1.0", "mode": "content_addressed_references"}
        return rendered

    def _hydrate_project_state(self, project_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        rendered = dict(payload)
        project_directory = self.project_dir(project_id).resolve()

        def hydrate(items: Any) -> list[Any]:
            if not isinstance(items, (list, tuple)):
                return []
            values: list[Any] = []
            for item in items:
                if not isinstance(item, Mapping) or "$ref" not in item:
                    values.append(item)
                    continue
                path = (project_directory / str(item["$ref"])).resolve()
                try:
                    path.relative_to(project_directory)
                except ValueError as exc:
                    raise ValueError(f"Project state reference escapes its archive: {path}") from exc
                if not path.is_file():
                    raise ValueError(f"Project state reference is missing: {path}")
                actual = json.loads(path.read_text(encoding="utf-8"))
                expected = str(item.get("content_hash", ""))
                if expected and content_hash(actual, length=24) != expected:
                    raise ValueError(f"Project state reference failed identity check: {path}")
                values.append(actual)
            return values

        rendered["evidence"] = hydrate(rendered.get("evidence", ()))
        rendered["lattice_history"] = hydrate(rendered.get("lattice_history", ()))
        rendered.pop("state_storage", None)
        return rendered

    def ensure_lattice_packages(self, project_id: str) -> Mapping[str, Mapping[str, str]]:
        project = self.load(project_id)
        if project is None:
            raise LookupError(f"Unknown project: {project_id}")
        from .lattice_archive import archive_lattice_package

        packages: dict[str, Mapping[str, str]] = {}
        project_directory = self.project_dir(project_id)
        for definition in project.lattice_history:
            package = archive_lattice_package(project_directory, project_id, definition)
            packages[str(definition.get("lattice_id", ""))] = {
                "definition": str(package.definition_path),
                "reproduction": str(package.reproduction_path),
                "provenance": str(package.provenance_path),
            }
        return packages

    def lattice_package_file(self, project_id: str, lattice_id: str, artifact: str) -> Path:
        if artifact not in {"definition", "reproduction", "provenance"}:
            raise LookupError(f"Unknown lattice package artifact: {artifact}")
        try:
            path = Path(self.ensure_lattice_packages(project_id)[lattice_id][artifact])
        except KeyError as exc:
            raise LookupError(f"Lattice {lattice_id!r} is not archived by {project_id!r}") from exc
        if not path.is_file():
            raise LookupError(f"Missing lattice package artifact: {path}")
        return path

    def snapshot_manifest(self, project_id: str) -> Path:
        directory = self.project_dir(project_id)
        manifest_directory = directory / "manifests"
        manifest_directory.mkdir(parents=True, exist_ok=True)
        records = [self._artifact_record(path, directory) for path in _manifest_artifact_paths(directory)]
        sequence = len(list(manifest_directory.glob("*.json"))) + 1
        payload = {
            "schema_version": "2.0",
            "project_id": project_id,
            "sequence": sequence,
            "created_at": _now(),
            "artifacts": records,
        }
        path = manifest_directory / f"{sequence:04d}-{content_hash(payload, length=16)}.json"
        _write_once(path, payload)
        return path

    def latest_manifest(self, project_id: str) -> Mapping[str, Any] | None:
        manifests = sorted((self.project_dir(project_id) / "manifests").glob("*.json"))
        return json.loads(manifests[-1].read_text(encoding="utf-8")) if manifests else None

    def audit(self, project_id: str) -> Mapping[str, Any]:
        manifest = self.latest_manifest(project_id)
        if manifest is None:
            return {"valid": False, "project_id": project_id, "issues": ["No manifest exists."]}
        directory = self.project_dir(project_id)
        issues: list[str] = []
        for record in manifest.get("artifacts", ()):
            if not isinstance(record, Mapping) or not record.get("path"):
                continue
            path = directory / str(record["path"])
            if not path.is_file():
                issues.append(f"Missing archived artifact: {record['path']}")
            elif _sha256_file(path) != record.get("sha256"):
                issues.append(f"Hash mismatch: {record['path']}")
        return {
            "valid": not issues,
            "project_id": project_id,
            "manifest_sequence": manifest.get("sequence"),
            "issues": issues,
        }

    @staticmethod
    def _artifact_record(path: Path, project_directory: Path) -> Mapping[str, Any]:
        relative = path.relative_to(project_directory).as_posix()
        return {
            "path": relative,
            "kind": relative.split("/", 1)[0],
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }


def create_planning_project(goal: str, context: str = "") -> ResearchProject:
    """Create a durable conversational project without inventing a task graph."""

    goal = goal.strip()
    context = context.strip()
    if len(goal) < 12:
        raise ValueError("Describe the accelerator research goal in at least one complete sentence")
    now = _now()
    return ResearchProject(
        schema_version=PROJECT_SCHEMA_VERSION,
        id="project-" + content_hash({"goal": goal, "context": context, "created_at": now}, length=14),
        revision=1,
        created_at=now,
        updated_at=now,
        goal=goal,
        context=context,
        title=_title_from_goal(goal),
        goal_interpretation="The configured model will design task-specific research programs.",
        machine_summary="Machine requirements will be taken from the conversation and saved evidence.",
        success_definition=(),
        knowns=(),
        unknowns=(),
        assumptions=(),
        hypotheses=(),
        tasks=(),
        current_task_id=None,
        status="planning",
        next_action="Begin the first evidence-producing research turn.",
        claim_boundary="Numerical claims require a saved program and JuTrack evidence.",
        model="codex",
    )


def create_project(
    goal: str,
    context: str = "",
    *,
    use_ai: bool = True,
    plan_data: Mapping[str, Any] | None = None,
) -> ResearchProject:
    """Compatibility alias for clients that created projects before v0.7."""

    del use_ai, plan_data
    return create_planning_project(goal, context)


def project_ai_available() -> bool:
    """Legacy UI fallback; all new reasoning is owned by the Codex runtime."""

    return True


def revise_project(
    project: ResearchProject,
    scientist_message: str,
    *,
    use_ai: bool = True,
    plan_data: Mapping[str, Any] | None = None,
) -> ResearchProject:
    """Record a scientist revision without compiling it into a hidden workflow."""

    del use_ai, plan_data
    message = scientist_message.strip()
    if not message:
        raise ValueError("Scientist guidance cannot be empty")
    return replace(
        project,
        schema_version=PROJECT_SCHEMA_VERSION,
        revision=project.revision + 1,
        updated_at=_now(),
        status="researching",
        next_action="The configured model will incorporate the latest scientist guidance.",
        dialogue=project.dialogue
        + ({"timestamp": _now(), "role": "scientist", "content": message},),
    )


def _title_from_goal(goal: str) -> str:
    compact = " ".join(goal.split())
    return compact if len(compact) <= 100 else compact[:97].rstrip() + "..."


def _write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError(f"immutable project artifact conflict: {path}")


def _manifest_artifact_paths(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for root in sorted(directory.iterdir()):
        if not root.is_dir() or root.name == "manifests" or root.name in MANIFEST_LIVE_ROOTS:
            continue
        paths.extend(path for path in sorted(root.rglob("*")) if path.is_file())
    return paths


def _sha256_file(path: Path) -> str:
    for attempt in range(8):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.025 * (attempt + 1))
    raise AssertionError("unreachable")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(value: str, *, prefix: str) -> str:
    """Small compatibility helper used by old report fixtures."""

    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()
    return (normalized or prefix)[:64]
