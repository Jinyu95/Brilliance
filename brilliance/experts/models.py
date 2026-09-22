"""Typed contracts for installable accelerator-physics expert modules."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from ..artifacts import json_safe

EXPERT_MODULE_SCHEMA_VERSION = "1.0"
EXPERT_TOOL_API_VERSION = "3"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{1,95}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9_.-]+)?$")
_KNOWLEDGE_STATUSES = {"draft", "reviewed", "validated", "deprecated"}
_MODULE_STATUSES = {"draft", "reviewed", "validated", "deprecated"}
_KNOWLEDGE_KINDS = {
    "foundation", "method", "protocol", "reference", "troubleshooting", "example"
}
_TOOL_AUTHORITIES = {"provisional", "deterministic", "jutrack", "external"}


class ExpertModuleError(ValueError):
    """Raised when a local expert module violates its public contract."""


def _identifier(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ExpertModuleError(
            f"{label} must match {_IDENTIFIER.pattern!r}: {value!r}"
        )
    return normalized


def _contained_file(root: Path, value: Any, label: str) -> Path:
    relative = Path(str(value or "").strip())
    if not str(relative) or relative.is_absolute():
        raise ExpertModuleError(f"{label} must be a relative file path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ExpertModuleError(f"{label} escapes the expert module: {relative}") from exc
    if not resolved.is_file():
        raise ExpertModuleError(f"{label} does not exist: {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class KnowledgeAsset:
    id: str
    title: str
    path: Path
    kind: str
    status: str
    summary: str
    tags: tuple[str, ...] = ()
    priority: int = 50

    @classmethod
    def from_dict(cls, root: Path, value: Mapping[str, Any]) -> "KnowledgeAsset":
        asset_id = _identifier(value.get("id"), "knowledge id")
        kind = str(value.get("kind", "method")).strip().lower()
        status = str(value.get("status", "draft")).strip().lower()
        if kind not in _KNOWLEDGE_KINDS:
            raise ExpertModuleError(f"knowledge {asset_id!r} has unsupported kind {kind!r}")
        if status not in _KNOWLEDGE_STATUSES:
            raise ExpertModuleError(
                f"knowledge {asset_id!r} has unsupported status {status!r}"
            )
        priority = int(value.get("priority", 50))
        if not 0 <= priority <= 100:
            raise ExpertModuleError(f"knowledge {asset_id!r} priority must be in [0, 100]")
        title = str(value.get("title") or asset_id).strip()
        summary = str(value.get("summary", "")).strip()
        if not title or not summary:
            raise ExpertModuleError(f"knowledge {asset_id!r} needs title and summary")
        return cls(
            id=asset_id,
            title=title,
            path=_contained_file(root, value.get("path"), f"knowledge {asset_id} path"),
            kind=kind,
            status=status,
            summary=summary,
            tags=tuple(str(item).strip().lower() for item in value.get("tags", ()) if str(item).strip()),
            priority=priority,
        )

    def as_dict(self, root: Path) -> dict[str, Any]:
        return {
            **json_safe(asdict(self)),
            "path": self.path.relative_to(root).as_posix(),
        }


@dataclass(frozen=True, slots=True)
class ExpertTool:
    id: str
    title: str
    description: str
    entrypoint: str
    validator_entrypoint: str | None
    authority: str
    output: str
    parameter_schema: Mapping[str, Any] = field(default_factory=dict)
    requires_lattice: bool = False
    timeout_s: int = 600
    state_scope: str = ""
    stateful_parameters: tuple[str, ...] = ()
    inherit_from: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpertTool":
        tool_id = _identifier(value.get("id"), "tool id")
        entrypoint = str(value.get("entrypoint", "")).strip()
        valid_entrypoint = (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*", entrypoint)
            or re.fullmatch(r"file:[^:]+\.py:[A-Za-z_][A-Za-z0-9_]*", entrypoint)
        )
        if not valid_entrypoint:
            raise ExpertModuleError(
                f"tool {tool_id!r} entrypoint must be 'package.module:function' or "
                "'file:relative.py:function'"
            )
        authority = str(value.get("authority", "provisional")).strip().lower()
        if authority not in _TOOL_AUTHORITIES:
            raise ExpertModuleError(f"tool {tool_id!r} has unsupported authority {authority!r}")
        validator_entrypoint = str(value.get("validator_entrypoint", "")).strip() or None
        if validator_entrypoint is not None and not (
            re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*",
                validator_entrypoint,
            )
            or re.fullmatch(
                r"file:[^:]+\.py:[A-Za-z_][A-Za-z0-9_]*",
                validator_entrypoint,
            )
        ):
            raise ExpertModuleError(
                f"tool {tool_id!r} validator_entrypoint has an invalid format"
            )
        if authority in {"deterministic", "jutrack"} and validator_entrypoint is None:
            raise ExpertModuleError(
                f"tool {tool_id!r} with authority {authority!r} requires a separate "
                "validator_entrypoint"
            )
        if validator_entrypoint == entrypoint:
            raise ExpertModuleError(
                f"tool {tool_id!r} validator_entrypoint must differ from its execution entrypoint"
            )
        schema = value.get("parameters", {"type": "object", "additionalProperties": False})
        if not isinstance(schema, Mapping) or schema.get("type", "object") != "object":
            raise ExpertModuleError(f"tool {tool_id!r} parameters must be an object schema")
        timeout_s = int(value.get("timeout_s", 600))
        if timeout_s < 1:
            raise ExpertModuleError(f"tool {tool_id!r} timeout_s must be positive")
        title = str(value.get("title") or tool_id).strip()
        description = str(value.get("description", "")).strip()
        output = str(value.get("output", "")).strip()
        if not title or not description or not output:
            raise ExpertModuleError(f"tool {tool_id!r} needs title, description, and output")
        state_scope = str(value.get("state_scope", "")).strip().lower()
        stateful_parameters = tuple(
            str(item).strip() for item in value.get("stateful_parameters", ()) if str(item).strip()
        )
        inherit_from_value = value.get("inherit_from", {})
        if not isinstance(inherit_from_value, Mapping):
            raise ExpertModuleError(f"tool {tool_id!r} inherit_from must be an object")
        inherit_from = {
            str(name).strip(): str(scope).strip().lower()
            for name, scope in inherit_from_value.items()
        }
        property_names = set(schema.get("properties", {}))
        unknown_state = sorted(
            (set(stateful_parameters) | set(inherit_from)) - property_names
        )
        if unknown_state:
            raise ExpertModuleError(
                f"tool {tool_id!r} state parameters are not public arguments: {unknown_state}"
            )
        scope_pattern = re.compile(r"^[a-z][a-z0-9_.-]{2,127}$")
        scopes = ([state_scope] if state_scope else []) + list(inherit_from.values())
        if any(not scope_pattern.fullmatch(scope) for scope in scopes):
            raise ExpertModuleError(f"tool {tool_id!r} has an invalid research-state scope")
        return cls(
            id=tool_id,
            title=title,
            description=description,
            entrypoint=entrypoint,
            validator_entrypoint=validator_entrypoint,
            authority=authority,
            output=output,
            parameter_schema=dict(schema),
            requires_lattice=bool(value.get("requires_lattice", False)),
            timeout_s=timeout_s,
            state_scope=state_scope,
            stateful_parameters=stateful_parameters,
            inherit_from=inherit_from,
        )

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class ExpertModule:
    id: str
    version: str
    title: str
    description: str
    status: str
    root: Path
    manifest_path: Path
    domains: tuple[str, ...]
    authors: tuple[str, ...]
    knowledge: tuple[KnowledgeAsset, ...]
    tools: tuple[ExpertTool, ...]
    enabled: bool = True
    schema_version: str = EXPERT_MODULE_SCHEMA_VERSION

    @classmethod
    def load(cls, manifest_path: Path | str) -> "ExpertModule":
        manifest = Path(manifest_path).resolve()
        if not manifest.is_file():
            raise ExpertModuleError(f"expert manifest does not exist: {manifest}")
        try:
            with manifest.open("rb") as handle:
                value = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ExpertModuleError(f"cannot parse expert manifest {manifest}: {exc}") from exc
        if str(value.get("schema_version", "")) != EXPERT_MODULE_SCHEMA_VERSION:
            raise ExpertModuleError(
                f"expert manifest {manifest} requires schema_version "
                f"{EXPERT_MODULE_SCHEMA_VERSION!r}"
            )
        root = manifest.parent.resolve()
        module_id = _identifier(value.get("id"), "expert module id")
        version = str(value.get("version", "")).strip()
        if not _VERSION.fullmatch(version):
            raise ExpertModuleError(f"expert module {module_id!r} needs a semantic version")
        title = str(value.get("title", "")).strip()
        description = str(value.get("description", "")).strip()
        status = str(value.get("status", "draft")).strip().lower()
        if status not in _MODULE_STATUSES:
            raise ExpertModuleError(
                f"expert module {module_id!r} has unsupported status {status!r}"
            )
        if not title or not description:
            raise ExpertModuleError(f"expert module {module_id!r} needs title and description")
        knowledge_values = value.get("knowledge", [])
        tool_values = value.get("tools", [])
        if not isinstance(knowledge_values, list) or not isinstance(tool_values, list):
            raise ExpertModuleError("knowledge and tools must be arrays of TOML tables")
        knowledge = tuple(KnowledgeAsset.from_dict(root, item) for item in knowledge_values)
        tools = tuple(ExpertTool.from_dict(item) for item in tool_values)
        for tool in tools:
            for label, entrypoint in (
                ("entrypoint", tool.entrypoint),
                ("validator", tool.validator_entrypoint),
            ):
                if entrypoint is None:
                    continue
                if entrypoint.startswith("file:"):
                    _, relative, _ = entrypoint.split(":", 2)
                    _contained_file(root, relative, f"tool {tool.id} {label}")
                else:
                    module_name, _ = entrypoint.split(":", 1)
                    try:
                        specification = importlib.util.find_spec(module_name)
                    except ModuleNotFoundError:
                        specification = None
                    if specification is None:
                        raise ExpertModuleError(
                            f"tool {tool.id!r} Python module is not installed: {module_name}"
                        )
        for label, identifiers in {
            "knowledge": [item.id for item in knowledge],
            "tool": [item.id for item in tools],
        }.items():
            duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
            if duplicates:
                raise ExpertModuleError(
                    f"expert module {module_id!r} has duplicate {label} ids: {duplicates}"
                )
        return cls(
            id=module_id,
            version=version,
            title=title,
            description=description,
            status=status,
            root=root,
            manifest_path=manifest,
            domains=tuple(str(item).strip().lower() for item in value.get("domains", ()) if str(item).strip()),
            authors=tuple(str(item).strip() for item in value.get("authors", ()) if str(item).strip()),
            knowledge=knowledge,
            tools=tools,
            enabled=bool(value.get("enabled", True)),
        )

    def as_dict(self, *, include_assets: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "domains": list(self.domains),
            "authors": list(self.authors),
            "enabled": self.enabled,
            "manifest": str(self.manifest_path),
        }
        if include_assets:
            value.update({
                "knowledge": [item.as_dict(self.root) for item in self.knowledge],
                "tools": [item.as_dict() for item in self.tools],
            })
        return value
