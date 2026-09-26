"""Compiler and developer checks for the simple ``EXPERT.md`` authoring format."""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

from .execution import ExpertToolContext, execute_expert_tool
from .loading import load_expert_source
from .models import (
    _MODULE_STATUSES,
    _VERSION,
    EXPERT_MODULE_SCHEMA_VERSION,
    ExpertModule,
    ExpertModuleError,
    ExpertTool,
    KnowledgeAsset,
    _identifier,
)

_FRONT_MATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"(?m)^#{1,4}\s+(.+?)\s*$")
_REQUIRED_SECTIONS = {
    "applicability": "State where the method is valid and where it is not.",
    "required inputs": "List physical inputs and units.",
    "method": "Give the ordered scientific procedure.",
    "gates and tolerances": "Define quantitative acceptance criteria.",
    "failure and recovery": "Distinguish physics, numerical, and software failures.",
    "required evidence": "Define required tables, plots, and provenance.",
    "sources": "Cite primary sources and identify inferences.",
}


def load_module_path(path: Path | str) -> ExpertModule:
    """Load either a simple expert directory/file or a legacy TOML manifest."""

    candidate = Path(path).resolve()
    if candidate.is_dir():
        legacy = candidate / "expert.toml"
        simple = candidate / "EXPERT.md"
        if legacy.is_file():
            return ExpertModule.load(legacy)
        if simple.is_file():
            return compile_simple_module(simple)
        raise ExpertModuleError(
            f"expert directory needs EXPERT.md (preferred) or expert.toml: {candidate}"
        )
    if candidate.name == "expert.toml":
        return ExpertModule.load(candidate)
    if candidate.name.casefold() == "expert.md":
        return compile_simple_module(candidate)
    raise ExpertModuleError(f"unsupported expert module source: {candidate}")


def compile_simple_module(path: Path | str) -> ExpertModule:
    """Compile human-facing Markdown and decorated functions into runtime contracts."""

    expert_path = Path(path).resolve()
    if not expert_path.is_file() or expert_path.name.casefold() != "expert.md":
        raise ExpertModuleError(f"simple expert module requires EXPERT.md: {expert_path}")
    root = expert_path.parent
    raw = expert_path.read_text(encoding="utf-8-sig")
    metadata, body = _parse_expert_markdown(raw, expert_path)
    module_id = _identifier(
        metadata.get("id") or f"local.{_slug(root.name)}",
        "expert module id",
    )
    version = str(metadata.get("version", "0.1.0")).strip()
    if not _VERSION.fullmatch(version):
        raise ExpertModuleError(f"expert module {module_id!r} needs a semantic version")
    status = str(metadata.get("status", "draft")).strip().lower()
    if status not in _MODULE_STATUSES:
        raise ExpertModuleError(
            f"expert module {module_id!r} has unsupported status {status!r}"
        )
    title = str(metadata.get("title") or _markdown_title(body) or root.name).strip()
    summary = str(metadata.get("summary") or _first_paragraph(body)).strip()
    if not title or not summary:
        raise ExpertModuleError("EXPERT.md needs a title and a one-line summary")
    domains = tuple(_list_value(metadata.get("domains", ())))
    authors = tuple(_list_value(metadata.get("authors", ())))
    tags = tuple(item.lower() for item in _list_value(metadata.get("tags", domains)))
    knowledge_status = str(metadata.get("knowledge_status", status)).strip().lower()
    knowledge = KnowledgeAsset.from_dict(root, {
        "id": f"{module_id}.knowledge",
        "title": title,
        "path": "EXPERT.md",
        "kind": str(metadata.get("kind", "protocol")),
        "status": knowledge_status,
        "summary": summary,
        "tags": tags,
        "priority": int(metadata.get("priority", 80)),
        "context_references": metadata.get("context_references", ()),
    })
    enabled = _boolean(metadata.get("enabled", True))
    # Disabled/deprecated modules must not execute Python during discovery.
    tools = _compile_tools(root, module_id) if enabled and status != "deprecated" else ()
    return ExpertModule(
        id=module_id,
        version=version,
        title=title,
        description=summary,
        status=status,
        root=root,
        manifest_path=expert_path,
        domains=tuple(item.lower() for item in domains),
        authors=authors,
        knowledge=(knowledge,),
        tools=tools,
        enabled=enabled,
        schema_version=EXPERT_MODULE_SCHEMA_VERSION,
    )


def doctor_module(path: Path | str) -> dict[str, Any]:
    """Return scientist-facing completeness checks without running calculations."""

    module = load_module_path(path)
    warnings: list[str] = []
    errors: list[str] = []
    if module.manifest_path.name.casefold() == "expert.md":
        body = module.manifest_path.read_text(encoding="utf-8-sig")
        headings = {item.strip().lower() for item in _HEADING.findall(body)}
        for heading, guidance in _REQUIRED_SECTIONS.items():
            if heading not in headings:
                warnings.append(f"Missing section '{heading.title()}': {guidance}")
    if not module.authors:
        warnings.append("No responsible expert is named yet.")
    if module.status in {"reviewed", "validated"}:
        warnings.append(
            f"Module status is {module.status!r}; confirm that human review evidence is archived."
        )
    examples = _example_paths(module.root)
    if module.tools and not examples:
        warnings.append("Tools are present but examples/ contains no executable JSON cases.")
    return {
        "passed": not errors,
        "module": module.as_dict(include_assets=True),
        "errors": errors,
        "warnings": warnings,
        "summary": (
            f"{module.id}: {len(module.knowledge)} knowledge asset(s), "
            f"{len(module.tools)} tool(s), {len(examples)} example case(s)."
        ),
    }


def test_module(path: Path | str) -> dict[str, Any]:
    """Execute optional JSON example cases through the real expert boundary."""

    from .registry import ExpertRegistry

    module = load_module_path(path)
    registry = ExpertRegistry((module,))
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="brilliance-expert-") as temporary:
        temp_root = Path(temporary)
        project = SimpleNamespace(
            id="expert-self-test",
            active_lattice_id=None,
            lattice_history=(),
        )
        context = ExpertToolContext(
            project_root=temp_root,
            project_directory=temp_root / "expert-self-test",
            workspace=temp_root / "expert-self-test" / "workspace",
            project=project,
            environment=SimpleNamespace(),
        )
        for example in _example_paths(module.root):
            try:
                value = json.loads(example.read_text(encoding="utf-8"))
                if not isinstance(value, Mapping):
                    raise ValueError("example must contain one JSON object")
                tool_id = _qualified_tool_id(module.id, str(value.get("tool", "")))
                _, declared_tool = registry.tool(tool_id)
                if declared_tool.requires_lattice:
                    records.append({
                        "example": example.name,
                        "tool": tool_id,
                        "status": "skipped",
                        "detail": "This example requires an accepted project lattice.",
                    })
                    continue
                execution = execute_expert_tool(
                    registry,
                    tool_id,
                    value.get("arguments", {}),
                    context=context,
                )
                expected_pass = bool(value.get("expect_validation_passed", True))
                actual_pass = execution["validation"]["passed"] is True
                records.append({
                    "example": example.name,
                    "tool": tool_id,
                    "status": "passed" if actual_pass == expected_pass else "failed",
                    "validation": execution["validation"],
                })
            except Exception as exc:  # developer command must report each case
                records.append({
                    "example": example.name,
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
    failures = [item for item in records if item["status"] == "failed"]
    return {
        "passed": not failures,
        "module_id": module.id,
        "tested": sum(item["status"] != "skipped" for item in records),
        "skipped": sum(item["status"] == "skipped" for item in records),
        "records": records,
        "warnings": ([] if records else ["No examples/*.json cases were found."]),
    }


def package_module(path: Path | str, destination: Path | str | None = None) -> Path:
    """Create a portable source ZIP without caches or generated runtime state."""

    module = load_module_path(path)
    root = module.root
    target = Path(destination).resolve() if destination else root.parent / f"{root.name}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    excluded_parts = {"__pycache__", ".pytest_cache", ".ruff_cache", ".brilliance"}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(root.rglob("*")):
            if (
                not source.is_file()
                or source.resolve() == target
                or any(part in excluded_parts for part in source.parts)
            ):
                continue
            archive.write(source, arcname=f"{root.name}/{source.relative_to(root).as_posix()}")
    return target


def _compile_tools(root: Path, module_id: str) -> tuple[ExpertTool, ...]:
    source = root / "tools.py"
    if not source.is_file():
        return ()
    imported = _load_tools_module(source, module_id)
    declared: list[tuple[str, Mapping[str, Any]]] = []
    validators: dict[str, str] = {}
    for name, value in vars(imported).items():
        tool_metadata = getattr(value, "__brilliance_tool__", None)
        if isinstance(tool_metadata, Mapping):
            declared.append((name, tool_metadata))
        validator_metadata = getattr(value, "__brilliance_validator__", None)
        if isinstance(validator_metadata, Mapping):
            target = _qualified_tool_id(module_id, str(validator_metadata["for_tool"]))
            if target in validators:
                raise ExpertModuleError(f"multiple validators target expert tool {target!r}")
            validators[target] = name
    tools: list[ExpertTool] = []
    aliases: dict[str, str] = {}
    for function_name, metadata in declared:
        requested_id = str(metadata.get("id") or function_name)
        tool_id = _qualified_tool_id(module_id, requested_id)
        aliases[function_name] = tool_id
        aliases[f"{module_id}.{function_name}"] = tool_id
        aliases[requested_id] = tool_id
        aliases[tool_id] = tool_id
    normalized_validators: dict[str, str] = {}
    for target, function_name in validators.items():
        normalized_validators[aliases.get(target, target)] = function_name
    for function_name, metadata in declared:
        requested_id = str(metadata.get("id") or function_name)
        tool_id = aliases[requested_id]
        title = str(metadata.get("title") or function_name.replace("_", " ").title())
        description = str(metadata.get("description") or f"Expert tool: {title}.").splitlines()[0]
        output = str(metadata.get("output") or f"Structured result from {title}.")
        validator_name = normalized_validators.get(tool_id)
        tools.append(ExpertTool.from_dict({
            "id": tool_id,
            "title": title,
            "description": description,
            "entrypoint": f"file:tools.py:{function_name}",
            "validator_entrypoint": (
                f"file:tools.py:{validator_name}" if validator_name else ""
            ),
            "authority": metadata.get("authority", "provisional"),
            "output": output,
            "requires_lattice": metadata.get("requires_lattice", False),
            "timeout_s": metadata.get("timeout_s", 600),
            "parameters": metadata.get("parameter_schema", {}),
            "state_scope": metadata.get("state_scope", ""),
            "stateful_parameters": metadata.get("stateful_parameters", ()),
            "inherit_from": metadata.get("inherit_from", {}),
        }))
    declared_ids = {item.id for item in tools}
    unused = sorted(set(normalized_validators) - declared_ids)
    if unused:
        raise ExpertModuleError(f"validators reference unknown expert tools: {unused}")
    return tuple(tools)


def _load_tools_module(source: Path, module_id: str) -> ModuleType:
    return load_expert_source(source.parent, source)


def _parse_expert_markdown(text: str, path: Path) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER.search(text)
    if match is None:
        return {}, text
    metadata: dict[str, Any] = {}
    for line_number, raw_line in enumerate(match.group(1).splitlines(), start=2):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ExpertModuleError(
                f"{path.name} front matter line {line_number} must be 'key: value'"
            )
        key, raw_value = line.split(":", 1)
        normalized_key = key.strip().lower().replace("-", "_")
        if not normalized_key:
            raise ExpertModuleError(f"{path.name} front matter contains an empty key")
        metadata[normalized_key] = _front_matter_value(raw_value.strip())
    return metadata, text[match.end():]


def _front_matter_value(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        return _list_value(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [
        item.strip().strip('"\'')
        for item in text.split(",")
        if item.strip().strip('"\'')
    ]


def _markdown_title(body: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", body)
    return match.group(1).strip() if match else ""


def _first_paragraph(body: str) -> str:
    cleaned = re.sub(r"(?m)^#{1,6}\s+.+$", "", body).strip()
    paragraphs = re.split(r"\r?\n\s*\r?\n", cleaned)
    return " ".join(paragraphs[0].split()) if paragraphs else ""


def _qualified_tool_id(module_id: str, value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ExpertModuleError("expert tool id must not be empty")
    if normalized.startswith(module_id + "."):
        return normalized
    return f"{module_id}.{normalized}"


def _slug(value: str) -> str:
    rendered = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_").lower()
    return rendered or "module"


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _example_paths(root: Path) -> tuple[Path, ...]:
    directory = root / "examples"
    return tuple(sorted(directory.glob("*.json"))) if directory.is_dir() else ()


__all__ = [
    "compile_simple_module",
    "doctor_module",
    "load_module_path",
    "package_module",
    "test_module",
]
