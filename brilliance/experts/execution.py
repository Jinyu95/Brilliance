"""Typed execution boundary for tools contributed by expert modules."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..artifacts import canonical_json, content_hash, json_safe
from ..lattice import LatticeIR
from .models import EXPERT_TOOL_API_VERSION, ExpertModule, ExpertModuleError, ExpertTool
from .registry import ExpertRegistry


@dataclass(frozen=True, slots=True)
class ExpertToolContext:
    """Stable host services passed to a locally installed expert tool."""

    project_root: Path
    project_directory: Path
    workspace: Path
    project: Any
    environment: Any
    compute_job_id: str | None = None
    cancellation_requested: Callable[[], bool] | None = None

    @property
    def project_id(self) -> str:
        return str(self.project.id)

    def active_lattice(self) -> LatticeIR | None:
        lattice_id = str(getattr(self.project, "active_lattice_id", "") or "")
        for item in reversed(tuple(getattr(self.project, "lattice_history", ()) or ())):
            if str(item.get("lattice_id", "")) == lattice_id:
                return LatticeIR.from_dict(item)
        return None

    def artifact_path(self, relative: str) -> Path:
        root = (self.workspace / "artifacts").resolve()
        destination = (root / relative).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise ValueError("expert tool artifact path escapes workspace/artifacts") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def is_cancellation_requested(self) -> bool:
        """Return true only after an explicit scientist cancellation request."""

        return bool(self.cancellation_requested and self.cancellation_requested())

    def raise_if_cancelled(self) -> None:
        """Stop a managed calculation at a safe numerical boundary."""

        if self.is_cancellation_requested():
            raise ExpertToolCancelled(
                f"managed expert calculation {self.compute_job_id or ''} was cancelled"
            )

    def research_contract(
        self,
        scope: str,
        *,
        subject_id: str | None = None,
        include_inactive: bool = False,
    ) -> Mapping[str, Any] | None:
        """Return the latest opaque contract in a domain-neutral state scope."""

        from ..research_state import ResearchStateStore

        return ResearchStateStore(self.project_directory).latest(
            scope,
            subject_id=subject_id,
            include_inactive=include_inactive,
        )

    def research_contracts(
        self,
        scope: str,
        *,
        subject_id: str | None = None,
        include_inactive: bool = False,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return every sibling contract in one opaque research scope."""

        from ..research_state import ResearchStateStore

        return ResearchStateStore(self.project_directory).find(
            scope,
            subject_id=subject_id,
            include_inactive=include_inactive,
        )


class ExpertToolCancelled(BaseException):
    """Cooperative stop signal that ordinary numerical exception handlers do not swallow."""


def execute_expert_tool(
    registry: ExpertRegistry,
    tool_id: str,
    arguments: Mapping[str, Any],
    *,
    context: ExpertToolContext,
    supplied_arguments_override: Mapping[str, Any] | None = None,
    argument_resolution_override: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    module, tool = registry.tool(tool_id)
    if argument_resolution_override is None:
        supplied_arguments = _validate_arguments(
            tool,
            arguments,
            apply_defaults=False,
            allow_missing_required=True,
        )
        normalized, argument_resolution = resolve_expert_arguments(
            tool, supplied_arguments, context=context
        )
    else:
        supplied_arguments = _validate_arguments(
            tool,
            supplied_arguments_override or {},
            apply_defaults=False,
            allow_missing_required=True,
        )
        normalized = _validate_arguments(tool, arguments, apply_defaults=True)
        argument_resolution = {
            str(name): str(source)
            for name, source in argument_resolution_override.items()
        }
        if set(argument_resolution) != set(normalized):
            raise ExpertModuleError(
                f"expert tool {tool.id!r} received an incomplete resolved-argument receipt"
            )
    if tool.requires_lattice and context.active_lattice() is None:
        raise ExpertModuleError(
            f"expert tool {tool.id!r} requires an accepted active lattice"
        )
    handler = _load_callable(
        module,
        tool.entrypoint,
        tool_id=tool.id,
        parameter_count=2,
        role="tool",
    )
    result = handler(context, normalized)
    if not isinstance(result, Mapping):
        raise ExpertModuleError(
            f"expert tool {tool.id!r} must return one JSON-compatible mapping"
        )
    safe_result = json_safe(result)
    validation = _validate_result(
        module,
        tool,
        context=context,
        arguments=normalized,
        result=safe_result,
    )
    validation_passed = validation["passed"] is True
    effective_authority = (
        tool.authority
        if module.status in {"reviewed", "validated"} and validation_passed
        else "provisional"
    )
    state_contract = _state_contract(tool, normalized, safe_result)
    identity = {
        "api_version": EXPERT_TOOL_API_VERSION,
        "module_id": module.id,
        "module_version": module.version,
        "tool_id": tool.id,
        "authority": effective_authority,
        "declared_authority": tool.authority,
        "module_status": module.status,
        "project_id": context.project_id,
        "active_lattice_id": getattr(context.project, "active_lattice_id", None),
        "supplied_arguments": supplied_arguments,
        "arguments": normalized,
        "argument_resolution": argument_resolution,
        "tool_implementation_hash": _tool_implementation_hash(module, tool),
        "state_contract": state_contract,
        "result": safe_result,
        "validation": validation,
    }
    return {
        "schema_version": "1.1",
        "execution_id": "expert-execution-" + content_hash(identity, length=20),
        **identity,
    }


def _validate_result(
    module: ExpertModule,
    tool: ExpertTool,
    *,
    context: ExpertToolContext,
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if tool.validator_entrypoint is None:
        return {
            "passed": False,
            "validator": None,
            "issues": [
                "No independent validator is registered; this execution is "
                "provisional and cannot be accepted as numerical evidence."
            ],
        }
    validator = _load_callable(
        module,
        tool.validator_entrypoint,
        tool_id=tool.id,
        parameter_count=3,
        role="validator",
    )
    verdict = validator(context, arguments, result)
    if not isinstance(verdict, Mapping):
        raise ExpertModuleError(
            f"expert tool {tool.id!r} validator must return one JSON-compatible mapping"
        )
    safe_verdict = json_safe(verdict)
    if not isinstance(safe_verdict, Mapping) or not isinstance(
        safe_verdict.get("passed"), bool
    ):
        raise ExpertModuleError(
            f"expert tool {tool.id!r} validator verdict needs boolean 'passed'"
        )
    rendered = dict(safe_verdict)
    if "scientific_passed" not in rendered and isinstance(result.get("passed"), bool):
        rendered["scientific_passed"] = bool(result.get("passed"))
    return rendered


def _load_callable(
    module: ExpertModule,
    entrypoint: str,
    *,
    tool_id: str,
    parameter_count: int,
    role: str,
) -> Callable[..., Any]:
    if entrypoint.startswith("file:"):
        _, relative, function_name = entrypoint.split(":", 2)
        source = (module.root / relative).resolve()
        try:
            source.relative_to(module.root)
        except ValueError as exc:
            raise ExpertModuleError(
                f"expert tool {tool_id!r} {role} source escapes its module"
            ) from exc
        if not source.is_file():
            raise ExpertModuleError(f"expert tool {role} source does not exist: {source}")
        from .loading import load_expert_source

        imported = load_expert_source(module.root, source)
    else:
        module_name, function_name = entrypoint.split(":", 1)
        imported = importlib.import_module(module_name)
    callable_value = getattr(imported, function_name, None)
    if not callable(callable_value):
        raise ExpertModuleError(
            f"expert tool {role} entrypoint is not callable: {entrypoint}"
        )
    signature = inspect.signature(callable_value)
    if len(signature.parameters) != parameter_count:
        expected = (
            "(context, arguments)"
            if parameter_count == 2
            else "(context, arguments, result)"
        )
        raise ExpertModuleError(
            f"expert tool {tool_id!r} {role} must accept exactly {expected}"
        )
    return callable_value


def _validate_arguments(
    tool: ExpertTool,
    arguments: Mapping[str, Any],
    *,
    apply_defaults: bool = True,
    allow_missing_required: bool = False,
) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise ExpertModuleError(f"expert tool {tool.id!r} arguments must be an object")
    schema = dict(tool.parameter_schema)
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ExpertModuleError(f"expert tool {tool.id!r} properties schema must be an object")
    required = {str(item) for item in schema.get("required", ())}
    missing = sorted(required - set(arguments))
    if missing and not allow_missing_required:
        raise ExpertModuleError(
            f"expert tool {tool.id!r} is missing required arguments: {missing}"
        )
    if schema.get("additionalProperties", True) is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ExpertModuleError(
                f"expert tool {tool.id!r} received unknown arguments: {unknown}"
            )
    normalized: dict[str, Any] = {}
    for name, value in arguments.items():
        rule = properties.get(name, {})
        if not isinstance(rule, Mapping):
            raise ExpertModuleError(f"expert tool {tool.id!r} schema for {name!r} is invalid")
        _validate_value(tool.id, str(name), value, rule)
        normalized[str(name)] = json_safe(value)
    if apply_defaults:
        for name, rule in properties.items():
            if name in normalized or not isinstance(rule, Mapping) or "default" not in rule:
                continue
            value = rule["default"]
            _validate_value(tool.id, str(name), value, rule)
            normalized[str(name)] = json_safe(value)
    return normalized


def resolve_expert_arguments(
    tool: ExpertTool,
    arguments: Mapping[str, Any],
    *,
    context: ExpertToolContext,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve explicit > inherited > default arguments and report every source."""

    from ..research_state import ResearchStateStore, infer_subject_id

    explicit = _validate_arguments(
        tool,
        arguments,
        apply_defaults=False,
        allow_missing_required=True,
    )
    merged = dict(explicit)
    sources = {name: "scientist_or_agent" for name in explicit}
    store = ResearchStateStore(context.project_directory)
    subject_id = infer_subject_id(context, explicit)

    def inherit(name: str, scope: str) -> None:
        if name in merged:
            return
        contract = store.latest(scope, subject_id=subject_id)
        values = contract.get("arguments", {}) if isinstance(contract, Mapping) else {}
        if isinstance(values, Mapping) and name in values:
            merged[name] = json_safe(values[name])
            sources[name] = f"research_state:{scope}"

    if tool.state_scope:
        for name in tool.stateful_parameters:
            inherit(name, tool.state_scope)
    for name, scope in tool.inherit_from.items():
        inherit(name, scope)

    normalized = _validate_arguments(tool, merged, apply_defaults=True)
    for name in normalized:
        sources.setdefault(name, "tool_default")
    return normalized, sources


def _state_contract(
    tool: ExpertTool,
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    declared = result.get("research_contract")
    if isinstance(declared, Mapping):
        contract = dict(declared)
        contract.setdefault("scope", tool.state_scope)
        contract.setdefault("arguments", dict(arguments))
        return contract
    if not tool.state_scope:
        return None
    candidate = result.get("candidate", {})
    source = result.get("source_parent", {})
    subject = (
        {"kind": "research_subject", "id": candidate.get("lattice_id")}
        if isinstance(candidate, Mapping) and candidate.get("lattice_id")
        else {}
    )
    return {
        "scope": tool.state_scope,
        "subject": subject,
        "parent_subject": dict(source) if isinstance(source, Mapping) else {},
        "arguments": {
            name: arguments[name]
            for name in tool.stateful_parameters
            if name in arguments
        },
        "constraints": [],
    }


def _tool_implementation_hash(module: ExpertModule, tool: ExpertTool) -> str:
    digest = hashlib.sha256()
    digest.update(module.id.encode("utf-8"))
    digest.update(module.version.encode("utf-8"))
    digest.update(canonical_json(tool.as_dict()).encode("utf-8"))
    sources: set[Path] = set()
    for entrypoint in (tool.entrypoint, tool.validator_entrypoint):
        if not entrypoint or not entrypoint.startswith("file:"):
            continue
        _, relative, _ = entrypoint.split(":", 2)
        sources.add((module.root / relative).resolve())
    # Expert tools may delegate pure calculations to sibling Python modules.
    # Hash every source file in the package so immutable execution receipts
    # cannot claim the same implementation after helper logic changes.
    sources.update(path.resolve() for path in module.root.rglob("*.py") if path.is_file())
    for source in sorted(sources, key=lambda path: path.relative_to(module.root).as_posix()):
        digest.update(source.relative_to(module.root).as_posix().encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()[:24]


def _validate_value(tool_id: str, name: str, value: Any, rule: Mapping[str, Any]) -> None:
    if value is None:
        if not rule or rule.get("nullable") is True:
            return
        raise ExpertModuleError(
            f"expert tool {tool_id!r} argument {name!r} must not be null"
        )
    expected = str(rule.get("type", ""))
    valid = {
        "": True,
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, (list, tuple)),
        "object": isinstance(value, Mapping),
    }.get(expected, False)
    if not valid:
        raise ExpertModuleError(
            f"expert tool {tool_id!r} argument {name!r} must have type {expected!r}"
        )
    if "enum" in rule and value not in rule["enum"]:
        raise ExpertModuleError(
            f"expert tool {tool_id!r} argument {name!r} must be one of {rule['enum']}"
        )
    if expected in {"number", "integer"}:
        number = float(value)
        if not math.isfinite(number):
            raise ExpertModuleError(
                f"expert tool {tool_id!r} argument {name!r} must be finite"
            )
        if "minimum" in rule and number < float(rule["minimum"]):
            raise ExpertModuleError(
                f"expert tool {tool_id!r} argument {name!r} is below its minimum"
            )
        if "maximum" in rule and number > float(rule["maximum"]):
            raise ExpertModuleError(
                f"expert tool {tool_id!r} argument {name!r} exceeds its maximum"
            )
    if expected == "string":
        if "minLength" in rule and len(value) < int(rule["minLength"]):
            raise ExpertModuleError(
                f"expert tool {tool_id!r} argument {name!r} is shorter than its minimum length"
            )
        if "maxLength" in rule and len(value) > int(rule["maxLength"]):
            raise ExpertModuleError(
                f"expert tool {tool_id!r} argument {name!r} exceeds its maximum length"
            )
    if expected == "array":
        if "minItems" in rule and len(value) < int(rule["minItems"]):
            raise ExpertModuleError(
                f"expert tool {tool_id!r} argument {name!r} has too few items"
            )
        if "maxItems" in rule and len(value) > int(rule["maxItems"]):
            raise ExpertModuleError(
                f"expert tool {tool_id!r} argument {name!r} has too many items"
            )
        item_rule = rule.get("items", {})
        if item_rule and not isinstance(item_rule, Mapping):
            raise ExpertModuleError(
                f"expert tool {tool_id!r} schema for {name!r}.items is invalid"
            )
        if isinstance(item_rule, Mapping):
            for index, item in enumerate(value):
                _validate_value(tool_id, f"{name}[{index}]", item, item_rule)
    if expected == "object":
        properties = rule.get("properties", {})
        if properties and not isinstance(properties, Mapping):
            raise ExpertModuleError(
                f"expert tool {tool_id!r} schema for {name!r}.properties is invalid"
            )
        properties = properties if isinstance(properties, Mapping) else {}
        required = {str(item) for item in rule.get("required", ())}
        missing = sorted(required - set(value))
        if missing:
            rendered = ", ".join(f"{name}.{item}" for item in missing)
            raise ExpertModuleError(
                f"expert tool {tool_id!r} is missing required arguments: {rendered}"
            )
        if rule.get("additionalProperties", True) is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                rendered = ", ".join(f"{name}.{item}" for item in unknown)
                raise ExpertModuleError(
                    f"expert tool {tool_id!r} received unknown arguments: {rendered}"
                )
        for child_name, child_value in value.items():
            child_rule = properties.get(child_name, {})
            if child_rule and not isinstance(child_rule, Mapping):
                raise ExpertModuleError(
                    f"expert tool {tool_id!r} schema for {name}.{child_name!r} is invalid"
                )
            if isinstance(child_rule, Mapping):
                _validate_value(
                    tool_id,
                    f"{name}.{child_name}",
                    child_value,
                    child_rule,
                )
