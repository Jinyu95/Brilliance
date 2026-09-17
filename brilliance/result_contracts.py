"""Read installed result annotations for help and zero-execution preflight."""
from __future__ import annotations

import collections.abc
import inspect
import json
import types
import typing
from typing import Any


def closed_result(cls):
    """Declare a TypedDict's public keys exhaustive; values remain untouched."""
    cls.__brilliance_closed_result__ = True
    return cls


def _schema(annotation, seen):
    if annotation in (Any, inspect.Signature.empty):
        return {"type": "unknown"}
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        variants = [_schema(item, seen) for item in args if item is not type(None)]
        return {**variants[0], "nullable": True} if len(variants) == 1 else {"type": "unknown"}
    if origin in tuple(item for name in ("Required", "NotRequired", "Annotated") if (item := getattr(typing, name, None)) is not None):
        return _schema(args[0], seen) if args else {"type": "unknown"}
    if isinstance(annotation, type) and hasattr(annotation, "__required_keys__") and hasattr(annotation, "__optional_keys__"):
        if annotation in seen:
            return {"type": "object", "name": annotation.__name__, "closed": False}
        try:
            hints = typing.get_type_hints(annotation, include_extras=True)
        except (NameError, TypeError, AttributeError):
            return {"type": "unknown"}
        return {"type": "object", "name": annotation.__name__,
                "closed": annotation.__dict__.get("__brilliance_closed_result__", False) is True
                          or annotation.__dict__.get("__closed__", False) is True,
                "required": sorted(annotation.__required_keys__),
                "properties": {key: _schema(value, {*seen, annotation}) for key, value in hints.items()}}
    if origin in (list, tuple, collections.abc.Sequence, collections.abc.Iterable):
        return {"type": "array", "items": _schema(args[0], seen) if args else {"type": "unknown"}}
    if origin in (dict, collections.abc.Mapping, collections.abc.MutableMapping) or annotation is dict:
        return {"type": "object", "closed": False}
    scalar = {str: "string", bool: "boolean", int: "number", float: "number", type(None): "null"}
    return {"type": scalar.get(annotation, "unknown")}


def return_contract(target) -> dict[str, Any] | None:
    """Describe a callable's installed return type without executing the callable."""
    try:
        target = inspect.unwrap(target)
        annotation = typing.get_type_hints(target, include_extras=True).get("return", Any)
        schema = _schema(annotation, set())
    except (NameError, TypeError, AttributeError, ValueError):
        return None
    return schema if schema["type"] != "unknown" else None


def parameter_contracts(target) -> dict[str, dict[str, Any]]:
    """Read explicitly annotated inputs without evaluating default values or code."""
    try:
        target = inspect.unwrap(target)
        parameters = inspect.signature(target).parameters
        hints = typing.get_type_hints(target, include_extras=True)
    except (NameError, TypeError, AttributeError, ValueError):
        return {}
    result = {}
    for name, parameter in parameters.items():
        annotation = hints.get(name, Any)
        unpack = (parameter.kind is inspect.Parameter.VAR_KEYWORD
                  and typing.get_origin(annotation) is getattr(typing, "Unpack", None))
        schema = _schema(typing.get_args(annotation)[0] if unpack else annotation, set())
        if schema["type"] != "unknown":
            result[name] = {**schema, "unpack": True} if unpack else schema
    return result


def _render_schema(schema, path, *, max_depth):
    lines = []

    def visit(item, path, depth):
        kind = item.get("type", "unknown")
        suffix = " or null" if item.get("nullable") else ""
        properties = item.get("properties", {})
        if kind == "object" and properties:
            lines.append(f"{path}: {'closed' if item.get('closed') else 'extensible'} mapping; keys: {', '.join(properties)}{suffix}")
            nullable = [key for key, value in properties.items() if value.get("nullable")]
            if nullable:
                lines.append(f"{path}: nullable values (not zero): {', '.join(nullable)}")
            if depth < max_depth:
                for key, value in properties.items():
                    if value.get("type") in {"object", "array"}:
                        visit(value, f"{path}[{key!r}]", depth + 1)
        elif kind == "array":
            lines.append(f"{path}: list{suffix}")
            if depth < max_depth:
                visit(item.get("items", {}), path + "[i]", depth + 1)
        else:
            lines.append(f"{path}: {kind}{suffix}")
    visit(schema, path, 0)
    return "\n".join(lines)


def render_return_contract(target, *, max_depth: int = 3) -> str:
    """Compact discoverable keys, including nested records and list elements."""
    schema = return_contract(target)
    if not schema:
        return ""
    return "Installed return contract (annotations; no calculation executed):\n" + _render_schema(
        schema, "result", max_depth=max_depth)


def render_parameter_contracts(target, *, max_depth: int = 3) -> str:
    """Expose structured argument keys; scalar parameters already have signatures."""
    lines = []
    for name, schema in parameter_contracts(target).items():
        if schema.get("properties") or schema.get("items", {}).get("properties"):
            lines.append(_render_schema(schema, f"argument {name!r}", max_depth=max_depth))
    if not lines:
        return ""
    return "Installed input contracts (annotations; no calculation executed):\n" + "\n".join(lines)


def compact_return_contracts(contracts, *, max_bytes: int = 8192):
    """Bound edit/preflight context; retain exact declared keys, never sample them."""
    def compact(item, depth):
        properties = item.get("properties", {})
        if properties:
            result = {"closed": bool(item.get("closed")), "keys": list(properties)}
            nullable = [key for key, value in properties.items() if value.get("nullable")]
            if nullable:
                result["nullable_values"] = nullable
            if depth:
                children = {key: compact(value, depth - 1) for key, value in properties.items()
                            if value.get("properties") or value.get("items")}
                if children:
                    result["nested"] = children
            return result
        if item.get("type") == "array":
            return {"items": compact(item.get("items", {}), depth)}
        return {"type": item.get("type", "unknown")}

    selected, omitted = {}, []
    for symbol, schema in sorted(contracts.items()):
        for depth in range(3, -1, -1):
            candidate = {**selected, symbol: compact(schema, depth)}
            if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= max_bytes:
                selected = candidate
                break
        else:
            omitted.append(symbol)
    return selected, omitted
