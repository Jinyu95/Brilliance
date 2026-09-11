"""Validate model messages before any tools run; keep rejected output private."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REJECTION_MESSAGES = {
    "output_token_limit": "The provider reached its output-token limit before completing the response.",
    "public_prose_too_long": "The model repeatedly exceeded the public-response length limit.",
    "incomplete_provider_response": "The provider repeatedly returned an incomplete response.",
    "malformed_tool_text": "The model repeatedly returned incomplete or malformed tool syntax.",
    "invalid_tool_arguments": "The model repeatedly returned tool arguments that do not match the advertised schema.",
    "invalid_tool_request": "The model repeatedly requested an unavailable or malformed tool call.",
}


def safe_response_metadata(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    """Allow only stop/budget facts and numeric usage, never provider payloads."""
    result = {}
    for key in ("requested_max_tokens", "next_max_tokens", "max_tokens_ceiling"):
        value = diagnostic.get(key)
        if type(value) is int and value >= 0:
            result[key] = value
    reason = diagnostic.get("finish_reason")
    if reason in ("length", "stop", "tool_calls", "content_filter", "stream_interrupted"):
        result["finish_reason"] = reason
    usage = diagnostic.get("usage")
    if isinstance(usage, Mapping):
        counts = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
            value = usage.get(key)
            if type(value) is int and value >= 0:
                counts[key] = value
        details = usage.get("completion_tokens_details")
        if isinstance(details, Mapping):
            value = details.get("reasoning_tokens")
            if type(value) is int and value >= 0:
                counts["reasoning_tokens"] = value
        result["usage"] = counts
    return result


def validate_tool_batch(
    calls: Any, tools: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Validate the entire batch before executing its first (possibly mutating) call.

    This checks the small schema vocabulary used by our advertised tool boundary.
    A program's free-form ``arguments`` object remains free-form; scientific input
    validation belongs to the program/API, not the model-protocol recovery loop.
    """
    if not isinstance(calls, list):
        return {"reason": "invalid_tool_request", "error": "tool_calls must be an array."}
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    seen: set[str] = set()
    for call in calls:
        if not isinstance(call, Mapping) or call.get("type", "function") != "function":
            return {"reason": "invalid_tool_request", "error": "Each tool call must be a function object."}
        identity = call.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            return {"reason": "invalid_tool_request", "error": "Tool calls need nonempty unique IDs."}
        seen.add(identity)
        function = call.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or name not in schemas:
            return {"reason": "invalid_tool_request", "error": "Choose a currently advertised function."}
        schema = schemas[name]
        try:
            raw = function.get("arguments", "{}")
            if not isinstance(raw, str):
                raise ValueError("Arguments must be a JSON-encoded object.")
            arguments = json.loads(raw, parse_constant=_invalid_constant, parse_float=_finite_float,
                                   object_pairs_hook=_unique_object)
            _validate(arguments, schema, "arguments")
        except (ValueError, TypeError, RecursionError) as exc:
            # JSON errors report syntax/location, never include the response payload.
            error = str(exc) if isinstance(exc, ValueError) else "Invalid JSON argument structure."
            return {"reason": "invalid_tool_arguments", "tool": name,
                    "error": error[:500], "parameters": schema}
    return None


def _invalid_constant(_value: str) -> None:
    raise ValueError("JSON arguments must not contain NaN or Infinity.")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON arguments must contain only finite numbers.")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON arguments contain a duplicate key.")
        result[key] = value
    return result


def _validate(value: Any, schema: Mapping[str, Any], label: str) -> None:
    expected = schema.get("type")
    checks = {
        "object": isinstance(value, dict),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "boolean": type(value) is bool,
        "number": type(value) in (int, float),
        "array": isinstance(value, list),
    }
    if expected and not checks.get(expected, False):
        raise ValueError(f"{label} must be {expected}.")
    if isinstance(value, dict) and "properties" in schema:
        properties = schema["properties"]
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"{label} is missing required fields: {', '.join(missing)}.")
        if set(value) - set(properties):
            raise ValueError(f"{label} contains fields not defined by the advertised tool schema.")
        for key, field in properties.items():
            if key in value:
                _validate(value[key], field, f"{label}.{key}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{label} must be one of the advertised enum values.")
    if type(value) in (int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{label} must be at least {schema['minimum']}.")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{label} must be at most {schema['maximum']}.")


def save_rejected_response(
    directory: Path, *, model: str, message: Mapping[str, Any],
    diagnostic: Mapping[str, Any], secrets: Sequence[str] = (),
) -> dict[str, Any]:
    """Save bounded rejected public/tool output, never prompts, headers or reasoning.

    The caller places this directory outside the public workspace. The returned
    metadata is safe for internal traces; the raw response is never fed back to
    the model or attached to chat. Diagnostic I/O failure must not stop recovery.
    """
    content = str(message.get("content") or "")
    def scrub(value: str) -> str:
        # Tool arguments are themselves JSON text: redact both plain secrets
        # and their escaped forms before the enclosing diagnostic is serialized.
        for secret in secrets:
            if not secret:
                continue
            variants = [secret]
            for _ in range(3):
                variants.append(json.dumps(variants[-1], ensure_ascii=True)[1:-1])
            for variant in sorted(set(variants), key=len, reverse=True):
                value = value.replace(variant, "[REDACTED]")
        return value

    calls = message.get("tool_calls")
    calls = calls if isinstance(calls, list) else []
    snippets = []
    for call in calls[:8]:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if isinstance(function, Mapping):
            arguments = str(function.get("arguments") or "")
            snippets.append({"name": scrub(str(function.get("name") or ""))[:160],
                             "arguments": scrub(arguments)[:16_000],
                             "arguments_truncated": len(arguments) > 16_000})
    payload = {
        "schema_version": "1.0", "model": model,
        **safe_response_metadata(diagnostic),
        "reason": str(diagnostic.get("reason") or "invalid_tool_request"),
        "content_sha256": hashlib.sha256(content.encode("utf-8", errors="surrogatepass")).hexdigest(),
        "content": scrub(content)[:32_000], "content_truncated": len(content) > 32_000,
        "tool_calls": snippets, "tool_calls_truncated": len(calls) > 8,
        "scope": "Rejected public content and tool fragments only; no prompts or reasoning channel.",
    }
    rendered = json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2)
    for secret in secrets:
        if secret:
            rendered = rendered.replace(secret, "[REDACTED]")
            rendered = rendered.replace(json.dumps(secret, ensure_ascii=True)[1:-1], "[REDACTED]")
    result = dict(diagnostic)
    target = directory / f"response-{time.time_ns()}.json"
    temporary = target.with_suffix(".tmp")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, target)
        result["diagnostic_path"] = f"codex/protocol_diagnostics/{target.name}"
    except OSError:
        result["diagnostic_saved"] = False
    return result
