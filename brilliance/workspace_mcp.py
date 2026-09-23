"""Offline MCP tools for reliable research-program authoring.

Codex normally edits files through its shell.  On Windows, embedding a complete
scientific program in ``powershell -Command`` can exceed CreateProcess' command
line limit before a file is created.  This small stdio MCP server transports the
source as JSON-RPC data and writes it atomically inside the active project.

The server deliberately exposes no numerical or physics workflow.  It only
provides the missing coding primitive: save task-specific Python and JSON input
without involving a shell command line.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

SERVER_NAME = "brilliance-workspace"
SERVER_VERSION = "1.0.0"
MAX_PROGRAM_BYTES = 128_000
MAX_JSON_BYTES = 2_000_000


class _InvalidUnicodeText(ValueError):
    """Reject malformed tool text without guessing at the intended source."""


def _utf8_bytes(text: str, field: str) -> bytes:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        line = text.count("\n", 0, exc.start) + 1
        column = exc.start - text.rfind("\n", 0, exc.start)
        raise _InvalidUnicodeText(
            f"{field} contains an unpaired Unicode surrogate U+{ord(text[exc.start]):04X} "
            f"at character {exc.start}, line {line}, column {column}. "
            "No file was changed. Resend valid Unicode text; source characters "
            "will not be silently replaced or reinterpreted."
        ) from exc


def _workspace() -> Path:
    root_value = str(os.environ.get("BRILLIANCE_PROJECT_ROOT", "")).strip()
    project_id = str(os.environ.get("BRILLIANCE_PROJECT_ID", "")).strip()
    if not root_value or not project_id:
        raise RuntimeError(
            "BRILLIANCE_PROJECT_ROOT and BRILLIANCE_PROJECT_ID are required"
        )
    root = Path(root_value).resolve()
    project = (root / project_id).resolve()
    try:
        project.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("configured project escapes the research-project root") from exc
    workspace = (project / "workspace").resolve()
    if not workspace.is_dir():
        raise RuntimeError(f"research workspace does not exist: {workspace}")
    return workspace


def _relative_workspace_path(
    workspace: Path,
    value: str,
    *,
    required_root: str,
    suffix: str,
) -> Path:
    _utf8_bytes(str(value), "path")
    raw = str(value).strip().replace("\\", "/")
    if raw.startswith("workspace/"):
        raw = raw[len("workspace/") :]
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("path must be relative to the project workspace")
    resolved = (workspace / candidate).resolve()
    allowed = (workspace / required_root).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"path must stay below workspace/{required_root}") from exc
    if resolved.suffix.casefold() != suffix:
        raise ValueError(f"path must use the {suffix} suffix")
    return resolved


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_research_program(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """Atomically save one model-authored Python program and inspect it."""

    workspace = _workspace()
    path = _relative_workspace_path(
        workspace,
        str(arguments.get("path", "")),
        required_root="programs",
        suffix=".py",
    )
    source = str(arguments.get("source", ""))
    if not source.strip():
        raise ValueError("source must not be empty")
    source = source if source.endswith("\n") else source + "\n"
    encoded = _utf8_bytes(source, "source")
    if len(encoded) > MAX_PROGRAM_BYTES:
        raise ValueError(
            f"program is {len(encoded)} bytes; split the scientific work into smaller "
            f"stages below {MAX_PROGRAM_BYTES} bytes"
        )
    _atomic_text_write(path, source)

    # Import lazily so the MCP process starts quickly and never initializes JuTrack.
    from .research_program import inspect_research_program

    inspection = inspect_research_program(workspace, path.relative_to(workspace).as_posix())
    return {
        "saved": True,
        "path": path.relative_to(workspace).as_posix(),
        "bytes": len(encoded),
        "lines": len(source.splitlines()),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "preflight": inspection,
    }


def replace_research_program_text(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """Apply one exact, bounded textual repair without resending a whole program."""

    workspace = _workspace()
    path = _relative_workspace_path(
        workspace,
        str(arguments.get("path", "")),
        required_root="programs",
        suffix=".py",
    )
    if not path.is_file():
        raise ValueError(f"program does not exist: {path.relative_to(workspace)}")
    old = str(arguments.get("old", ""))
    new = str(arguments.get("new", ""))
    _utf8_bytes(old, "old")
    _utf8_bytes(new, "new")
    if not old:
        raise ValueError("old must contain the exact source text to replace")
    source = path.read_text(encoding="utf-8")
    occurrences = source.count(old)
    if occurrences != 1:
        raise ValueError(
            f"expected old text exactly once, found {occurrences}; read the saved file and "
            "supply a less ambiguous edit"
        )
    updated = source.replace(old, new, 1)
    encoded = _utf8_bytes(updated, "source")
    if len(encoded) > MAX_PROGRAM_BYTES:
        raise ValueError("edited program exceeds the source-size boundary")
    _atomic_text_write(path, updated)

    from .research_program import inspect_research_program

    inspection = inspect_research_program(workspace, path.relative_to(workspace).as_posix())
    return {
        "saved": True,
        "path": path.relative_to(workspace).as_posix(),
        "bytes": len(encoded),
        "lines": len(updated.splitlines()),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "preflight": inspection,
    }


def write_research_arguments(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """Atomically save structured arguments next to research programs."""

    workspace = _workspace()
    path = _relative_workspace_path(
        workspace,
        str(arguments.get("path", "")),
        required_root="programs",
        suffix=".json",
    )
    value = arguments.get("value")
    if not isinstance(value, Mapping):
        raise ValueError("value must be one JSON object")
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    encoded = _utf8_bytes(rendered, "value")
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("arguments JSON exceeds the workspace-writing boundary")
    _atomic_text_write(path, rendered)
    return {
        "saved": True,
        "path": path.relative_to(workspace).as_posix(),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "write_research_program",
        "description": (
            "Create or replace a UTF-8 Python research program atomically below "
            "workspace/programs. Use this instead of PowerShell redirection, Set-Content, "
            "here-strings, or shell-embedded source. Returns the saved source preflight."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path such as programs/g0_reference.py",
                },
                "source": {"type": "string", "description": "Complete Python source"},
            },
            "required": ["path", "source"],
            "additionalProperties": False,
        },
    },
    {
        "name": "replace_research_program_text",
        "description": (
            "Repair one exact source fragment in an existing research program. The old text "
            "must occur exactly once. Use this for bounded post-preflight repairs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_research_arguments",
        "description": (
            "Create or replace a UTF-8 JSON arguments file atomically below "
            "workspace/programs. The value must be one JSON object."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path such as programs/g0_reference.run.json",
                },
                "value": {"type": "object"},
            },
            "required": ["path", "value"],
            "additionalProperties": False,
        },
    },
)


def _tool_result(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    handlers = {
        "write_research_program": write_research_program,
        "replace_research_program_text": replace_research_program_text,
        "write_research_arguments": write_research_arguments,
    }
    if name not in handlers:
        raise ValueError(f"unknown Brilliance workspace tool: {name}")
    try:
        result = handlers[name](arguments)
    except _InvalidUnicodeText as exc:
        # A malformed argument is a correctable tool result, not an MCP server
        # failure. Never save lossy source or reinterpret scientific code.
        result = {"saved": False, "error": "invalid_unicode", "message": str(exc)}
        return {
            "content": [{"type": "text", "text": json.dumps(result)}],
            "structuredContent": result,
            "isError": True,
        }
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(result, ensure_ascii=False, allow_nan=False),
            }
        ],
        "structuredContent": result,
        "isError": False,
    }


def _response(request: Mapping[str, Any]) -> dict[str, Any] | None:
    method = str(request.get("method", ""))
    request_id = request.get("id")
    if request_id is None:
        return None
    if method == "initialize":
        requested = request.get("params", {})
        protocol = (
            str(requested.get("protocolVersion", "2025-06-18"))
            if isinstance(requested, Mapping)
            else "2025-06-18"
        )
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": protocol,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": list(TOOLS)}}
    if method == "tools/call":
        params = request.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError("tools/call params must be an object")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ValueError("tool arguments must be an object")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": _tool_result(str(params.get("name", "")), arguments),
        }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def main() -> int:
    """Serve newline-delimited MCP JSON-RPC on stdin/stdout."""

    # MCP is UTF-8 regardless of the Windows console code page. Codex filters
    # inherited environment variables for MCP children; locale-decoding an en
    # dash as GBK/surrogateescape produces U+DC93 inside otherwise valid source.
    # Decode the wire bytes here, never try to repair the decoded source later.
    input_stream = getattr(sys.stdin, "buffer", sys.stdin)
    for line in input_stream:
        if not line.strip():
            continue
        request_id: Any = None
        try:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise ValueError("JSON-RPC request must be an object")
            request_id = request.get("id")
            response = _response(request)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"},
            }
        if response is not None:
            # ASCII JSON escapes preserve Unicode losslessly even if stdout is
            # attached to a legacy-code-page wrapper (or the request is invalid).
            sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
