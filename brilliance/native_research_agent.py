"""Small provider-native write/run/inspect loop for accelerator research.

The language model owns task-specific code and scientific decisions.  Brilliance
owns only a few durable operations: API lookup, atomic program edits, execution in
the configured worker, and user-visible checkpoints.  Numerical output or an exact
execution error is returned to the same model turn so it can revise from evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import inspect
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from .artifacts import json_safe
from .codex_backends import CodexBackend
from .experts import ExpertRegistry
from .model_protocol import (
    REJECTION_MESSAGES,
    _finite_float,
    _invalid_constant,
    _unique_object,
    save_rejected_response,
    safe_response_metadata,
    validate_tool_batch,
)
from .research_program import inspect_research_program
from .scientific_acceptance import (
    ACCEPTANCE_ENVIRONMENT,
    acceptance_instructions,
    compact_scientific_acceptance,
    continued_request_acceptance,
    load_expert_registry,
    optimization_acceptance_warnings,
    receipt_acceptance_issues,
)
from .workspace_inspection import read_workspace_artifact


class NativeResearchAgentError(RuntimeError):
    """Raised when the provider-native research loop cannot continue."""


@dataclass(frozen=True, slots=True)
class NativeResearchAgentResult:
    response: str
    execution_attempts: int
    completed_executions: int
    request_satisfied: bool = False
    managed_job_id: str | None = None
    last_error: str = ""
    continuation_needed: bool = False
    failed_program: str = ""
    completion_evidence: Mapping[str, Any] | None = None
    protocol_failure: Mapping[str, Any] | None = None


EventCallback = Callable[[Mapping[str, Any]], Awaitable[None] | None]
CompletionCallback = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    Awaitable[Mapping[str, Any]],
]
RejectionRecorder = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "complete_research",
            "description": (
                "Conclude the CURRENT scientist request with an evidence-based answer using an "
                "exact committed JuTrack receipt run_id from this project, including earlier runs "
                "listed in the eligible evidence index. Use when the requested deliverable "
                "already exists, even if its saved program marked request_satisfied=false as an "
                "intermediate stage. Assess the actual requested scope: a seed/component is not a "
                "complete machine, and a first-candidate request does not require later optimization. "
                "Include key results and limitations. This records a separate request-completion "
                "decision; it never rewrites evidence or reruns calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}, "evidence_run_id": {"type": "string"}},
                "required": ["answer", "evidence_run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_research",
            "description": (
                "Pause an unfinished request when a real scientist decision, missing information, "
                "or external blocker prevents further authorized work. Explain the blocker and "
                "ask one specific question. Do not pause merely to announce the next calculation."
            ),
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}, "question": {"type": "string"}},
                "required": ["reason", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_expert",
            "description": (
                "Read the complete installed expert method in bounded pages when the supplied "
                "context is incomplete. Use the module and asset IDs in the expert context. "
                "Read all pages of a selected constructive method before implementing it. "
                "No internet or simulation is used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                    "asset_id": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20000},
                },
                "required": ["module_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_artifact",
            "description": (
                "Read saved research evidence, runtime diagnostics, or optimizer checkpoints "
                "without executing code or starting JuTrack. Empty path lists the workspace. "
                "Use a JSON pointer to select exact values (e.g. /global_best/variables); "
                "offset/limit page JSON entries, directory entries, or text lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pointer": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_user",
            "description": (
                "Publish one concise scientific plan, finding, decision, or repair update "
                "in the live research chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["plan", "finding", "decision", "repair", "next"],
                    },
                    "message": {"type": "string"},
                },
                "required": ["kind", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_help",
            "description": (
                "Retrieve the relevant section of Brilliance's authoritative public "
                "accelerator/JuTrack API reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_program",
            "description": (
                "Atomically save one short task-specific Python research program below "
                "workspace/programs and run its zero-cost API preflight. For long programs, "
                "save a small syntactically valid skeleton, then use patch_program for focused additions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["name", "source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_program",
            "description": (
                "Replace one exact unique substring in a saved research program and "
                "rerun its zero-cost API preflight."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["name", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_program",
            "description": "Read a bounded line range from a saved research program.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "line_count": {"type": "integer", "minimum": 1, "maximum": 240},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_program",
            "description": (
                "Execute a preflighted saved program in the configured Python/Julia/JuTrack "
                "worker and return compact numerical evidence or the exact error. Long runs "
                "are automatically submitted as durable background calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "timeout_s": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 604800,
                        "description": "Runtime estimate for durable jobs (not a kill deadline); hard limit only for short foreground programs.",
                    },
                    "background": {"type": "boolean"},
                    "allow_repeat": {
                        "type": "boolean",
                        "description": (
                            "Run identical saved source and arguments again only when repeated "
                            "sampling or changed external state is scientifically intentional."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
)

_TOOLS = tuple({**tool, "function": {**tool["function"], "parameters": {
    **tool["function"]["parameters"], "additionalProperties": False,
}}} for tool in _TOOLS)

_POPULATION_OPTIMIZER_CALLS = frozenset(
    {"optimize_lattice_parameters", "particle_swarm_optimize"}
)
_DURABLE_PROGRAM_CALLS = _POPULATION_OPTIMIZER_CALLS


def supports_native_research_agent(backend: CodexBackend) -> bool:
    """Return whether the selected backend exposes an OpenAI-compatible chat API."""

    return backend.key in {"deepseek", "custom", "ollama", "lmstudio"} and bool(
        backend.base_url
    )


def _requires_durable_execution(inspection: Mapping[str, Any]) -> bool:
    calls = {str(item) for item in inspection.get("detected_calls", [])}
    return bool(calls & _DURABLE_PROGRAM_CALLS)


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _program_path(workspace: Path, name: str) -> Path:
    rendered = str(name).strip().replace("\\", "/")
    if rendered.startswith("workspace/programs/"):
        rendered = rendered[len("workspace/programs/") :]
    elif rendered.startswith("programs/"):
        rendered = rendered[len("programs/") :]
    if not rendered or Path(rendered).name != rendered:
        raise ValueError("program name must be one filename below workspace/programs")
    if not rendered.lower().endswith(".py"):
        rendered += ".py"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.py", rendered):
        raise ValueError(f"invalid research program name: {name!r}")
    root = (workspace / "programs").resolve()
    destination = (root / rendered).resolve()
    destination.relative_to(root)
    return destination


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _api_sections(reference: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", reference))
    if not matches:
        return [("API reference", reference)]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(reference)
        sections.append((match.group(1).strip(), reference[match.start() : end].strip()))
    return sections


def api_help(reference: str, query: str, *, max_chars: int = 12_000,
             expert_registry: ExpertRegistry | None = None) -> str:
    """Return the most relevant complete reference sections for one query."""

    if max_chars <= 0:
        return ""
    tokens = {
        item.casefold()
        for item in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(query))
        if item.casefold() not in {"the", "and", "for", "with", "from", "how", "use"}
    }
    ranked: list[tuple[int, int, str]] = []
    for index, (title, section) in enumerate(_api_sections(reference)):
        lowered = section.casefold()
        score = sum(8 if token in title.casefold() else lowered.count(token) for token in tokens)
        if score:
            ranked.append((score, -index, section))
    if not ranked:
        ranked = [(0, -index, section) for index, (_, section) in enumerate(_api_sections(reference))]
    from . import accelerator_api

    signatures: list[str] = []
    for name in sorted(tokens):
        target = next((getattr(accelerator_api, key) for key in dir(accelerator_api)
                       if not key.startswith("_") and key.casefold() == name), None)
        if target is None or not callable(target):
            continue
        try:
            signatures.append(f"api.{target.__name__}{inspect.signature(target)}")
            contract = inspect.getdoc(target)
            if contract:
                signatures.append(contract[:2400])
            if inspect.isclass(target):
                for method in sorted(tokens):
                    member = getattr(target, method, None)
                    if callable(member):
                        signatures.append(f"{target.__name__}.{method}{inspect.signature(member)}")
                        contract = inspect.getdoc(member)
                        if contract:
                            signatures.append(contract[:2400])
        except (ValueError, TypeError, AttributeError):
            continue
    if expert_registry is not None:
        for module in expert_registry.modules:
            library = expert_registry.python_library(module.id)
            module_requested = module.id.casefold() in str(query).casefold()
            requested_names = {name.casefold() for name in dir(library) if name.casefold() in tokens}
            if module_requested or requested_names:
                # Expert APIs need their own result schemas and examples, not an
                # unrelated core reference selected merely because it mentions optics.
                for asset in expert_registry.knowledge_assets(module.id):
                    if asset.status == "deprecated" or asset.kind != "reference" or asset.path.suffix.lower() != ".md":
                        continue
                    source = asset.path.read_text(encoding="utf-8-sig")
                    if requested_names and not any(name in source.casefold() for name in requested_names):
                        continue
                    locator = (f"Module reference: read_expert(module_id={module.id!r}, "
                               f"asset_id={asset.id!r}, offset=0, limit=12000)\n")
                    for index, (title, section) in enumerate(_api_sections(source)):
                        lowered = section.casefold()
                        score = sum(8 if token in title.casefold() else lowered.count(token) for token in tokens)
                        if score or module_requested:
                            ranked.append((1000 + score, -index, locator + section))
            for name in dir(library):
                if not module_requested and name.casefold() not in tokens:
                    continue
                target = getattr(library, name)
                signatures.append(f"api.expert_module({module.id!r}).{name}{inspect.signature(target)}")
                contract = inspect.getdoc(target)
                if contract:
                    signatures.append(contract[:2400])
                if inspect.isclass(target):
                    for method in sorted(tokens):
                        member = getattr(target, method, None)
                        if callable(member) and not method.startswith("_"):
                            signatures.append(f"{name}.{method}{inspect.signature(member)}")
                            contract = inspect.getdoc(member)
                            if contract:
                                signatures.append(contract[:2400])
    selected: list[str] = []
    if signatures:
        block = "Exact installed API signatures (authoritative):\n" + "\n".join(signatures)
        budget = min(max_chars, max(1000, max_chars // 2))
        if len(block) > budget:
            notice = "\n[Signature help truncated; request one function/method for its complete contract.]"
            block = block[:max(0, budget - len(notice))] + notice[:budget]
        selected.append(block)
    used = sum(len(item) + 2 for item in selected)
    for _, _, section in sorted(ranked, reverse=True):
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(section) <= remaining:
            text = section
        else:
            notice = "\n[Reference truncated; follow its read_expert locator or narrow the api_help query.]"
            text = section[:max(0, remaining - len(notice))] + notice[:remaining]
        selected.append(text)
        used += len(text) + 2
        if len(selected) >= 3:
            break
    return "\n\n".join(selected)


def _compact_table(table: Mapping[str, Any]) -> dict[str, Any]:
    rows = table.get("rows", [])
    if isinstance(rows, Mapping):
        preview = rows.get("preview", [])
        return {
            "title": table.get("title"),
            "columns": table.get("columns"),
            "rows": list(preview) if isinstance(preview, list) else [],
            "row_count": rows.get("item_count", 0),
            "preview_only": True,
        }
    return {
        "title": table.get("title"),
        "columns": table.get("columns"),
        "rows": list(rows[:8]) if isinstance(rows, list) else [],
        "row_count": len(rows) if isinstance(rows, list) else 0,
    }


def _compact_execution(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("kind") == "managed_expert_calculation":
        return {
            "ok": True,
            "kind": "managed_calculation",
            "job_id": payload.get("job_id"),
            "state": payload.get("state"),
            "detail": payload.get("detail"),
            "continues_without_chat_turn": payload.get("continues_without_chat_turn"),
            "progress": payload.get("progress"),
        }
    execution = payload.get("result", {})
    execution = execution if isinstance(execution, Mapping) else {}
    scientific = execution.get("result", {})
    scientific = scientific if isinstance(scientific, Mapping) else {}
    tables = scientific.get("tables", [])
    plots = scientific.get("plots", [])
    downloads = scientific.get("downloads", [])
    program = execution.get("program", {})
    program = program if isinstance(program, Mapping) else {}
    artifacts = execution.get("artifacts", {})
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    output_directory = str(execution.get("output_directory") or "")

    def output_path(value: Any, *, fallback: str = "") -> str | None:
        text = str(value or fallback)
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute() and output_directory:
            path = Path(output_directory) / path
        return path.as_posix()

    full_result = (str(artifacts["evaluation_summary"]).replace("\\", "/")
                   if artifacts.get("evaluation_summary") else
                   output_path("result.json") if output_directory else None)
    candidate = (str(artifacts["lattice_json"]).replace("\\", "/")
                 if artifacts.get("lattice_json") else output_path(scientific.get("candidate_lattice")))
    return {
        "ok": True,
        "kind": "completed_execution",
        "execution_valid": payload.get("execution_valid"),
        "scientific_passed": payload.get("scientific_passed"),
        "request_satisfied": scientific.get("request_satisfied"),
        "scientific_acceptance": compact_scientific_acceptance(
            scientific.get("scientific_acceptance", {})
            if isinstance(scientific.get("scientific_acceptance"), Mapping) else {}
        ),
        "run_id": execution.get("run_id"),
        "output_directory": output_directory.replace("\\", "/") or None,
        "program": {key: program[key] for key in ("source", "snapshot", "sha256", "python") if key in program},
        "full_result": full_result,
        "read_full_result": ({"tool": "read_artifact", "arguments": {"path": full_result, "pointer": "/data"}}
                             if full_result else None),
        "execution_artifact": payload.get("artifact"),
        "duration_s": execution.get("duration_s"),
        "working_lattice_action": execution.get("working_lattice_action"),
        "status": scientific.get("status"),
        "summary": scientific.get("summary"),
        "data": scientific.get("data", {}),
        "tables": [
            _compact_table(item) for item in tables[:8] if isinstance(item, Mapping)
        ]
        if isinstance(tables, list)
        else [],
        "plots": plots[:8] if isinstance(plots, list) else [],
        "downloads": downloads[:8] if isinstance(downloads, list) else [],
        "methods": scientific.get("methods", []),
        "limitations": scientific.get("limitations", []),
        "candidate_lattice": candidate,
        "candidate_lattice_relative": scientific.get("candidate_lattice"),
        "use_as_working_lattice": scientific.get("use_as_working_lattice"),
        "candidate_handoff": scientific.get("candidate_handoff"),
        "stdout_tail": str(execution.get("stdout", ""))[-6_000:],
        "stderr_tail": str(execution.get("stderr", ""))[-2_000:],
        "validation": payload.get("validation", {}),
    }


def _decode_command_json(stdout: str) -> Mapping[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"worker returned invalid JSON: {stdout[-2000:]}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("worker response is not one JSON object")
    return value


def _brief_error(stderr: str, stdout: str = "") -> str:
    text = (stderr or stdout).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
        start = text.find("{")
        if start >= 0:
            try:
                candidate, _ = json.JSONDecoder().raw_decode(text[start:])
                payload = candidate if isinstance(candidate, Mapping) else None
            except json.JSONDecodeError:
                pass
    if isinstance(payload, Mapping) and payload.get("error"):
        text = str(payload["error"])
    text = text.replace("\\n", "\n").replace('\\"', '"')
    markers = re.findall(
        r"(?:AssertionError|AttributeError|IndexError|KeyError|RuntimeError|"
        r"TypeError|ValueError):[^\r\n]+",
        text,
    )
    headline = markers[-1] if markers else (text.splitlines()[-1] if text else "unknown error")
    locals_line = next(
        (line for line in reversed(text.splitlines()) if "BRILLIANCE_FAILURE_LOCALS=" in line),
        "",
    )
    return (headline + ("\n" + locals_line if locals_line else ""))[-6000:]


def _result_ready_for_answer(receipt: Mapping[str, Any]) -> bool:
    """Recognize a complete presentable result without imposing a physics workflow."""

    if not (
        receipt.get("kind") == "completed_execution"
        and receipt.get("execution_valid") is True
        and receipt.get("scientific_passed") is True
        and receipt.get("request_satisfied") is True
        and receipt.get("status") == "completed"
    ):
        return False
    data = receipt.get("data", {})
    data = data if isinstance(data, Mapping) else {}
    # A requested preliminary candidate, smoke check, or diagnosed unmet
    # aspiration can be a complete answer. Do not turn narrative caveats into
    # hidden scientific goals after the explicit task contract says complete.
    return bool(
        str(receipt.get("summary", "")).strip()
        and (
            data
            or receipt.get("tables")
            or receipt.get("plots")
            or receipt.get("downloads")
            or receipt.get("candidate_lattice")
        )
    )


def _dsml_tool_calls(content: str, *, step: int) -> list[dict[str, Any]]:
    """Recover complete DSML envelopes, without guessing missing syntax or types."""

    calls: list[dict[str, Any]] = []
    envelope = re.fullmatch(
        r'[^<]*<[^>]*DSML[^>]*tool_calls\s*>(.*?)</[^>]*DSML[^>]*tool_calls>\s*',
        content, re.DOTALL | re.IGNORECASE,
    )
    if not envelope:
        return []
    invoke_pattern = re.compile(
        r'<[^>]*DSML[^>]*invoke\s+name="([^"]+)"[^>]*>(.*?)'
        r"</[^>]*DSML[^>]*invoke>",
        re.DOTALL | re.IGNORECASE,
    )
    parameter_pattern = re.compile(
        r'<[^>]*DSML[^>]*parameter\s+name="([^"]+)"([^>]*)>(.*?)'
        r"</[^>]*DSML[^>]*parameter>",
        re.DOTALL | re.IGNORECASE,
    )
    body = envelope.group(1)
    if invoke_pattern.sub("", body).strip():
        return []
    for index, invoke in enumerate(invoke_pattern.finditer(body)):
        arguments: dict[str, Any] = {}
        if parameter_pattern.sub("", invoke.group(2)).strip():
            return []
        for parameter in parameter_pattern.finditer(invoke.group(2)):
            name = parameter.group(1)
            if name in arguments:
                return []
            value: Any = html.unescape(parameter.group(3).strip())
            explicit_json = bool(re.search(r'\bstring\s*=\s*"false"', parameter.group(2)))
            if name == "arguments" or explicit_json:
                try:
                    value = json.loads(value, parse_constant=_invalid_constant,
                                       parse_float=_finite_float, object_pairs_hook=_unique_object)
                except (ValueError, RecursionError):
                    return []
            arguments[name] = value
        calls.append(
            {
                "id": f"dsml-{step}-{index}",
                "type": "function",
                "function": {
                    "name": invoke.group(1),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    return calls


async def _notify(callback: EventCallback | None, event: Mapping[str, Any]) -> None:
    if callback is None:
        return
    result = callback(event)
    if asyncio.iscoroutine(result):
        await result


@dataclass
class _ResponseBudget:
    current: int
    ceiling: int

    @classmethod
    def configured(cls) -> "_ResponseBudget":
        initial = _bounded_int(os.environ.get("BRILLIANCE_NATIVE_AGENT_MAX_TOKENS"),
                               16384, 1024, 262144)
        ceiling = _bounded_int(os.environ.get("BRILLIANCE_NATIVE_AGENT_MAX_TOKENS_CEILING"),
                               max(initial, 65536), 1024, 262144)
        return cls(min(initial, ceiling), ceiling)

    def grow(self) -> None:
        self.current = min(self.ceiling, self.current * 2)


async def _provider_completion(
    backend: CodexBackend,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    on_event: EventCallback | None = None,
    on_rejection: RejectionRecorder | None = None,
    budget: _ResponseBudget | None = None,
) -> Mapping[str, Any]:
    credential = str(os.environ.get(backend.api_key_env, "")).strip()
    if backend.api_key_env and not credential:
        raise NativeResearchAgentError(
            f"the configured credential variable {backend.api_key_env} is empty"
        )
    headers = {"Authorization": f"Bearer {credential}"} if credential else {}
    timeout_s = _bounded_int(
        os.environ.get("BRILLIANCE_NATIVE_AGENT_MODEL_TIMEOUT_S"), 180, 30, 900
    )
    budget = budget if budget is not None else _ResponseBudget.configured()
    max_tokens = budget.current

    def capture(message: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> Mapping[str, Any]:
        if diagnostic.get("reason") == "output_token_limit":
            budget.grow()
        metadata = {**diagnostic, "requested_max_tokens": max_tokens,
                    "next_max_tokens": budget.current, "max_tokens_ceiling": budget.ceiling}
        return on_rejection(message, metadata) if on_rejection else metadata
    endpoint = backend.base_url.rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                request: dict[str, Any] = {
                    "model": backend.model,
                    "messages": list(messages),
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                }
                if tools:
                    request.update({"tools": list(tools), "tool_choice": "auto"})
                if os.environ.get("BRILLIANCE_NATIVE_AGENT_STREAM", "1").lower() not in {"0", "false", "off"}:
                    request["stream"] = True
                    if os.environ.get("BRILLIANCE_NATIVE_AGENT_STREAM_USAGE", "false").lower() in {"1", "true", "on"}:
                        request["stream_options"] = {"include_usage": True}
                    async with client.stream("POST", endpoint, headers=headers, json=request) as response:
                        response.raise_for_status()
                        if "text/event-stream" not in response.headers.get("content-type", ""):
                            body = json.loads(await response.aread())
                            return _nonstream_message(body, capture)
                        return await _stream_message(response.aiter_lines(), on_event, capture)
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=request,
                )
                response.raise_for_status()
                body = response.json()
                return _nonstream_message(body, capture)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            last_error = exc
    raise NativeResearchAgentError(
        f"{backend.name} tool-loop request failed: {type(last_error).__name__}: {last_error}"
    )


def _rejected_message(
    message: Mapping[str, Any], diagnostic: Mapping[str, Any],
    on_rejection: RejectionRecorder | None,
) -> Mapping[str, Any]:
    saved = on_rejection(message, diagnostic) if on_rejection else diagnostic
    return {"role": "assistant", "content": "", "_output_rejection": dict(saved)}


def _nonstream_message(body: Mapping[str, Any], on_rejection: RejectionRecorder | None) -> Mapping[str, Any]:
    choice = body["choices"][0]
    message = choice["message"]
    if not isinstance(message, Mapping):
        raise TypeError("provider message is not an object")
    if choice.get("finish_reason") == "length":
        public = {key: message[key] for key in ("role", "content", "tool_calls") if key in message}
        return _rejected_message(public, {"reason": "output_token_limit",
            **safe_response_metadata({"finish_reason": "length", "usage": body.get("usage")})}, on_rejection)
    # Private reasoning from compatible providers is neither chat nor tool data.
    return {key: message[key] for key in ("role", "content", "tool_calls") if key in message}


def _public_model_content(content: str) -> str:
    # Never stream partial protocol prefixes before the next chunk supplies DSML.
    # Ordinary comparisons such as 'Dx < 0.1 mm' and Markdown remain visible.
    marker = re.search(r'<[A-Za-z/|｜]', content)
    if marker:
        return content[:marker.start()]
    if "DSML" in content:
        return content.split("<", 1)[0]
    return content.removesuffix("<")


async def _stream_message(
    lines: Any, on_event: EventCallback | None, on_rejection: RejectionRecorder | None = None,
) -> Mapping[str, Any]:
    """Assemble public content and fragmented tool calls, never expose reasoning_content."""
    prose_limit = 6_000
    preview_limit = 1_200
    content = ""
    calls: dict[int, dict[str, Any]] = {}
    last_publish = 0.0
    received = False
    finished = False
    finish_reason = None
    usage: dict[str, Any] = {}
    async for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            finished = True
            break
        payload = json.loads(raw)
        usage.update(safe_response_metadata({"usage": payload.get("usage")}).get("usage", {}))
        if payload.get("error"):
            raise ValueError("Provider stream returned an error")
        for choice in payload.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            received = True
            finished = finished or choice.get("finish_reason") is not None
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            content += str(delta.get("content") or "")
            for fragment in delta.get("tool_calls") or []:
                index = int(fragment.get("index", 0))
                call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if fragment.get("id"):
                    call["id"] = fragment["id"]
                for key in ("name", "arguments"):
                    call["function"][key] += str(fragment.get("function", {}).get(key) or "")
            # DSML embeds a tool's code/arguments in the content channel. Keep
            # its complete payload for the fallback parser, never count it as
            # prose or display the internal protocol as a research update.
            public_content = _public_model_content(content)
            rejected = "public_prose_too_long" if len(public_content) > prose_limit else ""
            if rejected:
                await _notify(on_event, {"kind": "assistant_text", "text":
                    "The model response needs a shorter, valid format. Retrying; saved calculations are unchanged."})
                # Return no tools, even if a partial function fragment arrived
                # before the excess prose. Exiting closes only this provider
                # response, never a running numerical process.
                return _rejected_message({"content": content, "tool_calls": list(calls.values())}, {
                    "reason": rejected, "public_characters": len(public_content),
                    "discarded_tool_fragments": len(calls),
                }, on_rejection)
            if public_content and len(public_content) <= preview_limit and time.monotonic() - last_publish > 0.3:
                await _notify(on_event, {"kind": "assistant_text", "text": public_content})
                last_publish = time.monotonic()
    if finish_reason == "length":
        # Drain the terminal usage chunk before recording truncation. No partial
        # call is ever returned, including one that happens to contain valid JSON.
        return _rejected_message({"content": content, "tool_calls": list(calls.values())}, {
            "reason": "output_token_limit", "finish_reason": "length", "usage": usage,
            "public_characters": len(_public_model_content(content)),
            "discarded_tool_fragments": len(calls),
        }, on_rejection)
    if not received or not finished:
        if on_rejection:
            return _rejected_message({"content": content, "tool_calls": list(calls.values())},
                                     {"reason": "incomplete_provider_response",
                                      "finish_reason": "stream_interrupted", "usage": usage}, on_rejection)
        raise ValueError("Provider stream ended before a completed response")
    public_content = _public_model_content(content)
    await _notify(on_event, {"kind": "assistant_text", "text": public_content})
    return {"role": "assistant", "content": content, **({"tool_calls": [calls[key] for key in sorted(calls)]} if calls else {})}


async def run_native_research_agent(
    backend: CodexBackend,
    *,
    project_root: Path,
    project_id: str,
    workspace: Path,
    python_executable: Path,
    process_environment: Mapping[str, str],
    turn_wall_time_s: int,
    project_goal: str,
    scientist_message: str,
    expert_context: str,
    project_state_context: str,
    api_reference: str,
    knowledge_lookup: Callable[..., Mapping[str, Any]] | None = None,
    expert_registry: ExpertRegistry | None = None,
    prior_receipt: Mapping[str, Any] | None = None,
    receipt_lookup: Callable[[str], Mapping[str, Any]] | None = None,
    require_execution: bool = True,
    on_event: EventCallback | None = None,
    completion: CompletionCallback | None = None,
    conversation_history: Sequence[Mapping[str, Any]] = (),
    consume_input: Callable[[], list[str]] | None = None,
    pause_requested: Callable[[], bool] | None = None,
) -> NativeResearchAgentResult:
    """Run a compact model/tool loop until evidence is ready or a job is submitted."""

    if not supports_native_research_agent(backend):
        raise NativeResearchAgentError(f"{backend.name} has no native research-agent endpoint")
    registry = (expert_registry if expert_registry is not None else
                load_expert_registry(project_directory=project_root / project_id))
    system = """You are Brilliance's accelerator-research coding agent. Answer a purely conceptual
request directly from supplied expert knowledge or saved evidence. When the request asks for a
calculation, plot, lattice inspection/change, or unsaved numerical value, write short task-specific
Python and run it through the configured JuTrack worker. Do not inspect, discover, install, or
repair Python, Julia, JuTrack, or packages;
the public API and worker are already configured. Use api_help for exact signatures. Execute early
and revise from the returned numerical evidence or exact error instead of producing a large
one-shot program. API contracts are authoritative: do not execute programs merely to discover
schemas, callback arguments, installed functions, or artifact paths. Expert context is guidance,
not a mandatory workflow. Prefer the documented generic parameterization, matching, PSO/MOPSO,
and mutable-lattice helpers over hand-written replacements; use SciPy directly only for genuinely
task-specific numerical logic. Never make a numerical
beam-dynamics claim without a JuTrack-backed execution. A valid calculation and achievement of a
requested design target are different facts. Every run.finish call must set request_satisfied=False
for an intermediate result or request_satisfied=True only when the current scientist request has
actually been answered. A saved program's intermediate flag is immutable evidence, not a permanent
ban on finishing the chat request: once the requested deliverable is supported by a completed valid
receipt, use complete_research(answer, evidence_run_id) to record your semantic scope assessment.
Do not rerun a calculation or edit result.json merely to flip its completion flag.
Working-lattice selection is separate from completing the chat. For a validated complete
candidate intended as the starting point for the scientist's subsequent work, explicitly set
run.finish(use_as_working_lattice=True, scientific_passed=True) after its numerical gates.
This selects a provisional working candidate; it does not certify a final design. Keep component
seeds, failed hypotheses, and comparison-only trials as branches. State whether the delivered
candidate was selected or merely archived, and what remains unvalidated. Never imply that
complete_research itself selects a lattice or satisfies missing scientific gates.
If a design result is physically poor, diagnose and
try another bounded change when one is available; do not present the first stable lattice as the
answer. Design knowledge must not supply or silently load a prebuilt lattice; a greenfield design
must construct its own candidate from the scientist's contract and analytic physics scales. Only a
lattice already created or explicitly imported in this scientist's project may be continued.
Every stage that constructs, matches, optimizes, or edits a lattice must pass the exact resulting LatticeIR to
run.finish(candidate_lattice=...). Choose the exact archived input by object scope and identity.
run.working_lattice_file and run.latest_candidate_lattice_file are conveniences ONLY when they
refer to the intended object; the latest complete ring must not be mistaken for its earlier unit.
Load older inputs from the saved lattice inventory with api.load_lattice(run.workspace / path),
preserve parent identity, and never reconstruct a matched lattice from remembered constants.
Publish concise plan/finding/repair updates with
update_user. Put tables and plots in
run.progress/run.finish so they appear in chat. End with a clear scientist-facing explanation based
only on tool receipts. Long optimizations may be submitted in the background and must not be polled
inside this turn. An unchanged program and arguments are cached; patch the program before rerunning
unless repeated sampling is scientifically intentional. Keep every chat update and the final answer
under 250 words. Use read_artifact to inspect saved results, diagnostics, and checkpoints directly;
do not write a new calculation just to read them. This loop has only the advertised function tools: pass a program's JSON input
directly in run_program.arguments; never seek an MCP argument writer or another shell/tool. Never
print private deliberation or XML/DSML tool syntax into chat."""
    system += """
Useful progress explains a decision, a computed change against the previous candidate, a failed
requirement, or the next evidence-producing action. Use update_user for these concise milestones.
Do not narrate repeated reconsideration, arithmetic scratch work, or API speculation. Analytical
references must publish their values; an object attribute list is not a physics calculation.
Follow the constructive method in relevant expert knowledge rather than merely using its labels.
An incomplete method excerpt is not a complete construction recipe: retrieve the remaining pages
with read_expert before implementing that method. Do not guess omitted topology or matching steps.
When a background result is intermediate, continue the original request after interpreting it.
A paragraph announcing the next calculation does not complete that request. Use pause_research
only when a real decision or blocker prevents authorized progress; do not ask for a generic continue.
State exactly which object was validated: unit, repeated-unit reference ring, full requested ring,
or nonlinear study. Success of one is not evidence that the others have been designed.
Deliver the evidence requested by this message before proposing optional later research. A request
for a first computed candidate is not authorization to withhold it until an operational machine,
chromatic correction, nonlinear optimization or a tune-selection decision is complete. Present the
candidate, its actual optics/lattice figure, computed tables and limitations first; ask about later
research separately. Never impose an unrequested tune target. High-level JuTrack API calculations
retain already-computed optics and bend-contribution data for the exact candidate: committing it
with run.finish automatically attaches available canonical views, without another simulation.
If data.presentation reports missing evidence, obtain only the missing requested evidence; do not
invent a profile or describe a figure that does not exist. Per-bend contributions must come from
api.bend_radiation_contributions and its ringpara consistency check, not an invented decomposition.
"""
    if str(process_environment.get("BRILLIANCE_INTERNET_ACCESS", "false")).lower() not in {"true", "1", "yes"}:
        system += "\nInternet search/access is disabled. Use installed expert knowledge and local project evidence only. Never fetch papers, websites, or remote lattices. Configured model inference and local worker IPC are separate from research internet access."
    system += """
Choose calculations from the current request and evidence, not keywords in the project title.
An optimization may need a bounded baseline/objective check before its production search.
Do not confuse that check with completion of optimization. A request to inspect, plot, recover,
or validate saved optimization evidence does not authorize rerunning the search. Reuse committed
checkpoints for such follow-ups. Actual optimizer calls must use durable controls and publish
measured generation/candidate evidence; commentary alone is not optimization.
Build larger programs incrementally: write a short syntactically valid skeleton with unique
comment markers, then replace one marker at a time with patch_program. Keep each edit focused
(aim for under 100 lines per tool call). Do not run a skeleton or unfinished objective. Run only
after the needed pieces are saved and preflight passes. Reuse saved code rather than repeatedly
rewriting a whole program. These are transport rules, not a prescribed scientific workflow."""
    reference_query = "\n".join(
        (project_goal, scientist_message, expert_context[:4_000])
    )
    relevant_api = api_help(api_reference, reference_query, max_chars=12_000, expert_registry=registry)
    user = f"""PROJECT GOAL
{project_goal.strip()}

CURRENT SCIENTIST REQUEST
{scientist_message.strip()}

RELEVANT EXPERT KNOWLEDGE
{expert_context or "No matching expert note was retrieved."}

SAVED PROJECT STATE
{project_state_context or "No accepted lattice or prior numerical result is available."}

REQUEST-RELEVANT PUBLIC API
{relevant_api}
"""
    from .research_session import conversation_context

    messages: list[Mapping[str, Any]] = [
        {"role": "system", "content": system},
        *conversation_context(conversation_history),
        {"role": "user", "content": user},
    ]
    acceptance_requirements = continued_request_acceptance(
        scientist_message, conversation_history, registry=registry)
    acceptance_history = [*conversation_history, {"role": "user", "content": scientist_message}]
    if acceptance_requirements:
        messages.append({"role": "system", "content": acceptance_instructions(
            acceptance_requirements, registry=registry)})

    def capture_rejection(message: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> Mapping[str, Any]:
        return save_rejected_response(
            workspace.parent / "codex" / "protocol_diagnostics", model=backend.model,
            message=message, diagnostic=diagnostic,
            secrets=(str(os.environ.get(backend.api_key_env, "")),),
        )

    response_budget = _ResponseBudget.configured()
    requester = completion or (
        lambda model_messages, model_tools: _provider_completion(
            backend, model_messages, model_tools, on_event=on_event, on_rejection=capture_rejection,
            budget=response_budget,
        )
    )
    max_steps = _bounded_int(os.environ.get("BRILLIANCE_NATIVE_AGENT_STEPS"), 32, 6, 50)
    execution_attempts = 0
    completed_executions = 0
    managed_job_id: str | None = None
    last_error = ""
    last_program = ""
    last_receipt: dict[str, Any] = dict(prior_receipt or {})
    prior_requires_assessment = bool(prior_receipt)
    if prior_requires_assessment:
        messages.append({"role": "system", "content": (
            "Resume from saved evidence. A prior program's request_satisfied flag belongs to its "
            "original scope, not automatically to this current request. Inspect/reuse its result; "
            "use complete_research only if that evidence actually answers the current request. "
            "Otherwise perform the next justified action without repeating the baseline.\n"
            + json.dumps({key: last_receipt[key] for key in ("run_id", "summary", "full_result")
                          if key in last_receipt}, ensure_ascii=False))})
    completion_receipts: dict[str, dict[str, Any]] = (
        {str(last_receipt["run_id"]): dict(last_receipt)} if last_receipt.get("run_id") else {})
    eligible_archived_runs: list[Mapping[str, Any]] = []
    if receipt_lookup is not None:
        try:
            index = receipt_lookup("")
            eligible_archived_runs = [dict(item) for item in index.get("eligible_runs", [])[:12]
                                     if isinstance(item, Mapping)]
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    if eligible_archived_runs:
        messages.append({"role": "system", "content": (
            "Committed successful execution index for complete_research (exact run IDs; no rerun needed). "
            "Earlier runs remain selectable. Select evidence whose actual scope answers the CURRENT "
            "scientist request; an index entry is not itself a declaration of task completion. "
            "Use full_result with read_artifact for details.\n"
            + json.dumps(json_safe(eligible_archived_runs), ensure_ascii=False))})
    requested_completion: dict[str, Any] | None = None
    reminded_intermediate_runs: dict[str, int] = {}
    requested_pause = ""
    premature_answers = 0
    force_finalize = False
    malformed_answers = 0
    execution_cache: dict[str, dict[str, Any]] = {}
    help_cache: set[str] = set()
    repeated_help = 0
    lookup_recovery_announced = False
    readonly_tools = {"api_help", "read_program", "read_artifact", "read_expert"}
    lookup_observations: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    session_path = workspace / "agent_sessions" / f"native-{time.time_ns()}.json"

    def record(kind: str, payload: Mapping[str, Any]) -> None:
        row = dict(json_safe(payload))
        if "source" in row:
            source = str(row.pop("source"))
            row["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
            row["source_lines"] = len(source.splitlines())
        trace.append({"kind": kind, **row})
        _atomic_text(
            session_path,
            json.dumps(
                {"schema_version": "1.0", "model": backend.model, "events": trace},
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
        )

    def guard_lookup(name: str, arguments: Mapping[str, Any], result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Suppress only repeatedly identical evidence, not fresh reads or tools."""
        signature = json.dumps({"tool": name, "arguments": arguments}, sort_keys=True, ensure_ascii=False)
        file_state: Any = None
        try:
            if name == "read_program":
                target = _program_path(workspace, str(arguments.get("name", "")))
            elif name == "read_artifact":
                relative = str(arguments.get("path", "")).replace("\\", "/")
                target = (workspace / relative.removeprefix("workspace/")).resolve()
            else:
                target = None
            if target is not None and target.is_relative_to(workspace.resolve()):
                stat = target.stat()
                file_state = (stat.st_mtime_ns, stat.st_size)
        except (OSError, ValueError):
            pass
        rendered = json.dumps(json_safe(result), sort_keys=True, ensure_ascii=False, allow_nan=False)
        fingerprint = hashlib.sha256((rendered + repr(file_state)).encode("utf-8")).hexdigest()
        previous = lookup_observations.get(signature, {})
        deliveries = int(previous.get("deliveries", 0)) + 1 if previous.get("fingerprint") == fingerprint else 1
        lookup_observations[signature] = {"fingerprint": fingerprint, "deliveries": deliveries}
        if deliveries <= 2:
            return result, None
        observed = result
        if len(rendered) > 1_600:
            observed = {key: result[key] for key in
                        ("ok", "error", "path", "pointer", "total", "next_offset", "summarized") if key in result}
            data = result.get("data")
            if isinstance(data, list) and data:
                observed.update({"data_count": len(data), "first_item": data[0], "last_item": data[-1],
                                 "observation_scope": "first and last item of the previously supplied page"})
            elif isinstance(data, Mapping):
                observed["data_keys"] = list(data)[:30]
            if len(json.dumps(json_safe(observed), ensure_ascii=False)) > 1_600:
                observed = {"ok": result.get("ok"), "result_keys": list(result),
                            "observation_scope": "full unchanged result was already supplied twice"}
        observation = {"tool": name, "arguments": dict(arguments), "observed": observed,
                       "result_sha256": fingerprint}
        return {"ok": True, "lookup_repeat_suppressed": True, "no_new_evidence": True,
                "observation": observation,
                "instruction": "This exact lookup returned unchanged evidence already supplied twice. Choose a different pointer/question, act on the evidence, or explain the actual result; repeating it cannot advance the request."}, observation

    def inspect_current_program(relative: str) -> dict[str, Any]:
        inspection = dict(inspect_research_program(workspace, relative, expert_registry=registry))
        source_path = workspace / relative
        if source_path.is_file():
            warnings = optimization_acceptance_warnings(
                source_path.read_text(encoding="utf-8-sig"), acceptance_requirements,
                registry=registry,
            )
            if warnings:
                inspection["warnings"] = warnings
        return inspection

    async def execute_tool(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal execution_attempts, completed_executions, managed_job_id, last_error
        nonlocal last_receipt
        nonlocal prior_requires_assessment
        nonlocal repeated_help
        nonlocal requested_pause
        nonlocal last_program
        nonlocal requested_completion
        nonlocal eligible_archived_runs
        trace_arguments = dict(arguments)
        source_for_trace = str(trace_arguments.pop("source", ""))
        if source_for_trace:
            trace_arguments["source_sha256"] = hashlib.sha256(
                source_for_trace.encode("utf-8")
            ).hexdigest()
            trace_arguments["source_lines"] = len(source_for_trace.splitlines())
        record("tool_call", {"name": name, "arguments": trace_arguments})
        if name == "complete_research":
            answer = arguments.get("answer")
            evidence_id = arguments.get("evidence_run_id")
            if not isinstance(answer, str) or not answer.strip() or len(answer) > 6_000:
                return {"ok": False, "error": "Provide a substantive evidence-based answer below 6000 characters."}
            evidence = completion_receipts.get(evidence_id) if isinstance(evidence_id, str) else None
            if isinstance(evidence_id, str) and evidence_id and receipt_lookup is not None:
                lookup = receipt_lookup(evidence_id)
                offered = lookup.get("receipt")
                # Recheck even a cached current receipt when the runtime can
                # verify its committed archive; do not accept stale provenance.
                evidence = None
                if isinstance(offered, Mapping) and offered.get("run_id") == evidence_id:
                    evidence = dict(offered)
                    completion_receipts[evidence_id] = evidence
                eligible_archived_runs = [dict(item) for item in lookup.get("eligible_runs", [])[:12]
                                         if isinstance(item, Mapping)]
            if not evidence or not evidence_id:
                local_runs = [{"run_id": identity, "summary": str(item.get("summary") or "")[:320]}
                              for identity, item in completion_receipts.items()
                              if item.get("execution_valid") is True and item.get("scientific_passed") is True
                              and item.get("status") == "completed"]
                return {"ok": False, "error": "evidence_run_id must exactly identify a committed successful receipt in this project whose scope supports the current scientist request. Choose an exact eligible run_id below; do not invent IDs or repeat the same invalid one.",
                        "eligible_runs": (eligible_archived_runs or local_runs)[:12]}
            if not (evidence.get("kind") == "completed_execution"
                    and evidence.get("execution_valid") is True
                    and evidence.get("scientific_passed") is True
                    and evidence.get("status") == "completed"):
                return {"ok": False, "error": "This receipt is not a completed, execution-valid, scientifically passed result; it cannot support this completion action."}
            acceptance_issues = receipt_acceptance_issues(evidence, acceptance_requirements)
            if acceptance_issues:
                return {"ok": False, "error": "This evidence does not satisfy the current scientific request.",
                        "acceptance_issues": acceptance_issues,
                        "instruction": "Preserve the valid evidence. Address the unmet installed expert checks; do not repeat a baseline or claim completion without the required evidence.\n"
                        + acceptance_instructions(acceptance_requirements, registry=registry)}
            proof = {
                "kind": "scientist_request_completion_from_saved_evidence",
                "evidence_run_id": evidence_id,
                "evidence_receipt_sha256": hashlib.sha256(json.dumps(json_safe(evidence), sort_keys=True).encode("utf-8")).hexdigest(),
                "evidence_full_result": evidence.get("full_result"),
                "evidence_candidate_lattice": evidence.get("candidate_lattice"),
                "program_request_satisfied": evidence.get("request_satisfied"),
                "execution_valid": True, "scientific_passed": True,
                "decision_authority": "model_assessment_of_current_scientist_request_scope",
            }
            # A completion decision must not silently masquerade as selection.
            # Use the committed execution action, never a model-authored claim
            # that a saved candidate has become the working lattice.
            handoff = ""
            if evidence.get("candidate_lattice"):
                if evidence.get("working_lattice_action") == "advance":
                    handoff = "Lattice handoff: this calculation selected its candidate as a provisional working lattice; this is not final-design certification."
                else:
                    handoff = "Lattice handoff: candidate archived as a branch, not selected as the working lattice. Any existing working lattice is unchanged."
                proof["lattice_handoff"] = handoff
            requested_completion = {"answer": answer.strip() + ("\n\n" + handoff if handoff else ""), "proof": proof}
            return {"ok": True, "completion_proposed": True, "evidence_run_id": evidence_id,
                    "instruction": "Completion will be committed after checking for newer scientist input or pause; numerical evidence is unchanged."}
        if name == "pause_research":
            reason = str(arguments.get("reason", "")).strip()
            question = str(arguments.get("question", "")).strip()
            if not reason or not question:
                return {"ok": False, "error": "State the actual blocker and one specific scientist question."}
            requested_pause = f"{reason}\n\n{question}"
            return {"ok": True, "paused": True, "request_satisfied": False}
        if name == "update_user":
            kind = str(arguments.get("kind", "progress")).strip().lower()
            message = str(arguments.get("message", "")).strip()
            if not message:
                return {"ok": False, "error": "message is empty"}
            if len(message) > 2000:
                return {"ok": False, "error": "Summarize this update below 2000 characters: decision, evidence/change, and next action. Put tables and plots in run.progress, not a long monologue."}
            await _notify(on_event, {"kind": "user_update", "update_kind": kind, "message": message})
            return {"ok": True, "published": True}
        if name == "api_help":
            query = str(arguments.get("query", "")).strip()
            identity = " ".join(sorted(set(re.findall(r"\w+", query.casefold()))))
            if identity in help_cache:
                repeated_help += 1
                return {"ok": True, "already_supplied": True, "query": query,
                        "instruction": "This exact reference was already supplied in this turn. Use its installed signatures to write or patch the saved program. Repeating the lookup adds no evidence."}
            help_cache.add(identity)
            return {"ok": True, "query": query, "reference": api_help(
                api_reference, query, expert_registry=registry)}
        if name == "read_artifact":
            return {"ok": True, **read_workspace_artifact(
                workspace, str(arguments.get("path", "")),
                pointer=str(arguments.get("pointer", "")),
                offset=arguments.get("offset", 0), limit=arguments.get("limit", 40),
            )}
        if name == "read_expert":
            if knowledge_lookup is None:
                return {"ok": False, "error": "No installed expert registry is available in this session"}
            return {"ok": True, **knowledge_lookup(
                module_id=str(arguments.get("module_id", "")),
                asset_id=str(arguments.get("asset_id", "")).strip() or None,
                offset=arguments.get("offset", 0), limit=arguments.get("limit", 16000),
            )}
        if name in {"write_program", "patch_program", "read_program", "run_program"}:
            try:
                path = _program_path(workspace, str(arguments.get("name", "")))
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
        else:
            return {"ok": False, "error": f"unknown tool: {name}"}

        if name == "write_program":
            repeated_help = 0
            source = str(arguments.get("source", ""))
            if not source.strip():
                return {"ok": False, "error": "program source is empty"}
            _atomic_text(path, source.rstrip() + "\n")
            relative = path.relative_to(workspace).as_posix()
            inspection = inspect_current_program(relative)
            await _notify(
                on_event,
                {
                    "kind": "program_written",
                    "program": relative,
                    "lines": inspection.get("source_lines"),
                    "passed": inspection.get("passed"),
                    "issues": inspection.get("issues", []),
                },
            )
            return {"ok": True, "program": relative, "preflight": inspection}

        if name == "patch_program":
            repeated_help = 0
            if not path.is_file():
                return {"ok": False, "error": f"program does not exist: {path.name}"}
            old = str(arguments.get("old", ""))
            new = str(arguments.get("new", ""))
            source = path.read_text(encoding="utf-8")
            occurrences = source.count(old) if old else 0
            if occurrences != 1:
                return {
                    "ok": False,
                    "error": f"old text must occur exactly once; found {occurrences}",
                }
            _atomic_text(path, source.replace(old, new, 1))
            relative = path.relative_to(workspace).as_posix()
            inspection = inspect_current_program(relative)
            await _notify(
                on_event,
                {
                    "kind": "program_patched",
                    "program": relative,
                    "passed": inspection.get("passed"),
                    "issues": inspection.get("issues", []),
                },
            )
            return {"ok": True, "program": relative, "preflight": inspection}

        if name == "read_program":
            if not path.is_file():
                return {"ok": False, "error": f"program does not exist: {path.name}"}
            lines = path.read_text(encoding="utf-8").splitlines()
            start = _bounded_int(arguments.get("start_line"), 1, 1, max(1, len(lines)))
            count = _bounded_int(arguments.get("line_count"), 160, 1, 240)
            selected = lines[start - 1 : start - 1 + count]
            return {
                "ok": True,
                "program": path.relative_to(workspace).as_posix(),
                "start_line": start,
                "end_line": start + len(selected) - 1,
                "total_lines": len(lines),
                "source": "\n".join(
                    f"{index:04d}: {line}" for index, line in enumerate(selected, start)
                ),
            }

        if not path.is_file():
            return {"ok": False, "error": f"program does not exist: {path.name}"}
        relative = path.relative_to(workspace).as_posix()
        inspection = inspect_current_program(relative)
        if not inspection.get("passed"):
            return {"ok": False, "error": "program preflight failed", "preflight": inspection}
        detected_calls = set(inspection.get("detected_calls", ()))
        background = bool(arguments.get("background", False))
        automatically_backgrounded = _requires_durable_execution(inspection) and not background
        if automatically_backgrounded:
            background = True
        timeout_s = _bounded_int(arguments.get("timeout_s"), 604800 if background else 600, 1, 604800)
        run_arguments = arguments.get("arguments", {})
        if not isinstance(run_arguments, Mapping):
            return {"ok": False, "error": "arguments must be one object"}
        signature = hashlib.sha256(
            path.read_bytes()
            + json.dumps(
                json_safe(run_arguments), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + str(timeout_s).encode("ascii")
            + str(background).encode("ascii")
        ).hexdigest()
        if signature in execution_cache and not bool(arguments.get("allow_repeat", False)):
            return {
                **execution_cache[signature],
                "cached": True,
                "instruction": (
                    "This exact source and arguments already ran in this turn. Patch the "
                    "program before rerunning, or set allow_repeat=true when repetition is "
                    "scientifically intentional."
                ),
            }
        if inspection.get("warnings"):
            record("scientific_preflight_advisory", {"program": relative, "warnings": inspection["warnings"]})
            await _notify(on_event, {"kind": "user_update", "update_kind": "decision", "message":
                "Expert preflight: " + " ".join(str(item) for item in inspection["warnings"])})
        argument_path = path.with_suffix(".run.json")
        _atomic_text(
            argument_path,
            json.dumps(
                {
                    "script": relative,
                    "arguments": json_safe(run_arguments),
                    "timeout_s": timeout_s,
                    "require_optimizer_evidence": bool(detected_calls & _POPULATION_OPTIMIZER_CALLS),
                },
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
        )
        execution_attempts += 1
        last_program = path.relative_to(workspace).as_posix()
        await _notify(
            on_event,
            {
                "kind": "run_started",
                "program": relative,
                "attempt": execution_attempts,
                "background": background,
                "automatically_backgrounded": automatically_backgrounded,
                "timeout_s": timeout_s,
            },
        )
        command = [
            str(python_executable),
            "-m",
            "brilliance.codex_tools",
            "--project-root",
            str(project_root),
            "--project-id",
            project_id,
            "submit-expert-job" if background else "run-expert-tool",
            "accelerator.run_long_research_program"
            if background
            else "accelerator.run_research_program",
            "--arguments-file",
            f"workspace/{argument_path.relative_to(workspace).as_posix()}",
        ]
        environment = dict(process_environment)
        environment.update(
            {
                "BRILLIANCE_PROJECT_ROOT": str(project_root),
                "BRILLIANCE_PROJECT_ID": project_id,
                ACCEPTANCE_ENVIRONMENT: json.dumps(list(acceptance_requirements)),
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            }
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(workspace.parent),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        wait_s = min(timeout_s + 90, max(180, int(turn_wall_time_s) + 90))
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=wait_s
            )
        except asyncio.TimeoutError:
            last_error = (
                "worker command did not return at its declared boundary; its process and "
                "workspace were left intact"
            )
            await _notify(
                on_event,
                {"kind": "run_failed", "attempt": execution_attempts, "error": last_error},
            )
            return {"ok": False, "error": last_error}
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            try:
                rejected_payload = _decode_command_json(stdout)
                rejected = _compact_execution(rejected_payload)
            except ValueError:
                rejected = {}
            if rejected:
                validation = rejected.get("validation", {})
                validation = validation if isinstance(validation, Mapping) else {}
                issues = validation.get("issues", [])
                last_error = (
                    "Result contract failed: "
                    + "; ".join(str(item) for item in issues)
                    if issues
                    else _brief_error(stderr, stdout)
                )
                rejected.update({"ok": False, "error": last_error})
                execution_cache[signature] = rejected
                last_receipt = rejected
                await _notify(
                    on_event,
                    {
                        "kind": "run_failed",
                        "attempt": execution_attempts,
                        "error": last_error,
                    },
                )
                return rejected
            last_error = _brief_error(stderr, stdout)
            await _notify(
                on_event,
                {"kind": "run_failed", "attempt": execution_attempts, "error": last_error},
            )
            return {"ok": False, "error": last_error, "preflight": inspection}
        try:
            payload = _decode_command_json(stdout)
            compact = _compact_execution(payload)
        except ValueError as exc:
            last_error = str(exc)
            await _notify(
                on_event,
                {"kind": "run_failed", "attempt": execution_attempts, "error": last_error},
            )
            return {"ok": False, "error": last_error}
        if compact.get("kind") == "managed_calculation":
            managed_job_id = str(compact.get("job_id") or "") or None
        else:
            completed_executions += 1
        last_error = ""
        execution_cache[signature] = compact
        last_receipt = compact
        prior_requires_assessment = False
        await _notify(
            on_event,
            {
                "kind": "run_completed",
                "attempt": execution_attempts,
                "receipt": compact,
            },
        )
        return compact

    def accept_input(additional: Sequence[str] = ()) -> bool:
        nonlocal force_finalize, require_execution, last_receipt, requested_pause
        nonlocal requested_completion, malformed_answers
        nonlocal acceptance_requirements
        incoming = list(additional) + (consume_input() if consume_input else [])
        for text in incoming:
            messages.append({"role": "user", "content": text})
            record("scientist_input", {"content": text})
        if incoming:
            for incoming_request in incoming:
                acceptance_requirements = continued_request_acceptance(
                    incoming_request, acceptance_history, registry=registry)
                acceptance_history.append({"role": "user", "content": incoming_request})
            # Re-evaluate the new request semantically, without the previous turn's
            # optimization/finalization constraint. Earlier evidence remains intact.
            force_finalize = False
            require_execution = False
            last_receipt = {}
            requested_pause = ""
            requested_completion = None
            malformed_answers = 0
            completion_receipts.clear()
            lookup_observations.clear()
            messages.append({"role": "system", "content": (
                "The scientist has steered this session. Address the latest request before "
                "continuing earlier work. Use JuTrack for any new numerical claims."
                " Prior request-specific expert requirements are superseded by this current set:\n"
                + (acceptance_instructions(acceptance_requirements, registry=registry)
                   or "No installed request-specific acceptance checks apply.")
            )})
        return bool(incoming)

    def paused_result() -> NativeResearchAgentResult:
        return NativeResearchAgentResult(
            response="Paused at your request. Saved programs and results are preserved. "
                     "Independent numerical jobs continue; stop those separately if needed.",
            execution_attempts=execution_attempts, completed_executions=completed_executions,
            managed_job_id=managed_job_id,
        )

    for _step in range(max_steps):
        if pause_requested and pause_requested():
            return paused_result()
        if accept_input():
            await _notify(on_event, {"kind": "input_received"})
        await _notify(on_event, {"kind": "model_started"})
        available_tools = _TOOLS if repeated_help < 2 else tuple(tool for tool in _TOOLS
            if tool["function"]["name"] not in {"api_help", "read_program"})
        if repeated_help >= 2 and not lookup_recovery_announced:
            messages.append({"role": "user", "content": "The same API references have already been returned. Stop repeating these lookups. Use the saved source and exact signatures already in context to write or patch the task program, or explain the specific missing information. Lookup tools are temporarily unavailable until a source edit; no scientific action is prescribed."})
            lookup_recovery_announced = True
        elif repeated_help < 2:
            lookup_recovery_announced = False
        assistant = dict(await requester(messages, () if force_finalize else available_tools))
        if pause_requested and pause_requested():
            return paused_result()
        # A request received during generation supersedes the unexecuted proposal.
        if accept_input():
            await _notify(on_event, {"kind": "input_received"})
            continue
        tool_calls = assistant.get("tool_calls") or []
        content = str(assistant.get("content") or "").strip()
        mixed_protocol = bool(tool_calls) and any(
            marker in content for marker in ("DSML", "tool_calls>", "invoke name="))
        if not tool_calls and "DSML" in content:
            tool_calls = _dsml_tool_calls(content, step=_step)
            if tool_calls:
                prefix = content.split("<", 1)[0].strip()
                content = prefix if len(prefix) <= 600 else ""
        rejection = assistant.get("_output_rejection")
        if not rejection and mixed_protocol:
            rejection = {"reason": "malformed_tool_text", "error": "Use one tool encoding, not mixed native calls and embedded protocol."}
        if not rejection and (len(content) > 6_000 or (
            not tool_calls and any(marker in content for marker in ("DSML", "tool_calls>", "invoke name="))
        )):
            rejection = {"reason": "public_prose_too_long" if len(content) > 6_000 else "malformed_tool_text",
                         "public_characters": len(content)}
        if not rejection and tool_calls:
            rejection = validate_tool_batch(tool_calls, () if force_finalize else available_tools)
        if rejection:
            # Keep rejected deliberation/protocol text out of the transcript
            # supplied to the next completion. Echoing it back reinforces the
            # provider's repetition and consumes the next context budget.
            diagnostic = dict(rejection) if isinstance(rejection, Mapping) else {"reason": "invalid_tool_request"}
            if not diagnostic.get("diagnostic_path"):
                diagnostic = dict(capture_rejection(assistant, diagnostic))
            reason = str(diagnostic.get("reason") or "invalid_tool_request")
            if reason not in REJECTION_MESSAGES:
                reason = "invalid_tool_request"
            malformed_answers += 1
            diagnostic["consecutive_attempts"] = malformed_answers
            record("model_output_rejected", diagnostic)
            if malformed_answers >= 3:
                message = REJECTION_MESSAGES[reason]
                summary = str(last_receipt.get("summary") or "")
                failure = {"type": "model_protocol", "reason": reason,
                           "diagnostic_path": diagnostic.get("diagnostic_path"),
                           "attempts": malformed_answers, "can_retry": True, "message": message,
                           **safe_response_metadata(diagnostic)}
                recovery_kind = "response-recovery" if reason in {"output_token_limit", "incomplete_provider_response"} else "format-repair"
                return NativeResearchAgentResult(
                    response=((f"Last saved result: {summary}\n\n" if summary else "")
                              + f"{message} After two automatic {recovery_kind} retries, research is paused. "
                              + (f"Last requested output budget: {diagnostic.get('requested_max_tokens', 'unknown')} tokens. "
                                 "Check the configured output budget/ceiling for this provider; repeating the same request may fail again. "
                                 if reason == "output_token_limit" else "")
                              + "This is a model-response problem, not a failed physics calculation; "
                              "the current request remains unfinished. Saved code and evidence are unchanged. "
                              "Use Retry from checkpoint to resume the unfinished request. "
                              "If it recurs, configure another model; you can also stop here. "
                              "Independent numerical jobs have not been stopped."),
                    execution_attempts=execution_attempts,
                    completed_executions=completed_executions,
                    request_satisfied=False,
                    managed_job_id=managed_job_id,
                    last_error=last_error,
                    protocol_failure=failure,
                )
            if reason == "output_token_limit":
                current = diagnostic.get("requested_max_tokens")
                next_budget = diagnostic.get("next_max_tokens")
                budget_note = (f"Increasing the output allowance from {current} to {next_budget} tokens. "
                               if isinstance(current, int) and isinstance(next_budget, int) and next_budget > current
                               else "The output allowance cannot increase further under this configuration. ")
                retry_text = (f"Response recovery {malformed_answers}/2: the provider hit its output-token limit. "
                              + budget_note + "Retrying with a smaller code-writing step; saved calculations are unchanged.")
            elif reason == "incomplete_provider_response":
                retry_text = (f"Response recovery {malformed_answers}/2: the provider stream ended without a complete response. "
                              "Retrying the unexecuted step; saved calculations are unchanged.")
            else:
                retry_text = (f"Model response format repair {malformed_answers}/2: "
                              + ("the public response was too long." if reason == "public_prose_too_long"
                                 else "the tool request did not match its contract.")
                              + " Retrying; saved calculations are unchanged.")
            await _notify(on_event, {"kind": "assistant_text", "text": retry_text})
            repair = {"role": "user", "content": (
                "[MODEL_FORMAT_REPAIR] The previous model response was discarded. "
                f"Reason: {reason}. No tools in that response executed. "
                + (("The provider hit its generation limit, not a physics or JSON-schema failure. "
                    "Send ONE small complete tool call. For a long program, write a short syntactically valid "
                    "skeleton with unique comment markers, then add one focused piece at a time with patch_program. "
                    "Aim for under 100 source lines per edit. Do not run unfinished code. Do not continue a "
                    "truncated JSON fragment: it was never saved. "
                    f"Next output allowance: {diagnostic.get('next_max_tokens', 'configured limit')} tokens. ")
                   if reason == "output_token_limit" else "")
                + "Do not repeat deliberation. Send the next provided function call with normal JSON "
                "arguments, or an evidence-based answer under 250 words. Code belongs in function "
                "arguments; its length is not restricted by the public-prose limit. "
                "Continue the CURRENT scientist request, not the original project goal. "
                "Reuse saved results; do not repeat a completed baseline or change scientific strategy "
                "just to repair response formatting.\n"
                + json.dumps({key: diagnostic[key] for key in ("tool", "error", "parameters")
                              if key in diagnostic}, ensure_ascii=False)
            )}
            if messages and str(messages[-1].get("content", "")).startswith("[MODEL_FORMAT_REPAIR]"):
                messages[-1] = repair
            else:
                messages.append(repair)
            continue
        # These are consecutive protocol failures, not a lifetime budget. A valid
        # response starts a fresh run; earlier successful work must not consume it.
        malformed_answers = 0
        record(
            "model",
            {
                "content": content,
                "tool_names": [
                    str(item.get("function", {}).get("name", ""))
                    for item in tool_calls
                    if isinstance(item, Mapping)
                    and isinstance(item.get("function"), Mapping)
                ],
            },
        )
        messages.append(
            {
                "role": "assistant",
                "content": content,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            }
        )
        if not tool_calls:
            if require_execution and execution_attempts == 0 and premature_answers < 2:
                premature_answers += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "This is a computational request and no saved program has been run. "
                            "Use the tools to obtain JuTrack evidence before answering."
                        ),
                    }
                )
                continue
            response = content or (
                "The research loop ended without a user-facing interpretation. "
                "No additional claim was accepted."
            )
            intermediate_id = str(last_receipt.get("run_id", ""))
            incomplete = bool(last_receipt) and (
                prior_requires_assessment or last_receipt.get("request_satisfied") is False
                or receipt_acceptance_issues(last_receipt, acceptance_requirements))
            if (content and not last_error and incomplete
                    and reminded_intermediate_runs.get(intermediate_id, 0) < 2):
                reminded_intermediate_runs[intermediate_id] = reminded_intermediate_runs.get(intermediate_id, 0) + 1
                messages.append({"role": "user", "content": (
                    f"Saved program flag: request_satisfied={str(last_receipt.get('request_satisfied')).lower()}. "
                    "This request still needs a scope assessment or further work. A saved program's "
                    "completion flag does not establish completion of the CURRENT request: "
                    "if the requested deliverable already exists in a valid completed receipt, use "
                    "complete_research with that exact run_id and an evidence-based answer; do not "
                    "rerun just to change a flag. Otherwise continue the next justified calculation "
                    "within the request; if a real scientist decision "
                    "or blocker prevents that, use pause_research to ask the specific question and "
                    "state the incomplete scope. Announcing a next action alone is not completion."
                )})
                continue
            # An explicit prose answer after at least one validated execution is the
            # model's final semantic decision. ``request_satisfied`` on the receipt is
            # still required for automatic finalization; this branch covers requests
            # such as auditing or classifying a deliberately rejected candidate.
            answered_from_evidence = bool(
                content and completed_executions and last_receipt and not last_error and not prior_requires_assessment
                and last_receipt.get("request_satisfied") is not False
                and not receipt_acceptance_issues(last_receipt, acceptance_requirements)
            )
            unmet_acceptance = receipt_acceptance_issues(last_receipt, acceptance_requirements)
            if unmet_acceptance:
                response = (
                    "The current request has not yet passed its required expert acceptance checks. "
                    "Existing evidence is preserved, but does not establish completion. "
                    + "; ".join(unmet_acceptance)
                )
            return NativeResearchAgentResult(
                response=response,
                execution_attempts=execution_attempts,
                completed_executions=completed_executions,
                request_satisfied=(
                    bool(not prior_requires_assessment and last_receipt.get("request_satisfied") is True
                         and not receipt_acceptance_issues(last_receipt, acceptance_requirements))
                    or answered_from_evidence
                    or bool(content and not require_execution and not execution_attempts and not incomplete
                            and not acceptance_requirements)
                ),
                managed_job_id=managed_job_id,
                last_error=last_error,
            )
        ready_this_step = False
        steered_this_step = False
        pending_batch: list[str] = []
        batch_message_index = len(messages) - 1
        redundant_observations: list[dict[str, Any]] = []
        # Tool-bearing model prose can contain extensive drafting/speculation.
        # Only explicit update_user milestones become permanent progress chat;
        # final answers and numerical run.progress events are presented separately.
        for raw_call in tool_calls:
            call = raw_call if isinstance(raw_call, Mapping) else {}
            function = call.get("function", {})
            function = function if isinstance(function, Mapping) else {}
            name = str(function.get("name", ""))
            activity_id = str(call.get("id") or f"step-{_step}-{name}")
            await _notify(on_event, {"kind": "tool_started", "id": activity_id, "name": name})
            lookup_executed = False
            try:
                arguments = json.loads(str(function.get("arguments") or "{}"))
                if not isinstance(arguments, Mapping):
                    raise TypeError("tool arguments are not an object")
                if requested_completion:
                    result = {"ok": False, "error": "Not executed: completion was proposed; no further actions run in this batch."}
                elif requested_pause or steered_this_step or (pause_requested and pause_requested()):
                    result = {"ok": False, "error": "Not executed: scientist changed or paused the request."}
                elif name not in {tool["function"]["name"] for tool in available_tools}:
                    result = {"ok": False, "error": f"Tool {name!r} is not available in this step. Use the currently supplied tool definitions."}
                else:
                    lookup_executed = name in readonly_tools
                    result = await execute_tool(name, arguments)
            except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if lookup_executed:
                result, redundant = guard_lookup(name, arguments, result)
                if redundant is not None:
                    redundant_observations.append(redundant)
            elif name in {"write_program", "patch_program", "run_program"} and result.get("ok") and not result.get("cached"):
                lookup_observations.clear()
            if (name == "run_program" and _result_ready_for_answer(result)
                    and not receipt_acceptance_issues(result, acceptance_requirements)):
                ready_this_step = True
            if (name == "run_program" and result.get("kind") == "completed_execution"
                    and result.get("run_id")):
                completion_receipts[str(result["run_id"])] = dict(result)
            record("tool_result", {"name": name, "result": result})
            preflight = result.get("preflight")
            preflight_failed = isinstance(preflight, Mapping) and preflight.get("passed") is False
            error_detail = "; ".join(str(issue) for issue in preflight.get("issues", [])) if preflight_failed else str(result.get("error", ""))
            await _notify(on_event, {"kind": "tool_finished", "id": activity_id, "name": name,
                "ok": result.get("ok", True) and not preflight_failed, "error": error_detail[:2000]})
            rendered_result = json.dumps(
                json_safe(result), ensure_ascii=False, allow_nan=False
            )
            if len(rendered_result) > 24_000:
                rendered_result = json.dumps(
                    {
                        "ok": result.get("ok"),
                        "truncated": True,
                        "content": rendered_result[:23_000],
                    },
                    ensure_ascii=False,
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id", "")),
                    "content": rendered_result,
                }
            )
            # Do not insert user messages between an assistant tool batch and its
            # replies. Collect at the boundary after every tool has a reply.
            if consume_input:
                pending_batch.extend(consume_input())
                steered_this_step = bool(pending_batch)
        if len(redundant_observations) == len(tool_calls):
            # The whole batch was read-only and added no evidence. Replace its
            # speculative prose + duplicate tool exchange with one bounded
            # observation, preserving the original full deliveries and tool
            # protocol pairs. Repeated blocked batches replace this reminder;
            # they cannot progressively fill context with the same narrative.
            del messages[batch_message_index:]
            reminder = {"role": "user", "content": (
                "[READ_ONLY_NO_PROGRESS] These exact read-only requests returned unchanged results "
                "already supplied twice. No new calculation or finding occurred. Do not repeat this "
                "lookup batch or the same planning paragraph. Use the observations below to choose "
                "a different lookup, perform the next justified action, or give an evidence-based "
                "interpretation with the actual incomplete scope. Missing keys are absent, not a "
                "reason to retry the same pointer. Other pointers, changed files, and all research "
                "tools remain available. This is not scientific completion or a physics rejection.\n"
                + json.dumps(json_safe(redundant_observations[:2]), ensure_ascii=False)
            )}
            if messages and str(messages[-1].get("content", "")).startswith("[READ_ONLY_NO_PROGRESS]"):
                messages[-1] = reminder
            else:
                messages.append(reminder)
            record("lookup_no_progress", {"observations": redundant_observations[:2]})
            await _notify(on_event, {"kind": "assistant_text", "text":
                "Those lookups returned unchanged evidence. The model is being redirected to a different action or an interpretation of the saved result."})
        if pause_requested and pause_requested():
            # Preserve consumed steering in the public conversation (already saved
            # by the service); the next turn receives it as history.
            return paused_result()
        if accept_input(pending_batch):
            await _notify(on_event, {"kind": "input_received"})
            continue
        if requested_pause:
            return NativeResearchAgentResult(
                response=requested_pause, execution_attempts=execution_attempts,
                completed_executions=completed_executions, request_satisfied=False,
                managed_job_id=managed_job_id, last_error=last_error,
            )
        if requested_completion:
            proof = dict(requested_completion["proof"])
            record("request_completed", {"completion_evidence": proof})
            return NativeResearchAgentResult(
                response=requested_completion["answer"], execution_attempts=execution_attempts,
                completed_executions=completed_executions, request_satisfied=True,
                managed_job_id=managed_job_id, completion_evidence=proof,
            )
        if ready_this_step:
            force_finalize = True
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The execution contract validates a complete, presentable JuTrack result. "
                        "Its declared tables, plots, and files already passed artifact validation. "
                        "Do not write verification code; give the scientist the concise result, "
                        "interpretation, assumptions, and limitations now."
                    ),
                }
            )
        if managed_job_id:
            return NativeResearchAgentResult(
                response=(
                    f"The long calculation was submitted as `{managed_job_id}`. It continues "
                    "independently, publishes live numerical progress, and can be cancelled by "
                    "the scientist without losing its checkpoint."
                ),
                execution_attempts=execution_attempts,
                completed_executions=completed_executions,
                request_satisfied=False,
                managed_job_id=managed_job_id,
                last_error=last_error,
            )
    if completed_executions and last_receipt:
        summary = str(last_receipt.get("summary") or "A numerical checkpoint was completed.")
        suffix = f" The latest refinement error was: {last_error}" if last_error else ""
        return NativeResearchAgentResult(
            response=(
                f"{summary} The coding loop reached its {max_steps}-step boundary after "
                f"{execution_attempts} execution attempt(s); the completed evidence is "
                f"attached and the unfinished refinement remains saved.{suffix}"
            ),
            execution_attempts=execution_attempts,
            completed_executions=completed_executions,
            request_satisfied=False,
            managed_job_id=managed_job_id,
            last_error=last_error,
            continuation_needed=not bool(last_error),
            failed_program=last_program if last_error else "",
        )
    raise NativeResearchAgentError(
        f"the research loop reached {max_steps} model/tool steps without a completed execution; "
        "saved source and diagnostics were preserved"
    )
