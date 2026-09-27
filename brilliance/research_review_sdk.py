"""Read the retained Codex SDK telemetry without replaying research or prompts.

SDK notifications are a bounded window, not a full audit log.  In particular,
an SDK turn can contain many model requests, and token-usage ``total`` values
are cumulative for a thread.  This module deliberately keeps those concepts
separate and never reads checkpoint copies or reports hidden reasoning text.
"""

from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


_EVENT_RETENTION = 256
_MAX_EVENT_FILES = 10_000
_MAX_FILE_BYTES = 16 * 1024 * 1024
_TOKEN_KEYS = {
    "input_tokens": ("inputTokens", "input_tokens", "prompt_tokens"),
    "output_tokens": ("outputTokens", "output_tokens", "completion_tokens"),
    "total_tokens": ("totalTokens", "total_tokens"),
    "cached_input_tokens": ("cachedInputTokens", "cached_input_tokens"),
    "reasoning_output_tokens": ("reasoningOutputTokens", "reasoning_output_tokens"),
    "cache_write_input_tokens": ("cacheWriteInputTokens", "cache_write_input_tokens"),
}
_TERMINAL_STATUSES = {"completed", "complete", "succeeded", "success", "failed", "error", "cancelled", "canceled", "declined", "interrupted"}
_FAILURE_STATUSES = {"failed", "error", "cancelled", "canceled", "declined", "interrupted"}
_SUCCESS_STATUSES = {"completed", "complete", "succeeded", "success"}
_PROGRAM_RUNNERS = {"run_research_program", "run_long_research_program"}
_PROGRAM_TOOLS = _PROGRAM_RUNNERS | {"write_research_program", "patch_research_program", "read_research_program"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _counter() -> dict[str, int]:
    return {"calls": 0, "completed": 0, "succeeded": 0, "failed": 0, "outcome_unknown": 0}


def _normal_tokens(value: Mapping[str, Any]) -> dict[str, int | None] | None:
    result = {
        target: next((_integer(value[key]) for key in aliases if _integer(value.get(key)) is not None), None)
        for target, aliases in _TOKEN_KEYS.items()
    }
    details = _mapping(value.get("input_tokens_details", value.get("prompt_tokens_details")))
    if result["cached_input_tokens"] is None:
        result["cached_input_tokens"] = _integer(details.get("cached_tokens"))
    details = _mapping(value.get("output_tokens_details", value.get("completion_tokens_details")))
    if result["reasoning_output_tokens"] is None:
        result["reasoning_output_tokens"] = _integer(details.get("reasoning_tokens"))
    if result["total_tokens"] is None and result["input_tokens"] is not None and result["output_tokens"] is not None:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result if any(value is not None for value in result.values()) else None


def _tool_name(item: Mapping[str, Any]) -> str | None:
    kind = str(item.get("type") or "")
    if kind == "commandExecution":
        return "exec_command"
    if kind == "fileChange":
        return "apply_patch"
    if kind == "webSearch":
        return "web.run"
    if kind in {"mcpToolCall", "dynamicToolCall"}:
        tool = str(item.get("tool") or item.get("name") or "unknown_tool")
        server = str(item.get("server") or "")
        return f"{server}.{tool}" if server else tool
    if kind in {"imageView", "image_view"}:
        return "view_image"
    if str(kind).startswith("collab") and str(kind).endswith("ToolCall"):
        return str(kind)
    return None


def _outcome(item: Mapping[str, Any], completed: bool) -> str:
    """Completion of a notification alone is not evidence of tool success."""
    if not completed:
        return "outcome_unknown"
    status = str(item.get("status") or "").casefold()
    result = _mapping(item.get("result"))
    if status in _FAILURE_STATUSES or item.get("error") or result.get("error") or result.get("isError") is True or result.get("ok") is False or item.get("success") is False:
        return "failed"
    if item.get("type") == "commandExecution":
        exit_code = item.get("exitCode", item.get("exit_code"))
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            return "succeeded" if exit_code == 0 else "failed"
        # Commands with no exit code can still be running in the background.
        return "outcome_unknown"
    if status in _SUCCESS_STATUSES or item.get("success") is True or result.get("isError") is False or result.get("ok") is True:
        return "succeeded"
    return "outcome_unknown"


def _program_invocation(item: Mapping[str, Any]) -> bool:
    if str(item.get("type") or "") in {"mcpToolCall", "dynamicToolCall"}:
        return str(item.get("tool") or item.get("name") or "").rsplit(".", 1)[-1] in _PROGRAM_RUNNERS
    if item.get("type") == "commandExecution":
        # Match only explicit Brilliance runner calls; shell file reads and
        # arbitrary Python commands are not reliably research executions.
        command = str(item.get("command") or "")
        return bool(re.search(r"\bbrilliance\.codex_tools\b", command) and re.search(r"\brun-expert-tool\s+['\"]?accelerator\.run_(?:long_)?research_program\b", command))
    return False


def _arguments(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("arguments")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return _mapping(value)


def _safe_identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value) else None


def _error_details(item: Mapping[str, Any], outcome: str) -> tuple[str | None, str | None]:
    """Classify failures without exporting provider text, commands or locals."""
    if outcome != "failed":
        return None, None
    exit_code = item.get("exitCode", item.get("exit_code"))
    if item.get("type") == "commandExecution" and isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code:
        return "NonzeroExitCode", f"Command exited with code {exit_code}."
    result = _mapping(item.get("result"))
    error = item.get("error") or result.get("error") or {}
    # Error bodies may embed credentials or whole commands. Only use them to
    # select fixed labels, never copy their text into the audit export.
    hint = str(error).casefold()
    for pattern, category, message in (
        (r"timeout|timed? ?out", "Timeout", "The tool reported a timeout."),
        (r"rate.?limit|too many requests|\b429\b", "RateLimit", "The tool reported a rate limit."),
        (r"unauthorized|authentication|invalid.?api.?key|\b401\b", "AuthenticationError", "The tool reported an authentication failure."),
        (r"forbidden|permission|\b403\b", "PermissionError", "The tool reported a permission failure."),
        (r"connection|network|dns", "ConnectionError", "The tool reported a connection failure."),
    ):
        if re.search(pattern, hint):
            return category, message
    status = str(item.get("status") or "").casefold()
    if status in {"cancelled", "canceled", "declined", "interrupted"}:
        return "Interrupted", "The tool was cancelled, declined or interrupted."
    return "ToolError", "The retained tool record explicitly reports failure."


def _online_arguments(item: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the supported web action's query/URL/pattern fields."""
    action = _mapping(item.get("action"))
    kind = str(action.get("type") or "")
    if kind == "search" or (not kind and isinstance(item.get("query"), str)):
        queries = action.get("queries")
        if not isinstance(queries, list):
            query = action.get("query", item.get("query"))
            queries = [query] if isinstance(query, str) else []
        return {"search_query": [{"q": query} for query in queries if isinstance(query, str)]}
    if kind in {"openPage", "open_page", "open"}:
        return {"open": [{"ref_id": action["url"]}]} if isinstance(action.get("url"), str) else {"open": []}
    if kind in {"findInPage", "find_in_page", "find"}:
        return {"find": [{"ref_id": action.get("url"), "pattern": action.get("pattern")}]} if isinstance(action.get("url"), str) and isinstance(action.get("pattern"), str) else {"find": []}
    return {}


def _online_result(item: Mapping[str, Any], outcome: str) -> dict[str, Any] | None:
    """Distinguish service errors from unretained search/read results."""
    raw_result = _mapping(item.get("result"))
    result: dict[str, Any] = {}
    if outcome == "failed":
        category, message = _error_details(item, outcome)
        result.update(ok=False, error=message, error_class=category)
    elif isinstance(raw_result.get("ok"), bool):
        result["ok"] = raw_result["ok"]
    elif isinstance(raw_result.get("isError"), bool):
        result["ok"] = not raw_result["isError"]
    elif isinstance(item.get("success"), bool):
        result["ok"] = item["success"]
    # Only results that were actually retained establish hit/content evidence.
    # Do not replace missing data with an empty result list.
    hits = raw_result.get("results", item.get("results"))
    if isinstance(hits, list):
        result["results"] = [{"url": hit["url"]} for hit in hits if isinstance(hit, Mapping) and isinstance(hit.get("url"), str)]
        if hits and not result["results"]:
            result.pop("results")  # Unsupported hit structure is not zero hits.
    text = raw_result.get("text")
    if isinstance(text, str):
        result["content_characters"] = len(text)
        result["content_returned"] = bool(text.strip())
    return result or None


def collect_sdk_metrics(project_dir: Path) -> dict[str, Any]:
    """Collect lower-bound SDK activity and properly scoped saved token totals.

    Only ``codex/events/*.json`` and ``codex/state.json`` are inspected.  Tool
    starts, updates, completions, and copies in a turn's item list are merged by
    thread/item ID.  Known-token fields are integers; unavailable ones are None.
    Cached/reasoning token counts are subsets of input/output, never additions.
    """
    project_dir = Path(project_dir)
    sources: list[str] = []
    source_manifest: dict[str, dict[str, Any]] = {}
    limitations: list[str] = []
    malformed: list[str] = []
    oversized: list[str] = []
    items: dict[tuple[str, str], dict[str, Any]] = {}
    turns: dict[tuple[str, str], dict[str, bool]] = {}
    threads: set[str] = set()
    # Scope, thread, turn (empty for a cumulative thread total).
    usage: dict[tuple[str, str, str], dict[str, int | None]] = {}
    token_snapshots = 0
    unidentified_items = 0
    schema_errors = 0
    recognized_events = 0

    def read(path: Path) -> Mapping[str, Any] | None:
        relative = path.relative_to(project_dir).as_posix()
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                oversized.append(relative)
                return None
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(value, Mapping):
                malformed.append(relative)
                return None
            source_manifest[relative] = {
                "path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            }
            return value
        except (OSError, UnicodeError, json.JSONDecodeError):
            malformed.append(relative)
            return None

    def register_turn(thread_id: str, turn_id: str, *, started: bool = False, completed: bool = False, failed: bool = False) -> None:
        if not turn_id or turn_id.startswith("native-agent-"):
            return
        row = turns.setdefault((thread_id, turn_id), {"started": False, "completed": False, "failed": False})
        row["started"] |= started
        row["completed"] |= completed
        row["failed"] |= failed

    def register_item(item: Mapping[str, Any], thread_id: str, turn_id: str, *, completed: bool, source: str) -> None:
        nonlocal unidentified_items
        name = _tool_name(item)
        if name is None:
            return
        item_id = str(item.get("id") or "")
        if not item_id:
            unidentified_items += 1
            return
        register_turn(thread_id, turn_id)
        key = (thread_id, item_id)
        row = items.setdefault(key, {
            "id": f"codex_sdk:{thread_id}:{item_id}", "name": name,
            "completed": False, "outcome": "outcome_unknown", "program": False,
            "source": source, "argument_keys": set(), "program_path": None,
            "run_id": None, "job_id": None, "error_class": None,
            "error_message": None, "outcome_rank": 0,
            "web_arguments": {}, "web_result": None,
        })
        terminal = completed or str(item.get("status") or "").casefold() in _TERMINAL_STATUSES
        outcome = _outcome(item, terminal)
        row["completed"] |= terminal
        row["program"] |= _program_invocation(item)
        arguments = _arguments(item)
        row["argument_keys"].update(_safe_identifier(key) or "[unusual argument key]" for key in arguments)
        if name.rsplit(".", 1)[-1] in _PROGRAM_TOOLS:
            program = arguments.get("name") or arguments.get("script") or arguments.get("path")
            if isinstance(program, str) and len(program) <= 512 and re.fullmatch(r"[A-Za-z0-9_. /\\:-]+", program):
                row["program_path"] = program.replace("\\", "/")
        raw_result = _mapping(item.get("result"))
        for field in ("run_id", "job_id"):
            row[field] = _safe_identifier(raw_result.get(field)) or row[field]
        # Preserve a known terminal result if a less informative duplicate or
        # start notification appears later in the retained file ordering.
        explicit = item.get("error") or raw_result.get("error") or any(isinstance(value, bool) for value in (item.get("success"), raw_result.get("isError"), raw_result.get("ok"))) or isinstance(item.get("exitCode", item.get("exit_code")), int)
        rank = 2 if explicit else 1
        if outcome != "outcome_unknown" and rank >= row["outcome_rank"]:
            row["outcome"] = outcome
            row["outcome_rank"] = rank
            row["source"] = source
            row["error_class"], row["error_message"] = _error_details(item, outcome)
        if name == "web.run":
            web_arguments = _online_arguments(item)
            if web_arguments and (not row["web_arguments"] or any(web_arguments.values())):
                row["web_arguments"] = web_arguments
                row["argument_keys"].update(web_arguments)
            web_result = _online_result(item, outcome)
            if web_result is not None:
                row["web_result"] = {**(row["web_result"] or {}), **web_result}
                row["source"] = source

    def register_usage(payload: Mapping[str, Any], thread_id: str, turn_id: str, *, scope_hint: str = "") -> bool:
        nonlocal token_snapshots
        value = _mapping(payload.get("tokenUsage", payload.get("token_usage", payload.get("usage", payload))))
        if isinstance(value.get("total"), Mapping):
            scope, value = "thread", _mapping(value["total"])
        elif isinstance(value.get("last"), Mapping):
            scope, value = "last", _mapping(value["last"])
        else:
            scope = scope_hint or ("turn" if turn_id else "last")
        normalized = _normal_tokens(value)
        if normalized is None:
            return False
        token_snapshots += 1
        key = (scope, thread_id or "unknown-thread", "" if scope == "thread" else turn_id)
        previous = usage.get(key)
        # A cumulative snapshot is not an increment.  Retain the most complete
        # (largest) observed total, also handling reordered notification files.
        rank = lambda row: (row.get("total_tokens") or 0, row.get("input_tokens") or 0, row.get("output_tokens") or 0)
        if previous is None or rank(normalized) >= rank(previous):
            usage[key] = normalized
        return True

    paths = sorted((project_dir / "codex" / "events").glob("*.json"))
    all_event_files = len(paths)
    if all_event_files > _MAX_EVENT_FILES:
        limitations.append(f"Only the newest {_MAX_EVENT_FILES} SDK event files were inspected; older events were omitted.")
        paths = paths[-_MAX_EVENT_FILES:]
    for path in paths:
        event = read(path)
        if event is None:
            continue
        method = str(event.get("method") or "")
        payload = _mapping(event.get("payload"))
        thread_id = str(payload.get("threadId") or payload.get("thread_id") or "")
        if not method.startswith(("item/", "turn/", "thread/")) and not (method == "error" and thread_id):
            continue
        recognized_events += 1
        sources.append(path.relative_to(project_dir).as_posix())
        if thread_id:
            threads.add(thread_id)
        turn = _mapping(payload.get("turn"))
        turn_id = str(payload.get("turnId") or payload.get("turn_id") or turn.get("id") or "")
        register_turn(thread_id, turn_id, started=method == "turn/started", completed=method == "turn/completed", failed=method == "turn/completed" and bool(turn.get("error") or str(turn.get("status") or "").casefold() in _FAILURE_STATUSES))
        if method in {"item/started", "item/updated", "item/completed"}:
            raw_item = payload.get("item", payload)
            if not isinstance(raw_item, Mapping):
                schema_errors += 1
            else:
                register_item(raw_item, thread_id, turn_id, completed=method == "item/completed", source=path.relative_to(project_dir).as_posix())
        if method in {"turn/started", "turn/completed"}:
            turn_items = turn.get("items") or []
            if not isinstance(turn_items, list):
                schema_errors += 1
                turn_items = []
            for item in turn_items:
                if isinstance(item, Mapping):
                    register_item(item, thread_id, turn_id, completed=False, source=path.relative_to(project_dir).as_posix())
                else:
                    schema_errors += 1
        if method in {"turn/tokenUsage/updated", "thread/tokenUsage/updated", "turn/completed"}:
            register_usage(payload, thread_id, turn_id, scope_hint="thread" if method == "thread/tokenUsage/updated" else "")
            if turn:
                register_usage(turn, thread_id, turn_id)

    state_path = project_dir / "codex" / "state.json"
    state = read(state_path) if state_path.is_file() else None
    state_usage = _mapping(_mapping(state).get("token_usage"))
    state_thread = str(state_usage.get("threadId") or _mapping(state).get("thread_id") or "")
    state_turn = str(state_usage.get("turnId") or _mapping(state).get("turn_id") or "")
    if state and (recognized_events or state_thread) and register_usage(state_usage, state_thread, state_turn, scope_hint="last"):
        sources.append(state_path.relative_to(project_dir).as_posix())
        if state_thread:
            threads.add(state_thread)
        register_turn(state_thread, state_turn)

    tool_counts: dict[str, dict[str, int]] = defaultdict(_counter)
    program_counts = _counter()
    tool_calls: list[dict[str, Any]] = []
    online_calls: list[dict[str, Any]] = []
    for row in items.values():
        counts = [tool_counts[row["name"]]]
        if row["program"]:
            counts.append(program_counts)
        for count in counts:
            count["calls"] += 1
            count["completed"] += int(row["completed"])
            count[row["outcome"]] += 1
        tool_calls.append({
            "id": row["id"], "runtime": "codex_sdk", "tool": row["name"],
            "outcome": "unknown" if row["outcome"] == "outcome_unknown" else row["outcome"],
            "source": row["source"], "event_index": None, "result_event_index": None,
            "argument_keys": sorted(row["argument_keys"]), "program": row["program_path"],
            "run_id": row["run_id"], "job_id": row["job_id"],
            "error_class": row["error_class"], "error_message": row["error_message"],
        })
        if row["name"] == "web.run":
            online_calls.append({
                "id": row["id"], "runtime": "codex_sdk", "tool": "web.run",
                "arguments": row["web_arguments"], "result": row["web_result"],
                "source": row["source"], "event_index": None, "result_event_index": None,
            })

    selected: list[tuple[tuple[str, str, str], dict[str, int | None]]] = []
    usage_threads = {key[1] for key in usage}
    for thread_id in sorted(usage_threads):
        if thread_id == "unknown-thread" and len(usage_threads) > 1:
            limitations.append("Token snapshots with no thread ID were excluded because they may duplicate a known thread's usage.")
            continue
        cumulative = usage.get(("thread", thread_id, ""))
        if cumulative is not None:
            selected.append((("thread", thread_id, ""), cumulative))
            continue
        turn_values = [(key, value) for key, value in usage.items() if key[1] == thread_id and key[0] == "turn"]
        if turn_values:
            selected.extend(turn_values)
        else:
            # Multiple last-request snapshots may be overlapping views.  With
            # no response IDs the largest one is a useful lower bound only.
            last_values = [(key, value) for key, value in usage.items() if key[1] == thread_id]
            if last_values:
                selected.append(max(last_values, key=lambda pair: pair[1].get("total_tokens") or 0))
    scopes = {key[0] for key, _ in selected}
    scope = {frozenset({"thread"}): "observed_thread_totals", frozenset({"turn"}): "observed_turn_totals", frozenset({"last"}): "last_request_only"}.get(frozenset(scopes), "mixed_observed_scopes" if scopes else "unavailable")
    tokens = {
        name: sum(value[name] for _, value in selected) if selected and all(value[name] is not None for _, value in selected) else None
        for name in _TOKEN_KEYS
    }
    tokens.update(scope=scope, snapshots=token_snapshots, threads=len(usage_threads))
    observed = bool(recognized_events or selected)
    if observed:
        limitations.extend([
            "SDK events have bounded retention (normally 256 files); observed tool and turn counts are lower bounds, not a complete historical audit.",
            "An SDK turn can contain multiple model requests. The number of model inference calls is unavailable from these retained notifications.",
            "Tool completion does not establish scientific validity; shell execution success requires a recorded zero exit code. Program runner calls can submit background work whose final status is recorded elsewhere.",
        ])
    if selected:
        limitations.append("Token totals use one cumulative total per observed thread when available, otherwise distinct turn totals or a last-request lower bound; state and notification snapshots are not added together. Cached input and reasoning output are subsets, not additional tokens.")
    if threads - usage_threads:
        limitations.append("Some observed SDK threads have no retained token totals; reported token usage excludes those missing threads.")
    if not selected and observed:
        limitations.append("No usable SDK token-usage snapshot was retained; token counts are unavailable.")
    if "last" in scopes:
        limitations.append("Last-request usage cannot be reconstructed into session usage; only the largest observed last-request snapshot per thread is reported as a lower bound.")
    if unidentified_items:
        limitations.append(f"{unidentified_items} tool notifications lacked item IDs and were excluded because their lifecycle could not be deduplicated safely.")
    if malformed:
        limitations.append(f"{len(malformed)} unreadable or malformed SDK files were skipped; counts may be incomplete.")
    if oversized:
        limitations.append(f"{len(oversized)} SDK files exceeded the 16 MiB per-file read limit and were skipped.")
    if schema_errors:
        limitations.append(f"{schema_errors} SDK item records had an unsupported structure and were skipped.")
    return {
        "source": "codex_sdk",
        "observed": observed,
        "turns": {"observed": len(turns), "started": sum(row["started"] for row in turns.values()), "completed": sum(row["completed"] for row in turns.values()), "failed": sum(row["failed"] for row in turns.values()), "model_requests": None},
        "tools": [{"name": name, **counts} for name, counts in sorted(tool_counts.items())],
        "tool_calls": tool_calls,
        "online_research_calls": online_calls,
        "program_executions": {"observed": program_counts.pop("calls"), **program_counts},
        "tokens": tokens,
        "sources": sources,
        "source_manifest": [source_manifest[path] for path in sources],
        "coverage": {"event_files": all_event_files, "recognized_events": recognized_events, "malformed_files": malformed, "oversized_files": oversized, "schema_errors": schema_errors, "retention_limit": _EVENT_RETENTION, "at_retention_limit": all_event_files >= _EVENT_RETENTION},
        "limitations": limitations,
    }
