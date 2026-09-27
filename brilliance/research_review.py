"""Read-only, evidence-based process statistics for a saved research project.

This is an audit of the research process, not a scientific result generator.
Only canonical project records are read; checkpoint copies and project revisions
are never recursively aggregated. No provider, application worker or Julia import
is needed to review an archive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from .research_review_scope import collect_scope
from .research_review_sdk import collect_sdk_metrics
from .research_review_web import summarize_web_calls


TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
                "reasoning_output_tokens", "cache_write_input_tokens")
FAILED_STATES = {"failed", "timed_out", "error"}


def _rate(numerator, denominator):
    return round(100 * numerator / denominator, 4) if denominator else None


def _tool_category(name):
    short = name.rsplit(".", 1)[-1]
    if name == "web.run" or short in {"search_web", "read_web_page"}:
        return "Online research"
    if short in {"write_program", "patch_program", "write_research_program", "patch_research_program", "apply_patch"}:
        return "Program authoring"
    if short in {"run_program", "run_research_program", "run_long_research_program", "exec_command"}:
        return "Execution"
    if short in {"api_help", "read_expert", "read_artifact", "read_program", "read_research_program"}:
        return "Reference and evidence reading"
    if short in {"update_user", "pause_research", "complete_research"}:
        return "Communication and research control"
    return "Other"


def _tool_error(result, preflight_failed):
    if preflight_failed:
        issues = (result.get("preflight") or {}).get("issues") or []
        return "SourcePreflightError", "; ".join(str(item) for item in issues[:3])[:500]
    error = result.get("error")
    if not error:
        return ("ToolError", "Tool returned ok=false without an error detail.") if result.get("ok") is False else (None, None)
    # Keep the headline, not traceback locals, generated code or file contents.
    headline = str(error).split("BRILLIANCE_FAILURE_LOCALS=", 1)[0].splitlines()[0][:500]
    match = re.match(r"([A-Za-z_][A-Za-z_0-9.]*(?:Error|Exception|Unavailable))\s*:", headline)
    return (match[1] if match else "ToolError"), headline


def _date(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
    except (TypeError, ValueError):
        return None


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _program(value):
    text = str(value or "").replace("\\", "/")
    if "/workspace/" in text:
        text = text.split("/workspace/", 1)[1]
    if text and "/" not in text:
        text = "programs/" + text
    return text


class Evidence:
    def __init__(self, project):
        self.project = project
        self.sources = {}
        self.warnings = []

    def relative(self, path):
        return path.relative_to(self.project).as_posix()

    def read(self, path):
        if not path.is_file():
            return None
        try:
            if not path.resolve().is_relative_to(self.project):
                raise ValueError("file resolves outside the project")
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("record exceeds 64 MiB")
            raw = path.read_bytes()
            self.sources[self.relative(path)] = {
                "path": self.relative(path), "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            return raw
        except (OSError, ValueError) as exc:
            self.warnings.append(f"Cannot read {self.relative(path)}: {exc}")
            return None

    def json(self, path):
        raw = self.read(path)
        if raw is None:
            return {}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data
        except (ValueError, UnicodeError) as exc:
            self.warnings.append(f"Invalid record {self.relative(path)}: {exc}")
            return {}


def _native_sessions(evidence):
    tools = defaultdict(lambda: {"calls": 0, "succeeded": 0, "failed": 0,
                                "unknown": 0, "skipped": 0})
    sessions, operations, responses, rejected = [], [], [], []
    run_names = {}
    duplicates = 0
    seen_responses = set()
    for path in sorted((evidence.project / "workspace/agent_sessions").glob("native-*.json")):
        data = evidence.json(path)
        events = data.get("events")
        if not isinstance(events, list):
            evidence.warnings.append(f"No event list in {evidence.relative(path)}")
            continue
        pending = defaultdict(deque)
        accepted = 0
        rejected_count = 0
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                evidence.warnings.append(f"Non-object event in {evidence.relative(path)}")
                continue
            kind = event.get("kind")
            if kind == "model":
                accepted += 1
            elif kind == "provider_response":
                identity = event.get("response_id")
                if identity and identity in seen_responses:
                    duplicates += 1
                    continue
                if identity:
                    seen_responses.add(identity)
                responses.append({**event, "model": event.get("returned_model") or data.get("model"),
                                  "source": evidence.relative(path)})
            elif kind == "model_output_rejected":
                rejected_count += 1
                rejected.append(event)
            elif kind == "tool_call":
                name = str(event.get("name") or "unknown")
                tools[name]["calls"] += 1
                arguments = event.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                operation = {"id": f"{path.stem}:{index}", "name": name,
                             "program": _program(arguments.get("name") or arguments.get("script"))
                             if name in {"write_program", "patch_program", "read_program", "run_program"} else None,
                             "argument_keys": sorted(str(key) for key in arguments),
                             "source_sha256": arguments.get("source_sha256"),
                             "source_lines": arguments.get("source_lines"),
                             "source": evidence.relative(path), "event_index": index,
                             "outcome": "unknown"}
                if name in {"search_web", "read_web_page"}:
                    operation["_web"] = {"id": operation["id"], "runtime": "native", "tool": name,
                        "arguments": arguments, "result": None, "source": evidence.relative(path),
                        "event_index": index, "result_event_index": None}
                operations.append(operation)
                pending[name].append(operation)
            elif kind == "tool_result":
                name = str(event.get("name") or "unknown")
                result = event.get("result") or {}
                if not isinstance(result, dict):
                    result = {}
                if not pending[name]:
                    # The loop records skipped/unavailable tools as results only.
                    tools[name]["skipped"] += 1
                    continue
                operation = pending[name].popleft()
                preflight = result.get("preflight") or {}
                failed = result.get("ok") is False or (isinstance(preflight, dict) and preflight.get("passed") is False)
                outcome = "failed" if failed else "succeeded" if result.get("ok") is True else "unknown"
                tools[name][outcome] += 1
                operation["outcome"] = outcome
                operation["result_event_index"] = index
                operation["error_class"], operation["error_message"] = _tool_error(result, failed and isinstance(preflight, dict) and preflight.get("passed") is False)
                if "_web" in operation:
                    operation["_web"].update(result=result, result_event_index=index)
                operation["write_saved"] = name in {"write_program", "patch_program"} and result.get("ok") is True
                operation["preflight_rejected"] = isinstance(preflight, dict) and preflight.get("passed") is False
                operation["run_id"] = result.get("run_id")
                operation["job_id"] = result.get("job_id")
                operation["deferred"] = result.get("kind") == "managed_calculation"
                if result.get("run_id"):
                    program = result.get("program") or {}
                    actual = program.get("source") if isinstance(program, dict) else program
                    run_names[result["run_id"]] = _program(actual) or operation["program"]
        for queue in pending.values():
            for operation in queue:
                tools[operation["name"]]["unknown"] += 1
        sessions.append({"source": evidence.relative(path), "runtime": "native",
                         "model": data.get("model"), "accepted_model_responses": accepted,
                         "rejected_model_responses": rejected_count, "events": len(events)})
    return {"tools": [{"name": k, "runtime": "native", **v} for k, v in sorted(tools.items())],
            "sessions": sessions, "operations": operations, "responses": responses,
            "rejected": rejected, "run_names": run_names, "duplicate_response_ids": duplicates,
            "web_calls": [operation["_web"] for operation in operations if "_web" in operation]}


def _usage(record):
    usage = record.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    details_in = usage.get("input_tokens_details", usage.get("prompt_tokens_details")) or {}
    details_out = usage.get("output_tokens_details", usage.get("completion_tokens_details")) or {}
    details_in = details_in if isinstance(details_in, dict) else {}
    details_out = details_out if isinstance(details_out, dict) else {}
    values = {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "cached_input_tokens": details_in.get("cached_tokens", usage.get("prompt_cache_hit_tokens")),
        "reasoning_output_tokens": details_out.get("reasoning_tokens"),
        "cache_write_input_tokens": details_in.get("cache_write_tokens"),
    }
    values = {k: _number(v) for k, v in values.items()}
    if values["total_tokens"] is None and values["input_tokens"] is not None and values["output_tokens"] is not None:
        values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
    return values


def _executions(evidence, native, scope):
    project = evidence.project
    names = dict(native["run_names"])
    for path in sorted((project / "workspace/artifacts").glob("expert-execution-*.json")):
        receipt = evidence.json(path)
        result = receipt.get("result") or {}
        if isinstance(result, dict) and result.get("run_id"):
            program = result.get("program") or {}
            if isinstance(program, dict):
                names[result["run_id"]] = _program(program.get("source"))
    runs = []
    inherited = 0
    for directory in sorted((project / "workspace/artifacts/programs").glob("program-*")):
        if not directory.is_dir():
            continue
        if directory.name in scope["inherited_run_ids"]:
            inherited += 1
            continue
        status = evidence.json(directory / "runtime_status.json")
        result = evidence.json(directory / "result.json")
        starts = sorted((project / "workspace/progress/events" / directory.name).glob("runner-start*.json"))
        launch = evidence.json(starts[0]) if starts else {}
        end = _date(status.get("updated_at") or result.get("completed_at"))
        elapsed = _number(status.get("elapsed_s"))
        start = _date(launch.get("timestamp"))
        if start is None and end is not None and elapsed is not None:
            start = end - timedelta(seconds=elapsed)
        snapshot = evidence.read(directory / "program.py")
        digest = hashlib.sha256(snapshot).hexdigest() if snapshot is not None else None
        program = _program((launch.get("metrics") or {}).get("program")) or names.get(directory.name)
        if not program and digest:
            candidates = {op["program"] for op in native["operations"]
                          if op.get("source_sha256") == digest and op["program"]}
            if len(candidates) == 1:
                program = candidates.pop()
        state = status.get("status") or "unknown"
        if state == "completed" and status.get("returncode") not in (None, 0):
            state = "failed"
        if not status and result:
            # A scientific result can be written before a subsequent crash.
            state = "unknown"
        runs.append({"run_id": directory.name, "program": program,
                     "status": state, "elapsed_s": elapsed, "source_sha256": digest,
                     "started_at": start.isoformat() if start else None,
                     "updated_at": end.isoformat() if end else None,
                     "durable": status.get("durable"), "returncode": status.get("returncode"),
                     "scientific_passed": result.get("scientific_passed"),
                     "request_satisfied": result.get("request_satisfied"),
                     "source": evidence.relative(directory / ("runtime_status.json" if status else "result.json"))})
    return sorted(runs, key=lambda r: (r["started_at"] or "", r["run_id"])), inherited


def _repair_episodes(runs):
    # Recovery is an observable sequence, not an inferred intention. Changed
    # arguments/environment can recover unchanged code; report that distinction.
    pending = {}
    episodes = []
    for run in runs:
        name = run["program"]
        if not name or not run["started_at"]:
            continue
        if run["status"] in FAILED_STATES:
            item = pending.setdefault(name, {"program": name, "failure_run_ids": [],
                "success_run_id": None, "edit_operations": None, "status": "unresolved",
                "source_changed": None, "failure_source_sha256": run["source_sha256"],
                "last_failure_ended_at": run["updated_at"]})
            item["failure_run_ids"].append(run["run_id"])
            item["last_failure_ended_at"] = run["updated_at"]
        elif run["status"] == "completed" and name in pending:
            if not pending[name]["last_failure_ended_at"] or _date(pending[name]["last_failure_ended_at"]) > _date(run["started_at"]):
                continue
            item = pending.pop(name)
            item["status"] = "recovered"
            item["success_run_id"] = run["run_id"]
            if item["failure_source_sha256"] and run["source_sha256"]:
                item["source_changed"] = item["failure_source_sha256"] != run["source_sha256"]
            episodes.append(item)
    return episodes + list(pending.values())


def review_project(project_dir: Path | str) -> dict:
    """Return a reproducible snapshot without modifying the project or its jobs."""
    project = Path(project_dir).resolve()
    if not project.is_dir() or not (project / "revisions").is_dir():
        raise ValueError(f"Not a saved Brilliance project: {project}")
    evidence = Evidence(project)
    revisions = sorted((project / "revisions").glob("*.json"))
    meta = evidence.json(revisions[-1]) if revisions else {}
    state = evidence.json(project / "codex/state.json")
    origin = evidence.json(project / "workspace/RESTORED_CHECKPOINT.json")
    scope = collect_scope(project, meta, origin)
    evidence.warnings.extend(scope["warnings"])
    metrics = []

    def metric(identity, label, value, unit, definition, group, coverage="recorded"):
        metrics.append({"id": identity, "label": label, "value": value, "unit": unit,
                        "definition": definition, "coverage": coverage, "group": group})

    # Human messages are counted only in the canonical public conversation.
    humans, inherited_messages, excluded_messages = [], 0, 0
    conversation_complete = True
    conversation_files = sorted((project / "conversation").glob("*.json"))
    for path in conversation_files:
        exchange = evidence.json(path)
        messages = exchange.get("messages")
        if not isinstance(messages, list):
            if path.name not in scope["inherited_conversation_files"]:
                conversation_complete = False
                evidence.warnings.append(f"Conversation message list unavailable in {path.name}; human totals are unknown.")
            continue
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                if path.name not in scope["inherited_conversation_files"]:
                    conversation_complete = False
                    evidence.warnings.append(f"Malformed conversation message in {path.name}; human totals are unknown.")
                continue
            if message.get("role") != "user":
                continue
            timestamp = _date(message.get("timestamp") or exchange.get("timestamp"))
            if path.name in scope["inherited_conversation_files"]:
                inherited_messages += 1
                continue
            message_meta = message.get("meta") or {}
            if isinstance(message_meta, dict) and (any(message_meta.get(k) for k in ("automatic", "synthetic", "internal"))
                    or message_meta.get("kind") in {"result_review", "coding_repair", "automatic_continuation"}):
                excluded_messages += 1
                continue
            humans.append({"timestamp": timestamp.isoformat() if timestamp else None,
                           "source": evidence.relative(path), "message_index": index})
    initial = int(bool(humans) and not scope["branch"])
    human_count = len(humans) if conversation_files and scope["exact"] and conversation_complete else None
    human_coverage = "canonical conversation only; UI controls and external edits are separate/unlogged"
    metric("human_messages", "Human chat messages", human_count, "count",
           "User-role conversation entries; automatic notices and inherited branch history excluded.", "Human involvement", human_coverage)
    metric("initial_prompts", "Initial research prompts", initial if human_count is not None else None, "count",
           "First human message of an original project; a branch inherits its initial request.", "Human involvement")
    metric("human_interventions", "Follow-up human interventions", human_count - initial if human_count is not None else None, "count",
           "Human chat messages after the initial prompt; repeated 'continue' messages each count.", "Human involvement", human_coverage)
    metric("branch_resume_actions", "Branch creation / Continue actions", int(scope["branch"]), "count",
           "One recorded creation of this branch; not added to human chat messages.", "Human involvement")

    jobs = []
    for directory in sorted((project / "jobs").glob("job-*")):
        versions = sorted((directory / "revisions").glob("*.json"))
        if versions:
            job = evidence.json(versions[-1])
            if job:
                jobs.append(job)
    turns = [j for j in jobs if j.get("action") == "codex_turn" and j.get("state") != "queued"
             and (j.get("worker_pid") or (project / "jobs" / j.get("job_id", "") / "claim.json").exists())]
    explicit = [j for j in turns if (j.get("payload") or {}).get("explicit_request") is True]
    automatic = [j for j in turns if j not in explicit and ((j.get("payload") or {}).get("explicit_request") is False
                 or (j.get("payload") or {}).get("research_mode") in {"result_review", "final_report"}
                 or any((j.get("payload") or {}).get(k) for k in ("review_of", "repair_of", "report_only")))]
    unknown_turns = [j for j in turns if j not in explicit and j not in automatic]
    control_retries = [j for j in turns if (j.get("payload") or {}).get("protocol_retry_of")]
    metric("agent_turns", "Started agent turns", len(turns) if jobs else None, "count",
           "Distinct durable codex_turn jobs with worker PID or claim record; revisions count once.", "Agent activity")
    metric("automatic_turns", "Automatic continuation / review turns", len(automatic) if jobs else None, "count",
           "Turns explicitly marked automatic or linked to review/repair/report handoffs; not optimizer generations.", "Agent activity")
    metric("unclassified_turns", "Turns with unrecorded trigger", len(unknown_turns) if jobs else None, "count",
           "Started turns without sufficient trigger metadata; not assumed autonomous.", "Agent activity")
    metric("explicit_repair_turns", "Scheduled diagnosis / repair turns", sum(bool((j.get("payload") or {}).get("repair_of")) or (j.get("payload") or {}).get("recovery_mode") == "diagnose_and_repair" for j in turns) if jobs else None,
           "count", "Jobs linked to a failed execution via repair_of; separate from inline repairs.", "Repairs")
    metric("protocol_retry_controls", "Recorded model Retry actions", len(control_retries), "count",
           "Durable turns explicitly marked protocol_retry_of; not network retry attempts.", "Human involvement")
    cancellation_files = sorted((project / "jobs").glob("job-*/cancel.request.json"))
    for path in cancellation_files:
        evidence.json(path)
    metric("cancel_requests", "Recorded calculation cancellation requests", len(cancellation_files), "count",
           "Persisted cancellation request files; UI pause history is not durably complete.", "Human involvement")

    native = _native_sessions(evidence)
    sdk = collect_sdk_metrics(project)
    evidence.warnings.extend(sdk.get("limitations", []))
    native_count = sum(s["accepted_model_responses"] for s in native["sessions"])
    rejected_count = sum(s["rejected_model_responses"] for s in native["sessions"])
    metric("native_sessions", "Recorded native agent sessions", len(native["sessions"]), "count",
           "Canonical native trace files, excluding checkpoints and restored copies.", "Agent activity")
    metric("native_model_responses", "Native model reasoning / action steps", native_count if native["sessions"] else None, "count",
           "Accepted model response events; one response may request several tools. Not scientific optimizer iterations.", "Agent activity",
           "recorded responses; provider disconnects and unlogged requests excluded")
    metric("native_rejected_responses", "Rejected native model outputs", rejected_count if native["sessions"] else None, "count",
           "Recorded protocol/output-limit rejections, separate from accepted steps.", "Agent activity")
    if sdk.get("observed"):
        metric("sdk_observed_turns", "SDK turns visible in retained event log", sdk.get("turns", {}).get("observed"), "count",
               "SDK turns are not individual model inference calls; event retention can limit coverage.", "Agent activity", "retained SDK events")
    tools = native["tools"] + [{"runtime": "codex_sdk", "name": t["name"], "calls": t.get("calls", 0),
        "succeeded": t.get("succeeded", 0), "failed": t.get("failed", 0),
        "unknown": t.get("outcome_unknown", 0), "skipped": 0} for t in sdk.get("tools", [])]
    total_calls = sum(tool["calls"] for tool in tools)
    for tool in tools:
        known = tool["succeeded"] + tool["failed"]
        tool.update(category=_tool_category(tool["name"]),
                    success_rate_percent=_rate(tool["succeeded"], known),
                    outcome_coverage_percent=_rate(known, tool["calls"]),
                    call_share_percent=_rate(tool["calls"], total_calls))
    tool_calls = [{"id": op["id"], "runtime": "native", "tool": op["name"],
        **{key: op.get(key) for key in ("outcome", "source", "event_index", "result_event_index",
            "argument_keys", "program", "run_id", "job_id", "error_class", "error_message")}}
        for op in native["operations"]] + sdk.get("tool_calls", [])
    online = summarize_web_calls(native["web_calls"] + sdk.get("online_research_calls", []))
    metrics.extend(online.get("metrics", []))
    evidence.warnings.extend(online.get("warnings", []))
    metric("tool_calls", "Recorded tool invocations", sum(t["calls"] for t in tools) if native["sessions"] or sdk.get("observed") else None,
           "count", "Executed native tool_call events plus deduplicated SDK tool items; replies and display activity are not counted again.", "Tools")
    metric("distinct_tools", "Distinct tools used", len({t["name"] for t in tools if t["calls"]}) if native["sessions"] or sdk.get("observed") else None, "count",
           "Distinct recorded tool names across runtimes.", "Tools")
    known_outcomes = sum(tool["succeeded"] + tool["failed"] for tool in tools)
    metric("tool_success_rate", "Tool invocation success rate", _rate(sum(tool["succeeded"] for tool in tools), known_outcomes), "percent",
           "100 × succeeded / (succeeded + failed); unknown and skipped excluded. A background submission succeeding does not establish calculation success.", "Tools")
    metric("tool_outcome_coverage", "Tool outcome recording coverage", _rate(known_outcomes, total_calls), "percent",
           "100 × calls with known success/failure / recorded invocations; shows uncertainty behind success rates.", "Tools")
    for tool, label in (("write_program", "Program write operations"), ("patch_program", "Program patch operations")):
        ops = [op for op in native["operations"] if op["name"] == tool]
        metric(tool, label, sum(bool(op.get("write_saved")) for op in ops) if native["sessions"] else None, "count",
               "Native file writes/patches with ok=true, including saved files that then fail preflight. Overwrites count as writes; patches are not automatically repairs.", "Programs", "native trace only")
    metric("preflight_rejections", "Tool results failing program preflight", sum(bool(op.get("preflight_rejected")) for op in native["operations"]) if native["sessions"] else None,
           "count", "Explicit preflight passed=false in native tool results; not counted as launched program failures.", "Programs", "native trace only")

    runs, inherited_runs = _executions(evidence, native, scope)
    states = Counter(r["status"] for r in runs)
    succeeded = states["completed"]
    failed = sum(states[k] for k in FAILED_STATES)
    denominator = succeeded + failed
    metric("program_executions", "Archived program execution attempts", len(runs), "count",
           "Unique canonical programs/program-* run directories; background submissions and receipts are deduplicated by run ID.", "Programs")
    metric("program_successes", "Technically successful executions", succeeded, "count",
           "Runner status completed (zero exit where recorded); this does not certify physics acceptance.", "Programs")
    metric("program_failures", "Failed / timed-out executions", failed, "count",
           "Runner status failed, error or timed_out; preflight rejection before launch is not an execution.", "Programs")
    metric("program_success_rate", "Technical execution success rate", round(100 * succeeded / denominator, 4) if denominator else None, "percent",
           "100 × completed / (completed + failed + timed_out + error); running, cancelled and unknown runs excluded.", "Programs")
    metric("program_pending_or_unknown", "Executions outside success-rate denominator", len(runs) - denominator, "count",
           "Running, cancelled, interrupted or missing terminal status; never counted as success.", "Programs")
    metric("scientific_passes", "Executions declaring scientific_passed=true", sum(r["scientific_passed"] is True for r in runs), "count",
           "Saved explicit program declarations only; not independently certified by this review.", "Programs")
    metric("unique_executed_sources", "Distinct executed source snapshots", len({r["source_sha256"] for r in runs if r["source_sha256"]}), "count",
           "SHA-256-distinct archived program.py files; helper dependencies and changing parameters are not source versions.", "Programs")
    files = []
    for path in sorted((project / "workspace/programs").rglob("*")):
        if path.suffix.lower() not in {".py", ".jl"} or not path.is_file():
            continue
        raw = evidence.read(path)
        if raw is not None:
            files.append({"path": evidence.relative(path), "lines": len(raw.splitlines()),
                          "sha256": hashlib.sha256(raw).hexdigest()})
    metric("program_files", "Current saved program files", len(files), "count",
           "Current .py/.jl files under workspace/programs, including inherited helpers; not cumulative writes.", "Programs")
    metric("program_lines", "Current saved program source lines", sum(f["lines"] for f in files), "lines",
           "Physical lines including blanks/comments; counts current files once, not snapshots or generated candidates.", "Programs")
    episodes = _repair_episodes(runs)
    recovered = [episode for episode in episodes if episode["status"] == "recovered"]
    metric("failure_recovery_episodes", "Observed failure → success episodes", len(recovered), "count",
           "One or more failures of the same named program followed by technical success, ordered by launch time. Recovery may be code, input or environment repair.", "Repairs", "inferred from execution sequence; not a causal attribution")
    metric("changed_source_recoveries", "Recovery episodes with changed program source", sum(e["source_changed"] is True for e in recovered), "count",
           "Recovered episodes with different first-failure and recovery program.py hashes; helper-only changes are not visible.", "Repairs", "inferred from archived snapshots")
    metric("unresolved_failure_episodes", "Failure sequences without later success", sum(e["status"] == "unresolved" for e in episodes), "count",
           "Named programs still lacking an observed later successful execution.", "Repairs")

    token_records = [_usage(r) for r in native["responses"]]
    model_groups = defaultdict(list)
    for response in native["responses"]:
        model_groups[response.get("requested_model") or response.get("model") or "not recorded"].append(response)
    models = []
    for model, rows in sorted(model_groups.items()):
        usages = [_usage(row) for row in rows]
        models.append({"model": model, "runtime": "native",
            "returned_models": sorted({str(row["returned_model"]) for row in rows if row.get("returned_model")}),
            "reasoning_efforts": sorted({str(row["requested_reasoning_effort"]) for row in rows if row.get("requested_reasoning_effort")}),
            "sessions": sum(session["model"] == model for session in native["sessions"]),
            "responses": len(rows),
            **{field: sum(u[field] for u in usages if u.get(field) is not None)
               if any(u.get(field) is not None for u in usages) else None for field in TOKEN_FIELDS},
            "coverage": "accepted provider responses with saved metadata; unrecorded/rejected calls excluded"})
    native_tokens = {field: sum(r[field] for r in token_records if r.get(field) is not None)
                     if any(r.get(field) is not None for r in token_records) else None for field in TOKEN_FIELDS}
    covered = sum(r.get("input_tokens") is not None and r.get("output_tokens") is not None for r in token_records)
    rejected_usage = [_usage(r) for r in native["rejected"]]
    # Rejected responses have their own event rather than a provider_response.
    for field in TOKEN_FIELDS:
        values = [r[field] for r in rejected_usage if r.get(field) is not None]
        if values:
            native_tokens[field] = (native_tokens[field] or 0) + sum(values)
    token_totals = {}
    for field in TOKEN_FIELDS:
        values = [native_tokens.get(field), (sdk.get("tokens") or {}).get(field)]
        recorded = [value for value in values if value is not None]
        token_totals[field] = sum(recorded) if recorded else None
    token_fields = {}
    for field, label in (("input_tokens", "Recorded input tokens"), ("output_tokens", "Recorded output tokens"),
                         ("total_tokens", "Recorded total tokens"), ("cached_input_tokens", "Cached input tokens (subset)"),
                         ("reasoning_output_tokens", "Reasoning output tokens (subset)"),
                         ("cache_write_input_tokens", "Cache-write input tokens (provider detail)")):
        field_count = sum(record.get(field) is not None for record in token_records)
        sdk_available = (sdk.get("tokens") or {}).get(field) is not None
        token_fields[field] = {"native_accepted_records": native_count,
            "native_accepted_records_with_value": field_count,
            "native_rejected_records": rejected_count,
            "native_rejected_records_with_value": sum(record.get(field) is not None for record in rejected_usage),
            "sdk_observed": sdk.get("observed", False), "sdk_field_available": sdk_available}
        token_coverage = f"native field recorded for {field_count}/{native_count} accepted responses"
        if rejected_count:
            token_coverage += f"; {token_fields[field]['native_rejected_records_with_value']}/{rejected_count} rejected responses"
        if sdk.get("observed"):
            token_coverage += f"; SDK: {(sdk.get('tokens') or {}).get('scope', 'unavailable') if sdk_available else 'field unavailable'}"
        if field_count < native_count or (sdk.get("observed") and not sdk_available):
            token_coverage += "; observed subtotal only, missing values excluded"
        metric(field, label, token_totals[field], "tokens",
               "Provider-reported usage. Total includes input and output; cached/reasoning fields are not added again. Missing detail remains unknown.", "Token usage", token_coverage)
    metric("token_coverage_percent", "Native response token coverage", round(100 * covered / native_count, 4) if native_count else None, "percent",
           "Accepted native responses with both input and output usage divided by accepted response events; not billing completeness.", "Token usage")

    boundaries = [_date(meta.get("created_at"))]
    boundaries += [_date(j.get("updated_at")) for j in jobs]
    boundaries += [_date(r.get("updated_at")) for r in runs]
    boundaries += [_date(h.get("timestamp")) for h in humans]
    boundaries = [b for b in boundaries if b is not None]
    span = (max(boundaries) - min(boundaries)).total_seconds() if boundaries else None
    metric("observed_wall_time_s", "Observed research time span", span, "seconds",
           "Time from project creation to latest recorded job/run/human activity; includes waiting and idle gaps, not human labor time.", "Timing")
    durations = [r["elapsed_s"] for r in runs if r["elapsed_s"] is not None]
    metric("summed_program_elapsed_s", "Sum of program elapsed times", sum(durations) if durations else None, "seconds",
           "Sum of recorded per-program elapsed seconds; concurrent runs overlap. This is not CPU time or end-to-end wall time.", "Timing")
    environment = evidence.json(project / "workspace/RESEARCH_ENVIRONMENT.json")
    compute = environment.get("compute") or {}
    environment_summary = {"compute": {k: compute[k] for k in ("backend", "local_worker_processes", "threads_per_worker") if k in compute},
        "internet_access": (environment.get("execution") or {}).get("internet_access"),
        "auto_continue": (environment.get("interaction") or {}).get("auto_continue"),
        "max_autonomous_increments": (environment.get("interaction") or {}).get("max_autonomous_increments")}
    if runs and not all(r["program"] for r in runs):
        evidence.warnings.append("Some execution paths could not be identified; their recovery episodes are not counted.")
    if covered < native_count or not token_records:
        evidence.warnings.append("Token recording is incomplete or unavailable; missing usage is not zero.")
    if native["duplicate_response_ids"]:
        evidence.warnings.append(f"Deduplicated {native['duplicate_response_ids']} repeated provider response IDs for token accounting.")
    evidence.warnings.extend([
        "This report counts saved activity, not scientific optimizer generations, candidate evaluations, or proof that a design meets requirements.",
        "Human intervention counts cover saved chat and the explicitly listed durable controls. UI pauses, external edits, off-platform discussions and human labor time are not fully logged.",
        "Tokens are observed provider usage, not a billing reconciliation. Disconnected/unrecorded calls and provider retries may be missing; cached and reasoning tokens are subsets.",
        "Program edits are not necessarily bug fixes. Failure-to-success sequences are observable recovery episodes, not proof that a particular edit caused recovery.",
        "Scheduled repair turns and failure-to-success episodes can overlap; do not add them as independent repair counts.",
        "Counts are a read-only snapshot; concurrent research can change files while this review runs.",
    ])
    if scope["branch"]:
        if scope["exact"]:
            evidence.warnings.append("Metrics are local to this branch. Inherited conversation/executions identified by the source checkpoint manifest are excluded; current saved program inventory may contain inherited files. Ancestor model usage is not summed.")
        else:
            evidence.warnings.append("Branch ancestry could not be resolved exactly. Human intervention counts are unknown; archive execution metrics include possible inherited runs. Native traces and jobs remain local. Do not compare these execution totals as branch-local counts.")
            for row in metrics:
                if row["group"] in {"Programs", "Repairs"}:
                    row["coverage"] = "workspace archive including possible inherited evidence"
    return {"schema_version": "brilliance-research-review/v1",
        "project": {"id": meta.get("id", project.name), "title": meta.get("title", project.name),
            "path": str(project), "created_at": meta.get("created_at"),
            "reviewed_at": datetime.now(timezone.utc).isoformat(), "status": state.get("status", meta.get("status")),
            "branch_origin": {k: origin[k] for k in ("source_project", "source_checkpoint", "continuation") if k in origin},
            "backend": {k: (state.get("backend") or {}).get(k) for k in ("name", "model", "wire_api")}},
        "metrics": metrics, "tools": tools, "tool_calls": tool_calls, "online_research": online,
        "executions": runs, "repair_episodes": episodes,
        "program_files": files, "sessions": native["sessions"], "models": models, "human_messages": humans,
        "token_usage": {"native": native_tokens, "sdk": sdk.get("tokens"), "observed_total": token_totals},
        "environment": environment_summary,
        "coverage": {"inherited_human_messages_excluded": inherited_messages,
            "inherited_executions_excluded": inherited_runs, "synthetic_or_undated_messages_excluded": excluded_messages,
            "native_responses_with_usage": covered, "native_accepted_responses": native_count,
            "token_fields": token_fields, "conversation_complete": conversation_complete,
            "sdk": sdk.get("coverage"), "branch_local": scope["branch"] and scope["exact"],
            "branch_scope_exact": scope["exact"], "branch_source_manifest": scope.get("source_manifest")},
        "sdk": {key: value for key, value in sdk.items() if key != "online_research_calls"},
        "sources": sorted(set(evidence.sources) | set(sdk.get("sources", []))),
        "source_manifest": list({**evidence.sources, **{item['path']: item for item in sdk.get('source_manifest', [])}}.values()),
        "branch_source_manifest": scope.get("source_manifest_record"),
        "warnings": list(dict.fromkeys(evidence.warnings))}


def resolve_project(target: str, root: Path) -> Path:
    candidate = Path(target)
    if target != "latest":
        if candidate.is_dir():
            return candidate.resolve()
        candidate = root / target
        if candidate.is_dir():
            return candidate.resolve()
        raise ValueError(f"Project not found: {target}")
    found = []
    for directory in root.glob("project-*"):
        revisions = sorted((directory / "revisions").glob("*.json"))
        if not revisions:
            continue
        try:
            data = json.loads(revisions[-1].read_text(encoding="utf-8"))
            created = _date(data.get("created_at"))
            if created:
                found.append((created, directory.name, directory))
        except (OSError, ValueError):
            continue
    if not found:
        raise ValueError(f"No saved projects found in {root}")
    return max(found)[2].resolve()


def main(argv=None):
    from .research_review_output import write_review
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", default="latest", help="Project ID, directory, or latest (default)")
    parser.add_argument("--project-root", type=Path, default=Path("research_projects"))
    parser.add_argument("--output", type=Path, help="Output directory (default: research_reviews/PROJECT_ID)")
    args = parser.parse_args(argv)
    try:
        project = resolve_project(args.project, args.project_root.resolve())
        output = (args.output or Path("research_reviews") / project.name).resolve()
        if output.is_relative_to(project):
            raise ValueError("Choose an output directory outside the source project to keep its evidence unchanged.")
        report = review_project(project)
        write_review(report, output)
        print(f"Reviewed {report['project']['id']}")
        for key in ("human_interventions", "agent_turns", "native_model_responses", "tool_calls", "program_success_rate", "failure_recovery_episodes", "total_tokens"):
            row = next(m for m in report["metrics"] if m["id"] == key)
            value = row["value"]
            display = f"{value:,}" if isinstance(value, int) else f"{value:,.2f}" if isinstance(value, float) else "not recorded"
            print(f"{row['label']}: {display} {row['unit']}")
        print(f"Report: {output / 'report.md'}")
        print(f"JSON and CSV audit tables: {output}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Review failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
