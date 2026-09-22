"""Execution and validation for saved, model-authored research programs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_hash, json_safe
from .jutrack_lint import JUTRACK_API_CONTRACT_VERSION, lint_jutrack_api
from .lattice import LatticeIR
from .parallel_safety import lint_worker_factory_captures
from .public_api_lint import lint_public_api_calls

MAX_RESEARCH_PROGRAM_BYTES = 128_000
_LATTICE_PRODUCING_CALLS = frozenset({
    "lattice",
    "periodic_cell",
    "cell_ring",
    "bounded_response_match",
    "bounded_fast_response_match",
    "continuation_match",
    "optimize_lattice_parameters",
    "match_jutrack_line",
    "insert_element_in_drift",
    "insert_element_between",
    "remove_inserted_element",
    "resize_inserted_element",
    "replace_lattice_elements",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _research_failure_context(stderr: str) -> str:
    """Keep the Python cause ahead of trailing native shutdown noise.

    Raw stderr remains archived. This bounded excerpt is used by both the
    scientist-facing error and the narrow coding-error recovery classifier.
    """
    limit = 4000
    marker = "Traceback (most recent call last):"
    locals_match = re.search(r"(?m)^BRILLIANCE_FAILURE_LOCALS=[^\r\n]*", stderr)
    diagnostic = stderr[:locals_match.start()] if locals_match else stderr
    start = diagnostic.rfind(marker)
    if start >= 0:
        traceback = diagnostic[start:]
        exception = re.search(r"(?m)^(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?::[^\r\n]*)?\r?$", traceback)
        if exception:
            # Preserve the exception TYPE and the final calling frame even if
            # the traceback or exception message exceeds the display budget.
            terminal = exception.group(0).rstrip("\r")[:1000]
            frames = traceback[:exception.start()].rstrip()
            room = limit - len(terminal) - 1
            if len(frames) > room:
                prefix = marker + "\n... earlier frames omitted; see archived stderr ...\n"
                frames = prefix + frames[-(room - len(prefix)):]
            selected = frames + "\n" + terminal
            if locals_match and len(selected) + len(locals_match.group(0)) + 1 <= limit:
                selected += "\n" + locals_match.group(0)
            return selected
    # Native aborts or incomplete tracebacks may lack a Python exception line.
    # Retain both ends rather than replacing an early failure with late noise.
    text = stderr.strip()
    if len(text) <= limit:
        return text or "No stderr output; see the archived stdout and process exit code."
    separator = "\n... diagnostic truncated; see archived stderr ...\n"
    head = (limit - len(separator)) // 2
    return text[:head] + separator + text[-(limit - len(separator) - head):]


def _workspace_path(workspace: Path, value: str, *, suffix: str | None = None) -> Path:
    candidate = Path(value)
    # The runner already resolves relative paths from the project workspace, but
    # people and coding agents naturally spell them as ``workspace/programs/x.py``.
    # Accept that unambiguous form instead of producing ``workspace/workspace``.
    if (
        not candidate.is_absolute()
        and candidate.parts
        and candidate.parts[0].casefold() == workspace.name.casefold()
    ):
        candidate = Path(*candidate.parts[1:])
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"research program path escapes workspace: {value}") from exc
    if suffix is not None and resolved.suffix.casefold() != suffix:
        raise ValueError(f"research program must use a {suffix} file: {resolved}")
    return resolved


def _current_lattice(context: Any, workspace: Path) -> LatticeIR | None:
    active = context.active_lattice()
    if active is not None:
        return active
    pointer = workspace / "WORKING_LATTICE.json"
    if not pointer.is_file():
        return None
    try:
        value = json.loads(pointer.read_text(encoding="utf-8"))
        raw_path = str(value.get("lattice_json", "")) if isinstance(value, Mapping) else ""
        lattice_path = _workspace_path(workspace, raw_path, suffix=".json")
        from .accelerator_api import load_lattice

        return load_lattice(lattice_path)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _contains_finish_call(node: ast.AST) -> bool:
    """Return whether an unconditionally evaluated expression calls ``.finish``."""

    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "finish"
        for item in ast.walk(node)
    )


def _normal_finish_states(
    statements: list[ast.stmt], incoming: set[bool] | None = None,
    *, issues: list[str] | None = None,
) -> set[bool]:
    """Conservatively track finish calls along normal top-level control-flow paths."""

    states = set({False} if incoming is None else incoming)
    for statement in statements:
        if not states:
            break
        if isinstance(statement, (ast.Raise, ast.Return)):
            states = set()
            continue
        if isinstance(statement, ast.If):
            body = _normal_finish_states(statement.body, states, issues=issues)
            alternate = (
                _normal_finish_states(statement.orelse, states, issues=issues)
                if statement.orelse
                else set(states)
            )
            states = body | alternate
            continue
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            states = _normal_finish_states(statement.body, states, issues=issues)
            continue
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            # A loop may execute zero times, so retain the incoming path too.
            states |= _normal_finish_states(statement.body, states, issues=issues)
            if statement.orelse:
                states = _normal_finish_states(statement.orelse, states, issues=issues)
            continue
        if isinstance(statement, ast.Try):
            paths = _normal_finish_states(statement.body, states, issues=issues)
            for handler in statement.handlers:
                paths |= _normal_finish_states(handler.body, states, issues=issues)
            if statement.orelse:
                paths = _normal_finish_states(statement.orelse, paths, issues=issues)
            if statement.finalbody:
                paths = _normal_finish_states(statement.finalbody, paths, issues=issues)
            states = paths
            continue
        # Definitions do not execute their bodies at definition time. For all
        # other simple statements, a finish call is on the current normal path.
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if _contains_finish_call(statement):
                if True in states and issues is not None:
                    issues.append(
                        f"line {statement.lineno}: run.finish may overwrite an earlier result on "
                        "the same execution path. finish saves a result; it does not exit Python. "
                        "After an early finish, use raise SystemExit(0) at program scope, or "
                        "return from the enclosing stage function. Use progress for intermediate results."
                    )
                states = {True for _state in states}
    return states


def inspect_research_program(workspace: Path | str, script: str, *, expert_registry: Any = None) -> Mapping[str, Any]:
    """Check known contracts; report uncertain publication analysis as advice.

    Python functions, imported helpers and dynamic keyword mappings make full
    publication control flow undecidable here. The actual worker/result checks
    remain responsible for requiring an output and validating its exact evidence.
    """

    root = Path(workspace).resolve()
    source = _workspace_path(root, script, suffix=".py")
    issues: list[str] = []
    warnings: list[str] = []
    tree: ast.AST | None = None
    source_bytes = 0
    source_lines = 0
    if not source.is_file():
        issues.append(f"research program does not exist: {source}")
    else:
        try:
            raw_source = source.read_bytes()
            source_bytes = len(raw_source)
            rendered_source = raw_source.decode("utf-8-sig")
            source_lines = len(rendered_source.splitlines())
            if source_bytes > MAX_RESEARCH_PROGRAM_BYTES:
                issues.append(
                    f"research program has {source_bytes} bytes; split it below "
                    f"{MAX_RESEARCH_PROGRAM_BYTES} bytes"
                )
            tree = ast.parse(rendered_source, filename=str(source))
        except (OSError, UnicodeError, SyntaxError) as exc:
            issues.append(f"research program source is not valid UTF-8 Python: {exc}")
    calls: set[str] = set()
    return_contracts: dict[str, Any] = {}
    finish_publishes_candidate = False
    if tree is not None:
        issues.extend(lint_jutrack_api(tree))
        issues.extend(lint_worker_factory_captures(tree))
        issues.extend(lint_public_api_calls(tree, registry=expert_registry, project_directory=root.parent,
                                           return_contracts=return_contracts))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name):
                calls.add(function.id)
            elif isinstance(function, ast.Attribute):
                calls.add(function.attr)
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "finish"
                and any(keyword.arg == "candidate_lattice" for keyword in node.keywords)
            ):
                finish_publishes_candidate = True
        if "open_research_run" not in calls:
            warnings.append("open_research_run() is optional; stdout and run output files are archived automatically.")
        if "finish" not in calls:
            warnings.append("run.finish(...) is optional; a successful plain program produces an execution result.")
        elif isinstance(tree, ast.Module) and False in _normal_finish_states(tree.body, issues=warnings):
            warnings.append(
                "Static analysis cannot establish publication on every normal top-level path; "
                "publication inside main(), imported helpers or dynamic calls is permitted. "
                "A successful plain program produces an execution result even without finish()."
            )
        if calls & _LATTICE_PRODUCING_CALLS and not finish_publishes_candidate:
            warnings.append(
                "A lattice-producing call was detected without an explicit candidate_lattice keyword at finish. "
                "If publishing a lattice, pass its exact candidate and evaluation (directly or through a helper); "
                "diagnostic-only programs need not publish a new candidate."
            )
    from .experts import ExpertRegistry
    registry = expert_registry
    if registry is None and "expert_module" in calls:
        registry = ExpertRegistry.current()
    durable_calls = calls & set(getattr(registry, "durable_python_calls", ()))
    from .result_contracts import compact_return_contracts
    contracts, omitted_contracts = compact_return_contracts(return_contracts)
    return {
        "passed": not issues,
        "source": (
            source.relative_to(root).as_posix()
            if source.is_relative_to(root)
            else str(source)
        ),
        "sha256": _sha256(source) if source.is_file() else None,
        "source_bytes": source_bytes,
        "source_lines": source_lines,
        "issues": issues,
        "warnings": warnings,
        "detected_calls": sorted(calls),
        "installed_return_contracts": contracts,
        "omitted_return_contracts": omitted_contracts,
        "durable_required": bool(durable_calls),
        "durable_expert_calls": sorted(durable_calls),
        "has_progress": "progress" in calls,
        "contract": f"saved_research_program_source/{JUTRACK_API_CONTRACT_VERSION}",
    }
def run_research_program(
    context: Any, arguments: Mapping[str, Any], *, durable: bool = False
) -> Mapping[str, Any]:
    """Run one inspectable Python program with immutable inputs and archived outputs."""

    require_python = getattr(context.environment, "require_active_python", None)
    if callable(require_python):
        require_python(operation="Saved research-program runner")
    workspace = Path(context.workspace).resolve()
    source = _workspace_path(workspace, str(arguments["script"]), suffix=".py")
    inspection = inspect_research_program(workspace, str(arguments["script"]))
    if not inspection["passed"]:
        raise ValueError(
            f"research program violates JuTrack API contract "
            f"{JUTRACK_API_CONTRACT_VERSION} or failed source preflight:\n- "
            + "\n- ".join(str(item) for item in inspection["issues"])
        )
    supplied = arguments.get("arguments", {})
    if not isinstance(supplied, Mapping):
        raise ValueError("research program arguments must be an object")
    timeout_s = int(arguments.get("timeout_s", 604_800 if durable else 600))
    if not 1 <= timeout_s <= 604_800:
        raise ValueError("timeout_s must be between 1 second and 7 days")
    identity = {
        "script_sha256": _sha256(source),
        "arguments": json_safe(supplied),
        "project_id": context.project_id,
        "active_lattice_id": getattr(context.project, "active_lattice_id", None),
        "started_at": _now(),
    }
    run_id = "program-" + content_hash(identity, length=20)
    output = context.artifact_path(f"programs/{run_id}")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "program.py"
    shutil.copyfile(source, snapshot)
    arguments_file = output / "arguments.json"
    arguments_file.write_text(
        json.dumps(json_safe(supplied), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result_file = output / "result.json"
    stdout_file = output / "stdout.txt"
    stderr_file = output / "stderr.txt"
    progress_file = workspace / "progress" / f"{run_id}.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    cancel_file = None
    if context.compute_job_id:
        cancel_file = (
            Path(context.project_directory)
            / "jobs"
            / str(context.compute_job_id)
            / "cancel.request.json"
        ).resolve()
    active_lattice_file = output / "active-lattice.json"
    active_lattice = _current_lattice(context, workspace)
    if active_lattice is not None:
        active_lattice_file.write_text(
            json.dumps(active_lattice.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        active_lattice_file = None
    latest_candidate_file = output / "latest-candidate.json"
    latest_candidate = _latest_program_candidate(workspace, exclude=output)
    if latest_candidate is not None:
        shutil.copyfile(latest_candidate, latest_candidate_file)
    else:
        latest_candidate_file = None
    environment = context.environment.project_process_environment(
        workspace, existing=os.environ
    )
    # Retired workflow-policy flags must not leak in from an older host/job.
    environment.pop("BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE", None)
    environment.pop("BRILLIANCE_REQUEST_ACCEPTANCE", None)
    environment.update(
        {
            "BRILLIANCE_WORKSPACE": str(workspace),
            "BRILLIANCE_RUN_OUTPUT": str(output),
            "BRILLIANCE_RUN_ARGUMENTS": str(arguments_file),
            "BRILLIANCE_RUN_RESULT": str(result_file),
            "BRILLIANCE_RUN_PROGRESS": str(progress_file),
            "BRILLIANCE_COMPUTE_JOB_ID": str(context.compute_job_id or ""),
            # Prevent Windows console code pages from corrupting scientific
            # symbols or raising UnicodeEncodeError in an otherwise valid run.
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if cancel_file is not None:
        environment["BRILLIANCE_RUN_CANCEL"] = str(cancel_file)
    if active_lattice_file is not None:
        environment["BRILLIANCE_WORKING_LATTICE"] = str(active_lattice_file)
    if latest_candidate_file is not None:
        environment["BRILLIANCE_LATEST_CANDIDATE_LATTICE"] = str(latest_candidate_file)
    command = [
        str(context.environment.python_executable),
        "-u",
        "-m",
        "brilliance.program_runner",
        str(snapshot),
    ]
    from .research_api import ResearchRun, _write_json

    reporting = ResearchRun(
        workspace=workspace, output_directory=output, arguments_file=arguments_file,
        result_file=result_file, progress_file=progress_file, cancellation_file=cancel_file,
        working_lattice_file=active_lattice_file,
        latest_candidate_lattice_file=latest_candidate_file,
        compute_job_id=context.compute_job_id,
    )
    reporting.progress(
        "Starting saved research program", event_key="runner-start",
        detail="Using the configured simulation runtime. Numerical results will appear as evaluations finish.",
        metrics={"program": str(source.relative_to(workspace)), "durable": durable,
                 "runtime_estimate_s": timeout_s},
    )
    started = time.monotonic()
    estimate_reported = False
    last_status = -30.0
    with stdout_file.open("wb") as stdout_handle, stderr_file.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed - last_status >= 30.0:
                _write_json(output / "runtime_status.json", {
                    "status": "running", "pid": process.pid, "elapsed_s": elapsed,
                    "durable": durable, "runtime_estimate_s": timeout_s,
                    "runtime_estimate_exceeded": elapsed > timeout_s,
                    "stdout_file": str(stdout_file), "stderr_file": str(stderr_file),
                    "updated_at": _now(),
                })
                last_status = elapsed
            if context.is_cancellation_requested():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                _write_json(output / "runtime_status.json", {
                    "status": "cancelled", "pid": process.pid, "elapsed_s": elapsed,
                    "durable": durable, "reason": "explicit_scientist_cancellation", "updated_at": _now(),
                })
                context.raise_if_cancelled()
            if elapsed > timeout_s and durable and not estimate_reported:
                reporting.progress(
                    "Research calculation continues beyond its estimate", kind="next",
                    event_key="runner-runtime-estimate",
                    detail=(f"The {timeout_s}-second estimate has elapsed. The managed calculation "
                            "is still running and was not interrupted. Keep monitoring, or use Cancel "
                            "to stop it explicitly; another Continue message is not needed."),
                    metrics={"elapsed_s": elapsed, "process_id": process.pid,
                             "runtime_estimate_s": timeout_s, "still_running": True},
                )
                estimate_reported = True
            if elapsed > timeout_s and not durable:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                _write_json(output / "runtime_status.json", {
                    "status": "timed_out", "pid": process.pid, "elapsed_s": elapsed,
                    "durable": False, "timeout_s": timeout_s, "updated_at": _now(),
                })
                raise TimeoutError(
                    f"research program exceeded its declared {timeout_s} s runtime; "
                    f"partial files are preserved in {output}"
                )
            time.sleep(0.2)
    duration = time.monotonic() - started
    _write_json(output / "runtime_status.json", {
        "status": "completed" if process.returncode == 0 else "failed",
        "pid": process.pid, "elapsed_s": duration, "returncode": process.returncode,
        "durable": durable, "runtime_estimate_s": timeout_s,
        "runtime_estimate_exceeded": duration > timeout_s, "updated_at": _now(),
    })
    stdout = stdout_file.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_file.read_text(encoding="utf-8", errors="replace")
    if process.returncode != 0:
        raise RuntimeError(
            f"research program exited with code {process.returncode}: {_research_failure_context(stderr)}\n"
            f"Exact archived output directory: {output}\n"
            f"Stdout: {stdout_file}\nStderr: {stderr_file}"
        )
    if not result_file.is_file():
        # Only this invocation's output directory is collected: other jobs may
        # be writing elsewhere in the workspace at the same time.
        generated = [path for path in sorted(output.rglob("*"))
                     if path.is_file() and path.resolve().is_relative_to(output)]
        images = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
        internal = {"program.py", "arguments.json", "result.json", "runtime_status.json",
                    "active-lattice.json", "latest-candidate.json"}
        plots = [{"path": path.relative_to(output).as_posix(), "title": path.stem}
                 for path in generated if path.suffix.lower() in images]
        downloads = [{"path": path.relative_to(output).as_posix(), "title": path.name}
                     for path in generated
                     if path.name not in internal and path.suffix.lower() not in images]
        from .output_tables import output_tables
        _write_json(result_file, {
            "schema_version": "1.0", "status": "completed",
            "summary": "Python program completed. Its output is preserved below.",
            "data": {"stdout": stdout[-20_000:], "stderr": stderr[-20_000:],
                     "returncode": process.returncode, "publication": "automatic_execution_output"},
            "tables": output_tables(stdout), "plots": plots, "downloads": downloads,
            "scientific_passed": None, "request_satisfied": None,
            "use_as_working_lattice": False,
            "limitations": ["Process completion is not scientific validation."],
        })
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if not isinstance(result, Mapping):
        raise ValueError("research program result must be one JSON object")
    files = [
        path.relative_to(output).as_posix()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    candidate = None
    candidate_artifact = None
    raw_candidate = result.get("candidate_lattice")
    if raw_candidate:
        candidate_path = (output / str(raw_candidate)).resolve()
        try:
            candidate_path.relative_to(output)
        except ValueError as exc:
            raise ValueError("candidate lattice escapes the program output directory") from exc
        candidate = LatticeIR.from_dict(
            json.loads(candidate_path.read_text(encoding="utf-8"))
        ).as_dict()
        candidate_artifact = str(candidate_path)
    from .scientific_reporting import normalize_scientific_outcome

    result = normalize_scientific_outcome(result)
    advance = (
        bool(result.get("use_as_working_lattice"))
        and candidate is not None
    )
    return {
        "run_id": run_id,
        # This flag describes execution, never scientific acceptance.
        "passed": True,
        "scientific_passed": result.get("scientific_passed"),
        "working_lattice_action": "advance" if advance else "branch",
        "program": {
            "source": source.relative_to(workspace).as_posix(),
            "snapshot": str(snapshot),
            "sha256": identity["script_sha256"],
            "python": str(context.environment.python_executable),
        },
        "arguments": json_safe(supplied),
        "active_lattice_id": identity["active_lattice_id"],
        "duration_s": duration,
        "returncode": process.returncode,
        "stdout": stdout[-20_000:],
        "stderr": stderr[-20_000:],
        "result": dict(result),
        "output_directory": str(output),
        "files": files,
        "candidate": candidate,
        "artifacts": {
            "lattice_json": candidate_artifact,
            "evaluation_summary": str(result_file),
        },
        "research_contract": {
            "scope": "accelerator.research.program",
            "subject": {"kind": "research_run", "id": run_id},
            "arguments": {
                "script": source.relative_to(workspace).as_posix(),
                "arguments": json_safe(supplied),
            },
            "constraints": {},
            "decisions": [str(result.get("summary", ""))],
        },
    }


def _latest_program_candidate(workspace: Path, *, exclude: Path | None = None) -> Path | None:
    """Return the newest archived candidate, including a scientifically rejected one."""

    artifact_root = (workspace / "artifacts" / "programs").resolve()
    if not artifact_root.is_dir():
        return None
    excluded = exclude.resolve() if exclude is not None else None
    results = sorted(
        artifact_root.glob("*/result.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for result_file in results:
        if excluded is not None and result_file.parent.resolve() == excluded:
            continue
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            relative = str(payload.get("candidate_lattice") or "").strip()
            if not relative:
                continue
            candidate = (result_file.parent / relative).resolve()
            candidate.relative_to(result_file.parent.resolve())
            LatticeIR.from_dict(json.loads(candidate.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        return candidate
    return None


def validate_research_program(
    _context: Any,
    _arguments: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate provenance, result shape, and declared artifact paths."""

    issues: list[str] = []
    output = Path(str(execution.get("output_directory", ""))).resolve()
    program = execution.get("program", {})
    snapshot = Path(str(program.get("snapshot", ""))).resolve() if isinstance(program, Mapping) else Path()
    if not snapshot.is_file() or _sha256(snapshot) != str(program.get("sha256", "")):
        issues.append("archived program snapshot is missing or failed its SHA-256 identity")
    result = execution.get("result")
    if not isinstance(result, Mapping):
        issues.append("program result is not an object")
        result = {}
    if str(result.get("status", "")) not in {"completed", "infeasible", "partial"}:
        issues.append("result status must be completed, infeasible, or partial")
    if not str(result.get("summary", "")).strip():
        issues.append("result summary is empty")
    tables = result.get("tables", [])
    plots = result.get("plots", [])
    downloads = result.get("downloads", [])
    if not isinstance(tables, list):
        issues.append("result tables must be a list")
        tables = []
    if not isinstance(plots, list):
        issues.append("result plots must be a list")
        plots = []
    if not isinstance(downloads, list):
        issues.append("result downloads must be a list")
        downloads = []
    for index, table in enumerate(tables):
        if not isinstance(table, Mapping) or not isinstance(table.get("rows", []), list):
            issues.append(f"table {index} must be an object with a rows list")
        elif any(not isinstance(row, Mapping) for row in table.get("rows", [])):
            issues.append(f"table {index} rows must be objects keyed by column name")
    for index, plot in enumerate(plots):
        if not isinstance(plot, Mapping) or not str(plot.get("path", "")):
            issues.append(f"plot {index} must declare a path")
            continue
        path = (output / str(plot["path"])).resolve()
        try:
            path.relative_to(output)
        except ValueError:
            issues.append(f"plot {index} escapes the program output directory")
            continue
        if not path.is_file():
            issues.append(f"plot {index} does not exist: {plot['path']}")
    for index, download in enumerate(downloads):
        if not isinstance(download, Mapping) or not str(download.get("path", "")):
            issues.append(f"download {index} must declare a path")
            continue
        path = (output / str(download["path"])).resolve()
        try:
            path.relative_to(output)
        except ValueError:
            issues.append(f"download {index} escapes the program output directory")
            continue
        if not path.is_file():
            issues.append(f"download {index} does not exist: {download['path']}")
    candidate_path = result.get("candidate_lattice")
    checked_lattice_id = None
    candidate = None
    if candidate_path:
        path = (output / str(candidate_path)).resolve()
        try:
            path.relative_to(output)
        except ValueError:
            issues.append("candidate lattice escapes the program output directory")
        else:
            try:
                candidate = LatticeIR.from_dict(json.loads(path.read_text(encoding="utf-8")))
                checked_lattice_id = candidate.lattice_id
            except Exception as exc:
                issues.append(f"candidate lattice is invalid: {type(exc).__name__}: {exc}")
    if result.get("use_as_working_lattice") is True and not candidate_path:
        issues.append("use_as_working_lattice=true requires an explicit candidate lattice")
    scientific_passed = result.get("scientific_passed")
    if scientific_passed is not None and not isinstance(scientific_passed, bool):
        issues.append("scientific_passed must be true, false, or null")
    request_satisfied = result.get("request_satisfied")
    if request_satisfied is not None and not isinstance(request_satisfied, bool):
        issues.append("request_satisfied must be true, false, or null")
    candidate_verification = None
    if candidate is not None and result.get("candidate_evaluation"):
        from .candidate_evidence import validate_saved_candidate_evaluation
        candidate_verification = validate_saved_candidate_evaluation(
            candidate, result["candidate_evaluation"], output)
    # The application's validator checks execution and archive integrity only.
    # Explicit scientific reports are preserved, not reinterpreted as policy.
    acceptance = result.get("scientific_acceptance", {})
    if execution.get("returncode") not in (None, 0):
        issues.append("program exited unsuccessfully")
    return {
        "passed": not issues,
        "scientific_passed": scientific_passed if isinstance(scientific_passed, bool) else None,
        "request_satisfied": (
            request_satisfied if isinstance(request_satisfied, bool) else None
        ),
        "scientific_acceptance": acceptance,
        "candidate_verification": candidate_verification,
        "validator": "saved_research_program_contract_v1",
        "issues": issues,
        "checked_values": {
            "program_sha256": program.get("sha256") if isinstance(program, Mapping) else None,
            "returncode": execution.get("returncode"),
            "table_count": len(tables),
            "plot_count": len(plots),
            "download_count": len(downloads),
            "candidate_lattice_id": checked_lattice_id,
            "candidate_evaluation_verified": (
                candidate_verification["passed"] if candidate_verification else None),
        },
        "claim_authority": (
            "Program provenance and output structure are validated. Scientific authority comes "
            "from the methods and JuTrack evidence recorded by the program."
        ),
    }
