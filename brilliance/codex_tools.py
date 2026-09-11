"""Small command surface exposed to the Codex research runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .accelerator_api import evaluate_candidate, load_lattice
from .agent_tools import (
    AgentEvidenceStore,
    agent_tool_catalog,
    compact_agent_evidence,
    execute_agent_tool,
)
from .artifacts import content_hash, json_safe
from .experts import ExpertRegistry, ExpertToolContext, execute_expert_tool
from .experts.execution import resolve_expert_arguments
from .jobs import JobState, ResearchJobStore, launch_research_job
from .julia_runtime import load_pyjutrack
from .jutrack_service import JuTrackService
from .lattice import LatticeIR
from .projects import ProjectStore
from .research_environment import ResearchEnvironment
from .research_program import inspect_research_program
from .research_state import ResearchStateStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brilliance-tool",
        description=(
            "Inspect a Brilliance project, execute saved research programs, and archive "
            "their evidence."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("BRILLIANCE_PROJECT_ROOT", "research_projects"),
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("BRILLIANCE_PROJECT_ID", ""),
        required=not bool(os.environ.get("BRILLIANCE_PROJECT_ID")),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "bootstrap",
        help="Show the prepared accelerator environment, project state, and relevant tool routes",
    )
    commands.add_parser("environment", help="Show configured Python, Julia, and JuTrack paths")
    commands.add_parser("status", help="Show the compact project and active-lattice state")
    check_program = commands.add_parser(
        "check-program",
        help="Syntax-check and lint a saved research program without running JuTrack",
    )
    check_program.add_argument("script")
    commands.add_parser(
        "working-lattice",
        help="Show the current provisional or canonical lattice and its cached evaluation",
    )
    select_working = commands.add_parser(
        "select-working-lattice",
        help="Select a passing archived branch after explicit scientist acceptance",
    )
    select_working.add_argument(
        "execution_artifact",
        help="Passing expert execution JSON under workspace/artifacts",
    )
    commands.add_parser("capabilities", help="List callable deterministic research tools")
    commands.add_parser("experts", help="List installed versioned expert knowledge and local tools")
    expert = commands.add_parser("expert", help="Show one installed expert module")
    expert.add_argument("module_id")
    knowledge = commands.add_parser(
        "expert-knowledge", help="Retrieve bounded installed expertise for a precise question"
    )
    knowledge.add_argument("--query", required=True)
    knowledge.add_argument("--module", default=None)
    expert_run = commands.add_parser(
        "run-expert-tool", help="Run a typed tool contributed by an installed expert module"
    )
    expert_run.add_argument("tool_id")
    expert_run.add_argument("--arguments-json", default="{}")
    expert_run.add_argument(
        "--arguments-file",
        default=None,
        help="UTF-8 JSON file containing one argument object (preferred on Windows)",
    )
    expert_run.add_argument(
        "--full-output",
        action="store_true",
        help="Print the complete execution instead of the compact model-facing receipt",
    )
    expert_submit = commands.add_parser(
        "submit-expert-job",
        help="Submit a long expert calculation as a durable background job",
    )
    expert_submit.add_argument("tool_id")
    expert_submit.add_argument("--arguments-json", default="{}")
    expert_submit.add_argument("--arguments-file", default=None)
    expert_status = commands.add_parser(
        "expert-job-status",
        help="Show managed expert calculations and their latest numerical progress",
    )
    expert_status.add_argument("job_id", nargs="?", default=None)
    expert_status.add_argument(
        "--json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    expert_cancel = commands.add_parser(
        "cancel-expert-job",
        help="Request cooperative cancellation of a managed expert calculation",
    )
    expert_cancel.add_argument("job_id")
    run = commands.add_parser("run-tool", help="Run one registered JuTrack/evidence tool")
    run.add_argument("tool_id")
    run.add_argument("--arguments-json", default="{}")
    run.add_argument(
        "--arguments-file",
        default=None,
        help="UTF-8 JSON file containing one argument object (preferred on Windows)",
    )
    # Tolerate the obvious direct spellings while keeping the model-visible
    # catalog limited to the same two read-only operations.
    for command_name, tool_id in (
        ("inspect-evidence", "inspect_evidence"),
        ("query-evidence", "query_evidence"),
    ):
        direct = commands.add_parser(command_name, help=argparse.SUPPRESS)
        direct.add_argument("--arguments-json", default="{}")
        direct.add_argument("--arguments-file", default=None)
        direct.set_defaults(direct_tool_id=tool_id)

    publish = commands.add_parser(
        "publish-lattice",
        help="Validate and archive a canonical lattice from a JSON specification",
    )
    publish.add_argument("specification")

    evaluate = commands.add_parser(
        "evaluate-lattice",
        help="Evaluate an unpublished full-ring candidate with JuTrack",
    )
    evaluate.add_argument("specification")
    evaluate.add_argument("--no-radiation", action="store_true")

    progress = commands.add_parser(
        "progress",
        help="Publish a concise scientific update to the live research conversation",
    )
    progress.add_argument(
        "--kind",
        choices=("plan", "finding", "decision", "repair", "next"),
        default="finding",
    )
    progress.add_argument("--stage", required=True)
    progress.add_argument("--detail", default="")

    commands.add_parser("artifacts", help="List archived Codex/JuTrack result artifacts")
    commands.add_parser(
        "research-state",
        help="Show the active cross-domain experimental contracts for this project",
    )
    commands.add_parser(
        "plot-optimization",
        help="Render the latest archived optimizer convergence with Matplotlib",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    store = ProjectStore(root)
    project = store.load(args.project_id)
    if project is None:
        raise LookupError(f"Research project not found: {args.project_id}")
    project_directory = root / project.id
    environment = ResearchEnvironment.load()
    expert_registry = ExpertRegistry.from_environment(
        environment, project_directory=project_directory
    )
    if args.command == "bootstrap":
        from .codex_runtime import ensure_codex_workspace

        ensure_codex_workspace(root, project, environment)
        _print_json(_bootstrap(project, root, environment, expert_registry=expert_registry))
        return 0
    if args.command == "environment":
        _print_json(ResearchEnvironment.load().manifest(project_root=root))
        return 0
    if args.command == "plot-optimization":
        from .optimization_plotting import render_latest_optimization_plot

        _print_json(render_latest_optimization_plot(project_directory))
        return 0
    if args.command == "status":
        _print_json(_project_status(project, root))
        return 0
    if args.command == "check-program":
        inspection = inspect_research_program(
            project_directory / "workspace", args.script, expert_registry=expert_registry)
        _print_json(inspection)
        return 0 if inspection["passed"] else 2
    if args.command == "research-state":
        _print_json(ResearchStateStore(project_directory).summary())
        return 0
    if args.command == "select-working-lattice":
        _print_json(
            _select_working_lattice(
                project_directory,
                Path(args.execution_artifact),
                project=project,
                root=root,
            )
        )
        store.snapshot_manifest(project.id)
        return 0
    if args.command == "capabilities":
        tools = [
            item
            for item in agent_tool_catalog(project)
            if item.get("id") != "write_and_run_jutrack_code"
        ]
        _print_json(
            {
                "project_id": project.id,
                "active_lattice_id": project.active_lattice_id,
                "tools": tools,
                "expert_tools": expert_registry.tool_catalog(),
                "coding_fallback": {
                    "available": True,
                    "rule": (
                        "Write and run new bounded JuTrack code in workspace when no registered "
                        "tool matches; archive plots and tables in workspace/artifacts."
                    ),
                },
            }
        )
        return 0
    if args.command == "experts":
        _print_json(expert_registry.catalog(goal=project.goal))
        return 0
    if args.command == "expert":
        _print_json(expert_registry.module(args.module_id).as_dict(include_assets=True))
        return 0
    if args.command == "expert-knowledge":
        _print_json(
            {
                "query": args.query,
                "module": args.module,
                "registry_fingerprint": expert_registry.fingerprint,
                "context": expert_registry.render_context(
                    args.query,
                    max_chars=environment.expert_max_context_chars,
                    module_id=args.module,
                ),
            }
        )
        return 0
    if args.command == "run-expert-tool":
        supplied_arguments = _json_argument_source(
            args.arguments_json, arguments_file=args.arguments_file
        )
        _, declared_tool = expert_registry.tool(args.tool_id)
        context = ExpertToolContext(
            project_root=root,
            project_directory=project_directory,
            workspace=project_directory / "workspace",
            project=project,
            environment=environment,
        )
        arguments, argument_resolution = resolve_expert_arguments(
            declared_tool, supplied_arguments, context=context
        )
        if _managed_expert_job_required(
            declared_tool.timeout_s,
            args.tool_id,
            arguments,
            turn_wall_time_s=environment.turn_wall_time_s,
        ):
            job = _submit_expert_job(
                root,
                project,
                tool_id=args.tool_id,
                arguments=arguments,
                supplied_arguments=supplied_arguments,
                argument_resolution=argument_resolution,
            )
            _print_json(_managed_job_receipt(root, job))
            return 0
        execution = execute_expert_tool(
            expert_registry, args.tool_id, supplied_arguments, context=context
        )
        artifact = context.workspace / "artifacts" / f"{execution['execution_id']}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        rendered = (
            json.dumps(json_safe(execution), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        if artifact.exists():
            if artifact.read_text(encoding="utf-8") != rendered:
                raise RuntimeError(f"expert execution artifact conflict: {artifact}")
        else:
            artifact.write_text(rendered, encoding="utf-8", newline="\n")
        ResearchStateStore(project_directory).record_execution(execution, artifact=artifact)
        _record_working_lattice(project_directory, execution)
        store.snapshot_manifest(project.id)
        rendered_execution = {**execution, "artifact": str(artifact)}
        _print_json(
            rendered_execution
            if args.full_output
            else _compact_expert_execution(rendered_execution)
        )
        return 0 if execution["validation"]["passed"] is True else 2
    if args.command == "submit-expert-job":
        arguments = _json_argument_source(args.arguments_json, arguments_file=args.arguments_file)
        _, declared_tool = expert_registry.tool(args.tool_id)
        context = ExpertToolContext(
            project_root=root,
            project_directory=project_directory,
            workspace=project_directory / "workspace",
            project=project,
            environment=environment,
        )
        normalized, argument_resolution = resolve_expert_arguments(
            declared_tool, arguments, context=context
        )
        job = _submit_expert_job(
            root,
            project,
            tool_id=args.tool_id,
            arguments=normalized,
            supplied_arguments=arguments,
            argument_resolution=argument_resolution,
        )
        _print_json(
            {
                **_managed_job_receipt(root, job),
                "argument_resolution": argument_resolution,
            }
        )
        return 0
    if args.command == "expert-job-status":
        jobs = ResearchJobStore(root)
        selected = (
            [jobs.load(project.id, args.job_id)]
            if args.job_id
            else [item for item in jobs.list(project.id) if item.action == "expert_tool"][-12:]
        )
        if args.job_id and (not selected or selected[0] is None):
            raise LookupError(f"Managed expert calculation not found: {args.job_id}")
        _print_json(
            {
                "project_id": project.id,
                "calculations": [
                    (
                        _managed_job_receipt(root, item)
                        if args.job_id
                        else _managed_job_summary(root, item)
                    )
                    for item in selected
                    if item is not None
                ],
            }
        )
        return 0
    if args.command == "cancel-expert-job":
        job = ResearchJobStore(root).request_cancel(project.id, args.job_id)
        _print_json(_managed_job_receipt(root, job))
        return 0
    if args.command in {"run-tool", "inspect-evidence", "query-evidence"}:
        arguments = _json_argument_source(args.arguments_json, arguments_file=args.arguments_file)
        evidence = execute_agent_tool(
            project,
            getattr(args, "direct_tool_id", None) or args.tool_id,
            arguments,
            project_root=root,
        )
        store.snapshot_manifest(project.id)
        _print_json(compact_agent_evidence(evidence))
        return 0 if bool(evidence.get("deterministic_validation", {}).get("passed")) else 2
    if args.command == "publish-lattice":
        lattice = _load_lattice_specification(Path(args.specification))
        # Publication means executable JuTrack state, not merely schema-valid JSON.
        jt = load_pyjutrack()
        JuTrackService(jt).build_lattice(lattice)
        known = {str(item.get("lattice_id")) for item in project.lattice_history}
        if lattice.lattice_id in known and project.active_lattice_id == lattice.lattice_id:
            updated = project
        else:
            history = tuple(project.lattice_history)
            if lattice.lattice_id not in known:
                history += (lattice.as_dict(),)
            updated = replace(
                project,
                revision=project.revision + 1,
                updated_at=_now(),
                status="researching",
                next_action="Research is continuing from the published JuTrack lattice.",
                model="codex",
                lattice_history=history,
                active_lattice_id=lattice.lattice_id,
            )
            store.save(updated)
        packages = store.ensure_lattice_packages(project.id)
        store.snapshot_manifest(project.id)
        _print_json(
            {
                "accepted": True,
                "validator": "canonical_lattice_schema_and_jutrack_build",
                "project_id": project.id,
                "project_revision": updated.revision,
                "lattice_id": lattice.lattice_id,
                "physical_fingerprint": lattice.physical_fingerprint,
                "element_count": len(lattice.elements),
                "total_length_m": lattice.total_length_m,
                "bend_sum_rad": lattice.bend_sum_rad,
                "periodicity": lattice.periodicity,
                "archive": packages[lattice.lattice_id],
                "scientific_status": "provisional",
            }
        )
        return 0
    if args.command == "evaluate-lattice":
        candidate = _load_lattice_specification(Path(args.specification))
        result = evaluate_candidate(
            candidate,
            include_radiation=not bool(args.no_radiation),
            include_profile=True,
            include_chromaticity=True,
        )
        artifact = (
            root
            / project.id
            / "workspace"
            / "artifacts"
            / f"candidate-evaluation-{candidate.lattice_id}.json"
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(json_safe(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        linear = dict(result.get("linear_optics", {}))
        linear.pop("twiss_profile", None)
        linear.pop("one_turn_matrix", None)
        radiation = dict(result.get("radiation", {}))
        radiation.pop("jutrack_ring_parameters", None)
        geometry = dict(result.get("geometry", {}))
        passed = bool(geometry.get("ring_bend_closure_passed")) and bool(linear.get("stable"))
        _print_json(
            {
                "candidate_lattice_id": candidate.lattice_id,
                "passed_baseline_gate": passed,
                "accepted_project_state": False,
                "inventory": result.get("inventory", {}),
                "geometry": geometry,
                "linear_optics": linear,
                "chromaticity": result.get("chromaticity", {}),
                "radiation": radiation,
                "full_result": str(artifact),
                "requires_design_gate": False,
                "next_action": (
                    "Publish this exact candidate if its assumptions and results are defensible."
                    if passed
                    else "Revise this candidate; it was not published or accepted."
                ),
            }
        )
        return 0 if passed else 2
    if args.command == "progress":
        payload = {
            "schema_version": "1.0",
            "project_id": project.id,
            "timestamp": _now(),
            "kind": args.kind,
            "stage": " ".join(str(args.stage).split())[:240],
            "detail": " ".join(str(args.detail).split())[:1_000],
        }
        digest = content_hash(payload, length=16)
        stamp = "".join(character for character in payload["timestamp"] if character.isdigit())[:20]
        path = root / project.id / "workspace" / "progress" / f"{stamp}-{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
        except FileExistsError:
            pass
        _print_json({**payload, "published": True})
        return 0
    if args.command == "artifacts":
        evidence_store = AgentEvidenceStore(root)
        rows = []
        directory = root / project.id / "agent_evidence"
        for path in sorted(directory.glob("agent-*.json")) if directory.is_dir() else ():
            try:
                evidence = evidence_store.load(project.id, path.stem)
            except (LookupError, OSError, ValueError):
                continue
            validation = evidence.get("deterministic_validation", {})
            rows.append(
                {
                    "artifact_id": path.stem,
                    "tool_id": evidence.get("tool_id"),
                    "created_at": evidence.get("created_at"),
                    "lattice_id": evidence.get("lattice_id"),
                    "passed": validation.get("passed") if isinstance(validation, Mapping) else None,
                    "validator": (
                        validation.get("validator") if isinstance(validation, Mapping) else None
                    ),
                }
            )
        _print_json({"project_id": project.id, "artifacts": rows})
        return 0
    if args.command == "working-lattice":
        _print_json(_working_lattice(project_directory, project=project, root=root))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def _managed_expert_job_required(
    declared_timeout_s: int,
    tool_id: str,
    arguments: Mapping[str, Any],
    *,
    turn_wall_time_s: int,
) -> bool:
    """Keep durable or no-longer-fittable programs outside the model turn."""

    explicitly_long = (
        str(tool_id).casefold() == "accelerator.run_long_research_program"
        or int(declared_timeout_s) > int(turn_wall_time_s)
    )
    if explicitly_long:
        return True
    raw_deadline = str(os.environ.get("BRILLIANCE_MODEL_TURN_DEADLINE_EPOCH", "")).strip()
    if not raw_deadline:
        return False
    try:
        remaining = float(raw_deadline) - time.time()
    except ValueError:
        return False
    requested = int(arguments.get("timeout_s", declared_timeout_s))
    return requested + 30 > remaining


def _submit_expert_job(
    root: Path,
    project: Any,
    *,
    tool_id: str,
    arguments: Mapping[str, Any],
    supplied_arguments: Mapping[str, Any] | None = None,
    argument_resolution: Mapping[str, str] | None = None,
) -> Any:
    from .presentation import ResearchConversationStore
    from .research_recovery import scientist_request
    from .scientific_acceptance import (
        ACCEPTANCE_ENVIRONMENT,
        continued_request_acceptance,
        environment_acceptance_requirements,
        load_expert_registry,
    )

    origin = scientist_request(root, project.id)
    required = (environment_acceptance_requirements() if ACCEPTANCE_ENVIRONMENT in os.environ else
                continued_request_acceptance(str(origin.get("content", "")),
                    ResearchConversationStore(root).messages(project.id),
                    registry=load_expert_registry(project_directory=root / project.id)))
    jobs = ResearchJobStore(root)
    job = jobs.submit(
        project.id,
        project.revision,
        action="expert_tool",
        payload={
            "tool_id": str(tool_id),
            "arguments": json_safe(arguments),
            "supplied_arguments": json_safe(supplied_arguments or {}),
            "argument_resolution": json_safe(argument_resolution or {}),
            "scientist_request": origin,
            "scientific_acceptance_requirements": list(required),
        },
    )
    if job.state is JobState.QUEUED:
        launch_research_job(root, job)
    return jobs.load(project.id, job.job_id) or job


def _managed_job_receipt(root: Path, job: Any) -> dict[str, Any]:
    progress = None
    progress_directory = root / job.project_id / "workspace" / "progress"
    paths = (
        sorted(
            progress_directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if progress_directory.is_dir()
        else []
    )
    for path in paths:
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("compute_job_id") == job.job_id:
            progress = dict(candidate)
            break
    worker_log = root / job.project_id / "jobs" / job.job_id / "worker.log"
    launch_history = ResearchJobStore(root).launch_history(job.project_id, job.job_id)
    latest_launch = launch_history[-1] if launch_history else None
    artifact: Path | None = None
    diagnostic: Path | None = None
    if "artifact:" in str(job.detail):
        artifact = Path(str(job.detail).rsplit("artifact:", 1)[1].strip())
    if "diagnostic:" in str(job.detail):
        diagnostic = Path(str(job.detail).rsplit("diagnostic:", 1)[1].strip())
    completed_result = None
    if artifact is not None and artifact.is_file():
        try:
            execution = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            execution = None
        if isinstance(execution, Mapping):
            completed_result = _compact_expert_execution(
                {
                    **dict(execution),
                    "artifact": str(artifact),
                }
            )
    return {
        "kind": "managed_expert_calculation",
        "job_id": job.job_id,
        "project_id": job.project_id,
        "tool_id": job.payload.get("tool_id"),
        "resolved_arguments": _compact_argument_values(job.payload.get("arguments", {})),
        "supplied_arguments": _compact_argument_values(
            job.payload.get("supplied_arguments", {})
        ),
        "argument_resolution": job.payload.get("argument_resolution", {}),
        "state": job.state.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "worker_pid": job.worker_pid,
        "detail": job.detail,
        "progress": progress,
        "result": completed_result,
        "artifact": str(artifact) if artifact is not None else None,
        "diagnostic": str(diagnostic) if diagnostic is not None else None,
        "worker_log": str(worker_log) if worker_log.is_file() else None,
        "launch": latest_launch,
        "launch_attempts": len(launch_history),
        "continues_without_chat_turn": job.state
        in {
            JobState.QUEUED,
            JobState.RUNNING,
            JobState.CANCEL_REQUESTED,
        },
        "scientist_controls": {
            "status": f"expert-job-status {job.job_id}",
            "cancel": f"cancel-expert-job {job.job_id}",
        },
    }


def _managed_job_summary(root: Path, job: Any) -> dict[str, Any]:
    """Return enough job state for routing without replaying its evidence archive."""

    receipt = _managed_job_receipt(root, job)
    execution = receipt.get("result")
    execution_summary = None
    if isinstance(execution, Mapping):
        result = execution.get("result", {})
        result = result if isinstance(result, Mapping) else {}
        execution_summary = {
            "execution_id": execution.get("execution_id"),
            "execution_valid": execution.get("execution_valid"),
            "scientific_passed": execution.get("scientific_passed"),
            "candidate": _compact_model_value(result.get("candidate", {})),
            "next_action": result.get("next_action"),
            "artifact": execution.get("artifact"),
        }
    return {
        key: receipt.get(key)
        for key in (
            "kind",
            "job_id",
            "tool_id",
            "state",
            "created_at",
            "updated_at",
            "worker_pid",
            "detail",
            "progress",
            "launch",
            "launch_attempts",
            "continues_without_chat_turn",
            "scientist_controls",
        )
    } | {"result": execution_summary}


def _project_status(project: Any, root: Path) -> dict[str, Any]:
    active = next(
        (
            item
            for item in reversed(project.lattice_history)
            if item.get("lattice_id") == project.active_lattice_id
        ),
        None,
    )
    active_summary = None
    if isinstance(active, Mapping):
        lattice = LatticeIR.from_dict(active)
        package = root / project.id / "state" / "lattices" / lattice.lattice_id
        active_summary = {
            "lattice_id": lattice.lattice_id,
            "title": lattice.title,
            "energy_gev": lattice.energy_gev,
            "periodicity": lattice.periodicity,
            "element_count": len(lattice.elements),
            "total_length_m": lattice.total_length_m,
            "bend_sum_rad": lattice.bend_sum_rad,
            "negative_angle_bends": sum(
                (item.angle_rad or 0.0) < 0.0
                for item in lattice.elements
                if item.kind in {"SBEND", "RBEND"}
            ),
            "definition": str(package / "lattice.json"),
            "reproduction": str(package / "reproduce.py"),
        }
    return {
        "project_id": project.id,
        "revision": project.revision,
        "goal": project.goal,
        "context": project.context,
        "status": project.status,
        "active_lattice": active_summary,
        "working_lattice": _working_lattice(root / project.id, project=project, root=root),
        "research_state": ResearchStateStore(root / project.id).summary(),
        "workspace": str(root / project.id / "workspace"),
        "artifacts_directory": str(root / project.id / "workspace" / "artifacts"),
        "managed_calculations": [
            _managed_job_summary(root, item)
            for item in ResearchJobStore(root).list(project.id)
            if item.action == "expert_tool"
        ][-8:],
        "rules": [
            "Treat the newest scientist message as the immediate task.",
            "Use JuTrack for numerical beam-dynamics claims.",
            "Use configured runtimes; do not search for Python, Julia, or JuTrack.",
            "Resolve the current research subject and inherited contract before modifying it.",
            "Use installed expertise when relevant; write bounded code when no tool fits.",
            "Separate output validity from whether a scientific hypothesis or target passed.",
            "Do not edit immutable archive directories directly.",
        ],
    }


def _working_lattice(
    project_directory: Path,
    *,
    project: Any,
    root: Path,
) -> dict[str, Any] | None:
    pointer = project_directory / "workspace" / "WORKING_LATTICE.json"
    if pointer.is_file():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping):
            return dict(payload)
    if project.active_lattice_id:
        package = (
            root
            / project.id
            / "state"
            / "lattices"
            / str(project.active_lattice_id)
            / "lattice.json"
        )
        return {
            "status": "canonical",
            "lattice_id": project.active_lattice_id,
            "lattice_json": str(package),
            "cached_evaluation": None,
        }
    # A passing branch is deliberately not *accepted* merely because a program
    # produced it. It is nevertheless the exact candidate subsequent requests refer to.
    # Exposing that branch here prevents the model from scraping large
    # execution JSON files and guessing which lattice is current.
    latest = _latest_validated_lattice_branch(project_directory)
    if latest is not None:
        return latest
    return None


def _latest_validated_lattice_branch(project_directory: Path) -> dict[str, Any] | None:
    workspace = (project_directory / "workspace").resolve()
    artifact_root = workspace / "artifacts"
    if not artifact_root.is_dir():
        return None
    executions = sorted(
        artifact_root.glob("expert-execution-*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for execution_path in executions:
        try:
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(execution, Mapping):
            continue
        validation = execution.get("validation", {})
        result = execution.get("result", {})
        if not (
            isinstance(validation, Mapping)
            and validation.get("passed") is True
            and isinstance(result, Mapping)
            and result.get("passed") is True
        ):
            continue
        artifacts = result.get("artifacts", {})
        candidate = result.get("candidate", {})
        if not isinstance(artifacts, Mapping) or not isinstance(candidate, Mapping):
            continue
        raw_path = str(artifacts.get("lattice_json", "")).strip()
        if not raw_path:
            continue
        lattice_path = Path(raw_path)
        if not lattice_path.is_absolute():
            lattice_path = workspace / lattice_path
        resolved = lattice_path.resolve()
        try:
            relative = resolved.relative_to(workspace).as_posix()
        except ValueError:
            continue
        if not resolved.is_file():
            continue
        try:
            lattice = load_lattice(resolved)
        except (OSError, ValueError, TypeError, KeyError):
            continue
        candidate_id = str(candidate.get("lattice_id", ""))
        if candidate_id and lattice.lattice_id != candidate_id:
            continue
        evaluation_path = Path(str(artifacts.get("evaluation_summary", "")))
        relative_evaluation = None
        if evaluation_path.is_file():
            try:
                relative_evaluation = evaluation_path.resolve().relative_to(workspace).as_posix()
            except ValueError:
                pass
        return {
            "schema_version": "1.0",
            "status": "latest_validated_branch",
            "accepted": False,
            "source_execution_id": execution.get("execution_id"),
            "source_tool_id": execution.get("tool_id"),
            "lattice_id": lattice.lattice_id,
            "physical_fingerprint": lattice.physical_fingerprint,
            "title": lattice.title,
            "energy_gev": lattice.energy_gev,
            "periodicity": lattice.periodicity,
            "element_count": len(lattice.elements),
            "circumference_m": lattice.total_length_m,
            "lattice_json": relative,
            "cached_evaluation": relative_evaluation,
            "instruction": (
                "Use this exact latest validated branch for follow-up inspection or modification. "
                "It remains unaccepted and must not be described as canonical."
            ),
        }
    return None


def _record_working_lattice(
    project_directory: Path,
    execution: Mapping[str, Any],
) -> None:
    validation = execution.get("validation", {})
    result = execution.get("result", {})
    if not (
        isinstance(validation, Mapping)
        and validation.get("passed") is True
        and isinstance(result, Mapping)
        and result.get("passed") is True
    ):
        return
    # A passing output contract is not enough; the program must explicitly
    # authorize this scientifically passing candidate as the working lattice.
    if str(result.get("working_lattice_action", "branch")).strip().lower() != "advance":
        return
    artifacts = result.get("artifacts", {})
    candidate = result.get("candidate", {})
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(candidate, Mapping)
        or not str(candidate.get("lattice_id", "")).strip()
    ):
        return
    lattice_path = Path(str(artifacts.get("lattice_json", "")))
    if not lattice_path.is_file():
        return
    workspace = (project_directory / "workspace").resolve()
    try:
        relative_lattice = lattice_path.resolve().relative_to(workspace).as_posix()
    except ValueError:
        return
    evaluation_path = Path(str(artifacts.get("evaluation_summary", "")))
    relative_evaluation = None
    if evaluation_path.is_file():
        try:
            relative_evaluation = evaluation_path.resolve().relative_to(workspace).as_posix()
        except ValueError:
            relative_evaluation = None
    lattice = load_lattice(lattice_path)
    payload = {
        "schema_version": "1.0",
        "status": "provisional",
        "updated_at": _now(),
        "source_execution_id": execution.get("execution_id"),
        "source_tool_id": execution.get("tool_id"),
        "lattice_id": lattice.lattice_id,
        "physical_fingerprint": lattice.physical_fingerprint,
        "title": lattice.title,
        "energy_gev": lattice.energy_gev,
        "periodicity": lattice.periodicity,
        "element_count": len(lattice.elements),
        "circumference_m": lattice.total_length_m,
        "lattice_json": relative_lattice,
        "cached_evaluation": relative_evaluation,
        "instruction": (
            "Use this exact working lattice for follow-up inspection and modification; "
            "do not restart from another seed."
        ),
    }
    pointer = workspace / "WORKING_LATTICE.json"
    pointer.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _select_working_lattice(
    project_directory: Path,
    execution_artifact: Path,
    *,
    project: Any,
    root: Path,
) -> dict[str, Any]:
    """Promote one immutable passing branch to the provisional working pointer."""

    workspace = (project_directory / "workspace").resolve()
    artifact_root = (workspace / "artifacts").resolve()
    supplied = execution_artifact
    if not supplied.is_absolute():
        supplied = project_directory / supplied
    resolved = supplied.resolve()
    try:
        relative_execution = resolved.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError(
            "working-lattice selection requires an execution artifact under workspace/artifacts"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"expert execution artifact does not exist: {resolved}")
    execution = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(execution, Mapping):
        raise ValueError("expert execution artifact must contain one JSON object")
    validation = execution.get("validation", {})
    result = execution.get("result", {})
    if not (
        isinstance(validation, Mapping)
        and validation.get("passed") is True
        and isinstance(result, Mapping)
        and result.get("passed") is True
    ):
        raise ValueError("only a passing, deterministically validated branch can be selected")
    artifacts = result.get("artifacts", {})
    candidate = result.get("candidate", {})
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(candidate, Mapping)
        or not str(candidate.get("lattice_id", "")).strip()
    ):
        raise ValueError("execution does not contain a selectable lattice candidate")
    selection_id = (
        "working-selection-"
        + content_hash(
            {
                "execution_id": execution.get("execution_id"),
                "lattice_id": candidate.get("lattice_id"),
            }
        )[:20]
    )
    selection = {
        "schema_version": "1.0",
        "selection_id": selection_id,
        "selected_at": _now(),
        "source_execution_artifact": (
            Path("workspace") / "artifacts" / relative_execution
        ).as_posix(),
        "source_execution_id": execution.get("execution_id"),
        "source_tool_id": execution.get("tool_id"),
        "lattice_id": candidate.get("lattice_id"),
        "physical_fingerprint": candidate.get("physical_fingerprint"),
        "reason": "Explicit scientist acceptance of a passing archived branch.",
    }
    selection_path = artifact_root / f"{selection_id}.json"
    rendered_selection = json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if selection_path.exists():
        previous = json.loads(selection_path.read_text(encoding="utf-8"))
        # Selection time is provenance, but repeated acceptance of the same exact
        # execution must remain idempotent.
        selection["selected_at"] = previous.get("selected_at", selection["selected_at"])
        rendered_selection = (
            json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        if selection_path.read_text(encoding="utf-8") != rendered_selection:
            raise RuntimeError(f"working-lattice selection conflict: {selection_path}")
    else:
        selection_path.write_text(rendered_selection, encoding="utf-8", newline="\n")
    selected_execution = {
        **dict(execution),
        "execution_id": selection_id,
        "tool_id": "brilliance.select_working_lattice",
        "result": {
            **dict(result),
            "working_lattice_action": "advance",
        },
    }
    _record_working_lattice(project_directory, selected_execution)
    working = _working_lattice(project_directory, project=project, root=root)
    if not isinstance(working, Mapping) or working.get("lattice_id") != candidate.get("lattice_id"):
        raise RuntimeError("selected lattice was not recorded as the working lattice")
    return {
        "selected": True,
        "selection": selection,
        "selection_artifact": str(selection_path),
        "working_lattice": working,
    }


def _bootstrap(
    project: Any,
    root: Path,
    environment: ResearchEnvironment,
    *,
    expert_registry: ExpertRegistry | None = None,
) -> dict[str, Any]:
    registry = expert_registry or ExpertRegistry.from_environment(
        environment, project_directory=root / project.id
    )
    catalog = agent_tool_catalog(project)
    available = [str(item["id"]) for item in catalog if item.get("available")]
    command = (
        f"{environment.manifest(project_root=root)['python']['command']} "
        f"-m brilliance.codex_tools --project-root {str(root)!r} --project-id {project.id!r}"
    )
    research_state = ResearchStateStore(root / project.id).summary()
    active = bool(
        project.active_lattice_id
        or _working_lattice(root / project.id, project=project, root=root)
        or research_state.get("contracts")
    )
    return {
        "role": "expert_accelerator_research_agent",
        "project": _project_status(project, root),
        "environment": environment.manifest(project_root=root),
        "prepared_context": {
            "handbook": str(root / project.id / "workspace" / "ACCELERATOR_RESEARCH_HANDBOOK.md"),
            "api_reference": str(root / project.id / "workspace" / "ACCELERATOR_API_REFERENCE.md"),
            "tool_catalog": str(root / project.id / "workspace" / "TOOL_CATALOG.json"),
            "runtime": str(root / project.id / "workspace" / "RESEARCH_ENVIRONMENT.json"),
            "accelerator_api": "brilliance.accelerator_api",
            "expert_catalog": str(root / project.id / "workspace" / "EXPERT_CATALOG.json"),
            "expert_context": str(root / project.id / "workspace" / "EXPERT_CONTEXT.md"),
        },
        "command_prefix": command,
        "select_working_lattice_command": (
            f"{command} select-working-lattice workspace/artifacts/EXPERT_EXECUTION.json"
        ),
        "available_registered_tools": available,
        "installed_expert_modules": [
            {"id": item.id, "version": item.version, "title": item.title}
            for item in registry.modules
        ],
        "available_expert_tools": [item["id"] for item in registry.tool_catalog()],
        "persistent_research_state": research_state,
        "recommended_route": {
            "state": "established_research" if active else "new_research_subject",
            "instruction": (
                "Classify the newest request, resolve its exact subject and persisted contract, "
                "then retrieve only relevant expertise. Use a registered tool when it fits or "
                "write a bounded saved experiment when it does not. Do not infer a workflow "
                "from the original project title."
            ),
            "specialized_resources": {
                "expert_knowledge": f"{command} expert-knowledge --query CURRENT_QUESTION",
                "saved_programs": "Write task-specific Python under workspace/programs/ and execute it through an accelerator program tool.",
                "inspect_evidence": (
                    f"{command} run-tool inspect_evidence --arguments-file "
                    "workspace/evidence_args.json"
                ),
            },
        },
        "prohibited_bootstrap_work": [
            "searching for Python or Julia executables",
            "inspecting pyJuTrack source or signatures",
            "studying demo lattices to learn the API",
            "scanning the Brilliance repository for schemas or tool implementations",
        ],
    }


def _load_lattice_specification(path: Path) -> LatticeIR:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Lattice specification does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Lattice specification must be a JSON object")
    if payload.get("lattice_id"):
        return LatticeIR.from_dict(payload)
    elements = payload.get("elements", payload.get("rows"))
    if not isinstance(elements, list) or not elements:
        raise ValueError("Lattice specification requires a non-empty ordered elements list")
    return LatticeIR.create(
        title=str(payload.get("title", "Generated JuTrack lattice")),
        energy_gev=float(payload["energy_gev"]),
        periodicity=int(payload.get("periodicity", 1)),
        elements=elements,
        source=str(payload.get("source", f"Research source file {resolved.name}")),
        parent_lattice_id=(
            str(payload["parent_lattice_id"])
            if payload.get("parent_lattice_id") is not None
            else None
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def _json_argument(value: str) -> dict[str, Any]:
    rendered = str(value).strip()
    if rendered.startswith("@"):
        raw = Path(rendered[1:]).read_bytes()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            rendered = raw.decode("utf-16")
        else:
            rendered = raw.decode("utf-8-sig")
    parsed = json.loads(rendered or "{}")
    if not isinstance(parsed, Mapping):
        raise ValueError("Tool arguments must be one JSON object")
    return dict(parsed)


def _json_argument_source(
    value: str,
    *,
    arguments_file: str | None,
) -> dict[str, Any]:
    if arguments_file:
        if str(value).strip() not in {"", "{}"}:
            raise ValueError("use either --arguments-json or --arguments-file, not both")
        return _json_argument("@" + str(arguments_file))
    return _json_argument(value)


def _compact_expert_execution(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Bound arbitrary program output while the immutable artifact keeps exact values."""

    validation = execution.get("validation", {})
    validation = validation if isinstance(validation, Mapping) else {}
    arguments = execution.get("arguments", {})
    supplied = execution.get("supplied_arguments", {})
    scientific_passed = validation.get("scientific_passed")
    if not isinstance(scientific_passed, bool):
        result = execution.get("result", {})
        scientific_passed = (
            result.get("passed")
            if isinstance(result, Mapping) and isinstance(result.get("passed"), bool)
            else None
        )
    return {
        key: execution.get(key)
        for key in (
            "schema_version",
            "execution_id",
            "api_version",
            "module_id",
            "module_version",
            "tool_id",
            "authority",
            "started_at",
            "completed_at",
            "artifact",
        )
    } | {
        "argument_keys": sorted(arguments) if isinstance(arguments, Mapping) else [],
        "supplied_argument_keys": sorted(supplied) if isinstance(supplied, Mapping) else [],
        "argument_resolution": execution.get("argument_resolution", {}),
        "arguments_stored_in_full_artifact": True,
        "tool_implementation_hash": execution.get("tool_implementation_hash"),
        "state_contract": _compact_argument_values(execution.get("state_contract", {})),
        "result": _compact_model_value(execution.get("result", {})),
        "validation": _compact_model_value(validation),
        "execution_valid": validation.get("passed") is True,
        "scientific_passed": scientific_passed,
    }

def _compact_argument_values(value: Any) -> Any:
    """Keep model-facing receipts bounded while the immutable job retains exact values."""

    if not isinstance(value, Mapping):
        return json_safe(value)
    compact: dict[str, Any] = {}
    for name, item in value.items():
        safe = json_safe(item)
        try:
            size = len(json.dumps(safe, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            size = 0
        if isinstance(safe, list) and (len(safe) > 16 or size > 6_000):
            compact[str(name)] = {
                "item_count": len(safe),
                "stored_in_full_job_or_execution": True,
            }
        elif isinstance(safe, Mapping) and size > 6_000:
            compact[str(name)] = {
                "field_count": len(safe),
                "stored_in_full_job_or_execution": True,
            }
        else:
            compact[str(name)] = safe
    return compact


def _compact_model_value(value: Any, *, depth: int = 0) -> Any:
    """Recursively bound arbitrary expert output for a model-facing receipt."""

    safe = json_safe(value)
    if depth >= 6:
        if isinstance(safe, Mapping):
            return {
                "field_count": len(safe),
                "stored_in_full_job_or_execution": True,
            }
        if isinstance(safe, list):
            return {
                "item_count": len(safe),
                "stored_in_full_job_or_execution": True,
            }
        return safe
    if isinstance(safe, list):
        if len(safe) > 16:
            return {
                "item_count": len(safe),
                "preview": [
                    _compact_model_value(item, depth=depth + 1) for item in safe[:3]
                ],
                "stored_in_full_job_or_execution": True,
            }
        return [_compact_model_value(item, depth=depth + 1) for item in safe]
    if isinstance(safe, Mapping):
        items = list(safe.items())
        if depth > 0 and len(items) > 32:
            # Preserve actual scalar measurements before large nested tables or
            # profiles. Alphabetical/first-eight selection could hide the most
            # important result solely because its key appeared late.
            preview: dict[str, Any] = {}
            remaining = 6_000
            prioritized = sorted(items, key=lambda pair:
                0 if pair[1] is None or isinstance(pair[1], (bool, int, float))
                else 1 if isinstance(pair[1], str) and len(pair[1]) <= 200 else 2)
            for name, item in prioritized:
                compact = _compact_model_value(item, depth=depth + 1)
                entry_size = len(json.dumps({str(name): compact}, ensure_ascii=False, separators=(",", ":")))
                if entry_size > remaining and isinstance(item, (Mapping, list)):
                    compact = {"field_count" if isinstance(item, Mapping) else "item_count": len(item),
                               "stored_in_full_job_or_execution": True}
                    entry_size = len(json.dumps({str(name): compact}, ensure_ascii=False, separators=(",", ":")))
                if entry_size <= remaining:
                    preview[str(name)] = compact
                    remaining -= entry_size
            return {
                "field_count": len(items),
                "preview": preview,
                "omitted_field_count": len(items) - len(preview),
                "stored_in_full_job_or_execution": True,
            }
        return {
            str(name): _compact_model_value(item, depth=depth + 1)
            for name, item in items
        }
    if isinstance(safe, str) and len(safe) > 2_000:
        return safe[:2_000] + "... [stored in full job or execution]"
    return safe


def _print_json(value: Any) -> None:
    # Keep the command transport ASCII-safe even when a user launches this
    # helper from a legacy Windows console. JSON consumers reconstruct the
    # original scientific symbols from \u escapes without data loss.
    print(json.dumps(json_safe(value), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=True,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
