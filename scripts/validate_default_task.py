"""First-use release smoke test, separate from the agent's physics workflow.

``start`` submits the unmodified welcome-screen prompt through the chat service.
``audit`` checks the resulting project and independently evaluates its saved MBA.
No lattice templates, user projects, optimizer settings, or prompts are modified.
The observer never cancels running jobs. A pass is a first linear candidate, not
validation of emittance optimality, nonlinear acceptance, or a production machine.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from brilliance import accelerator_api as api
from brilliance.artifacts import json_safe
from brilliance.candidate_evidence import (
    read_numerical_evidence,
    validate_saved_candidate_evaluation,
)
from brilliance.chat_service import ResearchChatService
from brilliance.jobs import ResearchJobStore
from brilliance.lattice import LatticeIR
from brilliance.launcher import _load_project_dotenv
from brilliance.presentation import ResearchConversationStore
from brilliance.research_environment import ResearchEnvironment
from brilliance.research_recovery import scientist_request
from brilliance.scientific_acceptance import (
    assess_scientific_acceptance,
    request_acceptance_requirements,
)

if __package__:
    from .release_trial_support import (
        MANIFEST,
        begin_manifest,
        browser_checks,
        digest,
        fresh_checks,
        observe_job_liveness,
        read_json,
        selected_presentation,
        trace_audit,
    )
else:
    from release_trial_support import (  # direct configured-Python script invocation
        MANIFEST,
        begin_manifest,
        browser_checks,
        digest,
        fresh_checks,
        observe_job_liveness,
        read_json,
        selected_presentation,
        trace_audit,
    )


class _Examples(HTMLParser):
    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "button" and "data-prompt" in values:
            self.prompts.append(values["data-prompt"])


def default_prompt() -> str:
    parser = _Examples()
    parser.feed((Path(__file__).resolve().parents[1] / "brilliance/web/static/index.html").read_text(encoding="utf-8"))
    return parser.prompts[0]


def scoped_candidate(candidate: LatticeIR, requirements: dict) -> tuple[LatticeIR | None, dict[str, Any]]:
    """Reject an isolated unit/reference ring masquerading as the default MBA."""
    cell_scope = candidate.metadata.get("evaluation_scope") == "periodic_cell"
    periods = candidate.metadata.get("intended_repetitions") if cell_scope else candidate.periodicity
    rows = api.one_period_elements(candidate)
    bends = sum(abs(float(row.get("angle_rad") or 0)) > 0 for row in rows)
    angle = sum(float(row.get("angle_rad") or 0) for row in rows)
    checks = {
        "requested_energy": math.isclose(candidate.energy_gev, requirements["energy_gev"], rel_tol=0, abs_tol=1e-12),
        "requested_cell_count": periods == requirements["periods"],
        "requested_bends": bends == requirements["expected_bends"] if "expected_bends" in requirements else bends >= 3,
        "cell_bend_closure": math.isclose(angle, 2 * math.pi / requirements["periods"], rel_tol=0, abs_tol=1e-10),
    }
    if not all(checks.values()):
        return None, {"passed": False, "checks": checks, "periods": periods, "bends_per_cell": bends}
    if cell_scope:
        ring = api.cell_ring(title=candidate.title + " — independent full-ring check",
            energy_gev=candidate.energy_gev, cell=rows, periods=requirements["periods"],
            source=f"Exact repetition of saved candidate {candidate.lattice_id}")
    else:
        ring = candidate
        # Do not replace an inconsistent saved full ring with its first cell.
        count = len(rows)
        fields = ("kind", "length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4")
        provenance = {"source_element", "source_element_name", "source_period", "mirror_source_element", "family", "design_role"}
        def physical(item):
            return [*[getattr(item, key) for key in fields],
                    {key: value for key, value in item.fields.items() if key not in provenance}]
        pattern = [physical(item) for item in candidate.elements[:count]]
        checks["identical_explicit_periods"] = (
            len(candidate.elements) == requirements["periods"] * count and all(
                [physical(item) for item in candidate.elements[start:start + count]] == pattern
                for start in range(0, len(candidate.elements), count)))
    checks["complete_ring_closure"] = abs(ring.bend_sum_rad - 2 * math.pi) < 1e-9
    if "circumference_m" in requirements:
        checks["requested_circumference"] = math.isclose(ring.total_length_m, requirements["circumference_m"], rel_tol=0, abs_tol=1e-9)
    if "maximum_circumference_m" in requirements:
        checks["requested_circumference_cap"] = ring.total_length_m <= requirements["maximum_circumference_m"]
    return ring, {"passed": all(checks.values()), "checks": checks,
                  "bends_per_cell": bends, "circumference_m": ring.total_length_m}


def completion_run_id(conversation: list[dict[str, Any]], origin: dict[str, str]) -> str | None:
    """Recognize a separate task decision, never rewrite a numerical receipt."""
    for row in reversed(conversation):
        proof = row.get("meta", {}).get("completion_evidence")
        if not isinstance(proof, dict):
            continue
        if (row.get("role") == "assistant"
                and proof.get("kind") == "scientist_request_completion_from_saved_evidence"
                and proof.get("scientist_request", {}).get("id") == origin.get("id")
                and proof.get("execution_valid") is True
                and proof.get("scientific_passed") is True
                and len(str(proof.get("evidence_receipt_sha256", ""))) == 64):
            return str(proof.get("evidence_run_id") or "") or None
    return None


def audit(project_directory: Path, *, verify_jutrack: bool = False,
          browser_receipt: Path | None = None, environment=None) -> dict[str, Any]:
    directory = project_directory.resolve()
    root, project_id = directory.parent, directory.name
    service = ResearchChatService(root)
    project = service.store.load(project_id)
    if project is None:
        raise ValueError("Project not found")
    report: dict[str, Any] = {"project_id": project_id, "passed": False, "status": "failed", "checks": {}, "scope":
        "Fresh unattended first full-MBA candidate, selected evidence and rendered chat; not nonlinear or optimal design qualification."}
    checks = report["checks"]
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        report["reason"] = "No fresh-trial manifest. A historical/component replay cannot earn an unattended-release pass."
        return report
    manifest = read_json(manifest_path)
    environment = environment or ResearchEnvironment.load()
    checks.update(fresh_checks(manifest, directory, environment))
    checks["unchanged_trial_prompt"] = project.goal == manifest["prompt"]
    report["elapsed_s"] = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(manifest["started_at"])).total_seconds())
    jobs = ResearchJobStore(root).list(project_id)
    report["job_liveness"] = observe_job_liveness(directory, jobs)
    active = report["job_liveness"]["active_job_ids"]
    report["running_jobs"] = active
    checks["no_stale_workers"] = not report["job_liveness"]["stale_job_ids"]
    checks["active_worker_pid_available"] = not report["job_liveness"]["unverifiable_job_ids"]
    checks["no_running_jobs"] = not active
    report["validation_provenance"] = "New isolated project; source/configuration freeze and model-authored executed source checked. Internal automatic repairs are counted, not hidden."
    conversation = ResearchConversationStore(root).messages(project_id)
    checks["one_persisted_user_request"] = (
        [row["content"] for row in conversation if row.get("role") == "user"] == [manifest["prompt"]])
    report["failed_jobs"] = [job.job_id for job in jobs if job.state.value == "failed"]
    checks["no_terminal_execution_failure"] = bool(jobs) and jobs[-1].state.value == "complete"
    # Only committed executions count. Loose result.json files are not evidence.
    executions = []
    for path in (directory / "workspace/artifacts").glob("expert-execution-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        if isinstance(result, dict) and isinstance(result.get("program"), dict):
            executions.append((path.stat().st_mtime_ns, path, payload))
    report["process"] = trace_audit(directory, jobs, manifest["prompt"], [item[2] for item in executions])
    checks.update(report["process"]["checks"])
    if not checks["no_stale_workers"] or not checks["active_worker_pid_available"]:
        report.update(status="failed", reason=(
            "Stored active-job metadata does not identify a live worker. The worker exited, could not be observed, "
            "or never claimed its job; inspect the recorded worker log. Other live jobs, if any, were left running. "
            "No job was resumed, cancelled, or changed by this observer."))
        return report
    if active:
        report.update(status="pending", reason="Research or independent numerical jobs are still active. This observer never cancels them.")
        return report
    completed_run = completion_run_id(conversation, scientist_request(root, project_id))
    matching = [item for item in executions
                if item[2].get("result", {}).get("run_id") == completed_run
                and item[2].get("validation", {}).get("passed") is True
                and item[2].get("validation", {}).get("scientific_passed") is True]
    # Never replace the selected result with the most recent file or an earlier
    # successful component when the model did not actually close the request.
    matched_candidates = [item for item in matching
                          if isinstance(item[2]["result"].get("candidate"), dict)]
    checks["selected_completion_receipt"] = len(matched_candidates) == 1
    if not checks["selected_completion_receipt"]:
        report["reason"] = "No unique committed complete candidate selected by the agent's final answer. Intermediate success is not request completion."
        return report
    _, artifact, execution = matched_candidates[0]
    result = execution["result"]
    receipt = result["result"]
    candidate = LatticeIR.from_dict(result["candidate"])
    report["artifact"] = str(artifact)
    report["summary"] = receipt.get("summary")
    checks["execution_valid"] = execution.get("validation", {}).get("passed") is True
    checks["request_declared_complete"] = bool(matching)
    report["task_completion_run_id"] = completed_run
    snapshot = Path(result.get("program", {}).get("snapshot", "")).resolve()
    output_directory = Path(result.get("output_directory", "")).resolve()
    checks["output_in_fresh_project"] = output_directory.is_relative_to(directory)
    if not checks["output_in_fresh_project"]:
        report["reason"] = "Selected output is outside this fresh project."
        return report
    checks["construction_source_preserved"] = snapshot.is_file() and snapshot.is_relative_to(directory)
    candidate_path = (output_directory / str(receipt.get("candidate_lattice", ""))).resolve()
    result_path = output_directory / "result.json"
    checks["candidate_json_preserved"] = candidate_path.is_relative_to(output_directory) and candidate_path.is_file()
    checks["committed_result_bytes_match"] = result_path.is_file() and read_json(result_path) == receipt
    checks["committed_candidate_bytes_match"] = checks["candidate_json_preserved"] and read_json(candidate_path) == result["candidate"]
    if not checks["committed_result_bytes_match"] or not checks["committed_candidate_bytes_match"]:
        report["reason"] = "The selected saved result/candidate disagrees with the committed execution."
        return report
    verified = validate_saved_candidate_evaluation(candidate, receipt.get("candidate_evaluation"), output_directory)
    checks["exact_native_candidate_receipt"] = verified["passed"]
    report["candidate_id"] = candidate.lattice_id
    report["result_sha256"] = digest(result_path)
    ring, scope = scoped_candidate(candidate, manifest["requirements"])
    report["candidate_scope"] = scope
    checks["requested_mba_scope"] = scope["passed"]
    checks["table_produced"] = any(table.get("rows") for table in receipt.get("tables", []))
    plots = [output_directory / str(plot["path"]) for plot in receipt.get("plots", []) if isinstance(plot, dict)]
    checks["plot_file_produced"] = any(path.is_file() and path.is_relative_to(output_directory) for path in plots)
    proof = next(row["meta"]["completion_evidence"] for row in reversed(conversation)
                 if (row.get("meta", {}).get("completion_evidence") or {}).get("evidence_run_id") == completed_run)
    # _messages is the same read-only presentation assembler used by snapshot;
    # do not call jobs.active/recover from this audit (they can alter job state).
    snapshot_messages = service._messages(project, jobs[-1] if jobs else None)
    checks.update(selected_presentation(snapshot_messages, proof, candidate.lattice_id))
    checks.update(browser_checks(browser_receipt, directory=directory, candidate_id=candidate.lattice_id,
                                 run_id=completed_run, result_path=result_path, source_root=Path(environment.source_root)))
    required = request_acceptance_requirements(manifest["prompt"])
    acceptance = assess_scientific_acceptance(receipt.get("scientific_acceptance", {}), required=required,
                                              lattice=candidate, evidence_directory=output_directory)
    checks["current_expert_request_acceptance"] = bool(required) and acceptance["passed"]
    report["scientific_acceptance"] = acceptance
    machine = None
    # The original limits must come from exact independently checked numerical
    # evidence, not the candidate's measured dimensions recycled as targets.
    for item in receipt.get("scientific_acceptance", {}).values():
        if isinstance(item, dict) and item.get("measurement_receipt"):
            measured = read_numerical_evidence(candidate, item["measurement_receipt"], output_directory=output_directory)
            if isinstance(measured.get("machine_contract"), dict):
                machine = measured["machine_contract"]
                break
    checks["measured_machine_contract_present"] = machine is not None
    report["machine_contract"] = machine
    checks["independent_jutrack_check"] = False
    if ring is not None and scope["passed"] and verify_jutrack and machine is not None and verified["passed"]:
        engineering = api.audit_lattice_constraints(api.one_period_elements(candidate), energy_gev=machine["energy_gev"],
            expected_length_m=machine["expected_cell_length_m"], expected_angle_rad=machine["expected_cell_angle_rad"],
            expected_bends=machine["expected_bends"], require_mirror_symmetry=True, limits=machine["limits"],
            minimum_magnet_length_m=machine["minimum_magnet_length_m"], minimum_drift_length_m=machine["minimum_drift_length_m"])
        value = api.evaluate(ring, include_radiation=True, include_profile=True)
        profile = value.linear_optics.twiss_profile
        # The default conventional MBA places the insertion boundary at START.
        start = profile[0] if profile else {}
        dx = float(start.get("Dx", math.inf))
        dpx = float(start.get("Dpx", math.inf))
        radiation = value.radiation
        physics = {
            "original_engineering_envelope": engineering["passed"],
            "stable": value.linear_optics.stable,
            "closed_ring": value.geometry["ring_bend_closure_passed"],
            "positive_damping_and_emittance": radiation.get("physical_validity", {}).get("passed") is True,
            "achromatic_start": abs(dx) <= machine["achromat_dx_tolerance_m"] and abs(dpx) <= machine["achromat_dpx_tolerance"],
        }
        report["independent_jutrack"] = {"checks": physics, "emittance_pm": radiation.get("emittance_pm"),
            "start_dx_m": dx, "start_dpx": dpx, "lattice_id": candidate.lattice_id}
        checks["independent_jutrack_check"] = all(physics.values())
    report["passed"] = all(checks.values())
    report["status"] = "passed" if report["passed"] else "failed"
    if not report["passed"] and all(value for key, value in checks.items()
            if key not in {"independent_jutrack_check", "browser_selected_result_rendered"}):
        report.update(status="pending", reason="Saved execution passed; independent JuTrack and rendered-browser verification are still pending.")
    report["failed_checks"] = [name for name, passed in checks.items() if not passed]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--root", type=Path, required=True, help="Dedicated validation project root, not user history")
    start.add_argument("--prompt", help="Exact ordinary user prompt; never supplemented with test instructions")
    start.add_argument("--requirements", type=Path, help="Audit-only JSON requirements; required for a custom prompt")
    check = sub.add_parser("audit")
    check.add_argument("project", type=Path)
    check.add_argument("--verify-jutrack", action="store_true")
    check.add_argument("--output", type=Path)
    check.add_argument("--browser-receipt", type=Path, help="JSON receipt from tests/scientific_history_ui.cjs")
    observe = sub.add_parser("watch", help="Read-only bounded observer; never cancels or steers any job")
    observe.add_argument("project", type=Path)
    observe.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    _load_project_dotenv()
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Default-task validation")
    if environment.internet_access:
        raise RuntimeError("Disable research internet access for this validation. Model API inference is still needed.")
    if args.command == "start":
        if args.prompt is not None and args.requirements is None:
            raise ValueError("A custom prompt needs an audit-only --requirements JSON; do not infer unspecified machine requirements")
        requirements = read_json(args.requirements) if args.requirements else {"energy_gev": 3.0, "periods": 32}
        validate_requirements(requirements)
        prompt = args.prompt if args.prompt is not None else default_prompt()
        begin_manifest(args.root, prompt, requirements, environment)
        service = ResearchChatService(args.root)
        created = service.create(prompt)
        launch = {"project": str(service.root / created["project"]["id"]), "prompt": prompt,
                  "action": created["action"], "job_id": (created.get("job") or {}).get("id"),
                  "manifest": str(args.root / MANIFEST)}
        with (args.root / "release_launch.json").open("x", encoding="utf-8") as handle:
            json.dump(launch, handle, indent=2)
        print(json.dumps(launch))
        return 0
    if args.command == "watch":
        if not math.isfinite(args.seconds) or args.seconds < 0:
            raise ValueError("Observer duration must be finite and nonnegative")
        deadline = time.monotonic() + args.seconds
        while True:
            result = audit(args.project, environment=environment)
            print(json.dumps({key: result.get(key) for key in ("status", "reason", "elapsed_s", "running_jobs", "process")}), flush=True)
            if not result.get("running_jobs") or time.monotonic() >= deadline:
                print("Observer ended. No running calculation was interrupted or altered.")
                return 0
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    result = json_safe(audit(args.project, verify_jutrack=args.verify_jutrack,
                            browser_receipt=args.browser_receipt, environment=environment))
    rendered = json.dumps(result, ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 2 if result.get("status") == "pending" else 1


def validate_requirements(requirements: dict) -> None:
    allowed = {"energy_gev", "periods", "expected_bends", "circumference_m", "maximum_circumference_m"}
    if set(requirements) - allowed or not {"energy_gev", "periods"} <= requirements.keys():
        raise ValueError("Audit requirements need energy_gev and periods; optional expected_bends, circumference_m or maximum_circumference_m only")
    for name, value in requirements.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
        if name in {"periods", "expected_bends"} and (not isinstance(value, int) or (name == "expected_bends" and value < 3)):
            raise ValueError(f"{name} must be an integer (MBA bend count at least 3)")


if __name__ == "__main__":
    raise SystemExit(main())
