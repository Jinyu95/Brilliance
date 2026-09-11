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
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from brilliance import accelerator_api as api
from brilliance.artifacts import json_safe
from brilliance.chat_service import ResearchChatService
from brilliance.jobs import ResearchJobStore
from brilliance.lattice import LatticeIR
from brilliance.launcher import _load_project_dotenv
from brilliance.presentation import ResearchConversationStore
from brilliance.research_environment import ResearchEnvironment
from brilliance.research_recovery import scientist_request


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


def scoped_candidate(candidate: LatticeIR) -> tuple[LatticeIR | None, dict[str, Any]]:
    """Reject an isolated unit/reference ring masquerading as the default MBA."""
    cell_scope = candidate.metadata.get("evaluation_scope") == "periodic_cell"
    periods = candidate.metadata.get("intended_repetitions") if cell_scope else candidate.periodicity
    rows = api.one_period_elements(candidate)
    bends = sum(abs(float(row.get("angle_rad") or 0)) > 0 for row in rows)
    angle = sum(float(row.get("angle_rad") or 0) for row in rows)
    checks = {
        "energy_3_gev": math.isclose(candidate.energy_gev, 3.0, abs_tol=1e-12),
        "32_mba_cells": periods == 32,
        "multiple_bends_per_achromat": bends >= 3,
        "cell_bend_closure": math.isclose(angle, 2 * math.pi / 32, rel_tol=0, abs_tol=1e-10),
    }
    if not all(checks.values()):
        return None, {"passed": False, "checks": checks, "periods": periods, "bends_per_cell": bends}
    if cell_scope:
        ring = api.cell_ring(title=candidate.title + " — independent full-ring check",
            energy_gev=candidate.energy_gev, cell=rows, periods=32,
            source=f"Exact repetition of saved candidate {candidate.lattice_id}")
    else:
        ring = candidate
        # Do not replace an inconsistent saved full ring with its first cell.
        count = len(rows)
        fields = ("kind", "length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4")
        pattern = [[getattr(item, key) for key in fields] for item in candidate.elements[:count]]
        checks["identical_explicit_periods"] = (
            len(candidate.elements) == 32 * count and all(
                [[getattr(item, key) for key in fields] for item in candidate.elements[start:start + count]] == pattern
                for start in range(0, len(candidate.elements), count)))
    checks["complete_ring_closure"] = abs(ring.bend_sum_rad - 2 * math.pi) < 1e-9
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


def audit(project_directory: Path, *, verify_jutrack: bool = False) -> dict[str, Any]:
    directory = project_directory.resolve()
    root, project_id = directory.parent, directory.name
    service = ResearchChatService(root)
    project = service.store.load(project_id)
    if project is None:
        raise ValueError("Project not found")
    report: dict[str, Any] = {"project_id": project_id, "passed": False, "checks": {}, "scope":
        "First-use linear MBA candidate and saved chat attachments; not a nonlinear design qualification."}
    checks = report["checks"]
    checks["unchanged_default_prompt"] = project.goal == default_prompt()
    jobs = ResearchJobStore(root).list(project_id)
    active = [job.job_id for job in jobs if job.state.value in {"running", "queued"}]
    report["running_jobs"] = active
    checks["no_running_jobs"] = not active
    report["validation_provenance"] = (
        "This audits saved results and task completion; operator-assisted recovery must be "
        "disclosed separately before claiming a cold unattended pass.")
    conversation = ResearchConversationStore(root).messages(project_id)
    checks["one_persisted_user_request"] = (
        [row["content"] for row in conversation if row.get("role") == "user"] == [default_prompt()])
    report["failed_jobs"] = [job.job_id for job in jobs if job.state.value == "failed"]
    checks["no_terminal_execution_failure"] = bool(jobs) and jobs[-1].state.value == "complete"
    # Only committed executions count. Loose result.json files are not evidence.
    executions = []
    for path in (directory / "workspace/artifacts").glob("expert-execution-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        if isinstance(result, dict) and isinstance(result.get("program"), dict):
            executions.append((path.stat().st_mtime_ns, path, payload))
    candidates = [item for item in executions if isinstance(item[2]["result"].get("candidate"), dict)]
    if not candidates:
        report["reason"] = "No committed candidate. Running work is preserved." if active else "No committed candidate."
        return report
    completed_run = completion_run_id(conversation, scientist_request(root, project_id))
    matching = [item for item in executions
                if item[2].get("result", {}).get("run_id") == completed_run
                and item[2].get("validation", {}).get("passed") is True
                and item[2].get("validation", {}).get("scientific_passed") is True]
    # A semantic completion must reference a committed run. It cannot invent a
    # lattice, upgrade a seed's scope, or bypass the independent physics checks.
    matched_candidates = [item for item in matching
                          if isinstance(item[2]["result"].get("candidate"), dict)]
    _, artifact, execution = max(matched_candidates or candidates, key=lambda item: item[0])
    result = execution["result"]
    receipt = result["result"]
    candidate = LatticeIR.from_dict(result["candidate"])
    report["artifact"] = str(artifact)
    report["summary"] = receipt.get("summary")
    checks["execution_valid"] = execution.get("validation", {}).get("passed") is True
    checks["request_declared_complete"] = receipt.get("request_satisfied") is True or bool(matching)
    report["task_completion_run_id"] = completed_run
    snapshot = Path(result.get("program", {}).get("snapshot", ""))
    output_directory = Path(result.get("output_directory", ""))
    checks["construction_source_preserved"] = snapshot.is_file() and snapshot.is_relative_to(directory)
    checks["candidate_json_preserved"] = (output_directory / str(receipt.get("candidate_lattice", ""))).is_file()
    ring, scope = scoped_candidate(candidate)
    report["candidate_scope"] = scope
    checks["requested_mba_scope"] = scope["passed"]
    checks["table_produced"] = any(table.get("rows") for table in receipt.get("tables", []))
    plots = [output_directory / str(plot["path"]) for plot in receipt.get("plots", []) if isinstance(plot, dict)]
    checks["plot_file_produced"] = any(path.is_file() and path.is_relative_to(output_directory) for path in plots)
    snapshot_messages = service.snapshot(project_id)["messages"]
    attachments = [item for row in snapshot_messages for item in row.get("attachments", [])]
    checks["image_exposed_in_chat"] = any(item.get("type") == "image" for item in attachments)
    checks["table_exposed_in_chat"] = any(item.get("type") == "evidence_summary" for item in attachments)
    # Interpretation must come from the agent, not just the managed formatter.
    checks["agent_interpretation_after_result"] = any(
        row.get("role") == "assistant" and len(row.get("content", "")) >= 80
        and row.get("meta", {}).get("runtime") != "managed_compute"
        and row.get("timestamp", "") > receipt.get("completed_at", "z") for row in conversation)
    checks["independent_jutrack_check"] = False
    if ring is not None and scope["passed"] and verify_jutrack:
        engineering = api.audit_lattice_constraints(ring.elements, energy_gev=ring.energy_gev,
            expected_length_m=ring.total_length_m, expected_angle_rad=2 * math.pi)
        value = api.evaluate(ring, include_radiation=True, include_profile=True)
        profile = value.linear_optics.twiss_profile
        # The default conventional MBA places the insertion boundary at START.
        start = profile[0] if profile else {}
        dx = float(start.get("Dx", math.inf))
        dpx = float(start.get("Dpx", math.inf))
        radiation = value.radiation
        physics = {
            "default_engineering_envelope": engineering["passed"],
            "stable": value.linear_optics.stable,
            "closed_ring": value.geometry["ring_bend_closure_passed"],
            "positive_damping_and_emittance": radiation.get("physical_validity", {}).get("passed") is True,
            "achromatic_start": abs(dx) <= 1e-4 and abs(dpx) <= 1e-4,
        }
        report["independent_jutrack"] = {"checks": physics, "emittance_pm": radiation.get("emittance_pm"),
            "start_dx_m": dx, "start_dpx": dpx, "lattice_id": candidate.lattice_id}
        checks["independent_jutrack_check"] = all(physics.values())
    report["passed"] = all(checks.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--root", type=Path, required=True, help="Dedicated validation project root, not user history")
    check = sub.add_parser("audit")
    check.add_argument("project", type=Path)
    check.add_argument("--verify-jutrack", action="store_true")
    check.add_argument("--output", type=Path)
    args = parser.parse_args()
    _load_project_dotenv()
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Default-task validation")
    if environment.internet_access:
        raise RuntimeError("Disable research internet access for this validation. Model API inference is still needed.")
    if args.command == "start":
        if args.root.resolve() == (Path(__file__).resolve().parents[1] / "research_projects"):
            raise ValueError("Use a separate validation root; leave user projects untouched")
        service = ResearchChatService(args.root)
        created = service.create(default_prompt())
        print(json.dumps({"project": str(service.root / created["project"]["id"]), "prompt": default_prompt()}))
        return 0
    result = json_safe(audit(args.project, verify_jutrack=args.verify_jutrack))
    rendered = json.dumps(result, ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
