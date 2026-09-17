"""Opt-in complete geometry/strength optimization in Brilliance's real worker."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from brilliance.research_environment import ResearchEnvironment
from brilliance.research_program import (
    inspect_research_program,
    run_research_program,
    validate_research_program,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research_projects/project-ab41b8de97b4c4/workspace/artifacts/programs/program-cdc6de955273696dd0f0/candidate_lattice.json"
FINGERPRINT = "7f1948f55ad61417350605625b34d97d5e28f0820318a9c7f5c4de24f29bc893"
PROGRAM = Path(__file__).with_name("fixtures") / "native_tme_geometry_program.py"
RESTORE_PROGRAM = Path(__file__).with_name("fixtures") / "native_tme_geometry_restore_program.py"


def test_geometry_optimizer_is_a_durable_public_expert_operation(tmp_path):
    shutil.copyfile(PROGRAM, tmp_path / "program.py")
    inspection = inspect_research_program(tmp_path, "program.py")
    assert inspection["passed"], inspection["issues"]
    assert inspection["durable_required"] is True
    assert inspection["durable_expert_calls"] == ["optimize_tme_unit_geometry"]


@pytest.mark.skipif(os.environ.get("BRILLIANCE_RUN_TME_GEOMETRY_TESTS") != "1",
                    reason="Opt-in production-budget JuTrack geometry optimization")
def test_joint_geometry_search_native_worker_and_publication(tmp_path, monkeypatch):
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Native unit geometry optimization")
    assert not environment.internet_access
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == FINGERPRINT
    project_id = "project-tme-geometry-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    output_root = Path(os.environ.get("BRILLIANCE_TME_GEOMETRY_OUTPUT", str(tmp_path)))
    workspace = output_root / project_id / "workspace"
    (workspace / "inputs").mkdir(parents=True)
    shutil.copyfile(SOURCE, workspace / "inputs/unit.json")
    shutil.copyfile(PROGRAM, workspace / "program.py")
    context = SimpleNamespace(
        workspace=workspace, project_directory=workspace.parent, project_id=project_id,
        compute_job_id="tme-geometry-test", project=SimpleNamespace(active_lattice_id=None),
        active_lattice=lambda: None, is_cancellation_requested=lambda: False,
        artifact_path=lambda name: workspace / "artifacts" / name, environment=environment,
    )
    arguments = {"script": "program.py", "arguments": {}, "timeout_s": 1800}
    execution = run_research_program(context, arguments, durable=True)
    validation = validate_research_program(context, arguments, execution)
    assert execution["returncode"] == 0, execution
    assert validation["passed"], validation
    assert validation["candidate_verification"]["passed"], validation
    output = Path(execution["output_directory"])
    search = json.loads((output / "geometry_search.json").read_text(encoding="utf-8"))
    assert search["passed"] and len(search["candidates"]) >= 2
    assert search["resumed_completed_result"] is True
    assert search["initialization"]["all_feasible"] is True
    assert search["initialization"]["feasible_particles"] >= 500
    optimizer = search["optimizer"]
    assert optimizer["budget"]["effective"]["population_size"] >= 500
    assert optimizer["budget"]["effective"]["generations"] >= 80
    assert optimizer["convergence"]["initial_feasible_count"] == optimizer["budget"]["effective"]["population_size"]
    assert optimizer["evaluation_cache"]["enabled"] is False
    assert optimizer["evaluation_cache"]["writes"] == 0
    original = api.one_period_elements(api.load_lattice(SOURCE))
    lengths = []
    for row in search["candidates"]:
        assert all(row["checks"].values()), row["checks"]
        elements = row["elements"]
        assert len(elements) == 11 and len(row["boundary_twiss"]) == 8
        assert math.fsum(e["length_m"] for e in elements) == pytest.approx(2.72, abs=1e-12)
        assert elements[5] == {**original[5], "length_m": elements[5]["length_m"]}
        assert elements[5]["angle_rad"] == original[5]["angle_rad"]
        assert row["complete_mba_accepted"] is False
        assert row["verified_unit_checkpoint"]
        for left, right in zip(elements[:5], reversed(elements[6:])):
            for key in ("kind", "length_m", "k1_m2"):
                assert left[key] == right[key]
        lengths.append(tuple(e["length_m"] for e in elements))
    assert lengths[0] != tuple(e["length_m"] for e in original)
    assert search["candidates"][0]["emittance_pm"] < 95.33989106199819
    assert execution["result"]["plots"] and execution["result"]["tables"]
    assert execution["result"]["request_satisfied"] is False
    # A real second process/output root, not only clearing Python's capture cache.
    # Its fixture forbids every native numerical entry point and the optimizer.
    shutil.copyfile(RESTORE_PROGRAM, workspace / "restore.py")
    resume_arguments = {"script": "restore.py", "arguments": {"previous_output": str(output)}, "timeout_s": 120}
    replay = run_research_program(context, resume_arguments, durable=True)
    replay_validation = validate_research_program(context, resume_arguments, replay)
    assert replay_validation["passed"], replay_validation
    assert replay_validation["candidate_verification"]["passed"], replay_validation
    assert replay["result"]["candidate_evaluation"]["evaluation_id"] == execution["result"]["candidate_evaluation"]["evaluation_id"]
    assert Path(replay["output_directory"]) != output
    assert list((Path(replay["output_directory"]) / "evidence_reuse").glob("*.json"))
    assert replay["result"]["plots"] and replay["result"]["tables"]
    assert any(item.get("kind") == "optics_lattice" for item in replay["result"]["plots"])
    assert (Path(replay["output_directory"]) / "computed_presentation.json").is_file()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == FINGERPRINT
    print("NATIVE_GEOMETRY_SEARCH " + json.dumps({
        "output_directory": str(output), "initialization": search.get("initialization"),
        "candidates": [{"emittance_pm": row["emittance_pm"],
            "lengths_m": [e["length_m"] for e in row["elements"]],
            "near_k1": row["k1_near_m2"], "far_k1": row["k1_far_m2"]}
            for row in search["candidates"]], "app_result_validated": validation["passed"],
        "new_worker_recovery_validated": replay_validation["passed"],
        "new_worker_output": replay["output_directory"],
    }))


@pytest.mark.skipif(not os.environ.get("BRILLIANCE_TME_RECOVERY_WORKSPACE"),
                    reason="Opt-in new worker publication from this test's preserved same-project campaign")
def test_preserved_native_campaign_auto_presents_without_a_simulation_rerun():
    workspace = Path(os.environ["BRILLIANCE_TME_RECOVERY_WORKSPACE"]).resolve()
    previous = Path(os.environ["BRILLIANCE_TME_RECOVERY_OUTPUT"]).resolve()
    assert previous.parent == workspace / "artifacts" / "programs"
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Native evidence presentation regression")
    assert not environment.internet_access
    script = workspace / "restore_auto_presentation.py"
    shutil.copyfile(RESTORE_PROGRAM, script)
    context = SimpleNamespace(
        workspace=workspace, project_directory=workspace.parent, project_id=workspace.parent.name,
        compute_job_id="tme-presentation-test", project=SimpleNamespace(active_lattice_id=None),
        active_lattice=lambda: None, is_cancellation_requested=lambda: False,
        artifact_path=lambda name: workspace / "artifacts" / name, environment=environment,
    )
    arguments = {"script": script.name, "arguments": {"previous_output": str(previous)}, "timeout_s": 120}
    replay = run_research_program(context, arguments, durable=True)
    validation = validate_research_program(context, arguments, replay)
    assert replay["returncode"] == 0 and validation["passed"], (replay, validation)
    report = replay["result"]
    original = json.loads((previous / "result.json").read_text(encoding="utf-8"))
    assert report["candidate_evaluation"]["evaluation_id"] == original["candidate_evaluation"]["evaluation_id"]
    assert report["data"]["presentation"]["simulation_rerun"] is False
    assert len(report["plots"]) == 1 and report["plots"][0]["kind"] == "optics_lattice"
    figure = Path(replay["output_directory"]) / report["plots"][0]["path"]
    assert figure.read_bytes().startswith(b"\x89PNG")
    print("NATIVE_ARCHIVED_PRESENTATION " + json.dumps({"output_directory": replay["output_directory"],
        "figure": str(figure), "same_calculation": True, "new_physics_run": False,
        "app_result_validated": validation["passed"]}))
