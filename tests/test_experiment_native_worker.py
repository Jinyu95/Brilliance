"""Opt-in native experiment selection/publication using historical test inputs.

The historical witness is copied into an isolated test project, never installed
as an expert seed or offered as evidence of fresh autonomous design capability.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import validate_saved_candidate_evaluation
from brilliance.experiment_candidates import CandidateLedger
from brilliance.numerical_checkpoints import read_numerical_checkpoint
from brilliance.research_environment import ResearchEnvironment
from brilliance.research_program import (
    inspect_research_program,
    run_research_program,
    validate_research_program,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKSPACE = ROOT / "research_projects/project-ab41b8de97b4c4/workspace"
PROGRAM = Path(__file__).with_name("fixtures") / "experiment_worker_program.py"
INPUTS = {
    "unit.json": (
        "artifacts/programs/program-cdc6de955273696dd0f0/candidate_lattice.json",
        "7f1948f55ad61417350605625b34d97d5e28f0820318a9c7f5c4de24f29bc893",
    ),
    "cell_witness.json": (
        "cell_7ba_3p5gev_v1.json",
        "62ad72b78a1d8e345f12656ae9ee1937de75edfa20747455dd7dcbbf9d25cc3c",
    ),
    "interface.json": (
        "unit_v2_interface.json",
        "2e408722fd3c213f00ffcc2bc6e00708805d7d6f4a90736f9cce61b15f7153ae",
    ),
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_experiment_worker_fixture_uses_supported_public_interfaces(tmp_path):
    shutil.copyfile(PROGRAM, tmp_path / "program.py")
    inspection = inspect_research_program(tmp_path, "program.py")
    assert inspection["passed"], inspection["issues"]


@pytest.mark.skipif(
    os.environ.get("BRILLIANCE_RUN_EXPERIMENT_WORKER_TESTS") != "1",
    reason="Opt-in historical-input native experiment/managed-publication integration",
)
@pytest.mark.parametrize("fail_reporting", [False, True], ids=["publication", "reporting-recovery"])
def test_experiment_native_ring_selection_and_disk_only_recovery(tmp_path, monkeypatch, fail_reporting):
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Native experiment integration")
    assert environment.internet_access is False, "The test does not authorize internet research"
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    monkeypatch.delenv("BRILLIANCE_PROJECT_WORKSPACE", raising=False)
    monkeypatch.delenv("BRILLIANCE_WORKSPACE", raising=False)
    project_id = "project-native-experiment-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    output_root = Path(os.environ.get("BRILLIANCE_EXPERIMENT_TEST_OUTPUT", str(tmp_path)))
    workspace = output_root / project_id / "workspace"
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True)
    for name, (relative, fingerprint) in INPUTS.items():
        source = SOURCE_WORKSPACE / relative
        assert source.is_file(), f"Missing exact historical integration input: {source}"
        assert _sha(source) == fingerprint, relative
        shutil.copyfile(source, inputs / name)
    shutil.copyfile(PROGRAM, workspace / "program.py")
    context = SimpleNamespace(
        workspace=workspace, project_directory=workspace.parent, project_id=project_id,
        compute_job_id="native-experiment", project=SimpleNamespace(active_lattice_id=None),
        active_lattice=lambda: None, is_cancellation_requested=lambda: False,
        artifact_path=lambda name: workspace / "artifacts" / name, environment=environment,
    )
    arguments = {"script": "program.py", "arguments": {
        "fail_reporting": fail_reporting, "remove_pair": True,
    }, "timeout_s": 600}
    execution = run_research_program(context, arguments, durable=True)
    validation = validate_research_program(context, arguments, execution)
    assert execution["returncode"] == 0, execution
    assert validation["passed"], validation
    assert validation["candidate_verification"]["passed"], validation
    result, output = execution["result"], Path(execution["output_directory"])
    assert result["request_satisfied"] is False
    assert result["use_as_working_lattice"] is False
    assert (output / "result.json").is_file()
    published = api.load_lattice(output / result["candidate_lattice"])
    assert published.total_length_m == pytest.approx(800, abs=1e-9)
    assert published.periodicity == 32
    archived = validate_saved_candidate_evaluation(published, result["candidate_evaluation"], output)
    assert archived["passed"], archived

    # Recovery must not call Julia, rematch, rerun the objective, or regenerate a
    # unit. The ledger can be reopened independently of its original process.
    def no_new_physics(*args, **kwargs):
        pytest.fail("Disk-only experiment recovery attempted another numerical calculation")

    for name in ("evaluate", "evaluate_periodic_cell", "match_jutrack_line"):
        monkeypatch.setattr(api, name, no_new_physics)
    latest = read_numerical_checkpoint(output_directory=output)
    assert latest["kind"] == "experiment_integration"
    diagnostics = latest["result"]
    assert all(diagnostics["gates"].values()), diagnostics["gates"]
    assert diagnostics["unit_reference_emittance_pm"] == pytest.approx(95.33989106199819, rel=1e-6)
    assert diagnostics["parent_match"]["callback_evaluations"] > 1
    assert diagnostics["parent_match"]["complete_cell_accepted"] is False
    restored_match = read_numerical_checkpoint(
        diagnostics["parent_match"]["numerical_checkpoint"], output_directory=output,
    )
    assert restored_match["result"]["elements"] == diagnostics["parent_match"]["elements"]
    selection = diagnostics["selection"]
    assert selection["selected"] is not None
    assert len(diagnostics["entries"]) == 3
    assert diagnostics["parent_selection"]["selected"] in diagnostics["entries"]
    assert diagnostics["scalar_trial"]["feasible"] is False
    assert (diagnostics["scalar_trial"]["measurements"]["emittance_pm"]
            != diagnostics["parent_trial"]["measurements"]["emittance_pm"])
    assert len(diagnostics["bend_contributions"]["rows"]) == 7
    assert diagnostics["removal"] is not None
    assert diagnostics["removal"]["match"]["callback_evaluations"] > 0
    assert len(diagnostics["removal"]["before_strengths"]) == 2
    assert all(abs(v) > 0.1 for v in diagnostics["removal"]["before_strengths"].values())
    assert diagnostics["removal"]["changes"]["requires_native_rebuild"] is True
    assert diagnostics["removal"]["changes"]["physics_verified"] is False
    if not diagnostics["removal"]["match"]["match_success"]:
        assert selection["selected"] == diagnostics["parent_selection"]["selected"]

    restored_ledger = CandidateLedger.open(output / "experiment", workspace=workspace)
    assert restored_ledger.entries() == diagnostics["entries"]
    assert restored_ledger.select() == selection
    chosen_cell = api.load_lattice(output / "selected_cell.json")
    assert api.one_period_elements(chosen_cell) == selection["selected"]["candidate"]["elements"]
    assert diagnostics["ring_evaluation"]["radiation"]["emittance_pm"] > 0
    assert abs(diagnostics["ring_evaluation"]["linear_optics"]["Dx_m"]) <= 1e-4
    if fail_reporting:
        assert result["status"] == "partial"
        assert (output / "reporting_failure.json").is_file()
        assert not (output / "experiment_cell_optics.svg").exists()
    else:
        assert result["status"] == "completed"
        assert result["tables"] and result["plots"]
        assert (output / "experiment_cell_optics.svg").is_file()
    for relative, fingerprint in INPUTS.values():
        assert _sha(SOURCE_WORKSPACE / relative) == fingerprint
    print("NATIVE_EXPERIMENT_INTEGRATION " + json.dumps({
        "scope": "historical witness, not a fresh autonomous design",
        "output_directory": str(output), "reporting_failure": fail_reporting,
        "ring_emittance_pm": diagnostics["ring_evaluation"]["radiation"]["emittance_pm"],
        "ID_Dx_m": diagnostics["ring_evaluation"]["linear_optics"]["Dx_m"],
        "pair_removal_match_success": diagnostics["removal"]["match"]["match_success"],
        "pair_removal_callbacks": diagnostics["removal"]["match"]["callback_evaluations"],
        "recorded_trials": len(diagnostics["entries"]),
        "checks": diagnostics["gates"], "app_validation": validation["passed"],
    }, sort_keys=True))
