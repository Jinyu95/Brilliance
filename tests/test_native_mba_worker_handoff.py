"""Opt-in actual saved-unit -> match -> ring -> managed publication regression.

Unlike component mocks this launches brilliance.program_runner with the configured
Python/Julia environment, then runs the same parent result validator as the app.
Historical inputs remain regression-only; nothing is installed as a design seed.
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
from brilliance.numerical_checkpoints import read_numerical_checkpoint
from brilliance.research_environment import ResearchEnvironment
from brilliance.research_program import (
    inspect_research_program,
    run_research_program,
    validate_research_program,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKSPACE = ROOT / "research_projects/project-ab41b8de97b4c4/workspace"
PROGRAM = Path(__file__).with_name("fixtures") / "native_mba_worker_program.py"
INPUTS = {
    "unit.json": ("artifacts/programs/program-cdc6de955273696dd0f0/candidate_lattice.json",
                  "7f1948f55ad61417350605625b34d97d5e28f0820318a9c7f5c4de24f29bc893"),
    "cell_witness.json": ("cell_7ba_3p5gev_v1.json",
                          "62ad72b78a1d8e345f12656ae9ee1937de75edfa20747455dd7dcbbf9d25cc3c"),
    "interface.json": ("unit_v2_interface.json",
                       "2e408722fd3c213f00ffcc2bc6e00708805d7d6f4a90736f9cce61b15f7153ae"),
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_worker_fixture_uses_supported_public_interfaces(tmp_path):
    shutil.copyfile(PROGRAM, tmp_path / "program.py")
    inspection = inspect_research_program(tmp_path, "program.py")
    assert inspection["passed"], inspection["issues"]


@pytest.mark.skipif(os.environ.get("BRILLIANCE_RUN_MBA_WORKER_TESTS") != "1",
                    reason="Opt-in historical-input JuTrack worker/publication integration")
@pytest.mark.parametrize("fail_reporting", [False, True], ids=["normal-publication", "reporting-failure"])
def test_saved_unit_through_real_worker_to_ring_and_recoverable_result(tmp_path, monkeypatch, fail_reporting):
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Saved-unit worker integration")
    assert environment.internet_access is False, "This integration must not use internet research"
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    # Real projects have unique IDs. Reusing one would collide with the configured
    # Windows Julia junction for a different pytest workspace.
    project_id = "project-native-handoff-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    output_root = Path(os.environ.get("BRILLIANCE_MBA_HANDOFF_OUTPUT", str(tmp_path)))
    workspace = output_root / project_id / "workspace"
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True)
    for name, (relative, fingerprint) in INPUTS.items():
        source = SOURCE_WORKSPACE / relative
        assert source.is_file(), f"Required exact regression history missing: {source}"
        assert _sha(source) == fingerprint, relative
        shutil.copyfile(source, inputs / name)
    shutil.copyfile(PROGRAM, workspace / "program.py")
    context = SimpleNamespace(
        workspace=workspace, project_directory=workspace.parent,
        project_id=project_id, compute_job_id="native-handoff",
        project=SimpleNamespace(active_lattice_id=None), active_lattice=lambda: None,
        is_cancellation_requested=lambda: False,
        artifact_path=lambda name: workspace / "artifacts" / name, environment=environment,
    )
    arguments = {"script": "program.py", "arguments": {"fail_reporting": fail_reporting}, "timeout_s": 600}
    execution = run_research_program(context, arguments, durable=True)
    validation = validate_research_program(context, arguments, execution)
    assert execution["returncode"] == 0
    assert validation["passed"], validation
    assert validation["candidate_verification"]["passed"], validation
    result = execution["result"]
    assert result["request_satisfied"] is False
    assert result["use_as_working_lattice"] is False
    output = Path(execution["output_directory"])
    assert (output / "result.json").is_file()
    assert (output / "assembled_cell.json").is_file()
    assert (output / "explicit_ring.json").is_file()
    published = api.load_lattice(output / result["candidate_lattice"])
    assert len(published.elements) == 32 * 77
    assert published.total_length_m == pytest.approx(800, abs=1e-9)
    archived = validate_saved_candidate_evaluation(published, result["candidate_evaluation"], output)
    assert archived["passed"], archived

    # Recovery is disk-only. No JuTrack or matching campaign is allowed here.
    monkeypatch.setattr(api, "match_jutrack_line", lambda *a, **kw: pytest.fail("Recovery reran matching"))
    monkeypatch.setattr(api, "evaluate", lambda *a, **kw: pytest.fail("Recovery reran ring physics"))
    monkeypatch.setattr(api, "evaluate_periodic_cell", lambda *a, **kw: pytest.fail("Recovery reran unit physics"))
    # Immutable checkpoints survive pointer updates and worker termination.
    checkpoints = []
    for path in (output / "numerical_checkpoints").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("kind") == "integration_diagnostics" and "result" in raw:
            checkpoints.append(raw)
    assert len(checkpoints) == 1, list((output / "numerical_checkpoints").glob("*"))
    # The reader validates both the immutable payload and its descriptor.
    latest = read_numerical_checkpoint(output_directory=output)
    diagnostics = latest["result"]
    assert latest["kind"] == "integration_diagnostics"
    assert all(diagnostics["gates"].values()), diagnostics["gates"]
    match = diagnostics["match"]
    restored_match = read_numerical_checkpoint(match["numerical_checkpoint"], output_directory=output)
    assert restored_match["result"]["elements"] == match["elements"]
    assert restored_match["result"]["incoming_twiss"] == match["incoming_twiss"]
    assert restored_match["result"]["measurement"]["endpoint"] == match["measurement"]["endpoint"]
    assert match["callback_evaluations"] > 1
    cell = api.load_lattice(output / "assembled_cell.json")
    assert api.one_period_elements(cell)[:11] == match["elements"]
    reference_emittance = diagnostics["unit_evaluation"]["radiation"]["emittance_pm"]
    ring_emittance = diagnostics["ring_evaluation"]["radiation"]["emittance_pm"]
    assert reference_emittance == pytest.approx(95.33989106199819, rel=1e-6)
    assert ring_emittance > 0
    assert len(diagnostics["bend_contributions"]["rows"]) == 7
    if fail_reporting:
        assert result["status"] == "partial"
        assert "report" in result["summary"].lower()
        assert not (output / "complete_cell_optics.svg").exists()
    else:
        assert result["status"] == "completed"
        assert result["tables"] and result["plots"]
        assert (output / "complete_cell_optics.svg").is_file()
    for relative, fingerprint in INPUTS.values():
        assert _sha(SOURCE_WORKSPACE / relative) == fingerprint
    print("NATIVE_WORKER_HANDOFF " + json.dumps({
        "intentional_reporting_failure": fail_reporting, "output_directory": str(output),
        "unit_reference_emittance_pm": reference_emittance, "ring_emittance_pm": ring_emittance,
        "ID_Dx_m": diagnostics["id_optics"]["Dx"],
        "sampled_beta_max_m": diagnostics["sampled_beta_max_m"],
        "native_match_callback_evaluations": match["callback_evaluations"],
        "checks": diagnostics["gates"], "app_result_validated": validation["passed"],
    }, sort_keys=True))
