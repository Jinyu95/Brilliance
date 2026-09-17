"""Scope, normalization and exact archived HD3/parent measurement regressions."""
import hashlib
import json
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
from expert_modules.mba_design.physics_references import homogeneous_bend_tme_reference

FIXTURES = Path(__file__).parent / "fixtures/mba_52956"
PROGRAM = Path(__file__).parent / "fixtures/mba_measurement_worker.py"


def test_archived_benchmark_hashes_and_reference_not_multiplied():
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf8"))
    for name, digest in manifest["canonical_sha256"].items():
        data = json.loads((FIXTURES / name).read_text(encoding="utf8"))
        assert hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == digest
    parent = api.load_lattice(FIXTURES / "parent_cell.json")
    reference = homogeneous_bend_tme_reference(parent)
    assert reference["reference_emittance_pm"] == pytest.approx(8.536481718596535)
    ring = api.cell_ring(title="Reference multiplicity check", energy_gev=3.5,
        cell=api.one_period_elements(parent), periods=32, source="test")
    assert homogeneous_bend_tme_reference(ring)["reference_emittance_pm"] == pytest.approx(reference["reference_emittance_pm"])
    assert 52.9562402016157/reference["reference_emittance_pm"] == pytest.approx(6.203521778265678, rel=1e-5)


def test_replay_program_uses_public_apis(tmp_path):
    shutil.copyfile(PROGRAM, tmp_path / "program.py")
    inspection = inspect_research_program(tmp_path, "program.py")
    assert inspection["passed"], inspection["issues"]


def test_reserved_slots_use_regions_global_mirrors_and_exact_names():
    from expert_modules.mba_design.hybrid_shape import _installation_slots, hybrid_mba_bend_roles
    rows = api.one_period_elements(api.load_lattice(FIXTURES / "parent_cell.json"))
    roles = hybrid_mba_bend_roles(rows)
    left = [i for i in roles["regions"]["left_bump"]["element_indices"] if rows[i]["kind"] == "DRIFT"][:2]
    slots = {label: [rows[i]["name"]] for label, i in zip(("SF_L", "SD_L", "SF_R", "SD_R"),
        [*left, len(rows)-1-left[0], len(rows)-1-left[1]])}
    points, position = [], 0.
    for row in [*rows, {"length_m": 0}]:
        points.append({"s_m": position, "Dx": .04, "Dpx": 0., "beta_x": 2., "beta_y": 3.})
        position += row["length_m"]
    measured = _installation_slots(rows, points, roles, slots)
    assert measured["SF_L"]["region"] == measured["SD_L"]["region"] == "left_bump"
    assert measured["SF_L"]["max_Dx_m"] == .04
    wrong = {**slots, "SD_L": slots["SF_L"]}
    with pytest.raises(ValueError, match="distinct"):
        _installation_slots(rows, points, roles, wrong)
    suppressor = next(i for i in roles["regions"]["left_suppressor"]["element_indices"] if rows[i]["kind"] == "DRIFT")
    with pytest.raises(ValueError, match="outer inter-bend"):
        _installation_slots(rows, points, roles, {**slots, "SD_L": [rows[suppressor]["name"]]})


def test_radiation_budget_cannot_be_compensated_by_good_shape(monkeypatch):
    from expert_modules.mba_design import hybrid_measurement as module
    parent = api.load_lattice(FIXTURES / "parent_cell.json")
    machine = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf8"))["machine_contract"]
    monkeypatch.setattr(api, "build_jutrack_profile_sampler", lambda *a, **kw: object())
    monkeypatch.setattr(module, "evaluate_hybrid_mba_shape", lambda *a, **kw: {
        "stable": True, "passed": True, "checks": {"positive_damping": True, "achromatic_ID": True},
        "metrics": {"D_bump_L_m": .1, "D_bump_R_m": .1, "D_core_m": .001, "beta_max_m": 20.},
        "radiation": {"emittance_pm": 5412., "I2": .44, "I5": .00013, "energy_spread": .001}})
    observer = module.HybridMeasurement(parent, machine_contract=machine, shape_requirements={},
        maximum_emittance_pm=250., budget_provenance={"test_only": True})
    actual = observer(api.one_period_elements(parent))
    assert actual["shape_passed"] and not actual["feasible"]
    assert not actual["checks"]["emittance_budget"] and actual["emittance_pm"] == 5412.
    assert actual["topology_exhausted"] is False
    assert actual["quadrupole_bounds"]["independent_knobs_at_limit"] is None


def test_reporting_rejects_another_candidates_evaluation():
    from brilliance.evaluation_results import evaluated_geometry
    from expert_modules.mba_design.radiation_reporting import mba_radiation_report
    parent = api.load_lattice(FIXTURES / "parent_cell.json")
    hd3 = api.load_lattice(FIXTURES / "hd3_cell.json")
    evaluation = {"engine": "JuTrack", "linear_optics": {}, "radiation": {},
        "geometry": {}, "evaluation_geometry": evaluated_geometry(hd3)}
    with pytest.raises(ValueError, match="candidate differ"):
        mba_radiation_report(parent, evaluation)


@pytest.mark.skipif(os.environ.get("BRILLIANCE_RUN_MBA_MEASUREMENTS") != "1", reason="Opt-in native benchmark replay, no optimization")
def test_saved_parent_and_hd3_in_actual_worker(tmp_path):
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="MBA measurement regression")
    assert not environment.internet_access
    name = "project-measurement-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    workspace = Path(os.environ.get("BRILLIANCE_MEASUREMENT_OUTPUT", str(tmp_path))) / name / "workspace"
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True)
    for path in FIXTURES.glob("*.json"):
        shutil.copyfile(path, inputs / path.name)
    shutil.copyfile(PROGRAM, workspace / "program.py")
    context = SimpleNamespace(workspace=workspace, project_directory=workspace.parent, project_id=name,
        compute_job_id="measurement-regression", project=SimpleNamespace(active_lattice_id=None), active_lattice=lambda: None,
        is_cancellation_requested=lambda: False, artifact_path=lambda name: workspace / "artifacts" / name, environment=environment)
    arguments = {"script": "program.py", "arguments": {}, "timeout_s": 600}
    execution = run_research_program(context, arguments, durable=True)
    assert execution["returncode"] == 0, execution
    validation = validate_research_program(context, arguments, execution)
    assert validation["passed"], validation
    data = execution["result"]["data"]
    assert data["proof"]["passed"]
    assert data["proof"]["metrics"]["final_emittance_pm"] == pytest.approx(52.9562402016157, rel=1e-7)
    assert data["proof"]["metrics"]["sampled_beta_max_m"] == pytest.approx(26.3971926664, rel=1e-7)
    assert data["ring_checks"]["circumference_m"] == pytest.approx(800., abs=1e-8)
    assert data["ring_checks"]["tune_x"] == pytest.approx(.21045237321799687, abs=1e-8)
    assert data["ring_checks"]["tune_y"] == pytest.approx(.13111322672441664, abs=1e-8)
    assert data["ring_checks"]["chromaticity_x"] == pytest.approx(-314.9666543717, rel=1e-7)
    assert data["ring_checks"]["chromaticity_y"] == pytest.approx(-202.3515041971, rel=1e-7)
    assert data["hd3_comparison"]["measurement_paths_agree"]
    assert not data["hd3_comparison"]["search_eligible"]
    assert data["hd3_comparison"]["quadrupole_bounds"]["physical_magnets_at_limit"] == 4
    assert data["hd3_comparison"]["quadrupole_bounds"]["physical_magnets"] == 28
    assert data["radiation_interpretation"]["region_fractions"]["outer_pairs"] == pytest.approx(.685, abs=.002)
    print("MBA_MEASUREMENT_REPLAY", execution["output_directory"], json.dumps(data["hd3_comparison"]))
