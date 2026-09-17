"""Replay immutable benchmark cells; NO unit generation, matching or optimization."""
import json

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import read_numerical_evidence
from brilliance.research_api import open_research_run

run = open_research_run()
mba = api.expert_module("laboratory.mba_design")
inputs = run.workspace / "inputs"
manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf8"))
machine = manifest["machine_contract"]
parent, unit, assembled, hd3 = [api.load_lattice(inputs / name) for name in
    ("parent_cell.json", "unit_cell.json", "assembled_cell.json", "hd3_cell.json")]
evaluation = api.evaluate_periodic_cell(parent, include_profile=True)
checkpoint = run.checkpoint_candidate(parent, evaluation, name="conventional_52956")
proof = mba.verify_conventional_candidate(parent, unit=unit, assembled_candidate=assembled,
    unit_groups=manifest["unit_groups"], machine_contract=machine)
assert proof["passed"], proof["issues"]
full_ring = read_numerical_evidence(parent, proof["measurement_receipt"])["full_ring_evaluation"]
run.artifact_path("explicit_ring.json").write_text(json.dumps(full_ring["candidate"], indent=2), encoding="utf8")
ring_snapshot = api.evaluation_snapshot(full_ring)
ring_parameters = full_ring["radiation"]["jutrack_ring_parameters"]
ring_checks = {"circumference_m": ring_parameters["circumference"],
    "tune_x": ring_snapshot["linear_optics"]["tune_x"], "tune_y": ring_snapshot["linear_optics"]["tune_y"],
    "chromaticity_x": ring_parameters["chromx"], "chromaticity_y": ring_parameters["chromy"]}
bends = api.bend_radiation_contributions(parent)
interpretation = mba.mba_radiation_report(parent, evaluation, bend_contributions=bends)
shape_targets = dict(min_bump_m=.03, min_contrast=2., max_core_dispersion_m=.025,
    max_core_h_m=8e-5, min_usable_drift_m=.2, min_total_usable_drift_m=.4,
    max_drift_slope=.02, max_beta_m=30., peak_slope_tolerance=1e-3,
    id_dispersion_tolerance_m=1e-4, id_slope_tolerance=1e-5, max_sample_step_m=.05)
# 250 pm is ONLY a rejection-test ceiling, not an approved research requirement.
observer = mba.HybridMeasurement(hd3, machine_contract=machine, shape_requirements=shape_targets,
    maximum_emittance_pm=250., budget_provenance={"test_only": "reject archived multi-nanometre HD3; not production approval"})
search_view = observer(api.one_period_elements(hd3))
check_view = mba.evaluate_hybrid_mba_shape(hd3, thresholds=shape_targets,
    expected_cell_length_m=25., expected_cell_angle_rad=machine["expected_cell_angle_rad"],
    limits=machine["limits"], minimum_magnet_length_m=.2, minimum_drift_length_m=.05)
keys = ("D_bump_L_m", "D_bump_R_m", "D_core_m", "H_core_max_m", "global_max_dispersion_m")
for key in keys:
    assert abs(search_view[key] - check_view["metrics"][key]) <= 1e-10, key
assert search_view["shape"]["cell_fingerprint"] == check_view["cell_fingerprint"]
assert not search_view["feasible"] and not search_view["checks"]["emittance_budget"]
# Reproduce the OLD unit-name/absolute-envelope definitions as a diagnostic only.
profile = check_view["profile"]
old = {"legacy_outer_unit_max_abs_Dx_m": max(abs(p["Dx"]) for p in profile
        if p.get("name", "").split(":")[0] in {"U001", "U002", "U006", "U007"}),
       "legacy_core_unit_max_abs_Dx_m": max(abs(p["Dx"]) for p in profile
        if p.get("name", "").split(":")[0] in {"U003", "U004", "U005"})}
comparison = {"candidate_id": hd3.lattice_id, "cell_fingerprint": check_view["cell_fingerprint"],
    "canonical": check_view["metrics"], "legacy_different_definitions": old,
    "radiation": check_view["radiation"], "peaks": check_view["peaks"],
    "search_eligible": search_view["feasible"], "radiation_budget_test_only_pm": 250.,
    "quadrupole_bounds": search_view["quadrupole_bounds"],
    "measurement_paths_agree": True, "topology_exhausted": False}
run.artifact_path("hd3_measurement_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf8")
run.artifact_path("parent_radiation_interpretation.json").write_text(json.dumps(interpretation, indent=2), encoding="utf8")
with run.reporting(checkpoint=checkpoint):
    run.finish(summary="The saved 52.956-pm conventional prototype reproduces. HD3 measurement definitions reconciled; radiation-rejected trial remains separate.",
        data={"proof": proof, "ring_checks": ring_checks, "radiation_interpretation": interpretation, "hd3_comparison": comparison},
        tables=[{"title": "TME normalization of the exact conventional parent", "rows": [
            {"Quantity": key, "Value": interpretation[key]} for key in ("measured_emittance_pm",
            "reference_emittance_pm_Jx1", "measured_over_reference_Jx1")]},
            {"title": "Physical bend roles and I5 contributions", "rows": interpretation["bend_roles"]}],
        candidate_lattice=parent, candidate_evaluation=evaluation,
        acceptance={"conventional_mba_linear": proof}, scientific_passed=True, request_satisfied=False,
        use_as_working_lattice=False, limitations=["Regression replay, not a new optimized design.",
            "No chromatic correction/nonlinear qualification. Test radiation ceiling is not a production approval."])
