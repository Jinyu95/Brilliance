"""Synthetic measurement/receipt contracts, not accelerator feasibility tests."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import numerical_evidence_directory as numerical_evidence_context
from brilliance.candidate_evidence import read_numerical_evidence, record_numerical_evidence
from brilliance.experts import ExpertRegistry
from brilliance.lattice_fingerprint import cell_fingerprint
from brilliance.scientific_acceptance import assess_scientific_acceptance
from expert_modules.mba_design import outcome_claims as claims
from expert_modules.mba_design.acceptance import HYBRID_LINEAR
from expert_modules.mba_design.hybrid_shape import hybrid_mba_shape_diagnostics


def cell(k1=1.0):
    rows = [api.drift("ID_L", 1.0)]
    for index in range(7):
        rows.append(api.sector_bend(f"B{index+1}", 1.0, 2*math.pi/(32*7)))
        if index in (1, 4):
            rows.extend([api.drift(f"D{index+1}A", .45), api.quadrupole(f"Q{index+1}", .1, k1),
                         api.drift(f"D{index+1}B", .45)])
        elif index < 6:
            rows.append(api.drift(f"D{index+1}", 1.0))
    rows.append(api.drift("ID_R", 1.0))
    return api.periodic_cell(title="Synthetic outcome fixture", energy_gev=3.5, elements=rows,
                             intended_repetitions=32, source="test only")


@pytest.fixture
def machine():
    return {"energy_gev": 3.5, "periods": 32, "expected_cell_length_m": 15.,
            "expected_cell_angle_rad": 2*math.pi/32, "expected_bends": 7,
            "minimum_magnet_length_m": .01, "minimum_drift_length_m": .01,
            "limits": api.magnet_strength_limits(3.5), "id_drift_names": ["ID_L", "ID_R"],
            "id_length_m": 2., "achromat_dx_tolerance_m": 1e-4, "achromat_dpx_tolerance": 1e-4,
            "maximum_beta_m": 10., "maximum_emittance_pm": None, "maximum_energy_spread": None,
            "limit_provenance": {"original": "synthetic machine contract, not facility defaults"}}


@pytest.fixture
def thresholds():
    return {"min_bump_m": .05, "min_contrast": 2., "min_usable_drift_m": .1,
            "min_total_usable_drift_m": .2, "max_drift_slope": .01, "max_beta_m": 10.,
            "max_core_dispersion_m": .01, "max_core_h_m": .001,
            "peak_slope_tolerance": 1e-4, "id_dispersion_tolerance_m": 1e-4,
            "id_slope_tolerance": 1e-4, "max_sample_step_m": .05}


def matrix():
    value = np.eye(6)
    for index in (0, 2):
        value[index:index+2, index:index+2] = [[0, 1], [-1, 0]]
    return value.tolist()


def radiation(candidate):
    k1 = next(row.k1_m2 for row in candidate.elements if row.kind == "KQUAD")
    return {"I2": .2, "I3": .03, "I4": 0., "I5": 1e-7*k1, "Jx": 1., "Jy": 1., "Jz": 2.,
            "emittance_pm": 100*k1, "energy_spread": .001}


def profile():
    result = []
    for position in np.linspace(0, 15, 1501):
        s = min(position, 15-position)
        if s <= 1:
            dx, dpx = 0., 0.
        elif s <= 2:
            dx, dpx = .04*(1-math.cos(math.pi*(s-1))), .04*math.pi*math.sin(math.pi*(s-1))
        elif s <= 3:
            dx, dpx = .08, 0.
        elif s <= 4:
            dx, dpx = .005+.0375*(1+math.cos(math.pi*(s-3))), -.0375*math.pi*math.sin(math.pi*(s-3))
        else:
            dx, dpx = .005, 0.
        result.append({"s_m": float(position), "Dx": dx, "Dpx": dpx if position <= 7.5 else -dpx,
                       "beta_x": 2., "beta_y": 3., "alpha_x": 0., "alpha_y": 0.})
    return result


@pytest.fixture
def synthetic_measurements(monkeypatch, machine):
    calls = []

    def measure(candidate, *, thresholds, **kwargs):
        calls.append((candidate.physical_fingerprint, kwargs["max_step_m"]))
        rows = api.one_period_elements(candidate)
        shape = hybrid_mba_shape_diagnostics(rows, profile(), thresholds=thresholds,
            sampling={"engine": "JuTrack", "method": "physical_element_prefix_map_checked",
                      "map_equivalent": True, "cell_fingerprint": cell_fingerprint(rows)}, lattice_id=candidate.lattice_id)
        audit = api.audit_lattice_constraints(rows, energy_gev=3.5, expected_length_m=15.,
            expected_angle_rad=2*math.pi/32, expected_bends=7, require_mirror_symmetry=True,
            minimum_magnet_length_m=.01, minimum_drift_length_m=.01, limits=machine["limits"])
        shape.update(scope="complete_cell_shape_gate", stable=True, construction_audit=audit,
                     one_turn_matrix=matrix(), machine_contract=machine, radiation=radiation(candidate))
        shape["checks"].update(stable=True, construction_and_mirror=audit["passed"],
                               ring_bend_closure=True, positive_damping=True)
        shape["passed"] = all(shape["checks"].values())
        return shape

    def full(candidate, **kwargs):
        class Result:
            def as_dict(self):
                return {"linear_optics": {"one_turn_matrix": matrix(), "twiss_profile": profile()},
                        "radiation": radiation(candidate)}
        return Result()

    monkeypatch.setattr(claims, "evaluate_hybrid_mba_shape", measure)
    monkeypatch.setattr(claims.api, "evaluate", full)
    return calls


def verify(candidate, machine, thresholds, output):
    return claims.verify_hmba_candidate(candidate, machine_contract=machine, shape_requirements=thresholds,
        sampling_steps_m=[.05, .025], relative_tolerance=1e-4, output_directory=output)


def test_final_ring_check_enforces_separate_hybrid_radiation_budget(tmp_path, machine, thresholds, synthetic_measurements):
    machine["hybrid_radiation_budget"] = {"maximum_emittance_pm": 10., "provenance": {"test_only": True}}
    report = verify(cell(), machine, thresholds, tmp_path)
    assert not report["passed"]
    assert "declared hybrid radiation budget exceeded" in report["issues"]
    assert machine["maximum_emittance_pm"] is None  # aspiration not rewritten as original requirement


def test_arbitrary_program_candidate_can_pass_without_classical_pipeline(tmp_path, machine, thresholds, synthetic_measurements):
    candidate = cell()
    result = verify(candidate, machine, thresholds, tmp_path)
    assert result["passed"], result["issues"]
    with numerical_evidence_context(tmp_path):
        assert claims.validate_hmba_linear(result, candidate) == []
    assert len(synthetic_measurements) == 2
    assert "pair_optics" not in result and "chromatic_correction" not in result
    checked = assess_scientific_acceptance({HYBRID_LINEAR: result}, required=(HYBRID_LINEAR,),
        lattice=candidate, registry=ExpertRegistry.discover(), evidence_directory=tmp_path)
    assert checked["passed"], checked["issues"]
    wrong_run = assess_scientific_acceptance({HYBRID_LINEAR: result}, required=(HYBRID_LINEAR,),
        lattice=candidate, registry=ExpertRegistry.discover(), evidence_directory=tmp_path/"another_run")
    assert not wrong_run["passed"]


def test_wrong_rows_and_self_certified_flags_cannot_pass(tmp_path, machine, thresholds, synthetic_measurements):
    candidate = cell()
    result = verify(candidate, machine, thresholds, tmp_path)
    result["passed"] = True
    with numerical_evidence_context(tmp_path):
        assert claims.validate_hmba_linear(result, cell(.8))
        assert claims.validate_hmba_linear({"passed": True, "checks": {"everything": True}}, candidate)
        fake = {"measurement_receipt": record_numerical_evidence(candidate, {"schema": claims.SCHEMA, "passed": True})}
        assert claims.validate_hmba_linear(fake, candidate)


def test_changed_archive_is_detected(tmp_path, machine, thresholds, synthetic_measurements):
    candidate = cell()
    result = verify(candidate, machine, thresholds, tmp_path)
    receipt = copy.deepcopy(result["measurement_receipt"])
    receipt["sha256"] = "0"*64
    with numerical_evidence_context(tmp_path):
        assert claims.validate_hmba_linear({"measurement_receipt": receipt}, candidate)


@pytest.mark.parametrize("mutation", ["radiation", "sampling", "achromat"])
def test_raw_numerical_failures_override_all_passed_flags(tmp_path, machine, thresholds, synthetic_measurements, mutation):
    candidate = cell()
    result = verify(candidate, machine, thresholds, tmp_path)
    with numerical_evidence_context(tmp_path):
        measured = read_numerical_evidence(candidate, result["measurement_receipt"])
        if mutation == "radiation":
            measured["full_ring_evaluation"]["radiation"]["Jx"] = -1.
        elif mutation == "sampling":
            measured["coarse_shape"]["metrics"]["D_core_m"] *= 2
        else:
            measured["full_ring_evaluation"]["linear_optics"]["twiss_profile"][0]["Dx"] = .1
        result["measurement_receipt"] = record_numerical_evidence(candidate, measured)
        result["passed"] = True
        assert claims.validate_hmba_linear(result, candidate)


def test_local_optimizer_has_algorithm_neutral_measured_evidence(tmp_path, machine, thresholds, synthetic_measurements):
    parent, candidate = cell(), cell(.8)
    baseline, selected = verify(parent, machine, thresholds, tmp_path), verify(candidate, machine, thresholds, tmp_path)
    selected["optimization"] = {"method": "bounded local objective with an inner achromat callback",
        "termination_reason": "local tolerance", "selection_reason": "20 percent lower measured emittance",
        "selected_index": 1, "trials": [
            {"candidate": parent.as_dict(), "measurement_receipt": baseline["measurement_receipt"]},
            {"candidate": candidate.as_dict(), "measurement_receipt": selected["measurement_receipt"]}]}
    with numerical_evidence_context(tmp_path):
        assert claims.validate_mba_optimization(selected, candidate) == []
        selected["optimization"]["selected_index"] = 0
        assert claims.validate_mba_optimization(selected, candidate)
        selected["optimization"]["trials"] = selected["optimization"]["trials"][:1]
        assert claims.validate_mba_optimization(selected, candidate)


def test_improvement_cannot_be_claimed_by_changing_original_limits(tmp_path, machine, thresholds, synthetic_measurements):
    parent, candidate = cell(), cell(.8)
    baseline = verify(parent, machine, thresholds, tmp_path)
    altered = copy.deepcopy(machine)
    altered["maximum_beta_m"] = 20.
    selected = verify(candidate, altered, thresholds, tmp_path)
    selected["optimization"] = {"method": "local", "termination_reason": "done", "selection_reason": "lower",
        "selected_index": 1, "trials": [
            {"candidate": parent.as_dict(), "measurement_receipt": baseline["measurement_receipt"]},
            {"candidate": candidate.as_dict(), "measurement_receipt": selected["measurement_receipt"]}]}
    with numerical_evidence_context(tmp_path):
        assert any("original machine limits" in item for item in claims.validate_mba_optimization(selected, candidate))


def test_optimizer_baseline_must_obey_actual_hardware_limits(tmp_path, machine, thresholds, synthetic_measurements):
    parent = cell(2 * machine["limits"]["max_abs_k1_m2"])
    candidate = cell(.8)
    baseline = verify(parent, machine, thresholds, tmp_path)
    selected = verify(candidate, machine, thresholds, tmp_path)
    assert not baseline["passed"] and selected["passed"]
    # Optional emittance caps are legitimately unspecified; hardware limits are not.
    assert machine["maximum_emittance_pm"] is None
    selected["optimization"] = {"method": "local", "termination_reason": "done", "selection_reason": "lower",
        "selected_index": 1, "trials": [
            {"candidate": parent.as_dict(), "measurement_receipt": baseline["measurement_receipt"]},
            {"candidate": candidate.as_dict(), "measurement_receipt": selected["measurement_receipt"]}]}
    with numerical_evidence_context(tmp_path):
        issues = claims.validate_mba_optimization(selected, candidate)
    assert any("original machine requirements" in item and "field limit" in item for item in issues)


def test_optimizer_baseline_may_fail_shape_without_failing_machine_contract(tmp_path, machine, thresholds, synthetic_measurements):
    parent, candidate = cell(), cell(.8)
    baseline, selected = verify(parent, machine, thresholds, tmp_path), verify(candidate, machine, thresholds, tmp_path)
    with numerical_evidence_context(tmp_path):
        measured = read_numerical_evidence(parent, baseline["measurement_receipt"])
        measured["shape"]["checks"]["left_bump_height"] = False
        measured["shape"]["passed"] = False
        baseline["measurement_receipt"] = record_numerical_evidence(parent, measured)
        selected["optimization"] = {"method": "local", "termination_reason": "done", "selection_reason": "lower",
            "selected_index": 1, "trials": [
                {"candidate": parent.as_dict(), "measurement_receipt": baseline["measurement_receipt"]},
                {"candidate": candidate.as_dict(), "measurement_receipt": selected["measurement_receipt"]}]}
        assert claims.validate_mba_optimization(selected, candidate) == []


def test_none_optional_caps_do_not_remove_engineering_bounds(machine):
    machine.update(maximum_beta_m=None, maximum_emittance_pm=None, maximum_energy_spread=None)
    assert claims._machine(cell(), machine) == []
    assert any("field limit" in issue for issue in claims._machine(cell(2 * machine["limits"]["max_abs_k1_m2"]), machine))


def test_original_limit_and_sampling_failures_stay_partial(tmp_path, machine, thresholds, synthetic_measurements):
    candidate = cell()
    machine["maximum_emittance_pm"] = 50.
    result = verify(candidate, machine, thresholds, tmp_path)
    assert not result["passed"] and any("machine cap" in item for item in result["issues"])
    with numerical_evidence_context(tmp_path):
        assert claims.validate_hmba_linear(result, candidate)
    assert (tmp_path/result["measurement_receipt"]["path"]).is_file()


def test_partial_deliverable_is_pending_not_missing_execution():
    result = assess_scientific_acceptance({}, required=(HYBRID_LINEAR,), lattice=cell(), registry=ExpertRegistry.discover())
    assert result["status"] == "pending"
    assert result["checks"][HYBRID_LINEAR]["passed"] is False


def test_machine_provenance_cannot_be_omitted(machine):
    del machine["limit_provenance"]
    with pytest.raises(ValueError, match="provenance"):
        claims._machine(cell(), machine)
