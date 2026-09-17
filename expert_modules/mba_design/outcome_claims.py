"""Final measured MBA outcomes, independent of construction and search methods.

No matching, optimization or lattice modification occurs here. Archived raw
JuTrack measurements are rechecked; a model-authored passed flag is never used.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import numpy as np

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import read_numerical_evidence, record_numerical_evidence
from brilliance.lattice import LatticeIR
from brilliance.lattice_optimization import radiation_physical_validity

from .hybrid_shape import evaluate_hybrid_mba_shape, validate_hybrid_mba_shape_report

SCHEMA = "mba_hybrid_outcome/v1"


def _machine(candidate, contract):
    required = {
        "energy_gev", "periods", "expected_cell_length_m", "expected_cell_angle_rad",
        "expected_bends", "minimum_magnet_length_m", "minimum_drift_length_m", "limits",
        "id_drift_names", "id_length_m", "achromat_dx_tolerance_m", "achromat_dpx_tolerance",
        "maximum_beta_m", "maximum_emittance_pm", "maximum_energy_spread", "limit_provenance",
    }
    if required - contract.keys() or not isinstance(contract["limit_provenance"], Mapping) or not contract["limit_provenance"]:
        raise ValueError("Declare original machine requirements, explicit optional caps and limit_provenance")
    budget = contract.get("hybrid_radiation_budget")
    if budget is not None:
        if (not isinstance(budget, Mapping) or not isinstance(budget.get("provenance"), Mapping)
                or not budget["provenance"] or isinstance(budget.get("maximum_emittance_pm"), bool)
                or not isinstance(budget.get("maximum_emittance_pm"), (int, float))
                or not math.isfinite(budget["maximum_emittance_pm"]) or budget["maximum_emittance_pm"] <= 0):
            raise ValueError("Hybrid radiation budget requires a positive ceiling and explicit provenance")
    periods = contract["periods"]
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
        raise ValueError("periods must be a positive integer")
    declared = candidate.metadata.get("intended_repetitions", candidate.metadata.get("repetitions", candidate.periodicity))
    if candidate.energy_gev != contract["energy_gev"] or declared != periods:
        raise ValueError("machine contract energy/repetitions differ from the exact candidate")
    for name in ("achromat_dx_tolerance_m", "achromat_dpx_tolerance", "id_length_m"):
        if not math.isfinite(contract[name]) or contract[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("maximum_beta_m", "maximum_emittance_pm", "maximum_energy_spread"):
        if contract[name] is not None and (not math.isfinite(contract[name]) or contract[name] <= 0):
            raise ValueError(f"{name} must be positive or explicitly None")
    rows = api.one_period_elements(candidate)
    if not isinstance(contract["limits"], Mapping) or not contract["limits"]:
        raise ValueError("explicit original magnet limits are required; no inferred defaults")
    audit = api.audit_lattice_constraints(
        rows, energy_gev=contract["energy_gev"], expected_length_m=contract["expected_cell_length_m"],
        expected_angle_rad=contract["expected_cell_angle_rad"], expected_bends=contract["expected_bends"],
        minimum_magnet_length_m=contract["minimum_magnet_length_m"],
        minimum_drift_length_m=contract["minimum_drift_length_m"], limits=contract["limits"], require_mirror_symmetry=True,
        length_bounds_m=contract.get("magnet_length_bounds_m"))
    names = contract["id_drift_names"]
    by_name = {row["name"]: row for row in rows}
    if not names or len(set(names)) != len(names) or any(name not in by_name or by_name[name]["kind"] != "DRIFT" for name in names):
        raise ValueError("id_drift_names must identify distinct physical ID drifts")
    issues = list(audit["issues"])
    if abs(sum(by_name[name]["length_m"] for name in names) - contract["id_length_m"]) > 1e-9:
        issues.append("original insertion-straight length changed")
    if abs(contract["expected_cell_angle_rad"] * periods - 2 * math.pi) > 1e-9:
        issues.append("ring bend closure failed")
    return issues


def _measurement_issues(candidate, measured):
    if measured.get("schema") != SCHEMA:
        return ["missing independently measured hybrid outcome"]
    machine, shape, coarse = measured["machine_contract"], measured["shape"], measured["coarse_shape"]
    issues = _machine(candidate, machine) + validate_hybrid_mba_shape_report(shape, candidate)
    for name, expected in (("id_dispersion_tolerance_m", "achromat_dx_tolerance_m"), ("id_slope_tolerance", "achromat_dpx_tolerance")):
        if shape["thresholds"][name] > machine[expected]:
            issues.append("shape thresholds relax an original achromat requirement")
    tolerance = measured["relative_tolerance"]
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("relative_tolerance must be finite and between zero and one")
    steps = measured["sampling_steps_m"]
    if len(steps) != 2 or not 0 < steps[1] < steps[0]:
        raise ValueError("supply positive coarse/fine sampling steps in decreasing order")
    for name in ("D_bump_L_m", "D_bump_R_m", "D_core_m", "H_core_max_m", "beta_max_m"):
        if not math.isclose(shape["metrics"][name], coarse["metrics"][name], rel_tol=tolerance, abs_tol=1e-15):
            issues.append("sampling has not converged: " + name)
    full = measured["full_ring_evaluation"]
    matrix = np.asarray(full["linear_optics"]["one_turn_matrix"], dtype=float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)) or any(abs(np.trace(matrix[i:i+2, i:i+2])) >= 2 for i in (0, 2)):
        issues.append("independent full ring is unstable")
    radiation = full["radiation"]
    budget = machine.get("hybrid_radiation_budget")
    if budget is not None and radiation.get("emittance_pm", math.inf) > budget["maximum_emittance_pm"]:
        issues.append("declared hybrid radiation budget exceeded")
    boundary = full["linear_optics"]["twiss_profile"][0]
    if abs(boundary["Dx"]) > machine["achromat_dx_tolerance_m"] or abs(boundary["Dpx"]) > machine["achromat_dpx_tolerance"]:
        issues.append("independent full-ring boundary is not achromatic")
    for axis in ("x", "y"):
        bounds = machine.get(f"id_beta_{axis}_bounds_m")
        if bounds is not None and not bounds[0] <= boundary[f"beta_{axis}"] <= bounds[1]:
            issues.append(f"original ID beta_{axis} bounds exceeded")
    if not radiation_physical_validity(radiation)["passed"]:
        issues.append("independent full-ring radiation is nonphysical")
    for name in ("I2", "I4", "I5", "emittance_pm", "energy_spread"):
        if not math.isclose(shape["radiation"][name], radiation[name], rel_tol=tolerance, abs_tol=1e-15):
            issues.append("periodic/full-ring radiation mismatch: " + name)
    for key, cap in (("beta_max_m", "maximum_beta_m"), ("emittance_pm", "maximum_emittance_pm"), ("energy_spread", "maximum_energy_spread")):
        value = shape["metrics"][key] if key == "beta_max_m" else radiation[key]
        if machine[cap] is not None and value > machine[cap]:
            issues.append("original machine cap exceeded: " + cap)
    return issues


def verify_hmba_candidate(candidate, *, machine_contract, shape_requirements,
                          sampling_steps_m, relative_tolerance, output_directory=None):
    """Measure the final candidate from any program; never run a design pipeline.

    machine_contract contains original limits/provenance; shape_requirements
    separately contains approved shape targets accepted by the shape diagnostic.
    Output directories are supplied by the managed runtime unless standalone.
    A failed shape still archives a useful exact-candidate measurement checkpoint.
    Requested chromatic correction or DA/MA needs separate computed evidence;
    this measurement alone cannot complete a compound scientific request.
    """
    candidate = api.load_lattice(candidate) if not isinstance(candidate, LatticeIR) else candidate
    machine = copy.deepcopy(dict(machine_contract))
    _machine(candidate, machine)
    if len(sampling_steps_m) != 2 or not 0 < sampling_steps_m[1] < sampling_steps_m[0]:
        raise ValueError("supply positive coarse/fine sampling steps in decreasing order")
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must be finite and between zero and one")
    measured = {"schema": SCHEMA, "machine_contract": machine,
                "sampling_steps_m": list(sampling_steps_m), "relative_tolerance": relative_tolerance}
    for key, step in zip(("coarse_shape", "shape"), sampling_steps_m):
        measured[key] = evaluate_hybrid_mba_shape(candidate, thresholds=shape_requirements,
            expected_cell_length_m=machine["expected_cell_length_m"], expected_cell_angle_rad=machine["expected_cell_angle_rad"],
            limits=machine["limits"], max_step_m=step, minimum_magnet_length_m=machine["minimum_magnet_length_m"],
            minimum_drift_length_m=machine["minimum_drift_length_m"], installation_slots=machine.get("installation_slots"))
    ring = api.cell_ring(title="Independent hybrid outcome verification", energy_gev=candidate.energy_gev,
                         cell=api.one_period_elements(candidate), periods=machine["periods"], source="exact selected candidate replay")
    measured["full_ring_evaluation"] = api.evaluate(ring, include_profile=True, include_radiation=True).as_dict()
    receipt = record_numerical_evidence(candidate, measured, output_directory=output_directory)
    try:
        issues = _measurement_issues(candidate, measured)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        issues = ["Incomplete numerical outcome: " + str(exc)]
    return {"schema": SCHEMA, "scope": "linear_hybrid_shape_only", "measurement_receipt": receipt, "passed": not issues, "issues": issues,
            "metrics": measured["shape"].get("metrics", {}), "radiation": measured["full_ring_evaluation"].get("radiation", {}),
            "limitations": ["Linear hybrid shape only; chromatic correction and DA/MA are not claimed."]}


def validate_hmba_linear(report, candidate):
    try:
        return _measurement_issues(candidate, read_numerical_evidence(candidate, report["measurement_receipt"]))
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError) as exc:
        return ["Unverified hybrid outcome: " + str(exc)]


def validate_mba_optimization(report, candidate):
    """Algorithm-neutral search proof from distinct measured physical snapshots."""
    issues = validate_hmba_linear(report, candidate)
    try:
        search = report["optimization"]
        if any(not str(search.get(key, "")).strip() for key in ("method", "termination_reason", "selection_reason")):
            raise ValueError("describe the actual search method, termination and selection")
        trials = search["trials"]
        if not isinstance(trials, list) or len(trials) < 2:
            raise ValueError("one feasibility evaluation is not optimization evidence")
        selected = search["selected_index"]
        if isinstance(selected, bool) or not isinstance(selected, int) or not 0 <= selected < len(trials):
            raise ValueError("selected_index must identify a measured trial")
        selected_measurement = read_numerical_evidence(candidate, report["measurement_receipt"])
        fingerprints, values = [], []
        for trial in trials:
            subject = LatticeIR.from_dict(trial["candidate"])
            raw = read_numerical_evidence(subject, trial["measurement_receipt"])
            machine_issues = _machine(subject, raw["machine_contract"])
            if machine_issues:
                raise ValueError("search comparison violates original machine requirements: " + "; ".join(machine_issues))
            radiation = raw["full_ring_evaluation"]["radiation"]
            if not radiation_physical_validity(radiation)["passed"]:
                raise ValueError("search comparison includes a nonphysical equilibrium")
            fingerprints.append(subject.physical_fingerprint)
            values.append(radiation["emittance_pm"])
            if subject.energy_gev != candidate.energy_gev:
                raise ValueError("search comparison changes beam energy")
            if raw["machine_contract"] != selected_measurement["machine_contract"]:
                raise ValueError("search comparison changes original machine limits or provenance")
        if len(set(fingerprints)) < 2:
            raise ValueError("search evidence repeats the same physical snapshot")
        if fingerprints[selected] != candidate.physical_fingerprint or trials[selected]["measurement_receipt"] != report["measurement_receipt"]:
            raise ValueError("optimization selection is not the exact independently checked candidate")
        if values[selected] >= values[0]:
            issues.append("no independently measured emittance improvement; retain the campaign as partial research")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError) as exc:
        issues.append("Unverified optimization outcome: " + str(exc))
    return issues
