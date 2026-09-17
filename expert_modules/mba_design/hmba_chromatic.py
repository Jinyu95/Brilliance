"""Measured two-plane chromatic correction for the optional HMBA route.

Coordinates are physical ``K2 [m^-3]``, shared by the explicitly named finite
KSEXT magnets in a family, NOT integrated K2*L or a polynomial coefficient.
JuTrack KSEXT stores a scalar ``k2``; multipole tracking converts it to
``PolynomB[3] = k2 / 2`` (Julia indexing, Python polynomial index 2). See
JuTrack.jl/src/lattice/elements.jl and src/tracking/multipole.jl. No dipole
multipole field is edited here. Actual lengths and multiplicities enter the
measured response, and are separately reported for reproducibility.

The callback kernel is testable with labelled synthetic algebra. The production
wrapper uses JuTrack getchrom for every response measurement and reconstructs a
fresh JuTrack cell for independent verification. It does not publish a lattice,
establish cancellation optics, or claim nonlinear/full-ring validation.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from brilliance.lattice import LatticeInputError

SCHEMA = "hmba_two_plane_chromatic_correction/v1"
NORMALIZATION = {
    "coordinate": "K2 [m^-3], common scalar for each explicitly named family member",
    "jutrack_field": "KSEXT.k2",
    "tracking_polynomial": "PolynomB[3] = k2 / 2 (Julia); polynomial index 2 in Python",
    "integrated_strength": "K2L [m^-2] = K2 [m^-3] * physical length [m]",
}


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LatticeInputError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise LatticeInputError(f"{name} must be finite and positive")
    return number


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise LatticeInputError(f"{name} must be a positive integer")
    return int(value)


def _pair(value: Any, name: str, *, positive: bool = False) -> np.ndarray:
    if isinstance(value, Mapping):
        value = [value["horizontal"], value["vertical"]]
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise LatticeInputError(f"{name} needs two finite horizontal/vertical values")
    if positive and np.any(array <= 0):
        raise LatticeInputError(f"{name} needs positive horizontal/vertical values")
    return array


def _xy(value: np.ndarray | None) -> dict | None:
    return None if value is None else {"horizontal": float(value[0]), "vertical": float(value[1])}


def _compile_families(rows, families):
    saved = deepcopy([dict(row) for row in rows])
    names = [row["name"] for row in saved]
    if len(set(names)) != len(names):
        raise LatticeInputError("Chromatic correction requires unique physical element names")
    if not isinstance(families, Mapping) or len(families) < 2:
        raise LatticeInputError(
            "Two-plane correction requires at least two explicit sextupole families"
        )
    by_name = dict(zip(names, saved))
    taken: set[str] = set()
    compiled = []
    for name, supplied in families.items():
        if not isinstance(name, str) or not name or not isinstance(supplied, Mapping):
            raise LatticeInputError(
                "Each sextupole family needs a name and a mapping specification"
            )
        members = supplied["element_names"]
        if isinstance(members, (str, bytes)) or not members or len(set(members)) != len(members):
            raise LatticeInputError(f"{name}: element_names must contain distinct physical names")
        if taken.intersection(members):
            raise LatticeInputError("A sextupole magnet cannot belong to two correction families")
        if set(members) - set(by_name):
            raise LatticeInputError(
                f"{name}: unknown sextupole names {sorted(set(members) - set(by_name))}"
            )
        bounds = _pair(supplied["k2_bounds_m3"], f"{name}.k2_bounds_m3")
        if bounds[0] >= bounds[1]:
            raise LatticeInputError(f"{name}: k2_bounds_m3 must be increasing")
        step = _positive(supplied["response_step_m3"], f"{name}.response_step_m3")
        magnets = [by_name[member] for member in members]
        for row in magnets:
            if row.get("kind") != "KSEXT":
                raise LatticeInputError(
                    f"{row['name']}: correction members must be actual KSEXT magnets"
                )
            _positive(row["length_m"], f"{row['name']}.length_m")
            if row.get("k2_m3") is None or not math.isfinite(float(row["k2_m3"])):
                raise LatticeInputError(
                    f"{row['name']}: physical k2_m3 must be declared and finite"
                )
        strengths = [float(row["k2_m3"]) for row in magnets]
        if any(
            not math.isclose(value, strengths[0], rel_tol=1e-12, abs_tol=1e-12)
            for value in strengths[1:]
        ):
            raise LatticeInputError(
                f"{name}: family members must initially share one physical K2 coordinate"
            )
        compiled.append(
            {
                "name": name,
                "element_names": list(members),
                "bounds": bounds,
                "step": step,
                "initial": strengths[0],
                "lengths": [float(row["length_m"]) for row in magnets],
            }
        )
        taken.update(members)
    return saved, compiled


def measure_and_solve_chromaticity(
    rows: Sequence[Mapping[str, Any]],
    families: Mapping[str, Mapping[str, Any]],
    measure: Callable[[list[dict]], Sequence[float] | Mapping[str, float]],
    verify: Callable[[list[dict]], Sequence[float] | Mapping[str, float]],
    *,
    target_chromaticities: Sequence[float],
    chromaticity_tolerance: Sequence[float],
    periods: int,
    max_response_condition: float,
    evaluation_budget: int,
    max_iterations: int = 3,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Measure A_ij=d(xi_i)/d(K2_j), solve within limits, independently verify.

    Both callbacks must return the two FULL-RING-equivalent chromaticities for
    exactly the supplied physical rows. ``verify`` must independently reconstruct
    the candidate, not return A*delta or a cached measurement. Numerical callback
    tests do not validate accelerator physics. API/runtime exceptions propagate;
    a budget outcome retains measured progress but never becomes a physics pass.

    Required family entries: ``element_names``, ``k2_bounds_m3`` (lower, upper),
    ``response_step_m3``. Family members may have different finite lengths.
    """
    from scipy.optimize import lsq_linear

    saved, compiled = _compile_families(rows, families)
    target = _pair(target_chromaticities, "target_chromaticities")
    tolerance = _pair(chromaticity_tolerance, "chromaticity_tolerance", positive=True)
    periods = _count(periods, "periods")
    budget = _count(evaluation_budget, "evaluation_budget")
    iterations = _count(max_iterations, "max_iterations")
    condition_limit = _positive(max_response_condition, "max_response_condition")
    if condition_limit < 1:
        raise LatticeInputError("max_response_condition cannot be smaller than one")
    if not callable(measure) or not callable(verify):
        raise LatticeInputError(
            "Both measurement and independent verification callbacks are required"
        )
    if measure is verify:
        raise LatticeInputError("Independent verification must not reuse the measurement callback")

    initial = np.asarray([family["initial"] for family in compiled])
    lower = np.asarray([family["bounds"][0] for family in compiled])
    upper = np.asarray([family["bounds"][1] for family in compiled])
    history: list[dict] = []
    baseline = None
    selected = initial.copy()
    selected_xi = None
    selected_score = math.inf
    corrected = None
    response_report: dict = {}
    unconstrained_violations: dict = {}
    checks = {
        "response_full_rank": False,
        "response_conditioned": False,
        "within_strength_limits": bool(np.all((initial >= lower) & (initial <= upper))),
        "target_met": False,
        "independent_verification": False,
    }

    def proposal(position):
        value = deepcopy(saved)
        by_name = {row["name"]: row for row in value}
        for family, strength in zip(compiled, position):
            for name in family["element_names"]:
                by_name[name]["k2_m3"] = float(strength)
        return value

    def strength_table(position):
        table = []
        for family, strength in zip(compiled, position):
            lo, hi = family["bounds"]
            absolute_limit = max(abs(lo), abs(hi))
            for name, length in zip(family["element_names"], family["lengths"]):
                table.append(
                    {
                        "family": family["name"],
                        "element": name,
                        "length_m": length,
                        "K2_m3": float(strength),
                        "K2L_m2": float(strength * length),
                        "lower_K2_m3": float(lo),
                        "upper_K2_m3": float(hi),
                        "strength_fraction": abs(float(strength)) / absolute_limit,
                        "family_multiplicity_per_cell": len(family["element_names"]),
                        "family_multiplicity_full_ring": periods * len(family["element_names"]),
                        "within_limit": bool(lo <= strength <= hi),
                    }
                )
        return table

    def outcome(status):
        table = strength_table(selected)
        passed = status == "corrected" and all(checks.values())
        return {
            "schema": SCHEMA,
            "status": status,
            "passed": passed,
            "checks": deepcopy(checks),
            "normalization": deepcopy(NORMALIZATION),
            "periods": periods,
            "chromaticity_scope": "full_ring_equivalent",
            "target_chromaticities": _xy(target),
            "chromaticity_tolerance": _xy(tolerance),
            "baseline_chromaticities": _xy(baseline),
            "candidate_chromaticities": _xy(selected_xi),
            "corrected_chromaticities": _xy(corrected),
            "residuals": _xy(None if selected_xi is None else selected_xi - target),
            "candidate_rows": proposal(selected),
            "corrected_rows": proposal(selected) if passed else None,
            "family_coordinates_m3": {
                family["name"]: float(value) for family, value in zip(compiled, selected)
            },
            "strengths": table,
            "max_strength_fraction": max(row["strength_fraction"] for row in table),
            "unconstrained_strength_violations": deepcopy(unconstrained_violations),
            "response": deepcopy(response_report),
            "evaluations": len(history),
            "evaluation_budget": budget,
            "evaluation_history": deepcopy(history),
            "full_ring_validation_required": True,
            "limitations": [
                "Two-plane chromatic correction only; pair cancellation and nonlinear acceptance are separate gates.",
                "A resource limit or incomplete local solve is not topology exhaustion.",
            ],
        }

    class BudgetReached(Exception):
        pass

    def evaluate(position, purpose, callback=measure):
        nonlocal selected, selected_xi, selected_score
        if len(history) >= budget:
            raise BudgetReached
        # All proposals, including difference probes, respect family limits.
        if np.any(position < lower) or np.any(position > upper):
            raise LatticeInputError(
                "Internal chromatic proposal exceeded explicit family strength bounds"
            )
        xi = _pair(callback(proposal(position)), "measured chromaticities")
        score = float(np.linalg.norm((xi - target) / tolerance))
        # Response probes are diagnostic, not proposed corrected candidates.
        if purpose in {"baseline", "corrected_candidate", "independent_verification"}:
            if score < selected_score or purpose == "independent_verification":
                selected, selected_xi, selected_score = position.copy(), xi.copy(), score
        record = {
            "evaluation": len(history) + 1,
            "purpose": purpose,
            "family_coordinates_m3": {
                family["name"]: float(value) for family, value in zip(compiled, position)
            },
            "chromaticities": _xy(xi),
            "residuals": _xy(xi - target),
            "scaled_residual_norm": score,
        }
        history.append(record)
        if progress is not None:
            progress(deepcopy(record))
        return xi

    if not checks["within_strength_limits"]:
        return outcome("strength_limit")
    try:
        baseline = evaluate(initial, "baseline")
        current, current_xi = initial.copy(), baseline.copy()
        for iteration in range(iterations):
            response = np.zeros((2, len(compiled)))
            response_report = {
                "iteration": iteration + 1,
                "coordinate": "K2 [m^-3]",
                "family_order": [family["name"] for family in compiled],
                "row_order": ["xi_x_full_ring", "xi_y_full_ring"],
                "columns": [],
                "complete": False,
            }
            for j, family in enumerate(compiled):
                lo = max(float(lower[j]), float(current[j] - family["step"]))
                hi = min(float(upper[j]), float(current[j] + family["step"]))
                if hi <= lo:
                    raise LatticeInputError(
                        f"{family['name']}: response step is not representable at current K2"
                    )
                a, b = current.copy(), current.copy()
                a[j], b[j] = lo, hi
                xa = (
                    current_xi
                    if lo == current[j]
                    else evaluate(a, f"response_minus:{family['name']}")
                )
                xb = (
                    current_xi
                    if hi == current[j]
                    else evaluate(b, f"response_plus:{family['name']}")
                )
                response[:, j] = (xb - xa) / (hi - lo)
                response_report["columns"].append(
                    {
                        "family": family["name"],
                        "lower_probe_K2_m3": lo,
                        "upper_probe_K2_m3": hi,
                        "dxi_dK2": response[:, j].tolist(),
                    }
                )
            singular = np.linalg.svd(response, compute_uv=False)
            rank = int(np.linalg.matrix_rank(response))
            condition = float(np.linalg.cond(response))
            response_report.update(
                matrix=response.tolist(),
                singular_values=singular.tolist(),
                rank=rank,
                condition_number=condition if math.isfinite(condition) else None,
                condition_limit=condition_limit,
                complete=True,
            )
            checks["response_full_rank"] = rank == 2
            checks["response_conditioned"] = (
                math.isfinite(condition) and condition <= condition_limit
            )
            if not checks["response_full_rank"]:
                return outcome("rank_deficient")
            if not checks["response_conditioned"]:
                return outcome("ill_conditioned")
            if np.all(np.abs(current_xi - target) <= tolerance):
                corrected = evaluate(current, "independent_verification", verify)
                checks["target_met"] = bool(np.all(np.abs(corrected - target) <= tolerance))
                checks["independent_verification"] = checks["target_met"]
                return outcome("corrected" if checks["target_met"] else "verification_failed")
            rhs = target - current_xi
            unbounded = current + np.linalg.lstsq(response, rhs, rcond=None)[0]
            unconstrained_violations = {
                family["name"]: {
                    "requested_K2_m3": float(value),
                    "lower_K2_m3": float(lo),
                    "upper_K2_m3": float(hi),
                }
                for family, value, lo, hi in zip(compiled, unbounded, lower, upper)
                if value < lo or value > hi
            }
            solution = lsq_linear(
                response / tolerance[:, None],
                rhs / tolerance,
                bounds=(lower - current, upper - current),
                method="bvls",
                tol=1e-12,
            )
            if not solution.success or not np.all(np.isfinite(solution.x)):
                return outcome("not_converged")
            current = np.clip(current + solution.x, lower, upper)
            current_xi = evaluate(current, "corrected_candidate")
            if np.all(np.abs(current_xi - target) <= tolerance):
                corrected = evaluate(current, "independent_verification", verify)
                checks["target_met"] = bool(np.all(np.abs(corrected - target) <= tolerance))
                checks["independent_verification"] = checks["target_met"]
                return outcome("corrected" if checks["target_met"] else "verification_failed")
        return outcome("strength_limit" if unconstrained_violations else "not_converged")
    except BudgetReached:
        return outcome("resource_limit")


def solve_chromatic_correction(
    editor: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    energy_gev: float,
    periods: int,
    families: Mapping[str, Mapping[str, Any]],
    target_chromaticities: Sequence[float],
    chromaticity_tolerance: Sequence[float],
    max_response_condition: float,
    evaluation_budget: int,
    max_iterations: int = 3,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """JuTrack-backed wrapper for one explicitly supplied periodic cell.

    Ring-equivalent xi is P times JuTrack's one-cell getchrom. Final verification
    rebuilds the corrected physical cell independently. An exact full-ring and
    momentum-step/orbit-convergence check is still required before design claims.
    JuTrack currently uses a forward momentum step of 1e-8 about its supplied
    origin in getchrom, not a separate off-momentum closed-orbit solve. Therefore
    this stage expects the ID-boundary achromat already checked by its caller;
    off-momentum/nonlinear claims cannot be inferred from its response matrix.
    The passed-in mutable editor is restored even on API/runtime exceptions.
    """
    from brilliance import accelerator_api as api
    from brilliance.jutrack_service import JuTrackService

    energy = _positive(energy_gev, "energy_gev")
    count = _count(periods, "periods")
    original = editor.export_rows()

    def measure(candidate):
        editor.sync_rows(candidate)
        return count * np.asarray(
            editor.jt.getchrom(editor.lattice, dp=0.0, energy=energy * 1e9), dtype=float
        )

    def verify(candidate):
        specification = api.periodic_cell(
            title="Independent corrected HMBA chromaticity check",
            energy_gev=energy,
            elements=candidate,
            intended_repetitions=count,
            source="independent reconstruction for HMBA two-plane chromatic correction",
        )
        independent = JuTrackService(editor.jt).build_lattice(specification)
        return count * np.asarray(
            editor.jt.getchrom(independent, dp=0.0, energy=energy * 1e9), dtype=float
        )

    try:
        result = measure_and_solve_chromaticity(
            rows,
            families,
            measure,
            verify,
            target_chromaticities=target_chromaticities,
            chromaticity_tolerance=chromaticity_tolerance,
            periods=count,
            max_response_condition=max_response_condition,
            evaluation_budget=evaluation_budget,
            max_iterations=max_iterations,
            progress=progress,
        )
        result.update(
            engine="JuTrack",
            method="pyJuTrack.getchrom",
            energy_gev=energy,
            verification_method="fresh physical periodic-cell reconstruction",
        )
        result["limitations"].append(
            "JuTrack getchrom uses a forward momentum step of 1e-8 about the supplied origin; "
            "ID achromat, exact full-ring validation and momentum/orbit convergence must be checked separately."
        )
        return result
    finally:
        editor.sync_rows(original)
