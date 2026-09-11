"""Position-based H-MBA diagnostics; no lattice seed or prescribed optimizer.

The complete cell is the authority. These measurements deliberately do not use
emittance as a proxy for two outer chromatic regions and a low-H central core.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from brilliance.lattice import LatticeElementIR
from brilliance.lattice_fingerprint import cell_fingerprint

SCHEMA = "hybrid_mba_shape/v2"


def _rows(elements: Sequence[Any]) -> list[dict[str, Any]]:
    result = [asdict(item if isinstance(item, LatticeElementIR)
                     else LatticeElementIR.from_mapping(item)) for item in elements]
    if not result or len({row["name"] for row in result}) != len(result):
        raise ValueError("Use one nonempty physical cell with unique element names.")
    return result


def hybrid_mba_bend_roles(elements: Sequence[Any]) -> dict[str, Any]:
    """Assign roles by physical order, never by a historical unit/family name.

    The default H-MBA hypothesis uses the first/last two positive bends as
    DBA-like regions and the middle M-4 bends as the low-H core (M >= 5).
    Segmented/reverse-bend/combined-function architectures need a different
    explicit role model; silently counting their slices is not supported here.
    """
    rows = _rows(elements)
    bends = []
    position = 0.0
    for index, row in enumerate(rows):
        length = float(row["length_m"])
        angle = float(row.get("angle_rad") or 0)
        if not math.isfinite(length) or length < 0:
            raise ValueError("Element lengths must be finite and nonnegative.")
        if angle:
            if row["kind"] != "SBEND" or angle <= 0 or length <= 0:
                raise ValueError("This role model requires unsplit positive sector bends.")
            if any(float(row.get(key) or 0) for key in ("k1_m2", "k2_m3", "k3_m4")):
                raise ValueError("Combined-function bends need an explicitly authorized role model.")
            bends.append({"bend": f"B{len(bends) + 1}", "name": row["name"], "element_index": index,
                          "s_start_m": position, "s_end_m": position + length,
                          "length_m": length, "angle_rad": angle})
        position += length
    count = len(bends)
    if count < 5:
        raise ValueError("The two-outer-pairs H-MBA hypothesis requires at least five bends.")
    for index, bend in enumerate(bends):
        bend["role"] = "left_DBA" if index < 2 else "right_DBA" if index >= count - 2 else "low_H_core"
        bend["mirror_bend"] = f"B{count - index}"
    regions = {}
    for side, first, second in (("left_bump", bends[0], bends[1]),
                                ("right_bump", bends[-2], bends[-1])):
        if second["s_start_m"] <= first["s_end_m"]:
            raise ValueError(f"{side} needs a finite inter-bend chromatic region.")
        regions[side] = {"s_start_m": first["s_end_m"], "s_end_m": second["s_start_m"],
                         "between": [first["bend"], second["bend"]],
                         "element_indices": list(range(first["element_index"] + 1, second["element_index"]))}
    regions["core"] = {"s_start_m": bends[2]["s_start_m"], "s_end_m": bends[-3]["s_end_m"],
                       "bends": [bend["bend"] for bend in bends[2:-2]],
                       "element_indices": list(range(bends[2]["element_index"], bends[-3]["element_index"] + 1))}
    for side, first, second in (("left_suppressor", bends[1], bends[2]),
                                ("right_suppressor", bends[-3], bends[-2])):
        regions[side] = {"s_start_m": first["s_end_m"], "s_end_m": second["s_start_m"],
                         "element_indices": list(range(first["element_index"] + 1, second["element_index"]))}
    return {"bend_count": count, "bends": bends, "regions": regions, "cell_length_m": position,
            "core_element_indices": regions["core"]["element_indices"],
            "core_bend_indices": [bend["element_index"] for bend in bends[2:-2]],
            "outer_bend_indices": [bend["element_index"] for bend in bends[:2] + bends[-2:]],
            "symmetry_plane_m": position / 2, "central_bend": f"B{count // 2 + 1}" if count % 2 else None}


def _thresholds(values: Mapping[str, Any], *, stage: str = "full_hybrid") -> dict[str, float]:
    if stage not in {"full_hybrid", "bump_feasibility"}:
        raise ValueError("Unknown hybrid evaluation stage")
    from .hybrid_optics import DRIFT_REQUIREMENTS
    required = ("min_bump_m", "min_contrast", *DRIFT_REQUIREMENTS) + (("max_core_h_m",) if stage == "full_hybrid" else ())
    absent = [key for key in required if key not in values]
    if absent:
        raise ValueError("Declare H-MBA shape thresholds explicitly: " + ", ".join(absent))
    result = {"peak_slope_tolerance": 1e-4, "id_dispersion_tolerance_m": 1e-4,
              "id_slope_tolerance": 1e-4, "max_sample_step_m": 0.05, **dict(values)}
    allowed = set(required) | {"peak_slope_tolerance", "id_dispersion_tolerance_m",
                               "id_slope_tolerance", "max_sample_step_m"}
    if set(result) - allowed:
        raise ValueError("Unknown H-MBA threshold key for " + stage + ": " +
                         ", ".join(sorted(set(result)-allowed)) + ". Allowed: " + ", ".join(sorted(allowed)))
    for key, value in result.items():
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{key} must be a finite positive value.")
        result[key] = float(value)
    if result["min_contrast"] <= 1:
        raise ValueError("min_contrast must exceed one: a central hill is not a hybrid shape.")
    return result


def hybrid_mba_shape_diagnostics(
    elements: Sequence[Any], profile: Sequence[Mapping[str, Any]], *,
    thresholds: Mapping[str, Any], sampling: Mapping[str, Any] | None = None,
    lattice_id: str | None = None,
    stage: str = "full_hybrid",
) -> dict[str, Any]:
    """Measure actual sampled outer interior maxima, central envelope/H and ID.

    Inputs use the public JuTrack profile schema. No bend-exit substitution or
    interpolated extremum is manufactured. The dense sampler adds actual JuTrack
    Dpx=0 solutions, allowing a small slope tolerance without an excessive grid.
    ``passed`` here means shape only, not a full-machine acceptance decision.

    Check sampler.linear_metrics(...)["stable"] before calling this function;
    rejected trials have an empty profile plus their actual failure/trace values.
    Returns metrics (D_bump_L_m, D_bump_R_m, D_core_m, H_core_max_m,
    dispersion_contrast, shape_penalty), peaks, checks, and scaled residuals.
    Full evaluator and result examples: read_expert(module_id="laboratory.mba_design",
    asset_id="references/hybrid_shape_api.md").
    """
    rows = _rows(elements)
    roles = hybrid_mba_bend_roles(rows)
    limits = _thresholds(thresholds, stage=stage)
    if len(profile) < 3:
        raise ValueError(
            "A complete sampled one-cell optics profile is required. Check the sampler's "
            "stable flag and failure/trace_x/trace_y first: an unstable perturbed trial "
            "intentionally has an empty profile and is not an API failure."
        )
    keys = ("s_m", "Dx", "Dpx", "beta_x", "alpha_x")
    points = [{**{key: float(row[key]) for key in keys},
               **{key: float(row[key]) for key in ("beta_y", "alpha_y") if key in row},
               **{key: str(row[key]) for key in ("name", "source_element", "extremum_method") if key in row}}
              for row in profile]
    arrays = np.asarray([[row[key] for key in keys] for row in points])
    if not np.all(np.isfinite(arrays)) or np.any(arrays[:, 3] <= 0):
        raise ValueError("Shape diagnostics require finite optics and positive beta_x.")
    if any(not math.isfinite(p.get("beta_y", math.nan)) or p["beta_y"] <= 0 for p in points):
        raise ValueError("Usable-bump diagnostics require finite positive beta_y.")
    if np.any(np.diff(arrays[:, 0]) < -1e-10):
        raise ValueError("Profile coordinates must be in physical cell order.")
    clean = []
    for point in points:
        if clean and abs(clean[-1]["s_m"] - point["s_m"]) < 1e-12:
            if any(abs(clean[-1][key] - point[key]) > 1e-9 for key in keys[1:]):
                raise ValueError("Conflicting optical samples at the same position.")
            continue
        clean.append(point)
    points = clean
    if abs(points[0]["s_m"]) > 1e-9 or abs(points[-1]["s_m"] - roles["cell_length_m"]) > 1e-9:
        raise ValueError("Profile must cover exactly one full physical cell, including both ID boundaries.")
    for row in points:
        # Stable sum-of-squares form of gamma*D²+2*alpha*D*D'+beta*D'².
        row["H_x_m"] = (row["Dx"] ** 2 + (row["alpha_x"] * row["Dx"] + row["beta_x"] * row["Dpx"]) ** 2) / row["beta_x"]
    max_gap = max(b["s_m"] - a["s_m"] for a, b in zip(points, points[1:]))
    fingerprint = cell_fingerprint(rows)
    sampling = dict(sampling or {})
    coverage = (max_gap <= limits["max_sample_step_m"] * (1 + 1e-9)
                and sampling.get("engine") == "JuTrack"
                and sampling.get("method") == "physical_element_prefix_map_checked"
                and sampling.get("map_equivalent") is True
                and sampling.get("cell_fingerprint") == fingerprint)
    peaks = {}
    checks = {"sampled_profile_and_map": coverage}
    metrics: dict[str, Any] = {"max_sample_gap_m": max_gap}
    for side, region_name, metric in (("left", "left_bump", "D_bump_L_m"),
                                      ("right", "right_bump", "D_bump_R_m")):
        region = roles["regions"][region_name]
        start, end = region["s_start_m"], region["s_end_m"]
        candidates = [(index, row) for index, row in enumerate(points) if start - 1e-10 <= row["s_m"] <= end + 1e-10]
        if not candidates:
            raise ValueError(f"No optical samples in the {side} inter-bend region.")
        index, peak = max(candidates, key=lambda pair: pair[1]["Dx"])
        # Ignore roundoff slopes while bracketing. A genuinely flat-topped
        # interior maximum is valid when it has resolved rising/falling sides;
        # selecting the nearest +1e-16 plateau sample would falsely reject it.
        slope_noise_floor = 1e-12
        before = next((row for row in reversed(points[:index]) if row["Dpx"] > slope_noise_floor and row["s_m"] >= start), None)
        after = next((row for row in points[index + 1:] if row["Dpx"] < -slope_noise_floor and row["s_m"] <= end), None)
        # Sign bracketing and prominence exclude endpoints, minima and wholly
        # flat regions. Tiny roundoff alone is not evidence of a bump.
        interior = bool(start + 1e-9 < peak["s_m"] < end - 1e-9 and before and after
                        and peak["Dx"] > max(before["Dx"], after["Dx"]) + 1e-12)
        stationary = abs(peak["Dpx"]) <= limits["peak_slope_tolerance"]
        peaks[side] = {"s_m": peak["s_m"], "Dx_m": peak["Dx"], "Dpx": peak["Dpx"],
                       "interior_maximum": interior, "stationary": stationary,
                       "region_start_m": start, "region_end_m": end}
        metrics[metric] = peak["Dx"]
        checks[f"{side}_interior_maximum"] = interior
        checks[f"{side}_stationary_peak"] = stationary
        checks[f"{side}_bump_height"] = peak["Dx"] >= limits["min_bump_m"]
    core = roles["regions"]["core"]
    core_points = [row for row in points if core["s_start_m"] - 1e-10 <= row["s_m"] <= core["s_end_m"] + 1e-10]
    if not core_points:
        raise ValueError("No optical samples in the central core.")
    core_dx = max(abs(row["Dx"]) for row in core_points)
    core_h = max(row["H_x_m"] for row in core_points)
    core_h_mean = (sum((a["H_x_m"] + b["H_x_m"]) * (b["s_m"] - a["s_m"]) / 2
                       for a, b in zip(core_points, core_points[1:]))
                   / (core_points[-1]["s_m"] - core_points[0]["s_m"])) if len(core_points) > 1 else core_h
    # A numerically zero core has a disclosed floor; no division-by-zero or
    # Infinity may leak into JSON/checkpoints or silently create a passing gate.
    floor = 1e-12
    left_ratio = metrics["D_bump_L_m"] / max(core_dx, floor)
    right_ratio = metrics["D_bump_R_m"] / max(core_dx, floor)
    metrics.update({"D_core_m": core_dx, "H_core_max_m": core_h, "H_core_mean_m": core_h_mean,
                    "dispersion_contrast": (left_ratio + right_ratio) / 2,
                    "contrast_L": left_ratio, "contrast_R": right_ratio,
                    "contrast_denominator_floor_m": floor,
                    "ID_Dx_max_abs_m": max(abs(points[0]["Dx"]), abs(points[-1]["Dx"])),
                    "ID_Dpx_max_abs": max(abs(points[0]["Dpx"]), abs(points[-1]["Dpx"]))})
    checks.update({"left_contrast": left_ratio >= limits["min_contrast"],
                   "right_contrast": right_ratio >= limits["min_contrast"],
                   "achromatic_ID": metrics["ID_Dx_max_abs_m"] <= limits["id_dispersion_tolerance_m"]
                       and metrics["ID_Dpx_max_abs"] <= limits["id_slope_tolerance"]})
    # Smooth target residuals can drive continuation from a valid conventional
    # seed. Do not put the *final* shape target into production feasibility until
    # it has been reached; stability/engineering/achromat remain hard throughout.
    residuals = {
        "left_height": max(0.0, 1 - metrics["D_bump_L_m"] / limits["min_bump_m"]),
        "right_height": max(0.0, 1 - metrics["D_bump_R_m"] / limits["min_bump_m"]),
        "left_contrast": max(0.0, 1 - left_ratio / limits["min_contrast"]),
        "right_contrast": max(0.0, 1 - right_ratio / limits["min_contrast"]),
    }
    if stage == "full_hybrid":
        checks.update(low_core_dispersion=core_dx <= limits["max_core_dispersion_m"],
                      low_core_H=core_h <= limits["max_core_h_m"])
        residuals.update(core_dispersion=max(0.0, core_dx / limits["max_core_dispersion_m"] - 1),
                         core_H=max(0.0, core_h / limits["max_core_h_m"] - 1))
    from .hybrid_optics import usable_bump_drifts
    usability = usable_bump_drifts(rows, points, roles, limits, core_dx)
    checks.update(usability["checks"])
    metrics.update(usability["metrics"])
    # A plateau can span the entire inter-bend window: its rising/falling
    # shoulders then lie in the bounding bends. Requiring a sharp maximum
    # inside empty space rejects a useful achromatic DBA configuration.
    for side, pair in (("left", roles["bends"][:2]), ("right", roles["bends"][-2:])):
        start, end = pair[0]["s_end_m"], pair[1]["s_start_m"]
        window = [p for p in points if start-1e-9 <= p["s_m"] <= end+1e-9]
        peak = peaks[side]
        plateau = (checks[side+"_usable_drift"] and checks[side+"_usable_drift_total"]
            and all(abs(p["Dx"]-peak["Dx_m"]) <= 1e-8 and abs(p["Dpx"]) <= limits["peak_slope_tolerance"] for p in window)
            and any(p["Dx"] < peak["Dx_m"]-1e-8 and p["Dpx"] > 1e-12 for p in points
                    if pair[0]["s_start_m"] <= p["s_m"] < start)
            and any(p["Dx"] < peak["Dx_m"]-1e-8 and p["Dpx"] < -1e-12 for p in points
                    if end < p["s_m"] <= pair[1]["s_end_m"]))
        if plateau:
            checks[side+"_interior_maximum"] = True
            peak["interior_maximum"] = True
            peak["plateau_with_bend_shoulders"] = True
    for side in ("left", "right"):
        residuals[side + "_usable_drift"] = max(0., 1 - metrics[side + "_longest_usable_drift_m"] / limits["min_usable_drift_m"])
        residuals[side + "_usable_drift_total"] = max(0., 1 - metrics[side + "_usable_drift_m"] / limits["min_total_usable_drift_m"])
    residuals["beta_envelope"] = max(0., metrics["beta_max_m"] / limits["max_beta_m"] - 1)
    residuals["core_dispersion"] = max(0., core_dx / limits["max_core_dispersion_m"] - 1)
    metrics["shape_penalty"] = sum(value ** 2 for value in residuals.values()) + sum(
        1.0 for side in ("left", "right") if not checks[f"{side}_interior_maximum"])
    return {"schema": SCHEMA, "lattice_id": lattice_id, "cell_fingerprint": fingerprint,
            "scope": "shape_diagnostics_only", "stage": stage, "thresholds": limits, "roles": roles,
            "metrics": metrics, "peaks": peaks, "usable_drifts": usability["drifts"],
            "checks": checks, "passed": all(checks.values()),
            "residuals": residuals, "sampling": sampling, "profile": points, "cell_rows": rows,
            "limitations": ["Linear dispersion shape only; not chromatic correction or nonlinear acceptance."]}


def evaluate_hybrid_mba_shape(
    candidate: Any, *, thresholds: Mapping[str, Any], expected_cell_length_m: float,
    expected_cell_angle_rad: float, limits: Mapping[str, float] | None = None,
    max_step_m: float = 0.05, minimum_magnet_length_m: float = 0.0,
    minimum_drift_length_m: float = 0.0,
    stage: str = "full_hybrid",
) -> dict[str, Any]:
    """JuTrack-backed final shape evidence, bound to the exact saved candidate."""
    from brilliance import accelerator_api as api
    from brilliance.optics_sampling import JuTrackProfileSampler

    _thresholds(thresholds, stage=stage)

    lattice = api.load_lattice(candidate) if isinstance(candidate, (str, Path)) else candidate
    rows = api.one_period_elements(lattice)
    roles = hybrid_mba_bend_roles(rows)
    audit = api.audit_lattice_constraints(rows, energy_gev=lattice.energy_gev,
        expected_length_m=expected_cell_length_m, expected_angle_rad=expected_cell_angle_rad,
        expected_bends=roles["bend_count"], require_mirror_symmetry=True, limits=limits,
        minimum_magnet_length_m=minimum_magnet_length_m,
        minimum_drift_length_m=minimum_drift_length_m)
    sampler = JuTrackProfileSampler(lattice, max_step_m=max_step_m)
    linear = sampler.linear_metrics()
    if not linear.get("stable"):
        return {"schema": SCHEMA, "lattice_id": lattice.lattice_id, "stage": stage,
                "scope": "complete_cell_shape_gate" if stage == "full_hybrid" else "complete_cell_bump_feasibility_gate",
                "passed": False, "checks": {"stable": False}, "stable": False,
                "failure": linear.get("failure", "The complete periodic cell is unstable."),
                "construction_audit": audit, "cell_fingerprint": cell_fingerprint(rows),
                "sampling": linear["sampling"], "one_turn_matrix": linear["one_turn_matrix"]}
    report = hybrid_mba_shape_diagnostics(rows, linear["twiss_profile"], thresholds=thresholds,
        sampling=linear["sampling"], lattice_id=lattice.lattice_id, stage=stage)
    report.update({"scope": "complete_cell_shape_gate", "stable": True, "construction_audit": audit,
                   "one_turn_matrix": linear["one_turn_matrix"],
                   "machine_contract": {"energy_gev": lattice.energy_gev,
                       "expected_cell_length_m": expected_cell_length_m,
                       "expected_cell_angle_rad": expected_cell_angle_rad,
                       "minimum_magnet_length_m": minimum_magnet_length_m,
                       "minimum_drift_length_m": minimum_drift_length_m}})
    periods = int(lattice.metadata.get("intended_repetitions", lattice.metadata.get("repetitions", lattice.periodicity)))
    closure = abs(expected_cell_angle_rad * periods - 2 * math.pi) <= 1e-9
    radiation = api.jutrack_periodic_radiation_metrics(sampler.jt, sampler.physical_editor.lattice,
        energy_gev=lattice.energy_gev, periods=periods) if closure else {}
    report["radiation"] = radiation
    report["machine_contract"]["periods"] = periods
    report["checks"].update({"stable": True, "construction_and_mirror": audit["passed"],
                             "ring_bend_closure": closure,
                             "positive_damping": bool(radiation.get("physical_validity", {}).get("passed"))})
    report["passed"] = all(report["checks"].values())
    if stage == "bump_feasibility":
        report["scope"] = "complete_cell_bump_feasibility_gate"
        report["limitations"].append("Intermediate usable-bump feasibility only; emittance recovery, chromatic correction and final acceptance remain pending. Source beta/dispersion preservation is optional, never inferred from ancestry.")
    from brilliance.result_presentation import remember_candidate_evidence
    remember_candidate_evidence(lattice, linear=linear)
    return report


def validate_hybrid_mba_shape_report(report: Any, lattice: Any = None) -> list[str]:
    """Recompute numeric gates; a caller's ``passed=True`` is not evidence.

    This is read-only validation of archived JuTrack evidence, not a rerun.
    Candidate identity, physical cell fingerprint, sampler provenance, machine
    audit and every shape check must agree. No radiation scalar can substitute.
    """
    from brilliance import accelerator_api as api

    if not isinstance(report, Mapping) or report.get("schema") != SCHEMA:
        return ["Missing supported hybrid-MBA shape evidence."]
    if report.get("stage", "full_hybrid") != "full_hybrid":
        return ["Bump feasibility is an intermediate stage, not final hybrid-MBA acceptance; core quality remains unvalidated."]
    if report.get("scope") != "complete_cell_shape_gate" or report.get("stable") is not True:
        return ["Hybrid acceptance needs complete stable-cell JuTrack shape evidence."]
    try:
        rows = api.one_period_elements(lattice) if lattice is not None else report["cell_rows"]
        if lattice is not None and report.get("lattice_id") != lattice.lattice_id:
            return ["Hybrid shape evidence belongs to a different lattice."]
        if report.get("cell_fingerprint") != cell_fingerprint(rows):
            return ["Hybrid shape evidence does not match these physical cell elements."]
        recomputed = hybrid_mba_shape_diagnostics(rows, report["profile"], thresholds=report["thresholds"],
            sampling=report["sampling"], lattice_id=report.get("lattice_id"))
        contract = report["machine_contract"]
        if lattice is not None and not math.isclose(lattice.energy_gev, float(contract["energy_gev"]), rel_tol=1e-12):
            return ["Hybrid shape evidence belongs to a different beam energy."]
        audit = api.audit_lattice_constraints(rows, energy_gev=float(contract["energy_gev"]),
            expected_length_m=float(contract["expected_cell_length_m"]),
            expected_angle_rad=float(contract["expected_cell_angle_rad"]),
            expected_bends=recomputed["roles"]["bend_count"], require_mirror_symmetry=True,
            limits=report["construction_audit"]["limits"],
            minimum_magnet_length_m=float(contract.get("minimum_magnet_length_m", 0.0)),
            minimum_drift_length_m=float(contract.get("minimum_drift_length_m", 0.0)))
        matrix = np.asarray(report["one_turn_matrix"], dtype=float)
        stable = bool(matrix.shape == (6, 6) and np.all(np.isfinite(matrix))
                      and abs(np.trace(matrix[:2, :2])) < 2 and abs(np.trace(matrix[2:4, 2:4])) < 2)
        from brilliance.lattice_optimization import radiation_physical_validity
        periods = int(contract["periods"])
        closure = periods > 0 and abs(float(contract["expected_cell_angle_rad"]) * periods - 2 * math.pi) <= 1e-9
        if lattice is not None:
            declared = int(lattice.metadata.get("intended_repetitions", lattice.metadata.get("repetitions", lattice.periodicity)))
            closure = closure and declared == periods
        checks = {**recomputed["checks"], "stable": stable, "construction_and_mirror": audit["passed"],
                  "ring_bend_closure": closure,
                  "positive_damping": radiation_physical_validity(report["radiation"])["passed"]}
        # Retain the meaning of older, explicitly preservation-gated archives;
        # new descendants need no baseline optical equality unless requested.
        preserve = report.get("preserve_core_optics", "core_optics_preserved" in report.get("checks", {}))
        if type(preserve) is not bool:
            return ["Invalid optical-preservation policy."]
        if preserve and not report.get("core_optics_reference"):
            return ["Explicit optical preservation requires a measured source-core optical reference."]
        if report.get("core_optics_reference"):
            from .hybrid_optics import compare_core_optics
            optical = compare_core_optics(report["core_optics_reference"], report["profile"],
                tolerances=report["core_optics"]["tolerances"])
            if preserve:
                checks["core_optics_preserved"] = optical["passed"]
            if optical != report.get("core_optics"):
                return ["Core optical comparison disagrees with measured evidence."]
        issues = ["Hybrid shape check failed: " + key for key, passed in checks.items() if not passed]
        if report.get("passed") is not all(checks.values()) or report.get("checks") != checks:
            issues.append("Hybrid shape flags disagree with recomputed evidence.")
        if (report.get("metrics") != recomputed["metrics"] or report.get("peaks") != recomputed["peaks"]
                or report.get("usable_drifts") != recomputed["usable_drifts"]):
            issues.append("Hybrid shape measurements disagree with the archived profile.")
        if report.get("roles") != recomputed["roles"]:
            issues.append("Hybrid bend roles disagree with physical order.")
        return issues
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
        return [f"Incomplete or invalid hybrid shape evidence: {type(exc).__name__}: {exc}"]
