"""A fixed-geometry two-family TME/M-TME experiment, not an MBA controller."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import verify_candidate_evaluation
from brilliance.jutrack_service import native_element_options
from brilliance.lattice import LatticeElementIR
from brilliance.numerical_checkpoints import write_numerical_checkpoint as _write_checkpoint
from brilliance.optics_sampling import UnsupportedOpticsSampling
from brilliance.population_optimization import _publish_progress

from .physics_references import tme_targets
from .result_types import TmeUnitResult

_BOUNDARY = ("beta_x_m", "beta_y_m", "alpha_x", "alpha_y", "Dx_m", "Dpx", "Dy_m", "Dpy")


def write_numerical_checkpoint(kind, result, *, output_directory, checks_passed=None):
    return {**_write_checkpoint(kind, result, output_directory=output_directory, checks_passed=checks_passed),
            "source_output_directory": str(Path(output_directory).resolve())}


def _output_locations(output_directory):
    managed = os.environ.get("BRILLIANCE_RUN_OUTPUT")
    root = Path(managed or output_directory).resolve() if managed or output_directory else None
    storage = Path(output_directory).resolve() if output_directory is not None else root
    if managed and storage is not None and not storage.is_relative_to(root):
        raise ValueError("output_directory must be the managed run directory or its child; "
                         "omit it for managed storage. A workspace optimizer checkpoint is a separate checkpoint_path.")
    return storage, root


def _trace_limits(margin_x, margin_y):
    return {"equation": "abs(trace(M_plane)) <= 2 - trace_margin_plane",
            "maximum_abs_trace_x": 2.0 - margin_x, "maximum_abs_trace_y": 2.0 - margin_y,
            "margin_x": margin_x, "margin_y": margin_y,
            "interpretation": "Absolute distance from |trace|=2; not a normalized |trace|/2 threshold."}


def _managed_callbacks(progress_callback, cancellation_check, *, title, event_key):
    """Keep host progress and cancellation independent of optional project hooks."""
    for name, callback in (("progress_callback", progress_callback), ("cancellation_check", cancellation_check)):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None")
    run = None
    if os.environ.get("BRILLIANCE_RUN_OUTPUT") and not getattr(progress_callback, "_managed_numerical_observer", False):
        from brilliance.research_api import open_research_run
        run = open_research_run()

    def cancel():
        if run is not None:
            run.raise_if_cancelled()
        if cancellation_check is not None and cancellation_check() is True:
            raise InterruptedError("the scientist requested cancellation")

    def progress(update):
        if run is not None:
            metrics = {key: value for key, value in update.items()
                       if isinstance(value, (str, int, float, bool))}
            for label in ("best", "level", "trace_limits"):
                for key, value in (update.get(label) or {}).items():
                    if isinstance(value, (str, int, float, bool)):
                        metrics[f"{label}.{key}"] = value
            try:
                run.progress(title, kind="progress", event_key=event_key,
                    detail=str(update.get("phase", "numerical progress")), metrics=metrics)
            except Exception:
                pass  # UI failures cannot erase numerical work or disable cancellation.
        _publish_progress(progress_callback, str(update.get("phase", "progress")), update)

    progress._managed_numerical_observer = run is not None or getattr(progress_callback, "_managed_numerical_observer", False)
    return progress, cancel


def _check_central_bend(bend):
    options = native_element_options(LatticeElementIR.from_mapping(bend))
    if options.get("rad", 0) != 0:
        raise ValueError("The unit strength search requires conservative periodic optics (rad=0); supplied fields were not changed")
    for entrance, exit_, default in (("e1", "e2", 0.), ("fint1", "fint2", 0.),
                                   ("FringeBendEntrance", "FringeBendExit", 1),
                                   ("FringeQuadEntrance", "FringeQuadExit", 0)):
        if options.get(entrance, default) != options.get(exit_, default):
            raise ValueError(f"Central bend is not self-mirrored: {entrance} != {exit_}; supplied fields were not changed")
    for name in ("PolynomA", "PolynomB", "T1", "T2", "KickAngle", "FringeIntM0", "FringeIntP0"):
        if np.any(np.asarray(options.get(name, 0)) != 0):
            raise ValueError(f"A pure self-mirrored TME bend cannot certify nonzero {name}; supplied fields were not changed")
    for name in ("R1", "R2"):
        if name in options and not (np.array_equal(options[name], np.eye(6)) or not np.any(options[name])):
            raise ValueError(f"Central bend transform {name} has no supported mirror certificate")
    return {"passed": True, "native_options": options, "supplied_fields_preserved": True}


def _unit_rows(bend, lengths, near, far):
    d1, d2, d3, qnear, qfar = lengths
    left = [api.drift("TME:D3_L", d3), api.quadrupole("TME:QFAR_L", qfar, far, family="TME_FAR"),
            api.drift("TME:D2_L", d2), api.quadrupole("TME:QNEAR_L", qnear, near, family="TME_NEAR"),
            api.drift("TME:D1_L", d1)]
    right = [{**copy.deepcopy(row), "name": row["name"].replace("_L", "_R")} for row in reversed(left)]
    return [*left, copy.deepcopy(dict(bend)), *right]


def _cell(rows, energy, periods):
    return api.periodic_cell(title="Measured fixed-geometry TME/M-TME unit", energy_gev=energy,
                             elements=rows, intended_repetitions=periods, source="design_tme_unit: native two-family experiment")


def _linear_checks(linear, margin_x, margin_y, beta_cap):
    return {"stable_two_plane_map": linear.get("stable") is True,
            "trace_margin_x": abs(float(linear.get("trace_x", math.inf))) <= 2.0 - margin_x,
            "trace_margin_y": abs(float(linear.get("trace_y", math.inf))) <= 2.0 - margin_y,
            "sampled_beta_limit": beta_cap is None or float(linear.get("maximum_beta_m", math.inf)) <= beta_cap}


class _NativeUnitEvaluator:
    def __init__(self, rows, energy, periods, margin_x, margin_y, beta_cap, strength_cap):
        self.editor = api.build_jutrack_editor(_cell(rows, energy, periods))
        self.knobs = api.compile_family_parameterization(
            rows, field="k1_m2", families=["TME_NEAR", "TME_FAR"], bounds=(-strength_cap, strength_cap), role="outer_design")
        self.energy, self.periods = energy, periods
        self.margin_x, self.margin_y, self.beta_cap = margin_x, margin_y, beta_cap

    def __call__(self, position):
        self.editor.apply_parameterization(self.knobs, position)
        linear = api.jutrack_linear_metrics(self.editor.jt, self.editor.lattice, include_profile=True)
        checks = _linear_checks(linear, self.margin_x, self.margin_y, self.beta_cap)
        radiation = None
        if all(checks.values()):
            radiation = api.jutrack_periodic_radiation_metrics(
                self.editor.jt, self.editor.lattice, energy_gev=self.energy, periods=self.periods)
        checks["physical_radiation"] = bool(radiation and radiation.get("computed") is True
                                            and radiation.get("physical_validity", {}).get("passed") is True)
        return {"position": list(map(float, position)), "passed": all(checks.values()), "checks": checks,
                "linear_optics": {key: value for key, value in linear.items() if key != "twiss_profile"},
                "radiation": radiation, "emittance_pm": float(radiation["emittance_pm"]) if radiation else None}


def _distinct(records, count, cap, distance):
    chosen = []
    for row in sorted(records, key=lambda item: item["emittance_pm"]):
        if all(np.linalg.norm((np.asarray(row["position"]) - old["position"]) / cap) >= distance for old in chosen):
            chosen.append(row)
            if len(chosen) >= count:
                break
    return chosen


def _grid_regions(feasible, axis):
    """Eight-neighbour regions on this measured grid, not continuous basins."""
    indices = {float(value): index for index, value in enumerate(axis)}
    unassigned = {(indices[abs(r["position"][0])], indices[abs(r["position"][1])]): r for r in feasible}
    regions = []
    while unassigned:
        first = min(unassigned)
        pending, members = [first], [unassigned.pop(first)]
        while pending:
            i, j = pending.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                neighbour = (i + di, j + dj)
                if neighbour in unassigned:
                    members.append(unassigned.pop(neighbour))
                    pending.append(neighbour)
        regions.append(members)
    return sorted(regions, key=lambda region: min(r["emittance_pm"] for r in region))


def _strength_search(evaluate, *, cap, resolutions, local_starts, local_budget, progress, cancel,
                     checkpoint_directory=None, initial_strengths=()):
    cache, levels, local_attempts = {}, [], []

    def measured(position):
        key = tuple(map(float, position))
        if key not in cache:
            if cancel:
                cancel()
            cache[key] = evaluate(key)
        return cache[key]

    anchors = [measured(pair) for pair in initial_strengths]
    directions = [np.asarray(pair) for pair in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                                (1, 1), (-1, 1), (1, -1), (-1, -1))]
    for resolution in resolutions:
        before = len(cache)
        grid = []
        for polarity in (1, -1):
            axis = np.linspace(0., cap, resolution)[1:]
            for near in axis:
                for far in axis:
                    grid.append(measured((polarity * near, -polarity * far)))
        feasible = [row for row in grid if row["passed"]]
        level = {"resolution": resolution, "grid_proposals": len(grid), "feasible_grid_points": len(feasible),
                 "grid_best_emittance_pm": min((r["emittance_pm"] for r in feasible), default=None), "polarities": {}}
        local_results = []
        for polarity in (1, -1):
            branch = [row for row in feasible if row["position"][0] * polarity > 0]
            regions = _grid_regions(branch, axis)
            budget = max(1, local_starts // 2)
            starts = [min(region, key=lambda r: r["emittance_pm"]) for region in regions[:budget]]
            branch_anchors = [r for r in anchors if r["passed"] and r["position"][0] * polarity > 0]
            if branch_anchors:
                anchor = min(branch_anchors, key=lambda r: r["emittance_pm"])
                starts = [anchor, *[r for r in starts if r["position"] != anchor["position"]]][:budget]
            for row in sorted(branch, key=lambda r: r["emittance_pm"]):
                if len(starts) >= budget:
                    break
                if all(np.linalg.norm((np.asarray(row["position"]) - old["position"]) / cap)
                       >= max(.03, 1.5 / (resolution - 1)) for old in starts):
                    starts.append(row)
            for start in starts:
                local_before = len(cache)
                current, step, evaluations = start, cap / (resolution - 1), 0
                while evaluations + len(directions) <= local_budget and step > cap * 1e-7:
                    proposals = []
                    for direction in directions:
                        point = np.asarray(current["position"]) + step * direction
                        point[0] = polarity * np.clip(polarity * point[0], np.finfo(float).eps * cap, cap)
                        point[1] = -polarity * np.clip(-polarity * point[1], np.finfo(float).eps * cap, cap)
                        proposals.append(measured(point))
                    evaluations += len(directions)
                    best = min([current, *[r for r in proposals if r["passed"]]], key=lambda row: row["emittance_pm"])
                    if best["emittance_pm"] < current["emittance_pm"] * (1. - 1e-12):
                        current = best
                    else:
                        step *= .5
                local_results.append(current)
                local_attempts.append({"resolution": resolution, "polarity": polarity,
                    "start_position": start["position"], "start_emittance_pm": start["emittance_pm"],
                    "final_position": current["position"], "final_emittance_pm": current["emittance_pm"],
                    "proposal_count": evaluations, "unique_evaluations": len(cache) - local_before,
                    "termination": "strength_resolution_reached" if step <= cap * 1e-7 else "proposal_budget_exhausted",
                    "final_strength_step_m2": step})
            branch_polished = [r for r in local_results if r["position"][0] * polarity > 0]
            refined_positions = {tuple(r["position"]) for r in starts}
            refined_regions = [any(tuple(r["position"]) in refined_positions for r in region) for region in regions]
            level["polarities"][str(polarity)] = {"feasible_grid_points": len(branch), "distinct_local_starts": len(starts),
                "feasible_supplied_strengths": len(branch_anchors),
                "feasible_grid_regions": [{"region": index + 1, "grid_points": len(region),
                    "best_emittance_pm": min(r["emittance_pm"] for r in region), "locally_refined": refined_regions[index],
                    "best_position": min(region, key=lambda r: r["emittance_pm"])["position"]}
                    for index, region in enumerate(regions)],
                "unrefined_grid_regions": sum(not refined for refined in refined_regions),
                "polished_best_emittance_pm": min((r["emittance_pm"] for r in branch_polished), default=None)}
        level.update(unique_evaluations=len(cache) - before,
                     polished_best_emittance_pm=min((r["emittance_pm"] for r in local_results), default=None))
        previous = levels[-1]["polished_best_emittance_pm"] if levels else None
        current = level["polished_best_emittance_pm"]
        level["change_from_previous_resolution_pm"] = previous - current if previous is not None and current is not None else None
        level["relative_improvement_from_previous_resolution"] = (
            (previous - current) / previous if previous is not None and current is not None and previous > 0 else None)
        levels.append(level)
        if checkpoint_directory is not None:
            level["numerical_checkpoint"] = write_numerical_checkpoint("tme_unit_scan_level", {
                "resolution_levels": levels, "local_attempts": local_attempts,
                "samples": [r if r["passed"] else {"position": r["position"], "passed": False,
                    "checks": r["checks"], "emittance_pm": None} for r in cache.values()],
                "complete_mba_accepted": False}, output_directory=checkpoint_directory)
        _publish_progress(progress, "unit_strength_scan", {"level": level, "evaluations": len(cache)})
    return list(cache.values()), levels, local_attempts


def design_tme_unit(*, energy_gev: float, bend: Mapping[str, Any], d1_m: float, d2_m: float, d3_m: float,
                    quadrupole_near_length_m: float, quadrupole_far_length_m: float,
                    expected_length_m: float, reference_repetitions: int, limits: Mapping[str, float],
                    maximum_beta_m: float | None, trace_margin_x: float, trace_margin_y: float,
                    minimum_magnet_length_m: float, minimum_drift_length_m: float,
                    scan_resolutions: Sequence[int] = (25, 49), local_starts: int = 12,
                    local_max_evaluations: int = 240, candidates_to_return: int = 4, profile_step_m: float = .05,
                    initial_strengths: Sequence[Sequence[float]] = (),
                    output_directory: Path | str | None = None,
                    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
                    cancellation_check: Callable[[], None] | None = None) -> TmeUnitResult:
    """Design complete fixed-geometry units by native emittance, not TME equality.

    Rows are d3,Qfar,d2,Qnear,d1,B,d1,Qnear,d2,Qfar,d3. The supplied homogeneous
    SBEND, including all edge fields, is preserved; only two mirror-paired K1
    values vary. Both opposite-sign assignments are scanned and locally refined
    at every resolution. Geometry, energy-dependent limits, trace margins and
    sampled-beta cap are explicit inputs. No unit-boundary achromat, phase/tune,
    alpha or bend-centre equality is optimized. Several distinct feasible units
    are rechecked with JuTrack, including explicit repeated reference rings.
    Optional initial_strengths are remeasured and retained as local starts so a
    geometry-search winner is not lost between finite grid points.
    This is a seed experiment, never complete-MBA or machine acceptance.
    """
    lengths = tuple(map(float, (d1_m, d2_m, d3_m, quadrupole_near_length_m, quadrupole_far_length_m)))
    if any(not math.isfinite(value) or value <= 0 for value in lengths):
        raise ValueError("Every unit drift and quadrupole length must be positive and finite")
    if bend.get("kind") != "SBEND" or float(bend.get("angle_rad") or 0) <= 0:
        raise ValueError("A TME unit requires one explicitly supplied positive homogeneous SBEND")
    if any(float(bend.get(key) or 0) != 0 for key in ("k1_m2", "k2_m3", "k3_m4")):
        raise ValueError("This TME experiment does not vary or allow combined-function bend strengths")
    if (isinstance(reference_repetitions, bool) or not isinstance(reference_repetitions, int)
            or reference_repetitions < 1 or abs(reference_repetitions * float(bend["angle_rad"]) - 2 * math.pi) > 1e-9):
        raise ValueError("Reference repetitions must be an integer giving exact 2*pi bend closure")
    for name, value in (("trace_margin_x", trace_margin_x), ("trace_margin_y", trace_margin_y)):
        if not math.isfinite(value) or not 0 <= value < 2:
            raise ValueError(f"{name} must be finite in [0,2)")
    if maximum_beta_m is not None and (not math.isfinite(maximum_beta_m) or maximum_beta_m <= 0):
        raise ValueError("maximum_beta_m must be positive or explicitly None")
    if not math.isfinite(profile_step_m) or profile_step_m <= 0:
        raise ValueError("profile_step_m must be positive and finite")
    resolutions = tuple(scan_resolutions)
    if (len(resolutions) < 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in resolutions)
            or any(b <= a for a, b in zip(resolutions, resolutions[1:]))):
        raise ValueError("scan_resolutions needs at least two increasing integers >=3")
    for name, value, minimum in (("local_starts", local_starts, 2), ("local_max_evaluations", local_max_evaluations, 8),
                                 ("candidates_to_return", candidates_to_return, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    cap = float(limits["max_abs_k1_m2"])
    anchors = []
    for pair in initial_strengths:
        values = tuple(map(float, pair))
        if (len(values) != 2 or not all(math.isfinite(v) and abs(v) <= cap for v in values)
                or values[0] * values[1] >= 0):
            raise ValueError("initial_strengths requires finite in-bound opposite-sign near/far pairs")
        anchors.append(values)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("max_abs_k1_m2 must be positive and finite")
    bend_mirror = _check_central_bend(bend)
    storage, evidence_root = _output_locations(output_directory)
    rows = _unit_rows(bend, lengths, cap / 2, -cap / 2)
    audit = api.audit_lattice_constraints(rows, energy_gev=energy_gev, expected_length_m=expected_length_m,
        expected_angle_rad=float(bend["angle_rad"]), expected_bends=1, require_mirror_symmetry=True,
        limits=limits, minimum_magnet_length_m=minimum_magnet_length_m, minimum_drift_length_m=minimum_drift_length_m)
    if not audit["passed"]:
        raise ValueError("Fixed unit geometry violates its declared contract: " + "; ".join(audit["issues"]))
    progress_callback, cancellation_check = _managed_callbacks(progress_callback, cancellation_check,
        title="Fixed-geometry unit strength scan", event_key="tme-unit-scan")
    trace_limits = _trace_limits(trace_margin_x, trace_margin_y)
    _publish_progress(progress_callback, "resolved_unit_constraints", {"trace_limits": trace_limits,
        "maximum_beta_m": maximum_beta_m, "expected_length_m": expected_length_m})
    cancellation_check()
    native = _NativeUnitEvaluator(rows, energy_gev, reference_repetitions, trace_margin_x, trace_margin_y, maximum_beta_m, cap)
    samples, levels, attempts = _strength_search(native, cap=cap, resolutions=resolutions,
        local_starts=local_starts, local_budget=local_max_evaluations, progress=progress_callback,
        cancel=cancellation_check, checkpoint_directory=evidence_root, initial_strengths=anchors)
    feasible = [row for row in samples if row["passed"]]
    # Preserve both feasible polarity branches even if one has larger emittance.
    selected = []
    for polarity in (1, -1):
        selected.extend(_distinct([r for r in feasible if r["position"][0] * polarity > 0], 1, cap, .005))
    for row in _distinct(feasible, candidates_to_return * 8, cap, .005):
        if all(np.linalg.norm((np.asarray(row["position"]) - old["position"]) / cap) >= .005 for old in selected):
            selected.append(row)
    candidates, rejected = [], []
    for trial in selected:
        polarity = "near_positive" if trial["position"][0] > 0 else "near_negative"
        if (len({r["position"][0] > 0 for r in feasible}) == 2
                and sum(c["polarity"] == polarity for c in candidates) >= candidates_to_return - 1):
            continue  # Reserve a place for the other feasible sign branch.
        if cancellation_check:
            cancellation_check()
        candidate_rows = _unit_rows(bend, lengths, *trial["position"])
        candidate = _cell(candidate_rows, energy_gev, reference_repetitions)
        evaluation = api.evaluate_periodic_cell(candidate, include_profile=True)
        linear, radiation = evaluation["linear_optics"], evaluation["radiation"]
        checks = _linear_checks(linear, trace_margin_x, trace_margin_y, maximum_beta_m)
        candidate_audit = api.audit_lattice_constraints(candidate_rows, energy_gev=energy_gev,
            expected_length_m=expected_length_m, expected_angle_rad=float(bend["angle_rad"]),
            expected_bends=1, require_mirror_symmetry=True, limits=limits,
            minimum_magnet_length_m=minimum_magnet_length_m, minimum_drift_length_m=minimum_drift_length_m)
        checks["construction_and_engineering"] = candidate_audit["passed"]
        checks["physical_radiation"] = bool(radiation.get("computed") and radiation.get("physical_validity", {}).get("passed"))
        if not all(checks.values()):
            rejected.append({"position": trial["position"], "checks": checks})
            continue
        saved = None
        if storage is not None:
            bound = verify_candidate_evaluation(candidate, evaluation, storage)
            if not bound["passed"]:
                raise ValueError("Fresh unit evaluation could not be archived: " + "; ".join(bound["issues"]))
            saved = write_numerical_checkpoint("candidate_evaluation", {"name": "tme_unit",
                "candidate_lattice": candidate.as_dict(), "candidate_evaluation": bound["receipt"],
                "boundary_twiss": {key: float(linear[key]) for key in _BOUNDARY}}, output_directory=evidence_root)
        ring = api.cell_ring(title="Identical-unit reference ring, not the requested MBA", energy_gev=energy_gev,
                             cell=candidate_rows, periods=reference_repetitions, source="design_tme_unit: independent reference check")
        ring_evaluation = api.evaluate(ring, include_profile=False, include_chromaticity=False)
        ring_result = ring_evaluation.as_dict() if hasattr(ring_evaluation, "as_dict") else dict(ring_evaluation)
        ring_radiation = ring_result.get("radiation", {})
        checks["reference_ring_stable"] = ring_result.get("linear_optics", {}).get("stable") is True
        checks["reference_ring_physical_radiation"] = ring_radiation.get("physical_validity", {}).get("passed") is True
        checks["reference_emittance_agrees"] = bool(ring_radiation.get("computed") and math.isclose(
            float(ring_radiation["emittance_pm"]), float(radiation["emittance_pm"]), rel_tol=1e-4, abs_tol=1e-8))
        if storage is not None:
            ring_bound = verify_candidate_evaluation(ring, ring_evaluation, storage)
            if not ring_bound["passed"]:
                raise ValueError("Reference-ring evaluation could not be archived: " + "; ".join(ring_bound["issues"]))
        else:
            ring_bound = {"receipt": None}
        diagnostic = {"status": "not_evaluated", "required_for_emittance_search": False}
        try:
            profile = api.build_jutrack_profile_sampler(candidate, max_step_m=profile_step_m).linear_metrics(
                sample_fractions={str(bend["name"]): [.5]})
            if profile.get("stable") and profile.get("sampling", {}).get("validated"):
                centre = next(r for r in profile["twiss_profile"] if r.get("requested_sample")
                              and r.get("source_element") == bend["name"] and r.get("sample_fraction") == .5)
                diagnostic = {"status": "measured", "centre": centre, "sampling": profile["sampling"],
                              "maximum_beta_m": max(max(r["beta_x"], r["beta_y"]) for r in profile["twiss_profile"])}
                checks["sampled_beta_limit"] = maximum_beta_m is None or diagnostic["maximum_beta_m"] <= maximum_beta_m
            else:
                diagnostic = {"status": "unavailable", "diagnostic_validation_passed": False,
                    "reason": profile.get("failure") or "Optional interior profile was not validated against the physical map",
                    "sampling": profile.get("sampling")}
        except UnsupportedOpticsSampling as exc:
            # Unsupported true edge fields must survive, not be erased for a plot.
            diagnostic = {"status": "unavailable", "reason": str(exc), "physical_bend_unchanged": True}
        if not all(checks.values()):
            rejected.append({"position": trial["position"], "checks": checks})
            continue
        verified_candidate = {"position": trial["position"], "k1_near_m2": trial["position"][0],
            "k1_far_m2": trial["position"][1], "polarity": polarity,
            "elements": candidate_rows, "periodic_cell": candidate.as_dict(), "evaluation": evaluation,
            "reference_ring_evaluation": ring_result, "reference_ring_receipt": ring_bound["receipt"],
            "boundary_twiss": {key: float(linear[key]) for key in _BOUNDARY}, "checks": checks,
            "construction_and_engineering": candidate_audit,
            "emittance_pm": float(radiation["emittance_pm"]), "numerical_checkpoint": saved,
            "bend_centre_diagnostic": diagnostic,
            "optics_profile": (profile["twiss_profile"] if diagnostic["status"] == "measured"
                               else linear.get("twiss_profile", [])),
            "beta_sampling_scope": "native_map_checked_profile" if diagnostic["status"] == "measured" else "native_element_boundaries",
            "complete_mba_accepted": False}
        if storage is not None:
            verified_candidate["verified_unit_checkpoint"] = write_numerical_checkpoint(
                "verified_tme_unit", verified_candidate, output_directory=evidence_root, checks_passed=True)
        candidates.append(verified_candidate)
        if len(candidates) >= candidates_to_return:
            break
    candidates.sort(key=lambda row: row["emittance_pm"])
    reference = tme_targets(energy_gev=energy_gev, bend_length_m=float(bend["length_m"]), bend_angle_rad=float(bend["angle_rad"])).as_dict()
    result = {"method": "deterministic_two_family_native_emittance_search", "engine": "JuTrack",
        "passed": bool(candidates), "candidates": candidates, "fixed_geometry_audit": audit,
        "central_bend_mirror_audit": bend_mirror,
        "provided_constraints": {"energy_gev": energy_gev, "expected_length_m": expected_length_m,
            "maximum_beta_m": maximum_beta_m, "trace_margin_x": trace_margin_x, "trace_margin_y": trace_margin_y,
            "minimum_magnet_length_m": minimum_magnet_length_m, "minimum_drift_length_m": minimum_drift_length_m,
            "limits": dict(limits), "reference_repetitions": reference_repetitions, "trace_limits": trace_limits},
        "knobs": [{"name": "K1_near", "elements": ["TME:QNEAR_L", "TME:QNEAR_R"]},
                  {"name": "K1_far", "elements": ["TME:QFAR_L", "TME:QFAR_R"]}],
        "strength_bounds_m2": [-cap, cap], "reference_physics": reference, "resolution_levels": levels,
        "polarity_comparison": [{"polarity": polarity, "verified_candidate_count": len(branch),
            "best_verified_emittance_pm": min((r["emittance_pm"] for r in branch), default=None)}
            for polarity in ("near_positive", "near_negative")
            for branch in [[r for r in candidates if r["polarity"] == polarity]]],
        "resolution_convergence_demonstrated": False,
        "local_attempts": attempts, "unique_evaluations": len(samples), "feasible_evaluations": len(feasible),
        "supplied_initial_strengths": [list(pair) for pair in anchors],
        "scan_samples": [{"k1_near_m2": r["position"][0], "k1_far_m2": r["position"][1],
                          "emittance_pm": r["emittance_pm"] if r["passed"] else None, "passed": r["passed"]} for r in samples],
        "rejected_final_candidates": rejected, "complete_mba_accepted": False, "optimality_demonstrated": False,
        "limitations": ["Fixed geometry only; geometry-bank selection belongs to the scientist or research program.",
                        "Both polarity branches were considered. A failed finite scan does not prove infeasibility.",
                        "Reference-ring emittance is not the emittance of a subsequently assembled MBA.",
                        "Beta maxima are sampled, not continuously certified; preserved edge fields may prevent centre diagnostics."]}
    if storage is not None:
        result["numerical_checkpoint"] = write_numerical_checkpoint("tme_unit_search", result, output_directory=evidence_root,
                                                                  checks_passed=bool(candidates))
    return result
