"""Bounded open-line matching; every optics evaluation belongs to JuTrack.

This is a boundary-value operation, not an MBA designer. It does not assemble
a periodic cell, change geometry requirements, or certify a storage ring.
"""
from __future__ import annotations

import copy
import math
import os
import time
from contextlib import nullcontext
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from .api_result_types import LineMatchResult, MatchingStartPlan
from .lattice_optimization import LatticeParameterization, _apply_design_proposal, _target_arrays
from .line_optics import JuTrackLineEvaluator
from .research_outcomes import ResearchInfeasible

_ENDPOINT_FIELDS = {"beta_x_m", "beta_y_m", "alpha_x", "alpha_y", "Dx_m", "Dpx", "Dy_m", "Dpy"}


class _UnevaluableTrial(Exception):
    """Abort this local start, never replace failed physics with a flat residual."""


def plan_matching_starts(
    parameterization: LatticeParameterization, *, limits: Mapping[str, float],
    start_positions: Sequence[Sequence[float]] = (), max_starts: int | None = None,
    seed: int = 0,
) -> MatchingStartPlan:
    """Preflight a matching start budget without loading Julia or running physics.

    Supplied starts take priority, in input order, with exact duplicates merged.
    Add the unchanged baseline if valid and not already supplied, then fill spare
    slots with seeded Latin-hypercube points. max_starts is a hard TOTAL cap,
    not an extra allowance for an implicit baseline. Excess unique positions
    are returned in deferred_positions, never discarded or run beyond that cap.
    When omitted, the budget is max(64, 10*knobs, unique supplied + baseline).

    Validate every supplied position, including deferred ones, against both
    knob and engineering bounds; never clip or display-round them. An invalid
    automatic baseline is recorded in skipped. Positions are proposal coordinates;
    declared exact-sum projections are applied later, not physics-validated here.
    Duplicate provenance is retained in supplied_indices (zero-based input rows).
    This is a work plan, not evidence of feasible geometry or a successful match.
    """
    n = len(parameterization.knobs)
    if n == 0:
        raise ValueError("line matching requires at least one declared knob")
    for name, value, minimum in (("max_starts", max_starts, 1), ("seed", seed, 0)):
        if value is None and name == "max_starts":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    lower, upper = np.asarray(parameterization.bounds, dtype=float).T.copy()
    for i, knob in enumerate(parameterization.knobs):
        limit_key = {"k1_m2": "max_abs_k1_m2", "k2_m3": "max_abs_k2_m3", "k3_m4": "max_abs_k3_m4"}.get(knob.field)
        if limit_key:
            if limit_key not in limits:
                raise ValueError(f"Matching {knob.field} requires an explicit engineering limit: {limit_key}")
            cap = float(limits[limit_key])
            if not math.isfinite(cap) or cap <= 0:
                raise ValueError(f"invalid engineering limit: {limit_key}")
            lower[i], upper[i] = max(lower[i], -cap), min(upper[i], cap)
    if not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower):
        raise ValueError("line knobs need finite nonempty engineering-bounded intervals")

    def in_bounds(value):
        return (value.shape == (n,) and np.all(np.isfinite(value))
                and np.all(value >= lower) and np.all(value <= upper))

    positions, sources, provenance = [], [], []
    seen: dict[tuple[float, ...], int] = {}
    supplied_count = duplicate_count = 0
    for index, position in enumerate(start_positions):
        try:
            value = np.asarray(position, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"matching start_positions[{index}] must be a numeric vector of {n} knobs") from exc
        if not in_bounds(value):
            raise ValueError(f"matching start_positions[{index}] must contain {n} finite values "
                             "inside the engineering-bounded knobs")
        supplied_count += 1
        key = tuple(value.tolist())
        if key in seen:
            duplicate_count += 1
            provenance[seen[key]].append(index)
            continue
        seen[key] = len(positions)
        positions.append(value.tolist())
        sources.append("supplied")
        provenance.append([index])

    baseline = np.asarray(parameterization.initial_position, dtype=float)
    baseline_key = tuple(baseline.tolist())
    skipped = []
    if not in_bounds(baseline):
        skipped.append({"source": "baseline", "position": baseline.tolist(),
                        "reason": "baseline_outside_engineering_bounds"})
    elif baseline_key not in seen:
        positions.append(baseline.tolist())
        sources.append("baseline")
        provenance.append([])
    count = max(64, 10 * n, len(positions)) if max_starts is None else max_starts
    deferred, deferred_sources, deferred_indices = positions[count:], sources[count:], provenance[count:]
    positions, sources, provenance = positions[:count], sources[:count], provenance[:count]
    generated_count = count - len(positions)
    if generated_count:
        generated = lower + qmc.LatinHypercube(n, seed=seed).random(generated_count) * (upper - lower)
        positions.extend(generated.tolist())
        sources.extend(["generated"] * generated_count)
        provenance.extend([[] for _ in range(generated_count)])
    return {"positions": positions, "sources": sources, "supplied_indices": provenance,
            "deferred_positions": deferred, "deferred_sources": deferred_sources,
            "deferred_supplied_indices": deferred_indices, "skipped": skipped,
            "effective_knob_bounds": np.column_stack((lower, upper)).tolist(),
            "budget": {"requested_max_starts": max_starts, "total_starts": count,
                       "supplied_count": supplied_count, "duplicate_supplied_count": duplicate_count,
                       "baseline_included": baseline.tolist() in positions,
                       "generated_count": generated_count, "deferred_count": len(deferred)}}


def match_jutrack_line(
    elements: Sequence[Mapping[str, Any]], parameterization: LatticeParameterization, *,
    energy_gev: float, initial_twiss: Mapping[str, float], targets: Mapping[str, Any],
    expected_length_m: float, expected_angle_rad: float, limits: Mapping[str, float],
    maximum_beta_m: float | None,
    minimum_magnet_length_m: float = 0.0, minimum_drift_length_m: float = 0.0,
    profile_step_m: float = 0.25, start_positions: Sequence[Sequence[float]] = (),
    max_starts: int | None = None, max_nfev: int = 500, seed: int = 0,
    finite_difference_fraction: float = 1e-4,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> LineMatchResult:
    """Match a line from a fixed measured incoming state to explicit exit targets.

    The incoming state requires beta_x_m, beta_y_m, alpha_x, alpha_y, Dx_m,
    Dpx, Dy_m, Dpy. Targets are named exit values with positive absolute
    tolerances, e.g. {"Dx_m": (0.0, 1e-4), "Dpx": (0.0, 1e-5)}. No tune,
    ID beta, geometry or physics requirement is inferred. maximum_beta_m=None
    explicitly means no declared beta cap; maxima are sampled, not certified
    continuous extrema. All propagation uses the installed JuTrack twissline.

    Normalize coordinates by knob bounds and residuals by physical tolerance.
    Supplied starts have priority, exact duplicates are removed, then the valid
    baseline and a reproducible Latin hypercube fill the budget. max_starts is
    a hard total cap; overflow is deferred. Use plan_matching_starts for cheap
    preflight. Default max(64, 10*knobs, unique supplied + baseline). A failed start does not prove
    infeasibility. Invalid trials terminate that start without constant penalty
    maps. Unexpected API/program errors propagate. Cancellation is checked on
    every callback. max_nfev is per-start SciPy work, not a hard wall timeout;
    actual calls including finite differences and fresh replay are counted.

    Return exact elements, residual table, attempts, and separate
    solver_terminated / solver_success / match_success outcomes. Fresh JuTrack
    replay must pass all boundary, engineering and sampled-beta checks.
    complete_cell_accepted is always False: assemble and independently evaluate
    the actual periodic cell/ring before selecting a machine candidate.
    """
    names, target, tolerance = _target_arrays(targets)
    if set(names) - _ENDPOINT_FIELDS:
        raise ValueError("line targets must be exit Twiss/dispersion fields: " + ", ".join(sorted(_ENDPOINT_FIELDS)))
    n = len(parameterization.knobs)
    if n == 0:
        raise ValueError("line matching requires at least one declared knob")
    if (isinstance(finite_difference_fraction, bool) or not math.isfinite(finite_difference_fraction)
            or not 0 < finite_difference_fraction < 0.1):
        raise ValueError("finite_difference_fraction must be positive and less than 0.1")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be an integer >= 1")
    start_plan = plan_matching_starts(parameterization, limits=limits, start_positions=start_positions,
                                      max_starts=max_starts, seed=seed)
    count = start_plan["budget"]["total_starts"]
    lower, upper = np.asarray(start_plan["effective_knob_bounds"], dtype=float).T
    width = upper - lower
    seeds = np.asarray(start_plan["positions"], dtype=float)
    configuration = dict(energy_gev=energy_gev, initial_twiss=initial_twiss,
                         expected_length_m=expected_length_m, expected_angle_rad=expected_angle_rad,
                         limits=limits, maximum_beta_m=maximum_beta_m,
                         minimum_magnet_length_m=minimum_magnet_length_m,
                         minimum_drift_length_m=minimum_drift_length_m, profile_step_m=profile_step_m)
    evaluator = JuTrackLineEvaluator(elements, **configuration)
    if os.environ.get("BRILLIANCE_RUN_OUTPUT") and (progress_callback is None or cancellation_check is None):
        from .research_api import open_research_run

        run = open_research_run()
        if cancellation_check is None:
            cancellation_check = run.raise_if_cancelled
        if progress_callback is None:
            def progress_callback(update):
                final = update.get("phase") == "line_match_result"
                rows = update.get("residuals", update.get("best_boundary_residuals", []))
                run.progress("JuTrack boundary match verified" if final and update["match_success"] else
                             "JuTrack boundary match not verified" if final else "Matching the declared line boundary",
                             detail=("Native propagation, engineering limits and exit tolerances; "
                                     "complete-cell acceptance is a separate check."),
                             event_key="jutrack-line-match:" + ",".join(parameterization.variable_names),
                             kind="finding" if final else "progress",
                             metrics={key: value for key, value in update.items()
                                      if isinstance(value, (bool, int, float))},
                             tables=[{"title": "Measured exit residuals", "rows": rows}] if rows else [])
    calls = 0
    best: dict[str, Any] | None = None
    attempts = []
    last_publish = 0.0
    last_checkpoint = 0.0
    current_start = 0
    active_attempt: dict[str, Any] | None = None
    contract = {**copy.deepcopy(configuration), "targets": copy.deepcopy(dict(targets)),
                "finite_difference_fraction": finite_difference_fraction, "max_nfev_per_start": max_nfev}

    def pending_starts():
        # Include the in-flight start until it has finished. Resuming that one
        # may repeat partial work, but never silently loses an uncompleted seed.
        return {"start_plan": start_plan, "completed_starts": len(attempts),
                "remaining_start_positions": start_plan["positions"][len(attempts):] + start_plan["deferred_positions"],
                "remaining_start_sources": start_plan["sources"][len(attempts):] + start_plan["deferred_sources"]}

    def save_progress(*, force=False):
        nonlocal last_checkpoint
        if not os.environ.get("BRILLIANCE_RUN_OUTPUT"):
            return
        now = time.monotonic()
        if not force and last_checkpoint and now - last_checkpoint < 60.0:
            return
        from .numerical_checkpoints import write_numerical_checkpoint

        # Only observed numerical state: fresh independent replay and complete
        # machine acceptance have NOT occurred. Save before optional reporting.
        write_numerical_checkpoint("jutrack_line_match_progress", {
            "engine": "JuTrack", "scope": "open_line_boundary_match_progress",
            "match_success": False, "complete_cell_accepted": False,
            "complete_cell_acceptance": {"status": "not_evaluated"},
            "elements": best["elements"] if best else None,
            "position": best["position"] if best else None,
            "incoming_twiss": dict(initial_twiss), "contract": contract,
            "residuals": best["residuals"] if best else [],
            "measurement": best["measurement"] if best else None,
            "attempts": attempts, "active_attempt": active_attempt,
            "callback_evaluations": calls, "configured_starts": count,
            "effective_knob_bounds": start_plan["effective_knob_bounds"],
            "checks": {"fresh_jutrack_verification": False}, **pending_starts(),
        }, checks_passed=None)
        last_checkpoint = now

    def check_cancel():
        if cancellation_check is not None:
            cancellation_check()

    def publish(update):
        if progress_callback is not None:
            # Presentation code may format its copy, never the numerical record.
            progress_callback(copy.deepcopy(update))

    def boundary(result):
        values = np.array([float(result["endpoint"][name]) for name in names])
        residual = (values - target) / tolerance
        rows = [{"observable": name, "value": float(value), "target": float(goal),
                 "tolerance": float(tol), "scaled_residual": float(res), "passed": bool(abs(res) <= 1.0)}
                for name, value, goal, tol, res in zip(names, values, target, tolerance, residual)]
        return residual, rows

    def evaluate(position):
        nonlocal calls, best, last_publish
        check_cancel()
        try:
            rows = _apply_design_proposal(parameterization, position, elements)
        except ResearchInfeasible as exc:
            raise _UnevaluableTrial(str(exc)) from exc
        if active_attempt is not None and active_attempt["physical_initial_position"] is None:
            active_attempt["physical_initial_position"] = [
                float(rows[knob.target_indices[0]][knob.field]) for knob in parameterization.knobs]
        calls += 1
        result = evaluator.evaluate(rows)
        if not result["evaluable"]:
            raise _UnevaluableTrial(str(result.get("diagnostics", result.get("geometry", {}))))
        residual, residual_rows = boundary(result)
        if not np.all(np.isfinite(residual)):
            raise _UnevaluableTrial("JuTrack returned a nonfinite boundary residual")
        # This is a numerical inequality residual, NOT a physics propagation model.
        beta_residual = (max(0.0, float(result["metrics"]["maximum_beta_m"]) - maximum_beta_m)
                         / (maximum_beta_m * 1e-3) if maximum_beta_m is not None else 0.0)
        vector = np.append(residual, beta_residual)
        matched = bool(result["passed"] and all(row["passed"] for row in residual_rows))
        rank = (not matched, float(vector @ vector))
        if best is None or rank < best["rank"]:
            best = {"rank": rank, "elements": copy.deepcopy(rows), "position": list(map(float, position)),
                    "measurement": result, "residuals": residual_rows, "match_success": matched}
        save_progress()
        if progress_callback is not None and (calls == 1 or time.monotonic() - last_publish >= 5.0):
            publish({"phase": "line_matching_evaluation", "start": current_start,
                               "total_starts": count, "callback_evaluations": calls,
                               "best_boundary_residuals": best["residuals"],
                               "best_sampled_beta_m": best["measurement"]["metrics"]["maximum_beta_m"],
                               "match_success": False, "complete_cell_accepted": False})
            last_publish = time.monotonic()
        return vector

    for index, initial in enumerate(seeds):
        current_start = index + 1
        check_cancel()
        before = calls
        attempt: dict[str, Any] = {"start": index, "source": start_plan["sources"][index],
                                   "initial_position": initial.tolist(), "physical_initial_position": None,
                                   "solver_terminated": False, "solver_success": False,
                                   "solver_status": None, "nfev": None, "njev": None,
                                   "final_position": None, "residual_jacobian_rank": None,
                                   "residual_jacobian_singular_values": []}
        active_attempt = attempt
        if np.any(initial < lower) or np.any(initial > upper):
            attempt.update(reason="baseline_outside_engineering_bounds", callback_evaluations=0)
        else:
            try:
                solved = least_squares(lambda z: evaluate(lower + np.clip(z, 0, 1) * width),
                                       (initial - lower) / width, bounds=(np.zeros(n), np.ones(n)),
                                       max_nfev=max_nfev, diff_step=finite_difference_fraction,
                                       x_scale=1.0, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                attempt.update(solver_terminated=True, solver_success=bool(solved.success),
                               solver_status=int(solved.status), message=str(solved.message),
                               nfev=int(solved.nfev), njev=int(solved.njev or 0),
                               final_position=(lower + solved.x * width).tolist())
                # These are numerical residual sensitivities, never invented transport maps.
                singular = np.linalg.svd(np.asarray(solved.jac), compute_uv=False)
                attempt["residual_jacobian_singular_values"] = singular.tolist()
                attempt["residual_jacobian_rank"] = int(np.linalg.matrix_rank(solved.jac))
            except _UnevaluableTrial as exc:
                attempt.update(reason="unevaluable_line_trial", diagnostic=str(exc),
                               message="This local start stopped; no constant physics residual was fabricated.")
            attempt["callback_evaluations"] = calls - before
        attempts.append(attempt)
        active_attempt = None
        save_progress(force=True)
        if progress_callback is not None:
            publish({"phase": "line_matching", "completed_starts": index + 1, "total_starts": count,
                               "callback_evaluations": calls, "solver_terminated": attempt["solver_terminated"],
                               "solver_success": attempt["solver_success"],
                               "best_boundary_residuals": best["residuals"] if best else [],
                               "match_success": False, "complete_cell_accepted": False})
        if best is not None and best["match_success"]:
            break
    fresh = None
    residual_rows = best["residuals"] if best else []
    verified = False
    if best is not None:
        check_cancel()
        fresh = JuTrackLineEvaluator(best["elements"], **configuration).evaluate()
        calls += 1
        if fresh["evaluable"]:
            _, residual_rows = boundary(fresh)
            verified = bool(fresh["passed"] and all(row["passed"] for row in residual_rows))
    result = {
        "engine": "JuTrack", "scope": "open_line_boundary_match",
        "solver_terminated": any(item["solver_terminated"] for item in attempts),
        "solver_success": any(item["solver_success"] for item in attempts),
        "match_success": verified, "complete_cell_accepted": False,
        "complete_cell_acceptance": {"status": "not_evaluated"},
        "reason": "boundary_and_limits_verified" if verified else "boundary_match_not_verified",
        "elements": best["elements"] if best else None,
        "position": best["position"] if best else None, "incoming_twiss": dict(initial_twiss),
        "residuals": residual_rows, "measurement": fresh, "attempts": attempts,
        "callback_evaluations": calls, "configured_starts": count,
        **pending_starts(), "contract": contract,
        "effective_knob_bounds": np.column_stack((lower, upper)).tolist(),
        "checks": {"boundary_targets": bool(fresh and fresh["evaluable"] and all(row["passed"] for row in residual_rows)),
                   "line_and_engineering_limits": bool(fresh and fresh["passed"]),
                   "fresh_jutrack_verification": verified},
        "limitations": ["Open-line matching is not complete-cell stability, radiation or machine acceptance.",
                        "A failed bounded search does not diagnose insufficient space or topology exhaustion.",
                        "Beta limits apply to the reported JuTrack samples, not a certified continuous maximum."],
    }
    if os.environ.get("BRILLIANCE_RUN_OUTPUT"):
        from .numerical_checkpoints import write_numerical_checkpoint

        result["numerical_checkpoint"] = write_numerical_checkpoint(
            "jutrack_line_match", result, checks_passed=verified)
    if progress_callback is not None:
        publication = nullcontext()
        if all(os.environ.get(key) for key in (
                "BRILLIANCE_RUN_OUTPUT", "BRILLIANCE_WORKSPACE", "BRILLIANCE_RUN_ARGUMENTS", "BRILLIANCE_RUN_RESULT")):
            from .research_api import open_research_run

            publication = open_research_run().reporting(checkpoint=result["numerical_checkpoint"])
        # This callback formats an already verified, durably saved result. It is
        # deliberately separate from numerical evaluations and earlier callbacks.
        with publication:
            publish({"phase": "line_match_result", "match_success": verified,
                               "complete_cell_accepted": False, "callback_evaluations": calls,
                               "residuals": residual_rows, "checks": result["checks"]})
    return result
