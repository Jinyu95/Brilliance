"""Measured numerical responses and bounded corrections, independent of physics models.

Callbacks must return the actual observables in a fixed order. For a mutable
simulation, use a private lattice and assign *every* requested coordinate on
every call. These helpers do not supply transport maps or scientific acceptance.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _vector(value: Any, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
    if size is not None and result.size != size:
        raise ValueError(f"{name} must contain {size} entries")
    return result.copy()


def _bounds(value: Any, size: int) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(value, dtype=float)
    if result.shape != (size, 2) or not np.all(np.isfinite(result)):
        raise ValueError(f"bounds must be a finite ({size}, 2) array of [lower, upper] pairs")
    lower, upper = result[:, 0].copy(), result[:, 1].copy()
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be smaller than its upper bound")
    return lower, upper


def _diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = (float(singular_values[0] / singular_values[-1])
                 if rank == min(matrix.shape) and singular_values[-1] > 0 else None)
    return {"rank": rank, "nullity": matrix.shape[1] - rank,
            "singular_values": singular_values.tolist(), "condition_number": condition}


def measure_response(
    evaluate: Callable[[np.ndarray], Sequence[float]],
    position: Sequence[float],
    *,
    steps: Sequence[float],
    bounds: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Measure a vector-valued callback's Jacobian without leaving declared bounds.

    ``steps`` contains one positive finite-difference displacement per coordinate,
    in its native units. Central differences are used when both probes fit;
    otherwise a bounded one-sided difference uses the side with more room.
    Probe errors propagate: failed evaluations never become zero derivatives.

    The callback is evaluated at ``position`` again in ``finally``, also on a
    failed probe, to restore a privately owned mutable simulation. A restoration
    failure cannot hide the original error. The repeated baseline and its change
    are recorded so the caller can inspect numerical/state reproducibility.
    Rank/conditioning here use native units, not a physical controllability test.
    """
    x = _vector(position, "position")
    step = _vector(steps, "steps", x.size)
    lower, upper = _bounds(bounds, x.size)
    if np.any(step <= 0):
        raise ValueError("steps must be positive")
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError("position must lie within bounds")
    output_size: int | None = None

    def call(at: np.ndarray) -> np.ndarray:
        return _vector(evaluate(at.copy()), "evaluate output", output_size)

    probes: list[dict[str, Any]] = []
    schemes: list[str] = []
    original_error: BaseException | None = None
    try:
        baseline = call(x)
        output_size = baseline.size
        matrix = np.empty((baseline.size, x.size))
        for index, requested_step in enumerate(step):
            left, right = x[index] - lower[index], upper[index] - x[index]
            if left >= requested_step and right >= requested_step:
                offsets = [-requested_step, requested_step]
                schemes.append("central")
            else:
                offsets = [min(requested_step, right) if right >= left
                           else -min(requested_step, left)]
                schemes.append("forward" if offsets[0] > 0 else "backward")
            sampled = []
            actual_offsets = []
            for offset in offsets:
                trial = x.copy()
                trial[index] = np.clip(x[index] + offset, lower[index], upper[index])
                actual_offset = trial[index] - x[index]
                if actual_offset == 0:
                    raise ValueError(f"step for coordinate {index} is below floating-point resolution")
                values = call(trial)
                probes.append({"coordinate": index, "offset": float(actual_offset),
                               "position": trial.tolist(), "values": values.tolist()})
                sampled.append(values)
                actual_offsets.append(actual_offset)
            if len(sampled) == 2:
                matrix[:, index] = ((sampled[1] - sampled[0])
                                    / (actual_offsets[1] - actual_offsets[0]))
            else:
                matrix[:, index] = (sampled[0] - baseline) / actual_offsets[0]
        if not np.all(np.isfinite(matrix)):
            raise ValueError("measured response contains nonfinite derivatives")
    except BaseException as exc:
        original_error = exc
        raise
    finally:
        try:
            restored = call(x)
        except Exception as exc:
            if original_error is None:
                raise RuntimeError("response measurement could not restore its baseline") from exc
            original_error.restoration_error = exc
            if hasattr(original_error, "add_note"):
                original_error.add_note(f"Baseline restoration also failed: {type(exc).__name__}: {exc}")
    return {"position": x.tolist(), "values": baseline.tolist(), "matrix": matrix.tolist(),
            "bounds": np.column_stack((lower, upper)).tolist(), "steps": step.tolist(),
            "schemes": schemes, "probes": probes, "evaluations": len(probes) + 2,
            "restored_values": restored.tolist(),
            "baseline_repeat_difference": (restored - baseline).tolist(),
            "diagnostic_scaling": "native coordinate and observable units",
            **_diagnostics(matrix)}


def bounded_response_correction(
    response: Mapping[str, Any],
    target: Sequence[float],
    *,
    bounds: Sequence[Sequence[float]],
    tolerance: Sequence[float],
) -> dict[str, Any]:
    """Propose bounded absolute coordinates from a previously measured response.

    Solve ``values + matrix @ (coordinates - position) = target``. For reachable
    targets a linear program minimizes the maximum ``abs(coordinate) / scale``,
    where each explicit engineering scale is ``max(abs(lower), abs(upper))``.
    This exploits available null-space freedom instead of merely taking a
    pseudoinverse. If the exact linear target is infeasible, bounded least
    squares finds a tolerance-weighted compromise and reports its residuals.

    Every output needs an explicit positive tolerance in its own units. Neither
    solver termination nor predicted satisfaction certifies a nonlinear match:
    apply the proposal to a private candidate and remeasure with the simulator.
    Bounds should limit the trusted response neighborhood as well as hardware.
    """
    from scipy.optimize import linprog, lsq_linear

    position = _vector(response["position"], "response position")
    baseline = _vector(response["values"], "response values")
    target_values = _vector(target, "target", baseline.size)
    tolerances = _vector(tolerance, "tolerance", baseline.size)
    if np.any(tolerances <= 0):
        raise ValueError("tolerance must be positive for every observable")
    matrix = np.asarray(response["matrix"], dtype=float)
    if matrix.shape != (baseline.size, position.size) or not np.all(np.isfinite(matrix)):
        raise ValueError("response matrix must be finite with shape (observables, coordinates)")
    lower, upper = _bounds(bounds, position.size)
    scales = np.maximum(np.abs(lower), np.abs(upper))
    scaled_matrix = matrix * scales[None, :] / tolerances[:, None]
    rhs = (target_values - baseline + matrix @ position) / tolerances
    if not np.all(np.isfinite(scaled_matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError("response scaling overflowed; review units, bounds, and tolerances")
    size = position.size
    # Variables are normalized absolute coordinates z and their infinity norm t.
    inequalities = np.vstack((np.column_stack((np.eye(size), -np.ones(size))),
                              np.column_stack((-np.eye(size), -np.ones(size)))))
    exact = linprog(
        np.r_[np.zeros(size), 1.0], A_ub=inequalities, b_ub=np.zeros(2 * size),
        A_eq=np.column_stack((scaled_matrix, np.zeros(baseline.size))), b_eq=rhs,
        bounds=[*(zip(lower / scales, upper / scales)), (0, None)], method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    if exact.success:
        normalized = exact.x[:size]
        method = "exact_target_minimax"
        solver_terminated = True
        message = exact.message
    else:
        fit = lsq_linear(scaled_matrix, rhs, bounds=(lower / scales, upper / scales),
                         tol=1e-12, lsmr_tol="auto", max_iter=300)
        normalized = fit.x
        method = "bounded_tolerance_weighted_least_squares"
        solver_terminated = bool(fit.success)
        message = fit.message
    coordinates = np.clip(normalized * scales, lower, upper)
    predicted = baseline + matrix @ (coordinates - position)
    residuals = predicted - target_values
    return {"coordinates": coordinates.tolist(), "target": target_values.tolist(),
            "predicted_values": predicted.tolist(), "predicted_residuals": residuals.tolist(),
            "tolerance": tolerances.tolist(),
            "within_tolerance": bool(np.all(np.abs(residuals) <= tolerances)),
            "method": method, "solver_terminated": solver_terminated, "solver_message": message,
            "exact_target_solver_message": exact.message,
            "normalization_scales": scales.tolist(),
            "peak_normalized_utilization": float(np.max(np.abs(coordinates) / scales)),
            "requires_simulator_remeasurement": True,
            "diagnostic_scaling": "coordinate engineering scales / observable tolerances",
            **_diagnostics(scaled_matrix)}
