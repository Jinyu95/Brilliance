"""Callback-based weighted fitting with explicit, bound-aware finite differences.

No automatic differentiation or machine-specific model is used here. Numerical
termination and local Jacobian diagnostics do not establish scientific validity.
"""
from collections import deque

import numpy as np
from scipy.optimize import least_squares


def fit_response_model(
    model, initial, lower, upper, parameter_scale, difference_step, observed, sigma,
    *, parameter_names, mask=None, prior_mean=None, prior_sigma=None, max_nfev,
    ftol=1e-8, xtol=1e-8, gtol=1e-8, progress=None,
):
    """Fit a deterministic ``model(parameter_vector) -> observation_vector``.

    Scales, absolute finite-difference steps and uncertainties may be positive
    scalars or vectors. Bounds and prior means are parameter-length vectors;
    infinite bounds are permitted. The observation mask is copied once: invalid
    active data/predictions fail, while inactive values never enter the fit.

    Central differences use the requested step where both sides fit in bounds;
    otherwise a one-sided step is shortened only as needed to stay in bounds.
    Priors add ``(parameters - prior_mean) / prior_sigma`` to the residual but
    are excluded from identifiability diagnostics. ``max_nfev`` limits SciPy's
    objective evaluations, NOT the total number of underlying model calls.

    Progress receives a fresh history record after each successful model call.
    Exceptions propagate; they are never converted into large penalty residuals.
    Final prediction and data Jacobian are independently recalculated and counted.
    Returned inactive residuals/nonfinite inactive predictions are JSON ``null``;
    a rank-deficient condition number is also ``null``, not infinity.
    """
    def vector(value, name, size=None, *, broadcast=False, finite=True):
        array = np.array(value, dtype=float, copy=True)
        if broadcast and array.ndim == 0:
            array = np.full(size, array.item())
        if array.ndim != 1 or not array.size or (size is not None and array.size != size):
            raise ValueError(f"{name} must be a nonempty vector of length {size}")
        if np.isnan(array).any() or (finite and not np.isfinite(array).all()):
            raise ValueError(f"{name} must contain valid {'finite ' if finite else ''}numbers")
        return array

    x0 = vector(initial, "initial")
    count = x0.size
    lo = vector(lower, "lower", count, finite=False)
    hi = vector(upper, "upper", count, finite=False)
    scale = vector(parameter_scale, "parameter_scale", count, broadcast=True)
    steps = vector(difference_step, "difference_step", count, broadcast=True)
    if np.any(lo >= hi) or np.any(x0 < lo) or np.any(x0 > hi):
        raise ValueError("Bounds must be ordered strictly and contain initial")
    if np.any(scale <= 0) or np.any(steps <= 0):
        raise ValueError("parameter_scale and difference_step must be strictly positive")
    if isinstance(parameter_names, (str, bytes)):
        raise ValueError("parameter_names must be a sequence, not a single string")
    names = list(parameter_names)
    if (len(names) != count or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != count):
        raise ValueError("parameter_names must be distinct nonempty strings, one per parameter")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)) or max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")
    for name, value in (("ftol", ftol), ("xtol", xtol), ("gtol", gtol)):
        if value is not None and (not np.isfinite(value) or value <= np.finfo(float).eps):
            raise ValueError(f"{name} must be None or finite and greater than machine epsilon")
    if all(value is None for value in (ftol, xtol, gtol)):
        raise ValueError("At least one termination tolerance must be enabled")
    if not callable(model) or (progress is not None and not callable(progress)):
        raise ValueError("model and optional progress must be callable")

    data = np.array(observed, dtype=float, copy=True)
    if data.ndim != 1 or not data.size:
        raise ValueError("observed must be a nonempty vector")
    active = np.ones(data.size, dtype=bool) if mask is None else np.array(mask, copy=True)
    if active.shape != data.shape or active.dtype != np.dtype(bool) or not active.any():
        raise ValueError("mask must be a boolean observation-length vector with active data")
    uncertainty = np.array(sigma, dtype=float, copy=True)
    if uncertainty.ndim == 0:
        uncertainty = np.full(data.size, uncertainty.item())
    if (uncertainty.shape != data.shape or not np.isfinite(data[active]).all()
            or not np.isfinite(uncertainty[active]).all() or np.any(uncertainty[active] <= 0)):
        raise ValueError("Active observed values must be finite and sigma finite and positive")
    if (prior_mean is None) != (prior_sigma is None):
        raise ValueError("prior_mean and prior_sigma must be supplied together")
    mean = None if prior_mean is None else vector(prior_mean, "prior_mean", count)
    prior = None if prior_sigma is None else vector(prior_sigma, "prior_sigma", count, broadcast=True)
    if prior is not None and np.any(prior <= 0):
        raise ValueError("prior_sigma must be strictly positive")

    evaluations = 0
    history = deque(maxlen=128)
    cache_x = cache_prediction = None

    def weighted(prediction):
        residual = (prediction[active] - data[active]) / uncertainty[active]
        if not np.isfinite(residual).all():
            raise ValueError("Active weighted residuals must be finite")
        return residual

    def evaluate(parameters, phase, *, reuse=False):
        nonlocal evaluations, cache_x, cache_prediction
        if not np.isfinite(parameters).all():
            raise ValueError("Model parameter trials must be finite")
        if reuse and cache_x is not None and np.array_equal(parameters, cache_x):
            return cache_prediction.copy()
        evaluations += 1
        prediction = np.array(model(parameters.copy()), dtype=float, copy=True)
        if prediction.shape != data.shape or not np.isfinite(prediction[active]).all():
            raise ValueError("Model prediction must match observed shape and be finite on the active mask")
        residual = weighted(prediction)
        chi_square = float(residual @ residual)
        if not np.isfinite(chi_square):
            raise ValueError("Weighted chi-square overflowed")
        record = {"evaluation": evaluations, "phase": phase, "chi_square": chi_square}
        history.append(record)
        if progress is not None:
            progress(dict(record))
        cache_x, cache_prediction = parameters.copy(), prediction.copy()
        return prediction

    def differences(parameters, baseline, phase):
        jacobian = np.empty((int(active.sum()), count))
        for column, requested in enumerate(steps):
            left, right = parameters[column] - lo[column], hi[column] - parameters[column]
            plus, minus = parameters.copy(), parameters.copy()
            if left >= requested and right >= requested:
                plus[column] += requested
                minus[column] -= requested
                if plus[column] == parameters[column] or minus[column] == parameters[column]:
                    raise ValueError("difference_step is not representable at the parameter value")
                denominator = plus[column] - minus[column]
                delta = evaluate(plus, phase)[active] - evaluate(minus, phase)[active]
            else:
                direction = 1 if right >= left else -1
                available = right if direction == 1 else left
                plus[column] = np.clip(parameters[column] + direction * min(requested, available),
                                       lo[column], hi[column])
                denominator = plus[column] - parameters[column]
                if denominator == 0 or not np.isfinite(denominator):
                    raise ValueError("difference_step is not representable within the parameter bounds")
                delta = evaluate(plus, phase)[active] - baseline[active]
            jacobian[:, column] = delta / denominator / uncertainty[active]
        if not np.isfinite(jacobian).all():
            raise ValueError("Finite-difference weighted Jacobian must be finite")
        return jacobian

    def residual(parameters):
        values = weighted(evaluate(parameters, "objective"))
        if prior is not None:
            values = np.concatenate((values, (parameters - mean) / prior))
        if not np.isfinite(values @ values):
            raise ValueError("Combined data and prior chi-square must be finite")
        return values

    def jacobian(parameters):
        nonlocal cache_x, cache_prediction
        baseline = evaluate(parameters, "jacobian_base", reuse=True)
        values = differences(parameters, baseline, "jacobian")
        cache_x, cache_prediction = parameters.copy(), baseline.copy()
        return values if prior is None else np.vstack((values, np.diag(1 / prior)))

    result = least_squares(residual, x0, jac=jacobian, bounds=(lo, hi), x_scale=scale,
                           max_nfev=int(max_nfev), ftol=ftol, xtol=xtol, gtol=gtol, method="trf")
    prediction = evaluate(result.x, "final_prediction")
    final_jacobian = differences(result.x, prediction, "final_jacobian") * scale
    if not np.isfinite(final_jacobian).all():
        raise ValueError("Scaled data Jacobian must be finite")
    singular = np.linalg.svd(final_jacobian, compute_uv=False)
    if not np.isfinite(singular).all():
        raise ValueError("Data Jacobian singular values must be finite")
    threshold = float(np.finfo(float).eps * max(final_jacobian.shape) * singular[0])
    rank = int(np.count_nonzero(singular > threshold))
    condition = float(singular[0] / singular[-1]) if rank == count else None
    final_weighted = weighted(prediction)
    prior_residual = np.array([]) if prior is None else (result.x - mean) / prior
    if not np.isfinite([result.cost, result.optimality, prior_residual @ prior_residual]).all():
        raise ValueError("Final solver and prior diagnostics must be finite")

    def json_vector(values):
        return [float(value) if np.isfinite(value) else None for value in values]

    physical = np.full(data.size, np.nan)
    normalized = physical.copy()
    physical[active] = prediction[active] - data[active]
    normalized[active] = final_weighted
    return {
        "parameter_names": names, "parameter_values": result.x.tolist(),
        "parameter_scale": scale.tolist(), "difference_step": steps.tolist(),
        "parameters": dict(zip(names, result.x.tolist())), "prediction": json_vector(prediction),
        "physical_residuals": json_vector(physical), "weighted_residuals": json_vector(normalized),
        "active_mask": active.tolist(), "chi_square": float(final_weighted @ final_weighted),
        "prior_chi_square": float(prior_residual @ prior_residual),
        "solver": {"status": int(result.status), "message": str(result.message),
                   "success": bool(result.success), "nfev": int(result.nfev),
                   "njev": int(result.njev), "optimality": float(result.optimality),
                   "cost": float(result.cost), "success_scope": "numerical termination only"},
        "callback_evaluations": evaluations, "history": list(history), "history_limit": 128,
        "identifiability": {"singular_values": singular.tolist(), "rank": rank,
                            "parameter_count": count, "rank_deficient": rank < count,
                            "condition_number": condition, "rank_threshold": threshold,
                            "jacobian": final_jacobian.tolist(),
                            "jacobian_definition": "data residual derivative times parameter_scale; no priors"},
        "active_bounds": {"mask": result.active_mask.tolist(),
                          "lower": [name for name, flag in zip(names, result.active_mask) if flag == -1],
                          "upper": [name for name, flag in zip(names, result.active_mask) if flag == 1]},
    }
