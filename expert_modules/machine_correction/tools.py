"""Independent offline LOCO primitives; no accelerator-control connection."""
from __future__ import annotations

import numpy as np

from brilliance.expert_sdk import python_tool

from .fitting import fit_response_model as _fit
from .fitting import load_fit_checkpoint as _load_fit_checkpoint
from .measurements import align_observations as _align
from .measurements import read_loco_measurements as _read_measurements
from .measurements import read_named_settings as _read_settings
from .native_response import jutrack_response as _response
from .native_response import validate_jutrack_response as _validate_response

read_loco_measurements = python_tool()(_read_measurements)
read_named_settings = python_tool()(_read_settings)
align_observations = python_tool()(_align)
fit_response_model = python_tool(durable=True)(_fit)
load_fit_checkpoint = python_tool()(_load_fit_checkpoint)
jutrack_response = python_tool()(_response)
validate_jutrack_response = python_tool()(_validate_response)


def _finite(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numbers")
    return array


@python_tool()
def calibrate_response(response_m_per_rad, cross_response_m_per_rad, *,
                       bpm_matrices, corrector_gain, corrector_cross_gain,
                       dispersion_m=None, energy_shift_per_kick=None) -> dict:
    """Apply explicit device calibration to a native coupled orbit response.

    Rows are all x then all y; columns are named H/V commanded kicks. Each
    BPM matrix is [[g_xx, g_xy], [g_yx, g_yy]], output = matrix @ true orbit.
    Main/cross gains multiply native kicks in the commanded/orthogonal plane
    at the SAME corrector, not columns from another corrector location.
    Optional energy_shift_per_kick is d(delta)/d(commanded kick), in rad^-1.
    Do not add this term to a model already including that energy response.
    """
    main = _finite(response_m_per_rad, "response")
    cross = _finite(cross_response_m_per_rad, "cross response")
    if main.ndim != 2 or not main.size or main.shape[0] % 2 or cross.shape != main.shape:
        raise ValueError("response and cross response must share shape (2*N_BPM, N_corrector)")
    n, m = main.shape[0] // 2, main.shape[1]
    matrices = _finite(bpm_matrices, "BPM matrices")
    gains = _finite(corrector_gain, "corrector gain")
    leaks = _finite(corrector_cross_gain, "corrector cross gain")
    if matrices.shape != (n, 2, 2) or gains.shape != (m,) or leaks.shape != (m,):
        raise ValueError("supply N_BPM 2x2 matrices and N_corrector main/cross gains")
    with np.errstate(over="ignore", invalid="ignore"):
        true_response = main * gains[None, :] + cross * leaks[None, :]
    _finite(true_response, "calibrated corrector response")
    dispersion = None
    if dispersion_m is not None:
        dispersion = _finite(dispersion_m, "dispersion")
        if dispersion.shape != (2 * n,):
            raise ValueError("dispersion_m must have all-x-then-all-y length 2*N_BPM")
    if energy_shift_per_kick is not None:
        shift = _finite(energy_shift_per_kick, "energy shift per kick")
        if dispersion is None or shift.shape != (m,):
            raise ValueError("energy shift requires dispersion and N_corrector coefficients")
        with np.errstate(over="ignore", invalid="ignore"):
            true_response += np.outer(dispersion, shift)
        _finite(true_response, "energy-shift response")
    pairs = np.stack((true_response[:n], true_response[n:]), axis=1)
    measured = np.einsum("nij,njk->nik", matrices, pairs)
    _finite(measured, "calibrated BPM response")
    output = {"response_m_per_rad": np.concatenate((measured[:, 0], measured[:, 1])).tolist(),
              "row_order": "all_x_then_all_y"}
    if dispersion is not None:
        pairs_d = np.stack((dispersion[:n], dispersion[n:]), axis=1)
        measured_d = np.einsum("nij,nj->ni", matrices, pairs_d)
        _finite(measured_d, "calibrated BPM dispersion")
        output["dispersion_m"] = np.concatenate((measured_d[:, 0], measured_d[:, 1])).tolist()
    return output


@python_tool()
def rf_slope_to_dispersion(slope_m_per_hz, *, rf_frequency_hz: float,
                           slip_factor: float) -> dict:
    """Convert dx/df to dx/d(delta) using delta = -df/(f_RF * eta).

    eta = alpha_c - 1/gamma^2 must be supplied for the actual reference state.
    This first-order fixed-revolution-frequency relation is not a 6D RF model.
    Apply the same multiplier in absolute value to slope uncertainties.
    """
    slope = _finite(slope_m_per_hz, "RF slope")
    if not slope.size:
        raise ValueError("RF slope must not be empty")
    frequency = float(rf_frequency_hz)
    eta = float(slip_factor)
    if not np.isfinite(frequency) or frequency <= 0 or not np.isfinite(eta) or eta == 0:
        raise ValueError("positive finite RF frequency and nonzero finite slip factor are required")
    factor = -frequency * eta
    if not np.isfinite(factor):
        raise ValueError("RF conversion factor overflowed; check units")
    with np.errstate(over="ignore", invalid="ignore"):
        dispersion = slope * factor
    if not np.all(np.isfinite(dispersion)):
        raise ValueError("RF conversion overflowed; check units")
    return {"dispersion_m": dispersion.tolist(), "slope_to_dispersion_hz": factor,
            "rf_frequency_hz": frequency, "slip_factor": eta}


@python_tool()
def bounded_correction_step(jacobian, error, lower_delta, upper_delta, *,
                            actuator_names, actuator_scale, sigma,
                            regularization: float = 0.0,
                            step_fraction: float = 1.0) -> dict:
    """Propose bounded actuator deltas solving J*delta + error approximately zero.

    error = current - target; J = d(observable)/d(actuator), calculated from
    the fitted JuTrack model in approved hardware units. Bounds are allowable
    DELTAS, not absolute settings, and must contain zero. Minimize weighted
    residual squared plus regularization*sum((delta/actuator_scale)**2).
    Returned residuals are LINEAR PREDICTIONS; reevaluate the exact candidate
    with JuTrack, then obtain approval before any real machine change.
    """
    from scipy.optimize import lsq_linear

    matrix = _finite(jacobian, "jacobian")
    residual = _finite(error, "error")
    low = _finite(lower_delta, "lower_delta")
    high = _finite(upper_delta, "upper_delta")
    scales = _finite(actuator_scale, "actuator_scale")
    if matrix.ndim != 2 or not matrix.size or residual.shape != (matrix.shape[0],):
        raise ValueError("jacobian must be nonempty (observations, actuators), error an observation vector")
    count = matrix.shape[1]
    if isinstance(actuator_names, (str, bytes)):
        raise ValueError("actuator_names must be a sequence of distinct names")
    names = list(actuator_names)
    if (len(names) != count or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != count):
        raise ValueError("provide one unique nonempty name per actuator")
    if any(vector.shape != (count,) for vector in (low, high, scales)):
        raise ValueError("bounds and actuator_scale must contain one value per actuator")
    if np.any(low >= high) or np.any(low > 0) or np.any(high < 0) or np.any(scales <= 0):
        raise ValueError("ordered delta bounds must contain zero; actuator scales must be positive")
    noise = np.broadcast_to(_finite(sigma, "sigma"), residual.shape)
    if np.any(noise <= 0):
        raise ValueError("sigma must be positive")
    penalty, fraction = float(regularization), float(step_fraction)
    if not np.isfinite(penalty) or penalty < 0 or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("regularization must be nonnegative; step_fraction must be in (0,1]")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        data_matrix = matrix * scales[None, :] / noise[:, None]
        target = -residual / noise
    _finite(data_matrix, "scaled data Jacobian")
    _finite(target, "weighted target")
    with np.errstate(over="ignore", invalid="ignore"):
        initial_cost = float(target @ target)
    if not np.isfinite(initial_cost):
        raise ValueError("Weighted correction objective overflowed; check units and uncertainties")
    system = data_matrix
    if penalty:
        system = np.vstack((system, np.sqrt(penalty) * np.eye(count)))
        target = np.concatenate((target, np.zeros(count)))
    fit = lsq_linear(system, target, bounds=(low / scales, high / scales))
    delta = fit.x * scales * fraction
    predicted = residual + matrix @ delta
    _finite(delta, "proposed correction")
    _finite(predicted, "predicted error")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        final_cost = float((predicted / noise) @ (predicted / noise))
    if not np.isfinite(final_cost):
        raise ValueError("Predicted correction objective overflowed")
    singular = np.linalg.svd(data_matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(data_matrix))
    return {"actuator_names": names, "delta": delta.tolist(),
            "proposal": dict(zip(names, delta.tolist())),
            "weighted_rms_before": float(np.sqrt(initial_cost / residual.size)),
            "predicted_weighted_rms_after": float(np.sqrt(final_cost / residual.size)),
            "predicted_error": predicted.tolist(), "data_singular_values": singular.tolist(),
            "data_rank": rank, "regularization": penalty, "step_fraction": fraction,
            "solver_terminated": bool(fit.success), "solver_message": str(fit.message),
            "status": "offline_linear_proposal", "machine_applied": False}
