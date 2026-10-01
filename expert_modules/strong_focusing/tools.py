"""Diagnostics for an externally computed uncoupled periodic transfer map."""
from __future__ import annotations

import numpy as np
from brilliance.expert_sdk import python_tool


@python_tool()
def periodic_plane_report(matrix, *, symplectic_tolerance: float) -> dict:
    """Inspect a real 2x2 map in (x [m], p_x/p0) at fixed reference momentum.

    Call separately for the two uncoupled transverse planes. Do not pass a
    diagonal block of a coupled map. Tolerance applies to abs(det(M)-1).
    Matched Twiss and wrapped phase are returned only for a symplectic,
    strictly elliptic map; identity and half-integer boundaries are excluded.
    This tool does not track, infer full-ring tune, or certify nonlinear stability.
    """
    m = np.asarray(matrix, dtype=float)
    tolerance = float(symplectic_tolerance)
    if m.shape != (2, 2) or not np.all(np.isfinite(m)):
        raise ValueError("matrix must be a finite real 2 x 2 array")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("symplectic_tolerance must be positive and finite")
    with np.errstate(over="ignore", invalid="ignore"):
        determinant = float(np.linalg.det(m))
        half_trace = float(np.trace(m) / 2)
    if not np.isfinite(determinant) or not np.isfinite(half_trace):
        raise ValueError("map diagnostics overflowed; check units")
    symplectic = abs(determinant - 1.0) <= tolerance
    elliptic = abs(half_trace) < 1.0
    defined = bool(symplectic and elliptic and m[0, 1] != 0)
    result = {
        "determinant": determinant, "symplectic_error": abs(determinant - 1.0),
        "symplectic_tolerance": tolerance, "symplectic_within_tolerance": bool(symplectic),
        "half_trace": half_trace, "trace_stability_margin": 1.0 - abs(half_trace),
        "periodic_solution_defined": defined,
        "phase_advance_rad_wrapped": None, "phase_fraction_wrapped": None,
        "beta_m": None, "alpha": None, "gamma_per_m": None,
    }
    if defined:
        sine = np.copysign(np.sqrt((1.0 - half_trace) * (1.0 + half_trace)), m[0, 1])
        phase = float(np.arctan2(sine, half_trace) % (2.0 * np.pi))
        beta = float(m[0, 1] / sine)
        alpha = float((m[0, 0] - m[1, 1]) / (2.0 * sine))
        gamma = float((1.0 + alpha * alpha) / beta)
        if not np.all(np.isfinite([beta, alpha, gamma])):
            raise ValueError("matched Twiss overflowed; map is ill-conditioned")
        result.update(phase_advance_rad_wrapped=phase,
                      phase_fraction_wrapped=phase / (2.0 * np.pi),
                      beta_m=beta, alpha=alpha, gamma_per_m=gamma)
    return result
