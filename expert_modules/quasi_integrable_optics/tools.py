"""Octupole construction and invariant diagnostics; use JuTrack for dynamics."""
from __future__ import annotations

import numpy as np
from brilliance.expert_sdk import python_tool


@python_tool()
def octupole_strength_profile(beta_m, *, beta_reference_m: float,
                              k3_reference_m4: float) -> dict:
    """Return k3(s) = k3_reference * (beta_reference/beta(s))**3.

    beta_m is the common horizontal/vertical beta sampled in an ideal
    dispersion-free insert. k3 is the normalized third field derivative in
    m^-4 (JuTrack KOCT convention), not PolynomB[4] or an integrated strength.
    This sets a hypothesis; matching and segmentation must be verified separately.
    """
    beta = np.asarray(beta_m, dtype=float)
    reference = float(beta_reference_m)
    strength = float(k3_reference_m4)
    if beta.ndim != 1 or not beta.size or not np.all(np.isfinite(beta)) or np.any(beta <= 0):
        raise ValueError("beta_m must be a nonempty positive finite vector")
    if not np.isfinite(reference) or reference <= 0 or not np.isfinite(strength):
        raise ValueError("reference beta must be positive and k3 finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        k3 = strength * (reference / beta) ** 3
        invariant_coefficient = strength * np.float64(reference) ** 3
    if not np.all(np.isfinite(k3)) or not np.isfinite(invariant_coefficient):
        raise ValueError("profile overflowed; check units and beta values")
    return {"beta_m": beta.tolist(), "k3_m4": k3.tolist(),
            "k3_beta_cubed_per_m": float(invariant_coefficient)}


@python_tool()
def invariant_variation(values, *, normalization_scale: float) -> dict:
    """Describe one particle's candidate invariant at a fixed observation plane.

    The positive scale is explicit so a near-zero invariant cannot silently
    produce an ill-conditioned relative error. Losses must be recorded elsewhere;
    a truncated surviving trajectory cannot establish long-term stability.
    """
    data = np.asarray(values, dtype=float)
    scale = float(normalization_scale)
    if data.ndim != 1 or data.size < 2 or not np.all(np.isfinite(data)):
        raise ValueError("values must contain at least two finite samples")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_scale must be positive and finite")
    return {"samples": int(data.size), "initial": float(data[0]),
            "mean": float(data.mean()), "normalization_scale": scale,
            "rms_variation_over_scale": float(np.std(data) / scale),
            "maximum_departure_from_initial_over_scale": float(np.max(np.abs(data - data[0])) / scale),
            "endpoint_change_over_scale": float((data[-1] - data[0]) / scale)}
