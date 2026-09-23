"""Reusable measurements of exact, mutable JuTrack lattices.

No lattice construction, transport approximation, optimizer, or design policy
lives here. Numerical maps and radiation are supplied by JuTrack. Successful
measurement is deliberately distinct from satisfying a scientific request.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .lattice_optimization import (
    jutrack_linear_metrics,
    jutrack_periodic_radiation_metrics,
)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _periods(value: int) -> int:
    count = int(value)
    if isinstance(value, bool) or count != value or count < 1:
        raise ValueError("periods must be a positive integer")
    return count


def _native_bridge(jt: Any) -> Any:
    # The current Python find_closed_orbit wrapper converts a heterogeneous
    # (orbit, matrix) tuple to one numpy array. Bypass only that conversion.
    bridge = getattr(jt, "_jl", None)
    if bridge is None and hasattr(jt, "get_julia"):
        bridge = jt.get_julia()
    return bridge


def _closed_orbit(jt: Any, lattice: Any, *, dp: float, energy_ev: float,
                  tolerance: float, max_iterations: int) -> np.ndarray:
    bridge = _native_bridge(jt)
    kwargs = {"energy": energy_ev, "tol": tolerance, "max_iter": max_iterations}
    if bridge is None:
        result = jt.find_closed_orbit(lattice, dp=dp, **kwargs)
    else:
        native_lattice = (jt.Lattice(lattice).julia_object
                          if isinstance(lattice, list)
                          else getattr(lattice, "julia_object", lattice))
        result = bridge.find_closed_orbit(native_lattice, dp, **kwargs)
    # JuTrack's public native contract is (six-coordinate orbit, 6x6 map).
    if len(result) != 2:
        raise TypeError("JuTrack find_closed_orbit must return (orbit, matrix)")
    orbit = np.asarray(result[0], dtype=float)
    if orbit.shape != (6,):
        raise TypeError(f"JuTrack closed orbit must have shape (6,); got {orbit.shape}")
    return orbit


def _native_map(jt: Any, lattice: Any, orbit: np.ndarray, *, dp: float,
                energy_ev: float) -> np.ndarray:
    bridge = _native_bridge(jt)
    if bridge is None:
        raw_orbit = orbit.copy()
    elif hasattr(jt, "_to_julia_float_vector"):
        raw_orbit = jt._to_julia_float_vector(orbit.tolist())
    else:
        raw_orbit = bridge.Vector(orbit)
    matrix = np.asarray(jt.findm66(lattice, dp=dp, order=0,
                                 E0=energy_ev, orb=raw_orbit), dtype=float)
    if matrix.shape != (6, 6):
        raise TypeError(f"JuTrack findm66 must return a 6x6 matrix; got {matrix.shape}")
    return matrix


def _tunes(matrix: np.ndarray, coupling_tolerance: float) -> tuple[list[float], str | None]:
    if not np.isfinite(matrix).all():
        return [], "native map contains nonfinite values"
    cross = max(float(np.max(np.abs(matrix[:2, 2:4]))),
                float(np.max(np.abs(matrix[2:4, :2]))))
    if cross > coupling_tolerance:
        return [], "transverse coupling exceeds this uncoupled measurement's tolerance"
    result = []
    for start, plane in ((0, "x"), (2, "y")):
        block = matrix[start:start + 2, start:start + 2]
        cosine = float(np.trace(block)) / 2.0
        determinant = float(np.linalg.det(block))
        if abs(determinant - 1.0) > 1e-5:
            return [], f"{plane} map is not a conservative symplectic 2x2 block"
        if abs(cosine) >= 1.0:
            return [], f"{plane} map is unstable or on an integer/half-integer boundary"
        phase = math.acos(cosine)
        if block[0, 1] < 0.0:
            phase = 2.0 * math.pi - phase
        result.append(phase / (2.0 * math.pi))
    return result, None


def _wrapped_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 0.5) % 1.0 - 0.5


def jutrack_chromaticity_metrics(
    jt: Any,
    lattice: Any,
    *,
    energy_gev: float,
    dp_step: float,
    periods: int = 1,
    orbit_tolerance: float = 1e-10,
    max_orbit_iterations: int = 30,
    coupling_tolerance: float = 1e-8,
    check_half_step: bool = True,
) -> dict[str, Any]:
    """Measure dQ/d(delta) around independently verified off-momentum orbits.

    ``lattice`` is the actual ring or one complete periodic block. ``periods``
    scales the *unwrapped derivative* for identical repeated blocks; it is not
    a substitute for evaluating the selected explicit ring. Use periods=1
    for an explicit ring. Energy is GeV and dp_step is relative momentum.

    This method is for conservative, uncoupled transverse motion with RF and
    radiation disabled by the caller. It rejects appreciably coupled maps;
    it does not mislabel eigenmodes as horizontal/vertical tunes. Each native
    closed orbit is checked by another native pass, then native findm66 is
    evaluated at that orbit, including sextupole feed-down. No origin-map or
    beta-times-gradient chromaticity surrogate is used.

    The native orbit solver uses 0.01 * orbit_tolerance internally. Its
    stopping residual is not the closure residual of the returned orbit;
    the tighter solve leaves margin for independent verification against the
    unchanged caller tolerance. Both tolerances are recorded in the result.

    With check_half_step=True, returns the half-step derivative plus both
    estimates and their difference. Step convergence remains the caller's
    responsibility; steps must resolve the local tune branch. ``computed``
    does not mean chromatic targets, nonlinear acceptance, or a design passed.
    Physical measurement failures return computed=False, never zero values;
    runtime/bridge errors propagate with their original diagnostics.
    """
    energy_ev = _positive(energy_gev, "energy_gev") * 1e9
    step = _positive(dp_step, "dp_step")
    tolerance = _positive(orbit_tolerance, "orbit_tolerance")
    solve_tolerance = _positive(0.01 * tolerance, "internal orbit solver tolerance")
    coupling = _positive(coupling_tolerance, "coupling_tolerance")
    count = _periods(periods)
    iterations = _periods(max_orbit_iterations)
    result: dict[str, Any] = {
        "computed": False,
        "method": "JuTrack closed orbit + native findm66; central tune difference",
        "evaluation_scope": "input_periodic_lattice",
        "periods": count,
        "energy_gev": energy_ev / 1e9,
        "dp_step": step,
        "orbit_tolerance": tolerance,
        "orbit_solver_tolerance": solve_tolerance,
        "max_orbit_iterations": iterations,
        "coupling_tolerance": coupling,
        "assumptions": "uncoupled transverse motion; RF and radiation off",
        "full_ring_validation_required": count != 1,
        "samples": [],
    }
    offsets = [0.0, -step, step]
    if check_half_step:
        offsets.extend([-step / 2.0, step / 2.0])
    by_dp: dict[float, np.ndarray] = {}
    for dp in offsets:
        orbit = _closed_orbit(jt, lattice, dp=dp, energy_ev=energy_ev,
                              tolerance=solve_tolerance, max_iterations=iterations)
        sample: dict[str, Any] = {"dp": dp, "closed_orbit": orbit.tolist()}
        result["samples"].append(sample)
        if not np.isfinite(orbit).all():
            result["reason"] = "native closed-orbit result contains nonfinite coordinates"
            return result
        beam = jt.Beam(orbit.reshape(1, 6).copy(), energy=energy_ev)
        jt.linepass(lattice, beam)
        output = np.asarray(beam.r, dtype=float)
        if output.shape != (1, 6):
            raise TypeError("JuTrack orbit verification must return one six-coordinate particle")
        lost = bool(np.any(np.asarray(beam.lost_flag)))
        residual = output[0, :4] - orbit[:4]
        residual_norm = float(np.max(np.abs(residual)))
        momentum_residual = max(abs(float(orbit[5]) - dp),
                                abs(float(output[0, 5]) - dp))
        sample.update({"transverse_residual": residual.tolist(),
                       "transverse_residual_max_abs": residual_norm,
                       "momentum_residual_abs": momentum_residual, "particle_lost": lost})
        if (lost or not np.isfinite(output).all() or residual_norm > tolerance
                or momentum_residual > tolerance):
            result["reason"] = "closed orbit failed independent native tracking verification"
            return result
        matrix = _native_map(jt, lattice, orbit, dp=dp, energy_ev=energy_ev)
        tune, error = _tunes(matrix, coupling)
        if error:
            result["reason"] = error
            return result
        sample["fractional_tunes"] = tune
        sample["transverse_map"] = matrix[:4, :4].tolist()
        by_dp[dp] = np.asarray(tune)
    estimates = []
    for width in ([step, step / 2.0] if check_half_step else [step]):
        minus = _wrapped_difference(by_dp[-width], by_dp[0.0])
        plus = _wrapped_difference(by_dp[width], by_dp[0.0])
        per_period = (plus - minus) / (2.0 * width)
        estimates.append({"dp_step": width, "per_period": per_period.tolist(),
                          "identical_ring": (per_period * count).tolist()})
    selected = estimates[-1]
    result.update({
        "computed": True,
        "chromaticity_x": selected["identical_ring"][0],
        "chromaticity_y": selected["identical_ring"][1],
        "per_period_chromaticities": selected["per_period"],
        "derivative_dp_step": selected["dp_step"],
        "step_estimates": estimates,
        "step_difference": ((np.asarray(estimates[-1]["identical_ring"])
                             - np.asarray(estimates[0]["identical_ring"])).tolist()
                            if check_half_step else None),
    })
    return result


def jutrack_design_metrics(
    jt: Any,
    lattice: Any,
    *,
    energy_gev: float,
    dp_step: float,
    periods: int = 1,
    include_profile: bool = True,
    orbit_tolerance: float = 1e-10,
    max_orbit_iterations: int = 30,
    coupling_tolerance: float = 1e-8,
    check_half_step: bool = True,
) -> dict[str, Any]:
    """One unchanged candidate -> native linear, radiation and chromatic data.

    Intended for ideal centred periodic lattices. Linear/radiation entries
    reuse the existing public JuTrack helpers; chromaticity uses verified
    off-momentum closed orbits. Geometry is measured from actual elements,
    never intended dimensions. Returned acceptance is *not assessed*: callers
    apply their own optics/engineering/request limits and periodic-ring tests.
    Use this same function in the objective and final report to avoid mixing
    approximate search observables with different final measurements.
    A native Julia DomainError raised during radiation evaluation is returned
    as an unmeasured trial with its original diagnostic, not invented damping
    or emittance values. Chromaticity is skipped for that failed joint trial.
    Other native/runtime exceptions propagate; unavailable radiation is not
    evidence for a particular physical cause such as negative damping.
    """
    count = _periods(periods)
    _positive(energy_gev, "energy_gev")
    _positive(dp_step, "dp_step")
    length = sum(float(element.len) for element in lattice)
    angle = sum(float(getattr(element, "angle", 0.0)) for element in lattice)
    linear = jutrack_linear_metrics(jt, lattice, include_profile=include_profile,
                                    energy_gev=energy_gev)
    result: dict[str, Any] = {
        "computed": False, "acceptance": "not_assessed",
        "geometry": {"period_length_m": length, "period_bend_angle_rad": angle,
                     "period_element_count": len(lattice), "periods": count,
                     "ring_length_m": count * length, "ring_bend_angle_rad": count * angle},
        "linear": linear,
        "radiation": {"computed": False, "reason": "linear optics not stable"},
        "chromaticity": {"computed": False, "reason": "linear optics not stable"},
        "full_ring_validation_required": count != 1,
    }
    if not linear.get("stable", False):
        return result
    try:
        result["radiation"] = jutrack_periodic_radiation_metrics(
            jt, lattice, energy_gev=energy_gev, periods=count)
    except Exception as exc:
        # Native radiation routines may throw before returning their integrals
        # (for example sqrt of an invalid equilibrium). Do not infer a cause
        # or fabricate I2/I4/Jz values that JuTrack did not return. This handler
        # deliberately excludes Python errors and Julia MethodError/runtime
        # failures, even when they occur in the same native call.
        if not (type(exc).__name__ == "JuliaError"
                and type(exc).__module__.split(".")[0] == "juliacall"
                and "DomainError" in str(exc)):
            raise
        result["radiation"] = {
            "computed": False, "failure_kind": "native_radiation_domain_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "physical_cause": "not established; radiation values unavailable",
        }
        result["chromaticity"] = {
            "computed": False,
            "reason": "not evaluated after native radiation domain error",
        }
        return result
    result["chromaticity"] = jutrack_chromaticity_metrics(
        jt, lattice, energy_gev=energy_gev, dp_step=dp_step, periods=count,
        orbit_tolerance=orbit_tolerance, max_orbit_iterations=max_orbit_iterations,
        coupling_tolerance=coupling_tolerance, check_half_step=check_half_step)
    result["computed"] = bool(result["radiation"].get("computed")
                              and result["chromaticity"].get("computed"))
    return result
