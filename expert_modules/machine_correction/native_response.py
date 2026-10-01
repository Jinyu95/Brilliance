"""Coupled orbit response from native JuTrack tracking, without importing Julia.

This module consumes an already configured ``pyJuTrack`` module and a complete
ring. It implements boundary-condition algebra, not element transport maps.
Coordinates are [x, px, y, py, z, delta]; indices identify element EXITS and are
zero based. All perturbations use private corrector copies. No lattice settings
are changed, no environment is configured, and no hardware commands are issued.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np


class NativeResponseError(RuntimeError):
    """Native tracking did not establish a usable response measurement."""


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _steps(value: float | Sequence[float]) -> np.ndarray:
    step = np.asarray(value, dtype=float)
    if step.ndim == 0:
        step = np.full(6, float(step))
    if step.shape != (6,) or not np.isfinite(step).all() or np.any(step <= 0):
        raise ValueError("coordinate_step must be a positive scalar or six positive steps")
    return step


def _indices(values: Sequence[int], size: int, name: str) -> list[int]:
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or int(value) != value:
            raise ValueError(f"{name} must contain zero-based integer indices")
        index = int(value)
        if not 0 <= index < size:
            raise ValueError(f"{name} index {index} is outside the complete ring")
        result.append(index)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _prepare(jt: Any, lattice: Any, bpms: Sequence[int], correctors: Sequence[int],
             planes: Sequence[str]) -> tuple[Any, list[int], list[int], list[str]]:
    lat = lattice if hasattr(lattice, "julia_object") else jt.Lattice(lattice)
    if len(lat) == 0:
        raise ValueError("A complete, nonempty ring is required")
    for index, element in enumerate(lat):
        kind = str(getattr(element, "eletype", type(element).__name__)).upper()
        excluded = ("RF", "CAV", "_SC", "SPACECHARGE", "WAKE", "BEAMBEAM",
                    "LONGITUDINAL", "RADIATION")
        if any(token in kind for token in excluded) or hasattr(element, "volt"):
            raise ValueError(f"Unsupported RF/collective/time-dependent element {index}: {kind}")
        if float(getattr(element, "rad", 0)) != 0:
            raise ValueError(f"Radiation is enabled at element {index}; it will not be disabled")
    bpm = _indices(bpms, len(lat), "bpm_indices")
    cor = _indices(correctors, len(lat), "corrector_indices")
    plane = [str(p).lower() for p in planes]
    if len(cor) != len(plane) or any(p not in ("x", "y") for p in plane):
        raise ValueError("corrector_planes must contain one 'x' or 'y' per corrector")
    if len(set(bpm)) != len(bpm) or len(set(zip(cor, plane))) != len(cor):
        raise ValueError("Duplicate BPM indices or corrector (index, plane) channels")
    for index in cor:
        element = lat[index]
        if not all(hasattr(element, key) for key in ("xkick", "ykick")):
            raise ValueError(f"Element {index} is not a native two-plane CORRECTOR")
    return lat, bpm, cor, plane


def _track(jt: Any, lattice: Any, particles: np.ndarray, energy: float,
           refpts: Sequence[int] = ()) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    beam = jt.Beam(np.asarray(particles, dtype=float).copy(), energy=energy)
    refs = sorted(set(refpts))
    saved = jt.linepass(lattice, beam, refpts=refs) if refs else jt.linepass(lattice, beam)
    final = np.asarray(beam.r, dtype=float).copy()
    lost = np.asarray(beam.lost_flag, dtype=bool).ravel()
    if final.shape != particles.shape or lost.size != len(particles):
        raise NativeResponseError("JuTrack returned inconsistent particle or loss-flag shapes")
    if lost.any() or not np.isfinite(final).all():
        raise NativeResponseError("Native orbit/map probe was lost or became nonfinite")
    records = {}
    if refs:
        if len(saved) != len(refs):
            raise NativeResponseError("JuTrack did not return every requested exit reference point")
        for index, coordinates in zip(refs, saved):
            sample = np.asarray(coordinates, dtype=float).copy()
            if sample.shape != particles.shape or not np.isfinite(sample).all():
                raise NativeResponseError("Invalid native reference-point coordinates")
            records[index] = sample
    return final, records


def _closed_orbit(jt: Any, lattice: Any, dp: float, energy: float,
                  tolerance: float, max_iterations: int) -> tuple[np.ndarray, dict[str, Any]]:
    bridge = getattr(jt, "_jl", None)
    kwargs = dict(energy=energy, tol=min(tolerance * .01, 1e-12), max_iter=max_iterations)
    # Bypass pyJuTrack's heterogeneous tuple-to-array conversion. Deliberately
    # discard the native solver's matrix: its delta column is not usable here.
    if bridge is not None:
        solved = bridge.find_closed_orbit(lattice.julia_object, float(dp), **kwargs)
    else:
        solved = jt.find_closed_orbit(lattice, dp=float(dp), **kwargs)
    orbit = np.asarray(solved[0], dtype=float).copy()
    if orbit.shape != (6,) or not np.isfinite(orbit).all():
        raise NativeResponseError("Native closed orbit is not a finite six-coordinate vector")
    if abs(orbit[5] - dp) > tolerance:
        raise NativeResponseError("Native closed orbit did not preserve the prescribed delta")
    orbit[4] = 0.0  # longitudinal origin is arbitrary in the supported static ring
    final, _ = _track(jt, lattice, orbit[None, :], energy)
    residual = final[0] - orbit
    error = float(np.max(np.abs(residual[[0, 1, 2, 3, 5]])))
    if error > tolerance:
        raise NativeResponseError(f"Closed-orbit closure {error:.3g} exceeds {tolerance:.3g}")
    return orbit, dict(verified=True, maximum_transverse_delta_residual=error,
                      residual=residual.tolist(), path_increment_m=float(residual[4]),
                      orbit=orbit.tolist(), tolerance=tolerance)


def _maps(jt: Any, lattice: Any, orbit: np.ndarray, energy: float, steps: np.ndarray,
          refpts: Sequence[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    # One native full-ring pass, no segment resets or analytical transport model.
    particles = np.tile(orbit, (13, 1))
    for coordinate in range(6):
        particles[1 + 2 * coordinate, coordinate] += steps[coordinate]
        particles[2 + 2 * coordinate, coordinate] -= steps[coordinate]
    _, saved = _track(jt, lattice, particles, energy, sorted(set(refpts) | {len(lattice) - 1}))
    matrices = {-1: np.eye(6)}
    baselines = {-1: orbit.copy()}
    for index, sample in saved.items():
        matrices[index] = ((sample[1::2] - sample[2::2]) / (2 * steps[:, None])).T
        if not np.isfinite(matrices[index]).all():
            raise NativeResponseError("Native finite-difference map contains nonfinite coefficients")
        baselines[index] = sample[0]
    return matrices, baselines


def _solve(matrix: np.ndarray, rhs: np.ndarray, limit: float) -> tuple[np.ndarray, float]:
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > limit:
        raise NativeResponseError(f"Response boundary is singular/ill-conditioned: {condition:.3g}")
    return np.linalg.solve(matrix, rhs), condition


def _boundary(matrix: np.ndarray, mode: str) -> np.ndarray:
    transverse = np.eye(4) - matrix[:4, :4]
    if mode == "fixed_momentum":
        return transverse
    if mode != "fixed_path_length":
        raise ValueError("mode must be 'fixed_momentum' or 'fixed_path_length'")
    boundary = np.empty((5, 5))
    boundary[:4, :4] = transverse
    boundary[:4, 4] = -matrix[:4, 5]
    boundary[4, :4] = matrix[4, :4]
    boundary[4, 4] = matrix[4, 5]
    return boundary


def _copy_element(jt: Any, element: Any) -> Any:
    copier = getattr(jt, "deepcopy_element", deepcopy)
    return copier(element)


def _kick_derivative(jt: Any, element: Any, entrance: np.ndarray, plane: str,
                     step: float, energy: float) -> np.ndarray:
    private = _copy_element(jt, element)
    local = jt.Lattice([private])
    attribute = plane + "kick"
    original = float(getattr(private, attribute))
    outputs = []
    try:
        for sign in (1, -1):
            setattr(private, attribute, original + sign * step)
            final, _ = _track(jt, local, entrance[None, :], energy)
            outputs.append(final[0])
    finally:
        setattr(private, attribute, original)
    return (outputs[0] - outputs[1]) / (2 * step)


def jutrack_response(
    jt: Any, lattice: Any, *, bpm_indices: Sequence[int],
    corrector_indices: Sequence[int], corrector_planes: Sequence[str], energy_ev: float,
    kick_step: float = 1e-6, coordinate_step: float | Sequence[float] = 1e-6,
    orbit_tolerance: float = 1e-10, mode: str = "fixed_momentum", dp: float = 0.0,
    max_iterations: int = 50, condition_limit: float = 1e12,
) -> dict[str, Any]:
    """Measure a complete coupled linear ORM about a verified native closed orbit.

    Output rows are [all x BPMs, all y BPMs]; columns follow the supplied
    corrector channels. ``cross_response_m_per_rad`` uses an orthogonal kick at
    the SAME corrector, for calibration/leakage fits. It is not a row swap.

    ``fixed_path_length`` permits delta to vary to preserve the baseline
    one-turn z increment. This is a static five-variable response constraint,
    not a cavity-inclusive synchronous orbit or a six-dimensional RF model.
    ``dispersion_m`` is dx/d(delta), dy/d(delta) at fixed momentum, in BOTH modes.
    """
    energy = _positive(energy_ev, "energy_ev")
    kick = _positive(kick_step, "kick_step")
    tolerance = _positive(orbit_tolerance, "orbit_tolerance")
    limit = _positive(condition_limit, "condition_limit")
    steps = _steps(coordinate_step)
    if not np.isfinite(dp) or dp <= -1:
        raise ValueError("dp must be finite and greater than -1")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    _boundary(np.eye(6), mode)  # validate mode before tracking
    lat, bpms, cors, planes = _prepare(jt, lattice, bpm_indices, corrector_indices, corrector_planes)
    orbit, closure = _closed_orbit(jt, lat, dp, energy, tolerance, int(max_iterations))
    refs = sorted(set(bpms + cors + [i - 1 for i in cors if i > 0]))
    maps, baselines = _maps(jt, lat, orbit, energy, steps, refs)
    turn = maps[len(lat) - 1]
    if not np.allclose(turn[5], np.eye(6)[5], atol=1e-7, rtol=0):
        raise NativeResponseError("Native map does not conserve delta")
    if not np.allclose(turn[:, 4], np.eye(6)[:, 4], atol=1e-7, rtol=0):
        raise NativeResponseError("Native map is not invariant under longitudinal translation")
    eigenvalues = np.linalg.eigvals(turn[:4, :4])
    if np.max(np.abs(np.abs(eigenvalues) - 1)) > 1e-5:
        raise NativeResponseError("Transverse one-turn map is unstable or nonconservative")
    boundary = _boundary(turn, mode)
    _, condition = _solve(boundary, np.zeros(len(boundary)), limit)
    eta, dispersion_condition = _solve(np.eye(4) - turn[:4, :4], turn[:4, 5], limit)
    eta_start = np.r_[eta, 0.0, 1.0]
    dispersion = np.array([(maps[i] @ eta_start)[[0, 2]] for i in bpms]).T.ravel()
    responses = [np.empty((2 * len(bpms), len(cors))) for _ in range(2)]
    momentum_response = [np.empty(len(cors)) for _ in range(2)]
    source_condition = 0.0
    for column, (index, plane) in enumerate(zip(cors, planes)):
        for channel, direction in enumerate((plane, "y" if plane == "x" else "x")):
            kick_exit = _kick_derivative(jt, lat[index], baselines[index - 1], direction, kick, energy)
            source, local_condition = _solve(maps[index], kick_exit, limit)
            source_condition = max(source_condition, local_condition)
            end_change = turn @ source
            rhs = end_change[:4] if mode == "fixed_momentum" else np.r_[end_change[:4], -end_change[4]]
            solution, _ = _solve(boundary, rhs, limit)
            start = np.r_[solution[:4], 0.0, solution[4] if len(solution) == 5 else 0.0]
            momentum_response[channel][column] = start[5]
            readings = [(maps[bpm] @ (start + (source if bpm >= index else 0)))[[0, 2]] for bpm in bpms]
            responses[channel][:, column] = np.asarray(readings).T.ravel()
    return dict(
        computed=True, response_m_per_rad=responses[0].tolist(),
        cross_response_m_per_rad=responses[1].tolist(), dispersion_m=dispersion.tolist(),
        delta_per_rad=momentum_response[0].tolist(), cross_delta_per_rad=momentum_response[1].tolist(),
        boundary_condition_number=condition, dispersion_condition_number=dispersion_condition,
        maximum_cumulative_map_condition_number=source_condition, closure=closure,
        bpm_devices=[dict(index=i, name=str(getattr(lat[i], "name", "")), identity=f"exit:{i}") for i in bpms],
        corrector_devices=[dict(index=i, plane=p, name=str(getattr(lat[i], "name", "")), identity=f"{i}:{p}") for i, p in zip(cors, planes)],
        one_turn_map=turn.tolist(), metadata=dict(
            mode=mode, energy_ev=energy, dp=float(dp), element_count=len(lat),
            row_order="x BPMs followed by y BPMs", index_convention="zero-based element exits",
            coordinate_order=["x", "px", "y", "py", "z", "delta"],
            kick_step=kick, coordinate_step=steps.tolist(), map_probe_particles=13,
            method="native cumulative central-coordinate tracking and local corrector kick derivatives",
            dispersion_method="periodic solution of native central-coordinate maps",
            scope="complete static conservative ring; no RF, radiation, or collective effects",
            parameter_units="native canonical kick; m/rad at delta=0; no mm conversion",
            integration_convergence_checked=False, input_lattice_modified=False,
        ),
    )


def _path_orbit(jt: Any, lattice: Any, initial: np.ndarray, path: float, energy: float,
                steps: np.ndarray, tolerance: float, iterations: int, limit: float) -> np.ndarray:
    orbit = initial.copy()
    for _ in range(iterations):
        maps, baselines = _maps(jt, lattice, orbit, energy, steps, [])
        end = baselines[len(lattice) - 1]
        error = np.r_[end[:4] - orbit[:4], end[4] - orbit[4] - path]
        if abs(end[5] - orbit[5]) > tolerance:
            raise NativeResponseError("Fixed-path orbit does not conserve delta")
        if np.max(np.abs(error)) <= tolerance:
            return orbit
        # The fifth boundary row has the opposite sign to the transverse rows.
        change, _ = _solve(_boundary(maps[len(lattice) - 1], "fixed_path_length"),
                           np.r_[error[:4], -error[4]], limit)
        accepted = False
        for factor in (1.0, .5, .25, .125, .0625):
            trial = orbit.copy()
            trial[[0, 1, 2, 3, 5]] += factor * change
            final, _ = _track(jt, lattice, trial[None, :], energy)
            residual = np.r_[final[0, :4] - trial[:4], final[0, 4] - trial[4] - path]
            if np.max(np.abs(residual)) < np.max(np.abs(error)):
                orbit, accepted = trial, True
                break
        if not accepted:
            raise NativeResponseError("Native fixed-path orbit line search failed")
    raise NativeResponseError("Native fixed-path orbit did not converge")


def validate_jutrack_response(
    jt: Any, lattice: Any, *, columns: Sequence[int] | None = None,
    response: dict[str, Any] | None = None, dispersion_step: float = 1e-5,
    **options: Any,
) -> dict[str, Any]:
    """Slow independent +/-kick closed-orbit checks of selected ORM columns.

    Recomputes closed orbits on privately modified rings, not the Green-function
    formula. Also checks dispersion with native closed orbits at +/-delta.
    Passing an existing ``response`` avoids repeating the fast measurement;
    metadata checks settings and device labels, NOT magnet fields. The caller
    must bind the supplied response to the same saved lattice snapshot/hash;
    equal element counts and labels alone do not establish provenance.
    """
    if response is None:
        response = jutrack_response(jt, lattice, **options)
    meta = response["metadata"]
    for key in ("energy_ev", "dp", "mode"):
        if key in options and options[key] != meta[key]:
            raise ValueError(f"Validation setting {key} differs from the supplied response")
    bpms = [d["index"] for d in response["bpm_devices"]]
    cors = [d["index"] for d in response["corrector_devices"]]
    planes = [d["plane"] for d in response["corrector_devices"]]
    for key, expected in (("bpm_indices", bpms), ("corrector_indices", cors), ("corrector_planes", planes)):
        if key in options and list(options[key]) != expected:
            raise ValueError(f"Validation setting {key} differs from the supplied response")
    lat, _, _, _ = _prepare(jt, lattice, bpms, cors, planes)
    if len(lat) != meta["element_count"]:
        raise ValueError("Validation lattice length differs from the supplied response")
    for device in response["bpm_devices"] + response["corrector_devices"]:
        if str(getattr(lat[device["index"]], "name", "")) != device["name"]:
            raise ValueError("Validation device name differs from the supplied response")
    selected = list(range(len(cors))) if columns is None else _indices(columns, len(cors), "columns")
    if len(set(selected)) != len(selected):
        raise ValueError("Validation columns must be unique")
    energy, dp, mode = meta["energy_ev"], meta["dp"], meta["mode"]
    tolerance = _positive(options.get("orbit_tolerance", response["closure"]["tolerance"]), "orbit_tolerance")
    kick = _positive(options.get("kick_step", meta["kick_step"]), "kick_step")
    dstep = _positive(dispersion_step, "dispersion_step")
    steps = _steps(meta["coordinate_step"])
    iterations = int(options.get("max_iterations", 50))
    limit = _positive(options.get("condition_limit", 1e12), "condition_limit")
    baseline, closure = _closed_orbit(jt, lat, dp, energy, tolerance, iterations)
    measured = np.empty((2 * len(bpms), len(selected)))
    for output_column, column in enumerate(selected):
        private = _copy_element(jt, lat[cors[column]])
        elements = list(lat)
        elements[cors[column]] = private
        perturbed = jt.Lattice(elements)
        attribute = planes[column] + "kick"
        original = float(getattr(private, attribute))
        readings = []
        try:
            for sign in (1, -1):
                setattr(private, attribute, original + sign * kick)
                if mode == "fixed_momentum":
                    orbit, _ = _closed_orbit(jt, perturbed, dp, energy, tolerance, iterations)
                else:
                    orbit = _path_orbit(jt, perturbed, baseline, closure["path_increment_m"],
                                        energy, steps, tolerance * .01, iterations, limit)
                _, saved = _track(jt, perturbed, orbit[None, :], energy, bpms)
                readings.append(np.array([saved[i][0, [0, 2]] for i in bpms]).T.ravel())
        finally:
            setattr(private, attribute, original)
        measured[:, output_column] = (readings[0] - readings[1]) / (2 * kick)
    dispersion_readings = []
    for sign in (1, -1):
        orbit, _ = _closed_orbit(jt, lat, dp + sign * dstep, energy, tolerance, iterations)
        _, saved = _track(jt, lat, orbit[None, :], energy, bpms)
        dispersion_readings.append(np.array([saved[i][0, [0, 2]] for i in bpms]).T.ravel())
    dispersion = (dispersion_readings[0] - dispersion_readings[1]) / (2 * dstep)
    expected = np.asarray(response["response_m_per_rad"])[:, selected]
    difference = measured - expected
    return dict(
        columns=selected, closed_orbit_response_m_per_rad=measured.tolist(),
        maximum_absolute_difference_m_per_rad=float(np.max(np.abs(difference))),
        relative_frobenius_difference=float(np.linalg.norm(difference) / max(np.linalg.norm(measured), 1e-30)),
        finite_difference_dispersion_m=dispersion.tolist(),
        maximum_dispersion_difference_m=float(np.max(np.abs(dispersion - np.asarray(response["dispersion_m"])))),
        kick_step=kick, dispersion_step=dstep, closure=closure,
        method="independent central-kick closed orbits and central-delta closed orbits",
        input_lattice_modified=False,
    )
