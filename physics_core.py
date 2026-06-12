"""Reusable physics evaluation and search helpers for lattice scripts."""

from __future__ import annotations

import json
import math
import os
import re
from contextlib import contextmanager
from typing import Any, Callable, Sequence

RESULT_JSON_BEGIN = "--- LATTICE RESULT JSON ---"
RESULT_JSON_END = "--- END LATTICE RESULT JSON ---"


def _to_float(value: Any) -> float | None:
    """Best-effort conversion for pyJuTrack and numpy scalar-like values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    try:
        return float(value[0])
    except Exception:
        return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    numeric = _to_float(value)
    if numeric is not None:
        return numeric
    return str(value)


@contextmanager
def suppressed_output():
    """Silence noisy stdout/stderr from pyJuTrack calls."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


def check_cell_stability(jt: Any, cell: Sequence[Any]) -> dict[str, Any]:
    """Check one-cell linear stability via the trace criterion."""
    try:
        lattice = jt.Lattice(list(cell))
        with suppressed_output():
            m66 = jt.findm66(lattice, 0.0, 0)
        trace_x = abs(_to_float(m66[0, 0] + m66[1, 1]) or math.inf)
        trace_y = abs(_to_float(m66[2, 2] + m66[3, 3]) or math.inf)
        stable = (
            math.isfinite(trace_x)
            and math.isfinite(trace_y)
            and trace_x < 2.0
            and trace_y < 2.0
        )
        return {
            "stable": stable,
            "trace_x": trace_x,
            "trace_y": trace_y,
        }
    except Exception as exc:
        return {
            "stable": False,
            "trace_x": None,
            "trace_y": None,
            "error": str(exc),
        }


def evaluate_cell(
    jt: Any,
    cell: Sequence[Any],
    *,
    energy_gev: float | None = None,
) -> dict[str, Any]:
    """Evaluate properties of a single accelerator cell as a transfer line.

    Computes the 6×6 transfer matrix M = jt.findm66(lat, 0.0, 0) and extracts:
      T56     = M[4,5]  path-length / momentum coupling  (isochronous cell → 0)
      D_exit  = M[0,5]  dispersion at cell exit          (achromatic cell → 0)
      Dp_exit = M[1,5]  dispersion slope at exit         (achromatic cell → 0)
      trace_x = M[0,0]+M[1,1]  stability criterion  (stable → |·| < 2)
      trace_y = M[2,2]+M[3,3]
      phase_advance_{x,y}_deg  betatron phase advance across the cell

    Coordinate convention (pyJuTrack / AT-style):
      [0] x    [1] px    [2] y    [3] py    [4] path/t    [5] δ=Δp/p
    """
    result: dict[str, Any] = {"lattice_mode": "cell", "n_cells": 1}
    if energy_gev is not None:
        fe = _to_float(energy_gev)
        if fe is not None and math.isfinite(fe):
            result["energy_gev"] = fe
    try:
        lat = jt.Lattice(list(cell))
        with suppressed_output():
            M    = jt.findm66(lat, 0.0, 0)
            spos = jt.spos(lat)
        cell_length = _to_float(spos[-1]) if len(spos) else None
        result["cell_length"]  = cell_length
        result["circumference"] = cell_length  # alias for compatibility

        T56     = _to_float(M[4, 5])
        D_exit  = _to_float(M[0, 5])
        Dp_exit = _to_float(M[1, 5])
        trace_x = _to_float(M[0, 0] + M[1, 1])
        trace_y = _to_float(M[2, 2] + M[3, 3])

        result.update({
            "T56":     T56,
            "D_exit":  D_exit,
            "Dp_exit": Dp_exit,
            "trace_x": trace_x,
            "trace_y": trace_y,
        })

        if trace_x is not None and abs(trace_x) < 2.0:
            result["phase_advance_x_deg"] = math.degrees(
                math.acos(max(-1.0, min(1.0, trace_x / 2.0)))
            )
        if trace_y is not None and abs(trace_y) < 2.0:
            result["phase_advance_y_deg"] = math.degrees(
                math.acos(max(-1.0, min(1.0, trace_y / 2.0)))
            )

        is_stable      = bool(trace_x is not None and trace_y is not None
                               and abs(trace_x) < 2.0 and abs(trace_y) < 2.0)
        is_achromatic  = bool(D_exit is not None and Dp_exit is not None
                               and abs(D_exit) < 1e-4 and abs(Dp_exit) < 1e-4)
        is_isochronous = bool(T56 is not None and abs(T56) < 1e-5)

        result.update({
            "is_linearly_stable": is_stable,
            "is_achromatic":      is_achromatic,
            "is_isochronous":     is_isochronous,
            "stable":             is_stable,   # alias for score_design_result
        })
    except Exception as exc:
        result.update({
            "error":              str(exc),
            "stable":             False,
            "is_linearly_stable": False,
            "is_achromatic":      False,
            "is_isochronous":     False,
        })
    return result


def _extract_ringpara_value(ringpara: Any, key: str) -> float | None:
    if isinstance(ringpara, dict):
        return _to_float(ringpara.get(key))
    if hasattr(ringpara, key):
        return _to_float(getattr(ringpara, key))
    return None


def evaluate_lattice(
    jt: Any,
    cell: Sequence[Any],
    n_cells: int,
    *,
    lattice_mode: str = "auto",
    periodic_twiss: bool = False,
    energy_gev: float | None = None,
    energy_eV: float | None = None,
    energy_ev: float | None = None,
    energy: float | None = None,
    **extra_kwargs: Any,
) -> dict[str, Any]:
    """Compute deterministic lattice metrics for ring or linac workflows.

    Parameters
    ----------
    lattice_mode:
        "ring", "linac", or "auto". In auto mode, n_cells > 1 implies ring,
        otherwise linac.
    periodic_twiss:
        If True and lattice_mode is ring, evaluate periodic twiss at the cell
        entrance. For linac mode this is ignored.
    """
    resolved_energy_gev = _to_float(energy_gev)
    if resolved_energy_gev is None:
        energy_electron_volts = _to_float(energy_eV)
        if energy_electron_volts is None:
            energy_electron_volts = _to_float(energy_ev)
        if energy_electron_volts is not None and math.isfinite(energy_electron_volts):
            resolved_energy_gev = energy_electron_volts / 1e9
    if resolved_energy_gev is None:
        generic_energy = _to_float(energy)
        if generic_energy is not None and math.isfinite(generic_energy):
            resolved_energy_gev = generic_energy / 1e9 if abs(generic_energy) > 1e6 else generic_energy

    mode = (lattice_mode or "auto").strip().lower()
    if mode not in {"auto", "ring", "linac", "cell"}:
        mode = "auto"
    if mode == "auto":
        mode = "ring" if int(n_cells) > 1 else "linac"

    # Single-cell / transfer-line evaluation — full cell analysis
    if mode == "cell":
        return evaluate_cell(jt, cell, energy_gev=resolved_energy_gev)

    if mode == "ring":
        result = check_cell_stability(jt, cell)
    else:
        result = {
            "stable": None,
            "trace_x": None,
            "trace_y": None,
            "stability_checked": False,
        }

    result["lattice_mode"] = mode
    result["n_cells"] = n_cells
    if resolved_energy_gev is not None and math.isfinite(resolved_energy_gev):
        result["energy_gev"] = resolved_energy_gev
    if extra_kwargs:
        result["ignored_kwargs"] = sorted(str(key) for key in extra_kwargs)

    try:
        cell_lattice = jt.Lattice(list(cell))

        if mode == "ring":
            ring = jt.Lattice(list(cell) * n_cells)
            with suppressed_output():
                tunes = jt.gettune(ring)
                chrom = jt.getchrom(ring)
                _rp_kwargs = (
                    {"energy": resolved_energy_gev * 1e9}
                    if resolved_energy_gev is not None and math.isfinite(resolved_energy_gev)
                    else {}
                )
                try:
                    ringpara = jt.ringpara(ring, **_rp_kwargs)
                except TypeError:
                    # Older pyJuTrack builds do not accept the energy keyword.
                    ringpara = jt.ringpara(ring)
                spos = jt.spos(ring)

            dx_entrance = None
            dpx_entrance = None
            if periodic_twiss:
                try:
                    with suppressed_output():
                        twiss = jt.twissring(cell_lattice)
                    if len(twiss):
                        dx_entrance = _to_float(twiss[0].dx)
                        dpx_entrance = _to_float(twiss[0].dpx)
                except Exception:
                    dx_entrance = None
                    dpx_entrance = None

            result.update(
                {
                    "tune_nux": _to_float(tunes[0]) if len(tunes) > 0 else None,
                    "tune_nuy": _to_float(tunes[1]) if len(tunes) > 1 else None,
                    "chromaticity_xix": _to_float(chrom[0]) if len(chrom) > 0 else None,
                    "chromaticity_xiy": _to_float(chrom[1]) if len(chrom) > 1 else None,
                    "emittance": _extract_ringpara_value(ringpara, "emittx"),
                    "circumference": _to_float(spos[-1]) if len(spos) else None,
                    "dx_entrance": dx_entrance,
                    "dpx_entrance": dpx_entrance,
                    "periodic_twiss": bool(periodic_twiss),
                    # Radiation integrals and derived quantities
                    "alphac": _extract_ringpara_value(ringpara, "alphac"),
                    "U0_eV": _extract_ringpara_value(ringpara, "U0"),
                    "sigma_E": _extract_ringpara_value(ringpara, "sigma_E"),
                    "damping_time_x": _extract_ringpara_value(ringpara, "dampingtime_x"),
                    "damping_time_y": _extract_ringpara_value(ringpara, "dampingtime_y"),
                    "damping_time_E": _extract_ringpara_value(ringpara, "dampingtime_E"),
                    "Jx": _extract_ringpara_value(ringpara, "Jx"),
                    "Jy": _extract_ringpara_value(ringpara, "Jy"),
                    "Je": _extract_ringpara_value(ringpara, "Je"),
                    "I1": _extract_ringpara_value(ringpara, "I1"),
                    "I2": _extract_ringpara_value(ringpara, "I2"),
                    "I3": _extract_ringpara_value(ringpara, "I3"),
                    "I4": _extract_ringpara_value(ringpara, "I4"),
                    "I5": _extract_ringpara_value(ringpara, "I5"),
                }
            )
        else:
            with suppressed_output():
                spos = jt.spos(cell_lattice)
            result.update(
                {
                    "tune_nux": None,
                    "tune_nuy": None,
                    "chromaticity_xix": None,
                    "chromaticity_xiy": None,
                    "emittance": None,
                    "circumference": _to_float(spos[-1]) if len(spos) else None,
                    "dx_entrance": None,
                    "dpx_entrance": None,
                    "periodic_twiss": False,
                }
            )
    except Exception as exc:
        result.setdefault("error", str(exc))

    return result


def _is_stable(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("stable"))
    return bool(value)


def _preserve_sign(reference: float, value: float) -> float:
    if reference > 0:
        return abs(value)
    if reference < 0:
        return -abs(value)
    return value


def search_stable_configuration(
    create_cell: Callable[[Sequence[float]], Sequence[Any]],
    initial_k1: Sequence[float],
    check_stability: Callable[[Sequence[Any]], Any],
    *,
    stages: Sequence[tuple[str, float, int]] | None = None,
    progress_every: int = 10000,
    rng=None,
) -> dict[str, Any]:
    """Run a shared staged search around a template operating point."""
    if rng is None:
        try:
            import numpy as np  # local import to avoid hard dependency for app code

            rng = np.random.default_rng()
        except Exception as exc:
            raise RuntimeError("numpy is required for stability search") from exc

    initial_k1 = list(initial_k1)
    if stages is None:
        stages = [
            ("template", 0.0, 1),
            ("perturb_30", 0.30, 20000),
            ("perturb_50", 0.50, 20000),
            ("perturb_100", 1.00, 10000),
        ]

    total_trials = 0
    initial_cell = create_cell(initial_k1)
    initial_report = check_stability(initial_cell)
    total_trials += 1
    if _is_stable(initial_report):
        return {
            "found": True,
            "best_k1": initial_k1,
            "stage": "template",
            "trials": total_trials,
            "report": initial_report,
        }

    for stage_name, perturbation, trials in stages:
        if stage_name == "template":
            continue

        for trial in range(trials):
            candidate = [
                _preserve_sign(base_value, base_value * (1.0 + rng.uniform(-perturbation, perturbation)))
                for base_value in initial_k1
            ]
            report = check_stability(create_cell(candidate))
            total_trials += 1
            if _is_stable(report):
                return {
                    "found": True,
                    "best_k1": candidate,
                    "stage": stage_name,
                    "trials": total_trials,
                    "report": report,
                }
            if progress_every and (trial + 1) % progress_every == 0:
                print(f"  {stage_name}: tried {trial + 1} configurations")

    return {
        "found": False,
        "best_k1": initial_k1,
        "stage": "failed",
        "trials": total_trials,
        "report": initial_report,
    }


def _tune_mod1(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    frac = value - math.floor(value)
    if frac < 0:
        frac += 1.0
    return frac


def _near_resonance(frac: float | None, tolerance: float = 0.03) -> bool:
    if frac is None:
        return False
    resonances = [0.0, 0.25, 1.0 / 3.0, 0.5]
    return any(abs(frac - resonance) <= tolerance for resonance in resonances)


def objective_from_result(
    result: dict[str, Any] | None,
    *,
    target_emittance: float | None = None,
    tune_window: tuple[float, float] = (0.05, 0.495),
    chrom_limit: float = 10.0,
) -> float:
    """Compute a scalar objective where smaller is better.

    This objective is designed for script-level optimization after a stable
    configuration has been found. Unstable or malformed results receive a
    large penalty so the optimizer naturally rejects them.
    """
    if not result:
        return 1e12

    stable = _to_bool(result.get("stable"))
    if stable is not True:
        return 1e9

    cost = 0.0

    emittance = _to_float(result.get("emittance"))
    if emittance is not None and math.isfinite(emittance) and emittance > 0:
        if target_emittance is not None and target_emittance > 0:
            ratio = emittance / target_emittance
            cost += ratio * ratio
        else:
            # Keep a soft preference toward low emittance even without a target.
            cost += emittance * 1e12
    else:
        cost += 5e6

    low, high = tune_window
    for key in ("tune_nux", "tune_nuy"):
        tune = _to_float(result.get(key))
        frac = _tune_mod1(tune)
        if frac is None:
            cost += 2e5
            continue
        if frac < low:
            cost += ((low - frac) * 100.0) ** 2
        elif frac > high:
            cost += ((frac - high) * 100.0) ** 2
        if _near_resonance(frac):
            cost += 250.0

    for key in ("chromaticity_xix", "chromaticity_xiy"):
        chrom = _to_float(result.get(key))
        if chrom is None:
            cost += 1e4
            continue
        exceed = abs(chrom) - chrom_limit
        if exceed > 0:
            cost += (exceed * 8.0) ** 2

    return float(cost)


def optimize_stable_k1(
    create_cell: Callable[[Sequence[float]], Sequence[Any]],
    initial_k1: Sequence[float],
    check_stability: Callable[[Sequence[Any]], Any],
    evaluate_result: Callable[[Sequence[Any]], dict[str, Any]],
    *,
    target_emittance: float | None = None,
    iterations: int = 4000,
    step_scale: float = 0.08,
    cooling: float = 0.999,
    progress_every: int = 500,
    rng=None,
) -> dict[str, Any]:
    """Refine a stable K1 vector with a lightweight stochastic search.

    Strategy:
    1. Start from a known-stable vector.
    2. Propose sign-preserving random perturbations.
    3. Keep only stable candidates.
    4. Accept improvements on a scalar objective and cool step size.
    """
    if rng is None:
        try:
            import numpy as np

            rng = np.random.default_rng()
        except Exception as exc:
            raise RuntimeError("numpy is required for optimization") from exc

    current_k1 = list(initial_k1)
    current_cell = create_cell(current_k1)
    current_stability = check_stability(current_cell)
    if not _is_stable(current_stability):
        return {
            "success": False,
            "reason": "initial_k1_not_stable",
            "best_k1": current_k1,
            "best_result": None,
            "best_objective": 1e12,
            "accepted": 0,
            "attempted": 0,
        }

    best_result = evaluate_result(current_cell)
    best_objective = objective_from_result(best_result, target_emittance=target_emittance)
    accepted = 0
    attempted = 0
    current_step = float(step_scale)

    for i in range(iterations):
        attempted += 1
        candidate_k1 = [
            _preserve_sign(base, base * (1.0 + rng.normal(0.0, current_step)))
            for base in current_k1
        ]
        candidate_cell = create_cell(candidate_k1)
        if not _is_stable(check_stability(candidate_cell)):
            current_step *= cooling
            continue

        candidate_result = evaluate_result(candidate_cell)
        candidate_objective = objective_from_result(
            candidate_result,
            target_emittance=target_emittance,
        )
        if candidate_objective < best_objective:
            current_k1 = candidate_k1
            best_result = candidate_result
            best_objective = candidate_objective
            accepted += 1

        current_step *= cooling
        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"  optimize: iter {i + 1}/{iterations}, accepted={accepted}, "
                f"best_obj={best_objective:.6g}"
            )

    return {
        "success": True,
        "best_k1": current_k1,
        "best_result": best_result,
        "best_objective": best_objective,
        "accepted": accepted,
        "attempted": attempted,
    }


def compute_dynamic_aperture(
    jt: Any,
    ring,
    *,
    nturns: int = 1024,
    amp_max: float = 0.025,
    amp_step: float = 0.0005,
    angle_steps: int = 20,
    energy_gev: float | None = None,
    dp: float = 0.0,
) -> dict[str, Any]:
    """Compute the dynamic aperture (DA) of a ring lattice.

    Uses ``jt.dynamic_aperture()`` when available, with a particle-tracking
    fallback for compatibility with older pyJuTrack builds.

    Parameters
    ----------
    ring:
        A ``jt.Lattice`` object for the full ring (cell × N_cells).
    nturns:
        Number of turns to track.  1024 gives a reasonable estimate quickly.
    amp_max:
        Maximum initial amplitude to probe [m].  Default 25 mm covers most
        storage-ring apertures.
    amp_step:
        Amplitude step for the boundary search [m].
    angle_steps:
        Number of angular directions (x,y) to probe.  20 gives good coverage.
    energy_gev:
        Beam energy [GeV].  Required for ``jt.dynamic_aperture`` energy param.
    dp:
        Off-momentum deviation δ = Δp/p₀.  Use 0 for on-momentum DA.

    Returns
    -------
    dict with keys:
        da_boundary : list of [x, y] pairs [m] at the DA boundary
        da_area_mm2 : approximate DA area [mm²] (polygon area of boundary)
        da_min_x_mm : minimum horizontal half-aperture [mm]
        da_min_y_mm : minimum vertical half-aperture [mm]
        nturns      : number of turns used
        method      : "jt.dynamic_aperture" or "tracking_fallback"
        error       : error message if computation failed (key absent on success)
    """
    energy_ev = (energy_gev * 1e9) if energy_gev is not None else None

    # ── Try native jt.dynamic_aperture ──────────────────────────────────────
    if hasattr(jt, "dynamic_aperture"):
        try:
            _kwargs: dict[str, Any] = {
                "nturns": nturns,
                "amp_max": amp_max,
                "amp_step": amp_step,
                "angle_steps": angle_steps,
                "dp": dp,
            }
            if energy_ev is not None:
                _kwargs["E"] = energy_ev
            with suppressed_output():
                da_raw, _ = jt.dynamic_aperture(ring, **_kwargs)
            # da_raw shape: (angle_steps, 2) → list of [x, y]
            import numpy as np  # noqa: PLC0415 — local import OK inside function
            da_array = np.asarray(da_raw)
            boundary = [[float(row[0]), float(row[1])] for row in da_array]
            return _da_summary(boundary, nturns=nturns, method="jt.dynamic_aperture")
        except Exception as exc:
            # Fall through to tracking fallback
            _native_err = str(exc)
    else:
        _native_err = "jt.dynamic_aperture not available"

    # ── Tracking fallback: probe a grid of (x, y) initial conditions ────────
    try:
        import numpy as np  # noqa: PLC0415

        angles = np.linspace(0.0, np.pi / 2, angle_steps, endpoint=True)
        boundary: list[list[float]] = []

        for theta in angles:
            last_alive = [0.0, 0.0]
            amp = amp_step
            while amp <= amp_max:
                x0 = amp * np.cos(theta)
                y0 = amp * np.sin(theta)
                # 6D coordinate: [x, px, y, py, t, delta]
                coords = np.array([[x0, 0.0, y0, 0.0, 0.0, dp]])
                try:
                    import copy  # noqa: PLC0415
                    test_ring = copy.deepcopy(ring)
                    beam_r = np.zeros((1, 6))
                    beam_r[0] = [x0, 0.0, y0, 0.0, 0.0, dp]
                    if energy_ev is not None:
                        beam = jt.Beam(beam_r, energy=energy_ev)
                    else:
                        beam = jt.Beam(beam_r, energy=3e9)
                    with suppressed_output():
                        jt.pringpass(test_ring, beam, num_turns=nturns)
                    lost = jt.check_lost(beam)
                    if not bool(lost[0]):
                        last_alive = [float(x0), float(y0)]
                        amp += amp_step
                    else:
                        break
                except Exception:
                    break
            boundary.append(last_alive)

        return _da_summary(boundary, nturns=nturns, method="tracking_fallback")
    except Exception as exc:
        return {
            "da_boundary": [],
            "da_area_mm2": None,
            "da_min_x_mm": None,
            "da_min_y_mm": None,
            "nturns": nturns,
            "method": "failed",
            "error": f"native: {_native_err}; fallback: {exc}",
        }


def _da_summary(
    boundary: list[list[float]],
    *,
    nturns: int,
    method: str,
) -> dict[str, Any]:
    """Compute scalar DA metrics from a boundary point list."""
    try:
        import numpy as np  # noqa: PLC0415

        pts = np.asarray(boundary)
        if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
            raise ValueError("empty boundary")

        xs = pts[:, 0] * 1e3  # → mm
        ys = pts[:, 1] * 1e3

        # Shoelace polygon area
        n = len(xs)
        area = abs(
            sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
        ) / 2.0

        return {
            "da_boundary": boundary,
            "da_area_mm2": float(area),
            "da_min_x_mm": float(xs[xs > 0].min()) if any(xs > 0) else 0.0,
            "da_min_y_mm": float(ys[ys > 0].min()) if any(ys > 0) else 0.0,
            "nturns": nturns,
            "method": method,
        }
    except Exception as exc:
        return {
            "da_boundary": boundary,
            "da_area_mm2": None,
            "da_min_x_mm": None,
            "da_min_y_mm": None,
            "nturns": nturns,
            "method": method,
            "error": str(exc),
        }


def score_design_result(
    result: dict[str, Any] | None,
    *,
    target_emittance: float | None = None,
    target_circumference: float | None = None,
    target_energy_gev: float | None = None,
    machine_class: str | None = None,
    full_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically score a result block for UI and review summaries."""
    _fc = full_constraints or {}

    if not result:
        return {"score": 0, "status": "missing", "issues": ["No result block found"]}

    # ── Infeasibility refusal — a valid outcome, not a failure ────────────────
    if result.get("infeasible") is True or result.get("status") == "infeasible-refusal":
        binding = result.get("binding_constraint", "unknown")
        return {
            "score": 85,
            "status": "infeasible-refusal",
            "issues": [],
            "critical_issues": [],
            "infeasible": True,
            "binding_constraint": binding,
        }

    mode = str(result.get("lattice_mode", "ring")).strip().lower()
    machine = str(machine_class or "unknown").strip().lower()

    # ── Cell / transfer-line scoring ──────────────────────────────────────────
    if mode == "cell":
        issues: list[str] = []
        critical_issues: list[str] = []
        score = 100
        if result.get("error"):
            critical_issues.append("Execution reported an error")
            score -= 50
        # Derive is_linearly_stable from traces if not explicitly present
        _is_ls = result.get("is_linearly_stable")
        if _is_ls is None:
            # infer from trace values or from the 'stable' field
            _tx = _to_float(result.get("trace_x"))
            _ty = _to_float(result.get("trace_y"))
            if _tx is not None and _ty is not None:
                _is_ls = abs(_tx) < 2.0 and abs(_ty) < 2.0
            else:
                _is_ls = bool(result.get("stable", False))
        if not _is_ls:
            critical_issues.append("Cell is not linearly stable (|trace| >= 2)")
            score -= 40
        if result.get("T56") is None:
            issues.append("T56 (momentum compaction contribution) not computed")
            score -= 10
        if result.get("D_exit") is None:
            issues.append("Dispersion at exit not computed")
            score -= 10
        # Phase-advance target check (if the task requested a specific mu)
        target_mu_x = _to_float(_fc.get("target_phase_advance_x_deg"))
        mu_x = _to_float(result.get("phase_advance_x_deg"))
        if target_mu_x is not None and mu_x is not None:
            mu_err = abs(mu_x - target_mu_x)
            if mu_err > 10.0:
                issues.append(
                    f"Horizontal phase advance {mu_x:.1f}° deviates from target "
                    f"{target_mu_x:.1f}° by {mu_err:.1f}°"
                )
                score -= 20
            elif mu_err > 3.0:
                issues.append(
                    f"Horizontal phase advance {mu_x:.1f}° deviates from target "
                    f"{target_mu_x:.1f}° by {mu_err:.1f}°"
                )
                score -= 10
        # Stability: trace-based check (reliable in this environment)
        # Symplecticity via numpy matmul is not available in some conda setups.
        issues.extend(critical_issues)
        status = "accepted" if score >= 70 and not critical_issues else "needs-review"
        if critical_issues:
            status = "invalid-physics"
        return {
            "score": max(score, 0),
            "status": status,
            "issues": issues,
            "critical_issues": critical_issues,
        }

    if mode == "linac":
        issues: list[str] = []
        critical_issues: list[str] = []
        score = 100
        length = _to_float(result.get("circumference"))
        if length is None or length <= 0:
            critical_issues.append("Missing or invalid beamline length")
            score -= 40
        if result.get("error"):
            critical_issues.append("Execution reported an error")
            score -= 40
        issues.extend(critical_issues)
        status = "accepted" if score >= 70 and not issues else "needs-review"
        if critical_issues:
            status = "invalid-physics"
        return {
            "score": max(score, 0),
            "status": status,
            "issues": issues,
            "critical_issues": critical_issues,
        }

    issues: list[str] = []
    critical_issues: list[str] = []
    score = 100

    stable = _to_bool(result.get("stable"))
    if stable is not True:
        return {
            "score": 0,
            "status": "unstable",
            "issues": ["Cell is not stable"],
            "critical_issues": ["Cell is not stable"],
        }

    if result.get("error"):
        critical_issues.append("Execution reported an error")
        score -= 40

    trace_x = _to_float(result.get("trace_x"))
    trace_y = _to_float(result.get("trace_y"))
    # Only penalise missing traces when we have no other stability evidence.
    # For ring mode: gettune succeeding implies the ring is already stable;
    # trace_x/y from findm66 is optional for ring scripts.
    _has_tunes = (_to_float(result.get("tune_nux")) is not None and
                  _to_float(result.get("tune_nuy")) is not None)
    for trace_value, label in ((trace_x, "trace_x"), (trace_y, "trace_y")):
        if trace_value is not None and not math.isfinite(trace_value):
            critical_issues.append(f"{label} is non-finite")
            score -= 20
        # Don't penalise absence if tunes computed (implicit stability evidence)

    for key in ("tune_nux", "tune_nuy"):
        tune = _to_float(result.get(key))
        frac = _tune_mod1(tune)
        if frac is None:
            issues.append(f"Missing {key}")
            score -= 15
            continue
        # Only penalise if very close to an integer tune (forbidden zone < 0.02 or > 0.98)
        if frac < 0.02 or frac > 0.98:
            issues.append(f"{key} (mod 1) value {frac:.3f} is dangerously close to an integer tune")
            score -= 20
        elif _near_resonance(frac):
            issues.append(f"{key} (mod 1) value {frac:.3f} is near a low-order resonance (0, 1/4, 1/3, 1/2)")
            score -= 15

    for key in ("chromaticity_xix", "chromaticity_xiy"):
        chrom = _to_float(result.get(key))
        if chrom is None:
            issues.append(f"Missing {key}")
            score -= 10
            continue
        if not math.isfinite(chrom):
            critical_issues.append(f"{key} is non-finite")
            score -= 25
        elif abs(chrom) > 5e3:
            critical_issues.append(f"{key} magnitude {abs(chrom):.0f} is physically implausible (>5000)")
            score -= 30
        # Natural chromaticity for MBA rings can legitimately be in the -50 to -500 range;
        # only warn (not penalise heavily) since correction is a separate step.

    # ── Robinson damping partition check (Jx + Jy + Je = 4) ────────────────────
    Jx = _to_float(result.get("Jx"))
    Jy = _to_float(result.get("Jy"))
    Je = _to_float(result.get("Je"))
    if Jx is not None and Jy is not None and Je is not None:
        J_sum = Jx + Jy + Je
        J_err = abs(J_sum - 4.0)
        if J_err > 0.10:
            critical_issues.append(
                f"Damping partition sum Jx+Jy+Je = {J_sum:.3f} ≠ 4.000 "
                f"(Robinson theorem violation, Δ = {J_err:.3f})"
            )
            score -= 25
        elif J_err > 0.02:
            issues.append(
                f"Damping partition sum Jx+Jy+Je = {J_sum:.3f} deviates from 4.000 "
                f"by {J_err:.3f} — check combined-function magnets or RF"
            )
            score -= 10

    # ── U0 vs. Cgamma·E⁴·I2/(2π) cross-check (electron rings) ──────────────
    _Cgamma = 8.85e-5   # m/GeV³ (Wiedemann/Sands constant; C_γ = r_e/(3(m_e c²)³))
    I2 = _to_float(result.get("I2"))
    U0_eV = _to_float(result.get("U0_eV"))
    energy_gev_val = _to_float(result.get("energy_gev"))
    if I2 is not None and U0_eV is not None and energy_gev_val is not None:
        if math.isfinite(I2) and math.isfinite(U0_eV) and math.isfinite(energy_gev_val):
            U0_analytic_eV = _Cgamma * (energy_gev_val ** 4) * I2 / (2.0 * math.pi) * 1e9
            if U0_analytic_eV > 0 and U0_eV > 0:
                U0_rel_err = abs(U0_eV - U0_analytic_eV) / max(U0_analytic_eV, 1.0)
                if U0_rel_err > 0.15:
                    issues.append(
                        f"U0 ({U0_eV:.1f} eV) differs from Cgamma·E⁴·I2/(2π) "
                        f"analytical estimate ({U0_analytic_eV:.1f} eV) by "
                        f"{U0_rel_err * 100:.0f}% — check radiation integral computation"
                    )
                    score -= 10

    emittance = _to_float(result.get("emittance"))
    if emittance is None or not math.isfinite(emittance) or emittance <= 0:
        critical_issues.append("Emittance is missing or nonphysical")
        score -= 40
    elif target_emittance is not None and target_emittance > 0:
        ratio = emittance / target_emittance
        if ratio > 100.0:
            critical_issues.append("Emittance is more than 100x above the target")
            score -= 40
        elif ratio > 20.0:
            issues.append("Emittance is more than 20x above the target")
            score -= 30
        elif ratio > 3.0:
            issues.append("Emittance is more than 3x above the target")
            score -= 20

    circumference = _to_float(result.get("circumference"))
    if target_circumference is not None and target_circumference > 0:
        if circumference is None or not math.isfinite(circumference) or circumference <= 0:
            critical_issues.append("Circumference is missing or invalid")
            score -= 20
        else:
            rel_err = abs(circumference - target_circumference) / target_circumference
            if rel_err > 0.40:
                critical_issues.append("Circumference differs from the target by more than 40%")
                score -= 30
            elif rel_err > 0.20:
                issues.append("Circumference differs from the target by more than 20%")
                score -= 20
            elif rel_err > 0.10:
                issues.append("Circumference differs from the target by more than 10%")
                score -= 10

    energy_gev = _to_float(result.get("energy_gev"))
    if target_energy_gev is not None and target_energy_gev > 0:
        if energy_gev is None or not math.isfinite(energy_gev) or energy_gev <= 0:
            critical_issues.append("Beam energy is missing or invalid")
            score -= 20
        else:
            rel_err = abs(energy_gev - target_energy_gev) / target_energy_gev
            if rel_err > 0.20:
                critical_issues.append("Beam energy differs from the target by more than 20%")
                score -= 30
            elif rel_err > 0.10:
                issues.append("Beam energy differs from the target by more than 10%")
                score -= 20
            elif rel_err > 0.05:
                issues.append("Beam energy differs from the target by more than 5%")
                score -= 10

    if machine == "booster" and target_emittance is not None and emittance is not None and emittance > 0:
        if emittance > max(10.0 * target_emittance, 1e-7):
            issues.append("Booster emittance is still far above the requested injector-quality range")
            score -= 10

    issues.extend(critical_issues)
    status = "accepted" if score >= 70 and not issues else "needs-review"
    if critical_issues:
        status = "invalid-physics"
    return {
        "score": max(score, 0),
        "status": status,
        "issues": issues,
        "critical_issues": critical_issues,
    }


def format_result_table(result: dict[str, Any]) -> str:
    """Render the canonical lattice result table."""
    lines = [
        "--- LATTICE RESULTS ---",
        "Parameter            | Value",
        "---------------------|-------------------",
        f"Stable               | {result.get('stable')}",
        f"Trace x              | {result.get('trace_x')}",
        f"Trace y              | {result.get('trace_y')}",
        f"Tune nux             | {result.get('tune_nux')}",
        f"Tune nuy             | {result.get('tune_nuy')}",
        f"Chromaticity xix     | {result.get('chromaticity_xix')}",
        f"Chromaticity xiy     | {result.get('chromaticity_xiy')}",
        f"Emittance [m.rad]    | {result.get('emittance')}",
        f"Circumference [m]    | {result.get('circumference')}",
        f"Dx at entrance [m]   | {result.get('dx_entrance')}",
        f"Dpx at entrance      | {result.get('dpx_entrance')}",
    ]
    # Radiation integrals and derived quantities (ring mode only)
    _rad_fields = [
        ("alphac", "Mom. compaction     |"),
        ("U0_eV",  "Energy loss/turn[eV]|"),
        ("sigma_E", "Energy spread       |"),
        ("damping_time_x", "Damp. time x [s]    |"),
        ("damping_time_y", "Damp. time y [s]    |"),
        ("damping_time_E", "Damp. time E [s]    |"),
        ("Jx", "Damp. partition Jx  |"),
        ("Jy", "Damp. partition Jy  |"),
        ("Je", "Damp. partition Je  |"),
        ("I1", "Rad. integral I1    |"),
        ("I2", "Rad. integral I2    |"),
        ("I3", "Rad. integral I3    |"),
        ("I4", "Rad. integral I4    |"),
        ("I5", "Rad. integral I5    |"),
    ]
    for key, label in _rad_fields:
        val = result.get(key)
        if val is not None:
            lines.append(f"{label} {val}")
    lines.append("--- END RESULTS ---")
    return "\n".join(lines)


def emit_result_report(result: dict[str, Any]) -> None:
    """Print both a human-readable table and a machine-readable JSON block."""
    serialized = {key: _serialize_value(value) for key, value in result.items()}
    print(format_result_table(serialized))
    print(RESULT_JSON_BEGIN)
    print(json.dumps(serialized, ensure_ascii=True, sort_keys=True))
    print(RESULT_JSON_END)


def parse_result_output(text: str) -> dict[str, Any] | None:
    """Parse structured results from stdout, preferring the JSON block."""
    # ── Infeasibility refusal block ──────────────────────────────────────────
    if "INFEASIBLE_DESIGN" in text:
        binding = None
        # Try to find a clean constraint name (stop before " and", newline, ".")
        for pat in [
            r"[Bb]inding constraint\s*[:\-]\s*([a-z_A-Z][a-z_A-Z0-9_]*)",
            r"[Bb]inding\s+[Cc]onstraint\s*[:\-]\s*([a-z_A-Z][a-z_A-Z0-9_]*)",
            r"Binding:\s*([a-z_A-Z][a-z_A-Z0-9_]*)",
        ]:
            m = re.search(pat, text)
            if m:
                binding = m.group(1).strip().rstrip('.').strip()
                break
        # Also scan for well-known constraint names if pattern didn't find one
        if binding is None:
            for name in ("target_emittance", "emittance", "circumference_m", "circumference",
                         "target_tune", "target_phase_advance"):
                if name in text.lower():
                    binding = name
                    break
        return {
            "infeasible": True,
            "status": "infeasible-refusal",
            "binding_constraint": binding,
            "stable": None,
            "lattice_mode": "ring",
        }

    json_match = re.search(
        rf"{re.escape(RESULT_JSON_BEGIN)}\s*(.+?)\s*{re.escape(RESULT_JSON_END)}",
        text,
        re.DOTALL,
    )
    if json_match:
        try:
            payload = json.loads(json_match.group(1).strip())
            if isinstance(payload, dict):
                # Normalise common alternative key names
                _aliases = {
                    "emittance_nm": ("emittance", lambda v: v * 1e-9),
                    "emittance_pm": ("emittance", lambda v: v * 1e-12),
                    "emittance_m_rad": ("emittance", lambda v: v),
                    "energy_GeV": ("energy_gev", lambda v: v),
                    "energy_gev_val": ("energy_gev", lambda v: v),
                    "n_cells_ring": ("n_cells", lambda v: v),
                    "max_Dx": ("Dx_max", lambda v: v),
                    "max_dx": ("Dx_max", lambda v: v),
                    "max_betax": ("beta_x_max", lambda v: v),
                    "max_betay": ("beta_y_max", lambda v: v),
                    "sigma_E_rel": ("sigma_E", lambda v: v),
                    "Robinson_sum": ("robinson_sum", lambda v: v),
                    # tune aliases (LLMs often omit the tune_ prefix)
                    "nux": ("tune_nux", lambda v: v),
                    "nuy": ("tune_nuy", lambda v: v),
                    "xix": ("chromaticity_xix", lambda v: v),
                    "xiy": ("chromaticity_xiy", lambda v: v),
                    "nu_x": ("tune_nux", lambda v: v),
                    "nu_y": ("tune_nuy", lambda v: v),
                    "xi_x": ("chromaticity_xix", lambda v: v),
                    "xi_y": ("chromaticity_xiy", lambda v: v),
                }
                for old_key, (new_key, transform) in _aliases.items():
                    if old_key in payload and new_key not in payload:
                        try:
                            payload[new_key] = transform(payload[old_key])
                        except Exception:
                            pass
                return payload
        except json.JSONDecodeError:
            pass

    block_match = re.search(
        r"---\s*LATTICE RESULTS\s*---(.+?)---\s*END RESULTS\s*---",
        text,
        re.DOTALL,
    )
    if not block_match:
        return None

    block = block_match.group(1)
    parsed: dict[str, Any] = {}
    patterns = [
        (r"Stable\s*[|:]\s*(\S+)", "stable", _to_bool),
        (r"Trace x\s*[|:]\s*([\d.eE+-]+)", "trace_x", _to_float),
        (r"Trace y\s*[|:]\s*([\d.eE+-]+)", "trace_y", _to_float),
        (r"Tune nux\s*[|:]\s*([\d.eE+-]+)", "tune_nux", _to_float),
        (r"Tune nuy\s*[|:]\s*([\d.eE+-]+)", "tune_nuy", _to_float),
        (r"Chromaticity xix\s*[|:]\s*([\d.eE+-]+)", "chromaticity_xix", _to_float),
        (r"Chromaticity xiy\s*[|:]\s*([\d.eE+-]+)", "chromaticity_xiy", _to_float),
        (r"Emittance(?:\s*\[.*?\])?\s*[|:]\s*([\d.eE+-]+)", "emittance", _to_float),
        (r"Circumference(?:\s*\[.*?\])?\s*[|:]\s*([\d.eE+-]+)", "circumference", _to_float),
        (r"Dx at entrance(?:\s*\[.*?\])?\s*[|:]\s*([\d.eE+-]+)", "dx_entrance", _to_float),
        (r"Dpx at entrance\s*[|:]\s*([\d.eE+-]+)", "dpx_entrance", _to_float),
        (r"nux\s*=\s*([\d.eE+-]+)", "tune_nux", _to_float),
        (r"nuy\s*=\s*([\d.eE+-]+)", "tune_nuy", _to_float),
        (r"xix\s*=\s*([\d.eE+-]+)", "chromaticity_xix", _to_float),
        (r"xiy\s*=\s*([\d.eE+-]+)", "chromaticity_xiy", _to_float),
        # Radiation integrals and derived quantities
        (r"Mom\.\s*compaction\s*[|:]\s*([\d.eE+-]+)", "alphac", _to_float),
        (r"Energy loss.*?\[eV\]\s*[|:]\s*([\d.eE+-]+)", "U0_eV", _to_float),
        (r"Energy spread\s*[|:]\s*([\d.eE+-]+)", "sigma_E", _to_float),
        (r"Damp\.\s*time x\s*[|:]\s*([\d.eE+-]+)", "damping_time_x", _to_float),
        (r"Damp\.\s*time y\s*[|:]\s*([\d.eE+-]+)", "damping_time_y", _to_float),
        (r"Damp\.\s*time E\s*[|:]\s*([\d.eE+-]+)", "damping_time_E", _to_float),
        (r"Damp\.\s*partition Jx\s*[|:]\s*([\d.eE+-]+)", "Jx", _to_float),
        (r"Damp\.\s*partition Jy\s*[|:]\s*([\d.eE+-]+)", "Jy", _to_float),
        (r"Damp\.\s*partition Je\s*[|:]\s*([\d.eE+-]+)", "Je", _to_float),
        (r"Rad\.\s*integral I1\s*[|:]\s*([\d.eE+-]+)", "I1", _to_float),
        (r"Rad\.\s*integral I2\s*[|:]\s*([\d.eE+-]+)", "I2", _to_float),
        (r"Rad\.\s*integral I3\s*[|:]\s*([\d.eE+-]+)", "I3", _to_float),
        (r"Rad\.\s*integral I4\s*[|:]\s*([\d.eE+-]+)", "I4", _to_float),
        (r"Rad\.\s*integral I5\s*[|:]\s*([\d.eE+-]+)", "I5", _to_float),
    ]
    for pattern, key, converter in patterns:
        hit = re.search(pattern, block)
        if hit:
            parsed[key] = converter(hit.group(1))
    return parsed or None
