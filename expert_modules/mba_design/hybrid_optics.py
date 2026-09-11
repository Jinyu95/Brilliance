"""Measured core-optics contracts and usable chromatic-region diagnostics."""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from brilliance.lattice_fingerprint import cell_fingerprint

TWISS_FIELDS = ("beta_x", "beta_y", "alpha_x", "alpha_y", "Dx", "Dpx")
CORE_TOLERANCES = {"beta_relative": .01, "alpha_absolute": .01,
                   "dispersion_m": 1e-4, "dispersion_slope": 1e-4}
DRIFT_REQUIREMENTS = ("min_usable_drift_m", "min_total_usable_drift_m",
                      "max_drift_slope", "max_beta_m", "max_core_dispersion_m")


def _at(profile, s):
    point = min(profile, key=lambda p: abs(p["s_m"] - s))
    if abs(point["s_m"] - s) > 1e-8:
        raise ValueError("Required physical boundary sample is missing; resample with JuTrack")
    return point


def usable_bump_drifts(rows, profile, roles, thresholds, core_dispersion_m):
    """Use exact field-free drift transport, never a fitted peak inside a magnet.

    Adjacent drifts/markers form one installation interval. Report the portion
    above both the absolute and core-relative dispersion requirements. A global
    peak may be in a quadrupole; it cannot substitute for installation space.
    """
    missing = set(DRIFT_REQUIREMENTS) - thresholds.keys()
    if missing:
        raise ValueError("Declare usable-bump thresholds: " + ", ".join(sorted(missing)))
    for key in DRIFT_REQUIREMENTS:
        if not math.isfinite(float(thresholds[key])) or thresholds[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    target = max(thresholds["min_bump_m"], thresholds["min_contrast"] * core_dispersion_m)
    inventory, checks, metrics = [], {"drift_sampling_complete": True}, {}
    starts = np.cumsum([0.] + [r["length_m"] for r in rows]).tolist()
    for side in ("left", "right"):
        indices = roles["regions"][side + "_bump"]["element_indices"]
        groups, group = [], []
        for i in indices:
            if rows[i]["kind"] in {"DRIFT", "MARKER"}:
                group.append(i)
            elif group:
                groups.append(group)
                group = []
        if group:
            groups.append(group)
        lengths = []
        for group in groups:
            s, end = starts[group[0]], starts[group[-1] + 1]
            length = end - s
            if length <= 0:
                continue
            if any(min(abs(p["s_m"]-boundary) for p in profile) > 1e-8 for boundary in (s,end)):
                checks["drift_sampling_complete"] = False
                inventory.append({"side": side, "elements": [rows[i]["name"] for i in group],
                    "s_start_m": s, "s_end_m": end, "usable_length_m": 0.,
                    "failure": "Missing physical drift entrance/exit samples"})
                continue
            entrance, exit_point = _at(profile, s), _at(profile, end)
            d, dp = entrance["Dx"], entrance["Dpx"]
            samples = [p for p in profile if s - 1e-9 <= p["s_m"] <= end + 1e-9]
            # Check the advertised field-free interval, not just its endpoints.
            transport = all(abs(p["Dx"] - (d + (p["s_m"] - s)*dp)) <= 1e-8
                            and abs(p["Dpx"] - dp) <= 1e-8 for p in samples)
            lo, hi = 0., length
            if abs(dp) < 1e-14:
                if d < target - 1e-10:
                    hi = 0.
            elif dp > 0:
                lo = max(lo, (target - d)/dp)
            else:
                hi = min(hi, (target - d)/dp)
            lo, hi = max(0., min(length, lo)), max(0., min(length, hi))
            usable = max(0., hi-lo) if transport and abs(dp) <= thresholds["max_drift_slope"] else 0.
            lengths.append(usable)
            beta_x = [p["beta_x"] for p in samples]
            beta_y = [p["beta_y"] for p in samples]
            inventory.append({"side": side, "elements": [rows[i]["name"] for i in group],
                "s_start_m": s, "s_end_m": end, "length_m": length,
                "Dx_min_m": min(d, exit_point["Dx"]), "Dx_max_m": max(d, exit_point["Dx"]),
                "Dpx": dp, "drift_transport_passed": transport,
                "usable_start_m": s+lo, "usable_end_m": s+max(lo, hi), "usable_length_m": usable,
                "beta_x_min_m": min(beta_x), "beta_x_max_m": max(beta_x),
                "beta_y_min_m": min(beta_y), "beta_y_max_m": max(beta_y),
                "beta_x_Dx_max_m2": max(p["beta_x"]*p["Dx"] for p in samples),
                "beta_y_Dx_max_m2": max(p["beta_y"]*p["Dx"] for p in samples)})
        metrics[side + "_usable_drift_m"] = sum(lengths)
        metrics[side + "_longest_usable_drift_m"] = max(lengths, default=0.)
        checks[side + "_usable_drift"] = max(lengths, default=0.) >= thresholds["min_usable_drift_m"] - 1e-9
        checks[side + "_usable_drift_total"] = sum(lengths) >= thresholds["min_total_usable_drift_m"] - 1e-9
    metrics["beta_max_m"] = max(max(p["beta_x"], p["beta_y"]) for p in profile)
    checks["beta_envelope"] = metrics["beta_max_m"] <= thresholds["max_beta_m"]
    checks["low_core_dispersion"] = core_dispersion_m <= thresholds["max_core_dispersion_m"]
    return {"checks": checks, "metrics": metrics, "drifts": inventory,
            "required_drift_dispersion_m": target, "passed": all(checks.values())}


def measure_core_optics(candidate, *, core_start_name: str, core_end_name: str,
                        max_step_m: float = .02) -> dict[str, Any]:
    """Measure the explicit source cell before modifying it; no saved seed lookup."""
    from brilliance import accelerator_api as api
    sampler = api.build_jutrack_profile_sampler(candidate, max_step_m=max_step_m)
    linear = sampler.linear_metrics()
    if not linear.get("stable"):
        raise ValueError("Core reference requires a stable, map-checked source cell")
    rows = api.one_period_elements(candidate)
    names = [r["name"] for r in rows]
    start, end = names.index(core_start_name), names.index(core_end_name)
    if start > end:
        raise ValueError("Core boundaries must be ordered")
    s0 = sum(r["length_m"] for r in rows[:start])
    s1 = sum(r["length_m"] for r in rows[:end+1])
    points = [deepcopy(p) for p in linear["twiss_profile"]
              if s0-1e-9 <= p["s_m"] <= s1+1e-9 and not p.get("extremum_method")]
    return {"schema": "hybrid_core_optics/v1", "source_lattice_id": candidate.lattice_id,
            "source_cell_fingerprint": cell_fingerprint(rows), "energy_gev": candidate.energy_gev,
            "core_start_name": core_start_name, "core_end_name": core_end_name,
            "s_start_m": s0, "s_end_m": s1, "max_step_m": max_step_m,
            "profile": points, "entrance": deepcopy(_at(points, s0)),
            "exit": deepcopy(_at(points, s1)), "sampling": linear["sampling"]}


def compare_core_optics(reference: Mapping[str, Any], profile: Sequence[Mapping[str, Any]], *,
                        tolerances: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Compare periodic optics at every reference sample, not only the centre.

    Requires identical spatial sampling for this fixed-geometry replacement
    route. Missing points fail closed; no interpolation manufactures a pass.
    """
    tol = {**CORE_TOLERANCES, **dict(tolerances or {})}
    if set(tol) != set(CORE_TOLERANCES) or any(not math.isfinite(float(v)) or v <= 0 for v in tol.values()):
        raise ValueError("Invalid core-optics tolerance contract")
    if reference.get("schema") != "hybrid_core_optics/v1" or not reference.get("profile"):
        raise ValueError("Measured source-core optical reference is required")
    maxima = {k: 0. for k in TWISS_FIELDS}
    comparisons = []
    for p in reference["profile"]:
        q = _at(profile, p["s_m"])
        errors = {}
        for key in TWISS_FIELDS:
            a, b = float(p[key]), float(q[key])
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Nonfinite core optics")
            if key.startswith("beta"):
                if a <= 0 or b <= 0:
                    raise ValueError("Core beta must be positive")
                error = abs(b/a-1)/tol["beta_relative"]
            else:
                scale = tol["alpha_absolute"] if key.startswith("alpha") else tol["dispersion_m"] if key == "Dx" else tol["dispersion_slope"]
                error = abs(b-a)/scale
            errors[key] = error
            maxima[key] = max(maxima[key], error)
        comparisons.append({"s_m": p["s_m"], "reference": {k:p[k] for k in TWISS_FIELDS},
                            "candidate": {k:q[k] for k in TWISS_FIELDS}, "scaled_errors": errors})
    return {"passed": all(v <= 1 for v in maxima.values()), "tolerances": tol,
            "max_scaled_errors": maxima, "sample_count": len(comparisons),
            "samples": comparisons, "source_lattice_id": reference["source_lattice_id"]}
