"""Local checks that numerical knobs change the physical outputs of a builder.

No transport model, optimizer, or scientific acceptance is supplied here.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _numbers(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a nonempty mapping of names to finite scalars")
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name or isinstance(value, (bool, str, bytes)):
            raise ValueError(f"{label} requires nonempty string names and numeric scalars")
        array = np.asarray(value)
        if array.ndim != 0:
            raise ValueError(f"{label}[{name!r}] must be a finite scalar")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{label}[{name!r}] must be finite")
        result[name] = number
    return result


def audit_parameterization(
    build: Callable[[Mapping[str, float]], Mapping[str, float]],
    position: Mapping[str, float],
    *,
    bounds: Mapping[str, Sequence[float]],
    step_fraction: float = 0.01,
    output_atol: float | Mapping[str, float] = 0.0,
    output_rtol: float = 1e-10,
) -> dict[str, Any]:
    """Probe named knobs against named *physical* scalar outputs, not metadata.

    Each nonfixed knob is perturbed in both available directions by at most
    ``step_fraction * (upper - lower)``. The exact nominal point is never rounded
    or projected; out-of-bound input is an error. Output keys must stay identical.
    Supply per-output absolute tolerances when different physical units need
    different resolutions. A locally flat result is not global ineffectiveness
    or evidence that a scientific goal is impossible.

    Callback failures are recorded, not replaced by zero responses. The callback
    is called at the nominal point again in ``finally`` to restore private mutable
    state. A failed or inconsistent repeat makes the audit inconclusive. Every
    call must assign all coordinates; do not use a shared live simulation here.
    """
    point = _numbers(position, "position")
    if not isinstance(bounds, Mapping) or set(bounds) != set(point):
        raise ValueError("bounds must name exactly the same knobs as position")
    if not np.isfinite(step_fraction) or not 0 < step_fraction <= 1:
        raise ValueError("step_fraction must be finite and in (0, 1]")
    if not np.isfinite(output_rtol) or output_rtol < 0:
        raise ValueError("output_rtol must be finite and nonnegative")
    limits = {}
    for name, value in point.items():
        pair = np.asarray(bounds[name], dtype=float)
        if pair.shape != (2,) or not np.all(np.isfinite(pair)) or pair[0] > pair[1]:
            raise ValueError(f"bounds[{name!r}] must be a finite [lower, upper] pair")
        low, high = map(float, pair)
        if not np.isfinite(high - low):
            raise ValueError(f"bounds[{name!r}] has nonfinite width")
        if not low <= value <= high:
            raise ValueError(f"position[{name!r}] lies outside its bounds; no projection applied")
        limits[name] = (low, high)

    baseline = _numbers(build(dict(point)), "build output")
    if isinstance(output_atol, Mapping):
        absolute = _numbers(output_atol, "output_atol")
        if set(absolute) != set(baseline):
            raise ValueError("output_atol must name exactly the physical outputs")
    else:
        absolute = {name: float(output_atol) for name in baseline}
    if any(not np.isfinite(v) or v < 0 for v in absolute.values()):
        raise ValueError("output_atol must be finite and nonnegative")

    def measure(at: Mapping[str, float]) -> dict[str, float]:
        values = _numbers(build(dict(at)), "build output")
        if set(values) != set(baseline):
            raise ValueError("build output names changed; compare the same physical quantities")
        return values

    def changes(values: Mapping[str, float]) -> dict[str, Any]:
        return {
            key: {"baseline": before, "value": values[key]}
            for key, before in baseline.items()
            if abs(values[key] - before) > absolute[key] + output_rtol * abs(before)
        }

    evaluations = 1
    rows = []
    restored = None
    restoration_error = None
    try:
        for name, value in point.items():
            low, high = limits[name]
            requested = step_fraction * (high - low)
            probes = []
            for direction, boundary in ((-1, low), (1, high)):
                if boundary == value:
                    continue
                distance = min(requested, abs(boundary - value))
                at = boundary if distance == abs(boundary - value) else value + direction * distance
                probe: dict[str, Any] = {"direction": direction, "requested_step": requested,
                                         "value": at, "actual_step": at - value}
                try:
                    if at == value:
                        raise ValueError("probe step is below floating-point resolution")
                    if not low <= at <= high:
                        raise ValueError("probe arithmetic left declared bounds; no projection applied")
                    proposal = {**point, name: at}
                    evaluations += 1
                    values = measure(proposal)
                    probe.update(status="measured", changed_outputs=changes(values))
                except Exception as exc:
                    probe.update(status="error", error=f"{type(exc).__name__}: {exc}")
                probes.append(probe)
            errors = any(probe["status"] == "error" for probe in probes)
            changed = sorted({key for probe in probes for key in probe.get("changed_outputs", {})})
            status = ("fixed" if low == high else "inconclusive" if errors else
                      "effective" if changed else "locally_flat")
            rows.append({"name": name, "value": value, "bounds": [low, high],
                         "status": status, "changed_outputs": changed, "probes": probes})
    finally:
        try:
            evaluations += 1
            restored = measure(point)
        except Exception as exc:
            restoration_error = f"{type(exc).__name__}: {exc}"

    repeat_changes = changes(restored) if restored is not None else {}
    repeat_ok = restored is not None and not repeat_changes
    # A drifting builder cannot establish a reliable local zero response.
    if not repeat_ok:
        for row in rows:
            if row["status"] != "fixed":
                row["probe_status"] = row["status"]
                row["status"] = "inconclusive"
    local_flat = [row["name"] for row in rows if row["status"] == "locally_flat"]
    inconclusive = [row["name"] for row in rows if row["status"] == "inconclusive"]
    return {
        "status": "inconclusive" if inconclusive or not repeat_ok else "locally_flat" if local_flat else "effective",
        "position": point, "baseline": baseline, "knobs": rows,
        "locally_flat_knobs": local_flat, "inconclusive_knobs": inconclusive,
        "fixed_knobs": [row["name"] for row in rows if row["status"] == "fixed"],
        "evaluations": evaluations,
        "baseline_repeat": {"consistent": repeat_ok, "changed_outputs": repeat_changes,
                            "error": restoration_error},
        "tolerances": {"absolute": absolute, "relative": float(output_rtol)},
        "interpretation": "Local physical-output sensitivity only; not a global rank, physics feasibility, or optimization result.",
    }
