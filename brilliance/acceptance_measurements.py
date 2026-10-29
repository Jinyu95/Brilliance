"""Connected acceptance from supplied survival masks, without tracking or targets.

These reducers measure sampled sets, not continuous apertures or injection capture.
Use the same reducer for a short optimization screen and longer validation, while
reporting their different turn counts and grids separately.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _grid(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional grid with at least three samples")
    if len(np.unique(array)) != array.size or np.sum(array == 0.0) != 1:
        raise ValueError(f"{name} must contain distinct coordinates and an exact measured zero")
    if not (np.any(array < 0) and np.any(array > 0)):
        raise ValueError(f"{name} must sample both signs around zero")
    return array


def _mask(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind not in "biuf" or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be a boolean or 0/1 array with shape {shape}")
    return array.astype(bool, copy=True)


def connected_momentum_acceptance(
    delta_grid: Sequence[float],
    survived: Any,
    *,
    location_names: Sequence[str] | None = None,
    target_interval: Sequence[float] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Reduce a (locations, delta) survival mask to zero-connected intervals.

    For standard local MA, supply tracks launched on the on-momentum closed orbit
    with a momentum kick (zero transverse coordinates in an ideal centred ring),
    not recentered on each off-momentum orbit. Finite-amplitude/all-phase survival
    is a separate measurement, not required by this reducer. It does not track or
    infer launch conditions: record them alongside the mask. A map/orbit failure
    is not a tracked loss and must not be silently encoded as False.

    Delta is fractional momentum offset (0.02 means 2%). Include an independently
    tracked zero at EVERY location; lost origins yield empty connected sets.
    Coordinates may be unordered. Each sign stops at its first lost sample,
    even when particles at larger offsets survive. A surviving scan endpoint
    is a censored lower bound, not a measured aperture boundary.

    Optional target_interval=[negative, positive] requires BOTH reaches at EVERY
    location. No average-survival compensation is used. ``strict=True`` requires
    sampled reaches strictly beyond the targets. Unobserved or bracketed targets
    are not demonstrated, distinct from a measured loss within the target domain.
    No target means request_satisfied=None. This does not certify survival between
    sampled offsets, over longer tracking, or with different launch conditions.
    """
    grid = _grid(delta_grid, "delta_grid")
    if not isinstance(strict, (bool, np.bool_)):
        raise ValueError("strict must be a boolean")
    supplied = np.asarray(survived)
    if supplied.ndim != 2 or supplied.shape[0] == 0:
        raise ValueError("survived must contain one row per sampled location")
    alive = _mask(supplied, (supplied.shape[0], grid.size), "survived")
    count = alive.shape[0]
    names = list(location_names) if location_names is not None else [str(i) for i in range(count)]
    if isinstance(location_names, (str, bytes)) or len(names) != count or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != count:
        raise ValueError("location_names must contain one distinct nonempty name per location")
    targets = None
    if target_interval is not None:
        target = np.asarray(target_interval, dtype=float)
        if target.shape != (2,) or not np.all(np.isfinite(target)) or not target[0] < 0 < target[1]:
            raise ValueError("target_interval must be a finite [negative, positive] pair in fractional delta")
        targets = {"negative": float(-target[0]), "positive": float(target[1])}
    origin = int(np.flatnonzero(grid == 0)[0])
    connected = np.zeros_like(alive)
    records, checks = [], []
    for row, name in enumerate(names):
        origin_alive = bool(alive[row, origin])
        connected[row, origin] = origin_alive
        result: dict[str, Any] = {"location": name, "origin_survived": origin_alive}
        for sign, label in ((-1, "negative"), (1, "positive")):
            indices = np.flatnonzero(grid * sign > 0)
            indices = indices[np.argsort(np.abs(grid[indices]))]
            reach, first_loss = 0.0, None
            if origin_alive:
                for index in indices:
                    if not alive[row, index]:
                        first_loss = float(grid[index])
                        break
                    connected[row, index] = True
                    reach = float(grid[index])
            side = {
                "reach_delta": reach, "first_lost_delta": first_loss,
                "scan_edge_censored": bool(origin_alive and first_loss is None),
                "disconnected_survivors": int(np.sum(alive[row, indices] & ~connected[row, indices])),
            }
            result[label] = side
            if targets is not None:
                required = targets[label]
                demonstrated = origin_alive and (abs(reach) > required if strict else abs(reach) >= required)
                observed_failure = not origin_alive or (first_loss is not None and abs(first_loss) <= required)
                status = "passed" if demonstrated else "failed" if observed_failure else "not_demonstrated"
                reason = ("sampled_reach" if demonstrated else "origin_lost" if not origin_alive else
                          "sampled_loss_within_target" if observed_failure else
                          "scan_too_narrow" if first_loss is None else "resolution_insufficient")
                checks.append({"location": name, "side": label, "target_delta": sign * required,
                    "satisfied": bool(demonstrated), "status": status, "reason": reason,
                    "shortfall_delta": max(0.0, required - abs(reach)),
                    "strict_boundary_not_exceeded": bool(strict and abs(reach) == required)})
        records.append(result)
    return {
        "schema": "connected_momentum_acceptance/v1", "delta_units": "fractional momentum offset",
        "locations": records, "delta_grid": grid.tolist(), "connected": connected.tolist(),
        "worst_sampled_interval": [max(r["negative"]["reach_delta"] for r in records),
                                   min(r["positive"]["reach_delta"] for r in records)],
        "origin_lost_locations": [r["location"] for r in records if not r["origin_survived"]],
        "target_interval": list(map(float, target_interval)) if targets is not None else None,
        "strict": bool(strict), "target_checks": checks,
        "request_satisfied": all(check["satisfied"] for check in checks) if targets is not None else None,
        "limitations": ["Contiguity is on the supplied samples; survival between samples is unproven.",
                        "Scan-edge reaches are censored lower bounds, not aperture boundaries.",
                        "Turns, launch conditions and candidate identity must accompany this report."],
    }


def connected_transverse_acceptance(
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    survived: Any,
    *,
    target_mask: Any | None = None,
) -> dict[str, Any]:
    """Measure the four-neighbour origin-connected component of an x-y grid.

    Coordinates are metres and survived has shape (len(y_grid), len(x_grid)).
    Both grids must be increasing, contain exact zero and sample both signs.
    Area is tensor-product trapezoidal quadrature of the binary nodal connected
    mask, not a proven continuous aperture. A component touching a scan boundary
    is censored there. Axis intercepts alone do not establish injection capture.

    If supplied, target_mask explicitly selects required grid samples. Every
    selected sample must be in the origin-connected component; islands do not
    count. Without a target, request_satisfied=None. No ellipse is invented.
    """
    from scipy.ndimage import label

    x, y = _grid(x_grid, "x_grid"), _grid(y_grid, "y_grid")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("x_grid and y_grid must be strictly increasing")
    alive = _mask(survived, (y.size, x.size), "survived")
    ix, iy = int(np.flatnonzero(x == 0)[0]), int(np.flatnonzero(y == 0)[0])
    labels, _ = label(alive, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    origin = int(labels[iy, ix])
    connected = labels == origin if origin else np.zeros_like(alive)

    def weights(grid):
        result = np.empty(grid.size)
        result[0], result[-1] = (grid[1] - grid[0]) / 2, (grid[-1] - grid[-2]) / 2
        result[1:-1] = (grid[2:] - grid[:-2]) / 2
        return result

    with np.errstate(over="ignore", invalid="ignore"):
        sampled_area = float(np.sum(connected * weights(y)[:, None] * weights(x)[None, :]))
    if not np.isfinite(sampled_area):
        raise ValueError("Grid spacing or sampled area is not representable; check coordinate units")
    targets = None if target_mask is None else _mask(target_mask, alive.shape, "target_mask")
    if targets is not None and not np.any(targets):
        raise ValueError("target_mask must select at least one measured sample")
    missing = None if targets is None else targets & ~connected
    return {
        "schema": "connected_transverse_acceptance/v1", "coordinate_units": "m",
        "x_grid": x.tolist(), "y_grid": y.tolist(), "origin_survived": bool(origin),
        "connected": connected.tolist(), "connected_sample_count": int(np.sum(connected)),
        "disconnected_survivor_count": int(np.sum(alive & ~connected)),
        "sampled_connected_area_m2": sampled_area,
        "area_convention": "tensor-product trapezoidal quadrature of binary nodal mask; not continuous aperture area",
        "scan_edge_censored": {"x_negative": bool(np.any(connected[:, 0])),
            "x_positive": bool(np.any(connected[:, -1])), "y_negative": bool(np.any(connected[0, :])),
            "y_positive": bool(np.any(connected[-1, :]))},
        "target_sample_count": None if targets is None else int(np.sum(targets)),
        "missing_target_samples": None if missing is None else np.argwhere(missing).tolist(),
        "target_missing_fraction": None if missing is None else float(np.sum(missing) / np.sum(targets)),
        "request_satisfied": None if missing is None else not bool(np.any(missing)),
        "limitations": ["Four-neighbour connectivity and area are sampled-grid diagnostics, not injection capture.",
                        "Turns, launch momenta, errors, apertures and candidate identity must accompany this report."],
    }
