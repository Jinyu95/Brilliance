"""Physical-region coordinates for a complete H-MBA; no stored lattice or unit template."""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .hybrid_shape import _rows, hybrid_mba_bend_roles


def hybrid_mba_regions(elements: Sequence[Any]) -> dict[str, Any]:
    """Inventory physical bump/suppressor/core regions and GLOBAL mirror partners.

    Names and family metadata are provenance only. Region boundaries follow the
    ordered physical bends. Local TME units need not remain internally symmetric.
    A genuinely asymmetric complete-cell input is reported, never silently repaired.
    """
    rows = _rows(elements)
    roles = hybrid_mba_bend_roles(rows)
    bins = {}
    for side in ("left", "right"):
        for role, suffix in (("bump_generator", "bump"), ("dispersion_suppressor", "suppressor")):
            for index in roles["regions"][f"{side}_{suffix}"]["element_indices"]:
                bins[index] = (role, side)
    for index in roles["core_element_indices"]:
        bins[index] = ("low_H_core", "centre")
    inventory, issues = [], []
    position = 0.
    bend_map = {bend["element_index"]: bend["bend"] for bend in roles["bends"]}
    for i, row in enumerate(rows):
        j = len(rows) - 1 - i
        other = rows[j]
        differences = [key for key in ("length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4")
                       if not math.isclose(float(row.get(key) or 0), float(other.get(key) or 0),
                                           rel_tol=1e-10, abs_tol=1e-12)]
        if row["kind"] != other["kind"]:
            differences.append("kind")
        if differences and i <= j:
            issues.append({"left": row["name"], "right": other["name"], "fields": differences})
        region, side = bins.get(i, ("outer_bend" if i in bend_map else "terminal_matching",
                                   "left" if i < len(rows) / 2 else "right"))
        inventory.append({"index": i, "name": row["name"], "kind": row["kind"],
            "s_start_m": position, "s_end_m": position + row["length_m"], "region": region,
            "side": side, "bend": bend_map.get(i), "mirror_index": j, "mirror_name": other["name"]})
        position += row["length_m"]
    return {"symmetry_scope": "complete_periodic_cell", "mirror_passed": not issues,
            "mirror_issues": issues, "elements": inventory, "bend_roles": roles,
            "region_names": ["bump_generator", "dispersion_suppressor", "low_H_core", "terminal_matching", "outer_bend"]}


class HybridMBARegionalGeometry:
    """Three structural coordinates with explicit, conservative length funding.

    Coordinates: full inter-bend bump length, full suppressor length (one side),
    and B1/(B1+B2) angle fraction. Region drift lengths and four outer bend
    angles/lengths change; core magnets/geometry, all K1, and ID straights stay untouched.
    The angle transfer carries a length transfer at the pair's mean radius,
    preserving total pair length and source identity (not increasing field in
    an initially equal-curvature pair already at its dipole field limit).
    An explicit list of terminal drift reservoirs funds the two length changes;
    every named reservoir is paired across the complete cell. Edge ID drifts
    and core/bump/suppressor members cannot be reservoirs. Bounds are supplied
    by the scientist/agent, not inferred from a facility or TME unit template.
    """
    variable_names = ("bump_length_m", "suppressor_length_m", "outer_bend_fraction")

    def __init__(self, elements: Sequence[Any], *, bounds: Mapping[str, Sequence[float]],
                 reservoir_element_names: Sequence[str], minimum_drift_length_m: float = 0.01):
        self.rows = _rows(elements)
        self.inventory = hybrid_mba_regions(self.rows)
        if not self.inventory["mirror_passed"]:
            raise ValueError(f"Complete-cell mirror mismatch: {self.inventory['mirror_issues']}")
        self.roles = self.inventory["bend_roles"]
        self.minimum = float(minimum_drift_length_m)
        if not math.isfinite(self.minimum) or self.minimum <= 0:
            raise ValueError("minimum_drift_length_m must be positive and finite")
        self.groups = {key: self.roles["regions"][f"left_{suffix}"]["element_indices"]
                       for key, suffix in (("bump_length_m", "bump"), ("suppressor_length_m", "suppressor"))}
        lookup = {row["name"]: i for i, row in enumerate(self.rows)}
        if isinstance(reservoir_element_names, str) or not reservoir_element_names:
            raise ValueError("Declare terminal drift reservoirs; cell length cannot be created for free")
        reservoirs = set()
        for name in reservoir_element_names:
            if name not in lookup:
                raise ValueError(f"Unknown reservoir {name!r}; select actual names from hybrid_mba_regions")
            i = lookup[name]
            i = min(i, len(self.rows) - 1 - i)
            if (i == 0 or self.rows[i]["kind"] != "DRIFT"
                    or self.inventory["elements"][i]["region"] != "terminal_matching"):
                raise ValueError(f"Reservoir {name!r} must be a non-ID terminal drift outside the bump/suppressor/core")
            reservoirs.add(i)
        self.reservoirs = sorted(reservoirs)
        self.base_lengths = {key: sum(self.rows[i]["length_m"] for i in indices)
                             for key, indices in self.groups.items()}
        self.reservoir_total = sum(self.rows[i]["length_m"] for i in self.reservoirs)
        bends = self.roles["bends"]
        self.angle_sum = bends[0]["angle_rad"] + bends[1]["angle_rad"]
        self.initial_position = [self.base_lengths[key] for key in self.variable_names[:2]] + [bends[0]["angle_rad"] / self.angle_sum]
        if set(bounds) != set(self.variable_names):
            raise ValueError(f"bounds must define exactly {self.variable_names}")
        self.bounds = [list(map(float, bounds[key])) for key in self.variable_names]
        for key, (lo, hi), initial in zip(self.variable_names, self.bounds, self.initial_position):
            if not all(math.isfinite(v) for v in (lo, hi)) or not lo < hi or not lo <= initial <= hi:
                raise ValueError(f"{key}: finite ordered bounds must contain the source coordinate")
        if not 0 < self.bounds[2][0] < self.bounds[2][1] < 1:
            raise ValueError("outer_bend_fraction bounds must stay strictly inside (0,1)")

    def apply(self, position: Sequence[float]) -> list[dict[str, Any]]:
        if len(position) != 3:
            raise ValueError("Grouped geometry has exactly three independent coordinates")
        values = list(map(float, position))
        if any(not math.isfinite(v) or not lo <= v <= hi for v, (lo, hi) in zip(values, self.bounds)):
            raise ValueError("Grouped geometry coordinates outside declared bounds")
        rows = deepcopy(self.rows)

        def allocate(indices, total):
            drifts = [i for i in indices if self.rows[i]["kind"] == "DRIFT"]
            fixed = sum(self.rows[i]["length_m"] for i in indices if i not in drifts)
            spare = total - fixed - len(drifts) * self.minimum
            weights = [self.rows[i]["length_m"] - self.minimum for i in drifts]
            if not drifts or min(weights) < 0 or sum(weights) <= 0 or spare < -1e-12:
                raise ValueError("Grouped geometry has insufficient drift budget at the declared physical minimum")
            for i, weight in zip(drifts, weights):
                length = self.minimum + max(0., spare) * weight / sum(weights)
                rows[i]["length_m"] = rows[len(rows) - 1 - i]["length_m"] = length

        for key, target in zip(self.variable_names[:2], values[:2]):
            allocate(self.groups[key], target)
        delta = sum(values[:2]) - sum(self.base_lengths.values())
        allocate(self.reservoirs, self.reservoir_total - delta)
        pair = self.roles["bends"][:2]
        radius = sum(bend["length_m"] for bend in pair) / self.angle_sum
        for bend, theta in zip(pair, (values[2] * self.angle_sum, (1 - values[2]) * self.angle_sum)):
            i = bend["element_index"]
            length = bend["length_m"] + radius * (theta - bend["angle_rad"])
            if length <= 0:
                raise ValueError("Outer angle transfer would produce a nonpositive bend length")
            rows[i]["angle_rad"] = rows[len(rows) - 1 - i]["angle_rad"] = theta
            rows[i]["length_m"] = rows[len(rows) - 1 - i]["length_m"] = length
        return rows

    def describe(self, *, minimum_population_per_knob: int = 10) -> dict[str, Any]:
        if type(minimum_population_per_knob) is not int or minimum_population_per_knob < 1:
            raise ValueError("minimum_population_per_knob must be a positive integer")
        return {"variable_names": list(self.variable_names), "initial_position": self.initial_position,
                "bounds": self.bounds, "outer_knob_count": 3,
                "minimum_population": 3 * minimum_population_per_knob,
                "symmetry_scope": "complete_periodic_cell", "reservoirs": [self.rows[i]["name"] for i in self.reservoirs],
                "regions": {key: [self.rows[i]["name"] for i in indices] for key, indices in self.groups.items()},
                "invariants": ["cell_length", "cell_angle", "core_geometry", "all_K1", "ID_edge_drifts"],
                "required_trial_checks": ["magnet_fields", "achromat_match", "stability", "core_radiation"]}
