"""MBA analytical initialization references, not simulated or stored lattices."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from brilliance.artifacts import json_safe
from brilliance.lattice import LatticeIR


@dataclass(frozen=True, slots=True)
class TMECellTargets:
    """Homogeneous-bend TME scale targets for initializing a JuTrack match."""

    energy_gev: float
    bend_length_m: float
    bend_angle_rad: float
    beta_x_center_m: float
    alpha_x_center: float
    dispersion_x_center_m: float
    dispersion_px_center: float
    minimum_emittance_m: float
    minimum_emittance_pm: float
    model: str = "homogeneous uncoupled TME, Jx=1"

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


def tme_targets(
    *,
    energy_gev: float,
    bend_length_m: float,
    bend_angle_rad: float,
    damping_partition_x: float = 1.0,
) -> TMECellTargets:
    """Return analytic TME center conditions as scales, never as a JuTrack result.

    The homogeneous-dipole formulas initialize a from-scratch match. Finite edge
    focusing and the assembled lattice must be evaluated independently with
    JuTrack.
    """

    values = (energy_gev, bend_length_m, bend_angle_rad, damping_partition_x)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("TME inputs must be finite")
    if energy_gev <= 0.0 or bend_length_m <= 0.0 or bend_angle_rad <= 0.0:
        raise ValueError("energy, bend length, and bend angle must be positive")
    if damping_partition_x <= 0.0:
        raise ValueError("damping_partition_x must be positive")
    gamma = energy_gev * 1.0e9 / 0.510998950e6
    cq_m = 3.8319e-13
    emittance_m = (
        cq_m * gamma * gamma * bend_angle_rad**3 / (12.0 * math.sqrt(15.0) * damping_partition_x)
    )
    return TMECellTargets(
        energy_gev=float(energy_gev),
        bend_length_m=float(bend_length_m),
        bend_angle_rad=float(bend_angle_rad),
        beta_x_center_m=bend_length_m / (2.0 * math.sqrt(15.0)),
        alpha_x_center=0.0,
        dispersion_x_center_m=bend_length_m * bend_angle_rad / 24.0,
        dispersion_px_center=0.0,
        minimum_emittance_m=emittance_m,
        minimum_emittance_pm=emittance_m * 1.0e12,
    )


def homogeneous_bend_tme_reference(
    candidate: LatticeIR | Path | str,
    *,
    damping_partition_x: float = 1.0,
) -> dict[str, Any]:
    """Return a geometry-weighted homogeneous-bend TME normalization scale.

    Unequal bend angles and lengths are combined through their ideal independent
    contributions, ``I2_j = theta_j**2/L_j`` and
    ``I5_j = abs(theta_j)**5/(12*sqrt(15)*L_j)``. This is a diagnostic
    reference, not a claim that one complete lattice can simultaneously attain
    every bend's isolated optimum.
    """

    from brilliance.accelerator_api import load_lattice

    lattice = load_lattice(candidate) if isinstance(candidate, (Path, str)) else candidate
    if not math.isfinite(float(damping_partition_x)) or damping_partition_x <= 0.0:
        raise ValueError("damping_partition_x must be a positive finite number")
    grouped: dict[str, dict[str, Any]] = {}
    total_i2 = 0.0
    total_i5 = 0.0
    divisor = 12.0 * math.sqrt(15.0)
    for index, item in enumerate(lattice.elements):
        angle = float(item.angle_rad or 0.0)
        if item.kind not in {"SBEND", "RBEND"} or abs(angle) <= 0.0:
            continue
        if item.length_m <= 0.0:
            raise ValueError(f"bending magnet {item.name!r} must have positive length")
        i2 = angle * angle / item.length_m
        i5 = abs(angle) ** 5 / (divisor * item.length_m)
        total_i2 += i2
        total_i5 += i5
        source_name = str(item.fields.get("source_element") or item.name)
        group = grouped.setdefault(
            source_name,
            {
                "bend": source_name,
                "first_element_index": index,
                "instances": 0,
                "length_sum_m": 0.0,
                "angle_sum_rad": 0.0,
                "I2_reference": 0.0,
                "I5_reference": 0.0,
            },
        )
        group["instances"] += 1
        group["length_sum_m"] += item.length_m
        group["angle_sum_rad"] += angle
        group["I2_reference"] += i2
        group["I5_reference"] += i5
    if not grouped or total_i2 <= 0.0:
        raise ValueError("candidate contains no finite nonzero-angle bending magnets")

    gamma = lattice.energy_gev * 1.0e9 / 0.510998950e6
    cq_m = 3.8319e-13
    emittance_m = (
        cq_m * gamma * gamma * total_i5 / (damping_partition_x * total_i2)
    )
    rows = []
    for group in sorted(grouped.values(), key=lambda row: row["first_element_index"]):
        instances = int(group["instances"])
        length = float(group["length_sum_m"]) / instances
        angle = float(group["angle_sum_rad"]) / instances
        rows.append(
            {
                "bend": str(group["bend"]),
                "instances": instances,
                "length_m": length,
                "angle_rad": angle,
                "beta_x_center_reference_m": length / (2.0 * math.sqrt(15.0)),
                "dispersion_x_center_reference_m": length * angle / 24.0,
                "mean_H_reference_m": length * angle * angle / divisor,
                "I2_reference": float(group["I2_reference"]),
                "I5_reference": float(group["I5_reference"]),
            }
        )
    return json_safe(
        {
            "model": "independent homogeneous uncoupled TME bends",
            "lattice_id": lattice.lattice_id,
            "damping_partition_x": float(damping_partition_x),
            "I2_reference": total_i2,
            "I5_reference": total_i5,
            "reference_emittance_m": emittance_m,
            "reference_emittance_pm": emittance_m * 1.0e12,
            "rows": rows,
            "interpretation": (
                "Geometry-weighted normalization scale only; the complete lattice must be "
                "matched and evaluated independently with JuTrack."
            ),
        }
    )
