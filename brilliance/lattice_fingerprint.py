"""Stable identity of represented physical cell fields, independent of expertise."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Sequence

from .lattice import LatticeElementIR


def cell_fingerprint(elements: Sequence[Any]) -> str:
    """Identify ordered physical scalar fields without interpretation metadata.

    This is a supplementary profile-to-cell identity, not a substitute for the
    full archived lattice identity (which also includes energy and other fields).
    """
    rows = [asdict(item if isinstance(item, LatticeElementIR)
                   else LatticeElementIR.from_mapping(item)) for item in elements]
    if not rows or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Use one nonempty physical cell with unique element names.")
    keys = ("name", "kind", "length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4")
    physical = [{key: row.get(key) for key in keys} for row in rows]
    return hashlib.sha256(json.dumps(physical, sort_keys=True, allow_nan=False).encode()).hexdigest()
