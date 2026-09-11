"""Topology-preserving lattice edits for programmable JuTrack research.

Insertion is expressed as replacement of one physical drift by
``left drift + inserted element + right drift``.  Removal performs the exact
inverse when the provenance tags are still present.  These operations never
change the longitudinal length of the lattice silently.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .artifacts import json_safe
from .lattice import LatticeElementIR, LatticeIR


def _row(value: LatticeElementIR | Mapping[str, Any]) -> dict[str, Any]:
    item = value if isinstance(value, LatticeElementIR) else LatticeElementIR.from_mapping(value)
    return json_safe(asdict(item))


def _materialize(
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [_row(item) for item in elements]
    names = [str(row["name"]) for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("lattice element names must be unique: " + ", ".join(duplicates))
    return rows


def insert_element_in_drift(
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    drift_name: str,
    inserted_element: LatticeElementIR | Mapping[str, Any],
    split_ratio: float = 0.5,
    minimum_drift_segment_m: float = 0.0,
) -> list[dict[str, Any]]:
    """Insert one element into a named drift while preserving total length.

    ``split_ratio`` applies to the drift length remaining after the inserted
    element length is removed.  A ratio of 0.5 therefore leaves equal drift
    segments on both sides, independent of the inserted magnet length.
    """

    rows = _materialize(elements)
    if not math.isfinite(float(split_ratio)) or not 0.0 <= split_ratio <= 1.0:
        raise ValueError("split_ratio must be between zero and one")
    if not math.isfinite(float(minimum_drift_segment_m)) or minimum_drift_segment_m < 0.0:
        raise ValueError("minimum_drift_segment_m must be finite and non-negative")
    matches = [index for index, row in enumerate(rows) if row["name"] == drift_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one drift named {drift_name!r}; found {len(matches)}")
    index = matches[0]
    drift = rows[index]
    if drift["kind"] != "DRIFT":
        raise ValueError(f"element {drift_name!r} is not a drift")
    inserted = _row(inserted_element)
    if inserted["name"] in {row["name"] for row in rows}:
        raise ValueError(f"inserted element name already exists: {inserted['name']!r}")
    drift_length = float(drift["length_m"])
    inserted_length = float(inserted["length_m"])
    remaining = drift_length - inserted_length
    if remaining < -1.0e-12:
        raise ValueError(
            f"inserted element length {inserted_length:g} m exceeds drift "
            f"{drift_name!r} length {drift_length:g} m"
        )
    remaining = max(0.0, remaining)
    left_length = remaining * float(split_ratio)
    right_length = remaining - left_length
    if min(left_length, right_length) + 1.0e-15 < minimum_drift_segment_m:
        raise ValueError(
            "insertion would leave a drift segment below minimum_drift_segment_m"
        )

    original_fields = dict(drift.get("fields", {}))
    edit_id = f"insert:{drift_name}:{inserted['name']}"

    def segment(side: str, length: float) -> dict[str, Any]:
        return {
            **drift,
            "name": f"{drift_name}:{side}",
            "length_m": length,
            "fields": {
                **original_fields,
                "lattice_edit": edit_id,
                "split_parent_drift": drift_name,
                "split_side": side.lower(),
            },
        }

    inserted = {
        **inserted,
        "fields": {
            **dict(inserted.get("fields", {})),
            "lattice_edit": edit_id,
            "inserted_in_drift": drift_name,
            "split_ratio": float(split_ratio),
        },
    }
    replacement = [segment("LEFT", left_length), inserted, segment("RIGHT", right_length)]
    edited = [*rows[:index], *replacement, *rows[index + 1 :]]
    before = sum(float(row["length_m"]) for row in rows)
    after = sum(float(row["length_m"]) for row in edited)
    if abs(before - after) > 1.0e-12:
        raise RuntimeError("drift insertion changed total lattice length")
    return edited


def insert_element_between(
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    left_element_name: str,
    right_element_name: str,
    inserted_element: LatticeElementIR | Mapping[str, Any],
    split_ratio: float = 0.5,
    minimum_drift_segment_m: float = 0.0,
) -> list[dict[str, Any]]:
    """Insert into the unique drift directly between two named elements."""

    rows = _materialize(elements)
    left = [index for index, row in enumerate(rows) if row["name"] == left_element_name]
    right = [index for index, row in enumerate(rows) if row["name"] == right_element_name]
    if len(left) != 1 or len(right) != 1:
        raise ValueError("left and right element names must each identify one physical element")
    if right[0] != left[0] + 2 or rows[left[0] + 1]["kind"] != "DRIFT":
        raise ValueError("the named elements do not enclose exactly one drift")
    return insert_element_in_drift(
        rows,
        drift_name=str(rows[left[0] + 1]["name"]),
        inserted_element=inserted_element,
        split_ratio=split_ratio,
        minimum_drift_segment_m=minimum_drift_segment_m,
    )


def remove_inserted_element(
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    element_name: str,
) -> list[dict[str, Any]]:
    """Remove an inserted element and reconstruct its original parent drift."""

    rows = _materialize(elements)
    matches = [index for index, row in enumerate(rows) if row["name"] == element_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one inserted element named {element_name!r}")
    index = matches[0]
    if index == 0 or index == len(rows) - 1:
        raise ValueError("inserted element must have a drift segment on each side")
    inserted = rows[index]
    left, right = rows[index - 1], rows[index + 1]
    fields = dict(inserted.get("fields", {}))
    parent = str(fields.get("inserted_in_drift", ""))
    edit_id = str(fields.get("lattice_edit", ""))
    if not parent or not edit_id:
        raise ValueError("element does not carry reversible drift-insertion provenance")
    for segment, side in ((left, "left"), (right, "right")):
        segment_fields = dict(segment.get("fields", {}))
        if (
            segment.get("kind") != "DRIFT"
            or segment_fields.get("split_parent_drift") != parent
            or segment_fields.get("split_side") != side
            or segment_fields.get("lattice_edit") != edit_id
        ):
            raise ValueError("neighboring drifts do not match the insertion provenance")
    restored_fields = {
        key: value
        for key, value in dict(left.get("fields", {})).items()
        if key not in {"lattice_edit", "split_parent_drift", "split_side"}
    }
    restored = {
        **left,
        "name": parent,
        "length_m": (
            float(left["length_m"])
            + float(inserted["length_m"])
            + float(right["length_m"])
        ),
        "fields": restored_fields,
    }
    edited = [*rows[: index - 1], restored, *rows[index + 2 :]]
    before = sum(float(row["length_m"]) for row in rows)
    after = sum(float(row["length_m"]) for row in edited)
    if abs(before - after) > 1.0e-12:
        raise RuntimeError("element removal changed total lattice length")
    return edited


def resize_inserted_element(
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    element_name: str,
    new_length_m: float,
    minimum_drift_segment_m: float = 0.0,
    preserve_integrated_strength: bool = True,
) -> list[dict[str, Any]]:
    """Resize an inserted magnet while preserving its host-drift length."""

    rows = _materialize(elements)
    matches = [index for index, row in enumerate(rows) if row["name"] == element_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one inserted element named {element_name!r}")
    index = matches[0]
    if index == 0 or index == len(rows) - 1:
        raise ValueError("inserted element must have a drift segment on each side")
    inserted, left, right = rows[index], rows[index - 1], rows[index + 1]
    fields = dict(inserted.get("fields", {}))
    edit_id = str(fields.get("lattice_edit", ""))
    parent = str(fields.get("inserted_in_drift", ""))
    if not edit_id or not parent:
        raise ValueError("element does not carry reversible drift-insertion provenance")
    for segment in (left, right):
        segment_fields = dict(segment.get("fields", {}))
        if segment.get("kind") != "DRIFT" or segment_fields.get("lattice_edit") != edit_id:
            raise ValueError("neighboring drift segments do not match the insertion")
    new_length = float(new_length_m)
    minimum = float(minimum_drift_segment_m)
    if not math.isfinite(new_length) or new_length <= 0.0:
        raise ValueError("new_length_m must be finite and positive")
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_drift_segment_m must be finite and non-negative")
    host_length = float(left["length_m"]) + float(inserted["length_m"]) + float(right["length_m"])
    remaining = host_length - new_length
    if remaining + 1.0e-15 < 2.0 * minimum:
        raise ValueError("resized element would violate the minimum neighboring drift length")
    ratio = float(fields.get("split_ratio", 0.5))
    left["length_m"] = remaining * ratio
    right["length_m"] = remaining - float(left["length_m"])
    old_length = float(inserted["length_m"])
    inserted["length_m"] = new_length
    if preserve_integrated_strength and old_length > 0.0:
        scale = old_length / new_length
        for key in ("k1_m2", "k2_m3", "k3_m4"):
            if inserted.get(key) is not None:
                inserted[key] = float(inserted[key]) * scale
    before = sum(float(row["length_m"]) for row in rows)
    rows[index - 1 : index + 2] = [left, inserted, right]
    after = sum(float(row["length_m"]) for row in rows)
    if abs(before - after) > 1.0e-12:
        raise RuntimeError("inserted-element resize changed total lattice length")
    return rows


def replace_lattice_elements(
    lattice: LatticeIR,
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    title: str,
    source: str,
    metadata: Mapping[str, Any] | None = None,
) -> LatticeIR:
    """Create a traceable lattice revision from a topology edit."""

    return LatticeIR.create(
        title=title,
        energy_gev=lattice.energy_gev,
        periodicity=lattice.periodicity,
        elements=_materialize(elements),
        source=source,
        parent_lattice_id=lattice.lattice_id,
        metadata={**dict(lattice.metadata), **dict(metadata or {})},
    )
