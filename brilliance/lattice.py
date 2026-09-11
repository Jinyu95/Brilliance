"""Canonical, unit-explicit accelerator lattice intermediate representation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .artifacts import content_hash, json_safe

LATTICE_SCHEMA_VERSION = "1.0"

SUPPORTED_ELEMENT_KINDS = {
    "MARKER", "DRIFT", "KQUAD", "QUAD", "KSEXT", "KOCT",
    "SBEND", "RBEND", "RFCA", "CORRECTOR",
}


class LatticeInputError(ValueError):
    """Invalid lattice data supplied by calling code, before a physics solver.

    This is not an unstable lattice or an infeasible matching result. Generated
    evaluators must guard their geometric proposals before constructing an IR,
    rather than catch arbitrary solver ValueErrors as infeasible particles.
    """


class LatticeTopologyError(LatticeInputError):
    """Calling code reused a compiled lattice with different physical elements.

    This is an editor/sampler lifetime error, not evidence of unstable optics.
    """


def require_same_lattice_topology(expected: Iterable[Mapping[str, Any]],
                                  proposed: Iterable[Mapping[str, Any]], *,
                                  operation: str = "JuTrack editor") -> None:
    """Check the complete element identity/order/type before mutating any field."""
    old, new = list(expected), list(proposed)
    identity = lambda row: (str(row["name"]), str(row["kind"]))
    mismatch = next((i for i, (a, b) in enumerate(zip(old, new))
                     if identity(a) != identity(b)), None)
    if len(old) == len(new) and mismatch is None:
        return
    detail = ""
    if mismatch is not None:
        detail = (f" First mismatch at element {mismatch + 1}: expected "
                  f"{identity(old[mismatch])!r}, received {identity(new[mismatch])!r}.")
    raise LatticeTopologyError(
        f"{operation} was built for {len(old)} elements, but received {len(new)}; "
        "element count, order, names, or kinds differ." + detail +
        " No fields were changed. Build a new editor/sampler for an inserted, removed, "
        "split, renamed or replaced element list. For interior optics, use "
        "api.build_jutrack_profile_sampler on the unsplit physical candidate with "
        "linear_metrics(sample_fractions={element_name: [0.5]})."
    )


def _finite(value: float, label: str) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not a physical scalar")
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LatticeInputError(f"{label} must be a finite numeric scalar") from exc
    if not math.isfinite(number):
        raise LatticeInputError(f"{label} must be finite")
    return number


def _periodicity(value: Any) -> int:
    number = _finite(value, "periodicity")
    if number < 1 or not number.is_integer():
        raise LatticeInputError("periodicity must be a positive integer")
    return int(number)


@dataclass(frozen=True, slots=True)
class LatticeElementIR:
    """One ordered element with explicit SI-unit fields."""

    name: str
    kind: str
    length_m: float
    k1_m2: float | None = None
    k2_m3: float | None = None
    k3_m4: float | None = None
    angle_rad: float | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise LatticeInputError("lattice element name must not be empty and must be a string")
        if not isinstance(self.kind, str):
            raise LatticeInputError("lattice element kind must be a string")
        kind = self.kind.upper().strip()
        if kind not in SUPPORTED_ELEMENT_KINDS:
            raise LatticeInputError(f"unsupported lattice element kind: {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        length = _finite(self.length_m, f"{self.name} length_m")
        if length < 0.0:
            raise LatticeInputError(f"{self.name} length_m must be non-negative")
        object.__setattr__(self, "length_m", length)
        if not isinstance(self.fields, Mapping):
            raise LatticeInputError("lattice element fields must be a mapping")
        for attribute in ("k1_m2", "k2_m3", "k3_m4", "angle_rad"):
            value = getattr(self, attribute)
            if value is not None:
                object.__setattr__(self, attribute, _finite(value, f"{self.name} {attribute}"))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "LatticeElementIR":
        if not isinstance(row, Mapping):
            raise LatticeInputError("lattice element row must be a mapping")
        known = {
            "name", "kind", "type", "length_m", "k1", "k1_m2", "k2", "k2_m3",
            "k3", "k3_m4", "angle", "angle_rad", "s_start_m", "s_end_m", "fields",
        }
        explicit_fields = row.get("fields", {})
        if not isinstance(explicit_fields, Mapping):
            raise LatticeInputError("lattice element fields must be a mapping")
        return cls(
            name=row.get("name", ""),
            kind=row.get("kind", row.get("type", "")),
            length_m=row.get("length_m", 0.0),
            k1_m2=row.get("k1_m2", row.get("k1")),
            k2_m3=row.get("k2_m3", row.get("k2")),
            k3_m4=row.get("k3_m4", row.get("k3")),
            angle_rad=row.get("angle_rad", row.get("angle")),
            fields={
                **dict(explicit_fields),
                **{key: value for key, value in row.items() if key not in known},
            },
        )


@dataclass(frozen=True, slots=True)
class LatticeIR:
    """Immutable lattice state passed between scientific work packages."""

    lattice_id: str
    title: str
    energy_gev: float
    periodicity: int
    elements: tuple[LatticeElementIR, ...]
    source: str
    parent_lattice_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = LATTICE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (not isinstance(self.lattice_id, str) or not self.lattice_id.strip()
                or not isinstance(self.title, str) or not self.title.strip()):
            raise LatticeInputError("lattice_id and title must not be empty strings")
        energy = _finite(self.energy_gev, "energy_gev")
        if energy <= 0.0:
            raise LatticeInputError("energy_gev must be positive")
        object.__setattr__(self, "energy_gev", energy)
        object.__setattr__(self, "periodicity", _periodicity(self.periodicity))
        if not self.elements:
            raise LatticeInputError("lattice must contain at least one element")
        if not isinstance(self.elements, (tuple, list)) or any(
                not isinstance(item, LatticeElementIR) for item in self.elements):
            raise LatticeInputError("lattice elements must contain LatticeElementIR objects")
        if not isinstance(self.metadata, Mapping):
            raise LatticeInputError("lattice metadata must be a mapping")

    @classmethod
    def create(
        cls,
        *,
        title: str,
        energy_gev: float,
        periodicity: int,
        elements: Iterable[LatticeElementIR | Mapping[str, Any]],
        source: str,
        parent_lattice_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LatticeIR":
        try:
            supplied = iter(elements)
        except TypeError as exc:
            raise LatticeInputError("lattice elements must be an iterable of element rows") from exc
        if metadata is not None and not isinstance(metadata, Mapping):
            raise LatticeInputError("lattice metadata must be a mapping")
        energy = _finite(energy_gev, "energy_gev")
        periods = _periodicity(periodicity)
        converted = tuple(
            item if isinstance(item, LatticeElementIR) else LatticeElementIR.from_mapping(item)
            for item in supplied
        )
        identity = {
            "schema_version": LATTICE_SCHEMA_VERSION,
            "title": title,
            "energy_gev": energy,
            "periodicity": periods,
            "elements": [asdict(item) for item in converted],
            "source": source,
            "parent_lattice_id": parent_lattice_id,
            "metadata": dict(metadata or {}),
        }
        return cls(
            lattice_id="lattice-" + content_hash(identity, length=16),
            title=title,
            energy_gev=energy,
            periodicity=periods,
            elements=converted,
            source=source,
            parent_lattice_id=parent_lattice_id,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_rows(
        cls,
        *,
        title: str,
        energy_gev: float,
        periodicity: int,
        rows: Iterable[Mapping[str, Any]],
        source: str,
        parent_lattice_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LatticeIR":
        try:
            supplied = iter(rows)
        except TypeError as exc:
            raise LatticeInputError("lattice rows must be an iterable of mappings") from exc
        materialized = []
        for row in supplied:
            if not isinstance(row, Mapping):
                raise LatticeInputError("lattice element row must be a mapping")
            materialized.append(dict(row))
        _validate_supplied_positions(materialized)
        return cls.create(
            title=title,
            energy_gev=energy_gev,
            periodicity=periodicity,
            elements=materialized,
            source=source,
            parent_lattice_id=parent_lattice_id,
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LatticeIR":
        """Rehydrate and integrity-check a serialized lattice artifact."""
        if not isinstance(data, Mapping):
            raise LatticeInputError("lattice artifact must be a mapping")
        missing = [name for name in ("title", "energy_gev", "periodicity", "source") if name not in data]
        if missing:
            raise LatticeInputError(f"lattice artifact is missing required fields: {', '.join(missing)}")
        rebuilt = cls.create(
            title=data["title"],
            energy_gev=data["energy_gev"],
            periodicity=data["periodicity"],
            elements=data.get("elements", ()),
            source=str(data["source"]),
            parent_lattice_id=(
                str(data["parent_lattice_id"])
                if data.get("parent_lattice_id") is not None else None
            ),
            metadata=data.get("metadata", {}),
        )
        claimed = str(data.get("lattice_id", ""))
        if claimed != rebuilt.lattice_id:
            raise ValueError(
                f"lattice artifact identity mismatch: claimed {claimed!r}, "
                f"derived {rebuilt.lattice_id!r}"
            )
        return rebuilt

    @property
    def total_length_m(self) -> float:
        return sum(item.length_m for item in self.elements)

    @property
    def bend_sum_rad(self) -> float:
        return sum(item.angle_rad or 0.0 for item in self.elements)

    @property
    def physical_fingerprint(self) -> str:
        """Hash the executable element sequence without prose provenance."""
        nonphysical_fields = {"family", "source_element", "source_period"}
        identity = {
            "energy_gev": self.energy_gev,
            "elements": [{
                "kind": item.kind,
                "length_m": item.length_m,
                "k1_m2": item.k1_m2,
                "k2_m3": item.k2_m3,
                "k3_m4": item.k3_m4,
                "angle_rad": item.angle_rad,
                "fields": {
                    key: value for key, value in item.fields.items()
                    if key not in nonphysical_fields
                },
            } for item in self.elements],
        }
        return "physical-" + content_hash(identity, length=20)

    def element_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        position = 0.0
        for item in self.elements:
            end = position + item.length_m
            row: dict[str, Any] = {
                "name": item.name,
                "type": item.kind,
                "length_m": item.length_m,
                "s_start_m": position,
                "s_end_m": end,
            }
            if item.k1_m2 is not None:
                row["k1"] = item.k1_m2
            if item.k2_m3 is not None:
                row["k2"] = item.k2_m3
            if item.k3_m4 is not None:
                row["k3"] = item.k3_m4
            if item.angle_rad is not None:
                row["angle_rad"] = item.angle_rad
            row.update(item.fields)
            rows.append(row)
            position = end
        return rows

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    def __getitem__(self, key: str) -> Any:
        """Provide read-only mapping compatibility for generated research code."""

        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


def _validate_supplied_positions(rows: list[Mapping[str, Any]]) -> None:
    expected = 0.0
    for index, row in enumerate(rows):
        if "s_start_m" in row:
            start = _finite(row["s_start_m"], f"element {index} s_start_m")
            if abs(start - expected) > 1e-9:
                raise LatticeInputError(
                    f"element {index} s_start_m={start:g} is not contiguous with {expected:g}"
                )
        length = _finite(row.get("length_m", 0.0), f"element {index} length_m")
        expected += length
        if "s_end_m" in row:
            end = _finite(row["s_end_m"], f"element {index} s_end_m")
            if abs(end - expected) > 1e-9:
                raise LatticeInputError(
                    f"element {index} s_end_m={end:g} differs from derived {expected:g}"
                )
