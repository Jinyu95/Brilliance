"""Stable typed adapter between :mod:`brilliance.lattice` and pyJuTrack.

Research agents never call pyJuTrack constructors directly on trusted paths.
This adapter is the single versioned API boundary and is covered with contract
tests independently of Julia availability.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .lattice import LatticeElementIR, LatticeInputError, LatticeIR

JUTRACK_SERVICE_VERSION = "1.3.0"

# Exact constructor options in the configured JuTrack elements.jl. Metadata
# stays metadata; declared physical options must never be silently discarded.
_TRANSFORMS = {"T1", "T2", "R1", "R2", "RApertures", "EApertures"}
_INTEGRATION = {"NumIntSteps", "MaxOrder", "rad", "PolynomA", "KickAngle",
                "FringeQuadEntrance", "FringeQuadExit"}
_BEND = _TRANSFORMS | _INTEGRATION | {"e1", "e2", "fint1", "fint2", "gap", "PolynomB",
    "FringeBendEntrance", "FringeBendExit", "FringeIntM0", "FringeIntP0"}
_OPTIONS = {"SBEND": _BEND, "RBEND": _BEND - {"e1", "e2"},
            "KQUAD": _TRANSFORMS | _INTEGRATION, "KSEXT": _TRANSFORMS | _INTEGRATION,
            "KOCT": _TRANSFORMS | _INTEGRATION, "QUAD": _TRANSFORMS | {"rad"},
            "DRIFT": _TRANSFORMS, "CORRECTOR": _TRANSFORMS,
            "RFCA": {"RApertures", "EApertures"}, "MARKER": set()}
_ALIASES = {key.casefold().replace("_", ""): key for key in _BEND}
_ALIASES.update(entranceangle="e1", exitangle="e2")
_UNSUPPORTED = {"fint", "fullgap", "hgap"}  # Ambiguous conventions, not aliases.
_INTEGER = _INTEGRATION - {"PolynomA", "KickAngle"} | {"FringeBendEntrance", "FringeBendExit"}
_ARRAY_SHAPES = {"T1": (6,), "T2": (6,), "R1": (6, 6), "R2": (6, 6),
    "RApertures": (6,), "EApertures": (6,), "KickAngle": (2,),
    "FringeIntM0": (5,), "FringeIntP0": (5,)}


def native_element_options(item: LatticeElementIR) -> dict[str, Any]:
    """Validate explicitly saved constructor settings; never invent defaults."""
    options: dict[str, Any] = {}
    for key, value in item.fields.items():
        normalized = key.casefold().replace("_", "")
        if normalized in _UNSUPPORTED:
            raise LatticeInputError(f"{item.name}: unsupported physical field {key!r}; use exact JuTrack fields")
        name = _ALIASES.get(normalized)
        if name is None:
            continue
        if name not in _OPTIONS.get(item.kind, set()):
            raise LatticeInputError(f"{item.name}: {item.kind} does not support explicit {name}; do not discard it")
        if name in _ARRAY_SHAPES or name in {"PolynomA", "PolynomB"}:
            array = np.asarray(value, dtype=float)
            valid_shape = (array.ndim == 1 and len(array) >= 4) if name.startswith("Polynom") else array.shape == _ARRAY_SHAPES[name]
            if not valid_shape or not np.isfinite(array).all():
                raise LatticeInputError(f"{item.name}.{name}: invalid native array shape or nonfinite values")
            converted = array.tolist()
        else:
            if isinstance(value, bool) or value is None:
                raise LatticeInputError(f"{item.name}.{name} must be an explicit finite number")
            converted = float(value)
            if not math.isfinite(converted):
                raise LatticeInputError(f"{item.name}.{name} must be finite")
            if name in _INTEGER:
                if not converted.is_integer() or converted < (1 if name == "NumIntSteps" else 0):
                    raise LatticeInputError(f"{item.name}.{name} must be a valid nonnegative integer (NumIntSteps >= 1)")
                converted = int(converted)
        if name in options and options[name] != converted:
            raise LatticeInputError(f"{item.name}: conflicting aliases for {name}")
        options[name] = converted
    return options


class JuTrackService:
    def __init__(self, module: Any):
        self.jt = module

    def build_lattice(self, lattice: LatticeIR) -> Any:
        elements = [self.build_element(item) for item in lattice.elements]
        return self.jt.Lattice(elements)

    def build_element(self, item: LatticeElementIR) -> Any:
        jt = self.jt
        kind = item.kind
        options = native_element_options(item)
        if kind == "MARKER":
            return jt.MARKER(item.name)
        if kind == "DRIFT":
            return jt.DRIFT(item.name, item.length_m, **options)
        if kind in {"KQUAD", "QUAD"}:
            if item.k1_m2 is None:
                raise ValueError(f"{kind} element {item.name!r} requires k1_m2")
            return getattr(jt, kind)(item.name, item.length_m, k1=item.k1_m2, **options)
        if kind == "KSEXT":
            if item.k2_m3 is None:
                raise ValueError(f"KSEXT element {item.name!r} requires k2_m3")
            return jt.KSEXT(item.name, item.length_m, k2=item.k2_m3, **options)
        if kind == "KOCT":
            if item.k3_m4 is None:
                raise ValueError(f"KOCT element {item.name!r} requires k3_m4")
            return jt.KOCT(item.name, item.length_m, k3=item.k3_m4, **options)
        if kind in {"SBEND", "RBEND"}:
            if item.angle_rad is None:
                raise ValueError(f"{kind} element {item.name!r} requires angle_rad")
            kwargs: dict[str, Any] = options
            if item.k1_m2 is not None:
                # Keep old archived lattice files reproducible through the
                # normal-multipole representation. New MBA construction rejects this.
                if "PolynomB" in kwargs and kwargs["PolynomB"][1] != item.k1_m2:
                    raise LatticeInputError(f"{item.name}: k1_m2 conflicts with saved PolynomB[1]")
                kwargs.setdefault("PolynomB", [0.0, item.k1_m2, 0.0, 0.0])
                kwargs["MaxOrder"] = max(int(kwargs.get("MaxOrder", 0)), 1)
            return getattr(jt, kind)(item.name, item.length_m, item.angle_rad, **kwargs)
        if kind == "RFCA":
            voltage = item.fields.get("voltage_v", item.fields.get("volt"))
            frequency = item.fields.get("frequency_hz", item.fields.get("freq"))
            if voltage is None or frequency is None:
                raise ValueError(f"RFCA element {item.name!r} requires voltage_v and frequency_hz")
            return jt.RFCA(item.name, item.length_m, float(voltage), float(frequency), **options)
        if kind == "CORRECTOR":
            return jt.CORRECTOR(
                item.name,
                item.length_m,
                xkick=float(item.fields.get("xkick_rad", 0.0)),
                ykick=float(item.fields.get("ykick_rad", 0.0)),
                **options,
            )
        raise ValueError(f"Unsupported JuTrack element kind: {kind}")
