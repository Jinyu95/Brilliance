"""Stable typed adapter between :mod:`brilliance.lattice` and pyJuTrack.

Research agents never call pyJuTrack constructors directly on trusted paths.
This adapter is the single versioned API boundary and is covered with contract
tests independently of Julia availability.
"""

from __future__ import annotations

from typing import Any

from .lattice import LatticeElementIR, LatticeIR

JUTRACK_SERVICE_VERSION = "1.1.0"


class JuTrackService:
    def __init__(self, module: Any):
        self.jt = module

    def build_lattice(self, lattice: LatticeIR) -> Any:
        elements = [self.build_element(item) for item in lattice.elements]
        return self.jt.Lattice(elements)

    def build_element(self, item: LatticeElementIR) -> Any:
        jt = self.jt
        kind = item.kind
        if kind == "MARKER":
            return jt.MARKER(item.name)
        if kind == "DRIFT":
            return jt.DRIFT(item.name, item.length_m)
        if kind in {"KQUAD", "QUAD"}:
            if item.k1_m2 is None:
                raise ValueError(f"{kind} element {item.name!r} requires k1_m2")
            return getattr(jt, kind)(item.name, item.length_m, k1=item.k1_m2)
        if kind == "KSEXT":
            if item.k2_m3 is None:
                raise ValueError(f"KSEXT element {item.name!r} requires k2_m3")
            return jt.KSEXT(item.name, item.length_m, k2=item.k2_m3)
        if kind == "KOCT":
            if item.k3_m4 is None:
                raise ValueError(f"KOCT element {item.name!r} requires k3_m4")
            return jt.KOCT(item.name, item.length_m, k3=item.k3_m4)
        if kind in {"SBEND", "RBEND"}:
            if item.angle_rad is None:
                raise ValueError(f"{kind} element {item.name!r} requires angle_rad")
            kwargs: dict[str, Any] = {}
            if item.k1_m2 is not None:
                # Keep old archived lattice files reproducible through the
                # normal-multipole representation. New MBA construction rejects this.
                kwargs["PolynomB"] = [0.0, item.k1_m2, 0.0, 0.0]
                kwargs["MaxOrder"] = 1
            return getattr(jt, kind)(item.name, item.length_m, item.angle_rad, **kwargs)
        if kind == "RFCA":
            voltage = item.fields.get("voltage_v", item.fields.get("volt"))
            frequency = item.fields.get("frequency_hz", item.fields.get("freq"))
            if voltage is None or frequency is None:
                raise ValueError(f"RFCA element {item.name!r} requires voltage_v and frequency_hz")
            return jt.RFCA(item.name, item.length_m, float(voltage), float(frequency))
        if kind == "CORRECTOR":
            return jt.CORRECTOR(
                item.name,
                item.length_m,
                xkick=float(item.fields.get("xkick_rad", 0.0)),
                ykick=float(item.fields.get("ykick_rad", 0.0)),
            )
        raise ValueError(f"Unsupported JuTrack element kind: {kind}")
