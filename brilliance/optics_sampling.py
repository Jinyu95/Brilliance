"""Map-checked, interior JuTrack optics samples without editing a physical lattice.

Private element prefixes never become a design or an archived replacement.
Scalar proposals reuse compiled JuTrack editors. True
dispersion extrema are located by propagating JuTrack Twiss objects, not by
interpolating plotted values.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq

from .lattice import LatticeElementIR, LatticeInputError, require_same_lattice_topology

_SUPPORTED = {"DRIFT", "KQUAD", "QUAD", "SBEND", "KSEXT", "KOCT", "MARKER"}
_EDGE_FIELDS = {
    "e1", "e2", "entrance_angle", "exit_angle", "entranceangle", "exitangle",
    "fint1", "fint2", "fint", "gap", "fullgap", "hgap",
    "fringebendentrance", "fringebendexit", "fringequadentrance", "fringequadexit",
    "polynoma", "polynomb", "t1", "t2", "xkick_rad", "ykick_rad",
    "rad", "kickangle", "fringeintm0", "fringeintp0",
}


class UnsupportedOpticsSampling(LatticeInputError):
    """Interior diagnostic is unavailable; intact native evaluation may still run."""


def _physical_rows(elements: Sequence[Any]) -> list[dict[str, Any]]:
    rows = [asdict(row if isinstance(row, LatticeElementIR)
                   else LatticeElementIR.from_mapping(row)) for row in elements]
    if not rows or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Optics sampling needs nonempty, uniquely named physical elements")
    for row in rows:
        kind, name = row["kind"], row["name"]
        if kind not in _SUPPORTED:
            raise UnsupportedOpticsSampling(f"Dense optics sampling does not support {kind} element {name!r}; use intact JuTrack optics/radiation without an unvalidated centre")
        if kind == "SBEND":
            if row["length_m"] <= 0.0:
                raise ValueError(f"Dense optics sampling needs a positive-length SBEND: {name!r}")
            if any(float(row.get(field) or 0.0) != 0.0
                   for field in ("k1_m2", "k2_m3", "k3_m4")):
                raise UnsupportedOpticsSampling("Dense optics sampling currently supports pure SBENDs only")
        if kind == "MARKER" and row["length_m"] != 0.0:
            raise ValueError("A diagnostic MARKER must have zero length")
        # These fields would require explicit edge/transform treatment.  Never
        # silently replicate an entrance kick at every diagnostic subdivision.
        for key, value in row["fields"].items():
            lowered = key.casefold().replace("_", "")
            if lowered in {item.replace("_", "") for item in _EDGE_FIELDS}:
                if value is not None and np.any(np.asarray(value, dtype=float) != 0.0):
                    raise UnsupportedOpticsSampling(f"Unsupported diagnostic edge/field {key!r} on {name!r}; intact optics and radiation remain available, but no centre is validated")
            if lowered in {"r1", "r2"} and value is not None:
                matrix = np.asarray(value, dtype=float)
                if matrix.shape != (6, 6) or not (np.array_equal(matrix, np.eye(6)) or not np.any(matrix)):
                    raise UnsupportedOpticsSampling(f"Unsupported diagnostic transform {key!r} on {name!r}")
    return rows


def _profile_row(point: Any, name: str, position: float, **metadata: Any) -> dict[str, Any]:
    row = {"name": name, "s_m": float(position),
           "beta_x": float(point.betax), "beta_y": float(point.betay),
           "alpha_x": float(point.alphax), "alpha_y": float(point.alphay),
           "Dx": float(point.dx), "Dpx": float(point.dpx),
           "Dy": float(point.dy), "Dpy": float(point.dpy), **metadata}
    if not all(math.isfinite(row[key]) for key in
               ("s_m", "beta_x", "beta_y", "alpha_x", "alpha_y", "Dx", "Dpx", "Dy", "Dpy")):
        raise ValueError("JuTrack returned nonfinite dense optics")
    if row["beta_x"] <= 0.0 or row["beta_y"] <= 0.0:
        raise ValueError("JuTrack returned nonpositive dense beta functions")
    return row


class JuTrackProfileSampler:
    """Sample one complete periodic cell, including genuine interior extrema.

    Supported physics is uncoupled, pure hard-edge DRIFT/KQUAD/QUAD/SBEND/
    KSEXT/KOCT/MARKER.  Unsupported edge/combined fields fail explicitly.  A
    changed element count/order/type requires a new sampler; lengths and
    supported strengths can change on each :meth:`linear_metrics` call.
    """

    def __init__(self, candidate: Any, *, max_step_m: float = 0.05,
                 map_rtol: float = 1.0e-7, map_atol: float = 1.0e-9):
        from . import accelerator_api as api

        for name, value in (("max_step_m", max_step_m), ("map_rtol", map_rtol),
                            ("map_atol", map_atol)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        self.max_step_m = float(max_step_m)
        self.map_rtol, self.map_atol = float(map_rtol), float(map_atol)
        self._api = api
        self._candidate = api.load_lattice(candidate) if isinstance(candidate, (str, Path)) else candidate
        self._energy_ev = float(self._candidate.energy_gev) * 1.0e9
        self.rows = _physical_rows(api.one_period_elements(self._candidate))
        self.physical_editor = api.build_jutrack_editor(self._cell(self.rows, "physical"))
        self.jt = self.physical_editor.jt
        self._counts = self._split_counts(self.rows)
        from .lattice_optimization import JuTrackLatticeEditor
        diagnostic = self._diagnostic_rows(self.rows)
        copies = [self.jt.deepcopy_element(element) for element in self.physical_editor.lattice]
        for element, row in zip(copies, diagnostic):
            element.name = row["name"]
        self.diagnostic_editor = JuTrackLatticeEditor(self.jt, self.jt.Lattice(copies), diagnostic)
        self._physical_singletons = [self.jt.Lattice([element]) for element in self.physical_editor.lattice]
        self._diagnostic_singletons = [self.jt.Lattice([element]) for element in self.diagnostic_editor.lattice]

    def _cell(self, rows: Sequence[Mapping[str, Any]], label: str) -> Any:
        return self._api.periodic_cell(
            title=f"Private {label} optics diagnostic", energy_gev=self._candidate.energy_gev,
            elements=rows, source="brilliance.optics_sampling",
            parent_lattice_id=self._candidate.lattice_id,
        )

    def _split_counts(self, rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
        return tuple(max(1, math.ceil(float(row["length_m"]) / self.max_step_m)) for row in rows)

    def _diagnostic_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for physical_index, row in enumerate(rows):
            piece = deepcopy(dict(row))
            piece["name"] = f"__prefix_{physical_index:06d}"
            piece["fields"]["diagnostic_source_element"] = row["name"]
            result.append(piece)
        return result

    def linear_metrics(self, elements: Sequence[Any] | None = None, *,
                       sample_fractions: Mapping[str, Sequence[float]] | None = None) -> dict[str, Any]:
        """Return physical-map optics and a map-checked dense JuTrack profile.

        Interior points are evaluated independently from each original physical
        entrance. Chaining subdivisions would multiply NumIntSteps and silently
        change finite-step KQUAD maps; this sampler never does that. Every full
        prefix must reproduce its physical element's 6x6 map and exact exit.

        A physically unstable or map-incompatible proposal returns no profile
        and cannot be certified by a downstream shape gate.  API/schema errors
        remain errors rather than being mislabeled as unstable physics.

        sample_fractions={"B": [0.5]} requests exact named-element interior
        locations in addition to the regular grid. Fractions lie strictly in
        (0,1); physical boundaries are already included. This measures bend
        centres or drift interiors without changing the physical topology.
        Each explicitly requested fraction also needs native prefix/suffix
        composition to reproduce the intact 6x6 map and eight-component exit.
        A full-length clone alone is not a certificate of interior transport.
        """
        from .lattice_fingerprint import cell_fingerprint

        rows = _physical_rows(self.rows if elements is None else elements)
        require_same_lattice_topology(self.rows, rows, operation="JuTrack profile sampler")
        requested: dict[str, set[float]] = {}
        if sample_fractions is not None:
            if not isinstance(sample_fractions, Mapping):
                raise LatticeInputError("sample_fractions must map physical element names to fractions in (0,1)")
            by_name = {row["name"]: row for row in rows}
            for name, fractions in sample_fractions.items():
                if name not in by_name or by_name[name]["length_m"] <= 0:
                    raise LatticeInputError(f"sample_fractions needs a positive-length physical element; received {name!r}")
                try:
                    values = list(fractions)
                    if any(isinstance(v, bool) for v in values):
                        raise ValueError("boolean fraction")
                    values = [float(v) for v in values]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise LatticeInputError(f"sample_fractions[{name!r}] must be a sequence of fractions in (0,1)") from exc
                if any(not math.isfinite(v) or not 0 < v < 1 for v in values):
                    raise LatticeInputError(f"sample_fractions[{name!r}] must contain finite fractions strictly in (0,1)")
                requested[name] = set(values)
        self.physical_editor.sync_rows(rows)
        self._counts = self._split_counts(rows)
        fractions_by_element = [sorted({piece/count for piece in range(1,count+1)} | requested.get(row["name"], set()))
                                for row,count in zip(rows,self._counts)]
        diagnostic_rows = self._diagnostic_rows(rows)
        self.diagnostic_editor.sync_rows(diagnostic_rows)
        self.rows = rows
        physical, diagnostic = self.physical_editor.lattice, self.diagnostic_editor.lattice
        original_map = np.asarray(self.jt.findm66(physical, dp=0.0, order=0, E0=self._energy_ev), dtype=float)
        sampled_map = np.asarray(self.jt.findm66(diagnostic, dp=0.0, order=0, E0=self._energy_ev), dtype=float)
        if original_map.shape != (6, 6) or sampled_map.shape != (6, 6):
            raise TypeError("JuTrack findm66 must return a complete 6x6 transfer matrix")
        finite = bool(np.all(np.isfinite(original_map)) and np.all(np.isfinite(sampled_map)))
        error = float(np.max(np.abs(original_map - sampled_map))) if finite else math.inf
        equivalent = finite and bool(np.allclose(original_map, sampled_map,
                                                  rtol=self.map_rtol, atol=self.map_atol))
        sampling = {
            "engine": "JuTrack", "method": "physical_element_prefix_map_checked",
            "validated": False,
            "map_equivalent": equivalent, "map_error": error,
            "map_rtol": self.map_rtol, "map_atol": self.map_atol,
            "max_step_m": self.max_step_m,
            "actual_max_step_m": max(row["length_m"] * max(b-a for a,b in zip([0.,*fractions[:-1]],fractions))
                                     for row,fractions in zip(rows,fractions_by_element)),
            "cell_fingerprint": cell_fingerprint(rows),
            "energy_eV": self._energy_ev,
            "physical_element_count": len(rows), "diagnostic_element_count": len(diagnostic_rows),
            "prefix_sample_count": sum(map(len, fractions_by_element)), "exact_extrema_count": 0,
            "requested_sample_fractions": {name: sorted(values) for name,values in requested.items()},
            "interior_composition_scope": "explicitly_requested_sample_fractions",
            "interior_composition_checks": [],
            "element_map_error_max": 0.0, "element_exit_error_max": 0.0,
            "interpretation": "Independent JuTrack partial-element propagation; original finite-step element exits and periodic cell are authoritative.",
        }
        result: dict[str, Any] = {
            "engine": "JuTrack", "stable": False, "twiss_profile": [], "sampling": sampling,
            "one_turn_matrix": original_map.tolist(),
            "trace_x": float(np.trace(original_map[:2, :2])),
            "trace_y": float(np.trace(original_map[2:4, 2:4])),
        }
        if not equivalent:
            result["failure"] = "Private diagnostic copies did not preserve the physical 6x6 map"
            return result
        if abs(result["trace_x"]) >= 2.0 or abs(result["trace_y"]) >= 2.0:
            result["failure"] = "The physical cell is linearly unstable"
            return result
        periodic = self.jt.periodicEdwardsTengTwiss(physical, 0.0, 0, E0=self._energy_ev)
        points = list(self.jt.twissline(periodic, physical, dp=0.0, order=0,
                                       refpts=list(range(1, len(physical) + 1)), E0=self._energy_ev))
        if len(points) != len(physical):
            raise ValueError("JuTrack profile did not cover every physical element exit")
        profile = [_profile_row(periodic, "START", 0.0)]
        position = 0.0
        entrance = periodic
        twiss_fields = ("betax", "betay", "alphax", "alphay", "dx", "dpx", "dy", "dpy")
        for index, (row, original_exit, fractions) in enumerate(zip(rows, points, fractions_by_element)):
            length = float(row["length_m"])
            prefix = diagnostic[index]
            prefix_lattice = self._diagnostic_singletons[index]
            original_element_map = np.asarray(self.jt.findm66(self._physical_singletons[index], dp=0.0, order=0, E0=self._energy_ev), dtype=float)
            prefix_full_map = np.asarray(self.jt.findm66(prefix_lattice, dp=0.0, order=0, E0=self._energy_ev), dtype=float)
            if original_element_map.shape != (6, 6) or prefix_full_map.shape != (6, 6):
                raise TypeError("JuTrack element findm66 must return a complete 6x6 map")
            local_error = float(np.max(np.abs(original_element_map - prefix_full_map)))
            sampling["element_map_error_max"] = max(sampling["element_map_error_max"], local_error)
            if not np.allclose(original_element_map, prefix_full_map, rtol=self.map_rtol, atol=self.map_atol):
                sampling["map_equivalent"] = False
                result["failure"] = f"The full prefix changed the physical map of {row['name']}"
                return result

            def propagated(distance: float) -> Any:
                if distance == 0.0:
                    return entrance
                prefix.len = float(distance)
                if row["angle_rad"] is not None:
                    prefix.angle = float(row["angle_rad"] * distance / length)
                return self.jt.twissline(entrance, prefix_lattice, dp=0.0, order=0, E0=self._energy_ev)

            def slope(distance: float) -> float:
                return float(propagated(distance).dpx)

            previous, previous_distance = entrance, 0.0
            try:
                # A full prefix is a numerical reproduction of the *original*
                # element, not a more finely integrated replacement for it.
                full_exit = self.jt.twissline(entrance, prefix_lattice, dp=0.0, order=0, E0=self._energy_ev)
                expected = np.asarray([float(getattr(original_exit, field)) for field in twiss_fields])
                actual = np.asarray([float(getattr(full_exit, field)) for field in twiss_fields])
                exit_error = float(np.max(np.abs(expected - actual)))
                sampling["element_exit_error_max"] = max(sampling["element_exit_error_max"], exit_error)
                if not np.allclose(expected, actual, rtol=self.map_rtol, atol=self.map_atol):
                    sampling["map_equivalent"] = False
                    result["failure"] = f"The full prefix did not reproduce the physical optics exit of {row['name']}"
                    return result
                for fraction in sorted(requested.get(row["name"], set())):
                    # Independent diagnostic closure: keep both native partial
                    # maps distinct from the intact original reference. Never
                    # use this divided transport for optics/radiation ranking.
                    interior = propagated(length * fraction)
                    prefix_map = np.asarray(self.jt.findm66(prefix_lattice, dp=0.0, order=0, E0=self._energy_ev), dtype=float)
                    prefix.len = float(length * (1.0 - fraction))
                    if row["angle_rad"] is not None:
                        prefix.angle = float(row["angle_rad"] * (1.0 - fraction))
                    suffix_map = np.asarray(self.jt.findm66(prefix_lattice, dp=0.0, order=0, E0=self._energy_ev), dtype=float)
                    if prefix_map.shape != (6, 6) or suffix_map.shape != (6, 6):
                        raise TypeError("JuTrack interior composition needs two complete 6x6 maps")
                    composed_map = suffix_map @ prefix_map
                    composed_exit = self.jt.twissline(interior, prefix_lattice, dp=0.0, order=0, E0=self._energy_ev)
                    transported = np.asarray([float(getattr(composed_exit, field)) for field in twiss_fields])
                    valid = bool(np.allclose(original_element_map, composed_map, rtol=self.map_rtol, atol=self.map_atol)
                                 and np.allclose(expected, transported, rtol=self.map_rtol, atol=self.map_atol))
                    sampling["interior_composition_checks"].append({
                        "source_element": row["name"], "sample_fraction": fraction, "passed": valid,
                        "map_error": float(np.max(np.abs(composed_map - original_element_map))),
                        "twiss_exit_error": float(np.max(np.abs(transported - expected))),
                    })
                    if not valid:
                        sampling["map_equivalent"] = False
                        result["failure"] = f"Native prefix/suffix composition did not reproduce intact {row['name']} at fraction {fraction}; interior diagnostic unvalidated"
                        return result
                for piece, fraction in enumerate(fractions, 1):
                    distance = length * fraction
                    at_exit = fraction == 1.0
                    point = original_exit if at_exit else propagated(distance)
                    d0, d1 = float(previous.dpx), float(point.dpx)
                    if length > 0.0 and d0 * d1 < 0.0 and min(abs(d0), abs(d1)) > 1.0e-12:
                        lower, upper = slope(previous_distance), slope(distance)
                        if lower * upper < 0.0:
                            root = float(brentq(slope, previous_distance, distance, xtol=1.0e-12, rtol=1.0e-12))
                            if previous_distance < root < distance:
                                profile.append(_profile_row(
                                    propagated(root), f"{row['name']}:Dx_extremum", position + root,
                                    source_element=row["name"],
                                    dispersion_extremum="maximum" if lower > 0.0 else "minimum",
                                    extremum_method="JuTrack_twissline_brentq"))
                                sampling["exact_extrema_count"] += 1
                    profile.append(_profile_row(
                        point, row["name"] if at_exit else f"{row['name']}:sample{piece}",
                        position + distance, source_element=row["name"],
                        physical_element_exit=at_exit, sample_fraction=fraction,
                        interior_composition_checked=fraction in requested.get(row["name"], set()),
                        requested_sample=fraction in requested.get(row["name"], set())))
                    previous, previous_distance = point, distance
            finally:
                prefix.len = length
                if row["angle_rad"] is not None:
                    prefix.angle = float(row["angle_rad"])
            position += length
            entrance = original_exit
        # JuTrack gettune names this keyword ``energy``, unlike its map/Twiss
        # routines' ``E0``. Python **kwargs alone does not validate Julia APIs.
        tune = [float(value) for value in self.jt.gettune(physical, energy=self._energy_ev)]
        if len(tune) < 2 or not all(math.isfinite(value) for value in tune[:2]):
            raise ValueError("JuTrack returned an invalid physical-cell tune")
        result.update(
            stable=True, twiss_profile=profile, tune_x=tune[0], tune_y=tune[1],
            Dx_m=float(periodic.dx), Dpx=float(periodic.dpx),
            beta_x_m=float(periodic.betax), beta_y_m=float(periodic.betay),
            maximum_beta_m=max(max(row["beta_x"], row["beta_y"]) for row in profile),
        )
        sampling["validated"] = True
        return result
