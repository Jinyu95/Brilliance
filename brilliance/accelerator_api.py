"""Stable expert-facing API for arbitrary JuTrack lattice experiments.

This is the API documented to the research agent.  It deliberately hides
pyJuTrack import discovery and the archive implementation while keeping lattice
construction completely programmable rather than template driven.
"""

from __future__ import annotations

import json
import math
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .api_result_types import MagnetStrengthLimits, PeriodicCellEvaluation
from .artifacts import json_safe
from .candidate_evidence import (
    restore_candidate_evaluation,  # noqa: F401 - verified archived result reader
)
from .evaluation_results import evaluation_snapshot  # noqa: F401 - public result reader
from .julia_runtime import load_pyjutrack
from .lattice import LatticeElementIR, LatticeIR, LatticeScopeError, lattice_cell_partition
from .lattice_editing import (
    insert_element_between,  # noqa: F401 - public expert API
    insert_element_in_drift,  # noqa: F401 - public expert API
    remove_inserted_element,  # noqa: F401 - public expert API
    replace_lattice_elements,  # noqa: F401 - public expert API
    resize_inserted_element,  # noqa: F401 - public expert API
)
from .population_optimization import read_optimization_checkpoint  # noqa: F401 - public API
from .research_outcomes import ResearchInfeasible  # noqa: F401 - public scientific outcome


def expert_module(module_id: str, *, registry: Any = None) -> Any:
    """Return an installed expert module's declared in-process Python tools.

    Resolve once per program/worker, then call the namespace's ordinary Python
    functions in numerical callbacks. Tools and policies belong to the selected
    module; no topic-specific implementation is imported by the core API.
    """
    from .experts import ExpertRegistry
    return (registry if registry is not None else ExpertRegistry.current()).python_library(module_id)


def __getattr__(name: str) -> Any:
    """Resolve module-declared legacy aliases for already archived programs.

    No alias is installed into core globals: removing/disabling its provider
    removes that capability, including after a prior successful lookup.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    from .experts import ExpertRegistry
    try:
        return ExpertRegistry.current().legacy_python_tool(name)
    except LookupError as exc:
        raise AttributeError(f"No core API or installed expert compatibility export named {name!r}") from exc


def jutrack() -> Any:
    """Return the configured pyJuTrack module without runtime discovery.

    Task-specific programs may use this escape hatch when a stable high-level
    helper does not cover the requested physics.  The configured environment,
    not the language model, determines the module and Julia runtime.
    """

    return load_pyjutrack()


def build_jutrack_lattice(candidate: LatticeIR | Path | str) -> Any:
    """Build a mutable pyJuTrack lattice from an explicit Brilliance lattice."""

    from .jutrack_service import JuTrackService

    lattice_ir = load_lattice(candidate) if isinstance(candidate, (Path, str)) else candidate
    lattice_cell_partition(lattice_ir)
    return JuTrackService(load_pyjutrack()).build_lattice(lattice_ir)


def build_jutrack_profile_sampler(candidate: LatticeIR | Path | str, *, max_step_m: float = 0.05) -> Any:
    """Compile a reusable dense, map-checked JuTrack profile sampler for one cell.

    ``sampler.linear_metrics(proposed_rows)`` updates scalar fields on private
    diagnostic copies, resolves interior dispersion extrema with JuTrack, and
    leaves the unsplit physical design and its saved parent unchanged.
    Request exact interior locations with ``sampler.linear_metrics(
    sample_fractions={"B": [0.5]})``; do not split a physical bend merely to
    inspect its centre. Each requested profile row identifies its source_element
    and sample_fraction. A changed element list requires a new sampler.
    """
    from .optics_sampling import JuTrackProfileSampler
    return JuTrackProfileSampler(candidate, max_step_m=max_step_m)


@dataclass(frozen=True, slots=True)
class LinearOpticsResult:
    stable: bool
    trace_x: float | None
    trace_y: float | None
    tune_x: float | None
    tune_y: float | None
    periodic_closure_max: float | None
    periodic_residuals: Mapping[str, Any]
    twiss_profile: tuple[Mapping[str, Any], ...]
    one_turn_matrix: tuple[tuple[float, ...], ...]
    raw: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return json_safe(self.raw)

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


@dataclass(frozen=True, slots=True)
class CandidateEvaluationResult:
    """Typed JuTrack result that prevents string/Boolean contract mistakes."""

    lattice_id: str
    geometry: Mapping[str, Any]
    inventory: Mapping[str, Any]
    linear_optics: LinearOpticsResult
    radiation: Mapping[str, Any]
    raw: Mapping[str, Any]

    @property
    def passed_baseline_gate(self) -> bool:
        closure_required = bool(self.geometry.get("ring_bend_closure_required", True))
        closure_passed = bool(self.geometry.get("ring_bend_closure_passed"))
        return (not closure_required or closure_passed) and self.linear_optics.stable

    @property
    def radiation_computed(self) -> bool:
        """Whether JuTrack radiation quantities are present and may be inspected."""

        return self.radiation.get("computed") is True

    def require_radiation(self) -> Mapping[str, Any]:
        """Return radiation data or raise one concise gate error.

        Generated research programs should use this method instead of guessing
        result keys recursively.  A missing radiation calculation is normally a
        geometry or stability result, not an unknown output schema.
        """

        if not self.radiation_computed:
            reason = str(self.radiation.get("reason") or "JuTrack radiation was not computed.")
            raise RuntimeError(f"Radiation unavailable: {reason}")
        return self.radiation

    def as_dict(self) -> dict[str, Any]:
        return json_safe(self.raw)

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


def element(
    kind: str,
    name: str,
    length_m: float = 0.0,
    *,
    k1_m2: float | None = None,
    k2_m3: float | None = None,
    k3_m4: float | None = None,
    angle_rad: float | None = None,
    family: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Create one schema-validated, SI-unit lattice element mapping.

    Physics fields are top-level: name, kind, length_m, k1_m2, k2_m3, k3_m4,
    angle_rad. Metadata is nested: row['fields']['family'] and
    row['fields']['design_role']. This is also the saved/loaded row format.
    """

    normalized_kind = str(kind).upper().strip()
    friendly = {
        "QUADRUPOLE": "quadrupole(...) or kind='KQUAD'",
        "SEXTUPOLE": "sextupole(...) or kind='KSEXT'",
        "BEND": "sector_bend(...) or kind='SBEND'",
    }
    if normalized_kind in friendly:
        raise ValueError(f"unsupported element token {kind!r}; use {friendly[normalized_kind]}")
    metadata = {key: value for key, value in fields.items() if value is not None}
    if family:
        metadata["family"] = family
    item = LatticeElementIR(
        name=name,
        kind=normalized_kind,
        length_m=length_m,
        k1_m2=k1_m2,
        k2_m3=k2_m3,
        k3_m4=k3_m4,
        angle_rad=angle_rad,
        fields=metadata,
    )
    return json_safe(asdict(item))


def marker(name: str, *, design_role: str | None = None, **fields: Any) -> dict[str, Any]:
    return element("MARKER", name, design_role=design_role, **fields)


def drift(
    name: str,
    length_m: float,
    *,
    design_role: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return element("DRIFT", name, length_m, design_role=design_role, **fields)


def quadrupole(
    name: str,
    length_m: float,
    k1_m2: float,
    *,
    family: str | None = None,
    design_role: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return element(
        "KQUAD",
        name,
        length_m,
        k1_m2=k1_m2,
        family=family,
        design_role=design_role,
        **fields,
    )


def sextupole(
    name: str,
    length_m: float,
    k2_m3: float,
    *,
    family: str | None = None,
    design_role: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return element(
        "KSEXT",
        name,
        length_m,
        k2_m3=k2_m3,
        family=family,
        design_role=design_role,
        **fields,
    )


def sector_bend(
    name: str,
    length_m: float,
    angle_rad: float,
    *,
    family: str | None = None,
    design_role: str = "primary_bend",
    **fields: Any,
) -> dict[str, Any]:
    return element(
        "SBEND",
        name,
        length_m,
        angle_rad=angle_rad,
        family=family,
        design_role=design_role,
        **fields,
    )


def reverse_bend(
    name: str,
    length_m: float,
    angle_rad: float,
    *,
    family: str | None = None,
    design_role: str = "reverse_bend",
    **fields: Any,
) -> dict[str, Any]:
    if angle_rad >= 0.0:
        raise ValueError("reverse_bend angle_rad must be negative")
    return element(
        "RBEND",
        name,
        length_m,
        angle_rad=angle_rad,
        family=family,
        design_role=design_role,
        **fields,
    )


def repeat_cell(
    cell: Sequence[LatticeElementIR | Mapping[str, Any]],
    periods: int,
    *,
    period_prefix: str = "P",
) -> list[dict[str, Any]]:
    """Expand one ordered cell into an explicit full ring with unique names."""

    if periods < 1:
        raise ValueError("periods must be positive")
    normalized = [
        item if isinstance(item, LatticeElementIR) else LatticeElementIR.from_mapping(item)
        for item in cell
    ]
    names = [item.name for item in normalized]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "cell element names must identify unique physical instances; duplicates: "
            + ", ".join(duplicates)
        )
    ring: list[dict[str, Any]] = []
    for period in range(1, periods + 1):
        for item in normalized:
            fields = {
                **dict(item.fields),
                "source_element": item.name,
                "source_period": period,
            }
            ring.append(
                json_safe(
                    asdict(
                        LatticeElementIR(
                            name=f"{period_prefix}{period:03d}:{item.name}",
                            kind=item.kind,
                            length_m=item.length_m,
                            k1_m2=item.k1_m2,
                            k2_m3=item.k2_m3,
                            k3_m4=item.k3_m4,
                            angle_rad=item.angle_rad,
                            fields=fields,
                        )
                    )
                )
            )
    return ring


def scale_element_lengths_to_total(
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    target_total_length_m: float,
    fixed_element_names: Sequence[str] = (),
    adjustable_kinds: Sequence[str] | None = None,
    minimum_length_m: float = 1.0e-6,
) -> list[dict[str, Any]]:
    """Scale authorized positive lengths to an exact lattice-length contract.

    This is a geometry primitive, not a lattice template. It preserves order,
    names, strengths, angles, and relative lengths of the adjustable elements.
    A final adjustable element absorbs floating-point residue. Callers choose
    which names or kinds are fixed and must recheck engineering field limits
    after changing bend lengths.
    """

    target = float(target_total_length_m)
    minimum = float(minimum_length_m)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_total_length_m must be finite and positive")
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_length_m must be finite and non-negative")
    fixed = {str(name) for name in fixed_element_names}
    kinds = {str(kind).upper() for kind in adjustable_kinds} if adjustable_kinds else None
    rows = [
        json_safe(
            asdict(
                item
                if isinstance(item, LatticeElementIR)
                else LatticeElementIR.from_mapping(item)
            )
        )
        for item in elements
    ]
    adjustable: list[int] = []
    fixed_length = 0.0
    adjustable_length = 0.0
    for index, row in enumerate(rows):
        length = float(row.get("length_m") or 0.0)
        authorized = (
            row["name"] not in fixed
            and length > 0.0
            and (kinds is None or str(row["kind"]).upper() in kinds)
        )
        if authorized:
            adjustable.append(index)
            adjustable_length += length
        else:
            fixed_length += length
    available = target - fixed_length
    if not adjustable or adjustable_length <= 0.0:
        raise ValueError("no positive authorized element lengths are available to scale")
    if available <= 0.0:
        raise ValueError(
            f"fixed elements occupy {fixed_length:.12g} m, exceeding the "
            f"{target:.12g} m total-length contract"
        )
    factor = available / adjustable_length
    for index in adjustable:
        rows[index]["length_m"] = float(rows[index]["length_m"]) * factor
        if rows[index]["length_m"] < minimum:
            raise ValueError(
                f"scaled element {rows[index]['name']} is shorter than "
                f"minimum_length_m={minimum:.12g}"
            )
    residual = target - sum(float(row.get("length_m") or 0.0) for row in rows)
    rows[adjustable[-1]]["length_m"] += residual
    return rows


def uniquify_element_names(
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    separator: str = "#",
) -> list[dict[str, Any]]:
    """Give repeated physical instances deterministic unique names.

    The first occurrence keeps its name and later occurrences receive ``#2``,
    ``#3``, and so on. Physics fields and family labels are unchanged.
    """

    if not separator:
        raise ValueError("separator must be nonempty")
    rows = [
        json_safe(
            asdict(
                item
                if isinstance(item, LatticeElementIR)
                else LatticeElementIR.from_mapping(item)
            )
        )
        for item in elements
    ]
    counts: dict[str, int] = {}
    occupied = {str(row["name"]) for row in rows}
    assigned: set[str] = set()
    for row in rows:
        original = str(row["name"])
        counts[original] = counts.get(original, 0) + 1
        if counts[original] == 1 and original not in assigned:
            assigned.add(original)
            continue
        occurrence = counts[original]
        candidate = f"{original}{separator}{occurrence}"
        while candidate in occupied or candidate in assigned:
            occurrence += 1
            candidate = f"{original}{separator}{occurrence}"
        fields = dict(row.get("fields") or {})
        fields.setdefault("source_element_name", original)
        row["fields"] = fields
        row["name"] = candidate
        assigned.add(candidate)
    return rows


def mirror_elements(
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    name_prefix: str = "R:",
) -> list[dict[str, Any]]:
    """Reflect an ordered section while preserving all physical fields.

    ``elements`` must be ordered from the outer boundary toward the symmetry
    plane. The returned copy is reversed for travel away from that plane.
    Bend angles are intentionally not sign-flipped: a geometrically symmetric
    storage-ring achromat normally bends in the same direction on both halves.
    Families are preserved so mirror partners remain tied, while instance names
    receive a prefix and stay independently addressable.
    """

    prefix = str(name_prefix)
    if not prefix:
        raise ValueError("name_prefix must be nonempty")
    mirrored: list[dict[str, Any]] = []
    for supplied in reversed(elements):
        item = (
            supplied
            if isinstance(supplied, LatticeElementIR)
            else LatticeElementIR.from_mapping(supplied)
        )
        fields = dict(item.fields)
        fields.setdefault("mirror_source_element", item.name)
        mirrored.append(
            json_safe(
                asdict(
                    LatticeElementIR(
                        name=prefix + item.name,
                        kind=item.kind,
                        length_m=item.length_m,
                        k1_m2=item.k1_m2,
                        k2_m3=item.k2_m3,
                        k3_m4=item.k3_m4,
                        angle_rad=item.angle_rad,
                        fields=fields,
                    )
                )
            )
        )
    return mirrored


def symmetric_cell_elements(
    left: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    center: LatticeElementIR | Mapping[str, Any] | None = None,
    name_prefix: str = "R:",
) -> list[dict[str, Any]]:
    """Assemble left + optional FULL central element + mirror(left).

    The left section excludes the centre element. For an odd-bend cell pass
    the entire central bend length and angle, never a half bend. For symmetry
    between elements omit center. No numerical lattice or bend count is assumed.
    """
    supplied = list(left) + ([] if center is None else [center])
    rows = [json_safe(asdict(item if isinstance(item, LatticeElementIR)
                            else LatticeElementIR.from_mapping(item))) for item in supplied]
    rows.extend(mirror_elements(left, name_prefix=name_prefix))
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("symmetric cell has duplicate element names")
    return rows


def audit_lattice_constraints(
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    *,
    energy_gev: float,
    expected_length_m: float,
    expected_angle_rad: float,
    expected_bends: int | None = None,
    require_mirror_symmetry: bool = False,
    minimum_magnet_length_m: float = 0.0,
    minimum_drift_length_m: float = 0.0,
    limits: Mapping[str, float] | None = None,
    length_bounds_m: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Audit geometry and hard-edge field limits before any JuTrack call.

    Counts nonzero-angle elements (not hidden physical-magnet groupings). A
    passed construction audit is NOT stability, achromaticity, or design success.
    Call again for every proposed candidate, including diagnostic scans.
    Optional length_bounds_m maps physical kinds (SBEND, KQUAD, KSEXT, KOCT,
    DRIFT) to [lower, upper] in metres; no MBA defaults are imposed here.
    """
    for value in (energy_gev, expected_length_m):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("energy and expected length must be positive finite numbers")
    if not math.isfinite(expected_angle_rad):
        raise ValueError("expected angle must be finite")
    for value in (minimum_magnet_length_m, minimum_drift_length_m):
        if not math.isfinite(value) or value < 0:
            raise ValueError("minimum lengths must be finite and nonnegative")
    envelope = dict(limits) if limits is not None else magnet_strength_limits(energy_gev)
    if not math.isclose(float(envelope.get("energy_gev", energy_gev)), energy_gev, rel_tol=1e-12):
        raise ValueError("field limits belong to a different beam energy")
    for key in ("dipole_field_t", "max_abs_k1_m2", "max_abs_k2_m3"):
        if not math.isfinite(float(envelope[key])) or float(envelope[key]) <= 0:
            raise ValueError(f"invalid engineering limit: {key}")
    items = [item if isinstance(item, LatticeElementIR) else LatticeElementIR.from_mapping(item) for item in elements]
    for kind, pair in (length_bounds_m or {}).items():
        if kind not in {"SBEND", "RBEND", "KQUAD", "QUAD", "KSEXT", "KOCT", "DRIFT"} or len(pair) != 2 or not all(math.isfinite(x) and x >= 0 for x in pair) or pair[0] > pair[1]:
            raise ValueError(f"invalid physical length bounds for {kind}")
    issues: list[str] = []
    if not items:
        issues.append("empty lattice")
    if len({item.name for item in items}) != len(items):
        issues.append("duplicate element names")
    length = sum(item.length_m for item in items)
    angle = sum(float(item.angle_rad or 0) for item in items)
    bends = sum(abs(float(item.angle_rad or 0)) > 0 for item in items)
    if not math.isclose(length, expected_length_m, rel_tol=0, abs_tol=1e-9):
        issues.append(f"length {length:g} m != required {expected_length_m:g} m")
    if not math.isclose(angle, expected_angle_rad, rel_tol=0, abs_tol=1e-12):
        issues.append(f"bend sum {angle:g} rad != required {expected_angle_rad:g} rad")
    if expected_bends is not None and bends != expected_bends:
        issues.append(f"bend count {bends} != required {expected_bends}")
    margins = []
    brho = energy_gev / 0.299792458
    for item in items:
        minimum = minimum_drift_length_m if item.kind == "DRIFT" else minimum_magnet_length_m
        if item.kind != "MARKER" and (item.length_m <= 0 or item.length_m < minimum):
            issues.append(f"{item.name}: length below physical minimum")
        pair = (length_bounds_m or {}).get(item.kind)
        if pair is not None and not pair[0] - 1e-12 <= item.length_m <= pair[1] + 1e-12:
            issues.append(f"{item.name}: length outside declared {item.kind} range")
        field = brho * abs(float(item.angle_rad or 0)) / item.length_m if item.length_m > 0 else 0.0
        fractions = {"dipole": field / envelope["dipole_field_t"],
                     "quadrupole": abs(float(item.k1_m2 or 0)) / envelope["max_abs_k1_m2"],
                     "sextupole": abs(float(item.k2_m3 or 0)) / envelope["max_abs_k2_m3"]}
        if item.k3_m4:
            cap = envelope.get("max_abs_k3_m4")
            if cap is None or not math.isfinite(float(cap)) or cap <= 0:
                issues.append(f"{item.name}: octupole requires an explicit field/aperture limit")
            else:
                fractions["octupole"] = abs(float(item.k3_m4)) / cap
        for kind, fraction in fractions.items():
            if not math.isfinite(fraction) or fraction > 1 + 1e-12:
                issues.append(f"{item.name}: {kind} exceeds field limit")
        margins.append({"element": item.name, "dipole_field_t": field, **{f"{k}_limit_fraction": v for k, v in fractions.items()}})
    if require_mirror_symmetry:
        for left, right in zip(items, reversed(items)):
            if left.kind != right.kind or any(not math.isclose(float(getattr(left, key) or 0), float(getattr(right, key) or 0), rel_tol=1e-10, abs_tol=1e-12)
                    for key in ("length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4")):
                issues.append(f"mirror mismatch: {left.name} / {right.name}")
                break
    return {"passed": not issues, "issues": issues, "length_m": length, "angle_rad": angle,
            "bends": bends, "magnet_margins": margins, "limits": envelope,
            "scope": "construction_and_engineering_only"}


def lattice_geometry_summary(
    elements: (
        LatticeIR
        | Path
        | str
        | Sequence[LatticeElementIR | Mapping[str, Any]]
    ),
) -> dict[str, Any]:
    """Return deterministic order/length/angle/name checks before JuTrack.

    Returns a dictionary with element_count, physical_bend_count (also
    bend_count), total_length_m, total_angle_rad, unique_names and
    duplicate_names. Use summary["total_length_m"] and
    summary["total_angle_rad"]; these are not the length_m/angle_rad keys
    returned by audit_lattice_constraints. No optics or radiation is computed.
    """

    if isinstance(elements, (Path, str)):
        sequence: Sequence[LatticeElementIR | Mapping[str, Any]] = load_lattice(elements).elements
    elif isinstance(elements, LatticeIR):
        sequence = elements.elements
    else:
        sequence = elements
    normalized = [
        item if isinstance(item, LatticeElementIR) else LatticeElementIR.from_mapping(item)
        for item in sequence
    ]
    names = [item.name for item in normalized]
    bends = [item for item in normalized if abs(float(item.angle_rad or 0.0)) > 0.0]
    return {
        "element_count": len(normalized),
        "physical_bend_count": len(bends),
        # Concise compatibility alias for model-written programs. Both values
        # count physical nonzero-angle elements, never bend families.
        "bend_count": len(bends),
        "total_length_m": sum(float(item.length_m) for item in normalized),
        "total_angle_rad": sum(float(item.angle_rad or 0.0) for item in normalized),
        "unique_names": len(names) == len(set(names)),
        "duplicate_names": sorted({name for name in names if names.count(name) > 1}),
    }


def magnet_strength_limits(
    energy_gev: float,
    *,
    dipole_field_t: float = 2.1,
    quadrupole_pole_tip_field_t: float = 1.2,
    sextupole_pole_tip_field_t: float = 1.0,
    quadrupole_radius_m: float = 0.02,
    sextupole_radius_m: float = 0.02,
    octupole_pole_tip_field_t: float | None = None,
    octupole_radius_m: float | None = None,
    length_m: float | None = None,
) -> MagnetStrengthLimits:
    """Field-derived limits for an ultra-relativistic unit-charge beam.

    Normalized Kn = (1/Brho) d^n By/dx^n: Kn_max = n! Bpole/(Brho*r^n).
    These local strengths are independent of length. With length_m, also return
    max_abs_angle_rad=Bmax*L/Brho and max_abs_k1l_m1/k2l_m2/k3l_m3=Kn_max*L.
    JuTrack stores scalar k2/k3 as derivatives; PolynomB uses k2/2 and k3/6.
    Default exploratory envelope: 2.1 T dipole, 1.2 T quadrupole and 1.0 T
    sextupole pole-tip fields. These are design assumptions, not universal
    demonstrated limits or a combined-function magnet feasibility certificate.
    Reference radii remain 20 mm; specify smaller apertures explicitly rather
    than silently borrowing small-bore prototype gradients. Explicit saved
    project envelopes override these defaults and must remain reproducible.
    Octupole limits require BOTH explicit field and radius; no technology default
    is invented. Applicability of any field envelope depends on the declared magnet technology.
    """

    values = {
        "energy_gev": float(energy_gev),
        "dipole_field_t": float(dipole_field_t),
        "quadrupole_pole_tip_field_t": float(quadrupole_pole_tip_field_t),
        "sextupole_pole_tip_field_t": float(sextupole_pole_tip_field_t),
        "quadrupole_radius_m": float(quadrupole_radius_m),
        "sextupole_radius_m": float(sextupole_radius_m),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("energy, field limits, and reference radii must be finite and positive")
    brho = values["energy_gev"] / 0.299792458
    result = {
        **values,
        "brho_tm": brho,
        "minimum_bend_radius_m": brho / values["dipole_field_t"],
        "max_abs_k1_m2": values["quadrupole_pole_tip_field_t"]
        / (values["quadrupole_radius_m"] * brho),
        "max_abs_k2_m3": 2.0
        * values["sextupole_pole_tip_field_t"]
        / (values["sextupole_radius_m"] ** 2 * brho),
    }
    if (octupole_pole_tip_field_t is None) != (octupole_radius_m is None):
        raise ValueError("Supply both octupole pole-tip field and radius")
    if octupole_pole_tip_field_t is not None:
        if any(not math.isfinite(v) or v <= 0 for v in (octupole_pole_tip_field_t, octupole_radius_m)):
            raise ValueError("octupole field and radius must be positive and finite")
        result.update(octupole_pole_tip_field_t=float(octupole_pole_tip_field_t),
                      octupole_radius_m=float(octupole_radius_m),
                      max_abs_k3_m4=6 * octupole_pole_tip_field_t / (brho * octupole_radius_m**3))
    if length_m is not None:
        if not math.isfinite(length_m) or length_m <= 0:
            raise ValueError("length_m must be positive and finite")
        result.update(length_m=float(length_m), max_abs_angle_rad=dipole_field_t * length_m / brho,
                      max_abs_k1l_m1=result["max_abs_k1_m2"] * length_m,
                      max_abs_k2l_m2=result["max_abs_k2_m3"] * length_m)
        if "max_abs_k3_m4" in result:
            result["max_abs_k3l_m3"] = result["max_abs_k3_m4"] * length_m
    return result


def lattice(
    *,
    title: str,
    energy_gev: float,
    periodicity: int,
    elements: Iterable[LatticeElementIR | Mapping[str, Any]],
    source: str,
    parent_lattice_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LatticeIR:
    """Create an explicit stored sequence, never implicitly repeat its rows.

    periodicity counts cells ALREADY present. Use periodic_cell for one cell
    intended for later repetition; cell_ring expands rows to a repeated ring.
    """

    candidate = LatticeIR.create(
        title=title,
        energy_gev=energy_gev,
        periodicity=periodicity,
        elements=elements,
        source=source,
        parent_lattice_id=parent_lattice_id,
        metadata=metadata,
    )
    lattice_cell_partition(candidate)
    return candidate


def periodic_cell(
    *,
    title: str,
    energy_gev: float,
    elements: Sequence[LatticeElementIR | Mapping[str, Any]],
    source: str,
    intended_repetitions: int | None = None,
    parent_lattice_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LatticeIR:
    """Create one periodic matching unit, explicitly not a complete storage ring.

    JuTrack may solve the periodic map of this sequence. Ring bend closure and
    radiation are intentionally deferred until distinct matching units have
    been assembled into an achromat and the achromat has been repeated to 2*pi.
    """

    supplied_metadata = dict(metadata or {})
    # Structural scope belongs to this constructor, not to its parent ring.
    if supplied_metadata.get("evaluation_scope") != "periodic_cell":
        supplied_metadata.pop("intended_repetitions", None)
    combined_metadata = {
        **supplied_metadata,
        "evaluation_scope": "periodic_cell",
        "ring_complete": False,
        "repetitions": 1,
        "cell_element_count": len(elements),
    }
    if intended_repetitions is not None:
        combined_metadata["intended_repetitions"] = intended_repetitions
    return lattice(
        title=title,
        energy_gev=energy_gev,
        periodicity=1,
        elements=elements,
        source=source,
        parent_lattice_id=parent_lattice_id,
        metadata=combined_metadata,
    )


def cell_ring(
    *,
    title: str,
    energy_gev: float,
    cell: Sequence[LatticeElementIR | Mapping[str, Any]],
    periods: int,
    source: str,
    parent_lattice_id: str | None = None,
    design_constraints: Sequence[Mapping[str, Any] | str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LatticeIR:
    """Create an explicit repeated full ring from an arbitrary programmed cell."""

    supplied_metadata = dict(metadata or {})
    supplied_metadata.pop("intended_repetitions", None)
    combined_metadata = {
        **supplied_metadata,
        "evaluation_scope": "full_ring",
        "ring_complete": True,  # declared scope, not a numerical closure certificate
        "repetitions": periods,
        "cell_element_count": len(cell),
    }
    if design_constraints is not None:
        combined_metadata["design_constraints"] = [
            dict(item) if isinstance(item, Mapping) else {"description": str(item)}
            for item in design_constraints
        ]
    return lattice(
        title=title,
        energy_gev=energy_gev,
        periodicity=periods,
        elements=repeat_cell(cell, periods),
        source=source,
        parent_lattice_id=parent_lattice_id,
        metadata=combined_metadata,
    )


def save_lattice(path: Path | str, candidate: LatticeIR) -> Path:
    """Save a candidate specification; publication remains a separate validated action."""

    destination = Path(path).resolve()
    lattice_cell_partition(candidate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(candidate.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def load_lattice(path: Path | str) -> LatticeIR:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Lattice file must contain one JSON object")
    if payload.get("lattice_id"):
        return LatticeIR.from_dict(payload)
    return LatticeIR.create(
        title=str(payload["title"]),
        energy_gev=float(payload["energy_gev"]),
        periodicity=payload.get("periodicity", 1),
        elements=payload.get("elements", payload.get("rows", ())),
        source=str(payload.get("source", f"candidate file {Path(path).name}")),
        parent_lattice_id=(
            str(payload["parent_lattice_id"])
            if payload.get("parent_lattice_id") is not None
            else None
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def one_period_elements(
    candidate: LatticeIR | Path | str,
) -> list[dict[str, Any]]:
    """Return physical rows for one declared cell, removing at most one archive layer.

    An already-periodic cell is returned unchanged: it may contain repeated
    physical subunits whose names/provenance must not be flattened. For a full
    ring, restore the first period's source names only when one common outer
    repetition layer is unambiguous. Never recursively strip P001-like names;
    those may identify distinct physical subunits within the requested cell.
    """

    lattice_ir = load_lattice(candidate) if isinstance(candidate, (Path, str)) else candidate
    count, _ = lattice_cell_partition(lattice_ir)
    if lattice_ir.metadata.get("evaluation_scope") == "periodic_cell":
        rows = [json_safe(asdict(item)) for item in lattice_ir.elements]
        if len({row["name"] for row in rows}) != len(rows):
            raise ValueError("periodic cell contains duplicate physical element names")
        return rows
    selected = lattice_ir.elements[:count]
    sources = [str(item.fields.get("source_element") or "") for item in selected]
    prefixes = [item.name[:-(len(source) + 1)]
                if source and item.name.endswith(":" + source) else None
                for item, source in zip(selected, sources)]
    restore_outer_layer = (
        all(sources) and len(set(sources)) == count
        and all(prefixes) and len(set(prefixes)) == 1
        and all(item.fields.get("source_period") == 1 for item in selected)
    )
    rows = []
    for item in selected:
        row = asdict(item)
        fields = dict(row.get("fields") or {})
        if restore_outer_layer:
            row["name"] = str(fields.pop("source_element"))
            fields.pop("source_period", None)
        row["fields"] = fields
        rows.append(json_safe(row))
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("period extraction produced duplicate source element names")
    return rows


def build_jutrack_editor(
    candidate: LatticeIR | Path | str,
) -> Any:
    """Build one mutable JuTrack lattice and compile its name-index editor.

    Reuse for scalar length/strength/angle changes only. Insertion, removal,
    splitting, renaming, reordering or type changes require a new editor made
    from the changed candidate. Interior diagnostics use a profile sampler.
    """

    from .jutrack_service import JuTrackService

    lattice_ir = load_lattice(candidate) if isinstance(candidate, (Path, str)) else candidate
    lattice_cell_partition(lattice_ir)
    jt = load_pyjutrack()
    mutable = JuTrackService(jt).build_lattice(lattice_ir)
    return JuTrackLatticeEditor(jt, mutable, [asdict(item) for item in lattice_ir.elements])


def evaluate_candidate(
    candidate: LatticeIR | Path | str,
    *,
    include_radiation: bool = True,
    include_profile: bool = False,
    include_chromaticity: bool = False,
) -> dict[str, Any]:
    """Evaluate a complete candidate with JuTrack without accepting project state."""

    return evaluate(
        candidate,
        include_radiation=include_radiation,
        include_profile=include_profile,
        include_chromaticity=include_chromaticity,
    ).as_dict()


def evaluate(
    candidate: LatticeIR | Path | str,
    *,
    include_radiation: bool = True,
    include_profile: bool = False,
    include_chromaticity: bool = False,
) -> CandidateEvaluationResult:
    """Run JuTrack and return a typed result while preserving complete raw evidence.

    Use api.evaluation_snapshot(result) for the same geometry/optics/radiation
    reading contract as evaluate_periodic_cell; retain result itself for finish.
    """

    lattice_ir = load_lattice(candidate) if isinstance(candidate, (Path, str)) else candidate
    lattice_cell_partition(lattice_ir)
    jt = load_pyjutrack()
    linear = _linear_evaluation(jt, lattice_ir)
    if not include_profile:
        linear.pop("twiss_profile", None)
        linear.pop("one_turn_matrix", None)
    closure_tolerance_rad = 1.0e-9
    closure_error_rad = lattice_ir.bend_sum_rad - 2.0 * math.pi
    result: dict[str, Any] = {
        "engine": "JuTrack",
        "candidate": lattice_ir.as_dict(),
        "inventory": lattice_inventory(lattice_ir),
        "geometry": {
            "evaluation_scope": str(lattice_ir.metadata.get("evaluation_scope", "full_ring")),
            "circumference_m": lattice_ir.total_length_m,
            "bend_sum_rad": lattice_ir.bend_sum_rad,
            "ring_closure_error_rad": closure_error_rad,
            "ring_closure_tolerance_rad": closure_tolerance_rad,
            "ring_bend_closure_passed": abs(closure_error_rad) < closure_tolerance_rad,
            "ring_bend_closure_required": (
                lattice_ir.metadata.get("evaluation_scope") != "periodic_cell"
            ),
        },
        "linear_optics": linear,
        "accepted": False,
        "publication_required": True,
    }
    if include_chromaticity and bool(linear.get("stable")):
        result["chromaticity"] = natural_chromaticity(lattice_ir, _loaded_jutrack=jt)
    ring_closed = bool(result["geometry"]["ring_bend_closure_passed"])
    periodic_scope = result["geometry"]["evaluation_scope"] == "periodic_cell"
    if include_radiation and bool(linear.get("stable")) and ring_closed and not periodic_scope:
        result["radiation"] = {"computed": True, **_radiation_evaluation(jt, lattice_ir)}
    elif include_radiation:
        result["radiation"] = {
            "computed": False,
            "reason": (
                "Radiation was not claimed for a periodic matching cell; assemble and close "
                "the complete ring first."
                if periodic_scope
                else "Radiation was not claimed because the candidate does not close to 2*pi "
                f"total ring bend: error={closure_error_rad:.12g} rad, "
                f"tolerance={closure_tolerance_rad:.12g} rad."
                if not ring_closed
                else "Radiation was not claimed because the JuTrack one-turn map is unstable."
            ),
        }
    from .evaluation_results import evaluated_geometry
    result["evaluation_geometry"] = evaluated_geometry(lattice_ir)
    safe = json_safe(result)
    from .candidate_evidence import record_evaluation
    safe["evaluation_receipt"] = record_evaluation(lattice_ir, safe)
    linear_raw = dict(safe["linear_optics"])
    linear_result = LinearOpticsResult(
        stable=bool(linear_raw.get("stable")),
        trace_x=_optional_number(linear_raw.get("trace_x")),
        trace_y=_optional_number(linear_raw.get("trace_y")),
        tune_x=_optional_number(linear_raw.get("tune_x")),
        tune_y=_optional_number(linear_raw.get("tune_y")),
        periodic_closure_max=_optional_number(linear_raw.get("periodic_closure_max")),
        periodic_residuals=dict(linear_raw.get("periodic_residuals", {})),
        twiss_profile=tuple(
            dict(item) for item in linear_raw.get("twiss_profile", ()) if isinstance(item, Mapping)
        ),
        one_turn_matrix=tuple(
            tuple(float(value) for value in row)
            for row in linear_raw.get("one_turn_matrix", ())
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
        ),
        raw=linear_raw,
    )
    return CandidateEvaluationResult(
        lattice_id=lattice_ir.lattice_id,
        geometry=dict(safe["geometry"]),
        inventory=dict(safe["inventory"]),
        linear_optics=linear_result,
        radiation=dict(safe.get("radiation", {})),
        raw=safe,
    )


def evaluate_periodic_cell(
    candidate: LatticeIR | Path | str,
    *,
    periods: int | None = None,
    include_profile: bool = False,
) -> PeriodicCellEvaluation:
    """Evaluate one periodic cell without explicitly expanding the full ring.

    This is the fast path for seed solving and optimizer loops.  It returns a
    fixed result schema with explicit linear and radiation gates.  Radiation
    integrals are scaled to the declared identical ring only when the one-cell
    optics are stable and ``periods * cell_bend`` closes to ``2*pi``.  A final
    selected design must still pass :func:`evaluate` on its explicit full ring.
    api.evaluation_snapshot(result) provides the common read-only result schema.
    """

    lattice_ir = load_lattice(candidate) if isinstance(candidate, (Path, str)) else candidate
    lattice_cell_partition(lattice_ir)
    if lattice_ir.metadata.get("evaluation_scope") != "periodic_cell":
        raise LatticeScopeError("evaluate_periodic_cell requires a candidate created by periodic_cell; "
                                "use api.periodic_cell with the confirmed single-cell rows and intended_repetitions")
    declared = periods if periods is not None else lattice_ir.metadata.get("intended_repetitions")
    if declared is None:
        raise LatticeScopeError("periods is required when intended_repetitions is not declared")
    try:
        count = int(declared)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LatticeScopeError("periods must be a positive integer") from exc
    if isinstance(declared, bool) or count < 1 or count != declared:
        raise LatticeScopeError("periods must be a positive integer")

    jt = load_pyjutrack()
    from .jutrack_service import JuTrackService

    mutable = JuTrackService(jt).build_lattice(lattice_ir)
    linear = jutrack_linear_metrics(jt, mutable, include_profile=include_profile)
    if bool(linear.get("stable")):
        for plane in ("x", "y"):
            tune = linear.get(f"tune_{plane}")
            if isinstance(tune, (int, float)) and math.isfinite(float(tune)):
                linear[f"predicted_ring_fractional_tune_{plane}"] = (
                    float(tune) * count
                ) % 1.0
    closure_tolerance_rad = 1.0e-9
    predicted_bend_sum_rad = lattice_ir.bend_sum_rad * count
    closure_error_rad = predicted_bend_sum_rad - 2.0 * math.pi
    closure_passed = abs(closure_error_rad) < closure_tolerance_rad
    if bool(linear.get("stable")) and closure_passed:
        radiation = {
            "computed": True,
            **jutrack_periodic_radiation_metrics(
                jt,
                mutable,
                energy_gev=lattice_ir.energy_gev,
                periods=count,
            ),
        }
    else:
        radiation = {
            "computed": False,
            "reason": (
                "Radiation was not computed because the periodic JuTrack map is unstable."
                if not bool(linear.get("stable"))
                else "Radiation was not computed because the repeated periodic cell does not "
                f"close to 2*pi: error={closure_error_rad:.12g} rad, "
                f"tolerance={closure_tolerance_rad:.12g} rad."
            ),
        }
    result = json_safe(
        {
            "engine": "JuTrack",
            "candidate_lattice_id": lattice_ir.lattice_id,
            "evaluation_scope": "one_periodic_cell_scaled_to_identical_ring",
            "periods": count,
            "geometry": {
                "cell_length_m": lattice_ir.total_length_m,
                "predicted_ring_length_m": lattice_ir.total_length_m * count,
                "cell_bend_sum_rad": lattice_ir.bend_sum_rad,
                "predicted_ring_bend_sum_rad": predicted_bend_sum_rad,
                "ring_closure_error_rad": closure_error_rad,
                "ring_closure_tolerance_rad": closure_tolerance_rad,
                "ring_bend_closure_passed": closure_passed,
            },
            "linear_optics": linear,
            "radiation": radiation,
            "full_ring_validation_required": True,
        }
    )
    from .evaluation_results import evaluated_geometry
    result["evaluation_geometry"] = evaluated_geometry(lattice_ir, reference_periods=count)
    from .candidate_evidence import record_evaluation
    result["evaluation_receipt"] = record_evaluation(lattice_ir, result, effective_periods=count)
    return result


def _linear_evaluation(jt: Any, lattice: LatticeIR) -> dict[str, Any]:
    """Direct JuTrack periodic-optics evaluation used by :func:`evaluate`."""

    from .lattice_optimization import _is_jutrack_twiss_domain_error

    ring = build_jutrack_lattice(lattice)
    if len(ring) != len(lattice.elements):
        raise ValueError("JuTrack lattice does not match the saved element count")
    try:
        matrix = np.asarray(jt.findm66(ring, dp=0.0), dtype=float)
    except ResearchInfeasible as exc:
        return {"stable": False, "trace_x": None, "trace_y": None,
                "twiss_profile": [], "periodic_closure_max": math.inf,
                "model_limits": [f"{type(exc).__name__}: {exc}"]}
    if matrix.shape != (6, 6):
        raise TypeError(f"JuTrack findm66 must return a 6x6 matrix; got {matrix.shape}")
    if not bool(np.all(np.isfinite(matrix))):
        return {
            "stable": False,
            "trace_x": None,
            "trace_y": None,
            "twiss_profile": [],
            "periodic_closure_max": math.inf,
            "model_limits": ["JuTrack returned a non-finite one-turn map."],
        }
    trace_x = float(np.trace(matrix[:2, :2]))
    trace_y = float(np.trace(matrix[2:4, 2:4]))
    if abs(trace_x) >= 2.0 or abs(trace_y) >= 2.0:
        return {
            "stable": False,
            "trace_x": trace_x,
            "trace_y": trace_y,
            "twiss_profile": [],
            "periodic_closure_max": math.inf,
            "one_turn_matrix": matrix.tolist(),
            "model_limits": ["JuTrack one-turn map is linearly unstable."],
        }
    def reject_periodic_optics(reason: str) -> dict[str, Any]:
        return {
            "stable": False, "trace_stable": True,
            "trace_x": trace_x, "trace_y": trace_y, "twiss_profile": [],
            "periodic_closure_max": math.inf, "one_turn_matrix": matrix.tolist(),
            "solver": "JuTrack findm66 + periodicEdwardsTengTwiss + twissline",
            "model_limits": [
                "The map passed the trace test but no finite periodic JuTrack Twiss solution "
                f"was obtained: {reason}"
            ],
        }

    try:
        periodic = jt.periodicEdwardsTengTwiss(ring, 0.0, 0)
        if getattr(periodic, "mode", 1) == 0:
            return reject_periodic_optics("JuTrack periodic Twiss returned invalid mode=0")
        start = _twiss_point("START", 0.0, periodic)
        physical_fields = ("beta_x", "beta_y", "alpha_x", "alpha_y", "Dx", "Dpx", "Dy", "Dpy")
        if (not all(math.isfinite(start[key]) for key in physical_fields)
                or start["beta_x"] <= 0.0 or start["beta_y"] <= 0.0):
            return reject_periodic_optics("periodic Twiss has nonfinite or nonpositive physical optics")
        endpoint = jt.twissline(periodic, ring, dp=0.0, order=0)
        points = jt.twissline(
            periodic,
            ring,
            dp=0.0,
            order=0,
            refpts=list(range(1, len(ring) + 1)),
        )
        if len(points) != len(lattice.elements):
            raise ValueError("JuTrack profile does not cover every element exit")
        profile = [start]
        position = 0.0
        for element, point in zip(lattice.elements, points):
            position += element.length_m
            profile.append(_twiss_point(element.name, position, point))
        end = _twiss_point("END", position, endpoint)
        if any(not all(math.isfinite(row[key]) for key in physical_fields)
               or row["beta_x"] <= 0.0 or row["beta_y"] <= 0.0
               for row in [*profile, end]):
            return reject_periodic_optics("Twiss propagation has nonfinite or nonpositive physical optics")
        start_beta_x = start["beta_x"]
        start_beta_y = start["beta_y"]
        residuals = {
            "beta_x": (float(getattr(endpoint, "betax")) - start_beta_x) / start_beta_x,
            "alpha_x": float(getattr(endpoint, "alphax")) - float(getattr(periodic, "alphax")),
            "beta_y": (float(getattr(endpoint, "betay")) - start_beta_y) / start_beta_y,
            "alpha_y": float(getattr(endpoint, "alphay")) - float(getattr(periodic, "alphay")),
            "Dx": float(getattr(endpoint, "dx")) - float(getattr(periodic, "dx")),
            "Dpx": float(getattr(endpoint, "dpx")) - float(getattr(periodic, "dpx")),
            "Dy": float(getattr(endpoint, "dy")) - float(getattr(periodic, "dy")),
            "Dpy": float(getattr(endpoint, "dpy")) - float(getattr(periodic, "dpy")),
        }
        tune = tuple(float(value) for value in jt.gettune(ring))
        if len(tune) != 2:
            raise TypeError(f"JuTrack gettune must return two tunes; got {len(tune)}")
        if not all(math.isfinite(value) for value in tune):
            return reject_periodic_optics("JuTrack returned a non-finite tune")
        if not all(math.isfinite(value) for value in residuals.values()):
            return reject_periodic_optics("JuTrack returned non-finite periodic residuals")
    except ResearchInfeasible as exc:
        return reject_periodic_optics(f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        if not _is_jutrack_twiss_domain_error(exc):
            raise
        return reject_periodic_optics(f"{type(exc).__name__}: {exc}")
    return {
        "stable": True,
        "trace_stable": True,
        "trace_x": trace_x,
        "trace_y": trace_y,
        "tune_x": tune[0],
        "tune_y": tune[1],
        "periodic_closure_max": max(abs(value) for value in residuals.values()),
        "periodic_residuals": residuals,
        "beta_x_m": start["beta_x"], "beta_y_m": start["beta_y"],
        "alpha_x": start["alpha_x"], "alpha_y": start["alpha_y"],
        "Dx_m": start["Dx"], "Dpx": start["Dpx"],
        "Dy_m": start["Dy"], "Dpy": start["Dpy"],
        "maximum_beta_m": max(max(row["beta_x"], row["beta_y"]) for row in profile),
        "twiss_profile": profile,
        "one_turn_matrix": matrix.tolist(),
        "solver": "JuTrack findm66 + periodicEdwardsTengTwiss + twissline",
        "model_limits": ["JuTrack hard-edge magnetic model represented by the saved lattice."],
    }


def _twiss_point(name: str, position_m: float, point: Any) -> dict[str, Any]:
    row = {
        "name": name,
        "s_m": position_m,
        "beta_x": float(getattr(point, "betax")),
        "alpha_x": float(getattr(point, "alphax")),
        "beta_y": float(getattr(point, "betay")),
        "alpha_y": float(getattr(point, "alphay")),
        "Dx": float(getattr(point, "dx")),
        "Dpx": float(getattr(point, "dpx")),
        "Dy": float(getattr(point, "dy")),
        "Dpy": float(getattr(point, "dpy")),
    }
    for source, target in (("mux", "mu_x_rad"), ("muy", "mu_y_rad")):
        value = getattr(point, source, None)
        if value is not None and math.isfinite(float(value)):
            row[target] = float(value)
    return row


def _radiation_evaluation(
    jt: Any,
    lattice: LatticeIR,
    *,
    ring: Any | None = None,
) -> dict[str, Any]:
    """Direct JuTrack ``ringpara`` evaluation with physical sanity checks."""

    evaluated_ring = build_jutrack_lattice(lattice) if ring is None else ring
    values = jt.ringpara(
        evaluated_ring,
        energy=lattice.energy_gev * 1.0e9,
        print_summary=False,
    )
    result = {
        "computed": True,
        **{key: float(values[key]) for key in ("I1", "I2", "I3", "I4", "I5", "Jx", "Jy")},
    }
    result.update(
        {
            "Jz": float(values["Je"]),
            "emittance_m": float(values["emittx"]),
            "emittance_pm": float(values["emittx"]) * 1.0e12,
            "energy_spread": float(values["sigma_E"]),
            "energy_loss_per_turn_eV": float(values["U0"]),
            "method": "JuTrack ringpara",
            "physical_element_treatment": "intact_original_elements",
            "physical_element_count": len(evaluated_ring),
            "jutrack_ring_parameters": json_safe(dict(values)),
        }
    )
    robinson_sum = result["Jx"] + result["Jy"] + result["Jz"]
    result["robinson_sum"] = robinson_sum
    checks = {
        "finite_radiation_quantities": all(
            math.isfinite(result[key])
            for key in ("I2", "I3", "I4", "I5", "Jx", "Jy", "Jz", "emittance_pm")
        ),
        "positive_I2": result["I2"] > 0.0,
        "positive_I3": result["I3"] > 0.0,
        "nonnegative_I5": result["I5"] >= 0.0,
        "positive_horizontal_damping": result["Jx"] > 0.0,
        "positive_vertical_damping": result["Jy"] > 0.0,
        "positive_longitudinal_damping": result["Jz"] > 0.0,
        "positive_natural_emittance": result["emittance_pm"] > 0.0,
        "robinson_sum": math.isfinite(robinson_sum) and abs(robinson_sum - 4.0) <= 5.0e-3,
    }
    result["physical_validity"] = {
        "passed": all(checks.values()),
        "checks": checks,
        "interpretation": (
            "Physical radiation equilibrium."
            if all(checks.values())
            else "Invalid radiation equilibrium; do not treat the emittance as a valid result."
        ),
    }
    return result


def _bend_radiation_contribution_evaluation(
    jt: Any,
    ring: Any,
    lattice: LatticeIR,
) -> dict[str, Any]:
    """Evaluate exact JuTrack ``elrad`` contributions and group repeated bends."""

    points = jt.twissring(
        ring,
        dp=0.0,
        refpts=list(range(1, len(ring) + 1)),
    )
    if len(points) != len(lattice.elements):
        raise RuntimeError(
            "JuTrack twissring did not return one exit point per lattice element"
        )
    bridge = getattr(jt, "_jl", None)
    if bridge is None:
        bridge = getattr(jt.twissring, "__globals__", {}).get("_jl")
    radiation_module = getattr(bridge, "JuTrack", None) if bridge is not None else None
    radiation_module = radiation_module or bridge
    if radiation_module is None or not hasattr(radiation_module, "elrad"):
        raise RuntimeError("configured pyJuTrack does not expose JuTrack elrad")

    grouped: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(lattice.elements):
        if item.kind not in {"SBEND", "RBEND"} or abs(item.angle_rad or 0.0) <= 0.0:
            continue
        entrance = points[index - 1] if index > 0 else points[-1]
        raw = radiation_module.elrad(ring[index], entrance, points[index])
        values = [float(value) for value in raw]
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"JuTrack elrad returned invalid data for {item.name!r}")
        source_name = str(item.fields.get("source_element") or item.name)
        source_period = item.fields.get("source_period")
        group = grouped.setdefault(
            source_name,
            {
                "bend": source_name,
                "instances": 0,
                "first_element_index": index,
                "periods": [],
                "length_sum_m": 0.0,
                "angle_sum_rad": 0.0,
                "H_weight_m-2": 0.0,
                **{f"I{order}": 0.0 for order in range(1, 7)},
            },
        )
        group["instances"] += 1
        if source_period is not None:
            group["periods"].append(int(source_period))
        group["length_sum_m"] += item.length_m
        group["angle_sum_rad"] += float(item.angle_rad or 0.0)
        curvature = abs(float(item.angle_rad or 0.0) / item.length_m)
        group["H_weight_m-2"] += curvature**3 * item.length_m
        for order, value in enumerate(values[:6], start=1):
            group[f"I{order}"] += value

    if not grouped:
        raise ValueError("candidate contains no nonzero-angle bending magnets")

    totals = {
        key: sum(float(group[key]) for group in grouped.values())
        for key in ("I1", "I2", "I3", "I4", "I5", "I6")
    }
    rows: list[dict[str, Any]] = []
    for group in sorted(grouped.values(), key=lambda row: row["first_element_index"]):
        instances = int(group["instances"])
        h_weight = float(group["H_weight_m-2"])
        row = {
            "bend": str(group["bend"]),
            "instances": instances,
            "length_m": float(group["length_sum_m"]) / instances,
            "angle_rad": float(group["angle_sum_rad"]) / instances,
            "mean_H_m": float(group["I5"]) / h_weight if h_weight > 0.0 else None,
            **{key: float(group[key]) for key in ("I1", "I2", "I3", "I4", "I5")},
            "I2_fraction": (
                float(group["I2"]) / totals["I2"] if totals["I2"] != 0.0 else None
            ),
            "I4_fraction": (
                float(group["I4"]) / totals["I4"] if totals["I4"] != 0.0 else None
            ),
            "I5_fraction": (
                float(group["I5"]) / totals["I5"] if totals["I5"] != 0.0 else None
            ),
            "I5_over_I2": (
                float(group["I5"]) / float(group["I2"])
                if float(group["I2"]) != 0.0
                else None
            ),
        }
        if group["periods"]:
            row["source_periods"] = sorted(set(group["periods"]))
        rows.append(row)

    reference = _radiation_evaluation(jt, lattice, ring=ring)
    checks = {
        key: math.isclose(
            totals[key],
            float(reference[key]),
            rel_tol=2.0e-10,
            abs_tol=1.0e-13,
        )
        for key in ("I2", "I4", "I5")
    }
    return {
        "computed": True,
        "engine": "JuTrack",
        "method": "JuTrack twissring + per-element elrad, checked against ringpara",
        "lattice_id": lattice.lattice_id,
        "grouping": "source_element across repeated periods",
        "rows": rows,
        "totals": totals,
        "ringpara": {key: float(reference[key]) for key in ("I2", "I4", "I5")},
        "consistency": {
            "passed": all(checks.values()),
            "checks": checks,
            "interpretation": (
                "Per-bend sums reproduce the authoritative JuTrack ringpara integrals."
                if all(checks.values())
                else "Per-bend sums do not reproduce ringpara; do not use the decomposition."
            ),
        },
    }


def bend_radiation_contributions(
    candidate: LatticeIR | Path | str,
    *,
    _loaded_jutrack: Any | None = None,
) -> dict[str, Any]:
    """Return per-physical-bend I1--I5 and mean-H evidence from JuTrack.

    Repeated instances are grouped by their archived ``source_element`` name,
    yielding B1...BM rows for an identical-cell ring.  Contributions come from
    JuTrack's own analytical ``elrad`` implementation and are accepted only
    when their I2/I4/I5 sums reproduce the same ring's ``ringpara`` values.

    A ``periodic_cell`` with declared ``intended_repetitions`` is expanded
    unchanged into its identical full ring for this calculation. The returned
    ``lattice_id`` remains the supplied candidate; ``evaluated_lattice_id``
    identifies that derived ring. No period count or partition is inferred.
    """

    lattice = load_lattice(candidate) if isinstance(candidate, (Path, str)) else candidate
    lattice_cell_partition(lattice)
    source_candidate = lattice
    derived_periods = None
    if lattice.metadata.get("evaluation_scope") == "periodic_cell":
        declared = lattice.metadata.get("intended_repetitions")
        if declared is None:
            raise LatticeScopeError(
                "per-bend radiation of a periodic_cell requires declared intended_repetitions; "
                "use api.periodic_cell(..., intended_repetitions=P) or supply an explicit full ring"
            )
        # The partition validator already rejects nonfinite/noninteger metadata;
        # use the same strict numeric declaration as evaluate_periodic_cell.
        derived_periods = int(declared)
        if isinstance(declared, bool) or derived_periods != declared:
            raise LatticeScopeError("intended_repetitions must be a positive integer")
    closure_error = lattice.bend_sum_rad * (derived_periods or 1) - 2.0 * math.pi
    if not math.isfinite(closure_error) or abs(closure_error) >= 1.0e-9:
        raise ValueError(
            "per-bend radiation evidence requires exact 2*pi bend closure; "
            f"error={closure_error:.12g} rad"
        )
    if derived_periods is not None:
        lattice = cell_ring(
            title=f"{lattice.title} — identical ring for per-bend radiation",
            energy_gev=lattice.energy_gev,
            cell=lattice.elements,
            periods=derived_periods,
            source="bend_radiation_contributions: explicit repetition of the supplied periodic cell",
            parent_lattice_id=lattice.lattice_id,
        )
    jt = _loaded_jutrack or load_pyjutrack()
    from .jutrack_service import JuTrackService

    ring = JuTrackService(jt).build_lattice(lattice)
    evidence = json_safe(_bend_radiation_contribution_evaluation(jt, ring, lattice))
    if derived_periods is not None:
        evidence.update({
            "lattice_id": source_candidate.lattice_id,
            "evaluated_lattice_id": lattice.lattice_id,
            "evaluation_scope": "explicit_full_ring_from_declared_periodic_cell",
            "periods": derived_periods,
            "cell_element_count": len(source_candidate.elements),
            "ring_element_count": len(lattice.elements),
            "ring_length_m": lattice.total_length_m,
            "ring_bend_sum_rad": lattice.bend_sum_rad,
        })
    return evidence


def natural_chromaticity(
    candidate: LatticeIR | Path | str,
    *,
    momentum_offset: float = 0.0,
    _loaded_jutrack: Any | None = None,
) -> dict[str, Any]:
    """Return JuTrack's direct natural chromaticity for an explicit lattice.

    This is the authoritative fast path for a chromaticity query.  It wraps
    ``pyJuTrack.getchrom`` so research code does not need to rediscover that
    API or replace it with an approximate beta-weighted quadrupole sum.
    """

    if not math.isfinite(float(momentum_offset)):
        raise ValueError("momentum_offset must be finite")
    lattice_ir = load_lattice(candidate) if isinstance(candidate, (Path, str)) else candidate
    jt = _loaded_jutrack or load_pyjutrack()
    from .jutrack_service import JuTrackService

    ring = JuTrackService(jt).build_lattice(lattice_ir)
    values = jt.getchrom(
        ring,
        dp=float(momentum_offset),
        energy=float(lattice_ir.energy_gev) * 1.0e9,
    )
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise RuntimeError("JuTrack getchrom did not return horizontal and vertical values")
    horizontal, vertical = float(values[0]), float(values[1])
    if not (math.isfinite(horizontal) and math.isfinite(vertical)):
        raise RuntimeError("JuTrack getchrom returned non-finite chromaticity")
    repetitions = int(lattice_ir.metadata.get("repetitions", lattice_ir.periodicity))
    scope = str(lattice_ir.metadata.get("evaluation_scope", "full_ring"))
    return {
        "engine": "JuTrack",
        "method": "pyJuTrack.getchrom",
        "lattice_id": lattice_ir.lattice_id,
        "evaluation_scope": scope,
        "momentum_offset": float(momentum_offset),
        "horizontal": horizontal,
        "vertical": vertical,
        "per_period_horizontal": (
            horizontal / repetitions if scope == "full_ring" and repetitions > 0 else horizontal
        ),
        "per_period_vertical": (
            vertical / repetitions if scope == "full_ring" and repetitions > 0 else vertical
        ),
        "periods": repetitions,
    }


def dynamic_aperture(
    candidate_or_ring: LatticeIR | Path | str | Any,
    *,
    nturns: int = 1024,
    amp_max_m: float = 0.025,
    amp_step_m: float = 0.0005,
    angle_steps: int = 20,
    energy_gev: float | None = None,
    momentum_offset: float = 0.0,
    _loaded_jutrack: Any | None = None,
) -> dict[str, Any]:
    """Use JuTrack's native dynamic-aperture calculation on a lattice or ring.

    The positional bridge fallback handles pyJuTrack versions whose public
    wrapper accepts keyword arguments while the Julia method is positional.
    """

    if nturns < 1 or angle_steps < 2:
        raise ValueError("nturns must be positive and angle_steps must be at least two")
    if amp_max_m <= 0.0 or amp_step_m <= 0.0:
        raise ValueError("dynamic-aperture amplitudes must be positive")
    jt = _loaded_jutrack or load_pyjutrack()
    lattice_ir = None
    if isinstance(candidate_or_ring, (LatticeIR, Path, str)):
        lattice_ir = (
            load_lattice(candidate_or_ring)
            if isinstance(candidate_or_ring, (Path, str))
            else candidate_or_ring
        )
        ring = build_jutrack_lattice(lattice_ir)
        energy_gev = lattice_ir.energy_gev if energy_gev is None else energy_gev
    else:
        ring = candidate_or_ring
    energy_ev = float(energy_gev if energy_gev is not None else 3.0) * 1.0e9
    kwargs = {
        "nturns": int(nturns),
        "amp_max": float(amp_max_m),
        "amp_step": float(amp_step_m),
        "angle_steps": int(angle_steps),
        "E": energy_ev,
        "dp": float(momentum_offset),
    }
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            try:
                raw, _ = jt.dynamic_aperture(ring, **kwargs)
            except Exception:
                bridge = getattr(jt, "_jl", None)
                if bridge is None:
                    bridge = getattr(jt.dynamic_aperture, "__globals__", {}).get("_jl")
                if bridge is None:
                    raise
                raw, _ = bridge.dynamic_aperture(
                    getattr(ring, "julia_object", ring),
                    int(nturns),
                    float(amp_max_m),
                    float(amp_step_m),
                    int(angle_steps),
                    energy_ev,
                    float(momentum_offset),
                )
        boundary = [
            [float(row[0]), float(row[1])]
            for row in np.asarray(raw, dtype=float)
        ]
    except Exception as exc:
        return {
            "engine": "JuTrack",
            "method": "jt.dynamic_aperture",
            "passed": False,
            "boundary_m": [],
            "error": f"{type(exc).__name__}: {exc}",
            "nturns": int(nturns),
            "momentum_offset": float(momentum_offset),
        }
    points = np.asarray(boundary, dtype=float)
    xs_mm = points[:, 0] * 1.0e3
    ys_mm = points[:, 1] * 1.0e3
    area_mm2 = 0.5 * abs(
        float(np.dot(xs_mm, np.roll(ys_mm, -1)) - np.dot(ys_mm, np.roll(xs_mm, -1)))
    )
    radii_mm = np.hypot(xs_mm, ys_mm)
    return {
        "engine": "JuTrack",
        "method": "jt.dynamic_aperture",
        "passed": bool(len(points) >= 2 and np.all(np.isfinite(points))),
        "boundary_m": boundary,
        # Legacy alias retained for archived plotting consumers.
        "da_boundary": boundary,
        "area_mm2": area_mm2,
        "da_area_mm2": area_mm2,
        "horizontal_mm": float(abs(xs_mm[int(np.argmin(np.abs(ys_mm)))])),
        "vertical_mm": float(abs(ys_mm[int(np.argmin(np.abs(xs_mm)))])),
        "minimum_radius_mm": float(np.min(radii_mm)) if len(radii_mm) else 0.0,
        "nturns": int(nturns),
        "momentum_offset": float(momentum_offset),
        "energy_gev": energy_ev / 1.0e9,
        "lattice_id": lattice_ir.lattice_id if lattice_ir is not None else None,
    }


def lattice_inventory(candidate: LatticeIR) -> dict[str, Any]:
    bends = [
        item
        for item in candidate.elements
        if item.kind in {"SBEND", "RBEND"} and abs(item.angle_rad or 0.0) > 0.0
    ]
    positive = [item for item in bends if (item.angle_rad or 0.0) > 0.0]
    negative = [item for item in bends if (item.angle_rad or 0.0) < 0.0]
    repetitions = int(candidate.metadata.get("repetitions", candidate.periodicity))
    return {
        "element_count": len(candidate.elements),
        "physical_bend_count": len(bends),
        "positive_bend_count": len(positive),
        "negative_bend_count": len(negative),
        "periodicity": candidate.periodicity,
        "declared_repetitions": repetitions,
        "positive_bends_per_period": (len(positive) / repetitions if repetitions > 0 else None),
        "negative_bends_per_period": (len(negative) / repetitions if repetitions > 0 else None),
        "quadrupole_count": sum(item.kind in {"KQUAD", "QUAD"} for item in candidate.elements),
        "sextupole_count": sum(item.kind == "KSEXT" for item in candidate.elements),
    }


def optics_lattice_plot_data(
    profile: Iterable[Mapping[str, Any]],
    elements: Iterable[Mapping[str, Any]],
    *,
    max_points: int = 1400,
) -> dict[str, Any]:
    """Return one-cell beta, dispersion, and lattice-strip data on one s axis."""

    from .plotting import optics_lattice_plot_data as build_plot_data

    return build_plot_data(profile, elements, max_points=max_points)


def save_optics_lattice_plot(
    path: Path | str,
    profile: Iterable[Mapping[str, Any]],
    elements: Iterable[Mapping[str, Any]],
    *,
    title: str = "One-cell optics and lattice",
    subtitle: str = "",
    max_points: int = 1400,
) -> Path:
    """Save the canonical combined beta/dispersion/lattice artifact.

    SVG uses the dependency-free renderer. PNG and PDF use Matplotlib. All
    formats consume the same plot payload and therefore retain beta functions,
    dispersion, and magnet glyphs on one longitudinal scope.
    """

    destination = Path(path)
    payload = optics_lattice_plot_data(profile, elements, max_points=max_points)
    if destination.suffix.lower() == ".svg":
        from .plotting import render_optics_lattice_svg

        return render_optics_lattice_svg(
            payload, destination, title=title, subtitle=subtitle
        )
    if destination.suffix.lower() in {".png", ".pdf"}:
        from .plotting import render_optics_lattice_matplotlib

        return render_optics_lattice_matplotlib(
            payload, destination, title=title, subtitle=subtitle
        )
    raise ValueError("combined optics/lattice plot path must end in .svg, .png, or .pdf")


def save_bend_radiation_contribution_plot(
    path: Path | str,
    evidence: Mapping[str, Any],
    *,
    title: str = "Per-bend radiation-integral contributions",
    subtitle: str = "I2 and I5 fractions of the full ring",
) -> Path:
    """Save measured per-bend I2/I5 fractions as SVG, PNG, or PDF.

    SVG is dependency-free; PNG/PDF use headless Matplotlib. All formats plot
    the same saved rows without recalculating or renormalizing the integrals.
    """

    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".svg", ".png", ".pdf"}:
        raise ValueError("bend-radiation plot path must end in .svg, .png, or .pdf")
    if suffix == ".svg":
        from .plotting import render_bend_radiation_contributions_svg as render
    else:
        from .plotting import render_bend_radiation_contributions_matplotlib as render
    return render(evidence, destination, title=title, subtitle=subtitle)


def ring_survey_plot_rows(
    elements: Iterable[Mapping[str, Any]],
    *,
    bend_samples: int = 4,
) -> list[dict[str, Any]]:
    """Return a floor-plan survey derived from the explicit lattice definition."""

    from .plotting import ring_plot_rows

    return ring_plot_rows(elements, bend_samples=bend_samples)


def particle_swarm_optimize(
    evaluate: Any,
    *,
    bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    objective_names: Sequence[str],
    initial_position: Sequence[float] | Mapping[str, float],
    variable_names: Sequence[str] | None = None,
    population_size: int | None = None,
    generations: int | None = None,
    seed: int = 0,
    selection: str = "knee",
    seed_positions: Sequence[Sequence[float]] | None = None,
    initial_velocity_fraction: float = 0.10,
    progress_callback: Any | None = None,
    evaluate_batch: Any | None = None,
    evaluation_batch_size: int | None = None,
    cancellation_check: Any | None = None,
    checkpoint_path: str | Path | None = None,
    evaluation_cache_directory: str | Path | None = None,
    cache_evaluations: bool = True,
    campaign_identity: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the public bounded PSO/parent-offspring optimization kernel.

    The callback owns all physics evaluation. For multiple objectives the
    optimizer combines parents and PSO-generated offspring, then performs
    constrained nondominated sorting and crowding selection.

    Defaults are max(500, 10 * number_of_knobs) particles and 80 generations.
    Managed research enforces scientist-configured [optimization] floors and
    records requested/effective budgets. Resuming preserves the saved swarm.
    Set cache_evaluations=False for fast callbacks to disable per-evaluation
    cache files (ignoring any cache directory), but keep generation checkpoints.
    """

    from .population_optimization import particle_swarm_optimize as optimize

    return optimize(
        evaluate,
        bounds=bounds,
        objective_names=objective_names,
        initial_position=initial_position,
        variable_names=variable_names,
        population_size=population_size,
        generations=generations,
        seed=seed,
        selection=selection,
        seed_positions=seed_positions,
        initial_velocity_fraction=initial_velocity_fraction,
        progress_callback=progress_callback,
        evaluate_batch=evaluate_batch,
        evaluation_batch_size=evaluation_batch_size,
        cancellation_check=cancellation_check,
        checkpoint_path=checkpoint_path,
        evaluation_cache_directory=evaluation_cache_directory,
        cache_evaluations=cache_evaluations,
        campaign_identity=campaign_identity,
        resume=resume,
    )


def process_batch_evaluator(
    evaluator_factory: Any,
    *,
    workers: int | None = None,
    verbose: int = 0,
) -> Any:
    """Create a persistent process-isolated evaluator for PSO/MOPSO.

    The factory runs once in each worker process.  JuTrack users must create
    the pyJuTrack module, lattice, and mutable editor inside that factory;
    live Julia objects are deliberately never transferred between processes.
    """

    from .population_optimization import ProcessBatchEvaluator
    from .research_environment import ResearchEnvironment

    if workers is None:
        environment = ResearchEnvironment.load()
        workers = (
            environment.local_worker_processes
            if environment.compute_backend == "local_process"
            else 1
        )
    return ProcessBatchEvaluator(evaluator_factory, workers=workers, verbose=verbose)


def parallel_evaluator(evaluator_factory: Any, *, workers: int | None = None,
                       on_result: Any = None, cancellation_check: Any = None) -> Any:
    """Resource-budgeted process evaluator for arbitrary scans and optimizers.

    The zero-argument factory initializes JuTrack inside each isolated worker.
    The returned context manager accepts batches and returns ordered results.
    on_result(index, result, status) executes only in the coordinator.
    """
    from .parallel_evaluation import ParallelEvaluator
    return ParallelEvaluator(evaluator_factory, workers=workers, on_result=on_result,
                             cancellation_check=cancellation_check)


# General lattice-optimization building blocks are re-exported here so model-
# written accelerator studies need only the stable public API.  JuTrack physics
# remains in callbacks supplied by the study; these helpers own safe lattice
# mutation, symmetry, continuation matching, and the outer population campaign.
from .experiment_candidates import CandidateLedger  # noqa: E402,F401
from .experiment_geometry import (  # noqa: E402,F401
    prepare_length_budget,
    remove_mirrored_quadrupole_pair,
)
from .experiments import (  # noqa: E402,F401
    ExperimentDefinition,
    JuTrackCellEvaluator,
    LatticeExperiment,
    MatchingOperation,
)
from .lattice_optimization import (  # noqa: E402,F401
    JuTrackLatticeEditor,
    LatticeKnob,
    LatticeParameterization,
    LinearSumConstraint,
    bounded_fast_response_match,
    bounded_response_match,
    compile_family_parameterization,
    compile_lattice_parameterization,
    continuation_match,
    feasible_basin_seed_positions,
    infer_mirror_symmetric_parameterization,
    jutrack_linear_metrics,
    jutrack_periodic_radiation_metrics,
    jutrack_radiation_metrics,
    local_seed_positions,
    optimize_lattice_parameters,
    plan_lattice_optimization,
    search_feasible_parameter_seeds,
)
from .line_matching import match_jutrack_line, plan_matching_starts  # noqa: E402,F401
from .line_optics import JuTrackLineEvaluator, evaluate_jutrack_line  # noqa: E402,F401


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
