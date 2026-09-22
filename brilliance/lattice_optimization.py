"""General parameterization and nested matching for lattice optimization.

The population optimizer should not know accelerator physics.  This module is
the bridge between a numerical position vector and an ordered lattice: it owns
element-field knobs, exact linear sum constraints, mirror symmetry, and an
optional in-objective matching operation.  JuTrack remains behind callbacks
provided by the caller.

The API is intentionally usable from model-written research code.  A research
agent can declare knobs and objectives for a new problem without adding a new
template to Brilliance, while the fragile parts (indexing, symmetry, bounds,
continuation, and response matching) remain tested platform code.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .lattice import LatticeInputError, require_same_lattice_topology
from .population_optimization import (
    OptimizerInputError,
    ProcessBatchEvaluator,
    _publish_progress,
    particle_swarm_optimize,
)
from .research_outcomes import ResearchInfeasible

LatticeRows = Sequence[Mapping[str, Any]]
MetricEvaluator = Callable[[list[dict[str, Any]]], Mapping[str, Any]]
ObjectiveEvaluator = Callable[[list[dict[str, Any]]], Mapping[str, Any]]
PositionBatchEvaluator = Callable[[Sequence[np.ndarray]], Sequence[Mapping[str, Any]]]
PositionBatchEvaluatorFactory = Callable[
    [list[dict[str, Any]], list[float]], PositionBatchEvaluator
]
WorkerPhysicsFactory = Callable[
    [list[dict[str, Any]]], tuple[MetricEvaluator, ObjectiveEvaluator]
]

_PHYSICAL_FIELDS = {"length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4"}
_FIELD_KINDS = {
    "length_m": None,
    "angle_rad": {"SBEND", "RBEND"},
    "k1_m2": {"KQUAD"},
    "k2_m3": {"KSEXT"},
    "k3_m4": {"KOCT"},
}


def _rows(elements: LatticeRows) -> list[dict[str, Any]]:
    rows = [copy.deepcopy(dict(item)) for item in elements]
    if not rows:
        raise LatticeInputError("lattice parameterization requires at least one element")
    names: set[str] = set()
    for index, row in enumerate(rows):
        name = str(row.get("name", "")).strip()
        if not name:
            raise LatticeInputError(f"element {index} needs a name")
        if name in names:
            raise LatticeInputError(f"duplicate element name {name!r}")
        names.add(name)
        row["name"] = name
        row["kind"] = str(row.get("kind", row.get("type", ""))).upper().strip()
        fields = row.get("fields")
        if fields is None:
            row["fields"] = {}
        elif not isinstance(fields, Mapping):
            raise LatticeInputError(f"{name}.fields must be an object")
        else:
            row["fields"] = copy.deepcopy(dict(fields))
    return rows


def _metadata(row: Mapping[str, Any], key: str) -> Any:
    fields = row.get("fields")
    if isinstance(fields, Mapping) and key in fields:
        return fields[key]
    return row.get(key)


def _field_value(row: Mapping[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        aliases = {
            "angle_rad": "angle",
            "k1_m2": "k1",
            "k2_m3": "k2",
            "k3_m4": "k3",
        }
        value = row.get(aliases.get(field, ""))
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{row.get('name', 'element')}.{field} must be finite")
    return number


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True, slots=True)
class LatticeKnob:
    """One scalar variable applied to one element or a symmetry-related group."""

    name: str
    field: str
    target_indices: tuple[int, ...]
    target_names: tuple[str, ...]
    lower: float
    upper: float
    initial: float
    unit: str = ""
    role: str = "outer_design"
    relabel_family: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "field": self.field,
            "target_indices": list(self.target_indices),
            "target_names": list(self.target_names),
            "bounds": [self.lower, self.upper],
            "initial": self.initial,
            "unit": self.unit,
            "role": self.role,
            "relabel_family": self.relabel_family,
        }


@dataclass(frozen=True, slots=True)
class LinearSumConstraint:
    """Project a set of physical element fields onto an exact requested sum."""

    field: str
    target: float
    variable_names: tuple[str, ...]
    mode: str = "scale"

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "target": self.target,
            "variable_names": list(self.variable_names),
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class LatticeParameterization:
    """Compiled map from a bounded vector to an explicit ordered lattice."""

    baseline: tuple[dict[str, Any], ...]
    knobs: tuple[LatticeKnob, ...]
    sum_constraints: tuple[LinearSumConstraint, ...] = ()

    @property
    def variable_names(self) -> list[str]:
        return [item.name for item in self.knobs]

    @property
    def bounds(self) -> list[list[float]]:
        return [[item.lower, item.upper] for item in self.knobs]

    @property
    def initial_position(self) -> list[float]:
        return [item.initial for item in self.knobs]

    def as_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for knob in self.knobs:
            counts[knob.field] = counts.get(knob.field, 0) + 1
        return {
            "variable_count": len(self.knobs),
            "variable_count_by_field": counts,
            "variables": [item.as_dict() for item in self.knobs],
            "sum_constraints": [item.as_dict() for item in self.sum_constraints],
        }

    def projected_position(self, position: Sequence[float]) -> list[float]:
        """Return physical knob values after exact sum projections."""

        values = np.asarray(position, dtype=float)
        if values.shape != (len(self.knobs),) or not np.all(np.isfinite(values)):
            raise ValueError("position must contain one finite value per lattice knob")
        lower = np.asarray([item.lower for item in self.knobs], dtype=float)
        upper = np.asarray([item.upper for item in self.knobs], dtype=float)
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("lattice position lies outside its declared bounds")
        projected = values.copy()
        index_by_name = {item.name: index for index, item in enumerate(self.knobs)}
        for constraint in self.sum_constraints:
            knob_indices = [index_by_name[name] for name in constraint.variable_names]
            if constraint.mode == "bounded_sum":
                from .experiment_geometry import project_weighted_sum

                projected[knob_indices] = project_weighted_sum(
                    projected[knob_indices], lower[knob_indices], upper[knob_indices],
                    [len(self.knobs[index].target_indices) for index in knob_indices],
                    constraint.target,
                )
                continue
            current = sum(
                projected[index] * len(self.knobs[index].target_indices) for index in knob_indices
            )
            if constraint.mode != "scale":
                raise ValueError(f"unsupported sum-constraint mode {constraint.mode!r}")
            if not math.isfinite(float(current)) or abs(float(current)) <= 1.0e-15:
                raise ValueError(
                    f"cannot scale {constraint.field} sum {current!r} to {constraint.target}"
                )
            factor = constraint.target / float(current)
            projected[knob_indices] *= factor
        if np.any(projected < lower - 1.0e-12) or np.any(projected > upper + 1.0e-12):
            raise ValueError(
                "exact sum projection moves at least one lattice knob outside its bounds"
            )
        return projected.tolist()

    def apply(
        self,
        position: Sequence[float],
        *,
        base_elements: LatticeRows | None = None,
    ) -> list[dict[str, Any]]:
        values = self.projected_position(position)
        rows = _rows(base_elements if base_elements is not None else self.baseline)
        if len(rows) != len(self.baseline):
            raise ValueError("base_elements does not match the compiled lattice length")
        for index, (row, baseline) in enumerate(zip(rows, self.baseline)):
            if str(row["name"]) != str(baseline["name"]):
                raise ValueError(f"base element {index} does not match the compiled lattice")
        for knob, value in zip(self.knobs, values):
            for index in knob.target_indices:
                rows[index][knob.field] = float(value)
                if knob.relabel_family:
                    fields = dict(rows[index].get("fields") or {})
                    fields["source_family"] = fields.get("source_family", fields.get("family"))
                    fields["family"] = knob.name
                    fields["optimization_family"] = knob.name
                    rows[index]["fields"] = fields
        return rows


class JuTrackLatticeEditor:
    """Compile JuTrack name lookups once and mutate element fields in place.

    ``pyJuTrack.findelem`` returns Julia-style 1-based indices, whereas its
    Python ``Lattice`` wrapper uses 0-based indexing.  The conversion happens
    here once.  Optimization code can then work only with element names and SI
    field names and cannot accidentally edit the neighbouring magnet.
    """

    _DIRECT_FIELDS = {
        "length_m": ("len", None),
        "angle_rad": ("angle", None),
        "k1_m2": ("k1", 1),
        "k2_m3": ("k2", 2),
        "k3_m4": ("k3", 3),
    }

    def __init__(self, jt: Any, lattice: Any, elements: LatticeRows):
        self.jt = jt
        self.lattice = lattice
        self.rows = _rows(elements)
        self.index_by_name: dict[str, int] = {}
        for row in self.rows:
            name = str(row["name"])
            found = [int(value) for value in jt.findelem(lattice, name=name)]
            if len(found) != 1:
                raise ValueError(
                    f"JuTrack lattice needs exactly one element named {name!r}; found {found}"
                )
            self.index_by_name[name] = found[0] - 1

    def _set(self, element: Any, field: str, value: float) -> None:
        attribute, polynomial_order = self._DIRECT_FIELDS[field]
        number = float(value)
        if hasattr(element, attribute):
            setattr(element, attribute, number)
            return
        if polynomial_order is not None and hasattr(element, "PolynomB"):
            element.PolynomB[polynomial_order] = number
            if hasattr(element, "MaxOrder"):
                element.MaxOrder = max(int(element.MaxOrder), polynomial_order)
            return
        raise AttributeError(f"JuTrack element {element.name!s} has no mutable field for {field}")

    def _get(self, element: Any, field: str) -> float | None:
        attribute, polynomial_order = self._DIRECT_FIELDS[field]
        if hasattr(element, attribute):
            return float(getattr(element, attribute))
        if polynomial_order is not None and hasattr(element, "PolynomB"):
            return float(element.PolynomB[polynomial_order])
        return None

    def sync_rows(self, elements: LatticeRows, *, lattice: Any | None = None) -> Any:
        """Set scalar fields; the editor is bound to one element count/order/type.

        Insertion/removal/splitting requires a new editor. Identity is validated
        for ALL rows before any field is changed, including same-count type edits.
        """

        target = self.lattice if lattice is None else lattice
        rows = _rows(elements)
        require_same_lattice_topology(self.rows, rows)
        from .jutrack_service import native_element_options
        from .lattice import LatticeElementIR
        for original, proposed in zip(self.rows, rows):
            if native_element_options(LatticeElementIR.from_mapping(original)) != native_element_options(LatticeElementIR.from_mapping(proposed)):
                raise LatticeInputError(
                    f"{proposed['name']}: native constructor fields changed; build a new JuTrack editor. "
                    "No scalar fields were changed."
                )
        for row in rows:
            name = str(row["name"])
            element = target[self.index_by_name[name]]
            for field in _PHYSICAL_FIELDS:
                value = _field_value(row, field)
                if value is not None:
                    self._set(element, field, value)
        return target

    def linear_metrics(
        self,
        elements: LatticeRows | None = None,
        *,
        include_profile: bool = False,
    ) -> dict[str, Any]:
        """Apply optional rows and evaluate the existing mutable JuTrack lattice."""

        if elements is not None:
            self.sync_rows(elements)
        return jutrack_linear_metrics(
            self.jt, self.lattice, include_profile=include_profile
        )

    def evaluate_candidate(
        self,
        elements: LatticeRows,
        *,
        energy_gev: float,
        periods: int,
        include_profile: bool = True,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Apply one explicit candidate and bind optics/radiation to its snapshot.

        Retain this result, not a position paired with the editor's later state.
        Radiation is skipped for an unstable candidate. Programming/API errors
        propagate rather than being mislabeled as infeasible physics. This is a
        sequential transaction: parallel workers must each own their editor.
        A selected periodic candidate still needs independent full-ring validation.
        output_path atomically saves the exact result before caller reporting.
        Use api.evaluation_snapshot(result) for safe missing-value handling.
        """
        if not math.isfinite(energy_gev) or energy_gev <= 0:
            raise ValueError("energy_gev must be finite and positive")
        if isinstance(periods, bool) or int(periods) != periods or periods < 1:
            raise ValueError("periods must be a positive integer")
        if elements is None:
            raise ValueError("evaluate_candidate requires explicit candidate elements")
        self.sync_rows(elements)
        snapshot = self.export_rows(base_elements=elements)
        linear = self.linear_metrics(include_profile=include_profile)
        radiation = None
        if linear.get("stable") is True:
            try:
                radiation = jutrack_periodic_radiation_metrics(
                    self.jt, self.lattice, energy_gev=energy_gev, periods=periods,
                )
            except Exception as exc:
                # Reject this proposal, not the whole search. Do not claim a
                # physical diagnosis or hide API/bridge/programming failures.
                if not (type(exc).__name__ == "JuliaError"
                        and type(exc).__module__.split(".")[0] == "juliacall"
                        and "DomainError" in str(exc)):
                    raise
                radiation = {"computed": False, "reason": "JuTrack ringpara domain error",
                             "error": str(exc), "physical_validity": {"passed": False},
                             "diagnosis": "Cause not established; exact trial elements retained."}
        result = {
            "schema": "brilliance_editor_evaluation/v1",
            "engine": "JuTrack",
            "elements": snapshot,
            "energy_gev": float(energy_gev),
            "periods": int(periods),
            "linear_optics": linear,
            "radiation": radiation,
            "passed": bool(
                radiation is not None
                and radiation.get("computed") is True
                and radiation.get("physical_validity", {}).get("passed") is True
            ),
            "full_ring_validation_required": True,
        }
        result["status"] = "evaluated" if result["passed"] else "rejected"
        result["issues"] = ([] if result["passed"] else [str(
            linear.get("reason") or linear.get("failure") or
            ("No stable periodic optics; boundary optics and radiation are unavailable."
             if linear.get("stable") is not True else
             (radiation or {}).get("reason") or "Radiation validity checks did not pass."))])
        if output_path is not None:
            from .artifacts import json_safe
            from .atomic_io import atomic_write_text
            atomic_write_text(Path(output_path), json.dumps(json_safe(result), ensure_ascii=False, indent=2))
        return result

    def apply_parameterization(
        self,
        parameterization: LatticeParameterization,
        position: Sequence[float],
        *,
        lattice: Any | None = None,
    ) -> Any:
        """Apply compiled knobs directly without rebuilding JuTrack elements."""

        target = self.lattice if lattice is None else lattice
        values = parameterization.projected_position(position)
        for knob, value in zip(parameterization.knobs, values):
            for name in knob.target_names:
                self._set(target[self.index_by_name[name]], knob.field, value)
        return target

    def export_rows(
        self,
        *,
        lattice: Any | None = None,
        base_elements: LatticeRows | None = None,
    ) -> list[dict[str, Any]]:
        """Materialize the mutable JuTrack state for the archive boundary."""

        source = self.rows if base_elements is None else _rows(base_elements)
        target = self.lattice if lattice is None else lattice
        result = _rows(source)
        for row in result:
            element = target[self.index_by_name[str(row["name"])]]
            for field in _PHYSICAL_FIELDS:
                value = self._get(element, field)
                if value is not None and (
                    field == "length_m" or _field_value(row, field) is not None
                ):
                    row[field] = value
        return result


def _is_jutrack_twiss_domain_error(exc: Exception) -> bool:
    """Recognize only JuTrack's documented propagation square-root failure."""
    return (type(exc).__name__ == "JuliaError"
            and type(exc).__module__.split(".")[0] == "juliacall"
            and "DomainError" in str(exc)
            and "negative sqrt argument in twissPropagate" in str(exc))


def jutrack_linear_metrics(
    jt: Any,
    lattice: Any,
    *,
    include_profile: bool = False,
) -> dict[str, Any]:
    """Evaluate a mutable JuTrack lattice without rebuilding a LatticeIR."""

    try:
        matrix = np.asarray(jt.findm66(lattice, dp=0.0), dtype=float)
        if matrix.shape != (6, 6):
            raise TypeError(f"JuTrack findm66 must return a 6x6 matrix; got {matrix.shape}")
        trace_x = float(np.trace(matrix[:2, :2]))
        trace_y = float(np.trace(matrix[2:4, 2:4]))
    except ResearchInfeasible as exc:
        return _physical_rejection(exc)
    if (
        not np.all(np.isfinite(matrix))
        or abs(trace_x) >= 2.0
        or abs(trace_y) >= 2.0
    ):
        return {"stable": False, "trace_x": trace_x, "trace_y": trace_y}
    try:
        periodic = jt.periodicEdwardsTengTwiss(lattice, 0.0, 0)
        # JuTrack explicitly returns mode=0 for an undecouplable/unstable map.
        # Do not confuse this sentinel with a failed bridge call or bad API.
        if getattr(periodic, "mode", 1) == 0:
            return {"stable": False, "trace_x": trace_x, "trace_y": trace_y,
                    "reason": "JuTrack periodic Twiss returned invalid mode=0"}
        tune = [float(item) for item in jt.gettune(lattice)]
        if len(tune) != 2:
            raise TypeError(f"JuTrack gettune must return two tunes; got {len(tune)}")
        metrics: dict[str, Any] = {
            "stable": True,
            "trace_x": trace_x,
            "trace_y": trace_y,
            "tune_x": tune[0],
            "tune_y": tune[1],
            "Dx_m": float(periodic.dx),
            "Dpx": float(periodic.dpx),
            "Dy_m": float(periodic.dy),
            "Dpy": float(periodic.dpy),
            "beta_x_m": float(periodic.betax),
            "beta_y_m": float(periodic.betay),
            "alpha_x": float(periodic.alphax),
            "alpha_y": float(periodic.alphay),
        }
        if (not all(math.isfinite(value) for key, value in metrics.items() if key != "stable")
                or metrics["beta_x_m"] <= 0 or metrics["beta_y_m"] <= 0):
            return {"stable": False, "trace_x": trace_x, "trace_y": trace_y,
                    "reason": "JuTrack periodic Twiss has nonfinite or nonpositive physical optics"}
        metrics["boundary_optics"] = {key: metrics[key] for key in (
            "Dx_m", "Dpx", "Dy_m", "Dpy", "beta_x_m", "beta_y_m", "alpha_x", "alpha_y")}
        metrics["boundary_location"] = "periodic lattice entrance, s=0"
        metrics["profile_sampling"] = "not_evaluated"
        metrics["maximum_beta_m"] = None
        if include_profile:
            points = jt.twissline(
                periodic,
                lattice,
                dp=0.0,
                order=0,
                refpts=list(range(1, len(lattice) + 1)),
            )
            def profile_row(point: Any, name: str, s_m: float) -> dict[str, Any]:
                return {
                    "name": name, "s_m": s_m,
                    "beta_x": float(point.betax), "beta_y": float(point.betay),
                    "alpha_x": float(point.alphax), "alpha_y": float(point.alphay),
                    "Dx": float(point.dx), "Dpx": float(point.dpx),
                    "Dy": float(point.dy), "Dpy": float(point.dpy),
                }

            # Same positional contract as evaluate(): START, then element exits.
            # Values without coordinates cannot be plotted or assigned to a bend.
            profile = [profile_row(periodic, "START", 0.0)]
            s_m = 0.0
            if len(points) != len(lattice):
                raise ValueError("JuTrack profile does not cover every element exit")
            for index, point in enumerate(points):
                element = lattice[index]
                s_m += float(element.len)
                profile.append(profile_row(point, str(element.name), s_m))
            if any(not math.isfinite(row[key]) for row in profile
                   for key in ("beta_x", "beta_y", "Dx", "Dpx")) or any(
                       row["beta_x"] <= 0 or row["beta_y"] <= 0 for row in profile):
                return {**metrics, "stable": False,
                        "reason": "JuTrack sampled profile contains nonphysical optics"}
            metrics["profile_sampling"] = "entrance_and_element_exits; interior extrema not guaranteed"
            metrics["maximum_beta_x_m"] = max(row["beta_x"] for row in profile)
            metrics["maximum_beta_y_m"] = max(row["beta_y"] for row in profile)
            metrics["maximum_beta_m"] = max(
                [metrics["beta_x_m"], metrics["beta_y_m"]]
                + [value for row in profile for value in (row["beta_x"], row["beta_y"])]
            )
            metrics["twiss_profile"] = profile
        return metrics
    except ResearchInfeasible as exc:
        return _physical_rejection(exc)
    except Exception as exc:
        # This is the specific physical-domain failure emitted by JuTrack's
        # _checked_sqrt in EdwardsTengTwiss.jl. Never classify arbitrary Julia
        # MethodError, Python RuntimeError, or profile-shape ValueError as optics.
        if not _is_jutrack_twiss_domain_error(exc):
            raise
        return {
            "stable": False,
            "trace_x": trace_x,
            "trace_y": trace_y,
            "failure": f"{type(exc).__name__}: {exc}",
        }


def radiation_physical_validity(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate whether radiation data describe a physical damped equilibrium.

    JuTrack remains the numerical authority.  This gate does not alter any
    ``ringpara`` value; it prevents a mathematically returned anti-damped or
    otherwise nonphysical result from being ranked as a low-emittance design.
    """

    values = {
        name: _finite_number(metrics.get(name))
        for name in ("I2", "I3", "I4", "I5", "Jx", "Jy", "Jz", "emittance_pm")
    }
    robinson_sum = (
        values["Jx"] + values["Jy"] + values["Jz"]
        if all(values[name] is not None for name in ("Jx", "Jy", "Jz"))
        else None
    )
    checks = {
        "finite_radiation_integrals": all(
            values[name] is not None for name in ("I2", "I3", "I4", "I5")
        ),
        "positive_I2": values["I2"] is not None and values["I2"] > 0.0,
        "positive_I3": values["I3"] is not None and values["I3"] > 0.0,
        "nonnegative_I5": values["I5"] is not None and values["I5"] >= 0.0,
        "positive_horizontal_damping": values["Jx"] is not None and values["Jx"] > 0.0,
        "positive_vertical_damping": values["Jy"] is not None and values["Jy"] > 0.0,
        "positive_longitudinal_damping": values["Jz"] is not None and values["Jz"] > 0.0,
        "positive_natural_emittance": (
            values["emittance_pm"] is not None and values["emittance_pm"] > 0.0
        ),
        "robinson_sum_consistent": (robinson_sum is not None and abs(robinson_sum - 4.0) <= 5.0e-3),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "robinson_sum": robinson_sum,
        "interpretation": (
            "physical damped equilibrium"
            if all(checks.values())
            else "invalid radiation equilibrium; do not rank or publish its emittance"
        ),
    }


def jutrack_radiation_metrics(jt: Any, lattice: Any, *, energy_gev: float) -> dict[str, Any]:
    """Return direct ``ringpara`` quantities with the same status as api.evaluate.

    ``computed`` means the call returned numerical data, NOT a valid equilibrium.
    All callers must separately check ``physical_validity['passed']``.
    """

    values = jt.ringpara(
        lattice,
        energy=float(energy_gev) * 1.0e9,
        print_summary=False,
    )
    metrics: dict[str, Any] = {
        "computed": True,
        "method": "JuTrack ringpara",
        "physical_element_treatment": "intact_original_elements",
        "physical_element_count": len(lattice),
        "I1": float(values["I1"]),
        "I2": float(values["I2"]),
        "I3": float(values["I3"]),
        "I4": float(values["I4"]),
        "I5": float(values["I5"]),
        "Jx": float(values["Jx"]),
        "Jy": float(values["Jy"]),
        "Jz": float(values["Je"]),
        "emittance_m": float(values["emittx"]),
        "emittance_pm": float(values["emittx"]) * 1.0e12,
        "energy_spread": float(values["sigma_E"]),
        "energy_loss_per_turn_eV": float(values["U0"]),
    }
    metrics["physical_validity"] = radiation_physical_validity(metrics)
    return metrics


def jutrack_periodic_radiation_metrics(
    jt: Any,
    periodic_cell_lattice: Any,
    *,
    energy_gev: float,
    periods: int,
) -> dict[str, Any]:
    """Evaluate one periodic cell and scale extensive radiation quantities.

    For a ring made from identical periodic cells, I1--I5 and the energy loss
    are extensive.  Damping partitions, natural emittance, and energy spread
    depend on ratios of those integrals and therefore do not change when the
    cell integrals are multiplied by the number of periods.  This avoids
    rebuilding and evaluating a thousands-of-elements ring for every optimizer
    particle.  The selected candidate must still receive a full-ring gate.

    Returns computed=True, method, I1-I5, Jx/Jy/Jz, emittance_m/emittance_pm,
    energy_spread, energy_loss_per_turn_eV and physical_validity['passed'].
    Test linear stability first. computed=True is not a physical acceptance;
    positive damping and emittance are checked separately by physical_validity.
    """

    count = int(periods)
    if isinstance(periods, bool) or count < 1 or count != periods:
        raise ValueError("periods must be a positive integer")
    metrics: dict[str, Any] = dict(
        jutrack_radiation_metrics(jt, periodic_cell_lattice, energy_gev=energy_gev)
    )
    for name in ("I1", "I2", "I3", "I4", "I5", "energy_loss_per_turn_eV"):
        metrics[name] = float(metrics[name]) * count
    metrics.update(
        {
            "evaluation_scope": "one_periodic_cell_scaled_to_identical_ring",
            "periods": count,
            "full_ring_validation_required": True,
        }
    )
    return metrics


def compile_lattice_parameterization(
    elements: LatticeRows,
    variable_specs: Sequence[Mapping[str, Any]],
    *,
    sum_constraints: Sequence[Mapping[str, Any]] = (),
) -> LatticeParameterization:
    """Compile explicit model-authored variable declarations.

    Each variable needs ``name``, ``field``, ``bounds`` and either
    ``element_names`` or ``indices``.  ``initial`` is inferred only when all
    controlled elements currently share the same value.
    """

    rows = _rows(elements)
    index_by_name = {str(row["name"]): index for index, row in enumerate(rows)}
    knobs: list[LatticeKnob] = []
    occupied: set[tuple[int, str]] = set()
    names: set[str] = set()
    for offset, supplied in enumerate(variable_specs):
        spec = dict(supplied)
        name = str(spec.get("name", "")).strip()
        field = str(spec.get("field", "")).strip()
        if not name or name in names:
            raise ValueError(f"variable {offset} needs a unique non-empty name")
        if field not in _PHYSICAL_FIELDS:
            raise ValueError(f"variable {name!r} has unsupported field {field!r}")
        target_names = [str(item) for item in spec.get("element_names", ())]
        target_indices = [int(item) for item in spec.get("indices", ())]
        if target_names and target_indices:
            raise ValueError(f"variable {name!r} cannot use both names and indices")
        if target_names:
            missing = [item for item in target_names if item not in index_by_name]
            if missing:
                raise ValueError(f"variable {name!r} has unknown targets: {missing}")
            target_indices = [index_by_name[item] for item in target_names]
        if not target_indices or any(index < 0 or index >= len(rows) for index in target_indices):
            raise ValueError(f"variable {name!r} needs valid target elements")
        target_indices = list(dict.fromkeys(target_indices))
        target_names = [str(rows[index]["name"]) for index in target_indices]
        allowed_kinds = _FIELD_KINDS[field]
        incompatible = [
            f"{rows[index]['name']} ({rows[index]['kind']})"
            for index in target_indices
            if allowed_kinds is not None and rows[index]["kind"] not in allowed_kinds
        ]
        if incompatible:
            raise ValueError(
                f"variable {name!r} cannot control {field} on: {', '.join(incompatible)}"
            )
        bounds = list(spec.get("bounds", ()))
        if len(bounds) != 2:
            raise ValueError(f"variable {name!r} needs [lower, upper] bounds")
        lower, upper = float(bounds[0]), float(bounds[1])
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(f"variable {name!r} bounds must be finite with lower < upper")
        seeds = [_field_value(rows[index], field) for index in target_indices]
        if any(value is None for value in seeds):
            raise ValueError(f"variable {name!r} targets an element without {field}")
        if spec.get("initial") is None:
            initial = float(seeds[0])
            if any(
                not math.isclose(initial, float(value), rel_tol=1e-10, abs_tol=1e-12)
                for value in seeds[1:]
            ):
                raise ValueError(
                    f"variable {name!r} targets unequal seed values; split the knob or provide initial"
                )
        else:
            initial = float(spec["initial"])
        if not lower <= initial <= upper:
            raise ValueError(f"variable {name!r} initial value lies outside its bounds")
        overlap = {(index, field) for index in target_indices} & occupied
        if overlap:
            raise ValueError(f"variable {name!r} overlaps an existing field target")
        occupied.update((index, field) for index in target_indices)
        names.add(name)
        knobs.append(
            LatticeKnob(
                name=name,
                field=field,
                target_indices=tuple(target_indices),
                target_names=tuple(target_names),
                lower=lower,
                upper=upper,
                initial=initial,
                unit=str(spec.get("unit", "")),
                role=str(spec.get("role", "outer_design")),
                relabel_family=bool(spec.get("relabel_family", False)),
            )
        )
    if not knobs:
        raise ValueError("at least one lattice variable is required")
    constraints: list[LinearSumConstraint] = []
    by_name = {item.name: item for item in knobs}
    for supplied in sum_constraints:
        spec = dict(supplied)
        field = str(spec.get("field", "")).strip()
        variable_names = tuple(str(item) for item in spec.get("variable_names", ()))
        if field not in _PHYSICAL_FIELDS or not variable_names:
            raise ValueError("sum constraints need a physical field and variable_names")
        unknown = [name for name in variable_names if name not in by_name]
        if unknown:
            raise ValueError(f"sum constraint has unknown variables: {unknown}")
        if any(by_name[name].field != field for name in variable_names):
            raise ValueError("every variable in a sum constraint must control its field")
        target = float(spec["target"])
        if not math.isfinite(target):
            raise ValueError("sum-constraint target must be finite")
        constraints.append(
            LinearSumConstraint(
                field=field,
                target=target,
                variable_names=variable_names,
                mode=str(spec.get("mode", "scale")),
            )
        )
    return LatticeParameterization(tuple(rows), tuple(knobs), tuple(constraints))


def infer_mirror_symmetric_parameterization(
    elements: LatticeRows,
    *,
    fields: Sequence[str] = ("length_m", "angle_rad", "k1_m2"),
    relative_length_bounds: Sequence[float] = (0.5, 1.5),
    relative_angle_bounds: Sequence[float] = (0.5, 1.5),
    k1_bounds: Sequence[float] | None = None,
    k2_bounds: Sequence[float] | None = None,
    k3_bounds: Sequence[float] | None = None,
    minimum_length_m: float = 1.0e-4,
    fixed_element_names: Sequence[str] = (),
    included_element_names_by_field: Mapping[str, Sequence[str]] | None = None,
    excluded_element_names_by_field: Mapping[str, Sequence[str]] | None = None,
    excluded_roles_by_field: Mapping[str, Sequence[str]] | None = None,
    exact_sum_by_field: Mapping[str, float] | None = None,
    role: str = "outer_design",
) -> LatticeParameterization:
    """Infer one independent knob per mirror pair and physical field.

    Only global mirror symmetry is imposed.  Existing family labels do not
    silently merge optically distinct positions; K1/K2/K3 families are split
    to the inferred mirror pairs in the generated lattice. Strength bounds
    must be supplied for every actually inferred strength field: element rows
    do not declare beam energy, aperture, or magnet technology. Geometry-only
    inference and excluded/fixed strength fields need no strength bounds.
    When ``included_element_names_by_field`` is provided, ONLY the explicitly
    listed element/field pairs are varied. List both global mirror partners;
    unlisted fields remain fixed, including lengths of strength-only magnets.
    """

    rows = _rows(elements)
    requested = tuple(dict.fromkeys(str(item) for item in fields))
    invalid = [field for field in requested if field not in _PHYSICAL_FIELDS]
    if invalid:
        raise ValueError(f"unsupported inferred fields: {invalid}")
    if not requested:
        raise ValueError("fields must not be empty")
    inclusions = None
    if included_element_names_by_field is not None:
        inclusions = {}
        known_names = {row["name"] for row in rows}
        for field, names in included_element_names_by_field.items():
            if field not in requested:
                raise OptimizerInputError(f"included field {field!r} must also appear in fields")
            if isinstance(names, (str, bytes)):
                raise OptimizerInputError(f"included {field} names must be a sequence, not a string")
            selected = {str(name) for name in names}
            unknown = selected - known_names
            if unknown:
                raise OptimizerInputError(f"unknown included {field} elements: {sorted(unknown)}")
            incompatible = [row["name"] for row in rows if row["name"] in selected
                            and (_field_value(row, field) is None or
                                 (_FIELD_KINDS[field] is not None and row["kind"] not in _FIELD_KINDS[field]))]
            if incompatible:
                raise OptimizerInputError(f"elements do not support included {field}: {incompatible}")
            inclusions[field] = selected
    fixed = {str(item) for item in fixed_element_names}
    name_exclusions = {
        str(field): {str(name) for name in names}
        for field, names in dict(excluded_element_names_by_field or {}).items()
    }
    exclusions = {
        str(field): {str(role_name) for role_name in roles}
        for field, roles in dict(excluded_roles_by_field or {}).items()
    }
    unit = {
        "length_m": "m",
        "angle_rad": "rad",
        "k1_m2": "m^-2",
        "k2_m3": "m^-3",
        "k3_m4": "m^-4",
    }
    specs: list[dict[str, Any]] = []
    missing_sum_fields = set(exact_sum_by_field or {}) - set(requested)
    if inclusions is not None:
        missing_sum_fields |= {field for field in (exact_sum_by_field or {}) if not inclusions.get(field)}
    if missing_sum_fields:
        raise OptimizerInputError(
            f"Exact sum fields have no selected variables: {sorted(missing_sum_fields)}. "
            "An inclusion map selects ONLY the listed element/field pairs. Include both "
            "mirror partners for each intended tunable field, or omit that field's exact sum "
            "only if it is intentionally fixed and audited separately."
        )
    for left in range((len(rows) + 1) // 2):
        right = len(rows) - 1 - left
        pair = (left,) if left == right else (left, right)
        left_kind = str(rows[left]["kind"])
        right_kind = str(rows[right]["kind"])
        if left_kind != right_kind:
            raise ValueError(
                "element order is not mirror symmetric: "
                f"{rows[left]['name']} ({left_kind}) vs {rows[right]['name']} ({right_kind})"
            )
        pair_names = [str(rows[index]["name"]) for index in pair]
        for field in requested:
            if inclusions is not None:
                selected = inclusions.get(field, set())
                if not selected.intersection(pair_names):
                    continue
                if not set(pair_names) <= selected:
                    raise OptimizerInputError(f"included {field} must list both global mirror partners: {pair_names}")
                if (set(pair_names) & (fixed | name_exclusions.get(field, set())) or any(
                    str(_metadata(rows[index], "design_role") or "") in exclusions.get(field, set())
                    for index in pair
                )):
                    raise OptimizerInputError(f"included {field} conflicts with fixed/excluded elements: {pair_names}")
            allowed_kinds = _FIELD_KINDS[field]
            if allowed_kinds is not None and left_kind not in allowed_kinds:
                continue
            seeds = [_field_value(rows[index], field) for index in pair]
            if any(value is None for value in seeds):
                continue
            if any(
                not math.isclose(float(seeds[0]), float(value), rel_tol=1e-10, abs_tol=1e-12)
                for value in seeds[1:]
            ):
                raise ValueError(
                    f"{field} is not mirror symmetric for {pair_names}; use explicit knobs"
                )
            if any(name in fixed for name in pair_names):
                continue
            if any(name in name_exclusions.get(field, set()) for name in pair_names):
                continue
            field_exclusions = exclusions.get(field, set())
            if any(
                str(_metadata(rows[index], "design_role") or "") in field_exclusions
                for index in pair
            ):
                continue
            seed = float(seeds[0])
            if field == "length_m":
                if seed <= 0.0:
                    continue
                lower = max(float(minimum_length_m), seed * float(relative_length_bounds[0]))
                upper = seed * float(relative_length_bounds[1])
            elif field == "angle_rad":
                if abs(seed) <= 1.0e-15:
                    continue
                proposed = [seed * float(item) for item in relative_angle_bounds]
                lower, upper = min(proposed), max(proposed)
            else:
                option = {"k1_m2": "k1_bounds", "k2_m3": "k2_bounds", "k3_m4": "k3_bounds"}[field]
                supplied = {"k1_m2": k1_bounds, "k2_m3": k2_bounds, "k3_m4": k3_bounds}[field]
                if supplied is None:
                    raise OptimizerInputError(
                        f"{option} is required to infer {field} knobs for {pair_names}; "
                        "lattice rows do not specify beam energy or magnet technology. "
                        "Use api.magnet_strength_limits(energy_gev=...) for K1/K2 or "
                        "supply your declared technology bounds. Outer design and inner "
                        "matching must respect the same engineering envelope."
                    )
                try:
                    lower, upper = map(float, supplied)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise OptimizerInputError(
                        f"{option} must contain two finite numbers with lower < upper"
                    ) from exc
                if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                    raise OptimizerInputError(
                        f"{option} must contain two finite numbers with lower < upper"
                    )
            label = pair_names[0] if len(pair_names) == 1 else f"{pair_names[0]}<->{pair_names[1]}"
            specs.append(
                {
                    "name": f"{field}:{label}",
                    "field": field,
                    "indices": list(pair),
                    "bounds": [lower, upper],
                    "initial": seed,
                    "unit": unit[field],
                    "role": role,
                    "relabel_family": field in {"k1_m2", "k2_m3", "k3_m4"},
                }
            )
    sums = []
    for field, target in dict(exact_sum_by_field or {}).items():
        field_specs = [spec for spec in specs if spec["field"] == field]
        variable_names = [spec["name"] for spec in field_specs]
        if not variable_names:
            raise OptimizerInputError(f"exact sum requested for {field} but it has no variables")
        controlled_indices = {
            int(index)
            for spec in field_specs
            for index in spec["indices"]
        }
        fixed_contribution = sum(
            float(value)
            for index, row in enumerate(rows)
            if index not in controlled_indices
            for value in (_field_value(row, field),)
            if value is not None
        )
        variable_target = float(target) - fixed_contribution
        if not math.isfinite(variable_target):
            raise ValueError(f"exact sum target for {field} is not finite")
        sums.append(
            {
                "field": field,
                "target": variable_target,
                "variable_names": variable_names,
                "mode": "scale",
            }
        )
    return compile_lattice_parameterization(rows, specs, sum_constraints=sums)


def compile_family_parameterization(
    elements: LatticeRows,
    *,
    field: str,
    families: Sequence[str],
    bounds: Sequence[float] | Mapping[str, Sequence[float]],
    role: str = "in_objective_matching",
) -> LatticeParameterization:
    """Compile declared power-supply families as matching or design knobs."""

    rows = _rows(elements)
    specs = []
    for family in families:
        name = str(family)
        targets = [
            index for index, row in enumerate(rows) if str(_metadata(row, "family") or "") == name
        ]
        if not targets:
            raise ValueError(f"lattice contains no elements in family {name!r}")
        family_bounds = bounds[name] if isinstance(bounds, Mapping) else bounds
        specs.append(
            {
                "name": f"{field}:{name}",
                "field": field,
                "indices": targets,
                "bounds": list(family_bounds),
                "role": role,
                "relabel_family": False,
            }
        )
    return compile_lattice_parameterization(rows, specs)


def _target_arrays(targets: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    names: list[str] = []
    values: list[float] = []
    tolerances: list[float] = []
    for name, supplied in targets.items():
        if isinstance(supplied, Mapping):
            target = float(supplied.get("target", 0.0))
            tolerance = float(supplied["tolerance"])
        else:
            target, tolerance = map(float, supplied)
        if not math.isfinite(target) or not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(f"match target {name!r} needs finite target and positive tolerance")
        names.append(str(name))
        values.append(target)
        tolerances.append(tolerance)
    if not names:
        raise ValueError("at least one matching target is required")
    return tuple(names), np.asarray(values), np.asarray(tolerances)


def _residual(
    metrics: Mapping[str, Any], names: Sequence[str], target: np.ndarray, tolerance: np.ndarray
) -> np.ndarray | None:
    metrics = _metric_result(metrics)
    if metrics.get("stable") is False or metrics.get("feasible") is False:
        return None
    values: list[float] = []
    for name in names:
        if name not in metrics:
            raise KeyError(f"evaluate_metrics omitted requested matching observable {name!r}")
        value = _callback_number(metrics[name], f"evaluate_metrics[{name!r}]")
        if not math.isfinite(value):
            raise ValueError(f"evaluate_metrics[{name!r}] must be finite unless stable=False or feasible=False")
        values.append(value)
    return (np.asarray(values, dtype=float) - target) / tolerance


def _callback_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar, not {type(value).__name__}")
    return float(value)


def _callback_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must return a mapping, not {type(value).__name__}")
    result = dict(value)
    for flag in ("stable", "feasible"):
        if flag in result:
            if not isinstance(result[flag], (bool, np.bool_)):
                raise TypeError(f"{label}[{flag!r}] must be a boolean")
            result[flag] = bool(result[flag])
    return result


def _metric_result(value: Any) -> dict[str, Any]:
    return _callback_mapping(value, "evaluate_metrics")


def _physical_rejection(exc: ResearchInfeasible) -> dict[str, Any]:
    return {"stable": False, "feasible": False, "failure": str(exc),
            "rejection_code": exc.code, "diagnostics": copy.deepcopy(exc.diagnostics)}


def _call_metrics(evaluate_metrics: MetricEvaluator, rows: list[dict[str, Any]]) -> dict[str, Any]:
    supplied = copy.deepcopy(rows)
    before = _physical_rows_key(supplied)
    try:
        result = _metric_result(evaluate_metrics(supplied))
    except ResearchInfeasible as exc:
        result = _physical_rejection(exc)
    if _physical_rows_key(supplied) != before:
        raise ValueError("evaluate_metrics changed physical lattice rows; matching callbacks must evaluate the supplied candidate without changing its coordinates")
    return result


def _objective_result(value: Any, objective_names: Sequence[str]) -> dict[str, Any]:
    """Validate callback protocol before invalid shapes can masquerade as physics."""
    result = _callback_mapping(value, "evaluate_objectives")
    violations = result.get("constraint_violations", {})
    if not isinstance(violations, Mapping):
        raise TypeError("evaluate_objectives['constraint_violations'] must be a mapping")
    clean = {}
    for name, supplied in violations.items():
        number = _callback_number(supplied, f"constraint_violations[{name!r}]")
        if math.isnan(number) or number == -math.inf:
            raise ValueError(f"constraint_violations[{name!r}] cannot be NaN or negative infinity")
        clean[str(name)] = max(0.0, number)
    result["constraint_violations"] = clean
    aggregate = result.get("constraint_violation", sum(clean.values()))
    aggregate = _callback_number(aggregate, "constraint_violation")
    if math.isnan(aggregate) or aggregate == -math.inf:
        raise ValueError("constraint_violation cannot be NaN or negative infinity")
    rejected = (result.get("stable") is False or result.get("feasible") is False
                or aggregate > 1e-12 or any(v > 1e-12 for v in clean.values()))
    objectives = result.get("objectives", result.get("objective_values", {}))
    if not isinstance(objectives, Mapping):
        raise TypeError("evaluate_objectives['objectives'] must be a mapping of objective names to real scalars")
    normalized = dict(objectives)
    for name in objective_names:
        if name not in objectives:
            if rejected:
                normalized[name] = math.inf
                continue
            raise KeyError(f"evaluate_objectives omitted requested objective {name!r}")
        if objectives[name] is None and rejected:
            number = math.inf
        else:
            number = _callback_number(objectives[name], f"objectives[{name!r}]")
        if not math.isfinite(number):
            if not rejected:
                raise ValueError(f"objectives[{name!r}] must be finite for an admissible candidate; explicitly reject invalid physics")
            number = math.inf
        normalized[name] = number
    result["objectives"] = normalized
    result["feasible"] = not rejected
    return result


def _call_objectives(evaluate_objectives: ObjectiveEvaluator, rows: list[dict[str, Any]],
                     objective_names: Sequence[str]) -> dict[str, Any]:
    supplied = copy.deepcopy(rows)
    before = _physical_rows_key(supplied)
    try:
        result = _objective_result(evaluate_objectives(supplied), objective_names)
    except ResearchInfeasible as exc:
        result = _objective_result(_physical_rejection(exc), objective_names)
    if _physical_rows_key(supplied) != before:
        raise ValueError("evaluate_objectives changed physical lattice rows; declare those variables in the matcher or return an independently validated candidate outside this optimization callback")
    return result


def _apply_design_proposal(parameterization, position, baseline):
    try:
        return parameterization.apply(position, base_elements=baseline)
    except ValueError as exc:
        if not str(exc).startswith(("exact sum projection moves", "cannot scale")):
            raise
        raise ResearchInfeasible(str(exc), code="exact_sum_projection_outside_bounds",
                                 diagnostics={"position": list(map(float, position))}) from exc


def bounded_response_match(
    elements: LatticeRows,
    parameterization: LatticeParameterization,
    evaluate_metrics: MetricEvaluator,
    targets: Mapping[str, Any],
    *,
    initial_position: Sequence[float] | None = None,
    max_iterations: int = 12,
    finite_difference_fraction: float = 2.0e-3,
    initial_trust_fraction: float = 0.15,
) -> dict[str, Any]:
    """Match declared knobs with a bounded finite-difference response matrix.

    Callbacks return named finite scalar observables, or explicitly reject a
    candidate with stable=False/feasible=False/ResearchInfeasible. They do not
    mutate the supplied rows. Programming and runtime exceptions propagate.
    """

    names, target, tolerance = _target_arrays(targets)
    position = np.asarray(
        parameterization.initial_position if initial_position is None else initial_position,
        dtype=float,
    )
    lower = np.asarray([item.lower for item in parameterization.knobs], dtype=float)
    upper = np.asarray([item.upper for item in parameterization.knobs], dtype=float)
    if position.shape != lower.shape or np.any(position < lower) or np.any(position > upper):
        raise ValueError("initial matching position does not fit the matching parameterization")
    span = upper - lower
    trust = float(initial_trust_fraction)
    evaluations = 0
    trace: list[dict[str, Any]] = []

    def evaluate(pos: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray | None]:
        nonlocal evaluations
        rows = parameterization.apply(pos, base_elements=elements)
        metrics = _call_metrics(evaluate_metrics, rows)
        evaluations += 1
        return rows, metrics, _residual(metrics, names, target, tolerance)

    rows, metrics, residual = evaluate(position)
    if residual is None:
        return {
            "success": False,
            "reason": "initial_matching_point_is_not_evaluable",
            "elements": rows,
            "position": position.tolist(),
            "metrics": metrics,
            "evaluations": evaluations,
            "trace": trace,
        }
    jacobian: np.ndarray | None = None
    for iteration in range(max(1, int(max_iterations)) + 1):
        norm = float(np.linalg.norm(residual))
        maximum = float(np.max(np.abs(residual)))
        trace.append(
            {
                "iteration": iteration,
                "residual_norm": norm,
                "maximum_normalized_residual": maximum,
                "trust_fraction": trust,
            }
        )
        if maximum <= 1.0:
            return {
                "success": True,
                "reason": "targets_reached",
                "elements": rows,
                "position": position.tolist(),
                "metrics": metrics,
                "evaluations": evaluations,
                "trace": trace,
            }
        if iteration >= max_iterations:
            break
        if jacobian is None:
            jacobian = np.zeros((len(residual), len(position)), dtype=float)
            for dimension in range(len(position)):
                step = max(1.0e-8, finite_difference_fraction * span[dimension])
                plus = position.copy()
                plus[dimension] = min(upper[dimension], position[dimension] + step)
                minus = position.copy()
                minus[dimension] = max(lower[dimension], position[dimension] - step)
                plus_residual = None
                if plus[dimension] != position[dimension]:
                    _, _, plus_residual = evaluate(plus)
                if plus_residual is not None:
                    jacobian[:, dimension] = (plus_residual - residual) / (
                        plus[dimension] - position[dimension]
                    )
                    continue
                # A backward evaluation is a stability/bound fallback, not a
                # second mandatory map call for every smooth response column.
                minus_residual = None
                if minus[dimension] != position[dimension]:
                    _, _, minus_residual = evaluate(minus)
                if minus_residual is not None:
                    jacobian[:, dimension] = (residual - minus_residual) / (
                        position[dimension] - minus[dimension]
                    )
        if not np.any(np.isfinite(jacobian)) or np.linalg.norm(jacobian) <= 1.0e-15:
            break
        # Solve in normalized knob coordinates.  A raw pseudoinverse followed
        # by component-wise clipping destroys correlated optics corrections
        # when K1 ranges or response columns differ by orders of magnitude.
        normalized_jacobian = jacobian * span[np.newaxis, :]
        singular_values = np.linalg.svd(normalized_jacobian, compute_uv=False)
        largest = float(singular_values[0]) if len(singular_values) else 0.0
        damping = max(1.0e-10, 1.0e-4 * largest * largest)
        normal = normalized_jacobian.T @ normalized_jacobian
        rhs = -(normalized_jacobian.T @ residual)
        normalized_direction = np.linalg.solve(normal + damping * np.eye(len(position)), rhs)
        maximum_direction = float(np.max(np.abs(normalized_direction)))
        if maximum_direction > trust:
            normalized_direction *= trust / maximum_direction
        directions = [normalized_direction * span]
        gradient = -(normalized_jacobian.T @ residual)
        gradient_maximum = float(np.max(np.abs(gradient)))
        if gradient_maximum > 0.0 and math.isfinite(gradient_maximum):
            directions.append((trust * gradient / gradient_maximum) * span)
        accepted = False
        previous_position = position.copy()
        previous_residual = residual.copy()
        for direction in directions:
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                proposal = np.clip(position + scale * direction, lower, upper)
                if np.array_equal(proposal, position):
                    continue
                trial_rows, trial_metrics, trial_residual = evaluate(proposal)
                if trial_residual is None:
                    continue
                if float(np.linalg.norm(trial_residual)) < norm - 1.0e-10:
                    position = proposal
                    rows, metrics, residual = trial_rows, trial_metrics, trial_residual
                    trust = min(0.5, trust * 1.25)
                    accepted = True
                    break
            if accepted:
                break
        if accepted:
            actual_step = position - previous_position
            denominator = float(actual_step @ actual_step)
            if denominator > 1.0e-20:
                prediction_error = residual - previous_residual - jacobian @ actual_step
                jacobian = jacobian + np.outer(prediction_error, actual_step) / denominator
        if not accepted:
            jacobian = None
            trust *= 0.5
            if trust < 1.0e-4:
                break
    return {
        "success": False,
        "reason": "response_match_did_not_reach_targets",
        "elements": rows,
        "position": position.tolist(),
        "metrics": metrics,
        "evaluations": evaluations,
        "trace": trace,
    }


def bounded_fast_response_match(
    elements: LatticeRows,
    parameterization: LatticeParameterization,
    evaluate_metrics: MetricEvaluator,
    targets: Mapping[str, Any],
    *,
    initial_position: Sequence[float] | None = None,
    max_iterations: int = 4,
    max_evaluations: int = 32,
    finite_difference_fraction: float = 2.0e-3,
    initial_trust_fraction: float = 0.35,
) -> dict[str, Any]:
    """Perform a warm-start response match with a strict evaluation budget.

    This is the inner matcher for population optimization.  It computes one
    finite-difference Jacobian, updates it with Broyden secants after accepted
    steps, and never restarts an adaptive continuation campaign for one
    particle.  A particle that cannot be rematched inside the declared budget
    is infeasible; it must not consume hundreds of hidden optics evaluations.

    Use :func:`bounded_response_match` or :func:`continuation_match` for a
    standalone repair where robustness matters more than throughput.
    """

    names, target, tolerance = _target_arrays(targets)
    position = np.asarray(
        parameterization.initial_position if initial_position is None else initial_position,
        dtype=float,
    )
    lower = np.asarray([item.lower for item in parameterization.knobs], dtype=float)
    upper = np.asarray([item.upper for item in parameterization.knobs], dtype=float)
    if position.shape != lower.shape or np.any(position < lower) or np.any(position > upper):
        raise ValueError("initial matching position does not fit the matching parameterization")
    budget = int(max_evaluations)
    if budget < 2:
        raise ValueError("max_evaluations must be at least 2")
    span = upper - lower
    trust = float(initial_trust_fraction)
    if not math.isfinite(trust) or not 0.0 < trust <= 1.0:
        raise ValueError("initial_trust_fraction must be in (0, 1]")
    evaluations = 0
    trace: list[dict[str, Any]] = []

    def evaluate(
        pos: np.ndarray,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray | None] | None:
        nonlocal evaluations
        if evaluations >= budget:
            return None
        rows = parameterization.apply(pos, base_elements=elements)
        metrics = _call_metrics(evaluate_metrics, rows)
        evaluations += 1
        return rows, metrics, _residual(metrics, names, target, tolerance)

    initial = evaluate(position)
    assert initial is not None
    rows, metrics, residual = initial
    if residual is None:
        return {
            "success": False,
            "reason": "initial_matching_point_is_not_evaluable",
            "elements": rows,
            "position": position.tolist(),
            "metrics": metrics,
            "evaluations": evaluations,
            "evaluation_budget": budget,
            "trace": trace,
        }
    maximum = float(np.max(np.abs(residual)))
    if maximum <= 1.0:
        return {
            "success": True,
            "reason": "targets_reached",
            "elements": rows,
            "position": position.tolist(),
            "metrics": metrics,
            "evaluations": evaluations,
            "evaluation_budget": budget,
            "trace": [{"iteration": 0, "maximum_normalized_residual": maximum}],
        }

    # One forward response matrix.  Backward differences are used only for an
    # invalid/bounded forward point and are still charged to the hard budget.
    jacobian = np.zeros((len(residual), len(position)), dtype=float)
    for dimension in range(len(position)):
        if evaluations >= budget:
            break
        step = max(1.0e-8, finite_difference_fraction * span[dimension])
        plus = position.copy()
        plus[dimension] = min(upper[dimension], position[dimension] + step)
        trial = evaluate(plus) if plus[dimension] != position[dimension] else None
        trial_residual = None if trial is None else trial[2]
        if trial_residual is not None:
            jacobian[:, dimension] = (trial_residual - residual) / (
                plus[dimension] - position[dimension]
            )
            continue
        minus = position.copy()
        minus[dimension] = max(lower[dimension], position[dimension] - step)
        trial = evaluate(minus) if minus[dimension] != position[dimension] else None
        trial_residual = None if trial is None else trial[2]
        if trial_residual is not None:
            jacobian[:, dimension] = (residual - trial_residual) / (
                position[dimension] - minus[dimension]
            )

    reason = "fast_response_match_did_not_reach_targets"
    for iteration in range(1, max(1, int(max_iterations)) + 1):
        norm = float(np.linalg.norm(residual))
        maximum = float(np.max(np.abs(residual)))
        trace.append(
            {
                "iteration": iteration,
                "residual_norm": norm,
                "maximum_normalized_residual": maximum,
                "trust_fraction": trust,
                "evaluations": evaluations,
            }
        )
        if maximum <= 1.0:
            reason = "targets_reached"
            break
        if evaluations >= budget:
            reason = "fast_response_match_evaluation_budget_exhausted"
            break
        if not np.all(np.isfinite(jacobian)) or np.linalg.norm(jacobian) <= 1.0e-15:
            reason = "fast_response_matrix_is_singular"
            break

        normalized_jacobian = jacobian * span[np.newaxis, :]
        singular_values = np.linalg.svd(normalized_jacobian, compute_uv=False)
        largest = float(singular_values[0]) if len(singular_values) else 0.0
        damping = max(1.0e-10, 1.0e-4 * largest * largest)
        normal = normalized_jacobian.T @ normalized_jacobian
        rhs = -(normalized_jacobian.T @ residual)
        normalized_direction = np.linalg.solve(normal + damping * np.eye(len(position)), rhs)
        maximum_direction = float(np.max(np.abs(normalized_direction)))
        if maximum_direction > trust:
            normalized_direction *= trust / maximum_direction
        direction = normalized_direction * span
        previous_position = position.copy()
        previous_residual = residual.copy()
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            proposal = np.clip(position + scale * direction, lower, upper)
            if np.array_equal(proposal, position):
                continue
            trial = evaluate(proposal)
            if trial is None:
                reason = "fast_response_match_evaluation_budget_exhausted"
                break
            trial_rows, trial_metrics, trial_residual = trial
            if trial_residual is None:
                continue
            if float(np.linalg.norm(trial_residual)) < norm - 1.0e-10:
                position = proposal
                rows, metrics, residual = trial_rows, trial_metrics, trial_residual
                accepted = True
                trust = min(0.75, trust * 1.35)
                break
        if not accepted:
            if evaluations >= budget:
                reason = "fast_response_match_evaluation_budget_exhausted"
            else:
                reason = "fast_response_match_has_no_descent_step"
            break
        actual_step = position - previous_position
        denominator = float(actual_step @ actual_step)
        if denominator > 1.0e-20:
            prediction_error = residual - previous_residual - jacobian @ actual_step
            jacobian = jacobian + np.outer(prediction_error, actual_step) / denominator

    success = residual is not None and float(np.max(np.abs(residual))) <= 1.0
    return {
        "success": success,
        "reason": "targets_reached" if success else reason,
        "elements": rows,
        "position": position.tolist(),
        "metrics": metrics,
        "evaluations": evaluations,
        "evaluation_budget": budget,
        "trace": trace,
    }


def _interpolate_rows(
    baseline: LatticeRows, target: LatticeRows, fraction: float
) -> list[dict[str, Any]]:
    left = _rows(baseline)
    right = _rows(target)
    if len(left) != len(right):
        raise ValueError("continuation lattices have different element counts")
    result = copy.deepcopy(left)
    for index, (base, goal) in enumerate(zip(left, right)):
        if base["name"] != goal["name"] or base["kind"] != goal["kind"]:
            raise ValueError(f"continuation element {index} changes identity or topology")
        for field in _PHYSICAL_FIELDS:
            base_value = _field_value(base, field)
            goal_value = _field_value(goal, field)
            if base_value is None and goal_value is None:
                continue
            if base_value is None or goal_value is None:
                raise ValueError(f"continuation cannot add or remove {field} on {base['name']}")
            result[index][field] = base_value + fraction * (goal_value - base_value)
        result[index]["fields"] = copy.deepcopy(dict(goal.get("fields") or {}))
    return result


def continuation_match(
    baseline_elements: LatticeRows,
    target_elements: LatticeRows,
    matching_parameterization: LatticeParameterization,
    evaluate_metrics: MetricEvaluator,
    targets: Mapping[str, Any],
    *,
    initial_matching_position: Sequence[float] | None = None,
    initial_steps: int = 1,
    minimum_step_fraction: float = 1.0 / 256.0,
    max_match_iterations: int = 12,
) -> dict[str, Any]:
    """Transport a stable matched solution to new geometry with adaptive steps."""

    position = list(
        matching_parameterization.initial_position
        if initial_matching_position is None
        else initial_matching_position
    )
    baseline_match = bounded_response_match(
        baseline_elements,
        matching_parameterization,
        evaluate_metrics,
        targets,
        initial_position=position,
        max_iterations=max_match_iterations,
    )
    total_evaluations = int(baseline_match["evaluations"])
    trace = [{"fraction": 0.0, "accepted": bool(baseline_match["success"])}]
    if not baseline_match["success"]:
        return {
            **baseline_match,
            "reason": "baseline_does_not_satisfy_matching_contract",
            "fraction_reached": 0.0,
            "continuation_trace": trace,
        }
    position = list(baseline_match["position"])
    accepted_elements = list(baseline_match["elements"])
    accepted_metrics = dict(baseline_match["metrics"])
    fraction = 0.0
    step = 1.0 / max(1, int(initial_steps))
    attempts = 0
    while fraction < 1.0 - 1.0e-12 and attempts < 2048:
        attempts += 1
        trial_fraction = min(1.0, fraction + step)
        geometry = _interpolate_rows(baseline_elements, target_elements, trial_fraction)
        trial = bounded_response_match(
            geometry,
            matching_parameterization,
            evaluate_metrics,
            targets,
            initial_position=position,
            max_iterations=max_match_iterations,
        )
        total_evaluations += int(trial["evaluations"])
        trace.append(
            {
                "fraction": trial_fraction,
                "step": step,
                "accepted": bool(trial["success"]),
                "reason": trial["reason"],
                "evaluations": int(trial["evaluations"]),
            }
        )
        if trial["success"]:
            fraction = trial_fraction
            position = list(trial["position"])
            accepted_elements = list(trial["elements"])
            accepted_metrics = dict(trial["metrics"])
            step = min(1.0 - fraction, step * 1.5) if fraction < 1.0 else 0.0
            continue
        step *= 0.5
        if step < minimum_step_fraction:
            break
    return {
        "success": fraction >= 1.0 - 1.0e-12,
        "reason": "target_geometry_matched"
        if fraction >= 1.0 - 1.0e-12
        else "continuation_stalled",
        "fraction_reached": fraction,
        "elements": accepted_elements,
        "position": position,
        "metrics": accepted_metrics,
        "evaluations": total_evaluations,
        "continuation_trace": trace,
    }


def local_seed_positions(
    parameterization: LatticeParameterization,
    *,
    relative_step: float = 0.03,
    maximum: int | None = None,
) -> list[list[float]]:
    """Return deterministic one-variable-at-a-time seeds around the baseline."""

    initial = np.asarray(parameterization.initial_position, dtype=float)
    bounds = np.asarray(parameterization.bounds, dtype=float)
    width = bounds[:, 1] - bounds[:, 0]
    seeds: list[list[float]] = []
    for dimension in range(len(initial)):
        for sign in (-1.0, 1.0):
            position = initial.copy()
            position[dimension] = np.clip(
                position[dimension] + sign * relative_step * width[dimension],
                bounds[dimension, 0],
                bounds[dimension, 1],
            )
            if not np.array_equal(position, initial):
                seeds.append(position.tolist())
                if maximum is not None and len(seeds) >= maximum:
                    return seeds
    return seeds


def feasible_basin_seed_positions(
    parameterization: LatticeParameterization,
    *,
    count: int,
    seed: int = 0,
    relative_sigma: float = 0.01,
) -> list[list[float]]:
    """Fill a population around the accepted seed without losing diversity."""

    requested = max(0, int(count))
    if requested == 0:
        return []
    initial = np.asarray(parameterization.initial_position, dtype=float)
    bounds = np.asarray(parameterization.bounds, dtype=float)
    width = bounds[:, 1] - bounds[:, 0]
    seeds = local_seed_positions(parameterization, maximum=requested)
    rng = np.random.default_rng(int(seed))
    while len(seeds) < requested:
        proposal = np.clip(
            initial + rng.normal(0.0, relative_sigma, len(initial)) * width,
            bounds[:, 0],
            bounds[:, 1],
        )
        seeds.append(proposal.tolist())
    return seeds


def search_feasible_parameter_seeds(
    elements: LatticeRows,
    parameterization: LatticeParameterization,
    evaluate_metrics: MetricEvaluator,
    *,
    count: int = 8,
    attempts: int = 2_000,
    seed: int = 0,
    targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Find stable finite starts before a response match or optimization.

    Accelerator matching frequently starts from an unstable map, where a local
    response matrix does not exist. This bounded search samples both the local
    neighbourhood and the full declared knob box, retains only points whose
    callback reports ``stable=True``, and ranks them by optional normalized
    target distance. It is deliberately physics-neutral: the caller still
    defines the elements, knobs, JuTrack evaluator, targets, and acceptance gate.
    """

    requested = int(count)
    budget = int(attempts)
    if requested < 1:
        raise ValueError("count must be at least 1")
    if budget < requested:
        raise ValueError("attempts must be at least count")
    names: tuple[str, ...] = ()
    target = tolerance = np.asarray([], dtype=float)
    if targets:
        names, target, tolerance = _target_arrays(targets)
    initial = np.asarray(parameterization.initial_position, dtype=float)
    bounds = np.asarray(parameterization.bounds, dtype=float)
    lower, upper = bounds[:, 0], bounds[:, 1]
    width = upper - lower
    rng = np.random.default_rng(int(seed))
    accepted: list[tuple[float, list[float], dict[str, Any]]] = []
    stable_count = 0

    for index in range(budget):
        if index == 0:
            position = initial.copy()
        elif index % 2:
            # Expand smoothly from the supplied physical seed instead of using
            # one arbitrary fixed perturbation scale.
            progress = min(1.0, index / max(1.0, budget - 1.0))
            sigma = 0.01 + 0.34 * progress
            position = np.clip(
                initial + rng.normal(0.0, sigma, len(initial)) * width,
                lower,
                upper,
            )
        else:
            position = rng.uniform(lower, upper)
        rows = parameterization.apply(position, base_elements=elements)
        metrics = _call_metrics(evaluate_metrics, rows)
        if "stable" not in metrics and "feasible" not in metrics:
            raise KeyError("Feasible seed screening requires an explicit stable or feasible boolean")
        if metrics.get("stable") is False or metrics.get("feasible") is False:
            continue
        numeric = [
            float(value)
            for value in metrics.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric and not all(math.isfinite(value) for value in numeric):
            raise ValueError("Feasible seed callback returned nonfinite scalar metrics without an explicit physical rejection")
        stable_count += 1
        residual = _residual(metrics, names, target, tolerance) if names else None
        score = float(np.linalg.norm(residual)) if residual is not None else 0.0
        accepted.append((score, position.tolist(), metrics))
        accepted.sort(key=lambda item: item[0])
        if len(accepted) > requested:
            accepted.pop()

    return {
        "success": bool(accepted),
        "attempts": budget,
        "stable_evaluations": stable_count,
        "requested_count": requested,
        "seeds": [
            {"score": score, "position": position, "metrics": metrics}
            for score, position, metrics in accepted
        ],
    }


@dataclass(frozen=True, slots=True)
class _LatticeWorkerEvaluatorFactory:
    """Serializable recipe for one process-local nested JuTrack evaluator."""

    physics_factory: WorkerPhysicsFactory
    baseline_elements: list[dict[str, Any]]
    design_parameterization: LatticeParameterization
    matching_parameterization: LatticeParameterization
    match_targets: dict[str, Any]
    objective_names: tuple[str, ...]
    baseline_matching_position: tuple[float, ...]
    max_match_iterations: int
    inner_match_evaluation_budget: int

    def __call__(self) -> Callable[[np.ndarray], Mapping[str, Any]]:
        callbacks = self.physics_factory(_rows(self.baseline_elements))
        if not isinstance(callbacks, tuple) or len(callbacks) != 2:
            raise TypeError(
                "worker_physics_factory must return (evaluate_metrics, evaluate_objectives)"
            )
        evaluate_metrics, evaluate_objectives = callbacks
        if not callable(evaluate_metrics) or not callable(evaluate_objectives):
            raise TypeError("worker physics callbacks must be callable")
        metric_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        warm_starts = _MatchingWarmStartBank(
            self.design_parameterization.bounds,
            self.design_parameterization.initial_position,
            self.baseline_matching_position,
        )

        def cached_metrics(rows: list[dict[str, Any]]) -> Mapping[str, Any]:
            key = _physical_rows_key(rows)
            if key not in metric_cache:
                metric_cache[key] = _call_metrics(evaluate_metrics, rows)
            return metric_cache[key]

        def evaluate(position: np.ndarray) -> Mapping[str, Any]:
            try:
                matching_start, warm_start_source = warm_starts.nearest(position)
                target = _apply_design_proposal(self.design_parameterization, position, self.baseline_elements)
                matched = bounded_fast_response_match(
                    target,
                    self.matching_parameterization,
                    cached_metrics,
                    self.match_targets,
                    initial_position=matching_start,
                    max_iterations=min(4, max(1, int(self.max_match_iterations))),
                    max_evaluations=self.inner_match_evaluation_budget,
                )
                if not matched["success"]:
                    return {
                        "objectives": {name: math.inf for name in self.objective_names},
                        "constraint_violations": {"in_objective_matching": 1.0},
                        "feasible": False,
                        "matching": {
                            "reason": matched["reason"],
                            "evaluations": matched.get("evaluations", 0),
                            "evaluation_budget": matched.get("evaluation_budget"),
                        },
                    }
                matched_rows = copy.deepcopy(matched["elements"])
                warm_starts.record(position, matched["position"])
                raw = _call_objectives(evaluate_objectives, matched_rows, self.objective_names)
                raw["matching"] = {
                    "reason": matched["reason"],
                    "evaluations": matched["evaluations"],
                    "position": matched["position"],
                    "metrics": matched["metrics"],
                    "warm_start_source": warm_start_source,
                    "elements": copy.deepcopy(matched_rows),
                }
                return raw
            except ResearchInfeasible as exc:
                return _objective_result(_physical_rejection(exc), self.objective_names)

        return evaluate


def _physical_rows_key(rows: LatticeRows) -> tuple[Any, ...]:
    return tuple(
        (
            str(row["name"]),
            str(row["kind"]),
            *(
                None
                if _field_value(row, field) is None
                else float(_field_value(row, field))
                for field in sorted(_PHYSICAL_FIELDS)
            ),
        )
        for row in rows
    )


class _MatchingWarmStartBank:
    """Bounded nearest-neighbour warm starts for repeated inner matching."""

    def __init__(
        self,
        bounds: Sequence[Sequence[float]],
        outer_position: Sequence[float],
        matching_position: Sequence[float],
        *,
        maximum: int = 512,
    ) -> None:
        array = np.asarray(bounds, dtype=float)
        self._lower = array[:, 0]
        self._width = array[:, 1] - array[:, 0]
        self._maximum = max(2, int(maximum))
        self._entries: list[tuple[np.ndarray, np.ndarray]] = [
            (
                np.asarray(outer_position, dtype=float).copy(),
                np.asarray(matching_position, dtype=float).copy(),
            )
        ]

    def nearest(self, outer_position: Sequence[float]) -> tuple[list[float], str]:
        position = np.asarray(outer_position, dtype=float)
        normalized = (position - self._lower) / self._width
        distances = [
            float(np.sum((((candidate - self._lower) / self._width) - normalized) ** 2))
            for candidate, _matching in self._entries
        ]
        index = int(np.argmin(distances))
        source = "baseline" if index == 0 else "nearest_successful_candidate"
        return self._entries[index][1].tolist(), source

    def record(
        self, outer_position: Sequence[float], matching_position: Sequence[float]
    ) -> None:
        self._entries.append(
            (
                np.asarray(outer_position, dtype=float).copy(),
                np.asarray(matching_position, dtype=float).copy(),
            )
        )
        if len(self._entries) > self._maximum:
            # Preserve the accepted baseline and the most recent local basin.
            del self._entries[1]


def plan_lattice_optimization(
    design_parameterization: LatticeParameterization,
    matching_parameterization: LatticeParameterization,
    *,
    population_size: int | None = None,
    generations: int | None = None,
    minimum_population_per_knob: float = 0.0,
    minimum_production_generations: int = 0,
) -> dict[str, Any]:
    """Validate ownership and size a campaign from its ACTUAL compiled knobs.

    Pure Python: no JuTrack import, callbacks, workers or optimization. Call
    before creating simulation objects and show ``knob_table`` to the user.
    Explicit budgets are respected in managed and standalone research. Callers
    may optionally declare their own minimum budget, but there is no imposed
    population-per-knob rule or production generation count.
    This is a coding preflight, NOT a claim of physical feasibility/convergence.
    """
    def integer(value: Any, name: str, minimum: int) -> int:
        number = _finite_number(value)
        if isinstance(value, bool) or number is None or number < minimum or number != int(number):
            raise OptimizerInputError(f"{name} must be an integer >= {minimum}")
        return int(number)

    count = len(design_parameterization.knobs)
    if not count:
        raise OptimizerInputError("optimization needs at least one outer design knob")
    ratio = _finite_number(minimum_population_per_knob)
    if ratio is None or ratio < 0:
        raise OptimizerInputError("minimum_population_per_knob must be finite and non-negative")
    minimum_generations = integer(minimum_production_generations, "minimum_production_generations", 0)
    generation_count = integer(generations, "generations", 1) if generations is not None else None
    requested_population = integer(population_size, "population_size", 2) if population_size is not None else None
    from .optimization_budget import resolve_population_budget

    budget = resolve_population_budget(count, requested_population, generation_count)
    generation_count = budget["effective"]["generations"]
    if generation_count < minimum_generations:
        raise OptimizerInputError(
            f"generations={generations} is below the caller-declared minimum of {minimum_generations}"
        )
    design_ownership = {(name, knob.field) for knob in design_parameterization.knobs
                        for name in knob.target_names}
    matching_ownership = {(name, knob.field) for knob in matching_parameterization.knobs
                          for name in knob.target_names}
    overlap = sorted(design_ownership & matching_ownership)
    if overlap:
        rendered = ", ".join(f"{name}.{field}" for name, field in overlap[:8])
        raise OptimizerInputError(
            "outer design and in-objective matcher cannot own the same lattice field: " + rendered
        )
    required_population = max(2, math.ceil(ratio * count))
    population = budget["effective"]["population_size"]
    if population_size is None:
        population = max(population, required_population)
    by_field = design_parameterization.as_dict()["variable_count_by_field"]
    if population < required_population:
        raise OptimizerInputError(
            f"population_size={population} is below the caller-declared minimum "
            f"of {required_population} for {count} outer knobs (by field: {by_field})."
        )
    if budget["effective"] != {"population_size": population, "generations": generation_count}:
        budget["effective"] = {"population_size": population, "generations": generation_count}
        budget["summary"] = (f"PSO budget: {population} particles for {generation_count} "
                             f"generations ({count} knobs), including the caller-declared planner minimum.")
    table = []
    for owner, parameterization in (("outer_design", design_parameterization),
                                     ("inner_matching", matching_parameterization)):
        for knob in parameterization.knobs:
            table.append({"owner": owner, "knob": knob.name, "field": knob.field,
                          "elements": ", ".join(knob.target_names), "unit": knob.unit,
                          "lower": knob.lower, "upper": knob.upper, "initial": knob.initial})
    return {
        "outer_knobs": count,
        "matching_knobs": len(matching_parameterization.knobs),
        "outer_knobs_by_field": by_field,
        "population_size": population,
        "generations": generation_count,
        "minimum_population": required_population,
        "population_per_outer_knob": population / count,
        "minimum_population_per_knob": ratio,
        "minimum_production_generations": minimum_generations,
        "knob_table": table,
        "sum_constraints": design_parameterization.as_dict()["sum_constraints"],
        "feasibility_checked": False,
        "budget": budget,
    }


def optimize_lattice_parameters(
    baseline_elements: LatticeRows,
    design_parameterization: LatticeParameterization,
    matching_parameterization: LatticeParameterization,
    evaluate_metrics: MetricEvaluator,
    evaluate_objectives: ObjectiveEvaluator,
    *,
    match_targets: Mapping[str, Any],
    objective_names: Sequence[str],
    population_size: int | None = None,
    generations: int | None = None,
    seed: int = 0,
    selection: str = "first_objective",
    seed_positions: Sequence[Sequence[float]] | None = None,
    max_match_iterations: int = 12,
    inner_match_evaluation_budget: int = 32,
    feasible_basin_population_fraction: float = 0.8,
    initial_velocity_fraction: float = 0.02,
    require_feasible_initial_population: bool = True,
    initialization_attempts_per_particle: int = 100,
    initial_feasible_sigma_fraction: float = 0.002,
    minimum_population_per_knob: float = 0.0,
    minimum_production_generations: int = 0,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    cancellation_check: Callable[[], None] | None = None,
    batch_evaluator_factory: PositionBatchEvaluatorFactory | None = None,
    worker_physics_factory: WorkerPhysicsFactory | None = None,
    parallel_workers: int | None = None,
    evaluation_batch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
    evaluation_cache_directory: str | Path | None = None,
    campaign_identity: str | None = None,
    resume: bool = False,
    local_polish_evaluations: int = 0,
    local_polish_initial_step_fraction: float = 0.005,
) -> dict[str, Any]:
    """Run outer PSO with matching performed inside every objective call.

    ``evaluate_objectives`` owns the scientific quantities and hard constraints.
    It receives only a successfully matched explicit lattice and must return the
    normal population-optimizer mapping (``objectives``, optional
    ``constraint_violations``, and optional compact metrics).

    For process-parallel JuTrack evaluation, provide ``worker_physics_factory``.
    It receives baseline rows and must build process-local JuTrack state before
    returning ``(evaluate_metrics, evaluate_objectives)``.  Live Julia objects
    must never be captured by that factory.
    """

    if isinstance(objective_names, (str, bytes)):
        raise OptimizerInputError("objective_names must be a nonempty sequence of distinct names")
    objective_names = tuple(objective_names)
    if (not objective_names or any(not isinstance(name, str) or not name.strip() for name in objective_names)
            or len(set(objective_names)) != len(objective_names)):
        raise OptimizerInputError("objective_names must be a nonempty sequence of distinct names")
    if not callable(evaluate_metrics) or not callable(evaluate_objectives):
        raise TypeError("evaluate_metrics and evaluate_objectives must be callable")
    _target_arrays(match_targets)  # Validate shape before any callback or worker starts.
    plan = plan_lattice_optimization(
        design_parameterization, matching_parameterization,
        population_size=population_size, generations=generations,
        minimum_population_per_knob=minimum_population_per_knob,
        minimum_production_generations=minimum_production_generations,
    )
    if resume:
        if checkpoint_path is None:
            raise OptimizerInputError("resume=True requires checkpoint_path")
        from .optimization_budget import resolve_population_budget
        from .population_optimization import _load_pso_checkpoint

        saved = _load_pso_checkpoint(Path(checkpoint_path).resolve())
        resumed_budget = resolve_population_budget(
            len(design_parameterization.knobs), population_size, generations,
            resumed_population_size=int(saved["contract"]["population_size"]),
            completed_generation=int(saved["completed_generation"]),
            saved_generation_target=(saved.get("budget", {}).get("effective", {}).get("generations")
                                     if saved.get("budget") else None),
            checkpoint_complete=bool(saved.get("complete")),
        )
        plan.update(resumed_budget["effective"])
        plan["population_per_outer_knob"] = plan["population_size"] / plan["outer_knobs"]
        plan["budget"] = resumed_budget
    population_size, generations = plan["population_size"], plan["generations"]
    _publish_progress(progress_callback, "budget", {
        "summary": plan["budget"]["summary"], "budget": plan["budget"],
        "requested_population_size": plan["budget"]["requested"]["population_size"],
        "requested_generations": plan["budget"]["requested"]["generations"],
        "population_size": population_size, "total_generations": generations,
    })
    baseline = _rows(baseline_elements)
    if parallel_workers is None and worker_physics_factory is not None:
        from .research_environment import ResearchEnvironment

        configured_compute = ResearchEnvironment.load()
        worker_count = (
            configured_compute.local_worker_processes
            if configured_compute.compute_backend == "local_process"
            else 1
        )
    else:
        worker_count = int(parallel_workers) if parallel_workers is not None else 1
    worker_count = min(worker_count, max(1, int(population_size)))
    if worker_count < 1:
        raise OptimizerInputError("parallel_workers must be positive")
    if batch_evaluator_factory is not None and worker_physics_factory is not None:
        raise OptimizerInputError(
            "use either batch_evaluator_factory or worker_physics_factory, not both"
        )
    if worker_count > 1 and worker_physics_factory is None and batch_evaluator_factory is None:
        raise OptimizerInputError(
            "parallel JuTrack evaluation requires worker_physics_factory so every process "
            "can build independent Julia and lattice state"
        )
    if worker_count > 1 and worker_physics_factory is not None:
        from .parallel_safety import validate_worker_factory

        validate_worker_factory(worker_physics_factory)
    basin_fraction = float(feasible_basin_population_fraction)
    if not math.isfinite(basin_fraction) or not 0.0 <= basin_fraction <= 1.0:
        raise OptimizerInputError("feasible_basin_population_fraction must be between 0 and 1")
    population_ratio = plan["minimum_population_per_knob"]
    minimum_generations = plan["minimum_production_generations"]
    attempts_per_particle = int(initialization_attempts_per_particle)
    if attempts_per_particle < 1:
        raise OptimizerInputError("initialization_attempts_per_particle must be positive")
    seed_sigma = float(initial_feasible_sigma_fraction)
    if not math.isfinite(seed_sigma) or not 0.0 < seed_sigma <= 0.25:
        raise OptimizerInputError("initial_feasible_sigma_fraction must be in the interval (0, 0.25]")
    inner_budget = int(inner_match_evaluation_budget)
    if inner_budget < 2:
        raise OptimizerInputError("inner_match_evaluation_budget must be at least 2")
    polish_budget = int(local_polish_evaluations)
    if polish_budget < 0:
        raise OptimizerInputError("local_polish_evaluations cannot be negative")
    if 0 < polish_budget < 2 * len(design_parameterization.knobs):
        raise OptimizerInputError(
            "a local polish must poll both directions of every outer knob; "
            f"use at least {2 * len(design_parameterization.knobs)} evaluations"
        )
    polish_step = float(local_polish_initial_step_fraction)
    if not math.isfinite(polish_step) or not 0.0 < polish_step <= 0.25:
        raise OptimizerInputError("local_polish_initial_step_fraction must be in (0, 0.25]")
    metric_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def cached_metrics(rows: list[dict[str, Any]]) -> Mapping[str, Any]:
        key = _physical_rows_key(rows)
        if key not in metric_cache:
            metric_cache[key] = _call_metrics(evaluate_metrics, rows)
        return metric_cache[key]

    _publish_progress(progress_callback, "baseline_matching", {
        "outer_knobs": len(design_parameterization.knobs),
        "matching_knobs": len(matching_parameterization.knobs),
        "parallel_workers": worker_count, "production_generations": 0,
    })
    baseline_match = bounded_response_match(
        baseline,
        matching_parameterization,
        cached_metrics,
        match_targets,
        max_iterations=max_match_iterations,
    )
    if not baseline_match["success"]:
        target_names, target_values, target_tolerances = _target_arrays(match_targets)
        observations = [{"observable": name, "measured": baseline_match.get("metrics", {}).get(name),
                         "target": float(value), "tolerance": float(tolerance)}
                        for name, value, tolerance in zip(target_names, target_values, target_tolerances)]
        raise ResearchInfeasible(
            "The baseline matcher did not reach the declared targets within this attempt. "
            "No production optimization was started; this does not establish physical impossibility.",
            code="baseline_matching_not_reached",
            diagnostics={"baseline_match": baseline_match, "targets": dict(match_targets),
                "target_observations": observations,
                "design_knobs": design_parameterization.as_dict(),
                "matching_knobs": matching_parameterization.as_dict(),
                "max_match_iterations": max_match_iterations, "production_generations": 0},
        )
    matched_baseline = list(baseline_match["elements"])
    baseline_matching_position = list(baseline_match["position"])
    if worker_physics_factory is not None and worker_count > 1:
        external_batch_evaluator = ProcessBatchEvaluator(
            _LatticeWorkerEvaluatorFactory(
                physics_factory=worker_physics_factory,
                baseline_elements=matched_baseline,
                design_parameterization=design_parameterization,
                matching_parameterization=matching_parameterization,
                match_targets=dict(match_targets),
                objective_names=tuple(str(name) for name in objective_names),
                baseline_matching_position=tuple(baseline_matching_position),
                max_match_iterations=int(max_match_iterations),
                inner_match_evaluation_budget=inner_budget,
            ),
            workers=worker_count,
        )
    elif batch_evaluator_factory is not None:
        external_batch_evaluator = batch_evaluator_factory(
            matched_baseline, baseline_matching_position
        )
    else:
        external_batch_evaluator = None

    def close_external_batch_evaluator() -> None:
        close = getattr(external_batch_evaluator, "close", None)
        if callable(close):
            close()
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    matched_lattices: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    matching_evaluation_counts: list[int] = []
    matching_successes = 0
    matching_failures = 0
    warm_starts = _MatchingWarmStartBank(
        design_parameterization.bounds,
        design_parameterization.initial_position,
        baseline_matching_position,
    )

    def evaluate(position: np.ndarray) -> Mapping[str, Any]:
        nonlocal matching_successes, matching_failures
        if cancellation_check is not None:
            cancellation_check()
        key = tuple(map(float, position))
        if key in cache:
            return cache[key]
        try:
            matching_start, warm_start_source = warm_starts.nearest(position)
            target = _apply_design_proposal(design_parameterization, position, matched_baseline)
            matched = bounded_fast_response_match(
                target,
                matching_parameterization,
                cached_metrics,
                match_targets,
                initial_position=matching_start,
                max_iterations=min(4, max(1, int(max_match_iterations))),
                max_evaluations=inner_budget,
            )
            matching_evaluation_counts.append(int(matched.get("evaluations", 0)))
            if not matched["success"]:
                matching_failures += 1
                result = {
                    "objectives": {str(name): math.inf for name in objective_names},
                    "constraint_violations": {"in_objective_matching": 1.0},
                    "feasible": False,
                    "matching": {
                        "reason": matched["reason"],
                        "evaluations": matched.get("evaluations", 0),
                        "evaluation_budget": matched.get("evaluation_budget"),
                    },
                }
            else:
                matching_successes += 1
                matched_rows = copy.deepcopy(matched["elements"])
                warm_starts.record(position, matched["position"])
                raw = _call_objectives(evaluate_objectives, matched_rows, objective_names)
                raw["matching"] = {
                    "reason": matched["reason"],
                    "evaluations": matched["evaluations"],
                    "position": matched["position"],
                    "metrics": matched["metrics"],
                    "warm_start_source": warm_start_source,
                    "elements": copy.deepcopy(matched_rows),
                }
                result = raw
                matched_lattices[key] = copy.deepcopy(matched_rows)
        except ResearchInfeasible as exc:
            result = _objective_result(_physical_rejection(exc), objective_names)
        cache[key] = result
        return result

    def evaluate_batch(positions: Sequence[np.ndarray]) -> list[Mapping[str, Any]]:
        nonlocal matching_successes, matching_failures
        if external_batch_evaluator is None:
            return [evaluate(position) for position in positions]
        results: list[Mapping[str, Any] | None] = [None] * len(positions)
        missing_positions: list[np.ndarray] = []
        missing_indices: list[int] = []
        for index, position in enumerate(positions):
            if cancellation_check is not None:
                cancellation_check()
            key = tuple(map(float, position))
            if key in cache:
                results[index] = cache[key]
            else:
                missing_positions.append(np.asarray(position, dtype=float).copy())
                missing_indices.append(index)
        if missing_positions:
            evaluated = list(external_batch_evaluator(missing_positions))
            if len(evaluated) != len(missing_positions):
                raise ValueError("batch evaluator returned a different number of candidate results")
            for result_index, position, supplied in zip(
                missing_indices, missing_positions, evaluated
            ):
                raw = _objective_result(supplied, objective_names)
                key = tuple(map(float, position))
                cache[key] = raw
                results[result_index] = raw
                matching_info = raw.get("matching", {})
                if isinstance(matching_info, Mapping):
                    if "elements" in matching_info:
                        actual = _rows(matching_info["elements"])
                        require_same_lattice_topology(matched_baseline, actual)
                        matched_lattices[key] = copy.deepcopy(actual)
                    count = int(matching_info.get("evaluations", 0))
                    matching_evaluation_counts.append(count)
                    if raw.get("feasible") is False and "in_objective_matching" in dict(
                        raw.get("constraint_violations", {})
                    ):
                        matching_failures += 1
                    else:
                        matching_successes += 1
        return [dict(item or {}) for item in results]

    initial_array = np.asarray(design_parameterization.initial_position, dtype=float)
    initialization: dict[str, Any]
    if resume:
        supplied_seeds: list[Sequence[float]] = []
        initialization = {
            "strategy": "restored_from_optimizer_checkpoint",
            "required_all_feasible": require_feasible_initial_population,
            "population_size": int(population_size),
            "checkpoint_path": str(checkpoint_path),
        }
    else:
        _publish_progress(progress_callback, "baseline_objective_validation", {
            "production_generations": 0,
            "matching_evaluations": baseline_match.get("evaluations", 0),
        })
        try:
            initial_evaluation = dict(evaluate(initial_array))
        except Exception:
            close_external_batch_evaluator()
            raise
        initial_violations = dict(initial_evaluation.get("constraint_violations", {}))
        initial_is_feasible = bool(
            initial_evaluation.get(
                "feasible",
                all(float(value) <= 1.0e-12 for value in initial_violations.values()),
            )
        ) and not any(float(value) > 1.0e-12 for value in initial_violations.values())
        if require_feasible_initial_population and not initial_is_feasible:
            close_external_batch_evaluator()
            raise ResearchInfeasible(
                "The initial seed failed the complete optimization objective contract. "
                "No all-feasible production population was started.",
                code="initial_objective_not_feasible",
                diagnostics={"initial_evaluation": initial_evaluation,
                    "matched_baseline": matched_baseline, "production_generations": 0},
            )

    if not resume and require_feasible_initial_population:
        requested = max(0, int(population_size) - 1)
        accepted_seeds: list[list[float]] = []
        seen = {tuple(round(float(value), 14) for value in initial_array)}
        attempted = 0
        rejected = 0
        supplied = list(seed_positions or ())
        lower = np.asarray([item[0] for item in design_parameterization.bounds], dtype=float)
        upper = np.asarray([item[1] for item in design_parameterization.bounds], dtype=float)
        width = upper - lower
        screening_batch_size = int(
            evaluation_batch_size
            if evaluation_batch_size is not None
            else max(worker_count, min(64, max(1, math.ceil(int(population_size) / 10))))
        )

        def prepared(candidates: Sequence[Sequence[float]]) -> list[np.ndarray]:
            nonlocal rejected
            proposals: list[np.ndarray] = []
            for candidate in candidates:
                proposal = np.asarray(candidate, dtype=float)
                if proposal.shape != initial_array.shape or not np.all(np.isfinite(proposal)):
                    rejected += 1
                    continue
                proposal = np.clip(proposal, lower, upper)
                key = tuple(round(float(value), 14) for value in proposal)
                if key in seen:
                    continue
                seen.add(key)
                proposals.append(proposal)
            return proposals

        def screen(candidates: Sequence[Sequence[float]]) -> None:
            nonlocal attempted, rejected
            if len(accepted_seeds) >= requested:
                return
            proposals = prepared(candidates)
            for start in range(0, len(proposals), screening_batch_size):
                if len(accepted_seeds) >= requested:
                    break
                batch = proposals[start : start + screening_batch_size]
                _publish_progress(progress_callback, "feasible_population_screening", {
                    "screened_candidates": attempted,
                    "accepted_nonbaseline_seeds": len(accepted_seeds),
                    "required_nonbaseline_seeds": requested,
                    "rejected_candidates": rejected,
                    "pending_batch_size": len(batch), "parallel_workers": worker_count,
                    "production_generations": 0,
                })
                try:
                    results = evaluate_batch(batch)
                except Exception:
                    close_external_batch_evaluator()
                    raise
                attempted += len(batch)
                for proposal, result in zip(batch, results):
                    violations = dict(result.get("constraint_violations", {}))
                    feasible = bool(
                        result.get(
                            "feasible",
                            all(float(value) <= 1.0e-12 for value in violations.values()),
                        )
                    ) and all(float(value) <= 1.0e-12 for value in violations.values())
                    if feasible and len(accepted_seeds) < requested:
                        accepted_seeds.append(proposal.tolist())
                    elif not feasible:
                        rejected += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "feasible_population_screening",
                            "screened_candidates": attempted,
                            "accepted_nonbaseline_seeds": len(accepted_seeds),
                            "required_nonbaseline_seeds": requested,
                            "rejected_candidates": rejected,
                            "evaluation_backend": (
                                "process_batch"
                                if worker_physics_factory is not None and worker_count > 1
                                else "batch"
                                if external_batch_evaluator is not None
                                else "serial"
                            ),
                        }
                    )

        supplied_and_local: list[Sequence[float]] = list(supplied)
        supplied_and_local.extend(
            local_seed_positions(
                design_parameterization,
                relative_step=min(0.002, seed_sigma),
                maximum=requested,
            )
        )
        screen(supplied_and_local)
        rng = np.random.default_rng(int(seed))
        maximum_attempts = max(requested, requested * attempts_per_particle)
        while len(accepted_seeds) < requested and attempted < maximum_attempts:
            count = min(screening_batch_size, maximum_attempts - attempted)
            candidates: list[np.ndarray] = []
            for offset in range(count):
                progress = (attempted + offset) / max(1, maximum_attempts)
                # Start close to the proven design basin, then broaden gradually.
                sigma = seed_sigma * (1.0 + 7.0 * progress**2)
                candidates.append(
                    np.clip(
                        initial_array + rng.normal(0.0, sigma, len(initial_array)) * width,
                        lower,
                        upper,
                    )
                )
            before = attempted
            screen(candidates)
            if attempted == before:
                break
        if len(accepted_seeds) < requested:
            close_external_batch_evaluator()
            raise ResearchInfeasible(
                "This initialization attempt did not find an all-feasible population: "
                f"accepted {len(accepted_seeds) + 1}/{requested + 1} total particles "
                f"after {attempted} screened candidates. Production optimization did not start.",
                code="feasible_population_not_found",
                diagnostics={"accepted_particles": len(accepted_seeds) + 1,
                    "required_particles": requested + 1, "screened_candidates": attempted,
                    "baseline": matched_baseline, "accepted_positions": accepted_seeds,
                    "production_generations": 0},
            )
        supplied_seeds = accepted_seeds
        initialization = {
            "strategy": "batched_screened_feasible_basin_random_search",
            "required_all_feasible": True,
            "population_size": int(population_size),
            "feasible_generation_zero": int(population_size),
            "screened_candidates": attempted,
            "rejected_candidates": rejected,
            "accepted_nonbaseline_seeds": len(accepted_seeds),
            "attempts_per_particle_limit": attempts_per_particle,
            "initial_sigma_fraction": seed_sigma,
            "screening_batch_size": screening_batch_size,
            "screening_backend": (
                "process_batch"
                if worker_physics_factory is not None and worker_count > 1
                else "batch"
                if external_batch_evaluator is not None
                else "serial"
            ),
        }
    elif not resume and seed_positions is None:
        local_count = min(
            max(0, int(population_size) - 1),
            max(0, round((int(population_size) - 1) * basin_fraction)),
        )
        supplied_seeds = feasible_basin_seed_positions(
            design_parameterization,
            count=local_count,
            seed=seed,
        )
        initialization = {
            "strategy": "unscreened_local_basin",
            "required_all_feasible": False,
            "supplied_seed_count": len(supplied_seeds),
        }
    elif not resume:
        supplied_seeds = list(seed_positions)
        initialization = {
            "strategy": "caller_supplied_unscreened",
            "required_all_feasible": False,
            "supplied_seed_count": len(supplied_seeds),
        }
    try:
        optimization = particle_swarm_optimize(
            evaluate,
            bounds=design_parameterization.bounds,
            objective_names=objective_names,
            initial_position=design_parameterization.initial_position,
            variable_names=design_parameterization.variable_names,
            population_size=population_size,
            generations=generations,
            seed=seed,
            selection=selection,
            seed_positions=supplied_seeds,
            initial_velocity_fraction=initial_velocity_fraction,
            progress_callback=progress_callback,
            evaluate_batch=(evaluate_batch if external_batch_evaluator is not None else None),
            evaluation_batch_size=evaluation_batch_size,
            cancellation_check=cancellation_check,
            checkpoint_path=checkpoint_path,
            evaluation_cache_directory=evaluation_cache_directory,
            campaign_identity=campaign_identity,
            resume=resume,
        )
    finally:
        close_external_batch_evaluator()
    local_polish = {
        "completed": False,
        "method": "bounded_deterministic_coordinate_poll",
        "evaluations": 0,
        "initial_step_fraction": polish_step,
        "all_dimensions_polled": False,
        "initial_objective": None,
        "final_objective": None,
        "relative_improvement": None,
    }
    if polish_budget:
        primary = str(objective_names[0])
        best_serial = dict(optimization["best_candidate"])
        position = np.asarray(
            [best_serial["variables"][name] for name in design_parameterization.variable_names],
            dtype=float,
        )
        lower = np.asarray([item.lower for item in design_parameterization.knobs], dtype=float)
        upper = np.asarray([item.upper for item in design_parameterization.knobs], dtype=float)
        span = upper - lower
        initial_objective = float(best_serial["objectives"][primary])
        best_objective = initial_objective
        best_raw: Mapping[str, Any] | None = None
        best_position = position.copy()
        evaluations = 0
        dimensions_polled: set[int] = set()
        step = polish_step
        while evaluations < polish_budget and step >= 1.0e-5:
            improved_in_sweep = False
            for dimension in range(len(position)):
                for direction in (-1.0, 1.0):
                    if evaluations >= polish_budget:
                        break
                    proposal = position.copy()
                    proposal[dimension] = np.clip(
                        proposal[dimension] + direction * step * span[dimension],
                        lower[dimension],
                        upper[dimension],
                    )
                    if proposal[dimension] == position[dimension]:
                        dimensions_polled.add(dimension)
                        continue
                    raw = dict(evaluate(proposal))
                    evaluations += 1
                    dimensions_polled.add(dimension)
                    objectives = raw.get("objectives", {})
                    value = objectives.get(primary) if isinstance(objectives, Mapping) else None
                    violations = raw.get("constraint_violations", {})
                    feasible = bool(raw.get("feasible")) and isinstance(violations, Mapping) and all(
                        float(item) <= 1.0e-12 for item in violations.values()
                    )
                    if feasible and _finite_number(value) is not None and float(value) < best_objective:
                        position = proposal
                        best_position = proposal.copy()
                        best_objective = float(value)
                        best_raw = raw
                        improved_in_sweep = True
                if evaluations >= polish_budget:
                    break
            if not improved_in_sweep:
                step *= 0.5
            if len(dimensions_polled) == len(position) and evaluations >= polish_budget:
                break
        if best_raw is not None:
            violations = dict(best_raw.get("constraint_violations", {}))
            metrics = {
                key: value
                for key, value in best_raw.items()
                if key
                not in {
                    "objectives",
                    "objective_values",
                    "constraint_violations",
                    "constraint_violation",
                    "feasible",
                }
            }
            optimization["best_candidate"] = {
                "candidate_id": "local-polish-best",
                "variables": {
                    name: float(value)
                    for name, value in zip(design_parameterization.variable_names, best_position)
                },
                "objectives": {
                    str(name): float(best_raw["objectives"][name])
                    for name in objective_names
                },
                "feasible": True,
                "constraint_violation": sum(max(0.0, float(value)) for value in violations.values()),
                "constraint_violations": violations,
                "metrics": metrics,
            }
        local_polish = {
            "completed": len(dimensions_polled) == len(position),
            "method": "bounded_deterministic_coordinate_poll",
            "evaluations": evaluations,
            "evaluation_budget": polish_budget,
            "initial_step_fraction": polish_step,
            "final_step_fraction": step,
            "all_dimensions_polled": len(dimensions_polled) == len(position),
            "initial_objective": initial_objective,
            "final_objective": best_objective,
            "relative_improvement": (
                max(0.0, initial_objective - best_objective)
                / max(abs(initial_objective), 1.0e-30)
            ),
        }
    best = optimization["best_candidate"]
    key = tuple(
        float(best["variables"][name]) for name in design_parameterization.variable_names
    )
    selected_elements = matched_lattices.get(key)
    if selected_elements is None:
        matching_info = best.get("metrics", {}).get("matching", {})
        if isinstance(matching_info, Mapping) and "elements" in matching_info:
            selected_elements = _rows(matching_info["elements"])
            require_same_lattice_topology(matched_baseline, selected_elements)
        matching_position = (
            matching_info.get("position") if isinstance(matching_info, Mapping) else None
        )
        # Older checkpoints may only contain full-precision coordinates. New
        # receipts carry the exact post-match rows, including process workers.
        if selected_elements is None and isinstance(matching_position, (list, tuple)):
            selected_geometry = design_parameterization.apply(
                [best["variables"][name] for name in design_parameterization.variable_names],
                base_elements=matched_baseline,
            )
            selected_elements = matching_parameterization.apply(
                matching_position, base_elements=selected_geometry
            )
    history = list(optimization.get("convergence", {}).get("history", []))
    primary_name = str(objective_names[0])
    generation_zero_best = (
        float(history[0].get("best", {}).get(primary_name, math.inf)) if history else math.inf
    )
    final_best = (
        float(history[-1].get("best", {}).get(primary_name, math.inf)) if history else math.inf
    )
    improvement = generation_zero_best - final_best
    convergence_diagnostic = {
        "primary_objective": primary_name,
        "generation_zero_best": generation_zero_best,
        "final_generation_best": final_best,
        "absolute_improvement_after_generation_zero": improvement,
        "improved_after_generation_zero": (
            math.isfinite(improvement)
            and improvement > 1.0e-12 * max(1.0, abs(generation_zero_best))
        ),
        "front_stagnated_after_initialization": not (
            math.isfinite(improvement)
            and improvement > 1.0e-12 * max(1.0, abs(generation_zero_best))
        ),
        "population_per_outer_knob": (
            int(population_size) / len(design_parameterization.knobs)
        ),
        "minimum_population_per_knob": population_ratio,
        "requested_generations": int(generations),
        "minimum_production_generations": minimum_generations,
        # One bounded campaign can demonstrate progress or stagnation, but not
        # architectural saturation.  That requires independent starts, a
        # plateau window, local polish, and active-constraint diagnosis.
        "saturation_demonstrated": False,
    }
    return {
        **optimization,
        "optimization_plan": plan,
        "parameterization": design_parameterization.as_dict(),
        "matching_parameterization": matching_parameterization.as_dict(),
        "selected_elements": selected_elements,
        "selected_matching": best.get("metrics", {}).get("matching"),
        "baseline_matching": {
            "position": baseline_matching_position,
            "metrics": baseline_match["metrics"],
            "evaluations": baseline_match["evaluations"],
        },
        "matched_baseline_elements": matched_baseline,
        "initialization": initialization,
        "convergence_diagnostic": convergence_diagnostic,
        "local_polish": local_polish,
        "cached_candidate_count": len(cache),
        "cached_linear_state_count": len(metric_cache),
        "inner_matching_performance": {
            "strategy": "nearest_feasible_warm_start_bounded_response_with_broyden_updates",
            "attempts": matching_successes + matching_failures,
            "successes": matching_successes,
            "failures": matching_failures,
            "evaluation_budget_per_candidate": inner_budget,
            "total_linear_optics_evaluations": sum(matching_evaluation_counts),
            "mean_linear_optics_evaluations_per_candidate": (
                sum(matching_evaluation_counts) / len(matching_evaluation_counts)
                if matching_evaluation_counts
                else 0.0
            ),
            "maximum_linear_optics_evaluations_for_one_candidate": (
                max(matching_evaluation_counts) if matching_evaluation_counts else 0
            ),
            "candidate_evaluation_backend": (
                "process_batch"
                if worker_physics_factory is not None and worker_count > 1
                else "batch"
                if external_batch_evaluator is not None
                else "serial"
            ),
            "parallel_workers": worker_count,
        },
    }
