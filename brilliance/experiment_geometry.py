"""Small exact-geometry operations for model-authored numerical experiments.

These functions neither call a simulator nor establish optical feasibility.
They preserve declared requirements and produce inspectable proposals.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, replace
from typing import Any, Mapping, Sequence

from .jutrack_service import native_element_options
from .lattice import LatticeElementIR, LatticeInputError
from .lattice_editing import _materialize
from .lattice_optimization import LatticeParameterization, LinearSumConstraint
from .population_optimization import OptimizerInputError


def project_weighted_sum(values, lower, upper, weights, target) -> list[float]:
    """Project onto a positive-weight sum and box, without a physics callback.

    This is the Euclidean box/hyperplane projection, not proportional scaling.
    Every feasible declared weighted sum can be represented, including bounds.
    """
    vectors = [list(map(float, item)) for item in (values, lower, upper, weights)]
    point, lo, hi, multiplicity = vectors
    goal = float(target)
    if not point or any(len(item) != len(point) for item in vectors):
        raise OptimizerInputError("weighted sum requires nonempty equal-length vectors")
    if (not all(math.isfinite(x) for item in vectors for x in item) or not math.isfinite(goal)
            or any(a > b for a, b in zip(lo, hi)) or any(w <= 0 for w in multiplicity)):
        raise OptimizerInputError("weighted sum needs finite ordered bounds and positive weights")
    minimum = math.fsum(a * w for a, w in zip(lo, multiplicity))
    maximum = math.fsum(b * w for b, w in zip(hi, multiplicity))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise OptimizerInputError("weighted bounds overflow the finite sum domain")
    rounding = 8 * math.ulp(max(abs(goal), abs(minimum), abs(maximum)))
    if goal < minimum - rounding or goal > maximum + rounding:
        raise OptimizerInputError(f"weighted sum target {goal!r} is outside attainable [{minimum!r}, {maximum!r}]")
    left = min((a - x) / w for x, a, w in zip(point, lo, multiplicity))
    right = max((b - x) / w for x, b, w in zip(point, hi, multiplicity))
    result = point
    for _ in range(100):
        multiplier = left + (right - left) / 2
        result = [min(b, max(a, x + multiplier * w)) for x, a, b, w in zip(point, lo, hi, multiplicity)]
        total = math.fsum(x * w for x, w in zip(result, multiplicity))
        if total == goal:
            break
        if total < goal:
            left = multiplier
        else:
            right = multiplier
    # Remove floating-point bisection residue using only available box slack.
    remainder = goal - math.fsum(x * w for x, w in zip(result, multiplicity))
    for i, w in enumerate(multiplicity):
        result[i] = min(hi[i], max(lo[i], result[i] + remainder / w))
        remainder = goal - math.fsum(x * weight for x, weight in zip(result, multiplicity))
        if remainder == 0:
            break
    if abs(remainder) > rounding:
        raise OptimizerInputError("weighted sum could not be resolved within floating-point precision")
    return result


def prepare_length_budget(parameterization: LatticeParameterization, *, total_length_m: float,
                          tolerance_m: float = 1e-9) -> dict[str, Any]:
    """Derive a full-sequence length budget from actual fixed rows and free knobs.

    Each controlled physical index contributes once, including globally mirrored
    pairs or larger groups. Returns the existing parameterization type with a
    bounded exact-sum projection and consistent feasible initial coordinates.
    Other fields and their independent sum constraints remain unchanged.
    A feasible length allocation is not stable optics or an accepted design.
    """
    if not isinstance(parameterization, LatticeParameterization):
        raise TypeError("parameterization must be a compiled LatticeParameterization")
    if (isinstance(total_length_m, bool) or not math.isfinite(total_length_m) or total_length_m <= 0
            or isinstance(tolerance_m, bool) or not math.isfinite(tolerance_m) or tolerance_m < 0):
        raise OptimizerInputError("total_length_m must be positive and tolerance_m nonnegative, both finite")
    rows = _materialize(parameterization.baseline)
    names = parameterization.variable_names
    if len(names) != len(set(names)):
        raise OptimizerInputError("length budgets require unique compiled knob names")
    length_knobs = [knob for knob in parameterization.knobs if knob.field == "length_m"]
    controlled = set()
    for knob in length_knobs:
        indices = knob.target_indices
        if (not indices or len(indices) != len(set(indices))
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(rows) for i in indices)
                or tuple(rows[i]["name"] for i in indices) != knob.target_names):
            raise OptimizerInputError(f"length knob {knob.name!r} has inconsistent physical targets")
        if controlled.intersection(indices):
            raise OptimizerInputError("length knobs cannot own overlapping physical elements")
        controlled.update(indices)
        if (not all(math.isfinite(v) for v in (knob.lower, knob.upper, knob.initial))
                or knob.lower < 0 or knob.lower >= knob.upper or not knob.lower <= knob.initial <= knob.upper):
            raise OptimizerInputError(f"length knob {knob.name!r} has invalid nonnegative bounds or initial value")
    occupied, other_constraints = set(), []
    free_names = tuple(knob.name for knob in length_knobs)
    fixed = math.fsum(float(row["length_m"]) for i, row in enumerate(rows) if i not in controlled)
    free_target = float(total_length_m) - fixed
    for constraint in parameterization.sum_constraints:
        selected = set(constraint.variable_names)
        if (len(selected) != len(constraint.variable_names) or selected - set(names)
                or selected & occupied):
            raise OptimizerInputError("duplicate, unknown or overlapping sum-constraint variables")
        occupied.update(selected)
        if constraint.field == "length_m":
            if selected != set(free_names) or not math.isclose(constraint.target, free_target, rel_tol=0, abs_tol=tolerance_m):
                raise OptimizerInputError("existing length constraint conflicts with the full fixed/free budget; do not stack overlapping budgets")
        else:
            other_constraints.append(constraint)
    multiplicities = [len(knob.target_indices) for knob in length_knobs]
    lower = [knob.lower for knob in length_knobs]
    upper = [knob.upper for knob in length_knobs]
    minimum = fixed + math.fsum(a * w for a, w in zip(lower, multiplicities))
    maximum = fixed + math.fsum(b * w for b, w in zip(upper, multiplicities))
    if length_knobs:
        feasible = project_weighted_sum([knob.initial for knob in length_knobs], lower, upper, multiplicities, free_target)
        by_name = dict(zip(free_names, feasible))
        knobs = tuple(replace(knob, initial=by_name.get(knob.name, knob.initial)) for knob in parameterization.knobs)
        constraints = (*other_constraints, LinearSumConstraint("length_m", free_target, free_names, "bounded_sum"))
        prepared = LatticeParameterization(tuple(rows), knobs, constraints)
    else:
        if not math.isclose(fixed, total_length_m, rel_tol=0, abs_tol=tolerance_m):
            raise OptimizerInputError(f"all lengths are fixed at {fixed!r} m, not requested {total_length_m!r} m")
        prepared = replace(parameterization, baseline=tuple(rows))
    actual_initial = prepared.projected_position(prepared.initial_position)
    prepared = replace(prepared, knobs=tuple(replace(knob, initial=value) for knob, value in zip(prepared.knobs, actual_initial)))
    actual_length = math.fsum(row["length_m"] for row in prepared.apply(prepared.initial_position))
    if not math.isclose(actual_length, total_length_m, rel_tol=0, abs_tol=tolerance_m):
        raise OptimizerInputError("compiled length budget does not reproduce the requested total")
    return {"parameterization": prepared, "initial_position": prepared.initial_position,
            "budget": {"total_length_m": float(total_length_m), "fixed_length_m": fixed,
                       "free_length_target_m": free_target, "minimum_total_length_m": minimum,
                       "maximum_total_length_m": maximum, "initial_total_length_m": actual_length,
                       "initial_length_error_m": actual_length - total_length_m,
                       "fixed_element_names": [row["name"] for i, row in enumerate(rows) if i not in controlled],
                       "free_variables": [{"name": knob.name, "element_names": list(knob.target_names),
                                           "multiplicity": count, "bounds_m": [knob.lower, knob.upper]}
                                          for knob, count in zip(length_knobs, multiplicities)],
                       "independent_length_degrees_of_freedom": max(0, len(length_knobs) - 1)},
            "feasible": True, "scope": "length_budget_only", "physics_verified": False}


def _verify_global_mirror(rows):
    items = [LatticeElementIR.from_mapping(row) for row in rows]
    for i in range((len(items) + 1) // 2):
        left, right = items[i], items[-1 - i]
        if left.kind != right.kind or any(not math.isclose(float(getattr(left, key) or 0), float(getattr(right, key) or 0),
                rel_tol=1e-10, abs_tol=1e-12) for key in ("length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4")):
            raise LatticeInputError(f"global physical mirror mismatch: {left.name!r} / {right.name!r}")
        options, partner = native_element_options(left), native_element_options(right)
        for entry, exit_ in (("e1", "e2"), ("fint1", "fint2"), ("FringeBendEntrance", "FringeBendExit"),
                             ("FringeQuadEntrance", "FringeQuadExit"), ("FringeIntM0", "FringeIntP0")):
            if entry in options or exit_ in options:
                a, b = options.pop(entry, None), options.pop(exit_, None)
                if a is not None:
                    options[exit_] = a
                if b is not None:
                    options[entry] = b
        # Coordinate transforms/skew/kicks need a dedicated mirror convention;
        # this structural operation must not certify it from matching names.
        for option in (options, partner):
            for key in ("T1", "T2", "PolynomA", "KickAngle"):
                if any(float(value) != 0 for value in option.get(key, [])):
                    raise LatticeInputError(f"global mirror verification does not support nonzero {key}")
            for key in ("R1", "R2"):
                matrix = option.get(key)
                if matrix is not None and not (all(value == 0 for row in matrix for value in row)
                        or all(value == float(i == j) for i, row in enumerate(matrix) for j, value in enumerate(row))):
                    raise LatticeInputError(f"global mirror verification does not support nontrivial {key}")
        if options != partner:
            raise LatticeInputError(f"global native-field mirror mismatch: {left.name!r} / {right.name!r}")
        if left.kind in {"RFCA", "CORRECTOR"}:
            raise LatticeInputError(f"global mirror verification of {left.kind} requires an explicit operation-specific convention")


def remove_mirrored_quadrupole_pair(elements: Sequence[LatticeElementIR | Mapping[str, Any]], *,
                                    element_names: Sequence[str]) -> dict[str, Any]:
    """Replace exactly two globally mirrored quadrupoles by equal-length drifts.

    Explicit names and the supplied full-sequence mirror cut define the pair;
    stale local-family/mirror metadata never selects it. No cyclic rotation,
    field rematch, bend modification, or native evaluation occurs. Both names,
    longitudinal spans and passive aperture settings are retained. Rebuild any
    native editor/sampler before evaluating this proposed topology.
    """
    if isinstance(element_names, (str, bytes)) or len(element_names) != 2 or len(set(element_names)) != 2:
        raise LatticeInputError("element_names must specify exactly two distinct saved names")
    rows = _materialize(elements)
    by_name = {row["name"]: i for i, row in enumerate(rows)}
    if any(name not in by_name for name in element_names):
        raise LatticeInputError("quadrupole removal names must exist exactly in the supplied sequence")
    indices = sorted(by_name[name] for name in element_names)
    if indices[0] + indices[1] != len(rows) - 1:
        raise LatticeInputError("selected quadrupoles are not global mirror partners in the supplied sequence")
    if any(rows[i]["kind"] not in {"KQUAD", "QUAD"} for i in indices):
        raise LatticeInputError("only a physical quadrupole pair can be removed by this operation")
    _verify_global_mirror(rows)
    before_length = math.fsum(row["length_m"] for row in rows)
    before_angle = math.fsum(row.get("angle_rad") or 0 for row in rows)
    changes = []
    for i in indices:
        original = copy.deepcopy(rows[i])
        if original["length_m"] <= 0:
            raise LatticeInputError("removed quadrupoles must have finite positive physical length")
        if original.get("angle_rad") not in (None, 0, 0.0):
            raise LatticeInputError("a quadrupole with a declared bend angle cannot be removed while preserving the bend budget")
        options = native_element_options(LatticeElementIR.from_mapping(original))
        fields = {key: value for key, value in options.items() if key in {"RApertures", "EApertures"}}
        fields.update(replaced_quadrupole=original, geometry_operation="quadrupole_to_equal_length_drift")
        rows[i] = asdict(LatticeElementIR(name=original["name"], kind="DRIFT", length_m=original["length_m"], fields=fields))
        changes.append({"index": i, "name": original["name"], "before": original, "after": copy.deepcopy(rows[i])})
    _verify_global_mirror(rows)
    return {"elements": rows, "changes": changes,
            "invariants": {"total_length_m": before_length, "total_bend_angle_rad": before_angle,
                           "length_preserved": math.fsum(row["length_m"] for row in rows) == before_length,
                           "bend_angle_preserved": math.fsum(row.get("angle_rad") or 0 for row in rows) == before_angle,
                           "global_physical_mirror_checked": True, "element_names_and_order_preserved": True},
            "requires_native_rebuild": True, "physics_verified": False,
            "scope": "geometry_and_supported_physical_mirror_fields_only"}
