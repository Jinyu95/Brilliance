"""Composable numerical experiments, independent of a particular design route.

The model defines a question, measurements and permitted edits. This layer owns
proposal -> optional dependent match -> measurement -> durable candidate record.
Feasibility under an experiment is NOT scientific acceptance or a ring receipt.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .lattice_optimization import LatticeParameterization
from .research_outcomes import ResearchInfeasible


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _operation_identity(operation: Callable) -> str:
    try:
        source = inspect.getsource(operation)
    except (TypeError, OSError):
        try:
            source = inspect.getsource(type(operation))
        except (TypeError, OSError):
            source = f"{type(operation).__module__}.{type(operation).__qualname__}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _constructor_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Physical fields consumed by the installed builder beyond mutable scalars.

    Metadata is not a constructor setting. RF/corrector fields, however, are
    consumed directly by JuTrackService rather than native_element_options and
    must participate in rebuild and inner-matcher ownership checks too.
    """
    from .jutrack_service import native_element_options
    from .lattice import LatticeElementIR

    element = LatticeElementIR.from_mapping(row)
    result = native_element_options(element)
    if element.kind == "RFCA":
        for canonical, alias in (("voltage_v", "volt"), ("frequency_hz", "freq")):
            value = element.fields.get(canonical, element.fields.get(alias))
            if not _finite(value):
                raise ValueError(f"{element.name}: {canonical} must be a finite constructor value")
            result[canonical] = float(value)
    elif element.kind == "CORRECTOR":
        for name in ("xkick_rad", "ykick_rad"):
            value = element.fields.get(name, 0.0)
            if not _finite(value):
                raise ValueError(f"{element.name}: {name} must be a finite constructor value")
            result[name] = float(value)
    return result


@dataclass(frozen=True)
class ExperimentDefinition:
    """Named scalar measurements, objective directions and explicit hard bounds.

    Numeric constraints use lower/upper and a positive residual scale. Boolean
    constraints use equals. Put requirements and proposed choices (including
    callback closure parameters) separately in provenance; no hidden defaults
    for energy, geometry, tunes or acceptable accelerator performance exist.
    Objectives are aspirations, never implicit validity gates.
    """

    name: str
    scope: str
    objectives: Mapping[str, str]
    constraints: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name.strip() or not self.scope.strip():
            raise ValueError("Experiment name and physical evaluation scope are required")
        if not self.objectives or any(v not in {"min", "max"} for v in self.objectives.values()):
            raise ValueError("Objectives must map measurement names to 'min' or 'max'")
        for name in [*self.objectives, *self.constraints]:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Measurement names must be nonempty strings")
        for name, rule in self.constraints.items():
            if not isinstance(rule, Mapping) or set(rule) - {"lower", "upper", "scale", "equals", "unit", "source"}:
                raise ValueError(f"Invalid constraint definition for {name}")
            if "equals" in rule:
                if type(rule["equals"]) is not bool or set(rule) & {"lower", "upper", "scale"}:
                    raise ValueError(f"{name}: equals is a boolean constraint, not a numerical target")
            else:
                if not set(rule) & {"lower", "upper"}:
                    raise ValueError(f"{name}: declare a numeric bound or boolean equals")
                if not _finite(rule.get("scale")) or rule["scale"] <= 0:
                    raise ValueError(f"{name}: numerical constraints require a positive finite scale")
                if any(not _finite(rule[key]) for key in ("lower", "upper") if key in rule):
                    raise ValueError(f"{name}: bounds must be finite")
                if rule.get("lower", -math.inf) > rule.get("upper", math.inf):
                    raise ValueError(f"{name}: lower exceeds upper")
        # Freeze caller-owned nested dictionaries by detaching at the boundary.
        for key in ("objectives", "constraints", "provenance"):
            object.__setattr__(self, key, copy.deepcopy(dict(getattr(self, key))))

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy({"name": self.name, "scope": self.scope,
                              "objectives": dict(self.objectives),
                              "constraints": dict(self.constraints),
                              "provenance": dict(self.provenance)})

    def assess(self, measurements: Mapping[str, Any]) -> dict[str, Any]:
        checks = {}
        for name, rule in self.constraints.items():
            value = measurements.get(name)
            if "equals" in rule:
                passed = type(value) is bool and value == rule["equals"]
                violation = 0.0 if passed else 1.0
            elif _finite(value):
                violation = max(0.0, rule.get("lower", -math.inf) - value,
                                value - rule.get("upper", math.inf)) / rule["scale"]
                passed = violation == 0.0
                if not math.isfinite(violation):
                    violation = None
            else:
                passed, violation = False, None
            checks[name] = {"passed": bool(passed), "violation": violation}
        for name in self.objectives:
            checks[f"objective_available:{name}"] = {
                "passed": _finite(measurements.get(name)),
                "violation": 0.0 if _finite(measurements.get(name)) else None,
            }
        return checks


@dataclass(frozen=True)
class MatchingOperation:
    """Dependent matching with exclusive ownership of declared element fields.

    solve(rows, parameters) calls a tested matching operation and returns its
    match_success (line matcher) or success (response matcher), plus elements.
    Solver termination is deliberately not used as matching success.
    """

    parameters: LatticeParameterization
    solve: Callable[[list[dict], LatticeParameterization], Mapping[str, Any]]
    contract: Mapping[str, Any]

    def apply(self, rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], Mapping[str, Any]]:
        before = copy.deepcopy(list(rows))
        from .lattice import require_same_lattice_topology

        require_same_lattice_topology(self.parameters.baseline, before)
        names = [r["name"] for r in before]
        if names != [r["name"] for r in self.parameters.baseline]:
            raise ValueError("Changed topology requires a newly compiled matching operation")
        owned = {(name, knob.field) for knob in self.parameters.knobs for name in knob.target_names}
        knobs = []
        for knob in self.parameters.knobs:
            values = [before[i][knob.field] for i in knob.target_indices]
            if any(v != values[0] for v in values):
                raise ValueError(f"Matching knob {knob.name} has unequal physical mirror values")
            knobs.append(replace(knob, initial=float(values[0])))
        parameters = replace(self.parameters, baseline=tuple(copy.deepcopy(before)), knobs=tuple(knobs))
        # Validate exact initial values before spending numerical evaluations.
        parameters.projected_position(parameters.initial_position)
        result = self.solve(copy.deepcopy(before), parameters)
        if not isinstance(result, Mapping):
            raise TypeError("Matching operation must return the tested matcher's result mapping")
        success = result.get("match_success", result.get("success"))
        if type(success) is not bool:
            raise TypeError("Matching result requires boolean match_success or success")
        returned = result.get("elements")
        if returned is not None and (not isinstance(returned, Sequence) or isinstance(returned, (str, bytes))):
            raise TypeError("Matching elements must be an explicit sequence of element rows")
        if success and not returned:
            raise TypeError("Successful matching requires exact returned elements; inputs are not a result")
        after = copy.deepcopy(list(returned) if returned else before)
        if len(after) != len(before):
            raise ValueError("An inner matcher cannot change topology")
        for original, final in zip(before, after):
            for key in set(original) | set(final):
                if key == "fields":
                    if _constructor_fields(original) != _constructor_fields(final):
                        raise ValueError("An inner matcher changed native constructor fields")
                    continue
                if original.get(key) != final.get(key) and (original["name"], key) not in owned:
                    raise ValueError(f"Inner matcher changed unowned field {original['name']}.{key}")
        for knob in parameters.knobs:
            values = [after[i][knob.field] for i in knob.target_indices]
            if any(not _finite(v) or not knob.lower <= v <= knob.upper for v in values):
                raise ValueError(f"Inner matcher returned out-of-bound values for {knob.name}")
            if any(v != values[0] for v in values):
                raise ValueError(f"Inner matcher broke declared mirror group {knob.name}")
        by_name = {knob.name: knob for knob in parameters.knobs}
        for constraint in parameters.sum_constraints:
            measured_sum = math.fsum(
                after[index][constraint.field]
                for name in constraint.variable_names
                for index in by_name[name].target_indices
            )
            if not math.isclose(measured_sum, constraint.target, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Inner matcher broke declared {constraint.field} sum constraint")
        return after, {"matched": success, "reason": result.get("reason"),
                       "solver_success": result.get("solver_success"),
                       "numerical_checkpoint": result.get("numerical_checkpoint")}


class LatticeExperiment:
    """An optimizer callback that archives exact final rows before returning.

    measure(rows) returns named scalar measurements. A topology edit is an
    explicit evaluate_rows hypothesis, never a mutation of compiled knobs. With
    inner_match, outer and inner fields cannot overlap. All callbacks must use
    JuTrack for accelerator simulation; other measurement topics can use this
    execution contract without MBA-specific state or objectives.
    """

    def __init__(self, definition: ExperimentDefinition,
                 parameterization: LatticeParameterization | None = None, *,
                 measure: Callable[[list[dict]], Mapping[str, Any]], directory: Path | str,
                 inner_match: MatchingOperation | None = None, workspace: Path | str | None = None):
        from .experiment_candidates import CandidateLedger

        self.definition = copy.deepcopy(definition)
        self.parameterization = copy.deepcopy(parameterization)
        self.measure = measure
        self.inner_match = copy.deepcopy(inner_match)
        if inner_match and parameterization:
            outer = {(n, k.field) for k in parameterization.knobs for n in k.target_names}
            inner = {(n, k.field) for k in inner_match.parameters.knobs for n in k.target_names}
            if outer & inner:
                raise ValueError(f"Outer and matching fields have overlapping ownership: {sorted(outer & inner)}")
        contract = {"definition": definition.as_dict(),
                    "parameters": parameterization.as_dict() if parameterization else None,
                    "baseline": list(parameterization.baseline) if parameterization else None,
                    "measurement": getattr(measure, "contract", {}),
                    "measurement_source": _operation_identity(measure),
                    "matching": ({"parameters": inner_match.parameters.as_dict(),
                                  "contract": dict(inner_match.contract),
                                  "source": _operation_identity(inner_match.solve)} if inner_match else None)}
        self.ledger = CandidateLedger(directory, contract=contract, objectives=definition.objectives,
                                      workspace=workspace)
        self._initial_contract_state = self._contract_state()
        self._measurement_callback = measure
        self._matching_callback = self.inner_match.solve if self.inner_match else None

    def _contract_state(self):
        from .experiment_candidates import _hash, _plain

        return _hash(_plain({"definition": self.definition.as_dict(),
                            "parameters": self.parameterization.as_dict() if self.parameterization else None,
                            "baseline": self.parameterization.baseline if self.parameterization else None,
                            "measurement": getattr(self.measure, "contract", {}),
                            "matching": ({"parameters": self.inner_match.parameters.as_dict(),
                                          "contract": self.inner_match.contract} if self.inner_match else None)}))

    def evaluate(self, position: Sequence[float]) -> dict[str, Any]:
        """PSO/MOPSO-compatible callback; objectives are minimized after direction conversion."""
        if self.parameterization is None:
            raise ValueError("evaluate(position) requires a compiled parameterization")
        rows = self.parameterization.apply(position)
        return self.evaluate_rows(rows, provenance={"proposed_position": list(map(float, position)),
                                                   "physical_position": self.parameterization.projected_position(position)})

    def evaluate_rows(self, rows: Sequence[Mapping[str, Any]], *, label: str = "candidate",
                      provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate an explicit hypothesis; every completed result is durable, not only incumbents."""
        if (self.measure is not self._measurement_callback
                or (self.inner_match.solve if self.inner_match else None) is not self._matching_callback
                or self._contract_state() != self._initial_contract_state):
            raise ValueError("Experiment contract changed; create a separately contracted branch")
        exact = copy.deepcopy(list(rows))
        measurements: Mapping[str, Any] = {}
        checks, match = {}, None
        try:
            if self.inner_match:
                exact, match = self.inner_match.apply(exact)
                checks["dependent_match"] = {"passed": match["matched"],
                                              "violation": 0.0 if match["matched"] else 1.0}
            supplied = copy.deepcopy(exact)
            measurements = self.measure(supplied)
            if supplied != exact:
                raise ValueError("Measurement callback changed the lattice; declare edits or matching ownership")
            if not isinstance(measurements, Mapping):
                raise TypeError("Experiment measurements must be a mapping of named scalar values")
            checks.update(self.definition.assess(measurements))
        except ResearchInfeasible as exc:
            checks["physical_attempt"] = {"passed": False, "violation": None}
            measurements = {"physical_rejection": exc.code, "reason": str(exc)}
            checks.update(self.definition.assess(measurements))
        except Exception as exc:
            # Retain the attempted coordinates but do not disguise software errors as physics.
            self.ledger.record(candidate={"elements": exact}, measurements={},
                               checks={"execution_completed": False},
                               provenance={"label": label, "error": f"{type(exc).__name__}: {exc}"})
            raise
        entry = self.ledger.record(candidate={"elements": exact}, measurements=measurements, checks=checks,
                                   provenance={**dict(provenance or {}), "label": label, "matching": match})
        objectives = {name: (float(measurements[name]) * (1 if direction == "min" else -1)
                             if _finite(measurements.get(name)) else None)
                      for name, direction in self.definition.objectives.items()}
        # The PSO kernel converts unavailable values to infinity internally.
        # Keep this public result JSON-safe for scientific checkpoints and chat.
        violations = {name: check["violation"] for name, check in checks.items()}
        return {"objectives": objectives, "constraint_violations": violations,
                "feasible": all(check["passed"] for check in checks.values()),
                "candidate_id": entry["candidate_id"], "measurements": dict(measurements)}

    def comparison_table(self, *, limit: int = 12) -> dict[str, Any]:
        """Small truthful table from the ledger, never a second selection algorithm."""
        if type(limit) is not int or limit < 1:
            raise ValueError("comparison table limit must be a positive integer")
        selection = self.ledger.select()
        selected = selection.get("selected")
        selected_id = selected["candidate_id"] if selected else None
        # Show the selected result and recent alternatives, not 40,000 trial rows.
        visible = ([selected] if selected else []) + [
            e for e in self.ledger.recent_entries(limit=limit)
            if e["candidate_id"] != selected_id][:limit - bool(selected)]
        return {"title": self.definition.name, "total_candidates": selection["counts"]["evaluated"], "rows": [
            {"Candidate": e["candidate_id"], "Hypothesis": e["provenance"].get("label", ""),
             "Selected": e["candidate_id"] == selected_id,
             "Experiment constraints passed": all(c["passed"] for c in e["checks"].values()),
             **{name: e["measurements"].get(name) for name in dict.fromkeys(
                 [*self.definition.objectives, *self.definition.constraints])}}
            for e in visible]}


class JuTrackCellEvaluator:
    """Fast native periodic-cell measurements, rebuilding only when necessary.

    Energy and periods are explicit. Engineering checks use the supplied limits;
    optics acceptance belongs to ExperimentDefinition. Boundary Dx is NOT a
    dispersion-bump measurement. Beta maximum is sampled at element boundaries.
    Selected candidates still require high-level cell and explicit-ring receipts.
    Each parallel worker must create its own evaluator/native Julia objects.
    """

    def __init__(self, *, energy_gev: float, periods: int, expected_length_m: float,
                 expected_angle_rad: float, limits: Mapping[str, Any],
                 minimum_magnet_length_m: float, minimum_drift_length_m: float):
        if not _finite(energy_gev) or energy_gev <= 0 or type(periods) is not int or periods <= 0:
            raise ValueError("Positive energy and integer periods are required")
        self.contract = {"engine": "JuTrack", "energy_gev": float(energy_gev), "periods": periods,
                         "expected_length_m": expected_length_m, "expected_angle_rad": expected_angle_rad,
                         "limits": copy.deepcopy(dict(limits)),
                         "minimum_magnet_length_m": minimum_magnet_length_m,
                         "minimum_drift_length_m": minimum_drift_length_m}
        self._editor = None
        self._constructor_key = None

    def __call__(self, rows: list[dict]) -> dict[str, Any]:
        from . import accelerator_api as api

        args = self.contract
        audit = api.audit_lattice_constraints(rows, **{k: v for k, v in args.items() if k not in {"engine", "periods"}})
        metrics = {"geometry_valid": audit["passed"], "stable": False, "radiation_valid": False,
                   "cell_length_m": math.fsum(float(r["length_m"]) for r in rows),
                   "cell_angle_rad": math.fsum(float(r.get("angle_rad") or 0) for r in rows),
                   "emittance_pm": None, "maximum_beta_m": None, "Dx_m": None, "Dpx": None,
                   "trace_x": None, "trace_y": None, "I2": None, "I4": None, "I5": None,
                   "full_ring_validation_required": True}
        if not audit["passed"]:
            return metrics
        key = json.dumps([(r["name"], r["kind"], _constructor_fields(r))
                          for r in rows], sort_keys=True)
        if self._editor is None or key != self._constructor_key:
            candidate = api.periodic_cell(title="Experiment trial", energy_gev=args["energy_gev"],
                                          elements=rows, intended_repetitions=args["periods"],
                                          source="brilliance.experiments.JuTrackCellEvaluator")
            self._editor = api.build_jutrack_editor(candidate)
            self._constructor_key = key
        evaluated = self._editor.evaluate_candidate(rows, energy_gev=args["energy_gev"],
                                                    periods=args["periods"], include_profile=True)
        linear, radiation = evaluated["linear_optics"], evaluated.get("radiation") or {}
        metrics.update({k: v for k, v in linear.items() if k != "twiss_profile"})
        metrics.update({k: radiation.get(k) for k in ("emittance_pm", "I2", "I3", "I4", "I5", "Jx", "Jy", "Jz", "energy_spread")})
        metrics["radiation_valid"] = radiation.get("physical_validity", {}).get("passed") is True
        unavailable = [k for k, v in metrics.items() if isinstance(v, Real)
                       and not isinstance(v, bool) and not math.isfinite(v)]
        for key in unavailable:
            metrics[key] = None
        if unavailable:
            metrics["nonfinite_native_diagnostics"] = unavailable
            metrics["stable"] = False
            metrics["radiation_valid"] = False
        return metrics
