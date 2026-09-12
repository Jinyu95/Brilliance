"""Experimental radiation-aware continuation from an accepted periodic MBA.

This is not the legacy outer-DBA replacement method. Connectivity, lengths and
angles remain unchanged; all global mirror-paired KQUAD strengths may rematch.
The numerical kernel is testable without Julia. Only the adapter below supplies
physics measurements, through the existing JuTrack API. No lattice is published
or declared a finished H-MBA here.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SCHEMA = "radiation_aware_hybrid_continuation/v1"
SLOT_NAMES = ("SF_L", "SD_L", "SF_R", "SD_R")


@dataclass(frozen=True)
class HybridContinuationContract:
    """Physical requirements; all values are explicit, never facility defaults.

    Dispersion and H have units m, emittance pm rad, beta m. Slopes, trace
    margins and relative energy spread are dimensionless. Core H/dispersion and
    slot slope/dispersion are continued from their measured parent values. All
    other limits, especially the ABSOLUTE emittance cap, apply at every step.
    """
    slot_dispersion_goals_m: Mapping[str, float]
    core_dispersion_goal_m: float
    core_h_goal_m: float
    max_emittance_pm: float
    max_energy_spread: float
    max_beta_m: float
    max_slot_slope: float
    id_dispersion_tolerance_m: float
    id_slope_tolerance: float
    stability_trace_margin: float
    dispersion_scale_m: float
    h_scale_m: float

    def __post_init__(self):
        values = asdict(self)
        slots = values.pop("slot_dispersion_goals_m")
        if set(slots) != set(SLOT_NAMES):
            raise ValueError(f"Declare dispersion goals for exactly {SLOT_NAMES}")
        for key, value in {**values, **slots}.items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.stability_trace_margin >= 2:
            raise ValueError("stability_trace_margin must be smaller than 2")
        object.__setattr__(self, "slot_dispersion_goals_m", deepcopy(dict(slots)))


@dataclass(frozen=True)
class HybridContinuationSettings:
    """Disclosed numerical settings, not physical acceptance limits."""
    initial_step: float = 0.05
    minimum_step: float = 1e-4
    maximum_step: float = 0.1
    trust_fraction: float = 0.1
    max_attempts: int = 40
    max_iterations: int = 100
    max_evaluations_per_attempt: int = 2000
    finite_difference_step: float = 1e-5
    optimizer_ftol: float = 1e-9
    feasibility_tolerance: float = 1e-8
    replay_rtol: float = 1e-7
    replay_atol: float = 1e-12

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("max_attempts", "max_iterations", "max_evaluations_per_attempt"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if not self.minimum_step <= self.initial_step <= self.maximum_step <= 1:
            raise ValueError("Require minimum_step <= initial_step <= maximum_step <= 1")
        if self.trust_fraction > 1 or self.finite_difference_step >= self.trust_fraction:
            raise ValueError("Finite-difference step must be smaller than trust_fraction <= 1")


def _checked_measurement(supplied: Mapping[str, Any]) -> dict:
    """Fail loudly on schema/API errors, not silently as physical instability."""
    value = deepcopy(dict(supplied))
    if type(value.get("valid")) is not bool:
        raise ValueError("Measurement needs an explicit boolean valid flag")
    for key in ("trace_x", "trace_y"):
        if not math.isfinite(float(value[key])):
            raise ValueError(f"Nonfinite {key}")
    if not value["valid"]:
        if not isinstance(value.get("failure"), str) or not value["failure"]:
            raise ValueError("Invalid physical measurements require a failure description")
        return value
    for key in ("emittance_pm", "energy_spread", "beta_max_m", "core_dispersion_m", "core_h_m"):
        number = float(value[key])
        if not math.isfinite(number) or number < 0 or (key in {"emittance_pm", "beta_max_m"} and number == 0):
            raise ValueError(f"Invalid {key}")
        value[key] = number
    for key in ("id_dx_m", "id_dpx"):
        a = np.asarray(value[key], dtype=float)
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise ValueError(f"{key} needs the two physical cell-boundary values")
        value[key] = a.tolist()
    if set(value["slots"]) != set(SLOT_NAMES):
        raise ValueError("Measurement is missing a reserved sextupole slot")
    for name in SLOT_NAMES:
        for key in ("min_dx_m", "max_abs_dpx"):
            number = float(value["slots"][name][key])
            if not math.isfinite(number) or (key == "max_abs_dpx" and number < 0):
                raise ValueError(f"Invalid {name}.{key}")
            value["slots"][name][key] = number
    return value


def hybrid_constraint_slacks(measured, parent, contract, fraction):
    """Dimensionless inequalities >= 0; identical constraints in solve and replay.

    No contrast ratio is minimized. Four finite chromatic slots and the actual
    core envelope are constrained separately. The objective is emittance.
    """
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Continuation fraction must be in [0,1]")
    c = contract
    slacks = {
        "physical_validity": 1.0 if measured["valid"] else -1.0,
        "stable_x": (2 - c.stability_trace_margin - abs(measured["trace_x"])) / 2,
        "stable_y": (2 - c.stability_trace_margin - abs(measured["trace_y"])) / 2,
    }
    names = ["emittance", "energy_spread", "beta", "core_dispersion", "core_H"]
    names += [f"ID_{side}_{field}_{sign}" for side in ("L", "R")
              for field in ("Dx", "Dpx") for sign in ("plus", "minus")]
    names += [f"{name}_{field}" for name in SLOT_NAMES for field in ("dispersion", "slope")]
    if not measured["valid"]:
        return {**slacks, **dict.fromkeys(names, -1.0)}
    interpolate = lambda initial, final: (1 - fraction) * initial + fraction * final
    core_d = interpolate(parent["core_dispersion_m"], c.core_dispersion_goal_m)
    core_h = interpolate(parent["core_h_m"], c.core_h_goal_m)
    slacks.update(emittance=1 - measured["emittance_pm"] / c.max_emittance_pm,
                  energy_spread=1 - measured["energy_spread"] / c.max_energy_spread,
                  beta=1 - measured["beta_max_m"] / c.max_beta_m,
                  core_dispersion=(core_d - measured["core_dispersion_m"]) / c.dispersion_scale_m,
                  core_H=(core_h - measured["core_h_m"]) / c.h_scale_m)
    for i, side in enumerate(("L", "R")):
        for field, key, tolerance in (("Dx", "id_dx_m", c.id_dispersion_tolerance_m),
                                       ("Dpx", "id_dpx", c.id_slope_tolerance)):
            v = measured[key][i] / tolerance
            slacks[f"ID_{side}_{field}_plus"] = 1 - v
            slacks[f"ID_{side}_{field}_minus"] = 1 + v
    for name in SLOT_NAMES:
        p, m = parent["slots"][name], measured["slots"][name]
        target = interpolate(p["min_dx_m"], c.slot_dispersion_goals_m[name])
        slope_cap = interpolate(max(p["max_abs_dpx"], c.max_slot_slope), c.max_slot_slope)
        slacks[f"{name}_dispersion"] = (m["min_dx_m"] - target) / c.dispersion_scale_m
        slacks[f"{name}_slope"] = (slope_cap - m["max_abs_dpx"]) / c.max_slot_slope
    return slacks


def _failed(slacks, tolerance):
    return {key: value for key, value in slacks.items() if value < -tolerance}


def _reproduces(first, second, settings):
    if first["valid"] != second["valid"]:
        return False
    if not first["valid"]:
        return False
    scalar = ("emittance_pm", "energy_spread", "trace_x", "trace_y", "beta_max_m",
              "core_dispersion_m", "core_h_m")
    a, b = [first[k] for k in scalar], [second[k] for k in scalar]
    for key in ("id_dx_m", "id_dpx"):
        a.extend(first[key]); b.extend(second[key])
    for name in SLOT_NAMES:
        for key in ("min_dx_m", "max_abs_dpx"):
            a.append(first["slots"][name][key]); b.append(second["slots"][name][key])
    return bool(np.allclose(a, b, rtol=settings.replay_rtol, atol=settings.replay_atol))


def _check_target_direction(parent, contract, settings):
    """Reject actual direction changes, not roundoff between mirror partners."""
    violations = []
    def check(name, requested, measured, increasing):
        tolerance = settings.replay_atol + settings.replay_rtol * max(abs(requested), abs(measured))
        wrong = measured - requested if increasing else requested - measured
        if wrong > tolerance:
            direction = "at least" if increasing else "no greater than"
            violations.append(
                f"{name}: requested={requested:.17g} m must be {direction} "
                f"parent={measured:.17g} m (direction tolerance={tolerance:.3g} m)"
            )
    for name in SLOT_NAMES:
        check(name, contract.slot_dispersion_goals_m[name], parent["slots"][name]["min_dx_m"], True)
    check("core_dispersion", contract.core_dispersion_goal_m, parent["core_dispersion_m"], False)
    if violations:
        raise ValueError("Hybrid target direction is invalid: " + "; ".join(violations))


def hybrid_continuation_summary(state):
    """Canonical compact facts; campaign termination is not candidate acceptance."""
    records = state.get("records", [])
    measurement = state.get("measurement", {})
    return {
        "status": state["status"],
        "fraction": state["fraction"],
        "requested_fraction": state.get("requested_fraction"),
        "has_accepted_progress": bool(state.get("accepted_steps")),
        "accepted_step_count": len(state.get("accepted_steps", [])),
        "attempt_count": len(records),
        "evaluations": sum(r.get("evaluations", 0) for r in records),
        "optimization_started": bool(records),
        "last_solver_termination": records[-1].get("solver_termination") if records else None,
        "emittance_pm": measurement.get("emittance_pm"),
        "requires_independent_acceptance": True,
        "request_satisfied": False,
    }


class _AttemptBudget(Exception):
    pass


class _Cancelled(Exception):
    pass


def radiation_aware_hybrid_continuation(
    measure: Callable, verify: Callable, *, initial_position: Sequence[float],
    bounds: Sequence[Sequence[float]], contract: HybridContinuationContract,
    measurement_identity: str, settings: HybridContinuationSettings | None = None,
    end_fraction: float = 1.0, progress: Callable | None = None,
    checkpoint: Callable | None = None, cancelled: Callable | None = None,
) -> dict:
    """Bounded fixed-topology SLSQP continuation with radiation and replay gates.

    measure(position) and verify(position) implement the documented measurement
    schema. verify must bypass trial caches; the JuTrack adapter uses a new editor
    and sampler. Both receive physical strength coordinates, not normalized ones.
    Exceptions from physics APIs propagate. Solver termination is not acceptance.

    A completed unsuccessful solve restarts from the last ACCEPTED position and
    halves the target step. Resource cutoffs stop with their own status instead. Checkpoints preserve that position and the immutable parent cap.
    No automatic resume or geometry release is hidden in this first implementation.
    A later campaign must explicitly retain the ORIGINAL radiation cap.
    """
    from scipy.optimize import minimize

    settings = settings or HybridContinuationSettings()
    contract = HybridContinuationContract(**asdict(contract))
    if not isinstance(measurement_identity, str) or not measurement_identity.strip():
        raise ValueError("An explicit source/evaluator identity is required")
    if not math.isfinite(end_fraction) or not 0 <= end_fraction <= 1:
        raise ValueError("end_fraction must be in [0,1]")
    x0, box = np.asarray(initial_position, dtype=float), np.asarray(bounds, dtype=float)
    if (x0.ndim != 1 or not len(x0) or box.shape != (len(x0), 2)
            or not np.all(np.isfinite(x0)) or not np.all(np.isfinite(box))
            or np.any(box[:, 0] >= box[:, 1]) or np.any(x0 < box[:, 0]) or np.any(x0 > box[:, 1])):
        raise ValueError("Supply an in-bound finite position and one nonzero-width bound per knob")
    lo, width = box[:, 0], box[:, 1] - box[:, 0]
    original = _checked_measurement(measure(x0.copy()))
    parent = _checked_measurement(verify(x0.copy()))
    if not _reproduces(original, parent, settings):
        raise ValueError("Zero-deformation replay failed; do not start hybrid shaping")
    failed = _failed(hybrid_constraint_slacks(parent, parent, contract, 0.0), settings.feasibility_tolerance)
    if failed:
        raise ValueError(f"Parent violates declared machine limits: {failed}")
    # Identity-only evaluation does not impose a future deformation direction.
    if end_fraction > 0:
        _check_target_direction(parent, contract, settings)
    identity = {"schema": SCHEMA, "measurement_identity": measurement_identity,
                "initial_position": x0.tolist(), "bounds": box.tolist(),
                "contract": asdict(contract), "settings": asdict(settings)}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    state = {"schema": SCHEMA, "contract_id": identity_hash, "contract": identity,
             "status": "running", "fraction": 0.0, "position": x0.tolist(),
             "requested_fraction": end_fraction,
             "parent_measurement": parent, "measurement": parent, "records": [],
             "accepted_steps": [], "request_satisfied": False,
             "limitation": "A matched dispersion/radiation stage is not nonlinear H-MBA validation."}
    def save():
        state["has_accepted_progress"] = bool(state["accepted_steps"])
        state["accepted_step_count"] = len(state["accepted_steps"])
        state["summary"] = hybrid_continuation_summary(state)
        if checkpoint:
            checkpoint(deepcopy(state))
    save()
    step = settings.initial_step
    for _ in range(settings.max_attempts):
        if state["fraction"] >= end_fraction:
            break
        if cancelled and cancelled():
            state["status"] = "cancelled"; break
        target = min(end_fraction, state["fraction"] + step)
        accepted = np.asarray(state["position"], dtype=float)
        z0 = (accepted - lo) / width
        lower = np.maximum(0.0, z0 - settings.trust_fraction)
        upper = np.minimum(1.0, z0 + settings.trust_fraction)
        cache, feasible = {}, []
        calls = 0
        objective_calls = constraint_calls = major_iterations = 0
        def evaluate(z):
            nonlocal calls
            if cancelled and cancelled():
                raise _Cancelled()
            z = np.asarray(z, dtype=float)
            key = z.tobytes()
            if key not in cache:
                if calls >= settings.max_evaluations_per_attempt:
                    raise _AttemptBudget()
                calls += 1
                p = lo + np.clip(z, 0.0, 1.0) * width
                m = _checked_measurement(measure(p.copy()))
                s = hybrid_constraint_slacks(m, parent, contract, target)
                cache[key] = (p, m, s)
                if not _failed(s, settings.feasibility_tolerance):
                    feasible.append((m["emittance_pm"], p.copy(), deepcopy(m)))
            return cache[key]
        def objective(z):
            nonlocal objective_calls
            objective_calls += 1
            _, m, s = evaluate(z)
            if m["valid"]:
                return m["emittance_pm"] / parent["emittance_pm"]
            # Finite search penalty only; invalid points can NEVER be accepted.
            return contract.max_emittance_pm / parent["emittance_pm"] + 10 + sum(min(0., v)**2 for v in s.values())
        def constraints(z):
            nonlocal constraint_calls
            constraint_calls += 1
            return np.asarray(list(evaluate(z)[2].values()))
        def iteration_completed(_position):
            nonlocal major_iterations
            major_iterations += 1
        termination = "not_started"
        resource_cutoff = None
        solution = None
        record = {"target_fraction": target, "from_fraction": state["fraction"],
                  "start_position": state["position"][:], "step": target - state["fraction"],
                  "variable_count": len(x0), "evaluation_budget": settings.max_evaluations_per_attempt,
                  "iteration_budget": settings.max_iterations,
                  "forward_difference_stencil_size": len(x0) + 1}
        try:
            solution = minimize(objective, z0, method="SLSQP", bounds=list(zip(lower, upper)),
                constraints={"type": "ineq", "fun": constraints}, callback=iteration_completed,
                options={"maxiter": settings.max_iterations, "ftol": settings.optimizer_ftol,
                         "eps": settings.finite_difference_step})
            termination = str(solution.message)
            record["solver_success"] = bool(solution.success)
            record["solver_status"] = getattr(solution, "status", None)
            record["solver_iterations"] = getattr(solution, "nit", major_iterations)
            record["solver_objective_evaluations"] = getattr(solution, "nfev", None)
            record["solver_gradient_evaluations"] = getattr(solution, "njev", None)
            if getattr(solution, "status", None) == 9:
                resource_cutoff = "iteration_budget_exhausted"
            evaluate(solution.x)
        except _AttemptBudget:
            termination = "evaluation_budget_exhausted"
            resource_cutoff = termination
        except _Cancelled:
            termination = "cancelled"
            state["status"] = "cancelled"
        record.update(evaluations=calls, solver_termination=termination, accepted=False,
                      resource_cutoff=resource_cutoff, major_iterations_completed=major_iterations,
                      objective_calls=objective_calls, constraint_calls=constraint_calls)
        record.setdefault("solver_success", False)
        if cache:
            best_position, best_measurement, best_slacks = min(
                cache.values(), key=lambda v: sum(min(0., s)**2 for s in v[2].values()))
            record["best_trial_failed_constraints"] = _failed(best_slacks, settings.feasibility_tolerance)
            record["best_trial"] = {"position": best_position.tolist(),
                                    "measurement": deepcopy(best_measurement),
                                    "constraint_slacks": dict(best_slacks)}
        if feasible and state["status"] != "cancelled":
            _, candidate, measured = min(feasible, key=lambda v: v[0])
            replay = _checked_measurement(verify(candidate.copy()))
            failures = _failed(hybrid_constraint_slacks(replay, parent, contract, target), settings.feasibility_tolerance)
            if not _reproduces(measured, replay, settings):
                record["replay_failure"] = "Selected position did not reproduce independently"
            elif failures:
                record["replay_failed_constraints"] = failures
            else:
                state.update(position=candidate.tolist(), fraction=target, measurement=replay)
                record.update(accepted=True, emittance_pm=replay["emittance_pm"],
                              accepted_position=candidate.tolist(),
                              accepted_measurement=deepcopy(replay),
                              accepted_constraint_slacks=hybrid_constraint_slacks(replay, parent, contract, target))
                state["accepted_steps"].append({"fraction": target, "position": candidate.tolist(), "measurement": replay})
                # A resource-limited feasible point is progress, not convergence.
                # Do not enlarge the next target on an unfinished local solve.
                if resource_cutoff is None and bool(getattr(solution, "success", False)):
                    step = min(settings.maximum_step, step * 1.5)
        state["records"].append(record)
        # A numerical resource cutoff is not evidence against the target. Stop
        # and retain the accepted branch; never halve into a false topology stall.
        if state["status"] != "cancelled" and resource_cutoff and not record["accepted"]:
            state["status"] = resource_cutoff
            record["next_action"] = "review_numerical_budget_without_changing_physical_contract"
        elif state["status"] != "cancelled" and not record["accepted"]:
            step = (target - state["fraction"]) / 2
            if step < settings.minimum_step:
                state["status"] = "continuation_stalled"
        save()  # Raw evidence is durable before caller-owned display callbacks.
        if progress:
            progress(deepcopy(record))
        if state["status"] != "running":
            break
    if state["status"] == "running":
        state["status"] = ("identity_passed" if end_fraction == 0 else
                           "continuation_targets_met" if state["fraction"] >= end_fraction else
                           "attempt_budget_exhausted")
    # Restore mutable evaluator state to the returned accepted candidate.
    if state["status"] != "cancelled":
        restored = _checked_measurement(measure(np.asarray(state["position"])))
        if not _reproduces(state["measurement"], restored, settings):
            raise ValueError("Final accepted-position replay failed")
    save()
    return state


def _slot_ranges(rows, roles, slots, minimum_slot_length_m):
    """Require disjoint, finite, globally paired field-free installation slots."""
    if set(slots) != set(SLOT_NAMES):
        raise ValueError(f"Reserve exactly {SLOT_NAMES}")
    if not math.isfinite(minimum_slot_length_m) or minimum_slot_length_m <= 0:
        raise ValueError("minimum_slot_length_m must be explicit and positive")
    lookup = {r["name"]: i for i, r in enumerate(rows)}
    starts = np.cumsum([0.0] + [float(r["length_m"]) for r in rows])
    occupied, ranges, indices = set(), {}, {}
    for name in SLOT_NAMES:
        names = slots[name]
        if isinstance(names, str) or not names:
            raise ValueError(f"{name} needs a nonempty ordered list of physical element names")
        idx = [lookup[n] for n in names]
        if idx != list(range(idx[0], idx[-1] + 1)) or set(idx) & occupied:
            raise ValueError("Slots must be contiguous, nonoverlapping physical intervals")
        region = roles["regions"]["left_bump" if name.endswith("_L") else "right_bump"]
        if not set(idx) <= set(region["element_indices"]):
            raise ValueError(f"{name} must be between the corresponding outer bend pair")
        if any(rows[i]["kind"] not in {"DRIFT", "MARKER"} for i in idx):
            raise ValueError("Reserve actual drifts/markers, not a peak inside a magnet")
        length = float(starts[idx[-1] + 1] - starts[idx[0]])
        if length < minimum_slot_length_m - 1e-12:
            raise ValueError(f"{name} is too short for the declared installation requirement")
        occupied.update(idx)
        indices[name] = idx
        ranges[name] = (float(starts[idx[0]]), float(starts[idx[-1] + 1]))
    for family in ("SF", "SD"):
        if indices[family + "_R"] != [len(rows) - 1 - i for i in reversed(indices[family + "_L"])]:
            raise ValueError("Corresponding SF/SD slots must be global mirror partners")
    return ranges


def _profile_measurement(linear, radiation, ranges, core_region):
    """Extract measured slot endpoints and sampled core envelopes, not line Twiss."""
    profile = linear["twiss_profile"]
    fields = ("s_m", "Dx", "Dpx", "beta_x", "beta_y", "alpha_x")
    values = np.asarray([[p[key] for key in fields] for p in profile], dtype=float)
    if (values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values))
            or np.any(values[:, 3:5] <= 0) or np.any(np.diff(values[:, 0]) < -1e-10)):
        raise ValueError("Require a finite ordered profile with positive beta in both planes")
    def at(s):
        point = min(profile, key=lambda p: abs(p["s_m"] - s))
        if abs(point["s_m"] - s) > 1e-8:
            raise ValueError("Missing physical slot-boundary sample")
        return point
    slots = {}
    for name, (start, end) in ranges.items():
        first, last = at(start), at(end)
        points = [p for p in profile if start - 1e-10 <= p["s_m"] <= end + 1e-10]
        # In a field-free slot Dx is linear and Dpx constant: endpoints suffice.
        for p in points:
            expected = first["Dx"] + (p["s_m"] - start) * first["Dpx"]
            if abs(p["Dx"] - expected) > 1e-8 or abs(p["Dpx"] - first["Dpx"]) > 1e-8:
                raise ValueError("Reserved slot does not reproduce field-free dispersion transport")
        slots[name] = {"min_dx_m": min(first["Dx"], last["Dx"]),
                       "max_abs_dpx": max(abs(first["Dpx"]), abs(last["Dpx"])),
                       "length_m": end - start}
    core = [p for p in profile if core_region["s_start_m"] - 1e-10 <= p["s_m"] <= core_region["s_end_m"] + 1e-10]
    if not core:
        raise ValueError("Missing central-core optics")
    hx = lambda p: (p["Dx"]**2 + (p["alpha_x"]*p["Dx"] + p["beta_x"]*p["Dpx"])**2) / p["beta_x"]
    result = {"valid": True, "trace_x": linear["trace_x"], "trace_y": linear["trace_y"],
              "emittance_pm": radiation["emittance_pm"], "energy_spread": radiation["energy_spread"],
              "radiation": deepcopy(radiation),
              "beta_max_m": max(max(p["beta_x"], p["beta_y"]) for p in profile),
              "core_dispersion_m": max(abs(p["Dx"]) for p in core),
              "core_h_m": max(hx(p) for p in core),
              "id_dx_m": [profile[0]["Dx"], profile[-1]["Dx"]],
              "id_dpx": [profile[0]["Dpx"], profile[-1]["Dpx"]], "slots": slots}
    return _checked_measurement(result)


def match_periodic_hybrid(
    parent: Any, *, periods: int, slots: Mapping[str, Sequence[str]],
    minimum_slot_length_m: float, limits: Mapping[str, float],
    minimum_magnet_length_m: float, minimum_drift_length_m: float,
    contract: Mapping[str, Any], max_sample_step_m: float,
    settings: Mapping[str, Any] | None = None, end_fraction: float = 1.0,
    progress: Callable | None = None, checkpoint: Callable | None = None,
    cancelled: Callable | None = None,
) -> dict:
    """Experimental JuTrack complete-periodic-cell hybrid shaping, no topology edits.

    Load an accepted parent, reserve SF_L/SD_L/SF_R/SD_R by actual drift names,
    and supply all physical caps. Every global mirror-paired KQUAD (including
    core quadrupoles) is controlled by ONE constrained solver. All lengths,
    bend angles, dipole gradients and ID length stay exactly unchanged.

    Uses api.one_period_elements, infer_mirror_symmetric_parameterization,
    build_jutrack_profile_sampler and jutrack_periodic_radiation_metrics. No new
    JuTrack function is assumed. A fresh sampler independently replays each
    accepted position. end_fraction=0 runs the zero-deformation gate only.

    Output candidate_lattice is a diagnostic branch, never automatically selected.
    Even continuation_targets_met requires the separate full_hybrid, full-ring,
    sampling-convergence, sextupole-strength and nonlinear validation gates.
    """
    from brilliance import accelerator_api as api
    from brilliance.lattice_fingerprint import cell_fingerprint

    from .hybrid_regions import hybrid_mba_regions

    if type(periods) is not int or periods <= 0:
        raise ValueError("periods must be a positive integer; do not infer by rounding")
    if not math.isfinite(max_sample_step_m) or max_sample_step_m <= 0:
        raise ValueError("Declare a positive max_sample_step_m")
    for name, value in (("minimum_magnet_length_m", minimum_magnet_length_m),
                        ("minimum_drift_length_m", minimum_drift_length_m)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    rows = deepcopy(api.one_period_elements(parent))
    inventory = hybrid_mba_regions(rows)
    if not inventory["mirror_passed"]:
        raise ValueError("Parent must already have complete-cell mirror symmetry")
    roles = inventory["bend_roles"]
    length = sum(r["length_m"] for r in rows)
    angle = sum(r.get("angle_rad") or 0 for r in rows)
    if not math.isclose(angle * periods, 2 * math.pi, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Declared periods do not close the actual cell bend angle")
    ranges = _slot_ranges(rows, roles, slots, minimum_slot_length_m)
    audit_options = dict(energy_gev=parent.energy_gev, expected_length_m=length,
                         expected_angle_rad=angle, expected_bends=roles["bend_count"],
                         require_mirror_symmetry=True, limits=limits,
                         minimum_magnet_length_m=minimum_magnet_length_m,
                         minimum_drift_length_m=minimum_drift_length_m)
    audit = api.audit_lattice_constraints(rows, **audit_options)
    if not audit["passed"]:
        raise ValueError(f"Parent construction/engineering audit failed: {audit}")
    kmax = float(limits["max_abs_k1_m2"])
    if not math.isfinite(kmax) or kmax <= 0:
        raise ValueError("Declare a positive quadrupole limit")
    knobs = api.infer_mirror_symmetric_parameterization(
        rows, fields=("k1_m2",), k1_bounds=(-kmax, kmax), role="periodic_hybrid_solver")
    def candidate_for(proposed):
        return api.periodic_cell(title="Radiation-aware hybrid continuation (experimental)",
            energy_gev=parent.energy_gev, elements=proposed, source=SCHEMA,
            intended_repetitions=periods, parent_lattice_id=parent.lattice_id)
    source_cell = candidate_for(rows)
    sampler = api.build_jutrack_profile_sampler(source_cell, max_step_m=max_sample_step_m)
    def measure(position, *, independent=False):
        proposed = knobs.apply(position)
        checked = api.audit_lattice_constraints(proposed, **audit_options)
        if not checked["passed"]:
            raise ValueError(f"Strength-only parameterization violated the machine audit: {checked}")
        active = (api.build_jutrack_profile_sampler(candidate_for(proposed), max_step_m=max_sample_step_m)
                  if independent else sampler)
        linear = active.linear_metrics(proposed)
        if linear["sampling"].get("map_equivalent") is not True:
            raise ValueError("Diagnostic sampler failed physical map-equivalence verification")
        if not linear["stable"]:
            return {"valid": False, "trace_x": linear["trace_x"], "trace_y": linear["trace_y"],
                    "failure": linear["failure"]}
        radiation = api.jutrack_periodic_radiation_metrics(
            active.jt, active.physical_editor.lattice, energy_gev=parent.energy_gev, periods=periods)
        if not radiation["physical_validity"]["passed"]:
            return {"valid": False, "trace_x": linear["trace_x"], "trace_y": linear["trace_y"],
                    "failure": "Radiation physical-validity check failed"}
        return _profile_measurement(linear, radiation, ranges, roles["regions"]["core"])
    identity = dict(source_lattice_id=parent.lattice_id, cell_fingerprint=cell_fingerprint(rows),
                    periods=periods, energy_gev=parent.energy_gev, slots=dict(slots),
                    minimum_slot_length_m=minimum_slot_length_m, sampling_step_m=max_sample_step_m,
                    limits=dict(limits), minimum_magnet_length_m=minimum_magnet_length_m,
                    minimum_drift_length_m=minimum_drift_length_m)
    declared = HybridContinuationContract(**dict(contract))
    controls = HybridContinuationSettings(**dict(settings or {}))
    from brilliance.numerical_evidence import NumericalEvidenceArchive
    archive = NumericalEvidenceArchive.for_environment(
        "periodic_hybrid", {"source": identity, "contract": asdict(declared),
                            "settings": asdict(controls), "end_fraction": end_fraction},
        parent_lattice=parent.as_dict())

    def archive_checkpoint(state):
        # This happens before returning to generated research/reporting code.
        # Preserve all 18+ coordinates, original caps and the exact cell metadata.
        if archive is not None:
            archived_cell = candidate_for(knobs.apply(state["position"])).as_dict()
            archive.snapshot(state, archived_cell, summary=hybrid_continuation_summary(state))
        if checkpoint is not None:
            checkpoint(deepcopy(state))

    def is_cancelled():
        return bool((archive is not None and archive.run is not None
                     and archive.run.cancellation_requested()) or (cancelled and cancelled()))

    try:
        result = radiation_aware_hybrid_continuation(
            measure, lambda position: measure(position, independent=True),
            initial_position=knobs.initial_position, bounds=knobs.bounds,
            contract=declared, settings=controls,
            measurement_identity=json.dumps(identity, sort_keys=True, allow_nan=False),
            end_fraction=end_fraction, progress=progress, checkpoint=archive_checkpoint,
            cancelled=is_cancelled)
        selected_rows = knobs.apply(result["position"])
        selected_cell = candidate_for(selected_rows)
        result["candidate_lattice"] = selected_cell.as_dict()
        result["source"] = identity
        result["parameterization"] = knobs.as_dict()
        result["installation_slots"] = {k: {"s_start_m": v[0], "s_end_m": v[1]} for k, v in ranges.items()}
        if archive is not None:
            # An explicit IR repetition, NOT a new JuTrack full-ring validation.
            # It prevents reporting wrappers from mislabelling one cell as a ring.
            ring = api.cell_ring(title="Archived hybrid pilot ring (validation required)",
                energy_gev=parent.energy_gev, cell=selected_rows, periods=periods,
                source=SCHEMA, parent_lattice_id=selected_cell.lattice_id)
            result["numerical_evidence_directory"] = str(archive.directory)
            result["full_ring_validation_required"] = True
            archive.complete(result, ring_lattice=ring.as_dict())
        return result
    except Exception as exc:
        if archive is not None:
            archive.failed(exc)
        raise
