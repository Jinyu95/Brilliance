"""Reusable, worker-local H-MBA measurements and bounded local responses.

No lattice construction, numerical seed or fixed optimization sequence lives
here. The scientist/agent chooses the knobs, matcher, goals and optimizer.
"""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .hybrid_regions import hybrid_mba_regions
from .hybrid_shape import _rows, _thresholds, hybrid_mba_bend_roles, hybrid_mba_shape_diagnostics


def hybrid_mba_quadrupole_knobs(elements: Sequence[Any], *, max_abs_k1_m2: float,
                              regions: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Return independently owned K1 knobs paired by full-cell physical mirror.

    Output is directly accepted by compile_lattice_parameterization. Pairing
    does NOT identify similarly named unit families or impose unit symmetry.
    The caller selects disjoint subsets for shaping and achromat matching.
    """
    rows = _rows(elements)
    inventory = hybrid_mba_regions(rows) if regions is not None else None
    if inventory is not None and (isinstance(regions, str) or set(regions) - set(inventory["region_names"])):
        raise ValueError("regions must list physical region names from hybrid_mba_regions")
    bound = float(max_abs_k1_m2)
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("max_abs_k1_m2 must be finite and positive")
    result = []
    for index, left in enumerate(rows[: (len(rows) + 1) // 2]):
        right = rows[len(rows) - 1 - index]
        if left["kind"] != right["kind"] or any(
            not math.isclose(float(left.get(field) or 0), float(right.get(field) or 0),
                             rel_tol=1e-10, abs_tol=1e-12)
            for field in ("length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4")
        ):
            raise ValueError(f"Source full-cell mirror mismatch: {left['name']} / {right['name']}")
        if left["kind"] != "KQUAD":
            continue
        if inventory is not None and inventory["elements"][index]["region"] not in regions:
            continue
        value = float(left.get("k1_m2") or 0)
        if abs(value) > bound:
            raise ValueError(f"{left['name']} is outside the declared K1 envelope")
        result.append({"name": f"K1:{left['name']}", "field": "k1_m2",
                       "element_names": list(dict.fromkeys([left["name"], right["name"]])),
                       "bounds": [-bound, bound], "initial": value})
    return result


class HybridMBAShapeEvaluator:
    """Reuse one JuTrack cell and sampler inside each worker process.

    ``linear_metrics`` serves an independently configured inner matcher.
    ``measure`` returns stable/feasible/shape_passed separately; an unstable
    proposal always retains its failure and traces, never an empty metrics dict.
    Create this object in a worker factory, never pickle a live Julia object.
    """

    def __init__(self, candidate: Any, *, thresholds: Mapping[str, Any],
                 expected_cell_length_m: float, expected_cell_angle_rad: float,
                 limits: Mapping[str, float] | None = None, max_step_m: float = .05,
                 minimum_magnet_length_m: float = 0.0, minimum_drift_length_m: float = 0.0):
        from brilliance import accelerator_api as api

        self.api = api
        self.candidate = api.load_lattice(candidate) if isinstance(candidate, (str, Path)) else candidate
        self.rows = api.one_period_elements(self.candidate)
        self.roles = hybrid_mba_bend_roles(self.rows)
        self.thresholds = _thresholds(thresholds)
        if not math.isfinite(float(max_step_m)) or max_step_m <= 0 or max_step_m > self.thresholds["max_sample_step_m"]:
            raise ValueError("max_step_m must be positive and no coarser than the declared shape sampling threshold")
        self.expected_cell_length_m = float(expected_cell_length_m)
        self.expected_cell_angle_rad = float(expected_cell_angle_rad)
        self.minimum_magnet_length_m = float(minimum_magnet_length_m)
        self.minimum_drift_length_m = float(minimum_drift_length_m)
        self.limits = dict(limits) if limits is not None else api.magnet_strength_limits(self.candidate.energy_gev)
        audit = self._audit(self.rows)
        if not audit["passed"]:
            raise ValueError("Source construction audit failed: " + "; ".join(audit["issues"]))
        self.sampler = api.build_jutrack_profile_sampler(self.candidate, max_step_m=max_step_m)
        self.editor = self.sampler.physical_editor

    def _audit(self, rows):
        return self.api.audit_lattice_constraints(rows, energy_gev=self.candidate.energy_gev,
            expected_length_m=self.expected_cell_length_m, expected_angle_rad=self.expected_cell_angle_rad,
            expected_bends=self.roles["bend_count"], require_mirror_symmetry=True, limits=self.limits,
            minimum_magnet_length_m=self.minimum_magnet_length_m,
            minimum_drift_length_m=self.minimum_drift_length_m)

    def linear_metrics(self, elements: Sequence[Any] | None = None) -> dict[str, Any]:
        """Cheap physical-cell optics with explicit signed Dx_m/Dpx and stability."""
        rows = self.rows if elements is None else _rows(elements)
        audit = self._audit(rows)
        if not audit["passed"]:
            return {"stable": False, "feasible": False, "failure": "; ".join(audit["issues"]),
                    "construction_audit": audit}
        result = self.editor.linear_metrics(rows)
        if not result.get("stable"):
            result.setdefault("failure", "The physical periodic cell is linearly unstable")
        return result

    def measure(self, elements: Sequence[Any] | None = None, *,
                include_radiation: bool = False) -> dict[str, Any]:
        """Flat numeric shape metrics plus complete diagnostics, not acceptance.

        Feasibility requires stable, symmetric, bounded, exact length/angle and
        achromatic optics. Final shape targets are NOT feasibility constraints
        on a conventional starting seed. Request acceptance still requires
        evaluate_hybrid_mba_shape on the exact archived winning candidate.
        """
        rows = self.rows if elements is None else _rows(elements)
        audit = self._audit(rows)
        if not audit["passed"]:
            return {"stable": False, "feasible": False, "shape_passed": False,
                    "failure": "; ".join(audit["issues"]), "construction_audit": audit}
        linear = self.sampler.linear_metrics(rows)
        if not linear["stable"]:
            return {**linear, "stable": False, "feasible": False, "shape_passed": False,
                    "failure": linear.get("failure", "The physical periodic cell is unstable"),
                    "construction_audit": audit}
        report = hybrid_mba_shape_diagnostics(rows, linear["twiss_profile"],
            thresholds=self.thresholds, sampling=linear["sampling"])
        # A continuous search surrogate must not add a 0/1 jump merely because
        # numerical slope bracketing changes on a near-flat conventional seed.
        # Retain a soft, resolved-peak objective: prominence over *both* outer
        # window boundaries, at a numerical search scale of 0.1% of bump height.
        # This is not an acceptance tolerance; the full interior/slope gate
        # remains mandatory on each selected continuation checkpoint.
        prominence_scale = self.thresholds["min_bump_m"] * 1e-3
        prominences = {}
        for side, peak in report["peaks"].items():
            endpoints = [min(report["profile"], key=lambda row: abs(row["s_m"] - position))["Dx"]
                         for position in (peak["region_start_m"], peak["region_end_m"])]
            prominences[side] = float(peak["Dx_m"] - max(endpoints))
        search_penalty = sum(value ** 2 for value in report["residuals"].values())
        search_penalty += sum(max(0., 1. - value / prominence_scale) ** 2 for value in prominences.values())
        result = {**report["metrics"], "Dx_m": linear["Dx_m"], "Dpx": linear["Dpx"],
                  "continuous_shape_penalty": search_penalty,
                  "search_peak_prominence_scale_m": prominence_scale,
                  "outer_peak_prominence_m": prominences,
                  "trace_x": linear["trace_x"], "trace_y": linear["trace_y"],
                  "maximum_beta_m": linear["maximum_beta_m"], "stable": True,
                  "feasible": bool(report["checks"]["achromatic_ID"] and report["checks"]["sampled_profile_and_map"]),
                  "shape_passed": report["passed"], "shape_report": report,
                  "construction_audit": audit}
        if include_radiation:
            periods = int(self.candidate.metadata.get("intended_repetitions",
                self.candidate.metadata.get("repetitions", self.candidate.periodicity)))
            if abs(periods * self.expected_cell_angle_rad - 2 * math.pi) > 1e-9:
                raise ValueError("Radiation requires the declared repeated cell to close 2*pi")
            radiation = self.api.jutrack_periodic_radiation_metrics(self.editor.jt, self.editor.lattice,
                energy_gev=self.candidate.energy_gev, periods=periods)
            result.update(radiation=radiation, emittance_pm=radiation.get("emittance_pm"))
            result["feasible"] &= bool(radiation.get("physical_validity", {}).get("passed"))
            decomposition = self._bend_radiation(rows, radiation, periods)
            result.update(decomposition)
        if not result["feasible"]:
            result["failure"] = "Stable proposal did not pass achromat or requested radiation feasibility"
        return result

    def _bend_radiation(self, rows, radiation, periods):
        """Exact JuTrack elrad, checked against ringpara; cell integrals, not a sampled H proxy."""
        jt, lattice = self.editor.jt, self.editor.lattice
        bridge = getattr(jt, "_jl", None)
        radiation_module = getattr(bridge, "JuTrack", bridge)
        if radiation_module is None or not hasattr(radiation_module, "elrad"):
            raise RuntimeError("Configured JuTrack must expose elrad for core radiation protection")
        points = list(jt.twissring(lattice, dp=0., refpts=list(range(1, len(rows) + 1))))
        bends = []
        roles = hybrid_mba_bend_roles(rows)
        for bend in roles["bends"]:
            i = bend["element_index"]
            values = [float(v) for v in radiation_module.elrad(lattice[i], points[i - 1], points[i])]
            if len(values) != 7 or not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"Invalid JuTrack elrad contribution for {bend['name']}")
            bends.append({"bend": bend["bend"], "name": bend["name"], "role": bend["role"],
                          **{f"I{n}": values[n-1] for n in (2, 4, 5)}})
        checks = {key: math.isclose(sum(row[key] for row in bends) * periods, radiation[key],
                                   rel_tol=2e-10, abs_tol=1e-13) for key in ("I2", "I4", "I5")}
        if not all(checks.values()):
            raise RuntimeError(f"JuTrack per-bend integrals disagree with ringpara: {checks}")
        return {"bend_radiation": {"engine": "JuTrack", "method": "elrad checked against ringpara",
                    "scope": "one_physical_cell", "rows": bends, "consistency": checks},
                "I5_core_m1": sum(row["I5"] for row in bends if row["role"] == "low_H_core"),
                "I2_core_m1": sum(row["I2"] for row in bends if row["role"] == "low_H_core")}

    def objective(self, elements: Sequence[Any], *, core_limits: Mapping[str, float] | None = None,
                  emittance_ceiling_pm: float | None = None) -> dict[str, Any]:
        """Independent shape objectives with optional, explicitly adopted ceilings.

        No source-optics or baseline H/I5/emittance ceiling is implicit. If the
        scientist adopts additional ceilings, declare any subset of D_core_m,
        H_core_max_m and I5_core_m1. They are independent, non-compensating
        constraints, not an equality to the source. Final acceptance is separate.
        """
        keys = ("D_core_m", "H_core_max_m", "I5_core_m1")
        core_limits = {} if core_limits is None else dict(core_limits)
        if not set(core_limits) <= set(keys) or any(not math.isfinite(float(v)) or float(v) <= 0 for v in core_limits.values()):
            raise ValueError(f"core_limits supports positive finite ceilings for {keys}")
        if emittance_ceiling_pm is not None and (not math.isfinite(emittance_ceiling_pm) or emittance_ceiling_pm <= 0):
            raise ValueError("emittance_ceiling_pm must be positive and finite")
        metrics = self.measure(elements, include_radiation=True)
        names = ("bump_deficit", "core_dispersion_m", "core_I5_m1")
        if not metrics.get("stable"):
            return {"feasible": False, "objectives": {k: float("inf") for k in names},
                    "constraint_violations": {"physical_trial": 1.}, "failure": metrics.get("failure")}
        violations = {k: max(0., metrics[k] / float(limit) - 1) for k, limit in core_limits.items()}
        if not metrics["feasible"]:
            violations["achromat_radiation_or_construction"] = 1.
        if emittance_ceiling_pm is not None:
            violations["emittance"] = max(0., metrics["emittance_pm"] / emittance_ceiling_pm - 1)
        residuals = metrics["shape_report"]["residuals"]
        bump = sum(residuals[k] ** 2 for k in ("left_height", "right_height"))
        bump += sum(max(0., 1 - value / metrics["search_peak_prominence_scale_m"]) ** 2
                    for value in metrics["outer_peak_prominence_m"].values())
        return {"feasible": not any(v > 1e-12 for v in violations.values()),
                "objectives": dict(zip(names, (bump, metrics["D_core_m"], metrics["I5_core_m1"]))),
                "constraint_violations": violations, "core_limits": dict(core_limits),
                "metrics": {key: metrics[key] for key in (*keys, "D_bump_L_m", "D_bump_R_m", "dispersion_contrast", "emittance_pm")}}

    def response(self, variable_specs: Sequence[Mapping[str, Any]], *,
                 metric_names: Sequence[str] = ("D_bump_L_m", "D_bump_R_m", "D_core_m", "H_core_max_m"),
                 inner_match: Callable[[list[dict[str, Any]]], Mapping[str, Any]] | None = None,
                 finite_difference_fraction: float = 1e-4, max_halvings: int = 10,
                 require_feasible: bool = True) -> dict[str, Any]:
        """Adaptive signed central/one-sided response, with optional inner match.

        Matcher callback returns {success, elements, ...}. Every perturbation
        is rematched independently from the same source. It may not overwrite
        the outer coordinates. Missing/unstable columns are explicitly recorded,
        never filled with zeros or extrapolated into a large unvalidated step.
        """
        fraction = float(finite_difference_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= .1:
            raise ValueError("finite_difference_fraction must be in (0, 0.1]")
        if isinstance(max_halvings, bool) or int(max_halvings) != max_halvings or not 0 <= max_halvings <= 30:
            raise ValueError("max_halvings must be an integer from 0 to 30")
        parameterization = self.api.compile_lattice_parameterization(self.rows, variable_specs)
        initial = np.asarray(parameterization.initial_position, dtype=float)
        names = list(metric_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("metric_names must be nonempty and unique")

        def evaluate(position):
            rows = parameterization.apply(position)
            if inner_match is not None:
                matched = inner_match(deepcopy(rows))
                if not isinstance(matched, Mapping) or matched.get("success") is not True:
                    return {"stable": False, "feasible": False,
                            "failure": "Inner achromat matching failed", "matching": matched}
                rows = _rows(matched["elements"])
                lookup = {row["name"]: row for row in rows}
                for knob, value in zip(parameterization.knobs, position):
                    if any(not math.isclose(float(lookup[name].get(knob.field) or 0), float(value),
                                            abs_tol=1e-12, rel_tol=1e-10) for name in knob.target_names):
                        raise ValueError(f"Inner matcher overwrote outer knob {knob.name!r}; ownership must be disjoint")
            return self.measure(rows)

        def usable(metrics):
            return (metrics.get("stable") is True and (not require_feasible or metrics.get("feasible") is True)
                    and all(name in metrics and math.isfinite(float(metrics[name])) for name in names))

        baseline = evaluate(initial)
        if not usable(baseline):
            return {"success": False, "reason": "baseline_not_evaluable", "baseline": baseline, "columns": []}
        columns = []
        for index, knob in enumerate(parameterization.knobs):
            attempts = []
            accepted = {}
            for halving in range(int(max_halvings) + 1):
                step = (knob.upper - knob.lower) * fraction / (2 ** halving)
                for sign in (1, -1):
                    if sign in accepted:
                        continue
                    position = initial.copy()
                    position[index] = np.clip(initial[index] + sign * step, knob.lower, knob.upper)
                    delta = float(position[index] - initial[index])
                    if abs(delta) < 1e-12:
                        continue
                    metrics = evaluate(position)
                    valid = usable(metrics)
                    attempts.append({"delta": delta, "stable": metrics.get("stable", False),
                                     "feasible": metrics.get("feasible", False), "usable": valid,
                                     "failure": metrics.get("failure"),
                                     "trace_x": metrics.get("trace_x"), "trace_y": metrics.get("trace_y")})
                    if valid:
                        accepted[sign] = (delta, metrics)
                if len(accepted) == 2:
                    break
            if len(accepted) == 2:
                plus, minus = accepted[1], accepted[-1]
                derivative = {name: (float(plus[1][name]) - float(minus[1][name])) / (plus[0] - minus[0]) for name in names}
                scheme = "signed_two_sided_secant"
            elif accepted:
                delta, metrics = next(iter(accepted.values()))
                derivative = {name: (float(metrics[name]) - float(baseline[name])) / delta for name in names}
                scheme = "signed_one_sided"
            else:
                derivative, scheme = None, "unresolved"
            columns.append({"name": knob.name, "element_names": list(knob.target_names),
                            "field": knob.field, "derivative": derivative, "scheme": scheme,
                            "accepted_deltas": [item[0] for item in accepted.values()], "attempts": attempts})
        complete = all(column["derivative"] is not None for column in columns)
        jacobian = ([[column["derivative"][name] for column in columns] for name in names] if complete else None)
        return {"success": complete, "baseline": baseline, "metric_names": names, "columns": columns,
                "jacobian": jacobian, "require_feasible": require_feasible,
                "limitations": ["Local derivative only. Validate bounded proposals after matching; do not extrapolate to a distant shape target.",
                                "Scale observables by their physical tolerances before interpreting response rank or condition numbers."]}
