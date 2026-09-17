"""Shared observation callback, not an HMBA construction or optimization route."""
from __future__ import annotations

import copy
import math

from brilliance import accelerator_api as api
from brilliance.lattice import LatticeIR

from .hybrid_shape import evaluate_hybrid_mba_shape
from .outcome_claims import _machine


class HybridMeasurement:
    """Native shape/radiation callback with explicit continuation eligibility.

    Supply original machine_contract, shape_requirements, maximum_emittance_pm
    and nonempty budget_provenance. The latter ceiling is a separate approved
    search allocation, never an inferred emittance aspiration. A rejected point
    retains real metrics but feasible=False: optimizers must not select it.
    This callback does not match, generate starts, or assert topology exhaustion.
    The final checker uses the SAME evaluate_hybrid_mba_shape measurement path.
    A new geometry needs its own matched feasible initialization; topology edits
    need a new callback. No native transport model is reimplemented.
    """
    def __init__(self, candidate, *, machine_contract, shape_requirements,
                 maximum_emittance_pm, budget_provenance, max_step_m=.05):
        self.base = candidate if isinstance(candidate, LatticeIR) else api.load_lattice(candidate)
        if isinstance(maximum_emittance_pm, bool) or not math.isfinite(maximum_emittance_pm) or maximum_emittance_pm <= 0 or not isinstance(budget_provenance, dict) or not budget_provenance:
            raise ValueError("Declare a positive emittance budget and its approval/proposal provenance before searching")
        self.machine = copy.deepcopy(machine_contract)
        self.thresholds = copy.deepcopy(shape_requirements)
        self.cap, self.provenance = maximum_emittance_pm, copy.deepcopy(budget_provenance)
        budget = {"maximum_emittance_pm": self.cap, "provenance": self.provenance}
        if "hybrid_radiation_budget" in self.machine and self.machine["hybrid_radiation_budget"] != budget:
            raise ValueError("Search and checking declare different hybrid radiation budgets")
        self.machine["hybrid_radiation_budget"] = budget
        self.step = max_step_m
        _machine(self.base, self.machine)  # schema preflight, not parent acceptance
        self.sampler = api.build_jutrack_profile_sampler(self.base, max_step_m=max_step_m)

    def __call__(self, elements):
        candidate = api.periodic_cell(title="Measured hybrid trial", energy_gev=self.base.energy_gev,
            elements=elements, intended_repetitions=self.machine["periods"], source="physical_mba_regions/v1")
        issues = _machine(candidate, self.machine)
        if issues:
            return {"feasible": False, "evaluation_status": "construction_rejected", "issues": issues,
                    "lattice_id": candidate.lattice_id, "physical_fingerprint": candidate.physical_fingerprint,
                    "topology_exhausted": False}
        shape = evaluate_hybrid_mba_shape(candidate, thresholds=self.thresholds,
            expected_cell_length_m=self.machine["expected_cell_length_m"],
            expected_cell_angle_rad=self.machine["expected_cell_angle_rad"], limits=self.machine["limits"],
            minimum_magnet_length_m=self.machine["minimum_magnet_length_m"],
            minimum_drift_length_m=self.machine["minimum_drift_length_m"], max_step_m=self.step, sampler=self.sampler,
            installation_slots=self.machine.get("installation_slots"))
        metrics, radiation = shape.get("metrics", {}), shape.get("radiation", {})
        measured_epsilon = radiation.get("emittance_pm")
        physical = shape.get("checks", {}).get("positive_damping", False)
        budget_ok = physical and measured_epsilon is not None and measured_epsilon <= self.cap
        checks = {"stable": shape.get("stable", False), "radiation_physical": physical,
            "emittance_budget": bool(budget_ok), "achromat": bool(shape.get("checks", {}).get("achromatic_ID")),
            "original_beta_cap": self.machine["maximum_beta_m"] is None or metrics.get("beta_max_m", math.inf) <= self.machine["maximum_beta_m"],
            "original_emittance_cap": self.machine["maximum_emittance_pm"] is None or measured_epsilon is not None and measured_epsilon <= self.machine["maximum_emittance_pm"],
            "original_energy_spread_cap": self.machine["maximum_energy_spread"] is None or radiation.get("energy_spread", math.inf) <= self.machine["maximum_energy_spread"]}
        quads = [row for row in api.one_period_elements(candidate) if row["kind"] in {"KQUAD", "QUAD"}]
        cap = self.machine["limits"]["max_abs_k1_m2"]
        saturated = [row["name"] for row in quads if abs(float(row.get("k1_m2") or 0)) >= cap - 1e-6]
        return {**metrics, "emittance_pm": measured_epsilon, "I2": radiation.get("I2"), "I5": radiation.get("I5"),
            "stable": shape.get("stable", False), "feasible": all(checks.values()),
            "evaluation_status": "eligible_continuation" if all(checks.values()) else "numerically_rejected",
            "request_satisfied": False, "shape_passed": shape["passed"], "checks": checks,
            "shape": shape, "lattice_id": candidate.lattice_id, "physical_fingerprint": candidate.physical_fingerprint,
            "quadrupole_bounds": {"physical_magnets": len(quads), "physical_magnets_at_limit": len(saturated),
                "element_names": saturated, "absolute_tolerance_m2": 1e-6,
                "independent_knobs_at_limit": None, "note": "Independent knob count requires the actual parameterization; not inferred from magnet count."},
            "maximum_emittance_pm": self.cap, "budget_provenance": self.provenance,
            "machine_contract": copy.deepcopy(self.machine), "topology_exhausted": False}
