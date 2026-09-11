"""Offline grouped H-MBA geometry experiment, preserving the source and project history.

This exercises public expert tools and generic MOPSO, not a stored design or an
LLM workflow. Three structural coordinates, disjoint inner quadrupole matching,
independent shape/core objectives, explicit core ceilings and complete evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from brilliance import accelerator_api as api
from brilliance.research_environment import ResearchEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    env = ResearchEnvironment.load()
    env.require_active_python(operation="grouped hybrid geometry verification")
    if env.internet_access:
        raise ValueError("This test requires internet research to be disabled")
    if args.output.exists():
        raise ValueError("Use a fresh output directory; prior evidence is never overwritten")
    args.output.mkdir(parents=True)
    digest = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    candidate = api.load_lattice(args.candidate)
    rows = api.one_period_elements(candidate)
    mba = api.expert_module("laboratory.mba_design")
    inventory = mba.hybrid_mba_regions(rows)
    roles = inventory["bend_roles"]
    energy, periods = candidate.energy_gev, int(candidate.metadata.get("intended_repetitions", candidate.periodicity))
    limits = api.magnet_strength_limits(energy)
    length = sum(row["length_m"] for row in rows)
    angle = sum(row.get("angle_rad") or 0 for row in rows)
    thresholds = {"min_bump_m": .005, "min_contrast": 1.5, "max_core_dispersion_m": .05,
                  "max_core_h_m": 1e-4, "max_sample_step_m": .02,
                  "peak_slope_tolerance": 1e-4, "id_dispersion_tolerance_m": 1e-4, "id_slope_tolerance": 1e-4}
    started = time.monotonic()
    evaluator = mba.HybridMBAShapeEvaluator(candidate, thresholds=thresholds,
        expected_cell_length_m=length, expected_cell_angle_rad=angle, limits=limits, max_step_m=.02)
    baseline = evaluator.measure(include_radiation=True)
    if not baseline["feasible"]:
        raise RuntimeError(f"Baseline is not feasible: {baseline.get('failure')}")
    # Original, not degraded shaped-descendant reference. 1e-8 relative numerical allowance.
    core_limits = {key: baseline[key] * (1 + 1e-8) for key in ("D_core_m", "H_core_max_m", "I5_core_m1")}
    thresholds["min_bump_m"] = min(baseline["D_bump_L_m"], baseline["D_bump_R_m"]) * 1.15
    # Transfer funds from explicitly inventoried non-ID terminal drifts; never the straight.
    reservoir_names = [r["name"] for r in inventory["elements"] if r["region"] == "terminal_matching"
                       and r["side"] == "left" and r["kind"] == "DRIFT" and r["index"] != 0]
    bump = roles["regions"]["left_bump"]
    suppressor = roles["regions"]["left_suppressor"]
    lb = bump["s_end_m"] - bump["s_start_m"]
    ls = suppressor["s_end_m"] - suppressor["s_start_m"]
    fraction = roles["bends"][0]["angle_rad"] / sum(b["angle_rad"] for b in roles["bends"][:2])
    geometry_bounds = {"bump_length_m": [lb*.96, lb*1.04], "suppressor_length_m": [ls*.96, ls*1.04],
                       "outer_bend_fraction": [fraction*.9, fraction*1.1]}
    geometry = mba.HybridMBARegionalGeometry(rows, bounds=geometry_bounds, reservoir_element_names=reservoir_names)
    specs = mba.hybrid_mba_quadrupole_knobs(rows, max_abs_k1_m2=limits["max_abs_k1_m2"], regions=["bump_generator"])
    matching_specs = specs[:2]  # Explicit hypothesis, response and match success checked below.
    emittance_ceiling = max(200., baseline["emittance_pm"])

    def make_worker():
        from brilliance import accelerator_api as api
        mba = api.expert_module("laboratory.mba_design")
        cell = api.periodic_cell(title="Regional geometry worker", energy_gev=energy, elements=rows,
                                 source="offline regional test", intended_repetitions=periods)
        local = mba.HybridMBAShapeEvaluator(cell, thresholds=thresholds,
            expected_cell_length_m=length, expected_cell_angle_rad=angle, limits=limits, max_step_m=.02)
        structure = mba.HybridMBARegionalGeometry(rows, bounds=geometry_bounds, reservoir_element_names=reservoir_names)
        matching = api.compile_lattice_parameterization(rows, matching_specs)

        def evaluate(position):
            proposed = structure.apply(position)
            matched = api.bounded_fast_response_match(proposed, matching, local.linear_metrics,
                {"Dx_m": (0., 1e-7), "Dpx": (0., 1e-7)}, max_evaluations=24, finite_difference_fraction=1e-4)
            if not matched["success"]:
                return {"feasible": False, "objectives": {k: float("inf") for k in ("bump_deficit", "core_dispersion_m", "core_I5_m1")},
                        "constraint_violations": {"inner_match": 1.}, "reason": matched["reason"]}
            result = local.objective(matched["elements"], core_limits=core_limits, emittance_ceiling_pm=emittance_ceiling)
            result["matched_elements"] = matched["elements"]
            result["match_evaluations"] = matched["evaluations"]
            return result
        return evaluate

    local_evaluate = make_worker()
    pilot = local_evaluate(geometry.initial_position)
    if not pilot["feasible"]:
        raise RuntimeError(f"Baseline objective pilot failed: {pilot}")
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    save("baseline.json", baseline)
    save("contract.json", {"geometry": geometry.describe(), "matching_knobs": matching_specs,
                           "core_limits": core_limits, "thresholds": thresholds, "inventory": inventory})
    print(json.dumps({"phase": "baseline", "elapsed_s": time.monotonic()-started,
        "objectives": pilot["objectives"], "core_limits": core_limits, "geometry": geometry.describe()}), flush=True)
    if args.pilot_only:
        return
    population = geometry.describe()["minimum_population"]
    rng = np.random.default_rng(918)
    lo, hi = np.asarray(geometry.bounds).T
    seeds = [geometry.initial_position]
    screened, rejection_counts = 0, {}
    with api.process_batch_evaluator(make_worker, workers=args.workers) as batch:
        # Every production seed must actually match and respect core constraints.
        while len(seeds) < population and screened < 600:
            sigma = .01 if screened < 200 else .1
            positions = np.clip(np.asarray(geometry.initial_position) + rng.normal(size=(30, 3))*sigma*(hi-lo), lo, hi)
            evaluations = batch(list(positions))
            for position, result in zip(positions, evaluations):
                screened += 1
                if result["feasible"]:
                    seeds.append(position.tolist())
                else:
                    for key, value in result.get("constraint_violations", {}).items():
                        if value > 1e-12:
                            rejection_counts[key] = rejection_counts.get(key, 0) + 1
            print(json.dumps({"phase": "feasible_screening", "screened": screened,
                "accepted": len(seeds), "rejections": rejection_counts}), flush=True)
        if len(seeds) < population:
            save("screening.json", {"screened": screened, "feasible": len(seeds), "rejections": rejection_counts})
            raise RuntimeError("Insufficient core-protected matched initial particles; this is recorded screening evidence, not geometry exhaustion")
        history = []
        def progress(event):
            history.append(event)
            save("progress.json", history)
            print(json.dumps({"phase": "mopso", **event}), flush=True)
        result = api.particle_swarm_optimize(local_evaluate, bounds=geometry.bounds,
            initial_position=geometry.initial_position, variable_names=geometry.variable_names,
            objective_names=("bump_deficit", "core_dispersion_m", "core_I5_m1"),
            population_size=population, generations=args.generations, seed=918,
            seed_positions=seeds[:population], evaluate_batch=batch, progress_callback=progress,
            initial_velocity_fraction=.002, checkpoint_path=args.output / "checkpoint.json",
            selection="first_objective")
    save("optimization.json", result)
    winner = result["best_candidate"]
    evaluated = local_evaluate(winner["position"])
    final_rows = evaluated["matched_elements"]
    final = api.cell_ring(title="Grouped geometry diagnostic", energy_gev=energy, cell=final_rows,
                         periods=periods, source="offline grouped geometry test", parent_lattice_id=candidate.lattice_id)
    final_measured = evaluator.measure(final_rows, include_radiation=True)
    final_gate = mba.evaluate_hybrid_mba_shape(final, thresholds=thresholds,
        expected_cell_length_m=length, expected_cell_angle_rad=angle, limits=limits, max_step_m=.02)
    save("candidate.json", final.as_dict())
    save("final_metrics.json", final_measured)
    save("final_shape_gate.json", final_gate)
    api.save_optics_lattice_plot(args.output / "optics.png", final_gate["profile"], final_rows,
        title="Grouped outer-geometry search: one periodic cell", subtitle=f"Shape gate passed: {final_gate['passed']}; core protection: {evaluated['feasible']}")
    preserved = hashlib.sha256(args.candidate.read_bytes()).hexdigest() == digest
    summary = {"elapsed_s": time.monotonic()-started, "source_preserved": preserved,
        "population": population, "generations": args.generations, "screened": screened,
        "rejections": rejection_counts, "selected_coordinates": winner["position"],
        "core_limits": core_limits, "initial_objectives": pilot["objectives"],
        "final_objectives": evaluated["objectives"], "shape_passed": final_gate["passed"],
        "core_protection_passed": evaluated["feasible"], "interpretation": "Bounded geometry experiment, not topology exhaustion or nonlinear validation"}
    save("summary.json", summary)
    print(json.dumps({"phase": "complete", **summary}), flush=True)
    if not preserved:
        raise RuntimeError("Source lattice unexpectedly changed")


if __name__ == "__main__":
    main()
