"""Offline, from-scratch modified-TME method experiment (not an agent seed asset).

Run with the configured interpreter. Numerical settings are searched anew; no
saved lattice is read. The output is validation evidence, never a module template.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from brilliance import accelerator_api as api
from brilliance.research_environment import ResearchEnvironment


def unit_rows(position, *, length, angle):
    bf, qf, gf, kf, kd = map(float, position)
    lb, lq, gap = length * bf, length * qf, length * gf
    end = (length - lb - 4 * lq - 4 * gap) / 2
    left = [api.drift("END", end), api.quadrupole("QF", lq, kf, family="QF"),
            api.drift("G1", gap), api.quadrupole("QD", lq, kd, family="QD"), api.drift("G2", gap)]
    return api.symmetric_cell_elements(left, center=api.sector_bend("B", lb, angle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy", type=float, default=3.5)
    parser.add_argument("--periods", type=int, default=32)
    parser.add_argument("--bends", type=int, default=7)
    parser.add_argument("--circumference", type=float, default=800.0)
    parser.add_argument("--straight", type=float, default=5.0)
    parser.add_argument("--matching-budget", type=float, default=4.0)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    runtime = ResearchEnvironment.load()
    runtime.require_active_python(operation="modified-TME method validation")
    if runtime.internet_access:
        raise RuntimeError("This validation must run with research internet access disabled")
    if args.periods < 1 or args.bends < 3 or args.straight <= 0 or args.matching_budget <= 0:
        raise ValueError("invalid machine or length-allocation contract")
    if args.population < 50:
        raise ValueError("Use at least ten particles per independent knob (five knobs)")
    args.output.mkdir(parents=True, exist_ok=True)
    source_bytes = Path(__file__).read_bytes()
    archived_source = args.output / "method.py"
    if archived_source.resolve() != Path(__file__).resolve():
        shutil.copyfile(__file__, archived_source)
    started = time.monotonic()
    # Hypothesis: M-2 interior units + two half-angle end bends. The separate
    # outer budget includes both ends; it is reserved, never stolen later.
    length = (args.circumference / args.periods - args.straight - args.matching_budget) / (args.bends - 2)
    angle = 2 * math.pi / (args.periods * (args.bends - 1))
    repetitions = args.periods * (args.bends - 1)
    limits = api.magnet_strength_limits(args.energy)
    maximum = limits["max_abs_k1_m2"]
    bounds = [(0.20, 0.36), (0.08, 0.14), (0.015, 0.04), (0.05, maximum), (-maximum, -0.05)]
    initial = [0.25, 0.11, 0.02, maximum / 2, -maximum / 2]
    rows = unit_rows(initial, length=length, angle=angle)
    cell = api.periodic_cell(title="From-scratch unit experiment", energy_gev=args.energy,
        elements=rows, intended_repetitions=repetitions, source=__file__)
    editor = api.build_jutrack_editor(cell)
    evaluations = 0
    cache = {}

    def objective(position):
        nonlocal evaluations
        key = tuple(float(x) for x in position)
        if key in cache:
            return cache[key]
        evaluations += 1
        invalid = {"objectives": {"emittance_pm": math.inf}, "feasible": False}
        if float(position[0]) + 4 * float(position[1]) + 4 * float(position[2]) >= 1:
            return invalid
        candidate = unit_rows(position, length=length, angle=angle)
        audit = api.audit_lattice_constraints(candidate, energy_gev=args.energy, expected_length_m=length,
            expected_angle_rad=angle, expected_bends=1, require_mirror_symmetry=True,
            minimum_magnet_length_m=0.1, minimum_drift_length_m=0.02, limits=limits)
        if not audit["passed"]:
            return invalid
        metrics = editor.linear_metrics(candidate, include_profile=True)
        if not metrics["stable"] or max(abs(metrics["trace_x"]), abs(metrics["trace_y"])) > 1.98 or metrics["maximum_beta_m"] > 100:
            return invalid
        radiation = api.jutrack_periodic_radiation_metrics(editor.jt, editor.lattice,
            energy_gev=args.energy, periods=repetitions)
        if not radiation["physical_validity"]["passed"]:
            return invalid
        result = {"objectives": {"emittance_pm": radiation["emittance_pm"]}, "feasible": True,
                  "metrics": {"maximum_beta_m": metrics["maximum_beta_m"]}}
        cache[key] = result
        return result

    rng = np.random.default_rng(args.seed)
    feasible = []
    for attempt in range(12000):
        position = [rng.uniform(lo, hi) for lo, hi in bounds]
        if objective(position)["feasible"]:
            feasible.append(position)
        if len(feasible) == args.population:
            break
    if len(feasible) < args.population:
        raise RuntimeError(f"Insufficient feasible initial population: {len(feasible)}; no optimizer started")
    print(json.dumps({"stage": "feasible_population", "size": len(feasible), "evaluations": evaluations,
                      "unit_length_m": length, "angle_rad": angle}), flush=True)
    initial_best = min(objective(x)["objectives"]["emittance_pm"] for x in feasible)

    def progress(*values):
        event = values[-1]
        if isinstance(event, dict) and "generation" in event and "best" in event:
            print(json.dumps({"stage": "optimization", "generation": event["generation"], "best": event["best"]}), flush=True)

    result = api.particle_swarm_optimize(objective, bounds=bounds, objective_names=["emittance_pm"],
        initial_position=feasible[0], seed_positions=feasible[1:], population_size=args.population,
        generations=args.generations, seed=args.seed, variable_names=["bend_fraction", "quad_fraction", "gap_fraction", "QF", "QD"],
        progress_callback=progress)
    (args.output / "optimizer.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    best = [result["best_candidate"]["variables"][name] for name in
            ("bend_fraction", "quad_fraction", "gap_fraction", "QF", "QD")]
    selected = unit_rows(best, length=length, angle=angle)
    cell = api.periodic_cell(title="Optimized modified-TME unit (not an MBA)", energy_gev=args.energy,
        elements=selected, intended_repetitions=repetitions, source=__file__)
    full_ring = api.cell_ring(title="Closed repeated-unit reference ring (not MBA)", energy_gev=args.energy,
        cell=selected, periods=repetitions, source=__file__, parent_lattice_id=cell.lattice_id)
    verified = api.evaluate(full_ring, include_profile=False)
    periodic = api.evaluate(cell, include_profile=True)
    rad = verified.require_radiation()
    # A separate diagnostic split samples the physical bend centre. It does not
    # replace the archived unsplit bend or count as two magnets in the design.
    split = []
    for row in selected:
        if row["name"] == "B":
            split.extend([api.sector_bend("B:L", row["length_m"] / 2, angle / 2),
                          api.sector_bend("B:R", row["length_m"] / 2, angle / 2)])
        else:
            split.append(row)
    sampled = api.periodic_cell(title="Centre sampling only", energy_gev=args.energy, elements=split,
        intended_repetitions=repetitions, source=__file__)
    centre_eval = api.evaluate(sampled, include_profile=True)
    split_map_error = float(np.max(np.abs(np.asarray(centre_eval.linear_optics.one_turn_matrix)
                                         - np.asarray(periodic.linear_optics.one_turn_matrix))))
    split_radiation = api.evaluate_periodic_cell(sampled, periods=repetitions)["radiation"]
    selected_audit = api.audit_lattice_constraints(selected, energy_gev=args.energy,
        expected_length_m=length, expected_angle_rad=angle, expected_bends=1,
        require_mirror_symmetry=True, limits=limits)
    centre = next(row for row in centre_eval.linear_optics.twiss_profile if row["name"] == "B:L")
    bend_length = next(row["length_m"] for row in selected if row["name"] == "B")
    reference = api.tme_targets(energy_gev=args.energy, bend_length_m=bend_length, bend_angle_rad=angle)
    final = float(rad["emittance_pm"])
    comparison = []
    for observable, achieved, analytic in [("beta_x_m", centre["beta_x"], reference.beta_x_center_m),
        ("alpha_x", centre["alpha_x"], 0), ("Dx_m", centre["Dx"], reference.dispersion_x_center_m),
        ("Dpx", centre["Dpx"], 0), ("emittance_pm", final, reference.minimum_emittance_pm)]:
        comparison.append({"observable": observable, "achieved": achieved, "homogeneous_TME_reference": analytic})
    api.save_lattice(args.output / "unit.json", cell)
    api.save_lattice(args.output / "repeated_unit_ring.json", full_ring)
    api.save_optics_lattice_plot(args.output / "unit_optics.png", centre_eval.linear_optics.twiss_profile, selected,
        title="Optimized modified-TME unit: JuTrack optics", subtitle="Seed experiment only; not a complete MBA design")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = result["convergence"]["history"]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot([row["generation"] for row in history], [row["best"]["emittance_pm"] for row in history])
    ax.set(xlabel="Generation", ylabel="Best feasible emittance [pm rad]", title="Modified-TME unit optimization")
    ax.grid(alpha=0.25)
    fig.savefig(args.output / "optimization.png", dpi=160)
    plt.close(fig)
    ratio = final / reference.minimum_emittance_pm
    report = {"scope": "modified_TME_unit_only", "method": "fresh feasible sampling then PSO; no stored lattice, no tune targets",
        "method_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "contract": {**vars(args), "output": str(args.output)}, "length_budget": {"unit_m": length, "total_outer_matching_m": args.matching_budget},
        "initial_best_emittance_pm": initial_best, "final_emittance_pm": final, "tme_ratio": ratio,
        "bend_centre_comparison": comparison, "full_ring_stable": verified.linear_optics.stable,
        "periodic_closure_max": verified.linear_optics.periodic_closure_max, "radiation": rad,
        "construction_audit": selected_audit, "split_map_error": split_map_error,
        "split_emittance_relative_error": abs(split_radiation["emittance_pm"] / final - 1),
        "optimizer": result, "limits": limits, "evaluations": evaluations, "duration_s": time.monotonic() - started,
        "passed": bool(verified.linear_optics.stable and verified.linear_optics.periodic_closure_max < 1e-6
                       and selected_audit["passed"] and split_map_error < 1e-8
                       and abs(split_radiation["emittance_pm"] / final - 1) < 1e-3
                       and final > 0 and ratio < 10 and final < initial_best)}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "initial_best_emittance_pm", "final_emittance_pm", "tme_ratio", "duration_s", "bend_centre_comparison")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
