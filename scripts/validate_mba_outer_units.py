"""Explicit-input native route validation; never an installed design seed."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from brilliance.research_environment import ResearchEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--starts", type=int, default=64)
    args = parser.parse_args()
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="outer-unit native route validation")
    os.environ.update(environment.process_environment())
    from brilliance import accelerator_api as api
    from brilliance.lattice import LatticeIR
    from expert_modules.mba_design.outer_units import (
        assemble_mba_outer_units,
        match_mba_outer_units,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    machine = json.loads(args.contract.read_text(encoding="utf-8"))["machine"]

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

    start = time.monotonic()
    def progress(update):
        if update["completed_starts"] == 1 or update["completed_starts"] % 100 == 0:
            print(json.dumps(update), flush=True)
    assembly = assemble_mba_outer_units(unit["elements"], energy_gev=machine["energy_gev"],
        periods=machine["periods"], bends_per_cell=machine["bends_per_cell"],
        cell_length_m=machine["cell_length_m"], straight_length_m=machine["id_straight_length_m"],
        limits=machine["limits"], minimum_magnet_length_m=machine["minimum_magnet_length_m"],
        minimum_drift_length_m=machine["minimum_drift_length_m"],
        end_bend_fraction=machine["terminal_angle_rad"] / machine["unit_angle_rad"])
    save("assembly.json", assembly)
    print(json.dumps({"assembled": {k: assembly["geometry"][k] for k in ("passed", "length_m", "angle_rad", "bends")},
                      "outer_unit_length_m": assembly["outer_unit_length_m"]}), flush=True)
    result = match_mba_outer_units(assembly, energy_gev=machine["energy_gev"], periods=machine["periods"],
        limits=machine["limits"], maximum_beta_m=machine["maximum_beta_m"],
        minimum_magnet_length_m=machine["minimum_magnet_length_m"], minimum_drift_length_m=machine["minimum_drift_length_m"],
        max_starts=args.starts, progress_callback=progress)
    save("matching.json", result)
    if not result["matched"]:
        raise SystemExit("No matched complete cell; exact attempted results saved")
    cell = LatticeIR.from_dict(result["candidate"])
    api.save_lattice(args.output / "candidate_cell.json", cell)
    ring = api.cell_ring(title="Outer-unit route validation ring", energy_gev=machine["energy_gev"],
        cell=result["elements"], periods=machine["periods"], source="explicit fresh outer-unit validation")
    api.save_lattice(args.output / "candidate_ring.json", ring)
    evaluation = api.evaluate(ring, include_profile=True, include_radiation=True).as_dict()
    save("ring_evaluation.json", evaluation)
    sampled = api.build_jutrack_profile_sampler(cell, max_step_m=.05).linear_metrics()
    save("cell_profile.json", sampled)
    bend = api.bend_radiation_contributions(ring)
    save("bend_radiation.json", bend)
    linear, radiation = evaluation["linear_optics"], evaluation["radiation"]
    boundary = linear["twiss_profile"][0]
    beta_max = max(max(p["beta_x"], p["beta_y"]) for p in sampled["twiss_profile"])
    checks = {"ring_stable": linear["stable"], "radiation_valid": radiation["physical_validity"]["passed"],
              "achromat_Dx": abs(boundary["Dx"]) <= 1e-4, "achromat_Dpx": abs(boundary["Dpx"]) <= 1e-5,
              "beta_limit": beta_max <= machine["maximum_beta_m"],
              "circumference": abs(ring.total_length_m - machine["circumference_target_m"]) < 1e-8}
    summary = {"checks": checks, "passed": all(checks.values()), "ring_emittance_pm": radiation["emittance_pm"],
               "maximum_beta_m": beta_max, "boundary": boundary, "elapsed_s": time.monotonic() - start,
               "matching_evaluations": result["evaluations"], "matching_elapsed_s": result["elapsed_s"]}
    save("summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["passed"]:
        raise SystemExit("Complete-ring gate failed; artifacts retained")


if __name__ == "__main__":
    main()
