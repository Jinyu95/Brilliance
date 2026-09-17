"""New managed worker: reuse a completed campaign, then publish its exact result."""
from pathlib import Path
from unittest.mock import patch

from brilliance import accelerator_api as api
from brilliance.lattice import LatticeIR
from brilliance.research_api import open_research_run

run = open_research_run()
source = api.load_lattice(run.workspace / "inputs/unit.json")
previous = Path(run.arguments["previous_output"])
mba = api.expert_module("laboratory.mba_design")
with patch.object(api, "particle_swarm_optimize", side_effect=AssertionError("Recovery reran PSO")), \
     patch.object(api, "evaluate", side_effect=AssertionError("Recovery reran ring physics")), \
     patch.object(api, "evaluate_periodic_cell", side_effect=AssertionError("Recovery reran optics")), \
     patch.object(api, "jutrack_linear_metrics", side_effect=AssertionError("Recovery reran screening")):
    result = mba.optimize_tme_unit_geometry(
        energy_gev=source.energy_gev, bend=api.one_period_elements(source)[5],
        expected_length_m=source.total_length_m, reference_repetitions=192,
        limits=api.magnet_strength_limits(source.energy_gev),
        maximum_beta_m=100., trace_margin_x=.01, trace_margin_y=.01,
        minimum_magnet_length_m=.15, minimum_drift_length_m=.05,
        seed=20260915, geometry_candidates=3, scan_resolutions=(25, 49),
        local_starts=12, local_max_evaluations=240, candidates_to_return=4,
        checkpoint_path=previous / "tme_geometry_swarm.json", resume=True,
    )
assert result["resumed_completed_result"]
best = result["candidates"][0]
candidate = LatticeIR.from_dict(best["periodic_cell"])
checkpoint = run.checkpoint_candidate(candidate, best["evaluation"], name="recovered_unit")
with run.reporting(checkpoint=checkpoint):
    run.finish(summary="Recovered the original unit in a new worker without rerunning numerical work.",
        candidate_lattice=candidate, candidate_evaluation=best["evaluation"],
        scientific_passed=True, request_satisfied=False, use_as_working_lattice=False,
        tables=[{"title": "Original measured unit", "rows": [{"Emittance (pm)": best["emittance_pm"]}]}],
        limitations=["Unit-reference result, not a complete MBA."])
