"""Managed geometry-search regression; history supplies only E/U/angle/fields.

No saved length allocation or quadrupole strength seeds this search. Numerical
beta/stability safeguards are explicit test hypotheses, not machine approval.
"""
import json
from unittest.mock import patch

from brilliance import accelerator_api as api
from brilliance import candidate_evidence
from brilliance.lattice import LatticeIR
from brilliance.research_api import open_research_run

run = open_research_run()
mba = api.expert_module("laboratory.mba_design")
source = api.load_lattice(run.workspace / "inputs/unit.json")
original = api.one_period_elements(source)
bend = dict(original[5])
inputs = dict(
    energy_gev=source.energy_gev, bend=bend,
    expected_length_m=source.total_length_m, reference_repetitions=192,
    limits=api.magnet_strength_limits(source.energy_gev),
    maximum_beta_m=100., trace_margin_x=.01, trace_margin_y=.01,
    minimum_magnet_length_m=.15, minimum_drift_length_m=.05,
    seed=20260915, geometry_candidates=3,
    scan_resolutions=(25, 49), local_starts=12, local_max_evaluations=240,
    candidates_to_return=4,
)
result = mba.optimize_tme_unit_geometry(**inputs)
# Simulate a later worker with no in-memory captures. Re-publication must read
# verified native archives, never buy another optimization or final evaluation.
candidate_evidence._captured.clear()
with patch.object(api, "particle_swarm_optimize", side_effect=AssertionError("Resume reran PSO")), \
     patch.object(api, "evaluate", side_effect=AssertionError("Resume reran full-ring physics")), \
     patch.object(api, "evaluate_periodic_cell", side_effect=AssertionError("Resume reran unit optics")), \
     patch.object(api, "jutrack_linear_metrics", side_effect=AssertionError("Resume reran screening")):
    restored = mba.optimize_tme_unit_geometry(**inputs, resume=True)
assert restored["resumed_completed_result"]
assert [c["elements"] for c in result["candidates"]] == [c["elements"] for c in restored["candidates"]]
result = restored
run.artifact_path("geometry_search.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
assert result["passed"], result.get("initialization")
best = result["candidates"][0]
candidate = LatticeIR.from_dict(best["periodic_cell"])
checkpoint = run.checkpoint_candidate(candidate, best["evaluation"], name="geometry_optimized_unit")
with run.reporting(checkpoint=checkpoint):
    plot = api.save_optics_lattice_plot(
        run.artifact_path("geometry_optimized_unit_optics.svg"), best["optics_profile"],
        best["elements"], title="Geometry-optimized periodic unit: native optics and intact bend",
    )
    run.finish(
        summary="Joint geometry/strength search produced verified periodic units; complete MBA matching remains separate.",
        data={"search": "geometry_search.json"},
        tables=[{"title": "Verified geometry-optimized units", "rows": [
            {"Reference emittance (pm)": row["emittance_pm"],
             "Bend length (m)": row["elements"][5]["length_m"],
             "Near quadrupole length (m)": row["elements"][3]["length_m"],
             "Far quadrupole length (m)": row["elements"][1]["length_m"],
             "K1 near (m⁻²)": row["k1_near_m2"], "K1 far (m⁻²)": row["k1_far_m2"]}
            for row in result["candidates"]]}],
        plots=[plot], candidate_lattice=candidate, candidate_evaluation=best["evaluation"],
        scientific_passed=True, request_satisfied=False, use_as_working_lattice=False,
        limitations=["100 m sampled beta cap and 0.01 trace margins are provisional test safeguards.",
                     "Neither a proven global minimum nor complete-MBA acceptance is claimed."],
    )
