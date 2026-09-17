"""Native unit-design regression, run by the configured application worker.

Historical input supplies geometry only, never strengths or a design template.
Safeguards below are explicit TEST hypotheses, not scientist-approved limits.
"""
import json

from brilliance import accelerator_api as api
from brilliance.lattice import LatticeIR
from brilliance.research_api import open_research_run

run = open_research_run()
mba = api.expert_module("laboratory.mba_design")
source = api.load_lattice(run.workspace / "inputs/unit.json")
rows = api.one_period_elements(source)
assert len(rows) == 11
geometry = run.arguments.get("geometry", {})
bend = dict(rows[5])
if "bend_length_m" in geometry:
    bend["length_m"] = geometry["bend_length_m"]
result = mba.design_tme_unit(
    energy_gev=source.energy_gev, bend=bend,
    d1_m=geometry.get("d1_m", rows[4]["length_m"]),
    d2_m=geometry.get("d2_m", rows[2]["length_m"]),
    d3_m=geometry.get("d3_m", rows[0]["length_m"]),
    quadrupole_near_length_m=geometry.get("near_length_m", rows[3]["length_m"]),
    quadrupole_far_length_m=geometry.get("far_length_m", rows[1]["length_m"]),
    expected_length_m=source.total_length_m, reference_repetitions=192,
    limits=api.magnet_strength_limits(source.energy_gev),
    maximum_beta_m=100.0, trace_margin_x=.01, trace_margin_y=.01,
    minimum_magnet_length_m=.15, minimum_drift_length_m=.05,
    scan_resolutions=tuple(run.arguments.get("resolutions", [25, 49])),
    local_starts=12, local_max_evaluations=240, candidates_to_return=4,
)
run.artifact_path("search.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
assert result["passed"], {key: result[key] for key in ("resolution_levels", "rejected_final_candidates")}
best = result["candidates"][0]
candidate = LatticeIR.from_dict(best["periodic_cell"])
checkpoint = run.checkpoint_candidate(candidate, best["evaluation"], name="best_verified_unit")
with run.reporting(checkpoint=checkpoint):
    plot = api.save_optics_lattice_plot(
        run.artifact_path("best_unit_optics.svg"), best["optics_profile"],
        best["elements"], title="Verified periodic unit: optics and intact physical bend",
    )
    run.finish(
        summary="Deterministic unit search completed under explicitly provisional test safeguards; not a complete MBA.",
        data={"search": "search.json", "resolution_levels": result["resolution_levels"]},
        tables=[{"title": "Verified periodic-unit alternatives", "rows": [
            {"Polarity": row["polarity"], "K1 near (m⁻²)": row["k1_near_m2"],
             "K1 far (m⁻²)": row["k1_far_m2"], "Reference emittance (pm)": row["emittance_pm"]}
            for row in result["candidates"]]}],
        plots=[plot], candidate_lattice=candidate, candidate_evaluation=best["evaluation"],
        scientific_passed=True, request_satisfied=False, use_as_working_lattice=False,
        limitations=["100 m beta cap and 0.01 trace margins are provisional validation inputs, not approved machine limits.",
                     "Fixed-geometry search does not demonstrate global optimality or complete-MBA acceptance."],
    )
