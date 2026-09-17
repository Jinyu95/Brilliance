"""Regression-only saved-unit handoff, executed by the real managed worker.

The input terminal is a historical numerical witness, not an expert-module seed.
Perturb it and rematch with JuTrack to exercise the complete publication path.
No TME optimization, facility template, or alternate transport model is used.
"""
import copy
import json
import math

from brilliance import accelerator_api as api
from brilliance.numerical_checkpoints import write_numerical_checkpoint
from brilliance.research_api import open_research_run

run = open_research_run()
options = run.arguments
unit = api.load_lattice(run.workspace / "inputs/unit.json")
witness = api.load_lattice(run.workspace / "inputs/cell_witness.json")
saved = json.loads((run.workspace / "inputs/interface.json").read_text(encoding="utf-8"))
periods, bends_per_cell = 32, 7
energy = unit.energy_gev
unit_rows = api.one_period_elements(unit)
unit_eval = api.evaluate_periodic_cell(unit, periods=192, include_profile=True)
unit_checkpoint = run.checkpoint_candidate(unit, unit_eval, name="saved_unit")
unit_view = api.evaluation_snapshot(unit_eval)
assert unit_view["linear_optics"]["stable"]
unit_exit = unit_view["linear_optics"]["twiss_profile"][-1]
state_keys = {"beta_x_m": "beta_x", "beta_y_m": "beta_y", "alpha_x": "alpha_x",
              "alpha_y": "alpha_y", "Dx_m": "Dx", "Dpx": "Dpx", "Dy_m": "Dy", "Dpy": "Dpy"}
incoming = {key: float(unit_exit[column]) for key, column in state_keys.items()}
for key, value in incoming.items():
    assert math.isclose(value, saved["incoming_unit_edge_state"][key], rel_tol=1e-7, abs_tol=1e-9), key

# Only this project's exact terminal is used as a regression witness.
terminal = copy.deepcopy(api.one_period_elements(witness)[:11])
assert [row["name"] for row in terminal] == [
    "T:D0", "T:Q1", "T:D1", "T:Q2", "T:D2", "T:B", "T:D3", "T:Q3", "T:D4", "T:Q4", "ID:L"]
limits = api.magnet_strength_limits(energy)
cap = limits["max_abs_k1_m2"]
for row in terminal:
    if row["kind"] == "KQUAD":
        row["k1_m2"] *= 0.99
knobs = api.compile_lattice_parameterization(terminal, [
    {"name": name, "field": "k1_m2", "element_names": [name], "bounds": [-cap, cap]}
    for name in ("T:Q1", "T:Q2", "T:Q3", "T:Q4")])
targets = {"alpha_x": (0.0, 1e-4), "alpha_y": (0.0, 1e-4),
           "Dx_m": (0.0, 1e-4), "Dpx": (0.0, 1e-5)}
match = api.match_jutrack_line(
    terminal, knobs, energy_gev=energy, initial_twiss=incoming, targets=targets,
    expected_length_m=5.7, expected_angle_rad=math.pi / 192,
    limits=limits, maximum_beta_m=100.0,
    minimum_magnet_length_m=0.15, minimum_drift_length_m=0.05,
    max_starts=2, max_nfev=100, seed=20260915,
)
# The runtime has saved the exact match BEFORE its optional final progress event.
assert match["match_success"], match["residuals"]
assert match["numerical_checkpoint"]
assert match["complete_cell_accepted"] is False
returned_terminal = match["elements"]
core_rows = api.repeat_cell(unit_rows, bends_per_cell - 2, period_prefix="C")
cell_rows = returned_terminal + api.mirror_elements(returned_terminal, name_prefix="R:") + core_rows
cell = api.periodic_cell(
    title="Integration replay: measured unit and native-rematched terminal",
    energy_gev=energy, elements=cell_rows, intended_repetitions=periods,
    source=__file__, parent_lattice_id=unit.lattice_id,
)
api.save_lattice(run.artifact_path("assembled_cell.json"), cell)
cell_eval = api.evaluate_periodic_cell(cell, include_profile=True)
cell_checkpoint = run.checkpoint_candidate(cell, cell_eval, name="complete_cell")
ring = api.cell_ring(
    title="Integration replay: explicit 32-cell ring", energy_gev=energy,
    cell=cell_rows, periods=periods, source=__file__, parent_lattice_id=cell.lattice_id,
)
api.save_lattice(run.artifact_path("explicit_ring.json"), ring)
ring_eval = api.evaluate(ring, include_radiation=True, include_profile=True)
ring_checkpoint = run.checkpoint_candidate(ring, ring_eval, name="full_ring")

# Independent complete-cell checks are numerical work, not optional formatting.
cell_view, ring_view = api.evaluation_snapshot(cell_eval), api.evaluation_snapshot(ring_eval)
periodic_start = cell_view["linear_optics"]["twiss_profile"][0]
periodic_incoming = {key: float(periodic_start[column]) for key, column in state_keys.items()}
profile = api.evaluate_jutrack_line(
    cell_rows, energy_gev=energy, initial_twiss=periodic_incoming,
    expected_length_m=25.0, expected_angle_rad=2 * math.pi / periods,
    limits=limits, maximum_beta_m=100.0,
    minimum_magnet_length_m=0.15, minimum_drift_length_m=0.05, profile_step_m=0.1,
)
id_row = next(row for row in profile["twiss_profile"] if row["name"] == "ID:L")
interface_rows = [profile["twiss_profile"][0]] + [
    row for row in profile["twiss_profile"]
    if row["name"] in {"R:T:D0", *(f"C{i:03d}:{unit_rows[-1]['name']}" for i in range(1, 6))}]
interface_residuals = [{"station": row["name"], "observable": key,
                        "measured": row[column], "unit_boundary": incoming[key],
                        "passed": math.isclose(float(row[column]), incoming[key], rel_tol=1e-5, abs_tol=1e-6)}
                       for row in interface_rows for key, column in state_keys.items()]
bend_evidence = api.bend_radiation_contributions(ring)
audit = api.audit_lattice_constraints(
    cell_rows, energy_gev=energy, expected_length_m=25.0,
    expected_angle_rad=2 * math.pi / periods, expected_bends=7, limits=limits,
    minimum_magnet_length_m=0.15, minimum_drift_length_m=0.05,
)
gates = {
    "terminal_match": match["match_success"], "geometry_and_fields": audit["passed"],
    "periodic_cell_stable": cell_view["linear_optics"]["stable"],
    "explicit_ring_stable": ring_view["linear_optics"]["stable"],
    "full_ring_circumference": math.isclose(ring_view["geometry"]["ring_circumference_m"], 800, abs_tol=1e-9),
    "physical_damping": ring_view["radiation"]["physical_validity"]["passed"],
    "id_Dx": abs(id_row["Dx"]) <= targets["Dx_m"][1],
    "id_Dpx": abs(id_row["Dpx"]) <= targets["Dpx"][1],
    "id_alpha_x": abs(id_row["alpha_x"]) <= targets["alpha_x"][1],
    "id_alpha_y": abs(id_row["alpha_y"]) <= targets["alpha_y"][1],
    "unit_interfaces_present": len(interface_rows) == 7,
    "unit_interface_optics": all(row["passed"] for row in interface_residuals),
    "sampled_beta_limit": profile["checks"]["maximum_beta"],
    "seven_bend_contributions": len(bend_evidence["rows"]) == 7,
    "per_bend_integrals_reconcile": bend_evidence["consistency"]["passed"],
    "cell_ring_radiation_agreement": math.isclose(
        cell_view["radiation"]["emittance_pm"], ring_view["radiation"]["emittance_pm"], rel_tol=1e-6),
}
diagnostics = {"gates": gates, "id_optics": id_row, "interface_residuals": interface_residuals,
               "sampled_beta_max_m": profile["metrics"]["maximum_beta_m"],
               "bend_contributions": bend_evidence, "match": match,
               "unit_evaluation": unit_view, "cell_evaluation": cell_view, "ring_evaluation": ring_view,
               "candidate_checkpoints": [unit_checkpoint, cell_checkpoint, ring_checkpoint]}
diagnostic_checkpoint = write_numerical_checkpoint("integration_diagnostics", diagnostics,
                                                  checks_passed=all(gates.values()))
assert all(gates.values()), gates

# Nothing scientific runs inside this marked optional reporting phase.
with run.reporting(checkpoint=ring_checkpoint):
    if options.get("fail_reporting"):
        raise AttributeError("Intentional reporting failure after verified numerical checkpoints")
    plot = api.save_optics_lattice_plot(
        run.artifact_path("complete_cell_optics.svg"), cell_view["linear_optics"]["twiss_profile"],
        cell_rows, title="Measured complete periodic cell: optics and lattice",
    )
    run.finish(
        summary="Saved-unit integration replay completed through the explicit 32-cell ring; no new design is certified.",
        data={"integration_diagnostics": diagnostic_checkpoint, "gates": gates,
              "ring_evaluation": ring_view, "match": match},
        tables=[{"title": "Complete-cell checks", "rows": [{"Check": key, "Passed": val} for key, val in gates.items()]},
                {"title": "Per-bend radiation contributions", "rows": bend_evidence["rows"]}],
        plots=[plot], candidate_lattice=ring, candidate_evaluation=ring_eval,
        scientific_passed=all(gates.values()), request_satisfied=False, use_as_working_lattice=False,
        limitations=["This is a historical-input worker integration regression, not a fresh autonomous design.",
                     "The repeated-unit reference emittance is not the 7BA ring emittance.",
                     "No nonlinear acceptance or full design certification; the 100 m beta cap is the saved prototype hypothesis."],
    )
