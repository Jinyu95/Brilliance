"""Integration witness for experiment selection, native physics and publication.

This is deliberately NOT a greenfield accelerator design example or installed
seed. Exact historical inputs exercise a bounded terminal rematch, complete-cell
assembly and an optional, genuinely nonzero quadrupole-removal hypothesis.
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
saved_interface = json.loads((run.workspace / "inputs/interface.json").read_text(encoding="utf-8"))
periods, energy = 32, unit.energy_gev
limits = api.magnet_strength_limits(energy)
cap = limits["max_abs_k1_m2"]
unit_rows = api.one_period_elements(unit)
unit_evaluation = api.evaluate_periodic_cell(unit, periods=192, include_profile=True)
unit_view = api.evaluation_snapshot(unit_evaluation)
assert unit_view["linear_optics"]["stable"]
state_keys = {"beta_x_m": "beta_x", "beta_y_m": "beta_y", "alpha_x": "alpha_x",
              "alpha_y": "alpha_y", "Dx_m": "Dx", "Dpx": "Dpx", "Dy_m": "Dy", "Dpy": "Dpy"}
unit_exit = unit_view["linear_optics"]["twiss_profile"][-1]
incoming = {key: float(unit_exit[column]) for key, column in state_keys.items()}
for key, value in incoming.items():
    assert math.isclose(value, saved_interface["incoming_unit_edge_state"][key],
                        rel_tol=1e-7, abs_tol=1e-9), key

targets = {"alpha_x": (0.0, 1e-4), "alpha_y": (0.0, 1e-4),
           "Dx_m": (0.0, 1e-4), "Dpx": (0.0, 1e-5)}


def rematch_terminal(rows):
    """One small native search; never reoptimizes or regenerates the saved unit."""
    knobs = api.compile_lattice_parameterization(rows, [
        {"name": row["name"], "field": "k1_m2", "element_names": [row["name"]],
         "bounds": [-cap, cap]} for row in rows if row["kind"] == "KQUAD"
    ])
    return api.match_jutrack_line(
        rows, knobs, energy_gev=energy, initial_twiss=incoming, targets=targets,
        expected_length_m=5.7, expected_angle_rad=math.pi / 192,
        limits=limits, maximum_beta_m=100.0,
        minimum_magnet_length_m=0.15, minimum_drift_length_m=0.05,
        max_starts=2, max_nfev=100, seed=20260915,
    )


def assemble_at_id(rows):
    """Start at the physical ID symmetry plane, not an arbitrary unit boundary."""
    return (api.mirror_elements(rows, name_prefix="R:")
            + api.repeat_cell(unit_rows, 5, period_prefix="C") + copy.deepcopy(rows))


terminal = copy.deepcopy(api.one_period_elements(witness)[:11])
assert terminal[-1]["name"] == "ID:L"
for row in terminal:
    if row["kind"] == "KQUAD":
        row["k1_m2"] *= 0.99
parent_match = rematch_terminal(terminal)
assert parent_match["match_success"], parent_match["residuals"]
assert parent_match["callback_evaluations"] > 1
assert parent_match["numerical_checkpoint"]
assert parent_match["complete_cell_accepted"] is False
parent_rows = assemble_at_id(parent_match["elements"])
parent_cell = api.periodic_cell(
    title="Integration witness: terminal rematched and cell cut at ID centre",
    energy_gev=energy, elements=parent_rows, intended_repetitions=periods,
    source=__file__, parent_lattice_id=unit.lattice_id,
)
api.save_lattice(run.artifact_path("parent_cell.json"), parent_cell)

definition_arguments = {
    "name": "saved-unit-composition-integration",
    "scope": "periodic_cell",
    "objectives": {"emittance_pm": "min"},
    "constraints": {
        "stable": {"equals": True},
        "radiation_valid": {"equals": True},
        "terminal_matched": {"equals": True},
        "geometry_valid": {"equals": True},
        "Dx_m": {"lower": -1e-4, "upper": 1e-4, "scale": 1e-4},
        "Dpx": {"lower": -1e-5, "upper": 1e-5, "scale": 1e-5},
        "maximum_beta_m": {"upper": 100.0, "scale": 100.0},
    },
    "provenance": {
        "scope": "historical-input integration, not greenfield design",
        "hypothesis": "the saved prototype's 100 m beta envelope, not a new machine requirement",
        "source_project": "project-ab41b8de97b4c4",
        "source_inputs": ["inputs/unit.json", "inputs/cell_witness.json", "inputs/interface.json"],
    },
}
match_context = {"terminal_matched": True}
native_measure = api.JuTrackCellEvaluator(
    energy_gev=energy, periods=periods, expected_length_m=25.0,
    expected_angle_rad=2 * math.pi / periods, limits=limits,
    minimum_magnet_length_m=0.15, minimum_drift_length_m=0.05,
)


def measure(rows):
    # The built-in native evaluator is also under test. Final publication below
    # independently compares it with the high-level periodic and full-ring APIs.
    metrics = native_measure(rows)
    metrics["terminal_matched"] = match_context["terminal_matched"]
    return metrics


measure.contract = native_measure.contract


experiment = api.LatticeExperiment(
    api.ExperimentDefinition(**definition_arguments), parameterization=None,
    measure=measure, directory=run.artifact_path("experiment"),
)
parent_trial = experiment.evaluate_rows(parent_rows, label="native-rematched historical parent")
parent_selection = experiment.ledger.select()
assert parent_selection["selected"] is not None, parent_selection

# Exercise reuse of one native editor after a scalar-only paired-field update.
# This is explicitly an unmatched diagnostic, not a new feasible design claim.
scalar_rows = copy.deepcopy(parent_rows)
for row in scalar_rows:
    if row["name"] in {"T:Q1", "R:T:Q1"}:
        row["k1_m2"] *= 0.999
match_context["terminal_matched"] = False
scalar_trial = experiment.evaluate_rows(scalar_rows, label="unmatched scalar-only paired-quadrupole diagnostic")
assert scalar_trial["feasible"] is False
assert experiment.ledger.select()["selected"] == parent_selection["selected"]

removal = None
if options.get("remove_pair", True):
    # These are nonzero physical quadrupoles in the saved matching region. Their
    # removal is a falsifiable structural trial, not a guaranteed successful edit.
    removed_names = ["T:Q4", "R:T:Q4"]
    before_strengths = {row["name"]: row["k1_m2"] for row in parent_rows
                        if row["name"] in removed_names}
    assert len(before_strengths) == 2 and all(abs(value) > 0.1 for value in before_strengths.values())
    changed = api.remove_mirrored_quadrupole_pair(parent_rows, element_names=removed_names)
    assert changed["requires_native_rebuild"] and not changed["physics_verified"]
    removed_terminal = copy.deepcopy(changed["elements"][-11:])
    removal_match = rematch_terminal(removed_terminal)
    removal_rows = assemble_at_id(removal_match["elements"] or removed_terminal)
    match_context["terminal_matched"] = bool(removal_match["match_success"])
    removal_trial = experiment.evaluate_rows(removal_rows, label="nonzero terminal-pair removal and native rematch")
    removal = {"before_strengths": before_strengths, "changes": changed,
               "match": removal_match, "trial": removal_trial}

selection = experiment.ledger.select()
selected = selection["selected"]
assert selected is not None, selection
selected_rows = selected["candidate"]["elements"]
selected_cell = api.periodic_cell(
    title="Experiment-selected periodic cell (historical integration witness)",
    energy_gev=energy, elements=selected_rows, intended_repetitions=periods,
    source=__file__, parent_lattice_id=parent_cell.lattice_id,
)
api.save_lattice(run.artifact_path("selected_cell.json"), selected_cell)
cell_evaluation = api.evaluate_periodic_cell(selected_cell, periods=periods, include_profile=True)
cell_checkpoint = run.checkpoint_candidate(selected_cell, cell_evaluation, name="experiment_selected_cell")
ring = api.cell_ring(
    title="Experiment-selected explicit 32-cell ring (integration witness)",
    energy_gev=energy, cell=selected_rows, periods=periods, source=__file__,
    parent_lattice_id=selected_cell.lattice_id,
)
api.save_lattice(run.artifact_path("explicit_ring.json"), ring)
ring_evaluation = api.evaluate(ring, include_radiation=True, include_profile=True)
ring_checkpoint = run.checkpoint_candidate(ring, ring_evaluation, name="experiment_selected_ring")
cell_view, ring_view = api.evaluation_snapshot(cell_evaluation), api.evaluation_snapshot(ring_evaluation)
optics, radiation = ring_view["linear_optics"], ring_view["radiation"]
bend_evidence = api.bend_radiation_contributions(ring)
gates = {
    "parent_native_terminal_match": parent_match["match_success"],
    "periodic_cell_stable": cell_view["linear_optics"]["stable"],
    "explicit_ring_stable": optics["stable"],
    "complete_ring_800m": math.isclose(ring_view["geometry"]["ring_circumference_m"], 800, abs_tol=1e-9),
    "physical_radiation": radiation["physical_validity"]["passed"],
    "id_achromat": abs(optics["Dx_m"]) <= 1e-4 and abs(optics["Dpx"]) <= 1e-5,
    "beta_envelope": optics["maximum_beta_m"] <= 100.0,
    "cell_ring_radiation_agreement": math.isclose(
        cell_view["radiation"]["emittance_pm"], radiation["emittance_pm"], rel_tol=1e-6),
    "fast_evaluator_cell_radiation_agreement": math.isclose(
        selected["measurements"]["emittance_pm"], cell_view["radiation"]["emittance_pm"], rel_tol=1e-6),
    "seven_bend_contributions": len(bend_evidence["rows"]) == 7,
    "per_bend_integrals_reconcile": bend_evidence["consistency"]["passed"],
}
diagnostics = {
    "definition": definition_arguments, "gates": gates, "parent_match": parent_match,
    "parent_trial": parent_trial, "parent_selection": parent_selection,
    "selection": selection, "entries": experiment.ledger.entries(), "removal": removal,
    "scalar_trial": scalar_trial, "bend_contributions": bend_evidence,
    "unit_reference_emittance_pm": unit_view["radiation"]["emittance_pm"],
    "cell_evaluation": cell_view, "ring_evaluation": ring_view,
    "cell_checkpoint": cell_checkpoint, "ring_checkpoint": ring_checkpoint,
}
diagnostic_checkpoint = write_numerical_checkpoint(
    "experiment_integration", diagnostics, checks_passed=all(gates.values()),
)
assert all(gates.values()), gates

with run.reporting(checkpoint=ring_checkpoint):
    if options.get("fail_reporting"):
        raise AttributeError("Intentional presentation failure after persistent experiment selection")
    plot = api.save_optics_lattice_plot(
        run.artifact_path("experiment_cell_optics.svg"),
        cell_view["linear_optics"]["twiss_profile"], selected_rows,
        title="Measured selected periodic cell: lattice and optics",
    )
    run.finish(
        summary="Saved-input experiment integration completed through the explicit ring; no fresh design is certified.",
        data={"experiment_integration": diagnostic_checkpoint, "gates": gates,
              "selected_experiment_candidate": selected},
        tables=[{"title": "Complete-ring integration checks", "rows": [
            {"Check": key, "Passed": value} for key, value in gates.items()]},
                {"title": "Per-bend radiation contributions", "rows": bend_evidence["rows"]}],
        plots=[plot], candidate_lattice=ring, candidate_evaluation=ring_evaluation,
        scientific_passed=all(gates.values()), request_satisfied=False, use_as_working_lattice=False,
        limitations=[
            "Historical-input integration only; this does not establish autonomous greenfield design.",
            "The optional nonzero quadrupole-pair removal is allowed to fail; its numerical record remains archived.",
            "Unit-reference and complete-ring emittance are distinct; nonlinear dynamics is not evaluated.",
        ],
    )
