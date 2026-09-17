"""Generic setup regression: no archived designs, JuTrack or physics search."""
from __future__ import annotations

import ast
import copy

import pytest

from brilliance import accelerator_api as api
from brilliance.population_optimization import OptimizerInputError
from brilliance.public_api_lint import lint_public_api_calls
from brilliance.research_recovery import coding_failure_summary, program_failure_guidance


def parameterizations(*, explicit=False):
    left = []
    for i in range(8):
        left.extend([
            {"name": f"D{i}", "kind": "DRIFT", "length_m": 0.3},
            {"name": f"Q{i}", "kind": "KQUAD", "length_m": 0.25, "k1_m2": 0.4},
        ])
    for i in range(6):
        left.append({"name": f"M{i}", "kind": "KQUAD", "length_m": 0.25,
                     "k1_m2": 0.0, "fields": {"family": f"M{i}"}})
    right = [dict(copy.deepcopy(row), name="R:" + row["name"]) for row in reversed(left)]
    rows = left + [{"name": "B", "kind": "SBEND", "length_m": 0.8, "angle_rad": 0.1}] + right
    drifts = [r["name"] for r in rows if r["kind"] == "DRIFT"]
    quads = [r["name"] for r in rows if r["kind"] == "KQUAD" and not r.get("fields")]
    options = ({"included_element_names_by_field": {"length_m": drifts, "k1_m2": quads}}
               if explicit else {"fixed_element_names": [r["name"] for r in rows
                                                          if r["name"] not in drifts + quads]})
    design = api.infer_mirror_symmetric_parameterization(
        rows, fields=("length_m", "k1_m2"), k1_bounds=(-2, 2),
        exact_sum_by_field={"length_m": sum(r["length_m"] for r in rows)}, **options,
    )
    matching = api.compile_family_parameterization(
        rows, field="k1_m2", families=[f"M{i}" for i in range(6)],
        bounds=(-2, 2), role="in_objective_matching",
    )
    return rows, design, matching


def test_actual_dimension_explains_archived_failure():
    _, design, matching = parameterizations()
    plan = api.plan_lattice_optimization(design, matching, population_size=240, generations=25)
    assert plan["outer_knobs"] == 24
    assert plan["outer_knobs_by_field"] == {"length_m": 16, "k1_m2": 8}
    assert plan["matching_knobs"] == 6
    assert plan["minimum_population"] == 240
    assert len(plan["knob_table"]) == 30
    assert plan["feasibility_checked"] is False
    with pytest.raises(OptimizerInputError, match="240 for 24 outer knobs"):
        api.plan_lattice_optimization(design, matching, population_size=160, generations=25)


def test_explicit_field_selection_preserves_geometry_and_ownership():
    rows, design, matching = parameterizations(explicit=True)
    plan = api.plan_lattice_optimization(design, matching, population_size=160, generations=25)
    assert plan["outer_knobs"] == 16
    assert plan["outer_knobs_by_field"] == {"length_m": 8, "k1_m2": 8}
    assert api.plan_lattice_optimization(design, matching)["population_size"] == 500
    position = [(k.lower + k.upper) / 2 for k in design.knobs]
    position[0] *= 1.1
    changed = design.apply(position)
    assert sum(r["length_m"] for r in changed) == pytest.approx(sum(r["length_m"] for r in rows))
    for old, new in zip(rows, changed):
        if old["kind"] == "KQUAD":
            assert old["length_m"] == new["length_m"]
        if old.get("fields"):
            assert old["k1_m2"] == new["k1_m2"]
    for a, b in zip(changed, reversed(changed)):
        assert a["length_m"] == pytest.approx(b["length_m"])
        assert a.get("k1_m2") == b.get("k1_m2")


@pytest.mark.parametrize("selection, message", [
    ({"length_m": ["D0"]}, "both global mirror partners"),
    ({"length_m": ["imaginary"]}, "unknown included"),
    ({"k1_m2": ["D0", "R:D0"]}, "do not support"),
    ({"angle_rad": ["B"]}, "must also appear"),
    ({"length_m": "D0"}, "not a string"),
])
def test_invalid_inclusion_fails_explicitly(selection, message):
    rows, _, _ = parameterizations()
    with pytest.raises(OptimizerInputError, match=message):
        api.infer_mirror_symmetric_parameterization(
            rows, fields=("length_m", "k1_m2"), k1_bounds=(-2, 2),
            included_element_names_by_field=selection,
        )


def test_conflicting_inclusions_cannot_silently_drop_knobs():
    rows, _, _ = parameterizations()
    with pytest.raises(OptimizerInputError, match="conflicts with fixed"):
        api.infer_mirror_symmetric_parameterization(
            rows, fields=("length_m",), fixed_element_names=["D0"],
            included_element_names_by_field={"length_m": ["D0", "R:D0"]},
        )


def test_exact_angle_sum_without_selected_bends_is_a_repairable_setup_error():
    rows, _, _ = parameterizations()
    with pytest.raises(OptimizerInputError, match="Exact sum fields have no selected variables"):
        api.infer_mirror_symmetric_parameterization(
            rows, fields=("length_m", "angle_rad"),
            included_element_names_by_field={"length_m": [r["name"] for r in rows if r["kind"] == "DRIFT"]},
            exact_sum_by_field={"length_m": sum(r["length_m"] for r in rows), "angle_rad": 0.1},
        )


def test_source_check_catches_missing_selected_field_without_reading_rows():
    issues = lint_public_api_calls(ast.parse('''
from brilliance import accelerator_api as api
design = api.infer_mirror_symmetric_parameterization(
    rows, fields=("length_m", "angle_rad"),
    included_element_names_by_field={"length_m": drift_names},
    exact_sum_by_field={"length_m": L_CELL, "angle_rad": THETA_CELL})
'''))
    assert len(issues) == 1
    assert "angle_rad" in issues[0]


def test_source_check_accepts_explicit_angle_and_length_selection():
    assert lint_public_api_calls(ast.parse('''
from brilliance import accelerator_api as api
design = api.infer_mirror_symmetric_parameterization(
    rows, fields=("length_m", "angle_rad"),
    included_element_names_by_field={"length_m": drift_names, "angle_rad": bend_names},
    exact_sum_by_field={"length_m": L_CELL, "angle_rad": THETA_CELL})
''')) == []


@pytest.mark.parametrize("options", [
    {"population_size": 159}, {"population_size": 160.5}, {"population_size": True},
    {"minimum_population_per_knob": float("nan")}, {"generations": 2},
    {"generations": 25.5}, {"minimum_production_generations": -1},
    {"parallel_workers": 0}, {"parallel_workers": -1}, {"parallel_workers": 2},
    {"feasible_basin_population_fraction": 2}, {"initialization_attempts_per_particle": 0},
    {"initial_feasible_sigma_fraction": 0}, {"inner_match_evaluation_budget": 1},
    {"local_polish_evaluations": -1}, {"local_polish_evaluations": 1},
    {"local_polish_initial_step_fraction": 0},
])
def test_invalid_budget_is_repairable_and_never_runs_a_callback(options):
    rows, design, matching = parameterizations(explicit=True)

    def no_physics(*args):
        pytest.fail("Setup errors must be caught before any physics callback")

    with pytest.raises(OptimizerInputError):
        api.optimize_lattice_parameters(
            rows, design, matching, no_physics, no_physics,
            match_targets={"Dx_m": (0, 1e-4)}, objective_names=("emittance_pm",),
            **options,
        )


def test_shared_field_ownership_is_repairable():
    _, design, _ = parameterizations()
    with pytest.raises(OptimizerInputError, match="cannot own the same lattice field"):
        api.plan_lattice_optimization(design, design)


def test_old_population_failure_gets_specific_repair_without_reclassifying_physics():
    error = "ValueError: population_size=160 is below the declared production minimum of 240 for 24 outer knobs"
    assert coding_failure_summary(error) == error
    guidance = program_failure_guidance(error)
    assert "not evidence" in guidance["explanation"]
    assert "included_element_names_by_field" in guidance["repair_hint"]
    assert coding_failure_summary("ValueError: no stable periodic solution") is None


def test_nested_optimizer_still_evaluates_rematched_rows(tmp_path):
    # Synthetic algebraic callbacks test optimizer integration, NOT accelerator physics.
    rows = [{"name": "Q", "kind": "KQUAD", "length_m": 0.2, "k1_m2": 0.5,
             "fields": {"family": "Q"}},
            {"name": "M", "kind": "KQUAD", "length_m": 0.2, "k1_m2": 0.05,
             "fields": {"family": "M"}}]
    design = api.compile_family_parameterization(rows, field="k1_m2", families=["Q"], bounds=(-2, 2))
    matching = api.compile_family_parameterization(rows, field="k1_m2", families=["M"], bounds=(-2, 2))
    evaluated = []

    def metrics(proposal):
        return {"stable": True, "Dx_m": proposal[1]["k1_m2"] - 0.1 * proposal[0]["k1_m2"]}

    def objectives(proposal):
        assert abs(metrics(proposal)["Dx_m"]) <= 1e-5
        evaluated.append(proposal)
        return {"objectives": {"loss": (proposal[0]["k1_m2"] - 0.3) ** 2}}

    result = api.optimize_lattice_parameters(
        rows, design, matching, metrics, objectives,
        match_targets={"Dx_m": (0, 1e-5)}, objective_names=("loss",),
        population_size=10, generations=2, minimum_production_generations=0,
        checkpoint_path=tmp_path / "checkpoint.json", seed=5,
    )
    assert evaluated
    assert result["selected_elements"] is not None
    assert result["optimization_plan"]["outer_knobs"] == 1
    assert result["convergence_diagnostic"]["minimum_production_generations"] == 0
