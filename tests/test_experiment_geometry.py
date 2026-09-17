"""General length budgets/topology edits need no simulator or task template."""
import copy
import math
from dataclasses import replace

import pytest

from brilliance import accelerator_api as api
from brilliance.experiment_geometry import (
    prepare_length_budget,
    project_weighted_sum,
    remove_mirrored_quadrupole_pair,
)
from brilliance.lattice import LatticeInputError
from brilliance.lattice_optimization import LinearSumConstraint, compile_lattice_parameterization
from brilliance.population_optimization import OptimizerInputError


@pytest.fixture(autouse=True)
def no_simulator(monkeypatch):
    monkeypatch.setattr(api, "load_pyjutrack", lambda: pytest.fail("Geometry helper ran a simulator"))


def rows():
    return api.symmetric_cell_elements(
        [api.drift("D0", .4), api.quadrupole("Q", .2, 1.23456789), api.drift("D1", .4)],
        center=api.sector_bend("B", .8, .1))


def parameterization(*, sum_constraints=()):
    return compile_lattice_parameterization(rows(), [
        {"name": "drift_lengths", "field": "length_m", "element_names": ["D0", "D1", "R:D1", "R:D0"],
         "bounds": [.1, 1.]},
        {"name": "quad_strength", "field": "k1_m2", "element_names": ["Q", "R:Q"], "bounds": [-3., 3.]},
    ], sum_constraints=sum_constraints)


def test_feasible_projection_does_not_reject_when_proportional_scaling_would():
    # Scaling [2,5] to sum6 violates y>=4; bounded projection is [1.5,4.5].
    assert project_weighted_sum([2., 5.], [1., 4.], [2., 5.], [1., 1.], 6.) == pytest.approx([1.5, 4.5])


@pytest.mark.parametrize("target", [1.6, 3.8, 5.2])
def test_budget_subtracts_all_fixed_rows_and_counts_multiple_mirror_pairs(target):
    original = parameterization()
    before = copy.deepcopy(original)
    result = prepare_length_budget(original, total_length_m=target)
    prepared = result["parameterization"]
    assert original == before
    assert isinstance(prepared, type(original))
    assert result["budget"]["fixed_length_m"] == pytest.approx(1.2)  # Fixed bend plus two strength-only quads.
    assert result["budget"]["free_length_target_m"] == pytest.approx(target - 1.2)
    assert result["budget"]["free_variables"][0]["multiplicity"] == 4
    assert result["budget"]["fixed_element_names"] == ["Q", "B", "R:Q"]
    assert prepared.initial_position == result["initial_position"]
    assert result["physics_verified"] is False
    for trial in ([.1, -2.], [.45, 0.], [1., 2.9]):
        candidate = prepared.apply(trial)
        assert math.fsum(row["length_m"] for row in candidate) == pytest.approx(target, abs=1e-12)
        assert candidate[1]["length_m"] == .2
        assert candidate[1]["k1_m2"] == trial[1]


def test_two_variable_length_groups_are_projected_at_every_proposal():
    original = compile_lattice_parameterization(rows(), [
        {"name": "outer", "field": "length_m", "element_names": ["D0", "R:D0"], "bounds": [.1, .5]},
        {"name": "inner", "field": "length_m", "element_names": ["D1", "R:D1"], "bounds": [.3, .9]},
    ])
    prepared = prepare_length_budget(original, total_length_m=3.2)["parameterization"]
    for point in ([.1, .3], [.5, .3], [.1, .9], [.5, .9]):
        projected = prepared.projected_position(point)
        assert .1 <= projected[0] <= .5 and .3 <= projected[1] <= .9
        assert math.fsum(row["length_m"] for row in prepared.apply(point)) == pytest.approx(3.2, abs=1e-12)


def test_other_field_sum_and_full_precision_strengths_survive():
    original = compile_lattice_parameterization(rows(), [
        {"name": "drifts", "field": "length_m", "element_names": ["D0", "R:D0"], "bounds": [.1, 1.]},
        {"name": "bend", "field": "angle_rad", "element_names": ["B"], "bounds": [.05, .2]},
    ], sum_constraints=[{"field": "angle_rad", "target": .1, "variable_names": ["bend"]}])
    result = prepare_length_budget(original, total_length_m=3.0)
    candidate = result["parameterization"].apply(result["initial_position"])
    assert candidate[1]["k1_m2"] == 1.23456789
    assert candidate[3]["angle_rad"] == .1
    assert result["parameterization"].sum_constraints[0] == original.sum_constraints[0]


@pytest.mark.parametrize("total", [1., 6.])
def test_impossible_budget_is_rejected_before_any_callback(total):
    with pytest.raises(OptimizerInputError, match="outside attainable"):
        prepare_length_budget(parameterization(), total_length_m=total)


def test_all_lengths_fixed_are_audited_without_inventing_a_length_knob():
    original = compile_lattice_parameterization(rows(), [{"name": "Q", "field": "k1_m2",
        "element_names": ["Q", "R:Q"], "bounds": [-3., 3.]}])
    report = prepare_length_budget(original, total_length_m=2.8)
    assert report["budget"]["free_variables"] == []
    assert report["parameterization"].variable_names == ["Q"]
    with pytest.raises(OptimizerInputError, match="all lengths are fixed"):
        prepare_length_budget(original, total_length_m=3.)


def test_existing_equivalent_budget_is_replaced_not_stacked():
    original = parameterization(sum_constraints=[{"field": "length_m", "target": 2.6,
                                                  "variable_names": ["drift_lengths"]}])
    prepared = prepare_length_budget(original, total_length_m=3.8)["parameterization"]
    assert len(prepared.sum_constraints) == 1
    assert prepared.sum_constraints[0].mode == "bounded_sum"
    with pytest.raises(OptimizerInputError, match="conflicts"):
        prepare_length_budget(original, total_length_m=3.)


def test_duplicate_sum_or_overlapping_physical_targets_are_not_silently_applied():
    original = parameterization()
    constraint = LinearSumConstraint("length_m", 1.6, ("drift_lengths",))
    with pytest.raises(OptimizerInputError, match="overlapping"):
        prepare_length_budget(replace(original, sum_constraints=(constraint, constraint)), total_length_m=2.8)
    repeated = replace(original.knobs[0], name="duplicate")
    with pytest.raises(OptimizerInputError, match="overlapping physical"):
        prepare_length_budget(replace(original, knobs=(*original.knobs, repeated)), total_length_m=2.8)


def test_named_pair_removal_preserves_span_names_bends_and_parent():
    original = rows()
    original[1]["fields"]["mirror_source_element"] = "obsolete-local-pair"
    original[-2]["fields"]["mirror_source_element"] = "another-obsolete-pair"
    before = copy.deepcopy(original)
    result = remove_mirrored_quadrupole_pair(original, element_names=["Q", "R:Q"])
    assert original == before
    assert result["requires_native_rebuild"] is True and result["physics_verified"] is False
    assert result["invariants"]["length_preserved"] and result["invariants"]["bend_angle_preserved"]
    assert result["invariants"]["global_physical_mirror_checked"]
    for i, (old, new) in enumerate(zip(original, result["elements"])):
        assert new["name"] == old["name"] and new["length_m"] == old["length_m"]
        if i in (1, 5):
            assert new["kind"] == "DRIFT" and new["k1_m2"] is None
            assert new["fields"]["replaced_quadrupole"] == old
        else:
            assert new == old


def test_pair_removal_retains_passive_apertures_but_not_magnet_integration_options():
    original = rows()
    aperture = [-.02, .02, -.02, .02, -1., 1.]
    for index in (1, 5):
        original[index]["fields"].update(RApertures=aperture, NumIntSteps=8)
    result = remove_mirrored_quadrupole_pair(original, element_names=["Q", "R:Q"])
    for index in (1, 5):
        assert result["elements"][index]["fields"]["RApertures"] == aperture
        assert "NumIntSteps" not in result["elements"][index]["fields"]


@pytest.mark.parametrize("names", [["Q"], ["Q", "Q"], ["Q", "missing"], ["D0", "R:D0"], ["Q", "B"], "Q"])
def test_invalid_or_nonglobal_named_pair_is_rejected(names):
    with pytest.raises(LatticeInputError):
        remove_mirrored_quadrupole_pair(rows(), element_names=names)


def test_global_mirror_checks_unselected_geometry_and_native_settings():
    for field, value in (("length_m", .6), ("k1_m2", 2.)):
        original = rows()
        original[0 if field == "length_m" else 1][field] = value
        with pytest.raises(LatticeInputError, match="mirror mismatch"):
            remove_mirrored_quadrupole_pair(original, element_names=["Q", "R:Q"])
    original = rows()
    original[1]["fields"]["NumIntSteps"] = 7
    original[-2]["fields"]["NumIntSteps"] = 9
    with pytest.raises(LatticeInputError, match="native-field mirror mismatch"):
        remove_mirrored_quadrupole_pair(original, element_names=["Q", "R:Q"])
