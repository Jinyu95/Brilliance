"""Exact bounded matching starts; optional native reachability, not a machine seed."""
from __future__ import annotations

import ast
import copy
import math
import os

import pytest

from brilliance import accelerator_api as api


def parameters(initial=.7, bounds=(-2., 2.)):
    rows = [api.quadrupole("Q", .2, initial)]
    return api.compile_lattice_parameterization(rows, [
        {"name": "Q strength", "field": "k1_m2", "element_names": ["Q"], "bounds": list(bounds)}])


def plan(knobs=None, **kwargs):
    return api.plan_matching_starts(knobs or parameters(), limits=api.magnet_strength_limits(3.5), **kwargs)


def test_eight_explicit_starts_fit_eight_start_budget_without_extra_baseline_error():
    starts = [[-.8], [-.6], [-.4], [-.2], [.0], [.2], [.4], [.6]]
    before = copy.deepcopy(starts)
    schedule = plan(start_positions=starts, max_starts=8)
    assert schedule["positions"] == before == starts
    assert schedule["sources"] == ["supplied"] * 8
    assert schedule["deferred_positions"] == [[.7]]
    assert schedule["deferred_sources"] == ["baseline"]
    assert schedule["budget"]["total_starts"] == 8
    assert schedule["budget"]["supplied_count"] == 8
    assert schedule["budget"]["deferred_count"] == 1
    assert schedule["budget"]["baseline_included"] is False
    assert schedule["budget"]["generated_count"] == 0


def test_maximum_one_honors_supplied_warm_start_and_preserves_exact_baseline():
    warm = .12345678901234567
    schedule = plan(start_positions=[[warm]], max_starts=1)
    assert schedule["positions"] == [[warm]]
    assert schedule["sources"] == ["supplied"]
    assert schedule["deferred_positions"] == [[.7]]
    assert schedule["budget"]["requested_max_starts"] == 1
    assert schedule["budget"]["total_starts"] == 1
    assert plan(max_starts=1)["positions"] == [[.7]]


def test_duplicate_supplied_baseline_is_one_start_not_a_budget_failure():
    schedule = plan(start_positions=[[.7], [.7], [.7]], max_starts=1)
    assert schedule["positions"] == [[.7]]
    assert schedule["sources"] == ["supplied"]
    assert schedule["budget"]["duplicate_supplied_count"] == 2
    assert schedule["budget"]["baseline_included"] is True
    assert schedule["supplied_indices"] == [[0, 1, 2]]
    assert schedule["deferred_positions"] == []


def test_overflow_is_an_ordered_recoverable_queue_not_silently_discarded():
    supplied = [[-.5], [.2], [.3], [.4]]
    schedule = plan(start_positions=supplied, max_starts=2)
    assert schedule["positions"] == supplied[:2]
    assert schedule["deferred_positions"] == supplied[2:] + [[.7]]
    assert schedule["deferred_sources"] == ["supplied", "supplied", "baseline"]
    assert schedule["budget"]["deferred_count"] == 3
    assert schedule["positions"] + schedule["deferred_positions"] == supplied + [[.7]]


def test_deduplication_does_not_round_nearby_but_distinct_physical_starts():
    near = math.nextafter(.7, 1.)
    schedule = plan(start_positions=[[.7], [near]], max_starts=2)
    assert schedule["positions"] == [[.7], [near]]
    assert schedule["budget"]["duplicate_supplied_count"] == 0
    assert schedule["deferred_positions"] == []


def test_default_budget_keeps_all_supplied_starts_when_they_exceed_default():
    supplied = [[-.9 + i / 100.] for i in range(70)]
    schedule = plan(start_positions=supplied)
    assert schedule["positions"] == supplied + [[.7]]
    assert schedule["budget"]["total_starts"] == 71
    assert schedule["budget"]["deferred_count"] == 0
    assert schedule["budget"]["generated_count"] == 0


def test_generated_starts_are_reproducible_inside_engineering_bounds():
    knobs = parameters(bounds=(-10., 10.))
    limits = api.magnet_strength_limits(3.5)
    first = api.plan_matching_starts(knobs, limits=limits, max_starts=9, seed=41)
    second = api.plan_matching_starts(knobs, limits=limits, max_starts=9, seed=41)
    assert first == second
    assert first["positions"][0] == [.7]
    assert first["sources"] == ["baseline"] + ["generated"] * 8
    assert first["budget"]["generated_count"] == 8
    bound = limits["max_abs_k1_m2"]
    assert all(-bound <= row[0] <= bound for row in first["positions"])


@pytest.mark.parametrize("options", [{"max_starts": 0}, {"max_starts": True}, {"max_starts": 1.5},
                                     {"seed": -1}, {"seed": True}, {"start_positions": [[math.nan]]},
                                     {"start_positions": [[math.inf]]}, {"start_positions": [[.1, .2]]},
                                     {"start_positions": [[2.1]]}])
def test_invalid_inputs_fail_preflight_without_loading_jutrack(monkeypatch, options):
    monkeypatch.setattr(api, "jutrack", lambda: pytest.fail("Start planning must not load JuTrack"))
    with pytest.raises(ValueError):
        plan(**options)


def test_engineering_out_of_bounds_seed_is_rejected_not_clipped():
    limits = api.magnet_strength_limits(3.5)
    cap = limits["max_abs_k1_m2"]
    with pytest.raises(ValueError):
        api.plan_matching_starts(parameters(bounds=(-20., 20.)), limits=limits,
                                 start_positions=[[cap + .01]], max_starts=1)


def test_invalid_automatic_baseline_is_explicitly_skipped_not_changed_to_fit():
    knobs = parameters(initial=5., bounds=(-20., 20.))
    schedule = plan(knobs, start_positions=[[.2]], max_starts=1)
    assert schedule["positions"] == [[.2]]
    assert schedule["skipped"] == [{"source": "baseline", "position": [5.],
                                    "reason": "baseline_outside_engineering_bounds"}]
    assert schedule["budget"]["baseline_included"] is False


def test_even_a_deferred_input_is_validated_before_any_work_is_scheduled():
    with pytest.raises(ValueError, match=r"start_positions\[1\]"):
        plan(start_positions=[[.2], [20.]], max_starts=1)


def test_planner_help_and_nested_result_contract_are_available_without_physics(monkeypatch):
    from brilliance import line_matching
    from brilliance.native_research_agent import api_help
    from brilliance.public_api_lint import lint_public_api_calls
    from brilliance.result_contracts import return_contract
    monkeypatch.setattr(line_matching, "JuTrackLineEvaluator", lambda *a, **k: pytest.fail("Planning/help must not evaluate physics"))
    help_text = api_help("", "plan_matching_starts", max_chars=12000)
    assert "api.plan_matching_starts(" in help_text
    assert "deferred_positions" in help_text and "supplied_indices" in help_text
    assert "requested_max_starts" in help_text and "duplicate_supplied_count" in help_text
    actual = plan(start_positions=[[.2]], max_starts=1)
    schema = return_contract(api.plan_matching_starts)
    assert schema["closed"]
    assert set(schema["properties"]) == set(actual)
    assert schema["properties"]["budget"]["closed"]
    assert set(schema["properties"]["budget"]["properties"]) == set(actual["budget"])
    match_schema = return_contract(api.match_jutrack_line)
    assert match_schema["properties"]["start_plan"] == schema
    source = "from brilliance import accelerator_api as api\nplan = api.plan_matching_starts(knobs, limits=limits)\n"
    assert lint_public_api_calls(ast.parse(source + "count = plan['budget']['total_starts']")) == []
    issues = lint_public_api_calls(ast.parse(source + "count = plan['budget']['made_up_count']"))
    assert len(issues) == 1 and "made_up_count" in issues[0]


@pytest.mark.parametrize("supplied,budget", [([[.7], [-.8], [-.6], [-.4], [-.2], [0.], [.2], [.4]], 8),
                                          ([[.2]], 1)])
def test_match_executes_bounded_plan_and_preserves_unattempted_starts(monkeypatch, supplied, budget):
    """Mock transport checks scheduling only, never a physical matching claim."""
    from brilliance import line_matching
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    evaluated = []
    initial = dict(beta_x_m=2., beta_y_m=3., alpha_x=0., alpha_y=0.,
                   Dx_m=.01, Dpx=0., Dy_m=0., Dpy=0.)
    class FakeLine:
        def __init__(self, elements, **configuration):
            self.rows = copy.deepcopy(elements)
        def evaluate(self, elements=None):
            evaluated.append(copy.deepcopy(self.rows if elements is None else elements))
            return {"evaluable": True, "passed": True, "endpoint": {**initial, "Dx_m": 0.},
                    "metrics": {"maximum_beta_m": 3.}}
    monkeypatch.setattr(line_matching, "JuTrackLineEvaluator", FakeLine)
    knobs = parameters()
    result = api.match_jutrack_line(knobs.baseline, knobs, energy_gev=3.5,
        initial_twiss=initial, targets={"Dx_m": (0., 1e-5)}, expected_length_m=.2,
        expected_angle_rad=0., limits=api.magnet_strength_limits(3.5), maximum_beta_m=100.,
        start_positions=supplied, max_starts=budget, max_nfev=3)
    assert result["match_success"]
    assert len(result["attempts"]) == 1
    assert evaluated[0][0]["k1_m2"] == pytest.approx(supplied[0][0], abs=1e-14)
    assert result["start_plan"]["positions"] == supplied
    assert result["start_plan"]["positions"][len(result["attempts"]):] == supplied[1:]
    assert result["start_plan"]["budget"]["total_starts"] == budget
    assert result["completed_starts"] == 1
    assert result["remaining_start_positions"] == supplied[1:] + result["start_plan"]["deferred_positions"]
    assert not result["complete_cell_accepted"]


@pytest.mark.skipif(os.environ.get("BRILLIANCE_RUN_JUTRACK_TESTS") != "1",
                    reason="Opt-in configured native JuTrack matching")
def test_single_supplied_native_solution_is_executed_not_blocked_by_added_baseline(monkeypatch):
    """A supplied exact physical target is the sole start; no broad search."""
    from brilliance.research_environment import ResearchEnvironment
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="native single-start schedule regression")
    for key, value in environment.process_environment().items():
        monkeypatch.setenv(key, value)
    known = [api.drift("D0", .2), api.quadrupole("QF", .2, 1.2), api.drift("D1", .3),
             api.sector_bend("B", .8, .02), api.drift("D2", .2),
             api.quadrupole("QD", .2, -1.), api.drift("D3", .25),
             api.quadrupole("QF2", .2, .6), api.drift("D4", .3)]
    initial = dict(beta_x_m=3., beta_y_m=4., alpha_x=.3, alpha_y=-.2,
                   Dx_m=.01, Dpx=-.002, Dy_m=0., Dpy=0.)
    configuration = dict(energy_gev=3.5, initial_twiss=initial, maximum_beta_m=100.,
                         limits=api.magnet_strength_limits(3.5),
                         expected_length_m=sum(row["length_m"] for row in known),
                         expected_angle_rad=.02, profile_step_m=.5)
    measured = api.evaluate_jutrack_line(known, **configuration)
    assert measured["passed"]
    proposed = copy.deepcopy(known)
    for row in proposed:
        if row["name"] in {"QF", "QD", "QF2"}:
            row["k1_m2"] *= .8
    knobs = api.compile_lattice_parameterization(proposed, [
        {"name": name, "field": "k1_m2", "element_names": [name], "bounds": [-3., 3.]}
        for name in ("QF", "QD", "QF2")])
    tolerances = {"beta_x_m": 1e-4, "beta_y_m": 1e-4, "alpha_x": 1e-5,
                  "alpha_y": 1e-5, "Dx_m": 1e-7, "Dpx": 1e-7}
    targets = {key: (measured["endpoint"][key], tolerance) for key, tolerance in tolerances.items()}
    supplied = [[1.2, -1., .6]]
    matched = api.match_jutrack_line(proposed, knobs, targets=targets, start_positions=supplied,
                                    max_starts=1, max_nfev=12, **configuration)
    assert matched["match_success"], matched
    assert len(matched["attempts"]) == 1
    assert not matched["complete_cell_accepted"]
    replay = api.evaluate_jutrack_line(matched["elements"], **configuration)
    assert replay["passed"]
    for key, (target, tolerance) in targets.items():
        assert replay["endpoint"][key] == pytest.approx(target, abs=tolerance)
    assert matched["measurement"]["endpoint"] == pytest.approx(replay["endpoint"], rel=1e-10, abs=1e-12)
