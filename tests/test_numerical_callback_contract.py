"""Callback faults are coding failures; rejected physics is an explicit outcome.

All calculations here are algebraic test doubles, never lattice-design evidence.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from brilliance import lattice_optimization as numerical
from brilliance.research_outcomes import ResearchInfeasible


def problem():
    rows = [
        {"name": "Q", "kind": "KQUAD", "length_m": 0.2, "k1_m2": 0.5},
        {"name": "M", "kind": "KQUAD", "length_m": 0.2, "k1_m2": 0.05},
    ]
    outer = numerical.compile_lattice_parameterization(rows, [
        {"name": "design", "field": "k1_m2", "element_names": ["Q"], "bounds": [-2, 2]}])
    inner = numerical.compile_lattice_parameterization(rows, [
        {"name": "match", "field": "k1_m2", "element_names": ["M"], "bounds": [-2, 2]}])
    return rows, outer, inner


def metrics(rows):
    return {"stable": True, "Dx_m": rows[1]["k1_m2"] - 0.1 * rows[0]["k1_m2"]}


def objective(rows):
    return {"objectives": {"loss": (rows[0]["k1_m2"] - 0.3) ** 2}}


def reject(_rows):
    raise ResearchInfeasible("Trial violates a declared physical constraint",
                             code="physical_test_limit", diagnostics={"limit": 1.0})


def optimize(metric=metrics, objectives=objective, **options):
    rows, outer, inner = problem()
    return numerical.optimize_lattice_parameters(
        rows, outer, inner, metric, objectives,
        match_targets={"Dx_m": (0.0, 1e-5)}, objective_names=("loss",),
        population_size=10, generations=2, minimum_production_generations=0,
        **options,
    )


@pytest.mark.parametrize("matcher", [numerical.bounded_response_match, numerical.bounded_fast_response_match])
@pytest.mark.parametrize("error", [TypeError, KeyError, ValueError, RuntimeError, AttributeError, OSError])
def test_matchers_propagate_callback_exceptions_unchanged(matcher, error):
    rows, _, inner = problem()
    original = error("callback defect, not instability")

    def broken(_rows):
        raise original

    with pytest.raises(error) as caught:
        matcher(rows, inner, broken, {"Dx_m": (0.0, 1e-5)})
    assert caught.value is original


@pytest.mark.parametrize("matcher", [numerical.bounded_response_match, numerical.bounded_fast_response_match])
@pytest.mark.parametrize("callback", [reject, lambda _: {"stable": False}, lambda _: {"feasible": False}])
def test_matchers_accept_explicit_physical_rejection(matcher, callback):
    rows, _, inner = problem()
    result = matcher(rows, inner, callback, {"Dx_m": (0.0, 1e-5)})
    assert result["success"] is False
    assert result["evaluations"] == 1
    if callback is reject:
        assert result["metrics"]["rejection_code"] == "physical_test_limit"


@pytest.mark.parametrize("matcher", [numerical.bounded_response_match, numerical.bounded_fast_response_match])
def test_physical_rejection_at_forward_difference_permits_backward_response(matcher):
    rows, _, inner = problem()

    def bounded_physics(proposal):
        if proposal[1]["k1_m2"] > 0.05:
            return reject(proposal)
        return metrics(proposal)

    result = matcher(rows, inner, bounded_physics, {"Dx_m": (-0.01, 1e-5)})
    assert result["success"] is True
    assert abs(result["metrics"]["Dx_m"] + 0.01) <= 1e-5


def test_only_declared_exact_sum_projection_is_translated_to_physical_rejection():
    class Projection:
        def apply(self, *args, **kwargs):
            raise ValueError("exact sum projection moves at least one lattice knob outside its bounds")

    with pytest.raises(ResearchInfeasible) as caught:
        numerical._apply_design_proposal(Projection(), [0.5], [])
    assert caught.value.code == "exact_sum_projection_outside_bounds"

    class Broken:
        def apply(self, *args, **kwargs):
            raise ValueError("unknown or malformed lattice field")

    with pytest.raises(ValueError) as caught:
        numerical._apply_design_proposal(Broken(), [0.5], [])
    assert not isinstance(caught.value, ResearchInfeasible)


@pytest.mark.parametrize("callback,error", [
    (lambda _: [], TypeError),
    (lambda _: {"stable": "false", "Dx_m": 0.0}, TypeError),
    (lambda _: {"stable": True}, KeyError),
    (lambda _: {"Dx_m": [0.0]}, TypeError),
    (lambda _: {"Dx_m": np.array([0.0])}, TypeError),
    (lambda _: {"Dx_m": "0.0"}, TypeError),
    (lambda _: {"Dx_m": float("nan")}, ValueError),
])
def test_required_matching_observables_have_actual_scalar_schema(callback, error):
    rows, _, inner = problem()
    with pytest.raises(error):
        numerical.bounded_response_match(rows, inner, callback, {"Dx_m": (0.0, 1e-5)})


def test_metric_names_and_shape_remain_checked_after_a_successful_first_call():
    rows, _, inner = problem()
    calls = []

    def later_broken(proposal):
        calls.append(1)
        return {"stable": True, "Dx_m": 1.0} if len(calls) == 1 else {"stable": True, "Dx": 1.0}

    with pytest.raises(KeyError, match="Dx_m"):
        numerical.bounded_fast_response_match(rows, inner, later_broken, {"Dx_m": (0.0, 1e-5)})
    assert len(calls) == 2


@pytest.mark.parametrize("error", [TypeError, KeyError, ValueError, RuntimeError, OSError])
def test_seed_screening_propagates_fault_on_first_attempt(error):
    rows, outer, _ = problem()
    calls = []

    def broken(_rows):
        calls.append(1)
        raise error("bad seed evaluator")

    with pytest.raises(error):
        numerical.search_feasible_parameter_seeds(rows, outer, broken, count=1, attempts=50)
    assert len(calls) == 1


@pytest.mark.parametrize("callback", [reject, lambda _: {"stable": False}, lambda _: {"feasible": False}])
def test_seed_screening_keeps_explicit_physics_rejections_normal(callback):
    rows, outer, _ = problem()
    result = numerical.search_feasible_parameter_seeds(rows, outer, callback, count=1, attempts=3)
    assert result["success"] is False
    assert result["attempts"] == 3


def test_seed_screening_can_use_general_feasible_flag():
    rows, outer, _ = problem()
    result = numerical.search_feasible_parameter_seeds(
        rows, outer, lambda _: {"feasible": True, "observable": 1.0}, count=1, attempts=2)
    assert result["success"] is True


@pytest.mark.parametrize("callback,error", [
    (lambda _: {"Dx_m": 0.0}, KeyError),
    (lambda _: {"stable": True, "Dx_m": [0.0]}, TypeError),
    (lambda _: {"stable": True, "Dx_m": float("inf")}, ValueError),
])
def test_seed_screening_validates_flags_and_target_shapes(callback, error):
    rows, outer, _ = problem()
    with pytest.raises(error):
        numerical.search_feasible_parameter_seeds(rows, outer, callback, count=1, attempts=2,
                                                  targets={"Dx_m": (0.0, 1e-5)})


@pytest.mark.parametrize("returned,error", [
    (None, TypeError),
    ([], TypeError),
    ({"objectives": [1.0]}, TypeError),
    ({"objectives": {"wrong_name": 1.0}}, KeyError),
    ({"objectives": {"loss": [1.0]}}, TypeError),
    ({"objectives": {"loss": np.array([1.0])}}, TypeError),
    ({"objectives": {"loss": "1.0"}}, TypeError),
    ({"objectives": {"loss": float("inf")}}, ValueError),
    ({"objectives": {"loss": 1.0}, "feasible": "yes"}, TypeError),
    ({"objectives": {"loss": 1.0}, "constraint_violations": [0]}, TypeError),
    ({"objectives": {"loss": 1.0}, "constraint_violations": {"x": float("nan")}}, ValueError),
])
def test_objective_schema_fails_before_screening_or_population(monkeypatch, returned, error):
    calls = []

    def should_not_start(*args, **kwargs):
        pytest.fail("Malformed baseline objective must not start population optimization")

    monkeypatch.setattr(numerical, "particle_swarm_optimize", should_not_start)

    def malformed(_rows):
        calls.append(1)
        return returned

    with pytest.raises(error):
        optimize(objectives=malformed)
    assert len(calls) == 1


@pytest.mark.parametrize("callback", [reject, lambda _: {"feasible": False},
                                      lambda _: {"constraint_violations": {"limit": 1.0}}])
def test_baseline_objective_can_reject_physics_without_a_fictitious_objective(callback):
    with pytest.raises(ResearchInfeasible) as caught:
        optimize(objectives=callback)
    assert caught.value.code == "initial_objective_not_feasible"
    assert caught.value.diagnostics["initial_evaluation"]["feasible"] is False


@pytest.mark.parametrize("error", [TypeError, KeyError, ValueError, RuntimeError, OSError])
def test_nested_serial_objective_fault_is_not_masked_during_screening(error):
    calls = []

    def later_fault(rows):
        calls.append(1)
        if len(calls) > 1:
            raise error("objective failed during seed screening")
        return objective(rows)

    with pytest.raises(error, match="objective failed"):
        optimize(objectives=later_fault)
    assert len(calls) == 2


def worker(callbacks):
    rows, outer, inner = problem()
    return numerical._LatticeWorkerEvaluatorFactory(
        physics_factory=lambda _rows: callbacks, baseline_elements=rows,
        design_parameterization=outer, matching_parameterization=inner,
        match_targets={"Dx_m": (0.0, 1e-5)}, objective_names=("loss",),
        baseline_matching_position=tuple(inner.initial_position), max_match_iterations=4,
        inner_match_evaluation_budget=32,
    )()


@pytest.mark.parametrize("fault_in", ["metrics", "objectives"])
@pytest.mark.parametrize("error", [TypeError, KeyError, RuntimeError, ValueError])
def test_process_local_nested_evaluator_propagates_faults(fault_in, error):
    def broken(_rows):
        raise error("worker callback error")

    callbacks = (broken, objective) if fault_in == "metrics" else (metrics, broken)
    evaluate = worker(callbacks)
    with pytest.raises(error, match="worker callback error"):
        evaluate(np.array([0.5]))


def test_process_local_evaluator_retains_exact_matched_rows():
    seen = []

    def record(rows):
        seen.append(copy.deepcopy(rows))
        return objective(rows)

    result = worker((metrics, record))(np.array([0.9]))
    assert result["matching"]["elements"] == seen[-1]
    assert abs(metrics(result["matching"]["elements"])["Dx_m"]) < 1e-5
    seen[-1][0]["k1_m2"] = 999
    assert result["matching"]["elements"][0]["k1_m2"] == 0.9


def test_objective_callback_cannot_mutate_evaluated_rows_behind_the_matcher():
    def mutating(rows):
        rows[1]["k1_m2"] = 1.5
        return objective(rows)

    with pytest.raises(ValueError, match="changed physical lattice rows"):
        optimize(objectives=mutating)


def test_metrics_callback_cannot_change_coordinates_and_preserves_parent_on_error():
    rows, _, inner = problem()
    original = copy.deepcopy(rows)

    def mutating(proposal):
        proposal[1]["k1_m2"] = 1.5
        return metrics(proposal)

    with pytest.raises(ValueError, match="evaluate_metrics changed physical lattice rows"):
        numerical.bounded_response_match(rows, inner, mutating, {"Dx_m": (0.0, 1e-5)})
    assert rows == original


def test_selected_rows_and_matching_receipt_are_exact_and_survive_resume(tmp_path):
    seen = []

    def record(rows):
        seen.append(copy.deepcopy(rows))
        return objective(rows)

    checkpoint = tmp_path / "algebraic-checkpoint.json"
    first = optimize(objectives=record, checkpoint_path=checkpoint, seed=7)
    assert first["selected_elements"] in seen
    assert first["selected_elements"] == first["selected_matching"]["elements"]
    resumed = optimize(objectives=record, checkpoint_path=checkpoint, seed=7, resume=True)
    assert resumed["selected_elements"] == first["selected_elements"]


def test_linear_cache_key_never_rounds_distinct_float_positions_together():
    rows, _, _ = problem()
    changed = copy.deepcopy(rows)
    changed[0]["k1_m2"] = np.nextafter(changed[0]["k1_m2"], 1.0)
    assert numerical._physical_rows_key(rows) != numerical._physical_rows_key(changed)


def native_double(**overrides):
    matrix = np.eye(6)
    matrix[:4, :4] = [[0., 1., 0., 0.], [-1., 0., 0., 0.],
                      [0., 0., 0., 1.], [0., 0., -1., 0.]]
    values = dict(findm66=lambda *a, **k: matrix,
        periodicEdwardsTengTwiss=lambda *a, **k: SimpleNamespace(mode=1, dx=0., dpx=0., dy=0., dpy=0., betax=1., betay=1., alphax=0., alphay=0.),
        gettune=lambda *a, **k: [0.25, 0.25])
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("operation", ["findm66", "periodicEdwardsTengTwiss", "gettune"])
@pytest.mark.parametrize("error", [TypeError, KeyError, ValueError, RuntimeError, OSError])
def test_native_optics_bridge_never_masks_program_or_runtime_fault(operation, error):
    def broken(*args, **kwargs):
        raise error("native bridge failure")

    with pytest.raises(error, match="native bridge failure"):
        numerical.jutrack_linear_metrics(native_double(**{operation: broken}), [])


def test_native_invalid_mode_is_explicit_physics_without_calling_gettune():
    jt = native_double(periodicEdwardsTengTwiss=lambda *a, **k: SimpleNamespace(mode=0),
        gettune=lambda *a, **k: pytest.fail("Invalid mode must be rejected before tune calculation"))
    result = numerical.jutrack_linear_metrics(jt, [])
    assert result["stable"] is False
    assert "mode=0" in result["reason"]


def test_native_known_twiss_domain_error_is_narrowly_classified():
    JuliaError = type("JuliaError", (Exception,), {"__module__": "juliacall"})

    def physical(*args, **kwargs):
        raise JuliaError("DomainError with -1: negative sqrt argument in twissPropagate(mode=1)")

    result = numerical.jutrack_linear_metrics(native_double(periodicEdwardsTengTwiss=physical), [])
    assert result["stable"] is False

    def method_error(*args, **kwargs):
        raise JuliaError("MethodError: no method matching twissPropagate")

    with pytest.raises(JuliaError, match="MethodError"):
        numerical.jutrack_linear_metrics(native_double(periodicEdwardsTengTwiss=method_error), [])


def test_native_wrong_profile_length_is_not_a_physics_rejection():
    jt = native_double(twissline=lambda *a, **k: [])
    with pytest.raises(ValueError, match="does not cover every element exit"):
        numerical.jutrack_linear_metrics(jt, [SimpleNamespace(name="D", len=1.)], include_profile=True)
