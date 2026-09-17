"""Execution contracts; toy callbacks test plumbing, never accelerator physics."""
import copy
import json

import pytest

from brilliance import accelerator_api as api
from brilliance.experiments import ExperimentDefinition, LatticeExperiment, MatchingOperation
from brilliance.research_outcomes import ResearchInfeasible


def rows():
    return [api.quadrupole("left", .2, 1.), api.drift("space", 1.),
            api.quadrupole("right", .2, 1.)]


def definition(**kwargs):
    return ExperimentDefinition("contract test", "synthetic plumbing only", {"cost": "min"},
                                {"stable": {"equals": True},
                                 "residual": {"lower": -.1, "upper": .1, "scale": .1}}, **kwargs)


def parameters(field="k1_m2"):
    name = "space" if field == "length_m" else "left"
    return api.compile_lattice_parameterization(rows(), [
        {"name": "knob", "field": field, "element_names": [name], "bounds": [.1, 2.]}])


def test_objective_is_not_acceptance_and_missing_is_not_zero():
    d = definition()
    assert all(c["passed"] for c in d.assess({"cost": 1e12, "stable": True, "residual": 0}).values())
    for missing in ({"cost": None, "stable": True, "residual": 0},
                    {"cost": 1, "stable": "True", "residual": 0},
                    {"cost": 1, "stable": True},
                    {"cost": float("nan"), "stable": True, "residual": 0}):
        assert not all(c["passed"] for c in d.assess(missing).values())
    assert d.assess({"cost": 1, "stable": True, "residual": .2})["residual"]["violation"] == pytest.approx(1)


@pytest.mark.parametrize("rule", [{"lower": 0}, {"equals": 0}, {"upper": 1, "scale": 0},
                                  {"lower": 2, "upper": 1, "scale": 1}, {"target": 0},
                                  {"equals": True, "lower": 0}])
def test_bad_constraint_rejected_before_worker(rule):
    with pytest.raises(ValueError):
        ExperimentDefinition("x", "test", {"f": "min"}, {"x": rule})


def test_late_feasible_candidate_retained_before_selection_and_reporting(tmp_path):
    def measure(r):
        x = r[0]["k1_m2"]
        return {"cost": x, "stable": True, "residual": 0 if x > 1 else .2}

    ex = LatticeExperiment(definition(), parameters(), measure=measure, directory=tmp_path / "study", workspace=tmp_path)
    bad = ex.evaluate([.2])
    stale = ex.ledger.entries()
    good = ex.evaluate([1.1])
    assert not bad["feasible"] and good["feasible"]
    selected = ex.ledger.select()["selected"]
    assert selected["candidate_id"] == good["candidate_id"]
    assert len(stale) == 1  # A stale user list cannot control framework selection.
    assert selected["candidate"]["elements"][0]["k1_m2"] == 1.1
    with pytest.raises(RuntimeError):
        raise RuntimeError("optional reporting failed")
    restored = LatticeExperiment(definition(), parameters(), measure=measure, directory=tmp_path / "study", workspace=tmp_path)
    assert restored.ledger.select()["selected"]["candidate_id"] == good["candidate_id"]
    table = restored.comparison_table(limit=1)
    assert len(table["rows"]) == 1 and table["rows"][0]["Selected"]


def test_exclusive_inner_matching_and_measured_returned_rows(tmp_path):
    outer = parameters("length_m")
    inner = parameters()

    def solve(r, p):
        assert r[1]["length_m"] == 1.5  # exact outer proposal reaches the matcher
        return {"match_success": True, "solver_success": True, "elements": p.apply([.8], base_elements=r)}

    def measure(r):
        return {"cost": r[0]["k1_m2"], "stable": True, "residual": 0}

    operation = MatchingOperation(inner, solve, {"purpose": "synthetic unit test"})
    with pytest.raises(ValueError, match="overlapping ownership"):
        LatticeExperiment(definition(), inner, measure=measure, directory=tmp_path / "bad", workspace=tmp_path, inner_match=operation)
    ex = LatticeExperiment(definition(), outer, measure=measure, directory=tmp_path / "good", workspace=tmp_path, inner_match=operation)
    result = ex.evaluate([1.5])
    assert result["objectives"]["cost"] == .8
    selected = ex.ledger.select()["selected"]
    assert selected["candidate"]["elements"][1]["length_m"] == 1.5
    assert selected["candidate"]["elements"][0]["k1_m2"] == .8
    assert selected["provenance"]["matching"]["matched"] is True


def test_solver_termination_is_not_match_success(tmp_path):
    operation = MatchingOperation(parameters(), lambda r, p: {
        "solver_success": True, "match_success": False, "elements": r}, {})
    ex = LatticeExperiment(definition(), measure=lambda r: {"cost": 1, "stable": True, "residual": 0},
                           directory=tmp_path / "study", workspace=tmp_path, inner_match=operation)
    outcome = ex.evaluate_rows(rows())
    assert outcome["feasible"] is False
    assert ex.ledger.select()["selected"] is None


def test_inner_match_cannot_change_unowned_field():
    def solve(r, p):
        r[1]["length_m"] += .1
        return {"success": True, "elements": r}
    with pytest.raises(ValueError, match="unowned"):
        MatchingOperation(parameters(), solve, {}).apply(rows())


def test_mutating_measurement_and_api_errors_do_not_become_physics(tmp_path):
    def mutating(r):
        r[0]["k1_m2"] = 99
        return {"cost": 1, "stable": True, "residual": 0}
    ex = LatticeExperiment(definition(), measure=mutating, directory=tmp_path / "study", workspace=tmp_path)
    with pytest.raises(ValueError, match="changed the lattice"):
        ex.evaluate_rows(rows())
    assert ex.ledger.entries()[0]["candidate"]["elements"] == rows()
    assert ex.ledger.select()["selected"] is None


def test_explicit_physics_rejection_keeps_candidate(tmp_path):
    def reject(r):
        raise ResearchInfeasible("unmatched", code="native_rejection", diagnostics={})
    ex = LatticeExperiment(definition(), measure=reject, directory=tmp_path / "study", workspace=tmp_path)
    result = ex.evaluate_rows(rows())
    assert result["feasible"] is False
    assert ex.ledger.entries()[0]["candidate"]["elements"] == rows()
    assert result["objectives"]["cost"] is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_general_optimizer_consumes_experiment_callback(tmp_path):
    # Numerical plumbing only: explicitly small standalone test, not a physics campaign.
    d = ExperimentDefinition("scalar test", "synthetic", {"cost": "min", "reward": "max"}, {})
    ex = LatticeExperiment(d, parameters(), measure=lambda r: {
        "cost": r[0]["k1_m2"] ** 2, "reward": r[0]["k1_m2"]}, directory=tmp_path / "study", workspace=tmp_path)
    result = api.particle_swarm_optimize(ex.evaluate, bounds=ex.parameterization.bounds,
        initial_position=[1.], objective_names=["cost", "reward"], population_size=4, generations=2)
    assert result["best_candidate"]["feasible"]
    assert ex.ledger.select()["pareto_front"]
    assert ex.evaluate([1.])["objectives"]["reward"] == -1


def test_native_evaluator_rebuilds_after_physical_topology_change(monkeypatch):
    calls = []
    class Editor:
        def evaluate_candidate(self, r, **kw):
            calls.append(("evaluate", copy.deepcopy(r)))
            return {"linear_optics": {"stable": True, "Dx_m": .01},
                    "radiation": {"emittance_pm": 50, "physical_validity": {"passed": True}}}
    monkeypatch.setattr(api, "build_jutrack_editor", lambda c: calls.append(("build", c)) or Editor())
    evaluate = api.JuTrackCellEvaluator(energy_gev=3, periods=32, expected_length_m=1.4,
        expected_angle_rad=0, limits=api.magnet_strength_limits(3), minimum_magnet_length_m=.1,
        minimum_drift_length_m=.1)
    original = rows()
    assert evaluate(original)["radiation_valid"]
    original[0]["k1_m2"] = .5
    evaluate(original)
    replaced = [api.drift("left", .2), api.drift("space", 1.), api.drift("right", .2)]
    evaluate(replaced)
    assert [name for name, _ in calls].count("build") == 2


def test_unknown_objective_direction_fails():
    with pytest.raises(ValueError):
        ExperimentDefinition("bad", "test", {"score": "maximize"}, {})


def test_mutated_contract_cannot_relabel_history(tmp_path):
    ex = LatticeExperiment(definition(), measure=lambda r: {}, directory=tmp_path / "study", workspace=tmp_path)
    ex.definition.constraints["residual"]["upper"] = 99
    with pytest.raises(ValueError, match="contract changed"):
        ex.evaluate_rows(rows())
    assert ex.ledger.entries() == []


def test_same_names_different_types_require_recompiled_matcher():
    changed = rows()
    changed[0] = api.drift("left", .2)
    with pytest.raises(ValueError, match="kinds differ"):
        MatchingOperation(parameters(), lambda r, p: {"success": True, "elements": r}, {}).apply(changed)


def test_replacing_measurement_callback_requires_new_experiment(tmp_path):
    ex = LatticeExperiment(definition(), measure=lambda r: {}, directory=tmp_path / "study", workspace=tmp_path)
    ex.measure = lambda r: {"cost": 1, "stable": True, "residual": 0}
    with pytest.raises(ValueError, match="contract changed"):
        ex.evaluate_rows(rows())
    assert not ex.ledger.entries()


def test_rejected_json_safe_results_are_understood_by_pso(tmp_path):
    def measured(r):
        feasible = r[0]["k1_m2"] > .8
        return {"cost": r[0]["k1_m2"] if feasible else None, "stable": feasible, "residual": 0}
    ex = LatticeExperiment(definition(), parameters(), measure=measured, directory=tmp_path / "study", workspace=tmp_path)
    bad = ex.evaluate([.2])
    json.dumps(bad, allow_nan=False)
    result = api.particle_swarm_optimize(ex.evaluate, bounds=ex.parameterization.bounds,
        initial_position=[1.], objective_names=["cost"], population_size=4, generations=2)
    assert result["best_candidate"]["feasible"]
    assert ex.ledger.select()["selected"]["measurements"]["cost"] > .8
