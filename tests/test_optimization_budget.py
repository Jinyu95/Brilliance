"""Budget resolution and checkpoint integration; no accelerator calculations."""

import json

import pytest

from brilliance import accelerator_api
from brilliance.optimization_budget import resolve_population_budget
from brilliance.population_optimization import particle_swarm_optimize, read_optimization_checkpoint


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in ("BRILLIANCE_RUN_OUTPUT", "BRILLIANCE_RUNTIME_CONFIG", "BRILLIANCE_SOURCE_ROOT"):
        monkeypatch.delenv(name, raising=False)


def managed_config(monkeypatch, tmp_path, text="[optimization]\n"):
    config = tmp_path / "runtime.toml"
    config.write_text(text, encoding="utf-8")
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("BRILLIANCE_RUNTIME_CONFIG", str(config))
    return config


@pytest.mark.parametrize("dimensions,expected", [(1, 500), (50, 500), (51, 510), (100, 1000)])
def test_omitted_generic_defaults_are_production_sized(dimensions, expected):
    report = resolve_population_budget(dimensions, None, None)
    assert report["effective"] == {"population_size": expected, "generations": 80}
    assert report["requested"] == {"population_size": None, "generations": None}


def test_explicit_standalone_kernel_probe_remains_small():
    report = resolve_population_budget(2, 4, 1)
    assert report["effective"] == {"population_size": 4, "generations": 1}
    assert not report["managed_research"]
    assert not report["adjusted"]


def test_managed_small_model_budget_is_raised_not_rejected(monkeypatch, tmp_path):
    config = managed_config(monkeypatch, tmp_path)
    report = resolve_population_budget(100, 60, 30)
    assert report["effective"] == {"population_size": 1000, "generations": 80}
    assert report["requested"] == {"population_size": 60, "generations": 30}
    assert report["configuration"] == str(config)
    assert report["adjusted"]
    assert "Requested 60 particles / 30 generations" in report["summary"]


def test_scientist_can_change_all_floors_in_runtime_toml(monkeypatch, tmp_path):
    managed_config(monkeypatch, tmp_path, '''[optimization]
minimum_population = 8
minimum_population_per_knob = 2
minimum_generations = 3
''')
    report = resolve_population_budget(6, 4, 1)
    assert report["effective"] == {"population_size": 12, "generations": 3}


@pytest.mark.parametrize("line", [
    "minimum_population = 3", "minimum_population = true",
    "minimum_population_per_knob = 0", "minimum_generations = 0",
    'minimum_generations = "5"', "minimum_generations = 2.5",
])
def test_invalid_scientist_config_is_explicit(monkeypatch, tmp_path, line):
    managed_config(monkeypatch, tmp_path, "[optimization]\n" + line)
    with pytest.raises(ValueError, match=r"Runtime \[optimization\]"):
        resolve_population_budget(2, 4, 1)


def test_larger_requested_campaign_is_not_shrunk(monkeypatch, tmp_path):
    managed_config(monkeypatch, tmp_path)
    report = resolve_population_budget(3, 700, 160)
    assert report["effective"] == report["requested"]
    assert not report["adjusted"]


def test_existing_small_campaign_is_not_resized_or_misrepresented(monkeypatch, tmp_path):
    managed_config(monkeypatch, tmp_path)
    report = resolve_population_budget(3, 60, 30, resumed_population_size=60)
    assert report["effective"] == {"population_size": 60, "generations": 30}
    assert report["below_production_floor"]
    assert "new campaign/checkpoint" in " ".join(report["notes"])
    assert "not a production-sized new search" in report["summary"]


def test_resume_default_does_not_rewind_completed_generations(monkeypatch, tmp_path):
    managed_config(monkeypatch, tmp_path)
    report = resolve_population_budget(3, None, None, resumed_population_size=60,
                                       completed_generation=120)
    assert report["effective"] == {"population_size": 60, "generations": 120}


@pytest.mark.parametrize("requested,target,completed,is_complete,expected", [
    (None, 30, 12, False, 30),
    (None, None, 30, True, 30),
    (30, None, 30, True, 30),
    (40, 30, 30, True, 40),
    (None, None, 12, False, 12),
    (20, None, 12, False, 20),
])
def test_resume_never_automatically_buys_a_new_production_budget(
        monkeypatch, tmp_path, requested, target, completed, is_complete, expected):
    managed_config(monkeypatch, tmp_path)
    report = resolve_population_budget(
        3, None, requested, resumed_population_size=60, completed_generation=completed,
        saved_generation_target=target, checkpoint_complete=is_complete)
    assert report["effective"] == {"population_size": 60, "generations": expected}


class CancelBeforeEvaluation(Exception):
    pass


@pytest.mark.parametrize("kernel", [particle_swarm_optimize, accelerator_api.particle_swarm_optimize])
def test_real_public_entry_emits_budget_before_any_evaluation(monkeypatch, tmp_path, kernel):
    managed_config(monkeypatch, tmp_path)
    updates = []

    def no_calculation(_position):
        pytest.fail("Budget test must not run an objective")

    def cancel():
        raise CancelBeforeEvaluation

    with pytest.raises(CancelBeforeEvaluation):
        kernel(no_calculation, bounds=[[-1, 1]] * 3, initial_position=[0] * 3,
               objective_names=["f"], population_size=8, generations=2,
               progress_callback=updates.append, cancellation_check=cancel)
    assert updates[0]["phase"] == "budget"
    assert updates[0]["population_size"] == 500
    assert updates[0]["total_generations"] == 80
    assert updates[0]["requested_population_size"] == 8


def objective(position):
    return {"objectives": {"f": float(position[0] ** 2)}, "feasible": True}


def test_budget_recorded_with_checkpoint_and_original_history_survives_resume(monkeypatch, tmp_path):
    # This is a 1D unit-test callback, not a scientific optimization campaign.
    checkpoint = tmp_path / "small-checkpoint.json"
    original = particle_swarm_optimize(objective, bounds=[[-1, 1]], initial_position=[0.5],
                                      objective_names=["f"], population_size=4, generations=1,
                                      checkpoint_path=checkpoint)
    old_bytes = checkpoint.read_bytes()
    old_state = json.loads(old_bytes)
    assert old_state["budget"]["effective"] == {"population_size": 4, "generations": 1}
    managed_config(monkeypatch, tmp_path, '''[optimization]
minimum_population = 16
minimum_population_per_knob = 10
minimum_generations = 2
''')
    resumed = particle_swarm_optimize(objective, bounds=[[-1, 1]], initial_position=[0.5],
                                     objective_names=["f"], population_size=16, generations=2,
                                     checkpoint_path=checkpoint, resume=True)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["contract"] == old_state["contract"]
    assert state["contract_fingerprint"] == old_state["contract_fingerprint"]
    assert state["history"][:2] == old_state["history"]
    assert state["evaluation_count"] == old_state["evaluation_count"] + 4
    assert resumed["checkpoint"]["resumed"]
    assert resumed["budget"]["effective"] == {"population_size": 4, "generations": 2}
    assert resumed["budget"]["below_production_floor"]
    assert read_optimization_checkpoint(checkpoint)["budget"] == resumed["budget"]
    assert original["budget"]["effective"]["population_size"] == 4


@pytest.mark.parametrize("with_budget", [True, False])
@pytest.mark.parametrize("requested_steps", [None, 1])
def test_completed_small_checkpoint_postprocessing_resume_performs_no_evaluation(
        monkeypatch, tmp_path, with_budget, requested_steps):
    checkpoint = tmp_path / "completed-checkpoint.json"
    particle_swarm_optimize(objective, bounds=[[-1, 1]], initial_position=[0.5],
                            objective_names=["f"], population_size=4, generations=1,
                            checkpoint_path=checkpoint)
    if not with_budget:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        state.pop("budget", None)  # Older schema-1 checkpoints predate budgets.
        checkpoint.write_text(json.dumps(state), encoding="utf-8")
    original = checkpoint.read_bytes()
    old_state = json.loads(original)
    managed_config(monkeypatch, tmp_path)  # New campaigns require 500 x 80.

    def no_calculation(_position):
        pytest.fail("Postprocessing repair must not run more generations")

    recovered = particle_swarm_optimize(
        no_calculation, bounds=[[-1, 1]], initial_position=[0.5], objective_names=["f"],
        population_size=4, generations=requested_steps, checkpoint_path=checkpoint, resume=True)
    assert recovered["budget"]["effective"] == {"population_size": 4, "generations": 1}
    assert recovered["budget"]["below_production_floor"]
    assert recovered["evaluated_candidates"] == old_state["evaluation_count"]
    assert checkpoint.read_bytes() == original


def test_changed_physics_on_resume_still_rejected_without_overwriting_history(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    particle_swarm_optimize(objective, bounds=[[-1, 1]], initial_position=[0.5],
                            objective_names=["f"], population_size=4, generations=1,
                            checkpoint_path=checkpoint)
    original = checkpoint.read_bytes()
    managed_config(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Changed fields: bounds"):
        particle_swarm_optimize(objective, bounds=[[-2, 2]], initial_position=[0.5],
                                objective_names=["f"], population_size=60, generations=30,
                                checkpoint_path=checkpoint, resume=True)
    assert checkpoint.read_bytes() == original


def lattice_parameterizations():
    rows = [
        {"name": "Q", "kind": "KQUAD", "length_m": 0.2, "k1_m2": 0.5,
         "fields": {"family": "DESIGN"}},
        {"name": "M", "kind": "KQUAD", "length_m": 0.2, "k1_m2": 0.0,
         "fields": {"family": "MATCH"}},
    ]
    design = accelerator_api.compile_family_parameterization(
        rows, field="k1_m2", families=["DESIGN"], bounds=(-1, 1), role="outer_design")
    matching = accelerator_api.compile_family_parameterization(
        rows, field="k1_m2", families=["MATCH"], bounds=(-1, 1), role="in_objective_matching")
    return rows, design, matching


def test_lattice_planner_uses_same_budget_before_seeds_and_workers(monkeypatch, tmp_path):
    managed_config(monkeypatch, tmp_path, '''[optimization]
minimum_population = 12
minimum_population_per_knob = 10
minimum_generations = 3
''')
    rows, design, matching = lattice_parameterizations()
    plan = accelerator_api.plan_lattice_optimization(
        design, matching, population_size=4, generations=1,
        minimum_population_per_knob=0, minimum_production_generations=0)
    assert plan["population_size"] == 12
    assert plan["generations"] == 3
    assert plan["budget"]["requested"] == {"population_size": 4, "generations": 1}

    def kernel_boundary(_evaluate, **options):
        assert options["population_size"] == 12
        assert options["generations"] == 3
        assert len(options["seed_positions"]) == 9  # 80% of 11 nonbaseline members
        raise CancelBeforeEvaluation

    from brilliance import lattice_optimization
    monkeypatch.setattr(lattice_optimization, "particle_swarm_optimize", kernel_boundary)
    updates = []
    with pytest.raises(CancelBeforeEvaluation):
        accelerator_api.optimize_lattice_parameters(
            rows, design, matching,
            lambda candidate: {"stable": True, "m": candidate[1]["k1_m2"]},
            lambda candidate: {"objectives": {"f": candidate[0]["k1_m2"] ** 2}, "feasible": True},
            match_targets={"m": {"target": 0, "tolerance": 1e-6}}, objective_names=["f"],
            population_size=4, generations=1, require_feasible_initial_population=False,
            progress_callback=updates.append)
    assert updates[0]["phase"] == "budget"
    assert updates[0]["population_size"] == 12
