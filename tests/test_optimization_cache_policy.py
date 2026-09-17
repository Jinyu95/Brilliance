"""Fast callbacks can retain swarm checkpoints without per-particle files."""
from __future__ import annotations

import json

import pytest

from brilliance import accelerator_api as api
from brilliance import population_optimization as optimization


@pytest.fixture(autouse=True)
def standalone_kernel(monkeypatch):
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)


def options(tmp_path, **overrides):
    return dict(bounds=[[-1., 1.]], initial_position=[.35], objective_names=["f"],
                population_size=4, generations=2, seed=12,
                checkpoint_path=tmp_path / "swarm.json", **overrides)


def objective(position):
    return {"objectives": {"f": float(position[0] ** 2), "g": float((position[0] - .5) ** 2)}, "feasible": True}


@pytest.mark.parametrize("kernel", [optimization.particle_swarm_optimize, api.particle_swarm_optimize])
def test_cache_disabled_never_reads_or_writes_particle_files_and_resumes_completed_checkpoint(tmp_path, monkeypatch, kernel):
    def no_cache_io(*args, **kwargs):
        pytest.fail("Disabled evaluation cache must not invoke cache IO helpers")
    monkeypatch.setattr(optimization, "_read_cached_evaluation", no_cache_io)
    monkeypatch.setattr(optimization, "_write_cached_evaluation", no_cache_io)
    calls = []
    def evaluate(position):
        calls.append(position.copy())
        return objective(position)
    kwargs = options(tmp_path, cache_evaluations=False)
    result = kernel(evaluate, **kwargs)
    checkpoint = kwargs["checkpoint_path"]
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint.exists() and stored["complete"]
    assert "cache_evaluations" not in stored["contract"]
    assert len(calls) == 12
    assert list(tmp_path.rglob("*.json")) == [checkpoint]
    assert not (tmp_path / "swarm.json.evaluations").exists()
    assert result["evaluation_cache"] == {"requested": False, "enabled": False, "directory": None,
        "supplied_directory_ignored": False, "generation_checkpoints_retained": True, "hits": 0, "writes": 0}
    old_calls = len(calls)
    resumed = kernel(evaluate, **kwargs, resume=True)
    assert len(calls) == old_calls
    assert resumed["convergence"]["history"] == result["convergence"]["history"]
    assert resumed["best_candidate"] == result["best_candidate"]
    assert resumed["checkpoint"]["resumed"]


def test_explicit_cache_directory_is_ignored_and_existing_files_preserved(tmp_path, monkeypatch):
    directory = tmp_path / "existing-cache"
    directory.mkdir()
    old_file = directory / "prior.json"
    old_file.write_text('{"keep":true}', encoding="utf-8")
    before = old_file.read_bytes()
    kwargs = options(tmp_path, cache_evaluations=False, evaluation_cache_directory=directory)
    result = api.particle_swarm_optimize(objective, **kwargs)
    assert old_file.read_bytes() == before
    assert list(directory.iterdir()) == [old_file]
    assert result["evaluation_cache"]["supplied_directory_ignored"] is True


def test_batched_multiobjective_evaluation_retains_generation_checkpoint_without_files(tmp_path):
    calls = []
    def batch(positions):
        calls.extend(positions)
        return [objective(position) for position in positions]
    kwargs = options(tmp_path, cache_evaluations=False, evaluate_batch=batch, evaluation_batch_size=3)
    kwargs["objective_names"] = ["f", "g"]
    result = optimization.particle_swarm_optimize(lambda _: pytest.fail("Use the supplied batch"), **kwargs)
    assert len(calls) == 12
    assert len(result["convergence"]["history"]) == 3
    assert result["checkpoint"]["completed_generation"] == 2
    assert list(tmp_path.rglob("*.json")) == [kwargs["checkpoint_path"]]


def test_default_cache_remains_enabled_and_policy_can_change_on_resume(tmp_path):
    kwargs = options(tmp_path)
    result = api.particle_swarm_optimize(objective, **kwargs)
    assert result["evaluation_cache"]["enabled"]
    assert result["evaluation_cache"]["writes"] > 0
    cache = tmp_path / "swarm.json.evaluations"
    files_before = {path: path.read_bytes() for path in cache.rglob("*.json")}
    resumed = api.particle_swarm_optimize(lambda _: pytest.fail("Completed swarm requires no new physics"),
                                        **kwargs, cache_evaluations=False, resume=True)
    assert resumed["convergence"]["history"] == result["convergence"]["history"]
    assert not resumed["evaluation_cache"]["enabled"]
    assert {path: path.read_bytes() for path in cache.rglob("*.json")} == files_before


def test_generation_extension_with_cache_disabled_only_evaluates_new_generation(tmp_path):
    kwargs = options(tmp_path, cache_evaluations=False)
    result = api.particle_swarm_optimize(objective, **kwargs)
    calls = []
    def evaluate(position):
        calls.append(position.copy())
        return objective(position)
    kwargs["generations"] = 3
    resumed = api.particle_swarm_optimize(evaluate, **kwargs, resume=True)
    assert len(calls) == 4
    assert resumed["convergence"]["history"][:3] == result["convergence"]["history"]
    assert not (tmp_path / "swarm.json.evaluations").exists()


@pytest.mark.parametrize("invalid", [0, 1, "false", None, []])
def test_cache_policy_is_strict_boolean_before_calculation_or_file_creation(tmp_path, invalid):
    with pytest.raises(optimization.OptimizerInputError, match="cache_evaluations must be a boolean"):
        api.particle_swarm_optimize(lambda _: pytest.fail("No calculation"), **options(tmp_path, cache_evaluations=invalid))
    assert not list(tmp_path.iterdir())
