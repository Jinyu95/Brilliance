"""Bounded population optimizers used by registered accelerator services.

The numerical optimizer is deliberately independent of JuTrack.  A caller
supplies the expensive physics evaluation and this module owns only proposal,
constraint handling, and population selection.  This keeps the scientific
boundary explicit and makes the evolutionary mechanics unit-testable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .artifacts import content_hash, json_safe
from .atomic_io import atomic_write_text

Evaluation = Mapping[str, Any]
Evaluator = Callable[[np.ndarray], Evaluation]
BatchEvaluator = Callable[[Sequence[np.ndarray]], Sequence[Evaluation]]
EvaluatorFactory = Callable[[], Evaluator]

OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION = "1.0"


class OptimizerInputError(ValueError):
    """A repairable optimizer call-contract error, not physics infeasibility.

    Raised only while validating the caller's arguments or callback contract.
    Physics evaluation exceptions and incompatible saved checkpoints are not
    converted to this type. ValueError compatibility is deliberately retained.
    """


def _integer_option(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    message = (f"{name} must be an integer from {minimum} through {maximum}"
               if maximum is not None else f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
        valid = not isinstance(value, (bool, np.bool_)) and float(value) == result
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerInputError(message) from exc
    if not valid or result < minimum or (maximum is not None and result > maximum):
        raise OptimizerInputError(message)
    return result


def _numeric_array(value: Any, message: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerInputError(message) from exc


def _names(value: Any, message: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise OptimizerInputError(message)
    try:
        names = tuple(value)
    except TypeError as exc:
        raise OptimizerInputError(message) from exc
    if (not names or any(not isinstance(item, str) or not item.strip() for item in names)
            or len(set(names)) != len(names)):
        raise OptimizerInputError(message)
    return names


class ProcessBatchEvaluator:
    """Evaluate independent candidates in persistent, isolated processes.

    ``evaluator_factory`` is serialized, but the evaluator it creates is not.
    A JuTrack factory should import the configured pyJuTrack module and build
    its mutable lattice inside the worker.  The worker then reuses that state
    for later candidates assigned to the same process.
    """

    def __init__(
        self,
        evaluator_factory: EvaluatorFactory,
        *,
        workers: int | None = None,
        verbose: int = 0,
    ) -> None:
        if not callable(evaluator_factory):
            raise TypeError("evaluator_factory must be callable")
        from .parallel_safety import validate_worker_factory

        validate_worker_factory(evaluator_factory)
        from .parallel_evaluation import ParallelEvaluator

        def factory():
            evaluate = evaluator_factory()
            return lambda position: dict(evaluate(np.asarray(position, dtype=float)))

        self._parallel = ParallelEvaluator(factory, workers=workers)
        self._workers = self._parallel.workers
        self._closed = False

    @property
    def workers(self) -> int:
        return self._workers

    def __call__(self, positions: Sequence[np.ndarray]) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("process batch evaluator is closed")
        if not positions:
            return []
        results = self._parallel(
            np.asarray(position, dtype=float).tolist()
            for position in positions
        )
        return [dict(item) for item in results]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._parallel.close()

    def __enter__(self) -> "ProcessBatchEvaluator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _checkpoint_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return json_safe(
        {
            "candidate_id": candidate["candidate_id"],
            "position": np.asarray(candidate["position"], dtype=float).tolist(),
            "velocity": np.asarray(candidate["velocity"], dtype=float).tolist(),
            "objective_values": list(candidate["objective_values"]),
            "constraint_violations": dict(candidate.get("constraint_violations", {})),
            "constraint_violation": candidate.get("constraint_violation", math.inf),
            "feasible": bool(candidate.get("feasible")),
            "raw": dict(candidate.get("raw", {})),
        }
    )


def _restore_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "position": np.asarray(candidate["position"], dtype=float),
        "velocity": np.asarray(candidate["velocity"], dtype=float),
        "objective_values": tuple(float(item) for item in candidate["objective_values"]),
        "constraint_violations": {
            str(name): float(value)
            for name, value in dict(candidate.get("constraint_violations", {})).items()
        },
        "constraint_violation": float(candidate.get("constraint_violation", math.inf)),
        "feasible": bool(candidate.get("feasible")),
        "raw": dict(candidate.get("raw", {})),
    }


def _write_pso_checkpoint(
    path: Path | None,
    *,
    contract: Mapping[str, Any],
    contract_fingerprint: str,
    mode: str,
    completed_generation: int,
    next_candidate_id: int,
    evaluation_count: int,
    rng_state: Mapping[str, Any],
    population: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    complete: bool,
    personal_best: Sequence[Mapping[str, Any]] | None = None,
    global_best: Mapping[str, Any] | None = None,
    budget: Mapping[str, Any] | None = None,
) -> None:
    if path is None:
        return
    payload = {
        "schema_version": OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION,
        "contract": dict(contract),
        "contract_fingerprint": contract_fingerprint,
        "mode": mode,
        "completed_generation": int(completed_generation),
        "next_candidate_id": int(next_candidate_id),
        "evaluation_count": int(evaluation_count),
        "rng_state": dict(rng_state),
        "population": [_checkpoint_candidate(item) for item in population],
        "personal_best": (
            [_checkpoint_candidate(item) for item in personal_best]
            if personal_best is not None
            else None
        ),
        "global_best": _checkpoint_candidate(global_best) if global_best is not None else None,
        "history": list(history),
        "complete": bool(complete),
        "budget": dict(budget or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(json_safe(payload), ensure_ascii=False,
                                     allow_nan=False, indent=2) + "\n")


def _load_pso_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"optimization checkpoint does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"optimization checkpoint is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("optimization checkpoint must contain one JSON object")
    if payload.get("schema_version") != OPTIMIZATION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("optimization checkpoint schema is not supported")
    return dict(payload)


def read_optimization_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read committed optimizer evidence without evaluating or resuming a job.

    This is a reporting/recovery view, not a proof of current scientific validity.
    It works even when plotting failed after the final generation was saved.
    """
    source = Path(path).resolve()
    state = _load_pso_checkpoint(source)
    contract = state["contract"]
    if content_hash(contract, length=32) != state["contract_fingerprint"]:
        raise ValueError("optimization checkpoint contract fingerprint mismatch")
    names, objectives = contract["variable_names"], contract["objective_names"]
    candidates = [_restore_candidate(item) for item in
                  (state.get("personal_best") or state["population"])]
    if not candidates:
        raise ValueError("optimization checkpoint contains no evaluated population")
    front = _first_front(candidates)
    selected = (_restore_candidate(state["global_best"]) if len(objectives) == 1
                else _representative(front or candidates, contract["selection"]))
    return {
        "checkpoint_path": str(source), "contract": dict(contract),
        "completed_generation": state["completed_generation"],
        "complete": bool(state.get("complete")),
        "evaluated_candidates": state["evaluation_count"],
        "best_candidate": _serial_candidate(selected, names, objectives),
        "pareto_candidates": [_serial_candidate(item, names, objectives) for item in front],
        "history": state["history"],
        "budget": dict(state.get("budget") or {}),
        "scope": "archived_optimizer_evidence_requires_selected_candidate_revalidation",
    }


def _evaluation_cache_path(
    directory: Path | None,
    contract_fingerprint: str,
    position: Sequence[float],
) -> Path | None:
    if directory is None:
        return None
    key = content_hash(
        {
            "contract_fingerprint": contract_fingerprint,
            "position": [float(value) for value in position],
        },
        length=32,
    )
    return directory / f"candidate-{key}.json"


def _read_cached_evaluation(
    directory: Path | None,
    contract_fingerprint: str,
    position: Sequence[float],
) -> dict[str, Any] | None:
    path = _evaluation_cache_path(directory, contract_fingerprint, position)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("contract_fingerprint") != contract_fingerprint:
        return None
    result = payload.get("evaluation")
    return dict(result) if isinstance(result, Mapping) else None


def _write_cached_evaluation(
    directory: Path | None,
    contract_fingerprint: str,
    position: Sequence[float],
    evaluation: Mapping[str, Any],
) -> None:
    path = _evaluation_cache_path(directory, contract_fingerprint, position)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "contract_fingerprint": contract_fingerprint,
        "position": [float(value) for value in position],
        "evaluation": dict(evaluation),
    }
    atomic_write_text(path, json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False) + "\n")


def bounded_pattern_search(
    evaluate: Evaluator,
    *,
    bounds: Sequence[Sequence[float]],
    objective_names: Sequence[str],
    initial_position: Sequence[float],
    variable_names: Sequence[str] | None = None,
    max_evaluations: int = 48,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a small derivative-free local search with hard bounds.

    This is the deliberately inexpensive optimizer used for interactive
    refinement.  It is not presented as a global or multi-objective method.
    """
    lower, upper, names, objectives, initial = _validated_inputs(
        bounds, objective_names, initial_position, variable_names
    )
    budget = _integer_option(max_evaluations, "max_evaluations", 2, 4096)
    rng = np.random.default_rng(_integer_option(seed, "seed", 0))
    if not callable(evaluate):
        raise OptimizerInputError("evaluate must be callable")
    current = _candidate(initial, np.zeros_like(initial), evaluate, objectives, 0)
    evaluated = [current]
    best = current
    width = upper - lower
    step = 0.20 * width
    order = np.arange(len(initial))
    generation = 0
    history = [_history_row(generation, [best], objectives)]
    while len(evaluated) < budget and np.max(step / width) > 1e-4:
        rng.shuffle(order)
        improved = False
        for dimension in order:
            for sign in (1.0, -1.0):
                if len(evaluated) >= budget:
                    break
                proposal = current["position"].copy()
                proposal[dimension] = np.clip(
                    proposal[dimension] + sign * step[dimension],
                    lower[dimension], upper[dimension],
                )
                if np.array_equal(proposal, current["position"]):
                    continue
                candidate = _candidate(
                    proposal, np.zeros_like(initial), evaluate, objectives, len(evaluated)
                )
                evaluated.append(candidate)
                if _single_key(candidate) < _single_key(best):
                    best = candidate
                    current = candidate
                    improved = True
                    break
            if len(evaluated) >= budget:
                break
        generation += 1
        if not improved:
            step *= 0.5
        history.append(_history_row(generation, [best], objectives))
    front = _first_front(evaluated)
    return _result(
        algorithm={
            "name": "bounded_pattern_search",
            "mode": "quick_local",
            "proposal": "bounded coordinate pattern steps",
            "environmental_selection": "best constrained scalar incumbent",
        },
        candidates=evaluated,
        front=front,
        selected=best,
        names=names,
        objectives=objectives,
        lower=lower,
        upper=upper,
        history=history,
        complete=len(evaluated) >= 2,
        evaluation_count=len(evaluated),
    )


def particle_swarm_optimize(
    evaluate: Evaluator,
    *,
    bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    objective_names: Sequence[str],
    initial_position: Sequence[float] | Mapping[str, float],
    variable_names: Sequence[str] | None = None,
    population_size: int | None = None,
    generations: int | None = None,
    seed: int = 0,
    selection: str = "knee",
    seed_positions: Sequence[Sequence[float]] | None = None,
    initial_velocity_fraction: float = 0.10,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    evaluate_batch: BatchEvaluator | None = None,
    evaluation_batch_size: int | None = None,
    cancellation_check: Callable[[], None] | None = None,
    checkpoint_path: str | Path | None = None,
    evaluation_cache_directory: str | Path | None = None,
    cache_evaluations: bool = True,
    campaign_identity: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run PSO, with NSGA-style replacement for multiple objectives.

    For one objective this is conventional constrained PSO with personal and
    global best memories.  For two or more objectives, PSO equations generate
    offspring, then parents and offspring are combined and a complete
    non-dominated-sort/crowding environmental selection chooses the next
    population.  Particle identities therefore do not force survival.

    Omitted budgets use configured defaults (64 particles / 40 generations when
    unconfigured). Explicit budgets are honored, not increased automatically.
    Neither a budget nor termination proves convergence. Resume retains the saved swarm.
    ``cache_evaluations=False`` skips all per-evaluation file caching, including
    any supplied cache directory, while retaining complete generation checkpoints.
    Interrupted partial generations may then need reevaluation on resume.
    """
    lower, upper, names, objectives, initial = _validated_inputs(
        bounds, objective_names, initial_position, variable_names
    )
    if not isinstance(cache_evaluations, bool):
        raise OptimizerInputError("cache_evaluations must be a boolean")
    requested_size = (_integer_option(population_size, "population_size", 4)
                      if population_size is not None else None)
    requested_steps = (_integer_option(generations, "generations", 1)
                       if generations is not None else None)
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
    if resume and checkpoint is None:
        raise OptimizerInputError("resume=True requires checkpoint_path")
    state = _load_pso_checkpoint(checkpoint) if resume else None
    from .optimization_budget import resolve_population_budget

    budget = resolve_population_budget(
        len(initial), requested_size, requested_steps,
        resumed_population_size=int(state["contract"]["population_size"]) if state else None,
        completed_generation=int(state["completed_generation"]) if state else 0,
        saved_generation_target=(state.get("budget", {}).get("effective", {}).get("generations")
                                 if state and state.get("budget") else None),
        checkpoint_complete=bool(state.get("complete")) if state else False,
    )
    size = budget["effective"]["population_size"]
    steps = budget["effective"]["generations"]
    if not isinstance(selection, str) or selection not in {"knee", "first_objective"}:
        raise OptimizerInputError("selection must be 'knee' or 'first_objective'")
    try:
        velocity_fraction = float(initial_velocity_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerInputError("initial_velocity_fraction must be between 0 and 0.5") from exc
    if not math.isfinite(velocity_fraction) or not 0.0 <= velocity_fraction <= 0.5:
        raise OptimizerInputError("initial_velocity_fraction must be between 0 and 0.5")
    rng = np.random.default_rng(_integer_option(seed, "seed", 0))
    for callback_name, callback in (("evaluate", evaluate), ("evaluate_batch", evaluate_batch),
                                    ("progress_callback", progress_callback),
                                    ("cancellation_check", cancellation_check)):
        if (callback is not None or callback_name == "evaluate") and not callable(callback):
            raise OptimizerInputError(f"{callback_name} must be callable")
    width = upper - lower
    positions = _latin_hypercube(rng, size, lower, upper)
    positions[0] = initial
    supplied_seeds = _validated_seed_positions(
        seed_positions, lower=lower, upper=upper, dimensions=len(initial)
    )
    for index, supplied in enumerate(supplied_seeds[: max(0, size - 1)], 1):
        positions[index] = supplied
    velocities = rng.uniform(
        -velocity_fraction, velocity_fraction, size=(size, len(initial))
    ) * width
    progress_interval = max(1, math.ceil(size / 10))
    batch_size = (
        progress_interval
        if evaluation_batch_size is None
        else _integer_option(evaluation_batch_size, "evaluation_batch_size", 1)
    )
    evaluation_cache = (
        None if not cache_evaluations else
        Path(evaluation_cache_directory).resolve()
        if evaluation_cache_directory is not None
        else checkpoint.with_name(f"{checkpoint.name}.evaluations")
        if checkpoint is not None
        else None
    )
    contract = {
        "optimizer": "particle_swarm_optimize",
        "bounds": np.column_stack((lower, upper)).tolist(),
        "variable_names": list(names),
        "objective_names": list(objectives),
        "population_size": size,
        "seed": int(seed),
        "selection": selection,
        "initial_velocity_fraction": velocity_fraction,
        "campaign_identity": str(campaign_identity or ""),
    }
    contract_fingerprint = content_hash(contract, length=32)
    cache_hits = 0
    cache_writes = 0

    def publish_evaluation_progress(
        generation: int, completed: int, *, force: bool = False
    ) -> None:
        if force or completed == 1 or completed == size or completed % progress_interval == 0:
            _publish_progress(
                progress_callback,
                "population_evaluation",
                {
                    "generation": generation,
                    "total_generations": steps,
                    "evaluated_in_generation": completed,
                    "population_size": size,
                    "fraction_complete": completed / size,
                },
            )

    def evaluate_population(
        candidate_positions: Sequence[np.ndarray],
        candidate_velocities: Sequence[np.ndarray],
        candidate_ids: Sequence[int],
        generation: int,
    ) -> list[dict[str, Any]]:
        nonlocal cache_hits, cache_writes
        produced: list[dict[str, Any]] = []
        total = len(candidate_positions)
        for start in range(0, total, batch_size):
            if cancellation_check is not None:
                cancellation_check()
            stop = min(total, start + batch_size)
            position_batch = [
                np.asarray(item, dtype=float).copy()
                for item in candidate_positions[start:stop]
            ]
            raw_batch: list[Evaluation | None] = [None] * len(position_batch)
            missing_positions: list[np.ndarray] = []
            missing_indices: list[int] = []
            for local_index, position in enumerate(position_batch):
                cached = _read_cached_evaluation(
                    evaluation_cache, contract_fingerprint, position
                ) if evaluation_cache is not None else None
                if cached is None:
                    missing_positions.append(position)
                    missing_indices.append(local_index)
                else:
                    raw_batch[local_index] = cached
                    cache_hits += 1
            if missing_positions and evaluate_batch is None:
                for local_index, position in zip(missing_indices, missing_positions):
                    if cancellation_check is not None:
                        cancellation_check()
                    raw = dict(evaluate(position))
                    raw_batch[local_index] = raw
                    if evaluation_cache is not None:
                        _write_cached_evaluation(
                            evaluation_cache, contract_fingerprint, position, raw
                        )
                        cache_writes += 1
            elif missing_positions:
                evaluated = list(evaluate_batch(missing_positions))
                if len(evaluated) != len(missing_positions):
                    raise OptimizerInputError(
                        "evaluate_batch must return exactly one result per position"
                    )
                for local_index, position, supplied in zip(
                    missing_indices, missing_positions, evaluated
                ):
                    raw = dict(supplied)
                    raw_batch[local_index] = raw
                    if evaluation_cache is not None:
                        _write_cached_evaluation(
                            evaluation_cache, contract_fingerprint, position, raw
                        )
                        cache_writes += 1
            produced.extend(
                _candidate_from_raw(
                    position,
                    np.asarray(velocity, dtype=float),
                    raw,
                    objectives,
                    candidate_id,
                )
                for position, velocity, raw, candidate_id in zip(
                    position_batch,
                    candidate_velocities[start:stop],
                    [dict(item or {}) for item in raw_batch],
                    candidate_ids[start:stop],
                )
            )
            completed = stop
            publish_evaluation_progress(generation, completed, force=True)
        return produced

    expected_mode = "single_objective" if len(objectives) == 1 else "multi_objective"
    resumed = False
    if resume:
        if state.get("contract_fingerprint") != contract_fingerprint:
            saved_contract = state.get("contract", {})
            changed = [key for key, value in contract.items()
                       if saved_contract.get(key) != value]
            raise ValueError(
                "optimization checkpoint does not match bounds, variables, objectives, "
                "population, selection, seed, or campaign identity. "
                f"Changed fields: {', '.join(changed) or 'contract fingerprint'}. "
                "The checkpoint is preserved. Read completed results with "
                "read_optimization_checkpoint(checkpoint_path), then revalidate the "
                "selected candidate; use a new campaign name for changed physics."
            )
        if state.get("mode") != expected_mode:
            raise ValueError("optimization checkpoint mode does not match this campaign")
        population = [_restore_candidate(item) for item in state.get("population", [])]
        if len(population) != size:
            raise ValueError("optimization checkpoint has an invalid population size")
        history = [dict(item) for item in state.get("history", [])]
        completed_generation = int(state.get("completed_generation", -1))
        if completed_generation < 0 or completed_generation > steps or not history:
            raise ValueError("optimization checkpoint has an invalid completed generation")
        next_id = int(state.get("next_candidate_id", size))
        evaluation_count = int(state.get("evaluation_count", size * (completed_generation + 1)))
        rng.bit_generator.state = dict(state["rng_state"])
        resumed = True
        _publish_progress(
            progress_callback,
            "resume",
            {
                "checkpoint_path": str(checkpoint),
                "completed_generation": completed_generation,
                "remaining_generations": steps - completed_generation,
            },
        )
    _publish_progress(progress_callback, "budget", {
        "summary": budget["summary"], "budget": budget,
        "requested_population_size": requested_size, "requested_generations": requested_steps,
        "population_size": size, "total_generations": steps,
    })
    if not resume:
        population = evaluate_population(
            list(positions), list(velocities), list(range(size)), 0
        )
        history = [_history_row(0, population, objectives)]
        _publish_progress(
            progress_callback,
            "generation",
            {**history[-1], "total_generations": steps},
        )
        completed_generation = 0
        next_id = size
        evaluation_count = size

    if len(objectives) == 1:
        if resumed:
            personal_best = [
                _restore_candidate(item)
                for item in state.get("personal_best", [])
            ]
            if len(personal_best) != size:
                raise ValueError("optimization checkpoint has invalid personal-best state")
            global_best = _restore_candidate(state["global_best"])
        else:
            personal_best = [_copy_candidate(item) for item in population]
            global_best = min(personal_best, key=_single_key)
            _write_pso_checkpoint(
                checkpoint,
                contract=contract,
                contract_fingerprint=contract_fingerprint,
                mode=expected_mode,
                completed_generation=completed_generation,
                next_candidate_id=next_id,
                evaluation_count=evaluation_count,
                rng_state=rng.bit_generator.state,
                population=population,
                history=history,
                personal_best=personal_best,
                global_best=global_best,
                complete=completed_generation >= steps,
                budget=budget,
            )
        for generation in range(completed_generation + 1, steps + 1):
            inertia = 0.85 - 0.45 * ((generation - 1) / max(1, steps - 1))
            next_positions = []
            next_velocities = []
            for index, particle in enumerate(population):
                r1 = rng.random(len(initial)); r2 = rng.random(len(initial))
                velocity = (
                    inertia * particle["velocity"]
                    + 1.7 * r1 * (personal_best[index]["position"] - particle["position"])
                    + 1.7 * r2 * (global_best["position"] - particle["position"])
                )
                velocity = np.clip(velocity, -0.35 * width, 0.35 * width)
                position, velocity = _bounded_move(
                    particle["position"], velocity, lower, upper
                )
                next_positions.append(position)
                next_velocities.append(velocity)
                next_id += 1
            next_population = evaluate_population(
                next_positions,
                next_velocities,
                list(range(next_id - size, next_id)),
                generation,
            )
            evaluation_count += size
            for index, child in enumerate(next_population):
                if _single_key(child) < _single_key(personal_best[index]):
                    personal_best[index] = _copy_candidate(child)
            population = next_population
            global_best = min([global_best, *personal_best], key=_single_key)
            history.append(_history_row(generation, personal_best, objectives))
            _publish_progress(
                progress_callback,
                "generation",
                {**history[-1], "total_generations": steps},
            )
            completed_generation = generation
            _write_pso_checkpoint(
                checkpoint,
                contract=contract,
                contract_fingerprint=contract_fingerprint,
                mode=expected_mode,
                completed_generation=completed_generation,
                next_candidate_id=next_id,
                evaluation_count=evaluation_count,
                rng_state=rng.bit_generator.state,
                population=population,
                history=history,
                personal_best=personal_best,
                global_best=global_best,
                complete=completed_generation >= steps,
                budget=budget,
            )
        candidates = personal_best
        front = _first_front(candidates)
        selected = min(candidates, key=_single_key)
        algorithm = {
            "name": "particle_swarm_optimization",
            "mode": "single_objective",
            "proposal": "bounded PSO velocity with personal and global best memories",
            "environmental_selection": "personal-best retention",
            "population_size": size,
            "generations": steps,
            "structured_seed_count": min(len(supplied_seeds), size - 1),
            "initial_velocity_fraction": velocity_fraction,
        }
    else:
        if not resumed:
            _write_pso_checkpoint(
                checkpoint,
                contract=contract,
                contract_fingerprint=contract_fingerprint,
                mode=expected_mode,
                completed_generation=completed_generation,
                next_candidate_id=next_id,
                evaluation_count=evaluation_count,
                rng_state=rng.bit_generator.state,
                population=population,
                history=history,
                complete=completed_generation >= steps,
                budget=budget,
            )
        for generation in range(completed_generation + 1, steps + 1):
            ranked = _rank_and_crowding(population)
            inertia = 0.85 - 0.45 * ((generation - 1) / max(1, steps - 1))
            child_positions = []
            child_velocities = []
            for index, particle in enumerate(population):
                leader_a = _tournament(rng, ranked)
                leader_b = _tournament(rng, ranked)
                r1 = rng.random(len(initial)); r2 = rng.random(len(initial))
                velocity = (
                    inertia * particle["velocity"]
                    + 1.55 * r1 * (leader_a["position"] - particle["position"])
                    + 1.55 * r2 * (leader_b["position"] - particle["position"])
                )
                mutation_mask = rng.random(len(initial)) < (1.0 / len(initial))
                velocity += mutation_mask * rng.normal(0.0, 0.04, len(initial)) * width
                velocity = np.clip(velocity, -0.35 * width, 0.35 * width)
                position, velocity = _bounded_move(
                    particle["position"], velocity, lower, upper
                )
                child_positions.append(position)
                child_velocities.append(velocity)
                next_id += 1
            children = evaluate_population(
                child_positions,
                child_velocities,
                list(range(next_id - size, next_id)),
                generation,
            )
            evaluation_count += size
            # The defining multi-objective step: population identities do not
            # guarantee survival.  Parents and PSO children compete together.
            population = _environmental_select([*population, *children], size)
            history.append(_history_row(generation, population, objectives))
            _publish_progress(
                progress_callback,
                "generation",
                {**history[-1], "total_generations": steps},
            )
            completed_generation = generation
            _write_pso_checkpoint(
                checkpoint,
                contract=contract,
                contract_fingerprint=contract_fingerprint,
                mode=expected_mode,
                completed_generation=completed_generation,
                next_candidate_id=next_id,
                evaluation_count=evaluation_count,
                rng_state=rng.bit_generator.state,
                population=population,
                history=history,
                complete=completed_generation >= steps,
                budget=budget,
            )
        front = _first_front(population)
        candidates = population
        selected = _representative(front or population, selection)
        algorithm = {
            "name": "pso_proposal_nondominated_population_optimization",
            "mode": "multi_objective",
            "proposal": "bounded PSO leader velocity plus sparse mutation",
            "environmental_selection": (
                "parent_offspring_nondominated_sorting_and_crowding"
            ),
            "population_size": size,
            "generations": steps,
            "structured_seed_count": min(len(supplied_seeds), size - 1),
            "initial_velocity_fraction": velocity_fraction,
            "survivor_identity": "not tied to original particle identities",
        }
    result = _result(
        algorithm=algorithm,
        candidates=candidates,
        front=front,
        selected=selected,
        names=names,
        objectives=objectives,
        lower=lower,
        upper=upper,
        history=history,
        complete=True,
        evaluation_count=evaluation_count,
    )
    result["checkpoint"] = {
        "enabled": checkpoint is not None,
        "path": str(checkpoint) if checkpoint is not None else None,
        "resumed": resumed,
        "completed_generation": completed_generation,
    }
    result["budget"] = budget
    result["evaluation_cache"] = {
        "requested": cache_evaluations,
        "enabled": evaluation_cache is not None,
        "directory": str(evaluation_cache) if evaluation_cache is not None else None,
        "supplied_directory_ignored": not cache_evaluations and evaluation_cache_directory is not None,
        "generation_checkpoints_retained": checkpoint is not None,
        "hits": cache_hits,
        "writes": cache_writes,
    }
    return result


def _publish_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    phase: str,
    payload: Mapping[str, Any],
) -> None:
    """Publish best-effort optimizer progress without changing its numerics."""

    if callback is None:
        return
    try:
        callback({"phase": phase, **dict(payload)})
    except Exception:
        # A failed UI/progress sink must never invalidate a physics evaluation.
        return


def _validated_seed_positions(
    supplied: Sequence[Sequence[float]] | None,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    dimensions: int,
) -> list[np.ndarray]:
    if supplied is None:
        return []
    positions: list[np.ndarray] = []
    try:
        items = iter(supplied)
    except TypeError as exc:
        raise OptimizerInputError("seed_positions must be a sequence of numeric positions") from exc
    for index, item in enumerate(items):
        position = _numeric_array(item, f"seed_positions[{index}] must contain numeric values")
        if position.shape != (dimensions,):
            raise OptimizerInputError(
                f"seed_positions[{index}] must contain {dimensions} values"
            )
        if not np.all(np.isfinite(position)):
            raise OptimizerInputError(f"seed_positions[{index}] must be finite")
        if np.any(position < lower) or np.any(position > upper):
            raise OptimizerInputError(f"seed_positions[{index}] lies outside the bounds")
        positions.append(position.copy())
    return positions


def fast_nondominated_sort(candidates: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Return constrained non-dominated fronts as candidate indices."""
    dominated: list[list[int]] = [[] for _ in candidates]
    counts = [0] * len(candidates)
    fronts: list[list[int]] = [[]]
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if _dominates(candidates[left], candidates[right]):
                dominated[left].append(right); counts[right] += 1
            elif _dominates(candidates[right], candidates[left]):
                dominated[right].append(left); counts[left] += 1
        if counts[left] == 0:
            fronts[0].append(left)
    level = 0
    while level < len(fronts) and fronts[level]:
        next_front: list[int] = []
        for left in fronts[level]:
            for right in dominated[left]:
                counts[right] -= 1
                if counts[right] == 0:
                    next_front.append(right)
        if next_front:
            fronts.append(next_front)
        level += 1
    return fronts if fronts and fronts[0] else []


def _validated_inputs(
    bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    objective_names: Sequence[str],
    initial_position: Sequence[float],
    variable_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray]:
    if variable_names is not None:
        variable_names = _names(variable_names, "variable_names must be unique non-empty strings and match the bound count")
    if isinstance(bounds, Mapping):
        names = variable_names if variable_names is not None else _names(bounds, "named bounds must have unique non-empty string keys")
        if set(names) != set(bounds):
            raise OptimizerInputError("named bounds must have exactly the unique variable_names keys")
        bounds = [bounds[name] for name in names]
        variable_names = names
    array = _numeric_array(bounds, "bounds must contain numeric [lower, upper] pairs, either in variable_names order or keyed by those names")
    if array.ndim != 2 or array.shape[1] != 2 or not len(array):
        raise OptimizerInputError("bounds must be a non-empty sequence of [lower, upper] pairs")
    lower, upper = array[:, 0], array[:, 1]
    if not np.all(np.isfinite(array)) or np.any(lower >= upper):
        raise OptimizerInputError("every bound must be finite and satisfy lower < upper")
    names = tuple(variable_names) if variable_names is not None else tuple(f"x{index}" for index in range(len(lower)))
    if isinstance(initial_position, Mapping):
        if set(initial_position) != set(names):
            raise OptimizerInputError("named initial_position must have exactly the variable_names keys")
        initial_position = [initial_position[name] for name in names]
    initial = _numeric_array(initial_position, "initial_position must contain one finite numeric value per bound")
    if initial.shape != lower.shape or not np.all(np.isfinite(initial)):
        raise OptimizerInputError("initial_position must contain one finite value per bound")
    if np.any(initial < lower) or np.any(initial > upper):
        raise OptimizerInputError("initial_position must be inside all bounds")
    objectives = tuple(item.strip() for item in _names(objective_names, "objective_names must be non-empty and unique"))
    if not objectives or any(not item for item in objectives) or len(set(objectives)) != len(objectives):
        raise OptimizerInputError("objective_names must be non-empty and unique")
    if len(names) != len(initial) or len(set(names)) != len(names):
        raise OptimizerInputError("variable_names must be unique and match the bound count")
    return lower, upper, names, objectives, initial


def _candidate(
    position: np.ndarray,
    velocity: np.ndarray,
    evaluate: Evaluator,
    objective_names: tuple[str, ...],
    candidate_id: int,
) -> dict[str, Any]:
    raw = dict(evaluate(np.asarray(position, dtype=float).copy()))
    return _candidate_from_raw(
        position, velocity, raw, objective_names, candidate_id
    )


def _candidate_from_raw(
    position: np.ndarray,
    velocity: np.ndarray,
    evaluation: Mapping[str, Any],
    objective_names: tuple[str, ...],
    candidate_id: int,
) -> dict[str, Any]:
    raw = dict(evaluation)
    objective_map = raw.get("objectives", raw.get("objective_values", {}))
    objective_map = objective_map if isinstance(objective_map, Mapping) else {}
    values = []
    finite = True
    for name in objective_names:
        try:
            value = float(objective_map[name])
        except (KeyError, TypeError, ValueError):
            value = math.inf
        finite = finite and math.isfinite(value)
        values.append(value)
    violations = raw.get("constraint_violations", {})
    violations = violations if isinstance(violations, Mapping) else {}
    clean_violations: dict[str, float] = {}
    for name, value in violations.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.inf
        clean_violations[str(name)] = max(0.0, number) if math.isfinite(number) else math.inf
    supplied = raw.get("constraint_violation")
    if supplied is None:
        violation = sum(clean_violations.values())
    else:
        try:
            violation = max(0.0, float(supplied))
        except (TypeError, ValueError):
            violation = math.inf
    if not math.isfinite(violation):
        violation = math.inf
    feasible = bool(raw.get("feasible", violation <= 1e-12)) and finite and violation <= 1e-12
    return {
        "candidate_id": f"candidate-{candidate_id:06d}",
        "position": np.asarray(position, dtype=float).copy(),
        "velocity": np.asarray(velocity, dtype=float).copy(),
        "objective_values": tuple(values),
        "constraint_violations": clean_violations,
        "constraint_violation": violation,
        "feasible": feasible,
        "raw": raw,
    }


def _copy_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **candidate,
        "position": np.asarray(candidate["position"], dtype=float).copy(),
        "velocity": np.asarray(candidate["velocity"], dtype=float).copy(),
        "objective_values": tuple(candidate["objective_values"]),
        "constraint_violations": dict(candidate["constraint_violations"]),
        "raw": dict(candidate["raw"]),
    }


def _single_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    values = tuple(float(item) for item in candidate["objective_values"])
    if candidate.get("feasible") is True:
        return (0.0, values[0], *values[1:])
    return (1.0, float(candidate.get("constraint_violation", math.inf)), *values)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_feasible = left.get("feasible") is True
    right_feasible = right.get("feasible") is True
    if left_feasible != right_feasible:
        return left_feasible
    left_violation = float(left.get("constraint_violation", math.inf))
    right_violation = float(right.get("constraint_violation", math.inf))
    if not left_feasible:
        return left_violation < right_violation - 1e-15
    left_values = np.asarray(left["objective_values"], dtype=float)
    right_values = np.asarray(right["objective_values"], dtype=float)
    return bool(np.all(left_values <= right_values) and np.any(left_values < right_values))


def _first_front(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    fronts = fast_nondominated_sort(candidates)
    return [candidates[index] for index in fronts[0]] if fronts else []


def _crowding(candidates: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> dict[int, float]:
    distance = {index: 0.0 for index in indices}
    if len(indices) <= 2:
        return {index: math.inf for index in indices}
    objective_count = len(candidates[indices[0]]["objective_values"])
    for objective in range(objective_count):
        ordered = sorted(indices, key=lambda index: candidates[index]["objective_values"][objective])
        distance[ordered[0]] = distance[ordered[-1]] = math.inf
        low = float(candidates[ordered[0]]["objective_values"][objective])
        high = float(candidates[ordered[-1]]["objective_values"][objective])
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            continue
        for offset in range(1, len(ordered) - 1):
            if math.isinf(distance[ordered[offset]]):
                continue
            previous = float(candidates[ordered[offset - 1]]["objective_values"][objective])
            following = float(candidates[ordered[offset + 1]]["objective_values"][objective])
            distance[ordered[offset]] += (following - previous) / (high - low)
    return distance


def _rank_and_crowding(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ranked: list[Mapping[str, Any]] = []
    for rank, front in enumerate(fast_nondominated_sort(candidates)):
        crowding = _crowding(candidates, front)
        for index in front:
            ranked.append({"candidate": candidates[index], "rank": rank, "crowding": crowding[index]})
    return ranked


def _environmental_select(
    candidates: Sequence[Mapping[str, Any]], size: int
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for front in fast_nondominated_sort(candidates):
        if len(selected) + len(front) <= size:
            selected.extend(candidates[index] for index in front)
            continue
        crowding = _crowding(candidates, front)
        ordered = sorted(front, key=lambda index: crowding[index], reverse=True)
        selected.extend(candidates[index] for index in ordered[: size - len(selected)])
        break
    return selected


def _tournament(
    rng: np.random.Generator, ranked: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    left, right = rng.integers(0, len(ranked), size=2)
    a, b = ranked[int(left)], ranked[int(right)]
    key_a = (int(a["rank"]), -float(a["crowding"]))
    key_b = (int(b["rank"]), -float(b["crowding"]))
    return (a if key_a <= key_b else b)["candidate"]


def _representative(
    candidates: Sequence[Mapping[str, Any]], selection: str
) -> Mapping[str, Any]:
    feasible = [item for item in candidates if item.get("feasible") is True]
    pool = feasible or list(candidates)
    if not pool:
        raise RuntimeError("optimizer produced no candidates")
    if not feasible:
        return min(pool, key=_single_key)
    if selection == "first_objective" or len(pool[0]["objective_values"]) == 1:
        return min(pool, key=lambda item: float(item["objective_values"][0]))
    values = np.asarray([item["objective_values"] for item in pool], dtype=float)
    low = np.min(values, axis=0); high = np.max(values, axis=0)
    normalized = (values - low) / np.where(high > low, high - low, 1.0)
    index = int(np.argmin(np.sqrt(np.mean(normalized ** 2, axis=1))))
    return pool[index]


def _history_row(
    generation: int,
    candidates: Sequence[Mapping[str, Any]],
    objective_names: Sequence[str],
) -> dict[str, Any]:
    feasible = [item for item in candidates if item.get("feasible") is True]
    source = feasible or candidates
    best = {
        name: min(float(item["objective_values"][index]) for item in source)
        for index, name in enumerate(objective_names)
    }
    return {
        "generation": generation,
        "population_size": len(candidates),
        "feasible_count": len(feasible),
        "front_size": len(_first_front(candidates)),
        "best": best,
        "minimum_constraint_violation": min(
            float(item.get("constraint_violation", math.inf)) for item in candidates
        ),
    }


def _bounded_move(
    position: np.ndarray,
    velocity: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    proposal = position + velocity
    clipped = np.clip(proposal, lower, upper)
    hit = clipped != proposal
    velocity = velocity.copy()
    velocity[hit] *= -0.25
    return clipped, velocity


def _latin_hypercube(
    rng: np.random.Generator, size: int, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    unit = np.empty((size, len(lower)), dtype=float)
    for dimension in range(len(lower)):
        unit[:, dimension] = (rng.permutation(size) + rng.random(size)) / size
    return lower + unit * (upper - lower)


def _serial_candidate(
    candidate: Mapping[str, Any], names: Sequence[str], objectives: Sequence[str]
) -> dict[str, Any]:
    raw = dict(candidate.get("raw", {}))
    raw.pop("objectives", None); raw.pop("objective_values", None)
    raw.pop("constraint_violations", None); raw.pop("constraint_violation", None)
    raw.pop("feasible", None)
    return {
        "candidate_id": candidate["candidate_id"],
        "position": [float(value) for value in candidate["position"]],
        "variables": {
            name: float(value) for name, value in zip(names, candidate["position"])
        },
        "objectives": {
            name: float(value)
            for name, value in zip(objectives, candidate["objective_values"])
        },
        "feasible": bool(candidate.get("feasible")),
        "constraint_violation": float(candidate.get("constraint_violation", math.inf)),
        "constraint_violations": dict(candidate.get("constraint_violations", {})),
        "metrics": raw,
    }


def _result(
    *,
    algorithm: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    front: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    names: Sequence[str],
    objectives: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
    history: Sequence[Mapping[str, Any]],
    complete: bool,
    evaluation_count: int,
) -> dict[str, Any]:
    primary = str(objectives[0])
    initial_best = (
        float(history[0].get("best", {}).get(primary, math.inf))
        if history
        else math.inf
    )
    final_best = (
        float(history[-1].get("best", {}).get(primary, math.inf))
        if history
        else math.inf
    )
    absolute_improvement = initial_best - final_best
    improvement_threshold = 1.0e-12 * max(1.0, abs(initial_best))
    improved = math.isfinite(absolute_improvement) and absolute_improvement > improvement_threshold
    initial_feasible = int(history[0].get("feasible_count", 0)) if history else 0
    final_feasible = int(history[-1].get("feasible_count", 0)) if history else 0
    return {
        "algorithm": dict(algorithm),
        "evaluated_candidates": int(evaluation_count),
        "variable_names": list(names),
        "bounds": {
            name: [float(low), float(high)]
            for name, low, high in zip(names, lower, upper)
        },
        "objectives": list(objectives),
        "population": [_serial_candidate(item, names, objectives) for item in candidates],
        "pareto_candidates": [_serial_candidate(item, names, objectives) for item in front],
        "best_candidate": _serial_candidate(selected, names, objectives),
        "convergence": {
            "complete": bool(complete),
            "history": list(history),
            "primary_objective": primary,
            "generation_zero_best": initial_best,
            "final_generation_best": final_best,
            "absolute_improvement_after_generation_zero": absolute_improvement,
            "improved_after_generation_zero": improved,
            "front_stagnated_after_initialization": not improved,
            "initial_feasible_count": initial_feasible,
            "final_feasible_count": final_feasible,
            "initial_feasible_fraction": (
                initial_feasible / len(candidates) if candidates else 0.0
            ),
            "population_per_dimension": (
                len(candidates) / len(names) if names else 0.0
            ),
            # This kernel records what happened in one campaign.  It never
            # upgrades stagnation into a scientific saturation claim.
            "saturation_demonstrated": False,
        },
    }
