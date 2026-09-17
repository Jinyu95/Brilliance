"""Scientist-configured population-search budgets, independent of physics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


_DEFAULTS = {"minimum_population": 500, "minimum_population_per_knob": 10,
             "minimum_generations": 80}


def resolve_population_budget(
    dimensions: int, population_size: int | None, generations: int | None,
    *, resumed_population_size: int | None = None, completed_generation: int = 0,
    saved_generation_target: int | None = None, checkpoint_complete: bool = False,
) -> dict[str, Any]:
    """Resolve requested/effective budget without executing an evaluation.

    Explicit small standalone calls remain possible for kernel tests. Managed
    research takes production floors only from the scientist's runtime TOML,
    never from a model-selected ``pilot`` switch. An existing swarm is resumed
    at its exact saved size; changing it requires a separate named campaign.
    """
    managed = bool(os.environ.get("BRILLIANCE_RUN_OUTPUT"))
    policy = dict(_DEFAULTS)
    config_path = None
    if managed:
        root = Path(os.environ.get("BRILLIANCE_SOURCE_ROOT") or Path(__file__).resolve().parent.parent)
        config_path = Path(os.environ.get("BRILLIANCE_RUNTIME_CONFIG") or
                           root / "brilliance" / "config" / "research_runtime.toml")
        if not config_path.is_absolute():
            config_path = root / config_path
        with config_path.open("rb") as stream:
            payload = tomllib.load(stream).get("optimization", {})
        if not isinstance(payload, dict):
            raise ValueError("Runtime [optimization] must be a TOML table")
        for name, default in _DEFAULTS.items():
            value = payload.get(name, default)
            minimum = 4 if name == "minimum_population" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Runtime [optimization].{name} must be an integer >= {minimum}")
            policy[name] = value
    population_floor = max(policy["minimum_population"],
                           policy["minimum_population_per_knob"] * dimensions)
    size = population_floor if population_size is None else population_size
    steps = policy["minimum_generations"] if generations is None else generations
    if managed and resumed_population_size is None:
        size = max(size, population_floor)
        steps = max(steps, policy["minimum_generations"])
    notes = []
    if resumed_population_size is not None:
        if size != resumed_population_size:
            notes.append("Continue the exact archived swarm without resizing or discarding history. "
                         "Use a new campaign/checkpoint to change population size.")
        size = resumed_population_size
        # Repairing reporting must never buy another production campaign. The
        # saved target is authoritative; only an explicitly larger requested
        # generation target extends it. Old checkpoints may not contain budget
        # metadata, so a completed one ends at its recorded generation.
        saved_target = (saved_generation_target if saved_generation_target is not None
                        else completed_generation if checkpoint_complete
                        else generations if generations is not None else completed_generation)
        steps = max(completed_generation, saved_target,
                    generations if generations is not None else saved_target)
        if saved_generation_target is None and not checkpoint_complete and generations is None:
            notes.append("The old checkpoint has no recorded generation target; no additional "
                         "generations were inferred. Specify a larger target to extend it.")
    below_floor = size < population_floor or steps < policy["minimum_generations"]
    if managed and below_floor:
        notes.append("This is a preserved smaller prior campaign, not a production-sized new search. "
                     "Use a new campaign/checkpoint for a new production search.")
    adjusted = (population_size is not None and size != population_size) or (
        generations is not None and steps != generations)
    summary = f"PSO budget: {size} particles for {steps} generations ({dimensions} knobs)."
    if adjusted:
        summary += (f" Requested {population_size if population_size is not None else 'default'}"
                    f" particles / {generations if generations is not None else 'default'} generations;")
        summary += " preserved checkpoint size." if resumed_population_size is not None else " applied configured research floors."
    if notes:
        summary += " " + " ".join(notes)
    return {
        "managed_research": managed,
        "configuration": str(config_path) if config_path is not None else None,
        "dimensions": dimensions,
        "requested": {"population_size": population_size, "generations": generations},
        "effective": {"population_size": size, "generations": steps},
        "policy": {**policy, "population_floor": population_floor},
        "adjusted": adjusted,
        "resumed_population_size": resumed_population_size,
        "saved_generation_target": saved_generation_target,
        "below_production_floor": below_floor,
        "notes": notes,
        "summary": summary,
    }
