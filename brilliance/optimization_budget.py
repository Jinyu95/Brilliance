"""Explicit numerical budgets, with defaults rather than imposed research effort."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


_DEFAULTS = {"default_population": 64, "default_generations": 40}


def resolve_population_budget(
    dimensions: int, population_size: int | None, generations: int | None,
    *, resumed_population_size: int | None = None, completed_generation: int = 0,
    saved_generation_target: int | None = None, checkpoint_complete: bool = False,
) -> dict[str, Any]:
    """Honor explicit budgets; resume the exact saved swarm and generation target.

    Defaults are computational conveniences, not physical requirements or evidence
    of convergence. Legacy minimum_* config keys are read as defaults only. A
    reporting-only resume never silently schedules additional generations.
    """
    managed = bool(os.environ.get("BRILLIANCE_RUN_OUTPUT"))
    policy = dict(_DEFAULTS)
    config_path = None
    notes = []
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
        for name, legacy in (("default_population", "minimum_population"),
                             ("default_generations", "minimum_generations")):
            value = payload.get(name, payload.get(legacy, policy[name]))
            minimum = 4 if name == "default_population" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Runtime [optimization].{name} must be an integer >= {minimum}")
            policy[name] = value
        if any(name.startswith("minimum_") for name in payload):
            notes.append("Legacy minimum_* settings are defaults, not enforced floors; migrate to default_population/default_generations.")
    size = policy["default_population"] if population_size is None else population_size
    steps = policy["default_generations"] if generations is None else generations
    if resumed_population_size is not None:
        if size != resumed_population_size:
            notes.append("Continue the exact archived swarm. Use a new campaign/checkpoint to change population size.")
        size = resumed_population_size
        saved_target = (saved_generation_target if saved_generation_target is not None
                        else completed_generation if checkpoint_complete
                        else generations if generations is not None else completed_generation)
        steps = max(completed_generation, saved_target,
                    generations if generations is not None else saved_target)
        if saved_generation_target is None and not checkpoint_complete and generations is None:
            notes.append("No recorded generation target; specify a larger target to extend this checkpoint.")
    adjusted = (population_size is not None and size != population_size) or (
        generations is not None and steps != generations)
    summary = f"PSO budget: {size} particles for {steps} generations ({dimensions} knobs)."
    if notes:
        summary += " " + " ".join(notes)
    return {
        "managed_research": managed,
        "configuration": str(config_path) if config_path is not None else None,
        "dimensions": dimensions,
        "requested": {"population_size": population_size, "generations": generations},
        "effective": {"population_size": size, "generations": steps},
        "policy": policy,
        "adjusted": adjusted,
        "resumed_population_size": resumed_population_size,
        "saved_generation_target": saved_generation_target,
        "notes": notes,
        "summary": summary,
    }
