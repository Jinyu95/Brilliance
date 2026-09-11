"""Matplotlib presentation for archived optimizer histories.

Optimizer results are scientific data, not prose.  This module turns the typed
history already archived by Brilliance into a figure without rerunning JuTrack,
matching, or the optimizer.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


class OptimizationPlotUnavailable(LookupError):
    """Raised when a project has no plottable archived optimization history."""


def save_optimization_convergence_plot(
    result: Mapping[str, Any],
    output_directory: Path | str,
    *,
    title: str = "Optimization convergence",
    source: str = "archived optimizer evidence",
) -> dict[str, Any]:
    """Render one optimizer result with Matplotlib and export its numeric rows."""

    rows = optimization_history_rows(result)
    if not rows:
        raise OptimizationPlotUnavailable("The optimizer result contains no numeric history rows")

    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifact_root = next(
        (parent for parent in (destination, *destination.parents) if parent.name == "artifacts"),
        None,
    )
    matplotlib_cache = (
        artifact_root.parent / "runtime" / "matplotlib"
        if artifact_root is not None
        else destination / ".matplotlib"
    )
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    generations = [row["generation"] for row in rows]
    emittance = [row.get("emittance_pm") for row in rows]
    beta = [row.get("maximum_beta_m") for row in rows]
    front_size = [row.get("front_size") for row in rows]

    baseline = _finite(_mapping(result.get("baseline")).get("emittance_pm"))
    selected = _finite(_mapping(result.get("selected")).get("emittance_pm"))
    final_emittance = selected if selected is not None else _last_finite(emittance)
    first_emittance = _first_finite(emittance)

    figure, (axis_emittance, axis_diagnostics) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.0]},
        constrained_layout=True,
    )
    figure.suptitle(title, fontsize=15, fontweight="bold")

    _plot_numeric_series(
        axis_emittance,
        generations,
        emittance,
        color="#1d4ed8",
        label="Best feasible emittance",
        linewidth=2.2,
    )
    if baseline is not None:
        axis_emittance.axhline(
            baseline,
            color="#6b7280",
            linewidth=1.3,
            linestyle="--",
            label=f"Baseline: {baseline:.3f} pm rad",
        )
    if final_emittance is not None:
        final_generation = generations[-1]
        axis_emittance.scatter(
            [final_generation],
            [final_emittance],
            color="#dc2626",
            edgecolor="white",
            linewidth=0.8,
            s=58,
            zorder=5,
            label=f"Latest candidate: {final_emittance:.3f} pm rad",
        )
        axis_emittance.annotate(
            f"{final_emittance:.3f} pm rad",
            (final_generation, final_emittance),
            xytext=(-8, 11),
            textcoords="offset points",
            ha="right",
            color="#991b1b",
            fontsize=9,
        )
    axis_emittance.set_ylabel(r"Natural emittance $\epsilon_x$ (pm rad)")
    axis_emittance.grid(True, alpha=0.25)
    axis_emittance.legend(loc="best", frameon=False)

    _plot_numeric_series(
        axis_diagnostics,
        generations,
        beta,
        color="#059669",
        label=r"Best-candidate $\beta_{max}$",
        linewidth=1.8,
    )
    axis_diagnostics.set_ylabel(r"$\beta_{max}$ (m)", color="#047857")
    axis_diagnostics.tick_params(axis="y", labelcolor="#047857")
    axis_diagnostics.grid(True, alpha=0.25)
    diagnostic_handles, diagnostic_labels = axis_diagnostics.get_legend_handles_labels()

    if any(value is not None for value in front_size):
        axis_front = axis_diagnostics.twinx()
        _plot_numeric_series(
            axis_front,
            generations,
            front_size,
            color="#7c3aed",
            label="Pareto-front size",
            linewidth=1.5,
            drawstyle="steps-post",
        )
        axis_front.set_ylabel("Pareto-front size", color="#6d28d9")
        axis_front.tick_params(axis="y", labelcolor="#6d28d9")
        axis_front.yaxis.set_major_locator(MaxNLocator(integer=True))
        front_handles, front_labels = axis_front.get_legend_handles_labels()
        diagnostic_handles += front_handles
        diagnostic_labels += front_labels
    axis_diagnostics.legend(
        diagnostic_handles,
        diagnostic_labels,
        loc="best",
        frameon=False,
    )
    axis_diagnostics.set_xlabel("Generation")
    axis_diagnostics.xaxis.set_major_locator(MaxNLocator(integer=True))

    population = _int_or_none(_mapping(result.get("optimization_summary")).get("population"))
    evaluated = _int_or_none(result.get("evaluated_candidates"))
    detail = [source]
    if population is not None:
        detail.append(f"population {population}")
    if evaluated is not None:
        detail.append(f"{evaluated:,} candidate evaluations")
    if baseline is not None and final_emittance is not None and baseline > 0.0:
        improvement = 100.0 * (baseline - final_emittance) / baseline
        detail.append(f"{improvement:.1f}% below baseline")
    axis_emittance.set_title("  |  ".join(detail), fontsize=8.5, color="#4b5563", pad=7)

    png_path = destination / "optimization-convergence.png"
    svg_path = destination / "optimization-convergence.svg"
    csv_path = destination / "optimization-history.csv"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    _write_history_csv(csv_path, rows)

    return {
        "png": str(png_path),
        "svg": str(svg_path),
        "csv": str(csv_path),
        "history_rows": len(rows),
        "generation_start": generations[0],
        "generation_end": generations[-1],
        "baseline_emittance_pm": baseline,
        "generation_zero_emittance_pm": first_emittance,
        "selected_emittance_pm": final_emittance,
        "relative_improvement": (
            (baseline - final_emittance) / baseline
            if baseline is not None and final_emittance is not None and baseline > 0.0
            else None
        ),
        "population": population,
        "evaluated_candidates": evaluated,
        "selected_lattice_id": str(_mapping(result.get("selected")).get("lattice_id", "")),
    }


def render_latest_optimization_plot(project_directory: Path | str) -> dict[str, Any]:
    """Find and render the newest archived optimizer execution in a project."""

    project = Path(project_directory).resolve()
    artifact_root = project / "workspace" / "artifacts"
    candidates = sorted(
        artifact_root.glob("expert-execution-*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    for path in candidates:
        try:
            execution = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result = execution.get("result") if isinstance(execution, Mapping) else None
        if not isinstance(result, Mapping) or not optimization_history_rows(result):
            continue
        output_directory = _optimization_output_directory(artifact_root, result, path.stem)
        rendered = save_optimization_convergence_plot(
            result,
            output_directory,
            title="MBA emittance optimization convergence",
            source="Archived JuTrack/MOPSO evidence",
        )
        rendered.update(
            {
                "execution_path": str(path),
                "execution_id": str(execution.get("execution_id", path.stem)),
                "tool_id": str(execution.get("tool_id", "")),
                "passed": bool(result.get("passed")),
            }
        )
        return rendered
    raise OptimizationPlotUnavailable(
        "No archived optimizer execution with numeric convergence history is available"
    )


def optimization_history_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize optimizer history and discard incomplete values safely."""

    convergence = result.get("convergence")
    history: Any = convergence.get("history") if isinstance(convergence, Mapping) else convergence
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            continue
        generation = _finite(item.get("generation"))
        if generation is None:
            generation = float(index)
        best = _mapping(item.get("best"))
        row = {
            "generation": int(generation),
            "emittance_pm": _finite(best.get("emittance_pm")),
            "maximum_beta_m": _finite(best.get("maximum_beta_m", best.get("max_beta_m"))),
            "normalized_strength_rms": _finite(best.get("normalized_strength_rms")),
            "feasible_count": _int_or_none(item.get("feasible_count")),
            "population_size": _int_or_none(item.get("population_size")),
            "front_size": _int_or_none(item.get("front_size")),
            "minimum_constraint_violation": _finite(item.get("minimum_constraint_violation")),
        }
        if any(
            row[key] is not None
            for key in ("emittance_pm", "maximum_beta_m", "normalized_strength_rms")
        ):
            rows.append(row)
    return sorted(rows, key=lambda row: row["generation"])


def _optimization_output_directory(
    artifact_root: Path, result: Mapping[str, Any], fallback: str
) -> Path:
    progress = _mapping(result.get("artifacts")).get("optimization_progress")
    if progress:
        candidate = Path(str(progress)).resolve().parent
        try:
            candidate.relative_to(artifact_root.resolve())
        except ValueError:
            pass
        else:
            return candidate
    return artifact_root / "plots" / fallback


def _plot_numeric_series(axis, x_values, y_values, **kwargs) -> None:
    pairs = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
    if not pairs:
        return
    axis.plot(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        marker="o" if len(pairs) <= 60 else None,
        markersize=3.2,
        **kwargs,
    )


def _write_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "generation",
        "emittance_pm",
        "maximum_beta_m",
        "normalized_strength_rms",
        "feasible_count",
        "population_size",
        "front_size",
        "minimum_constraint_violation",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    number = _finite(value)
    return int(number) if number is not None else None


def _first_finite(values: Sequence[float | None]) -> float | None:
    return next((value for value in values if value is not None), None)


def _last_finite(values: Sequence[float | None]) -> float | None:
    return next((value for value in reversed(values) if value is not None), None)
