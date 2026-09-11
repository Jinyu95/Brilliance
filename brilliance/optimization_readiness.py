"""End-to-end readiness probe for generic JuTrack population optimization.

This is deliberately a software contract probe, not a lattice-design template.
It verifies that process-local JuTrack instances can evaluate a population and
that dedicated quadrupole knobs restore two dispersion coordinates inside each
outer objective call.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

from . import accelerator_api as api
from .lattice import LatticeIR
from .lattice_optimization import (
    JuTrackLatticeEditor,
    compile_family_parameterization,
    compile_lattice_parameterization,
    jutrack_linear_metrics,
    optimize_lattice_parameters,
)
from .research_environment import ResearchEnvironment


def _probe_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period in range(8):
        prefix = f"OPT_CHECK:P{period + 1:02d}:"
        specifications = (
            ("DRIFT", "D1", 0.5, None, None, "OUTER_LENGTH"),
            ("KQUAD", "QF", 0.2, 2.0, None, "MATCH_QF"),
            ("DRIFT", "D2", 0.5, None, None, None),
            ("SBEND", "B1", 1.0, None, math.pi / 8.0, None),
            ("DRIFT", "D3", 0.5, None, None, None),
            ("KQUAD", "QD", 0.2, -3.0, None, "MATCH_QD"),
            ("DRIFT", "D4", 0.5, None, None, None),
            ("SBEND", "B2", 1.0, None, math.pi / 8.0, None),
        )
        for kind, name, length, k1, angle, family in specifications:
            row: dict[str, Any] = {
                "name": prefix + name,
                "kind": kind,
                "length_m": length,
                "fields": {"family": family} if family else {},
            }
            if k1 is not None:
                row["k1_m2"] = k1
            if angle is not None:
                row["angle_rad"] = angle
            rows.append(row)
    return rows


def _probe_physics_factory(
    baseline_rows: list[dict[str, Any]],
) -> tuple[Any, Any]:
    lattice_ir = LatticeIR.from_rows(
        title="Brilliance optimization readiness probe",
        energy_gev=3.0,
        periodicity=1,
        rows=baseline_rows,
        source="brilliance.optimization_readiness",
    )
    jt = api.jutrack()
    lattice = api.build_jutrack_lattice(lattice_ir)
    editor = JuTrackLatticeEditor(jt, lattice, baseline_rows)

    def metrics(rows: list[dict[str, Any]]) -> Mapping[str, Any]:
        editor.sync_rows(rows)
        return jutrack_linear_metrics(jt, lattice)

    def objectives(rows: list[dict[str, Any]]) -> Mapping[str, Any]:
        evaluated = dict(metrics(rows))
        if not evaluated.get("stable"):
            return {
                "objectives": {"beta_x_m": math.inf, "quadrupole_burden": math.inf},
                "constraint_violations": {"linear_stability": 1.0},
                "feasible": False,
                "worker_pid": os.getpid(),
            }
        quadrupole_burden = sum(
            abs(float(row.get("k1_m2", 0.0)))
            for row in rows
            if row.get("kind") == "KQUAD"
        )
        return {
            "objectives": {
                "beta_x_m": float(evaluated["beta_x_m"]),
                "quadrupole_burden": quadrupole_burden,
            },
            "constraint_violations": {},
            "feasible": True,
            "worker_pid": os.getpid(),
        }

    return metrics, objectives


def probe_optimization_runtime(
    environment: ResearchEnvironment | None = None,
    *,
    workers: int = 2,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run a real process-parallel JuTrack nested-dispersion optimization."""

    runtime = environment or ResearchEnvironment.load()
    runtime.require_active_python(operation="JuTrack optimization readiness check")
    started = time.monotonic()
    rows = _probe_rows()
    metrics, objectives = _probe_physics_factory(rows)
    baseline = dict(metrics(rows))
    if not baseline.get("stable"):
        raise RuntimeError(f"optimization probe baseline is unstable: {baseline}")
    target_dx = float(baseline["Dx_m"])
    target_dpx = float(baseline["Dpx"])
    design = compile_lattice_parameterization(
        rows,
        [
            {
                "name": "outer_drift_length_m",
                "field": "length_m",
                "element_names": [row["name"] for row in rows if row["fields"].get("family") == "OUTER_LENGTH"],
                "bounds": [0.49, 0.51],
            }
        ],
    )
    matching = compile_family_parameterization(
        rows,
        field="k1_m2",
        families=("MATCH_QF", "MATCH_QD"),
        bounds=(-5.0, 5.0),
    )
    output = Path(output_directory).resolve() if output_directory is not None else None
    checkpoint = output / "optimization-checkpoint.json" if output is not None else None
    result = optimize_lattice_parameters(
        rows,
        design,
        matching,
        metrics,
        objectives,
        match_targets={
            "Dx_m": {"target": target_dx, "tolerance": 1.0e-5},
            "Dpx": {"target": target_dpx, "tolerance": 1.0e-5},
        },
        objective_names=("beta_x_m", "quadrupole_burden"),
        population_size=4,
        generations=1,
        seed=17,
        minimum_population_per_knob=0.0,
        minimum_production_generations=0,
        worker_physics_factory=_probe_physics_factory,
        parallel_workers=max(1, int(workers)),
        evaluation_batch_size=4,
        checkpoint_path=checkpoint,
    )
    selected = result.get("selected_elements")
    if not isinstance(selected, list):
        raise RuntimeError("optimization probe did not return the selected matched lattice")
    final_metrics = dict(metrics(selected))
    dx_residual = abs(float(final_metrics["Dx_m"]) - target_dx)
    dpx_residual = abs(float(final_metrics["Dpx"]) - target_dpx)
    if not final_metrics.get("stable") or dx_residual > 1.0e-5 or dpx_residual > 1.0e-5:
        raise RuntimeError(
            "optimization probe selected a lattice outside its inner dispersion contract: "
            f"stable={final_metrics.get('stable')}, dDx={dx_residual}, dDpx={dpx_residual}"
        )
    worker_pids = sorted(
        {
            int(candidate.get("metrics", {}).get("worker_pid"))
            for candidate in result.get("population", [])
            if candidate.get("metrics", {}).get("worker_pid") is not None
        }
    )
    return {
        "schema_version": "1.0",
        "status": "ready",
        "ready": True,
        "duration_s": time.monotonic() - started,
        "configured_python": str(runtime.python_executable),
        "candidate_evaluation_backend": result["inner_matching_performance"][
            "candidate_evaluation_backend"
        ],
        "parallel_workers": result["inner_matching_performance"]["parallel_workers"],
        "worker_pids": worker_pids,
        "population_size": result["algorithm"]["population_size"],
        "generations": result["algorithm"]["generations"],
        "mode": result["algorithm"]["mode"],
        "feasible_generation_zero": result["initialization"].get(
            "feasible_generation_zero"
        ),
        "inner_matching": result["inner_matching_performance"],
        "dispersion_target": {"Dx_m": target_dx, "Dpx": target_dpx},
        "selected_dispersion": {
            "Dx_m": float(final_metrics["Dx_m"]),
            "Dpx": float(final_metrics["Dpx"]),
            "Dx_residual_m": dx_residual,
            "Dpx_residual": dpx_residual,
        },
        "checkpoint": result["checkpoint"],
        "evaluation_cache": result["evaluation_cache"],
        "detail": (
            "Real JuTrack candidates were evaluated in isolated worker processes; every "
            "outer proposal was rematched to the declared two-coordinate dispersion contract."
        ),
    }

