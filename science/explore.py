"""Parameter-space exploration harness for autonomous lattice discovery.

Generates structured design prompts across a multi-dimensional parameter grid,
calls the BRILLIANCE agent pipeline for each grid point, and collects results
into a searchable database.

No pyJuTrack or Julia dependency — runs in pure Python.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Sequence

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Lattice factor F per family (dimensionless — from Sands scaling theory)
_LATTICE_FACTORS: dict[str, float] = {
    "FODO":  0.144,
    "DBA":   0.022,
    "TBA":   0.017,
    "5BA":   0.012,    # interpolated between TBA and 7BA
    "6BA":   0.0075,   # interpolated
    "7BA":   0.0046,
    "8BA":   0.0035,   # interpolated
    "9BA":   0.0028,
    "H7BA":  0.0030,   # hybrid 7BA (reverse bend)
    "H6BA":  0.0040,
    "MBA":   0.0030,
}

# Default bends per cell per family
_DEFAULT_BENDS: dict[str, int] = {
    "FODO": 2, "DBA": 2, "TBA": 3,
    "5BA": 5, "6BA": 6, "7BA": 7, "8BA": 8, "9BA": 9,
    "H7BA": 7, "H6BA": 6, "MBA": 7,
}

# ── Physical constants (same as physics_core — local copies for standalone use) ──
_Cq = 3.8319e-13       # m  — quantum diffusion constant
_M_E_EV = 0.51099895069e6   # eV


def theoretical_emittance_mrad(
    energy_gev: float, n_cells: int, n_bends_per_cell: int, family: str,
) -> float:
    """Analytical emittance floor from ε₀ = Cq·γ²·θ³·F."""
    gamma = energy_gev * 1e9 / _M_E_EV
    theta = 2.0 * math.pi / (n_cells * n_bends_per_cell)
    F = _LATTICE_FACTORS.get(family.upper(), 0.010)
    return _Cq * (gamma ** 2) * (theta ** 3) * F


@dataclass
class GridPoint:
    """One point in the design parameter space."""
    energy_gev: float
    n_cells: int
    family: str
    n_bends_per_cell: int | None = None   # None → inferred from family
    # Optional target constraints (None = let agent choose)
    target_emittance_pm: float | None = None
    target_circumference_m: float | None = None
    # Optional topology overrides
    allow_variable_dipoles: bool = False
    allow_asymmetric: bool = False
    label: str = ""

    def __post_init__(self):
        if self.n_bends_per_cell is None:
            self.n_bends_per_cell = _DEFAULT_BENDS.get(
                self.family.upper(), 2
            )

    @property
    def analytical_emittance_pm(self) -> float:
        """Theoretical emittance floor in pm·rad."""
        return theoretical_emittance_mrad(
            self.energy_gev, self.n_cells,
            self.n_bends_per_cell or 2, self.family,
        ) * 1e12

    def to_prompt(self) -> str:
        """Generate the natural-language design prompt for this grid point."""
        family = self.family.upper()
        if family == "FODO":
            prompt = (
                f"Design a {self.energy_gev} GeV FODO electron storage ring "
                f"with {self.n_cells} cells, 2 sector dipoles per cell. "
            )
        elif family in ("DBA", "TBA"):
            achr = "double-bend" if family == "DBA" else "triple-bend"
            prompt = (
                f"Design a {self.energy_gev} GeV {achr} achromat storage ring "
                f"with {self.n_cells} cells. "
            )
        elif family in ("7BA", "9BA", "5BA", "6BA", "8BA"):
            n_bends = int(family[0])
            prompt = (
                f"Design a {self.energy_gev} GeV {n_bends}-bend achromat "
                f"storage ring with {self.n_cells} superperiods "
                f"and a mirror-symmetric cell. "
            )
        elif family in ("H7BA", "H6BA"):
            n_bends = int(family[1])
            prompt = (
                f"Design a {self.energy_gev} GeV hybrid {n_bends}-bend achromat "
                f"storage ring with {self.n_cells} superperiods. "
            )
        else:
            n_bends = self.n_bends_per_cell or 4
            prompt = (
                f"Design a {self.energy_gev} GeV {family} storage ring "
                f"with {self.n_cells} cells and {n_bends} dipoles per cell. "
            )

        # Exploration levers
        if self.allow_variable_dipoles:
            prompt += (
                "Let the dipole bending angles vary freely -- do not force "
                "any fixed grading ratio. Discover the optimal angle distribution "
                "that minimizes emittance. "
            )
        if self.allow_asymmetric:
            prompt += (
                "Do not assume mirror symmetry -- allow each quadrupole on the "
                "left and right sides of the cell centre to take independent K1. "
            )

        # Compute the full set of required physics quantities
        prompt += (
            "Compute the fractional tunes, natural chromaticities, natural "
            "horizontal emittance, momentum compaction factor, radiation "
            "integrals I1–I5, damping partition numbers (verify Robinson sum "
            "Jx+Jy+Je = 4), energy loss per turn U0, energy spread, and "
            "damping times. Report the circumference. "
        )

        if self.target_emittance_pm is not None:
            prompt += (
                f"Target natural horizontal emittance below "
                f"{self.target_emittance_pm:.1f} pm·rad. "
            )
        if self.target_circumference_m is not None:
            prompt += (
                f"Target circumference around {self.target_circumference_m:.0f} m. "
            )

        return prompt.strip()


@dataclass
class GridSpec:
    """Specification for a parameter sweep."""
    energies_gev: list[float] = field(default_factory=lambda: [3.0, 6.0])
    n_cells_values: list[int] = field(default_factory=lambda: [16, 20, 24, 32])
    families: list[str] = field(default_factory=lambda: ["FODO", "DBA", "7BA"])
    n_bends_overrides: dict[str, int] | None = None
    allow_variable_dipoles: bool = False
    allow_asymmetric: bool = False
    target_emittance_pm: float | None = None
    target_circumference_m: float | None = None

    def generate(self) -> list[GridPoint]:
        """Generate all grid points as a flattened list."""
        points: list[GridPoint] = []
        bends_override = self.n_bends_overrides or {}
        for e, nc, fam in product(
            self.energies_gev, self.n_cells_values, self.families,
        ):
            nb = bends_override.get(fam)
            points.append(GridPoint(
                energy_gev=e,
                n_cells=nc,
                family=fam,
                n_bends_per_cell=nb,
                allow_variable_dipoles=self.allow_variable_dipoles,
                allow_asymmetric=self.allow_asymmetric,
                target_emittance_pm=self.target_emittance_pm,
                target_circumference_m=self.target_circumference_m,
            ))
        return points


@dataclass
class ExplorationConfig:
    """Top-level configuration for an exploration run."""
    grid: GridSpec
    n_runs_per_point: int = 3           # repeat each point for statistics
    max_messages: int = 30               # per-run message budget
    model: str = ""                      # LLM model override (empty = from env)
    output_dir: Path | None = None
    dry_run: bool = False                # print prompts without running
    resume: bool = True                  # skip points with existing results

    def __post_init__(self):
        if self.output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = _PROJECT_ROOT / "science" / "results" / ts
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class DesignSpaceExplorer:
    """Orchestrates a parameter-space sweep using the BRILLIANCE agent pipeline."""

    def __init__(self, config: ExplorationConfig):
        self.config = config
        self.points = config.grid.generate()
        self._results_db: Path = config.output_dir / "results.jsonl"
        self._manifest_path: Path = config.output_dir / "manifest.json"

    # ── public API ────────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Execute the full sweep.  Returns a summary dict."""
        self._save_manifest()

        all_summaries: list[dict[str, Any]] = []
        total = len(self.points) * self.config.n_runs_per_point
        done = 0

        for point in self.points:
            for run_idx in range(self.config.n_runs_per_point):
                run_label = f"{point.family}_{point.energy_gev:.0f}GeV_{point.n_cells}c_r{run_idx}"
                point.label = run_label

                # Resume support: skip existing
                if self.config.resume and self._has_result(run_label):
                    done += 1
                    continue

                if self.config.dry_run:
                    print(f"[dry] {run_label}")
                    print(f"      {point.to_prompt()[:120]}...")
                    done += 1
                    continue

                print(
                    f"\n{'='*60}\n"
                    f"[{done + 1}/{total}]  {run_label}\n"
                    f"  family={point.family}  E={point.energy_gev:.1f} GeV  "
                    f"cells={point.n_cells}  bends/cell={point.n_bends_per_cell}\n"
                    f"  analytical ε₀ ≈ {point.analytical_emittance_pm:.1f} pm·rad\n"
                    f"{'='*60}"
                )

                result = self._run_one(point, run_label)
                self._append_result(result)
                all_summaries.append(result)
                done += 1

                # Brief pause between runs to avoid rate-limiting
                if not self.config.dry_run and done < total:
                    time.sleep(1.0)

        summary = self._build_summary(all_summaries)
        (self.config.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        return summary

    def prompts_only(self) -> list[dict[str, str]]:
        """Return all prompts without running anything (for review)."""
        return [
            {"label": f"{p.family}_{p.energy_gev:.0f}GeV_{p.n_cells}c",
             "family": p.family,
             "energy_gev": p.energy_gev,
             "n_cells": p.n_cells,
             "n_bends_per_cell": p.n_bends_per_cell,
             "analytical_emittance_pm": p.analytical_emittance_pm,
             "prompt": p.to_prompt()}
            for p in self.points
        ]

    # ── internal ──────────────────────────────────────────────────────────

    def _has_result(self, label: str) -> bool:
        if not self._results_db.exists():
            return False
        for line in self._results_db.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("label") == label:
                    return True
            except json.JSONDecodeError:
                continue
        return False

    def _run_one(self, point: GridPoint, label: str) -> dict[str, Any]:
        """Run the agent pipeline for a single grid point."""
        from agents.team import run_design
        from agents.session import parse_lattice_results, assess_lattice_results
        from agents.constraints import parse_design_constraints
        from physics_core import (
            parse_result_output, score_design_result,
            compute_brightness_fom, measure_lattice_factor_F,
            fit_scaling_exponent,
        )

        prompt = point.to_prompt()
        constraints = parse_design_constraints(prompt)

        # Use exploration mode when the grid point removes human priors
        _explore = (
            point.allow_variable_dipoles
            or point.allow_asymmetric
        )

        t0 = time.perf_counter()

        # Read EXECUTOR from env (same logic as run_design.py)
        _use_docker = os.environ.get("EXECUTOR", "docker").lower() == "docker"

        try:
            team_result = asyncio.run(
                run_design(
                    prompt,
                    use_docker=_use_docker,
                    max_messages=self.config.max_messages,
                    exploration_mode=_explore,
                )
            )
        except Exception as exc:
            elapsed_s = time.perf_counter() - t0
            return {
                "label": label,
                "family": point.family,
                "energy_gev": point.energy_gev,
                "n_cells": point.n_cells,
                "n_bends_per_cell": point.n_bends_per_cell,
                "analytical_emittance_pm": point.analytical_emittance_pm,
                "prompt": prompt,
                "success": False,
                "error": str(exc),
                "elapsed_s": elapsed_s,
                "message_count": 0,
                "result": None,
                "score": 0,
                "status": "error",
                "brightness_A_per_m_rad": None,
                "F_empirical": None,
                "timestamp": datetime.now().isoformat(),
            }

        elapsed_s = time.perf_counter() - t0

        messages = list(getattr(team_result, "messages", []))
        message_count = len(messages)

        # Extract the last structured result from any message
        parsed_result = None
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue
            parsed = parse_lattice_results(content)
            if parsed:
                parsed_result = parsed
                break

        assessment = assess_lattice_results(parsed_result, constraints=constraints)
        score = assessment.get("score", 0)
        status = assessment.get("status", "unknown")

        brightness = compute_brightness_fom(parsed_result or {})
        F_empirical = measure_lattice_factor_F(parsed_result or {})

        return {
            "label": label,
            "family": point.family,
            "energy_gev": point.energy_gev,
            "n_cells": point.n_cells,
            "n_bends_per_cell": point.n_bends_per_cell,
            "analytical_emittance_pm": point.analytical_emittance_pm,
            "prompt": prompt,
            "success": status == "accepted",
            "error": None,
            "elapsed_s": elapsed_s,
            "message_count": message_count,
            "result": parsed_result,
            "assessment": assessment,
            "score": score,
            "status": status,
            "brightness_A_per_m_rad": brightness,
            "F_empirical": F_empirical,
            "timestamp": datetime.now().isoformat(),
        }

    def _append_result(self, result: dict[str, Any]) -> None:
        with self._results_db.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _save_manifest(self) -> None:
        manifest = {
            "config": {
                "energies_gev": self.config.grid.energies_gev,
                "n_cells_values": self.config.grid.n_cells_values,
                "families": self.config.grid.families,
                "n_runs_per_point": self.config.n_runs_per_point,
                "max_messages": self.config.max_messages,
                "model": self.config.model or os.environ.get("LLM_MODEL", "unknown"),
                "allow_variable_dipoles": self.config.grid.allow_variable_dipoles,
                "allow_asymmetric": self.config.grid.allow_asymmetric,
            },
            "n_points": len(self.points),
            "n_total_runs": len(self.points) * self.config.n_runs_per_point,
            "created": datetime.now().isoformat(),
        }
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _build_summary(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(summaries)
        if n == 0:
            return {"n_runs": 0}

        n_success = sum(1 for s in summaries if s.get("success"))
        n_stable = sum(
            1 for s in summaries
            if s.get("result") and s["result"].get("stable") is True
        )

        families = {}
        for s in summaries:
            f = s.get("family", "unknown")
            if f not in families:
                families[f] = {"n": 0, "n_success": 0, "n_stable": 0}
            families[f]["n"] += 1
            if s.get("success"):
                families[f]["n_success"] += 1
            if s.get("result") and s["result"].get("stable") is True:
                families[f]["n_stable"] += 1

        return {
            "n_runs": n,
            "n_success": n_success,
            "success_rate": n_success / n,
            "n_stable": n_stable,
            "stable_rate": n_stable / n,
            "by_family": families,
            "output_dir": str(self.config.output_dir),
        }


# ── Convenience builders for each PRL candidate experiment ──────────────────


def candidate1_dipole_grading_sweep(
    energies_gev: list[float] | None = None,
    n_cells_values: list[int] | None = None,
) -> DesignSpaceExplorer:
    """Experiment: optimize dipole angle distribution in 7BA cells.

    Uses variable dipole angles, symmetric cells, 7 quad families.
    Sweeps energy from 2–8 GeV and cell count from 16–36.
    """
    cfg = ExplorationConfig(
        grid=GridSpec(
            energies_gev=energies_gev or [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            n_cells_values=n_cells_values or [16, 20, 24, 28, 32, 36],
            families=["7BA"],
            allow_variable_dipoles=True,
        ),
        n_runs_per_point=5,
        max_messages=36,
    )
    return DesignSpaceExplorer(cfg)


def candidate2_nbend_sweep(
    energies_gev: list[float] | None = None,
    n_cells_values: list[int] | None = None,
) -> DesignSpaceExplorer:
    """Experiment: sweep over number of bends per cell from 2 to 10.

    Discovers whether non-standard n_bends values achieve better
    performance than the canonical {2, 3, 7, 9}.
    """
    cfg = ExplorationConfig(
        grid=GridSpec(
            energies_gev=energies_gev or [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
            n_cells_values=n_cells_values or [12, 16, 20, 24, 32, 40],
            families=["FODO", "DBA", "TBA", "5BA", "6BA", "7BA", "8BA", "9BA"],
            n_bends_overrides={
                "FODO": 2, "DBA": 2, "TBA": 3,
                "5BA": 5, "6BA": 6, "7BA": 7, "8BA": 8, "9BA": 9,
            },
            allow_variable_dipoles=True,
        ),
        n_runs_per_point=3,
        max_messages=36,
    )
    return DesignSpaceExplorer(cfg)


def candidate3_scaling_sweep(
    family: str = "7BA",
    energy_gev: float = 6.0,
) -> DesignSpaceExplorer:
    """Experiment: measure empirical θ³ scaling by sweeping n_cells.

    At fixed energy and lattice family, vary n_cells to change θ.
    The analytical theory predicts ε ∝ θ³ ∝ 1/n_cells³.
    The empirical exponent is measured by fit_scaling_exponent().
    """
    cfg = ExplorationConfig(
        grid=GridSpec(
            energies_gev=[energy_gev],
            n_cells_values=[8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48],
            families=[family],
        ),
        n_runs_per_point=5,
        max_messages=32,
    )
    return DesignSpaceExplorer(cfg)


def candidate5_asymmetry_sweep(
    energies_gev: list[float] | None = None,
) -> DesignSpaceExplorer:
    """Experiment: test whether asymmetric quadrupole placement helps.

    Allows independent K1 left and right of the mirror plane.
    """
    cfg = ExplorationConfig(
        grid=GridSpec(
            energies_gev=energies_gev or [3.0, 6.0],
            n_cells_values=[20, 24],
            families=["7BA", "DBA"],
            allow_asymmetric=True,
        ),
        n_runs_per_point=5,
        max_messages=40,
    )
    return DesignSpaceExplorer(cfg)
