"""Analysis tools for exploration results — Pareto frontiers, anomaly detection,
scaling-law fitting, and structured comparisons.

All functions are pure Python (no pyJuTrack, no Julia).  They operate on the
JSONL database produced by ``DesignSpaceExplorer``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from physics_core import (
    _to_float,
    _to_bool,
    compute_pareto_front,
    compute_brightness_fom,
    measure_lattice_factor_F,
    fit_scaling_exponent,
    theoretical_emittance_scaling,
)


class ResultDatabase:
    """Load and query the exploration results JSONL file."""

    def __init__(self, results_jsonl: Path | str):
        self.path = Path(results_jsonl)
        self._rows: list[dict[str, Any]] | None = None

    @property
    def rows(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = self._load()
        return self._rows

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def reload(self) -> None:
        """Force re-read from disk."""
        self._rows = None

    # ── filters ──────────────────────────────────────────────────────────

    def stable(self) -> list[dict[str, Any]]:
        """Return runs that produced a stable design."""
        return [
            r for r in self.rows
            if r.get("result") and r["result"].get("stable") is True
            and not r.get("error")
        ]

    def by_family(self, family: str) -> list[dict[str, Any]]:
        """Filter to one lattice family."""
        return [r for r in self.rows if r.get("family", "").upper() == family.upper()]

    def by_energy(self, energy_gev: float, tol: float = 0.05) -> list[dict[str, Any]]:
        """Filter to runs at a specific energy (±tol)."""
        return [
            r for r in self.rows
            if abs(r.get("energy_gev", float("nan")) - energy_gev) < tol
        ]

    def successful(self) -> list[dict[str, Any]]:
        """Return runs that completed without errors."""
        return [r for r in self.rows if not r.get("error") and r.get("success")]

    def best_per_point(self) -> list[dict[str, Any]]:
        """For each unique (family, energy, n_cells), keep the best run by score."""
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in self.rows:
            key = (
                r.get("family", ""),
                r.get("energy_gev", 0.0),
                r.get("n_cells", 0),
            )
            groups[key].append(r)

        best: list[dict[str, Any]] = []
        for group in groups.values():
            group.sort(key=lambda x: x.get("score", 0), reverse=True)
            best.append(group[0])
        return best

    # ── statistics ───────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics across all runs."""
        stable = self.stable()
        successful = self.successful()
        total = len(self.rows)

        if total == 0:
            return {"n_total": 0}

        emittances = [
            _to_float(r["result"].get("emittance"))
            for r in stable
            if r.get("result")
        ]
        emittances = [e for e in emittances if e is not None and math.isfinite(e)]

        F_emp = [
            r.get("F_empirical")
            for r in stable
            if r.get("F_empirical") is not None
        ]

        scores = [r.get("score", 0) for r in self.rows]

        families = defaultdict(int)
        for r in self.rows:
            families[r.get("family", "unknown")] += 1

        return {
            "n_total": total,
            "n_successful": len(successful),
            "n_stable": len(stable),
            "success_rate": len(successful) / total,
            "stable_rate": len(stable) / total,
            "emittance_min_pm": min(e * 1e12 for e in emittances) if emittances else None,
            "emittance_max_pm": max(e * 1e12 for e in emittances) if emittances else None,
            "emittance_median_pm": (
                sorted(emittances)[len(emittances) // 2] * 1e12
                if emittances else None
            ),
            "F_empirical_mean": (
                sum(F_emp) / len(F_emp) if F_emp else None
            ),
            "score_mean": sum(scores) / len(scores) if scores else 0.0,
            "score_max": max(scores) if scores else 0,
            "by_family": dict(families),
        }

    def family_stats(self) -> dict[str, dict[str, Any]]:
        """Per-family statistics."""
        out: dict[str, dict[str, Any]] = {}
        for family in set(r.get("family", "") for r in self.rows):
            if not family:
                continue
            rows = self.by_family(family)
            stable_rows = [r for r in rows if r.get("result") and r["result"].get("stable") is True]
            emittances = [
                _to_float(r["result"].get("emittance"))
                for r in stable_rows if r.get("result")
            ]
            emittances = [e for e in emittances if e is not None and math.isfinite(e)]
            F_vals = [
                r.get("F_empirical") for r in stable_rows
                if r.get("F_empirical") is not None
            ]
            scores = [r.get("score", 0) for r in rows]

            out[family] = {
                "n_runs": len(rows),
                "n_stable": len(stable_rows),
                "stable_rate": len(stable_rows) / len(rows) if rows else 0.0,
                "emittance_min_pm": min(e * 1e12 for e in emittances) if emittances else None,
                "emittance_median_pm": (
                    sorted(emittances)[len(emittances) // 2] * 1e12
                    if emittances else None
                ),
                "F_empirical_mean": sum(F_vals) / len(F_vals) if F_vals else None,
                "score_mean": sum(scores) / len(scores) if scores else 0.0,
            }
        return out


# ── Anomaly detection ───────────────────────────────────────────────────────


def detect_anomalies(
    db: ResultDatabase,
    *,
    z_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """Flag runs where the emittance is anomalously far from the theoretical floor.

    An "anomaly" here is not necessarily an error — it could be a discovery.
    A negative anomaly (emittance MUCH lower than theoretical) suggests the
    agent found a non-canonical solution.  A positive anomaly (MUCH higher)
    suggests an error or a peculiar lattice topology.

    Returns a list of anomalous runs with z-score and interpretation.
    """
    stable = db.stable()
    if len(stable) < 5:
        return []

    # Compute emittance ratio = emittance / theoretical_floor
    ratios: list[tuple[int, float]] = []
    for i, r in enumerate(stable):
        emit = _to_float(r["result"].get("emittance")) if r.get("result") else None
        e_gev = r.get("energy_gev", 0.0)
        nc = r.get("n_cells", 1)
        nb = r.get("n_bends_per_cell", 2)
        fam = r.get("family", "FODO")

        if emit is None or emit <= 0:
            continue
        theory = theoretical_emittance_scaling(e_gev, nc, nb,
                                                _lattice_F(fam))
        if theory is None or theory <= 0:
            continue
        ratios.append((i, emit / theory))

    if not ratios:
        return []

    mean_log = sum(math.log(rt) for _, rt in ratios) / len(ratios)
    var_log = sum((math.log(rt) - mean_log) ** 2 for _, rt in ratios) / len(ratios)
    std_log = math.sqrt(var_log) if var_log > 0 else 1.0

    anomalies: list[dict[str, Any]] = []
    for i, ratio in ratios:
        z = (math.log(ratio) - mean_log) / std_log
        if abs(z) >= z_threshold:
            r = stable[i]
            anomaly_type = (
                "unusually_low_emittance" if ratio < 1.0
                else "unusually_high_emittance"
            )
            anomalies.append({
                "label": r.get("label", ""),
                "family": r.get("family", ""),
                "energy_gev": r.get("energy_gev"),
                "n_cells": r.get("n_cells"),
                "emittance_pm": (
                    _to_float(r["result"].get("emittance")) * 1e12
                    if r.get("result") else None
                ),
                "emittance_ratio": ratio,
                "z_score": z,
                "type": anomaly_type,
                "note": (
                    f"Emittance is {ratio:.1f}× the theoretical floor "
                    f"({z:+.1f}σ). "
                    + (
                        "Potentially a novel design discovery — worth expert review."
                        if anomaly_type == "unusually_low_emittance"
                        else "Possible error or degenerate solution — verify."
                    )
                ),
            })
    return anomalies


def _lattice_F(family: str) -> float:
    """Theoretical lattice factor F per family."""
    defaults = {
        "FODO": 0.144, "DBA": 0.022, "TBA": 0.017,
        "5BA": 0.012, "6BA": 0.0075, "7BA": 0.0046,
        "8BA": 0.0035, "9BA": 0.0028, "H7BA": 0.0030,
        "H6BA": 0.0040, "MBA": 0.0030,
    }
    return defaults.get(family.upper(), 0.010)


# ── Comparison to reference designs ─────────────────────────────────────────


def compare_to_reference(
    db: ResultDatabase,
    *,
    reference_emittance_pm: float | None = None,
    reference_brightness: float | None = None,
) -> dict[str, Any]:
    """Compare the best agent designs to known reference lattices.

    If ``reference_emittance_pm`` and ``reference_brightness`` are None,
    reasonable defaults for each family are used:

        FODO 3 GeV:    ε₀ ~ 5000 pm·rad
        DBA 3 GeV:     ε₀ ~ 100 pm·rad
        7BA 6 GeV:     ε₀ ~ 50 pm·rad
    """
    _REF_EMITTANCE_PM: dict[str, float] = {
        "FODO": 5000.0,
        "DBA": 100.0,
        "TBA": 80.0,
        "7BA": 50.0,
        "9BA": 30.0,
        "H7BA": 40.0,
        "H6BA": 60.0,
    }

    stable = db.stable()
    comparisons: list[dict[str, Any]] = []

    for r in stable:
        family = r.get("family", "")
        ref_emit_pm = reference_emittance_pm or _REF_EMITTANCE_PM.get(family.upper())
        if ref_emit_pm is None:
            continue

        emit = _to_float(r["result"].get("emittance")) if r.get("result") else None
        if emit is None:
            continue
        emit_pm = emit * 1e12

        improvement = None
        if ref_emit_pm > 0:
            improvement = (ref_emit_pm - emit_pm) / ref_emit_pm

        comparisons.append({
            "label": r.get("label", ""),
            "family": family,
            "emittance_pm": emit_pm,
            "reference_emittance_pm": ref_emit_pm,
            "improvement_pct": round(improvement * 100, 1) if improvement is not None else None,
        })

    comparisons.sort(key=lambda c: c.get("improvement_pct", -999), reverse=True)
    return {
        "n_comparisons": len(comparisons),
        "top_improvements": comparisons[:10],
        "best_improvement_pct": (
            comparisons[0]["improvement_pct"] if comparisons else None
        ),
        "n_beat_reference": sum(
            1 for c in comparisons
            if c.get("improvement_pct") is not None and c["improvement_pct"] > 0
        ),
    }


def summarize_by_family(db: ResultDatabase) -> list[dict[str, Any]]:
    """Generate a structured summary per (family, energy) combination.

    Suitable for table generation in a paper.
    """
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in db.stable():
        key = (r.get("family", ""), r.get("energy_gev", 0.0))
        groups[key].append(r)

    rows: list[dict[str, Any]] = []
    for (family, energy), runs in sorted(groups.items()):
        emittances = [
            _to_float(r["result"].get("emittance")) * 1e12
            for r in runs if r.get("result")
        ]
        emittances = [e for e in emittances if e is not None and math.isfinite(e)]

        circumferences = [
            _to_float(r["result"].get("circumference"))
            for r in runs if r.get("result")
        ]
        circumferences = [c for c in circumferences if c is not None and math.isfinite(c)]

        tunes_x = [
            _to_float(r["result"].get("tune_nux"))
            for r in runs if r.get("result")
        ]
        tunes_x = [t for t in tunes_x if t is not None and math.isfinite(t)]

        F_vals = [
            r.get("F_empirical") for r in runs
            if r.get("F_empirical") is not None
        ]

        brightness_vals = [
            r.get("brightness_A_per_m_rad") for r in runs
            if r.get("brightness_A_per_m_rad") is not None
        ]

        rows.append({
            "family": family,
            "energy_gev": energy,
            "n_runs_stable": len(runs),
            "emittance_min_pm": min(emittances) if emittances else None,
            "emittance_max_pm": max(emittances) if emittances else None,
            "emittance_mean_pm": (
                sum(emittances) / len(emittances) if emittances else None
            ),
            "emittance_std_pm": _std(emittances),
            "circumference_mean_m": (
                sum(circumferences) / len(circumferences) if circumferences else None
            ),
            "tune_x_mean": sum(tunes_x) / len(tunes_x) if tunes_x else None,
            "F_empirical_mean": sum(F_vals) / len(F_vals) if F_vals else None,
            "brightness_mean_Apm": (
                sum(brightness_vals) / len(brightness_vals)
                if brightness_vals else None
            ),
        })

    return rows


def _std(values: list[float]) -> float | None:
    """Sample standard deviation."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) if var > 0 else 0.0


# ── Batch exploration runner ─────────────────────────────────────────────────

def run_pilot(
    families: list[str] | None = None,
    energies: list[float] | None = None,
    n_cells_values: list[int] | None = None,
    *,
    n_runs: int = 2,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run a small pilot sweep for validation before committing to full scale.

    Default: 2 energies × 3 families × 3 cell counts × 2 runs = 36 total.
    Good for catching prompt issues, API errors, and feasibility problems.
    """
    from science.explore import ExplorationConfig, GridSpec, DesignSpaceExplorer

    cfg = ExplorationConfig(
        grid=GridSpec(
            energies_gev=energies or [3.0, 6.0],
            n_cells_values=n_cells_values or [16, 20, 24],
            families=families or ["FODO", "DBA", "7BA"],
        ),
        n_runs_per_point=n_runs,
        max_messages=28,
        dry_run=dry_run,
    )
    explorer = DesignSpaceExplorer(cfg)

    if dry_run:
        prompts = explorer.prompts_only()
        print(json.dumps(prompts, indent=2, ensure_ascii=False))
        return {"dry_run": True, "n_prompts": len(prompts), "prompts": prompts}

    return explorer.run()
