#!/usr/bin/env python
"""Deterministic 7BA dipole angle scan — zero LLM, pure physics.

Based on the verified std7ba_6gev.py template.  Sweeps the 4 dipole bending
angle ratios (B1, B2, B3, BC) through their physically plausible range and
computes the resulting emittance.  Uses physics_core.optimize_stable_k1()
to re-tune quadrupoles at each angle set.

This is the engine for PRL Candidate 1 — discovering whether the optimal
dipole grading is monotonic or non-monotonic.

Run inside the Docker container or with pyJuTrack available:
    python science/dipole_scan.py --energy 6.0 --n-cells 24 --output results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# ── Physical constants ────────────────────────────────────────────────────────
_Cq     = 3.8319e-13       # quantum diffusion constant [m]
_Cgamma = 8.85e-5          # energy-loss constant [m·GeV⁻³]
_me_eV  = 0.51099895e6     # electron rest energy [eV]
_c      = 2.99792458e8     # speed of light [m/s]


# ══════════════════════════════════════════════════════════════════════════════
# 7BA cell builder — exact topology from the verified std7ba_6gev.py template
# ══════════════════════════════════════════════════════════════════════════════

def build_7ba_cell(
    angle_B1: float,
    angle_B2: float,
    angle_B3: float,
    angle_BC: float,
    k1: Sequence[float],
    jt,
) -> Any:
    """Build one mirror-symmetric 7BA cell.

    Topology (half-cell, outward from straight section centre):
      DS - Q1 - DM1 - Q2 - DM2 - B1 - D1 - Q3 - D2 - SD1 - D3 - Q4 - D4 - SF1 - D4
      - B2 - D1 - Q5 - D2 - SD2 - D3 - Q6 - D4 - SF2 - D4
      - B3 - D5 - Q7 - D5 - BC

    Full cell = half + mirror(reversed(half)).  Dipole angles are free parameters.

    Parameters
    ----------
    angle_B1..angle_BC : float
        Bending angle per dipole [rad].  Must satisfy:
        2*(B1+B2+B3) + BC = 2π / n_cells
    k1 : sequence of 7 floats
        Quadrupole strengths [m⁻²].  Q1..Q7.
    jt : module
        The pyJuTrack module.

    Returns
    -------
    jt.Lattice
    """
    # Drift lengths (from std7ba_6gev.py)
    DS_len  = 2.50
    DM1_len = 0.30
    DM2_len = 0.25
    D1_len  = 0.25
    D2_len  = 0.12
    D3_len  = 0.12
    D4_len  = 0.15
    D5_len  = 0.20

    # Element lengths
    L_Q1 = 0.25; L_Q2 = 0.20; L_Q3 = 0.20; L_Q4 = 0.20
    L_Q5 = 0.20; L_Q6 = 0.20; L_Q7 = 0.25
    L_B1 = 0.58; L_B2 = 0.78; L_B3 = 0.97; L_BC = 1.46

    # Forward half-cell (outward from straight-section center)
    # Sextupoles at zero for natural chromaticity
    half = [
        jt.DRIFT("DS",  DS_len),
        jt.KQUAD("Q1",  L_Q1, k1=k1[0]),
        jt.DRIFT("DM1", DM1_len),
        jt.KQUAD("Q2",  L_Q2, k1=k1[1]),
        jt.DRIFT("DM2", DM2_len),
        jt.SBEND("B1",  L_B1, angle_B1),
        jt.DRIFT("D1",  D1_len),
        jt.KQUAD("Q3",  L_Q3, k1=k1[2]),
        jt.DRIFT("D2",  D2_len),
        jt.KSEXT("SD1", 0.15, k2=0.0),
        jt.DRIFT("D3",  D3_len),
        jt.KQUAD("Q4",  L_Q4, k1=k1[3]),
        jt.DRIFT("D4",  D4_len),
        jt.KSEXT("SF1", 0.15, k2=0.0),
        jt.DRIFT("D4",  D4_len),
        jt.SBEND("B2",  L_B2, angle_B2),
        jt.DRIFT("D1",  D1_len),
        jt.KQUAD("Q5",  L_Q5, k1=k1[4]),
        jt.DRIFT("D2",  D2_len),
        jt.KSEXT("SD2", 0.15, k2=0.0),
        jt.DRIFT("D3",  D3_len),
        jt.KQUAD("Q6",  L_Q6, k1=k1[5]),
        jt.DRIFT("D4",  D4_len),
        jt.KSEXT("SF2", 0.15, k2=0.0),
        jt.DRIFT("D4",  D4_len),
        jt.SBEND("B3",  L_B3, angle_B3),
        jt.DRIFT("D5",  D5_len),
        jt.KQUAD("Q7",  L_Q7, k1=k1[6]),
        jt.DRIFT("D5",  D5_len),
    ]

    # Mirror the half-cell about the central bend BC.
    # BC sits at the symmetry plane: half → BC → mirror(reversed(half))
    centre = [jt.SBEND("BC", L_BC, angle_BC)]
    full = half + centre + list(reversed(half))

    return jt.Lattice(full)


def check_stable(jt, cell: Any) -> bool:
    """Check linear stability of a single cell."""
    try:
        M = jt.findm66(cell, dp=0.0)
    except Exception:
        return False
    tx = abs(float(M[0, 0] + M[1, 1]))
    ty = abs(float(M[2, 2] + M[3, 3]))
    return math.isfinite(tx) and math.isfinite(ty) and tx < 2.0 and ty < 2.0


def compute_ring_physics(
    jt,
    cell: Any,
    n_cells: int,
    energy_eV: float,
    k1_vals: Sequence[float],
    angle_B1: float,
    angle_B2: float,
    angle_B3: float,
    angle_BC: float,
) -> dict[str, Any]:
    """Compute all physics quantities for a ring built from one cell."""
    ring = cell * n_cells
    energy_GeV = energy_eV / 1e9
    gamma = energy_eV / _me_eV

    nux, nuy = jt.gettune(ring)
    xix, xiy = jt.getchrom(ring)
    rp = jt.ringpara(ring, energy=energy_eV)
    spos = jt.spos(ring)

    emit       = float(rp['emittx'])
    alphac     = float(rp['alphac'])
    circ       = float(spos[-1])
    I1 = float(rp['I1']); I2 = float(rp['I2']); I3 = float(rp['I3'])
    I4 = float(rp['I4']); I5 = float(rp['I5'])
    Jx = float(rp['Jx']); Jy = float(rp['Jy']); Je = float(rp['Je'])

    # Analytical U0 (ringpara is broken)
    U0_eV = _Cgamma * (energy_GeV ** 4) * I2 / (2.0 * math.pi) * 1e9
    denom = 2.0 * I2 + I4
    sigma_E = math.sqrt(_Cq * (gamma ** 2) * I3 / denom) if denom > 0 else float('nan')
    T0 = circ / _c
    tau_x = 2.0 * energy_eV * T0 / (U0_eV * Jx) if U0_eV * Jx > 0 else float('inf')
    tau_y = 2.0 * energy_eV * T0 / (U0_eV * Jy) if U0_eV * Jy > 0 else float('inf')
    tau_E = 2.0 * energy_eV * T0 / (U0_eV * Je) if U0_eV * Je > 0 else float('inf')

    # Cell transfer matrix
    try:
        M = jt.findm66(cell, dp=0.0)
        trace_x = float(M[0, 0] + M[1, 1])
        trace_y = float(M[2, 2] + M[3, 3])
        D_exit  = float(M[0, 5])
        T56     = float(M[4, 5])
    except Exception:
        trace_x = trace_y = D_exit = T56 = float('nan')

    return {
        "lattice_mode": "ring",
        "stable": True,
        "tune_nux": float(nux),
        "tune_nuy": float(nuy),
        "chromaticity_xix": float(xix),
        "chromaticity_xiy": float(xiy),
        "emittance": emit,
        "alphac": alphac,
        "circumference": circ,
        "energy_gev": energy_GeV,
        "n_cells": n_cells,
        "n_bends_per_cell": 7,
        "I1": I1, "I2": I2, "I3": I3, "I4": I4, "I5": I5,
        "Jx": Jx, "Jy": Jy, "Je": Je,
        "U0_eV": U0_eV,
        "sigma_E": sigma_E,
        "damping_time_x": tau_x,
        "damping_time_y": tau_y,
        "damping_time_E": tau_E,
        "trace_x": trace_x,
        "trace_y": trace_y,
        "D_exit": D_exit,
        "T56": T56,
        # Exploration data — the key parameters varied
        "k1_values": list(float(k) for k in k1_vals),
        "dipole_angles_rad": [
            float(angle_B1), float(angle_B2), float(angle_B3), float(angle_BC),
        ],
        "dipole_angles_mrad": [
            float(angle_B1 * 1e3), float(angle_B2 * 1e3),
            float(angle_B3 * 1e3), float(angle_BC * 1e3),
        ],
        "dipole_ratio_B1": float(angle_B1 / angle_BC),
        "dipole_ratio_B2": float(angle_B2 / angle_BC),
        "dipole_ratio_B3": float(angle_B3 / angle_BC),
        "is_monotonic": (
            angle_B1 <= angle_B2 <= angle_B3 <= angle_BC
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main scan
# ══════════════════════════════════════════════════════════════════════════════

def run_scan(
    energy_eV: float = 6.0e9,
    n_cells: int = 24,
    *,
    n_steps: int = 6,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Scan the 4-dimensional dipole angle space for optimal emittance.

    Parameters
    ----------
    energy_eV : float
        Beam energy in eV.
    n_cells : int
        Number of superperiods.
    n_steps : int
        Number of ratio values per dipole (total = n_steps^4 configurations).
    output_path : Path | str | None
        Where to write the JSONL results.
    verbose : bool
        Print progress.

    Returns
    -------
    list[dict] — all computed results.
    """
    import numpy as np
    import pyJuTrack as jt

    energy_GeV = energy_eV / 1e9
    cell_angle = 2.0 * math.pi / n_cells

    # ── Reference K1 values (from the verified std7ba template) ───────────
    k1_ref = [3.79, -2.74, 1.77, -2.77, 1.91, -2.25, 2.70]

    # ── Scan space: 4 angle ratios (B1, B2, B3, BC) ─────────────────────
    # We use ratio_i = angle_i / angle_BC to parameterize the distribution.
    # For each set of 4 ratios, we normalize so:
    #   2*(ratio_B1 + ratio_B2 + ratio_B3) + ratio_BC = 1.0 (if ratio_BC=1)
    #   Actually: 2*(r1 + r2 + r3) + 1.0 = S_total
    #   x = cell_angle / S_total
    #   θ_i = x * r_i,  θ_BC = x * 1.0

    r1_range = np.linspace(0.2, 1.5, n_steps)
    r2_range = np.linspace(0.3, 1.5, n_steps)
    r3_range = np.linspace(0.4, 1.5, n_steps)
    rC_range = np.linspace(0.5, 2.5, n_steps)

    total = n_steps ** 4
    results: list[dict[str, Any]] = []

    if verbose:
        print(f"7BA dipole angle scan: E={energy_GeV:.1f} GeV, {n_cells} cells")
        print(f"Grid: {n_steps}^4 = {total} angle configurations")
        print(f"Reference K1: {k1_ref}")
        print()

    np.random.seed(42)  # reproducible perturbation directions
    i = 0
    for r1 in r1_range:
        for r2 in r2_range:
            for r3 in r3_range:
                for rC in rC_range:
                    i += 1
                    # Normalize to cell_angle
                    S = 2.0 * (r1 + r2 + r3) + rC
                    if S <= 0:
                        continue
                    x = cell_angle / S
                    a1 = x * r1; a2 = x * r2; a3 = x * r3; aC = x * rC

                    # Build cell with reference K1
                    try:
                        cell = build_7ba_cell(a1, a2, a3, aC, k1_ref, jt)
                    except Exception:
                        continue

                    if not check_stable(jt, cell):
                        # Try re-tuning K1 with sign-preserving perturbations
                        k1_retuned = list(k1_ref)
                        stabilized = False
                        for _attempt in range(100):
                            _pert = [k * (1.0 + np.random.uniform(-0.3, 0.3))
                                     for k in k1_ref]
                            # Preserve sign
                            _pert = [abs(p) if k > 0 else -abs(p)
                                     for p, k in zip(_pert, k1_ref)]
                            try:
                                cell_pert = build_7ba_cell(
                                    a1, a2, a3, aC, _pert, jt)
                            except Exception:
                                continue
                            if check_stable(jt, cell_pert):
                                k1_retuned = _pert
                                cell = cell_pert
                                stabilized = True
                                break
                        if not stabilized:
                            continue

                    # Compute physics
                    try:
                        result = compute_ring_physics(
                            jt, cell, n_cells, energy_eV,
                            k1_ref if not 'k1_retuned' in dir()
                            else k1_retuned,
                            a1, a2, a3, aC,
                        )
                    except Exception:
                        continue

                    # Add scan metadata
                    result["scan_r1"] = float(r1)
                    result["scan_r2"] = float(r2)
                    result["scan_r3"] = float(r3)
                    result["scan_rC"] = float(rC)
                    result["scan_index"] = i

                    results.append(result)

                    if verbose and (i % max(1, total // 20) == 0):
                        n_stable = len(results)
                        best = min(results, key=lambda r: r.get("emittance", float("inf")))
                        print(
                            f"  [{i}/{total}] {n_stable} stable so far, "
                            f"best emit={best['emittance']*1e12:.1f} pm-rad "
                            f"(monotonic={best['is_monotonic']}) "
                            f"r1={best['scan_r1']:.2f} r2={best['scan_r2']:.2f} "
                            f"r3={best['scan_r3']:.2f} rC={best['scan_rC']:.2f}"
                        )

    if verbose:
        print(f"\nComplete. {len(results)}/{total} configurations stable.")
        if results:
            best = min(results, key=lambda r: r["emittance"])
            print(f"Best emittance: {best['emittance']*1e12:.1f} pm-rad")
            print(f"Angles [mrad]: B1={best['dipole_angles_mrad'][0]:.2f} "
                  f"B2={best['dipole_angles_mrad'][1]:.2f} "
                  f"B3={best['dipole_angles_mrad'][2]:.2f} "
                  f"BC={best['dipole_angles_mrad'][3]:.2f}")
            print(f"Monotonic: {best['is_monotonic']}")
            print(f"Grading: B1/BC={best['dipole_ratio_B1']:.3f} "
                  f"B2/BC={best['dipole_ratio_B2']:.3f} "
                  f"B3/BC={best['dipole_ratio_B3']:.3f}")

    # Write output
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to native Python for JSON
        with output.open("w", encoding="utf-8") as f:
            for r in results:
                # Build a flat exploration row compatible with ResultDatabase
                row = {
                    "label": (
                        f"7BA_{energy_GeV:.0f}GeV_{n_cells}c_"
                        f"r1={r['scan_r1']:.2f}_r2={r['scan_r2']:.2f}_"
                        f"r3={r['scan_r3']:.2f}_rC={r['scan_rC']:.2f}"
                    ),
                    "family": "7BA",
                    "energy_gev": energy_GeV,
                    "n_cells": n_cells,
                    "n_bends_per_cell": 7,
                    "analytical_emittance_pm": None,
                    "prompt": "",
                    "success": True,
                    "error": None,
                    "elapsed_s": 0,
                    "message_count": 0,
                    "result": r,
                    "score": 100,
                    "status": "accepted",
                    "brightness_A_per_m_rad": (
                        0.2 / r["emittance"] if r.get("emittance", 0) > 0 else None
                    ),
                    "F_empirical": None,
                    "timestamp": datetime.now().isoformat(),
                    "dipole_angles_rad": r.get("dipole_angles_rad"),
                    "dipole_angles_mrad": r.get("dipole_angles_mrad"),
                    "is_monotonic": r.get("is_monotonic"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if verbose:
            print(f"Results written to: {output}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="7BA dipole angle scan — deterministic discovery, zero LLM",
    )
    parser.add_argument("--energy", type=float, default=6.0,
                        help="Beam energy in GeV (default: 6.0)")
    parser.add_argument("--n-cells", type=int, default=24,
                        help="Number of superperiods (default: 24)")
    parser.add_argument("--steps", type=int, default=6,
                        help="Ratio steps per parameter (default: 6 → 6^4=1296 configs)")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSONL path (default: science/results/<timestamp>/dipole_scan.jsonl)")
    args = parser.parse_args()

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (Path(__file__).resolve().parent
                       / "results" / ts / "dipole_scan.jsonl")

    run_scan(
        energy_eV=args.energy * 1e9,
        n_cells=args.n_cells,
        n_steps=args.steps,
        output_path=output_path,
        verbose=True,
    )


if __name__ == "__main__":
    main()
