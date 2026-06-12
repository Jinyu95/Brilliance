"""FODO lattice template for pyJuTrack.

Source: FODO.m (AT/MATLAB, validated)
Machine: Simple FODO ring, 3 GeV
Structure: QF – B – QD – B, 20 cells (40 bends total)
Circumference: ~100 m

AT→pyJuTrack conversion:
  Quadrupoles: K1 identical (no conversion)
  Sextupoles:  K2_JuTrack = 2 × K2_AT  (AT factorial normalisation)
  Octupoles:   K3_JuTrack = 6 × K3_AT
"""

import numpy as np
import pyJuTrack as jt

energy = 3.0e9   # eV
N_cells = 20

# 2 SBEND per cell × 20 cells = 40 bends summing to 2π
cell_angle = 2 * np.pi / N_cells
bend_angle = cell_angle / 2       # angle per SBEND

# ── Quadrupoles (K1 from FODO.m, unchanged) ──────────────────────
def QF(): return jt.KQUAD("QF", 0.5,  1.2)    # L=0.5 m, K1= 1.2 m⁻²
def QD(): return jt.KQUAD("QD", 0.5, -1.2)    # L=0.5 m, K1=-1.2 m⁻²

# ── Dipole ────────────────────────────────────────────────────────
def B(): return jt.SBEND("B", 1.0, bend_angle)  # L=1.0 m, sector bend

# ── Sextupoles (K2_AT=±1 → K2_JuTrack=±2) ───────────────────────
def SF(): return jt.KSEXT("SF", 0.1,  2.0)    # L=0.1 m, K2= 2.0 m⁻³
def SD(): return jt.KSEXT("SD", 0.1, -2.0)    # L=0.1 m, K2=-2.0 m⁻³

# ── Drifts ────────────────────────────────────────────────────────
def Dr():  return jt.DRIFT("Dr",  0.50)        # full cell-boundary drift
def D2():  return jt.DRIFT("D2",  0.20)        # short drift around sextupoles
def DH():  return jt.DRIFT("DH",  0.25)        # half drift at cell boundary

# ── Cell layout (from FODO.m) ─────────────────────────────────────
# HalfDr – B – D2 – SF – D2 – QF – Dr – B – D2 – SD – D2 – QD – HalfDr
# Consecutive half-drifts at cell boundaries merge to a full 0.5 m drift.
cell = [
    DH(), B(), D2(), SF(), D2(),
    QF(),
    Dr(),
    B(), D2(), SD(), D2(),
    QD(),
    DH(),
]

ring = jt.Lattice(cell * N_cells)

# ── Diagnostics ───────────────────────────────────────────────────
nux, nuy = jt.gettune(ring)
print(f"FODO 3 GeV | {N_cells} cells | C≈{N_cells*5:.0f} m")
print(f"Tunes: νx={nux:.6f}  νy={nuy:.6f}")
