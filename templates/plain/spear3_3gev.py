"""SPEAR3-type standard-cell lattice for pyJuTrack.

Source: spear3.m (AT/MATLAB, SLAC SPEAR3 validated design, 3 GeV)
Machine type: 3rd-generation synchrotron light source
Structure: 18 standard cells × 2 SBEND per cell = 36 bends
Circumference: ~234 m (full SPEAR3 uses 14 standard + 4 matching cells;
               here the matching cells are approximated as standard cells and
               the bend angle is adjusted to close the ring at 18 cells)

Key parameters (from spear3.m):
  Quadrupoles: QF K1=1.769, QD K1=-1.542, QFC K1=1.749 (values unchanged)
  Sextupoles: SF PolynomB(3)=32.0477/2 (AT) → K2_JuTrack=32.048
              SD PolynomB(3)=-38.802/2  (AT) → K2_JuTrack=-38.802

AT→pyJuTrack conversion:
  Quadrupoles: K1 unchanged
  Sextupoles:  K2_JuTrack = 2 × AT_PolynomB[2]
"""

import numpy as np
import pyJuTrack as jt

energy = 3.0e9   # eV  (SPEAR3 design energy)
N_cells = 18     # standard cells only (see note above)

# Bend angle adjusted for ring closure with 18 cells × 2 bends/cell = 36 total
# SPEAR3 actual SBEND angle = 0.18479957 rad (per bend in 14-cell arc)
# Here: cell_angle/2 = π/18 ≈ 0.17453 rad (slightly smaller, ~5% adjustment)
cell_angle  = 2 * np.pi / N_cells
bend_angle  = cell_angle / 2

# ── Quadrupoles (from spear3.m, K1 unchanged) ─────────────────────
# QF:  standard arc focusing quad,   L=0.3534 m, K1= 1.7687 m⁻²
# QD:  standard arc defocusing quad, L=0.1635 m, K1=-1.5425 m⁻²
# QFC: combined matching/arc quad,   L=0.5124 m, K1= 1.7486 m⁻²
def QF():  return jt.KQUAD("QF",  0.3533895,  1.768672904054)
def QD():  return jt.KQUAD("QD",  0.1634591, -1.542474230359)
def QFC(): return jt.KQUAD("QFC", 0.5123803,  1.748640831069)

# ── Dipole (SPEAR3 main arc bend, length preserved, angle scaled) ─
def B(): return jt.SBEND("B", 1.5048, bend_angle)

# ── Sextupoles (from spear3.m; K2_JuTrack = 2 × AT_PolynomB[2]) ─
# spear3.m: SF.PolynomB(3) = 32.0477093/2  → K2_JuTrack = 32.048 m⁻³
# spear3.m: SD.PolynomB(3) = -38.80153/2   → K2_JuTrack =-38.802 m⁻³
def SF(): return jt.KSEXT("SF", 0.21,  32.048)
def SD(): return jt.KSEXT("SD", 0.25, -38.802)

# ── Drifts (from spear3.m; BPMs & correctors absorbed into adjacent drifts)
# DC1 = DC1A(1.405934) + DC1B(0.12404)            = 1.5300 m
# DC2 = DC2A(0.11577) + COR_L(0.15) + DC2B(0.11581) = 0.3816 m
# DC3 = DC3A(0.05322) + DC3B(0.16368)              = 0.2169 m
# DC4 = DC4A(0.15921) + DC4B(0.04442)              = 0.2036 m
# DC5 = DC5A(0.09058) + COR_L(0.15) + DC5B(0.36139) = 0.6020 m
# DC6 = DC6A(0.110646) + DC6B(0.063166)             = 0.1738 m  (each side of QFC)
def DC1(): return jt.DRIFT("DC1", 1.5300)
def DC2(): return jt.DRIFT("DC2", 0.3816)
def DC3(): return jt.DRIFT("DC3", 0.2169)
def DC4(): return jt.DRIFT("DC4", 0.2036)
def DC5(): return jt.DRIFT("DC5", 0.6020)
def DC6(): return jt.DRIFT("DC6", 0.1738)

# ── Standard SPEAR3 cell (from HCEL1+HCEL2 in spear3.m) ───────────
# Symmetric layout about the centre QFC:
#   [DC1 QF DC2 QD DC3 BEND DC4 SD DC5 SF DC6] QFC [DC6 SF DC5 SD DC4 BEND DC3 QD DC2 QF DC1]
cell = [
    DC1(),
    QF(), DC2(), QD(), DC3(),
    B(),
    DC4(), SD(), DC5(), SF(), DC6(),
    QFC(),
    DC6(), SF(), DC5(), SD(), DC4(),
    B(),
    DC3(), QD(), DC2(), QF(),
    DC1(),
]

ring = jt.Lattice(cell * N_cells)

# ── Diagnostics ───────────────────────────────────────────────────
nux, nuy = jt.gettune(ring)
circ = N_cells * (2 * 1.5300 + 2 * 0.3816 + 2 * 0.2169 + 2 * 0.2036 + 2 * 0.6020 +
                  2 * 0.1738 + 2 * 1.5048 + 2 * 0.21 + 2 * 0.25 +
                  2 * 0.3534 + 2 * 0.1635 + 0.5124)
print(f"SPEAR3-type 3 GeV | {N_cells} cells | C≈{circ:.1f} m")
print(f"Tunes: νx={nux:.6f}  νy={nuy:.6f}")
