"""Double-Bend Achromat (DBA) lattice template for pyJuTrack.

Source: dba.m (AT/MATLAB, validated)
Machine: Compact DBA test ring, 3 GeV
Structure: 4 cells × 2 RBEND (π/4 each) = 8 bends, 2π total
Circumference: ~60 m

Key design choices:
  - RBEND elements provide edge-focusing (e1=e2=angle/2 each end)
  - Mirror symmetry about the centre sextupole (QFM/M)
  - Dispersion zero at straights (achromatic condition)

AT→pyJuTrack conversion:
  Quadrupoles: K1 identical
  Sextupoles:  K2_JuTrack = 2 × K2_AT
"""

import numpy as np
import pyJuTrack as jt

energy = 3.0e9   # eV  (not specified in dba.m; using 3 GeV default)
N_cells = 4

# 2 RBEND per cell × 4 cells = 8 bends → total = 8 × π/4 = 2π ✓
bend_angle = np.pi / 4   # 45° rectangular bend

# ── Quadrupoles (from dba.m, no conversion) ───────────────────────
def QD():  return jt.KQUAD("QD",  0.2,  -3.2243)    # defocusing quad
def QF():  return jt.KQUAD("QF",  0.2,   4.60156)   # focusing quad
def QDM(): return jt.KQUAD("QDM", 0.2,  -1.95898)   # matching cell defoc quad
def QFM(): return jt.KQUAD("QFM", 0.2,   3.57926)   # matching cell foc quad

# ── Dipole (RBEND with symmetric edge angles) ─────────────────────
def B(): return jt.RBEND("B", 1.0, bend_angle)

# ── Sextupoles (from dba.m; K2_JuTrack = 2 × K2_AT) ─────────────
# dba.m: SD K2_AT=-35.45865  → K2_JuTrack=-70.917
# dba.m: SF K2_AT= 13.50225  → K2_JuTrack= 27.005
def SD(): return jt.KSEXT("SD", 0.1, -70.917)
def SF(): return jt.KSEXT("SF", 0.1,  27.005)

# ── Drifts (exact lengths from dba.m) ─────────────────────────────
def DR_24():  return jt.DRIFT("DR_24",  2.4)
def DR_03():  return jt.DRIFT("DR_03",  0.3)
def DR_04():  return jt.DRIFT("DR_04",  0.4)
def DR_031(): return jt.DRIFT("DR_031", 0.313086)

# ── Symmetric DBA cell (from dba.m, markers removed) ─────────────
# Layout (half-cell mirror about QFM):
#   DR_24 QD QD DR_03 QF QF DR_03 QD QD DR_031
#   B  DR_031  QDM QDM SD DR_04 SF QFM
#   [mirror]
#   QFM SF DR_04 SD QDM QDM DR_031 B DR_031
#   QD QD DR_03 QF QF DR_03 QD QD DR_24
#
# Note: QD QD back-to-back reproduces the split-quad convention in dba.m.
cell = [
    DR_24(),
    QD(), QD(), DR_03(), QF(), QF(), DR_03(), QD(), QD(),
    DR_031(), B(), DR_031(),
    QDM(), QDM(), SD(), DR_04(), SF(), QFM(),
    # centre of cell (marker in dba.m has zero length)
    QFM(),
    SF(), DR_04(), SD(), QDM(), QDM(),
    DR_031(), B(), DR_031(),
    QD(), QD(), DR_03(), QF(), QF(), DR_03(), QD(), QD(),
    DR_24(),
]

ring = jt.Lattice(cell * N_cells)

# ── Diagnostics ───────────────────────────────────────────────────
nux, nuy = jt.gettune(ring)
print(f"DBA 3 GeV | {N_cells} cells | 8 RBEND × 45°")
print(f"Tunes: νx={nux:.6f}  νy={nuy:.6f}")
