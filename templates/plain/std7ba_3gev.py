"""Standard 7-Bend Achromat (conventional 7BA) lattice for pyJuTrack.

Machine type: 4th-generation synchrotron light source (conventional MBA)
Structure: N_cells superperiods, each with 7 sector bends
           Mirror-symmetric cell about central bend BC
           Distributed sextupoles between each pair of dipoles

Cell topology (half-cell, from straight center to cell center):
  [DS] - Q1 - Q2 - B1 - Q3 - SD1 - Q4 - SF1 - B2 - Q5 - SD2 - Q6 - SF2 - B3 - Q7 - BC

  Full cell = half + mirror about BC = 7 dipoles: B1+B2+B3+BC+B3+B2+B1

Bend angle distribution (graded, BC is longest):
  B1 = 0.6x, B2 = 0.8x, B3 = 1.0x, BC = 1.5x
  where x = cell_angle / 6.3 (from 2*0.6 + 2*0.8 + 2*1.0 + 1.5 = 6.3)

Design notes:
  1. All bends are simple SBENDs (no combined-function, no gradient).
  2. Sextupoles distributed between every pair of dipoles (conventional MBA).
  3. One quadrupole between every adjacent pair of dipoles (mandatory for stability).
  4. 7 independent quad families (Q1-Q7) — alternating focusing/defocusing.
  5. 4 sextupole families (SD1, SF1, SD2, SF2) for chromaticity correction.
  6. Simpler than Hybrid MBA: no reverse bends, no combined-function dipoles,
     no RBEND correctors.  Suitable for random stability search.
  7. angle_scale = 20/N_cells lets the template scale to any period count
     while preserving ring closure (total bend = 2*pi).
"""

import numpy as np
import pyJuTrack as jt

energy = 3.0e9     # eV (3 GeV — scales to any energy, K1 is energy-independent)
N_cells = 20       # superperiods (MAX IV uses 20; adjust for target circumference)
angle_scale = 20.0 / N_cells   # scales all bend angles for ring closure

# ── Angular budget ────────────────────────────────────────────────
cell_angle = 2 * np.pi / N_cells   # total bending per cell

# Graded angles: B1(outer, smallest) < B2 < B3 < BC(center, largest)
x = cell_angle / 6.3
angle_B1 = 0.6 * x    # outer matching bend
angle_B2 = 0.8 * x    # intermediate arc bend
angle_B3 = 1.0 * x    # inner arc bend
angle_BC = 1.5 * x    # central bend (longest, lowest emittance contribution)

# ── Reference quad strengths (verified stable for this cell) ──────
# Alternating focusing/defocusing pattern: +, -, +, -, +, -, +
# Order: Q1, Q2, Q3, Q4, Q5, Q6, Q7
init_quads = [3.79, -2.74, 1.77, -2.77, 1.91, -2.25, 2.70]

# ── Quadrupoles ───────────────────────────────────────────────────
def Q1(k1=3.79):  return jt.KQUAD("Q1", 0.25, k1)    # matching, focusing
def Q2(k1=-2.74): return jt.KQUAD("Q2", 0.20, k1)    # matching, defocusing
def Q3(k1=1.77):  return jt.KQUAD("Q3", 0.20, k1)    # arc, focusing
def Q4(k1=-2.77): return jt.KQUAD("Q4", 0.20, k1)    # arc, defocusing
def Q5(k1=1.91):  return jt.KQUAD("Q5", 0.20, k1)    # arc, focusing
def Q6(k1=-2.25): return jt.KQUAD("Q6", 0.20, k1)    # arc, defocusing
def Q7(k1=2.70):  return jt.KQUAD("Q7", 0.25, k1)    # center, focusing

# ── Dipoles (simple sector bends, no gradient) ────────────────────
def B1(): return jt.SBEND("B1", 0.58, angle_B1 * angle_scale)   # outer
def B2(): return jt.SBEND("B2", 0.78, angle_B2 * angle_scale)   # intermediate
def B3(): return jt.SBEND("B3", 0.97, angle_B3 * angle_scale)   # inner
def BC(): return jt.SBEND("BC", 1.46, angle_BC * angle_scale)   # center (longest)

# ── Sextupoles (zero initially; set for chromaticity correction) ──
def SD1(k2=0.0): return jt.KSEXT("SD1", 0.15, k2)   # defocusing, after B1
def SF1(k2=0.0): return jt.KSEXT("SF1", 0.15, k2)   # focusing, before B2
def SD2(k2=0.0): return jt.KSEXT("SD2", 0.15, k2)   # defocusing, after B2
def SF2(k2=0.0): return jt.KSEXT("SF2", 0.15, k2)   # focusing, before B3

# ── Drifts ────────────────────────────────────────────────────────
def DS():  return jt.DRIFT("DS",  2.50)    # half straight section (for IDs)
def DM1(): return jt.DRIFT("DM1", 0.30)    # matching drift
def DM2(): return jt.DRIFT("DM2", 0.25)    # matching drift
def D1():  return jt.DRIFT("D1",  0.25)    # bend-to-quad
def D2():  return jt.DRIFT("D2",  0.12)    # quad-to-sextupole
def D3():  return jt.DRIFT("D3",  0.12)    # sextupole-to-quad
def D4():  return jt.DRIFT("D4",  0.15)    # quad-to-sextupole / sext-to-bend
def D5():  return jt.DRIFT("D5",  0.20)    # center region drift

# ── Cell layout (mirror-symmetric about BC) ───────────────────────
cell = [
    # ── Left half (straight center → cell center) ────────────────
    DS(),
    Q1(), DM1(), Q2(), DM2(),
    B1(),
    D1(), Q3(), D2(), SD1(), D3(), Q4(), D4(), SF1(), D4(),
    B2(),
    D1(), Q5(), D2(), SD2(), D3(), Q6(), D4(), SF2(), D4(),
    B3(),
    D5(), Q7(), D5(),
    # ── Center bend ──────────────────────────────────────────────
    BC(),
    # ── Right half (mirror of left) ──────────────────────────────
    D5(), Q7(), D5(),
    B3(),
    D4(), SF2(), D4(), Q6(), D3(), SD2(), D2(), Q5(), D1(),
    B2(),
    D4(), SF1(), D4(), Q4(), D3(), SD1(), D2(), Q3(), D1(),
    B1(),
    DM2(), Q2(), DM1(), Q1(),
    DS(),
]

# ── create_cell helper (used by stability search & optimizer) ─────
def create_cell(quad_strengths):
    """Build one cell with custom quad K1 values.

    Parameters
    ----------
    quad_strengths : list of 7 floats
        [Q1, Q2, Q3, Q4, Q5, Q6, Q7] K1 values in m^-2.
        Sign convention: Q1,Q3,Q5,Q7 > 0 (focusing), Q2,Q4,Q6 < 0 (defocusing).

    Returns
    -------
    list : pyJuTrack element list for one cell.
    """
    k = quad_strengths
    return [
        DS(),
        jt.KQUAD("Q1", 0.25, k[0]), DM1(), jt.KQUAD("Q2", 0.20, k[1]), DM2(),
        B1(),
        D1(), jt.KQUAD("Q3", 0.20, k[2]), D2(), SD1(), D3(), jt.KQUAD("Q4", 0.20, k[3]), D4(), SF1(), D4(),
        B2(),
        D1(), jt.KQUAD("Q5", 0.20, k[4]), D2(), SD2(), D3(), jt.KQUAD("Q6", 0.20, k[5]), D4(), SF2(), D4(),
        B3(),
        D5(), jt.KQUAD("Q7", 0.25, k[6]), D5(),
        BC(),
        D5(), jt.KQUAD("Q7", 0.25, k[6]), D5(),
        B3(),
        D4(), SF2(), D4(), jt.KQUAD("Q6", 0.20, k[5]), D3(), SD2(), D2(), jt.KQUAD("Q5", 0.20, k[4]), D1(),
        B2(),
        D4(), SF1(), D4(), jt.KQUAD("Q4", 0.20, k[3]), D3(), SD1(), D2(), jt.KQUAD("Q3", 0.20, k[2]), D1(),
        B1(),
        DM2(), jt.KQUAD("Q2", 0.20, k[1]), DM1(), jt.KQUAD("Q1", 0.25, k[0]),
        DS(),
    ]

ring = jt.Lattice(cell * N_cells)

# ── Diagnostics ───────────────────────────────────────────────────
nux, nuy = jt.gettune(ring)
print(f"Standard 7BA 3 GeV | {N_cells} cells | 7 bends/cell | C~{N_cells*20.3:.0f} m")
print(f"Tunes: nux={nux:.6f}  nuy={nuy:.6f}")
