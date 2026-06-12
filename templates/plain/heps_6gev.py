"""HEPS-type Multi-Bend Achromat (MBA) lattice for pyJuTrack.

Source: HEPS_RING_V2p4c.m (AT/MATLAB, HEPS official design, 6 GeV)
Machine type: 4th-generation synchrotron light source (ultra-low emittance)
Structure: N_full_periods periods × 2 half-cells = 2*N_full_periods half-cells
Circumference: ~1360 m (HEPS design: 1360.4 m, scales with N_full_periods)

Cell topology per half-period (CELLA):
  [Straight] – B1 – [arc1: Q3 SD Q4 Q4b SF Q5 Q6] – B2 – Q7 – BD –
  Q8 Q8b – BC(centre bend) – Q8b Q8 – BD – Q7 – B2 –
  [arc2: Q6 Q5 SF Q4b Q4 SD Q3] – B1 – [Straight]

Physics modelling notes:
  1. Each multi-slice dipole group (BLG1–BLG5) → single SBEND (total L & angle).
  2. Combined-function magnets Q4 (angle=-0.001341 rad) and Q8 (angle=-0.005377
     rad) have their bending restored via thin RBEND correctors (Q4b, Q8b) placed
     adjacent to the quad. These provide (a) correct ring closure and (b) critical
     vertical edge focusing that pyJuTrack SBENDs alone cannot supply.
  3. Octupoles OF1–OF4 included at zero strength (DA optimisation, not linear).
  4. CELLA/CELLB averaged into one quad family per type.
  5. angle_scale = 24/N_full_periods lets the template scale to any period count
     while preserving ring closure (total bend = 2π).

AT→pyJuTrack conversion:
  Quadrupoles: K1 unchanged
  Sextupoles:  K2_JuTrack = 2 × K2_AT  (AT factorial normalisation)
  Octupoles:   K3_JuTrack = 6 × K3_AT  (not used here; set to zero)
"""

import numpy as np
import pyJuTrack as jt

energy  = 6.0e9   # eV  (HEPS design energy 6 GeV)
N_full_periods = 24           # HEPS has 24 super-periods (change to e.g. 20)
angle_scale = 24.0 / N_full_periods   # scales all bend angles for ring closure

# ── Angular budgets (from HEPS_RING_V2p4c.m, at 24 periods) ─────
# Each half-cell contributes ½ × (2π / N_full_periods) of bending.
# At 24 periods the reference angles per half-cell are:
#   B1  (BLG1 group)            : 0.021595 rad
#   B2  (BLG2 group)            : 0.014606 rad
#   BD  (combined-function)     : 0.023869 rad  (×2 per half-cell)
#   BC  (centre bend BLG3+DS)   : 0.024194 rad  (×1)
#   Q4b (reverse-bend corrector): −0.001341 rad  (×2)
#   Q8b (reverse-bend corrector): −0.005377 rad  (×2)
# Total = 2×0.021595 + 2×0.014606 + 2×0.023869 + 0.024194
#       + 2×(−0.001341) + 2×(−0.005377) ≈ 0.130898 rad = π/24 ✓
# All angles are multiplied by angle_scale when N_full_periods ≠ 24.

# ── Reference quad strengths (HEPS V2p4c, 24 periods) ────────────
# Order: Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9
init_quads = [3.964, -3.921, -3.760, 2.670, 3.942, -3.817, 3.923, 3.319, -3.903]

# ── Quadrupoles (averaged CELLA/CELLB families from HEPS) ─────────
# K2 values are V2p4c optimised strengths; R-subscript variants averaged
def Q1():  return jt.KQUAD("Q1",  0.255,  3.964)    # K1 = 3.964 m⁻²
def Q2():  return jt.KQUAD("Q2",  0.205, -3.921)    # K1 =-3.921
def Q3():  return jt.KQUAD("Q3",  0.180, -3.760)    # K1 =-3.760
def Q4():  return jt.KQUAD("Q4",  0.180,  2.670)    # combined-function quad part
def Q5():  return jt.KQUAD("Q5",  0.205,  3.942)    # K1 = 3.942
def Q6():  return jt.KQUAD("Q6",  0.340, -3.817)    # K1 =-3.817
def Q7():  return jt.KQUAD("Q7",  0.385,  3.923)    # K1 = 3.923
def Q8():  return jt.KQUAD("Q8",  0.610,  3.319)    # combined-function quad part
def Q9():  return jt.KQUAD("Q9",  0.260, -3.903)    # K1 =-3.903

# ── Combined-function reverse-bend correctors ────────────────────
# Q4 and Q8 in the real machine have small reverse-bend angles.
# Modelled as thin RBENDs adjacent to the quad to restore:
#   (a) correct total bend → ring closure
#   (b) vertical edge focusing (pyJuTrack RBEND has e1=e2=angle/2)
def Q4b(): return jt.RBEND("Q4b", 0.001, -0.001341 * angle_scale)
def Q8b(): return jt.RBEND("Q8b", 0.001, -0.005377 * angle_scale)

# ── Dipoles (single-element representation of each slice group) ──
# B1/B5 (BLG1 group): 5 slices × 0.2998 m = 1.499 m total, angle=0.021595 rad
# B2/B4 (BLG2 group): 5 slices × 0.2016 m = 1.008 m total, angle=0.014606 rad
# BD    (combined bd): 2 × 0.5486 m = 1.097 m total,        angle=0.023869 rad
# BC    (BLG3+DS):     0.4285 m × 2 = 0.857 m total,        angle=0.024194 rad
def B1(): return jt.SBEND("B1", 1.499, 0.021595 * angle_scale)   # outer arc dipole
def B2(): return jt.SBEND("B2", 1.008, 0.014606 * angle_scale)   # inner arc dipole
def BD(): return jt.SBEND("BD", 1.097, 0.023869 * angle_scale)   # centre bend
def BC(): return jt.SBEND("BC", 0.857, 0.024194 * angle_scale)   # centre (reverse-arc)

# ── Sextupoles (from HEPS_RING_V2p4c.m; K2_JuTrack = 2 × K2_AT) ─
# ksd1=-116.52, ksd2=-146.15, ksd3=-124.53, ksd4=-54.25  → × 2
# ksf1= 171.32, ksf2= 182.77, ksf3=  61.61, ksf4= 130.53 → × 2
def SD1(): return jt.KSEXT("SD1", 0.315, -233.04)
def SD2(): return jt.KSEXT("SD2", 0.335, -292.31)
def SD3(): return jt.KSEXT("SD3", 0.335, -249.06)
def SD4(): return jt.KSEXT("SD4", 0.315, -108.49)
def SF1(): return jt.KSEXT("SF1", 0.150,  342.65)   # half-length (split at M1)
def SF2(): return jt.KSEXT("SF2", 0.150,  365.54)
# ── Octupoles (included at zero strength; set non-zero for DA optimisation) ─
def OF():  return jt.KOCT("OF", 0.270, 0.0)   # K3=0 template; K3_JuTrack = 6×K3_AT

# ── Drifts (from HEPS_RING_V2p4c.m, adjacent BPM/corrector absorbed) ─
# D1R=D1=3.043 m (half straight section – shared between CELLA and CELLB)
# At cell boundary the two half-straights add to 6.086 m per straight section.
def D1():   return jt.DRIFT("D1",   3.043)    # half straight section
def D2():   return jt.DRIFT("D2",   0.408)
def D3():   return jt.DRIFT("D3",   0.282)
def D4():   return jt.DRIFT("D4",   0.199)
def D5a():  return jt.DRIFT("D5a",  0.103)
def D5b():  return jt.DRIFT("D5b",  0.102)
def D6a():  return jt.DRIFT("D6a",  0.339)
def D6b():  return jt.DRIFT("D6b",  0.246)
def D7a():  return jt.DRIFT("D7a",  0.099)
def D7b():  return jt.DRIFT("D7b",  0.084)
def D7c():  return jt.DRIFT("D7c",  0.102)
def D8():   return jt.DRIFT("D8",   0.337)
def D9():   return jt.DRIFT("D9",   0.180)
def D10():  return jt.DRIFT("D10",  0.240)
def D11():  return jt.DRIFT("D11",  0.298)
def D12a(): return jt.DRIFT("D12a", 0.237)
def D12b(): return jt.DRIFT("D12b", 0.081)

# ── Half-period cell (CELLA from HEPS_RING_V2p4c.m) ──────────────
# Mirror-symmetric about the centre bend BC.
# Q4b/Q8b thin RBENDs restore combined-function reverse bends.
cell = [
    # ──── Upstream straight end ───────────────────────────────────
    D1(),
    Q1(), D2(), Q2(), D3(),
    # ──── Upstream arc: B1 triplet ────────────────────────────────
    B1(),
    D4(), Q3(), D5a(), SD1(), D5b(),
    Q4(), Q4b(),
    D6a(), SF1(), D6b(),
    Q5(), D7b(), OF(), D7a(), SD2(), D7c(),
    Q6(), D8(),
    # ──── Inner arc: B2 group ─────────────────────────────────────
    B2(),
    D9(), Q7(), D10(),
    # ──── Centre region with combined-function bends ──────────────
    BD(),
    D11(), Q8(), Q8b(), D12a(), Q9(), D12b(),
    BC(),   # central bend (BLG3+BLG3_DS)
    D12b(), Q9(), D12a(), Q8b(), Q8(), D11(),
    BD(),
    D10(), Q7(), D9(),
    # ──── Inner arc: B4=B2' (mirror) ─────────────────────────────
    B2(),
    D8(), Q6(), D7c(), SD3(), D7a(), OF(), D7b(),
    Q5(),
    D6b(), SF2(), D6a(),
    Q4b(), Q4(),
    D5b(), SD4(), D5a(), Q3(), D4(),
    # ──── Downstream arc: B5=B1' (mirror) ────────────────────────
    B1(),
    D3(), Q2(), D2(), Q1(),
    # ──── Downstream straight end ─────────────────────────────────
    D1(),
]

# ── create_cell helper (used by stability sampler & optimiser) ────
def create_cell(quad_strengths):
    """Build one half-cell with custom quad K1 values.

    Parameters
    ----------
    quad_strengths : list of 9 floats
        [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9] K1 values in m⁻².

    Returns
    -------
    list : pyJuTrack element list for one half-cell.
    """
    k = quad_strengths
    return [
        D1(),
        jt.KQUAD("Q1", 0.255, k[0]), D2(), jt.KQUAD("Q2", 0.205, k[1]), D3(),
        B1(),
        D4(), jt.KQUAD("Q3", 0.180, k[2]), D5a(), SD1(), D5b(),
        jt.KQUAD("Q4", 0.180, k[3]), Q4b(),
        D6a(), SF1(), D6b(),
        jt.KQUAD("Q5", 0.205, k[4]), D7b(), OF(), D7a(), SD2(), D7c(),
        jt.KQUAD("Q6", 0.340, k[5]), D8(),
        B2(),
        D9(), jt.KQUAD("Q7", 0.385, k[6]), D10(),
        BD(),
        D11(), jt.KQUAD("Q8", 0.610, k[7]), Q8b(), D12a(), jt.KQUAD("Q9", 0.260, k[8]), D12b(),
        BC(),
        D12b(), jt.KQUAD("Q9", 0.260, k[8]), D12a(), Q8b(), jt.KQUAD("Q8", 0.610, k[7]), D11(),
        BD(),
        D10(), jt.KQUAD("Q7", 0.385, k[6]), D9(),
        B2(),
        D8(), jt.KQUAD("Q6", 0.340, k[5]), D7c(), SD3(), D7a(), OF(), D7b(),
        jt.KQUAD("Q5", 0.205, k[4]),
        D6b(), SF2(), D6a(),
        Q4b(), jt.KQUAD("Q4", 0.180, k[3]),
        D5b(), SD4(), D5a(), jt.KQUAD("Q3", 0.180, k[2]), D4(),
        B1(),
        D3(), jt.KQUAD("Q2", 0.205, k[1]), D2(), jt.KQUAD("Q1", 0.255, k[0]),
        D1(),
    ]

# 2 * N_full_periods copies of the half-cell span the full ring
ring = jt.Lattice(cell * (N_full_periods * 2))

# ── Diagnostics ───────────────────────────────────────────────────
nux, nuy = jt.gettune(ring)
print(f"HEPS-type 6 GeV MBA | {N_full_periods*2} half-cells | C≈1360 m")
print(f"Tunes: νx={nux:.6f}  νy={nuy:.6f}")
