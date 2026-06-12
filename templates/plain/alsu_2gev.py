"""ALS-U type lattice for pyJuTrack.

Source: alsu.jl (JuTrack/Julia, ALS-U validated design, 2 GeV)
Machine type: 4th-generation synchrotron light source (ultra-low emittance)
Structure: 12 superperiods, each with a straight (STR) and arc (ARC) section
           2 × BEND1 + 2 × BEND2 + 5 × BEND3 = 9 bends per superperiod
           Total: 12 × 9 = 108 bends, angle = 2π/108 per bend ✓

Combined-function dipoles:
  BEND1: L=0.34 m, angle=0.058178 rad, PolynomB[1]=−2.828  (quad component)
  BEND2: L=0.50 m, angle=0.058178 rad, PolynomB[1]=−7.058  (stronger quad)
  BEND3: L=0.50 m, angle=0.058178 rad, PolynomB[1]=−7.058  (same as BEND2)
The negative quad component in the bends is essential for ALS-U's ultra-low
emittance: it acts as a focusing element inside the dipole (longitudinal
gradient / combined-function technique).

No AT→JuTrack conversion needed – alsu.jl is already in JuTrack (Julia)
convention. Values are used as-is.
"""

import numpy as np
import pyJuTrack as jt

energy = 2.0e9    # eV  (ALS-U design energy 2 GeV)
N_superperiods = 12
bend_angle = 2 * np.pi / (N_superperiods * 9)   # = 2π/108 ≈ 0.058178 rad

# ── Quadrupoles (from alsu.jl, no conversion) ─────────────────────
def QF1(): return jt.KQUAD("QF1", 0.18,   13.84045)   # outer foc quad
def QD1(): return jt.KQUAD("QD1", 0.14,  -13.77444)   # outer defoc quad
def QF2(): return jt.KQUAD("QF2", 0.19,   10.19367)   # inner foc quad
def QF3(): return jt.KQUAD("QF3", 0.115,  10.94517)
def QF4(): return jt.KQUAD("QF4", 0.305,  15.32064)
def QF5(): return jt.KQUAD("QF5", 0.305,  15.80000)
def QF6(): return jt.KQUAD("QF6", 0.305,  15.68564)

# ── Combined-function dipoles (from alsu.jl) ─────────────────────
# PolynomB=[0, K1, 0, 0]: K1 is the quadrupole component inside the bend.
# Negative K1 = defocusing in horizontal, focusing in vertical (inside bend).
def BEND1(): return jt.SBEND("BEND1", 0.34, bend_angle,
                              PolynomB=[0.0, -2.827967, 0.0, 0.0])
def BEND2(): return jt.SBEND("BEND2", 0.50, bend_angle,
                              PolynomB=[0.0, -7.057813, 0.0, 0.0])
def BEND3(): return jt.SBEND("BEND3", 0.50, bend_angle,
                              PolynomB=[0.0, -7.057813, 0.0, 0.0])

# ── Sextupoles (from alsu.jl, JuTrack convention, no conversion) ─
def SHH():  return jt.KSEXT("SHH",  0.075,     3.514648)
def SHH2(): return jt.KSEXT("SHH2", 0.075,  -929.577200)   # strong harmonic
def SD():   return jt.KSEXT("SD",   0.28,  -1367.684109)   # defocusing sext
def SF():   return jt.KSEXT("SF",   0.28,   1610.769824)   # focusing sext

# ── Drifts (from alsu.jl) ─────────────────────────────────────────
def DX():  return jt.DRIFT("DX",  0.0375)
def D12(): return jt.DRIFT("D12", 0.075)
def D15(): return jt.DRIFT("D15", 0.225)
def D11(): return jt.DRIFT("D11", 0.500)    # straight section drift
def D11A(): return jt.DRIFT("D11A", 0.535)  # longer end of straight

# ── Straight section (5 drift segments, ~2.535 m total) ───────────
# In the ring: STR_B + ARC + STR_A (superperiod); the two half-straights
# from adjacent cells merge into a full 5.07 m straight section.
def STR_A_seg(): return [D11A(), D11(), D11(), D11(), D11()]
def STR_B_seg(): return [D11(), D11(), D11(), D11(), D11A()]

# ── Arc section (9 combined-function bends per arc) ───────────────
# From alsu.jl lines 43–49 (BPMs/correctors/markers omitted):
def ARC_seg():
    return [
        DX(), SHH(), D12(), QF1(), DX(), DX(), QD1(), DX(), DX(), SHH2(), D12(),
        BEND1(),
        DX(), DX(), SD(), D12(), QF2(), D12(), SF(), DX(), DX(), QF3(), D15(),
        BEND2(),
        DX(), DX(), QF4(), DX(), DX(),
        BEND3(), DX(), DX(), QF5(), D12(),
        BEND3(),
        DX(), DX(), QF6(), DX(), DX(),
        BEND3(), DX(), DX(), QF6(), DX(), DX(),
        BEND3(),
        DX(), DX(), QF5(), D12(),
        BEND3(), D12(), QF4(), DX(), DX(),
        BEND2(),
        D15(), QF3(), D12(), SF(), DX(), DX(), QF2(), D12(), SD(), DX(), DX(),
        BEND1(),
        DX(), DX(), SHH2(), DX(), DX(), QD1(), DX(), DX(), QF1(), D12(), SHH(), DX(),
    ]

# ── Build the single-superperiod cell ────────────────────────────
cell = STR_B_seg() + ARC_seg() + STR_A_seg()

ring = jt.Lattice(cell * N_superperiods)

# ── Diagnostics ───────────────────────────────────────────────────
nux, nuy = jt.gettune(ring)
str_len = 2 * (0.535 + 4 * 0.5)   # full straight per superperiod
print(f"ALS-U type 2 GeV | {N_superperiods} superperiods | 108 combined-function bends")
print(f"Tunes: νx={nux:.6f}  νy={nuy:.6f}")
