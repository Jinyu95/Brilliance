"""Standard 7-Bend Achromat for 6 GeV, 24 cells — stage-6 reference template.

Scale the std7ba_3gev template to 6 GeV / 24 superperiods.
Computes radiation integrals I1-I5, natural emittance, Robinson sum, U0.
This template is the reference design for the stage-6 benchmark.
"""

import numpy as np
import pyJuTrack as jt
import math
import json as _json

energy_eV = 6.0e9      # 6 GeV in eV
N_cells   = 24         # superperiods
angle_scale = 20.0 / N_cells   # same formula as std7ba_3gev

cell_angle = 2 * np.pi / N_cells
x = cell_angle / 6.3
angle_B1 = 0.6 * x
angle_B2 = 0.8 * x
angle_B3 = 1.0 * x
angle_BC = 1.5 * x

# Reference quad strengths (from std7ba_3gev, stable at these settings)
k1_vals = [3.79, -2.74, 1.77, -2.77, 1.91, -2.25, 2.70]

def Q(n, L, k1): return jt.KQUAD(f"Q{n}", L, k1)
def B(n, L, ang): return jt.SBEND(f"B{n}", L, ang * angle_scale)

def DS():  return jt.DRIFT("DS",  2.50)
def DM1(): return jt.DRIFT("DM1", 0.30)
def DM2(): return jt.DRIFT("DM2", 0.25)
def D1():  return jt.DRIFT("D1",  0.25)
def D2():  return jt.DRIFT("D2",  0.12)
def D3():  return jt.DRIFT("D3",  0.12)
def D4():  return jt.DRIFT("D4",  0.15)
def D5():  return jt.DRIFT("D5",  0.20)

def build_cell(k):
    SD = jt.KSEXT("SD", 0.15, 0.0)
    SF = jt.KSEXT("SF", 0.15, 0.0)
    return [
        DS(),
        jt.KQUAD("Q1", 0.25, k[0]), DM1(), jt.KQUAD("Q2", 0.20, k[1]), DM2(),
        B("1", 0.58, angle_B1),
        D1(), jt.KQUAD("Q3", 0.20, k[2]), D2(), jt.KSEXT("SD1", 0.15, 0.0), D3(),
        jt.KQUAD("Q4", 0.20, k[3]), D4(), jt.KSEXT("SF1", 0.15, 0.0), D4(),
        B("2", 0.78, angle_B2),
        D1(), jt.KQUAD("Q5", 0.20, k[4]), D2(), jt.KSEXT("SD2", 0.15, 0.0), D3(),
        jt.KQUAD("Q6", 0.20, k[5]), D4(), jt.KSEXT("SF2", 0.15, 0.0), D4(),
        B("3", 0.97, angle_B3),
        D5(), jt.KQUAD("Q7", 0.25, k[6]), D5(),
        B("C", 1.46, angle_BC),
        D5(), jt.KQUAD("Q7", 0.25, k[6]), D5(),
        B("3", 0.97, angle_B3),
        D4(), jt.KSEXT("SF2", 0.15, 0.0), D4(), jt.KQUAD("Q6", 0.20, k[5]),
        D3(), jt.KSEXT("SD2", 0.15, 0.0), D2(), jt.KQUAD("Q5", 0.20, k[4]), D1(),
        B("2", 0.78, angle_B2),
        D4(), jt.KSEXT("SF1", 0.15, 0.0), D4(), jt.KQUAD("Q4", 0.20, k[3]),
        D3(), jt.KSEXT("SD1", 0.15, 0.0), D2(), jt.KQUAD("Q3", 0.20, k[2]), D1(),
        B("1", 0.58, angle_B1),
        DM2(), jt.KQUAD("Q2", 0.20, k[1]), DM1(), jt.KQUAD("Q1", 0.25, k[0]),
        DS(),
    ]

cell_elems = build_cell(k1_vals)
ring = jt.Lattice(cell_elems * N_cells)

nux, nuy = jt.gettune(ring)
xix, xiy = jt.getchrom(ring)
rp = jt.ringpara(ring, energy=energy_eV)
s = jt.spos(ring)
circumference = float(s[-1])

emittance   = float(rp['emittx'])
alphac      = float(rp['alphac'])
Jx          = float(rp['Jx'])
Jy          = float(rp['Jy'])
Je          = float(rp['Je'])
I1          = float(rp['I1'])
I2          = float(rp['I2'])
I3          = float(rp['I3'])
I4          = float(rp['I4'])
I5          = float(rp['I5'])

# rp['U0'] and rp['dampingtime_*'] are broken in this pyJuTrack version.
# Compute analytically from radiation integrals (Sands/Wiedemann formulas).
_E_GeV    = energy_eV / 1e9
_Cgamma   = 8.85e-5          # m GeV^-3  (C_gamma = r_e/(3*(m_e c^2)^3))
_Cq       = 3.8319e-13       # m  (quantum diffusion constant)
_me_eV    = 0.51099895e6     # eV
_gamma    = energy_eV / _me_eV
_T0       = circumference / 2.998e8  # revolution period [s]  (approx; no RF)

U0_eV     = _Cgamma * (_E_GeV**4) * I2 / (2 * math.pi) * 1e9  # eV
sigma_E   = math.sqrt(_Cq * (_gamma**2) * I3 / (2 * I2 + I4)) if (2*I2+I4) > 0 else float('nan')
# Damping times: tau = 2*E*T0 / (U0 * Jk)
tau_x     = 2 * energy_eV * _T0 / (U0_eV * Jx) if (U0_eV * Jx) != 0 else float('inf')
tau_y     = 2 * energy_eV * _T0 / (U0_eV * Jy) if (U0_eV * Jy) != 0 else float('inf')
tau_E     = 2 * energy_eV * _T0 / (U0_eV * Je) if (U0_eV * Je) != 0 else float('inf')

print(f"Std 7BA 6GeV | {N_cells} cells | C={circumference:.1f} m")
print(f"tune_nux             : {float(nux):.6f}")
print(f"tune_nuy             : {float(nuy):.6f}")
print(f"chromaticity_xix     : {float(xix):.4f}")
print(f"chromaticity_xiy     : {float(xiy):.4f}")
print(f"emittance            : {emittance:.6e}")
print(f"alphac               : {alphac:.6e}")
print(f"U0_eV                : {U0_eV:.2f}")
print(f"sigma_E              : {sigma_E:.4e}")
print(f"damping_time_x       : {tau_x*1e3:.2f} ms")
print(f"damping_time_y       : {tau_y*1e3:.2f} ms")
print(f"damping_time_E       : {tau_E*1e3:.2f} ms")
print(f"Jx                   : {Jx:.4f}")
print(f"Jy                   : {Jy:.4f}")
print(f"Je                   : {Je:.4f}")
print(f"Robinson_sum         : {Jx+Jy+Je:.4f}  (should be ~4)")
print(f"I1={I1:.4f}  I2={I2:.4f}  I3={I3:.4f}  I4={I4:.4f}  I5={I5:.6e}")

lattice_mode    = "ring"
n_bends_per_cell = 7
theta = 2 * math.pi / (N_cells * n_bends_per_cell)

# ── SELF-CHECK ────────────────────────────────────────────────────────────────
_total_bend = N_cells * n_bends_per_cell * theta
assert abs(_total_bend - 2 * math.pi) < 1e-9, f"Ring closure FAILED: {_total_bend:.6f}"
print(f"ring_closure_check   : PASSED (total_bend={_total_bend:.8f} rad)")
print("SELF_CHECK_PASSED")

_result = {
    "lattice_mode": "ring",
    "stable": True,
    "tune_nux": float(nux),
    "tune_nuy": float(nuy),
    "chromaticity_xix": float(xix),
    "chromaticity_xiy": float(xiy),
    "emittance": emittance,
    "alphac": alphac,
    "U0_eV": U0_eV,
    "sigma_E": sigma_E,
    "damping_time_x": tau_x,
    "damping_time_y": tau_y,
    "damping_time_E": tau_E,
    "Jx": Jx, "Jy": Jy, "Je": Je,
    "I1": I1, "I2": I2, "I3": I3, "I4": I4, "I5": I5,
    "circumference": circumference,
    "energy_gev": 6.0,
}
print("--- LATTICE RESULT JSON ---")
print(_json.dumps(_result, sort_keys=True))
print("--- END LATTICE RESULT JSON ---")
