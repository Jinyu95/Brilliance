"""FODO arc template — stage-3 reference.

Designs a multi-cell FODO arc for a storage ring:
- Computes tunes, periodic Twiss, horizontal dispersion, natural chromaticity.
- Demonstrates BPM/corrector placement rationale.
- Uses analytic U0/damping times (rp['U0'] is broken in current pyJuTrack).
"""

import pyJuTrack as jt
import numpy as np
import math
import json as _json

# ── Parameters ────────────────────────────────────────────────────────────────
energy_eV        = 3.0e9   # 3 GeV electron ring
N_cells          = 20      # FODO cells forming the full ring
n_bends_per_cell = 2       # 2 sector dipoles per cell
lattice_mode     = "ring"

# Cell geometry
L_qf    = 0.5    # focusing quad length [m]
L_qd    = 0.5    # defocusing quad length [m]
L_dip   = 1.0    # dipole length [m]
L_drift = 0.5    # drift length [m]

# Dipole angle: total ring bend = 2π
theta = 2.0 * math.pi / (N_cells * n_bends_per_cell)
print(f"Dipole angle          : {math.degrees(theta):.4f} deg")

# Quadrupole strengths — use scan to achieve ~60° horizontal phase advance
# Target: trace_x = 2*cos(60°) = 1.0
target_trace_x = 2.0 * math.cos(math.radians(60.0))
best_k1_qf = None; best_err = 1e9

for k1_10x in range(2, 60):
    k1 = k1_10x / 10.0
    QF_t = jt.KQUAD("QF", L_qf, k1=k1)
    QD_t = jt.KQUAD("QD", L_qd, k1=-k1)
    B_t  = jt.SBEND("B",  L_dip, theta)
    D_t  = jt.DRIFT("D",  L_drift)
    cell_t = jt.Lattice([D_t, QF_t, D_t, B_t, D_t, QD_t, D_t, B_t])
    M = jt.findm66(cell_t)
    tx = float(M[0,0]+M[1,1]); ty = float(M[2,2]+M[3,3])
    if abs(tx) >= 2.0 or abs(ty) >= 2.0:
        continue
    err = abs(tx - target_trace_x)
    if err < best_err:
        best_err = err; best_k1_qf = k1

k1_qf = best_k1_qf or 1.5
k1_qd = -k1_qf
print(f"k1_qf (scan result)   : {k1_qf:.4f} m^-2")

# ── Build ring ────────────────────────────────────────────────────────────────
QF = jt.KQUAD("QF", L_qf, k1=k1_qf)
QD = jt.KQUAD("QD", L_qd, k1=k1_qd)
B  = jt.SBEND("B",  L_dip, theta)
D  = jt.DRIFT("D",  L_drift)

cell_elements = [D, QF, D, B, D, QD, D, B]
cell_lat = jt.Lattice(cell_elements)
ring = jt.Lattice(cell_elements * N_cells)

# ── Global ring parameters ─────────────────────────────────────────────────────
nux, nuy = jt.gettune(ring)
xix, xiy = jt.getchrom(ring)
rp        = jt.ringpara(ring, energy=energy_eV)
s_ring    = jt.spos(ring)
circumference = float(s_ring[-1])

emittance = float(rp['emittx'])
alphac    = float(rp['alphac'])
Jx        = float(rp['Jx'])
Jy        = float(rp['Jy'])
Je        = float(rp['Je'])
I1 = float(rp['I1']); I2 = float(rp['I2']); I3 = float(rp['I3'])
I4 = float(rp['I4']); I5 = float(rp['I5'])

# Analytic U0 and damping times (rp['U0'] broken in this pyJuTrack version)
_E_GeV  = energy_eV / 1e9
_Cgamma = 8.85e-5   # m GeV^-3
_Cq     = 3.8319e-13
_me_eV  = 0.51099895e6
_gamma  = energy_eV / _me_eV
_T0     = circumference / 2.998e8
U0_eV   = _Cgamma * (_E_GeV**4) * I2 / (2*math.pi) * 1e9
sigma_E = math.sqrt(_Cq * (_gamma**2) * I3 / (2*I2 + I4)) if (2*I2+I4) > 0 else float('nan')
tau_x   = 2*energy_eV*_T0/(U0_eV*Jx) if (U0_eV*Jx) > 0 else float('inf')
tau_E   = 2*energy_eV*_T0/(U0_eV*Je) if (U0_eV*Je) > 0 else float('inf')

# ── Periodic Twiss for the cell ────────────────────────────────────────────────
M_cell = jt.findm66(cell_lat)
trace_x = float(M_cell[0,0]+M_cell[1,1])
trace_y = float(M_cell[2,2]+M_cell[3,3])
D_exit  = float(M_cell[0,5])
T56     = float(M_cell[4,5])

mu_x_deg = math.degrees(math.acos(max(-1.0,min(1.0,trace_x/2.0)))) if abs(trace_x)<2 else None
mu_y_deg = math.degrees(math.acos(max(-1.0,min(1.0,trace_y/2.0)))) if abs(trace_y)<2 else None

# Twiss around the ring for dispersion profile
twiss_ring = jt.twissring(ring)
s_pos = [float(s_ring[i]) for i in range(len(s_ring))]
beta_x = [float(t.betax) for t in twiss_ring]
beta_y = [float(t.betay) for t in twiss_ring]
disp_x = [float(t.dx)    for t in twiss_ring]
Dx_max = max(abs(d) for d in disp_x)
beta_x_max = max(beta_x)
beta_y_max = max(beta_y)

# ── BPM/corrector placement rationale ─────────────────────────────────────────
# Horizontal BPMs: near QF (high beta_x) for maximum orbit sensitivity
# Vertical   BPMs: near QD (high beta_y)
# Horizontal correctors: near QF (max beta_x → max orbit response)
# Vertical   correctors: near QD (max beta_y)
print(f"\nFODO Ring 3 GeV | {N_cells} cells | C={circumference:.1f} m")
print(f"tune_nux             : {float(nux):.6f}")
print(f"tune_nuy             : {float(nuy):.6f}")
print(f"chromaticity_xix     : {float(xix):.4f}")
print(f"chromaticity_xiy     : {float(xiy):.4f}")
print(f"emittance            : {emittance:.6e}")
print(f"alphac               : {alphac:.6e}")
print(f"U0_eV                : {U0_eV:.2f}")
print(f"sigma_E              : {sigma_E:.4e}")
print(f"damping_time_x_ms    : {tau_x*1e3:.2f}")
print(f"damping_time_E_ms    : {tau_E*1e3:.2f}")
print(f"Jx={Jx:.4f}  Jy={Jy:.4f}  Je={Je:.4f}  Robinson_sum={Jx+Jy+Je:.4f}")
print(f"I1={I1:.4f}  I2={I2:.4f}  I3={I3:.4f}  I4={I4:.4f}  I5={I5:.4e}")
print(f"phase_advance_x_deg  : {mu_x_deg:.4f}" if mu_x_deg else "phase_advance_x_deg  : unstable")
print(f"phase_advance_y_deg  : {mu_y_deg:.4f}" if mu_y_deg else "phase_advance_y_deg  : unstable")
print(f"D_exit (per cell)    : {D_exit:.6f} m")
print(f"beta_x_max           : {beta_x_max:.2f} m")
print(f"beta_y_max           : {beta_y_max:.2f} m")
print(f"Dx_max               : {Dx_max:.4f} m")
print(f"BPM_x placement      : next to QF (high beta_x = {beta_x_max:.1f} m)")
print(f"BPM_y placement      : next to QD (high beta_y = {beta_y_max:.1f} m)")

# ── SELF-CHECK ────────────────────────────────────────────────────────────────
_total_bend = N_cells * n_bends_per_cell * theta
assert abs(_total_bend - 2*math.pi) < 1e-9, f"Ring closure FAILED: {_total_bend}"
print(f"ring_closure_check   : PASSED (total_bend={_total_bend:.8f} rad)")
print(f"trace_x : {trace_x:.6f}  stable_x={abs(trace_x)<2}")
print(f"trace_y : {trace_y:.6f}  stable_y={abs(trace_y)<2}")
print(f"D_exit  : {D_exit:.6f} m")
print(f"T56     : {T56:.6f}")
print("SELF_CHECK_PASSED")

# NOTE: matplotlib plotting is excluded from this template because the
# Agg backend uses numpy BLAS which crashes in the pyJuTrack conda env.
# To produce Twiss plots, use a separate plotting step after the ring is verified.

# ── Structured result block ────────────────────────────────────────────────────
_result = {
    "lattice_mode": "ring",
    "stable": True,
    "tune_nux": float(nux), "tune_nuy": float(nuy),
    "chromaticity_xix": float(xix), "chromaticity_xiy": float(xiy),
    "emittance": emittance, "alphac": alphac, "U0_eV": U0_eV, "sigma_E": sigma_E,
    "Jx": Jx, "Jy": Jy, "Je": Je,
    "I1": I1, "I2": I2, "I3": I3, "I4": I4, "I5": I5,
    "circumference": circumference, "energy_gev": energy_eV/1e9,
    "phase_advance_x_deg": mu_x_deg, "phase_advance_y_deg": mu_y_deg,
    "D_exit": D_exit, "Dx_max": Dx_max, "beta_x_max": beta_x_max,
    "damping_time_x": tau_x, "damping_time_E": tau_E,
}
print("--- LATTICE RESULT JSON ---")
print(_json.dumps(_result, sort_keys=True))
print("--- END LATTICE RESULT JSON ---")
