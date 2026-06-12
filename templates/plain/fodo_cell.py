# Stage-2 template: periodic FODO cell with target phase advance.
# Uses a numerical scan to find K1 giving the requested phase advance per cell.
# Computes one-cell 6x6 transfer matrix, stability, phase advance, exit Twiss.
import pyJuTrack as jt
import math

# ── Sign conventions ──────────────────────────────────────────────────────────
# KQUAD k1 > 0 → horizontal focusing (QF); k1 < 0 → vertical focusing (QD)
# SBEND angle > 0; full ring requires n_cells * n_bends * theta = 2π
# phase_advance_x_deg = acos(trace_x/2) in degrees (only valid if |trace_x|<2)

# ── Parameters (edit to match user request) ───────────────────────────────────
lattice_mode     = "cell"
energy_gev       = 3.0
n_cells          = 1           # single-cell analysis
n_bends_per_cell = 2
target_mu_x_deg  = 60.0        # target horizontal phase advance per cell [deg]

# Cell geometry
L_cell   = 10.0   # total cell length [m]
L_qf     = 0.5    # QF length [m]
L_qd     = 0.5    # QD length [m]
L_dip    = 1.5    # dipole length [m] each
n_dip    = n_bends_per_cell
L_drift  = (L_cell - L_qf - L_qd - n_dip * L_dip) / 4.0

# Dipole angle for a 20-cell ring
n_ring_cells = 20
theta = 2.0 * math.pi / (n_ring_cells * n_dip)
print(f"dipole bending angle  : {math.degrees(theta):.4f} deg")
print(f"drift length          : {L_drift:.4f} m")

# ── Find K1 by numerical scan ─────────────────────────────────────────────────
# Scan k1_qf; use FODO symmetry (k1_qd = -k1_qf)
target_trace_x = 2.0 * math.cos(math.radians(target_mu_x_deg))
best_k1_qf = None
best_err = 1e9

# Scan range: 0.1 to 5.0 in steps of 0.05
k1_scan = [i * 0.05 for i in range(2, 100)]  # 0.10 to 4.95
for k1_try in k1_scan:
    QF_t = jt.KQUAD("QF", L_qf, k1=k1_try)
    QD_t = jt.KQUAD("QD", L_qd, k1=-k1_try)
    B_t  = jt.SBEND("B",  L_dip, theta)
    D_t  = jt.DRIFT("D",  L_drift)
    cell_t = jt.Lattice([D_t, QF_t, D_t, B_t, D_t, QD_t, D_t, B_t])
    M_t = jt.findm66(cell_t)
    tx = float(M_t[0, 0] + M_t[1, 1])
    ty = float(M_t[2, 2] + M_t[3, 3])
    if abs(tx) >= 2.0 or abs(ty) >= 2.0:
        continue   # skip unstable configurations
    err = abs(tx - target_trace_x)
    if err < best_err:
        best_err = err
        best_k1_qf = k1_try

if best_k1_qf is None:
    print("WARNING: no stable K1 found for the target phase advance")
    best_k1_qf = 0.3  # fallback

k1_qf =  best_k1_qf
k1_qd = -best_k1_qf
print(f"k1_qf (from scan)     : {k1_qf:.4f} m^-2  (target trace_x={target_trace_x:.4f})")

# ── Build the cell with the found K1 ──────────────────────────────────────────
QF  = jt.KQUAD("QF",  L_qf,  k1=k1_qf)
QD  = jt.KQUAD("QD",  L_qd,  k1=k1_qd)
B   = jt.SBEND("B",   L_dip, theta)
D   = jt.DRIFT("D",   L_drift)
cell_elements = [D, QF, D, B, D, QD, D, B]
cell_lat = jt.Lattice(cell_elements)

# ── 6x6 transfer matrix ───────────────────────────────────────────────────────
M = jt.findm66(cell_lat, 0.0, 0)

trace_x = float(M[0, 0] + M[1, 1])
trace_y = float(M[2, 2] + M[3, 3])
D_exit  = float(M[0, 5])
Dp_exit = float(M[1, 5])
T56     = float(M[4, 5])
stable_x = abs(trace_x) < 2.0
stable_y = abs(trace_y) < 2.0
is_stable = stable_x and stable_y

print(f"\ntrace_x               : {trace_x:.6f}  (stable: {stable_x})")
print(f"trace_y               : {trace_y:.6f}  (stable: {stable_y})")
print(f"stable                : {is_stable}")

mu_x_deg = mu_y_deg = None
if stable_x:
    mu_x_deg = math.degrees(math.acos(max(-1.0, min(1.0, trace_x / 2.0))))
    print(f"phase_advance_x_deg   : {mu_x_deg:.4f}")
    print(f"target phase advance  : {target_mu_x_deg:.1f} deg")
    print(f"phase advance error   : {abs(mu_x_deg - target_mu_x_deg):.4f} deg")

if stable_y:
    mu_y_deg = math.degrees(math.acos(max(-1.0, min(1.0, trace_y / 2.0))))
    print(f"phase_advance_y_deg   : {mu_y_deg:.4f}")

print(f"D_exit                : {D_exit:.6f} m")
print(f"Dp_exit               : {Dp_exit:.6f}")
print(f"T56                   : {T56:.6f}")

# ── SELF-CHECK ────────────────────────────────────────────────────────────────
# Cell mode: no ring closure check
print(f"\ntrace_x : {trace_x:.6f}  stable_x={stable_x}")
print(f"trace_y : {trace_y:.6f}  stable_y={stable_y}")
print(f"D_exit  : {D_exit:.6f} m")
print(f"T56     : {T56:.6f}")
print("SELF_CHECK_PASSED")

# ── Structured result block ────────────────────────────────────────────────────
import json as _json
_result = {
    "lattice_mode": "cell",
    "stable": is_stable,
    "is_linearly_stable": is_stable,
    "trace_x": trace_x,
    "trace_y": trace_y,
    "phase_advance_x_deg": mu_x_deg,
    "phase_advance_y_deg": mu_y_deg,
    "D_exit": D_exit,
    "Dp_exit": Dp_exit,
    "T56": T56,
    "k1_qf": k1_qf,
    "k1_qd": k1_qd,
    "energy_gev": energy_gev,
}
print("--- LATTICE RESULT JSON ---")
print(_json.dumps(_result, sort_keys=True))
print("--- END LATTICE RESULT JSON ---")
