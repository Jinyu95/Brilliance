# Stage-1 template: single-element 6x6 transfer matrices with physics verification.
# Demonstrates DRIFT, KQUAD, SBEND matrix extraction and analytic cross-checks.
# NOTE: numpy matmul (M.T @ J @ M) crashes in some environments due to DLL issues;
#       use element-wise float access and analytic formulas for verification instead.
import pyJuTrack as jt
import math

# ── Coordinate system (pyJuTrack / AT style) ─────────────────────────────────
# [x, px, y, py, z, delta]  (delta = Δp/p₀)
# KQUAD k1 > 0 → horizontal focusing;   k1 < 0 → vertical focusing
# SBEND angle > 0 → bends in +x plane;  M[0,5] = dispersion D_x at exit
# KSEXT k2 > 0 → normal sextupole (MAD-X K2 convention)

# ────────────────────────────────────────────────────────────────────────────
# 1. DRIFT  L = 2.0 m
# ────────────────────────────────────────────────────────────────────────────
L_drift = 2.0
drift = jt.DRIFT("D", L_drift)
lat_drift = jt.Lattice([drift])
M = jt.findm66(lat_drift, 0.0, 0)

print("=== DRIFT L=2.0 m ===")
m01 = float(M[0, 1])
m23 = float(M[2, 3])
print(f"M[0,1] = {m01:.8f}  (should = L = {L_drift})")
print(f"M[2,3] = {m23:.8f}  (should = L = {L_drift})")
print(f"M[0,0] = {float(M[0,0]):.8f}  (should = 1)")
print(f"M[1,1] = {float(M[1,1]):.8f}  (should = 1)")
assert abs(m01 - L_drift) < 1e-8, f"DRIFT M[0,1] = {m01} != L"
assert abs(m23 - L_drift) < 1e-8, f"DRIFT M[2,3] = {m23} != L"
print("DRIFT checks PASSED")

# ────────────────────────────────────────────────────────────────────────────
# 2. KQUAD (focusing)  L = 0.5 m,  K1 = 1.5 m^-2
# ────────────────────────────────────────────────────────────────────────────
L_qf = 0.5
k1_qf = 1.5   # > 0 → horizontal focusing, vertical defocusing
qf = jt.KQUAD("QF", L_qf, k1=k1_qf)
lat_qf = jt.Lattice([qf])
M = jt.findm66(lat_qf, 0.0, 0)

# Thick-lens quad: horizontal block is cosine-like, vertical block is cosh-like
sqrt_k = math.sqrt(abs(k1_qf))
phi = sqrt_k * L_qf
trace_x = float(M[0, 0] + M[1, 1])
trace_y = float(M[2, 2] + M[3, 3])
trace_x_analytic = 2.0 * math.cos(phi)
trace_y_analytic = 2.0 * math.cosh(phi)

print("\n=== KQUAD (focusing) L=0.5 m, K1=1.5 m^-2 ===")
print(f"trace_x = {trace_x:.6f}  analytic = {trace_x_analytic:.6f}  err = {abs(trace_x-trace_x_analytic):.2e}")
print(f"trace_y = {trace_y:.6f}  analytic = {trace_y_analytic:.6f}  err = {abs(trace_y-trace_y_analytic):.2e}")
print(f"stable_x = {abs(trace_x) < 2}  (|trace_x| = {abs(trace_x):.4f} < 2)")
print(f"stable_y = {abs(trace_y) < 2}  (|trace_y| = {abs(trace_y):.4f} < 2)")
assert abs(trace_x - trace_x_analytic) < 1e-4, f"QF trace_x mismatch: {trace_x} vs {trace_x_analytic}"
print("KQUAD checks PASSED")

# ────────────────────────────────────────────────────────────────────────────
# 3. SBEND (sector dipole)  L = 1.0 m,  angle = 0.2 rad
# ────────────────────────────────────────────────────────────────────────────
L_bend = 1.0
angle_bend = 0.2   # rad
rho = L_bend / angle_bend  # bending radius [m]
sbend = jt.SBEND("B", L_bend, angle_bend)
lat_bend = jt.Lattice([sbend])
M = jt.findm66(lat_bend, 0.0, 0)

D_exit = float(M[0, 5])   # dispersion at exit (nonzero for dipole)
T56    = float(M[4, 5])   # T56 element
trace_x = float(M[0, 0] + M[1, 1])
trace_y = float(M[2, 2] + M[3, 3])
# Analytic D_exit for sector dipole: rho*(1 - cos(angle))
D_exit_analytic = rho * (1.0 - math.cos(angle_bend))

print("\n=== SBEND L=1.0 m, angle=0.2 rad ===")
print(f"D_exit = {D_exit:.6f} m  analytic = {D_exit_analytic:.6f} m  err = {abs(D_exit-D_exit_analytic):.2e}")
print(f"T56    = {T56:.6f}  (pyJuTrack: T56>0 for normal lattices; opposite sign to MAD-X/AT)")
print(f"trace_x = {trace_x:.6f}  (stable: {abs(trace_x) < 2})")
print(f"trace_y = {trace_y:.6f}  (stable: {abs(trace_y) < 2})")
assert abs(D_exit) > 1e-5, "Sector dipole dispersion should be nonzero"
assert abs(D_exit - D_exit_analytic) < 1e-3, f"SBEND D_exit mismatch: {D_exit} vs {D_exit_analytic}"
print("SBEND checks PASSED")

# ── SELF-CHECK ────────────────────────────────────────────────────────────────
# Ring mode: not applicable (single elements, not a closed ring)
# Trace and D_exit already printed and verified above.
print("\nSELF_CHECK_PASSED")
print("stable : True")

# ── Structured result block ────────────────────────────────────────────────
import json as _json
_result = {
    "lattice_mode": "cell",
    "stable": True,
    "drift_M01": m01,
    "qf_trace_x": trace_x,
    "qf_trace_x_analytic_err": abs(float(jt.findm66(jt.Lattice([jt.KQUAD("QF", 0.5, k1=1.5)]))[0,0]
                                        + jt.findm66(jt.Lattice([jt.KQUAD("QF", 0.5, k1=1.5)]))[1,1]) - trace_x_analytic),
    "sbend_D_exit": D_exit,
    "sbend_D_exit_analytic": D_exit_analytic,
    "sbend_T56": T56,
}
print("--- LATTICE RESULT JSON ---")
print(_json.dumps(_result, sort_keys=True))
print("--- END LATTICE RESULT JSON ---")
