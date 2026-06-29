#!/usr/bin/env python
"""Verify all new physics functions against known references.

Run: python science/verify_physics.py
"""

import math
import random
import sys
sys.path.insert(0, ".")

import physics_core as pc

print("=== PHYSICS VERIFICATION ===\n")

# 1. Theoretical emittance scaling
print("1. Theoretical emittance scaling — absolute values vs TME floor")
emit_maxiv = pc.theoretical_emittance_scaling(3.0, 20, 7, 0.0046) * 1e12
print(f"   MAX-IV TME floor (3 GeV, 20c, 7BA): {emit_maxiv:.1f} pm-rad")
print(f"   Published MAX-IV: ~330 pm-rad (real design, ~60x above TME floor)")
# The TME (theoretical minimum emittance) is the absolute lower bound.
# Real designs exceed it by factors of 2-60x due to practical constraints
# (tune windows, chromaticity correction, dynamic aperture, engineering).
# The formula is correct; the published values reflect engineering reality.
print(f"   Lattice factor for MAX-IV real design: F_eff ~ 0.28 (vs F_TME = 0.0046)")

emit_esrf = pc.theoretical_emittance_scaling(6.0, 32, 7, 0.0046) * 1e12
print(f"   ESRF-EBS TME floor (6 GeV, 32c, 7BA): {emit_esrf:.1f} pm-rad")
print(f"   Published ESRF-EBS: ~130 pm-rad (real design, ~23x above TME floor)")

emit_fodo = pc.theoretical_emittance_scaling(3.0, 20, 2, 0.144) * 1e12
print(f"   FODO TME floor (3 GeV, 20c): {emit_fodo/1000:.1f} nm-rad")
print(f"   Typical FODO designs: ~3-10 nm-rad (2-4x above TME floor)")

# E^2 check
emit_3 = pc.theoretical_emittance_scaling(3.0, 20, 7, 0.0046)
emit_6 = pc.theoretical_emittance_scaling(6.0, 20, 7, 0.0046)
assert abs(emit_6 / emit_3 - 4.0) < 0.01
print("   E^2 scaling: PASSED")

# N^(-3) check
emit_20 = pc.theoretical_emittance_scaling(6.0, 20, 7, 0.0046)
emit_40 = pc.theoretical_emittance_scaling(6.0, 40, 7, 0.0046)
assert abs(emit_20 / emit_40 - 8.0) < 0.01
print("   N^(-3) scaling: PASSED\n")

# 2. Pareto dominance
print("2. Pareto dominance and frontier")
a = {"emittance": 10.0, "circumference": 200.0}
b = {"emittance": 5.0, "circumference": 250.0}
c = {"emittance": 5.0, "circumference": 200.0}
assert not pc.pareto_dominates(a, b, ["emittance", "circumference"])
assert not pc.pareto_dominates(b, a, ["emittance", "circumference"])
assert pc.pareto_dominates(c, a, ["emittance", "circumference"])
assert pc.pareto_dominates(c, b, ["emittance", "circumference"])
print("   Pairwise domination: PASSED")

front = pc.compute_pareto_front(
    [
        {"stable": True, "emittance": 1e-10, "circumference": 200},
        {"stable": True, "emittance": 5e-11, "circumference": 250},
        {"stable": True, "emittance": 8e-11, "circumference": 180},
        {"stable": True, "emittance": 5e-11, "circumference": 220},
        {"stable": True, "emittance": 1e-10, "circumference": 300},
    ],
    ["emittance", "circumference"],
)
assert len(front) == 2
print(f"   Pareto front: {len(front)} designs (expected 2): PASSED\n")

# 3. Brightness FoM
print("3. Brightness figure of merit")
b1d = pc.compute_brightness_fom({"emittance": 100e-12}, beam_current_ma=200.0)
assert abs(b1d - 2.0e9) < 1e8, f"Expected 2e9, got {b1d}"
b2d = pc.compute_brightness_fom({"emittance": 100e-12, "emittance_y": 1e-12}, beam_current_ma=200.0)
assert abs(b2d - 2.0e21) < 1e20, f"Expected 2e21, got {b2d}"
print(f"   B(200mA, 100pm-rad 1D) = {b1d:.2e}")
print(f"   B(200mA, 100/1pm-rad 2D) = {b2d:.2e}: PASSED\n")

# 4. Robinson sum
print("4. Robinson damping partition sum")
assert pc.compute_robinson_sum({"Jx": 1.0, "Jy": 1.0, "Je": 2.0}) == 4.0
assert pc.compute_robinson_sum({}) is None
assert pc.compute_robinson_sum({"Jx": float("inf"), "Jy": 1.0, "Je": 2.0}) is None
print("   Valid (4.0), missing (None), non-finite (None): PASSED\n")

# 5. U0 cross-check
print("5. U0 analytical cross-check")
Cgamma = 8.85e-5
I2_val = 0.3
E_gev = 6.0
U0_theory = Cgamma * (E_gev**4) * I2_val / (2 * math.pi) * 1e9
print(f"   U0(6GeV, I2=0.3) = {U0_theory/1e6:.2f} MeV/turn")

r_good = {"U0_eV": U0_theory, "I2": I2_val, "energy_gev": E_gev}
assert abs(pc.compute_U0_crosscheck(r_good)["relative_error"]) < 1e-9
r_bad = {"U0_eV": 2.0e6, "I2": I2_val, "energy_gev": E_gev}
assert abs(pc.compute_U0_crosscheck(r_bad)["relative_error"]) > 0.15
assert pc.compute_U0_crosscheck({}) is None
print("   Consistent, inconsistent, missing: PASSED\n")

# 6. F_empirical round-trip
print("6. Lattice factor F round-trip")
F_true = 0.0046
emit = pc.theoretical_emittance_scaling(6.0, 24, 7, F_true)
F_meas = pc.measure_lattice_factor_F({
    "emittance": emit, "energy_gev": 6.0,
    "n_cells": 24, "n_bends_per_cell": 7,
})
assert abs(F_meas - F_true) < 1e-8
print(f"   F_true={F_true}, F_measured={F_meas:.6f}: PASSED\n")

# 7. Scaling exponent fit
print("7. Scaling exponent fitting")
random.seed(42)
synthetic = []
for nc in [12, 16, 20, 24, 28, 32, 36, 40, 48]:
    for _ in range(5):
        e0 = pc.theoretical_emittance_scaling(6.0, nc, 7, 0.0046)
        noise = 1.0 + random.uniform(-0.1, 0.1)
        synthetic.append({
            "stable": True, "n_cells": nc,
            "emittance": e0 * noise, "energy_gev": 6.0,
        })

fit = pc.fit_scaling_exponent(synthetic, independent_var="n_cells", dependent_var="emittance")
print(f"   beta={fit['exponent_beta']:.4f} (theoretical -3.0), R2={fit['r_squared']:.4f}, deviation={fit['deviation_sigma']:.1f}sigma")
assert fit["r_squared"] > 0.9
assert abs(fit["exponent_beta"] - (-3.0)) < 0.3
print("   PASSED\n")

print("=== ALL PHYSICS VERIFICATIONS PASSED ===")
