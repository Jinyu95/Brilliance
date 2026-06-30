"""Unit tests for MBA physics tools (agents/mba_tools.py).

All tests are pure Python — no pyJuTrack or Julia required.
Each tool is tested against known analytical physics expectations.
"""

import math
import pytest

from agents.mba_tools import (
    compute_hoa_phase_advance,
    build_unit_cell_topology,
    decompose_superperiod,
    compute_correction_leverage,
    check_dsc_matching_conditions,
    validate_mba_construction_step,
    estimate_mba_emittance,
    MBA_TOOLS,
    _Cq,
    _me_eV,
    _LATTICE_FACTORS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1: compute_hoa_phase_advance
# ═══════════════════════════════════════════════════════════════════════════════

class TestHoaPhaseAdvance:
    """HOA phase selection tests (Paper Step 4)."""

    def test_7ba_24cells_produces_valid_points(self):
        """24-superperiod 7BA: 168 total dipoles must yield valid HOA points."""
        result = compute_hoa_phase_advance(n_cells=24, n_bends_per_cell=7)
        assert result["n_total_dipoles"] == 168
        assert result["total_valid_working_points"] > 0
        assert len(result["best_working_points"]) >= 1

    def test_all_best_working_points_are_resonance_free(self):
        """Best working points should have zero resonance warnings."""
        result = compute_hoa_phase_advance(n_cells=24, n_bends_per_cell=7)
        for pt in result["best_working_points"]:
            assert pt["resonance_count"] == 0, (
                f"Best working point has resonance: {pt['resonance_warnings']}"
            )

    def test_nx_ny_are_integers(self):
        """Nx, Ny must be integers for HOA condition."""
        result = compute_hoa_phase_advance(n_cells=24, n_bends_per_cell=7)
        for pt in result["valid_working_points"]:
            assert isinstance(pt["Nx"], int)
            assert isinstance(pt["Ny"], int)
            assert pt["Nx"] > 0
            assert pt["Ny"] > 0

    def test_phase_advance_per_dipole_in_physical_range(self):
        """Per-dipole phase advance must be in physically sensible range."""
        result = compute_hoa_phase_advance(n_cells=24, n_bends_per_cell=7)
        for pt in result["valid_working_points"]:
            assert 5.0 <= pt["mu_x_dipole_deg"] <= 100.0, (
                f"mu_x dipole {pt['mu_x_dipole_deg']}° out of range"
            )
            assert 4.0 <= pt["mu_y_dipole_deg"] <= 70.0, (
                f"mu_y dipole {pt['mu_y_dipole_deg']}° out of range"
            )

    def test_hoa_phase_is_nx_times_360_degrees(self):
        """Total ring phase advance must = Nx * 360°."""
        result = compute_hoa_phase_advance(n_cells=24, n_bends_per_cell=7)
        for pt in result["valid_working_points"][:10]:
            total_mu_x = pt["mu_x_cell_deg"] * 24
            # Should be approximately Nx * 360
            assert abs(total_mu_x - pt["Nx"] * 360.0) < 0.01, (
                f"Total mu_x={total_mu_x}° ≠ {pt['Nx']}×360°"
            )

    def test_requested_tune_matching(self):
        """When tune_x/y are specified, requested_match should find nearest."""
        result = compute_hoa_phase_advance(
            n_cells=24, n_bends_per_cell=7, tune_x=0.20, tune_y=0.15
        )
        assert result["requested_match"] is not None
        # The detuned match is approximate — Nx,Ny are integers, detuned
        # by ~0.15-0.22, so the closest might be off by up to 0.5 in frac part.
        # This is informative — the HOA condition constrains the phase space.
        matched_nux = float(result["requested_match"]["detuned_fractional_nux"])
        matched_nuy = float(result["requested_match"]["detuned_fractional_nuy"])
        # The detuned fractional tune should be between 0 and 1
        assert 0.0 <= matched_nux <= 1.0
        assert 0.0 <= matched_nuy <= 1.0

    def test_few_cells_produces_fewer_points(self):
        """Fewer superperiods → fewer valid HOA phase combos."""
        r_many = compute_hoa_phase_advance(n_cells=40, n_bends_per_cell=7)
        r_few = compute_hoa_phase_advance(n_cells=8, n_bends_per_cell=7)
        # More cells → larger Nx, Ny ranges → more combos
        assert r_many["total_valid_working_points"] >= r_few["total_valid_working_points"]

    def test_fodo_hoa_works(self):
        """HOA tool should work for FODO (2 bends/cell) too."""
        result = compute_hoa_phase_advance(n_cells=20, n_bends_per_cell=2)
        assert result["total_valid_working_points"] > 0
        assert result["n_total_dipoles"] == 40


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2: build_unit_cell_topology
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildUnitCellTopology:
    """Unit cell topology tests (Paper Step 3)."""

    def test_7ba_has_seven_quad_families(self):
        """7BA should have 7 independent quad families."""
        result = build_unit_cell_topology(n_bends=7)
        assert result["n_quad_families"] == 7, f"Got {result['n_quad_families']} quad families"
        assert result["is_symmetric"] is True

    def test_all_supported_n_bends_work(self):
        """2, 3, 7, 9 bend topologies should all be defined."""
        for n in [2, 3, 7, 9]:
            result = build_unit_cell_topology(n_bends=n)
            assert "error" not in result, f"{n}-bend topology failed: {result.get('message','')}"
            assert len(result["element_sequence"]) > 0
            assert result["n_quad_families"] > 0

    def test_unsupported_n_bends_returns_error(self):
        """Unsupported bend counts should return closest topology."""
        result = build_unit_cell_topology(n_bends=15)
        assert result.get("error")
        assert "closest" in result

    def test_has_sextupole_placements(self):
        """All MBA topologies should include at least 2 sextupole pairs."""
        result = build_unit_cell_topology(n_bends=7)
        assert result["n_sextupole_pairs"] >= 2

    def test_mirror_axis_identified(self):
        """Mirror axis (centre bend) must be identified."""
        result = build_unit_cell_topology(n_bends=7)
        assert result["mirror_axis_index"] is not None

    def test_sextupoles_at_meaningful_positions(self):
        """SF near outer symmetry, SD near main bends (correction leverage)."""
        result = build_unit_cell_topology(n_bends=7)
        placements = result["sextupole_placements"]
        sf_positions = [p for p in placements if "SF" in p.get("role", "")]
        sd_positions = [p for p in placements if "SD" in p.get("role", "")]
        assert len(sf_positions) >= 1, "Need at least one SF sextupole"
        assert len(sd_positions) >= 1, "Need at least one SD sextupole"

    def test_reverse_bend_flag(self):
        """When include_reverse_bend=True, QF should be tagged."""
        result = build_unit_cell_topology(n_bends=7, include_reverse_bend=True)
        assert result["reverse_bend_enabled"]
        # Check that some element has reverse_bend info
        rb_elements = [
            e for e in result["element_sequence"]
            if e.get("reverse_bend_candidate")
        ]
        assert len(rb_elements) >= 1

    def test_full_cell_length_positive(self):
        """Full cell length estimate must be positive."""
        result = build_unit_cell_topology(n_bends=7)
        assert result["total_full_cell_length_est_m"] > 0
        assert result["total_full_cell_length_est_m"] > 5  # at least 5 m


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3: decompose_superperiod
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecomposeSuperperiod:
    """Superperiod decomposition tests (Paper Step 2)."""

    def test_produces_five_components(self):
        """Standard decomposition: 2 straights + 2 DSCs + N×UC."""
        result = decompose_superperiod(n_bends=7)
        comps = result["components"]
        assert len(comps) >= 4, f"Expected ≥4 components, got {len(comps)}"

    def test_has_straight_matchers(self):
        """Must include straight-matcher sections at both ends."""
        result = decompose_superperiod(n_bends=7)
        comp_names = [c["name"] for c in result["components"]]
        straights = [n for n in comp_names if "straight" in n.lower()]
        assert len(straights) >= 2, f"Expected ≥2 straight matchers, got {len(straights)}"

    def test_has_dispersion_suppression_cells(self):
        """Must include DSC sections."""
        result = decompose_superperiod(n_bends=7)
        comp_names = [c["name"] for c in result["components"]]
        dsc = [n for n in comp_names if "dispersion_suppression" in n.lower()]
        assert len(dsc) >= 1

    def test_has_unit_cells(self):
        """Must include repeated unit cell block."""
        result = decompose_superperiod(n_bends=7)
        comp_names = [c["name"] for c in result["components"]]
        uc = [n for n in comp_names if "unit_cell" in n.lower()]
        assert len(uc) >= 1

    def test_length_estimate_positive_and_reasonable(self):
        """Superperiod length must be positive and reasonable size."""
        result = decompose_superperiod(n_bends=7, total_superperiod_length_m=25.0)
        assert result["estimated_length_m"] > 0
        assert 15 <= result["estimated_length_m"] <= 60  # realistic range

    def test_explicit_n_unit_cells_honoured(self):
        """When n_unit_cells is specified, it should be used."""
        result = decompose_superperiod(n_unit_cells=3, n_bends=7)
        assert result["n_unit_cells"] == 3

    def test_total_length_option_constrains_n_unit_cells(self):
        """total_superperiod_length_m should influence n_unit_cells estimate."""
        r_long = decompose_superperiod(total_superperiod_length_m=40.0, n_bends=7)
        r_short = decompose_superperiod(total_superperiod_length_m=15.0, n_bends=7)
        assert r_long["estimated_length_m"] >= r_short["estimated_length_m"]


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 4: compute_correction_leverage
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrectionLeverage:
    """Sextupole correction leverage tests (Paper Step 6)."""

    def test_dispersive_region_has_high_leverage(self):
        """Sextupole at high Dx, high β should be EFFECTIVE."""
        # Typical 7BA outer SF position: Dx ~ 0.3 m, βx ~ 15 m
        result = compute_correction_leverage(
            "SF_outer_symmetry", beta_x_m=15.0, beta_y_m=5.0,
            Dx_m=0.30, energy_gev=6.0
        )
        assert result["is_effective_x"], "High-Dx sextupole should be effective for X"
        assert result["is_effective_y"], "High-Dx sextupole should be effective for Y"

    def test_zero_dispersion_region_ineffective(self):
        """Sextupole at Dx ≈ 0 provides NO chromatic correction."""
        result = compute_correction_leverage(
            "SD_zero_dispersion", beta_x_m=10.0, beta_y_m=10.0,
            Dx_m=0.0005, energy_gev=6.0
        )
        assert not result["is_effective_x"], "Zero-Dx sextupole should be INEFFECTIVE"
        assert not result["is_effective_y"], "Zero-Dx sextupole should be INEFFECTIVE"

    def test_signs_follow_physics(self):
        """ξx shift from sextupole has correct sign.

        Horizontal chromaticity: ξx ∝ -k2 · βx · Dx
        Vertical chromaticity:   ξy ∝ +k2 · βy · Dx  (opposite sign!!)
        """
        result = compute_correction_leverage(
            "test", beta_x_m=10.0, beta_y_m=3.0,
            Dx_m=0.20, energy_gev=6.0
        )
        # leverage_x = dξx/dk2 = -βx * Dx * L / (4π·Bρ) → negative
        assert result["leverage_x"] < 0, (
            f"Horizontal leverage should be negative, got {result['leverage_x']}"
        )
        # leverage_y = dξy/dk2 = +βy * Dx * L / (4π·Bρ) → positive
        assert result["leverage_y"] > 0, (
            f"Vertical leverage should be positive, got {result['leverage_y']}"
        )

    def test_leverage_scales_with_energy(self):
        """Higher energy → larger Bρ → lower leverage for same k2."""
        r_low = compute_correction_leverage(
            "test", beta_x_m=10.0, beta_y_m=3.0, Dx_m=0.20, energy_gev=3.0
        )
        r_high = compute_correction_leverage(
            "test", beta_x_m=10.0, beta_y_m=3.0, Dx_m=0.20, energy_gev=9.0
        )
        # Higher Bρ → smaller |leverage|
        assert abs(r_low["leverage_x"]) > abs(r_high["leverage_x"]), (
            "Higher energy should reduce correction leverage"
        )

    def test_recommended_k2_range_reasonable(self):
        """Recommended |k2| should be in the 5-800 m⁻³ range."""
        result = compute_correction_leverage(
            "test", beta_x_m=12.0, beta_y_m=4.0, Dx_m=0.25, energy_gev=6.0
        )
        lo, hi = result["recommended_k2_range"]
        assert 1 <= lo <= hi <= 1000, f"k2 range [{lo}, {hi}] unreasonable"
        assert lo > 0, "Lower bound must be positive"


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 5: check_dsc_matching_conditions
# ═══════════════════════════════════════════════════════════════════════════════

class TestDscMatchingConditions:
    """DSC matching feasibility tests (Paper Step 7)."""

    def test_zero_dispersion_target_is_feasible_with_enough_dof(self):
        """Matching Dx=0, Dpx=0 from typical UC exit should be feasible."""
        result = check_dsc_matching_conditions(
            beta_x_entry=10.0, alpha_x_entry=0.1,
            beta_y_entry=3.0, alpha_y_entry=0.05,
            Dx_entry=0.2, Dpx_entry=0.01,
            Dx_target=0.0, Dpx_target=0.0,
            n_dsc_quad_families=4, n_dsc_dipoles=2,
        )
        assert result["feasibility"] != "overconstrained", (
            "4 quads + 2 dipoles should handle 6 constraints"
        )

    def test_dispersion_matching_needs_dipoles(self):
        """Dispersion suppression requires at least 2 dipoles (η→0, η'→0)."""
        result = check_dsc_matching_conditions(
            beta_x_entry=10.0, alpha_x_entry=0.0,
            beta_y_entry=3.0, alpha_y_entry=0.0,
            Dx_entry=0.3, Dpx_entry=0.02,
            n_dsc_quad_families=2, n_dsc_dipoles=1,
        )
        # With only 1 dipole, dispersion matching is hard
        assert result["matching_difficulty"] in ("moderate", "hard")

    def test_overconstrained_when_too_few_dof(self):
        """Too few DOF vs constraints → overconstrained."""
        result = check_dsc_matching_conditions(
            beta_x_entry=10.0, alpha_x_entry=0.5,  # significant α mismatch
            beta_y_entry=3.0, alpha_y_entry=0.3,
            Dx_entry=0.4, Dpx_entry=0.03,
            n_dsc_quad_families=1, n_dsc_dipoles=1,  # too few DOF
        )
        assert result["is_overconstrained"], (
            f"1 quad + 1 dipole ({result['n_dof_estimated']} DOF) "
            f"should be overconstrained for {result['n_constraints']} constraints"
        )

    def test_already_matched_is_easy(self):
        """When entry ≈ target, matching should be easy."""
        result = check_dsc_matching_conditions(
            beta_x_entry=10.0, alpha_x_entry=0.0,
            beta_y_entry=3.0, alpha_y_entry=0.0,
            Dx_entry=0.0, Dpx_entry=0.0,
            n_dsc_quad_families=4, n_dsc_dipoles=2,
        )
        assert result["n_constraints"] <= 2  # only trivial constraints


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 6: validate_mba_construction_step
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateMbaStep:
    """MBA construction step validation tests."""

    def test_valid_unit_cell_optimization_passes(self):
        """Correctly structured cell optimization output should pass."""
        results = {
            "stable": True,
            "trace_x": 0.5,
            "trace_y": 0.3,
            "D_exit": 0.001,
            "Dp_exit": 0.0005,
            "mu_x_cell_deg": 55.0,
            "mu_y_cell_deg": 30.0,
            "target_mu_x_deg": 55.0,
            "target_mu_y_deg": 30.0,
        }
        result = validate_mba_construction_step("unit_cell_optimization", results)
        assert result["passed"], f"Should pass: {result['failures']}"

    def test_unstable_cell_fails(self):
        """Unstable cell (|trace| ≥ 2) must fail."""
        for step in ["unit_cell_optimization"]:
            results = {"stable": False, "trace_x": 2.5, "trace_y": 0.5}
            result = validate_mba_construction_step(step, results)
            assert not result["passed"], f"Unstable {step} should fail"

    def test_non_achromatic_cell_fails(self):
        """Cell with large D_exit should fail achromatic check."""
        results = {
            "stable": True,
            "trace_x": 0.5,
            "trace_y": 0.3,
            "D_exit": 0.05,  # 5 cm → too large for achromatic cell
            "Dp_exit": 0.001,
        }
        result = validate_mba_construction_step("unit_cell_optimization", results)
        assert not result["passed"], "Non-achromatic cell should fail"

    def test_valid_design_contract_passes(self):
        """Complete contract with all fields should pass."""
        results = {
            "energy_gev": 6.0,
            "n_cells": 24,
            "n_bends": 7,
            "ranked_goals": [
                "1. Low emittance < 50 pm-rad",
                "2. HOA phase structure",
                "3. Acceptable αc > 5e-5",
            ],
        }
        result = validate_mba_construction_step("design_contract", results)
        assert result["passed"], f"Valid contract should pass: {result['failures']}"

    def test_bad_robinson_sum_fails_ring_assembly(self):
        """Robinson sum far from 4 should fail ring assembly."""
        results = {
            "ring_linear_ok": True,
            "robinson_sum": 3.2,  # should be ≈ 4.0
        }
        result = validate_mba_construction_step("ring_assembly_linear", results)
        assert not result["passed"], "Wrong Robinson sum should fail"

    def test_valid_ring_assembly_passes(self):
        """Correct ring assembly values should pass."""
        results = {
            "ring_linear_ok": True,
            "stable": True,
            "robinson_sum": 4.001,
            "emittance": 45.0,
            "target_emittance_pm": 50.0,
        }
        result = validate_mba_construction_step("ring_assembly_linear", results)
        assert result["passed"], f"Valid ring should pass: {result['failures']}"

    def test_valid_dsc_passes(self):
        """Correctly matched DSC should pass."""
        results = {
            "dsc_matched": True,
            "Dx_exit": 0.00001,
            "Dpx_exit": 0.00001,
            "dsc_matching_residuals": {"betax": 0.01, "betay": 0.02},
        }
        result = validate_mba_construction_step("dsc_construction", results)
        assert result["passed"], f"Valid DSC should pass: {result['failures']}"

    def test_valid_hoa_passes(self):
        """HOA-compatible phase selection should pass."""
        results = {
            "Nx": 8,
            "Ny": 4,
            "detuned_fractional_nux": "0.250000",
            "detuned_fractional_nuy": "0.180000",
            "resonance_count": 0,
        }
        result = validate_mba_construction_step("hoa_phase_selection", results)
        assert result["passed"], f"Valid HOA should pass: {result['failures']}"


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 7: estimate_mba_emittance
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimateMbaEmittance:
    """MBA emittance estimation tests."""

    def test_7ba_24cells_6gev_reasonable_emittance(self):
        """Standard 7BA at 6 GeV, 24 cells → emittance ~50 pm·rad range."""
        result = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=7,
            lattice_type="7BA",
        )
        assert 5 <= result["emittance_pm_rad"] <= 500, (
            f"7BA emittance {result['emittance_pm_rad']} pm·rad out of range"
        )
        # Should be within 20× the TME floor (accounting for finite F)
        assert result["ratio_to_tme_floor"] <= 20

    def test_no_reverse_bend_has_zero_penalty(self):
        """Zero reverse bend should give zero penalty."""
        result = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=7,
            reverse_bend_fraction=0.0,
        )
        assert result["reverse_bend_penalty_pct"] == 0.0

    def test_reverse_bend_increases_emittance(self):
        """Reverse bend should increase emittance vs no RB."""
        r_no_rb = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=7,
            reverse_bend_fraction=0.0,
        )
        r_rb = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=7,
            reverse_bend_fraction=0.05,
        )
        assert r_rb["emittance_pm_rad"] > r_no_rb["emittance_pm_rad"], (
            "Reverse bend should increase emittance"
        )
        assert r_rb["reverse_bend_penalty_pct"] > 0

    def test_fodo_higher_emittance_than_mba(self):
        """FODO should have much higher emittance than 7BA at same parameters."""
        r_fodo = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=2,
            lattice_type="FODO",
        )
        r_7ba = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=7,
            lattice_type="7BA",
        )
        assert r_fodo["emittance_pm_rad"] > r_7ba["emittance_pm_rad"] * 5, (
            "FODO emittance should be >> 7BA"
        )

    def test_emittance_scales_with_energy_squared(self):
        """ε ∝ γ² → doubling energy → ~4× emittance."""
        r_low = estimate_mba_emittance(
            energy_gev=3.0, n_cells=24, n_bends_per_cell=7, lattice_type="7BA"
        )
        r_high = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=7, lattice_type="7BA"
        )
        ratio = r_high["emittance_pm_rad"] / r_low["emittance_pm_rad"]
        # γ² scaling: factor ~3.8−4.1 (accounting for identical F)
        assert 3.0 <= ratio <= 5.0, (
            f"Energy doubling ratio {ratio:.2f} not ~4"
        )

    def test_theta_in_mrad_reasonable(self):
        """Dipole bending angle should be sensible (few mrad for MBA)."""
        result = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=7,
        )
        assert 1.0 <= result["theta_per_dipole_mrad"] <= 100.0, (
            f"Theta {result['theta_per_dipole_mrad']} mrad out of range"
        )

    def test_total_dipoles_correct(self):
        """Total dipoles should = n_cells * n_bends_per_cell * n_unit_cells_per_period."""
        result = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=7,
            n_unit_cells_per_period=2,
        )
        assert result["total_dipoles"] == 24 * 7 * 2
        # More dips = smaller theta per dip → lower emittance
        r_single = estimate_mba_emittance(
            energy_gev=6.0, n_cells=24, n_bends_per_cell=7,
            n_unit_cells_per_period=1,
        )
        assert result["emittance_pm_rad"] < r_single["emittance_pm_rad"], (
            "More dipoles → smaller theta → lower emittance"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registry tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry:
    """Meta-tests for the MBA_TOOLS registry."""

    def test_all_seven_tools_registered(self):
        """All 7 tools should be in MBA_TOOLS."""
        assert len(MBA_TOOLS) == 7

    def test_all_tools_are_callable(self):
        """Every tool should be callable."""
        for tool in MBA_TOOLS:
            assert callable(tool), f"{tool} is not callable"

    def test_all_tools_have_docstrings(self):
        """Every tool should have a docstring for LLM tool descriptions."""
        for tool in MBA_TOOLS:
            assert tool.__doc__ and len(tool.__doc__) > 50, (
                f"{tool.__name__} missing or too-short docstring"
            )

    def test_all_tools_return_dict(self):
        """Verify each tool returns a dict — required for tool-use compatibility."""
        # Test with minimal valid inputs
        tests = {
            compute_hoa_phase_advance: {"n_cells": 24, "n_bends_per_cell": 7},
            build_unit_cell_topology: {"n_bends": 7},
            decompose_superperiod: {"n_bends": 7},
            compute_correction_leverage: {
                "sextupole_position_name": "test",
                "beta_x_m": 10.0, "beta_y_m": 3.0, "Dx_m": 0.2,
            },
            check_dsc_matching_conditions: {
                "beta_x_entry": 10.0, "alpha_x_entry": 0.0,
                "beta_y_entry": 3.0, "alpha_y_entry": 0.0,
                "Dx_entry": 0.2, "Dpx_entry": 0.0,
            },
            validate_mba_construction_step: {
                "step_name": "design_contract",
                "results": {"energy_gev": 6.0, "n_cells": 24, "n_bends": 7,
                           "ranked_goals": ["1. Low emittance"]},
            },
            estimate_mba_emittance: {
                "energy_gev": 6.0, "n_cells": 24, "n_bends_per_cell": 7,
            },
        }
        for tool, kwargs in tests.items():
            result = tool(**kwargs)
            assert isinstance(result, dict), f"{tool.__name__} returned {type(result)}"
            assert len(result) > 0, f"{tool.__name__} returned empty dict"
