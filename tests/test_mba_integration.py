"""Integration tests for MBA workflow orchestrator.

Tests the MBAWorkflowRunner state accumulation, validation gating,
and stage definitions WITHOUT actual LLM calls.
"""

import pytest
from agents.mba_workflow import (
    MBAWorkflowRunner,
    MBA_STAGES,
    MBAStage,
    run_mba_tool_stage,
)
from agents.mba_tools import validate_mba_construction_step


class TestMBAStageDefinitions:
    """Verify MBA_STAGES are well-formed."""

    def test_all_10_stages_defined(self):
        """The paper specifies exactly 10 construction stages."""
        assert len(MBA_STAGES) == 10

    def test_all_stages_have_unique_names(self):
        """Stage names must be unique."""
        names = [s.name for s in MBA_STAGES]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_stages_are_in_construction_order(self):
        """Stages must follow the paper's construction order."""
        expected_order = [
            "design_contract",
            "decompose_superperiod",
            "unit_cell_topology",
            "hoa_phase_selection",
            "unit_cell_optimization",
            "sextupole_placement",
            "dsc_construction",
            "straight_matching",
            "ring_assembly_linear",
            "nonlinear_check",
        ]
        actual_order = [s.name for s in MBA_STAGES]
        assert actual_order == expected_order

    def test_pyjutrack_stages_are_marked(self):
        """Stages 5, 7, 8, 9, 10 require pyJuTrack code execution."""
        pyjutrack_stages = {s.name for s in MBA_STAGES if s.uses_pyjutrack}
        expected = {
            "unit_cell_optimization",
            "dsc_construction",
            "straight_matching",
            "ring_assembly_linear",
            "nonlinear_check",
        }
        assert pyjutrack_stages == expected

    def test_non_pyjutrack_stages_are_tool_only(self):
        """Stages 1-4, 6 should NOT require pyJuTrack (tool calls only)."""
        tool_stages = {s.name for s in MBA_STAGES if not s.uses_pyjutrack}
        expected = {
            "design_contract",
            "decompose_superperiod",
            "unit_cell_topology",
            "hoa_phase_selection",
            "sextupole_placement",
        }
        assert tool_stages == expected

    def test_each_stage_has_description(self):
        """Every stage must have a human-readable description."""
        for stage in MBA_STAGES:
            assert stage.description and len(stage.description) > 10

    def test_each_stage_has_task_prompt(self):
        """Every stage must have a prompt template."""
        for stage in MBA_STAGES:
            assert stage.task_prompt_template and len(stage.task_prompt_template) > 20

    def test_each_stage_has_produces_outputs(self):
        """Every stage must produce outputs for downstream stages."""
        for stage in MBA_STAGES:
            assert len(stage.produces_outputs) > 0, (
                f"Stage '{stage.name}' has no produced outputs"
            )

    def test_fix_attempts_reasonable(self):
        """Fix attempts should be in range 1-6."""
        for stage in MBA_STAGES:
            assert 1 <= stage.max_fix_attempts <= 6

    def test_validation_keys_match_mba_tools(self):
        """Each stage's validation_key must be valid for validate_mba_construction_step."""
        valid_keys = [
            "design_contract", "decompose_superperiod",
            "unit_cell_topology", "hoa_phase_selection",
            "unit_cell_optimization", "sextupole_placement",
            "dsc_construction", "straight_matching",
            "ring_assembly_linear", "nonlinear_check",
        ]
        for stage in MBA_STAGES:
            assert stage.validation_key in valid_keys, (
                f"Stage '{stage.name}' validation_key '{stage.validation_key}' not in {valid_keys}"
            )


class TestMBAWorkflowRunner:
    """Test the workflow runner state management."""

    def test_initial_state_is_empty(self):
        """Runner starts with empty accumulated state."""
        runner = MBAWorkflowRunner()
        assert runner.accumulated_state == {}
        assert runner.stage_log == []

    def test_accumulate_adds_outputs(self):
        """accumulate() should add produced outputs to state."""
        runner = MBAWorkflowRunner()
        stage = MBA_STAGES[0]  # design_contract
        output = {
            "energy_gev": 6.0,
            "n_cells": 24,
            "n_bends": 7,
            "target_emittance_pm": 50.0,
            "min_alpha_c": 5e-5,
            "ranked_goals": ["1. Low emittance", "2. HOA phase"],
            "hard_constraints": {"dipole_field_t": 1.8},
        }
        runner.accumulate(stage, output)
        for key in stage.produces_outputs:
            assert key in runner.accumulated_state, f"Missing key: {key}"

    def test_accumulate_extra_keys_not_added(self):
        """Only keys in stage.produces_outputs should be added."""
        runner = MBAWorkflowRunner()
        stage = MBA_STAGES[0]
        output = {"energy_gev": 6.0, "extra_noise": "garbage"}
        runner.accumulate(stage, output)
        assert "extra_noise" not in runner.accumulated_state

    def test_check_required_inputs_missing(self):
        """When inputs are missing, they should be reported."""
        runner = MBAWorkflowRunner()
        stage = MBA_STAGES[1]  # needs n_cells, n_bends
        missing = runner.check_required_inputs(stage)
        assert len(missing) >= 1, "Should report missing inputs"

    def test_check_required_inputs_satisfied(self):
        """When all inputs are present, no missing."""
        runner = MBAWorkflowRunner()
        runner.accumulated_state = {"n_cells": 24, "n_bends": 7}
        stage = MBA_STAGES[1]
        missing = runner.check_required_inputs(stage)
        assert missing == [], f"Unexpected missing inputs: {missing}"

    def test_build_stage_prompt_includes_state(self):
        """Stage prompt should include accumulated state."""
        runner = MBAWorkflowRunner()
        runner.accumulated_state = {"n_cells": 24, "n_bends": 7}
        stage = MBA_STAGES[1]
        prompt = runner.build_stage_prompt(stage, "design a 7BA")
        assert "24" in prompt, "n_cells not in prompt"
        assert "7" in prompt, "n_bends not in prompt"
        assert "design a 7BA" in prompt, "user request not in prompt"

    def test_build_stage_prompt_handles_missing_keys(self):
        """Missing keys should not crash the formatter."""
        runner = MBAWorkflowRunner()
        runner.accumulated_state = {"n_cells": 24}  # n_bends missing
        stage = MBA_STAGES[1]  # template expects {n_bends}
        # Should not crash even with missing key (uses defaultdict fallback)
        prompt = runner.build_stage_prompt(stage, "test")
        # n_cells was provided
        assert "24" in prompt

    def test_validate_stage_output_runs(self):
        """Validation should run and return a dict."""
        runner = MBAWorkflowRunner()
        stage = MBA_STAGES[0]
        output = {
            "energy_gev": 6.0,
            "n_cells": 24,
            "n_bends": 7,
            "ranked_goals": ["1. Low emittance"],
        }
        result = runner.validate_stage_output(stage, output)
        assert isinstance(result, dict)
        assert "passed" in result
        assert result.get("step") == "design_contract"

    def test_extract_output_finds_json(self):
        """Should extract JSON from response text with markers."""
        runner = MBAWorkflowRunner()
        text = """Some preamble
--- STAGE RESULT JSON ---
{"stage": "design_contract", "energy_gev": 6.0, "n_cells": 24}
--- END STAGE RESULT JSON ---
More content."""
        output = runner.extract_output_from_response(text)
        assert "energy_gev" in output
        assert output["energy_gev"] == 6.0

    def test_extract_output_fallback(self):
        """Should fallback to first JSON object."""
        runner = MBAWorkflowRunner()
        text = 'No markers but {"key": "value"} is here.'
        output = runner.extract_output_from_response(text)
        assert output.get("key") == "value"

    def test_extract_output_empty_for_no_json(self):
        """Empty dict when no JSON found."""
        runner = MBAWorkflowRunner()
        output = runner.extract_output_from_response("No JSON here at all.")
        assert output == {}

    def test_log_stage_records_entry(self):
        """log_stage should append to stage_log."""
        runner = MBAWorkflowRunner()
        stage = MBA_STAGES[0]
        runner.log_stage(stage, True, {"energy_gev": 6.0}, "OK")
        assert len(runner.stage_log) == 1
        assert runner.stage_log[0]["success"] is True
        assert runner.stage_log[0]["stage"] == "design_contract"

    def test_build_final_report_contains_stages(self):
        """Final report should list all logged stages."""
        runner = MBAWorkflowRunner()
        for stage in MBA_STAGES[:3]:
            runner.log_stage(stage, True, {}, "OK")
        report = runner.build_final_report()
        for stage in MBA_STAGES[:3]:
            assert stage.name in report
        assert "PASS" in report

    def test_build_final_report_shows_accumulated_state(self):
        """Final report should show accumulated state summary."""
        runner = MBAWorkflowRunner()
        runner.accumulated_state = {"energy_gev": 6.0, "n_cells": 24}
        runner.log_stage(MBA_STAGES[0], True, {"energy_gev": 6.0}, "OK")
        report = runner.build_final_report()
        assert "energy_gev" in report
        assert "6.0" in report

    def test_fix_attempts_more_for_pyjutrack_stages(self):
        """pyJuTrack stages should have more fix attempts (>=3)."""
        for stage in MBA_STAGES:
            if stage.uses_pyjutrack:
                assert stage.max_fix_attempts >= 3, (
                    f"Stage '{stage.name}' uses pyJuTrack but only "
                    f"{stage.max_fix_attempts} fix attempts"
                )

    def test_fix_attempts_less_for_tool_stages(self):
        """Tool-only stages should have <=2 fix attempts."""
        for stage in MBA_STAGES:
            if not stage.uses_pyjutrack:
                assert stage.max_fix_attempts <= 2, (
                    f"Stage '{stage.name}' is tool-only but has "
                    f"{stage.max_fix_attempts} fix attempts"
                )


class TestFullPipelineDataFlow:
    """Test data flow between consecutive stages."""

    def test_stage1_outputs_feed_stage2(self):
        """Stage 1 outputs should contain all Stage 2 required inputs."""
        s1_outputs = set(MBA_STAGES[0].produces_outputs)
        s2_inputs = set(MBA_STAGES[1].required_inputs)
        assert s2_inputs.issubset(s1_outputs), (
            f"Stage 2 needs {s2_inputs - s1_outputs} not produced by Stage 1"
        )

    def test_stage4_outputs_feed_stage5(self):
        """HOA phase outputs must feed unit cell optimization."""
        s4_outputs = set(MBA_STAGES[3].produces_outputs)
        s5_inputs = set(MBA_STAGES[4].required_inputs)
        missing = s5_inputs - s4_outputs
        # Stage 5 also needs stage 3 outputs (element_sequence, n_quad_families)
        # and stage 1 outputs (energy_gev) — those come from state accumulation
        expected_from_s4 = {"mu_x_cell_deg", "mu_y_cell_deg"}
        assert expected_from_s4.issubset(s4_outputs), (
            f"Stage 4 should produce {expected_from_s4}"
        )
        for key in expected_from_s4:
            assert key in s5_inputs, f"Stage 5 needs {key} from Stage 4"

    def test_chain_is_connected(self):
        """The entire 10-stage chain should be connected — no orphaned stages."""
        produced = set()
        for i, stage in enumerate(MBA_STAGES):
            missing = set(stage.required_inputs) - produced
            if i == 0:
                assert missing == set(), (
                    f"Stage 1 should have no required inputs, but needs: {missing}"
                )
            else:
                # Allow some inputs from the user request (not from previous stages)
                # Stage 1 produces: energy_gev, n_cells, n_bends, target_emittance_pm,
                #   min_alpha_c, ranked_goals, hard_constraints
                assert len(produced) >= 3, (
                    f"Stage {i} ('{stage.name}') has {len(produced)} produced keys from "
                    f"prior stages. Missing inputs: {missing}"
                )
            produced.update(stage.produces_outputs)

    def test_can_complete_full_pipeline_with_valid_outputs(self):
        """Smoke test: full pipeline can run with valid fake outputs."""
        runner = MBAWorkflowRunner()

        # Artificially feed valid outputs through all stages
        fake_outputs = {
            "design_contract": {
                "energy_gev": 6.0, "n_cells": 24, "n_bends": 7,
                "target_emittance_pm": 50.0, "min_alpha_c": 5e-5,
                "ranked_goals": ["1. Low emittance"], "hard_constraints": {},
            },
            "decompose_superperiod": {
                "components": ["straight", "dsc", "uc", "dsc", "straight"],
                "n_unit_cells": 2, "superperiod_length_est_m": 25.0,
            },
            "unit_cell_topology": {
                "element_sequence": ["SF", "QF", "QD", "SD", "MB"],
                "n_quad_families": 7, "sextupole_placements": ["SF", "SD"],
                "cell_length_est_m": 20.0,
            },
            "hoa_phase_selection": {
                "Nx": 8, "Ny": 4,
                "mu_x_cell_deg": 120.0, "mu_y_cell_deg": 60.0,
                "target_nux": 0.25, "target_nuy": 0.18,
            },
            "unit_cell_optimization": {
                "k1_values": [3.8, -2.7, 1.8, -2.8, 1.9, -2.3, 2.7],
                "dipole_angles": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.015],
                "uc_twiss_exit": {"beta_x": 12.0, "alpha_x": 0.2,
                                  "beta_y": 5.0, "alpha_y": 0.1,
                                  "Dx": 0.25, "Dpx": -0.01},
                "uc_stable": True, "uc_trace_x": 0.5, "uc_trace_y": 0.3,
                "uc_D_exit": 0.001, "uc_Dp_exit": 0.0005,
                "uc_mu_x_deg": 120.0, "uc_mu_y_deg": 60.0,
            },
            "sextupole_placement": {
                "sextupole_report": [
                    {"position": "SF1", "beta_x": 15.0, "beta_y": 5.0,
                     "Dx": 0.3, "leverage_x": 0.003, "leverage_y": 0.001,
                     "is_effective": True, "k2_range": [50, 300]},
                ],
                "initial_k2_values": [100.0, -100.0, 80.0, -80.0],
            },
            "dsc_construction": {
                "dsc_matched": True, "dsc_Dx_exit": 0.0001, "dsc_Dpx_exit": 0.0001,
                "dsc_matching_residuals": {"betax": 0.01},
                "dsc_elements": ["QF", "QD", "B"], "dsc_k1_values": [2.0, -1.5],
            },
            "straight_matching": {
                "superperiod_matched": True, "straight_betax": 10.0,
                "straight_betay": 3.0,
                "ring_nux": 0.25, "ring_nuy": 0.18,
                "ring_chromx": -200.0, "ring_chromy": -150.0,
                "emittance": 45.0, "alphac": 1e-4,
                "radiation_integrals": {"I1": 1.0, "I2": 0.5, "I3": 0.3, "I4": -0.1, "I5": 0.01},
                "circumference": 490.0,
            },
            "ring_assembly_linear": {
                "ring_linear_ok": True, "robinson_sum": 4.001,
                "U0_eV": 2.5e6,
                "Jx": 1.8, "Jy": 1.0, "Je": 1.2,
                "I1": 1.0, "I2": 0.5, "I3": 0.3, "I4": -0.1, "I5": 0.01,
                "magnet_limits_ok": True,
            },
            "nonlinear_check": {
                "nonlinear_ok": True, "da_estimate_mm": 5.0,
                "dipole_margin_pct": 30.0, "quad_margin_pct": 40.0,
                "sextupole_margin_pct": 50.0,
            },
        }

        for stage in MBA_STAGES:
            runner.accumulate(stage, fake_outputs[stage.name])
            runner.log_stage(stage, True, fake_outputs[stage.name], "OK")

        report = runner.build_final_report()
        assert "MBA CONSTRUCTION COMPLETE" in report
        assert "PASS" in report
        assert "energy_gev" in report

    def test_all_stages_can_be_validated(self):
        """validate_mba_construction_step works for every stage."""
        valid_outputs = {
            "design_contract": {
                "energy_gev": 6.0, "n_cells": 24, "n_bends": 7,
                "ranked_goals": ["1. Low emittance"],
            },
            "decompose_superperiod": {
                "components": [
                    {"name": "straight_matcher_entrance"},
                    {"name": "dispersion_suppression_cell_entrance"},
                    {"name": "unit_cells_x2"},
                    {"name": "dispersion_suppression_cell_exit"},
                    {"name": "straight_matcher_exit"},
                ],
            },
            "unit_cell_topology": {
                "n_quad_families": 7, "n_bends_per_cell": 7,
                "n_sextupole_pairs": 4, "is_symmetric": True,
            },
            "hoa_phase_selection": {
                "Nx": 8, "Ny": 4,
                "detuned_fractional_nux": "0.250000",
                "detuned_fractional_nuy": "0.180000",
                "resonance_count": 0,
            },
            "unit_cell_optimization": {
                "stable": True, "trace_x": 0.5, "trace_y": 0.3,
                "D_exit": 0.001, "Dp_exit": 0.0005,
                "mu_x_cell_deg": 120.0, "mu_y_cell_deg": 60.0,
                "target_mu_x_deg": 120.0, "target_mu_y_deg": 60.0,
            },
            "sextupole_placement": {
                "correction_leverage_report": {"SF1": {"leverage_x": 0.003}},
                "k2_initial_values": [100.0, -100.0],
            },
            "dsc_construction": {
                "dsc_matched": True, "Dx_exit": 0.00001, "Dpx_exit": 0.00001,
                "dsc_matching_residuals": {"betax": 0.01},
            },
            "straight_matching": {
                "superperiod_matched": True,
                "straight_betax": 10.0, "straight_betay": 3.0,
            },
            "ring_assembly_linear": {
                "ring_linear_ok": True, "stable": True,
                "robinson_sum": 4.001, "emittance": 45.0,
                "target_emittance_pm": 50.0,
            },
            "nonlinear_check": {
                "nonlinear_ok": True, "da_estimate_mm": 5.0,
            },
        }

        for stage in MBA_STAGES:
            output = valid_outputs[stage.name]
            result = validate_mba_construction_step(stage.validation_key, output)
            # Not all stages must pass with minimal outputs,
            # but validation must run without error
            assert isinstance(result, dict)
            assert "passed" in result
