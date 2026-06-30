"""MBA multi-stage workflow orchestrator for BRILLIANCE.

Implements the deterministic 10-step MBA construction methodology from the
"Deterministic Construction of a Distributed-Sextupole MBA Lattice" paper.

Each stage uses the existing 4-agent pipeline (TaskPlanner -> CodeWriter ->
CodeRunner -> CodeReviewer) scoped to one construction step. Accumulated
state from previous stages is injected into each stage's prompt.

Architecture:
  User request -> MBAWorkflowRunner
    -> Stage 1: design_contract
    -> Stage 2: decompose_superperiod
    -> Stage 3: unit_cell_topology
    -> Stage 4: hoa_phase_selection
    -> Stage 5: unit_cell_optimization
    -> Stage 6: sextupole_placement
    -> Stage 7: dsc_construction
    -> Stage 8: straight_matching
    -> Stage 9: ring_assembly_linear
    -> Stage 10: nonlinear_check
    -> Final report
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .mba_tools import validate_mba_construction_step


# ═══════════════════════════════════════════════════════════════════════════════
# Stage definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MBAStage:
    """One stage of the MBA construction methodology."""
    name: str                        # e.g. "design_contract"
    description: str                 # human-readable
    task_prompt_template: str        # user-facing prompt for this stage
    required_inputs: list[str]       # keys needed from accumulated_state
    produces_outputs: list[str]      # keys added to accumulated_state
    validation_key: str              # key for validate_mba_construction_step
    max_fix_attempts: int = 2        # per-stage fix-attempt budget
    uses_pyjutrack: bool = False     # whether this stage runs pyJuTrack code


# The 10 MBA construction stages as defined in the paper
MBA_STAGES: list[MBAStage] = [
    MBAStage(
        name="design_contract",
        description="Define ranked goals and hard constraints for the MBA ring",
        task_prompt_template=(
            "MBA STAGE 1: DESIGN CONTRACT\n\n"
            "Define the MBA design contract. Use the tools:\n"
            "- estimate_mba_emittance to get the emittance budget\n"
            "- compare_lattice_families to find suitable topologies\n"
            "- check_design_feasibility to verify the request\n\n"
            "OUTPUT: energy_gev, n_cells, n_bends, target_emittance_pm,\n"
            "min_alpha_c, ranked_goals (ordered priority list),\n"
            "hard_constraints (dict of limits).\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=[],
        produces_outputs=["energy_gev", "n_cells", "n_bends",
                          "target_emittance_pm", "min_alpha_c",
                          "ranked_goals", "hard_constraints"],
        validation_key="design_contract",
        max_fix_attempts=2,
        uses_pyjutrack=False,
    ),
    MBAStage(
        name="decompose_superperiod",
        description="Decompose one superperiod into StraightMatcher + DSC + NxUnitCell + mirror",
        task_prompt_template=(
            "MBA STAGE 2: DECOMPOSE SUPERPERIOD\n\n"
            "Use the decompose_superperiod tool with:\n"
            "- n_bends={n_bends} (from stage 1)\n"
            "- total_superperiod_length_m from circumference/{n_cells}\n\n"
            "OUTPUT: components (ordered list), n_unit_cells,\n"
            "estimated superperiod length.\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=["n_cells", "n_bends"],
        produces_outputs=["components", "n_unit_cells",
                          "superperiod_length_est_m"],
        validation_key="decompose_superperiod",
        max_fix_attempts=2,
    ),
    MBAStage(
        name="unit_cell_topology",
        description="Build the generic symmetric unit cell topology",
        task_prompt_template=(
            "MBA STAGE 3: UNIT CELL TOPOLOGY\n\n"
            "Use the build_unit_cell_topology tool with n_bends={n_bends}\n"
            "and optionally include_reverse_bend=True.\n\n"
            "OUTPUT: element_sequence, n_quad_families,\n"
            "sextupole_placements (with leverage rationale),\n"
            "mirror_axis_index, total_cell_length_est_m.\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=["n_bends", "n_unit_cells"],
        produces_outputs=["element_sequence", "n_quad_families",
                          "sextupole_placements", "cell_length_est_m"],
        validation_key="unit_cell_topology",
        max_fix_attempts=2,
    ),
    MBAStage(
        name="hoa_phase_selection",
        description="Select phase advance via Higher-Order-Achromat conditions",
        task_prompt_template=(
            "MBA STAGE 4: HOA PHASE SELECTION\n\n"
            "Use compute_hoa_phase_advance with:\n"
            "- n_cells={n_cells}, n_bends_per_cell={n_bends}\n\n"
            "Screen per-dipole phase advances (8-95 deg X, 6-65 deg Y).\n"
            "Select the resonance-free working point.\n\n"
            "OUTPUT: Nx, Ny, mu_x_cell_deg, mu_y_cell_deg,\n"
            "mu_x_dipole_deg, mu_y_dipole_deg, detuned nux/nuy,\n"
            "resonance_warnings, selection rationale.\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=["n_cells", "n_bends"],
        produces_outputs=["Nx", "Ny", "mu_x_cell_deg", "mu_y_cell_deg",
                          "target_nux", "target_nuy"],
        validation_key="hoa_phase_selection",
        max_fix_attempts=2,
    ),
    MBAStage(
        name="unit_cell_optimization",
        description="Optimize the unit cell with interpretable physics scans",
        task_prompt_template=(
            "MBA STAGE 5: UNIT CELL OPTIMIZATION\n\n"
            "The unit cell topology and HOA target phase advance are:\n"
            "{accumulated_state}\n\n"
            "Write a pyJuTrack script that:\n"
            "1. Builds the unit cell from the topology in stage 3\n"
            "2. Scans K1 values to achieve target phase advance\n"
            "   (mu_x={mu_x_cell_deg} deg, mu_y={mu_y_cell_deg} deg)\n"
            "3. Scans reverse-bend angle (0, 2%, 5% of total bend)\n"
            "4. Reports: trace_x, trace_y, D_exit, Dp_exit, mu_x, mu_y,\n"
            "   Twiss at exit for each configuration\n"
            "5. Keeps the cell achromatic (D_exit ~ 0)\n\n"
            "OUTPUT JSON: stable, trace_x, trace_y, D_exit, Dp_exit,\n"
            "mu_x_deg, mu_y_deg, beta_x_exit, beta_y_exit, alpha_x_exit,\n"
            "alpha_y_exit, Dx_exit, Dpx_exit, k1_values, dipole_angles.\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=["element_sequence", "mu_x_cell_deg", "mu_y_cell_deg",
                         "n_quad_families", "energy_gev"],
        produces_outputs=["k1_values", "dipole_angles", "uc_twiss_exit",
                          "uc_stable", "uc_trace_x", "uc_trace_y",
                          "uc_D_exit", "uc_Dp_exit",
                          "uc_mu_x_deg", "uc_mu_y_deg"],
        validation_key="unit_cell_optimization",
        max_fix_attempts=4,
        uses_pyjutrack=True,
    ),
    MBAStage(
        name="sextupole_placement",
        description="Place sextupoles using correction leverage analysis",
        task_prompt_template=(
            "MBA STAGE 6: SEXTUPOLE PLACEMENT\n\n"
            "The optimized unit cell exit Twiss is:\n"
            "{accumulated_state}\n\n"
            "For each sextupole position in the unit cell topology,\n"
            "use compute_correction_leverage with the local Twiss values\n"
            "(beta_x, beta_y, Dx) at that s-position.\n\n"
            "NEVER place sextupoles at Dx ~ 0 — they provide NO correction.\n"
            "Place SF at high-beta_x, high-Dx locations.\n"
            "Place SD at high-beta_y, high-Dx locations.\n\n"
            "OUTPUT JSON: sextupole_report [{position, beta_x, beta_y, Dx,\n"
            "leverage_x, leverage_y, is_effective, k2_range}] for each position.\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=["uc_twiss_exit", "sextupole_placements", "energy_gev"],
        produces_outputs=["sextupole_report", "initial_k2_values"],
        validation_key="sextupole_placement",
        max_fix_attempts=2,
    ),
    MBAStage(
        name="dsc_construction",
        description="Construct Dispersion Suppression Cell matching UC exit to straight",
        task_prompt_template=(
            "MBA STAGE 7: DSC CONSTRUCTION\n\n"
            "The unit cell exit Twiss is:\n"
            "{accumulated_state}\n\n"
            "Use check_dsc_matching_conditions to verify feasibility.\n\n"
            "Write a pyJuTrack script to build the DSC as a constrained\n"
            "transfer line. The DSC must match UC exit Twiss to the\n"
            "straight section entrance (Dx=0, Dpx=0, alpha_y ~ 0).\n\n"
            "Keep the DSC as similar as possible to HALF the unit cell.\n"
            "Tuning knobs: bend angle/length, QF/QD gradients, drift lengths.\n\n"
            "OUTPUT JSON: dsc_matched, Dx_exit, Dpx_exit, matching_residuals,\n"
            "dsc_elements, dsc_k1_values.\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=["uc_twiss_exit", "cell_length_est_m", "energy_gev"],
        produces_outputs=["dsc_matched", "dsc_Dx_exit", "dsc_Dpx_exit",
                          "dsc_matching_residuals", "dsc_elements",
                          "dsc_k1_values"],
        validation_key="dsc_construction",
        max_fix_attempts=3,
        uses_pyjutrack=True,
    ),
    MBAStage(
        name="straight_matching",
        description="Match straight section and assemble superperiod",
        task_prompt_template=(
            "MBA STAGE 8: STRAIGHT MATCHING\n\n"
            "DSC outputs and previous stages:\n"
            "{accumulated_state}\n\n"
            "Write a pyJuTrack script to:\n"
            "1. Build the straight matcher (3-4 quadrupoles)\n"
            "2. Assemble: straight + DSC + N*UC + mirror DSC + straight\n"
            "3. Match beta_x, beta_y at straight centre with alpha=0\n"
            "4. Assemble ring from n_cells superperiods\n"
            "5. Verify ring closure (total bend = 2*pi)\n"
            "6. Detune from exact integer HOA (shift ~0.1-0.2)\n"
            "7. Compute ring tunes, chromaticities, radiation integrals\n\n"
            "OUTPUT JSON: superperiod_matched, straight_betax, straight_betay,\n"
            "ring_nux, ring_nuy, ring_chromx, ring_chromy, emittance, alphac,\n"
            "radiation_integrals, circumference.\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=["dsc_elements", "dsc_k1_values", "element_sequence",
                         "k1_values", "n_cells", "Nx", "Ny", "energy_gev"],
        produces_outputs=["superperiod_matched", "straight_betax", "straight_betay",
                          "ring_nux", "ring_nuy", "ring_chromx", "ring_chromy",
                          "emittance", "alphac", "radiation_integrals",
                          "circumference"],
        validation_key="straight_matching",
        max_fix_attempts=3,
        uses_pyjutrack=True,
    ),
    MBAStage(
        name="ring_assembly_linear",
        description="Assemble full ring and verify linear optics",
        task_prompt_template=(
            "MBA STAGE 9: RING ASSEMBLY (LINEAR)\n\n"
            "Previous stage outputs:\n"
            "{accumulated_state}\n\n"
            "Write a pyJuTrack script to:\n"
            "1. Assemble the complete ring from the previous stage\n"
            "2. Compute tunes, chromaticities, radiation integrals\n"
            "3. Compute Robinson sum Jx+Jy+Je (must be 4.0 +/- 0.05)\n"
            "4. Compute U0 analytically from I2 and energy\n"
            "5. Check magnet limits (dipole field, quad gradient, sextupole strength)\n"
            "6. Report all engineering spaces\n\n"
            "OUTPUT JSON: ring_linear_ok, nux, nuy, chromx, chromy, emittance,\n"
            "alphac, Jx, Jy, Je, robinson_sum, U0_eV, sigma_E, I1-I5,\n"
            "circumference, magnet_limits_ok.\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=["superperiod_matched", "n_cells", "energy_gev",
                         "target_emittance_pm"],
        produces_outputs=["ring_linear_ok", "robinson_sum", "U0_eV",
                          "Jx", "Jy", "Je", "I1", "I2", "I3", "I4", "I5",
                          "magnet_limits_ok"],
        validation_key="ring_assembly_linear",
        max_fix_attempts=3,
        uses_pyjutrack=True,
    ),
    MBAStage(
        name="nonlinear_check",
        description="Engineering constraints and nonlinear readiness check",
        task_prompt_template=(
            "MBA STAGE 10: NONLINEAR READINESS CHECK\n\n"
            "Previous stage outputs:\n"
            "{accumulated_state}\n\n"
            "DO NOT begin a full nonlinear optimization. This is a READINESS\n"
            "CHECK to verify the linear lattice is a valid starting point.\n\n"
            "Write a pyJuTrack script to:\n"
            "1. Compute quick on-momentum DA estimate (~6 angles, 512 turns)\n"
            "2. Check dipole field margin, quad gradient margin,\n"
            "   sextupole gradient margin\n"
            "3. Verify absorber/BPM/corrector space exists\n\n"
            "OUTPUT JSON: nonlinear_ok, da_estimate_mm, dipole_field_margin_pct,\n"
            "quad_gradient_margin_pct, sextupole_margin_pct.\n\n"
            "Original request: {user_request}"
        ),
        required_inputs=["ring_linear_ok", "energy_gev", "initial_k2_values"],
        produces_outputs=["nonlinear_ok", "da_estimate_mm",
                          "dipole_margin_pct", "quad_margin_pct",
                          "sextupole_margin_pct"],
        validation_key="nonlinear_check",
        max_fix_attempts=3,
        uses_pyjutrack=True,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Workflow runner
# ═══════════════════════════════════════════════════════════════════════════════

class MBAWorkflowRunner:
    """Runs the multi-stage MBA construction methodology.

    Each stage uses the existing 4-agent pipeline (run_design from team.py),
    scoped to one construction step. Accumulated state from previous stages is
    injected into the prompt via str.format().

    Parameters
    ----------
    stages : list[MBAStage] | None
        Custom stage list. Defaults to MBA_STAGES (10 stages from the paper).
    pause_between_stages : bool
        If True (not yet implemented), prompt user before each stage.
    """

    def __init__(
        self,
        stages: list[MBAStage] | None = None,
        pause_between_stages: bool = False,
    ):
        self.stages = stages or MBA_STAGES
        self.pause_between_stages = pause_between_stages
        self.accumulated_state: dict[str, Any] = {}
        self.stage_log: list[dict[str, Any]] = []

    def build_stage_prompt(self, stage: MBAStage, user_request: str) -> str:
        """Build the prompt for a stage by merging accumulated state.

        Missing format keys are left as-is (e.g. '{n_bends}') — the LLM
        will handle them as placeholders it should fill from context.
        """
        fmt = dict(self.accumulated_state)
        fmt["accumulated_state"] = json.dumps(self.accumulated_state, indent=2)
        fmt["user_request"] = user_request
        fmt["stage_name"] = stage.name
        fmt["stage_description"] = stage.description
        # Use str.format_map with a defaultdict so missing keys don't crash
        from collections import defaultdict
        return stage.task_prompt_template.format_map(
            defaultdict(str, **fmt)
        )

    def check_required_inputs(self, stage: MBAStage) -> list[str]:
        """Return list of missing required inputs."""
        missing = []
        for key in stage.required_inputs:
            if key not in self.accumulated_state:
                missing.append(key)
        return missing

    def validate_stage_output(
        self, stage: MBAStage, output: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate stage output using validate_mba_construction_step."""
        return validate_mba_construction_step(stage.validation_key, output)

    def extract_output_from_response(self, response_text: str) -> dict[str, Any]:
        """Extract structured JSON output from agent response.

        Looks for JSON block between '--- STAGE RESULT JSON ---' and
        '--- END STAGE RESULT JSON ---', or '--- LATTICE RESULT JSON ---'
        and '--- END LATTICE RESULT JSON ---'.
        Falls back to parsing the first JSON object in the text.
        """
        for start, end in [
            ("--- STAGE RESULT JSON ---", "--- END STAGE RESULT JSON ---"),
            ("--- LATTICE RESULT JSON ---", "--- END LATTICE RESULT JSON ---"),
        ]:
            if start in response_text and end in response_text:
                json_start = response_text.index(start) + len(start)
                json_end = response_text.index(end)
                json_str = response_text[json_start:json_end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Fallback: find first JSON object
        try:
            match = re.search(r'\{[^{}]*\}', response_text)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

        return {}

    def accumulate(self, stage: MBAStage, output: dict[str, Any]) -> None:
        """Add stage outputs to accumulated state."""
        for key in stage.produces_outputs:
            if key in output:
                self.accumulated_state[key] = output[key]

    def log_stage(
        self, stage: MBAStage, success: bool, output: dict[str, Any], note: str
    ) -> None:
        """Record stage execution result."""
        self.stage_log.append({
            "stage": stage.name,
            "description": stage.description,
            "success": success,
            "output_keys": list(output.keys()),
            "note": note,
        })

    def build_final_report(self) -> str:
        """Generate a human-readable report of the entire MBA construction."""
        lines = [
            "=" * 72,
            "MBA CONSTRUCTION COMPLETE — FINAL REPORT",
            "=" * 72,
            "",
        ]
        for entry in self.stage_log:
            status = "✓ PASS" if entry["success"] else "✗ FAIL"
            lines.append(
                f"[{status}] Stage: {entry['stage']} — {entry['description']}"
            )
            if entry["note"]:
                lines.append(f"      {entry['note']}")

        lines.append("")
        lines.append("-" * 72)
        lines.append("ACCUMULATED STATE SUMMARY")
        lines.append("-" * 72)
        for key, value in sorted(self.accumulated_state.items()):
            if isinstance(value, (list, dict)):
                lines.append(f"  {key}: {json.dumps(value)[:200]}")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience function for direct tool-only stages (no pyJuTrack)
# ═══════════════════════════════════════════════════════════════════════════════

def run_mba_tool_stage(
    stage: MBAStage,
    accumulated_state: dict[str, Any],
    user_request: str = "",
) -> dict[str, Any]:
    """For stages that use only tool calls (no pyJuTrack), run them directly.

    This is a lightweight alternative for stages 1-4 and 6 that don't need
    the full 4-agent pipeline with pyJuTrack code execution.

    Currently a placeholder — the LLM handles tool selection autonomously.
    Returns the accumulated state unchanged (LLM will update it via response).
    """
    return dict(accumulated_state)
