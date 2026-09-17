"""Actual saved-unit -> periodic outer-pair match -> full ring -> app publication."""
import json
import math

from brilliance import accelerator_api as api
from brilliance.lattice import LatticeIR
from brilliance.research_api import open_research_run

run = open_research_run()
mba = api.expert_module("laboratory.mba_design")
unit_raw = json.loads((run.workspace / "inputs/unit.json").read_text(encoding="utf-8"))
machine = json.loads((run.workspace / "inputs/contract.json").read_text(encoding="utf-8"))["machine"]
assembly = mba.assemble_mba_outer_units(unit_raw["elements"], energy_gev=machine["energy_gev"],
    periods=machine["periods"], bends_per_cell=machine["bends_per_cell"], cell_length_m=machine["cell_length_m"],
    straight_length_m=machine["id_straight_length_m"], limits=machine["limits"],
    minimum_magnet_length_m=machine["minimum_magnet_length_m"], minimum_drift_length_m=machine["minimum_drift_length_m"],
    end_bend_fraction=machine["terminal_angle_rad"] / machine["unit_angle_rad"])
assembled = api.periodic_cell(title="Unmatched repeated-unit MBA", energy_gev=machine["energy_gev"],
    elements=assembly["elements"], intended_repetitions=machine["periods"], source="outer-unit validation")
api.save_lattice(run.artifact_path("assembled_cell.json"), assembled)
matched = mba.match_mba_outer_units(assembly, energy_gev=machine["energy_gev"], periods=machine["periods"],
    limits=machine["limits"], maximum_beta_m=machine["maximum_beta_m"],
    minimum_magnet_length_m=machine["minimum_magnet_length_m"], minimum_drift_length_m=machine["minimum_drift_length_m"],
    trace_margin_x=machine["trace_margin_x"], trace_margin_y=machine["trace_margin_y"])
assert matched["matched"], "No verified complete cell in this validation"
cell = LatticeIR.from_dict(matched["candidate"])
checkpoint = run.checkpoint_candidate(cell, matched["evaluation"], name="outer_pair_cell")
api.save_lattice(run.artifact_path("matched_cell.json"), cell)
unit = api.periodic_cell(title="Original measured unit", energy_gev=machine["energy_gev"],
    elements=unit_raw["elements"], intended_repetitions=machine["reference_repetitions"], source="explicit regression input")
contract = dict(energy_gev=machine["energy_gev"], periods=machine["periods"],
    expected_cell_length_m=machine["cell_length_m"], expected_cell_angle_rad=2 * math.pi / machine["periods"],
    expected_bends=machine["bends_per_cell"], minimum_magnet_length_m=machine["minimum_magnet_length_m"],
    minimum_drift_length_m=machine["minimum_drift_length_m"], limits=machine["limits"],
    id_drift_names=assembly["id_drift_names"], id_length_m=assembly["id_length_m"],
    achromat_dx_tolerance_m=1e-4, achromat_dpx_tolerance=1e-5, maximum_beta_m=machine["maximum_beta_m"],
    maximum_emittance_pm=None, maximum_energy_spread=None,
    limit_provenance={"saved_contract": "inputs/contract.json", "original_statement": machine["limit_provenance"],
                      "acceptance_tolerances": "Dx 0.1 mm and Dpx 1e-5; validation criteria, not inferred machine dimensions"})
proof = mba.verify_conventional_candidate(cell, unit=unit, assembled_candidate=assembled,
    unit_groups=assembly["unit_groups"], machine_contract=contract)
assert proof["passed"], proof["issues"]
ring = api.cell_ring(title="Explicit verified 32-cell ring", energy_gev=machine["energy_gev"],
    cell=matched["elements"], periods=machine["periods"], source="exact outer-pair match")
api.save_lattice(run.artifact_path("explicit_ring.json"), ring)
bends = api.bend_radiation_contributions(ring)
with run.reporting(checkpoint=checkpoint):
    if run.arguments.get("fail_reporting"):
        raise ValueError("Intentional reporting failure after verified outer-unit matching")
    plot = api.save_optics_lattice_plot(run.artifact_path("one_cell_optics.svg"),
        matched["profile_verification"]["twiss_profile"], matched["elements"], title="Complete 7BA: matched outer two unit pairs")
    run.finish(summary="TME-derived outer-unit matching produced a verified achromatic 800 m ring. Emittance optimization remains separate.",
        data={"match": matched, "proof": proof, "assembly": assembly},
        tables=[{"title": "Whole-cell verification", "rows": [{"Quantity": k, "Value": v}
                 for k, v in proof["metrics"].items() if isinstance(v, (int, float))]},
                {"title": "Per-bend radiation contributions", "rows": bends["rows"]}],
        plots=[plot], candidate_lattice=cell, candidate_evaluation=matched["evaluation"],
        acceptance={"conventional_mba_linear": proof}, scientific_passed=True, request_satisfied=False,
        use_as_working_lattice=False, limitations=["Explicit saved-unit integration, not a fresh autonomous run.",
                                                 "No optimized emittance, chromatic correction or nonlinear dynamics claim."])
