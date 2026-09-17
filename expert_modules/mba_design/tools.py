"""Small MBA physics/measurement tools; scientific programs belong in projects."""

from brilliance.expert_sdk import acceptance, python_tool

from .acceptance import (
    CONVENTIONAL_LINEAR,
    CONVENTIONAL_LINEAR_INSTRUCTIONS,
    HYBRID_LINEAR,
    HYBRID_LINEAR_INSTRUCTIONS,
    MBA_OPTIMIZATION,
    MBA_OPTIMIZATION_INSTRUCTIONS,
    is_conventional_mba_design_request,
    is_hybrid_linear_request,
    is_mba_emittance_optimization_request,
)
from .conventional_claims import validate_conventional_linear, verify_conventional_candidate
from .hmba_chromatic import solve_chromatic_correction
from .hybrid_measurement import HybridMeasurement
from .hybrid_shape import (
    evaluate_hybrid_mba_shape,
    hybrid_mba_bend_roles,
    hybrid_mba_shape_diagnostics,
    validate_hybrid_mba_shape_report,
)
from .outcome_claims import validate_hmba_linear, validate_mba_optimization, verify_hmba_candidate
from .outer_units import assemble_mba_outer_units, match_mba_outer_units, mba_outer_unit_knobs
from .physics_references import TMECellTargets, homogeneous_bend_tme_reference, tme_targets
from .radiation_reporting import mba_radiation_report
from .tme_geometry import optimize_tme_unit_geometry
from .tme_unit import design_tme_unit

TMECellTargets = python_tool()(TMECellTargets)
tme_targets = python_tool()(tme_targets)
design_tme_unit = python_tool(durable=True)(design_tme_unit)
optimize_tme_unit_geometry = python_tool(durable=True)(optimize_tme_unit_geometry)
assemble_mba_outer_units = python_tool()(assemble_mba_outer_units)
mba_outer_unit_knobs = python_tool()(mba_outer_unit_knobs)
match_mba_outer_units = python_tool(durable=True)(match_mba_outer_units)
homogeneous_bend_tme_reference = python_tool()(homogeneous_bend_tme_reference)
mba_radiation_report = python_tool()(mba_radiation_report)
HybridMeasurement = python_tool()(HybridMeasurement)
hybrid_mba_bend_roles = python_tool()(hybrid_mba_bend_roles)
hybrid_mba_shape_diagnostics = python_tool()(hybrid_mba_shape_diagnostics)
evaluate_hybrid_mba_shape = python_tool()(evaluate_hybrid_mba_shape)
validate_hybrid_mba_shape_report = python_tool()(validate_hybrid_mba_shape_report)
solve_chromatic_correction = python_tool()(solve_chromatic_correction)
verify_hmba_candidate = python_tool()(verify_hmba_candidate)
verify_conventional_candidate = python_tool()(verify_conventional_candidate)


@acceptance(id=CONVENTIONAL_LINEAR, applies_to=is_conventional_mba_design_request,
            instructions=CONVENTIONAL_LINEAR_INSTRUCTIONS, requires_lattice=True)
def conventional_linear_acceptance(report, lattice):
    return validate_conventional_linear(report, lattice)


@acceptance(id=HYBRID_LINEAR, applies_to=is_hybrid_linear_request,
            instructions=HYBRID_LINEAR_INSTRUCTIONS, requires_lattice=True)
def hybrid_linear_acceptance(report, lattice):
    return validate_hmba_linear(report, lattice)


@acceptance(id=MBA_OPTIMIZATION, applies_to=is_mba_emittance_optimization_request,
            instructions=MBA_OPTIMIZATION_INSTRUCTIONS, requires_lattice=True)
def mba_optimization_acceptance(report, lattice):
    return validate_mba_optimization(report, lattice)
