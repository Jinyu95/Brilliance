"""Small MBA physics/measurement tools; scientific programs belong in projects."""

from brilliance.expert_sdk import python_tool

from .hmba_chromatic import solve_chromatic_correction
from .hybrid_shape import (
    evaluate_hybrid_mba_shape,
    hybrid_mba_bend_roles,
    hybrid_mba_shape_diagnostics,
    validate_hybrid_mba_shape_report,
)
from .physics_references import TMECellTargets, homogeneous_bend_tme_reference, tme_targets
from .radiation_reporting import mba_radiation_report
from .observables import hmba_optics_observables

TMECellTargets = python_tool()(TMECellTargets)
tme_targets = python_tool()(tme_targets)
homogeneous_bend_tme_reference = python_tool()(homogeneous_bend_tme_reference)
mba_radiation_report = python_tool()(mba_radiation_report)
hmba_optics_observables = python_tool()(hmba_optics_observables)
hybrid_mba_bend_roles = python_tool()(hybrid_mba_bend_roles)
hybrid_mba_shape_diagnostics = python_tool()(hybrid_mba_shape_diagnostics)
evaluate_hybrid_mba_shape = python_tool()(evaluate_hybrid_mba_shape)
validate_hybrid_mba_shape_report = python_tool()(validate_hybrid_mba_shape_report)
solve_chromatic_correction = python_tool()(solve_chromatic_correction)
