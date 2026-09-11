"""Public MBA capabilities, discovered from this expert package without core edits.

These tools accept ordinary Python lattice objects and arrays. They are used by
research programs through api.expert_module("laboratory.mba_design"). The legacy
names keep archived programs reproducible while this module remains installed.
"""

from brilliance.expert_sdk import acceptance, python_tool

from .acceptance import (
    HYBRID_INSTRUCTIONS,
    HYBRID_SHAPE,
    hybrid_optimization_warnings,
    is_hybrid_shape_request,
)
from .hybrid_regions import HybridMBARegionalGeometry as _HybridMBARegionalGeometry
from .hybrid_regions import hybrid_mba_regions as _hybrid_mba_regions
from .outer_dba import HybridMBAOuterReplacement as _HybridMBAOuterReplacement
from .outer_dba import build_outer_dba_section as _build_outer_dba_section
from .outer_dba import match_outer_dba_section as _match_outer_dba_section
from .hybrid_continuation import explore_hybrid_mba as _explore_hybrid_mba
from .hybrid_optics import measure_core_optics as _measure_core_optics
from .hybrid_optics import compare_core_optics as _compare_core_optics
from .hybrid_response import (
    HybridMBAShapeEvaluator as _HybridMBAShapeEvaluator,
)
from .hybrid_response import (
    hybrid_mba_quadrupole_knobs as _hybrid_mba_quadrupole_knobs,
)
from .hybrid_shape import (
    evaluate_hybrid_mba_shape as _evaluate_hybrid_mba_shape,
)
from .hybrid_shape import (
    hybrid_mba_bend_roles as _hybrid_mba_bend_roles,
)
from .hybrid_shape import (
    hybrid_mba_shape_diagnostics as _hybrid_mba_shape_diagnostics,
)
from .hybrid_shape import (
    validate_hybrid_mba_shape_report as _validate_hybrid_mba_shape_report,
)
from .physics_references import (
    TMECellTargets as _TMECellTargets,
)
from .physics_references import (
    homogeneous_bend_tme_reference as _homogeneous_bend_tme_reference,
)
from .physics_references import (
    tme_targets as _tme_targets,
)

TMECellTargets = python_tool(legacy_names=("TMECellTargets",))(_TMECellTargets)
tme_targets = python_tool(legacy_names=("tme_targets",))(_tme_targets)
homogeneous_bend_tme_reference = python_tool(legacy_names=("homogeneous_bend_tme_reference",))(_homogeneous_bend_tme_reference)
hybrid_mba_bend_roles = python_tool(legacy_names=("hybrid_mba_bend_roles",))(_hybrid_mba_bend_roles)
hybrid_mba_shape_diagnostics = python_tool(legacy_names=("hybrid_mba_shape_diagnostics",))(_hybrid_mba_shape_diagnostics)
evaluate_hybrid_mba_shape = python_tool(legacy_names=("evaluate_hybrid_mba_shape",))(_evaluate_hybrid_mba_shape)
validate_hybrid_mba_shape_report = python_tool()(_validate_hybrid_mba_shape_report)
HybridMBAShapeEvaluator = python_tool()(_HybridMBAShapeEvaluator)
hybrid_mba_quadrupole_knobs = python_tool()(_hybrid_mba_quadrupole_knobs)
HybridMBARegionalGeometry = python_tool()(_HybridMBARegionalGeometry)
hybrid_mba_regions = python_tool()(_hybrid_mba_regions)
HybridMBAOuterReplacement = python_tool()(_HybridMBAOuterReplacement)
build_outer_dba_section = python_tool()(_build_outer_dba_section)
match_outer_dba_section = python_tool()(_match_outer_dba_section)
explore_hybrid_mba = python_tool()(_explore_hybrid_mba)
measure_core_optics = python_tool()(_measure_core_optics)
compare_core_optics = python_tool()(_compare_core_optics)


@acceptance(
    id=HYBRID_SHAPE,
    applies_to=is_hybrid_shape_request,
    instructions=HYBRID_INSTRUCTIONS,
    preflight=hybrid_optimization_warnings,
    requires_lattice=True,
)
def hybrid_shape_acceptance(report, lattice) -> list[str]:
    """Validate complete-cell hybrid shape evidence against its archived candidate."""
    return _validate_hybrid_mba_shape_report(report, lattice)
