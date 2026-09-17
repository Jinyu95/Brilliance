"""Public read contracts for numerical dictionaries; no runtime conversion."""
from typing import Any, NotRequired, TypedDict

from .result_contracts import closed_result


@closed_result
class ScientificStage(TypedDict, total=False):
    stage_id: str | None
    hypothesis_id: str | None
    candidate_id: str | None
    parent_candidate_id: str | None
    outcome: str | None
    interpretation: str | None
    next_decision: str | None


@closed_result
class MagnetStrengthLimits(TypedDict):
    energy_gev: float
    dipole_field_t: float
    quadrupole_pole_tip_field_t: float
    sextupole_pole_tip_field_t: float
    quadrupole_radius_m: float
    sextupole_radius_m: float
    brho_tm: float
    minimum_bend_radius_m: float
    max_abs_k1_m2: float
    max_abs_k2_m3: float
    octupole_pole_tip_field_t: NotRequired[float]
    octupole_radius_m: NotRequired[float]
    max_abs_k3_m4: NotRequired[float]
    length_m: NotRequired[float]
    max_abs_angle_rad: NotRequired[float]
    max_abs_k1l_m1: NotRequired[float]
    max_abs_k2l_m2: NotRequired[float]
    max_abs_k3l_m3: NotRequired[float]


class ProfilePoint(TypedDict, total=False):
    name: str
    s_m: float
    beta_x: float
    beta_y: float
    alpha_x: float
    alpha_y: float
    Dx: float
    Dpx: float
    Dy: float
    Dpy: float


@closed_result
class BoundaryTwiss(TypedDict, total=False):
    beta_x_m: float
    beta_y_m: float
    alpha_x: float
    alpha_y: float
    Dx_m: float
    Dpx: float
    Dy_m: float
    Dpy: float


@closed_result
class LineMeasurement(TypedDict, total=False):
    engine: str
    scope: str
    evaluable: bool
    passed: bool
    endpoint: BoundaryTwiss
    metrics: dict[str, float]
    twiss_profile: list[ProfilePoint]
    geometry: dict[str, float]
    engineering: dict[str, Any]
    checks: dict[str, bool]
    diagnostics: list[str]
    sampling: dict[str, Any]


@closed_result
class BoundaryResidual(TypedDict):
    observable: str
    value: float
    target: float
    tolerance: float
    scaled_residual: float
    passed: bool


@closed_result
class MatchingStartBudget(TypedDict):
    requested_max_starts: int | None
    total_starts: int
    supplied_count: int
    duplicate_supplied_count: int
    baseline_included: bool
    generated_count: int
    deferred_count: int


@closed_result
class MatchingStartPlan(TypedDict):
    positions: list[list[float]]
    sources: list[str]
    supplied_indices: list[list[int]]
    deferred_positions: list[list[float]]
    deferred_sources: list[str]
    deferred_supplied_indices: list[list[int]]
    skipped: list[dict[str, Any]]
    effective_knob_bounds: list[list[float]]
    budget: MatchingStartBudget


@closed_result
class LineMatchAttempt(TypedDict, total=False):
    start: int
    source: str
    initial_position: list[float]
    physical_initial_position: list[float] | None
    solver_terminated: bool
    solver_success: bool
    solver_status: int | None
    message: str
    nfev: int | None
    njev: int | None
    final_position: list[float] | None
    residual_jacobian_singular_values: list[float]
    residual_jacobian_rank: int | None
    callback_evaluations: int
    reason: str
    diagnostic: str


@closed_result
class LineMatchResult(TypedDict, total=False):
    engine: str
    scope: str
    solver_terminated: bool
    solver_success: bool
    match_success: bool
    complete_cell_accepted: bool
    complete_cell_acceptance: dict[str, Any]
    reason: str
    elements: list[dict[str, Any]] | None
    position: list[float] | None
    incoming_twiss: BoundaryTwiss
    residuals: list[BoundaryResidual]
    measurement: LineMeasurement | None
    attempts: list[LineMatchAttempt]
    callback_evaluations: int
    configured_starts: int
    start_plan: MatchingStartPlan
    completed_starts: int
    remaining_start_positions: list[list[float]]
    remaining_start_sources: list[str]
    contract: dict[str, Any]
    effective_knob_bounds: list[list[float]]
    checks: dict[str, bool]
    limitations: list[str]
    numerical_checkpoint: dict[str, Any]


class LinearOpticsMapping(TypedDict, total=False):
    stable: bool | None
    trace_x: float | None
    trace_y: float | None
    tune_x: float | None
    tune_y: float | None
    beta_x_m: float | None
    beta_y_m: float | None
    alpha_x: float | None
    alpha_y: float | None
    Dx_m: float | None
    Dpx: float | None
    Dy_m: float | None
    Dpy: float | None
    maximum_beta_m: float | None
    twiss_profile: list[ProfilePoint]


class RadiationMapping(TypedDict, total=False):
    computed: bool
    reason: str | None
    I1: float | None
    I2: float | None
    I3: float | None
    I4: float | None
    I5: float | None
    Jx: float | None
    Jy: float | None
    Jz: float | None
    emittance_pm: float | None
    emittance_m: float | None
    energy_spread: float | None
    energy_loss_per_turn_eV: float | None
    physical_validity: dict[str, Any] | None


@closed_result
class PeriodicGeometry(TypedDict):
    cell_length_m: float
    predicted_ring_length_m: float
    cell_bend_sum_rad: float
    predicted_ring_bend_sum_rad: float
    ring_closure_error_rad: float
    ring_closure_tolerance_rad: float
    ring_bend_closure_passed: bool


class _LegacyGeometry(TypedDict, total=False):
    legacy_reported_geometry: dict[str, Any]


@closed_result
class EvaluatedGeometry(_LegacyGeometry):
    source: str
    row_geometry_available: bool
    candidate_lattice_id: str | None
    candidate_sha256: str | None
    physical_fingerprint: str | None
    energy_gev: float | None
    stored_scope: str | None
    stored_element_count: int | None
    cell_element_count: int | None
    stored_periods: int | None
    evaluated_periods: int | None
    stored_length_m: float | None
    cell_length_m: float | None
    ring_circumference_m: float | None
    reference_ring_circumference_m: float | None
    stored_bend_angle_rad: float | None
    cell_bend_angle_rad: float | None
    ring_bend_angle_rad: float | None
    reference_ring_bend_angle_rad: float | None


@closed_result
class PeriodicCellEvaluation(TypedDict):
    engine: str
    candidate_lattice_id: str
    evaluation_scope: str
    periods: int
    geometry: PeriodicGeometry
    linear_optics: LinearOpticsMapping
    radiation: RadiationMapping
    full_ring_validation_required: bool
    evaluation_geometry: EvaluatedGeometry
    evaluation_receipt: dict[str, Any]


@closed_result
class EvaluationChecks(TypedDict):
    linear_stable: bool | None
    ring_bend_closure_required: bool | None
    ring_bend_closure_passed: bool | None
    passed_baseline_gate: bool | None
    radiation_computed: bool
    radiation_physical_validity: bool | None
    full_ring_evaluated: bool


@closed_result
class EvaluationSnapshot(TypedDict):
    schema: str
    engine: str
    candidate_lattice_id: str | None
    scope: str
    geometry: EvaluatedGeometry
    linear_optics: LinearOpticsMapping
    radiation: RadiationMapping
    chromaticity: dict[str, Any]
    checks: EvaluationChecks
    acceptance: dict[str, Any]
    evaluation_receipt: dict[str, Any] | None
