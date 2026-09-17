"""Discoverable numerical handoff contracts; mappings are not transformed."""

from typing import Any, TypedDict

from brilliance.api_result_types import PeriodicCellEvaluation, ProfilePoint
from brilliance.result_contracts import closed_result


@closed_result
class OuterUnitAssembly(TypedDict):
    elements: list[dict[str, Any]]
    unit_groups: list[list[str]]
    core_unit_groups: list[list[str]]
    id_drift_names: list[str]
    id_length_m: float
    outer_unit_numbers: list[int]
    outer_unit_length_m: float
    source_unit_length_m: float
    end_bend_fraction: float
    geometry: dict[str, Any]
    physics_accepted: bool
    route: str


@closed_result
class OuterUnitMatch(TypedDict, total=False):
    matched: bool
    elements: list[dict[str, Any]]
    candidate: dict[str, Any]
    evaluation: PeriodicCellEvaluation
    profile_verification: dict[str, Any]
    attempts: list[dict[str, Any]]
    best: dict[str, Any] | None
    evaluations: int
    elapsed_s: float
    complete_ring_accepted: bool
    knobs: dict[str, Any]
    numerical_checkpoint: dict[str, Any]


class BoundaryTwiss(TypedDict):
    beta_x_m: float
    beta_y_m: float
    alpha_x: float
    alpha_y: float
    Dx_m: float
    Dpx: float
    Dy_m: float
    Dpy: float


class UnitCandidate(TypedDict, total=False):
    elements: list[dict[str, Any]]
    periodic_cell: dict[str, Any]
    evaluation: PeriodicCellEvaluation
    reference_ring_evaluation: dict[str, Any]
    reference_ring_receipt: dict[str, Any]
    boundary_twiss: BoundaryTwiss
    emittance_pm: float
    k1_near_m2: float
    k1_far_m2: float
    position: list[float]
    polarity: str
    checks: dict[str, bool]
    optics_profile: list[ProfilePoint]
    bend_centre_diagnostic: dict[str, Any]
    beta_sampling_scope: str
    construction_and_engineering: dict[str, Any]
    numerical_checkpoint: dict[str, Any]
    verified_unit_checkpoint: dict[str, Any]
    complete_mba_accepted: bool


class TmeUnitResult(TypedDict, total=False):
    passed: bool
    candidates: list[UnitCandidate]
    provided_constraints: dict[str, Any]
    numerical_checkpoint: dict[str, Any]
    complete_mba_accepted: bool


class GeometryFinalist(TypedDict):
    geometry: dict[str, float]
    outer_position: list[float]
    outer_emittance_pm: float
    verification: TmeUnitResult


class TmeGeometryResult(TmeUnitResult, total=False):
    optimizer: dict[str, Any]
    geometry_finalists: list[GeometryFinalist]
    initialization: dict[str, Any]
    trace_limits: dict[str, Any]
    budget: dict[str, Any]
    optimizer_started: bool
    contract: dict[str, Any]
    contract_sha256: str
