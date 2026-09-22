"""One read-only serialization contract for completed high-level JuTrack calls."""
from __future__ import annotations

import math
from typing import Any, Mapping

from .api_result_types import EvaluatedGeometry, EvaluationSnapshot
from .artifacts import json_safe
from .candidate_evidence import geometry_snapshot
from .lattice import LatticeIR, lattice_cell_partition

SCHEMA = "brilliance_evaluation_snapshot/v1"


def evaluated_geometry(candidate: LatticeIR, *, reference_periods: int | None = None) -> EvaluatedGeometry:
    """Capture exact evaluated rows at the simulator boundary, before its receipt."""
    snapshot = geometry_snapshot(candidate)
    metrics, binding = snapshot["canonical_metrics"], snapshot["binding"]
    count, stored_periods = lattice_cell_partition(candidate)
    length = metrics["stored_length_m"]
    angle = math.fsum(element.angle_rad or 0.0 for element in candidate.elements)
    cell_angles = [math.fsum(element.angle_rad or 0.0 for element in candidate.elements[start:start + count])
                   for start in range(0, len(candidate.elements), count)]
    return {
        "source": "evaluated_lattice_rows", "row_geometry_available": True,
        "candidate_lattice_id": candidate.lattice_id, "candidate_sha256": binding["candidate_sha256"],
        "physical_fingerprint": binding["physical_fingerprint"], "energy_gev": candidate.energy_gev,
        "stored_scope": snapshot["evaluation_scope"], "stored_element_count": len(candidate.elements),
        "cell_element_count": count, "stored_periods": stored_periods,
        "evaluated_periods": reference_periods if reference_periods is not None else stored_periods,
        "stored_length_m": length, "cell_length_m": metrics["cell_length_m"],
        "ring_circumference_m": metrics["ring_circumference_m"],
        "reference_ring_circumference_m": length * reference_periods if reference_periods is not None else None,
        "stored_bend_angle_rad": angle,
        "cell_bend_angle_rad": cell_angles[0] if all(math.isclose(value, cell_angles[0], rel_tol=0.0,
                                                                 abs_tol=1e-12) for value in cell_angles) else None,
        "ring_bend_angle_rad": angle if snapshot["evaluation_scope"] == "full_ring" else None,
        "reference_ring_bend_angle_rad": angle * reference_periods if reference_periods is not None else None,
    }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def evaluation_snapshot(evaluation: Any) -> EvaluationSnapshot:
    """Read geometry, optics, radiation and checks through one fixed dictionary.

    Accepts api.evaluate's typed result, api.evaluate_periodic_cell's dictionary,
    editor.evaluate_candidate's result, or saved JSON dictionaries. This never starts JuTrack, recalculates a
    lattice, expands a reference ring or accepts a research result. Preserve the
    ORIGINAL returned evaluation for finish(candidate_evaluation=...); this
    normalized view is for reading/reporting, not a replacement receipt payload.

    Keys: schema, engine, candidate_lattice_id, scope, geometry, linear_optics,
    radiation, chromaticity, checks, acceptance, evaluation_receipt. Missing
    numerical values are None, not zero. Geometry distinguishes actual stored
    lengths from an evaluated identical reference ring; no design targets enter.
    Technical baseline checks do not establish machine/request acceptance.
    """
    raw = evaluation.as_dict() if hasattr(evaluation, "as_dict") else evaluation
    if not isinstance(raw, Mapping) or raw.get("engine") != "JuTrack":
        raise TypeError("evaluation_snapshot requires a high-level JuTrack evaluation result")
    raw = json_safe(dict(raw))  # detached JSON values, including restored receipt dictionaries
    if raw.get("schema") == "brilliance_editor_evaluation/v1":
        from .accelerator_api import periodic_cell
        candidate = periodic_cell(title="Exact evaluated editor candidate", energy_gev=raw["energy_gev"],
                                  elements=raw["elements"], intended_repetitions=raw["periods"],
                                  source="JuTrackLatticeEditor.evaluate_candidate snapshot")
        raw["evaluation_geometry"] = evaluated_geometry(candidate, reference_periods=raw["periods"])
        raw["geometry"] = {"ring_bend_closure_required": False}
        raw["evaluation_scope"] = "periodic_cell"
    if raw.get("schema") == SCHEMA:
        return raw
    if not isinstance(raw.get("linear_optics"), Mapping) or not isinstance(raw.get("geometry"), Mapping):
        raise TypeError("evaluation_snapshot requires api.evaluate or api.evaluate_periodic_cell output; "
                        "an open-line measurement instead exposes its endpoint mapping")
    legacy_geometry = dict(raw.get("geometry") or {})
    scope = str(raw.get("evaluation_scope") or legacy_geometry.get("evaluation_scope") or "unknown")
    geometry = raw.get("evaluation_geometry")
    if not isinstance(geometry, Mapping):
        candidate = raw.get("candidate")
        if isinstance(candidate, Mapping):
            geometry = evaluated_geometry(LatticeIR.from_dict(candidate))
            geometry["source"] = "saved_evaluation_candidate_rows"
        else:
            # Old periodic results have no row snapshot. Preserve their reported
            # values for diagnosis, without inventing a stored ring or period count.
            geometry = {key: None for key in (
                "candidate_lattice_id", "candidate_sha256", "physical_fingerprint", "energy_gev",
                "stored_scope", "stored_element_count", "cell_element_count", "stored_periods",
                "evaluated_periods", "stored_length_m", "cell_length_m", "ring_circumference_m",
                "reference_ring_circumference_m", "stored_bend_angle_rad", "cell_bend_angle_rad",
                "ring_bend_angle_rad", "reference_ring_bend_angle_rad")}
            geometry.update(source="legacy_evaluation_without_rows", row_geometry_available=False,
                            legacy_reported_geometry=legacy_geometry)
    linear = dict(raw.get("linear_optics") or {})
    profile = list(linear.get("twiss_profile") or [])
    start = profile[0] if profile and isinstance(profile[0], Mapping) and profile[0].get("name") == "START" else {}
    for name, profile_name in (("beta_x_m", "beta_x"), ("beta_y_m", "beta_y"), ("alpha_x", "alpha_x"),
                               ("alpha_y", "alpha_y"), ("Dx_m", "Dx"), ("Dpx", "Dpx"),
                               ("Dy_m", "Dy"), ("Dpy", "Dpy")):
        linear[name] = _number(linear.get(name, start.get(profile_name)))
    for name in ("trace_x", "trace_y", "tune_x", "tune_y", "periodic_closure_max",
                 "predicted_ring_fractional_tune_x", "predicted_ring_fractional_tune_y"):
        linear[name] = _number(linear.get(name))
    sampled = [_number(row.get(name)) for row in profile if isinstance(row, Mapping) for name in ("beta_x", "beta_y")]
    maximum = _number(linear.get("maximum_beta_m", linear.get("beta_max_m")))
    linear.update(stable=_boolean(linear.get("stable")), twiss_profile=profile,
                  profile_available=bool(profile), maximum_beta_m=maximum if maximum is not None else max(
                      (value for value in sampled if value is not None), default=None),
                  maximum_beta_is_sampled=True, periodic_residuals=dict(linear.get("periodic_residuals") or {}),
                  one_turn_matrix=list(linear.get("one_turn_matrix") or []))
    linear["boundary_optics"] = {key: linear[key] if linear["stable"] is True else None for key in (
        "beta_x_m", "beta_y_m", "alpha_x", "alpha_y", "Dx_m", "Dpx", "Dy_m", "Dpy")}
    linear["boundary_optics_available"] = linear["stable"] is True and all(
        linear["boundary_optics"][key] is not None for key in ("beta_x_m", "beta_y_m", "Dx_m", "Dpx"))
    radiation = dict(raw.get("radiation") or {})
    radiation.setdefault("computed", False)
    radiation.setdefault("reason", None if radiation["computed"] else "Radiation was not requested or is unavailable in this evaluation.")
    radiation.setdefault("physical_validity", None)
    for name in ("I1", "I2", "I3", "I4", "I5", "Jx", "Jy", "Jz", "emittance_m", "emittance_pm",
                 "energy_spread", "energy_loss_per_turn_eV"):
        radiation.setdefault(name, None)
    chromaticity = dict(raw.get("chromaticity") or {})
    chromaticity.update(computed=bool(chromaticity), horizontal=_number(chromaticity.get("horizontal")),
                        vertical=_number(chromaticity.get("vertical")))
    required = _boolean(legacy_geometry.get("ring_bend_closure_required"))
    if required is None:
        required = scope != "periodic_cell"
    closed = _boolean(legacy_geometry.get("ring_bend_closure_passed"))
    stable = linear["stable"]
    baseline = (stable and (not required or closed)) if stable is not None else None
    physical = radiation.get("physical_validity")
    checks = {"linear_stable": stable, "ring_bend_closure_required": required,
              "ring_bend_closure_passed": closed, "passed_baseline_gate": baseline,
              "radiation_computed": radiation["computed"] is True,
              "radiation_physical_validity": _boolean(physical.get("passed")) if isinstance(physical, Mapping) else None,
              "full_ring_evaluated": geometry.get("stored_scope") == "full_ring" and scope == "full_ring"}
    return {"schema": SCHEMA, "engine": "JuTrack",
            "candidate_lattice_id": geometry.get("candidate_lattice_id") or raw.get("candidate_lattice_id"),
            "scope": scope, "geometry": geometry, "linear_optics": linear, "radiation": radiation,
            "chromaticity": chromaticity, "checks": checks,
            "acceptance": {"accepted": False, "status": "not_assessed", "publication_required": False,
                           "machine_requirements": "not_assessed", "scientific_request": "not_assessed",
                           "full_ring_validation_required": not checks["full_ring_evaluated"]},
            "evaluation_receipt": raw.get("evaluation_receipt")}
