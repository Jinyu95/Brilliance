"""Bind reported JuTrack results to the exact evaluated lattice snapshot.

Capture is cheap and process-local. Only a deliberately selected evaluation is
written by ResearchRun.finish. This is provenance, not a new physics gate, and
never invokes a simulator. It is an accidental-mismatch guard, not a sandbox
against arbitrary Python code with access to the same filesystem.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .artifacts import canonical_json, json_safe
from .atomic_io import atomic_write_text
from .lattice import LatticeIR, lattice_cell_partition

SCHEMA = "brilliance_candidate_evaluation/v1"
_MAX_CAPTURED_EVALUATIONS = 32
_captured: OrderedDict[str, dict[str, Any]] = OrderedDict()
_lock = threading.RLock()
_evidence_directory: ContextVar[Path | None] = ContextVar("numerical_evidence_directory", default=None)
_NEXT_ACTION = (
    "Evaluate the exact candidate made from the matcher's returned elements with "
    "api.evaluate(candidate) (or api.evaluate_periodic_cell for an explicit cell), "
    "then pass that returned result to finish(candidate_lattice=candidate, "
    "candidate_evaluation=evaluation). No optimizer rerun is required."
)
_LINE_NEXT_ACTION = (
    "Keep this open line separate from periodic-cell or ring acceptance. Report its "
    "saved JuTrack line-match numerical checkpoint, exact returned elements, incoming "
    "Twiss state and residuals; use run.reporting(checkpoint=the_saved_checkpoint) for "
    "optional presentation. A propagated line is not periodic evidence. Do not rerun "
    "matching merely to repair reporting or change its physical scope for publication."
)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _scope() -> str:
    configured = str(os.environ.get("BRILLIANCE_RUN_OUTPUT", "")).strip()
    return str(Path(configured).resolve()) if configured else ""


def _candidate(candidate: Any) -> LatticeIR:
    value = candidate.as_dict() if isinstance(candidate, LatticeIR) else candidate
    if not isinstance(value, Mapping):
        raise TypeError("candidate must be a lattice or serialized lattice")
    # Deep rehydration also detects mutation of a frozen object's nested fields.
    result = LatticeIR.from_dict(copy.deepcopy(value))
    lattice_cell_partition(result)
    return result


def _binding(candidate: LatticeIR, effective_periods: int | None) -> dict[str, Any]:
    count, repetitions = lattice_cell_partition(candidate)
    if effective_periods is None:
        effective_periods = repetitions
    if (isinstance(effective_periods, bool) or not isinstance(effective_periods, int)
            or effective_periods < 1):
        raise ValueError("effective evaluation periods must be a positive integer")
    return {
        "lattice_id": candidate.lattice_id,
        "candidate_sha256": _digest(candidate.as_dict()),
        "physical_fingerprint": candidate.physical_fingerprint,
        "energy_gev": candidate.energy_gev,
        "evaluation_scope": candidate.metadata.get("evaluation_scope", "full_ring"),
        "periodicity": candidate.periodicity,
        "cell_element_count": count,
        "repetitions": repetitions,
        "intended_repetitions": candidate.metadata.get("intended_repetitions"),
        "effective_periods": effective_periods,
    }


def record_evaluation(candidate: Any, payload: Mapping[str, Any], *,
                      effective_periods: int | None = None) -> dict[str, Any]:
    """Capture one completed high-level JuTrack evaluation before returning it.

    The returned opaque descriptor is embedded in the public evaluation result.
    Values supplied later by a caller cannot replace this detached capture.
    The bounded cache may evict optimizer trials; final-evaluate the selected
    candidate at the reporting boundary instead of relying on an old trial.
    """
    snapshot = _candidate(candidate)
    calculation = copy.deepcopy(json_safe(dict(payload)))
    calculation.pop("evaluation_receipt", None)
    evaluation_id = uuid4().hex
    archive = {"schema": SCHEMA, "evaluation_id": evaluation_id,
               "candidate": snapshot.as_dict(),
               "binding": _binding(snapshot, effective_periods),
               "calculation": calculation}
    descriptor = {"schema": SCHEMA, "evaluation_id": evaluation_id,
                  "content_sha256": _digest(archive),
                  "candidate_lattice_id": snapshot.lattice_id}
    with _lock:
        _captured[evaluation_id] = {"scope": _scope(), "archive": archive,
                                    "descriptor": descriptor}
        while len(_captured) > _MAX_CAPTURED_EVALUATIONS:
            _captured.popitem(last=False)
    return copy.deepcopy(descriptor)


def _unverified(issue: str, *, candidate: Any = None) -> dict[str, Any]:
    metadata = candidate.metadata if isinstance(candidate, LatticeIR) else (
        candidate.get("metadata", {}) if isinstance(candidate, Mapping) else {})
    is_line = isinstance(metadata, Mapping) and (
        metadata.get("not_a_ring") is True or metadata.get("evaluation_scope") in {
            "open_line", "transfer_line", "open_line_boundary_match", "uncoupled_hard_edge_transfer_line"})
    return {"status": "unverified", "passed": False, "issues": [issue],
            "next_action": _LINE_NEXT_ACTION if is_line else _NEXT_ACTION, "receipt": None,
            "canonical_metrics": {}, "table": None}


def _finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


_GEOMETRY_LENGTH_KEYS = frozenset({
    "stored_length_m", "cell_length_m", "ring_circumference_m",
    "projected_ring_circumference_m", "circumference_m", "predicted_ring_length_m",
})
_GEOMETRY_ABSOLUTE_TOLERANCE_M = 1e-9
_GEOMETRY_LABELS = {
    "stored_length_m": ("Stored element sequence length", "m"),
    "cell_length_m": ("Periodic cell length", "m"),
    "ring_circumference_m": ("Explicit full-ring circumference", "m"),
    "projected_ring_circumference_m": ("Projected circumference from intended repetitions (not a stored ring)", "m"),
    "stored_periods": ("Stored cells", "1"),
    "intended_repetitions": ("Intended repetitions of this cell", "1"),
}


def geometry_snapshot(candidate: Any) -> dict[str, Any]:
    """Derive geometry from exact saved rows without starting JuTrack.

    Stored length, a declared full ring and a future repetition of a single
    periodic cell are different quantities. No design target, title or
    evaluation summary participates in these sums. ``passed`` means the
    snapshot/partition is valid, never that the machine requirements or beam
    physics passed. A ring is reported only when its scope is explicit.
    """
    lattice = _candidate(candidate)
    count, periods = lattice_cell_partition(lattice)
    scope = str(lattice.metadata.get("evaluation_scope") or "undeclared")
    stored_length = math.fsum(element.length_m for element in lattice.elements)
    lengths = [math.fsum(element.length_m for element in lattice.elements[start:start + count])
               for start in range(0, len(lattice.elements), count)]
    equal_lengths = all(math.isclose(length, lengths[0], rel_tol=0.0,
                                    abs_tol=_GEOMETRY_ABSOLUTE_TOLERANCE_M) for length in lengths)
    # Do not present the first cell as a common length when the stored cells differ.
    cell_length = lengths[0] if equal_lengths else None
    intended = lattice.metadata.get("intended_repetitions")
    intended = int(float(intended)) if intended is not None else None  # already partition-validated
    projected = (stored_length * intended
                 if scope == "periodic_cell" and intended is not None else None)
    metrics = {
        "stored_length_m": stored_length,
        "cell_length_m": cell_length,
        "ring_circumference_m": stored_length if scope == "full_ring" else None,
        "projected_ring_circumference_m": projected,
        "stored_periods": periods,
        "intended_repetitions": intended,
    }
    description = (f"Candidate {lattice.lattice_id}; declared scope: {scope}. "
                   "Lengths are summed from the exact stored elements, not copied from design targets. "
                   "This is geometry provenance, not a physics or request-acceptance certificate.")
    if not equal_lengths:
        description += " Stored cells have different lengths; no common periodic-cell length is claimed."
    return {
        "schema": "brilliance_candidate_geometry/v1", "passed": True,
        "candidate_lattice_id": lattice.lattice_id,
        "binding": {**_binding(lattice, None), "evaluation_scope": scope}, "evaluation_scope": scope,
        "canonical_metrics": metrics, "stored_cell_lengths_m": lengths,
        "equal_stored_cell_lengths": equal_lengths,
        "table": {"title": "Exact candidate geometry — independent of design targets",
                  "kind": "candidate_geometry", "lattice_id": lattice.lattice_id,
                  "description": description,
                  "rows": [{"Quantity": _GEOMETRY_LABELS[key][0], "Value": value,
                            "Unit": _GEOMETRY_LABELS[key][1]}
                           for key, value in metrics.items() if value is not None]},
    }


def unbound_geometry_tables(tables: Sequence[Mapping[str, Any]], candidate_id: str | None) -> list[dict[str, Any]]:
    """Find explicitly labelled geometry tables lacking current-candidate lineage.

    This does not compare numbers or interpret arbitrary comparison columns.
    A row saying ``circumference, m, cell_scope_32_repeats=800, ring=None``
    needs source identity before any of those values can be treated as measured.
    Declared contracts, targets and hypotheses are not measurement claims.
    """
    labels = {"circumference", "ring circumference", "cell length", *_GEOMETRY_LENGTH_KEYS,
              *(label.casefold() for label, unit in _GEOMETRY_LABELS.values() if unit == "m")}
    labelled_nonmeasurement = {"contract", "targets", "target", "requirements", "reference", "hypothesis"}
    typed = {"candidate_geometry", "candidate_metrics", "candidate_evaluation"}
    unverified = []
    for index, table in enumerate(tables):
        if not isinstance(table, Mapping):
            continue
        kind = str(table.get("kind", "")).strip().casefold()
        identity = table.get("candidate_lattice_id", table.get("lattice_id"))
        if (kind in labelled_nonmeasurement or (identity is not None and identity != candidate_id)
                or (identity is not None and identity == candidate_id and kind in typed)):
            continue
        rows = table.get("rows", [])
        if not isinstance(rows, (list, tuple)):
            continue
        quantities = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            label = row.get("metric", row.get("quantity", row.get("Quantity")))
            unit = str(row.get("unit", row.get("Unit", ""))).strip().casefold()
            if isinstance(label, str) and label.strip().casefold() in labels and unit in {"m", "cm", "mm"}:
                quantities.add(label)
        if quantities:
            unverified.append({"table_index": index, "title": str(table.get("title", "Geometry table")),
                               "quantities": sorted(quantities), "reason": "missing_candidate_geometry_lineage"})
    return unverified


def compare_declared_geometry(data: Mapping[str, Any], snapshot: Mapping[str, Any], *,
                              tables: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Compare explicitly scoped current-candidate claims with exact row sums.

    Check direct metric keys and ``geometry``/``current_candidate`` containers;
    do not recursively reinterpret contracts, targets, parents or comparisons.
    Tables must identify this candidate and use a candidate geometry/metrics
    kind. Typed rows may supply a metric/value pair or canonical metric columns.
    Human prose and arbitrary comparison-column names are not machine claims.
    ``unverified_claims`` separately identifies unbound geometry tables; these
    are missing lineage, not numerical conflicts or a failed physics result.
    """
    canonical = snapshot.get("canonical_metrics", {})
    if snapshot.get("passed") is not True or not isinstance(canonical, Mapping):
        return {"passed": True, "conflicts": [], "checked_fields": [], "unverified_claims": [], "next_action": None}
    expected = dict(canonical)
    # Existing api.evaluate geometry uses circumference_m for the evaluated
    # stored sequence, including a cell. Retain that explicit legacy meaning.
    expected["circumference_m"] = canonical.get("stored_length_m")
    expected["predicted_ring_length_m"] = canonical.get("projected_ring_circumference_m")
    candidate_id = snapshot.get("candidate_lattice_id")
    conflicts: list[dict[str, Any]] = []
    checked: list[str] = []

    def check(metric: str, declared: Any, path: str) -> None:
        if metric not in expected:
            return
        measured = expected[metric]
        checked.append(path)
        same = declared is None if measured is None else (
            _finite(declared) and math.isclose(float(declared), float(measured), rel_tol=0.0,
                                              abs_tol=(_GEOMETRY_ABSOLUTE_TOLERANCE_M
                                                       if metric in _GEOMETRY_LENGTH_KEYS else 0.0)))
        if not same:
            conflicts.append({"metric": metric, "path": path,
                              "declared": json_safe(declared), "verified": measured})

    def current(values: Mapping[str, Any], path: str) -> None:
        # Explicitly different candidate identity denotes a comparison, not the selected one.
        identity = values.get("candidate_lattice_id", values.get("lattice_id"))
        if identity is not None and identity != candidate_id:
            return
        for key, value in values.items():
            check(key, value, f"{path}.{key}")
        for key in ("geometry", "current_candidate"):
            value = values.get(key)
            if isinstance(value, Mapping):
                current(value, f"{path}.{key}")

    current(data, "data")
    labels = {label: key for key, (label, _unit) in _GEOMETRY_LABELS.items()}
    labels.update({"Evaluated lattice length": "circumference_m", "Cell length": "cell_length_m",
                   "Repeated ring circumference": "predicted_ring_length_m"})
    for index, table in enumerate(tables):
        if (not isinstance(table, Mapping) or table.get("lattice_id") != candidate_id
                or table.get("kind") not in {"candidate_geometry", "candidate_metrics", "candidate_evaluation"}):
            continue
        rows = table.get("rows", [])
        if not isinstance(rows, (list, tuple)):
            continue
        for number, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            identity = row.get("candidate_lattice_id", row.get("lattice_id"))
            if identity is not None and identity != candidate_id:
                continue
            path = f"tables[{index}].rows[{number}]"
            current(row, path)
            metric = row.get("metric", row.get("quantity", row.get("Quantity")))
            if isinstance(metric, str):
                value_key = "value" if "value" in row else "Value"
                if value_key in row:
                    check(labels.get(metric, metric), row[value_key], f"{path}.{value_key}")
    return {
        "passed": not conflicts, "conflicts": conflicts, "checked_fields": checked,
        "unverified_claims": unbound_geometry_tables(tables, candidate_id),
        "geometry_absolute_tolerance_m": _GEOMETRY_ABSOLUTE_TOLERANCE_M,
        "geometry_relative_tolerance": 0.0,
        "next_action": (
            "Correct the reported geometry from the exact candidate's row sums. Keep intended machine "
            "requirements in a separately labelled target/contract section. Do not rerun matching or "
            "optimization merely to repair the report; the candidate is preserved unchanged."
            if conflicts else None),
    }


def _canonical_metrics(archive: Mapping[str, Any]) -> dict[str, Any]:
    """Known typed scalar fields only; do not guess values from arbitrary prose."""
    calculation = archive["calculation"]
    result: dict[str, Any] = {}
    fields = {
        "geometry": ("circumference_m", "cell_length_m", "predicted_ring_length_m",
                     "bend_sum_rad", "predicted_ring_bend_sum_rad", "ring_closure_error_rad"),
        "linear_optics": ("trace_x", "trace_y", "tune_x", "tune_y", "Dx_m", "Dpx",
                          "beta_x_m", "beta_y_m", "beta_max_m", "maximum_beta_m", "periodic_closure_max"),
        "radiation": ("emittance_pm", "I1", "I2", "I3", "I4", "I5", "Jx", "Jy", "Jz",
                      "energy_spread", "sigma_delta", "momentum_compaction"),
        "chromaticity": ("chromaticity_x", "chromaticity_y", "xi_x", "xi_y"),
    }
    for section, names in fields.items():
        values = calculation.get(section, {})
        if not isinstance(values, Mapping):
            continue
        for name in names:
            if _finite(values.get(name)):
                result[name] = values[name]
    linear = calculation.get("linear_optics", {})
    if isinstance(linear, Mapping) and isinstance(linear.get("stable"), bool):
        result["stable"] = linear["stable"]
    chromaticity = calculation.get("chromaticity", {})
    if isinstance(chromaticity, Mapping):
        for source, name in (("horizontal", "chromaticity_x"), ("vertical", "chromaticity_y")):
            if _finite(chromaticity.get(source)):
                result[name] = chromaticity[source]
    result["energy_gev"] = archive["binding"]["energy_gev"]
    return result


_METRIC_LABELS = {
    "energy_gev": ("Beam energy", "GeV"), "emittance_pm": ("Natural emittance εₓ", "pm rad"),
    "circumference_m": ("Evaluated lattice length", "m"), "cell_length_m": ("Cell length", "m"),
    "predicted_ring_length_m": ("Repeated ring circumference", "m"),
    "bend_sum_rad": ("Total bend angle", "rad"),
    "predicted_ring_bend_sum_rad": ("Repeated ring bend angle", "rad"),
    "ring_closure_error_rad": ("Ring bend closure error", "rad"),
    "trace_x": ("Horizontal matrix trace", "1"), "trace_y": ("Vertical matrix trace", "1"),
    "tune_x": ("Horizontal fractional tune νₓ (evaluated scope)", "1"),
    "tune_y": ("Vertical fractional tune νᵧ (evaluated scope)", "1"),
    "Dx_m": ("Boundary dispersion Dₓ", "m"), "Dpx": ("Boundary dispersion slope D′ₓ", "1"),
    "beta_x_m": ("Boundary βₓ", "m"), "beta_y_m": ("Boundary βᵧ", "m"),
    "beta_max_m": ("Maximum sampled β", "m"), "maximum_beta_m": ("Maximum sampled β", "m"),
    "periodic_closure_max": ("Maximum periodic closure residual (mixed components)", "mixed"),
    "I1": ("Radiation integral I₁", "m"), "I2": ("Radiation integral I₂", "m⁻¹"),
    "I3": ("Radiation integral I₃", "m⁻²"), "I4": ("Radiation integral I₄", "m⁻¹"),
    "I5": ("Radiation integral I₅", "m⁻¹"),
    "Jx": ("Horizontal damping partition Jₓ", "1"), "Jy": ("Vertical damping partition Jᵧ", "1"),
    "Jz": ("Longitudinal damping partition J𝓏", "1"),
    "energy_spread": ("Relative energy spread σδ", "1"), "sigma_delta": ("Relative momentum spread σδ", "1"),
    "momentum_compaction": ("Momentum compaction αc", "1"),
    "chromaticity_x": ("Horizontal chromaticity ξₓ", "1"), "xi_x": ("Horizontal chromaticity ξₓ", "1"),
    "chromaticity_y": ("Vertical chromaticity ξᵧ", "1"), "xi_y": ("Vertical chromaticity ξᵧ", "1"),
    "stable": ("Linear stability", "—"),
}


def _verified(archive: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _canonical_metrics(archive)
    binding = archive["binding"]
    scope = archive["calculation"].get("evaluation_scope", binding["evaluation_scope"])
    return {"status": "verified", "passed": True, "issues": [],
            "next_action": None, "receipt": dict(receipt),
            "candidate_lattice_id": archive["binding"]["lattice_id"],
            "binding": dict(archive["binding"]), "canonical_metrics": metrics,
            "table": {"title": "Exact archived candidate — verified JuTrack metrics",
                      "kind": "candidate_evaluation", "lattice_id": archive["binding"]["lattice_id"],
                      "description": (f"Candidate {binding['lattice_id']}; scope: {scope}; "
                                      f"effective repetitions: {binding['effective_periods']}. "
                                      "Snapshot provenance is verified; scientific acceptance is separate."),
                      "rows": [{"Quantity": _METRIC_LABELS[key][0], "Value": value,
                                "Unit": _METRIC_LABELS[key][1]} for key, value in metrics.items()]}}


def compare_declared_candidate_metrics(data: Mapping[str, Any], verification: Mapping[str, Any]) -> dict[str, Any]:
    """Find stale direct candidate scalars, never infer meaning of comparisons.

    Only exact top-level API metric keys are checked. Nested reports and names
    such as parent_emittance_pm remain independent comparison evidence.
    A 0.05% relative tolerance admits ordinary 3–4 significant-figure display
    rounding for physics results. Geometry lengths instead use an absolute
    1 nm arithmetic tolerance and no relative tolerance: design targets must
    not be substituted for the saved lattice length. These checks apply only
    to the human summary comparison;
    the archived evaluation/snapshot hashes remain exact and no physics gate
    is relaxed.
    """
    conflicts = []
    metrics = verification.get("canonical_metrics", {})
    if verification.get("passed") is True and isinstance(metrics, Mapping):
        for key, measured in metrics.items():
            if key not in data:
                continue
            declared = json_safe(data[key])
            geometry = key in _GEOMETRY_LENGTH_KEYS
            same = (isinstance(declared, bool) and declared is measured) if isinstance(measured, bool) else (
                _finite(declared) and math.isclose(float(declared), float(measured),
                                                 rel_tol=0.0 if geometry else 5e-4,
                                                 abs_tol=_GEOMETRY_ABSOLUTE_TOLERANCE_M if geometry else 1e-12))
            if not same:
                conflicts.append({"metric": key, "declared": json_safe(declared), "verified": measured})
    return {"passed": not conflicts, "conflicts": conflicts,
            "summary_relative_tolerance": 5e-4, "summary_absolute_tolerance": 1e-12,
            "geometry_absolute_tolerance_m": _GEOMETRY_ABSOLUTE_TOLERANCE_M,
            "geometry_relative_tolerance": 0.0,
            "next_action": (
                "Correct the report to use the selected evaluation's canonical metrics. "
                "Label parent/baseline comparisons separately. The candidate is already evaluated; "
                "do not rerun matching or optimization merely to repair this report."
                if conflicts else None)}


def verify_candidate_evaluation(candidate: Any, evaluation: Any, output: Path | str) -> dict[str, Any]:
    """Verify an explicitly selected runtime capture and archive it once.

    Missing, evicted, altered, cross-run or mismatched evidence is a recoverable
    unverified checkpoint, not an execution exception or a physics rejection.
    Storage may be a child directory of the current managed run; returned paths
    remain relative to that run's root for independent parent revalidation.
    """
    unverified = partial(_unverified, candidate=candidate)
    if evaluation is None:
        return unverified("No exact-candidate evaluation was selected for the archived lattice.")
    try:
        supplied = evaluation.as_dict() if hasattr(evaluation, "as_dict") else evaluation
        if not isinstance(supplied, Mapping):
            return unverified("candidate_evaluation must be the returned JuTrack evaluation result.")
        descriptor = supplied.get("evaluation_receipt")
        if not isinstance(descriptor, Mapping):
            return unverified("The supplied evaluation has no runtime-captured snapshot receipt.")
        with _lock:
            capture = copy.deepcopy(_captured.get(str(descriptor.get("evaluation_id", ""))))
        if capture is None:
            return unverified("The selected evaluation is not retained in this process; recover its verified archive before reporting.")
        output = Path(output).resolve()
        if capture["scope"] != _scope():
            return unverified("The evaluation belongs to a different managed research run.")
        archive_root = (Path(capture["scope"]) if capture["scope"] else
                        _evidence_directory.get() or output).resolve()
        try:
            output.relative_to(archive_root)
        except ValueError:
            return unverified("The evaluation storage directory leaves the current managed research run.")
        if dict(descriptor) != capture["descriptor"]:
            return unverified("The runtime evaluation descriptor was changed.")
        archive = capture["archive"]
        submitted = copy.deepcopy(json_safe(dict(supplied)))
        submitted.pop("evaluation_receipt", None)
        if _digest(submitted) != _digest(archive["calculation"]):
            return unverified("The supplied evaluation values were changed after JuTrack evaluated them.")
        actual = _candidate(candidate)
        if _binding(actual, archive["binding"]["effective_periods"]) != archive["binding"]:
            return unverified("The archived lattice is not the exact snapshot used for the selected JuTrack metrics.")
        text = canonical_json(archive) + "\n"
        path = (output / "candidate_evaluations" / f"{archive['evaluation_id']}.json").resolve()
        path.relative_to(output)
        relative = path.relative_to(archive_root).as_posix()
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                return unverified("An existing immutable candidate evaluation archive has different content.")
        else:
            atomic_write_text(path, text)
        receipt = {**capture["descriptor"], "path": relative,
                   "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        return _verified(archive, receipt)
    except (TypeError, ValueError, KeyError, OSError) as exc:
        return unverified(f"Candidate evaluation could not be bound: {type(exc).__name__}: {exc}")


def validate_saved_candidate_evaluation(candidate: Any, receipt: Any, output: Path | str) -> dict[str, Any]:
    """Recheck saved bytes and candidate identity in the parent validation process."""
    unverified = partial(_unverified, candidate=candidate)
    if not isinstance(receipt, Mapping):
        return unverified("The archived candidate has no selected numerical snapshot receipt.")
    try:
        output = Path(output).resolve()
        relative = Path(str(receipt.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            return unverified("The candidate evaluation path leaves this program's output directory.")
        path = (output / relative).resolve()
        path.relative_to(output)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt.get("sha256"):
            return unverified("The saved candidate evaluation failed its SHA-256 check.")
        archive = json.loads(raw.decode("utf-8"))
        if not isinstance(archive, Mapping):
            return unverified("The saved candidate evaluation must be a JSON object.")
        if (archive.get("schema") != SCHEMA or receipt.get("schema") != SCHEMA
                or archive.get("evaluation_id") != receipt.get("evaluation_id")
                or _digest(archive) != receipt.get("content_sha256")):
            return unverified("The candidate evaluation archive has an invalid identity.")
        actual = _candidate(candidate)
        archived = _candidate(archive["candidate"])
        binding = _binding(actual, archive["binding"]["effective_periods"])
        if (binding != archive["binding"]
                or _digest(archived.as_dict()) != binding["candidate_sha256"]
                or receipt.get("candidate_lattice_id") != actual.lattice_id):
            return unverified("Saved evaluation metrics and the archived candidate refer to different snapshots.")
        return _verified(archive, receipt)
    except (TypeError, ValueError, KeyError, OSError) as exc:
        return unverified(f"Saved candidate evaluation could not be verified: {type(exc).__name__}: {exc}")


def restore_candidate_evaluation(candidate: Any, receipt: Mapping[str, Any], *,
                                 output_directory: Path | str | None = None,
                                 source_output_directory: Path | str | None = None) -> dict[str, Any]:
    """Restore an exact archived JuTrack evaluation without numerical execution.

    Supply the saved candidate-evaluation *receipt*, not an unverified result
    dictionary. Hashes, exact candidate rows/energy/partition and current-run
    path containment are checked before restoring the original process capture.
    The returned dictionary has the original evaluation_id/content hash and
    can be passed to ResearchRun.checkpoint_candidate or finish. It is not a
    new calculation and does not upgrade physics or scientific acceptance.

    By default read this run only. To reuse an earlier worker's result explicitly,
    set source_output_directory to its preserved program output in this same
    project. This retains the original identity, not a new calculation. Another
    project's evidence is never accepted. Outside a managed run,
    supply the trusted archive root explicitly (or use the host evidence root).
    A child output_directory is allowed, but receipt paths remain root-relative.
    """
    root_value = _scope() or _evidence_directory.get() or output_directory
    if root_value is None or root_value == "":
        raise ValueError("Restoring candidate evidence requires a managed or explicit archive root")
    root = Path(root_value).resolve()
    if output_directory is not None:
        try:
            Path(output_directory).resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("Restoring evidence cannot leave the current managed run") from exc
    from .evidence_scope import evidence_source_root
    source_root = evidence_source_root(root, source_output_directory)
    snapshot = _candidate(candidate)
    if not isinstance(receipt, Mapping):
        raise ValueError("Restoring an evaluation requires its saved candidate-evaluation receipt")
    original_receipt = copy.deepcopy(dict(receipt))
    verified = validate_saved_candidate_evaluation(snapshot, original_receipt, source_root)
    if not verified["passed"]:
        raise ValueError("Cannot restore candidate evaluation: " + "; ".join(verified["issues"]))
    # Recheck the exact bytes read for hydration, not only a preceding path read.
    raw = (source_root / original_receipt["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != original_receipt["sha256"]:
        raise ValueError("Candidate evaluation changed while restoring its verified archive")
    archive = json.loads(raw.decode("utf-8"))
    calculation = archive["calculation"]
    if (calculation.get("engine") != "JuTrack"
            or not isinstance(calculation.get("linear_optics"), Mapping)
            or not isinstance(calculation.get("geometry"), Mapping)):
        raise ValueError("The receipt is not a high-level JuTrack lattice evaluation")
    descriptor = {key: original_receipt[key] for key in
                  ("schema", "evaluation_id", "content_sha256", "candidate_lattice_id")}
    evaluation_id = str(descriptor["evaluation_id"])
    if source_root != root:
        # Durable provenance sits outside immutable calculation values. A later
        # publication will archive the unchanged measurement under the new root.
        provenance = {"schema": "brilliance_evidence_reuse/v1", "evaluation_id": evaluation_id,
                      "source_output_directory": str(source_root), "source_receipt": original_receipt,
                      "destination_output_directory": str(root), "numerical_rerun": False}
        destination = root / "evidence_reuse" / f"{evaluation_id}.json"
        text = canonical_json(provenance) + "\n"
        if destination.exists() and destination.read_text(encoding="utf-8") != text:
            raise ValueError("An immutable evidence-reuse record already has different provenance")
        if not destination.exists():
            atomic_write_text(destination, text)
    with _lock:
        existing = _captured.get(evaluation_id)
        if existing is not None and _digest(existing["archive"]) != _digest(archive):
            raise ValueError("Restored evaluation conflicts with an existing immutable runtime capture")
        _captured[evaluation_id] = {"scope": _scope(), "archive": archive, "descriptor": descriptor,
                                   "restored_from_receipt": original_receipt, "restored_archive_root": str(source_root)}
        _captured.move_to_end(evaluation_id)
        while len(_captured) > _MAX_CAPTURED_EVALUATIONS:
            _captured.popitem(last=False)
    return {**copy.deepcopy(calculation), "evaluation_receipt": copy.deepcopy(descriptor)}


@contextmanager
def numerical_evidence_directory(directory: Path | str | None):
    """Set the host-owned archive root while independently checking an expert claim."""
    token = _evidence_directory.set(Path(directory).resolve() if directory is not None else None)
    try:
        yield
    finally:
        _evidence_directory.reset(token)


def _measurement_output(directory: Path | str | None) -> Path:
    hosted = _evidence_directory.get()
    configured = hosted or _scope() or directory
    if not configured:
        raise ValueError("numerical evidence requires the managed run's output directory")
    root = Path(configured).resolve()
    if directory is not None:
        try:
            Path(directory).resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("numerical evidence cannot override the host's current run directory") from exc
    return root


def record_numerical_evidence(candidate: Any, calculation: Mapping[str, Any], *,
                              output_directory: Path | str | None = None) -> dict[str, Any]:
    """Archive final diagnostic measurements using the existing snapshot receipt.

    Measurement adapters call this after computing raw values; it neither decides
    scientific acceptance nor replaces the main candidate evaluation. It guards
    accidental stale/changed evidence, not hostile code in the same Python process.
    An explicit output directory may be a child of the managed run. Receipt paths
    are always relative to the managed root, not the child storage directory.
    """
    output = _measurement_output(output_directory)
    storage = Path(output_directory).resolve() if output_directory is not None else output
    payload = copy.deepcopy(json_safe(dict(calculation)))
    payload.pop("evaluation_receipt", None)
    payload["evaluation_receipt"] = record_evaluation(candidate, payload)
    verified = verify_candidate_evaluation(candidate, payload, storage)
    if not verified["passed"]:
        raise ValueError("; ".join(verified["issues"]))
    return verified["receipt"]


def read_numerical_evidence(candidate: Any, receipt: Mapping[str, Any], *,
                            output_directory: Path | str | None = None) -> dict[str, Any]:
    """Read raw measurements only after replaying hash, scope and exact-row checks.

    The parent supplies the archive root; a model-written report cannot redirect
    this reader to a different run or another project.
    """
    output = _measurement_output(output_directory)
    verified = validate_saved_candidate_evaluation(candidate, receipt, output)
    if not verified["passed"]:
        raise ValueError("; ".join(verified["issues"]))
    path = output / receipt["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("numerical evidence changed while reading the verified archive")
    return json.loads(raw.decode("utf-8"))["calculation"]
