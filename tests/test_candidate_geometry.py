"""Exact row-summed geometry is available before any simulator succeeds.

Synthetic rows test application provenance/reporting, not accelerator designs.
"""
from __future__ import annotations

import copy
import math

import pytest

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import (
    compare_declared_candidate_metrics,
    compare_declared_geometry,
    geometry_snapshot,
    unbound_geometry_tables,
)


def cell(*, periods=32):
    # Regression shape: two incorrectly allocated 2.37 m terminals instead of
    # intended 2.4 m sections. Targets are deliberately not the row sums.
    return api.periodic_cell(
        title="Synthetic geometry arithmetic", energy_gev=3.5,
        elements=[api.drift("ID:L", 2.5), api.drift("T:L", 2.37),
                  api.drift("CORE", 15.2), api.drift("T:R", 2.37), api.drift("ID:R", 2.5)],
        source="synthetic geometry regression", intended_repetitions=periods,
        metadata={"design_constraints": [{"quantity": "circumference_m", "target": 800.0}]},
    )


def ring():
    return api.cell_ring(title="Synthetic expanded ring", energy_gev=3.5,
                         cell=api.one_period_elements(cell()), periods=32,
                         source="synthetic geometry regression",
                         design_constraints=[{"quantity": "circumference_m", "target": 800.0}])


def test_geometry_is_derived_from_saved_rows_without_jutrack(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Geometry reporting must not initialize Julia or run a simulator")

    monkeypatch.setattr(api, "load_pyjutrack", forbidden)
    monkeypatch.setattr(api, "evaluate", forbidden)
    candidate = ring()
    original = copy.deepcopy(candidate.as_dict())
    report = geometry_snapshot(candidate)
    metrics = report["canonical_metrics"]
    assert metrics["stored_length_m"] == math.fsum(el.length_m for el in candidate.elements)
    assert metrics["cell_length_m"] == pytest.approx(24.94)
    assert metrics["ring_circumference_m"] == pytest.approx(798.08)
    assert metrics["stored_periods"] == 32
    assert metrics["projected_ring_circumference_m"] is None
    assert report["binding"]["candidate_sha256"]
    assert report["binding"]["physical_fingerprint"] == candidate.physical_fingerprint
    assert report["table"]["lattice_id"] == candidate.lattice_id
    assert report["table"]["kind"] == "candidate_geometry"
    assert "not a physics" in report["table"]["description"]
    assert candidate.as_dict() == original
    assert all(row["Value"] != 800.0 for row in report["table"]["rows"])


def test_periodic_reference_ring_never_looks_like_a_stored_requested_ring():
    unit = api.periodic_cell(title="Synthetic reference unit", energy_gev=3.5,
                             elements=[api.drift("D", 3.04)], source="synthetic",
                             intended_repetitions=192)
    report = geometry_snapshot(unit)
    assert report["evaluation_scope"] == "periodic_cell"
    metrics = report["canonical_metrics"]
    assert metrics == {"stored_length_m": 3.04, "cell_length_m": 3.04,
                       "ring_circumference_m": None,
                       "projected_ring_circumference_m": 3.04 * 192,
                       "stored_periods": 1, "intended_repetitions": 192}
    assert any("not a stored ring" in row["Quantity"] for row in report["table"]["rows"])
    assert not compare_declared_geometry({"ring_circumference_m": 3.04 * 192}, report)["passed"]
    assert compare_declared_geometry({"projected_ring_circumference_m": 3.04 * 192}, report)["passed"]


def test_undeclared_ring_or_repetition_is_not_guessed():
    unit = cell(periods=None)
    metrics = geometry_snapshot(unit)["canonical_metrics"]
    assert metrics["projected_ring_circumference_m"] is None
    assert metrics["ring_circumference_m"] is None
    unspecified = api.lattice(title="Line with no ring scope", energy_gev=3.5, periodicity=1,
                               elements=[api.drift("D", 800.0)], source="synthetic")
    report = geometry_snapshot(unspecified)
    assert report["evaluation_scope"] == "undeclared"
    assert report["binding"]["evaluation_scope"] == "undeclared"
    assert report["canonical_metrics"]["stored_length_m"] == 800.0
    assert report["canonical_metrics"]["ring_circumference_m"] is None


def test_nonuniform_stored_cells_use_the_actual_full_sum_not_first_cell_times_periods():
    candidate = api.lattice(title="Explicit nonuniform ring", energy_gev=3.5, periodicity=2,
                            elements=[api.drift("D1", 1.0), api.drift("D2", 1.5)], source="synthetic",
                            metadata={"evaluation_scope": "full_ring", "repetitions": 2,
                                      "ring_complete": True, "cell_element_count": 1})
    report = geometry_snapshot(candidate)
    assert not report["equal_stored_cell_lengths"]
    assert report["stored_cell_lengths_m"] == [1.0, 1.5]
    assert report["canonical_metrics"]["cell_length_m"] is None
    assert report["canonical_metrics"]["ring_circumference_m"] == 2.5


def test_geometry_snapshot_rejects_mutated_identity_instead_of_binding_old_label():
    candidate = cell()
    candidate.metadata["intended_repetitions"] = 12
    with pytest.raises(ValueError, match="identity mismatch"):
        geometry_snapshot(candidate)


def test_geometry_comparison_catches_direct_and_explicit_nested_current_claims():
    report = geometry_snapshot(ring())
    check = compare_declared_geometry({"ring_circumference_m": 800.0,
                                      "geometry": {"circumference_m": 800.0},
                                      "current_candidate": {"geometry": {"cell_length_m": 25.0}}}, report)
    assert not check["passed"]
    assert {row["path"] for row in check["conflicts"]} == {
        "data.ring_circumference_m", "data.geometry.circumference_m",
        "data.current_candidate.geometry.cell_length_m"}
    assert {row["verified"] for row in check["conflicts"]} == {
        report["canonical_metrics"]["ring_circumference_m"], report["canonical_metrics"]["cell_length_m"]}
    assert "Do not rerun matching" in check["next_action"]


def test_targets_parent_and_reference_comparisons_are_not_false_current_claims():
    report = geometry_snapshot(ring())
    data = {"contract": {"circumference_m": 800.0, "cell_length_m": 25.0},
            "targets": {"ring_circumference_m": 800.0},
            "parent": {"geometry": {"circumference_m": 810.0}},
            "unit": {"cell_length_m": 3.04},
            "current_candidate": {"lattice_id": "different-candidate", "cell_length_m": 3.04}}
    untyped = [{"title": "Reference comparisons", "rows": [
        {"quantity": "circumference", "target": 800.0, "parent": 810.0,
         "unit_reference": 583.68, "cell_scope_32_repeats": 800.0}]}]
    check = compare_declared_geometry(data, report, tables=untyped)
    assert check["passed"]
    assert check["checked_fields"] == []


def test_matching_candidate_typed_tables_checked_without_guessing_comparison_columns():
    report = geometry_snapshot(ring())
    table = {"kind": "candidate_metrics", "lattice_id": report["candidate_lattice_id"], "rows": [
        {"metric": "ring_circumference_m", "value": 800.0},
        {"cell_length_m": 25.0, "parent_cell_length_m": 27.0},
        {"quantity": "circumference", "target": 800.0, "parent": 900.0},
        {"lattice_id": "other-row-candidate", "metric": "stored_length_m", "value": 12.0},
    ]}
    check = compare_declared_geometry({}, report, tables=[table])
    assert not check["passed"]
    assert {item["metric"] for item in check["conflicts"]} == {"ring_circumference_m", "cell_length_m"}
    other = {**table, "lattice_id": "another-lattice"}
    assert compare_declared_geometry({}, report, tables=[other])["passed"]
    assert compare_declared_geometry({}, report, tables=[report["table"]])["passed"]


def test_length_comparison_does_not_allow_physics_display_rounding_tolerance():
    snapshot = geometry_snapshot(ring())
    measured = snapshot["canonical_metrics"]["ring_circumference_m"]
    assert compare_declared_geometry({"ring_circumference_m": measured + 5e-10}, snapshot)["passed"]
    # 0.01 m used to fit the general 0.05% relative tolerance near 800 m.
    assert not compare_declared_geometry({"ring_circumference_m": measured + 0.01}, snapshot)["passed"]
    verified = {"passed": True, "canonical_metrics": {"circumference_m": measured, "emittance_pm": 37.870082}}
    assert not compare_declared_candidate_metrics({"circumference_m": measured + 0.01}, verified)["passed"]
    assert compare_declared_candidate_metrics({"emittance_pm": 37.87}, verified)["passed"]
    assert snapshot["canonical_metrics"]["ring_circumference_m"] == measured


@pytest.mark.parametrize("wrong", [True, "798.08", float("nan"), float("inf")])
def test_nonphysical_values_cannot_be_confirmed_as_exact_geometry(wrong):
    assert not compare_declared_geometry({"ring_circumference_m": wrong}, geometry_snapshot(ring()))["passed"]


def test_serialized_candidate_has_the_same_snapshot():
    candidate = ring()
    assert geometry_snapshot(candidate.as_dict()) == geometry_snapshot(candidate)


def test_archived_comparison_shape_is_unverified_not_a_numeric_or_physics_conflict():
    # Exact reporting pattern of the failing project: the author placed a
    # target in an untyped comparison column and read a nonexistent result key.
    tables = [{"title": "Cell and ring comparison", "rows": [
        {"quantity": "circumference", "unit": "m", "cell_scope_32_repeats": 800.0, "ring": None},
        {"quantity": "natural emittance", "unit": "pm rad", "cell_scope_32_repeats": 880.48, "ring": 880.48},
    ]}]
    original = copy.deepcopy(tables)
    report = compare_declared_geometry({}, geometry_snapshot(ring()), tables=tables)
    assert report["passed"]
    assert report["conflicts"] == []
    assert report["unverified_claims"] == [
        {"table_index": 0, "title": "Cell and ring comparison", "quantities": ["circumference"],
         "reason": "missing_candidate_geometry_lineage"}]
    assert tables == original  # caller quarantines; original narrative is preserved


@pytest.mark.parametrize("kind", ["contract", "targets", "requirements", "hypothesis", "reference"])
def test_unbound_geometry_detector_does_not_reclassify_explicit_nonmeasurement_tables(kind):
    table = {"kind": kind, "rows": [{"Quantity": "Ring circumference", "Value": 800.0, "Unit": "m"}]}
    assert not unbound_geometry_tables([table], "selected")


def test_unbound_geometry_requires_exact_label_and_physical_unit_without_parsing_prose():
    tables = [
        {"title": "Unrelated", "rows": [{"quantity": "Natural emittance", "unit": "m", "value": 1e-10}]},
        {"title": "Prose", "rows": [{"text": "The circumference is 800 m"}]},
        {"title": "Question", "rows": [{"quantity": "What should the cell length be?", "unit": "m"}]},
        {"title": "Length", "rows": [{"metric": "cell_length_m", "unit": "cm", "value": 2500.0}]},
        {"title": "No units", "rows": [{"quantity": "cell length", "value": 25.0}]},
    ]
    assert [entry["table_index"] for entry in unbound_geometry_tables(tables, "selected")] == [3]


def test_explicit_other_candidate_table_and_properly_bound_geometry_are_not_quarantined():
    report = geometry_snapshot(ring())
    other = {"lattice_id": "different-candidate", "rows": [
        {"quantity": "circumference", "unit": "m", "value": 810.0}]}
    assert unbound_geometry_tables([other, report["table"]], report["candidate_lattice_id"]) == []
    # Merely adding an id does not turn an untyped table into verified metrics.
    untyped = {**other, "lattice_id": report["candidate_lattice_id"]}
    assert unbound_geometry_tables([untyped], report["candidate_lattice_id"])
