"""Stored-cell contracts use synthetic rows only; no JuTrack or design search."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict

import pytest

from brilliance import accelerator_api as api
from brilliance.lattice import (
    LatticeInputError,
    LatticeIR,
    LatticeScopeError,
    lattice_cell_partition,
)


@pytest.fixture(autouse=True)
def no_numerical_runtime(monkeypatch):
    def fail_if_started(*args, **kwargs):
        pytest.fail("A lattice representation check must not start JuTrack")

    monkeypatch.setattr(api, "load_pyjutrack", fail_if_started)


def rows(count=35):
    return [api.drift(f"D{i:03d}", 0.1 + i * 0.001) for i in range(count)]


def legacy(*, count=35, periodicity=32, metadata=None, elements=None):
    """The lower-level IR remains able to inspect intact historical artifacts."""
    return LatticeIR.create(
        title="Synthetic archived candidate", energy_gev=3.5,
        periodicity=periodicity, elements=rows(count) if elements is None else elements,
        source="synthetic_scope_test.py", metadata=metadata,
    )


def test_unexpanded_odd_cell_is_rejected_at_public_construction():
    with pytest.raises(LatticeScopeError) as error:
        api.lattice(title="one cell, misdeclared", energy_gev=3.5,
                    periodicity=32, elements=rows(), source="synthetic.py")
    assert isinstance(error.value, LatticeInputError)
    message = str(error.value)
    assert "35" in message and "32" in message
    assert "api.periodic_cell" in message
    assert "api.cell_ring" in message


def test_periodic_cell_keeps_all_35_rows_with_32_intended_repetitions():
    candidate = api.periodic_cell(title="one cell", energy_gev=3.5, elements=rows(),
                                  intended_repetitions=32, source="synthetic.py")
    assert candidate.periodicity == 1
    assert lattice_cell_partition(candidate) == (35, 1)
    assert candidate.metadata["intended_repetitions"] == 32
    assert len(api.one_period_elements(candidate)) == 35


def test_integer_stored_partition_remains_valid_without_guessing_scope():
    supplied = rows(64)
    candidate = api.lattice(title="explicit sequence", energy_gev=3.5,
                            periodicity=32, elements=supplied, source="synthetic.py")
    assert lattice_cell_partition(candidate) == (2, 32)
    assert [r["name"] for r in api.one_period_elements(candidate)] == ["D000", "D001"]
    assert len(candidate.elements) == 64


def test_bend_angles_and_cell_like_names_do_not_silently_repair_scope():
    supplied = rows()
    supplied[0] = api.sector_bend("CELL:B", 0.8, 2.0 * math.pi / 32)
    candidate = legacy(elements=supplied)
    original = copy.deepcopy(candidate.as_dict())
    with pytest.raises(LatticeScopeError):
        api.one_period_elements(candidate)
    assert candidate.as_dict() == original


@pytest.mark.parametrize("with_identity", [True, False])
def test_malformed_legacy_stays_readable_and_unchanged(tmp_path, with_identity):
    candidate = legacy()
    payload = candidate.as_dict()
    if not with_identity:
        payload.pop("lattice_id")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    original_bytes = path.read_bytes()
    loaded = api.load_lattice(path)
    assert loaded.lattice_id == candidate.lattice_id
    assert loaded.as_dict() == candidate.as_dict()
    with pytest.raises(LatticeScopeError):
        api.one_period_elements(path)
    assert path.read_bytes() == original_bytes


def test_saving_invalid_scope_does_not_overwrite_existing_artifact(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text("preserve existing evidence", encoding="utf-8")
    with pytest.raises(LatticeScopeError):
        api.save_lattice(path, legacy())
    assert path.read_text(encoding="utf-8") == "preserve existing evidence"


@pytest.mark.parametrize("operation", [
    "build_jutrack_lattice", "build_jutrack_editor", "evaluate",
    "evaluate_candidate", "evaluate_periodic_cell",
])
def test_invalid_scope_fails_before_any_runtime_start(operation):
    with pytest.raises(LatticeScopeError):
        getattr(api, operation)(legacy())


@pytest.mark.parametrize("metadata", [
    {"cell_element_count": 1},
    {"cell_element_count": 4},
    {"repetitions": 1, "cell_element_count": 6},
    {"evaluation_scope": "periodic_cell", "repetitions": 2, "cell_element_count": 6},
    {"evaluation_scope": "full_ring", "ring_complete": False, "cell_element_count": 3},
])
def test_contradictory_partition_never_silently_slices_rows(metadata):
    candidate = legacy(count=6, periodicity=2, metadata=metadata)
    with pytest.raises(LatticeScopeError):
        api.one_period_elements(candidate)


@pytest.mark.parametrize("field", ["cell_element_count", "repetitions", "intended_repetitions"])
@pytest.mark.parametrize("value", [0, -1, 1.25, True, False, float("nan"), float("inf"), "wrong", [], {}])
def test_invalid_numeric_metadata_is_a_typed_input_error(field, value):
    candidate = legacy(count=6, periodicity=2, metadata={field: value})
    with pytest.raises(LatticeScopeError):
        api.one_period_elements(candidate)


def test_periodic_scope_cannot_hide_stale_element_count():
    candidate = legacy(count=35, periodicity=1, metadata={
        "evaluation_scope": "periodic_cell", "cell_element_count": 34,
        "intended_repetitions": 32,
    })
    with pytest.raises(LatticeScopeError):
        api.one_period_elements(candidate)


def test_ring_to_cell_to_ring_owns_structural_metadata():
    original_rows = rows(3)
    ring = api.cell_ring(title="ring", energy_gev=3.5, cell=original_rows,
                         periods=4, source="synthetic.py", metadata={"physics_note": "keep"})
    stale = dict(ring.metadata, intended_repetitions=99)
    cell = api.periodic_cell(title="cell", energy_gev=3.5, elements=original_rows,
                             source="synthetic.py", metadata=stale, intended_repetitions=4)
    assert cell.periodicity == 1
    assert cell.metadata["repetitions"] == 1
    assert cell.metadata["cell_element_count"] == 3
    assert cell.metadata["evaluation_scope"] == "periodic_cell"
    assert cell.metadata["ring_complete"] is False
    assert cell.metadata["intended_repetitions"] == 4
    assert cell.metadata["physics_note"] == "keep"
    assert stale["repetitions"] == 4
    expanded = api.cell_ring(title="expanded", energy_gev=3.5, cell=original_rows,
                             periods=4, source="synthetic.py", metadata=cell.metadata)
    assert expanded.metadata["evaluation_scope"] == "full_ring"
    assert expanded.metadata["ring_complete"] is True
    assert "intended_repetitions" not in expanded.metadata
    assert lattice_cell_partition(expanded) == (3, 4)
    assert len(api.one_period_elements(expanded)) == 3


def test_stale_future_repetitions_are_not_inherited_from_a_ring():
    cell = api.periodic_cell(title="cell", energy_gev=3.5, elements=rows(3),
                             source="synthetic.py", metadata={
                                 "evaluation_scope": "full_ring", "repetitions": 32,
                                 "cell_element_count": 99, "intended_repetitions": 99,
                             })
    assert "intended_repetitions" not in cell.metadata
    assert lattice_cell_partition(cell) == (3, 1)


def test_editing_an_existing_cell_can_preserve_its_intended_repetitions():
    parent = api.periodic_cell(title="parent", energy_gev=3.5, elements=rows(3),
                               source="synthetic.py", intended_repetitions=32)
    revised = api.periodic_cell(title="revised", energy_gev=3.5, elements=rows(4),
                                source="synthetic.py", metadata=parent.metadata)
    assert revised.metadata["intended_repetitions"] == 32
    assert lattice_cell_partition(revised) == (4, 1)


def test_nested_source_names_are_stripped_at_most_once():
    nested_rows = api.repeat_cell(rows(2), 3)
    cell = api.periodic_cell(title="assembled cell", energy_gev=3.5,
                             elements=nested_rows, source="synthetic.py",
                             intended_repetitions=4)
    cell_rows = api.one_period_elements(cell)
    assert cell_rows == [asdict(element) for element in cell.elements]
    ring = api.cell_ring(title="outer ring", energy_gev=3.5, cell=cell_rows,
                         periods=4, source="synthetic.py", metadata=cell.metadata)
    extracted = api.one_period_elements(ring)
    assert [r["name"] for r in extracted] == [r["name"] for r in nested_rows]
    assert len(set(r["name"] for r in extracted)) == 6
    assert extracted[0]["name"] == "P001:D000"
    assert extracted[-1]["name"] == "P003:D001"


def test_replace_single_cell_updates_count_without_touching_parent():
    parent = api.periodic_cell(title="parent", energy_gev=3.5, elements=rows(3),
                               source="synthetic.py", intended_repetitions=32)
    original = copy.deepcopy(parent.as_dict())
    revised = api.replace_lattice_elements(parent, rows(4), title="edited cell",
                                           source="synthetic_edit.py")
    assert lattice_cell_partition(revised) == (4, 1)
    assert len(api.one_period_elements(revised)) == 4
    assert revised.metadata["intended_repetitions"] == 32
    assert revised.parent_lattice_id == parent.lattice_id
    assert parent.as_dict() == original


def test_replace_expanded_ring_requires_explicit_updated_partition():
    ring = api.cell_ring(title="ring", energy_gev=3.5, cell=rows(2),
                         periods=3, source="synthetic.py")
    expanded_changed = api.repeat_cell(rows(3), 3)
    with pytest.raises(LatticeScopeError):
        api.replace_lattice_elements(ring, expanded_changed, title="new topology",
                                     source="synthetic_edit.py")
    revised = api.replace_lattice_elements(
        ring, expanded_changed, title="declared new topology", source="synthetic_edit.py",
        metadata={"cell_element_count": 3},
    )
    assert lattice_cell_partition(revised) == (3, 3)
    assert len(api.one_period_elements(revised)) == 3


def test_scalar_edit_preserves_expanded_ring_partition():
    ring = api.cell_ring(title="ring", energy_gev=3.5, cell=rows(2),
                         periods=3, source="synthetic.py")
    changed = [asdict(element) for element in ring.elements]
    for element in changed:
        element["length_m"] *= 1.1
    revised = api.replace_lattice_elements(ring, changed, title="length edit",
                                           source="synthetic_edit.py")
    assert lattice_cell_partition(revised) == (2, 3)
    assert [row["name"] for row in api.one_period_elements(revised)] == ["D000", "D001"]
