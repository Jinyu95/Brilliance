"""Per-bend scope conversion is explicit, exact and independent of design tools."""
import copy
import math
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from brilliance import result_presentation
from brilliance.jutrack_service import JuTrackService
from brilliance.lattice import LatticeScopeError


def cell(*, periods=4):
    return api.periodic_cell(
        title="Program-authored periodic cell", energy_gev=3.5, source="unit test",
        intended_repetitions=periods,
        elements=[api.drift("ID", 0.7234567890123456),
                  api.sector_bend("LEFT:B", 0.7, math.pi / 4),
                  api.quadrupole("Q", 0.35, 3.4261995199999995),
                  api.sector_bend("RIGHT:B", 0.8, math.pi / 4)],
    )


@pytest.fixture
def native(monkeypatch):
    """Algebraic native double exercises real row grouping, not beam physics."""
    built, retained = [], []

    def build(self, candidate):
        built.append(candidate)
        return candidate.elements

    def elrad(element, entrance, exit):
        i2 = element.angle_rad ** 2 / element.length_m
        return [0.0, i2, i2 / 10, i2 / 20, i2 / 30, 0.0, 0.0]

    def radiation(jt, candidate, *, ring):
        sums = [sum(elrad(item, None, None)[j] for item in ring
                    if item.kind in {"SBEND", "RBEND"}) for j in range(6)]
        return {f"I{j + 1}": value for j, value in enumerate(sums)}

    jt = SimpleNamespace(
        twissring=lambda ring, **kwargs: [object() for _ in ring],
        _jl=SimpleNamespace(elrad=elrad),
    )
    monkeypatch.setattr(JuTrackService, "build_lattice", build)
    monkeypatch.setattr(api, "_radiation_evaluation", radiation)
    monkeypatch.setattr(result_presentation, "remember_candidate_evidence",
                        lambda candidate, **kwargs: retained.append((candidate, kwargs)))
    monkeypatch.setattr(api, "load_pyjutrack", lambda: pytest.fail("Use the supplied native module"))
    return jt, built, retained


def test_declared_cell_and_explicit_ring_have_identical_bend_evidence(native):
    jt, built, retained = native
    candidate = cell()
    before = copy.deepcopy(candidate.as_dict())
    ring = api.cell_ring(title="Explicit ring", energy_gev=candidate.energy_gev,
                         cell=candidate.elements, periods=4, source="unit test")
    from_cell = api.bend_radiation_contributions(candidate, _loaded_jutrack=jt)
    from_ring = api.bend_radiation_contributions(ring, _loaded_jutrack=jt)

    for key in ("rows", "totals", "ringpara", "consistency", "method"):
        assert from_cell[key] == from_ring[key]
    assert from_cell["consistency"]["passed"] is True
    assert [row["bend"] for row in from_cell["rows"]] == ["LEFT:B", "RIGHT:B"]
    assert [row["instances"] for row in from_cell["rows"]] == [4, 4]
    assert from_cell["rows"][0]["source_periods"] == [1, 2, 3, 4]
    assert candidate.as_dict() == before
    assert from_cell["lattice_id"] == candidate.lattice_id
    assert from_cell["evaluated_lattice_id"] == built[0].lattice_id
    assert built[0].parent_lattice_id == candidate.lattice_id
    assert from_cell["evaluation_scope"] == "explicit_full_ring_from_declared_periodic_cell"
    assert from_cell["periods"] == 4
    assert from_cell["cell_element_count"] == len(candidate.elements)
    assert from_cell["ring_element_count"] == 4 * len(candidate.elements)
    assert from_cell["ring_bend_sum_rad"] == pytest.approx(2 * math.pi)
    assert from_cell["ring_length_m"] == pytest.approx(4 * candidate.total_length_m)
    assert retained[0][0] is candidate
    assert retained[0][1]["bends"]["lattice_id"] == candidate.lattice_id
    assert from_ring["lattice_id"] == ring.lattice_id
    assert "evaluated_lattice_id" not in from_ring
    assert built[1] is ring
    assert built[0].physical_fingerprint == ring.physical_fingerprint
    for i, element in enumerate(built[0].elements):
        source = candidate.elements[i % len(candidate.elements)]
        for field in ("kind", "length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4"):
            assert getattr(element, field) == getattr(source, field)


def test_saved_cell_path_uses_declared_repetitions(native, tmp_path):
    jt, built, _ = native
    candidate = cell()
    path = api.save_lattice(tmp_path / "candidate.json", candidate)
    result = api.bend_radiation_contributions(path, _loaded_jutrack=jt)
    assert result["lattice_id"] == candidate.lattice_id
    assert len(built[0].elements) == 16


def test_cell_missing_repetition_is_not_guessed_from_bend_sum(native):
    jt, built, _ = native
    candidate = cell(periods=None)
    with pytest.raises(LatticeScopeError, match="requires declared intended_repetitions"):
        api.bend_radiation_contributions(candidate, _loaded_jutrack=jt)
    assert built == []


@pytest.mark.parametrize("bad", [False, 0, -2, 1.5, float("nan"), float("inf"), "4"])
def test_invalid_repetition_metadata_fails_before_native_work(native, bad):
    jt, built, _ = native
    candidate = cell()
    candidate.metadata["intended_repetitions"] = bad
    with pytest.raises(LatticeScopeError, match="positive integer"):
        api.bend_radiation_contributions(candidate, _loaded_jutrack=jt)
    assert built == []


@pytest.mark.parametrize("key,value", [("cell_element_count", 2), ("repetitions", 2),
                                       ("ring_complete", True)])
def test_conflicting_stored_cell_scope_is_not_repaired(native, key, value):
    jt, built, _ = native
    candidate = cell()
    candidate.metadata[key] = value
    with pytest.raises(LatticeScopeError):
        api.bend_radiation_contributions(candidate, _loaded_jutrack=jt)
    assert built == []


@pytest.mark.parametrize("periods", [1, 3, 1000000000])
def test_nonclosing_declared_ring_is_rejected_before_expansion(native, monkeypatch, periods):
    jt, built, _ = native
    candidate = cell(periods=periods)
    monkeypatch.setattr(api, "cell_ring", lambda **kwargs: pytest.fail("Preflight closure before expansion"))
    with pytest.raises(ValueError, match="exact 2.pi bend closure"):
        api.bend_radiation_contributions(candidate, _loaded_jutrack=jt)
    assert built == []


def test_explicit_ring_closure_gate_is_unchanged(native):
    jt, built, _ = native
    candidate = cell()
    ring = api.cell_ring(title="Unclosed explicit ring", energy_gev=candidate.energy_gev,
                         cell=candidate.elements, periods=3, source="unit test")
    with pytest.raises(ValueError, match="exact 2.pi bend closure"):
        api.bend_radiation_contributions(ring, _loaded_jutrack=jt)
    assert built == []
