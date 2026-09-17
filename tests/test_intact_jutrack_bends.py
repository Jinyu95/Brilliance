"""Physical bends remain intact; interior diagnostics never supply radiation."""
from __future__ import annotations

import copy
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from brilliance import accelerator_api as api
from brilliance.jutrack_service import JuTrackService, native_element_options
from brilliance.lattice import LatticeElementIR, LatticeInputError
from brilliance.lattice_optimization import JuTrackLatticeEditor, jutrack_periodic_radiation_metrics
from brilliance.optics_sampling import JuTrackProfileSampler, UnsupportedOpticsSampling


def bend(fields=None, kind="SBEND"):
    return LatticeElementIR("B", kind, 0.8, angle_rad=2 * math.pi / 32, fields=fields or {})


def test_saved_edges_integration_and_polynomial_arrays_reach_native_constructor():
    calls = []
    jt = SimpleNamespace(SBEND=lambda *a, **kw: calls.append((a, kw)))
    fields = {"e1": .013, "e2": -.021, "NumIntSteps": 17, "fint1": .1, "fint2": .2,
              "gap": .03, "FringeBendEntrance": 1, "PolynomB": [0., .04, 0., 0.], "family": "B"}
    item = bend(fields)
    JuTrackService(jt).build_element(item)
    assert calls[0][0] == ("B", .8, item.angle_rad)
    assert calls[0][1] == {key: value for key, value in fields.items() if key != "family"}
    calls[0][1]["PolynomB"][1] = 8.0
    assert fields["PolynomB"][1] == .04


def test_explicit_quadrupole_numintsteps_not_lost_or_defaulted():
    calls = []
    jt = SimpleNamespace(KQUAD=lambda *a, **kw: calls.append(kw))
    JuTrackService(jt).build_element(LatticeElementIR("Q", "KQUAD", .2, k1_m2=2., fields={"NumIntSteps": 23}))
    assert calls == [{"k1": 2., "NumIntSteps": 23}]


@pytest.mark.parametrize("fields,kind", [({"e1": .01}, "RBEND"), ({"hgap": .03}, "SBEND"),
    ({"NumIntSteps": 0}, "SBEND"), ({"NumIntSteps": 2.5}, "SBEND"),
    ({"e1": float("nan")}, "SBEND"), ({"e1": .01, "entrance_angle": .02}, "SBEND"),
    ({"R1": [1, 2, 3]}, "SBEND")])
def test_unsupported_or_ambiguous_physics_settings_fail_before_julia(fields, kind):
    with pytest.raises(LatticeInputError):
        JuTrackService(SimpleNamespace()).build_element(bend(fields, kind))


def test_equivalent_edge_aliases_are_exact_and_conflicting_gradient_rejected():
    assert native_element_options(bend({"e1": .01, "entrance_angle": .01})) == {"e1": .01}
    item = LatticeElementIR("B", "SBEND", .8, k1_m2=.1, angle_rad=.02, fields={"PolynomB": [0., .2, 0., 0.]})
    with pytest.raises(LatticeInputError, match="conflicts"):
        JuTrackService(SimpleNamespace()).build_element(item)


def test_editor_rejects_constructor_change_before_touching_any_scalar():
    rows = [api.drift("D", .2), {**api.sector_bend("B", .8, .03), "fields": {"e1": .01}}]
    native = [SimpleNamespace(name="D", len=.2), SimpleNamespace(name="B", len=.8, angle=.03, e1=.01)]
    jt = SimpleNamespace(findelem=lambda lat, name: [i + 1 for i, elem in enumerate(lat) if elem.name == name])
    editor = JuTrackLatticeEditor(jt, native, rows)
    changed = copy.deepcopy(rows)
    changed[0]["length_m"] = 8.
    changed[1]["fields"]["e1"] = .02
    with pytest.raises(LatticeInputError, match="constructor fields changed"):
        editor.sync_rows(changed)
    assert native[0].len == .2 and native[1].e1 == .01


def test_radiation_receives_exact_intact_lattice_and_scales_only_extensive_values():
    physical = [object(), object(), object()]
    raw = dict(I1=1., I2=2., I3=3., I4=.1, I5=.02, Jx=.95, Jy=1., Je=2.05,
               emittx=42.e-12, sigma_E=.001, U0=10.)
    calls = []
    def ringpara(actual, **kw):
        assert actual is physical
        calls.append(kw)
        return raw.copy()
    result = jutrack_periodic_radiation_metrics(SimpleNamespace(ringpara=ringpara), physical, energy_gev=3.5, periods=32)
    assert calls == [{"energy": 3.5e9, "print_summary": False}]
    assert result["physical_element_count"] == 3
    assert result["physical_element_treatment"] == "intact_original_elements"
    for name in ("I1", "I2", "I3", "I4", "I5"):
        assert result[name] == raw[name] * 32
    assert result["emittance_pm"] == pytest.approx(42.)
    assert result["Jx"] == .95 and result["energy_loss_per_turn_eV"] == 320.


def test_full_ring_radiation_keeps_same_native_object():
    physical = [object()] * 9
    raw = dict(I1=1., I2=2., I3=3., I4=.1, I5=.02, Jx=.95, Jy=1., Je=2.05,
               emittx=42.e-12, sigma_E=.001, U0=10.)
    def ringpara(actual, **kw):
        assert actual is physical
        return raw.copy()
    result = api._radiation_evaluation(SimpleNamespace(ringpara=ringpara), SimpleNamespace(energy_gev=3.5), ring=physical)
    assert result["physical_element_count"] == 9
    assert result["jutrack_ring_parameters"] == raw


class SamplerDouble:
    """Deliberately nonphysical test double for validation control flow only."""
    Lattice = list
    deepcopy_element = staticmethod(copy.deepcopy)
    def findelem(self, line, *, name):
        return [i + 1 for i, elem in enumerate(line) if elem.name == name]
    def findm66(self, line, **kw):
        matrix = np.eye(6)
        matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = matrix[3, 3] = .5 ** (sum(elem.len for elem in line) / .8)
        if any(getattr(elem, "corrupt", False) for elem in line):
            matrix[0, 1] = 1.
        if any(getattr(elem, "corrupt_partial", False) and elem.len < .8 for elem in line):
            matrix[0, 1] = .1
        return matrix
    def periodicEdwardsTengTwiss(self, *a, **kw):
        return SimpleNamespace(betax=1., betay=2., alphax=0., alphay=0., dx=.1, dpx=0., dy=.02, dpy=.001)
    def twissline(self, tin, line, refpts=None, **kw):
        return [copy.copy(tin) for _ in refpts] if refpts else copy.copy(tin)
    def gettune(self, *a, **kw):
        return [.2, .3]


def test_sampler_uses_native_deepcopies_preserves_fields_and_checks_map(monkeypatch):
    jt = SamplerDouble()
    fields = {"NumIntSteps": 27}
    rows = [{**api.sector_bend("B", .8, .02), "fields": fields}]
    candidate = api.periodic_cell(title="Diagnostic test", energy_gev=3., elements=rows, source="test")
    original = candidate.as_dict()
    native = [SimpleNamespace(name="B", len=.8, angle=.02, NumIntSteps=27, e1=0., e2=0.)]
    monkeypatch.setattr(api, "build_jutrack_editor", lambda _: JuTrackLatticeEditor(jt, native, rows))
    sampler = JuTrackProfileSampler(candidate, max_step_m=1.)
    assert sampler.diagnostic_editor.lattice[0] is not native[0]
    measured = sampler.linear_metrics(sample_fractions={"B": [.5]})
    assert measured["sampling"]["validated"]
    assert measured["sampling"]["interior_composition_checks"][0]["passed"]
    assert len([row for row in measured["twiss_profile"] if row.get("sample_fraction") == .5]) == 1
    assert candidate.as_dict() == original
    assert vars(native[0]) == {"name": "B", "len": .8, "angle": .02, "NumIntSteps": 27, "e1": 0., "e2": 0.}
    assert sampler.diagnostic_editor.rows[0]["fields"]["NumIntSteps"] == 27
    sampler.diagnostic_editor.lattice[0].corrupt = True
    invalid = sampler.linear_metrics(sample_fractions={"B": [.5]})
    assert not invalid["sampling"]["validated"]
    assert not invalid["sampling"]["map_equivalent"]
    assert invalid["twiss_profile"] == []
    sampler.diagnostic_editor.lattice[0].corrupt = False
    sampler.diagnostic_editor.lattice[0].corrupt_partial = True
    invalid = sampler.linear_metrics(sample_fractions={"B": [.5]})
    assert not invalid["sampling"]["validated"]
    assert invalid["twiss_profile"] == []
    assert "prefix/suffix composition" in invalid["failure"]
    assert invalid["sampling"]["element_map_error_max"] == 0.0  # Full clone agrees, but interior does not.
    assert float(native[0].len) == .8 and float(sampler.diagnostic_editor.lattice[0].len) == .8
    sampler.diagnostic_editor.lattice[0].corrupt_partial = False
    original_twissline = jt.twissline
    def wrong_vertical_exit(tin, line, refpts=None, **kw):
        result = original_twissline(tin, line, refpts=refpts, **kw)
        if refpts is None and line[0].name.startswith("__prefix_"):
            result.dy += .1
        return result
    jt.twissline = wrong_vertical_exit
    invalid = sampler.linear_metrics(sample_fractions={"B": [.5]})
    assert not invalid["sampling"]["validated"]
    assert invalid["twiss_profile"] == []
    assert "physical optics exit" in invalid["failure"]


def test_unsupported_edge_centre_is_explicit_without_starting_julia(monkeypatch):
    candidate = api.periodic_cell(title="Edged bend", energy_gev=3., elements=[bend({"e1": .01})], source="test")
    monkeypatch.setattr(api, "build_jutrack_editor", lambda _: pytest.fail("No diagnostic should start"))
    with pytest.raises(UnsupportedOpticsSampling, match="no centre is validated"):
        JuTrackProfileSampler(candidate)


@pytest.fixture
def configured_native(monkeypatch):
    if os.environ.get("BRILLIANCE_RUN_JUTRACK_TESTS") != "1":
        pytest.skip("Opt-in intact native JuTrack regression")
    from brilliance.research_environment import ResearchEnvironment
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="intact bend regression")
    for key, value in environment.process_environment().items():
        monkeypatch.setenv(key, value)


def native_fixture_rows(*, edges=False):
    return [{**api.quadrupole("QF", .2, 2.), "fields": {"NumIntSteps": 23}}, api.drift("D1", .4),
        {**api.sector_bend("B", .8, 2 * math.pi / 32),
         "fields": {"e1": .013, "e2": .021, "NumIntSteps": 17} if edges else {"NumIntSteps": 17}},
        {**api.quadrupole("QD", .2, -2.), "fields": {"NumIntSteps": 19}}, api.drift("D2", .4)]


def test_real_native_original_edges_and_ringpara_match_original_elrad(configured_native):
    rows = native_fixture_rows(edges=True)
    candidate = api.periodic_cell(title="Intact bend native test", energy_gev=3., elements=rows, source="test", intended_repetitions=32)
    editor = api.build_jutrack_editor(candidate)
    jt, cell = editor.jt, editor.lattice
    physical = cell[2]
    assert float(physical.e1) == .013 and float(physical.e2) == .021 and int(physical.NumIntSteps) == 17
    assert int(cell[0].NumIntSteps) == 23 and int(cell[3].NumIntSteps) == 19
    assert api.jutrack_linear_metrics(jt, cell)["stable"]
    twiss = jt.twissring(cell, dp=0., refpts=list(range(1, len(cell) + 1)))
    native = jt._jl.JuTrack.elrad(physical, twiss[1], twiss[2])
    computed = api.jutrack_periodic_radiation_metrics(jt, cell, energy_gev=3., periods=32)
    for index, key in enumerate(("I1", "I2", "I3", "I4", "I5")):
        assert computed[key] == pytest.approx(float(native[index]) * 32, rel=2e-8, abs=1e-13)
    ring = api.cell_ring(title="Native closed ring", energy_gev=3., cell=rows, periods=32, source="test")
    evaluated = api.evaluate(ring, include_radiation=True, include_profile=False)
    for key in ("I2", "I4", "I5", "Jx", "emittance_pm"):
        assert evaluated.radiation[key] == pytest.approx(computed[key], rel=2e-5, abs=1e-13)
    with pytest.raises(UnsupportedOpticsSampling):
        api.build_jutrack_profile_sampler(candidate)
    assert float(physical.e1) == .013 and float(physical.e2) == .021 and float(physical.len) == .8


def test_real_native_centre_sampling_never_changes_physical_map_or_emittance(configured_native):
    rows = native_fixture_rows()
    candidate = api.periodic_cell(title="Intact diagnostic native test", energy_gev=3., elements=rows, source="test", intended_repetitions=32)
    sampler = api.build_jutrack_profile_sampler(candidate, max_step_m=.3)
    jt, cell = sampler.jt, sampler.physical_editor.lattice
    matrix_before = np.asarray(jt.findm66(cell), dtype=float).copy()
    radiation_before = api.jutrack_periodic_radiation_metrics(jt, cell, energy_gev=3., periods=32)
    result = sampler.linear_metrics(sample_fractions={"B": [.5]})
    assert result["stable"] and result["sampling"]["validated"]
    composition = result["sampling"]["interior_composition_checks"]
    assert len(composition) == 1 and composition[0]["passed"]
    assert composition[0]["source_element"] == "B" and composition[0]["sample_fraction"] == .5
    assert result["sampling"]["physical_element_count"] == len(rows)
    assert result["sampling"]["diagnostic_element_count"] == len(rows)
    assert any(row.get("source_element") == "B" and row.get("sample_fraction") == .5 for row in result["twiss_profile"])
    np.testing.assert_allclose(jt.findm66(cell), matrix_before, rtol=0., atol=0.)
    radiation_after = api.jutrack_periodic_radiation_metrics(jt, cell, energy_gev=3., periods=32)
    assert radiation_before == radiation_after
    assert len(cell) == len(rows) and float(cell[2].len) == .8 and int(cell[2].NumIntSteps) == 17
