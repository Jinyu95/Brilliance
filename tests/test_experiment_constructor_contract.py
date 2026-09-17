"""Constructor and matching contracts; mocked plumbing, no physics claims."""
import copy

import pytest

from brilliance import accelerator_api as api
from brilliance.experiments import JuTrackCellEvaluator, MatchingOperation


def _rows(kind="CORRECTOR", fields=None):
    return [api.quadrupole("Q", .2, 1.),
            {"name": "device", "kind": kind, "length_m": .2, "fields": fields or {}}]


def _parameters(rows):
    return api.compile_lattice_parameterization(rows, [
        {"name": "Q-strength", "field": "k1_m2", "element_names": ["Q"], "bounds": [.1, 2.]}])


@pytest.mark.parametrize("kind,fields,changed", [
    ("CORRECTOR", {}, {"xkick_rad": .001}),
    ("CORRECTOR", {}, {"ykick_rad": .002}),
    ("RFCA", {"voltage_v": 1e6, "frequency_hz": 500e6}, {"voltage_v": 2e6}),
    ("RFCA", {"volt": 1e6, "freq": 500e6}, {"freq": 450e6}),
])
def test_constructor_change_rebuilds_and_cannot_be_hidden_as_metadata(monkeypatch, kind, fields, changed):
    builds = []

    class Editor:
        def evaluate_candidate(self, rows, **kwargs):
            return {"linear_optics": {"stable": True}, "radiation": {}}

    monkeypatch.setattr(api, "audit_lattice_constraints", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(api, "build_jutrack_editor", lambda c: builds.append(c) or Editor())
    original = _rows(kind, fields)
    evaluate = JuTrackCellEvaluator(energy_gev=3., periods=32, expected_length_m=.4,
        expected_angle_rad=0., limits={}, minimum_magnet_length_m=.1, minimum_drift_length_m=.1)
    evaluate(original)
    proposed = copy.deepcopy(original)
    proposed[1]["fields"].update(changed)
    evaluate(proposed)
    assert len(builds) == 2

    def solve(rows, parameters):
        rows[1]["fields"].update(changed)
        return {"match_success": True, "elements": rows}

    with pytest.raises(ValueError, match="native constructor fields"):
        MatchingOperation(_parameters(original), solve, {}).apply(original)


def test_explicit_default_and_metadata_do_not_rebuild(monkeypatch):
    builds = []

    class Editor:
        def evaluate_candidate(self, rows, **kwargs):
            return {"linear_optics": {"stable": True}, "radiation": {}}

    monkeypatch.setattr(api, "audit_lattice_constraints", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(api, "build_jutrack_editor", lambda c: builds.append(c) or Editor())
    evaluate = JuTrackCellEvaluator(energy_gev=3., periods=32, expected_length_m=.4,
        expected_angle_rad=0., limits={}, minimum_magnet_length_m=.1, minimum_drift_length_m=.1)
    rows = _rows()
    evaluate(rows)
    rows[1]["fields"].update(xkick_rad=0., ykick_rad=0., family="new metadata")
    rows[0]["k1_m2"] = .5
    evaluate(rows)
    assert len(builds) == 1


@pytest.mark.parametrize("returned", [None, []])
def test_success_requires_returned_exact_rows(returned):
    rows = _rows()
    with pytest.raises(TypeError, match="Successful matching requires exact returned elements"):
        MatchingOperation(_parameters(rows), lambda r, p: {
            "success": True, "elements": returned}, {}).apply(rows)
    after, outcome = MatchingOperation(_parameters(rows), lambda r, p: {
        "success": False, "elements": returned}, {}).apply(rows)
    assert after == rows and outcome["matched"] is False


def test_matching_cannot_break_declared_sum_with_in_bound_coordinates():
    rows = [api.drift("left", 1.), api.drift("right", 1.)]
    parameters = api.compile_lattice_parameterization(rows, [
        {"name": name, "field": "length_m", "element_names": [name], "bounds": [.1, 2.]}
        for name in ("left", "right")], sum_constraints=[
            {"field": "length_m", "target": 2., "variable_names": ["left", "right"]}])

    def solve(rows, parameters):
        rows[0]["length_m"] = 1.2  # Individually in bound but total no longer equals 2 m.
        return {"success": True, "elements": rows}

    with pytest.raises(ValueError, match="sum constraint"):
        MatchingOperation(parameters, solve, {}).apply(rows)
    unchanged, result = MatchingOperation(parameters, lambda r, p: {
        "success": True, "elements": r}, {}).apply(rows)
    assert unchanged == rows and result["matched"]
