"""Open-line API contract plus opt-in configured native JuTrack transport."""
from __future__ import annotations

import copy
import math
import os
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from brilliance.lattice_optimization import JuTrackLatticeEditor
from brilliance.line_optics import JuTrackLineEvaluator, evaluate_jutrack_line

INITIAL = {"beta_x_m": 5.0, "beta_y_m": 7.0, "alpha_x": 0.2, "alpha_y": -0.3,
           "Dx_m": 0.04, "Dpx": 0.003, "Dy_m": 0.002, "Dpy": -0.001}


def options(rows, **overrides):
    return {"energy_gev": 3.5, "initial_twiss": INITIAL, "maximum_beta_m": None,
            "limits": api.magnet_strength_limits(3.5), "profile_step_m": 0.25,
            "expected_length_m": math.fsum(row["length_m"] for row in rows),
            "expected_angle_rad": math.fsum(row.get("angle_rad") or 0.0 for row in rows), **overrides}


class FakeNative:
    """Only a composable drift surrogate to check adapter plumbing, not physics."""
    Lattice = list
    deepcopy_element = staticmethod(copy.deepcopy)

    def __init__(self):
        self.calls = []
        self.inputs = []

    def findelem(self, line, *, name):
        return [i + 1 for i, element in enumerate(line) if element.name == name]

    def EdwardsTengTwiss(self, betax, betay, **kw):
        self.inputs.append((betax, betay, kw))
        return SimpleNamespace(betax=betax, betay=betay, mode=1, **kw)

    def twissline(self, tin, line, *, dp, order, E0, refpts=None):
        self.calls.append((dp, order, E0, refpts))
        previous = tin
        points = []
        for element in line:
            value = copy.copy(previous)
            for suffix in ("x", "y"):
                beta, alpha = getattr(previous, "beta" + suffix), getattr(previous, "alpha" + suffix)
                gamma = (1 + alpha**2) / beta
                setattr(value, "beta" + suffix, beta - 2 * element.len * alpha + element.len**2 * gamma)
                setattr(value, "alpha" + suffix, alpha - element.len * gamma)
                setattr(value, "d" + suffix, getattr(previous, "d" + suffix) + element.len * getattr(previous, "dp" + suffix))
            points.append(value)
            previous = value
        return [points[i - 1] for i in refpts] if refpts is not None else points[-1]


@pytest.fixture
def native(monkeypatch):
    jt = FakeNative()
    builds = []

    def build(candidate):
        rows = [copy.deepcopy(row) for row in api.one_period_elements(candidate)]
        line = [SimpleNamespace(name=row["name"], len=row["length_m"], k1=row.get("k1_m2") or 0.0,
                                angle=row.get("angle_rad") or 0.0) for row in rows]
        editor = JuTrackLatticeEditor(jt, line, rows)
        builds.append(editor)
        return editor

    monkeypatch.setattr(api, "build_jutrack_editor", build)
    return jt, builds


def test_complete_initial_state_explicit_energy_exit_indices_and_midpoint_profile(native):
    jt, builds = native
    rows = [api.drift("D1", 0.6), api.drift("D2", 0.4)]
    original = copy.deepcopy(rows)
    result = evaluate_jutrack_line(rows, **options(rows))
    assert result["passed"] and result["evaluable"]
    assert len(builds) == 1
    assert jt.inputs[0][2] == {"alphax": 0.2, "alphay": -0.3, "dx": 0.04,
                               "dpx": 0.003, "dy": 0.002, "dpy": -0.001}
    assert jt.calls[0] == (0.0, 0, 3.5e9, [1, 2])
    assert all(call[2] == 3.5e9 for call in jt.calls)
    assert result["endpoint"]["Dx_m"] == pytest.approx(0.043)
    assert result["endpoint"]["Dy_m"] == pytest.approx(0.001)
    assert result["twiss_profile"][0]["name"] == "START"
    assert result["twiss_profile"][-1]["s_m"] == pytest.approx(1.0)
    assert {row["source_element"] for row in result["twiss_profile"] if row.get("sample_fraction") == 0.5} == {"D1", "D2"}
    assert result["sampling"]["maximum_is_sampled"]
    assert "stable" not in result  # An open line is not a periodic stability certificate.
    assert rows == original


def test_hardware_or_length_infeasibility_does_not_start_native_worker(native):
    jt, builds = native
    rows = [api.quadrupole("Q", 0.3, 20.0)]
    result = evaluate_jutrack_line(rows, **options(rows))
    assert not result["evaluable"] and not result["passed"]
    assert "quadrupole exceeds" in result["diagnostics"][0]
    rows = [api.drift("D", 1.0)]
    result = evaluate_jutrack_line(rows, **options(rows, expected_length_m=2.0))
    assert not result["evaluable"]
    assert builds == [] and jt.calls == []


def test_finite_above_beta_cap_retains_endpoint_for_matcher(native):
    rows = [api.drift("D", 1.0)]
    result = evaluate_jutrack_line(rows, **options(rows, maximum_beta_m=1.0))
    assert result["evaluable"] and not result["passed"]
    assert result["endpoint"] and result["twiss_profile"]
    assert result["checks"]["finite_optics"] and not result["checks"]["maximum_beta"]


def test_editor_reused_and_infeasible_proposal_does_not_corrupt_physical_state(native):
    jt, builds = native
    rows = [api.quadrupole("Q", 0.3, 1.0)]
    evaluator = JuTrackLineEvaluator(rows, **options(rows))
    evaluator.evaluate()
    changed = [{**rows[0], "k1_m2": 1.2}]
    evaluator.evaluate(changed)
    assert len(builds) == 1 and builds[0].lattice[0].k1 == 1.2
    before = len(jt.calls)
    assert not evaluator.evaluate([{**rows[0], "k1_m2": 50.0}])["evaluable"]
    assert len(jt.calls) == before and builds[0].lattice[0].k1 == 1.2
    with pytest.raises(ValueError, match="count, order, names, or kinds"):
        evaluator.evaluate([api.drift("Different", 0.3)])


@pytest.mark.parametrize("error", [KeyError, TypeError, ValueError, RuntimeError, PermissionError])
def test_native_api_errors_are_not_disguised_as_infeasible_physics(native, error):
    jt, _ = native
    def fail(*args, **kwargs):
        raise error("exact native diagnostic")
    jt.twissline = fail
    rows = [api.drift("D", 1.0)]
    with pytest.raises(error, match="exact native diagnostic"):
        evaluate_jutrack_line(rows, **options(rows))


def test_documented_twiss_domain_failure_is_explicit_physical_outcome(native):
    jt, _ = native
    JuliaError = type("JuliaError", (Exception,), {"__module__": "juliacall"})
    def fail(*args, **kwargs):
        raise JuliaError("DomainError: negative sqrt argument in twissPropagate")
    jt.twissline = fail
    rows = [api.drift("D", 1.0)]
    result = evaluate_jutrack_line(rows, **options(rows))
    assert not result["evaluable"]
    assert "negative sqrt" in result["diagnostics"][0]


@pytest.mark.parametrize("incoming", [{key: value for key, value in INITIAL.items() if key != "Dpy"},
                                       {**INITIAL, "betax": 5.0}, {**INITIAL, "beta_y_m": -1.0}])
def test_full_incoming_state_required_without_undisclosed_zero_defaults(native, incoming):
    rows = [api.drift("D", 1.0)]
    with pytest.raises(ValueError):
        JuTrackLineEvaluator(rows, **options(rows, initial_twiss=incoming))
    assert native[1] == []


@pytest.fixture
def configured_native(monkeypatch):
    if os.environ.get("BRILLIANCE_RUN_JUTRACK_TESTS") != "1":
        pytest.skip("Opt-in configured native JuTrack open-line transport")
    from brilliance.research_environment import ResearchEnvironment
    env = ResearchEnvironment.load()
    env.require_active_python(operation="native line optics regression")
    for key, value in env.process_environment().items():
        monkeypatch.setenv(key, value)


def test_real_jutrack_drift_full_eight_component_state(configured_native):
    rows = [api.drift("D", 0.7)]
    result = evaluate_jutrack_line(rows, **options(rows))
    assert result["passed"]
    endpoint = result["endpoint"]
    for plane in ("x", "y"):
        beta, alpha = INITIAL[f"beta_{plane}_m"], INITIAL[f"alpha_{plane}"]
        gamma = (1 + alpha**2) / beta
        assert endpoint[f"beta_{plane}_m"] == pytest.approx(beta - 1.4 * alpha + 0.49 * gamma, rel=2e-7)
        assert endpoint[f"alpha_{plane}"] == pytest.approx(alpha - 0.7 * gamma, rel=2e-7)
    assert endpoint["Dx_m"] == pytest.approx(INITIAL["Dx_m"] + 0.7 * INITIAL["Dpx"], rel=2e-7)
    assert endpoint["Dy_m"] == pytest.approx(INITIAL["Dy_m"] + 0.7 * INITIAL["Dpy"], rel=2e-7)
    assert endpoint["Dpx"] == pytest.approx(INITIAL["Dpx"], rel=2e-7)
    assert endpoint["Dpy"] == pytest.approx(INITIAL["Dpy"], rel=2e-7)


@pytest.mark.parametrize("kind", ["QUAD", "KQUAD"])
def test_real_jutrack_quadrupole_has_opposite_vertical_sign_and_reuses_editor(configured_native, kind):
    rows = [{**api.quadrupole("Q", 0.3, 1.0), "kind": kind}]
    initial = {**INITIAL, "beta_x_m": 10.0, "beta_y_m": 10.0, "alpha_x": 0.0, "alpha_y": 0.0}
    evaluator = JuTrackLineEvaluator(rows, **options(rows, initial_twiss=initial))
    positive = evaluator.evaluate()
    negative = evaluator.evaluate([{**rows[0], "k1_m2": -1.0}])
    assert positive["passed"] and negative["passed"]
    assert positive["endpoint"]["beta_x_m"] < 10.0 < positive["endpoint"]["beta_y_m"]
    assert negative["endpoint"]["beta_y_m"] < 10.0 < negative["endpoint"]["beta_x_m"]
    assert negative["endpoint"]["beta_x_m"] == pytest.approx(positive["endpoint"]["beta_y_m"], rel=2e-7)
    assert evaluator.evaluate()["endpoint"] == pytest.approx(positive["endpoint"], rel=2e-7, abs=1e-10)


def test_real_jutrack_bend_generates_dispersion_from_zero_and_preserves_exact_rows(configured_native):
    rows = [api.drift("D0", 0.2), api.sector_bend("B", 1.0, 0.05), api.drift("D1", 0.2)]
    original = copy.deepcopy(rows)
    initial = {**INITIAL, "Dx_m": 0.0, "Dpx": 0.0, "Dy_m": 0.0, "Dpy": 0.0}
    evaluator = JuTrackLineEvaluator(rows, **options(rows, initial_twiss=initial))
    result = evaluator.evaluate()
    assert result["passed"]
    assert result["endpoint"]["Dx_m"] > 0.01 and result["endpoint"]["Dpx"] > 0.01
    assert result["endpoint"]["Dy_m"] == pytest.approx(0.0, abs=1e-10)
    matrix = evaluator.jt.findm66(evaluator.editor.lattice, dp=0.0, order=0, E0=3.5e9)
    assert result["endpoint"]["Dx_m"] == pytest.approx(float(matrix[0, 5]), rel=2e-6)
    assert result["endpoint"]["Dpx"] == pytest.approx(float(matrix[1, 5]), rel=2e-6)
    assert rows == original
