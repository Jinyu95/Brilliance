"""Matching mechanics and fresh verification. Synthetic algebra is not optics."""
import copy
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from brilliance import line_matching as matching

INITIAL = dict(beta_x_m=2.0, beta_y_m=3.0, alpha_x=0.0, alpha_y=0.0,
               Dx_m=0.01, Dpx=0.0, Dy_m=0.0, Dpy=0.0)


def setup():
    rows = [api.quadrupole("Q", 0.2, 0.7)]
    knobs = api.compile_lattice_parameterization(rows, [
        {"name": "q", "field": "k1_m2", "element_names": ["Q"], "bounds": [-2.0, 2.0]}])
    options = dict(energy_gev=3.5, initial_twiss=INITIAL, targets={"Dx_m": (0.0, 1e-5)},
                   expected_length_m=0.2, expected_angle_rad=0.0, limits=api.magnet_strength_limits(3.5),
                   maximum_beta_m=100, max_starts=2, max_nfev=40)
    return rows, knobs, options


class FakeNative:
    instances = []
    offset = 0.0
    constant = False
    evaluable = True
    excessive_beta = False

    def __init__(self, elements, **configuration):
        self.rows = copy.deepcopy(elements)
        self.index = len(self.instances)
        self.instances.append(self)

    def evaluate(self, elements=None):
        rows = elements if elements is not None else self.rows
        # Only a test double for the callback contract. Production has no callback override.
        value = 0.02414 if self.constant else float(rows[0]["k1_m2"]) - 0.2
        endpoint = {**INITIAL, "Dx_m": value + (self.offset if self.index else 0)}
        return {"evaluable": self.evaluable, "passed": self.evaluable and not self.excessive_beta,
                "endpoint": endpoint, "metrics": {"maximum_beta_m": 58657 if self.excessive_beta else 10},
                "diagnostics": ["synthetic trial is unevaluable"] if not self.evaluable else []}


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    FakeNative.instances = []
    for key, value in (("offset", 0.0), ("constant", False), ("evaluable", True), ("excessive_beta", False)):
        monkeypatch.setattr(FakeNative, key, value)
    monkeypatch.setattr(matching, "JuTrackLineEvaluator", FakeNative)


def test_success_is_fresh_boundary_match_not_a_complete_cell():
    rows, knobs, options = setup()
    before = copy.deepcopy(rows)
    result = api.match_jutrack_line(rows, knobs, **options)
    assert result["match_success"]
    assert result["checks"]["fresh_jutrack_verification"]
    assert result["elements"][0]["k1_m2"] == pytest.approx(0.2, abs=1e-5)
    assert result["complete_cell_accepted"] is False
    assert result["complete_cell_acceptance"]["status"] == "not_evaluated"
    assert len(FakeNative.instances) == 2
    assert rows == before
    assert all(row["passed"] for row in result["residuals"])
    assert result["attempts"][0]["residual_jacobian_rank"] == 1
    assert result["callback_evaluations"] > result["attempts"][0]["nfev"]


def test_solver_gtol_termination_does_not_mean_achromat_solved(monkeypatch):
    monkeypatch.setattr(FakeNative, "constant", True)
    rows, knobs, options = setup()
    result = api.match_jutrack_line(rows, knobs, **options)
    assert result["solver_terminated"] is True
    assert result["solver_success"] is True
    assert result["match_success"] is False
    assert result["residuals"][0]["value"] == pytest.approx(0.02414)
    assert result["residuals"][0]["scaled_residual"] == pytest.approx(2414)
    assert result["attempts"][0]["residual_jacobian_rank"] == 0
    assert result["complete_cell_accepted"] is False


def test_excessive_beta_cannot_be_compensated_by_zero_boundary_dispersion(monkeypatch):
    monkeypatch.setattr(FakeNative, "excessive_beta", True)
    rows, knobs, options = setup()
    result = api.match_jutrack_line(rows, knobs, **options)
    assert result["match_success"] is False
    assert result["checks"]["line_and_engineering_limits"] is False


def test_fresh_replay_disagreement_overrides_training_point(monkeypatch):
    monkeypatch.setattr(FakeNative, "offset", 0.025)
    rows, knobs, options = setup()
    result = api.match_jutrack_line(rows, knobs, **options)
    assert result["solver_success"]
    assert not result["match_success"]
    assert not result["checks"]["fresh_jutrack_verification"]
    assert result["residuals"][0]["value"] == pytest.approx(0.025, abs=1e-5)


def test_unevaluable_trials_not_flat_successful_matches(monkeypatch):
    monkeypatch.setattr(FakeNative, "evaluable", False)
    rows, knobs, options = setup()
    result = api.match_jutrack_line(rows, knobs, **options)
    assert not result["solver_terminated"]
    assert not result["solver_success"]
    assert not result["match_success"]
    assert result["elements"] is None
    assert result["callback_evaluations"] == 2
    assert all(row["reason"] == "unevaluable_line_trial" for row in result["attempts"])


def test_eight_supplied_starts_execute_exactly_eight_attempts_without_hidden_baseline(monkeypatch):
    monkeypatch.setattr(FakeNative, "evaluable", False)
    rows, knobs, options = setup()
    supplied = [[.7], [-.8], [-.6], [-.4], [-.2], [0.], [.2], [.4]]
    result = api.match_jutrack_line(rows, knobs, **{**options, "max_starts": 8,
                                                  "start_positions": supplied})
    assert result["configured_starts"] == result["completed_starts"] == 8
    assert result["callback_evaluations"] == 8
    assert [attempt["initial_position"] for attempt in result["attempts"]] == supplied
    assert all(attempt["source"] == "supplied" for attempt in result["attempts"])
    assert all(attempt["nfev"] is None and attempt["residual_jacobian_rank"] is None
               for attempt in result["attempts"])
    assert result["remaining_start_positions"] == []
    assert not result["match_success"]


def test_native_api_errors_propagate_not_recorded_as_physics(monkeypatch):
    def broken(*args):
        raise TypeError("wrong native API signature")
    monkeypatch.setattr(FakeNative, "evaluate", broken)
    rows, knobs, options = setup()
    with pytest.raises(TypeError, match="native API"):
        api.match_jutrack_line(rows, knobs, **options)


def test_cancellation_checked_before_simulation():
    def cancelled():
        raise InterruptedError("scientist cancelled")
    rows, knobs, options = setup()
    with pytest.raises(InterruptedError):
        api.match_jutrack_line(rows, knobs, cancellation_check=cancelled, **options)


@pytest.mark.parametrize("change", [{"targets": {"tune_x": (0.25, 1e-4)}}, {"max_starts": 0},
                                  {"max_nfev": True}, {"start_positions": [[100]]}])
def test_bad_contract_fails_before_native_calls(change):
    rows, knobs, options = setup()
    options.update(change)
    with pytest.raises(ValueError):
        api.match_jutrack_line(rows, knobs, **options)
    assert not FakeNative.instances


def test_managed_matching_has_live_progress_and_cancellation_without_model_boilerplate(monkeypatch, tmp_path):
    from brilliance import research_api
    from brilliance.native_research_agent import _requires_durable_execution

    events, cancellations = [], []
    run = SimpleNamespace(progress=lambda *args, **kwargs: events.append((args, kwargs)),
                          raise_if_cancelled=lambda: cancellations.append(True))
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(tmp_path))
    monkeypatch.setattr(research_api, "open_research_run", lambda: run)
    rows, knobs, options = setup()
    result = api.match_jutrack_line(rows, knobs, **options)
    assert result["match_success"] and cancellations
    assert events[0][0] == ("Matching the declared line boundary",)
    assert events[-1][0] == ("JuTrack boundary match verified",)
    assert events[-1][1]["tables"][0]["rows"] == result["residuals"]
    assert _requires_durable_execution({"detected_calls": ["match_jutrack_line"]})
