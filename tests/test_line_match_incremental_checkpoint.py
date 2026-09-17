"""Completed native observations survive later failure without becoming acceptance.

The native and solver doubles exercise persistence only, not accelerator physics.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from brilliance import accelerator_api as api
from brilliance import line_matching as matching
from brilliance import numerical_checkpoints
from brilliance.numerical_checkpoints import read_numerical_checkpoint

INCOMING = {"beta_x_m": 2.123456789012345, "beta_y_m": 3.456789012345678,
            "alpha_x": -.123456789012345, "alpha_y": .234567890123456,
            "Dx_m": .0123456789012345, "Dpx": -.00123456789012345,
            "Dy_m": 0., "Dpy": 0.}


@pytest.fixture
def bounded_match(monkeypatch, tmp_path):
    observed = []
    controls = {"raise_on_call": None}

    class NativeDouble:
        def __init__(self, elements, **configuration):
            self.rows = copy.deepcopy(elements)

        def evaluate(self, elements=None):
            rows = copy.deepcopy(elements if elements is not None else self.rows)
            observed.append(rows)
            if len(observed) == controls["raise_on_call"]:
                raise RuntimeError("synthetic native bridge failure on next start")
            endpoint = {**INCOMING, "Dx_m": .1 + rows[0]["k1_m2"]}
            return {"evaluable": True, "passed": True, "endpoint": endpoint,
                    "metrics": {"maximum_beta_m": 12.34567890123456},
                    "diagnostics": []}

    def one_evaluation_solver(callback, initial, **kwargs):
        residuals = callback(initial)
        return SimpleNamespace(success=False, status=0, message="synthetic bounded solver",
                               nfev=1, njev=0, x=initial.copy(),
                               jac=np.zeros((len(residuals), len(initial))))

    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(tmp_path))
    for key in ("BRILLIANCE_WORKSPACE", "BRILLIANCE_RUN_ARGUMENTS", "BRILLIANCE_RUN_RESULT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(matching, "JuTrackLineEvaluator", NativeDouble)
    monkeypatch.setattr(matching, "least_squares", one_evaluation_solver)
    rows = [api.quadrupole("Q", .2345678901234567, .3)]
    parameters = api.compile_lattice_parameterization(rows, [
        {"name": "Q-strength", "field": "k1_m2", "element_names": ["Q"], "bounds": [-2., 2.]}])
    options = {"energy_gev": 3.5, "initial_twiss": INCOMING,
               "targets": {"Dx_m": (0., 1e-5)},
               "expected_length_m": rows[0]["length_m"], "expected_angle_rad": 0.,
               "limits": api.magnet_strength_limits(3.5), "maximum_beta_m": 100.,
               "start_positions": [[.3], [.7]], "max_starts": 2, "max_nfev": 1,
               "cancellation_check": lambda: None,
               "progress_callback": lambda update: None}
    return SimpleNamespace(rows=rows, parameters=parameters, options=options,
                           observed=observed, controls=controls, output=tmp_path)


def _assert_provisional_saved_state(saved, observed):
    assert saved["kind"] == "jutrack_line_match_progress"
    assert saved["request_satisfied"] is False
    assert saved["use_as_working_lattice"] is False
    assert saved["checks_passed"] is not True
    result = saved["result"]
    assert result["match_success"] is False
    assert result["complete_cell_accepted"] is False
    assert result["checks"]["fresh_jutrack_verification"] is False
    assert result["elements"] == observed
    assert result["incoming_twiss"] == INCOMING
    assert result["position"] == [observed[0]["k1_m2"]]
    assert result["measurement"]["endpoint"]["Dx_m"] == .1 + observed[0]["k1_m2"]
    assert result["residuals"][0]["value"] == result["measurement"]["endpoint"]["Dx_m"]
    assert result["completed_starts"] == 1
    assert len(result["attempts"]) == 1
    assert result["configured_starts"] == 2
    assert len(result["remaining_start_positions"]) == 1
    assert result["remaining_start_positions"][0] == [.7]
    assert len(result["remaining_start_sources"]) == 1
    assert result["start_plan"]
    assert result["contract"]["initial_twiss"] == INCOMING
    return result


def test_next_native_failure_preserves_completed_start_and_recovery_needs_no_native(bounded_match, monkeypatch):
    run = bounded_match
    run.controls["raise_on_call"] = 2
    with pytest.raises(RuntimeError, match="synthetic native bridge failure"):
        matching.match_jutrack_line(run.rows, run.parameters, **run.options)
    assert len(run.observed) == 2

    def forbid_native(*args, **kwargs):
        raise AssertionError("Reading a checkpoint must not construct or run JuTrack")

    monkeypatch.setattr(matching, "JuTrackLineEvaluator", forbid_native)
    saved = read_numerical_checkpoint(output_directory=run.output)
    result = _assert_provisional_saved_state(saved, run.observed[0])
    assert result["callback_evaluations"] == 1
    assert len(run.observed) == 2
    # The immutable numerical record remains readable after the live rows change.
    run.rows[0]["k1_m2"] = 999.
    assert read_numerical_checkpoint(saved["descriptor"], output_directory=run.output) == saved


def test_completed_start_is_durable_before_progress_callback_failure(bounded_match):
    run = bounded_match

    def failed_report(update):
        if update.get("phase") == "line_matching":
            raise ValueError("synthetic progress formatting error")

    run.options["progress_callback"] = failed_report
    with pytest.raises(ValueError, match="synthetic progress formatting error"):
        matching.match_jutrack_line(run.rows, run.parameters, **run.options)
    assert len(run.observed) == 1
    saved = read_numerical_checkpoint(output_directory=run.output)
    _assert_provisional_saved_state(saved, run.observed[0])
    # No final fresh-verification checkpoint may appear after reporting stopped.
    snapshots = [json.loads(path.read_text(encoding="utf-8"))
                 for path in (run.output / "numerical_checkpoints").glob("*.json")
                 if path.name != "latest.json"]
    assert snapshots and all(item["kind"] == "jutrack_line_match_progress" for item in snapshots)


def test_long_inflight_start_checkpoints_are_throttled_and_remain_uncompleted(bounded_match, monkeypatch):
    run = bounded_match
    clock = {"now": 100.}
    writes = []
    original_write = numerical_checkpoints.write_numerical_checkpoint

    def recording_write(kind, result, **kwargs):
        if kind == "jutrack_line_match_progress":
            writes.append(clock["now"])
        return original_write(kind, result, **kwargs)

    def interrupted_solver(callback, initial, **kwargs):
        for instant in (100., 120., 159., 160.):
            clock["now"] = instant
            callback(initial)
        raise RuntimeError("synthetic interruption during first local solve")

    monkeypatch.setattr(matching.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(matching, "least_squares", interrupted_solver)
    monkeypatch.setattr(numerical_checkpoints, "write_numerical_checkpoint", recording_write)
    with pytest.raises(RuntimeError, match="synthetic interruption during first local solve"):
        matching.match_jutrack_line(run.rows, run.parameters, **run.options)
    assert writes and len(writes) <= 2
    assert all(b - a >= 60. for a, b in zip(writes, writes[1:]))
    saved = read_numerical_checkpoint(output_directory=run.output)
    assert saved["kind"] == "jutrack_line_match_progress"
    result = saved["result"]
    assert result["completed_starts"] == 0 and result["attempts"] == []
    assert result["remaining_start_positions"] == [[.3], [.7]]
    assert len(result["remaining_start_sources"]) == 2
    assert result["elements"] == run.observed[0]
    assert result["match_success"] is False
    assert result["checks"]["fresh_jutrack_verification"] is False


@pytest.mark.parametrize("phase", ["line_matching_evaluation", "line_matching"])
def test_reporting_cannot_edit_the_observed_residuals_in_later_checkpoint(bounded_match, phase):
    run = bounded_match
    run.controls["raise_on_call"] = 3
    run.options.update(start_positions=[[.3], [.7], [1.]], max_starts=3)

    def formatting_callback(update):
        # User reporting code may format or repurpose its own table. That is
        # not authority to change the stored native numerical observation.
        if update.get("phase") == phase:
            for row in update.get("best_boundary_residuals", []):
                row["value"] = 999.
                row["passed"] = True

    run.options["progress_callback"] = formatting_callback
    with pytest.raises(RuntimeError, match="synthetic native bridge failure"):
        matching.match_jutrack_line(run.rows, run.parameters, **run.options)
    saved = read_numerical_checkpoint(output_directory=run.output)
    result = saved["result"]
    assert saved["kind"] == "jutrack_line_match_progress"
    assert result["elements"] == run.observed[0]
    assert result["completed_starts"] == 2
    assert result["residuals"][0]["value"] == result["measurement"]["endpoint"]["Dx_m"]
    assert result["residuals"][0]["passed"] is False
    assert result["match_success"] is False


def test_final_reporting_cannot_change_returned_result_or_archived_fresh_verification(bounded_match):
    run = bounded_match

    def formatting_callback(update):
        if update.get("phase") == "line_match_result":
            update["residuals"][0]["value"] = 999.
            update["residuals"][0]["passed"] = True
            update["checks"]["fresh_jutrack_verification"] = True

    run.options["progress_callback"] = formatting_callback
    result = matching.match_jutrack_line(run.rows, run.parameters, **run.options)
    assert len(run.observed) == 3  # two starts and independent fresh replay
    saved = read_numerical_checkpoint(output_directory=run.output)
    assert saved["kind"] == "jutrack_line_match"
    assert result["match_success"] is False
    assert result["checks"]["fresh_jutrack_verification"] is False
    assert result["residuals"][0]["passed"] is False
    assert result["residuals"][0]["value"] == result["measurement"]["endpoint"]["Dx_m"]
    assert result["residuals"] == saved["result"]["residuals"]
    assert result["checks"] == saved["result"]["checks"]
