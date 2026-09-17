"""Durability/reporting boundaries; synthetic evidence, never accelerator design."""

import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence
from brilliance.numerical_checkpoints import (
    ReportingStopped,
    read_numerical_checkpoint,
    write_numerical_checkpoint,
)
from brilliance.research_api import ResearchRun
from brilliance.research_outcomes import ResearchInfeasible


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("BRILLIANCE_RUN_OUTPUT", "BRILLIANCE_SCIENTIFIC_ACCEPTANCE", "BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE"):
        monkeypatch.delenv(name, raising=False)
    candidate_evidence._captured.clear()


def make_run(tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "artifacts" / "programs" / "trial"
    output.mkdir(parents=True)
    return ResearchRun(workspace=workspace, output_directory=output, arguments_file=output / "arguments.json",
        result_file=output / "result.json", progress_file=None, cancellation_file=None,
        working_lattice_file=None, latest_candidate_lattice_file=None, compute_job_id=None)


def captured_candidate():
    candidate = api.periodic_cell(title="Synthetic provenance", energy_gev=3., intended_repetitions=8,
                                 elements=[api.drift("D", 1.)], source="checkpoint test double")
    measured = {"engine": "JuTrack", "linear_optics": {"stable": True, "Dx_m": .0034261995199999995}}
    measured["evaluation_receipt"] = candidate_evidence.record_evaluation(candidate, measured, effective_periods=8)
    return candidate, measured


def test_exact_detached_line_state_survives_mutation_and_later_failure(tmp_path):
    source = {"elements": [{"name": "Q", "k1_m2": 3.4261995199999995}],
              "incoming_twiss": {"Dx_m": .001234567891234}, "measurement": {"endpoint": {"Dx_m": 0.0}},
              "match_success": True, "complete_cell_accepted": False}
    before = copy.deepcopy(source)
    receipt = write_numerical_checkpoint("jutrack_line_match", source, output_directory=tmp_path, checks_passed=True)
    source["elements"][0]["k1_m2"] = 3.4262
    saved = read_numerical_checkpoint(receipt, output_directory=tmp_path)
    assert saved["result"] == before
    assert saved["checks_passed"] is True
    assert saved["request_satisfied"] is saved["use_as_working_lattice"] is False
    assert read_numerical_checkpoint(output_directory=tmp_path)["descriptor"] == receipt
    old_bytes = (tmp_path / receipt["path"]).read_bytes()
    with pytest.raises(ValueError, match="Out of range float"):
        write_numerical_checkpoint("invalid", {"value": math.nan}, output_directory=tmp_path)
    assert (tmp_path / receipt["path"]).read_bytes() == old_bytes
    assert read_numerical_checkpoint(output_directory=tmp_path)["descriptor"] == receipt


def test_checkpoint_tampering_and_scope_change_are_not_recoverable(tmp_path, monkeypatch):
    receipt = write_numerical_checkpoint("test", {"value": 1}, output_directory=tmp_path)
    with pytest.raises(ValueError, match="leaves"):
        read_numerical_checkpoint({**receipt, "path": "../outside.json"}, output_directory=tmp_path)
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(tmp_path / "different_run"))
    with pytest.raises(ValueError, match="cannot change"):
        read_numerical_checkpoint(receipt, output_directory=tmp_path)
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT")
    (tmp_path / receipt["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        read_numerical_checkpoint(receipt, output_directory=tmp_path)


def test_candidate_checkpoint_is_durable_before_optional_formatting(tmp_path):
    run = make_run(tmp_path)
    candidate, measured = captured_candidate()
    receipt = run.checkpoint_candidate(candidate, measured, name="complete_cell")
    saved = read_numerical_checkpoint(receipt, output_directory=run.output_directory)
    archive_path = run.output_directory / saved["result"]["candidate_evaluation"]["path"]
    assert archive_path.is_file()
    candidate_evidence._captured.clear()  # Recovery must not depend on live Python/JuTrack state.
    with pytest.raises(ReportingStopped, match="KeyError"):
        with run.reporting(checkpoint=receipt):
            raise KeyError("table_display_column")
    result = json.loads(run.result_file.read_text(encoding="utf-8"))
    assert result["status"] == "partial"
    assert result["request_satisfied"] is result["use_as_working_lattice"] is False
    assert result["scientific_passed"] is None  # Provenance is not full scientific acceptance.
    restored = json.loads((run.output_directory / result["candidate_lattice"]).read_text(encoding="utf-8"))
    assert restored == candidate.as_dict()
    assert candidate_evidence.validate_saved_candidate_evaluation(
        restored, result["candidate_evaluation"], run.output_directory)["passed"]
    assert result["data"]["reporting_failure"]["type"] == "KeyError"
    assert "table_display_column" in result["data"]["reporting_failure"]["traceback"]


def test_altered_candidate_capture_cannot_be_checkpointed(tmp_path):
    run = make_run(tmp_path)
    candidate, measured = captured_candidate()
    measured["linear_optics"]["Dx_m"] = 0.0
    with pytest.raises(ValueError, match="unverified"):
        run.checkpoint_candidate(candidate, measured)
    assert not (run.output_directory / "numerical_checkpoints" / "latest.json").exists()


@pytest.mark.parametrize("error", [InterruptedError("scientist cancelled"), KeyboardInterrupt(),
    ResearchInfeasible("No match", code="unmatched", diagnostics={})])
def test_reporting_boundary_never_converts_cancellation_or_physics_rejection(tmp_path, error):
    run = make_run(tmp_path)
    receipt = write_numerical_checkpoint("test", {"value": 1}, output_directory=run.output_directory)
    with pytest.raises(type(error)):
        with run.reporting(checkpoint=receipt):
            raise error
    assert not run.result_file.exists()


def test_unmarked_program_error_stays_an_error_despite_completed_checkpoint(tmp_path):
    run = make_run(tmp_path)
    write_numerical_checkpoint("test", {"value": 1}, output_directory=run.output_directory)
    with pytest.raises(TypeError):
        raise TypeError("new numerical calculation used a wrong API")
    assert not run.result_file.exists()


def test_failed_line_checkpoint_never_becomes_accepted_after_reporting_failure(tmp_path):
    run = make_run(tmp_path)
    receipt = write_numerical_checkpoint("jutrack_line_match", {
        "scope": "open_line_boundary_match", "match_success": False, "complete_cell_accepted": False,
        "checks": {"boundary_targets": False},
        "residuals": [{"observable": "Dx_m", "value": .03, "target": 0., "passed": False}],
        "elements": [{"name": "Q", "k1_m2": .3}], "incoming_twiss": {"Dx_m": .01},
        "measurement": {"endpoint": {"Dx_m": .03}},
    }, checks_passed=False, output_directory=run.output_directory)
    with pytest.raises(ReportingStopped):
        with run.reporting(checkpoint=receipt):
            raise ValueError("bad plot style")
    saved = json.loads(run.result_file.read_text(encoding="utf-8"))
    assert saved["scientific_passed"] is False
    assert saved["request_satisfied"] is False
    assert saved["candidate_lattice"] is None
    assert saved["data"]["match"]["measurement"]["endpoint"]["Dx_m"] == .03


@pytest.mark.parametrize("full_app", [False, True])
def test_managed_match_archives_exact_fresh_result_before_final_callback(tmp_path, monkeypatch, full_app):
    from brilliance import line_matching

    class NativeDouble:
        def __init__(self, elements, **_kwargs):
            self.elements = copy.deepcopy(elements)

        def evaluate(self, elements=None):
            selected = elements if elements is not None else self.elements
            return {"evaluable": True, "passed": True,
                    "endpoint": {"Dx_m": selected[0]["k1_m2"] - .2},
                    "metrics": {"maximum_beta_m": 2.0}}

    def presentation(update):
        if update["phase"] == "line_match_result":
            raise ValueError("optional final presentation failed")

    output = tmp_path
    if full_app:
        run = make_run(tmp_path)
        output = run.output_directory
        monkeypatch.setenv("BRILLIANCE_WORKSPACE", str(run.workspace))
        monkeypatch.setenv("BRILLIANCE_RUN_ARGUMENTS", str(run.arguments_file))
        monkeypatch.setenv("BRILLIANCE_RUN_RESULT", str(run.result_file))
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(output))
    monkeypatch.setattr(line_matching, "JuTrackLineEvaluator", NativeDouble)
    elements = [api.quadrupole("Q", .2, .4)]
    knobs = api.compile_lattice_parameterization(elements, [
        {"name": "Q", "field": "k1_m2", "element_names": ["Q"], "bounds": [-1., 1.]}])
    incoming = dict(beta_x_m=2., beta_y_m=3., alpha_x=0., alpha_y=0., Dx_m=.01, Dpx=0., Dy_m=0., Dpy=0.)
    with pytest.raises(ReportingStopped if full_app else ValueError, match="optional final presentation"):
        api.match_jutrack_line(elements, knobs, energy_gev=3., initial_twiss=incoming,
            targets={"Dx_m": (0., 1e-5)}, expected_length_m=.2, expected_angle_rad=0.,
            limits=api.magnet_strength_limits(3.), maximum_beta_m=100., max_starts=1,
            progress_callback=presentation, cancellation_check=lambda: None)
    saved = read_numerical_checkpoint(output_directory=output)
    assert saved["checks_passed"] is True
    assert saved["result"]["match_success"] is True
    assert saved["result"]["complete_cell_accepted"] is False
    assert saved["result"]["incoming_twiss"] == incoming
    assert abs(saved["result"]["measurement"]["endpoint"]["Dx_m"]) < 1e-5
    assert saved["result"]["elements"][0]["k1_m2"] == pytest.approx(.2, abs=1e-5)
    if full_app:
        partial = json.loads(run.result_file.read_text(encoding="utf-8"))
        assert partial["status"] == "partial"
        assert partial["request_satisfied"] is False
        assert partial["data"]["match"]["match_success"] is True


def test_marked_reporting_failure_returns_normal_partial_result_from_real_worker(tmp_path):
    from brilliance.research_program import inspect_research_program, run_research_program

    run = make_run(tmp_path)
    script = run.workspace / "checkpoint_and_report.py"
    script.write_text('''
from brilliance import accelerator_api as api
from brilliance.candidate_evidence import record_evaluation
from brilliance.research_api import open_research_run
run = open_research_run()
candidate = api.periodic_cell(title="Test capture, not physics", energy_gev=3.0,
    elements=[api.drift("D", 1.)], intended_repetitions=8, source="synthetic checkpoint test")
evaluation = {"engine": "JuTrack", "linear_optics": {"stable": True, "Dx_m": .002}}
evaluation["evaluation_receipt"] = record_evaluation(candidate, evaluation, effective_periods=8)
checkpoint = run.checkpoint_candidate(candidate, evaluation)
with run.reporting(checkpoint=checkpoint):
    raise KeyError("optional_plot_label")
    run.finish(summary="Unreachable presentation", candidate_lattice=candidate, candidate_evaluation=evaluation)
run.artifact_path("unsafe_next_stage.txt").write_text("must not be written")
''', encoding="utf-8")
    assert inspect_research_program(run.workspace, script.name)["passed"]

    def environment(*_args, **_kwargs):
        return {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                "BRILLIANCE_INTERNET_ACCESS": "false"}

    context = SimpleNamespace(workspace=run.workspace, project_id="checkpoint-test", compute_job_id=None,
        project=SimpleNamespace(active_lattice_id=None), active_lattice=lambda: None,
        is_cancellation_requested=lambda: False,
        artifact_path=lambda name: run.workspace / "artifacts" / name,
        environment=SimpleNamespace(python_executable=sys.executable, project_process_environment=environment))
    execution = run_research_program(context, {"script": script.name, "timeout_s": 30})
    assert execution["returncode"] == 0
    assert execution["result"]["status"] == "partial"
    assert execution["result"]["request_satisfied"] is False
    assert execution["working_lattice_action"] == "branch"
    output = Path(execution["output_directory"])
    assert not (output / "unsafe_next_stage.txt").exists()
    assert candidate_evidence.validate_saved_candidate_evaluation(
        execution["candidate"], execution["result"]["candidate_evaluation"], output)["passed"]
    assert "optional_plot_label" in execution["result"]["limitations"][0]
