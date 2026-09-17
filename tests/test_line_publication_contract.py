"""Public line-result and publication contracts; synthetic callbacks, no physics."""

import copy
import json

import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence, line_matching
from brilliance.numerical_checkpoints import write_numerical_checkpoint
from brilliance.research_api import ResearchRun
from brilliance.result_contracts import return_contract


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in ("BRILLIANCE_RUN_OUTPUT", "BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE",
                 "BRILLIANCE_SCIENTIFIC_ACCEPTANCE"):
        monkeypatch.delenv(name, raising=False)


def line():
    return api.lattice(title="Synthetic open line", energy_gev=3., periodicity=1,
        elements=[api.quadrupole("Q", .2, .7)], source="publication contract test",
        metadata={"evaluation_scope": "open_line", "not_a_ring": True})


@pytest.fixture
def run(tmp_path):
    output = tmp_path / "workspace" / "artifacts" / "programs" / "report"
    output.mkdir(parents=True)
    return ResearchRun(workspace=tmp_path / "workspace", output_directory=output,
        arguments_file=output / "arguments.json", result_file=output / "result.json",
        progress_file=None, cancellation_file=None, working_lattice_file=None,
        latest_candidate_lattice_file=None, compute_job_id=None)


@pytest.mark.parametrize("constant", [False, True], ids=["matched", "unmatched"])
def test_public_line_result_round_trips_json_without_permissive_converter(monkeypatch, constant):
    initial = dict(beta_x_m=2., beta_y_m=3., alpha_x=0., alpha_y=0.,
                   Dx_m=.01, Dpx=0., Dy_m=0., Dpy=0.)

    class SyntheticLine:
        def __init__(self, elements, **configuration):
            self.elements = copy.deepcopy(elements)

        def evaluate(self, elements=None):
            rows = self.elements if elements is None else elements
            dx = .02414 if constant else float(rows[0]["k1_m2"]) - .2
            return {"evaluable": True, "passed": True,
                "endpoint": {**initial, "Dx_m": dx}, "metrics": {"maximum_beta_m": 10.}}

    monkeypatch.setattr(line_matching, "JuTrackLineEvaluator", SyntheticLine)
    elements = [api.quadrupole("Q", .2, .7)]
    parameters = api.compile_lattice_parameterization(elements, [
        {"name": "Q strength", "field": "k1_m2", "element_names": ["Q"], "bounds": [-2., 2.]}])
    result = api.match_jutrack_line(elements, parameters, energy_gev=3., initial_twiss=initial,
        targets={"Dx_m": (0., 1e-5)}, expected_length_m=.2, expected_angle_rad=0.,
        limits=api.magnet_strength_limits(3.), maximum_beta_m=100., max_starts=2, max_nfev=40)
    assert all(type(row["passed"]) is bool for row in result["residuals"])
    saved = json.loads(json.dumps(result, allow_nan=False))
    assert saved["match_success"] is not constant
    assert saved["complete_cell_accepted"] is False
    assert saved["residuals"][0]["passed"] is not constant
    assert saved["elements"] == result["elements"]
    assert set(result) <= set(return_contract(api.match_jutrack_line)["properties"])


@pytest.mark.parametrize("boundary", ["direct", "saved", "finish"])
def test_open_line_advice_uses_checkpoint_not_periodic_evaluation(run, boundary):
    candidate = line()
    if boundary in {"direct", "saved"}:
        verify = (candidate_evidence.verify_candidate_evaluation if boundary == "direct"
                  else candidate_evidence.validate_saved_candidate_evaluation)
        result = verify(candidate, None, run.output_directory)
        assert result["passed"] is False and result["receipt"] is None
        advice = result["next_action"]
    else:
        path = run.finish(summary="Unmatched line diagnostic", candidate_lattice=candidate,
            scientific_passed=True, request_satisfied=False, use_as_working_lattice=False)
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["request_satisfied"] is False and result["use_as_working_lattice"] is False
        assert result["candidate_evaluation"] is None
        advice = " ".join(result["limitations"])
    assert "checkpoint" in advice.lower()
    assert "api.evaluate(" not in advice
    assert "api.evaluate_periodic_cell" not in advice


def test_unverified_periodic_candidate_still_requires_real_periodic_evaluation(run):
    candidate = api.periodic_cell(title="Unmeasured periodic example", energy_gev=3.,
        elements=[api.quadrupole("Q", .2, .7)], intended_repetitions=32, source="contract test")
    result = candidate_evidence.verify_candidate_evaluation(candidate, None, run.output_directory)
    assert result["passed"] is False and result["receipt"] is None
    assert "api.evaluate_periodic_cell" in result["next_action"]


def test_failed_line_checkpoint_remains_reportable_without_periodic_receipt(run):
    checkpoint = write_numerical_checkpoint("jutrack_line_match", {
        "engine": "JuTrack", "scope": "open_line_boundary_match", "elements": line().as_dict()["elements"],
        "match_success": False, "solver_success": True, "complete_cell_accepted": False,
        "residuals": [{"observable": "Dx_m", "value": .02414, "target": 0.,
                       "tolerance": 1e-5, "scaled_residual": 2414., "passed": False}]},
        output_directory=run.output_directory, checks_passed=False)
    run._publish_checkpoint_after_reporting_failure(checkpoint,
        {"type": "ValueError", "message": "optional synthetic formatting failure"})
    result = json.loads(run.result_file.read_text(encoding="utf-8"))
    assert result["data"]["match"]["residuals"][0]["passed"] is False
    assert result["candidate_evaluation"] is None
    assert result["request_satisfied"] is False and result["use_as_working_lattice"] is False
    assert result["scientific_passed"] is False


def test_line_match_return_contract_exposes_native_handoff_fields():
    schema = return_contract(api.match_jutrack_line)
    fields = schema["properties"]
    assert {"elements", "measurement", "residuals", "attempts", "numerical_checkpoint",
            "solver_terminated", "solver_success", "match_success", "complete_cell_accepted"} <= set(fields)
    assert fields["residuals"]["items"]["properties"]["passed"]["type"] == "boolean"
    assert "endpoint" in fields["measurement"]["properties"]
