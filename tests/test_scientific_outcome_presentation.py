"""Explicit failed numerical checks override prose, not useful partial science."""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance.research_api import ResearchRun
from brilliance.scientific_reporting import normalize_scientific_outcome


@pytest.mark.parametrize("extra,expected", [
    ({"scientific_passed": None}, "not_evaluated"),
    ({"scientific_passed": None, "data": {"reporting_failure": {"message": "plot error"}}}, "reporting_failed"),
    ({"scientific_passed": False}, "numerically_rejected"),
    ({"scientific_passed": True, "request_satisfied": True, "scientific_acceptance": {"passed": True}}, "accepted"),
])
def test_evidence_states_are_distinct(extra, expected):
    reported = normalize_scientific_outcome({"status": "completed", "summary": "test", **extra})
    assert reported["outcome_presentation"]["evidence_status"] == expected


def result(**overrides):
    return {"summary": "The achromat is solved.", "scientific_passed": False,
            "request_satisfied": False, "status": "completed",
            "scientific_stage": {"stage_id": "complete-cell", "outcome": "matched",
                                 "interpretation": "Dispersion suppression was solved.",
                                 "next_decision": "Inspect the residuals before continuing."},
            "data": {"gates": {"stable": True, "id_Dx": False, "boundary_beta": False}},
            "tables": [{"title": "Measured boundary", "rows": [{"Dx_m": .0377, "target_m": 0.0}]}],
            "plots": [{"path": "measured.png"}], "limitations": ["The boundary target was not met."],
            "scientific_acceptance": {"passed": True, "status": "passed", "checks": {}}, **overrides}


def test_failed_cell_flags_override_summary_and_stage_but_retain_real_data():
    original = result()
    snapshot = copy.deepcopy(original)
    actual = normalize_scientific_outcome(original)
    assert actual["outcome_presentation"]["evidence_status"] == "numerically_rejected"
    assert original == snapshot
    assert actual["scientific_stage"]["outcome"] == "checks_failed"
    assert actual["scientific_stage"]["interpretation"] == actual["summary"]
    assert "id_Dx" in actual["summary"] and "boundary_beta" in actual["summary"]
    assert "The achromat is solved." not in actual["summary"]
    assert actual["data"]["submitted_outcome_presentation"]["summary"] == original["summary"]
    assert actual["data"]["submitted_outcome_presentation"]["scientific_stage"] == original["scientific_stage"]
    assert actual["tables"][0] == original["tables"][0]
    assert actual["plots"] == original["plots"]
    assert actual["limitations"] == original["limitations"]
    assert "before advancing" in actual["scientific_stage"]["next_decision"]
    assert actual["status"] == "partial"


def test_successful_intermediate_survives_pending_whole_request():
    original = result(scientific_passed=True, status="partial",
                      summary="The unit was measured successfully; the ring remains pending.",
                      data={"gates": {"unit_stable": True}},
                      scientific_acceptance={"passed": False, "status": "pending", "checks": {
                          "complete_design": {"status": "pending", "passed": False}}})
    actual = normalize_scientific_outcome(original)
    assert actual["summary"] == original["summary"]
    assert actual["scientific_stage"] == original["scientific_stage"]
    assert actual["scientific_passed"] is True
    assert actual["request_satisfied"] is False
    assert "submitted_outcome_presentation" not in actual["data"]
    assert actual["outcome_presentation"]["request_acceptance_status"] == "pending"


def test_failed_candidate_cannot_recommend_a_success_dependent_next_stage():
    original = result(scientific_stage={"stage_id": "complete-cell",
        "interpretation": "The achromat is solved.",
        "next_decision": "Nonlinear stage (sextupoles, chromaticity, dynamic aperture) and unit re-optimization."})
    actual = normalize_scientific_outcome(original)
    assert "Nonlinear stage" not in actual["scientific_stage"]["next_decision"]
    assert "before advancing to a dependent stage" in actual["scientific_stage"]["next_decision"]
    assert "scientist" in actual["scientific_stage"]["next_decision"]
    assert actual["data"]["submitted_outcome_presentation"]["scientific_stage"]["next_decision"] == (
        original["scientific_stage"]["next_decision"])


def test_failure_next_action_policy_is_not_a_topic_or_word_filter():
    original = result(scientific_stage={"next_decision": "Publish the fitted response coefficients."})
    actual = normalize_scientific_outcome(original)
    assert actual["scientific_stage"]["next_decision"].startswith("Review the failed checks")
    assert actual["data"]["submitted_outcome_presentation"]["scientific_stage"]["next_decision"] == (
        "Publish the fitted response coefficients.")


def test_optional_diagnostic_and_historical_trial_failures_do_not_reject_current_result():
    original = result(scientific_passed=True, data={
        "checks": {"optional_diagnostic": {"required": False, "passed": False}},
        "attempts": [{"checks": {"convergence": False}}]})
    actual = normalize_scientific_outcome(original)
    assert actual["summary"] == original["summary"]
    assert actual["scientific_passed"] is True


def test_declared_final_failure_overrides_contradictory_success_flags():
    actual = normalize_scientific_outcome(result(
        scientific_passed=True, request_satisfied=True, use_as_working_lattice=True,
        candidate_lattice="candidate.json", candidate_handoff={"disposition": "working_selection_requested"}))
    assert actual["scientific_passed"] is False
    assert actual["request_satisfied"] is False
    assert actual["use_as_working_lattice"] is False
    assert actual["candidate_handoff"]["disposition"] == "archived_branch"


def test_failed_acceptance_does_not_invalidate_independently_valid_calculation():
    actual = normalize_scientific_outcome(result(scientific_passed=True, data={},
        scientific_acceptance={"passed": False, "status": "failed", "checks": {
            "complete_design": {"status": "failed", "passed": False, "issues": ["boundary mismatch"]}}}))
    assert actual["scientific_passed"] is True
    assert actual["request_satisfied"] is False
    assert "candidate acceptance checks failed" in actual["summary"]
    assert "complete_design" in actual["summary"]


def test_solver_termination_boundary_match_and_complete_cell_are_distinct():
    actual = normalize_scientific_outcome(result(scientific_passed=True, status="partial", data={"match": {
        "scope": "open_line_boundary_match", "solver_terminated": True, "solver_success": True,
        "match_success": False, "complete_cell_accepted": False,
        "complete_cell_acceptance": {"status": "not_evaluated"}}}))
    rows = {row["Check"]: row["Outcome"] for row in actual["tables"][-1]["rows"]}
    assert rows["Numerical solver terminated"] == "passed"
    assert rows["Numerical solver convergence"] == "passed"
    assert rows["Boundary targets matched"] == "failed"
    assert rows["Complete-cell acceptance"] == "not_evaluated"
    assert rows["Complete request acceptance"] == "failed_for_this_candidate"
    assert actual["scientific_passed"] is True  # The calculation can be valid while matching fails.
    assert "Boundary matching did not meet" in actual["summary"]


def test_matched_line_is_valid_intermediate_not_complete_cell_acceptance():
    original = result(scientific_passed=True, status="partial", summary="The line boundary was matched.", data={"match": {
        "scope": "open_line_boundary_match", "solver_terminated": True, "solver_success": False,
        "match_success": True, "complete_cell_accepted": False,
        "complete_cell_acceptance": {"status": "not_evaluated"}}})
    actual = normalize_scientific_outcome(original)
    assert actual["summary"] == original["summary"]
    assert actual["outcome_presentation"]["request_acceptance_status"] == "pending"
    rows = {row["Check"]: row["Outcome"] for row in actual["tables"][-1]["rows"]}
    assert rows["Numerical solver convergence"] == "failed"
    assert rows["Boundary targets matched"] == "passed"
    assert rows["Complete-cell acceptance"] == "not_evaluated"


def test_normalization_is_idempotent_and_never_uses_success_word_filtering():
    actual = normalize_scientific_outcome(result(summary="Neutral diagnostic statement."))
    assert "Neutral diagnostic statement." not in actual["summary"]
    assert normalize_scientific_outcome(actual) == actual
    assert actual["data"]["submitted_outcome_presentation"]["summary"] == "Neutral diagnostic statement."


def test_runtime_status_does_not_create_numerical_evidence_for_empty_unverified_report():
    actual = normalize_scientific_outcome(result(scientific_passed=None, request_satisfied=True,
                                                use_as_working_lattice=True, data={}, tables=[]))
    assert actual["tables"] == []
    assert actual["outcome_presentation"]["request_acceptance_status"] == "unverified"
    assert actual["outcome_presentation"]["heading"] == "Unverified research checkpoint"


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    monkeypatch.delenv("BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE", raising=False)
    workspace = tmp_path / "workspace"
    output = workspace / "artifacts" / "program"
    output.mkdir(parents=True)
    return ResearchRun(workspace=workspace, output_directory=output,
        arguments_file=output / "arguments.json", result_file=output / "result.json",
        progress_file=workspace / "progress" / "test.json", cancellation_file=None,
        working_lattice_file=None, latest_candidate_lattice_file=None, compute_job_id="outcome-test")


def test_finish_publishes_runtime_verdict_instead_of_failed_authored_conclusion(run):
    saved = json.loads(run.finish(summary="The achromat is solved.", scientific_passed=False,
        request_satisfied=False, data={"gates": {"id_Dx": False}},
        tables=[{"title": "Boundary values", "rows": [{"Dx": .0377}]}],
        scientific_stage={"interpretation": "Matched successfully.", "outcome": "matched"}
    ).read_text(encoding="utf-8"))
    event = json.loads(run.progress_file.read_text(encoding="utf-8"))
    assert "id_Dx" in event["detail"]
    assert event["interpretation"] == saved["summary"]
    assert event["outcome"] == "checks_failed"
    assert event["tables"][0]["rows"] == [{"Dx": .0377}]
    assert saved["data"]["submitted_outcome_presentation"]["summary"] == "The achromat is solved."


@pytest.mark.parametrize("status", ["complete", " COMPLETE "])
def test_complete_is_only_a_reporting_spelling_alias(run, status):
    saved = json.loads(run.finish(summary="A valid measured result.", status=status,
        scientific_passed=True, request_satisfied=True, data={"measurement": 2.0}
    ).read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["requested_status"] == "complete"
    assert saved["scientific_passed"] is True
    assert saved["request_satisfied"] is True


def test_complete_spelling_cannot_override_failed_numerical_checks(run):
    saved = json.loads(run.finish(summary="It is matched.", status="complete",
        scientific_passed=True, request_satisfied=True,
        data={"gates": {"boundary_Dx": False}}
    ).read_text(encoding="utf-8"))
    assert saved["status"] == "partial"
    assert saved["requested_status"] == "complete"
    assert saved["scientific_passed"] is False
    assert saved["request_satisfied"] is False
    assert "boundary_Dx" in saved["summary"]


def test_complete_spelling_cannot_override_failed_expert_acceptance(run, monkeypatch):
    from brilliance import scientific_acceptance

    monkeypatch.setattr(scientific_acceptance, "assess_scientific_acceptance", lambda *_args, **_kwargs: {
        "passed": False, "status": "failed", "required": ["complete_candidate"],
        "checks": {"complete_candidate": {"status": "failed", "passed": False,
                                            "issues": ["Boundary target was not met."]}},
    })
    saved = json.loads(run.finish(summary="The candidate meets the request.", status="complete",
        scientific_passed=True, request_satisfied=True, data={"measurement": 2.0}
    ).read_text(encoding="utf-8"))
    assert saved["status"] == "partial"
    assert saved["requested_status"] == "complete"
    assert saved["scientific_passed"] is True
    assert saved["request_satisfied"] is False
    assert "complete_candidate" in saved["summary"]


def test_parent_runner_rechecks_result_even_if_program_rewrites_public_prose(run):
    from brilliance.research_program import run_research_program

    source = run.workspace / "report.py"
    source.write_text('''
import json
from brilliance.research_api import open_research_run
run = open_research_run()
path = run.finish(summary="Candidate failed a boundary condition.", scientific_passed=False,
                  request_satisfied=False, data={"gates": {"boundary": False}})
result = json.loads(path.read_text(encoding="utf-8"))
result["summary"] = "The achromat is solved."
result["scientific_stage"]["interpretation"] = "All targets matched."
path.write_text(json.dumps(result), encoding="utf-8")
''', encoding="utf-8")

    def process_environment(*_args, **_kwargs):
        return {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                "BRILLIANCE_INTERNET_ACCESS": "false"}

    context = SimpleNamespace(workspace=run.workspace, project_id="outcome-only", compute_job_id=None,
        project=SimpleNamespace(active_lattice_id=None), active_lattice=lambda: None,
        is_cancellation_requested=lambda: False,
        artifact_path=lambda name: run.workspace / "artifacts" / name,
        environment=SimpleNamespace(python_executable=sys.executable,
                                    project_process_environment=process_environment))
    execution = run_research_program(context, {"script": source.name, "timeout_s": 30})
    assert execution["returncode"] == 0
    assert "The achromat is solved." not in execution["result"]["summary"]
    assert "boundary" in execution["result"]["summary"]
    assert execution["result"]["scientific_stage"]["interpretation"] == execution["result"]["summary"]
    assert execution["working_lattice_action"] == "branch"
