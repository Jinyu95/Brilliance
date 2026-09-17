"""Scientific milestones retain lineage; operational messages are not findings."""
import json
from dataclasses import replace

import pytest

from brilliance.research_api import ResearchRun


def make_run(tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "artifacts" / "programs" / "trial"
    output.mkdir(parents=True)
    return ResearchRun(workspace=workspace, output_directory=output,
                       arguments_file=output / "arguments.json", result_file=output / "result.json",
                       progress_file=workspace / "progress" / "trial.json",
                       cancellation_file=None, working_lattice_file=None,
                       latest_candidate_lattice_file=None, compute_job_id="job")


def test_progress_separates_activity_and_scientific_lineage(tmp_path):
    run = make_run(tmp_path)
    run.progress("Starting saved research program", event_key="start")
    assert json.loads(run.progress_file.read_text())['scientific_role'] == "activity"
    run.progress("Matched cell compared with its parent", kind="finding", stage_id="complete-cell",
                 hypothesis_id="matching-A", candidate_id="candidate-B", parent_candidate_id="candidate-A",
                 outcome="feasible", interpretation="Stable, but the objective became worse.",
                 next_decision="Retain A and change the matching allocation.")
    event = json.loads(run.progress_file.read_text())
    assert event["scientific_role"] == "milestone"
    assert event["stage_id"] == "complete-cell"
    assert event["candidate_id"] == "candidate-B"
    assert event["parent_candidate_id"] == "candidate-A"
    assert "worse" in event["interpretation"]
    assert len(list((run.progress_file.parent / "events" / "trial").glob("*.json"))) == 2


@pytest.mark.parametrize("status,passed,expected", [
    ("infeasible", None, "No feasible candidate"),
    ("completed", False, "failed its numerical checks"),
    ("partial", True, "work remains"),
    ("completed", True, "Calculation result"),
])
def test_finish_does_not_announce_failed_or_partial_attempt_as_ready(tmp_path, monkeypatch, status, passed, expected):
    monkeypatch.delenv("BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE", raising=False)
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    run = make_run(tmp_path)
    path = run.finish(summary="An explicit attempt outcome.", status=status, scientific_passed=passed,
                      scientific_stage={"stage_id": "stage-A", "interpretation": "A measured finding.",
                                        "next_decision": "Compare another hypothesis."})
    result = json.loads(path.read_text())
    event = json.loads(run.progress_file.read_text())
    assert expected in event["stage"]
    assert result["scientific_stage"]["stage_id"] == event["stage_id"] == "stage-A"
    assert event["scientific_role"] == "milestone"
    if status == "infeasible" or passed is False:
        assert "Review the failed checks" in event["next_decision"]
        assert result["data"]["submitted_outcome_presentation"]["scientific_stage"]["next_decision"] == (
            "Compare another hypothesis.")
    else:
        assert event["next_decision"] == "Compare another hypothesis."


def test_stage_metadata_validated_without_ui(tmp_path):
    run = replace(make_run(tmp_path), progress_file=None)
    with pytest.raises(ValueError, match="unknown scientific_stage"):
        run.finish(summary="Invalid", scientific_stage={"made_up": True})
