"""Presentation labels must not abort calculations or imply scientific acceptance."""
from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance.public_api_lint import lint_public_api_calls
from brilliance.research_api import ProgressInputError, ResearchRun


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    monkeypatch.delenv("BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE", raising=False)
    workspace = tmp_path / "workspace"
    output = workspace / "artifacts" / "programs" / "report"
    output.mkdir(parents=True)
    return ResearchRun(workspace=workspace, output_directory=output,
        arguments_file=output / "arguments.json", result_file=output / "result.json",
        progress_file=workspace / "progress" / "report.json", cancellation_file=None,
        working_lattice_file=None, latest_candidate_lattice_file=None, compute_job_id="job")


def event(run):
    return json.loads(run.progress_file.read_text(encoding="utf-8"))


def test_actual_terminal_scan_reporting_call_does_not_abort(run):
    # Reporting-only regression: these are saved scan counts, not a new simulation.
    run.progress("Stage 4a: bounded stable-start scan for the terminal",
                 kind="activity", scientific_role="activity", stage_id="terminal-match",
                 detail="Baseline scan of 64 terminal quadrupole combinations: 7 stable cells.",
                 metrics={"scan_trials": 64, "scan_stable": 7, "assembly_stable": False})
    saved = event(run)
    assert saved["kind"] == "progress"
    assert saved["requested_kind"] == "activity"
    assert saved["scientific_role"] == "activity"
    assert saved["metrics"]["scan_stable"] == 7
    assert saved["stage_id"] == "terminal-match"
    assert saved["outcome"] is None
    assert not run.result_file.exists()
    assert run.working_lattice_file is None
    # A later stage can now execute; no valid physics is fabricated by the label.
    run.progress("Terminal matching remains to be done", kind="next")
    assert event(run)["stage"] == "Terminal matching remains to be done"


@pytest.mark.parametrize("kind,expected,role", [
    ("plan", "plan", "milestone"), ("progress", "progress", "activity"),
    ("finding", "finding", "milestone"), ("decision", "decision", "milestone"),
    ("repair", "repair", "activity"), ("next", "next", "milestone"),
    (" Activity ", "progress", "activity"), ("MILESTONE", "finding", "milestone"),
    (" Finding ", "finding", "milestone"), ("", "progress", "activity"),
])
def test_normalized_labels_have_consistent_roles(run, kind, expected, role):
    run.progress("Evidence", kind=kind)
    assert event(run)["kind"] == expected
    assert event(run)["scientific_role"] == role


@pytest.mark.parametrize("kind", ["info", "accepted", "success", "arbitrary-topic-label"])
def test_unknown_kind_is_recorded_as_neutral_progress_not_success(run, kind):
    run.progress("Update", kind=kind, candidate_id="branch")
    saved = event(run)
    assert saved["kind"] == "progress"
    assert saved["scientific_role"] == "activity"
    assert saved["requested_kind"] == kind
    assert "neutral progress" in saved["presentation_warnings"][0]
    assert saved["candidate_id"] == "branch"
    assert saved["outcome"] is None
    assert "scientific_passed" not in saved
    assert "request_satisfied" not in saved
    assert not run.result_file.exists()


def test_alias_keeps_explicit_role_and_exact_artifacts_across_updates(run):
    plot = run.artifact_path("unit.svg")
    plot.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    tables = [{"title": "Measured values", "rows": [{"value": 3.4261995199999995}]}]
    run.progress("Unit", kind="milestone", scientific_role="activity", event_key="unit",
                 tables=tables, plots=[plot], candidate_id="unit-1")
    first = event(run)
    run.progress("Unit measurement retained", kind="activity", event_key="unit")
    second = event(run)
    assert first["scientific_role"] == "activity"  # Explicit caller role is respected.
    assert first["id"] == second["id"]
    assert second["tables"] == tables
    assert second["plots"] == first["plots"]
    assert len(list((run.progress_file.parent / "events" / "report").glob("*.json"))) == 1


@pytest.mark.parametrize("kwargs", [
    {"scientific_role": "accepted"}, {"event_key": " "},
    {"tables": ["a table"]}, {"tables": [{"rows": {"not": "a list"}}]},
    {"tables": [{"rows": ["not a row"]}]}, {"plots": [42]},
    {"plots": ["missing.svg"]}, {"plots": ["../../../outside.svg"]},
])
def test_malformed_content_has_specific_reporting_error(run, kwargs):
    with pytest.raises(ProgressInputError):
        run.progress("Malformed reporting", **kwargs)
    assert not run.progress_file.exists()
    assert not run.result_file.exists()


def test_access_failure_is_not_swallowed_as_a_label_problem(run, monkeypatch):
    def deny(*args):
        raise PermissionError("access denied")
    monkeypatch.setattr(ResearchRun, "_publish_progress", deny)
    with pytest.raises(PermissionError, match="access denied"):
        run.progress("Update", kind="activity")


def test_disabled_progress_remains_optional(run):
    replace(run, progress_file=None).progress("Update", kind="activity")
    assert not run.progress_file.exists()


@pytest.mark.parametrize("status", ["intermediate", " INTERMEDIATE ", "partial"])
def test_intermediate_finish_remains_partial_even_if_caller_claims_completion(run, status):
    path = run.finish(summary="Unit measured; complete cell remains pending.", status=status,
                      scientific_passed=True, request_satisfied=True)
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "partial"
    assert result["request_satisfied"] is False
    assert result["scientific_passed"] is True
    assert result["use_as_working_lattice"] is False
    assert result["scientific_stage"]["outcome"] == "partial"
    assert "work remains" in event(run)["stage"]
    if status.strip().lower() == "intermediate":
        assert result["requested_status"] == "intermediate"


def test_unknown_result_status_is_not_silently_accepted(run):
    with pytest.raises(ValueError, match="status must"):
        run.finish(summary="Do not turn an unknown status into success", status="accepted")
    assert not run.result_file.exists()


def test_reporting_aliases_pass_public_api_preflight():
    source = '''
from brilliance.research_api import open_research_run
run = open_research_run()
run.progress("Terminal scan", kind="activity", scientific_role="activity")
run.finish(summary="Still working", status="intermediate", request_satisfied=False)
'''
    assert lint_public_api_calls(ast.parse(source)) == []


def test_reporting_aliases_complete_in_the_actual_saved_program_runner(run):
    from brilliance.research_program import run_research_program

    source = run.workspace / "report.py"
    source.write_text('''
from brilliance.research_api import open_research_run
run = open_research_run()
run.progress("Terminal scan reported", kind="activity", scientific_role="activity")
run.finish(summary="Reporting-only replay; no numerical calculation performed.",
           status="intermediate", request_satisfied=False)
''', encoding="utf-8")
    original = source.read_bytes()

    def process_environment(*args, **kwargs):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        environment["BRILLIANCE_INTERNET_ACCESS"] = "false"
        return environment

    context = SimpleNamespace(
        workspace=run.workspace, project_id="reporting-only", compute_job_id=None,
        project=SimpleNamespace(active_lattice_id=None), active_lattice=lambda: None,
        is_cancellation_requested=lambda: False,
        artifact_path=lambda name: run.workspace / "artifacts" / name,
        environment=SimpleNamespace(python_executable=sys.executable,
                                    project_process_environment=process_environment),
    )
    execution = run_research_program(context, {"script": source.name, "timeout_s": 30})
    assert execution["returncode"] == 0
    assert execution["result"]["status"] == "partial"
    assert execution["result"]["request_satisfied"] is False
    assert execution["working_lattice_action"] == "branch"
    assert execution["candidate"] is None
    assert source.read_bytes() == original
    directory = run.progress_file.parent / "events" / execution["run_id"]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")]
    activity = next(row for row in rows if row["stage"] == "Terminal scan reported")
    assert activity["kind"] == "progress" and activity["requested_kind"] == "activity"
