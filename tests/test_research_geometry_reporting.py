"""Geometry/reporting boundaries only; no physics solver or model calls."""
import json
import math
from dataclasses import replace

import pytest

from brilliance import accelerator_api as api
from brilliance.research_api import ResearchRun


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


def ring():
    # Synthetic length-accounting fixture, NOT a design seed or stable lattice.
    return api.cell_ring(title="Geometry test", energy_gev=3.5, periods=32,
                         cell=[api.drift("D", 23.94), api.sector_bend("B", 1.0, math.tau / 32)],
                         source="synthetic geometry test")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_progress_distinguishes_required_and_actual_circumference(run):
    candidate = ring()
    run.progress("Assembly geometry", candidate_lattice=candidate,
                 metrics={"target_circumference_m": 800.0, "ring_circumference_m": 798.08})
    saved = read(run.progress_file)
    assert saved["candidate_id"] == candidate.lattice_id
    assert saved["geometry_metric_comparison"]["passed"]
    assert saved["candidate_geometry"]["canonical_metrics"]["cell_length_m"] == pytest.approx(24.94)
    assert saved["candidate_geometry"]["canonical_metrics"]["ring_circumference_m"] == pytest.approx(798.08)
    assert saved["metrics"]["target_circumference_m"] == 800
    assert "scientific_passed" not in saved


def test_progress_quarantines_contradictory_numeric_report_without_stopping_code(run):
    run.progress("Matched 800 m ring", detail="Claim based on a target, not rows.",
                 candidate_lattice=ring(), metrics={"circumference_m": 800},
                 tables=[{"title": "Model-written table", "rows": [{"C": 800}]}])
    saved = read(run.progress_file)
    assert saved["outcome"] == "reporting_conflict"
    assert "Matched 800" not in saved["stage"]
    assert saved["metrics"] == {}
    assert saved["submitted_report"]["stage"] == "Matched 800 m ring"
    assert saved["submitted_report"]["metrics"]["circumference_m"] == 800
    assert len(saved["tables"]) == 1
    assert saved["tables"][0]["kind"] == "candidate_geometry"
    run.progress("Next code executes")  # Reporting conflict is not a scientific crash.


def test_finish_checks_geometry_even_without_a_successful_jutrack_evaluation(run):
    path = run.finish(summary="Matched 800 m ring", candidate_lattice=ring(),
                      data={"geometry": {"circumference_m": 800}},
                      scientific_passed=True, request_satisfied=True, use_as_working_lattice=True)
    saved = read(path)
    assert not saved["data"]["geometry_metric_comparison"]["passed"]
    assert saved["data"]["candidate_geometry"]["canonical_metrics"]["ring_circumference_m"] == pytest.approx(798.08)
    assert saved["data"]["submitted_presentation"]["summary"] == "Matched 800 m ring"
    assert "Matched 800" not in saved["summary"]
    assert saved["request_satisfied"] is False
    assert saved["use_as_working_lattice"] is False
    assert saved["status"] == "partial"
    assert any(table.get("kind") == "candidate_geometry" for table in saved["tables"])


def test_repeated_progress_replaces_geometry_table_not_duplicates(run):
    candidate = ring()
    for _ in range(2):
        run.progress("Geometry", event_key="geometry", candidate_lattice=candidate)
    assert len(read(run.progress_file)["tables"]) == 1


def test_budget_announcement_is_not_execution_or_feasibility_evidence(run):
    controls = run.optimization_options("campaign")
    controls["progress_callback"]({"phase": "budget", "summary": "500 particles, 80 generations",
                                   "population_size": 500, "total_generations": 80})
    saved = read(run.progress_file)
    assert saved["kind"] == "decision"
    assert saved["event_key"] == "optimization-budget:campaign"
    assert not (run.output_directory / "optimizer_evidence.json").exists()
    controls["progress_callback"]({"phase": "generation", "generation": 0,
                                   "population_size": 500, "total_generations": 80,
                                   "feasible_count": 0, "front_size": 10})
    assert (run.output_directory / "optimizer_evidence.json").exists()
    assert len(list((run.progress_file.parent / "events" / "report").glob("*.json"))) == 2


def test_headless_finish_also_checks_geometry(run):
    run = replace(run, progress_file=None)
    result = read(run.finish(summary="Incorrect", candidate_lattice=ring(),
                             data={"cell_length_m": 25}, request_satisfied=True))
    assert result["request_satisfied"] is False
    assert not result["data"]["geometry_metric_comparison"]["passed"]


def test_latest_untyped_comparison_does_not_present_target_as_measurement(run):
    tables = [{"title": "Complete cell versus explicit ring", "rows": [
        {"quantity": "circumference", "unit": "m", "cell_scope_32_repeats": 800, "ring": None}]}]
    path = run.finish(summary="Matched 800 m ring", candidate_lattice=ring(),
                      tables=tables, scientific_passed=False, request_satisfied=False)
    saved = read(path)
    assert saved["data"]["geometry_metric_comparison"]["unverified_claims"]
    assert saved["data"]["submitted_presentation"]["tables"] == tables
    assert "Matched 800" not in saved["summary"]
    assert all(table["title"] != tables[0]["title"] for table in saved["tables"])
    assert saved["data"]["candidate_geometry"]["canonical_metrics"]["ring_circumference_m"] == pytest.approx(798.08)


def test_progress_without_candidate_does_not_certify_geometry_table(run):
    run.progress("Built 800 m", detail="Unbound claim", tables=[{"rows": [
        {"quantity": "circumference", "unit": "m", "value": 800}]}])
    saved = read(run.progress_file)
    assert saved["outcome"] == "unverified_geometry"
    assert saved["tables"] == []
    assert saved["submitted_report"]["tables"][0]["rows"][0]["value"] == 800
    assert "Built 800" not in saved["stage"]


def test_explicit_contract_does_not_require_a_lattice(run):
    table = {"kind": "contract", "rows": [{"quantity": "circumference", "unit": "m", "value": 800}]}
    run.progress("Machine contract", kind="decision", tables=[table])
    assert read(run.progress_file)["tables"] == [table]
