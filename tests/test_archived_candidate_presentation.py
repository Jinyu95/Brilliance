"""Presentation uses verified selected evidence, not the lifetime of a UI cache.

Synthetic runtime captures test provenance/rendering only; no lattice-design or
JuTrack-physics result is claimed by these fixtures.
"""
import copy
import json
import math
from pathlib import Path

import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence, result_presentation
from brilliance.research_api import ResearchRun


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    for name in ("BRILLIANCE_RUN_OUTPUT", "BRILLIANCE_WORKSPACE", "BRILLIANCE_SCIENTIFIC_ACCEPTANCE",
                 "BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE"):
        monkeypatch.delenv(name, raising=False)
    candidate_evidence._captured.clear()
    result_presentation._evidence.clear()
    monkeypatch.setattr(api, "load_pyjutrack", lambda: pytest.fail("Reporting reran JuTrack"))
    yield
    candidate_evidence._captured.clear()
    result_presentation._evidence.clear()


def make_run(tmp_path, monkeypatch, name="selected"):
    workspace = tmp_path / "workspace"
    output = workspace / "artifacts" / "programs" / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "program.py").write_text("# Preserved synthetic reporting fixture", encoding="utf-8")
    monkeypatch.setenv("BRILLIANCE_WORKSPACE", str(workspace))
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(output))
    return ResearchRun(workspace=workspace, output_directory=output,
                       arguments_file=output / "arguments.json", result_file=output / "result.json",
                       progress_file=None, cancellation_file=None, working_lattice_file=None,
                       latest_candidate_lattice_file=None, compute_job_id="presentation-test")


def make_candidate(*, strength=.5, full_ring=False):
    rows = [api.drift("D", .5), api.quadrupole("Q", .2, strength),
            api.sector_bend("B", .8, math.tau / 8)]
    if full_ring:
        return api.cell_ring(title="Synthetic repeated ring", energy_gev=3., cell=rows, periods=8,
                             source="Application provenance fixture")
    return api.periodic_cell(title="Synthetic periodic cell", energy_gev=3., elements=rows,
                             intended_repetitions=8, source="Application provenance fixture")


def capture(candidate, *, profile=True):
    period = candidate.total_length_m / (8 if candidate.metadata.get("evaluation_scope") == "full_ring" else 1)
    values = [{"name": "START" if i == 0 else f"point-{i}", "s_m": i * period / 4,
               "beta_x": 2. + i / 10, "beta_y": 3. + i / 10,
               "alpha_x": 0., "alpha_y": 0., "Dx": .001, "Dpx": 0.}
              for i in range(int(candidate.total_length_m / period) * 4 + 1)]
    result = {"engine": "JuTrack", "candidate_lattice_id": candidate.lattice_id,
              "evaluation_scope": candidate.metadata["evaluation_scope"],
              "geometry": {"cell_length_m": period},
              "linear_optics": {"stable": True, "trace_x": .2, "trace_y": .3,
                                "twiss_profile": values if profile else []},
              "radiation": {"computed": True, "emittance_pm": 12., "I2": .1,
                            "I4": .001, "I5": 2e-6}}
    result["evaluation_receipt"] = candidate_evidence.record_evaluation(candidate, result)
    return result


def finish(run, candidate, evaluation):
    run.finish(summary="Synthetic numerical checkpoint", status="partial", request_satisfied=False,
               candidate_lattice=candidate, candidate_evaluation=evaluation)
    return json.loads(run.result_file.read_text(encoding="utf-8"))


def assert_one_cell_plot(run, report, candidate):
    assert report["data"]["candidate_verification"]["passed"]
    optics = [plot for plot in report["plots"] if plot.get("kind") == "optics_lattice"]
    assert len(optics) == 1
    assert (run.output_directory / optics[0]["path"]).read_bytes().startswith(b"\x89PNG")
    manifest = json.loads((run.output_directory / "computed_presentation.json").read_text(encoding="utf-8"))
    expected = sum(row["length_m"] for row in api.one_period_elements(candidate))
    assert manifest["cell_length_m"] == pytest.approx(expected)
    assert manifest["profile"][-1]["s_m"] == pytest.approx(expected)
    assert report["data"]["presentation"]["simulation_rerun"] is False
    return manifest


def test_finish_recovers_best_candidate_optics_after_finalist_cache_eviction(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    selected = make_candidate()
    evaluation = capture(selected)
    result_presentation.remember_candidate_evidence(selected, linear=evaluation["linear_optics"])
    for i in range(result_presentation._MAX_CANDIDATES + 1):
        other = make_candidate(strength=.6 + i / 100)
        result_presentation.remember_candidate_evidence(other, linear=capture(other)["linear_optics"])
    assert selected.lattice_id not in result_presentation._evidence
    manifest = assert_one_cell_plot(run, finish(run, selected, evaluation), selected)
    assert manifest["profile"][0]["beta_x"] == evaluation["linear_optics"]["twiss_profile"][0]["beta_x"]


def test_cross_worker_restored_receipt_plots_one_period_without_simulation(tmp_path, monkeypatch):
    old_run = make_run(tmp_path, monkeypatch, "old")
    selected = make_candidate(full_ring=True)
    evaluation = capture(selected)
    receipt = candidate_evidence.verify_candidate_evaluation(selected, evaluation, old_run.output_directory)["receipt"]
    candidate_evidence._captured.clear()
    result_presentation._evidence.clear()
    new_run = make_run(tmp_path, monkeypatch, "new")
    restored = api.restore_candidate_evaluation(selected, receipt,
        source_output_directory=old_run.output_directory, output_directory=new_run.output_directory)
    manifest = assert_one_cell_plot(new_run, finish(new_run, selected, restored), selected)
    assert manifest["lattice_length_m"] == pytest.approx(8 * manifest["cell_length_m"])


@pytest.mark.parametrize("mode", ["edited_profile", "different_candidate"])
@pytest.mark.parametrize("cached", [False, True])
def test_unverified_caller_data_cannot_supply_canonical_optics(tmp_path, monkeypatch, mode, cached):
    run = make_run(tmp_path, monkeypatch)
    selected = make_candidate()
    evaluation = capture(selected)
    if cached:
        result_presentation.remember_candidate_evidence(selected, linear=evaluation["linear_optics"])
    if mode == "edited_profile":
        evaluation = copy.deepcopy(evaluation)
        evaluation["linear_optics"]["twiss_profile"][0]["beta_x"] = 12345.
    else:
        selected = make_candidate(strength=.9)
    report = finish(run, selected, evaluation)
    assert not report["data"]["candidate_verification"]["passed"]
    assert not report["plots"]
    assert not (run.output_directory / "computed_presentation.json").exists()


def test_no_profile_in_receipt_does_not_invent_an_optics_curve(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    selected = make_candidate()
    report = finish(run, selected, capture(selected, profile=False))
    assert report["data"]["candidate_verification"]["passed"]
    assert not report["plots"]


def test_restoring_archived_optics_retains_existing_exact_candidate_bend_evidence(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    selected = make_candidate()
    evaluation = capture(selected)
    # A failed decomposition is still diagnostic evidence, never a plotted I5 claim.
    bends = {"rows": [{"bend": "B", "I2": .1, "I5": 2e-6}],
             "consistency": {"passed": False}}
    result_presentation.remember_candidate_evidence(selected, bends=bends)
    manifest = assert_one_cell_plot(run, finish(run, selected, evaluation), selected)
    assert manifest["bend_radiation"] == bends
    assert not (run.output_directory / "bend_radiation_contributions.svg").exists()


def test_optional_presentation_write_failure_preserves_verified_result(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    selected = make_candidate()
    evaluation = capture(selected)
    original_write = Path.write_text

    def fail_manifest(path, *args, **kwargs):
        if path.name == "computed_presentation.json":
            raise PermissionError("Intentional presentation-only write failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest)
    report = finish(run, selected, evaluation)
    verification = report["data"]["candidate_verification"]
    assert verification["passed"]
    assert not report["plots"]
    assert "PermissionError" in report["data"]["presentation"]["errors"][0]
    recovered = candidate_evidence.read_numerical_evidence(
        selected, verification["receipt"], output_directory=run.output_directory)
    assert recovered["linear_optics"] == evaluation["linear_optics"]


def test_explicit_plot_is_not_duplicated_by_recovered_profile(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    selected = make_candidate()
    evaluation = capture(selected)
    receipt = candidate_evidence.verify_candidate_evaluation(selected, evaluation, run.output_directory)["receipt"]
    plots = [{"path": "scientist_selected.png", "kind": "optics_lattice"}]
    presentation = result_presentation.present_candidate_evidence(
        selected, run.output_directory, tables=[], plots=plots, candidate_receipt=receipt)
    assert presentation["optics_available"]
    assert presentation["generated_figures"] == []
    assert plots == [{"path": "scientist_selected.png", "kind": "optics_lattice"}]
