"""No simulation: exact same-project receipts survive a new worker process."""
from __future__ import annotations

import json

import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence as evidence
from brilliance.numerical_checkpoints import read_numerical_checkpoint, write_numerical_checkpoint
from brilliance.research_api import ResearchRun


def prepare(tmp_path, monkeypatch):
    workspace = tmp_path / "project" / "workspace"
    original = workspace / "artifacts" / "programs" / "original"
    destination = original.with_name("repaired")
    original.mkdir(parents=True)
    destination.mkdir()
    (original / "program.py").write_text("# preserved original numerical program\n", encoding="utf-8")
    monkeypatch.setenv("BRILLIANCE_WORKSPACE", str(workspace))
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(original))
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    evidence._captured.clear()
    subject = api.periodic_cell(title="Synthetic provenance", energy_gev=3., intended_repetitions=8,
                               elements=[api.drift("D", 1.)], source="test only")
    measured = {"engine": "JuTrack", "geometry": {"cell_length_m": 1.},
                "linear_optics": {"stable": False}, "radiation": {"computed": False}}
    measured["evaluation_receipt"] = evidence.record_evaluation(subject, measured, effective_periods=8)
    receipt = evidence.verify_candidate_evaluation(subject, measured, original)["receipt"]
    checkpoint = write_numerical_checkpoint("saved_stage", {"candidate": subject.as_dict(), "receipt": receipt})
    evidence._captured.clear()
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(destination))
    for name in ("evaluate", "evaluate_periodic_cell", "jutrack", "load_pyjutrack"):
        monkeypatch.setattr(api, name, lambda *a, **kw: pytest.fail("Recovery must not run numerical work"))
    return workspace, original, destination, subject, measured, receipt, checkpoint


def test_explicit_old_worker_reuse_publishes_exact_original_identity(tmp_path, monkeypatch):
    workspace, original, destination, subject, measured, receipt, checkpoint = prepare(tmp_path, monkeypatch)
    original_bytes = (original / receipt["path"]).read_bytes()
    state = read_numerical_checkpoint(checkpoint, source_output_directory=original)
    assert state["result"]["candidate"] == subject.as_dict()
    restored = api.restore_candidate_evaluation(subject, state["result"]["receipt"], source_output_directory=original)
    assert restored == measured
    run = ResearchRun(workspace=workspace, output_directory=destination,
        arguments_file=destination / "arguments.json", result_file=destination / "result.json",
        progress_file=None, cancellation_file=None, working_lattice_file=None,
        latest_candidate_lattice_file=None, compute_job_id="repair-test")
    run.checkpoint_candidate(subject, restored, name="recovered")
    path = run.finish(summary="Recovered exact prior measurement; no new simulation.",
        candidate_lattice=subject, candidate_evaluation=restored, scientific_passed=False,
        request_satisfied=False, use_as_working_lattice=False)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["candidate_evaluation"]["evaluation_id"] == receipt["evaluation_id"]
    assert evidence.validate_saved_candidate_evaluation(subject, saved["candidate_evaluation"], destination)["passed"]
    assert not saved["request_satisfied"] and not saved["scientific_passed"]
    provenance = json.loads(next((destination / "evidence_reuse").glob("*.json")).read_text(encoding="utf-8"))
    assert provenance["source_receipt"] == receipt and provenance["numerical_rerun"] is False
    assert (original / receipt["path"]).read_bytes() == original_bytes


@pytest.mark.parametrize("location", ["foreign", "workspace", "child", "missing_program"])
def test_reuse_rejects_non_program_or_foreign_archive(tmp_path, monkeypatch, location):
    workspace, original, _, subject, _, receipt, checkpoint = prepare(tmp_path, monkeypatch)
    source = {"foreign": tmp_path / "other-project" / "workspace" / "artifacts" / "programs" / "run",
              "workspace": workspace, "child": original / "child",
              "missing_program": original.with_name("missing")}[location]
    source.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        api.restore_candidate_evaluation(subject, receipt, source_output_directory=source)
    with pytest.raises(ValueError):
        read_numerical_checkpoint(checkpoint, source_output_directory=source)
    assert not evidence._captured


def test_reuse_does_not_trust_changed_metrics_or_checkpoint(tmp_path, monkeypatch):
    _, original, _, subject, _, receipt, checkpoint = prepare(tmp_path, monkeypatch)
    (original / receipt["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        api.restore_candidate_evaluation(subject, receipt, source_output_directory=original)
    (original / checkpoint["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        read_numerical_checkpoint(checkpoint, source_output_directory=original)
    assert not evidence._captured
