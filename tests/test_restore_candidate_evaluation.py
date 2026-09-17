"""Restore trusted saved-native receipts, not model-authored metrics or designs."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence as evidence
from brilliance.numerical_checkpoints import read_numerical_checkpoint
from brilliance.research_api import ResearchRun


@pytest.fixture(autouse=True)
def no_native_execution(monkeypatch):
    for key in ("BRILLIANCE_RUN_OUTPUT", "BRILLIANCE_SCIENTIFIC_ACCEPTANCE", "BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE"):
        monkeypatch.delenv(key, raising=False)
    evidence._captured.clear()
    for name in ("evaluate", "evaluate_periodic_cell", "jutrack", "load_pyjutrack", "build_jutrack_lattice"):
        monkeypatch.setattr(api, name, lambda *a, **kw: pytest.fail("Archive restoration must not start a simulation"))
    yield
    evidence._captured.clear()


def candidate(*, strength=.7346199519999999, energy=3.5, periods=8):
    return api.periodic_cell(title="Synthetic receipt restoration test", energy_gev=energy, intended_repetitions=periods,
        elements=[api.drift("D", .23123456789123), api.quadrupole("Q", .2, strength),
                  api.sector_bend("B", .8, 2 * math.pi / 8)], source="provenance test, not beam physics")


def archive(root, subject=None, *, stable=True):
    subject = subject or candidate()
    calculation = {"engine": "JuTrack", "candidate_lattice_id": subject.lattice_id,
        "geometry": {"cell_length_m": subject.total_length_m},
        "linear_optics": {"stable": stable, "Dx_m": .001234567891234},
        "radiation": {"computed": True, "emittance_pm": 42.8202012356789,
                      "physical_validity": {"passed": stable}}}
    calculation["evaluation_receipt"] = evidence.record_evaluation(subject, calculation, effective_periods=8)
    verification = evidence.verify_candidate_evaluation(subject, calculation, root)
    assert verification["passed"]
    return subject, calculation, verification["receipt"]


def test_restores_original_descriptor_exact_numbers_and_reuses_immutable_archive(tmp_path):
    subject, original, receipt = archive(tmp_path)
    path = tmp_path / receipt["path"]
    before = path.read_bytes()
    evidence._captured.clear()
    assert not evidence.verify_candidate_evaluation(subject, original, tmp_path)["passed"]
    restored = api.restore_candidate_evaluation(subject.as_dict(), receipt, output_directory=tmp_path)
    assert restored == original
    assert evidence.verify_candidate_evaluation(subject, restored, tmp_path)["receipt"] == receipt
    assert path.read_bytes() == before
    assert list(tmp_path.rglob("*.json")) == [path]
    record = evidence._captured[receipt["evaluation_id"]]
    assert record["restored_from_receipt"] == receipt
    restored["radiation"]["emittance_pm"] = 1.
    assert not evidence.verify_candidate_evaluation(subject, restored, tmp_path)["passed"]
    assert api.restore_candidate_evaluation(subject, receipt, output_directory=tmp_path) == original


def test_restored_candidate_supports_normal_checkpoint_and_publication(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    output = workspace / "artifacts" / "programs" / "restored"
    output.mkdir(parents=True)
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(output))
    subject, _, receipt = archive(output)
    evidence._captured.clear()
    restored = api.restore_candidate_evaluation(subject, receipt)
    run = ResearchRun(workspace=workspace, output_directory=output, arguments_file=output / "arguments.json",
        result_file=output / "result.json", progress_file=None, cancellation_file=None,
        working_lattice_file=None, latest_candidate_lattice_file=None, compute_job_id="restore-test")
    checkpoint = run.checkpoint_candidate(subject, restored, name="resumed_native_unit")
    payload = read_numerical_checkpoint(checkpoint, output_directory=output)["result"]
    assert payload["candidate_evaluation"] == receipt
    finished = run.finish(summary="Reused the exact archived native unit evaluation; no numerical rerun.",
                          candidate_lattice=subject, candidate_evaluation=restored,
                          scientific_passed=False, request_satisfied=False, use_as_working_lattice=False)
    result = json.loads(finished.read_text(encoding="utf-8"))
    assert result["candidate_evaluation"] == receipt
    assert result["scientific_passed"] is False and result["request_satisfied"] is False


@pytest.mark.parametrize("changed", [candidate(strength=.8), candidate(energy=3.), candidate(periods=16)])
def test_modified_candidate_energy_strength_or_partition_cannot_reuse_receipt(tmp_path, changed):
    _, _, receipt = archive(tmp_path)
    evidence._captured.clear()
    with pytest.raises(ValueError, match="different snapshots"):
        api.restore_candidate_evaluation(changed, receipt, output_directory=tmp_path)
    assert not evidence._captured


@pytest.mark.parametrize("key,value", [("sha256", "0" * 64), ("content_sha256", "0" * 64),
                                      ("evaluation_id", "invented"), ("path", "../outside.json")])
def test_edited_receipt_fields_rejected_before_hydration(tmp_path, key, value):
    subject, _, receipt = archive(tmp_path)
    evidence._captured.clear()
    with pytest.raises(ValueError):
        api.restore_candidate_evaluation(subject, {**receipt, key: value}, output_directory=tmp_path)
    assert not evidence._captured


def test_changed_archive_bytes_rejected_even_with_updated_file_digest(tmp_path):
    subject, _, receipt = archive(tmp_path)
    path = tmp_path / receipt["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calculation"]["radiation"]["emittance_pm"] = 1.
    path.write_text(json.dumps(payload), encoding="utf-8")
    changed = {**receipt, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    evidence._captured.clear()
    with pytest.raises(ValueError, match="invalid identity"):
        api.restore_candidate_evaluation(subject, changed, output_directory=tmp_path)
    assert not evidence._captured


def test_other_project_or_host_context_cannot_override_managed_root(tmp_path, monkeypatch):
    other = tmp_path / "other_project"
    subject, _, receipt = archive(other)
    evidence._captured.clear()
    current = tmp_path / "current_project"
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(current))
    with pytest.raises(ValueError, match="cannot leave"):
        api.restore_candidate_evaluation(subject, receipt, output_directory=other)
    with evidence.numerical_evidence_directory(other), pytest.raises(ValueError):
        api.restore_candidate_evaluation(subject, receipt)
    assert not evidence._captured


def test_child_storage_receipt_remains_managed_root_relative(tmp_path, monkeypatch):
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(tmp_path))
    subject, original, receipt = archive(tmp_path / "selected")
    evidence._captured.clear()
    restored = api.restore_candidate_evaluation(subject, receipt, output_directory=tmp_path / "selected")
    assert restored == original
    assert evidence.verify_candidate_evaluation(subject, restored, tmp_path / "selected")["receipt"] == receipt


def test_no_raw_metrics_or_non_evaluation_measurement_restoration(tmp_path):
    subject, original, _ = archive(tmp_path)
    evidence._captured.clear()
    with pytest.raises(ValueError):
        api.restore_candidate_evaluation(subject, original, output_directory=tmp_path)
    raw = {"engine": "JuTrack", "analysis_table": [{"value": 1.0}]}
    raw["evaluation_receipt"] = evidence.record_evaluation(subject, raw)
    receipt = evidence.verify_candidate_evaluation(subject, raw, tmp_path)["receipt"]
    evidence._captured.clear()
    with pytest.raises(ValueError, match="not a high-level"):
        api.restore_candidate_evaluation(subject, receipt, output_directory=tmp_path)
    assert not evidence._captured


def test_restoration_does_not_upgrade_failed_physics_and_requires_archive_root(tmp_path):
    subject, original, receipt = archive(tmp_path, stable=False)
    evidence._captured.clear()
    with pytest.raises(ValueError, match="requires a managed or explicit"):
        api.restore_candidate_evaluation(subject, receipt)
    restored = api.restore_candidate_evaluation(subject, receipt, output_directory=tmp_path)
    assert restored == original and restored["linear_optics"]["stable"] is False
    assert restored["radiation"]["physical_validity"]["passed"] is False


def test_restoring_existing_capture_does_not_invalidate_original_result(tmp_path):
    subject, original, receipt = archive(tmp_path)
    before = copy.deepcopy(original)
    restored = api.restore_candidate_evaluation(subject, receipt, output_directory=tmp_path)
    assert original == before == restored
    assert evidence.verify_candidate_evaluation(subject, original, tmp_path)["passed"]
