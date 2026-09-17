"""Same-run evidence subdirectories preserve run identity and parent validation."""
from __future__ import annotations

import json

import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence as evidence


@pytest.fixture(autouse=True)
def isolated_capture(monkeypatch):
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    evidence._captured.clear()
    yield
    evidence._captured.clear()


def candidate():
    return api.periodic_cell(title="Synthetic measurement evidence", energy_gev=3.,
        elements=[api.drift("D", 1.)], intended_repetitions=8, source="application evidence test")


def test_managed_subdirectory_returns_run_relative_receipt_and_parent_can_read(tmp_path, monkeypatch):
    output = tmp_path / "program-current"
    storage = output / "mba_verification"
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(output))
    lattice = candidate()
    calculation = {"schema": "synthetic_measurement/v1", "value": 12.5}
    receipt = evidence.record_numerical_evidence(lattice, calculation, output_directory=storage)
    assert receipt["path"].startswith("mba_verification/candidate_evaluations/")
    archive = output / receipt["path"]
    original = archive.read_bytes()
    assert evidence.read_numerical_evidence(lattice, receipt, output_directory=storage) == calculation
    assert evidence.validate_saved_candidate_evaluation(lattice, receipt, output)["passed"]
    # Parent process resolves receipts against the managed root, not the child's
    # storage destination. It does not need the producing process's capture.
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT")
    evidence._captured.clear()
    with evidence.numerical_evidence_directory(output):
        assert evidence.read_numerical_evidence(lattice, receipt) == calculation
        assert evidence.read_numerical_evidence(lattice, receipt, output_directory=storage) == calculation
    assert archive.read_bytes() == original


def test_hosted_standalone_measurement_can_use_child_storage(tmp_path):
    lattice = candidate()
    with evidence.numerical_evidence_directory(tmp_path):
        receipt = evidence.record_numerical_evidence(lattice, {"value": 2}, output_directory=tmp_path / "checks")
        assert receipt["path"].startswith("checks/candidate_evaluations/")
        assert evidence.read_numerical_evidence(lattice, receipt) == {"value": 2}


@pytest.mark.parametrize("target", ["sibling", "parent", "traversal"])
def test_managed_storage_cannot_escape_current_root(tmp_path, monkeypatch, target):
    output = tmp_path / "program-current"
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(output))
    destination = {"sibling": tmp_path / "program-other",
                   "parent": tmp_path,
                   "traversal": output / "child" / ".." / ".." / "program-other"}[target]
    with pytest.raises(ValueError, match="cannot override"):
        evidence.record_numerical_evidence(candidate(), {"value": 1}, output_directory=destination)
    assert not list(tmp_path.rglob("*.json"))


def test_capture_cannot_move_between_managed_runs_even_using_subdirectories(tmp_path, monkeypatch):
    current = tmp_path / "program-current"
    other = tmp_path / "program-other"
    lattice = candidate()
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(current))
    calculation = {"value": 1}
    calculation["evaluation_receipt"] = evidence.record_evaluation(lattice, calculation)
    rejected = evidence.verify_candidate_evaluation(lattice, calculation, other / "checks")
    assert not rejected["passed"]
    assert "leaves" in rejected["issues"][0]
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(other))
    rejected = evidence.verify_candidate_evaluation(lattice, calculation, other / "checks")
    assert not rejected["passed"]
    assert "different managed research run" in rejected["issues"][0]
    assert not list(tmp_path.rglob("*.json"))


def test_subdirectory_receipt_still_rejects_tampering_and_wrong_host(tmp_path, monkeypatch):
    output = tmp_path / "program-current"
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(output))
    lattice = candidate()
    receipt = evidence.record_numerical_evidence(lattice, {"value": 1}, output_directory=output / "checks")
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT")
    with evidence.numerical_evidence_directory(tmp_path / "program-other"):
        with pytest.raises(ValueError, match="cannot override"):
            evidence.read_numerical_evidence(lattice, receipt, output_directory=output)
        with pytest.raises(ValueError):
            evidence.read_numerical_evidence(lattice, receipt)
    archive = output / receipt["path"]
    payload = json.loads(archive.read_text(encoding="utf-8"))
    payload["calculation"]["value"] = 99
    archive.write_text(json.dumps(payload), encoding="utf-8")
    with evidence.numerical_evidence_directory(output):
        with pytest.raises(ValueError, match="SHA-256"):
            evidence.read_numerical_evidence(lattice, receipt)
