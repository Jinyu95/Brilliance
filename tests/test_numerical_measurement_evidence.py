"""Supplemental measurements use the same exact-snapshot archive as core results."""
import json

import pytest

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import (
    numerical_evidence_directory,
    read_numerical_evidence,
    record_numerical_evidence,
)


def candidate(length=1.0):
    return api.periodic_cell(title="Evidence scope", energy_gev=3.0,
                             elements=[api.drift("D", length)], intended_repetitions=8,
                             source="measurement_evidence_test")


def test_measurement_capture_is_detached_and_replayable(tmp_path, monkeypatch):
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(tmp_path))
    cell = candidate()
    data = {"schema": "custom_measurement/v1", "observations": [{"Dx": .1}]}
    receipt = record_numerical_evidence(cell, data)
    data["observations"][0]["Dx"] = .9
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT")
    with numerical_evidence_directory(tmp_path):
        restored = read_numerical_evidence(cell, receipt)
    assert restored["observations"][0]["Dx"] == .1
    assert restored["schema"] == "custom_measurement/v1"


def test_wrong_candidate_directory_and_changed_measurements_fail(tmp_path, monkeypatch):
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    cell = candidate()
    receipt = record_numerical_evidence(cell, {"value": 3}, output_directory=tmp_path)
    with pytest.raises(ValueError, match="different snapshots"):
        read_numerical_evidence(candidate(2), receipt, output_directory=tmp_path)
    with pytest.raises(ValueError):
        read_numerical_evidence(cell, receipt, output_directory=tmp_path / "other_run")
    path = tmp_path / receipt["path"]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["calculation"]["value"] = 0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        read_numerical_evidence(cell, receipt, output_directory=tmp_path)


def test_no_implicit_archive_directory_or_path_traversal(tmp_path, monkeypatch):
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    with pytest.raises(ValueError, match="output directory"):
        record_numerical_evidence(candidate(), {"value": 1})
    receipt = record_numerical_evidence(candidate(), {"value": 1}, output_directory=tmp_path)
    receipt["path"] = "../other.json"
    with pytest.raises(ValueError, match="leaves"):
        read_numerical_evidence(candidate(), receipt, output_directory=tmp_path)


def test_report_adapter_cannot_override_host_archive_root(tmp_path, monkeypatch):
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    receipt = record_numerical_evidence(candidate(), {"value": 1}, output_directory=tmp_path)
    with numerical_evidence_directory(tmp_path / "another_run"):
        with pytest.raises(ValueError, match="cannot override"):
            read_numerical_evidence(candidate(), receipt, output_directory=tmp_path)
