"""Parent-side geometry checks; synthetic receipts, no JuTrack/model runs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence as evidence
from brilliance import research_program


@pytest.fixture(autouse=True)
def isolated_evidence(monkeypatch):
    for key in ("BRILLIANCE_RUN_OUTPUT", "BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE",
                "BRILLIANCE_SCIENTIFIC_ACCEPTANCE"):
        monkeypatch.delenv(key, raising=False)
    evidence._captured.clear()
    yield
    evidence._captured.clear()


def ring():
    return api.cell_ring(title="Synthetic geometry fixture", energy_gev=3.5,
        cell=[api.drift("D", 24.), api.sector_bend("B", .94, 2 * math.pi / 32)],
        periods=32, source="application test; not a numerical design")


def result_for(directory, candidate, claim):
    directory.mkdir(parents=True, exist_ok=True)
    candidate_path = directory / "candidate.json"
    candidate_path.write_text(json.dumps(candidate.as_dict()), encoding="utf-8")
    # Exact synthetic receipt isolates the geometry check from numerical
    # provenance checks. No beam-physics result is produced by this test.
    calculation = {"engine": "JuTrack", "candidate_lattice_id": candidate.lattice_id,
        "evaluation_scope": "full_ring", "periods": 32,
        "geometry": {"circumference_m": candidate.total_length_m},
        "linear_optics": {"stable": True}, "radiation": {"computed": True, "emittance_pm": 10.}}
    calculation["evaluation_receipt"] = evidence.record_evaluation(candidate, calculation, effective_periods=32)
    receipt = evidence.verify_candidate_evaluation(candidate, calculation, directory)["receipt"]
    result = {"status": "completed", "summary": "Synthetic candidate report",
        "scientific_passed": True, "request_satisfied": True, "use_as_working_lattice": True,
        "candidate_lattice": candidate_path.name, "candidate_evaluation": receipt, "data": {}, "tables": []}
    if claim == "data":
        result["data"] = {"geometry": {"ring_circumference_m": 800.}}
    elif claim == "table":
        result["tables"] = [{"kind": "candidate_geometry", "lattice_id": candidate.lattice_id,
            "title": "Wrong claimed geometry", "rows": [{"metric": "ring_circumference_m", "value": 800.}]}]
    elif claim == "untyped":
        result["tables"] = [
            {"title": "Cell and ring comparison", "rows": [
                {"quantity": "circumference", "unit": "m", "cell_scope_32_repeats": 800., "ring": None}]},
            {"title": "Other diagnostics", "rows": [{"observations": 1}]},
        ]
    elif claim == "target":
        result["data"] = {"contract": {"ring_circumference_m": 800.},
                          "geometry": {"ring_circumference_m": 798.08}}
    elif claim == "tampered_snapshot":
        result["data"] = {"candidate_geometry": {"passed": True,
            "canonical_metrics": {"ring_circumference_m": 800.}}}
    return result


@pytest.mark.parametrize("claim", ["data", "table", "untyped", "target", "tampered_snapshot"])
def test_validator_recomputes_geometry_even_with_valid_numerical_receipt(tmp_path, claim):
    candidate = ring()
    result = result_for(tmp_path, candidate, claim)
    source = tmp_path / "program.py"
    source.write_text("# Synthetic archived source\n", encoding="utf-8")
    execution = {"output_directory": str(tmp_path), "returncode": 0, "result": result,
        "program": {"snapshot": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
    validation = research_program.validate_research_program(None, {}, execution)
    conflict = claim in {"data", "table"}
    pending = conflict or claim == "untyped"
    assert validation["passed"], validation["issues"]
    assert validation["candidate_verification"]["passed"]
    assert validation["candidate_metric_comparison"]["passed"]
    assert validation["geometry_metric_comparison"]["passed"] is not conflict
    assert validation["candidate_geometry"]["canonical_metrics"]["ring_circumference_m"] == pytest.approx(798.08)
    assert validation["request_satisfied"] is not pending
    assert validation["scientific_passed"] is (None if pending else True)
    if claim == "untyped":
        assert validation["geometry_metric_comparison"]["unverified_claims"][0]["table_index"] == 0
    assert (tmp_path / "candidate.json").is_file()


@pytest.mark.parametrize("claim", ["data", "table", "untyped", "target", "tampered_snapshot"])
def test_runner_rechecks_result_file_and_never_promotes_geometry_conflict(tmp_path, monkeypatch, claim):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "report.py"
    source.write_text('from brilliance.research_api import open_research_run\n'
                      'open_research_run().finish(summary="Synthetic report")\n', encoding="utf-8")
    candidate = ring()

    class CompletedProgram:
        pid = 12345
        returncode = 0

        def __init__(self, _command, *, env, **kwargs):
            # Simulate a program bypassing finish(), or modifying its result
            # afterwards. The real parent result-file boundary is exercised.
            output = Path(env["BRILLIANCE_RUN_OUTPUT"])
            payload = result_for(output, candidate, claim)
            Path(env["BRILLIANCE_RUN_RESULT"]).write_text(json.dumps(payload), encoding="utf-8")

        def poll(self):
            return 0

    monkeypatch.setattr(research_program.subprocess, "Popen", CompletedProgram)
    context = SimpleNamespace(workspace=workspace, project_id="geometry-test", compute_job_id=None,
        project=SimpleNamespace(active_lattice_id=None), active_lattice=lambda: None,
        is_cancellation_requested=lambda: False,
        artifact_path=lambda name: workspace / "artifacts" / name,
        environment=SimpleNamespace(python_executable="configured-python",
                                    project_process_environment=lambda *a, **k: dict(os.environ)))
    execution = research_program.run_research_program(context, {"script": source.name})
    result = execution["result"]
    conflict = claim in {"data", "table"}
    pending = conflict or claim == "untyped"
    assert execution["returncode"] == 0
    assert execution["working_lattice_action"] == ("branch" if pending else "advance")
    assert result["request_satisfied"] is not pending
    assert result["use_as_working_lattice"] is not pending
    assert result["status"] == ("partial" if pending else "completed")
    assert result["data"]["candidate_geometry"]["canonical_metrics"]["ring_circumference_m"] == pytest.approx(798.08)
    assert result["data"]["geometry_metric_comparison"]["passed"] is not conflict
    assert result["tables"][0]["kind"] == "candidate_geometry"
    assert Path(execution["artifacts"]["lattice_json"]).is_file()
    if pending:
        assert result["scientific_passed"] is None
        assert result["data"]["candidate_verification"]["passed"]
        assert result["data"]["submitted_presentation"]["summary"] == "Synthetic candidate report"
    if conflict:
        assert "needs correction" in result["summary"]
        assert len(result["tables"]) == 1
    if claim == "untyped":
        assert "not bound" in result["summary"]
        assert len(result["tables"]) == 2
        assert result["tables"][1]["title"] == "Other diagnostics"
        assert result["data"]["submitted_presentation"]["tables"][0]["rows"][0]["cell_scope_32_repeats"] == 800.
