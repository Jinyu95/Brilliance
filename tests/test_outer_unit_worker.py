"""Opt-in exact-current-unit numerical and normal worker-publication integration."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance.candidate_evidence import validate_saved_candidate_evaluation
from brilliance.research_environment import ResearchEnvironment
from brilliance.research_program import (
    inspect_research_program,
    run_research_program,
    validate_research_program,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research_projects/project-4e7dc8072920a1/workspace/stage_a"
PROGRAM = Path(__file__).with_name("fixtures") / "outer_unit_worker_program.py"


def test_worker_example_is_public_api_valid(tmp_path):
    shutil.copyfile(PROGRAM, tmp_path / "program.py")
    inspected = inspect_research_program(tmp_path, "program.py")
    assert inspected["passed"], inspected["issues"]


@pytest.mark.skipif(os.environ.get("BRILLIANCE_RUN_OUTER_UNIT_TESTS") != "1", reason="Opt-in real saved-unit complete-ring validation")
@pytest.mark.parametrize("fail_reporting", [False, True])
def test_actual_unit_to_matched_mba_and_app_result(tmp_path, monkeypatch, fail_reporting):
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="outer-unit worker validation")
    assert environment.internet_access is False
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    name = "project-outer-units-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    workspace = Path(os.environ.get("BRILLIANCE_OUTER_UNIT_OUTPUT", str(tmp_path))) / name / "workspace"
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True)
    shutil.copyfile(SOURCE / "unit.json", inputs / "unit.json")
    shutil.copyfile(SOURCE / "machine_contract.json", inputs / "contract.json")
    shutil.copyfile(PROGRAM, workspace / "program.py")
    context = SimpleNamespace(workspace=workspace, project_directory=workspace.parent, project_id=name,
        compute_job_id="outer-unit-validation", project=SimpleNamespace(active_lattice_id=None),
        active_lattice=lambda: None, is_cancellation_requested=lambda: False,
        artifact_path=lambda name: workspace / "artifacts" / name, environment=environment)
    arguments = {"script": "program.py", "arguments": {"fail_reporting": fail_reporting}, "timeout_s": 600}
    execution = run_research_program(context, arguments, durable=True)
    print("OUTER_UNIT_EXECUTION", execution["output_directory"], flush=True)
    assert execution["returncode"] == 0, execution
    validation = validate_research_program(context, arguments, execution)
    assert validation["passed"], validation
    result = execution["result"]
    output = Path(execution["output_directory"])
    candidate = json.loads((output / result["candidate_lattice"]).read_text(encoding="utf-8"))
    assert validate_saved_candidate_evaluation(candidate, result["candidate_evaluation"], output)["passed"]
    assert (output / "explicit_ring.json").is_file()
    assert (output / "matched_cell.json").is_file()
    assert result["request_satisfied"] is False  # no fourth-generation optimization claim
    if fail_reporting:
        assert result["status"] == "partial"
        assert result["scientific_stage"]["outcome"] == "reporting_pending"
        assert "Intentional reporting failure" in result["data"]["reporting_failure"]["message"]
        assert (output / result["data"]["numerical_checkpoint"]["path"]).is_file()
        assert result["use_as_working_lattice"] is False
    else:
        match = result["data"]["match"]
        assert match["matched"] and match["evaluations"] < 2000
        assert result["data"]["proof"]["passed"]
        assert (output / "one_cell_optics.svg").is_file()
        assert len(result["tables"]) >= 2
