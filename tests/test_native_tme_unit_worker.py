"""Opt-in native deterministic scan through the actual managed worker."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from brilliance.research_environment import ResearchEnvironment
from brilliance.research_program import (
    inspect_research_program,
    run_research_program,
    validate_research_program,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research_projects/project-ab41b8de97b4c4/workspace/artifacts/programs/program-cdc6de955273696dd0f0/candidate_lattice.json"
FINGERPRINT = "7f1948f55ad61417350605625b34d97d5e28f0820318a9c7f5c4de24f29bc893"


def test_unit_program_uses_public_expert_surface_and_durable_metadata(tmp_path):
    shutil.copyfile(Path(__file__).with_name("fixtures") / "native_tme_unit_program.py", tmp_path / "program.py")
    inspection = inspect_research_program(tmp_path, "program.py")
    assert inspection["passed"], inspection["issues"]
    assert inspection["durable_required"] is True
    assert inspection["durable_expert_calls"] == ["design_tme_unit"]


@pytest.mark.skipif(os.environ.get("BRILLIANCE_RUN_TME_WORKER_TESTS") != "1", reason="Opt-in native unit scan")
def test_native_complete_unit_search_and_managed_publication(tmp_path, monkeypatch):
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Native unit search regression")
    assert not environment.internet_access
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == FINGERPRINT
    project_id = "project-tme-test-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    output_root = Path(os.environ.get("BRILLIANCE_TME_TEST_OUTPUT", str(tmp_path)))
    workspace = output_root / project_id / "workspace"
    (workspace / "inputs").mkdir(parents=True)
    shutil.copyfile(SOURCE, workspace / "inputs/unit.json")
    shutil.copyfile(Path(__file__).with_name("fixtures") / "native_tme_unit_program.py", workspace / "program.py")
    context = SimpleNamespace(
        workspace=workspace, project_directory=workspace.parent, project_id=project_id,
        compute_job_id="tme-unit-test", project=SimpleNamespace(active_lattice_id=None),
        active_lattice=lambda: None, is_cancellation_requested=lambda: False,
        artifact_path=lambda name: workspace / "artifacts" / name, environment=environment,
    )
    options = json.loads(os.environ.get("BRILLIANCE_TME_TEST_OPTIONS", "{}"))
    arguments = {"script": "program.py", "arguments": options, "timeout_s": 900}
    execution = run_research_program(context, arguments, durable=True)
    validation = validate_research_program(context, arguments, execution)
    assert execution["returncode"] == 0, execution
    assert validation["passed"], validation
    assert validation["candidate_verification"]["passed"], validation
    output = Path(execution["output_directory"])
    search = json.loads((output / "search.json").read_text(encoding="utf-8"))
    assert search["passed"] and len(search["candidates"]) >= 2
    assert search["optimality_demonstrated"] is False
    assert len(search["resolution_levels"]) >= 2
    original = api.one_period_elements(api.load_lattice(SOURCE))[5]
    for row in search["candidates"]:
        assert all(row["checks"].values()), row["checks"]
        assert len(row["elements"]) == 11
        assert len(row["boundary_twiss"]) == 8
        assert row["elements"][5] == {**original, **({"length_m": options["geometry"]["bend_length_m"]}
                                                     if options.get("geometry") else {})}
        assert row["complete_mba_accepted"] is False
        assert row["numerical_checkpoint"]
        diagnostic = row["bend_centre_diagnostic"]
        assert diagnostic["status"] == "measured", diagnostic
        assert diagnostic["sampling"]["interior_composition_checks"]
        assert len(row["optics_profile"]) > len(row["elements"])
    assert execution["result"]["plots"]
    assert execution["result"]["request_satisfied"] is False
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == FINGERPRINT
    print("NATIVE_TME_SCAN " + json.dumps({
        "output_directory": str(output), "evaluations": search["unique_evaluations"],
        "levels": search["resolution_levels"], "candidates": [{
            key: row[key] for key in ("polarity", "emittance_pm", "k1_near_m2", "k1_far_m2", "boundary_twiss")
        } for row in search["candidates"]], "app_result_validated": validation["passed"],
    }))
