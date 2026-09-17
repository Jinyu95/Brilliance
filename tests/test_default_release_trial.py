"""Release observers exercise provenance and delivery without model/Julia calls."""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from scripts import release_trial_support as support
from scripts import validate_default_task as trial


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def environment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = source / "brilliance/config/runtime.toml"
    config.parent.mkdir(parents=True)
    config.write_text("# config", encoding="utf-8")
    native = source / "JuTrack.jl/python_integration/pyJuTrack.py"
    native.parent.mkdir(parents=True)
    native.write_text("def twissline(): pass\n", encoding="utf-8")
    return SimpleNamespace(source_root=source, config_path=config, pyjutrack_module=native,
        jutrack_root=source / "JuTrack.jl", julia_project=source / "JuTrack.jl",
        python_julia_environment=source / "julia_env", python_executable=source / "python.exe",
        julia_executable=source / "julia.exe", project_store=source / "research_projects",
        expert_paths=(source / "expert_modules",), internet_access=False,
        remote_mcp_enabled=False, local_mcp_enabled=True)


def job(name="initial", *, state="complete", **payload):
    return SimpleNamespace(job_id=name, action="codex_turn", state=SimpleNamespace(value=state),
                           payload={"initial": name == "initial", "message": "prompt", **payload})


def candidate(bends=7, periods=32, energy=3.0):
    return api.periodic_cell(title="Test geometry, not a physics claim", energy_gev=energy,
        elements=[{"name": f"B{i}", "kind": "SBEND", "length_m": 1.0,
                   "angle_rad": 2 * math.pi / periods / bends} for i in range(bends)],
        source="test", intended_repetitions=periods)


def test_manifest_is_exclusive_offline_and_contains_no_secrets(tmp_path, monkeypatch):
    env = environment(tmp_path)
    monkeypatch.setenv("BRILLIANCE_API_KEY", "secret-do-not-copy")
    monkeypatch.setattr(support, "describe_codex_backend", lambda: {"model": "local-test", "base_url": "https://user:secret@private/", "api_key": "secret-do-not-copy"})
    root = tmp_path / "trial"
    manifest = support.begin_manifest(root, "prompt", {"energy_gev": 3.0, "periods": 32}, env)
    assert "secret" not in json.dumps(manifest) and "base_url" not in manifest["configuration"]["backend"]
    with pytest.raises(ValueError, match="empty root"):
        support.begin_manifest(root, "prompt", {}, env)
    with pytest.raises(ValueError, match="user history"):
        support.begin_manifest(env.project_store, "prompt", {}, env)
    env.remote_mcp_enabled = True
    with pytest.raises(ValueError, match="remote MCP"):
        support.begin_manifest(tmp_path / "other", "prompt", {}, env)


def test_source_freeze_excludes_results_and_detects_real_edits(tmp_path, monkeypatch):
    env = environment(tmp_path)
    monkeypatch.setattr(support, "describe_codex_backend", lambda: {"model": "test"})
    root = tmp_path / "trial"
    manifest = support.begin_manifest(root, "prompt", {}, env)
    project = root / "project-test"
    project.mkdir()
    assert all(support.fresh_checks(manifest, project, env).values())
    write(project / "workspace/result.json", {"output": 42})
    write(env.source_root / "brilliance/__pycache__/cache.json", {})
    assert support.fresh_checks(manifest, project, env)["sources_frozen"]
    env.pyjutrack_module.write_text("changed", encoding="utf-8")
    assert not support.fresh_checks(manifest, project, env)["sources_frozen"]


@pytest.mark.parametrize("value", [{"energy_gev": 3, "periods": True}, {"energy_gev": 3, "periods": 32.0},
    {"energy_gev": float("nan"), "periods": 32}, {"periods": 32}, {"energy_gev": 3, "periods": 32, "expected_bends": 2}])
def test_requirements_never_invent_or_accept_invalid_values(value):
    with pytest.raises(ValueError):
        trial.validate_requirements(value)


def test_scope_uses_explicit_user_energy_cell_and_bend_count():
    requirements = {"energy_gev": 3.5, "periods": 24, "expected_bends": 9, "circumference_m": 216.0}
    ring, report = trial.scoped_candidate(candidate(9, 24, 3.5), requirements)
    assert report["passed"] and ring.periodicity == 24
    assert not trial.scoped_candidate(candidate(7, 24, 3.5), requirements)[1]["passed"]
    requirements["circumference_m"] = 800.0
    assert not trial.scoped_candidate(candidate(9, 24, 3.5), requirements)[1]["passed"]


def test_explicit_ring_repetition_checks_native_physical_fields():
    cell = candidate()
    ring = api.cell_ring(title="ring", energy_gev=3.0, cell=api.one_period_elements(cell), periods=32, source="test")
    data = ring.as_dict()
    data["elements"][7]["fields"]["e1"] = 0.02
    altered = type(ring).create(**{key: data[key] for key in
        ("title", "energy_gev", "periodicity", "elements", "source", "metadata")})
    assert not trial.scoped_candidate(altered, {"energy_gev": 3.0, "periods": 32})[1]["passed"]


def test_source_authorship_accepts_model_patches_but_not_operator_changes(tmp_path):
    program = tmp_path / "workspace/artifacts/programs/program-test/program.py"
    program.parent.mkdir(parents=True)
    program.write_text("print('model patched program')\n", encoding="utf-8")
    identity = support.digest(program)
    write(tmp_path / "workspace/agent_sessions/native.json", {"events": [
        {"kind": "tool_result", "name": "patch_program", "result": {"ok": True, "preflight": {"sha256": identity}}}]})
    executions = [{"result": {"program": {"snapshot": str(program), "sha256": identity}, "duration_s": 2}}]
    checked = support.trace_audit(tmp_path, [job()], "prompt", executions)
    assert all(checked["checks"].values())
    assert checked["committed_calculation_seconds"] == 2
    program.write_text("print('operator altered')\n", encoding="utf-8")
    assert not support.trace_audit(tmp_path, [job()], "prompt", executions)["checks"]["programs_authored_in_model_trace"]


def test_unrecorded_operator_turn_is_not_automatic_recovery(tmp_path):
    assert not support.trace_audit(tmp_path, [job(), job("manual")], "prompt", [])["checks"]["no_unattributed_agent_turns"]
    report = support.trace_audit(tmp_path, [job(), job("repair", repair_of="initial")], "prompt", [])
    assert report["checks"]["no_unattributed_agent_turns"] and report["automatic_repairs"] == 1


def test_old_or_unrelated_plot_and_table_cannot_satisfy_selected_answer():
    proof = {"evidence_run_id": "selected"}
    messages = [{"content": "Long unrelated text", "meta": {"completion_evidence": None}, "attachments": [
        {"type": "image", "lattice_id": "other"}, {"type": "evidence_summary", "tables": [{"rows": [1]}]}]},
        {"content": "Selected answer", "meta": {"completion_evidence": proof}, "attachments": []}]
    assert not support.selected_presentation(messages, proof, "correct")["selected_image_in_chat"]
    messages[-1]["attachments"] = [{"type": "image", "lattice_id": "wrong", "presentation_role": "selected_evidence"}]
    assert not support.selected_presentation(messages, proof, "correct")["selected_image_in_chat"]


def test_missing_manifest_cannot_turn_historical_success_into_fresh_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "ResearchChatService", lambda root: SimpleNamespace(store=SimpleNamespace(load=lambda _: object())))
    report = trial.audit(tmp_path / "project-history")
    assert not report["passed"] and "manifest" in report["reason"]


def test_active_job_returns_pending_without_snapshot_recovery_or_numerical_execution(tmp_path, monkeypatch):
    project = tmp_path / "project-fresh"
    project.mkdir()
    write(tmp_path / support.MANIFEST, {"prompt": "prompt", "started_at": support.now()})
    monkeypatch.setattr(trial, "fresh_checks", lambda *args: {"fresh": True})
    monkeypatch.setattr(trial, "trace_audit", lambda *args: {"checks": {}})
    monkeypatch.setattr(trial, "ResearchChatService", lambda root: SimpleNamespace(store=SimpleNamespace(load=lambda _: SimpleNamespace(goal="prompt"))))
    running = job(state="running")
    running.worker_pid = 12345
    monkeypatch.setattr(support, "_process_alive", lambda pid: True)
    monkeypatch.setattr(trial, "ResearchJobStore", lambda root: SimpleNamespace(list=lambda _: [running]))
    monkeypatch.setattr(trial, "ResearchConversationStore", lambda root: SimpleNamespace(messages=lambda _: [{"role": "user", "content": "prompt"}]))
    monkeypatch.setattr(api, "evaluate", lambda *a, **k: pytest.fail("Observer started Julia"))
    report = trial.audit(project, verify_jutrack=True, environment=object())
    assert report["status"] == "pending" and report["running_jobs"] == ["initial"]


def test_stale_running_job_fails_observation_without_state_reconciliation(tmp_path, monkeypatch):
    project = tmp_path / "project-fresh"
    project.mkdir()
    write(tmp_path / support.MANIFEST, {"prompt": "prompt", "started_at": support.now()})
    monkeypatch.setattr(trial, "fresh_checks", lambda *args: {"fresh": True})
    monkeypatch.setattr(trial, "trace_audit", lambda *args: {"checks": {}})
    monkeypatch.setattr(trial, "ResearchChatService", lambda root: SimpleNamespace(store=SimpleNamespace(load=lambda _: SimpleNamespace(goal="prompt"))))
    running = job(state="running")
    running.worker_pid = 12345
    monkeypatch.setattr(support, "_process_alive", lambda pid: False)
    monkeypatch.setattr(trial, "ResearchJobStore", lambda root: SimpleNamespace(list=lambda _: [running]))
    monkeypatch.setattr(trial, "ResearchConversationStore", lambda root: SimpleNamespace(messages=lambda _: [{"role": "user", "content": "prompt"}]))
    monkeypatch.setattr(api, "evaluate", lambda *a, **k: pytest.fail("Observer started Julia"))
    report = trial.audit(project, environment=object())
    assert report["status"] == "failed" and report["running_jobs"] == []
    assert report["job_liveness"]["stale_job_ids"] == ["initial"]
    assert running.state.value == "running"  # Observer did not reconcile persisted state.


def test_browser_receipt_binds_exact_result_assets_and_screenshot(tmp_path):
    source = tmp_path / "source"
    for name in ("app.js", "styles.css", "workbench.css", "index.html"):
        file = source / "brilliance/web/static" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(name, encoding="utf-8")
    result = tmp_path / "result.json"
    write(result, {"candidate": "selected"})
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"image")
    value = {"schema": "brilliance_rendered_selection/v1", "project_directory": str(tmp_path.resolve()),
        "candidate_id": "candidate", "run_id": "run", "result_sha256": support.digest(result),
        "assets": {name: support.digest(source / "brilliance/web/static" / name) for name in ("app.js", "styles.css", "workbench.css", "index.html")},
        "checks": dict.fromkeys(support.RENDER_CHECKS, True), "screenshot": str(shot), "screenshot_sha256": support.digest(shot)}
    receipt = tmp_path / "browser.json"
    write(receipt, value)
    args = dict(directory=tmp_path, candidate_id="candidate", run_id="run", result_path=result, source_root=source)
    assert support.browser_checks(receipt, **args)["browser_selected_result_rendered"]
    write(result, {"candidate": "new unrelated candidate"})
    assert not support.browser_checks(receipt, **args)["browser_selected_result_rendered"]


def test_completion_must_be_bound_to_this_scientist_request():
    proof = {"kind": "scientist_request_completion_from_saved_evidence", "scientist_request": {"id": "old"},
             "execution_valid": True, "scientific_passed": True, "evidence_receipt_sha256": "a" * 64,
             "evidence_run_id": "program-older"}
    rows = [{"role": "assistant", "meta": {"completion_evidence": copy.deepcopy(proof)}}]
    assert trial.completion_run_id(rows, {"id": "new"}) is None
    assert trial.completion_run_id(rows, {"id": "old"}) == "program-older"
