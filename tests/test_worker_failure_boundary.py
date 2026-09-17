"""Unexpected controller failures cannot leave their claimed job running."""
import json
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import pytest

from brilliance import jobs, research_environment, research_recovery


@pytest.fixture
def worker(tmp_path, monkeypatch):
    project = SimpleNamespace(id="project-worker-test", revision=1)
    monkeypatch.setattr(research_environment.ResearchEnvironment, "load", lambda: SimpleNamespace(
        require_active_python=lambda **kw: None))
    monkeypatch.setattr(jobs, "ProjectStore", lambda root: SimpleNamespace(load=lambda project_id: project))
    monkeypatch.setattr(research_recovery, "scientist_request", lambda *args: {})
    store = jobs.ResearchJobStore(tmp_path)
    job = store.submit(project.id, project.revision, action="codex_turn", payload={"message": "Inspect current evidence"})
    native = ModuleType("brilliance.codex_runtime")
    native.CodexRuntimeStore = lambda *args: SimpleNamespace(consume_pending=lambda: [])
    native.configured_auto_continue = lambda: False
    native.configured_max_increments = lambda: 8
    native.run_codex_turn = lambda *args, **kwargs: SimpleNamespace(
        status="complete", detail="Existing numerical result explained.", disposition="stop")
    monkeypatch.setitem(sys.modules, "brilliance.codex_runtime", native)
    return tmp_path, store, job, native


def failure_record(root, job):
    path = root / job.project_id / "jobs" / job.job_id / "worker-failure.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_startup_import_error_is_operational_terminal_failure(worker, capsys):
    root, store, job, native = worker
    del native.run_codex_turn  # Reproduces an in-flight mixed-version import.
    outcome = jobs.run_research_job(root, job.project_id, job.job_id)
    assert outcome.state is jobs.JobState.FAILED
    assert store.load(job.project_id, job.job_id) == outcome
    assert outcome.controller_status == "worker_execution_error"
    assert "not a scientific rejection" in outcome.detail
    diagnostic = failure_record(root, job)
    assert diagnostic["error_type"] == "ImportError"
    assert "run_codex_turn" in diagnostic["error"]
    assert "Traceback (most recent call last)" in diagnostic["traceback"]
    assert diagnostic["worker_pid"] == outcome.worker_pid
    assert diagnostic["python_executable"] == sys.executable
    assert diagnostic["scientific_failure_inferred"] is False
    assert diagnostic["independent_jobs_cancelled"] is False
    assert "ImportError" in capsys.readouterr().err


def test_unexpected_turn_exception_preserves_its_exact_traceback(worker):
    root, store, job, native = worker
    def broken_turn(*args, **kwargs):
        raise KeyError("unavailable controller field")
    native.run_codex_turn = broken_turn
    outcome = jobs.run_research_job(root, job.project_id, job.job_id)
    assert outcome.state is jobs.JobState.FAILED
    assert store.load(job.project_id, job.job_id).payload == job.payload
    diagnostic = failure_record(root, job)
    assert "broken_turn" in diagnostic["traceback"]
    assert diagnostic["error_type"] == "KeyError"
    assert "unavailable controller field" in diagnostic["error"]


def test_ordinary_success_has_no_failure_record(worker):
    root, store, job, _ = worker
    outcome = jobs.run_research_job(root, job.project_id, job.job_id)
    assert outcome.state is jobs.JobState.COMPLETE
    assert outcome.controller_status == "codex_complete"
    assert store.load(job.project_id, job.job_id) == outcome
    assert not list((root / job.project_id).rglob("worker-failure.json"))


def test_post_completion_failure_does_not_downgrade_committed_increment(worker):
    root, store, job, native = worker
    def broken_pending():
        raise RuntimeError("Could not read follow-up queue")
    native.CodexRuntimeStore = lambda *args: SimpleNamespace(consume_pending=broken_pending)
    outcome = jobs.run_research_job(root, job.project_id, job.job_id)
    assert outcome.state is jobs.JobState.COMPLETE
    assert outcome.controller_status == "worker_followup_error"
    assert "Existing numerical result explained" in outcome.detail
    assert "completed increment remains committed" in outcome.detail
    assert store.load(job.project_id, job.job_id) == outcome
    assert failure_record(root, job)["previous_job_state"] == "complete"


def test_failed_controller_does_not_change_independent_job_or_saved_evidence(worker):
    root, store, job, native = worker
    numerical = store.submit(job.project_id, job.project_revision, action="expert_tool",
                             payload={"tool_id": "accelerator.run_long_research_program"})
    numerical = store.claim(numerical.project_id, numerical.job_id)
    unchanged = asdict(numerical)
    evidence = root / job.project_id / "workspace" / "artifacts" / "accepted-evidence.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"verified": true}\n', encoding="utf-8")
    previous_bytes = evidence.read_bytes()
    def broken_turn(*args, **kwargs):
        raise ImportError("new helper unavailable during live edit")
    native.run_codex_turn = broken_turn
    jobs.run_research_job(root, job.project_id, job.job_id)
    assert asdict(store.load(numerical.project_id, numerical.job_id)) == unchanged
    assert not list((root / job.project_id).rglob("cancel.request.json"))
    assert evidence.read_bytes() == previous_bytes


def test_wrong_interpreter_remains_preclaim_without_mutating_job(worker, monkeypatch):
    root, store, job, _ = worker
    def wrong_interpreter(**kwargs):
        raise RuntimeError("Configured interpreter required")
    monkeypatch.setattr(research_environment.ResearchEnvironment, "load", lambda: SimpleNamespace(
        require_active_python=wrong_interpreter))
    with pytest.raises(RuntimeError, match="Configured interpreter"):
        jobs.run_research_job(root, job.project_id, job.job_id)
    assert store.load(job.project_id, job.job_id) == job
    assert not list((root / job.project_id).rglob("claim.json"))


def test_duplicate_claim_does_not_mark_another_workers_job_failed(worker):
    root, store, job, _ = worker
    running = store.claim(job.project_id, job.job_id)
    with pytest.raises(RuntimeError, match="not queued"):
        jobs.run_research_job(root, job.project_id, job.job_id)
    assert store.load(job.project_id, job.job_id) == running
    assert not list((root / job.project_id).rglob("worker-failure.json"))


def test_failed_diagnostic_storage_still_commits_operational_terminal_state(worker, monkeypatch):
    root, store, job, native = worker
    del native.run_codex_turn
    original = jobs._write_once
    def write(path, payload):
        if path.name == "worker-failure.json":
            raise PermissionError("Diagnostic temporarily locked")
        return original(path, payload)
    monkeypatch.setattr(jobs, "_write_once", write)
    outcome = jobs.run_research_job(root, job.project_id, job.job_id)
    assert outcome.state is jobs.JobState.FAILED
    assert "Diagnostic storage failed: PermissionError" in outcome.detail
    assert store.load(job.project_id, job.job_id) == outcome
