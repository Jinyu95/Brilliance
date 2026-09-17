"""Worker observation never invokes application recovery or mutates job state."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts import release_trial_support as support


def job(state, pid=None):
    return SimpleNamespace(job_id="job-test", state=SimpleNamespace(value=state), worker_pid=pid)


def launch(project, *, seconds_ago, pid=42):
    path = project / "jobs/job-test/launches/0001-result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"pid": pid, "started_at": (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()}), encoding="utf-8")


@pytest.fixture(autouse=True)
def forbid_mutation(monkeypatch):
    for name in ("active", "recover", "finish", "record_launch", "claim", "save"):
        monkeypatch.setattr(support.ResearchJobStore, name, lambda *a, **k: pytest.fail("Observer attempted job mutation/recovery"))


def test_live_claimed_worker_is_active(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "_process_alive", lambda pid: pid == 42)
    report = support.observe_job_liveness(tmp_path, [job("running", 42)])
    assert report["active_job_ids"] == ["job-test"] and report["stale_job_ids"] == []
    assert report["observations"][0]["process_alive"] is True


def test_dead_claimed_worker_is_stale_and_history_bytes_unchanged(tmp_path, monkeypatch):
    launch(tmp_path, seconds_ago=60)
    revision = tmp_path / "jobs/job-test/revisions/0002.json"
    revision.parent.mkdir()
    revision.write_text('{"state":"running","worker_pid":42}', encoding="utf-8")
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(support, "_process_alive", lambda pid: False)
    report = support.observe_job_liveness(tmp_path, [job("running", 42)])
    assert report["stale_job_ids"] == ["job-test"] and report["active_job_ids"] == []
    assert report["observations"][0]["worker_log"] == "jobs/job-test/worker.log"
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_unlaunched_queue_is_not_a_dead_process(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "_process_alive", lambda pid: pytest.fail("No PID exists to probe"))
    report = support.observe_job_liveness(tmp_path, [job("queued")])
    assert report["active_job_ids"] == ["job-test"]
    assert report["observations"][0]["observation"] == "queued_awaiting_launch"


@pytest.mark.parametrize("age,stale", [(0, False), (60, True)])
def test_dead_unclaimed_launch_observes_existing_start_grace(tmp_path, monkeypatch, age, stale):
    launch(tmp_path, seconds_ago=age)
    monkeypatch.setattr(support, "_process_alive", lambda pid: False)
    report = support.observe_job_liveness(tmp_path, [job("queued")])
    assert bool(report["stale_job_ids"]) is stale
    assert bool(report["active_job_ids"]) is not stale


def test_live_worker_that_never_claimed_job_is_not_indefinitely_pending(tmp_path, monkeypatch):
    launch(tmp_path, seconds_ago=180)
    monkeypatch.setattr(support, "_process_alive", lambda pid: True)
    report = support.observe_job_liveness(tmp_path, [job("queued")])
    assert report["stale_job_ids"] == ["job-test"]
    claim = tmp_path / "jobs/job-test/claim.json"
    claim.write_text('{"pid":42}', encoding="utf-8")
    assert support.observe_job_liveness(tmp_path, [job("queued")])["active_job_ids"] == ["job-test"]


def test_missing_running_pid_is_unverifiable_not_confirmed_active(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "_process_alive", lambda pid: pytest.fail("No valid PID"))
    report = support.observe_job_liveness(tmp_path, [job("running")])
    assert report["unverifiable_job_ids"] == ["job-test"]
    assert report["active_job_ids"] == []


def test_terminal_job_is_not_probed_or_reopened(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "_process_alive", lambda pid: pytest.fail("Terminal jobs require no probe"))
    report = support.observe_job_liveness(tmp_path, [job("complete", 42)])
    assert report["observations"] == report["active_job_ids"] == []
