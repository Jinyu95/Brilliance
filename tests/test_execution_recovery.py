"""Every failed saved program can reach bounded diagnosis, without executing jobs."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance import research_recovery
from brilliance.research_session import SessionWorkspace


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    project = SimpleNamespace(id="project-execution-recovery", revision=1)
    directory = root / project.id
    workspace = directory / "workspace"
    source = workspace / "programs" / "study.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Exact scientific source remains unchanged during diagnosis.\n", encoding="utf-8")
    candidate = workspace / "candidate_lattice.json"
    candidate.write_bytes(b'{"lattice_id":"exact-candidate","k1":3.4261995199999995}\n')
    checkpoint = workspace / "checkpoint.json"
    checkpoint.write_bytes(b'{"generation":30,"best_candidate":[3.4261995199999995]}\n')
    origin = {"id": "scientist-request-1", "content": "Inspect the current accelerator calculation."}
    state = SimpleNamespace(latest=origin, cancelled=False, active=False)
    submissions, launches, queued = [], [], []
    monkeypatch.setenv("BRILLIANCE_AUTO_REPAIR_ATTEMPTS", "3")
    monkeypatch.setattr(research_recovery, "scientist_request", lambda *_: state.latest)
    monkeypatch.setattr(research_recovery, "saved_recovery_context", lambda *_: "Exact candidate and checkpoint already exist.")

    def submit(*args, **kwargs):
        submissions.append(kwargs)
        return SimpleNamespace(job_id=f"diagnosis-{len(submissions)}")

    jobs = SimpleNamespace(cancellation_requested=lambda *_: state.cancelled,
                           active=lambda *_, **__: SimpleNamespace(job_id="active-agent") if state.active else None,
                           submit=submit)
    monkeypatch.setattr("brilliance.jobs.launch_research_job", lambda *args: launches.append(args))
    monkeypatch.setattr("brilliance.codex_runtime.CodexRuntimeStore",
                        lambda *_: SimpleNamespace(queue_input=queued.append))

    def schedule(job_id="failed-job", error="StopIteration", *, arguments=None,
                 script="programs/study.py", tool_id="accelerator.run_long_research_program",
                 scientist_origin=None):
        failure = workspace / "runtime_failures" / f"{job_id}.json"
        failure.parent.mkdir(exist_ok=True)
        if not failure.exists():
            failure.write_text(json.dumps({"error": error}), encoding="utf-8")
        job = SimpleNamespace(job_id=job_id, action="tool", payload={
            "tool_id": tool_id,
            "scientist_request": origin if scientist_origin is None else scientist_origin,
            "arguments": {"script": script, "arguments": arguments if arguments is not None else {"point": 1.0}},
        })
        return research_recovery.schedule_coding_repair(root, jobs, job, project, failure, error)

    return SimpleNamespace(root=root, project=project, directory=directory, workspace=workspace,
                           source=source, candidate=candidate, checkpoint=checkpoint, state=state,
                           origin=origin, schedule=schedule, submissions=submissions, launches=launches,
                           queued=queued)


@pytest.mark.parametrize("error, expected", [
    ('Traceback (most recent call last):\n  File "program.py", line 8, in <module>\n'
     '    selected = next(candidate for candidate in rows if candidate.name == name)\n'
     'StopIteration\nBRILLIANCE_FAILURE_LOCALS={"name":"B4"}', "StopIteration"),
    ("AssertionError", "AssertionError"),
    ("my_topic.BeamStudyAbort: an unexpected callback outcome", "BeamStudyAbort"),
    ("ValueError: no stable periodic solution", "ValueError"),
    ("RuntimeError: Julia bridge failed to initialize", "RuntimeError"),
    ("PermissionError: [WinError 5] Access denied", "PermissionError"),
    ("Worker exited unexpectedly without a traceback", "Worker exited unexpectedly"),
    ("", "Calculation could not complete"),
])
def test_unknown_failures_receive_diagnosis_not_a_whitelist_rejection(recovery, error, expected):
    before = {path: path.read_bytes() for path in
              (recovery.source, recovery.candidate, recovery.checkpoint)}
    record = recovery.schedule(error=error)
    assert record["scheduled"] is True
    assert record["mode"] == "diagnose_and_repair"
    assert record["attempt"] == 1
    assert expected in record["summary"]
    assert len(recovery.submissions) == len(recovery.launches) == 1
    payload = recovery.submissions[0]["payload"]
    assert payload["repair_request_id"] == recovery.origin["id"]
    message = payload["message"].casefold()
    assert "inspect the traceback" in message
    assert "do not rerun an unchanged failed attempt" in message
    assert "read-only diagnosis" in message
    assert "physical requirements" in message
    assert "configured" in message
    assert "install" in message and "do not" in message
    assert recovery.origin["content"] in payload["message"]
    for path, original in before.items():
        assert path.read_bytes() == original


@pytest.mark.parametrize("tool_id", ["accelerator.run_research_program", "accelerator.run_long_research_program"])
def test_both_saved_program_boundaries_are_eligible(recovery, tool_id):
    assert recovery.schedule(tool_id=tool_id)["scheduled"] is True


def test_duplicate_notifications_and_unchanged_attempts_do_not_launch_twice(recovery):
    record = recovery.schedule()
    assert recovery.schedule() == record
    assert len(recovery.launches) == 1
    assert recovery.schedule("same-source-new-job") == {
        "scheduled": False, "reason": "same_source_failed_again",
    }
    assert len(recovery.launches) == 1


@pytest.mark.parametrize("script", ["programs/study.py", "workspace/programs/study.py"])
def test_changed_arguments_are_a_new_attempt_without_source_mutation(recovery, script):
    before = recovery.source.read_bytes()
    first = recovery.schedule("first", script=script, arguments={"point": 1.0, "settings": {"samples": 16}})
    assert first["scheduled"] is True
    assert first["source_sha"] != "unavailable"
    # Mapping order alone is not a changed input.
    repeated = recovery.schedule("same-input", script=script,
                                 arguments={"settings": {"samples": 16}, "point": 1.0})
    assert repeated == {"scheduled": False, "reason": "same_source_failed_again"}
    changed = recovery.schedule("corrected-input", script=script,
                                arguments={"point": 1.0, "settings": {"samples": 32}})
    assert changed["scheduled"] is True and changed["attempt"] == 2
    assert changed["failure_key"] != first["failure_key"]
    assert changed["source_sha"] == first["source_sha"]
    assert recovery.source.read_bytes() == before
    assert len(recovery.launches) == 2


def test_changed_source_can_be_diagnosed_but_budget_still_applies(recovery, monkeypatch):
    monkeypatch.setenv("BRILLIANCE_AUTO_REPAIR_ATTEMPTS", "2")
    first = recovery.schedule("first")
    recovery.source.write_text("# Reviewed source correction.\n", encoding="utf-8")
    second = recovery.schedule("second")
    assert second["scheduled"] is True and second["attempt"] == 2
    assert first["source_sha"] != second["source_sha"]
    assert recovery.schedule("third", error="ValueError: another diagnostic") == {
        "scheduled": False, "reason": "repair_budget_exhausted",
    }
    assert len(recovery.launches) == 2


def test_workspace_prefix_is_not_a_way_to_evade_unchanged_attempt_guard(recovery):
    assert recovery.schedule("first", script="programs/study.py")["scheduled"] is True
    assert recovery.schedule("same-file", script="workspace/programs/study.py") == {
        "scheduled": False, "reason": "same_source_failed_again",
    }
    assert len(recovery.launches) == 1


def test_source_read_access_failure_still_gets_read_only_diagnosis(recovery, monkeypatch):
    read_text = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path.resolve() == recovery.source.resolve():
            raise PermissionError("source cannot be read")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    record = recovery.schedule(error="PermissionError: source cannot be read")
    assert record["scheduled"] is True
    assert record["mode"] == "diagnose_and_repair"
    assert record["source_sha"] == "unavailable"
    message = recovery.submissions[0]["payload"]["message"].casefold()
    assert "read-only diagnosis" in message
    assert "permissions" in message and "do not" in message


def test_program_path_outside_this_workspace_cannot_launch_recovery(recovery):
    record = recovery.schedule(script="../../other-project/workspace/program.py")
    assert record == {"scheduled": False, "reason": "program_outside_workspace"}
    assert not recovery.submissions and not recovery.launches and not recovery.queued


@pytest.mark.parametrize("guard", ["cancelled", "paused", "new_request", "missing_origin", "disabled"])
def test_scientist_and_budget_controls_still_block_diagnosis(recovery, monkeypatch, guard):
    origin = None
    if guard == "cancelled":
        recovery.state.cancelled = True
    elif guard == "paused":
        SessionWorkspace(recovery.directory).pause_file.write_text("scientist paused", encoding="utf-8")
    elif guard == "new_request":
        recovery.state.latest = {"id": "new-request", "content": "Stop and explain."}
    elif guard == "missing_origin":
        origin = {}
    else:
        monkeypatch.setenv("BRILLIANCE_AUTO_REPAIR_ATTEMPTS", "0")
    record = recovery.schedule(scientist_origin=origin)
    expected = ("scientist_paused_or_cancelled" if guard in {"cancelled", "paused"}
                else "request_changed_or_not_recorded" if guard in {"new_request", "missing_origin"}
                else "repair_disabled")
    assert record == {"scheduled": False, "reason": expected}
    assert not recovery.submissions and not recovery.launches and not recovery.queued


def test_existing_agent_gets_one_queued_diagnosis_not_a_second_job(recovery):
    recovery.state.active = True
    record = recovery.schedule()
    assert record["scheduled"] is True
    assert len(recovery.queued) == 1
    assert not recovery.submissions and not recovery.launches
    assert recovery.schedule() == record
    assert len(recovery.queued) == 1


def test_unrelated_tool_failure_is_not_a_saved_program_recovery(recovery):
    record = recovery.schedule(tool_id="accelerator.query_evidence")
    assert record["scheduled"] is False
    assert not recovery.submissions and not recovery.launches and not recovery.queued
