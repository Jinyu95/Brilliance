"""Reporting-contract failures are recoverable without retrying physics blindly."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from brilliance import research_recovery
from brilliance.research_recovery import coding_failure_summary, program_failure_guidance

LEGACY_KIND_ERROR = (
    "ValueError: progress kind must be plan, progress, finding, decision, repair, or next"
)


@pytest.mark.parametrize("headline", [
    LEGACY_KIND_ERROR,
    "ProgressInputError: progress table must contain a rows list",
    "brilliance.research_api.ProgressInputError: progress event_key must not be empty",
    "ValueError: scientific_role must be milestone or activity",
    "ValueError: each progress table must be a mapping with a rows list",
    "ValueError: progress table must contain a rows list",
    "ValueError: every progress table row must be a mapping of column names",
    "ValueError: each progress plot must be a path or a mapping with a path",
    "ValueError: progress plot must be stored below workspace/artifacts",
    "ValueError: progress plot does not exist: plots/tme.png",
    "ValueError: progress event_key must not be empty",
    "ValueError: status must be completed, infeasible, or partial",
])
def test_reporting_contracts_get_bounded_recovery(headline):
    traceback = (
        'Traceback (most recent call last):\n'
        '  File "program.py", line 216, in main\n'
        '    run.progress("TME unit complete", kind="activity")\n'
        '  File "brilliance/research_api.py", line 395, in progress\n'
        '    raise ValueError("progress kind must be plan, progress, finding, decision, repair, or next")\n'
        + headline + '\n'
    )
    assert coding_failure_summary(traceback) == headline
    guidance = program_failure_guidance(traceback)
    assert "reporting API error" in guidance["explanation"]
    assert "not evidence" in guidance["explanation"]
    hint = guidance["repair_hint"]
    assert "Correct only the reporting call" in hint
    assert "full-precision fields" in hint
    assert "Recover the exact saved stage" in hint
    assert "Do not regenerate a seed, repeat a completed search" in hint
    assert "intermediate stage remains partial" in hint
    assert "If needed results were never saved" in hint


@pytest.mark.parametrize("error", [
    "ValueError: no stable periodic solution",
    "ValueError: periodic ring bend angle does not close",
    "ValueError: negative emittance",
    "ValueError: required dispersion matching tolerance was not reached",
    "ValueError: optimizer has no feasible particle",
    "ValueError: progress table must contain a rows list: different unrelated failure",
    "ValueError: status must be completed, infeasible, or partial: physics rejection",
    "ValueError: bad progress",  # Deliberately not a broad 'progress' substring match.
    "RuntimeError: Julia bridge failed to initialize",
    "PermissionError: [WinError 5] Access denied",
    "FileNotFoundError: a plot or native library is missing",
    "OSError: progress file access denied",
    "TimeoutError: calculation ended",
    "InterruptedError: the scientist requested cancellation",
])
def test_other_failures_are_not_mistaken_for_reporting_contracts(error):
    assert coding_failure_summary(error) is None
    assert program_failure_guidance(error)["repair_hint"] == ""


def _repair_harness(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    project = SimpleNamespace(id="project-progress-test", revision=1)
    workspace = root / project.id / "workspace"
    workspace.mkdir(parents=True)
    source = workspace / "program.py"
    source.write_text('run.progress("TME complete", kind="activity")\n', encoding="utf-8")
    candidate = workspace / "tme_candidate.json"
    candidate.write_bytes(b'{"accepted": true, "k1_m2": 3.4261995199999995}\n')
    checkpoint = workspace / "checkpoint.json"
    checkpoint.write_bytes(b'{"generation": 30, "best_candidate": [3.4261995199999995]}\n')
    origin = {"id": "scientist-request", "content": "Design a conventional MBA."}
    monkeypatch.setattr(research_recovery, "scientist_request", lambda *_: origin)
    monkeypatch.setattr(research_recovery, "saved_recovery_context", lambda *_: "Saved TME checkpoint exists.")
    monkeypatch.setenv("BRILLIANCE_AUTO_REPAIR_ATTEMPTS", "2")
    submissions, launches = [], []

    def submit(*args, **kwargs):
        submissions.append(kwargs["payload"])
        return SimpleNamespace(job_id=f"repair-{len(submissions)}")

    monkeypatch.setattr("brilliance.jobs.launch_research_job", lambda *args: launches.append(args))
    jobs = SimpleNamespace(
        cancellation_requested=lambda *_: False,
        active=lambda *_, **__: None,
        submit=submit,
    )

    def schedule(job_id, error=LEGACY_KIND_ERROR):
        job = SimpleNamespace(job_id=job_id, payload={
            "tool_id": "accelerator.run_long_research_program",
            "scientist_request": origin,
            "arguments": {"script": "program.py"},
        })
        return research_recovery.schedule_coding_repair(
            root, jobs, job, project,
            workspace / "runtime_failures" / f"{job_id}.json", error,
        )

    return SimpleNamespace(
        schedule=schedule, source=source, candidate=candidate, checkpoint=checkpoint,
        submissions=submissions, launches=launches,
    )


def test_legacy_reporting_failure_schedules_once_without_mutating_saved_science(tmp_path, monkeypatch):
    harness = _repair_harness(tmp_path, monkeypatch)
    originals = {path: path.read_bytes() for path in
                 (harness.source, harness.candidate, harness.checkpoint)}
    record = harness.schedule("job-371b66c6db854b825927")
    assert record["scheduled"] is True
    assert record["attempt"] == 1
    assert record["summary"] == LEGACY_KIND_ERROR
    assert len(harness.submissions) == len(harness.launches) == 1
    message = harness.submissions[0]["message"]
    assert "Original request: Design a conventional MBA." in message
    assert "Correct only the reporting call" in message
    assert "Recover the exact saved stage" in message
    assert "do not claim the checkpoint contains them" in message
    assert harness.schedule("job-371b66c6db854b825927") == record
    assert len(harness.submissions) == len(harness.launches) == 1
    assert harness.schedule("unchanged-source") == {
        "scheduled": False, "reason": "same_source_failed_again",
    }
    for path, original in originals.items():
        assert path.read_bytes() == original


def test_reporting_recovery_remains_bounded_and_does_not_restart_search(tmp_path, monkeypatch):
    harness = _repair_harness(tmp_path, monkeypatch)
    original_candidate = harness.candidate.read_bytes()
    original_checkpoint = harness.checkpoint.read_bytes()
    assert harness.schedule("first")["attempt"] == 1
    # A different reporting-contract error may consume the remaining attempt.
    assert harness.schedule("second", "ProgressInputError: bad progress table")["attempt"] == 2
    assert harness.schedule("third", "ProgressInputError: bad progress plot") == {
        "scheduled": False, "reason": "repair_budget_exhausted",
    }
    assert len(harness.launches) == 2
    assert harness.candidate.read_bytes() == original_candidate
    assert harness.checkpoint.read_bytes() == original_checkpoint


@pytest.mark.parametrize("error", [
    "ValueError: no stable periodic solution",
    "PermissionError: [WinError 5] Access denied",
    "RuntimeError: Julia bridge failed to initialize",
])
def test_nonreporting_failures_receive_bounded_diagnosis(tmp_path, monkeypatch, error):
    harness = _repair_harness(tmp_path, monkeypatch)
    originals = {path: path.read_bytes() for path in
                 (harness.source, harness.candidate, harness.checkpoint)}
    record = harness.schedule("failed", error)
    assert record["scheduled"] is True
    assert record["mode"] == "diagnose_and_repair"
    assert len(harness.submissions) == len(harness.launches) == 1
    message = harness.submissions[0]["message"].casefold()
    assert "inspect the traceback" in message
    assert "do not rerun an unchanged failed attempt" in message
    assert "read-only diagnosis" in message
    for path, original in originals.items():
        assert path.read_bytes() == original


def test_reporting_recovery_respects_scientist_repair_setting(tmp_path, monkeypatch):
    harness = _repair_harness(tmp_path, monkeypatch)
    monkeypatch.setenv("BRILLIANCE_AUTO_REPAIR_ATTEMPTS", "0")
    assert harness.schedule("failed") == {"scheduled": False, "reason": "repair_disabled"}
    assert not harness.launches
