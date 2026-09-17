"""Regressions from the frozen fresh provider trial, without replaying that run.

These synthetic records reproduce flat PSO checkpoints and synchronous failures.
No saved project, production fixture or native numerical process is used here.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.release_trial_support import trace_audit


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def inspect(tmp_path, events):
    save(tmp_path / "workspace/agent_sessions/native-test.json", {"schema_version": "1.0", "events": events})
    job = SimpleNamespace(job_id="initial", action="codex_turn", state=SimpleNamespace(value="complete"),
                          payload={"initial": True, "message": "prompt"})
    return trace_audit(tmp_path, [job], "prompt", [])


def pair(name, arguments, result):
    return [{"kind": "tool_call", "name": name, "arguments": arguments},
            {"kind": "tool_result", "name": name, "result": result}]


def test_flat_native_checkpoint_is_one_campaign_not_zero_or_geometry_duplicate(tmp_path):
    """The actual native layout is unit_geometry.json + unit_geometry.geometry.json."""
    checkpoint = {
        "schema_version": "1.0", "mode": "single_objective", "complete": True,
        "completed_generation": 80, "evaluation_count": 40500,
        "contract_fingerprint": "swarm-identity", "contract": {
            "optimizer": "particle_swarm_optimize", "population_size": 500,
            "variable_names": [f"knob_{i}" for i in range(7)],
            "objective_names": ["emittance_pm"], "bounds": [[0.0, 1.0]] * 7,
            "seed": 0, "campaign_identity": "unit-geometry-campaign"},
        "history": [
            {"generation": 0, "population_size": 500, "feasible_count": 500,
             "best": {"emittance_pm": 1074.5504914575822}},
            {"generation": 80, "population_size": 500, "feasible_count": 500,
             "best": {"emittance_pm": 113.49154862010735}}],
        "budget": {"effective": {"population_size": 500, "generations": 80}},
    }
    save(tmp_path / "workspace/optimizations/unit_geometry.json", checkpoint)
    save(tmp_path / "workspace/optimizations/unit_geometry.geometry.json", {
        "contract": {"expected_length_m": 2.6}, "contract_sha256": "geometry-identity",
        "initialization": {"attempted": 1708, "feasible_particles": 500},
        "finished_result": {"optimizer": {"evaluated_candidates": 40500}},
        "verification_campaigns": {},
    })
    report = inspect(tmp_path, [])
    assert len(report["optimizer_campaigns"]) == 1
    assert report["recorded_optimizer_evaluations"] == 40500
    campaign = report["optimizer_campaigns"][0]
    assert campaign["evaluation_count"] == 40500 and campaign["completed_generation"] == 80
    assert campaign["complete"] is True


def test_synchronous_failure_and_within_turn_repair_not_hidden_by_completed_parent_job(tmp_path):
    """Actual trial: failed checkpoint(None), corrected partial report, durable submission."""
    events = pair("run_program", {"name": "contract.py"}, {
        "ok": False,
        "error": "ValueError: Cannot checkpoint an unverified candidate evaluation: No exact-candidate evaluation was selected for the archived lattice.",
        "preflight": {"passed": True, "source": "programs/contract.py", "sha256": "original"},
    })
    events += pair("patch_program", {"name": "contract.py", "old": "bad checkpoint", "new": "partial report"}, {
        "ok": True, "program": "programs/contract.py",
        "preflight": {"passed": True, "source": "programs/contract.py", "sha256": "corrected"},
    })
    events += pair("run_program", {"name": "contract.py"}, {
        "ok": True, "kind": "completed_execution", "execution_valid": True,
        "scientific_passed": False, "run_id": "program-corrected", "duration_s": 0.8083369,
        "program": {"source": "programs/contract.py", "sha256": "corrected"},
    })
    events += pair("run_program", {"name": "unit.py", "background": True}, {
        "ok": True, "kind": "managed_calculation", "job_id": "managed-unit", "state": "queued",
    })
    report = inspect(tmp_path, events)
    assert report["job_counts"] == {"complete": 1} and report["automatic_repairs"] == 0
    attempts = report["program_attempts"]
    assert attempts["run_program_requests"] == 3
    assert attempts["synchronous_execution_failures"] == 1
    assert attempts["synchronous_execution_completions"] == 1
    assert attempts["partial_scientific_results"] == 1
    assert attempts["managed_submissions"] == 1
    assert attempts["in_turn_execution_recoveries"] == 1
    assert attempts["preflight_rejections"] == attempts["api_rejections"] == 0


def test_preflight_and_tool_argument_rejection_are_not_launched_execution_failures(tmp_path):
    events = pair("run_program", {"name": "wrong_api.py"}, {
        "ok": False, "error": "program preflight failed", "preflight": {"passed": False}})
    events += pair("run_program", {"name": "wrong_arguments.py", "arguments": []}, {
        "ok": False, "error": "arguments must be one object"})
    events += pair("run_program", {"name": "missing.py"}, {
        "ok": False, "error": "program does not exist: missing.py"})
    attempts = inspect(tmp_path, events)["program_attempts"]
    assert attempts["run_program_requests"] == 3
    assert attempts["preflight_rejections"] == 1 and attempts["api_rejections"] == 2
    assert attempts["synchronous_execution_failures"] == 0
    assert attempts["in_turn_execution_recoveries"] == 0


def test_cached_failure_is_not_a_second_execution_or_a_repair(tmp_path):
    failure = {"ok": False, "error": "ValueError: callback failed",
               "preflight": {"passed": True, "source": "programs/scan.py"}}
    events = pair("run_program", {"name": "scan.py"}, failure)
    events += pair("run_program", {"name": "scan.py"}, {**failure, "cached": True})
    attempts = inspect(tmp_path, events)["program_attempts"]
    assert attempts["run_program_requests"] == 2
    assert attempts["synchronous_execution_failures"] == 1 and attempts["cached_results"] == 1
    assert attempts["in_turn_execution_recoveries"] == 0


def test_unrelated_program_success_does_not_claim_previous_failure_repaired(tmp_path):
    events = pair("run_program", {"name": "failed.py"}, {
        "ok": False, "error": "RuntimeError: native execution failed",
        "preflight": {"passed": True, "source": "programs/failed.py"}})
    events += pair("run_program", {"name": "different.py"}, {
        "ok": True, "kind": "completed_execution", "execution_valid": True,
        "scientific_passed": True, "program": {"source": "programs/different.py"}})
    attempts = inspect(tmp_path, events)["program_attempts"]
    assert attempts["synchronous_execution_failures"] == attempts["synchronous_execution_completions"] == 1
    assert attempts["in_turn_execution_recoveries"] == 0


def test_missing_checkpoints_are_unknown_not_zero_physics_work(tmp_path):
    report = inspect(tmp_path, [])
    assert report["optimizer_campaigns"] == []
    assert report["recorded_optimizer_evaluations"] is None


def test_reporting_checkpoint_recovery_and_model_limit_are_not_execution_failures(tmp_path):
    events = [{"kind": "model_output_rejected", "reason": "output_token_limit", "finish_reason": "length"}]
    save(tmp_path / "workspace/agent_sessions/native-test.json", {"events": events})
    execution = {"result": {"run_id": "program-reporting", "result": {
        "status": "partial", "scientific_stage": {"outcome": "reporting_pending"},
        "data": {"reporting_failure": {"type": "ValueError", "message": "unknown scientific_stage fields: stage"},
                 "numerical_checkpoint": {"kind": "candidate_evaluation", "sha256": "checkpoint-hash"}}}}}
    report = trace_audit(tmp_path, [], "prompt", [execution, execution])
    assert report["model_output_rejections"] == {"output_token_limit": 1}
    assert report["optional_reporting"]["failure_results"] == 1
    assert report["optional_reporting"]["checkpoint_preserving_recoveries"] == 1
    assert report["program_attempts"]["synchronous_execution_failures"] == 0
    assert report["program_attempts"]["in_turn_execution_recoveries"] == 0


def test_unknown_error_is_not_misclassified_as_an_api_or_physics_error(tmp_path):
    events = pair("run_program", {"name": "unknown.py"}, {"ok": False, "error": "Worker ended unexpectedly"})
    counts = inspect(tmp_path, events)["program_attempts"]
    assert counts["unclassified_run_errors"] == 1
    assert counts["api_rejections"] == counts["synchronous_execution_failures"] == 0


def test_failed_durable_submission_does_not_claim_native_job_executed(tmp_path):
    events = pair("run_program", {"name": "scan.py", "background": True}, {
        "ok": False, "error": "RuntimeError: submit command failed", "preflight": {"passed": True}})
    counts = inspect(tmp_path, events)["program_attempts"]
    assert counts["managed_submission_failures"] == 1
    assert counts["managed_submissions"] == counts["synchronous_execution_failures"] == 0
