"""Numerical claims must refer to the exact evaluated, archived snapshot.

Synthetic capture/mocked JuTrack only: these test application lineage, not beam
physics, optimizer performance, or a stored design template.
"""
from __future__ import annotations

import hashlib
import json
import math

import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence as evidence
from brilliance.research_api import ResearchRun
from brilliance.research_program import validate_research_program


@pytest.fixture(autouse=True)
def isolated_capture(monkeypatch):
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    monkeypatch.delenv("BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE", raising=False)
    monkeypatch.delenv("BRILLIANCE_SCIENTIFIC_ACCEPTANCE", raising=False)
    evidence._captured.clear()
    yield
    evidence._captured.clear()


def cell(*, k1=0.4, energy=3.0, periods=8, title="Synthetic cell"):
    return api.periodic_cell(title=title, energy_gev=energy,
                             elements=[api.drift("D", 1.0), api.quadrupole("Q", 0.2, k1),
                                       api.sector_bend("B", 1.0, 2 * math.pi / 8)],
                             source="synthetic lineage test", intended_repetitions=periods)


def capture(candidate, emittance=10.0):
    result = {"engine": "JuTrack", "candidate_lattice_id": candidate.lattice_id,
              "evaluation_scope": "one_periodic_cell_scaled_to_identical_ring",
              "periods": 8, "geometry": {"cell_length_m": candidate.total_length_m},
              "linear_optics": {"stable": True, "Dx_m": 1e-6, "Dpx": 0.0},
              "radiation": {"computed": True, "emittance_pm": emittance, "I2": 0.1,
                            "I4": 0.001, "I5": 2e-6}}
    result["evaluation_receipt"] = evidence.record_evaluation(candidate, result, effective_periods=8)
    return result


def make_run(tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "artifacts" / "programs" / "trial"
    output.mkdir(parents=True)
    return ResearchRun(workspace=workspace, output_directory=output,
                       arguments_file=output / "arguments.json", result_file=output / "result.json",
                       progress_file=None, cancellation_file=None, working_lattice_file=None,
                       latest_candidate_lattice_file=None, compute_job_id="snapshot-test")


def test_exact_selected_snapshot_is_archived_and_rechecked(tmp_path):
    candidate = cell()
    result = capture(candidate)
    report = evidence.verify_candidate_evaluation(candidate, result, tmp_path)
    assert report["passed"] is True
    assert report["canonical_metrics"]["emittance_pm"] == 10.0
    assert report["table"]["rows"]
    receipt = report["receipt"]
    path = tmp_path / receipt["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    archive = json.loads(path.read_text(encoding="utf-8"))
    assert archive["candidate"] == candidate.as_dict()
    assert archive["binding"]["effective_periods"] == 8
    assert evidence.validate_saved_candidate_evaluation(candidate, receipt, tmp_path)["passed"]
    # Idempotent commit uses the same immutable bytes, not another calculation.
    assert evidence.verify_candidate_evaluation(candidate, result, tmp_path)["receipt"] == receipt


def test_prematch_candidate_cannot_claim_matched_metrics_even_if_both_evaluated(tmp_path):
    prematch, matched = cell(k1=0.4), cell(k1=0.7)
    old = capture(prematch, 1000.0)
    new = capture(matched, 100.0)
    report = evidence.verify_candidate_evaluation(prematch, new, tmp_path)
    assert not report["passed"]
    assert "not the exact snapshot" in report["issues"][0]
    assert evidence.verify_candidate_evaluation(prematch, old, tmp_path)["passed"]
    assert evidence.verify_candidate_evaluation(matched, new, tmp_path)["passed"]


def test_no_implicit_fallback_to_cached_seed(tmp_path):
    candidate = cell()
    capture(candidate)
    report = evidence.verify_candidate_evaluation(candidate, None, tmp_path)
    assert report["status"] == "unverified"
    assert "finish(" in report["next_action"]
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("changed", [cell(k1=0.5), cell(energy=4.0), cell(periods=9), cell(title="Renamed")])
def test_candidate_identity_energy_and_periods_are_not_interchangeable(tmp_path, changed):
    report = evidence.verify_candidate_evaluation(changed, capture(cell()), tmp_path)
    assert not report["passed"]


def test_same_physical_fingerprint_does_not_hide_different_ring_periods(tmp_path):
    original, changed = cell(periods=8), cell(periods=9)
    assert original.physical_fingerprint == changed.physical_fingerprint
    assert not evidence.verify_candidate_evaluation(changed, capture(original), tmp_path)["passed"]


def test_mutated_returned_metrics_do_not_replace_detached_capture(tmp_path):
    candidate = cell()
    result = capture(candidate)
    result["radiation"]["emittance_pm"] = 1.0
    report = evidence.verify_candidate_evaluation(candidate, result, tmp_path)
    assert not report["passed"]
    assert "values were changed" in report["issues"][0]


def test_mutated_lattice_nested_metadata_is_rejected_without_crash(tmp_path):
    candidate = cell()
    result = capture(candidate)
    candidate.metadata["intended_repetitions"] = 12
    report = evidence.verify_candidate_evaluation(candidate, result, tmp_path)
    assert not report["passed"]
    assert "identity mismatch" in report["issues"][0]


def test_unissued_or_edited_descriptor_is_not_trusted(tmp_path):
    candidate = cell()
    result = capture(candidate)
    result["evaluation_receipt"]["content_sha256"] = "0" * 64
    assert not evidence.verify_candidate_evaluation(candidate, result, tmp_path)["passed"]
    result["evaluation_receipt"]["evaluation_id"] = "invented"
    assert not evidence.verify_candidate_evaluation(candidate, result, tmp_path)["passed"]


def test_run_scope_isolation(tmp_path, monkeypatch):
    output_a, output_b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(output_a))
    candidate = cell()
    result = capture(candidate)
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(output_b))
    assert not evidence.verify_candidate_evaluation(candidate, result, output_b)["passed"]


def test_cache_is_bounded_and_eviction_requires_final_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "_MAX_CAPTURED_EVALUATIONS", 2)
    candidate = cell()
    first = capture(candidate)
    capture(cell(k1=0.5))
    latest = capture(candidate)
    assert len(evidence._captured) == 2
    assert not evidence.verify_candidate_evaluation(candidate, first, tmp_path)["passed"]
    assert evidence.verify_candidate_evaluation(candidate, latest, tmp_path)["passed"]


def test_archived_bytes_and_archived_candidate_are_revalidated(tmp_path):
    candidate = cell()
    report = evidence.verify_candidate_evaluation(candidate, capture(candidate), tmp_path)
    receipt = report["receipt"]
    assert not evidence.validate_saved_candidate_evaluation(cell(k1=0.5), receipt, tmp_path)["passed"]
    path = tmp_path / receipt["path"]
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert not evidence.validate_saved_candidate_evaluation(candidate, receipt, tmp_path)["passed"]
    assert not evidence.verify_candidate_evaluation(candidate, capture(candidate), path)["passed"]


@pytest.mark.parametrize("relative", ["../outside.json", "C:/outside.json"])
def test_saved_receipt_cannot_escape_output(tmp_path, relative):
    assert not evidence.validate_saved_candidate_evaluation(cell(), {"path": relative}, tmp_path)["passed"]


def test_api_evaluate_adds_receipt_after_all_requested_physics(tmp_path, monkeypatch):
    candidate = api.cell_ring(title="Synthetic ring", energy_gev=3.0,
                              cell=api.one_period_elements(cell()), periods=8, source="test")
    monkeypatch.setattr(api, "load_pyjutrack", lambda: object())
    monkeypatch.setattr(api, "_linear_evaluation", lambda jt, lat: {"stable": True, "trace_x": 0.2})
    monkeypatch.setattr(api, "_radiation_evaluation", lambda jt, lat: {"emittance_pm": 10.0})
    monkeypatch.setattr(api, "natural_chromaticity", lambda *a, **kw: {"horizontal": -2.0, "vertical": -3.0})
    result = api.evaluate(candidate, include_chromaticity=True)
    report = evidence.verify_candidate_evaluation(candidate, result, tmp_path)
    assert report["passed"]
    assert report["canonical_metrics"]["chromaticity_y"] == -3.0
    assert report["canonical_metrics"]["emittance_pm"] == 10.0
    assert any(row["Quantity"] == "Natural emittance εₓ" and row["Unit"] == "pm rad"
               for row in report["table"]["rows"])
    assert candidate.lattice_id in report["table"]["description"]


def test_api_periodic_evaluation_captures_explicit_periods(tmp_path, monkeypatch):
    from brilliance.jutrack_service import JuTrackService
    candidate = cell()
    monkeypatch.setattr(api, "load_pyjutrack", lambda: object())
    monkeypatch.setattr(JuTrackService, "build_lattice", lambda self, lat: object())
    monkeypatch.setattr(api, "jutrack_linear_metrics", lambda *a, **kw: {"stable": True})
    monkeypatch.setattr(api, "jutrack_periodic_radiation_metrics", lambda *a, **kw: {"emittance_pm": 10.0})
    result = api.evaluate_periodic_cell(candidate, periods=8)
    report = evidence.verify_candidate_evaluation(candidate, result, tmp_path)
    assert report["passed"]
    assert report["binding"]["effective_periods"] == 8


def test_finish_archives_verified_snapshot_and_unverified_branch_without_crash(tmp_path):
    run = make_run(tmp_path)
    candidate = cell()
    result = capture(candidate)
    path = run.finish(summary="Synthetic measurement.", candidate_lattice=candidate,
                      candidate_evaluation=result, scientific_passed=True, request_satisfied=True)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["candidate_evaluation"]["path"]
    assert saved["data"]["candidate_verification"]["passed"]
    assert saved["scientific_passed"] is True
    path = run.finish(summary="Wrong snapshot; preserve this branch.", candidate_lattice=cell(k1=0.7),
                      candidate_evaluation=result, scientific_passed=True,
                      request_satisfied=True, use_as_working_lattice=True)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "partial"
    assert saved["scientific_passed"] is None
    assert saved["request_satisfied"] is False
    assert saved["use_as_working_lattice"] is False
    assert (run.output_directory / saved["candidate_lattice"]).is_file()


def execution_for(tmp_path, result, candidate):
    program = tmp_path / "program.py"
    program.write_text("# Synthetic provenance fixture\n", encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate.as_dict()), encoding="utf-8")
    return {"output_directory": str(tmp_path), "returncode": 0,
            "program": {"snapshot": str(program), "sha256": hashlib.sha256(program.read_bytes()).hexdigest()},
            "result": {"status": "completed", "summary": "A candidate result.",
                       "scientific_passed": True, "request_satisfied": True,
                       "candidate_lattice": candidate_path.name, **result}}


def test_parent_validator_downgrades_unbound_candidate_without_execution_error(tmp_path):
    validation = validate_research_program(None, {}, execution_for(tmp_path, {}, cell()))
    assert validation["passed"] is True
    assert validation["scientific_passed"] is None
    assert validation["request_satisfied"] is False
    assert validation["candidate_verification"]["status"] == "unverified"


def test_parent_validator_accepts_exact_bound_candidate(tmp_path):
    candidate = cell()
    report = evidence.verify_candidate_evaluation(candidate, capture(candidate), tmp_path)
    execution = execution_for(tmp_path, {"candidate_evaluation": report["receipt"]}, candidate)
    validation = validate_research_program(None, {}, execution)
    assert validation["passed"] is True
    assert validation["scientific_passed"] is True
    assert validation["request_satisfied"] is True
    assert validation["checked_values"]["candidate_evaluation_verified"] is True


def test_only_explicit_top_level_candidate_metric_conflicts_are_flagged(tmp_path):
    candidate = cell()
    report = evidence.verify_candidate_evaluation(candidate, capture(candidate), tmp_path)
    check = evidence.compare_declared_candidate_metrics({"emittance_pm": 1000.0, "stable": False,
                                                         "parent_emittance_pm": 2000.0}, report)
    assert not check["passed"]
    assert {item["metric"] for item in check["conflicts"]} == {"emittance_pm", "stable"}
    assert "do not rerun matching" in check["next_action"]
    assert evidence.compare_declared_candidate_metrics(
        {"emittance_pm": 10.000001, "baseline": {"emittance_pm": 2000.0}}, report)["passed"]


def test_parent_validator_downgrades_stale_summary_but_keeps_verified_snapshot(tmp_path):
    candidate = cell()
    report = evidence.verify_candidate_evaluation(candidate, capture(candidate), tmp_path)
    execution = execution_for(tmp_path, {"candidate_evaluation": report["receipt"],
                                         "data": {"emittance_pm": 1234.0}}, candidate)
    validation = validate_research_program(None, {}, execution)
    assert validation["passed"]
    assert validation["candidate_verification"]["passed"]
    assert not validation["candidate_metric_comparison"]["passed"]
    assert validation["scientific_passed"] is None
    assert validation["request_satisfied"] is False


def test_finish_downgrades_stale_metrics_without_rejecting_valid_snapshot(tmp_path):
    run = make_run(tmp_path)
    candidate = cell()
    path = run.finish(summary="Synthetic stale summary.", candidate_lattice=candidate,
                      candidate_evaluation=capture(candidate), data={"emittance_pm": 1000.0},
                      scientific_passed=True, request_satisfied=True, use_as_working_lattice=True)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["candidate_evaluation"]["path"]
    assert saved["data"]["candidate_verification"]["passed"]
    assert not saved["data"]["candidate_metric_comparison"]["passed"]
    assert saved["scientific_passed"] is None
    assert saved["request_satisfied"] is False
    assert saved["use_as_working_lattice"] is False
    assert saved["status"] == "partial"


def test_display_rounding_is_allowed_but_different_candidate_emittance_is_not(tmp_path):
    candidate = cell()
    report = evidence.verify_candidate_evaluation(candidate, capture(candidate, 37.870082), tmp_path)
    assert evidence.compare_declared_candidate_metrics({"emittance_pm": 37.87}, report)["passed"]
    assert not evidence.compare_declared_candidate_metrics({"emittance_pm": 39.335}, report)["passed"]
    # Display comparison never rounds or replaces the authoritative receipt.
    assert report["canonical_metrics"]["emittance_pm"] == 37.870082
