"""Scientific chat follows committed evidence, not the newest output directory."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from brilliance.chat_service import _rehydrate_completion_message, _separate_research_updates
from brilliance.codex_runtime import _turn_attachments, workspace_research_updates
from brilliance.presentation import deduplicate_chat_attachments


def saved_result(tmp_path, run_id, *, value, modified_at, working=False):
    directory = tmp_path / "project" / "workspace" / "artifacts" / "programs" / run_id
    directory.mkdir(parents=True)
    candidate = directory / "candidate_lattice.json"
    candidate.write_text(json.dumps({"lattice_id": "candidate-" + run_id}), encoding="utf-8")
    figure = directory / "optics.svg"
    figure.write_text(f'<svg xmlns="http://www.w3.org/2000/svg"><text>{value}</text></svg>', encoding="utf-8")
    result = directory / "result.json"
    result.write_text(json.dumps({
        "summary": f"Measured objective {value}", "status": "completed",
        "data": {"objective": value}, "scientific_passed": True,
        "tables": [{"title": "Measured objective", "rows": [{"objective": value}]}],
        "plots": [{"path": "optics.svg", "title": "Candidate optics"}],
        "candidate_lattice": candidate.name,
        "candidate_handoff": {"disposition": "working_selection_requested" if working else "archived_branch"},
        "scientific_stage": {"stage_id": "linear-feasibility", "outcome": "feasible"},
    }), encoding="utf-8")
    for path in directory.iterdir():
        os.utime(path, (modified_at, modified_at))
    return {"evidence_run_id": run_id, "evidence_full_result": str(result),
            "evidence_candidate_lattice": str(candidate), "execution_valid": True,
            "scientific_passed": True, "working_lattice_action": "advance" if working else "archive"}


def test_final_answer_uses_exact_older_receipt_and_restores_its_figures(tmp_path):
    proof = saved_result(tmp_path, "program-old", value=10, modified_at=10, working=True)
    saved_result(tmp_path, "program-new", value=12, modified_at=200)
    attachments = _turn_attachments(tmp_path, "project", 100, completion_evidence=proof)
    cards = [row for row in attachments if row["type"] == "evidence_summary"]
    assert len(cards) == 1
    assert cards[0]["run_id"] == "program-old"
    assert cards[0]["lattice_id"] == "candidate-program-old"
    assert cards[0]["summary"] == "Measured objective 10"
    assert cards[0]["presentation_role"] == "selected_evidence"
    assert cards[0]["candidate_handoff"]["disposition"] == "working_selection_requested"
    assert cards[0]["working_lattice_action"] == "advance"
    images = [row for row in attachments if row["type"] == "image"]
    assert len(images) == 1
    assert images[0]["path"] == "programs/program-old/optics.svg"
    assert images[0]["presentation_role"] == "selected_evidence"
    assert images[0]["lattice_id"] == "candidate-program-old"
    downloads = [row for row in attachments if row["type"] == "download"]
    assert [row["path"] for row in downloads] == ["programs/program-old/candidate_lattice.json"]
    alternatives = next(row for row in attachments if row["type"] == "alternative_results")
    assert alternatives["rows"] == [{"run_id": "program-new", "summary": "Measured objective 12",
                                     "status": "completed", "disposition": "not_selected_for_this_answer"}]


def test_answer_evidence_selection_does_not_claim_working_lattice_selection(tmp_path):
    proof = saved_result(tmp_path, "program-branch", value=10, modified_at=10)
    attachments = _turn_attachments(tmp_path, "project", 100, completion_evidence=proof)
    assert attachments[0]["presentation_role"] == "selected_evidence"
    assert attachments[0]["candidate_handoff"]["disposition"] == "archived_branch"


def test_broken_receipt_never_falls_back_to_newer_candidate(tmp_path):
    proof = saved_result(tmp_path, "program-old", value=10, modified_at=10)
    saved_result(tmp_path, "program-new", value=12, modified_at=200)
    proof["evidence_run_id"] = "wrong-run"
    result = _turn_attachments(tmp_path, "project", 100, completion_evidence=proof)
    assert len(result) == 1
    assert result[0]["presentation_role"] == "selected_evidence_unavailable"
    assert "12" not in result[0]["summary"]


def test_completion_cannot_attach_a_different_candidate_or_another_project(tmp_path):
    proof = saved_result(tmp_path, "program-old", value=10, modified_at=10)
    other = saved_result(tmp_path, "program-new", value=12, modified_at=200)
    proof["evidence_candidate_lattice"] = other["evidence_candidate_lattice"]
    result = _turn_attachments(tmp_path, "project", 100, completion_evidence=proof)
    assert result[0]["presentation_role"] == "selected_evidence_unavailable"
    result = _turn_attachments(tmp_path, "different-project", 100, completion_evidence=other)
    assert result[0]["presentation_role"] == "selected_evidence_unavailable"


def test_all_scientific_milestones_survive_many_runtime_updates(tmp_path):
    directory = tmp_path / "project" / "workspace" / "progress" / "events" / "run-1"
    directory.mkdir(parents=True)
    first = {"id": "contract", "timestamp": "0000", "stage": "Machine contract", "kind": "finding",
             "scientific_role": "milestone", "stage_id": "contract", "hypothesis_id": "route-A",
             "candidate_id": "candidate-A", "parent_candidate_id": "parent-A", "outcome": "validated",
             "interpretation": "The proposed geometry obeys the declared contract.",
             "next_decision": "Evaluate the optics."}
    (directory / "contract.json").write_text(json.dumps(first), encoding="utf-8")
    for index in range(70):
        update = {"id": f"runtime-{index}", "timestamp": f"{index + 1:04d}",
                  "stage": "Starting saved research program", "kind": "progress"}
        (directory / f"runtime-{index}.json").write_text(json.dumps(update), encoding="utf-8")
    updates = workspace_research_updates(tmp_path, "project", limit=None)
    assert len(updates) == 71
    milestones, activity = _separate_research_updates(updates)
    assert len(activity) == 70
    assert len(milestones) == 1
    for key in ("stage_id", "hypothesis_id", "candidate_id", "parent_candidate_id",
                "outcome", "scientific_role", "interpretation", "next_decision"):
        assert milestones[0][key] == first[key]
    assert len(workspace_research_updates(tmp_path, "project")) == 12


def test_explicit_scientific_progress_and_legacy_scientific_artifacts_stay_visible():
    updates = [{"kind": "progress", "scientific_role": "milestone", "stage": "Feasible region"},
               {"kind": "progress", "stage": "Measured response", "tables": [{"rows": []}]},
               {"kind": "repair", "stage": "Calculation failed"},
               {"kind": "finding", "stage": "Runtime", "scientific_role": "activity"}]
    milestones, activity = _separate_research_updates(updates)
    assert milestones == updates[:3]
    assert activity == updates[3:]


def test_selected_final_evidence_is_whole_even_after_progress_shows_same_artifacts(tmp_path):
    proof = saved_result(tmp_path, "program-old", value=10, modified_at=10, working=True)
    selected = _turn_attachments(tmp_path, "project", 100, completion_evidence=proof)
    progress = copy.deepcopy(selected)
    for item in progress:
        item.pop("presentation_role", None)
    messages = [{"id": "scientist", "role": "user", "content": "Evaluate it"},
                {"id": "progress", "role": "assistant", "attachments": progress},
                {"id": "final", "role": "assistant", "attachments": copy.deepcopy(selected)},
                {"id": "duplicate", "role": "assistant", "attachments": copy.deepcopy(selected)}]
    deduplicate_chat_attachments(messages)
    final = messages[2]["attachments"]
    assert any(item["type"] == "image" for item in final)
    assert next(item for item in final if item["type"] == "evidence_summary")["tables"]
    assert not messages[3]["attachments"]
    snapshot = copy.deepcopy(messages)
    deduplicate_chat_attachments(messages)
    assert messages == snapshot


def test_native_help_exposes_actual_scientific_reporting_signatures():
    from brilliance.native_research_agent import api_help
    reference = api_help("# API\n\nGeneral research tools.", "finish progress", max_chars=20000)
    assert "run.finish(" in reference
    assert "scientific_stage" in reference
    assert "run.progress(" in reference
    for name in ("stage_id", "hypothesis_id", "candidate_id", "parent_candidate_id",
                 "outcome", "scientific_role", "interpretation", "next_decision"):
        assert name in reference


def test_legacy_final_view_rehydrates_selected_receipt_without_changing_history(tmp_path):
    proof = saved_result(tmp_path, "program-old", value=10, modified_at=10, working=True)
    saved_result(tmp_path, "program-new", value=12, modified_at=200)
    message = {"role": "assistant", "content": "Original saved answer; attached 0 figures.",
               "meta": {"kind": "result", "completion_evidence": proof},
               "attachments": [{"type": "evidence_summary", "run_id": "program-new",
                                "summary": "Measured objective 12", "status": "completed"}]}
    history = tmp_path / "project" / "conversation.json"
    history.write_text(json.dumps(message), encoding="utf-8")
    original = history.read_bytes()
    snapshot = _rehydrate_completion_message(tmp_path, "project", message)
    assert snapshot["content"] == message["content"]
    assert message["attachments"][0]["run_id"] == "program-new"
    assert snapshot["attachments"][0]["run_id"] == "program-old"
    assert "1 figures" in snapshot["attachments"][0]["history_rehydrated"]
    assert any(row["type"] == "image" for row in snapshot["attachments"])
    assert history.read_bytes() == original
    assert _rehydrate_completion_message(tmp_path, "project", snapshot) is snapshot
    # Rendering mutation must not corrupt the cached, immutable source view.
    snapshot["attachments"][0]["tables"].clear()
    fresh = _rehydrate_completion_message(tmp_path, "project", message)
    assert fresh["attachments"][0]["tables"]


def test_historical_message_without_validated_proof_is_not_reinterpreted(tmp_path):
    proof = saved_result(tmp_path, "program-old", value=10, modified_at=10)
    proof["scientific_passed"] = False
    message = {"role": "assistant", "meta": {"completion_evidence": proof}, "attachments": []}
    assert _rehydrate_completion_message(tmp_path, "project", message) is message


def test_candidate_file_identity_overrides_stale_authored_data_and_canonical_metrics_are_not_duplicated(tmp_path):
    proof = saved_result(tmp_path, "program-old", value=10, modified_at=10)
    result_path = Path(proof["evidence_full_result"])
    value = json.loads(result_path.read_text(encoding="utf-8"))
    value["data"] = {"lattice_id": "wrong-candidate", "emittance_pm": 10,
                     "verified_candidate_metrics": {"emittance_pm": 10},
                     "design_status": "MISLEADING_IMPROVEMENT_FLAG", "engine": "internal-engine",
                     "declared_mirror_mismatch_count": 0, "scientific_passed": True,
                     "independent_diagnostic": 3.0}
    value["tables"] = [{"kind": "candidate_evaluation", "title": "Verified candidate",
                        "rows": [{"quantity": "Natural emittance εₓ", "value": 10, "unit": "pm rad"}]}]
    result_path.write_text(json.dumps(value), encoding="utf-8")
    card = _turn_attachments(tmp_path, "project", 100, completion_evidence=proof)[0]
    assert card["lattice_id"] == "candidate-program-old"
    quantities = next(table for table in card["tables"] if table["title"] == "Computed quantities")
    assert quantities["kind"] == "automatic_quantities"
    assert quantities["rows"] == [{"quantity": "independent_diagnostic", "value": 3.0}]


def test_selected_receipt_retains_all_declared_figures_and_keeps_lattice_download_first(tmp_path):
    proof = saved_result(tmp_path, "program-old", value=10, modified_at=10)
    result_path = Path(proof["evidence_full_result"])
    value = json.loads(result_path.read_text(encoding="utf-8"))
    for index in range(12):
        figure = f"figure-{index}.svg"
        data = f"data-{index}.csv"
        (result_path.parent / figure).write_text(f'<svg><text>{index}</text></svg>', encoding="utf-8")
        (result_path.parent / data).write_text(f"data\n{index}\n", encoding="utf-8")
        value["plots"].append({"path": figure})
        value.setdefault("downloads", []).append({"path": data})
    result_path.write_text(json.dumps(value), encoding="utf-8")
    attachments = _turn_attachments(tmp_path, "project", 100, completion_evidence=proof)
    assert sum(item["type"] == "image" for item in attachments) == 13
    downloads = [item for item in attachments if item["type"] == "download"]
    assert len(downloads) == 13
    assert downloads[0]["artifact_role"] == "candidate_definition"


def test_typed_result_without_plots_never_promotes_undeclared_stale_image(tmp_path):
    proof = saved_result(tmp_path, "program-old", value=10, modified_at=10)
    result_path = Path(proof["evidence_full_result"])
    value = json.loads(result_path.read_text(encoding="utf-8"))
    value["plots"] = []
    result_path.write_text(json.dumps(value), encoding="utf-8")
    attachments = _turn_attachments(tmp_path, "project", 100, completion_evidence=proof)
    assert not any(item["type"] == "image" for item in attachments)


def test_unfinished_program_figure_is_marked_unverified(tmp_path):
    artifact = tmp_path / "project" / "workspace" / "artifacts" / "unfinished.svg"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    attachments = _turn_attachments(tmp_path, "project", 0)
    assert len(attachments) == 1
    assert attachments[0]["presentation_role"] == "unverified_artifact"
    assert "unverified output" in attachments[0]["title"]


@pytest.mark.parametrize("prefix", ["workspace/artifacts/", "artifacts/", ""])
def test_native_completion_proof_resolves_supported_relative_receipt_paths(tmp_path, prefix):
    from brilliance.native_research_agent import _completion_proof
    saved_result(tmp_path, "program-old", value=10, modified_at=10, working=True)
    evidence = {"kind": "completed_execution", "execution_valid": True, "scientific_passed": True,
                "status": "completed", "request_satisfied": True, "run_id": "program-old",
                "full_result": prefix + "programs/program-old/result.json",
                "candidate_lattice": prefix + "programs/program-old/candidate_lattice.json",
                "working_lattice_action": "advance"}
    proof = _completion_proof(evidence, decision_authority="prose_completion_after_validated_request_result")
    attachments = _turn_attachments(tmp_path, "project", 100, completion_evidence=proof)
    assert attachments[0]["run_id"] == "program-old"
    assert attachments[0]["lattice_id"] == "candidate-program-old"
    assert attachments[0]["working_lattice_action"] == "advance"
    assert any(item["type"] == "image" for item in attachments)
    message = {"role": "assistant", "content": "Saved result.", "attachments": [],
               "meta": {"completion_evidence": proof}}
    restored = _rehydrate_completion_message(tmp_path, "project", message)
    assert restored["attachments"][0]["lattice_id"] == "candidate-program-old"
