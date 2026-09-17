"""Normal prose completion must bind attachments as explicitly selected completion does."""
import pytest

from brilliance.native_research_agent import _completion_proof, _result_ready_for_answer


def receipt():
    return {"kind": "completed_execution", "execution_valid": True, "scientific_passed": True,
            "status": "completed", "request_satisfied": True, "run_id": "run-A",
            "full_result": "workspace/artifacts/programs/run-A/result.json",
            "candidate_lattice": "workspace/artifacts/programs/run-A/candidate_lattice.json",
            "working_lattice_action": "advance", "summary": "A measured candidate.",
            "data": {"emittance_pm": 50.0}}


def test_prose_and_explicit_completion_share_exact_selected_receipt():
    evidence = receipt()
    assert _result_ready_for_answer(evidence)
    explicit = _completion_proof(evidence, decision_authority="explicit")
    prose = _completion_proof(evidence, decision_authority="prose")
    assert explicit["evidence_receipt_sha256"] == prose["evidence_receipt_sha256"]
    assert prose["evidence_run_id"] == "run-A"
    assert prose["working_lattice_action"] == "advance"
    assert prose["evidence_candidate_lattice"] == evidence["candidate_lattice"]


@pytest.mark.parametrize("field,value", [("execution_valid", False), ("scientific_passed", None),
                                        ("status", "partial"), ("run_id", None)])
def test_unverified_or_failed_candidate_cannot_acquire_completion_proof(field, value):
    evidence = {**receipt(), field: value}
    with pytest.raises(ValueError, match="valid completed scientific"):
        _completion_proof(evidence, decision_authority="prose")


def test_intermediate_result_requires_explicit_scope_decision_before_prose_completion():
    evidence = {**receipt(), "request_satisfied": False}
    assert not _result_ready_for_answer(evidence)
