"""Request-policy regressions; no model API, Julia startup, or saved project needed.

Load the actual policy modules directly so this focused suite requires only
Python and pytest, not the application's optional runtime integrations.
"""
from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acceptance = _load_module("acceptance_policy", "brilliance/scientific_acceptance.py")
mba = _load_module("mba_request_policy", "expert_modules/mba_design/acceptance.py")
HYBRID = ("hybrid_mba_shape",)
REQUEST = "Convert the current 7BA to a hybrid 7BA with two dispersion bumps."


def _user(content):
    return {"role": "user", "content": content}


@pytest.fixture
def registry():
    # The real MBA request classifier is used, not a substring approximation.
    # A second check exercises the topic-neutral host's replacement behavior.
    checks = (
        SimpleNamespace(id=HYBRID[0], applies_to=mba.is_hybrid_shape_request,
                        module_id="laboratory.mba_design", requires_lattice=True,
                        validator=lambda report, lattice: []),
        SimpleNamespace(id="orbit_response", applies_to=lambda text: text.startswith("Fit an orbit")),
    )
    return SimpleNamespace(
        acceptance_checks=checks,
        acceptance_check=lambda name: next(check for check in checks if check.id == name),
        module=lambda name: SimpleNamespace(version="test"),
    )


@pytest.fixture
def pending():
    return [_user(REQUEST), {"role": "assistant", "content": "Choose (a) or (b)."}]


@pytest.mark.parametrize("reply", [
    "b", "(b)", "[B]", "option b", "Option (b).", "2", "12.",
    "continue", "Please resume.", "go ahead", "继续", "yes", "No", "OK.",
    "do that", "Increase the bump target to 40 mm.", "Reduce the core dispersion.",
    "Continue, but keep the circumference fixed.", "Please keep the ID straight unchanged.",
    "Release the core quadrupoles.", "Choose option b.", "Yes, increase the target.",
])
def test_replies_inherit_pending_requirements(reply, registry, pending):
    assert acceptance.continued_request_acceptance(reply, pending, registry=registry) == HYBRID


@pytest.mark.parametrize("message", [
    "What files are in the workspace?", "What is this?", "Explain the previous result.",
    "What is a hybrid 7BA?", "Why does this fail?", "List the current files.",
    "Write a poem.", "hi", "Cancel this task.", "No, cancel this task.",
    "Do not create a hybrid 7BA.", "Start a new task: plot an orbit.", "", "   ",
])
def test_inspection_or_new_scope_does_not_inherit(message, registry, pending):
    assert acceptance.continued_request_acceptance(message, pending, registry=registry) == ()


def test_explicit_request_owns_its_scope(registry, pending):
    assert acceptance.continued_request_acceptance(REQUEST, registry=registry) == HYBRID
    assert acceptance.continued_request_acceptance(
        "Fit an orbit response matrix.", pending, registry=registry
    ) == ("orbit_response",)


def test_option_and_amendment_chain(registry, pending):
    history = pending + [
        _user("b"), {"role": "assistant", "content": "A candidate remains unaccepted."},
        _user("Increase the bump target to 40 mm."),
        {"role": "assistant", "content": "The intermediate calculation finished."},
    ]
    assert acceptance.continued_request_acceptance("continue", history, registry=registry) == HYBRID


def test_current_message_may_already_be_in_history(registry, pending):
    history = pending + [_user("b")]
    assert acceptance.continued_request_acceptance("b", history, registry=registry) == HYBRID


def test_history_is_not_mutated(registry, pending):
    original = deepcopy(pending)
    acceptance.continued_request_acceptance("b", pending, registry=registry)
    assert pending == original


@pytest.mark.parametrize("reply", ["b", "continue", "Increase the target."])
def test_reply_without_pending_request_has_no_requirements(reply, registry):
    assert acceptance.continued_request_acceptance(reply, registry=registry) == ()


@pytest.mark.parametrize("meta", [
    {"completion_evidence": {"run_id": "completed-run"}},
    {"runtime": "native_research_agent", "kind": "result", "request_satisfied": True},
])
def test_saved_completion_ends_inheritance(meta, registry, pending):
    history = pending + [{"role": "assistant", "content": "Done", "meta": meta}]
    assert acceptance.continued_request_acceptance("continue", history, registry=registry) == ()
    assert acceptance.continued_request_acceptance(REQUEST, history, registry=registry) == HYBRID


@pytest.mark.parametrize("meta", [
    {}, None, {"runtime": "native_research_agent", "kind": "result", "request_satisfied": False},
    {"runtime": "native_research_agent", "kind": "progress", "request_satisfied": True},
])
def test_intermediate_progress_is_not_completion(meta, registry, pending):
    history = pending + [{"role": "assistant", "content": "Calculation done", "meta": meta}]
    assert acceptance.continued_request_acceptance("continue", history, registry=registry) == HYBRID


def test_substantive_topic_change_stops_history_walk(registry, pending):
    history = pending + [_user("Write a poem."), {"role": "assistant", "content": "A poem."}]
    assert acceptance.continued_request_acceptance("continue", history, registry=registry) == ()


def test_new_expert_scope_is_used_by_next_followup(registry, pending):
    history = pending + [_user("Fit an orbit response matrix."), _user("keep the same settings")]
    assert acceptance.continued_request_acceptance("continue", history, registry=registry) == (
        "orbit_response",
    )


def test_blank_history_rows_do_not_erase_pending_request(registry, pending):
    history = pending + [_user("  "), {"role": "tool", "content": "diagnostics"}]
    assert acceptance.continued_request_acceptance("continue", history, registry=registry) == HYBRID


def test_inherited_requirement_rejects_missing_receipt_evidence(registry, pending):
    required = acceptance.continued_request_acceptance("b", pending, registry=registry)
    issues = acceptance.receipt_acceptance_issues({"validation": {"passed": True}}, required)
    assert len(issues) == 1 and HYBRID[0] in issues[0]


def test_inherited_requirement_leaves_missing_report_pending(registry, pending):
    required = acceptance.continued_request_acceptance("b", pending, registry=registry)
    verdict = acceptance.assess_scientific_acceptance({}, required=required, registry=registry)
    assert verdict["status"] == "pending"
    assert verdict["passed"] is False


def test_validated_receipt_is_not_blocked(registry, pending):
    required = acceptance.continued_request_acceptance("b", pending, registry=registry)
    receipt = {"validation": {"scientific_acceptance": {"checks": {
        HYBRID[0]: {"passed": True, "issues": []},
    }}}}
    assert acceptance.receipt_acceptance_issues(receipt, required) == []
