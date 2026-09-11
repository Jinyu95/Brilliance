from __future__ import annotations

import brilliance.scientific_acceptance as acceptance


def _requirements(text: str, *, registry=None) -> tuple[str, ...]:
    lowered = text.casefold()
    if "hybrid 7ba" in lowered or "dispersion bumps" in lowered:
        return ("hybrid_mba_shape",)
    if "other scientific request" in lowered:
        return ("other_check",)
    return ()


def _history(*messages):
    return list(messages)


def test_direct_request_owns_its_acceptance(monkeypatch):
    monkeypatch.setattr(acceptance, "request_acceptance_requirements", _requirements)
    assert acceptance.continued_request_acceptance(
        "Convert the current lattice to a hybrid 7BA with two dispersion bumps."
    ) == ("hybrid_mba_shape",)


def test_option_reply_inherits_pending_acceptance(monkeypatch):
    monkeypatch.setattr(acceptance, "request_acceptance_requirements", _requirements)
    history = _history(
        {"role": "user", "content": "Convert the current lattice to a hybrid 7BA."},
        {"role": "assistant", "content": "Choose (a) or (b)."},
    )
    assert acceptance.continued_request_acceptance("b", history) == ("hybrid_mba_shape",)


def test_amendment_inherits_pending_acceptance(monkeypatch):
    monkeypatch.setattr(acceptance, "request_acceptance_requirements", _requirements)
    history = _history(
        {"role": "user", "content": "Create two dispersion bumps in the current 7BA."},
        {"role": "assistant", "content": "The first trial is complete."},
    )
    assert acceptance.continued_request_acceptance(
        "Increase the bump target to 40 mm.", history
    ) == ("hybrid_mba_shape",)
    assert acceptance.continued_request_acceptance(
        "Continue, but keep the circumference fixed.", history
    ) == ("hybrid_mba_shape",)


def test_followup_chain_skips_prior_option_reply(monkeypatch):
    monkeypatch.setattr(acceptance, "request_acceptance_requirements", _requirements)
    history = _history(
        {"role": "user", "content": "Convert the current lattice to a hybrid 7BA."},
        {"role": "assistant", "content": "Choose (a) or (b)."},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "I changed the declared length bound."},
    )
    assert acceptance.continued_request_acceptance("continue", history) == (
        "hybrid_mba_shape",
    )


def test_unrelated_new_request_does_not_inherit(monkeypatch):
    monkeypatch.setattr(acceptance, "request_acceptance_requirements", _requirements)
    history = _history(
        {"role": "user", "content": "Convert the current lattice to a hybrid 7BA."},
        {"role": "assistant", "content": "The search is still pending."},
    )
    assert acceptance.continued_request_acceptance("What files are in the workspace?", history) == ()


def test_new_scientific_request_replaces_pending_scope(monkeypatch):
    monkeypatch.setattr(acceptance, "request_acceptance_requirements", _requirements)
    history = _history(
        {"role": "user", "content": "Convert the current lattice to a hybrid 7BA."},
        {"role": "assistant", "content": "The search is still pending."},
    )
    assert acceptance.continued_request_acceptance(
        "Start another scientific request.", history
    ) == ("other_check",)


def test_saved_completion_ends_inheritance(monkeypatch):
    monkeypatch.setattr(acceptance, "request_acceptance_requirements", _requirements)
    history = _history(
        {"role": "user", "content": "Convert the current lattice to a hybrid 7BA."},
        {
            "role": "assistant",
            "content": "Completed.",
            "meta": {
                "runtime": "native_research_agent",
                "kind": "result",
                "request_satisfied": True,
            },
        },
    )
    assert acceptance.continued_request_acceptance("continue", history) == ()
