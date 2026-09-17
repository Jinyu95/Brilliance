"""Intent selects physical outcomes; no deleted controller route is retained."""
from __future__ import annotations

import pytest

from brilliance.experts import ExpertRegistry
from brilliance.native_research_agent import _completion_proof
from brilliance.scientific_acceptance import (
    acceptance_instructions,
    continued_request_acceptance,
    optimization_acceptance_warnings,
    receipt_acceptance_issues,
    request_acceptance_requirements,
)
from expert_modules.mba_design.acceptance import (
    CONVENTIONAL_LINEAR,
    HYBRID_LINEAR,
    MBA_OPTIMIZATION,
)


@pytest.fixture(scope="module")
def registry():
    return ExpertRegistry.discover()


@pytest.mark.parametrize("user_text", [
    "Design a fouth-generation synchrotron light source with a 3.5 GeV, 32-cell 7BA storage ring. "
    "Circumference of the ring is around 800 m. Make a prototype first without considering nonlinear dynamics.",
    "Design a 7BA electron storage ring with 32 identical cells.",
    "Build a 6BA ring.", "We need a conventional 9BA lattice.", "Create a multi-bend achromat.",
    "Design a conventional 7BA, not a hybrid lattice.",
])
def test_fresh_conventional_request_requires_unit_transfer_and_full_cell(registry, user_text):
    assert request_acceptance_requirements(user_text, registry=registry) == (CONVENTIONAL_LINEAR,)


@pytest.mark.parametrize("user_text", [
    "We need a hybrid 7BA lattice with optimized emittance.", "I want a hybrid 9BA with optimised emittance.",
    "Design a H-6BA and optimize its emittance.", "Optimize the emittance of this H7BA.",
])
def test_hybrid_optimization_has_outcomes_not_an_algorithm(registry, user_text):
    assert set(request_acceptance_requirements(user_text, registry=registry)) == {HYBRID_LINEAR, MBA_OPTIMIZATION}


@pytest.mark.parametrize("user_text", [
    "Convert the accepted 7BA to a hybrid 7BA.", "I would like a hybrid MBA lattice.",
    "Could you create a H9BA?", "Create two outer dispersion bumps.",
    "Do not use an old controller. Design a hybrid 7BA.",
])
def test_hybrid_outcome_does_not_require_unrequested_optimization(registry, user_text):
    assert request_acceptance_requirements(user_text, registry=registry) == (HYBRID_LINEAR,)


@pytest.mark.parametrize("user_text", [
    "What is a hybrid 7BA?", "Explain how to design a 7BA.", "How would I build an MBA?",
    "Show instructions for designing a 7BA.", "I want a tutorial about a 7BA.",
    "Plot the optics of my hybrid 7BA.", "Create a one-cell optics plot of this 7BA.",
    "Give me the chromaticity of the hybrid 7BA.", "We need the chromaticities of this 7BA.",
    "List all quadrupoles in the H7BA cell.", "Correct the chromaticity of the hybrid 7BA.",
    "We need a plot for the 7BA.", "Do not design a 7BA.", "Don't build a 9BA.",
    "No need to design a 7BA.", "Do not hybridize the lattice.",
    "Optimize the current conventional 7BA.", "Change the current 7BA straight length.",
])
def test_diagnostics_modifications_negations_and_instructions_do_not_force_new_design(registry, user_text):
    assert request_acceptance_requirements(user_text, registry=registry) == ()


def test_upfront_guidance_describes_measured_deliverables(registry):
    conventional = acceptance_instructions((CONVENTIONAL_LINEAR,), registry=registry)
    assert "modified-TME unit" in conventional and "exact transfer" in conventional
    hybrid = acceptance_instructions((HYBRID_LINEAR, MBA_OPTIMIZATION), registry=registry)
    assert "HybridMeasurement" in hybrid and "No PSO generation" in hybrid
    assert "approved radiation budget" in hybrid


def test_general_matcher_has_no_method_selected_warning(registry):
    required = request_acceptance_requirements("Design a hybrid 7BA", registry=registry)
    assert optimization_acceptance_warnings("result = match_line(rows)", required, registry=registry) == []


@pytest.mark.parametrize("user_text,required", [
    ("Design a 7BA", {CONVENTIONAL_LINEAR}),
    ("Design a hybrid 7BA with optimized emittance", {HYBRID_LINEAR, MBA_OPTIMIZATION}),
])
def test_continuation_preserves_pending_outcomes_not_empty_success(registry, user_text, required):
    history = [{"role": "user", "content": user_text}, {"role": "assistant", "meta": {
        "completion_evidence": {"scientific_passed": True, "scientific_acceptance": {"checks": {}}}}}]
    assert set(continued_request_acceptance("Continue", history, registry=registry)) == required
    assert continued_request_acceptance("Plot the 7BA optics", history, registry=registry) == ()
    assert len(receipt_acceptance_issues({"validation": {"scientific_acceptance": {"checks": {}}}}, tuple(required))) == len(required)


def test_only_independent_matching_completion_clears_scope(registry):
    required = (CONVENTIONAL_LINEAR,)
    receipt = {"kind": "completed_execution", "execution_valid": True, "scientific_passed": True,
        "status": "completed", "run_id": "verified-run", "request_satisfied": True,
        "validation": {"scientific_acceptance": {"checks": {name: {"passed": True} for name in required}}}}
    history = [{"role": "user", "content": "Design a 7BA"}, {"role": "assistant", "meta": {
        "completion_evidence": _completion_proof(receipt, decision_authority="test")}}]
    assert continued_request_acceptance("Continue", history, registry=registry) == ()
