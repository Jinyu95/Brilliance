"""Bounded recovery of lattice representation mistakes, not physics rejection."""
from __future__ import annotations

import pytest

from brilliance.research_recovery import coding_failure_summary, program_failure_guidance


@pytest.mark.parametrize("headline", [
    "LatticeScopeError: periodic cell metadata conflicts with the saved physical rows",
    "brilliance.lattice.LatticeScopeError: explicit full-ring partition is inconsistent",
    "ValueError: lattice does not declare an integer periodic-cell partition",
    "ValueError: invalid cell_element_count in the lattice archive",
])
def test_scope_metadata_errors_receive_bounded_coding_recovery(headline):
    traceback = (
        'Traceback (most recent call last):\n'
        '  File "program.py", line 19, in main\n'
        '    rows = api.one_period_elements(candidate)\n'
        + headline + '\n'
    )
    assert coding_failure_summary(traceback) == headline
    guidance = program_failure_guidance(traceback)
    assert "one periodic cell" in guidance["explanation"]
    assert "not evidence" in guidance["explanation"]
    hint = guidance["repair_hint"]
    assert "exact archived candidate" in hint
    assert "machine contract in this project" in hint
    assert "api.periodic_cell(..., intended_repetitions=P)" in hint
    assert "api.cell_ring(..., periods=P)" in hint
    assert "Preserve every physical element" in hint
    assert "Do not infer scope from bend angle alone" in hint
    assert "do not regenerate a seed or restart optimization" in hint


@pytest.mark.parametrize("error", [
    "ValueError: no stable periodic solution",
    "ValueError: periodic cell has non-positive horizontal damping",
    "ValueError: periodic ring bend angle does not close",
    "ValueError: required dispersion matching tolerance was not reached",
    "ValueError: invalid cell_element_count in the lattice archive: physics rejection",
    "RuntimeError: Julia bridge failed to initialize",
    "PermissionError: [WinError 5] Access denied",
    "TimeoutError: bounded calculation ended",
])
def test_unrelated_physics_and_runtime_failures_are_not_reclassified(error):
    assert coding_failure_summary(error) is None
    assert program_failure_guidance(error)["repair_hint"] == ""


@pytest.mark.parametrize("error, expected_hint", [
    ("LatticeTopologyError: JuTrack row synchronization cannot change topology", "new editor"),
    ("ValueError: JuTrack row synchronization cannot change topology", "new editor"),
    ("ValueError: population_size=160 is below the declared production minimum of 240 for 24 outer knobs",
     "api.plan_lattice_optimization"),
])
def test_existing_targeted_repairs_remain_distinct(error, expected_hint):
    assert coding_failure_summary(error) == error
    assert expected_hint in program_failure_guidance(error)["repair_hint"]
