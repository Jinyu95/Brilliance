"""Numerical/control regression tests; toy callbacks are NOT JuTrack validation."""
from __future__ import annotations

import numpy as np
import pytest

from expert_modules.mba_design.periodic_hybrid import (
    SLOT_NAMES,
    HybridContinuationContract,
    HybridContinuationSettings,
    _checked_measurement,
    _profile_measurement,
    _slot_ranges,
    hybrid_constraint_slacks,
    radiation_aware_hybrid_continuation,
)


def contract(**changes):
    values = dict(slot_dispersion_goals_m=dict.fromkeys(SLOT_NAMES, .03),
        core_dispersion_goal_m=.015, core_h_goal_m=3e-5,
        max_emittance_pm=60., max_energy_spread=.002, max_beta_m=40.,
        max_slot_slope=.001, id_dispersion_tolerance_m=1e-4,
        id_slope_tolerance=1e-4, stability_trace_margin=.01,
        dispersion_scale_m=.01, h_scale_m=1e-5)
    return HybridContinuationContract(**{**values, **changes})


def toy(x):
    """A smooth constrained optimization problem, not an accelerator model."""
    u, v = x
    return dict(valid=True, trace_x=.1 + .2*u, trace_y=.4,
        emittance_pm=40. + 10*u*u + 5*v*v, energy_spread=.001,
        beta_max_m=20. + u, core_dispersion_m=.020 - .010*v,
        core_h_m=2e-5 + 1e-5*(u-v)**2, id_dx_m=[0., 0.], id_dpx=[0., 0.],
        slots={name: dict(min_dx_m=.020 + .020*u, max_abs_dpx=0.) for name in SLOT_NAMES})


def run(measure=toy, verify=toy, **kw):
    options = dict(initial_position=[0., 0.], bounds=[[0., 1.], [0., 1.]],
        contract=contract(), measurement_identity="toy-v1-not-beam-physics",
        settings=HybridContinuationSettings(max_attempts=80))
    return radiation_aware_hybrid_continuation(measure, verify, **{**options, **kw})


def test_zero_deformation_preserves_exact_position_without_optimizer(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: pytest.fail("no optimizer at lambda=0"))
    result = run(end_fraction=0.)
    assert result["status"] == "identity_passed"
    assert result["position"] == [0., 0.]
    assert result["fraction"] == 0
    assert result["records"] == []
    assert result["request_satisfied"] is False


def test_modest_continuation_meets_targets_with_radiation_objective():
    result = run()
    assert result["status"] == "continuation_targets_met"
    assert result["fraction"] == 1
    np.testing.assert_allclose(result["position"], [.5, .5], atol=1e-4)
    assert result["measurement"]["emittance_pm"] == pytest.approx(43.75, abs=.005)
    assert result["request_satisfied"] is False
    assert all(step["measurement"]["emittance_pm"] <= 60 + 1e-6 for step in result["accepted_steps"])


def test_intentionally_infeasible_radiation_cap_retains_last_accepted_state():
    result = run(contract=contract(max_emittance_pm=41.))
    assert result["status"] == "continuation_stalled"
    assert 0 < result["fraction"] < 1
    assert result["measurement"]["emittance_pm"] <= 41. + 1e-6
    assert result["position"] == result["accepted_steps"][-1]["position"]
    assert any(not trial["accepted"] for trial in result["records"])
    assert result["contract"]["contract"]["max_emittance_pm"] == 41.
    # The absolute cap does not ratchet upward with each accepted deformation.
    for step in result["accepted_steps"]:
        assert step["measurement"]["emittance_pm"] <= 41. + 1e-6


def test_failed_attempts_restart_from_last_accepted_position():
    result = run(contract=contract(max_emittance_pm=41.))
    previous = [0., 0.]
    for i, record in enumerate(result["records"]):
        assert record["start_position"] == previous
        if record["accepted"]:
            previous = next(s["position"] for s in result["accepted_steps"]
                            if s["fraction"] == record["target_fraction"])
        elif i+1 < len(result["records"]):
            assert result["records"][i+1]["step"] == pytest.approx(record["step"]/2)


def test_good_bump_with_huge_radiation_fails_even_with_valid_flag():
    parent = toy([0., 0.]); bad = toy([.5, .5]); bad["emittance_pm"] = 6000.
    slacks = hybrid_constraint_slacks(bad, parent, contract(), 1.)
    assert slacks["SF_L_dispersion"] == pytest.approx(0.)
    assert slacks["core_dispersion"] == pytest.approx(0.)
    assert slacks["emittance"] < 0


@pytest.mark.parametrize("field,value", [("beta_max_m", 100.), ("energy_spread", .01)])
def test_other_physics_caps_do_not_compensate_each_other(field, value):
    measured = toy([.5, .5]); measured[field] = value
    slacks = hybrid_constraint_slacks(measured, toy([0., 0.]), contract(), 1.)
    assert min(slacks.values()) < 0


def test_missing_slot_measurement_is_an_error_not_infeasibility():
    bad = toy([0., 0.]); del bad["slots"]["SF_R"]
    with pytest.raises(ValueError, match="slot"):
        _checked_measurement(bad)


def test_programming_error_propagates():
    def broken(x):
        if any(x):
            raise AttributeError("bad JuTrack method name")
        return toy(x)
    with pytest.raises(AttributeError, match="bad JuTrack"):
        run(measure=broken)


def test_baseline_replay_mismatch_blocks_search():
    def other(x):
        m = toy(x); m["emittance_pm"] += 1.
        return m
    with pytest.raises(ValueError, match="Zero-deformation"):
        run(verify=other)


def test_selected_candidate_replay_failure_never_promotes():
    def other(x):
        m = toy(x)
        if any(x):
            m["emittance_pm"] += 10.
        return m
    result = run(verify=other)
    assert result["status"] == "continuation_stalled"
    assert result["fraction"] == 0
    assert result["position"] == [0., 0.]
    assert any("replay_failure" in r for r in result["records"])


def test_independent_replay_checks_id_not_only_emittance():
    def other(x):
        m = toy(x)
        if any(x):
            m["id_dx_m"][1] = .01
        return m
    result = run(verify=other)
    assert result["fraction"] == 0


def test_machine_limit_failure_does_not_relax_parent_contract():
    with pytest.raises(ValueError, match="Parent violates"):
        run(contract=contract(max_beta_m=10.))


def test_slope_continuation_starts_from_measured_parent():
    parent = toy([0., 0.])
    for s in parent["slots"].values():
        s["max_abs_dpx"] = .005
    assert hybrid_constraint_slacks(parent, parent, contract(), 0.)["SF_L_slope"] == 0
    assert hybrid_constraint_slacks(parent, parent, contract(), 1.)["SF_L_slope"] < 0


def test_checkpoint_callbacks_cannot_mutate_solver_state():
    def corrupt(copy):
        copy["position"][0] = 999.
        copy["parent_measurement"]["emittance_pm"] = 0.
    result = run(checkpoint=corrupt, end_fraction=.1)
    assert result["position"][0] < 1
    assert result["parent_measurement"]["emittance_pm"] == 40.


def test_cancellation_keeps_last_accepted_state():
    stop = {"value": False}
    def progress(record):
        stop["value"] = record["accepted"]
    result = run(progress=progress, cancelled=lambda: stop["value"])
    assert result["status"] == "cancelled"
    assert len(result["accepted_steps"]) == 1
    assert result["position"] == result["accepted_steps"][-1]["position"]


def test_evaluation_budget_is_bounded():
    result = run(settings=HybridContinuationSettings(max_attempts=2, max_evaluations_per_attempt=2))
    assert all(r["evaluations"] <= 2 for r in result["records"])
    assert result["status"] == "attempt_budget_exhausted"


@pytest.mark.parametrize("setting", [dict(initial_step=0), dict(minimum_step=.2),
    dict(trust_fraction=2), dict(max_attempts=1.5), dict(max_iterations=True)])
def test_invalid_numerical_contract(setting):
    with pytest.raises(ValueError):
        HybridContinuationSettings(**setting)


@pytest.mark.parametrize("change", [dict(max_emittance_pm=0), dict(max_energy_spread=float("nan")),
    dict(stability_trace_margin=2), dict(slot_dispersion_goals_m={"SF_L": .03})])
def test_invalid_physical_contract(change):
    with pytest.raises(ValueError):
        contract(**change)


def test_unstable_callback_keeps_fixed_constraint_shape():
    parent = toy([0., 0.])
    invalid = _checked_measurement(dict(valid=False, trace_x=2.1, trace_y=.4, failure="unstable"))
    good = hybrid_constraint_slacks(parent, parent, contract(), .1)
    bad = hybrid_constraint_slacks(invalid, parent, contract(), .1)
    assert list(good) == list(bad)
    assert bad["stable_x"] < 0 and bad["physical_validity"] < 0


def test_toy_false_solver_success_is_not_accepted(monkeypatch):
    import scipy.optimize
    from types import SimpleNamespace
    def fake(fun, x, **kw):
        fun(x)
        return SimpleNamespace(x=x, success=True, message="success but targets unmet")
    monkeypatch.setattr(scipy.optimize, "minimize", fake)
    result = run()
    assert result["fraction"] == 0
    assert result["status"] == "continuation_stalled"


def slot_fixture():
    left = [dict(name=n, kind=k, length_m=l) for n, k, l in
            [("B1", "SBEND", 1.), ("SF_L", "DRIFT", .3), ("Q", "KQUAD", .2),
             ("SD_L", "DRIFT", .3), ("B2", "SBEND", 1.), ("B3", "SBEND", 1.)]]
    rows = left + [dict(name="B4", kind="SBEND", length_m=1.)]
    rows += [{**r, "name": r["name"].replace("_L", "_R") if "_L" in r["name"] else "R:"+r["name"]}
             for r in reversed(left)]
    slots = {k: [k] for k in SLOT_NAMES}
    roles = {"regions": {"left_bump": {"element_indices": [1,2,3]},
                         "right_bump": {"element_indices": [9,10,11]}}}
    return rows, roles, slots


def test_actual_slots_are_finite_distinct_and_mirrored():
    rows, roles, slots = slot_fixture()
    ranges = _slot_ranges(rows, roles, slots, .25)
    assert all(b-a == pytest.approx(.3) for a,b in ranges.values())


@pytest.mark.parametrize("fault", ["too_short", "magnet", "wrong_side", "overlap", "not_mirrored"])
def test_slot_layout_failures(fault):
    rows, roles, slots = slot_fixture()
    length = .25
    if fault == "too_short": length = .4
    if fault == "magnet": slots["SF_L"] = ["Q"]
    if fault == "wrong_side": slots["SF_L"] = ["SF_R"]
    if fault == "overlap": slots["SD_L"] = ["SF_L"]
    if fault == "not_mirrored": slots["SF_R"], slots["SD_R"] = slots["SD_R"], slots["SF_R"]
    with pytest.raises(ValueError):
        _slot_ranges(rows, roles, slots, length)


def test_stationary_drift_plateau_is_valid_without_isolated_peak():
    # Exact drift transport; no ring/accelerator design is implied by this fixture.
    profile = []
    for s in np.linspace(0, 10, 101):
        profile.append(dict(s_m=s, Dx=.03, Dpx=0., beta_x=10+s*s/10,
                            alpha_x=-s/10, beta_y=10+s*s/10))
    linear = dict(trace_x=.1, trace_y=.4, twiss_profile=profile)
    radiation = dict(emittance_pm=40., energy_spread=.001)
    ranges = dict(SF_L=(1.,2.), SD_L=(3.,4.), SD_R=(6.,7.), SF_R=(8.,9.))
    result = _profile_measurement(linear, radiation, ranges, dict(s_start_m=4., s_end_m=6.))
    assert result["valid"]
    assert result["core_h_m"] == pytest.approx(.03**2/10)
    assert all(s["min_dx_m"] == .03 for s in result["slots"].values())
    assert all(s["max_abs_dpx"] == 0 for s in result["slots"].values())


def test_guidance_no_longer_forces_shape_first_without_radiation():
    from expert_modules.mba_design.acceptance import HYBRID_INSTRUCTIONS
    assert "match_periodic_hybrid" in HYBRID_INSTRUCTIONS
    assert "EXPERIMENTAL" in HYBRID_INSTRUCTIONS
    assert "absolute emittance" in HYBRID_INSTRUCTIONS
    assert "not a mandatory conversion" in HYBRID_INSTRUCTIONS
    assert "No baseline H/I5/emittance ceiling is implicit" not in HYBRID_INSTRUCTIONS


def test_emittance_warning_acknowledges_hard_shape_constraints():
    from expert_modules.mba_design.acceptance import hybrid_optimization_warnings
    message = hybrid_optimization_warnings('api.particle_swarm_optimize(objective_names=["emittance_pm"])')
    assert len(message) == 1
    assert "appropriate when those constraints are active" in message[0]
    assert "Establish outer DBA replacement" not in message[0]


def test_request_classifier_remains_compatible():
    from expert_modules.mba_design.acceptance import is_hybrid_shape_request
    assert is_hybrid_shape_request("Convert current 7BA to a hybrid 7BA.")
    assert not is_hybrid_shape_request("What is a hybrid 7BA?")


def test_tool_registration_exposes_new_route_without_removing_legacy():
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "expert_modules/mba_design/tools.py"
    tree = ast.parse(source.read_text())
    assignments = {n.targets[0].id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert {"match_periodic_hybrid", "explore_hybrid_mba", "match_outer_dba_section", "tme_targets"} <= assignments
