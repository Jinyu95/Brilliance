"""Regression controls and saved measurements, not new JuTrack simulations."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_periodic_hybrid import contract, run, toy
from expert_modules.mba_design.periodic_hybrid import (
    HybridContinuationSettings, _check_target_direction, hybrid_continuation_summary,
)


def test_resource_cutoff_never_halves_to_a_false_stall():
    result = run(settings=HybridContinuationSettings(
        initial_step=.005, minimum_step=.005, maximum_step=.005,
        max_attempts=5, max_evaluations_per_attempt=2))
    assert result['status'] == 'evaluation_budget_exhausted'
    assert len(result['records']) == 1
    assert result['records'][0]['step'] == .005
    assert result['records'][0]['next_action'].startswith('review_numerical_budget')
    assert result['has_accepted_progress'] is False


def test_iteration_limit_is_not_physical_stall(monkeypatch):
    import scipy.optimize
    def limited(fun, x, **kw):
        fun(x)
        kw['callback'](x)
        return SimpleNamespace(x=x, success=False, status=9, message='Iteration limit reached',
                               nit=1, nfev=1, njev=1)
    monkeypatch.setattr(scipy.optimize, 'minimize', limited)
    result = run()
    assert result['status'] == 'iteration_budget_exhausted'
    record = result['records'][0]
    assert len(result['records']) == 1
    assert record['major_iterations_completed'] == 1
    assert record['solver_status'] == 9
    assert record['solver_iterations'] == 1


def test_accepted_point_survives_attempt_limit(monkeypatch):
    import scipy.optimize
    def advance(fun, x, **kw):
        selected = np.asarray([.025, .025])
        fun(selected)
        return SimpleNamespace(x=selected, success=False, status=9,
                               message='Iteration limit reached', nit=1, nfev=1, njev=1)
    monkeypatch.setattr(scipy.optimize, 'minimize', advance)
    result = run(end_fraction=.1, settings=HybridContinuationSettings(max_attempts=1))
    assert result['status'] == 'attempt_budget_exhausted'
    assert result['has_accepted_progress'] is True
    assert result['fraction'] == .05
    assert result['position'] == [.025, .025]
    assert result['summary']['accepted_step_count'] == 1
    assert result['summary']['request_satisfied'] is False
    assert result['records'][0]['accepted_constraint_slacks']


def test_accepted_budget_cutoff_does_not_grow_next_target(monkeypatch):
    import scipy.optimize
    calls = []
    def partial(fun, x, **kw):
        calls.append(x.copy())
        y = np.asarray([.025*len(calls), .025*len(calls)])
        fun(y)
        return SimpleNamespace(x=y, success=False, status=9, message='Iteration limit reached')
    monkeypatch.setattr(scipy.optimize, 'minimize', partial)
    result = run(end_fraction=.1, settings=HybridContinuationSettings(max_attempts=2))
    assert [r['step'] for r in result['records']] == [.05, .05]
    assert result['fraction'] == .1
    assert result['status'] == 'continuation_targets_met'
    assert all(r['resource_cutoff'] == 'iteration_budget_exhausted' for r in result['records'])


def test_real_slsqp_records_iterations_and_calls():
    result = run(end_fraction=.05)
    record = result['records'][0]
    assert record['variable_count'] == 2
    assert record['forward_difference_stencil_size'] == 3
    assert record['major_iterations_completed'] > 0
    assert record['objective_calls'] > 0
    assert record['constraint_calls'] > 0
    assert len(record['best_trial']['position']) == 2
    assert 'measurement' in record['best_trial']


def test_checkpoint_is_written_before_failing_display_callback():
    saved = []
    def broken(_record):
        raise RuntimeError('UI/reporting callback broke')
    with pytest.raises(RuntimeError, match='callback broke'):
        run(end_fraction=.05, checkpoint=lambda state: saved.append(state), progress=broken)
    assert saved[-1]['has_accepted_progress']
    assert saved[-1]['records'][-1]['accepted']
    assert saved[-1]['accepted_steps'][-1]['position'] == saved[-1]['position']


def test_archived_mirrored_sd_roundoff_is_not_a_route_change():
    # Exact saved values from program-2259bce13b0751fcea00, not recomputed optics.
    parent = toy([0., 0.])
    parent['slots']['SD_L']['min_dx_m'] = .012757260146263341
    parent['slots']['SD_R']['min_dx_m'] = .012757260146264335
    goals = {name: row['min_dx_m'] for name, row in parent['slots'].items()}
    goals['SD_R'] = goals['SD_L']
    _check_target_direction(parent, contract(slot_dispersion_goals_m=goals), HybridContinuationSettings())


def test_real_wrong_direction_names_the_slot_and_values():
    goals = dict.fromkeys(('SF_L', 'SD_L', 'SF_R', 'SD_R'), .03)
    goals['SD_R'] = .019
    with pytest.raises(ValueError, match=r'SD_R: requested=.*parent=.*tolerance'):
        _check_target_direction(toy([0., 0.]), contract(slot_dispersion_goals_m=goals),
                                HybridContinuationSettings())


def test_direction_tolerance_does_not_relax_achromat_or_hardware():
    from expert_modules.mba_design.periodic_hybrid import hybrid_constraint_slacks
    m = toy([0., 0.]); m['id_dx_m'] = [.00011, .00011]
    s = hybrid_constraint_slacks(m, toy([0., 0.]), contract(), 0.)
    assert s['ID_L_Dx_plus'] < 0


def test_identity_does_not_impose_future_direction(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, 'minimize', lambda *a, **k: pytest.fail('identity must not optimize'))
    result = run(end_fraction=0., contract=contract(slot_dispersion_goals_m=dict.fromkeys(
        ('SF_L', 'SD_L', 'SF_R', 'SD_R'), .001), core_dispersion_goal_m=.1))
    assert result['status'] == 'identity_passed'


def test_summary_does_not_equate_endpoint_with_design_success():
    result = run(end_fraction=.05)
    summary = hybrid_continuation_summary(result)
    assert summary['has_accepted_progress'] is True
    assert summary['request_satisfied'] is False
    assert summary['requires_independent_acceptance'] is True
    assert summary['requested_fraction'] == .05
