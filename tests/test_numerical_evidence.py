"""Real runtime I/O and finish integration; no Julia or browser execution."""
import json
import sys
from copy import deepcopy
from types import ModuleType

import numpy as np
import pytest

from brilliance.numerical_evidence import NumericalEvidenceArchive, collect_numerical_evidence
from brilliance.research_api import ResearchRun


def state():
    return dict(status='attempt_budget_exhausted', fraction=.02, position=list(range(18)),
                accepted_steps=[dict(fraction=.02, position=list(range(18)))],
                contract={'max_emittance_pm': 51.0},
                records=[dict(accepted=True, solver_termination='evaluation_budget_exhausted')])


def summary():
    return dict(status='attempt_budget_exhausted', fraction=.02, has_accepted_progress=True,
                accepted_step_count=1, attempt_count=1, optimization_started=True,
                request_satisfied=False)


def candidate():
    return dict(lattice_id='exact-cell', energy_gev=3.5, periodicity=1,
                elements=[dict(name='Q', kind='KQUAD', length_m=.2, k1_m2=3.0)],
                metadata={'evaluation_scope': 'periodic_cell', 'intended_repetitions': 32})


def create(output, *, run=None):
    return NumericalEvidenceArchive(output, 'periodic_hybrid', {'settings': {'max_attempts': 1}},
                                    parent_lattice={'lattice_id': 'parent'}, run=run)


def test_raw_arrays_and_deep_records_are_never_display_truncated(tmp_path):
    archive = create(tmp_path)
    raw = state()
    raw['many_records'] = [{'vector': np.arange(18)} for _ in range(70)]
    raw['deep'] = {'a': {'b': {'c': {'d': {'e': {'f': {'g': list(range(100))}}}}}}}
    archive.snapshot(raw, candidate(), summary=summary())
    evidence = collect_numerical_evidence(tmp_path)
    stored = json.loads((tmp_path / evidence['runs'][0]['files']['checkpoint']['path']).read_text())
    assert stored['position'] == list(range(18))
    assert len(stored['many_records']) == 70
    assert stored['many_records'][-1]['vector'] == list(range(18))
    assert len(stored['deep']['a']['b']['c']['d']['e']['f']['g']) == 100
    assert evidence['rows'][0]['has_accepted_progress'] is True
    assert evidence['optimization_started'] is True


def test_exact_cell_metadata_survives_even_when_wrapper_submits_parent(tmp_path):
    archive = create(tmp_path)
    original = candidate()
    archive.snapshot(state(), original, summary=summary())
    original['metadata'].clear()  # Generated wrapper mutates its own copy.
    files = collect_numerical_evidence(tmp_path)['runs'][0]['files']
    saved = json.loads((tmp_path / files['candidate_lattice']['path']).read_text())
    assert saved['metadata']['intended_repetitions'] == 32
    assert saved['lattice_id'] == 'exact-cell'


def test_separate_invocations_do_not_overwrite_candidates(tmp_path):
    a, b = create(tmp_path), create(tmp_path)
    a.snapshot(state(), candidate(), summary=summary())
    other = candidate(); other['elements'][0]['k1_m2'] = 2.0
    b.snapshot(state(), other, summary=summary())
    evidence = collect_numerical_evidence(tmp_path)
    assert len(evidence['runs']) == 2
    paths = [r['files']['candidate_lattice']['path'] for r in evidence['runs']]
    assert len(set(paths)) == 2
    assert all((tmp_path / p).is_file() for p in paths)


def test_existing_snapshot_remains_immutable(tmp_path):
    archive = create(tmp_path)
    archive.snapshot(state(), candidate(), summary=summary())
    path = archive.directory / archive.manifest['files']['checkpoint']['path']
    old = path.read_bytes()
    updated = state(); updated['fraction'] = .03
    archive.snapshot(updated, candidate(), summary={**summary(), 'fraction': .03})
    assert path.read_bytes() == old
    assert archive.manifest['files']['checkpoint']['path'] != str(path.relative_to(archive.directory))


def test_execution_failure_keeps_accepted_progress(tmp_path):
    archive = create(tmp_path)
    archive.snapshot(state(), candidate(), summary=summary())
    archive.failed(RuntimeError('reporting failed'))
    evidence = collect_numerical_evidence(tmp_path)
    assert evidence['runs'][0]['status'] == 'execution_error'
    assert evidence['runs'][0]['has_accepted_progress'] is True
    assert 'candidate_lattice' in evidence['runs'][0]['files']
    assert 'error' in evidence['runs'][0]['files']


def test_partial_snapshot_write_does_not_publish_a_mismatched_candidate(tmp_path, monkeypatch):
    archive = create(tmp_path)
    archive.snapshot(state(), candidate(), summary=summary())
    original = deepcopy(archive.manifest['files'])
    real = archive._file
    def fail_candidate(role, name, value):
        if role == 'candidate_lattice':
            raise OSError('disk full')
        real(role, name, value)
    monkeypatch.setattr(archive, '_file', fail_candidate)
    with pytest.raises(OSError, match='disk full'):
        archive.snapshot({**state(), 'fraction': .04}, candidate(), summary=summary())
    archive.failed(OSError('disk full'))
    saved = collect_numerical_evidence(tmp_path)['runs'][0]['files']
    assert saved['checkpoint']['sha256'] == original['checkpoint']['sha256']
    assert saved['candidate_lattice']['sha256'] == original['candidate_lattice']['sha256']


def test_corrupt_evidence_does_not_silently_pass(tmp_path):
    archive = create(tmp_path)
    archive.snapshot(state(), candidate(), summary=summary())
    path = archive.directory / archive.manifest['files']['checkpoint']['path']
    path.write_text('{}')
    with pytest.raises(ValueError, match='integrity failure'):
        collect_numerical_evidence(tmp_path)


def test_manifest_cannot_escape_its_output(tmp_path):
    archive = create(tmp_path)
    archive.manifest['files']['parent_lattice']['path'] = '../outside.json'
    archive._commit()
    with pytest.raises(ValueError, match='inside its archive'):
        collect_numerical_evidence(tmp_path)


def test_nonfinite_values_fail_instead_of_silently_changing_types(tmp_path):
    archive = create(tmp_path)
    with pytest.raises(ValueError):
        archive.snapshot({**state(), 'position': [float('nan')]}, candidate(), summary=summary())


def test_direct_unmanaged_calls_do_not_write_to_guessed_directories(monkeypatch):
    monkeypatch.delenv('BRILLIANCE_RUN_OUTPUT', raising=False)
    assert NumericalEvidenceArchive.for_environment('method', {}, parent_lattice={}) is None


def run_fixture(tmp_path, monkeypatch):
    # Registry validation is a separate unit. Exercise the actual finish/UI-data
    # code without loading a scientific expert or calling a simulator here.
    acceptance = ModuleType('brilliance.scientific_acceptance')
    acceptance.environment_acceptance_requirements = lambda: ()
    acceptance.load_expert_registry = lambda **kw: None
    acceptance.assess_scientific_acceptance = lambda *a, **kw: dict(
        passed=True, checks={}, required=[], status='passed')
    monkeypatch.setitem(sys.modules, 'brilliance.scientific_acceptance', acceptance)
    monkeypatch.delenv('BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE', raising=False)
    output = tmp_path / 'workspace' / 'artifacts' / 'programs' / 'test'
    output.mkdir(parents=True)
    return ResearchRun(workspace=tmp_path / 'workspace', output_directory=output,
                       arguments_file=output / 'arguments.json', result_file=output / 'result.json',
                       progress_file=None, cancellation_file=None, working_lattice_file=None,
                       latest_candidate_lattice_file=None, compute_job_id=None)


def test_normal_finish_attaches_runtime_evidence_even_if_wrapper_omits_it(tmp_path, monkeypatch):
    run = run_fixture(tmp_path, monkeypatch)
    archive = create(run.output_directory)
    archive.snapshot(state(), candidate(), summary=summary())
    result = json.loads(run.finish(summary='Program discarded its numerical result.', data={
        'position': {'head': [0, 1, 2, 3], 'n': 18}}).read_text())
    assert result['tables'][0]['rows'][0]['has_accepted_progress'] is True
    evidence = result['data']['numerical_evidence'][0]
    assert evidence['files']['candidate_lattice']['path'] in {d['path'] for d in result['downloads']}
    assert result['use_as_working_lattice'] is False
    assert result['candidate_lattice'] is None  # No silent replacement/selection.
    assert result['data']['position']['head'] == [0, 1, 2, 3]
    stored = json.loads((run.output_directory / evidence['files']['checkpoint']['path']).read_text())
    assert stored['position'] == list(range(18))


def test_local_optimizer_counts_as_execution_evidence(tmp_path, monkeypatch):
    run = run_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv('BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE', '1')
    archive = create(run.output_directory)
    archive.snapshot(state(), candidate(), summary=summary())
    result = json.loads(run.finish(summary='Bounded SLSQP attempt',
                                  scientific_passed=True, request_satisfied=True).read_text())
    assert result['request_satisfied'] is True
    assert result['status'] == 'completed'


def test_identity_only_does_not_fake_optimizer_execution(tmp_path, monkeypatch):
    run = run_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv('BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE', '1')
    archive = create(run.output_directory)
    archive.snapshot({**state(), 'status': 'identity_passed'}, candidate(), summary={
        **summary(), 'has_accepted_progress': False, 'optimization_started': False})
    result = json.loads(run.finish(summary='Identity only', scientific_passed=True,
                                  request_satisfied=True).read_text())
    assert result['status'] == 'partial'
    assert result['scientific_passed'] is True  # Missing work is not failed physics.
    assert result['request_satisfied'] is False
    assert result['data']['optimization_completion']['status'] == 'pending'
    assert result['data']['numerical_evidence']  # Existing identity evidence survives.


@pytest.mark.parametrize('status', ['partial', 'completed', 'infeasible'])
def test_optimizer_requirement_does_not_reject_unfinished_probes(tmp_path, monkeypatch, status):
    run = run_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv('BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE', '1')
    result = json.loads(run.finish(summary='Checked the objective callback before a search',
                                  status=status, scientific_passed=True,
                                  request_satisfied=False, use_as_working_lattice=False,
                                  data={'callback_evaluation_count': 1},
                                  tables=[{'title': 'Callback probe', 'rows': [{'value': 2.0}]}]).read_text())
    assert result['status'] == status
    assert result['request_satisfied'] is False
    assert result['use_as_working_lattice'] is False
    assert result['data']['callback_evaluation_count'] == 1
    assert result['tables'][0]['rows'] == [{'value': 2.0}]
    assert 'optimization_completion' not in result['data']


def test_summary_cannot_replace_optimization_execution_evidence(tmp_path, monkeypatch):
    run = run_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv('BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE', '1')
    result = json.loads(run.finish(summary='Optimization completed', scientific_passed=True,
                                  request_satisfied=True,
                                  data={'optimization_started': True, 'optimization_blocker': 'none'}).read_text())
    assert result['status'] == 'partial'
    assert result['request_satisfied'] is False
    assert 'not an optimization' in result['limitations'][0]


def test_default_live_progress_is_emitted_after_durable_snapshot(tmp_path):
    seen = []
    class Run:
        def progress(self, *args, **kw):
            seen.append(collect_numerical_evidence(tmp_path))
    archive = create(tmp_path, run=Run())
    archive.snapshot(state(), candidate(), summary=summary())
    assert seen[0]['runs'][0]['has_accepted_progress'] is True


def test_ring_definition_is_separately_labelled_and_not_validated(tmp_path):
    archive = create(tmp_path)
    archive.snapshot(state(), candidate(), summary=summary())
    ring = {**candidate(), 'lattice_id': 'explicit-ring', 'periodicity': 32}
    archive.complete({'request_satisfied': False}, ring_lattice=ring)
    evidence = collect_numerical_evidence(tmp_path)
    assert 'candidate_ring' in evidence['runs'][0]['files']
    assert archive.manifest['request_satisfied'] is False
    assert archive.manifest['use_as_working_lattice'] is False
