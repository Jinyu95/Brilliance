"""Actual expert-adapter/runtime integration with explicit API doubles, NOT JuTrack."""
from copy import deepcopy
import hashlib
import json
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from brilliance.numerical_evidence import collect_numerical_evidence
from expert_modules.mba_design.periodic_hybrid import match_periodic_hybrid
from test_periodic_hybrid import contract, toy
from dataclasses import asdict


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import brilliance
    from expert_modules.mba_design import periodic_hybrid as module
    output = tmp_path / 'workspace' / 'artifacts' / 'programs' / 'runtime-test'
    output.mkdir(parents=True)
    for key, value in {'BRILLIANCE_WORKSPACE': tmp_path / 'workspace', 'BRILLIANCE_RUN_OUTPUT': output,
                       'BRILLIANCE_RUN_ARGUMENTS': output / 'arguments.json',
                       'BRILLIANCE_RUN_RESULT': output / 'result.json',
                       'BRILLIANCE_RUN_PROGRESS': tmp_path / 'workspace' / 'progress.json'}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv('BRILLIANCE_RUN_CANCEL', raising=False)
    rows = [dict(name=n, kind=k, length_m=l, angle_rad=(2*math.pi/32/7 if k == 'SBEND' else None),
                 k1_m2=0. if k == 'KQUAD' else None) for n,k,l in [
        ('B1','SBEND',1),('SF_L','DRIFT',.3),('Q_L','KQUAD',.2),('SD_L','DRIFT',.3),
        ('B2','SBEND',1),('B3','SBEND',1),('B4','SBEND',1),('B5','SBEND',1),
        ('B6','SBEND',1),('SD_R','DRIFT',.3),('Q_R','KQUAD',.2),('SF_R','DRIFT',.3),('B7','SBEND',1)]]
    class IR:
        def __init__(self, elements, *, periods=1, metadata=None):
            self.energy_gev = 3.5
            self.elements = deepcopy(elements)
            self.lattice_id = hashlib.sha256(json.dumps(elements,sort_keys=True).encode()).hexdigest()[:16]
            self.periods = periods
            self.metadata = metadata or {}
        def as_dict(self):
            return dict(lattice_id=self.lattice_id, energy_gev=self.energy_gev, periodicity=self.periods,
                        elements=deepcopy(self.elements), metadata=deepcopy(self.metadata))
    class Knobs:
        initial_position = [0.]
        bounds = [[0.,1.]]
        def apply(self, x):
            result = deepcopy(rows)
            for r in result:
                if r['kind'] == 'KQUAD':
                    r['k1_m2'] = float(x[0])
            return result
        def as_dict(self):
            return {'variables': ['global_Q_pair']}
    class Sampler:
        def __init__(self, cell, *, max_step_m):
            self.jt = object()
            self.physical_editor = SimpleNamespace(lattice=cell.elements)
        def linear_metrics(self, proposed):
            self.physical_editor.lattice = proposed
            x = next(r['k1_m2'] for r in proposed if r['kind'] == 'KQUAD')
            return dict(stable=True,sampling={'map_equivalent':True}, x=x,trace_x=.1,trace_y=.4)
    api = ModuleType('brilliance.accelerator_api')
    api.one_period_elements = lambda parent: deepcopy(parent.elements)
    api.audit_lattice_constraints = lambda rows, **kw: {'passed': True}
    api.infer_mirror_symmetric_parameterization = lambda rows, **kw: Knobs()
    api.periodic_cell = lambda *, title,energy_gev,elements,source,intended_repetitions,parent_lattice_id: IR(
        elements, metadata={'evaluation_scope':'periodic_cell','intended_repetitions':intended_repetitions})
    api.build_jutrack_profile_sampler = Sampler
    api.jutrack_periodic_radiation_metrics = lambda *a,**kw: {'physical_validity':{'passed':True}}
    def ring(*, title, energy_gev, cell, periods, source, parent_lattice_id):
        return IR([deepcopy(r) for _ in range(periods) for r in cell], periods=periods)
    api.cell_ring = ring
    monkeypatch.setitem(sys.modules, 'brilliance.accelerator_api', api)
    monkeypatch.setattr(brilliance, 'accelerator_api', api, raising=False)
    fingerprint = ModuleType('brilliance.lattice_fingerprint')
    fingerprint.cell_fingerprint = lambda rows: 'explicit-test-double'
    monkeypatch.setitem(sys.modules, 'brilliance.lattice_fingerprint', fingerprint)
    regions = ModuleType('expert_modules.mba_design.hybrid_regions')
    regions.hybrid_mba_regions = lambda rows: {'mirror_passed':True, 'bend_roles': {'bend_count':7,
        'regions':{'left_bump':{'element_indices':[1,2,3]},'right_bump':{'element_indices':[9,10,11]},
                   'core':{'s_start_m':3.8,'s_end_m':6.8}}}}
    monkeypatch.setitem(sys.modules, 'expert_modules.mba_design.hybrid_regions', regions)
    monkeypatch.setattr(module, '_profile_measurement', lambda lin,*a: toy([lin['x'],lin['x']]))
    parent = IR(rows)
    config = dict(periods=32, slots={n:[n] for n in ('SF_L','SD_L','SF_R','SD_R')},
        minimum_slot_length_m=.25, limits={'max_abs_k1_m2':1.},
        minimum_magnet_length_m=.1, minimum_drift_length_m=.02, max_sample_step_m=.02,
        contract=asdict(contract()), end_fraction=.05)
    return output,parent,config,api


def test_unmodified_app_call_automatically_preserves_cell_ring_and_full_state(runtime):
    output,parent,config,api = runtime
    before = parent.as_dict()
    result = match_periodic_hybrid(parent, **config)  # NO user checkpoint/progress callbacks.
    assert result['has_accepted_progress']
    assert parent.as_dict() == before
    ev = collect_numerical_evidence(output)
    files = ev['runs'][0]['files']
    cell = json.loads((output / files['candidate_lattice']['path']).read_text())
    ring = json.loads((output / files['candidate_ring']['path']).read_text())
    assert cell == result['candidate_lattice']
    assert cell['metadata']['intended_repetitions'] == 32
    assert len(ring['elements']) == 32 * len(cell['elements'])
    assert sum(r['angle_rad'] or 0 for r in ring['elements']) == pytest.approx(2*math.pi)
    raw = json.loads((output / files['result']['path']).read_text())
    assert raw['position'] == result['position']
    assert raw['summary']['request_satisfied'] is False
    assert result['full_ring_validation_required'] is True
    assert (output.parents[2] / 'progress.json').is_file()


def test_report_callback_failure_does_not_lose_the_expert_candidate(runtime):
    output,parent,config,api = runtime
    def broken(_):
        raise RuntimeError('generated wrapper failed')
    with pytest.raises(RuntimeError, match='wrapper failed'):
        match_periodic_hybrid(parent, **config, progress=broken)
    ev = collect_numerical_evidence(output)
    assert ev['runs'][0]['has_accepted_progress']
    assert ev['runs'][0]['status'] == 'execution_error'
    checkpoint = json.loads((output / ev['runs'][0]['files']['checkpoint']['path']).read_text())
    assert checkpoint['accepted_step_count'] == 1
    assert checkpoint['records'][-1]['accepted']


def test_callback_cannot_mutate_raw_checkpoint(runtime):
    output,parent,config,api = runtime
    def truncate(state):
        state['position'] = {'head':[999], 'n':18}
        state['accepted_steps'].clear()
    result = match_periodic_hybrid(parent, **config, checkpoint=truncate)
    ev = collect_numerical_evidence(output)
    raw = json.loads((output / ev['runs'][0]['files']['checkpoint']['path']).read_text())
    assert raw['position'] == result['position']
    assert raw['accepted_step_count'] == 1


def test_ring_construction_error_keeps_accepted_cell(runtime):
    output,parent,config,api = runtime
    api.cell_ring = lambda **kw: (_ for _ in ()).throw(TypeError('ring adapter error'))
    with pytest.raises(TypeError, match='ring adapter error'):
        match_periodic_hybrid(parent, **config)
    ev = collect_numerical_evidence(output)
    assert ev['runs'][0]['has_accepted_progress']
    assert ev['runs'][0]['status'] == 'execution_error'
    assert 'candidate_lattice' in ev['runs'][0]['files']


def test_identity_metadata_and_no_optimization_are_retained(runtime):
    output,parent,config,api = runtime
    config['end_fraction'] = 0
    result = match_periodic_hybrid(parent, **config)
    assert result['status'] == 'identity_passed'
    ev = collect_numerical_evidence(output)
    assert not ev['optimization_started']
    assert not ev['runs'][0]['has_accepted_progress']
    assert 'result' in ev['runs'][0]['files']
