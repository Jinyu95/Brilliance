"""Offline JuTrack experiment on an explicitly supplied working cell; never publish.

The unsolved geometry below is an experiment, not an expert seed asset. All budgets
come from the input; neither source files nor accepted research history are edited.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from brilliance import accelerator_api as api
from brilliance.atomic_io import atomic_write_text
from brilliance.research_environment import ResearchEnvironment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--core-start',required=True)
    p.add_argument('--core-end',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trials-per-stage',type=int,default=4)
    p.add_argument('--starts',type=int,default=2)
    p.add_argument('--max-nfev',type=int,default=180)
    p.add_argument('--alternating-start',action='store_true')
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    env = ResearchEnvironment.load()
    env.require_active_python(operation='offline hybrid continuation validation')
    if env.internet_access:
        raise ValueError('Disable internet access for this validation')
    a.output.mkdir(parents=True,exist_ok=a.resume)
    def save(name,value):
        atomic_write_text(a.output/name,json.dumps(value,indent=2,allow_nan=False))
    digest = hashlib.sha256(a.source.read_bytes()).hexdigest()
    start = time.monotonic()
    source = api.load_lattice(a.source)
    mba = api.expert_module('laboratory.mba_design')
    rep = mba.HybridMBAOuterReplacement(api.one_period_elements(source),
        core_start_name=a.core_start,core_end_name=a.core_end)
    sampled = api.build_jutrack_profile_sampler(source,max_step_m=.02).linear_metrics()
    if not sampled['stable']:
        raise ValueError('Input periodic cell must be stable for this experiment')
    point = min(sampled['twiss_profile'],key=lambda row:abs(row['s_m']-rep.cell_length/2))
    tin = dict(betax=point['beta_x'],betay=point['beta_y'],dx=point['Dx'],alphax=0.,alphay=0.,dpx=0.)
    # An unsolved connectivity hypothesis, with no fitted K1 values.
    geometry = dict(suppressor_length_m=1.7,bump_length_m=2.6,quadrupole_length_m=.25,
        bend_lengths_m=[.6,.45],inner_bend_fraction=.6,half_straight_m=rep.minimum_half_straight,
        k1_values=[0.]*8,quadrupoles_per_region=[2,4,2])
    if a.alternating_start:
        kscale = .8*api.magnet_strength_limits(source.energy_gev)['max_abs_k1_m2']
        geometry['k1_values'] = [kscale*(-1)**i for i in range(8)]
    bounds = dict(suppressor_length_m=[1.,2.5],bump_length_m=[1.8,3.4],inner_bend_fraction=[.3,.7],
        suppressor_quad_length_m=[.18,.4],bump_quad_length_m=[.18,.4],terminal_quad_length_m=[.18,.4],
        inner_bend_length_m=[.35,.8],terminal_bend_length_m=[.3,.7])
    thresholds = dict(min_bump_m=.04,min_contrast=3.,max_core_dispersion_m=.025,max_sample_step_m=.02,
        min_usable_drift_m=.2,min_total_usable_drift_m=.5,max_drift_slope=.002,max_beta_m=100.)
    settings = dict(energy_gev=source.energy_gev,geometry=geometry,input_twiss=tin,
        parent_lattice_id=source.lattice_id,
        center_optics_bounds=dict(betax=[.01,100.],betay=[.01,100.],dx=[-.1,.1]),
        thresholds=thresholds,limits=api.magnet_strength_limits(source.energy_gev),
        continuation_targets_m=[.03,.045,.06],continuation_contrasts=[1.1,2.,3.],geometry_bounds=bounds,
        trials_per_stage=a.trials_per_stage,starts_per_trial=a.starts,max_nfev=a.max_nfev,random_seed=911)
    save('contract.json',dict(source=str(a.source),source_sha256=digest,replacement=rep.describe(),settings=settings))
    def progress(record):
        print(json.dumps({k:v for k,v in record.items() if k not in
            {'quadrupole_ownership','geometry_coordinates','matching_residuals'}},allow_nan=False),flush=True)
    resume = json.loads((a.output/'checkpoint.json').read_text(encoding='utf8')) if a.resume else None
    result = mba.explore_hybrid_mba(rep,**settings,progress=progress,
        checkpoint=lambda state:save('checkpoint.json',state),resume=resume)
    summary = dict(status=result['status'],duration_s=time.monotonic()-start,trials=len(result['records']),
        jutrack_evaluations=sum(r.get('evaluations',0) for r in result['records']),
        source_unchanged=hashlib.sha256(a.source.read_bytes()).hexdigest()==digest)
    if result['best']:
        best = result['best']
        save('candidate_lattice.json',best['candidate_lattice'])
        save('shape_report.json',best['report'])
        report = best['report']
        summary.update(metrics=report.get('metrics',{}),checks=report['checks'])
        if report.get('stable'):
            api.save_optics_lattice_plot(a.output/'candidate_one_cell.png',report['profile'],
                best['candidate_lattice']['elements'],title='Hybrid continuation: one cell (diagnostic candidate)')
    save('summary.json',summary)
    print(json.dumps(summary,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
