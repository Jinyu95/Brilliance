"""Re-evaluate an explicit same-project hybrid trial with core K1 released.

No source edit, publication, borrowed facility data or automatic change of cell budget.
This diagnostic distinguishes a working matcher API from successful hybrid physics.
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
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--core-start',required=True)
    p.add_argument('--core-end',required=True)
    p.add_argument('--target-bump',type=float)
    p.add_argument('--max-core-dispersion',type=float)
    a = p.parse_args()
    env = ResearchEnvironment.load()
    env.require_active_python(operation='hybrid rematch regression')
    if env.internet_access:
        raise ValueError('This test requires internet disabled')
    a.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        atomic_write_text(a.output/name,json.dumps(value,indent=2,allow_nan=False))
    started = time.monotonic()
    digest = hashlib.sha256(a.source.read_bytes()).hexdigest()
    parent = api.load_lattice(a.source)
    mba = api.expert_module('laboratory.mba_design')
    rows = api.one_period_elements(parent)
    rep = mba.HybridMBAOuterReplacement(rows,core_start_name=a.core_start,core_end_name=a.core_end)
    profile = api.build_jutrack_profile_sampler(parent,max_step_m=.02).linear_metrics()['twiss_profile']
    centre = min(profile,key=lambda p:abs(p['s_m']-rep.cell_length/2))
    tin = dict(betax=centre['beta_x'],betay=centre['beta_y'],dx=centre['Dx'],alphax=0.,alphay=0.,dpx=0.)
    outer,half = rows[rep.end+1:],rep.right_core_half()
    strengths = [r['k1_m2'] for r in outer+half if r['kind']=='KQUAD']
    station = next(p for p in profile if p['name']=='DBA_R:BUMP')
    limits = api.magnet_strength_limits(parent.energy_gev)
    fit = mba.match_outer_dba_section(outer,energy_gev=parent.energy_gev,input_twiss=tin,
        core_prefix=half,center_optics_bounds=dict(betax=[.01,200.],betay=[.01,200.],dx=[-.1,.1]),
        rematch_core_quadrupoles=True,bump_station_name='DBA_R:BUMP',
        target_bump_m=a.target_bump or station['Dx']*1.02,starts=[strengths],max_abs_k1_m2=limits['max_abs_k1_m2'],max_nfev=600,
        max_core_dispersion_m=a.max_core_dispersion,max_beta_m=100. if a.max_core_dispersion is not None else None)
    save('match.json',fit)
    cell = rep.assemble(fit['elements'],core_elements=rep.core_from_half(fit['core_prefix']))
    candidate = api.periodic_cell(title='Same-project core-rematch regression trial',energy_gev=parent.energy_gev,
        elements=cell,source=__file__,intended_repetitions=parent.metadata['intended_repetitions'],parent_lattice_id=parent.lattice_id)
    save('candidate_lattice.json',candidate.as_dict())
    thresholds = dict(min_bump_m=.04,min_contrast=3.,max_core_dispersion_m=.025,max_sample_step_m=.02,
        min_usable_drift_m=.2,min_total_usable_drift_m=.5,max_drift_slope=.002,max_beta_m=100.)
    gate = rep.evaluate(candidate,thresholds=thresholds,limits=limits)
    save('shape_report.json',gate)
    if gate.get('stable'):
        api.save_optics_lattice_plot(a.output/'candidate_one_cell.png',gate['profile'],cell,
            title='Rematched hybrid trial (diagnostic, not accepted design)')
    changed = [r['name'] for r,old in zip(rep.core_from_half(fit['core_prefix']),rep.core)
               if r['k1_m2'] != old['k1_m2']]
    summary = dict(line_matched=fit['matched'],core_quadrupoles_changed=changed,
        core_geometry_preserved=rep.core_preserved(cell,allow_quadrupole_rematch=True),
        checks=gate['checks'],metrics=gate.get('metrics',{}),
        cell_length_m=rep.cell_length,duration_s=time.monotonic()-started,
        source_unchanged=hashlib.sha256(a.source.read_bytes()).hexdigest()==digest)
    save('summary.json',summary)
    print(json.dumps(summary,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
