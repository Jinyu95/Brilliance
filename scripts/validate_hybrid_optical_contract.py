"""Offline hybrid-optics experiment; explicit input, no publication or seed asset.

Default: new periodic optics. --preserve-core-optics tests the optional stricter
source-equality hypothesis, never an implicit requirement for conversion.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from brilliance import accelerator_api as api
from brilliance.research_environment import ResearchEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--core-start', required=True)
    parser.add_argument('--core-end', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometries', type=int, default=8)
    parser.add_argument('--starts', type=int, default=4)
    parser.add_argument('--max-nfev', type=int, default=160)
    parser.add_argument('--random-seed', type=int, default=911)
    parser.add_argument('--quad-length-min', type=float, default=.16)
    parser.add_argument('--quad-length-max', type=float, default=.25)
    parser.add_argument('--long-suppressors', action='store_true')
    parser.add_argument('--preserve-core-optics', action='store_true')
    parser.add_argument('--min-bump', type=float, default=.04)
    parser.add_argument('--min-contrast', type=float, default=3.)
    parser.add_argument('--max-core-dispersion', type=float, default=.03)
    parser.add_argument('--max-beta', type=float, default=100.)
    args = parser.parse_args()
    env = ResearchEnvironment.load()
    env.require_active_python(operation='hybrid optical-contract validation')
    if env.internet_access:
        raise ValueError('Validation requires internet disabled')
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    started = time.monotonic()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    parent = api.load_lattice(args.source)
    periods = int(parent.metadata.get('intended_repetitions',
                  parent.metadata.get('repetitions',parent.periodicity)))
    mba = api.expert_module('laboratory.mba_design')
    rep = mba.HybridMBAOuterReplacement(api.one_period_elements(parent),
        core_start_name=args.core_start, core_end_name=args.core_end)
    ref = rep.measure_reference(parent)
    save('reference.json', ref)
    tin = {key: ref['exit'][field] for key, field in
           zip(('betax','betay','alphax','alphay','dx','dpx'),
               ('beta_x','beta_y','alpha_x','alpha_y','Dx','Dpx'))}
    core_max = max(abs(p['Dx']) for p in ref['profile'])
    matching_options = {}
    if not args.preserve_core_optics:
        centre = min(ref['profile'], key=lambda p:abs(p['s_m']-rep.cell_length/2))
        tin = dict(betax=centre['beta_x'],betay=centre['beta_y'],dx=centre['Dx'],
                   alphax=0.,alphay=0.,dpx=0.)
        matching_options = dict(core_prefix=rep.right_core_half(),
            center_optics_bounds=dict(betax=[.01,args.max_beta],betay=[.01,args.max_beta],dx=[-.1,.1]))
    thresholds = dict(min_bump_m=args.min_bump, min_contrast=args.min_contrast,
        max_core_dispersion_m=args.max_core_dispersion, max_sample_step_m=.02,
        min_usable_drift_m=.2, min_total_usable_drift_m=.5,
        max_drift_slope=.002, max_beta_m=args.max_beta)
    limits = api.magnet_strength_limits(parent.energy_gev)
    kmax = limits['max_abs_k1_m2']
    save('contract.json', dict(source=str(args.source), sha256=digest,
         replacement=rep.describe(), thresholds=thresholds, input_twiss=tin,
         preserve_core_optics=args.preserve_core_optics, matching_options=matching_options))
    print(json.dumps(dict(core_max_m=core_max, target_bump_m=thresholds['min_bump_m'], input=tin)), flush=True)
    if args.previous:
        old = rep.evaluate(api.load_lattice(args.previous), thresholds=thresholds, limits=limits,
                           preserve_core_optics=args.preserve_core_optics)
        save('previous_gate.json', old)
        print('Previous candidate checks:', json.dumps(old['checks']), flush=True)
    rng = np.random.default_rng(args.random_seed)
    trials = []
    for index in range(args.geometries):
        counts = (2,4,2) if index%2 == 0 else (2,2,2)
        q = rng.uniform(args.quad_length_min,args.quad_length_max)
        qlens = ([q*1.8]*counts[0] + [q]*counts[1] + [q*1.2]*counts[2]
                  if args.long_suppressors else [q]*sum(counts))
        fraction = rng.uniform(.35,.7)
        lens = [max(.35, rep.outer_angle*fraction*limits['minimum_bend_radius_m']*1.1),
                max(.35, rep.outer_angle*(1-fraction)*limits['minimum_bend_radius_m']*1.1)]
        free = rep.outer_length - rep.minimum_half_straight - sum(lens) - sum(qlens)
        allocation = rng.dirichlet([3.,5.,3.])*free
        geometry = dict(suppressor_length_m=sum(qlens[:counts[0]])+allocation[0],
            bump_length_m=sum(qlens[counts[0]:counts[0]+counts[1]])+allocation[1], quadrupole_length_m=q,
            quadrupole_lengths_m=qlens,
            bend_lengths_m=lens, inner_bend_fraction=fraction,
            half_straight_m=rep.minimum_half_straight, quadrupoles_per_region=counts)
        rows = mba.build_outer_dba_section(rep, **geometry, k1_values=[0.]*sum(counts))
        starts = rng.uniform(-kmax, kmax, (args.starts,sum(counts))).tolist()
        result = mba.match_outer_dba_section(rows, energy_gev=parent.energy_gev,
            input_twiss=tin, bump_station_name='DBA_R:BUMP',
            target_bump_m=thresholds['min_bump_m']*1.03, starts=starts,
            max_abs_k1_m2=kmax, max_nfev=args.max_nfev, **matching_options)
        save(f'match-{index}.json', dict(geometry=geometry, result=result))
        trial = dict(index=index, matched=result['matched'],
            max_scaled_residual=max(abs(v) for v in result['scaled_residuals']),
            evaluations=result['evaluations'])
        if result['matched']:
            candidate = api.cell_ring(title='Optical-contract trial', energy_gev=parent.energy_gev,
                cell=rep.assemble(result['elements']), periods=periods,
                source=__file__, parent_lattice_id=parent.lattice_id)
            gate = rep.evaluate(candidate, thresholds=thresholds, limits=limits,
                                preserve_core_optics=args.preserve_core_optics)
            save(f'candidate-{index}.json', candidate.as_dict())
            save(f'gate-{index}.json', gate)
            trial.update(passed=gate['passed'], checks=gate['checks'])
            if gate.get('stable'):
                api.save_optics_lattice_plot(args.output/f'optics-{index}.png', gate['profile'],
                    api.one_period_elements(candidate), title='Optical-contract hybrid trial')
        trials.append(trial)
        save('summary.json', dict(trials=trials, duration_s=time.monotonic()-started,
                                 source_unchanged=hashlib.sha256(args.source.read_bytes()).hexdigest()==digest))
        print(json.dumps(trial), flush=True)
        if trial.get('passed'):
            full = api.evaluate(candidate, include_radiation=True, include_profile=False)
            save('full_ring.json', dict(stable=full.linear_optics.stable, radiation=full.require_radiation()))
            break


if __name__ == '__main__':
    main()
