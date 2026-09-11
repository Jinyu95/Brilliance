# Local hybrid continuation and ordered geometry release

Use `mba = api.expert_module("laboratory.mba_design")`. This is an optional expert numerical
tool, not a fixed application workflow. It never creates a conventional seed, imports another
project, accesses the internet, changes a working pointer or publishes a final H-MBA claim.

The agent selects a physics hypothesis, the current source cell, explicit core boundaries,
connectivity, bounds, requirements and budget. The helper handles repetitive numerical work.
Do not rewrite the loop or inspect private package internals for each continuation step.

## Contract and ownership

`replacement = mba.HybridMBAOuterReplacement(rows, core_start_name=..., core_end_name=...)`
retains the source cell length, bend angle and required ID half-straight. Choose the physical
middle M-4-bend core; the same method applies to odd/even M >= 5. No solved strengths are supplied
by the module. Core bend geometry remains the initial structural hypothesis; **every globally
mirror-paired quadrupole**, including core quadrupoles, belongs exclusively to the inner matcher.
There are no within-unit strength ties and no source tune/beta/Dx targets.

`geometry` uses the exact keyword schema of build_outer_dba_section, INCLUDING k1_values:

- suppressor_length_m, bump_length_m (include quads, exclude bends);
- quadrupole_length_m, optionally quadrupole_lengths_m (one common length per region for this helper);
- bend_lengths_m = [inner, terminal], inner_bend_fraction;
- half_straight_m, quadrupoles_per_region, minimum_drift_length_m (optional);
- k1_values = initial outer strengths, one per outer quad, not necessarily matched.

`geometry_bounds` is a mapping from the following coordinates to [lower, upper]. Bounds must
contain the initial coordinate; omission means fixed. Release order is cumulative:

| After a bounded stage stalls | Newly available coordinates |
| --- | --- |
| All global K1 + free centre optics | No structural variables yet |
| Drift allocation | suppressor_length_m, bump_length_m |
| Outer angle split | inner_bend_fraction |
| Grouped magnet lengths | suppressor_quad_length_m, bump_quad_length_m, terminal_quad_length_m, inner_bend_length_m, terminal_bend_length_m |

The terminal drift reservoir funds changes. Unfunded geometries and excessive dipole fields
are rejected before simulation. This is at most eight outer coordinates, not dozens of K1
coordinates competing with an inner matcher. The geometry scout is NOT PSO: it alternates
bounded neighbourhood and full-bound proposals. For a larger PSO campaign, reuse the same
ownership/budget contracts and the generic optimizer's population/feasibility rules.

`input_twiss` contains betax/betay/alphax/alphay/dx/dpx: an initial symmetry-plane guess with
alphax=alphay=dpx=0. `center_optics_bounds` must have betax/betay/dx pairs. It must be wide enough
to explore realistic optics; source values can initialize them but are not constraints.
If measuring the source, `sampler.linear_metrics()` returns `twiss_profile`; the final shape
report returns `profile`. Public point fields are beta_x/beta_y/Dx/Dpx.

`thresholds` is the **bump_feasibility** contract from outer_dba_api.md. Do NOT pass max_core_h_m
at this initial stage; that is a later full_hybrid requirement. No H/I5/emittance ceiling or old
optical preservation is imposed. Independent beta/core-dispersion inequalities guide the line
fit; dense JuTrack periodic checks still decide actual drift usability, shape, stability,
achromat, magnet limits and positive damping. A stationary minimum cannot pass as a bump.

## Execute, show progress, and resume

```python
import json
from brilliance.atomic_io import atomic_write_text

checkpoint_path = run.artifact_path('hybrid/continuation.json')

def save_checkpoint(state):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(checkpoint_path, json.dumps(state, allow_nan=False))

def on_trial(t):
    measurements = dict(t.get('metrics', {}))
    measurements.update(trial=t['trial'], target_bump_m=t['target_bump_m'])
    residuals = [{'condition': k, 'scaled residual': v}
                 for k, v in t.get('matching_residuals', {}).items()]
    run.progress('Hybrid exploration: ' + t['stage'],
        detail=t.get('message') or (t['status'] + '; failed checks: ' +
               ', '.join(t.get('failed_checks', [])) + '; ' + t.get('reason', '')),
        metrics=measurements,
        tables=[{'title': 'Matching residuals (pass within ±1)', 'rows': residuals}] if residuals else [])

previous_checkpoint_or_None = (json.loads(checkpoint_path.read_text(encoding='utf8'))
                               if checkpoint_path.is_file() else None)
state = mba.explore_hybrid_mba(
    replacement, energy_gev=parent.energy_gev, parent_lattice_id=parent.lattice_id,
    geometry=declared_geometry, geometry_bounds=declared_geometry_bounds,
    input_twiss=centre_initial_guess, center_optics_bounds=centre_search_bounds,
    thresholds=declared_bump_requirements, limits=api.magnet_strength_limits(parent.energy_gev),
    continuation_targets_m=declared_increasing_bump_heights,
    continuation_contrasts=declared_nondecreasing_contrasts,  # optional; defaults to final contrast throughout
    trials_per_stage=trial_budget, starts_per_trial=multistarts, max_nfev=matching_budget,
    random_seed=campaign_seed,
    initialize_achromat=True,  # cold-start ID prefit; warm starts retain achieved shape
    progress=on_trial, checkpoint=save_checkpoint,  # optional cancelled=your_stop_callback
    resume=previous_checkpoint_or_None,
)
```

Callbacks receive ordinary JSON-compatible dictionaries. Write checkpoints atomically (e.g.
`brilliance.atomic_io.atomic_write_text`) in this project's workspace. Do not place the entire
checkpoint in model context. Resume with the same source/contract and saved state. The helper
rejects stale/different-source checkpoints. `max_trials=N` optionally returns a paused state
between trials; it does not kill a running numerical job. A changed contract needs a new campaign.

The last height/contrast must reach the FINAL contract. Intermediate lower contrasts are
explicit continuation hypotheses, not relaxed final success criteria. Do not derive a tighter
core-Dx bound by dividing a small initial bump by the final contrast: that can make the early
step harder than the final design. The independent declared core-dispersion envelope stays fixed.
If the current continuation path stalls, compare other starts/paths/connectivities; none is universal.

`progress(record)` emits a start event and a completion event for each trial. Completed records
contain status, stage, target_bump_m, active_geometry_variables, geometry_coordinates and,
when evaluated, matching_residuals (named, scaled), evaluations, saturated_strengths and
quadrupole_ownership. Periodic candidates also include metrics and failed_checks.
`knob_regions` lists the actual position-based region and global mirror partner for each
independent quadrupole. The technical half-core list includes its boundary quads; do not
confuse that propagation prefix with the narrower physical B3..B[M-2] low-H region.

Show a concise sentence explaining the changed physical variables and measured outcome, and
a small table at each material checkpoint. Don't flood chat with repeated schema/parameter
dumps. Matching callbacks are numerical, not new model turns. Coding exceptions propagate
separately; they are never counted as negative physics trials. Keep the last saved checkpoint.

Returned state includes records, warm_position, warm_geometry and best. `best`, if present,
contains candidate_lattice (a periodic-cell dictionary), report, position and geometry. The
source project is never overwritten. Before plotting, check report['stable']; use report['profile']
with exactly best['candidate_lattice']['elements'] and the standard combined one-cell plot.
For further API calls convert the dictionary with `api.load_lattice(...)` using its archived
path, or `LatticeIR.from_dict` as documented by api.help; do not pass dictionaries to APIs
expecting LatticeIR. Show per-bend radiation on an explicit closed ring in a separate diagnostic.

Status is one of:

- `bump_feasible`: independent periodic stage passed. This is an INTERMEDIATE milestone,
  not a finished hybrid design; retain request_satisfied=False until full shape/quality gates pass.
- `paused`: checkpoint preserved for cancellation or a requested trial slice.
- `search_budget_exhausted`: no final feasible bump shape found in the declared trial budget.
  This is NOT proof that H-MBA is impossible or that a topology is saturated. Inspect failed
  conditions and active limits; choose new bounds, connectivity, starts or deeper exploration.

Emittance recovery with verified bump constraints is a separate next stage, followed by
chromatic correction and nonlinear acceptance. No route change is forced merely by a timer.
