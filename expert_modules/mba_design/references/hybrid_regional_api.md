# Physical-region H-MBA geometry and protected-core optimization

This is an OPTIONAL refinement route for an existing shaped cell, not the default conversion
method. Start conversion with `outer_dba_api.md`: replace outer sections, solve new periodic
optics, obtain usable high-dispersion drift regions, then recover emittance separately.
Source beta/dispersion and baseline H/I5 ceilings are not default constraints.
Use the hard core limits below only when explicitly adopted for that later stage.

These optional tools operate on an existing complete cell (M >= 5, including even M). They do
not construct a seed, pick a facility lattice or mandate an optimization sequence.

## Global reflection and region membership

```python
from brilliance import accelerator_api as api
mba = api.expert_module("laboratory.mba_design")
rows = api.one_period_elements(candidate)
inventory = mba.hybrid_mba_regions(rows)
# mirror_passed, mirror_issues[{left,right,fields}], elements[{index,name,kind,
# s_start_m,s_end_m,region,side,mirror_index,mirror_name,bend}], bend_roles.
# region_names: bump_generator, dispersion_suppressor, low_H_core,
#               terminal_matching, outer_bend.
knobs = mba.hybrid_mba_quadrupole_knobs(rows, max_abs_k1_m2=limits["max_abs_k1_m2"],
    regions=["bump_generator", "dispersion_suppressor"])
```

Whole-cell mirror pairs need equal fields. Left/right halves of a former TME unit do NOT.
For example U001:QF pairs with U005:R:QF, while U001:R:QF is an independent pair. Region
membership comes from physical bend boundaries, not the example names. Publish the actual
inventory and select matching pairs using their measured signed response/rank. Do not silently
disable an audit or symmetrize a genuinely mismatched saved source.

## Three structural coordinates, explicit length funding

```python
geometry = mba.HybridMBARegionalGeometry(rows,
    bounds={"bump_length_m": bump_bounds,
            "suppressor_length_m": suppressor_bounds,
            "outer_bend_fraction": fraction_bounds},
    reservoir_element_names=declared_terminal_drift_names,
    minimum_drift_length_m=min_drift_length)
description = geometry.describe(minimum_population_per_knob=10)
# variable_names, initial_position, bounds, outer_knob_count=3,
# minimum_population=30, reservoirs, regions, invariants, required_trial_checks.
trial_rows = geometry.apply(position)
```

Both lengths are FULL one-side inter-bend lengths, including fixed magnets inside each region;
only drift allocation changes. The third coordinate is B1/(B1+B2), with the mirror pair tied.
Angle changes carry a length transfer at the pair-mean radius, preserving total outer-pair
angle and length. The original coordinates reproduce the original fields. This leaves the
core bends/lengths untouched and avoids increasing field simply because bend length was frozen.

Specify actual terminal drifts outside the bump/suppressor/core and excluding the first/last
ID drifts as reservoirs. Each named drift automatically includes its whole-cell mirror; naming
both is harmless. Extra bump/suppressor length reduces these reservoir lengths and vice versa.
No source file is changed. Coordinates outside bounds or without sufficient drift budget raise
ValueError before JuTrack; a physically rejected proposal is not evidence of route exhaustion.
Do not catch arbitrary programming errors as infeasibility. Audit dipole fields, minimum magnet
lengths, global symmetry, stability and achromat after the complete structural/matching proposal.

## Independent objectives and hard core protection

```python
measured = evaluator.measure(include_radiation=True)
# Returns I5_core_m1, I2_core_m1 and bend_radiation with scope='one_physical_cell'.
# Per-bend JuTrack elrad I2/I4/I5 sums are checked against the repeated-cell ringpara totals.
# Optional later-stage ceilings, explicitly adopted; do not set them automatically
# equal to the conventional baseline. Omit core_limits entirely for unconstrained shaping.
core_limits = {"D_core_m": permitted_core_dispersion,
               "H_core_max_m": permitted_core_H,
               "I5_core_m1": permitted_core_I5}

def objective(matched_rows):
    return evaluator.objective(matched_rows, core_limits=core_limits,
                               emittance_ceiling_pm=emittance_ceiling)
```

Output `objectives` contains **three minimization coordinates**: `bump_deficit`,
`core_dispersion_m`, `core_I5_m1`. `constraint_violations` reports EACH core ceiling separately;
core_limits=None (default) imposes none; any subset of the documented three ceilings is valid.
`feasible=False` cannot be offset by a better bump. The bump objective uses separate left/right
height deficits plus continuous prominence terms, not an emittance proxy. JuTrack/damping/ID
feasibility and any declared emittance ceiling also apply. A lower core-I5 fraction alone is
not protection: absolute I5_core and H_core can grow while the fraction shrinks.

Feed `geometry.apply(position)` into your disjoint inner quadrupole matcher, then this objective
into generic `api.particle_swarm_optimize`. This is a nonlinear coordinate transform, not a
`LatticeParameterization`: do NOT pass the geometry object to `optimize_lattice_parameters` as
if it were a collection of independent element-field knobs. Population minimum is based on
the THREE outer structural coordinates, not every affected element or the inner matching knobs.
Use process-local evaluators, fully screened feasible initial particles and persistent checkpoints.
Use a local pilot and then a worker pilot before production. Preserve complete front candidates;
never combine the best bump of one candidate with the best core of another in the result table.

`continuous_shape_penalty` remains available for local diagnostics/continuation, but is NOT
the preferred single objective for a protected-core design search. The final shape gate
still requires two resolved interior bumps, low core dispersion/H and achromat; core ceilings
must also pass for the SAME exported candidate. A short search that misses that gate is
unfinished work, not evidence that geometry release failed physically.
