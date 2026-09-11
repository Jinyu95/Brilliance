# Outer DBA replacement: usable drifts with freely rematched periodic optics

The conventional seed-generation pipeline is unchanged. Design NEW outer sections around
the existing middle M-4-bend core as an initial structural hypothesis. Source beta and
dispersion do NOT have to be preserved. Solve new periodic optics. Preserve old optical
values only if the scientist explicitly requests that additional constraint.
Core geometry is retained, but ALL global mirror-paired quadrupoles may rematch.
For local continuation without rewriting trial loops, first read `hybrid_continuation_api.md`.

## 1. Identify the current source; measure it for diagnostics

```python
mba = api.expert_module("laboratory.mba_design")
parent = api.load_lattice(run.working_lattice_file)  # confirm intended full-ring identity
rows = api.one_period_elements(parent)
replacement = mba.HybridMBAOuterReplacement(rows,
    core_start_name=first_core_element, core_end_name=last_core_element)
reference = replacement.measure_reference(parent, max_step_m=.02)
core_half = replacement.right_core_half()  # diagnostic centre-to-core-exit propagation
```

Choose complete optical interfaces from physical inventory, not unit labels alone.
The measurement checks stable JuTrack periodic optics and binds the reference to the source
fingerprint. Archive source_lattice_id, source_cell_fingerprint, profile, entrance, exit,
energy_gev, s_start_m/s_end_m, boundary names, max_step_m and sampling evidence.
Public profile keys are beta_x/beta_y, NOT betax/betay (the latter are JuTrack constructor keys).
Measuring the reference is useful but is NOT required by the default acceptance contract.

## 2. Generate unsolved outer connectivities

```python
counts = (2,4,2)  # suppressor, bump, terminal quads; explicit hypothesis, not a fixed design
outer = mba.build_outer_dba_section(replacement,
    suppressor_length_m=L_suppressor, bump_length_m=L_bump,
    quadrupole_length_m=Lq, bend_lengths_m=[Lb_inner,Lb_terminal],
    inner_bend_fraction=f_inner, half_straight_m=L_ID/2,
    quadrupoles_per_region=counts, k1_values=[0.]*sum(counts),
    minimum_drift_length_m=Ld_min)
```

Core exit → suppressor → inner bend → bump quads/installation drifts → terminal bend →
ID matching → ID centre. Region lengths include quads but exclude bends; terminal length
is derived from the exact budget. DBA_R:BUMP marks the middle installation drift.
The default (2,2,2) is minimal, not mandatory; try extra independent outer quads where needed.
Optional quadrupole_lengths_m provides one explicit length per quad (otherwise the common
quadrupole_length_m is used). For example, release grouped suppressor/bump/terminal lengths
without requiring the whole region to share one magnet length or violating pole-tip limits.
The bump count must be even for this builder. Other explicitly authored topologies are valid.
Do not spend core length or shorten the required ID straight to fit geometry.

## 3. Free symmetry-plane optics and outer one-pass matching

```python
# Declare finite, physical search bounds and an in-bound optical INITIAL GUESS.
# The symbols below are study inputs, not stored lattice constants.
tin = dict(betax=beta_x_guess, betay=beta_y_guess, dx=dispersion_guess,
           alphax=0., alphay=0., dpx=0.)
centre_bounds = dict(betax=[beta_x_min,beta_x_max],
                     betay=[beta_y_min,beta_y_max], dx=[Dx_min,Dx_max])
matched = mba.match_outer_dba_section(outer, energy_gev=parent.energy_gev,
    core_prefix=core_half, center_optics_bounds=centre_bounds, rematch_core_quadrupoles=True,
    input_twiss=tin, bump_station_name='DBA_R:BUMP', target_bump_m=D_goal,
    starts=declared_strength_vectors, max_abs_k1_m2=limits['max_abs_k1_m2'],
    max_nfev=300, id_dispersion_tolerance_m=1e-4, slope_tolerance=1e-4,
    alpha_tolerance=1e-4, bump_tolerance_m=1e-3)
```

Each start has one in-bound K1 per OUTER quad followed by one per HALF-CORE quad, optionally
followed by centre betax, betay, dx for independent optical starts. Otherwise tin supplies
the initial centre guess for each start. The three optical coordinates are optimized, NOT
fixed to tin. Alpha_x/alpha_y/Dpx vanish at the reflection plane by symmetry. The diagnostic
core half is propagated but never exported as extra magnets. The assembled periodic cell
must independently confirm the matched optics and stability; a line solve is not acceptance.
ALL outer/core strengths and centre optics belong to the inner matcher, disjoint from outer
geometry/angle proposals. No tune/emittance
target. Six residuals: ID alpha_x/alpha_y/Dx/Dpx and the drift station's Dx/Dpx.
Use a DRIFT or marker between drifts; bump_station_fraction=.5 supports a drift centre.
In a drift Dpx is constant and Dx linear: high Dx with zero Dpx is a plateau, not a sharp
isolated maximum. A regional peak in a quadrupole is acceptable only if drift space qualifies.
Provide core_prefix and center_optics_bounds together. Their omission selects a fixed-input
line solve: use this ONLY for a deliberately fixed boundary problem (e.g. explicit source-optics
preservation, where input_twiss is mapped from reference['exit']). It is not the default
conversion requirement. Scout new outer lengths, angles, quad lengths/counts and optical starts.

Return keys: matched, elements (outer only), core_prefix (matched diagnostic half-core),
quadrupole_ownership, position, variable_names, scaled_residuals, residual_names,
input_twiss, matched_input_twiss, center_optics_bounds, boundary_mode, bump_station, evaluations, starts, stage,
limitations. The winner is replayed before export. matched means a line solve, not an H-MBA.
diagnostics includes Jacobian singular values/rank and saturated strengths; starts includes
each trial's scaled residuals. Default initialization solves all six conditions together;
initialize_achromat=True optionally prefits the four ID conditions. boundary_mode records
free_symmetry_plane_optics or fixed_source_core_exit. A failed scout does not prove topology exhaustion.
Optional `max_core_dispersion_m` and `max_beta_m` add independent resulting-envelope
inequalities. They guide the line solve, not equality to source optics; dense periodic samples
remain the gate. With these options the residual list grows: use residual_names, not fixed indices.
The legacy rematch_core_quadrupoles=False default only reproduces archived fixed-core programs.

## 4. Independently gate the periodic cell

```python
core = replacement.core_from_half(matched['core_prefix'])  # restores unsplit global mirror pairs
candidate = api.cell_ring(title='Outer replacement trial', energy_gev=parent.energy_gev,
    cell=replacement.assemble(matched['elements'], core_elements=core), periods=periods,
    source='this research program', parent_lattice_id=parent.lattice_id)
thresholds = dict(min_bump_m=D_required, min_contrast=contrast_required,
    max_core_dispersion_m=core_dispersion_limit,
    min_usable_drift_m=minimum_contiguous_installation_length,
    min_total_usable_drift_m=minimum_total_installation_length_per_side,
    max_drift_slope=allowed_drift_Dpx, max_beta_m=beta_limit,
    max_sample_step_m=.02, id_dispersion_tolerance_m=1e-4, id_slope_tolerance=1e-4)
report = replacement.evaluate(candidate, thresholds=thresholds, limits=limits,
    stage='bump_feasibility')
```

Requirements are disclosed hypotheses, not copied constants or source-profile equality.
Beta and core-dispersion limits bound the NEW cell. No old beta/dispersion/H/I5/emittance
ceiling is implicit. Requested source-optics preservation is an additional, optional contract.
Use continuation/scouting before applying final targets to all initial particles. Only achieved
conditions earn a pass; no target reduction just to change a status flag.

Checks: resolved outer maxima/plateaus; field-free drift transport; contiguous/total usable
length; high Dx throughout usable space relative to the WHOLE core; low core dispersion;
global beta envelope; achromat; symmetry/stability; geometry/magnet bounds; positive damping.
Optional preservation: call measure_reference(parent) then evaluate(..., preserve_core_optics=True).
Only this mode adds periodic core-optics agreement at all reference samples. Its tolerances
default to 1% beta, .01 alpha, 1e-4 m Dx and 1e-4 Dpx (core_optics_tolerances overrides).
Without preserve_core_optics=True, an available reference comparison is DIAGNOSTIC and
may fail without rejecting a valid new shape. Do not interpret it as a required match.
`core_preserved(rows)` compares physical fields, ignoring known provenance labels; pass
`allow_quadrupole_rematch=True` to test retained geometry with changed K1. Unknown fields
and bend focusing are NOT ignored. evaluate uses this rematch-compatible structural contract
and still audits complete-cell symmetry. Never use raw row-dictionary equality on provenance.

Report includes passed, design_status, checks, metrics, usable_drifts, peaks, profile,
preserve_core_optics, radiation, replacement_contract, and optional core_optics_reference/
core_optics (diagnostic per-sample/max scaled errors). The milestone is USABLE_BUMPS_FEASIBLE.
Use report['design_status'], never hard-code success. For radiation use
report['radiation']['emittance_pm'] or ev.require_radiation(), not getattr.
Publish failed checks, usable-drift table, before/after core optics, combined one-cell plot and
per-bend I2/I4/I5. Do not swallow plotting/schema exceptions or claim an intrinsic hybrid penalty.

First-stage results are intermediate, request_satisfied=False. Final conversion evidence uses
replacement.evaluate(..., stage='full_hybrid') with additionally
max_core_h_m, then run.finish(acceptance={'hybrid_mba_shape': report}, candidate_lattice=candidate).
Later emittance recovery must retain the accepted shape/engineering contract. Chromatic correction
and nonlinear acceptance remain separate; every selected descendant needs its own checks.
