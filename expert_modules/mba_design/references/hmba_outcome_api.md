# HMBA observation contract

[Worked method](hmba_worked_method.md), [conventional seed](conventional_api.md),
[superperiod exploration](very_low_emittance.md): alternative routes.

## Shared regions and quantities

Localize dispersion in **non-bending correction regions** with internal quadrupoles
and finite SF/SD sites; inspect their actual lengths and beta functions.
Low-H bending does **not** require zero dispersion in every dipole: outer bends
generate/suppress bumps. Keep user straights achromatic within declared tolerances.

Use one physical-region definition for both search and reporting. In two-bump
7BA, B1–B2/B6–B7 contain correction regions; the B3–B5 core includes its intervening
elements. Count physical bends, not numerical slices. Use actual roles in other
layouts. Put SF/SD in distinct intervals per bump.

Report site Dx/Dpx/betas, peak positions/types, whole-core max |Dx| and H,
per-bend I2/I5 contributions, contrast, chromatic response and integrated strengths;
name the global maximum separately. Contrast is not a universal acceptance threshold.
Keep core degradation and site leverage visible separately from emittance.
Show one-cell optics with the magnet layout, identifying rejected trials.

## Measurement tools

Call `hmba_optics_observables(profile, rows, correction_regions=..., core_region=...)`
in both objectives and reports with contiguous real element names. Its correction
regions contain no bends; the core includes intervening elements. Read `api_help`.
Boundary samples do not prove interior extrema. For other regions use
`api.region_optics_summary`; audit knobs with `api.audit_parameterization`.

`mba.mba_radiation_report` normalizes TME. Its default grouping and
`evaluate_hybrid_mba_shape` cover unsplit positive homogeneous bends only.
Segmented/reverse/combined bends need explicit regions and native per-bend data.
`mba.solve_chromatic_correction` measures response; verify both targets.

Use consistent native derivatives and momentum steps for chromatic matching and
verification. Check convergence; central-difference tolerances do not apply to
legacy `getchrom`'s fixed-step forward difference.

## Evidence limits

Nominal −I is not acceptance: inspect pair maps, endpoint optics, finite sextupole lengths
and off-momentum behavior. Use actual periodicity; pairs may cross straights.
A QR/SVD nullspace preserves chromaticity locally; re-correct after finite moves.

Distinguish physical edges from internal interfaces; numerical accuracy is not a design knob.
Declare launch optics, locations, turns, RF, radiation, apertures and errors.
Use [injection-based targets](injection_acceptance.md); scan edges are lower bounds.
Report unmet requirements; failed searches are not impossibility proofs.

## Keep DA and momentum acceptance separate

For local MA, launch from the **on-momentum closed orbit**, add a momentum offset
and track the full ring. In an ideal centred lattice the initial transverse
coordinates are zero. Do not recenter the launch onto the off-momentum closed orbit.
Scan both signs, connected to delta=0. A single-location scan establishes only
that location's result; repeat around the ring for local MA.
Extend each sign toward its first tracked loss within a declared scan/resource cap.
A surviving cap is a lower bound, not a target or measured boundary.

DA varies transverse coordinates. Finite-amplitude off-momentum acceptance and
all-phase survival are separate robustness measurements, not the default MA
objective or mandatory MA gates. Preserve requested on-momentum DA separately.
Report map instability separately; never encode an untracked point as a tracked
loss because an orbit/map solve failed. Use identical tracking settings in comparisons.

## Acceptance objectives and bounded validation

Define "preserve DA" by agreed connected area, minimum radius and/or an explicit
required injection domain, with stated tolerances. Do not silently require every
historical boundary point to survive. Report internal losses and disconnected
islands; larger area alone does not prove injection usefulness. Without an agreed
DA threshold, retain MA–DA trade-offs rather than inventing a pass/fail rule.
Keep both MA signs and chosen DA measures separate in a Pareto search; do not let
an arbitrary lost-point penalty outweigh all attainable MA improvement.

Compare parent and trials at the same working integration settings. Establish
decision-relevant accuracy, then keep those settings fixed during the search.
Use bounded finalist checks of MA endpoints and DA measures; do not demand identical
pointwise survival masks or automatically double integration steps repeatedly.
Escalate only when measured uncertainty could change selection. Reuse completed
checks with matching candidate/settings; optical convergence alone does not prove
tracking convergence. Change objectives by warm-starting saved lattices, not old
scores or optimizer convergence state.

Definition: [standard MA tracking, Section III.A](https://doi.org/10.1103/PhysRevAccelBeams.27.094002).

Background: [radiation/chromatic trade-offs](https://arxiv.org/abs/2105.14492),
[cross-straight cancellation](https://arxiv.org/abs/2212.01870),
[distributed cancellation](https://proceedings.jacow.org/ipac2019/papers/moprb005.pdf).
