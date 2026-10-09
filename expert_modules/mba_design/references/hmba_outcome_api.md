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

For chromatic matching and acceptance, use the same native derivative method and
momentum step; demonstrate step convergence. Do not apply a central-difference
tolerance to legacy `getchrom`'s fixed-step forward difference.

## Evidence limits

Nominal −I is not acceptance: inspect pair maps, endpoint optics, finite sextupole lengths
and off-momentum behavior. Use actual periodicity; pairs may cross straights.
A QR/SVD nullspace preserves chromaticity locally; re-correct after finite moves.

Distinguish physical edges from internal interfaces and check integration convergence.
On-axis energy survival is not local positive/negative momentum acceptance.
Declare launch optics, locations, turns, RF, radiation, apertures and errors.
Use [injection-based targets](injection_acceptance.md); scan edges are lower bounds.
Report unmet requirements; failed searches are not impossibility proofs.

Background: [radiation/chromatic trade-offs](https://arxiv.org/abs/2105.14492),
[cross-straight cancellation](https://arxiv.org/abs/2212.01870),
[distributed cancellation](https://proceedings.jacow.org/ipac2019/papers/moprb005.pdf).
