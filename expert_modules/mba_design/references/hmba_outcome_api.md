# Conventional HMBA: established two-outer-bump design

This guide covers the established hybrid-MBA architecture: two outer
chromatic-correction regions surrounding low-H bending, between achromatic
user straights. The architecture is mature; these notes are draft knowledge,
not a certificate of numerical or engineering validation.

For ordinary HMBA requests, start here. For an existing candidate, preserve its
architecture unless the scientist requests a change or evidence justifies one.
A very-low-emittance aspiration alone does not invalidate this route. Distinct exploratory
superperiods are covered in [Very-low-emittance design](very_low_emittance.md);
they do not replace this method.

Use a measured conventional MBA as a seed when available, or construct the
complete hybrid cell directly. Jointly match outer bump/suppressor regions,
low-H core and user straights. Release geometry and global mirror-paired
strengths as needed; do not preserve each seed unit's internal symmetry or tune.
Provide finite SF/SD sites with useful dispersion and complementary betas.
Use the actual radiation and chromatic response to balance emittance, correction
strength and damping before tracking. Do not equate a dispersion peak with
successful hybridization. [Conventional initialization](conventional_api.md)
remains available without imposing it as a mandatory first stage.

Two dominant bumps belong in the sextupole correction regions; strongly suppress
dispersion elsewhere and retain low-H bending. Check Dx/Dpx/betas at sextupoles,
maximum absolute Dx and H/I5 across the core, and Dx/Dpx near zero in user
straights. Preserve localization during refinement; a central minimum alone is
insufficient.

## Decide from measurements

| Observation | Useful experiment |
| --- | --- |
| Low emittance, excessive sextupole demand | Co-vary site dispersion/betas and low-H bending optics; compare emittance, chromaticity and required integrated strengths. |
| Corrected chromaticity, poor acceptance | Examine pair maps, tune footprints and detuning before adding nonlinear families. |
| Saturated or ineffective controls | Measure response and singular values; test regrouping, relocation or geometry. |

Retain alternatives; failed searches cannot prove impossibility.
[Radiation/chromatic trade-offs](https://arxiv.org/abs/2105.14492).

## Sites, cancellation and nonlinear controls

Only for the conventional two-outer-bump 7BA hypothesis, B1–B2/B6–B7 are outer
regions and B3–B5 the core (physical bends, not slices). Both SF/SD sites can
occupy distinct drift intervals within each outer region. Other structures
need their actual optical-role regions, not this numbering rule.

Nominal −I is not a tracking result: evaluate complete two-plane pair maps,
endpoint optics, finite sextupole lengths and off-momentum departures.
Cancellation may cross a straight or cell boundary, releasing arc phase
constraints. Do not prescribe within-cell pairing.
[Cross-straight cancellation](https://arxiv.org/abs/2212.01870).

Use actual ring periodicity, resonances and symmetry breaking, not facility tunes.
[Distributed cancellation](https://proceedings.jacow.org/ipac2019/papers/moprb005.pdf).
A chromatic-response QR/SVD nullspace preserves chromaticities locally;
re-correct after finite moves. Add nonlinear controls for measured limitations.

## Linear optics first, then nonlinear dynamics

Optimize linear optics and geometry first, including radiation, damping, pair
maps and measured chromatic-correction feasibility. Then tune nonlinear families
and track shortlisted corrected candidates; revisit linear optics when needed.
Compare emittance, connected horizontal/vertical acceptance and both signs of
local momentum acceptance. Use [injection-based targets](injection_acceptance.md);
scan beyond them in both planes, and report surviving scan edges as lower bounds.
Screen cheaply, then refine turns, sampling and integration for finalists. For segmented fields, distinguish
physical edges from internal interfaces; demonstrate convergence without
spurious edge focusing. Report unsupported JuTrack field models explicitly.

`mba.mba_radiation_report` provides scoped TME normalization and bend grouping.
`mba.hybrid_mba_bend_roles` and `mba.evaluate_hybrid_mba_shape` are optional
two-bump diagnostics for symmetric positive homogeneous sector-bend MBAs (M≥5),
not universal validators. Use identical metric definitions in optimization and
reporting; label global peaks separately from useful sites. Unsupported tool
scope is not physical impossibility. For segmented/reverse/combined-function
bends, use native profiles and physical regions instead of this restricted shape
helper. `mba.solve_chromatic_correction` measures response.

Report candidate identity, launch optics/coordinates, turns, RF, radiation,
apertures and errors. On-axis energy survival is not ring-local momentum
acceptance; disconnected islands are not injection aperture. Compare papers
only with these scopes disclosed. Show lattice/optics and measured acceptance.
Report final magnet-family strengths and unmet requirements.
