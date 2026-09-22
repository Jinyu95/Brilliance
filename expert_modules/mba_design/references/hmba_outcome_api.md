# MBA design decisions: radiation, chromatic optics and acceptance

Compare hypotheses with JuTrack, not a prescribed route or facility template.

## Very-low-emittance architecture hypotheses

For uncoupled natural horizontal emittance,
`epsilon_x = Cq * gamma^2 * I5 / (Jx * I2)`,
`I5 = integral(Hx * abs(h)^3 ds)`, `Jx = 1 - I4/I2`.
Low Dx alone is not low H: Dpx and Twiss functions matter. Diagnose excitation
by physical bend/region; isolated-unit TME is not complete-ring emittance.

Explore many weak physical bends, balancing their smaller-angle emittance scale
against space/focusing costs. Numerical slicing alone creates no benefit.
Combined-function dipoles can alter focusing and damping partition; check I4,
positive damping and aperture-dependent combined fields. Longitudinal-gradient
bends redistribute curvature relative to H; distinguish physical modules from
numerical segments. Weak reverse bends may be dedicated magnets or bending
quadrupoles; include compensating positive bending and correct orbit/aperture
modeling. These mechanisms can cooperate, but none guarantees improvement.
[First-principles field/optics comparisons](https://arxiv.org/abs/1810.11286).

Think hierarchically: two hybrid MBAs may form a superperiod; two superperiods
may form a four-MBA repeating block. Explore one or two chromatic bumps across
the selected block, not automatically two per MBA. Jointly redesign interfaces:
concatenation alone neither removes bumps nor creates correction freedom.
Introduce independent families/phases; approximate reflection symmetry belongs
to the chosen block. This is optional, not a fixed layout or doubling sequence.

Preserve each MBA's achromatic boundaries (Dx/Dpx zero within tolerances) to
retain intermediate user straights. These can remain insertion-device sections
inside larger superperiods: verify magnet-free length, matched betas and device
requirements after rematching. Achromaticity alone is insufficient. Count MBAs,
user straights and repeating blocks separately; preserve circumference/counts.

Match radiation, chromatic and insertion optics together using grouped controls.
Judge larger blocks by measured benefits, not complexity.

## Decide from measurements

| Observation | Useful experiment |
| --- | --- |
| Low emittance, excessive sextupole demand | Co-vary site dispersion/betas and low-H bending optics; compare emittance, chromaticity and required integrated strengths. |
| Corrected chromaticity, poor acceptance | Examine pair maps, tune footprints and detuning before adding nonlinear families. |
| Saturated or ineffective controls | Measure response and singular values; test regrouping, relocation or geometry. |
| Improved bend profile | Rematch the complete ring; compare I2/I4/I5, damping and acceptance under consistent constraints. |

Retain alternatives. Failed searches do not prove impossibility.
[Radiation/chromatic trade-offs](https://arxiv.org/abs/2105.14492).

## Sites, cancellation and nonlinear controls

For a two-outer-bump 7BA hypothesis, B1–B2/B6–B7 are outer regions and B3–B5
the core, counting physical main bends, not slices or inherited unit names.
Both SF/SD sites can occupy distinct finite intervals in each outer region;
adjacent drifts may form an installation span. Measure site Dx/Dpx/betas and
core H/I5 separately. Other structures need their own optical-role regions.

Nominal −I is not a tracking result: evaluate complete two-plane pair maps,
endpoint optics, finite sextupole lengths and off-momentum departures.
Cancellation may cross a straight or cell boundary, releasing arc phase
constraints. Do not prescribe within-cell pairing.
[Cross-straight cancellation](https://arxiv.org/abs/2212.01870).

Use actual ring periodicity and relevant resonance terms to explore phase
neighborhoods, including symmetry breaking; do not copy facility tunes.
[Distributed cancellation](https://proceedings.jacow.org/ipac2019/papers/moprb005.pdf).
A measured chromatic-response QR/SVD nullspace exposes controls preserving
chromaticities locally; re-correct after finite moves. Investigate harmonic
families, positions or octupoles when measured detuning/resonances justify them.

## Joint search and trustworthy measurements

Compare emittance, connected transverse acceptance and positive/negative momentum acceptance
among chromatically corrected candidates. Screen cheaply, then refine turns,
sampling and integration for finalists. For segmented fields, distinguish
physical edges from internal interfaces; demonstrate convergence without
spurious edge focusing. Report unsupported JuTrack field models explicitly.

`mba.mba_radiation_report` provides scoped TME normalization and bend grouping.
`mba.hybrid_mba_bend_roles` and `mba.evaluate_hybrid_mba_shape` are optional
two-bump diagnostics for symmetric positive homogeneous sector-bend MBAs (M≥5),
not universal validators. Use identical metric definitions in optimization and
reporting; label global peaks separately from useful sites. Unsupported tool
scope is not physical impossibility. `mba.solve_chromatic_correction` measures response.

Report candidate identity, launch optics/coordinates, turns, RF, radiation,
apertures and errors. On-axis energy survival is not ring-local momentum
acceptance; disconnected islands are not injection aperture. Compare papers
only with these scopes disclosed. Show lattice/optics and measured acceptance.
