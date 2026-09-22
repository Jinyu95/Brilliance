# MBA design decisions: radiation, chromatic optics and acceptance

Test hypotheses with JuTrack; do not copy templates.

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

## Design optical regions, not concatenated MBAs

A possible block follows: half-straight, low-H bending/focusing section, bump,
low-H section, intermediate straight, low-H section, second bump, low-H section,
half-straight. These are optical roles, not a magnet sequence or universal
two-bump requirement. Declare the actual repeating block, boundary maps and
ring counts; a plot does not establish half-cell/full-period identity.

Jointly design sections, not pairs of finished MBAs. Do not preserve inherited
unit tunes or arbitrary internal periodic/achromat boundaries. Match Dx/Dpx to
zero at user straights with useful betas and magnet-free length. Seek low
dispersion/H elsewhere, not exactly zero throughout bends; measure radiation
cost in bump transitions.

Reserve finite sextupole sites with complementary beta weights; pair across
the block, including straights. Co-vary geometry, bending and focusing without
sacrificing low-H regions. Symmetry/grouped controls are optional aids. Report
bump-site optics, non-bump H/I5 and each user straight separately, not just
emittance or peak dispersion. Preserve machine requirements and straight counts.

## Decide from measurements

| Observation | Useful experiment |
| --- | --- |
| Low emittance, excessive sextupole demand | Co-vary site dispersion/betas and low-H bending optics; compare emittance, chromaticity and required integrated strengths. |
| Corrected chromaticity, poor acceptance | Examine pair maps, tune footprints and detuning before adding nonlinear families. |
| Saturated or ineffective controls | Measure response and singular values; test regrouping, relocation or geometry. |
| Improved bend profile | Rematch the complete ring; compare I2/I4/I5, damping and acceptance under consistent constraints. |

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
