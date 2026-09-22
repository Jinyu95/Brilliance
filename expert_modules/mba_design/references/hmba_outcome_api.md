# MBA design decisions: radiation, chromatic optics and acceptance

These are hypotheses, not a required route. Compare JuTrack candidates under
consistent requirements. Conventional, hybrid or higher-order-achromat layouts
may win; more bends are not automatically better.

## Decide from measurements

| Observation | Useful experiment |
| --- | --- |
| Low emittance but excessive sextupole demand | Vary sextupole-site dispersion and beta separation together with low-H bending optics; compare emittance, natural chromaticities and required integrated strengths. |
| Corrected chromaticity but poor acceptance | Examine pair maps, tune footprints, amplitude detuning and both-sign off-momentum tracking; distinguish cancellation errors from inadequate nonlinear controls. |
| Family strengths saturate or barely affect objectives | Measure response and its singular values; test family regrouping or site relocation before merely increasing search size. |
| A changed bend profile lowers emittance | Re-match the complete cell/ring and compare radiation, damping and acceptance at consistent field, space and aperture assumptions. |

The SAPS study optimizes emittance, natural chromaticity and sextupole strength
together, then tests nonlinear dynamics. None of these linear proxies guarantees
large acceptance. Its damping-time trade-off also shows why emittance alone is
insufficient. Learn these trade-offs, not its magnet settings or quoted aperture.
[Zhao, Jiao & Wang, 2021](https://arxiv.org/abs/2105.14492).

## Sites and cancellation

For the common two-outer-bump hypothesis, number bends by longitudinal order:
B1–B2 and B(M−1)–BM are outer regions, B3–B(M−2) the core. In a 7BA this means
B3–B5, regardless of inherited unit names. Both SF/SD slots on one side can be
distinct finite intervals inside that outer region; adjacent drifts may form one
installation span. Measure actual site Dx, Dpx and both betas, not just a peak in
a bend. Keep core H/I5 separate from bump contrast.

Nominal −I cancellation organizes a search; it is not a tracking result.
Evaluate complete two-plane pair maps, endpoint optics, finite sextupole lengths
and off-momentum departures. Cancellation can also cross a straight between
cells, changing the phase constraints on the arc. Do not hard-code within-cell
pairing as the only hybrid architecture.
[Tan et al., 2023](https://arxiv.org/abs/2212.01870).

## Phase and nonlinear controls

Choose phase neighborhoods using the actual ring periodicity and the resonance
terms of interest; account for symmetry breaking, insertion sections and tune
footprints. Do not copy a facility's fractional tunes. Higher-order-achromat
schemes offer distributed cancellation, with different off-momentum trade-offs.
Measure the two-plane chromatic response: a QR/SVD nullspace can expose remaining
family combinations while holding chromaticities to target. Re-correct after
finite moves. Test harmonic families, positions or octupoles when measured
detuning/resonances justify them—not because more magnets must help.
[SOLEIL HOA study, 2019](https://proceedings.jacow.org/ipac2019/papers/moprb005.pdf).

## Joint search and field shaping

Keep several matched, chromatically corrected candidates. Compare emittance,
connected transverse acceptance and positive/negative momentum acceptance rather
than optimizing an isolated scalar proxy. Use inexpensive screens to allocate
tracking effort, then increase turns, spatial sampling and integration accuracy
for finalists. This is a suggested experimental strategy, not a compulsory solver.

Reverse bends and longitudinal field redistribution are distinct hypotheses.
Compare under consistent length, angle, field and aperture constraints; inspect
I2/I4/I5 and damping after complete-ring rematching. First-principles comparisons
explain why local field changes cannot be judged from peak field alone.
[Riemann & Streun, 2019](https://arxiv.org/abs/1810.11286).
If using explicit segments, distinguish physical edges from internal interfaces
and demonstrate convergence without adding spurious edge focusing. If JuTrack
cannot represent the intended field/edge model reliably, report that limitation.

## Measurements and reporting

`mba.mba_radiation_report` supplies scoped TME normalization and per-bend grouping.
`mba.hybrid_mba_bend_roles` and `mba.evaluate_hybrid_mba_shape` are optional
two-bump diagnostics for symmetric positive homogeneous sector-bend MBAs (M≥5),
not universal hybrid validators. Use identical region/slot definitions in the
objective and final report; distinguish global peaks from useful site values.
Read signatures for thresholds. Unsupported scope is not a
physical impossibility. `mba.solve_chromatic_correction` measures response.

Report candidate identity, launch location/Twiss, coordinates, turns, RF,
radiation, physical apertures and error assumptions. On-axis energy survival is
not local momentum acceptance around the ring; disconnected surviving islands
are not an injection aperture. Compare papers only with these scopes disclosed.
Show one-cell lattice/optics, chromatic footprints and acceptance when measured.
A failed search is not proof that a topology cannot work.
