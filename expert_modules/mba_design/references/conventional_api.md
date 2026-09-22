# Conventional MBA: one practical design recipe

This is an optional starting method: other seeds and direct complete-cell searches
are legitimate. Use JuTrack and general matching/optimization APIs.

## Allocate the complete cell

For energy E, P identical cells and circumference C, use Lcell=C/P and
Θcell=2π/P. Reserve the requested straight; distinguish approximate from exact C.

For M complete bend units, one useful allocation is
`(M−4)*U + 4*Louter + Lstraight = Lcell`. Equal unit lengths and equal angles
are initial hypotheses, not physical laws. Half-angle end bends instead give
`(M−2)*θunit + 2*(θunit/2) = Θcell`. Recompute all budgets when changing angles.

`api.magnet_strength_limits` converts declared fields/apertures to strengths;
disclose assumptions. For relativistic electrons, Brho≈E/0.299792458
T·m, θmax=Bmax*L/Brho and Kn,max=n!*Bpole/(Brho*r**n). Longer magnets increase
integrated strength at a fixed local field limit. Dipole lengths around 0.8–1.5 m
and quadrupoles around 0.25–0.5 m are examples, not compulsory bounds.

## Obtain and measure a unit

A simple symmetric trial is
`D3–Qfar–D2–Qnear–D1–B–D1–Qnear–D2–Qfar–D3`.
Its intact physical bend retains its actual edge/fringe settings. The total
length is `Lb + 2*(Lnear+Lfar+D1+D2+D3)`. Near/far label position, not polarity;
both opposite-sign quadrupole assignments deserve consideration. Positive K1
focuses horizontally under the normal JuTrack convention.

With fixed geometry there are two independent strengths. A coarse scan followed
by bounded local refinement is straightforward; geometry can also vary under
the length/angle budget. SciPy and PSO/MOPSO are alternatives, not prescribed
stages. See [design decisions](hmba_outcome_api.md) for chromatic co-design.
Evaluate periodic stability and physical radiation with JuTrack. Do not
impose zero boundary dispersion, preserve arbitrary seed tunes, or use symmetry
identities αx=Dpx=0 as substitutes for minimizing actual emittance.

`api.periodic_cell(..., intended_repetitions=N)` and
`api.evaluate_periodic_cell(cell, periods=N)` provide a fast repeated-unit
reference when N times its bend angle closes to 2π. Its emittance is not the
requested MBA's emittance. Analytical bend-centre TME conditions are diagnostics;
compare centre optics only when sampling preserves the intact bend map.

## Assemble and rematch the complete MBA

Repeat exact measured rows with unique names and the allocated straight.
One achromat hypothesis rematches quadrupoles in the first two units and globally
mirrored last two units; for a 7BA the central B3–B5 units initially stay unchanged.
Global symmetry does not require each original outer unit to remain internally
symmetric. This is a useful initial knob choice, not a permanent core constraint.
Use saved full-precision rows/strengths, not reconstructed rounded values.

Use `api.compile_lattice_parameterization` and `api.bounded_response_match`, or
an appropriate SciPy solver, with a JuTrack callback measuring periodic ID Dx/Dpx.
Scale residuals by the actual tolerances. `api.build_jutrack_editor` avoids
reconstructing the native lattice for scalar updates. Solver termination,
matched residuals and complete-machine feasibility are separate checks.

Release geometry/strengths when measurements justify it. Inner matching
owns different knobs from the outer optimizer. Retain the exact post-match
elements, not the outer proposal before matching. Search size follows dimension,
feasible fraction and observed progress, not a universal particle-count rule.

## Check and interpret the selected result

Use `api.evaluation_snapshot(result)` for geometry, optics and radiation.
Check circumference, closure, stability, achromat, damping and engineering limits.
Evaluate the exact cell and explicit `api.cell_ring`; save before plotting.
Entrance beta is not the envelope: measure the profile and disclose sampling.

`mba.mba_radiation_report(candidate, evaluation,
bend_contributions=api.bend_radiation_contributions(candidate))` supplies scoped
TME normalization and physical-order I2/I4/I5. Diagnose core versus outer
contributions. A linear prototype is not a nonlinear-qualified light source.
