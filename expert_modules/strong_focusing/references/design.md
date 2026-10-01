# Design and recovery reasoning

## Define what stronger focusing should achieve

Specify the benefit: reduced beam envelope, shorter cells, a required phase
advance, or a match to an insertion. Keep field, bore, length, circumference,
momentum acceptance and correction freedom as explicit constraints. The task is
to find a useful feasible lattice, not maximize quadrupole strength.

Separate three stages: periodic-cell construction, complete-lattice matching,
and particle acceptance. Allocate computation to each stage. A promising cell
can fail when assembled with bends, straight sections and matching insertions.

## Establish an interpretable linear seed

Use a FODO or other specified alternating-gradient topology as the starting
family. Keep finite magnet lengths and required drift space in the candidate.
Thin-lens estimates can initialize a search, but final observables must come
from the actual ordered JuTrack lattice.

For a real uncoupled symplectic 2-by-2 cell map M, strict elliptic stability
requires abs(trace(M)/2) < 1. Write M12 = beta*sin(mu) and
M11 - M22 = 2*alpha*sin(mu). The sign of M12 selects the wrapped phase branch
with positive beta. Integer and half-integer boundaries need separate treatment;
the usual extraction is singular there [1]. Inspect both planes.

The helper reports this diagnostic but cannot identify coupled normal modes or
the integer part of the lattice tune. Check an actual 4D map for coupling before
using its plane blocks. An accelerated, damped or noncanonical map may not have
unit plane determinant; do not force it into this conservative test.

Scan a small set of bounded quadrupole-family strengths to locate feasible
regions before local matching. Record unstable points as failures, not invented
large beta values. Track phase continuously along a parameter scan to avoid
switching branches when an optimizer crosses a wrap.

## Embed and match the complete lattice

Match the required Twiss functions and dispersion at defined interfaces. Keep
geometry and total bending closure fixed for a ring. Check the full-ring tune
and periodic solution after adding matching sections; multiplying a wrapped
cell tune by cell count does not establish the complete tune.

Use independent knobs for independent conditions where possible. Diagnose the
scaled residual Jacobian when matching stalls: redundant knobs, inactive bounds,
an insufficient number of degrees of freedom and inconsistent constraints need
different remedies. The commissioning module's identifiability diagnostic is
optional; this module remains usable without it.

## Trade envelope against chromatic and engineering costs

Stronger focusing changes beta peaks and chromatic response. Compare candidate
cells using the same length, momentum range, magnet inventory and acceptance
definition [2]. Measure beta and dispersion maxima along elements, refining
long elements if endpoint samples miss internal peaks. Convert k1 into gradient
with the declared rigidity and species/sign convention; use engineering bore
and field limits supplied by the user rather than generic limits.

Evaluate a symmetric momentum grid and refine near the first failure. Preserve
the definition of delta and unwrap tune branches before computing derivatives.
If sextupole correction is available, include the actual corrected lattice in
the nonlinear comparison. A satisfactory linear chromaticity does not establish
finite-amplitude acceptance.

Use the beam covariance and aperture model consistently. For uncorrelated
betatron motion and momentum, sigma_x^2 = beta_x*epsilon_x + D_x^2*sigma_delta^2;
include correlations explicitly when this assumption fails. Declare whether
an aperture requirement uses an rms envelope, a finite distribution quantile,
or tracked survival. These are different requirements.

## Diagnose before redesigning

| Symptom | Check | Useful change |
| --- | --- | --- |
| One plane becomes unstable | Plane maps and phase branches | Rebalance focusing and defocusing families |
| Stable cell, failed ring match | Interface residuals and available knobs | Add matching freedom or relax an incompatible target |
| Good on-momentum optics, narrow bandwidth | Off-momentum optics and tune footprint | Revisit phase advance and chromatic correction |
| Smaller envelope but infeasible magnet | Gradient, bore and integrated strength together | Change cell length or distribute focusing |
| Good ideal lattice, large error sensitivity | Independent alignment/gradient errors with stated correction | Move working point or reserve correction strength |

Track a reproducible amplitude/momentum grid through the full lattice, retaining
lost-particle information and a stated turn count. Repeat the final comparison
with finer integration and independent error seeds. Space charge, radiation and
collective effects require separate configured models if material to the design.
Do not infer their absence from successful single-particle tracking.

## Sources

1. E. D. Courant and H. S. Snyder, *Theory of the alternating-gradient
   synchrotron*, Annals of Physics 3, 1–48 (1958),
   https://doi.org/10.1016/0003-4916(58)90012-5.
2. B. J. Holzer, *Lattice Design in High Energy Particle Accelerators*,
   https://arxiv.org/abs/1303.6514.

The staged search, diagnostics and reporting contract are recommendations for
this module. They are not a newly demonstrated accelerator design.
