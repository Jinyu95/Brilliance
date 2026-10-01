# Construction and diagnosis

## Select a claim that conventional magnets can support

The octupole-channel route approximates a transverse Hamiltonian with one
invariant. In two transverse degrees of freedom, that alone is insufficient for
Liouville integrability. Keep this distinct from the two-invariant constructions
of Danilov and Nagaitsev [1]. The practical goal here is a useful tune footprint
and connected acceptance under specified perturbations, not proof of integrability.

## Construct the optical environment

An IOTA-style starting hypothesis is a dispersion-free insert with equal beta
functions in both planes and an appropriately matched surrounding linear lattice
[2]. In an ideal drift centered at a round waist, beta(s) = beta_star +
s^2/beta_star. This provides an initial common-beta profile; verify the actual
JuTrack optics after embedding the insert in a physical ring.

Use Courant-Snyder normalization X = x/sqrt(beta),
P = (alpha*x + beta*p_x)/sqrt(beta) in the on-momentum paraxial limit. Here p_x
is normalized canonical momentum, approximately the trajectory slope in this
limit. Define analogous vertical coordinates. With JuTrack's derivative-strength
convention, an ideal normalized octupole Hamiltonian is

H = (P_X^2 + P_Y^2 + X^2 + Y^2)/2
    + g*(X^4 + Y^4 - 6*X^2*Y^2)/24,

where g = k3(s)*beta(s)^3 is constant and has units m^-1. Thus set the continuous
strength proportional to beta^-3. The helper returns this derivative strength;
do not confuse it with the cubic polynomial coefficient k3/6. Finite magnets and
their integration steps approximate the continuous profile and must be checked.

For the canonical IOTA construction, the return transport from insert exit to
entrance is the identity in the respective normalized frames, with an unwrapped
phase advance of 2*pi times an integer [2]. Let physical coordinates equal A(s)
times normalized coordinates. Check A_in^-1 M_return A_out against the identity.
The raw physical transfer matrix need not itself be the identity. Do not impose
an integer tune on the complete ring: the nonlinear insert contributes phase.

Keep quadrupole, bend, drift, and octupole lengths physically realizable. With
finite gaps, sample/integrate the strength profile over each magnet and test
segmentation convergence. Replacing one magnet with more numerical integration
steps is different from installing additional independent magnets.

## Optimize a useful operating domain

First establish the supported linear match and common-beta/dispersion residuals.
Then compare strength scale, insert beta/length, segmentation, and correction
freedom while preserving the declared ring and hardware constraints. Treat this
as a design hypothesis that can be rejected, not a prescribed search algorithm.

Sample both transverse actions and several launch phases. Axis-only launches can
miss unstable mixed-plane motion. For each launch, retain survival, turn count,
tunes from two observation windows, and candidate-invariant statistics. Use the
same launch coordinates and physical apertures for comparisons. Tune extraction
must handle frequency aliases and mode swapping; report the estimator and window
length instead of labelling all peak changes as chaotic diffusion.

Finite-octupole studies show a trade-off between strength, tune spread, and
acceptance [2]. Increase strength only when the accepted domain benefits. A
baseline can use the same magnet inventory with a different distribution of
strengths. Declare whether peak field, total integrated strength, or another
engineering resource is matched; these constraints are not interchangeable.

UMER studies provide another realistic single-invariant implementation and
discuss its design sensitivities [3]. Use such examples to choose tests rather
than copying their numerical tolerances as universal requirements.

## Diagnose deviations from the hypothesis

| Observation | Discriminating comparison | Likely next action |
| --- | --- | --- |
| Invariant varies strongly even on momentum | Check return matrix and normalization plane | Repair matching or coordinate conventions |
| Variation decreases with integration refinement | Hold physical magnets fixed and refine tracking | Resolve numerical error before optimizing |
| Variation decreases only with more physical segments | Compare continuous-profile approximation | Reconsider magnet count/gaps and field feasibility |
| On-momentum works; off-momentum fails | Dispersion, chromaticity, and synchrotron-off comparisons | Correct relevant perturbations and reassess bandwidth |
| Large tune spread comes with a shrinking stable region | Compare footprints on the same accepted domain | Reduce strength or change insert optics |

Evaluate invariants at a fixed observation plane using its fixed nominal
normalization. Do not renormalize separately for every turn to manufacture
conservation. Use an explicit finite scale for relative deviations near H=0.
Archive lost trajectories separately rather than taking statistics only from
their short surviving prefixes. Radiation damping and collective forces change
the Hamiltonian problem; distinguish those studies from conservative invariance.

## Acceptance and scientific boundary

Require finite-turn survival over the requested connected domain and momentum
range, measured useful tune spread, feasible magnets, and stated robustness.
Demonstrate sufficient turn count and integration resolution for the comparison.
A small invariant variation is supporting evidence, not a stability theorem.
Large incoherent tune spread alone does not prove Landau damping or increased
current limit: those require a beam-distribution and collective-response model.

## Sources

1. V. Danilov and S. Nagaitsev, *Nonlinear accelerator lattices with one and two
   analytic invariants*, PRSTAB 13, 084002 (2010),
   https://doi.org/10.1103/PhysRevSTAB.13.084002.
2. S. A. Antipov et al., *Single-Particle Dynamics in a Nonlinear Accelerator
   Lattice: Attaining a Large Tune Spread with Octupoles in IOTA*,
   https://arxiv.org/abs/1604.08565 — octupole profile, normalized construction,
   finite-segmentation and single-particle comparisons.
3. *Single-invariant nonlinear optics for a small electron recirculator*,
   PRAB 22, 041601 (2019), https://doi.org/10.1103/PhysRevAccelBeams.22.041601.

The benchmark domain, diagnosis table, and software choices are recommendations
for this module. No new lattice-performance result is claimed here.
