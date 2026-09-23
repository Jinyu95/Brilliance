# Very-low-emittance design: construction and search recipe

Transfer mechanisms from a scientist-validated lattice below 10 pm rad,
not its saved strengths, exact magnet counts or fitted lattice. The construction
principles are observed; the initialization/search strategy below is a proposed
way to recover them independently, not the reference's known original optimizer
or a success guarantee. Preserve the separate [conventional HMBA](hmba_outcome_api.md) route.

## 1. Allocate an optical block

For the exploratory two-arc hypothesis, start with this physical grammar:

```text
arc = half straight A -> upstream low-H bending/focusing -> field-shaped transition
  -> ONE localized correction region -> independent downstream transition/bending/focusing
  -> half straight B
block = arc + physical_mirror(arc)
```

`physical_mirror` is conceptual, not an API or simple list reversal. Preserve net
bending direction and validate finite magnet faces/segment interfaces under the
chosen coordinate convention.

The reflected arc provides the second bump and full intermediate user straight.
With P blocks, there are 2P user straights, block length C/P and net angle 2*pi/P.
Count joins once; declare half-cell/full-period identity and straight counts.
Allocate both straights, both bending sections, correction space and clearances
before matching. Enforce positive lengths and exact length/angle sums,
not by shortening a user straight afterward.

Design sections, not pairs of finished MBAs. Use independent upstream and downstream
parameters; an arc's sides need not be mirror copies. Apply symmetry boundary
conditions only at actual symmetry planes. Release inherited unit tunes,
two-family core grouping and internal periodic/achromat boundaries.
Repeating an identical symmetric arc twice adds no new design freedom.

## 2. Create a physically useful, adjustable seed

Alternate short focusing quadrupoles with defocusing-gradient positive bends
throughout each low-H section; do not put all focusing at the ends of a bend train.
Inspect actual K1, not element type/plot color: a bending quadrupole remains a
focusing station. Distinguish each physical magnet from its numerical slices;
splitting one bend adds no focusing stations.

Initialize a short bending/focusing motif with native periodic optics/radiation:
scan focusing and bend gradients within the declared engineering envelope and
retain several stable low-emittance settings: motif results are seed diagnostics,
not the requested ring. Analytical TME is a scale, not a
mandatory matching target. Use these as starting strengths, then release motif
periodicity and independent strengths when assembling the complete block.
Compare structurally distinct, matched hypotheses with different physical
focusing spacing under the same machine budgets; do not only retune one frozen layout.

Test weak reverse curvature at selected focusing stations while retaining their
focusing role. Compensate negative angles with positive bending. Start localized
longitudinal-gradient bends (LGB) at transitions into/out of the correction region;
not every positive bend needs the same segmented profile. Expose independent
physical-dipole angle and length controls, quadrupole lengths, drift allocation,
gradients and selected reverse angles under declared global symmetry.

Allow segment gradients to vary with field profiles. Do not require monotonic
profiles or a high-field centre: vary edge-versus-centre weights as well as tilt,
independently for different magnets. Include their controls in candidate construction.
One side-shared knob cannot explore local curvature and individual field profiles.
List free/fixed parameters, including aggregate bend length. Preserve correct
external edges/internal interfaces and verify native integration convergence.
Use apertures and combined-field limits, not unrelated historical strength caps.
Begin with grouped section budgets, then release individual physical controls
around matched seeds rather than randomizing every coordinate at once.
[LGB/reverse-bend mechanisms](https://arxiv.org/abs/1810.11286).

## 3. Match interfaces, not isolated components

Reserve finite SF/SD sites with complementary beta-times-dispersion weights,
retaining the quadrupoles: sextupoles/octupoles cannot replace their linear focusing.
Initially their strengths may be zero, but their lengths occupy real space.
The correction region need not sit at the exact geometrical centre.

Treat interface beta/alpha/Dx/Dpx as shared unknowns between adjoining sections.
With JuTrack propagation, match continuity, every user straight's Dx/Dpx, and
complete-block periodicity. At a true reflection plane alpha/Dpx may vanish;
do not impose this at arbitrary section boundaries. Sectional shooting can
initialize the solve, but recompute periodic optics on the assembled block.
Inner matching and outer search must own disjoint coordinates; evaluate/save
the returned matched elements, not the original structural proposal.
Scale coordinates and residuals by their physical ranges and declared tolerances.

Obtain a stable block, then continue toward useful insertion optics in modest
steps while controlling radiation. Optimize site Dx/Dpx/betas and non-bump H/I5
separately; suppress secondary humps, not require zero dispersion inside bends.
If a step fails, return to a matched checkpoint and reduce it or release the
responsible geometry. Changing strengths alone cannot relocate correction regions.

## 4. Evaluate one candidate consistently

Use one project-local evaluator for matching, refinement, selection and reporting
with candidate-bound regions and identical observables:

1. Construct the proposed geometry/fields and rematch. Reject invalid geometry
   or unmeasurable optics separately from runtime/API errors.
2. Measure native periodic optics and regional I2/I4/I5, emittance and damping.
3. Measure chromatic response at the allocated finite sextupoles, solve bounded
   correction strengths, apply them, and verify corrected chromaticities.
4. Return emittance, site optics, regional radiation, actual correction residuals
   and strength burden together. Retain a trade-off front of corrected candidates.

Use `jutrack_design_metrics` and `jutrack_chromaticity_metrics`; consult `api_help`.
Combine `region_optics_summary`, `measure_response` and `bounded_response_correction`.
An origin-map derivative is not closed-orbit chromaticity; predicted correction
is not verification. Check momentum-step convergence before ranking candidates.
Measure chromatic response in the same candidate evaluation as emittance and
regional optics, not just afterward. Fixed bump-height/beta targets are
initialization aids, not substitutes for measured correctability.

Retain declared beta envelopes, site optics, localization, achromat, damping and
engineering conditions; do not silently discard them. Revise assumptions explicitly.
Smaller K2 obtained through enormous beta is not automatically improved nonlinear optics.
JuTrack supplies epsilon_x=Cq*gamma^2*I5/(Jx*I2), Jx=1-I4/I2; low Dx alone is not
low H. Do not improve Jx by driving longitudinal damping toward zero.
Rematch after geometry changes and re-correct chromaticity before comparison.

## 5. Adapt the search from measured limitations

Start a bounded pilot before allocating a larger campaign. Scale sampling to
dimension and feasible fraction; use independent feasible starts, checkpointed
PSO/MOPSO or local refinement, and configured parallel resources. No universal
population count or iteration limit establishes convergence.

| Observed limitation | Next experiment |
| --- | --- |
| No stable matched starts | Restore feasibility with JuTrack boundary residuals; do not optimize identical penalty values. |
| Broad high-H bending | Vary physical focusing spacing/strength and curvature allocation; compare native I5, not dispersion alone. |
| Transition dominates I5 | Release its angle, length, segment profile and neighboring matching optics together. |
| Weak correction response | Vary insertion length/site separation and beta contrast; measure response rank and bounded residuals. |
| Good radiation but poor acceptance | Test two-plane correction-pair maps, off-momentum optics and remaining nonlinear controls. |

Do not restrict pair cancellation to one arc; it may cross a user straight.
Evaluate complete finite-length maps, not just report phase advances afterward.
[Cross-straight pairing](https://arxiv.org/abs/2212.01870).
Keep independent nonlinear families despite reflected linear geometry.
Do not force identical nonlinear strengths in both insertions. Chromatic correction
does not prove nonlinear acceptance: track several corrected candidates with
controls beyond the two chromatic combinations before selecting finalists.

Preserve alternatives; a failed restricted search is not structural impossibility.
Save solver diagnostics and explain each change with a scoped optics/radiation/
correction comparison. Verify selected explicit rings and acceptance as in the
conventional guide. Continue within authorized budgets; a chat pause is not saturation.
