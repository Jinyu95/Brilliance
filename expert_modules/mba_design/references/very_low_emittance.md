# Very-low-emittance design: layout and optics exploration

Transfer mechanisms from a scientist-validated lattice below 10 pm rad,
not its saved strengths, exact magnet counts or fitted lattice. This is proposed
exploration, not the reference's known original optimizer or a success guarantee.
Preserve the [conventional HMBA](hmba_outcome_api.md) route. For a fresh project,
do not load another project's fitted lattices unless the scientist explicitly requests reuse.

## 1. Search physical layouts

Separate discrete layout (physical magnet counts, ordering and roles), continuous
geometry and fields. A few prescribed insertion lengths are not a layout search.
For the two-arc hypothesis:

```text
arc = half straight A -> upstream low-H bending/focusing -> field-shaped transition
  -> ONE localized correction region -> independent downstream transition/bending/focusing
  -> half straight B
block = arc + physical_mirror(arc)
```

`physical_mirror` is conceptual, not an API: preserve net bending direction and
faces/segment interfaces. The reflected arc provides the second bump and full
intermediate user straight. P blocks give 2P user straights, length C/P and angle
2*pi/P. Count joins once; record half-cell/full-period identity and straight counts.
Budget both bending sections, correction space and straights before matching;
preserve positive lengths and exact length/angle sums. If circumference is an
upper bound, do not fill it automatically by lengthening dipoles.

Design sections, not pairs of finished MBAs. Use independent upstream and downstream
counts and parameters; they need not be mirror copies. Apply symmetry boundary
conditions only at actual symmetry planes. Release inherited tunes, two-family
core grouping and internal periodic/achromat boundaries.

Compare structurally distinct, matched hypotheses under the same machine budgets,
including dense many-weak-bend layouts. Insert/remove a physical bending/focusing unit,
redistribute units around bumps or change the correction-site ordering. Rematch
each child from current-project parents; preserve competing layouts, not parent optics.

## 2. Allocate magnets by role, not leftover space

Alternate short focusing quadrupoles with defocusing-gradient positive bends.
For dense low-H sections, start by exploring physical bends of a few tenths of a
metre and varied focusing spacing, not inherited metre-scale conventional defaults.
These are initialization scales, not mandatory bounds: longer transition LGBs may
be useful. Shortening bends alone does not guarantee lower emittance.

Do not set every bend to `remaining_length / bend_count` and freeze that allocation.
Expose independent physical-dipole angle and length controls, quadrupole lengths,
drift allocation and gradients under declared global symmetry. Begin with grouped
section budgets, then release individual physical controls. Enforce angle closure
through a dependent variable or normalized weights, not equal angles.
Uniform values are acceptable seeds, not compulsory ties; equality after a genuinely
independent search may be legitimate. Avoid uniformly rescaling every bend.

Scan focusing and bend gradients of a native periodic motif for several stable
seeds, then release motif periodicity. Inspect actual K1, not plot color:
a bending quadrupole is a focusing station. A physical magnet's numerical slices
add no focusing stations or independent low-emittance cells.

Test weak reverse curvature, retaining their focusing role. Compensate negative
angles with positive bending. Consider longitudinal-gradient bends (LGB) at transitions
into/out of the correction region. Allow segment gradients to vary with field profiles;
do not require monotonic profiles. Vary edge-versus-centre weights as well as tilt;
one side-shared knob cannot explore local curvature and individual field profiles.
Include their controls in candidate construction; preserve correct external
edges/internal interfaces. [Field-shaping mechanisms](https://arxiv.org/abs/1810.11286).
Use justified field/aperture limits; facility operating values are not universal
technology ceilings, especially for combined-function gradients.

## 3. Build a focusing section inside each bump

Reserve finite SF/SD sites with complementary beta-times-dispersion weights,
retaining the quadrupoles: sextupoles/octupoles cannot replace their linear focusing.
A useful optical-role hypothesis, with finite drifts/magnets, is:

```text
transition - QD - SD - QF - SF - QF - SD - QD - transition
```

QF/QD denote horizontal focusing/defocusing; QF may also carry weak reverse bending.
This is a role sketch, not fixed counts, symmetry or fitted strengths.
Compare alternative telescopes with independently positioned internal quadrupoles.
SF can lie near the bump peak, SD on its slopes; do not require equal dispersion
or Dpx=0 at every sextupole. Merely including quadrupoles is insufficient:
measure their K1, beta separation and finite-length chromatic response.
Compare K2, integrated K2L and family multiplicity with consistent conventions.

The region need not sit at the exact geometrical centre. Treat interface
beta/alpha/Dx/Dpx as shared unknowns; use JuTrack propagation for continuity,
achromatic user straights and complete-block periodicity.
Inner matching and outer search own disjoint coordinates; evaluate returned matched elements.
Rematch after geometry changes; changing strengths alone cannot relocate correction regions.
Inspect site Dx/Dpx/betas and non-bump maximum absolute Dx and H/I5; suppress secondary humps.
Fixed bump-height/beta targets are aids, not substitutes for measured correctability.

## 4. Verify the actual search freedoms and objective

Before a campaign, show an ordered physical-magnet/knob table: region, name, L,
angle, K1, free/fixed parameters, control owner and ties. Aggregate slices by physical identity.
Perturb independent lengths and angles nonuniformly and inspect the constructed
candidate: advertised knobs must actually change it, with only declared symmetry
ties and exact budgets. Plot internal bump quadrupoles and label their names/signs.

Use one project-local evaluator for matching, refinement, selection and reporting,
with candidate-bound regions and identical observables:

1. Construct, rematch and screen native stability.
2. Measure regional optics, I2/I4/I5, emittance, damping and natural chromaticities.
3. Measure chromatic response, solve bounded correction strengths for the declared
   target, apply them and verify corrected chromaticities.
4. Return emittance, correction residual and required peak normalized strength
   inside the same candidate evaluation used by the optimizer.

A post-search correction report does not feed back into optimization.
Recompute response when linear optics change. Predicted correction is not verification.
Use `jutrack_design_metrics`, `jutrack_chromaticity_metrics`,
`region_optics_summary`, `measure_response`, `bounded_response_correction` and `api_help`.
No hand-written transport surrogate replaces JuTrack.

Preserve alternatives: retain a trade-off front of corrected candidates and improve
both low-emittance and readily correctable seeds. Retain declared beta envelopes,
site optics, localization, achromat, damping and engineering conditions;
do not silently discard them. Revise assumptions explicitly.
Smaller K2 obtained through enormous beta is not automatically improved nonlinear optics.
Do not drive longitudinal damping toward zero to reduce emittance.

## 5. Explore quickly, then qualify

Use persistent parallel workers: initialize JuTrack once, build once per layout,
update native fields during continuous optimization. Topology changes need a new
native lattice/index map, not row synchronization. Optimize continuous vectors per layout;
do not reuse another topology's knob map. Cache evaluations; never reuse stale optics.
Profile complete candidate cost, separating matching, optics, radiation and correction.

Search periodic blocks with correct ring scaling; verify finalists as explicit rings.
Avoid tracking during this linear exploration. Allocate PSO/MOPSO and refinement effort
by feasibility/progress. With no stable matched starts, restore feasibility.
If a transition dominates I5, release its geometry/fields; weak correction calls for
new focusing/site arrangements. Report realized knob movement: tiny steps and solver
termination can indicate scaling/initialization stagnation, not exhausted layout freedom.
Failed searches are not structural impossibility.

Examine two-plane correction-pair maps, not just report phase advances afterward.
Keep independent nonlinear families despite reflected linear geometry.
Do not force identical nonlinear strengths in both insertions.
Chromatic correction does not prove nonlinear acceptance; track corrected alternatives.
Use [injection-based targets](injection_acceptance.md) for both transverse planes;
scan beyond them and retain these targets when optimizing momentum acceptance.
[Cross-straight pairing](https://arxiv.org/abs/2212.01870).

Save exact candidates, solver termination and integration settings before reporting;
verify momentum-step convergence and carry those settings into restarts.
Report distinct physical layouts evaluated and measured trade-offs, not just trial counts.
