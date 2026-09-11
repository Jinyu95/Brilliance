---
id: laboratory.mba_design
title: Constructive modified-TME and MBA design methods
summary: Construct modified-TME and conventional MBA seeds; hybrid conversion replaces outer sections with new DBA-like bump cells around a preserved core, then separately recovers core quality and emittance.
domains: [mba, lattice design, electron storage rings, synchrotron light sources]
authors: [Brilliance project team (draft maintainer)]
tags: [mba, 5ba, 6ba, 7ba, 9ba, tme, modified-tme, hybrid-mba, matching, radiation, optimization, jutrack]
version: 1.13.0
status: draft
knowledge_status: draft
kind: method
enabled: true
---

# Constructive modified-TME and MBA design methods

Module tools are available through `mba = api.expert_module("laboratory.mba_design")`.
See `references/physics_reference_api.md` for analytical reference signatures and
`references/hybrid_shape_api.md` for optional hybrid diagnostics and acceptance.
The `api.tme_targets` calls in the established seed method remain supported as
module-declared aliases; their formulas and the working construction method are unchanged.

## Scope and route choice

This note supplies construction methods, not stored lattice templates or a mandatory application
workflow. For greenfield conventional MBA work, the default route is **an optimized modified-TME
unit → independently matched complete MBA → complete-cell optimization**. Hybridization is a
later hypothesis, not a required step. Continue the exact current project lattice for follow-ups;
never reconstruct a successful matched cell from remembered constants.

The complete periodic MBA cell remains the final design authority. A good unit is a seed, not
proof of a good MBA. Nevertheless, skipping the unit solve and guessing seven bends with
alternating quadrupoles is NOT the TME-derived route. Direct complete-cell construction is a
legitimate alternative when requested or supported by a concrete argument. Do not require two
arbitrary complete-cell guesses before establishing a useful unit.

Only pure positive homogeneous dipoles are covered. No dipole gradients, reverse bends, or
longitudinal-gradient bends are authorized here. Do not import an existing facility lattice.
Historical regression outputs must never be executed as research stages or loaded as hidden seeds.

## Contract and length budget — before construction

Record E, number of periods P, bend count M, circumference, minimum straight, apertures, field
limits and minimum element lengths. Distinguish an exact circumference from an upper bound,
and an emittance aspiration from calculation validity. Lcell=C/P and theta_cell=2*pi/P.

Reserve the insertion straight and BOTH end-matching regions first:

    (M-2)*Lunit + Lmatch_left + Lmatch_right + Lstraight = Lcell

The matching budgets include their terminal bends, quadrupoles, and drifts. This is an initial
allocation, not a restriction on later optimization. Check that the terminal bend field/length
budget is feasible. Never construct an arc first and shorten the straight or append one long
drift to repair circumference.

One explicit initial bend-angle hypothesis is:

    theta_unit = theta_cell/(M-1)
    theta_end_left = theta_end_right = theta_unit/2
    (M-2)*theta_unit + 2*theta_unit/2 = theta_cell

Equal angles theta_cell/M are a different hypothesis, not the same scale. Neither distribution
is optimal by definition. Use each proposed individual bend angle for its TME reference,
NEVER the entire achromat angle as if it were an interior bend.

Use api.magnet_strength_limits(energy_gev=E, ...) and retain the same engineering envelope
through search, matching, diagnostics, and optimization. A diagnostic scan outside those
limits is not an admissible solution. Do not infer a field limit from a poor optimization result.

If the scientist has NOT specified circumference, do not turn an arbitrary first circumference
guess into a hard requirement. Unit length and terminal matching allocation are provisional
design variables. A short unit with QF at its field limit and emittance far above its reference
needs length/fraction exploration, not more generations of the same fixed-geometry strength
scan. If circumference IS fixed, redistribute the complete budget, keeping the required straight;
do not silently enlarge the ring. Recompute the analytical beta/dispersion references whenever
the proposed bend length changes.

### Choose an optical length scale, not an arbitrary small unit

The dimensionless focusing strength of a geometrically scaled unit is `K1*U**2`, not K1
alone. A useful unit-length scout is `chi = U*sqrt(k1_max)`. For the four-quad topology below,
try several chi values over roughly [4, 8], so `U = chi/sqrt(k1_max)`, with the geometry
fractions AND both strengths free at each scale. This interval is an initial numerical
hypothesis, not a universal TME condition or a validity bound. A dimensionless scout avoids
implicitly demanding unavailable focusing simply by choosing very short magnets and drifts.
It does not supersede dipole-field, minimum-length or circumference checks.

If circumference is free, include U as a sixth optimization coordinate alongside the fractions
and strengths, or compare a small bank of unit lengths before refining the promising basins.
Bound each complete-machine proposal by `(M-2)*U + 2*T + Lstraight` and any user-specified
circumference upper bound; the terminal budget T must still accommodate its independent match.
If circumference is exact, derive allowed U from the complete length allocation and release
terminal/core allocations together when necessary. Do not change an exact machine requirement.
No useful solution in a short-U/fixed-fraction strength scan means that initialization failed,
not that the magnet technology or the MBA concept is inadequate.

## U1 — Analytical reference, explicitly reported

For a homogeneous bend of length Lb and angle theta, at its physical centre:

    beta_TME = Lb/(2*sqrt(15))
    alpha_TME = 0
    eta_TME = Lb*theta/24
    eta_prime_TME = 0
    epsilon_TME = Cq*gamma**2*theta**3/(12*sqrt(15)*Jx)

These are small-angle, uncoupled homogeneous-bend references, not tune prescriptions or a
promise that finite bounded magnets attain them. api.tme_targets returns documented attributes
and as_dict(); publish their VALUES, not dir(reference). Distinguish analytic Jx=1 from measured
JuTrack damping. Never walk nested dictionaries for guessed physics-output keys.

## U2 — Construct a real symmetric unit

A useful optical construction hypothesis, from one unit boundary to the other, is:

    Dend – QF – D1 – QD – D2 – B – D2 – QD – D1 – QF – Dend

QF has positive K1 and QD negative K1 in the configured JuTrack convention. Both instances of a
family are tied. Solve their strengths; this is not a lattice with preassigned numerical settings.

With available unit length U, choose fractions fB, fQ, f1, f2:

    Lb=fB*U; Lq=fQ*U; d1=f1*U; d2=f2*U
    dend=(U-Lb-4*Lq-2*d1-2*d2)/2

For a small initial search tie d1=d2. A testable initial box is fB in [0.20,0.36], fQ in
[0.08,0.14], and f1=f2 in [0.015,0.04]. These fractions are hypotheses, not universal engineering
limits. Reject negative/too-short dend BEFORE constructing elements. Broaden fractions when
length, magnet limits, or diagnostics warrant it. A bare QF~1/Lb² formula cannot determine a
practical cell without quadrupole lengths and intervening geometry.

Construct left=[Dend,QF,D1,QD,D2], then:

    rows = api.symmetric_cell_elements(left, center=full_bend)

The centre is the FULL physical bend (Lb,theta), inserted once. Do not halve its length while
keeping the full angle, or duplicate a central half bend. For an even-bend complete cell whose
symmetry plane is between elements, omit center and budget the complete half-cell accordingly.
Tag family and design_role using the constructor keywords; do not guess a fields= constructor API.

Use api.audit_lattice_constraints for required length, total angle, bend count, mirror symmetry,
minimum lengths, and the energy-derived field envelope. Repeat the audit on every changed
candidate. This establishes construction validity, NOT stable optics or a successful design.

Mirror names differ but families must remain tied. Tag the original QF/QD constructors with
`family="QF"` / `family="QD"`; `api.symmetric_cell_elements` preserves those family labels on
both halves. For a strength-only local check, the exact mutation pattern is:

```python
strengths = api.compile_family_parameterization(
    rows, field="k1_m2", families=["QF", "QD"],
    bounds={"QF": [0.0, k1_max], "QD": [-k1_max, 0.0]}, role="outer_design")
# Each strength knob targets BOTH physical mirror partners, not just the left name.
editor.apply_parameterization(strengths, position)
linear = editor.linear_metrics()
# After this strength-only search (not a complete unit-geometry optimization):
selected_rows = strengths.apply(result["best_candidate"]["position"])
```

Changing only an element named `QF` leaves `R:QF` unchanged and destroys mirror symmetry.
For the full geometry search, write `build_unit_rows(U,fB,fQ,fgap,KF,KD)` from the construction
equations above, returning both halves with the same current fractions and family strengths.
Call `editor.linear_metrics(proposal_rows, include_profile=True)` to synchronize ALL represented
lengths and strengths on the existing JuTrack elements; then evaluate radiation on that current
editor lattice. Do not update a length in the JSON rows but leave the old length in JuTrack.
Audit the selected exported rows again before reporting a symmetric candidate.

## U3 — Solve and optimize the unit with JuTrack

With d1=d2 and a justified allocated U, a small search vector is [fB,fQ,fgap,K1_QF,K1_QD].
If U is provisional, use [U,fB,fQ,fgap,K1_QF,K1_QD] instead. There is NO inner achromat match
for the isolated periodic unit: its boundary dispersion is generally nonzero. Requiring zero
dispersion there would solve a different problem. Never constrain its tunes to 0.25/0.25 or a
remembered working point. Avoid inventing transfer matrices instead of using JuTrack.

1. Build one mutable JuTrack editor, then update scalar fields by element name per proposal.
2. Reject geometry and magnet violations before simulation. Evaluate periodic optics next.
3. Retain finite stable points with disclosed stability margin and beta envelope. A trace near
   ±2 signals a stability boundary; it does not by itself diagnose weak or strong focusing.
4. Use `editor.evaluate_candidate(rows, energy_gev=E, periods=P*(M-1))` to apply the
   trial and obtain its `elements`, `linear_optics`, `radiation`, and `passed` together.
   Reject invalid damping; minimize positive natural emittance. Do not minimize a signed
   unphysical emittance. Never scan optics into a list of strength tuples and then evaluate
   radiation on an editor left at the last trial. That pairs different candidates' data even
   within a fresh project. Retain the returned snapshot with its metrics. Before transfer,
   replay the selected snapshot and verify that its objective is reproduced.
5. Populate generation zero entirely with feasible sampled positions. For five independent knobs
   use at least 50 particles (60 for six knobs including U). If no feasible population exists, change geometry/search bounds
   before ranking objectives. Record counts, settings, and reasons for rejection.
6. Run generic PSO with geometry fractions and strengths released, not a frozen-geometry K1
   scan. Record generation-zero and final objectives. Optional local refinement must respect
   the same fields and geometry. Unit optimization need not have a target tune or achromat.

Use api.particle_swarm_optimize and run.optimization_options(...) for durable execution and
live numerical progress. best_candidate["variables"] is a NAME→VALUE mapping, not a position
array. Extract values in the declared variable-name order. Public API references give exact
callback and result signatures.

Specifically, compile_family_parameterization can name a knob `k1_m2:QF`, not `QF`.
Use `best_position = result["best_candidate"]["position"]` and
`selected_rows = parameterization.apply(best_position)`. The equivalent mapping lookup is
`[best_candidate["variables"][n] for n in result["variable_names"]]`. Do not guess bare family keys or
reconstruct a selected lattice from a separately maintained set of constants. Print and archive
the actual ordered variable names and selected values. A postprocessing error after optimization
does not invalidate the saved numerical checkpoint or justify rerunning the scientific search.

The lower-level fast radiation call is api.jutrack_periodic_radiation_metrics(jt, editor.lattice,
energy_gev=E, periods=P*(M-1)); the keyword is periods, not repetitions. Never convert a
TypeError, AttributeError, or KeyError into an infeasible particle. Before a large search,
test the callback at the already evaluated baseline and verify it reproduces that result.
If the baseline changes from stable to infeasible, inspect the exact API failure and rejection
reason before searching. Record failure categories rather than one anonymous unstable count.
Bound the strength coordinates themselves by the declared k1_max; do not sample a much larger
box and throw most proposals away afterward. Do not pass checkpoint_path twice: use either
the paths supplied by run.optimization_options or an explicitly managed custom campaign.

For the half-angle-end hypothesis above, repeating the unit P*(M-1) times closes 2*pi. That
**repeated-unit reference ring is NOT the requested MBA** and generally has a different
circumference. It validates unit radiation only. Do not change a bend angle silently just to
fit a different repetition count.

## U4 — Unit evidence before transfer

Archive the selected unsplit unit, exact parameters, and source. Show:

Complete and present this unit checkpoint BEFORE terminal assembly; do not defer its geometry
optimization until an assembled ring fails. This is a component candidate, not the working
complete ring. Archive `evaluation["elements"]` from the selected/replayed evaluation rather
than rebuilding from a separate list of remembered strengths.

- allocated unit length, bend angle, geometry and magnet margins;
- initial versus optimized emittance and optimization history;
- achieved beta_x, alpha_x, Dx, Dpx at the PHYSICAL bend centre against the analytical values;
- radiation integrals, positive damping partitions, periodic closure, beta maxima, and measured
  emittance/reference ratio (do not force an exact or universal modified-TME ratio);
- one combined unit lattice/beta/dispersion plot via api.save_optics_lattice_plot.

Save the selected UNSPLIT unit before diagnostic plotting. Use the general interior sampler:

```python
sampler = api.build_jutrack_profile_sampler(selected_unit, max_step_m=0.05)
sampled = sampler.linear_metrics(sample_fractions={physical_bend_name: [0.5]})
# Verify sampled['stable'] and sampled['sampling']['map_equivalent'] before reporting.
profile = sampled['twiss_profile']
centre = next(p for p in profile if p.get('requested_sample')
              and p['source_element'] == physical_bend_name and p['sample_fraction'] == 0.5)
api.save_optics_lattice_plot(plot_path, profile, api.one_period_elements(selected_unit))
```

This measures the PHYSICAL bend centre with JuTrack while keeping the unsplit map and lattice
authoritative. Do not pass a diagnostic split to the existing optimization editor: an editor
is compiled for one element count/order/name/type list. A genuinely different element list
requires a separate editor. Do not disable that guard or export slices as physical magnets.
Joining only element exits draws a misleading flat segment across a symmetric bend.
If this reporting stage fails after optimization, use api.read_optimization_checkpoint and
the saved program's builder to replay the selected candidate; do not rerun the search.

Check the selected result on an explicit closed repeated-unit ring. Its success validates only
the unit, not the requested ring. If it remains tens of times its homogeneous scale, diagnose
centre optics, fractions, saturation and convergence before assembling a larger MBA.

## A — Transfer, match, diagnose, then optimize the complete MBA

### A1 — Match a one-pass terminal transition before closing the achromat

For the symmetric, uncoupled pure-dipole unit above, use its solved periodic Twiss at the UNIT
BOUNDARY as input to an independent end transition. The input dispersion is NOT zero. Match
from the outer edge of the low-emittance core OUTWARD to the centre of the insertion straight.
This is a line-optics boundary-value problem, not a periodic stability search of a guessed ring.
One-pass propagation remains evaluable when an as-yet-unmatched complete ring is unstable;
a periodic-cell response matcher often cannot even form its first response matrix there.

A useful terminal topology hypothesis has at least four independent quadrupoles, two before
and two after its terminal bend:

    core boundary — d0 — QM1 — d1 — QM2 — d2 — Bend_end
                  — d3 — QM3 — d4 — QM4 — half insertion straight

The four QM strengths are solved, NOT inherited from QF/QD. Their signs are not prescribed by
the unit family names. Choose their lengths, drift gaps, and end-bend length inside the reserved
terminal budget and field limits. This is a connectivity hypothesis with free geometry, not a
stored lattice. Other topologies are legitimate if they provide sufficient independent optical
response. Two adjustable quads alone generally cannot meet four independent boundary conditions.

Use the existing configured JuTrack bridge and public editor, not hand-written transfer maps:

```python
# unit_editor contains the exact selected full, unsplit unit.
jt = unit_editor.jt
tin = jt.periodicEdwardsTengTwiss(unit_editor.lattice, 0.0, 0)
# transition_editor contains the chosen core-to-ID-centre sequence.
# matching is a compiled parameterization of its independent QM strengths.
def terminal_residual(position):
    transition_editor.apply_parameterization(matching, position)
    tout = jt.twissline(tin, transition_editor.lattice, dp=0.0, order=0)
    return [float(tout.alphax), float(tout.alphay),
            float(tout.dx) / dispersion_scale_m,
            float(tout.dpx) / dispersion_slope_scale]
```

Solve these four residuals toward zero with bounded `scipy.optimize.least_squares` (available
in the configured Python) or a generic bounded-response callback. For conditioning, reasonable
initial scales are dispersion_scale_m=0.01 m and dispersion_slope_scale=0.01; these are numerical
scales, NOT acceptance tolerances. Start from several independently sampled in-bound strengths,
not only the same failed local point. Use the same magnet envelope throughout. Archive actual
unscaled terminal alpha_x, alpha_y, Dx and Dpx; check alpha near zero, |Dx|<=1e-4 m and
|Dpx|<=1e-4 separately. Inspect response rank, active field bounds and beta along the line.
If starts converge to a nonzero residual or enormous beta, release the terminal geometry or
allocate additional terminal length within the complete machine contract. Do not run an
unbounded retry loop or claim this four-quad hypothesis always has a solution.

The zero-alpha conditions put both transverse waists at the insertion-centre mirror plane.
They are symmetry conditions, not fixed tunes or requested numerical beta targets. With zero
dispersion and slope, reflecting the transition connects that ID plane back to the unit's
boundary optics. For a different symmetry or coupled layout, formulate its appropriate boundary
conditions instead of applying these four scalar conditions blindly.

### A2 — Assemble and independently verify the complete periodic cell

After a successful terminal solve, export the exact current transition rows. For this unit route:

```python
transition_rows = transition_editor.export_rows()
complete_rows = (api.mirror_elements(transition_rows, name_prefix="LEFT:")
                 + api.repeat_cell(selected_unit_rows, M - 2)
                 + transition_rows)
```

The sequence starts and ends at insertion-centre symmetry planes. Each transition contains ONE
half straight, so both together count the required full straight exactly once per cell. If T is
one terminal length EXCLUDING its half straight, then `(M-2)*U + 2*T + Lstraight = Lcell`.
Each terminal carries theta_unit/2, so its bend sum remains the original theta_cell. This
construction applies to odd and even M; an even number of interior units places the core mirror
plane between units without splitting or duplicating a central physical bend.

Audit length, angle, bend count, fields and mirror symmetry again. Evaluate the COMPLETE periodic
cell with JuTrack; require stable transverse maps, achromatic insertion boundaries, acceptable
beta and valid radiation. Repeating a solved unit inside a matched achromat preserves its core
optics for this initialization, but the terminal contribution still changes total I5/I2 and
emittance. Compare measured values rather than assuming an increase or decrease.

Then construct the explicit ring using `api.cell_ring(..., cell=complete_rows, periods=P)` and
verify it with `api.evaluate(..., include_profile=False)` and `require_radiation()`, which uses
JuTrack ringpara. `api.bend_radiation_contributions` requires this explicit CLOSED FULL RING,
not the one-cell object; it groups repeated instances into per-physical-cell bends and checks
their I2/I4/I5 sums against ringpara. Plot only the one-cell optics with its corresponding layout.
Natural chromaticity, sextupole correction, nonlinear acceptance and RF are separate later checks;
a bare achromat with positive damping partitions is not an operational machine design.

### A3 — Diagnose transfer and optimize the complete MBA

Keep the original angle and full-cell length budgets. Place the solved interior units and
construct independent terminal transition/matching regions in their reserved space. End bends
and matching quadrupoles do not inherit all interior settings. Match the COMPLETE periodic
cell at the insertion boundary to Dx=0, Dpx=0 with disclosed tolerances (initially 1e-4 m, 1e-4).

Never tile one identical QF-QD-bend-QD-QF block around all M bends and call it an achromat.
For a 7BA, B3–B5 form the low-H core hypothesis; B1–B2 and B6–B7 can acquire transition/matching
roles during complete-cell optimization. Generalize by physical positions and symmetry, not
hard-coded seven-bend family names. An even-M cell has no single centre bend.

Use separate matching families; inspect response rank, residual history and stability when a
match fails. A failed local match does not prove a topology impossible. Compare repeated-unit
and actual MBA emittance explicitly. Compute per-bend H, I2, I4, I5 and I5 fractions using
api.bend_radiation_contributions. Show a real table and combined one-MBA-cell optics plot.

Outer I5 dominance calls for transition/angle/geometry changes; core dominance calls for core
retuning. A good unit can transfer poorly. Treat that as evidence about the construction, not
permission to call a high-emittance, unmatched result a completed design.

Use api.optimize_lattice_parameters or task-specific generic PSO callbacks for complete-cell
search. Release symmetric bend angles/lengths, quadrupole lengths/structural K1 and drift
allocations under exact total length and bend angle. The inner matcher exclusively owns its
own matching strengths. No coordinate may be overwritten by both layers. Do not freeze seed
tunes or call a fixed-geometry strength scan global optimization.

Use a fully feasible production population, normally at least ten times the independent
outer-knob count, process-parallel JuTrack evaluation and persistent checkpoints. A physically
correct objective and legal knobs matter more than forcing an optimizer into the first call.

## H — Optional hybrid and nonlinear branches

This module owns its hybrid tools, shaping policy and candidate-acceptance checks. Load its
Python namespace with `mba = api.expert_module("laboratory.mba_design")`; the application's
general JuTrack, lattice-editing, optimization and plotting tools remain under `api`. Read
`references/hybrid_shape_api.md` for the executable callback recipe and exact report contract.
The optional registered acceptance rule accompanies this module; it is not a built-in MBA
workflow in the Brilliance application.

Hybridization is justified by the request or inadequate chromatic leverage/nonlinear acceptance.
Missing an aspiration after a short campaign does not establish conventional saturation. Look
for a plateau, independent starts, local refinement and identified active constraints.

This branch modifies the exact accepted COMPLETE periodic MBA cell. Preserve its parent lattice,
source and the working unit/assembly pipeline. A follow-up to form an H-MBA is not permission to
restart TME design. Do not assume a previously optimized conventional MBA is already hybrid.

Default conversion route: **replace both outer matching sections with newly designed DBA-like
bump sections around the preserved middle M-4-bend core**. Do not begin by globally optimizing
the old conventional cell. For 7BA retain the full three TME-like core units, including their
quadrupoles and boundary drifts. Read `references/outer_dba_api.md` first.
Stage 1 is USABLE_BUMPS_FEASIBLE: two extended high-dispersion drift regions, spatial
separation from the core, stable achromat, geometry/magnet bounds, a disclosed beta envelope
and positive damping. Source beta and dispersion preservation is OPTIONAL, never the default.
Retaining core connectivity is a starting structural hypothesis, not a requirement to freeze
core quadrupole strengths or reproduce old periodic optics. Solve new centre optics and ALL
globally mirror-paired quadrupoles, including the core, together; publish
before/after diagnostics without treating deviations as failures. Emittance and further
core-H/I5 reduction are diagnostic initially, not baseline-equality or no-worsening gates.
Do not label stage 1 a final hybrid design. Final low-core-H quality remains a later gate.

### H1 — Assign roles by physical bend order, not inherited unit labels

Enumerate the M physical positive bends along ONE periodic cell, from its left ID boundary to
its right ID boundary, as B1 through BM. A diagnostic split of one magnet is still ONE physical
bend. Use the unsplit saved element list for identity, order and s positions; do not count
diagnostic slices as extra magnets. Publish the role-to-element map before changing the lattice.

For the seven-bend cell constructed by A2, the mapping is:

| Role | Saved physical element | Optical region |
| --- | --- | --- |
| B1 | left terminal bend | left DBA-like region |
| B2 | U001:B | left DBA-like region |
| B3 | U002:B | low-emittance core |
| B4 | U003:B | low-emittance core |
| B5 | U004:B | low-emittance core |
| B6 | U005:B | right DBA-like region |
| B7 | right terminal bend | right DBA-like region |

Resolve the actual terminal names from the archived rows; do not invent a terminal name.
The left DBA-like region is B1–B2, the low-H core is B3–B5, and the right DBA-like region is
B6–B7. U002/U004 are B3/B5, NOT outer bump units. Keep them in the low-H core even if an older
script called them "outer units". Unit labels record construction provenance, not H-MBA roles.

For general M >= 5, assign the first two and last two bends to the outer DBA-like regions;
the middle M-4 bends form the core. Thus 5BA has core B3, 6BA has core B3–B4, and 9BA has
core B3–B7. An even-M cell has a central mirror plane between bends, not a fictitious central
bend. Tie mirror pairs by their physical roles. If a cell wraps across its ID or its bend
count differs, establish the correct one-cell scope and count before doing this assignment.

Use `mba.hybrid_mba_regions(rows)` for the element-by-element region inventory and complete-cell
mirror audit. Classify quadrupoles by their PHYSICAL position, not a unit name: B1–B2 is the
`bump_generator`, B2–B3 is the `dispersion_suppressor`, and B3–B5 (general middle M-4 bends and
their internal gaps) is the `low_H_core`. These groups include the corresponding right-side
mirrors. A quadrupole named U002 can be in the B2–B3 suppressor BEFORE B3; do not classify it
as a core knob merely from the name. The physical bends B3/B5 themselves remain in the core.
For optional knob selection use `hybrid_mba_quadrupole_knobs(..., regions=[...])`.

Only complete-cell reflection is required. U001 left/right quadrupoles may have different
strengths; each must equal its partner in U005 ACROSS THE WHOLE CELL, not its partner across
the old isolated TME bend. The same applies to suppressor/core transitions. Do not reapply
the unit-generation symmetry rules to an H-MBA descendant. A real full-cell mismatch must
still be fixed explicitly or restored from a preserved symmetric candidate; never average
strengths silently or disable the final complete-cell audit.

### H2 — Diagnose the actual dispersion shape before optimizing

Obtain a JuTrack optical profile for the SAME current candidate and ONE full periodic cell.
Use Dx, Dpx, beta_x and alpha_x at resolved s positions. Sample inside the interbend regions
and core magnets, not only at magnet exits. The provided sampler propagates independent
partial elements from each ORIGINAL physical entrance and checks original maps/exits.
Do not chain subdivided quadrupoles: multiplying their integration steps changes the lattice
map. Never replace the physical design with diagnostic slices. Increase sampling around a
possible extremum until its location, slope and height are resolved. Do not infer a measured
interior maximum by fitting a curve through two bend-exit values.

Measure these quantities:

    D_bump_L = max Dx(s), for exit(B1) < s < entrance(B2)
    D_bump_R = max Dx(s), for exit(B[M-1]) < s < entrance(BM)
    D_core = max abs(Dx(s)), throughout entrance(B3) ... exit(B[M-2])
    Hx(s) = ((1 + alpha_x**2)/beta_x)*Dx**2 + 2*alpha_x*Dx*Dpx + beta_x*Dpx**2
    H_core = max Hx(s), throughout the same core
    dispersion_contrast = mean(D_bump_L, D_bump_R) / D_core

The bump windows are the spaces BETWEEN each outer bend pair, not B2/B6 exits or the outer
unit centres. Each maximum must be strictly interior to its window, have small |Dpx|, and
have a resolved rising-to-falling Dpx sign change on its two sides. A monotonic hill whose
largest outer value is a window edge is NOT a bump. A flat-topped interior maximum is valid
only with resolved rising and falling sides; a wholly flat interval is not evidence of a bump.
Ignore floating-point slope noise when locating those sides. Report both peak positions and
slopes, not just peak heights. H_core is a sampled maximum: refine the grid for selected
candidates near a shape threshold and report its sampling resolution.
The useful object is a DRIFT INTERVAL for sextupoles, not a mathematical peak in a magnet.
In a field-free drift Dx(s)=Dx_in+s*Dpx_in and Dpx is constant. A stationary high-dispersion
drift is a plateau; do not demand an isolated maximum inside it. The regional maximum may
lie in a quadrupole only if neighbouring physical drift intervals independently qualify.
Merge adjacent drifts/markers when measuring usable installation length. Report minimum
dispersion, slope, usable contiguous/total length and beta_x*Dx/beta_y*Dx for BOTH sides.
The entire usable interval must exceed both the absolute dispersion threshold and the
core-relative threshold. A narrow high spike or a small 1.5-times ripple is not evidence of
the broad, separated chromatic regions sought here. Choose contrast/width from the measured
baseline and chromatic leverage, not a copied figure or one universal numeric prescription.
Zero core dispersion requires an explicit numerical denominator floor for the ratio; never
accept an enormous ratio caused solely by a zero/near-zero denominator and tiny outer peaks.

Adopt and disclose candidate-specific shape thresholds before ranking the final candidates:
minimum outer peak heights, minimum contrast, maximum core dispersion and H, peak-slope and
sampling tolerances, and ID Dx/Dpx tolerances. These are configurable design requirements or
provisional hypotheses, NOT universal constants. Anchor them in the measured conventional
baseline, the requested contrast and eventual chromatic leverage. Require actual increases
in both outer bumps, not merely a smaller core denominator. No fixed 40-mm bump, 25-mm core,
or remembered tune is a universal H-MBA condition. Initially keep |Dx_ID| <= 1e-4 m and
|Dpx_ID| <= 1e-4 unless the scientist has specified tighter tolerances.

Use mba.hybrid_mba_bend_roles(cell_rows) for the physical role map. For repeated objective
evaluations, create api.build_jutrack_profile_sampler(candidate, max_step_m=...) once and use
sampler.linear_metrics(proposed_rows) for the current dense JuTrack profile and sampling
metadata. Pass the physical rows, that profile, declared thresholds and sampling metadata to
mba.hybrid_mba_shape_diagnostics. Read the module's API help; do not replace this
diagnostic with a Boolean invented from emittance or bend-exit dispersion. An incomplete or
insufficient profile is missing evidence, not permission to pass the gate.

Check `linear['stable']` BEFORE reading its profile. An unstable trial intentionally returns
an empty profile with `failure`, `trace_x` and `trace_y`; record that as a rejected PHYSICAL
proposal, not as an API/schema problem. Do not change family labels, ring scope, or runtime
configuration when the baseline passed and only a perturbed proposal lost stability. A
programming error is different: preserve its traceback and fix the actual call site.

For the final full_hybrid gate declare: min_bump_m, min_contrast, max_core_dispersion_m,
max_core_h_m, min_usable_drift_m, min_total_usable_drift_m, max_drift_slope, max_beta_m,
peak_slope_tolerance, id_dispersion_tolerance_m, id_slope_tolerance and
max_sample_step_m. min_contrast must exceed 1. Both sides must meet the contrast and peak
requirements individually even though the mean contrast is useful for presentation. Keep
the sampling step fine enough to test the declared peak-slope tolerance, and refine if needed.

### H3 — Optimize shape first by replacing the outer DBA-like sections

Use mba.HybridMBAOuterReplacement with explicit physical core_start_name/core_end_name.
Do not infer the interface from names alone: inspect source rows and include complete core
unit optics on both sides of B3..B(M-2). The class checks global symmetry and exactly M-4
preserved bends and derives each outer section's exact length/angle budget.

Design ONE outward section from core exit to ID centre, then reflect it across the WHOLE
cell. Its two bends are B[M-1] and BM; the reflected pair is B1/B2. New suppressor quadrupoles
connect the core boundary to the inner DBA bend; independently designed inter-bend optics
generate the bump; terminal quadrupoles close the achromat at the ID. Old matching strengths,
old per-unit symmetry and old terminal drift allocations are not fixed design constraints.

The module provides configurable unsolved outer connectivities via mba.build_outer_dba_section.
quadrupoles_per_region controls suppressor/bump/terminal counts: (2,2,2) is a minimal hypothesis;
(2,4,2) or (3,4,3) provides additional matching freedom around a central installation drift.
All lengths, angle split and initial strengths are explicit inputs.
It contains no validated seed or facility lattice. Other outer connectivities are legitimate.
Conserve the exact total angle and exact cell length; never spend the required ID straight.
Dipole K1/PolynomB, reverse bends and LGBs remain unauthorized.

Start with JuTrack one-pass propagation from the GLOBAL reflection plane through a diagnostic
core half and the new outer section. Vary centre betax/betay/dx inside explicit finite bounds;
alphax/alphay/dpx are zero by reflection symmetry, not copied from the old cell. Match the new
line's ID alpha_x/alpha_y/Dx/Dpx and a candidate inter-bend bump station's Dx/Dpx.
mba.match_outer_dba_section implements this six-residual, at-least-six-strength problem without
requiring a stable guessed periodic cell. It does NOT match a tune or minimize emittance.
A matched stationary station could be a minimum: only the final dense profile decides.
Prefer `mba.explore_hybrid_mba` for bounded numerical continuation and ordered geometry release;
read `references/hybrid_continuation_api.md`. It does not build conventional seeds or choose
the scientific contract. Supply explicit unsolved connectivity, limits, thresholds and bounds.
For custom one-pass trials use core_prefix=replacement.right_core_half(), center_optics_bounds
with betax/betay/dx, and **rematch_core_quadrupoles=True**. input_twiss is a centre INITIAL GUESS,
not a fixed core-exit target. Starts contain K1 for outer quadrupoles THEN half-core quadrupoles,
optionally followed by betax/betay/dx. The legacy default False exists only for reproducibility.
Export the returned half-core K1 with `replacement.core_from_half(fit['core_prefix'])` and pass
that unsplit core to `replacement.assemble(fit['elements'], core_elements=core)`. It restores
GLOBAL mirrored partners, not old within-unit symmetry. Core bend fields/geometry stay unchanged;
core quadrupoles may rematch. Provenance labels are not physical core-preservation constraints.
replacement.measure_reference(parent) is useful for diagnostics, not mandatory acceptance.
Its public profile keys beta_x/beta_y differ from the JuTrack keys betax/betay.
Target an intentional inter-bend installation drift or DBA_R:BUMP marker between drifts.
Dx is high and Dpx is small there; a plateau is valid. Diagnostic prefixes never split the
exported magnet. When matching fails, compare new outer lengths, angle allocation, and
connectivities, including more independent outer quadrupoles. Centre optics remain free;
there is no need to ask permission merely to change source beta or dispersion. Only when
the scientist explicitly requires optical preservation use the fixed measured core-exit
route (omit core_prefix/center_optics_bounds) and preserve_core_optics=True at acceptance.
Failed local starts are not topology exhaustion.

Use a modest grouped geometry scout with ALL global quadrupole pairs exclusively owned by
the inner line matcher, INCLUDING those in the core. Start with free strengths/centre optics.
When continuation stalls, release explicitly bounded bump/suppressor drift allocation first,
outer bend-angle distribution second, then grouped outer quadrupole/bend lengths. Previously
released groups stay available; the exact total length and bend angle never change. A length
increase is funded from legal outer drift slack, never the required insertion straight or by
silently enlarging the ring. The helper screens unfunded geometry before calling JuTrack.
Use independent resulting-core dispersion and beta envelopes for line guidance; do not impose
old beta/Dx, seed tunes or initial H/I5 ceilings. Dense periodic gates decide actual success.
Choose explicit continuation heights and, if useful, intermediate contrasts via
continuation_contrasts; the last step must reach the final shape contract. Do not force
core_Dx < initial_bump/final_contrast: smaller bumps need not be easier under that hidden
constraint. Do not repeat achromat-only initialization on a warm shape, undoing previous work.
Structural and matching
coordinates must be DISJOINT. B2–B3 suppressor-region geometry is rebuilt outside the declared
preserved core boundary. Generic (MO)PSO remains available for larger campaigns, but pilot the
callback first and collect a feasible initial population before production. A line match is
a seed hypothesis; the assembled cell must pass independent periodic JuTrack evaluation.
Save compact trial metrics and warm-state checkpoints using the helper callbacks. Resume the
same numerical campaign rather than asking the LLM to recode each failed trial. Treat coding
exceptions separately from measured infeasibility. Budget exhaustion is NOT physical impossibility
or topology saturation; inspect failed gates, active field limits and explored geometry before
choosing new bounds, connectivity or a larger generic optimization campaign.
No as-yet-unreached FINAL hybrid-shape target or low-core-H aspiration should prevent the
one-pass feasibility scout from exploring.

First use replacement.evaluate(..., stage="bump_feasibility") with explicit drift/contrast,
core-dispersion and beta-envelope requirements plus map-checked dense sampling. These bound
the RESULTING shape, not equality to the conventional lattice. Source-optics preservation
is opt-in with preserve_core_optics=True and a measured reference. Further core-H/I5 reduction
is deferred. Emittance
is not the dominant objective: if a scientist sets
an engineering emittance bound it is a CONSTRAINT or SECONDARY objective, otherwise report
it diagnostically. Positive damping is never optional. Use report['design_status']; never
hard-code a success title when report['passed'] is false. Publish failed checks and residuals.
Read radiation dictionaries with ev.require_radiation()['emittance_pm'], not getattr.
Publish before/after core optics and core-H/I5/emittance, plus the usable-drift inventory.

After bump feasibility, start SEPARATE emittance-recovery optimization and establish useful
core quality while preserving the verified bumps and achromat. Do not automatically impose
the original conventional core-H/I5 as hard ceilings: adopt realistic quality requirements
for this later stage and disclose them. This is where optional regional/core-aware refinement
tools can be used. The final full_hybrid gate below still requires a low-dispersion/low-H core.

### H4 — Hard hybrid-shape gate, then separate recovery stages

For an H-MBA formation request, request_satisfied may be true ONLY after the current exported
complete-cell candidate passes the hybrid shape diagnostic AND the independent stability,
symmetry, length/angle, engineering, positive-damping and achromat checks. Require two resolved
outer INTERIOR bumps and a low-dispersion/low-H central core. A stable low-emittance lattice
with a single central dispersion hill is NOT an H-MBA, regardless of its name or plot title.
Report shape failure as incomplete work with measured missing conditions; do not silently
convert the request to "find a lower-emittance conventional MBA".

Only an explicitly requested preservation contract adds a beta_x/beta_y/alpha_x/alpha_y/Dx/Dpx
comparison at all reference core samples. Parent identity alone does not require it.
The optional evaluator objective accepts core_limits=None (default), or a declared subset of
Dx/H/I5_core ceilings for later quality protection. Do not copy baseline H/I5 or emittance
as hidden hard ceilings into the initial bump-feasibility search. Archive any explicitly
adopted limits and the per-bend JuTrack decomposition with the final shape report.

Publish a real baseline-versus-candidate shape table, including each bump's s, Dx and Dpx,
core Dx/H, contrast, ID Dx/Dpx, beta maxima and emittance. Show one combined one-cell
lattice/beta/dispersion plot with B1..BM roles and marked peak/core regions, plus per-bend
I2/I4/I5 contributions. Archive the shape diagnostic with the EXACT exported candidate and
parent identity. Calculation validity alone does not satisfy the requested architecture.

For final evidence, call the full-candidate gate, not only the inexpensive objective diagnostic:

```python
mba = api.expert_module("laboratory.mba_design")
shape_report = replacement.evaluate(
    candidate, thresholds=shape_thresholds, stage="full_hybrid",
    limits=magnet_limits, max_step_m=profile_step,
    minimum_magnet_length_m=min_magnet_length,
    minimum_drift_length_m=min_drift_length,
)
# Preserve this report in the SAME finish call that archives this exact candidate:
# run.finish(..., candidate_lattice=candidate,
#            acceptance={"hybrid_mba_shape": shape_report})
```

The final report's schema is hybrid_mba_shape/v2. Its candidate-bound JuTrack profile and
independent construction checks are the evidence; naming a lattice "H7BA", reporting a
successful numerical optimizer, or borrowing an ancestor's report cannot satisfy the gate.

After the first-stage bump gate, recover core quality and emittance while retaining the
verified bump conditions as hard constraints. Once low-H quality is established, do not
trade it away silently. Recompute the appropriate stage gate on every selected recovery
candidate; the earlier parent's successful gate cannot validate its modified descendant.

Then, as a separate chromatic stage, identify or insert SF/SD at useful high beta*dispersion
sites, compute the chromatic response and match the requested chromaticities within strength
limits. Compare required sextupole strengths with the conventional baseline to test whether
hybridization helped. Extra harmonic families support later JuTrack nonlinear acceptance
optimization; two dispersion bumps alone do not prove chromatic correction or good DA/MA.

## Communication, provenance, and limitations

Every material candidate must use run.finish(candidate_lattice=...) with the exact result and
parent identity. Select exact saved inputs by scope: after assembly, latest_candidate_lattice_file
is the complete cell, NOT the optimized unit. Load that unit from its earlier archived path in
the saved lattice inventory; similarly preserve and select the solved terminal rows. Working/
latest pointers are conveniences only when they identify the intended object. Do not recreate
either object from rounded reported strengths or substitute the complete-cell Twiss for unit input.
Explain what was evaluated, what changed, what passed/failed, and the next justified action.
Publish useful tables/plots and numerical milestones, not repeated “let me reconsider” prose.

scientific_passed describes calculation validity; request_satisfied describes the user's request.
A seed is not an MBA; an MBA is not automatically hybrid; stable linear optics does not prove DA.

This module remains draft expert knowledge. Unit validation does not establish arbitrary MBA
assembly or hybrid design reliability. references/source_claim_matrix.json records existing
literature boundaries; no online search is needed for this method. The repository's
scripts/validate_tme_method.py is an offline method experiment that searches fresh parameters,
not an agent stage or a prebuilt seed. Its output must never initialize a scientist's project.
