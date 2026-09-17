# Conventional MBA: practical initialization for project-written code

For a greenfield conventional MBA, the default physical starting point is a
fresh, measured TME/M-TME unit, repetition into a complete MBA, then rematching
of the first two units and their globally mirrored last two units. Implement these choices in short
project-local code defining experiments with general numerical tools, not a prewritten design
controller, saved strengths, facility lattice or cross-project seed.

Use the core `LatticeExperiment` for proposal → dependent matching → native
measurement → persistent selection. Its `CandidateLedger` retains exact returned
elements for each completed trial, including late improvements. Do not maintain
another candidate pool or select prematch rows. Experiment feasibility remains
distinct from final cell/ring acceptance.

An alternative topology requires an explicit scientist request/approval or a
documented branch justified by physical reasoning and measured evidence. Merely
placing M equal bends among a few quadrupoles and finding stable tunes is not
an equivalent low-emittance initialization. Neither "prototype" nor permission
to write code justifies that shortcut. State the selected hypothesis and evidence
for departing from this starting point before a broad search. The **complete
cell**, not its isolated unit, remains the design authority.

## 1. Contract and physical budget

Record energy, periods P, bends per cell M, circumference C, required straight,
fields/apertures, minimum lengths, beta/stability limits and achromat tolerances.
Use `api.magnet_strength_limits` and `api.audit_lattice_constraints`; separate
machine requirements from proposed allocations. A Python dictionary is sufficient.

Reserve the COMPLETE cell before designing its unit:

```text
Lcell = C/P; Θcell = 2π/P
(M−4)*U + 4*Louter + Lstraight = Lcell
(M−2)*θunit + 2*θend = Θcell
```

The simplest initial allocation repeats M equal-length units, U=(Lcell−Lstraight)/M.
Uniform bend angles use θend=θunit. A declared half-angle end-bend hypothesis uses
θend=θunit/2 and θunit=Θcell/(M−1). Neither is a universal optimum.
When reusing a saved unit, retain its exact angle and geometry in the central
M−4 units; calculate the remaining outer-region budget, do not silently resize
the core or shorten a required straight. Report geometry edits explicitly.
Incompatible lengths or total angles are construction failures before matching.

Use `mba.assemble_mba_outer_units` to obtain actual ordered rows and provenance,
and `mba.mba_outer_unit_knobs` to inspect global mirror ownership. Their outputs
do not claim stability or an achromat. The initial outer drift redistribution is
a hypothesis; lengths/angle distribution may subsequently be released explicitly.
Keep bends intact, preserve edge/integration settings, and check engineering
bounds on the actual rows. Never tune geometry to a display-rounded dimension.
The repeated-unit reference ring is NOT the requested complete MBA.

Declare initial length ranges before the unit search: dipoles **0.8–1.5 m**;
quadrupoles, sextupoles and octupoles **0.25–0.5 m**. These are configurable
design allocations, not universal validity limits. `optimize_tme_unit_geometry`
applies the dipole/quad ranges by default and records them in its contract.
Use `length_bounds_m` to explicitly replace them (geometry keys:
`bend_length_m`, `quadrupole_near_length_m`, `quadrupole_far_length_m`,
`d1_m`, `d2_m`, `d3_m`). `[L,L]` fixes a length; `{}` explicitly opts out of
these allocations, retaining engineering minima. Disclose any revision;
incompatible budgets require a geometry decision, not silently shorter magnets.
Store physical-kind ranges as `machine["magnet_length_bounds_m"]`, e.g.
`{"SBEND": [0.8, 1.5], "KQUAD": [0.25, 0.5], "KSEXT": [0.25, 0.5], "KOCT": [0.25, 0.5]}`,
for independent candidate acceptance, including later geometry changes.

Use `api.magnet_strength_limits(E, length_m=L, ...)` for each proposed length.
It returns both local normalized-strength and integrated-strength limits:
`theta_max=Bmax*L/Brho`, `Kn_max=n!*Bpole/(Brho*r**n)`, `KnL_max=Kn_max*L`.
Length does not change the pole-tip bound on local Kn; it changes the available
integrated kick. JuTrack scalar k2/k3 are derivatives; PolynomB coefficients
are k2/2 and k3/6. Octupole field and aperture must be explicitly supplied.

## 2. Reference physics and a real unit solve

For a small-angle homogeneous bend of physical length Lb and angle θ, at its centre:

```text
βx,TME = Lb/(2√15);  αx,TME = 0
Dx,TME = Lb θ/24;    D′x,TME = 0
εTME = Cq γ² |θ|³/(12√15 Jx),  Cq = 3.8319×10⁻¹³ m
```

Use `mba.tme_targets(energy_gev=E, bend_length_m=Lb, bend_angle_rad=theta)`;
its `.as_dict()` contains `minimum_emittance_pm` and centre reference values.
Default Jx=1 is analytical, not measured damping. Recompute when Lb or θ changes.
The actual lattice uses εx=Cq γ² I5/(Jx I2); **γ², not γ³**.
Report the actual repeated-unit emittance; compare bend-centre optics only with
a verified map-preserving sample. Do not assert that finite-strength lattices
universally cannot approach the TME conditions. Exact equality is not
a mandatory gate: a practical M-TME unit may trade emittance for focusing,
chromaticity and geometry, with the measured trade-off explained.

A physically explicit **complete eleven-element** unit is
`D3–Qfar–D2–Qnear–D1–B–D1–Qnear–D2–Qfar–D3`.
Near/far describe location, not polarity: scan **both opposite-sign assignments**.
Positive K1 focuses horizontally; that does not require Qfar to be positive.
Keep B intact, including its saved edge/fringe/integration fields. Do not halve,
duplicate or replace it to manufacture a bend-centre condition.

Default to `mba.optimize_tme_unit_geometry`. Among geometric requirements only
the total unit length U and bend angle are fixed; lengths Lb, Lnear, Lfar,
D1, D2 and D3 are free within physical minima and the engineering envelope.
The exact sum `U = Lb + 2*(Lnear + Lfar + D1 + D2 + D3)` leaves **five geometry
coordinates plus two strengths**. Do not freeze equal quadrupole lengths, gaps,
or a previously successful geometry. Preserve declared edge/fringe/integration
fields while changing bend length, and recheck its field at each proposal.

The scalable search is joint PSO over those seven independent coordinates,
with one native JuTrack evaluation per proposal—not a full two-dimensional grid
nested inside every particle. Use the configured production budget and durable
checkpoint; record feasibility, active bounds and population history. Afterwards,
rescan a few shortlisted geometries with `mba.design_tme_unit`: both opposite-sign
strength assignments, local refinement of feasible basins, resolution comparison,
then fresh native verification. This conditional strength refinement creates
new candidates; retain both the original PSO result and exact refined rows.

Both tools require `energy_gev`, `bend` (including the fixed angle),
`expected_length_m`, `reference_repetitions`, `limits`, `maximum_beta_m`,
`trace_margin_x`, `trace_margin_y`, `minimum_magnet_length_m`, and
`minimum_drift_length_m`. The geometry optimizer's optional `initial_geometry`
is a starting proposal, not frozen dimensions. Its `geometry_candidates` controls
the finalist count. The fixed-geometry helper additionally requires `d1_m`,
`d2_m`, `d3_m`, `quadrupole_near_length_m`, and `quadrupole_far_length_m`;
retain it for diagnostics and same-objective, same-limits comparisons.
Trace margins and beta envelopes require explicit values and provenance.
Disclose proposed values for scientist approval; until approved, results remain a
provisional experiment. `maximum_beta_m=None` explicitly means no beta cap.
The trace gate is **`abs(trace(M_plane)) <= 2 - trace_margin_plane`**:
`.01` permits `|Tr| <= 1.99`; `.99` permits only `|Tr| <= 1.01`. This is not
a normalized `|Tr|/2` threshold. Both tools publish the resolved limits before
calculation; never silently reinterpret a supplied margin.

Both operations minimize positive physical emittance from the actual repeated
unit, with stable finite optics, geometry/field bounds and positive damping.
There is **no unit Dx=0 constraint, implicit low-tune band, or alpha/Dpx objective**.
Symmetric-point alpha/Dpx are diagnostics. Analytical TME comparisons use a
measured bend-centre sample only after its intact-map equivalence is verified;
an unavailable centre diagnostic does not block the native emittance solve.

Use the returned `result["candidates"]` directly, not a recursive search for row
lists: these are the globally ranked, verified finalists. Each contains exact
`elements`, `periodic_cell`, `evaluation`, `boundary_twiss`, `emittance_pm` and
reference-ring evidence. Detailed per-geometry/polarity results remain in
`geometry_finalists[i]["verification"]["candidates"]`. Never deduplicate by
length/angle: all unit candidates share those invariants. A failed finite search
is not impossibility, nor a boundary plateau unrestricted optimality.

Managed call/report pattern (the JSON contains the explicit physics arguments
above; do not substitute invented settings):

```python
import json
from brilliance.research_api import open_research_run
from brilliance.lattice import LatticeIR
from expert_modules.mba_design import tools as mba

run = open_research_run()
args = json.loads((run.workspace / "unit_arguments.json").read_text(encoding="utf-8"))
result = mba.optimize_tme_unit_geometry(
    **args, checkpoint_path=run.workspace / "optimizations" / "unit_geometry.json")
if result["passed"] and result["candidates"]:
    selected = result["candidates"][0]
    run.finish(summary=f"Measured unit reference emittance: {selected['emittance_pm']:.3f} pm; MBA assembly remains pending.",
        candidate_lattice=LatticeIR.from_dict(selected["periodic_cell"]),
        candidate_evaluation=selected["evaluation"], scientific_passed=True,
        request_satisfied=False, use_as_working_lattice=False)
else:
    run.finish(summary="No verified unit in this bounded search; inspect its feasibility diagnostics.",
        scientific_passed=False, request_satisfied=False)
```

Omit `output_directory` for managed storage, or select a child of
`run.output_directory`; a sibling workspace directory is not a numerical evidence
root. `checkpoint_path` may persist elsewhere **inside this project workspace**.
Default `resume="auto"` starts fresh only when neither checkpoint nor contract
sidecar exists; otherwise it verifies the pair before resuming. Explicit `True`
requires the pair; `False` refuses overwriting it. A partial/corrupt/incompatible
pair requires inspection, not deletion. Completed compatible work restores exact
same-project receipts without rerunning PSO; different-project evidence is forbidden.
Leave optional progress/cancellation hooks unset normally: managed numerical
progress and scientist cancellation are provided independently of custom hooks.

The unit's reference ring uses 2π/θunit repetitions for 2π closure (an integer for the chosen angle hypothesis). It is **not**
the requested MBA. Show the unit search history and combined unit optics/lattice
figure; label unverified bend-centre diagnostics unavailable, not failed physics.
Archive that measured unit before transferring it. Printing `tme_targets` while
directly searching an unrelated full-cell layout does not perform this stage.

## 3. Match the existing outer two unit pairs

For a 7BA the adjustable regions are B1/B2 and B6/B7; B3/B4/B5 remain the
central units. Generalize by physical order to M bends, including even M.
Do not invent two short Q–Q–B–Q–Q terminal lines around M−2 frozen units.
Do not make the scientist debug that unrelated boundary-value problem.

Assemble first. The complete periodic cell is the matching authority.
Select quadrupoles in the first TWO units, each tied only to its global mirror.
The four quadrupoles within a source TME unit need not remain internally tied.
Core magnets stay unchanged in this operation; their optical functions are
allowed to respond to the complete cell. Do not impose the isolated unit's
incoming Twiss state, seed tunes, or redundant alpha/Dpx symmetry targets.

`mba.match_mba_outer_units` composes the general `JuTrackLatticeEditor`,
`plan_matching_starts` and `bounded_response_match`. It measures periodic Dx/Dpx
at the actual ID boundary, retains engineering/stability/beta checks and checks
shortlisted cells with dense native profiles and radiation. No replacement maps,
facility strengths or previously solved project lattice are installed.
The agent can use these same general operations for a different declared method.

The minimal call sequence is:

```python
assembly = mba.assemble_mba_outer_units(
    saved_unit_rows, energy_gev=E, periods=P, bends_per_cell=M,
    cell_length_m=Lcell, straight_length_m=Lstraight, limits=limits,
    minimum_magnet_length_m=Lmag_min, minimum_drift_length_m=Ldrift_min,
    end_bend_fraction=f_end,  # explicit angle hypothesis, consistent with source unit
)
matched = mba.match_mba_outer_units(
    assembly, energy_gev=E, periods=P, limits=limits, maximum_beta_m=beta_cap,
    minimum_magnet_length_m=Lmag_min, minimum_drift_length_m=Ldrift_min,
    trace_margin_x=margin_x, trace_margin_y=margin_y,
    dx_tolerance_m=dx_tol, dpx_tolerance=dpx_tol,
)
```

Read `matched["matched"]` before using `candidate`/`evaluation`; a failed search
returns attempts and best observations, not an accepted lattice. In a managed
program, exact matched rows and evaluation are checkpointed before returning.
Keep `assembly["unit_groups"]` for the independent check below. All numerical
inputs above come from the current contract or a disclosed design allocation;
do not copy constants from a previous project. These operations leave the unit
designer unchanged and do not run complete-cell emittance optimization.

Use `assembly["id_drift_names"]` and `assembly["id_length_m"]` directly in
the acceptance contract. The straight consists of two boundary half-drifts;
never identify them by a threshold such as length > 90% of the full straight.

A repeated unit is not automatically a complete achromat after adding a long
straight or redistributing outer geometry. Screen complete-cell stability,
then match the achromat without freezing its tunes. An unstable seed is a
screening result, not a solver success or physical impossibility.
Record native evaluation counts, residuals, active bounds and elapsed time.
If no candidate meets requirements, retain the best exact observations and
diagnose the outer geometry/angle hypothesis. Do not call thousands of identical
stagnant evaluations scientific progress. Do not loosen original requirements.

## 4. Evidence, optimization and handoff

Use `mba.mba_radiation_report(candidate, exact_evaluation,
bend_contributions=api.bend_radiation_contributions(candidate))` for the TME
comparison and outer/core I5 grouping. It reports scope and both Jx=1 and
measured-Jx normalizations. Never multiply single-bend emittance by bend count:
identical-bend multiplicity cancels in I5/I2. A unit's measured/reference ratio
does not describe the complete MBA. Physical-order outer bends are B1, B2,
B(M−1), BM; do not classify them by inherited family names.

Retain the exact post-matching rows. Evaluate the selected periodic cell with
`api.evaluate_periodic_cell`; expand those same rows with `api.cell_ring` and
independently replay with `api.evaluate`. Compare radiation and per-bend sums.
Pass the exact selected object and its evaluation to `run.finish`; never rebuild
a matched lattice from rounded tables. Generic presentation already supplies
canonical figures and tables; use Matplotlib for a new research question.

Present one stage comparison, one combined one-cell optics/lattice figure, and
per-bend θ, L, ⟨H⟩, I2/I4/I5 and I5 fractions. H=γx Dx²+2αx Dx D′x+βx D′x²;
I5,j=∫H|h|³ds. Report beta and natural chromaticity alongside emittance.
Large terminal I5 motivates matching/angle/geometry revision; poor core I5
motivates core retuning. Release geometry and symmetric angle distribution in
complete-cell optimization; assign outer knobs and inner matching disjoint ownership.
No seed tune constraint. Require plateau, restarts and constraint evidence before
claiming exhaustion; neither extra generations nor hybridization is automatic.

Use `request_satisfied=True` only for the actual requested deliverable.
Chromatic correction, nonlinear acceptance and robustness are separate stages.
This route authorizes only pure positive homogeneous bends, not dipole gradients,
reverse bends or LGBs. A valid bare conventional cell is not an operational HMBA.

## Independent prototype check

Keep the solved `unit` and the first `assembled_candidate` as exact saved objects.
While assembling, retain `unit_groups`: M lists of the real assembly element
names, each listing one complete unit in physical order. Do not reconstruct
that mapping from a plot or rounded table. The final candidate may subsequently
rematch scalar fields; unit transfer is checked against the original assembly,
not imposed forever on the optimized cell.

```python
proof = mba.verify_conventional_candidate(
    candidate, unit=unit, assembled_candidate=assembled_candidate,
    unit_groups=unit_groups, machine_contract=machine,
)
# Use the exact candidate's already computed high-level evaluation:
run.finish(summary="Measured conventional MBA prototype",
           candidate_lattice=candidate, candidate_evaluation=candidate_evaluation,
           acceptance={"conventional_mba_linear": proof},
           scientific_passed=proof["passed"], request_satisfied=proof["passed"])
```

This measures supplied objects; it never builds a unit or runs an optimizer.
This verifier supports repeated-unit outer-pair construction; archived legacy
unit-and-terminal evidence remains readable, not a new-design instruction. A separately authorized alternative needs corresponding measured
evidence and an applicable validator before claiming request completion; do not
fabricate a unit-transfer history to satisfy this check.
`machine` contains `energy_gev`, `periods`, `expected_cell_length_m`,
`expected_cell_angle_rad`, `expected_bends`, `minimum_magnet_length_m`,
`minimum_drift_length_m`, `limits` from `api.magnet_strength_limits`,
`id_drift_names`, `id_length_m`, `achromat_dx_tolerance_m`,
`achromat_dpx_tolerance`, and nonempty `limit_provenance`. Include
`maximum_beta_m`, `maximum_emittance_pm`, `maximum_energy_spread`, using `None`
for unspecified hard caps; preserve `id_beta_x_bounds_m`/`id_beta_y_bounds_m`
only when specified. Optional `profile_step_m` controls diagnostic sampling.
There are no hybrid-shape targets in this conventional check. Original limits
and proposed allocations must remain distinguishable in the provenance.

Publish unit/assembly/final emittance, verified centre diagnostics when available,
and limitations. A call to `tme_targets`, a bend count or a self-written `passed=True`
cannot substitute for the measured unit and its transfer. An incomplete unit or
assembly stage can still be shown, with the complete-MBA request left pending.
