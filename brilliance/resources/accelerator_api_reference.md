# Brilliance accelerator API quick reference

Read this file instead of inspecting Brilliance or JuTrack source code. All calls run in the
configured Python/Julia/JuTrack environment. SI units are used at the Python boundary.

## Supported Python toolbox

`numpy`, `scipy`, `matplotlib` (Agg), `pandas`, `sympy`, `h5py`, and `joblib` are installed and
validated by `scripts/setup_env.py`. Use SciPy directly for least squares, root finding,
integration, sparse algebra, interpolation, and local/global optimization. Do not implement a
replacement numerical optimizer merely to avoid an import, and never run `pip` from a research
program.

## Installed expert capabilities

The core supplies general execution, lattice editing, JuTrack evaluation, plotting and
optimization. Topic-specific Python functions and acceptance rules come from enabled expert
modules. Use the module ID in the installed catalog:

```python
expert = api.expert_module(module_id)
# Call the declared functions on expert using their exact installed signatures.
```

Resolve this namespace once per program/worker, not per optimizer particle. Ask `api_help`
for the declared function name to retrieve its signature and documentation. Module knowledge
defines when to use it; the core does not prescribe a scientific workflow. Existing archived
`api.<name>` calls remain available only when their owning module declares that compatibility
alias and is installed/enabled. New programs should use the explicit module namespace.

An expert may require a named numerical report. Publish it together with the exact evidence:
`run.finish(..., acceptance={check_id: report}, candidate_lattice=candidate)`. Non-lattice studies
can omit the candidate if their validator allows that. Brilliance invokes the module's validator;
an unknown/missing validator or a failed check cannot establish request success or promote a
working candidate. The module owns the physics criteria; the application preserves evidence
and the active requirements across continuation and managed jobs.

## Composable experiments and exact candidate selection

For composed lattice searches use `ExperimentDefinition`, `LatticeExperiment`
and `CandidateLedger`. These are numerical operations, not an MBA workflow.
The expert/project defines allowed structures and physics measurements; Python
extensions remain supported. Do not rewrite candidate retention or selection.

- `ExperimentDefinition(name, scope, objectives, constraints, provenance)` maps
  objectives to `min`/`max`. Hard numeric constraints use `lower` and/or `upper`
  with positive residual `scale`; boolean constraints use `equals`. Separate
  requirements from proposed choices. Objectives are not implicit hard bounds.
- `prepare_length_budget(parameters, total_length_m=...)` derives fixed length
  from actual rows, counts each physical target (including mirror partners), and
  returns `parameterization`, `initial_position`, `budget`. Bounded exact-sum
  projection rejects impossible budgets before simulation. Never constrain the
  variable drifts alone to the total length of a line containing fixed magnets.
- `MatchingOperation(parameters, solve, contract)` calls `solve(rows, parameters)`
  using current proposal coordinates. Use the native line or response matcher.
  Outer/matching field ownership cannot overlap. Only returned rows are measured;
  solver termination is not matching success.
- `JuTrackCellEvaluator(...)` reuses a native editor for scalar changes and
  rebuilds on topology/constructor changes. Flat metrics include `stable`,
  `geometry_valid`, `radiation_valid`, `emittance_pm`, `Dx_m`, `Dpx`,
  `maximum_beta_m`, `I2`, `I4`, `I5`. Boundary Dx is not a bump metric; beta maximum
  is sampled. Required arguments: energy_gev, periods, expected_length_m,
  expected_angle_rad, limits, minimum_magnet_length_m, minimum_drift_length_m.
- `LatticeExperiment.evaluate(position)` applies proposal → optional dependent
  match → measurement → durable exact candidate record before returning the
  PSO/MOPSO callback result. Software exceptions are not physical infeasibility.

```python
# parameters and args describe this project's chosen experiment, not a saved solution.
prepared = api.prepare_length_budget(parameters, total_length_m=cell_length_m)
measure = api.JuTrackCellEvaluator(**args["native_evaluation"])
definition = api.ExperimentDefinition(**args["experiment"])
study = api.LatticeExperiment(
    definition, prepared["parameterization"], measure=measure,
    directory=run.workspace / "experiments" / "cell-search",
)
optimizer = api.particle_swarm_optimize(
    study.evaluate, bounds=study.parameterization.bounds,
    initial_position=study.parameterization.initial_position,
    variable_names=study.parameterization.variable_names,
    objective_names=list(definition.objectives),
    **run.optimization_options("cell-search"),
)
selected = study.ledger.select()["selected"]  # all trials, including late improvements
if selected is not None:  # no feasible result means None, not an invalid fallback
    exact_rows = selected["candidate"]["elements"]
    # Independently evaluate this exact cell AND explicit ring, checkpoint both
    # original receipts, then use run.reporting/run.finish as documented below.
```

Experiment checks are not native receipts or request acceptance. Final promotion
still uses exact-candidate and expert acceptance gates. Use
`study.comparison_table(limit=12)` for a bounded comparison, then one-cell combined
lattice/optics and per-bend radiation plots of the selected evidence.

`remove_mirrored_quadrupole_pair(rows, element_names=[left_name, right_name])`
verifies global physical symmetry and replaces only those magnets by equal-length
drifts, preserving names, spans and bends. It does not match or certify physics.
Recompile knobs/matching for changed topology and evaluate the hypothesis with
`study.evaluate_rows(changed_rows, label=...)` or a separately contracted branch.
Failure leaves earlier candidates intact; removing mandatory components or relaxing
machine requirements is not simplification.

The transactional archive is `experiments/<name>/candidates.sqlite3`.
`api.CandidateLedger.open(directory)` recovers exact entries/selection read-only
without importing the original callback or rerunning physics. Standalone calls
supply `workspace=...`; managed calls bind this project. Changed contracts require
a new directory. Put callback settings in provenance or `measure.contract`;
hidden mutable settings are not reproducible. This archive never supplies another
project's design seed.

## Saved program

In the provider-native coding loop, use `write_program` and pass JSON input directly as the
`arguments` object of `run_program`; there is no separate argument-writing or shell tool. In a
Codex-MCP session, the equivalent operations are `brilliance_workspace.write_research_program` and
`brilliance_workspace.write_research_arguments`. Both routes write atomically below
`workspace/programs/`. A first program should implement one bounded scientific route stage. The
source preflight rejects programs over 600 lines or 128 kB so an entire campaign cannot become one
fragile, unreviewable process.

Use the native `read_artifact` tool for saved evidence and diagnostics without starting a worker.
`path=""` lists workspace entries; `path="optimizations/campaign/checkpoint.json"` with
`pointer="/global_best/variables"` reads exact selected values. JSON pointers, directory entries,
and text lines have bounded pagination (`offset=0, limit=40`). A `summarized=true` response contains
descriptors, not full numerical values: select a narrower pointer. A completed optimizer checkpoint
can be recovered after a reporting error; do not repeat a completed search just to inspect it.

```python
from brilliance.research_api import open_research_run
from brilliance import accelerator_api as api

run = open_research_run()
args = run.arguments
candidate = api.load_lattice(run.workspace / args["candidate_path"])
# This example evaluates an explicitly stored ring. Use evaluate_periodic_cell
# for a single declared cell; the reporting reader below is the same.
evaluation = api.evaluate(candidate, include_profile=True, include_radiation=True)
checkpoint = run.checkpoint_candidate(candidate, evaluation)
# All numerical work above is durable BEFORE optional presentation below.
with run.reporting(checkpoint=checkpoint):
    view = api.evaluation_snapshot(evaluation)
    run.finish(
        summary="The saved candidate was evaluated; measured checks are attached.",
        data={"evaluation": view},
        tables=[{"title": "Evaluation checks", "rows": [
            {"Check": key, "Outcome": value} for key, value in view["checks"].items()
        ]}],
        candidate_lattice=candidate, candidate_evaluation=evaluation,
        use_as_working_lattice=False, request_satisfied=False,
        limitations=["Baseline calculation checks do not establish all machine requirements."],
    )
```

Every stage that constructs, matches, optimizes, or edits a lattice must pass its exact resulting
`LatticeIR` as `candidate_lattice=...`, including feasible and rejected hypotheses. Brilliance
archives it automatically. Choose the saved input by object scope and identity. Use
`run.working_lattice_file` or `run.latest_candidate_lattice_file` only when it is that object;
otherwise use `api.load_lattice(run.workspace / exact_archived_path)` from the saved lattice
inventory. The latest assembled ring is not its earlier unit or matching section. Never
reconstruct a successfully matched parent from old constants. Use `parent_lattice_id=parent.lattice_id` when wrapping a modified element sequence in
`api.lattice`, `api.periodic_cell`, or `api.cell_ring`.

Bind every candidate claim to its actual computed snapshot. The native line matcher's
`numerical_checkpoint` preserves exact returned elements, incoming/outgoing boundary state and
checks before its final progress event. Construct the final candidate from **those returned
elements**, then evaluate with `api.evaluate` (explicit ring) or `api.evaluate_periodic_cell`
(one periodic cell). Immediately call `run.checkpoint_candidate(candidate, evaluation, name=...)`.
This flushes the exact candidate and hash-bound native receipt to disk without rerunning JuTrack.
The evaluation cache alone is not durable: checkpoint the cell before computing the full ring,
and checkpoint the full ring before formatting any tables or plots. Pass the original return as
`candidate_evaluation=` to `finish`; the normalized reporting view is not a replacement receipt.
The original proposal and a rebuilt prematch array are not the matched candidate.
Missing/mismatched verification leaves an unverified archived checkpoint with a specific next
action, never silently promotes a working lattice. Only if an older failed program did not save
its evaluation receipt, evaluate the saved final rows once; do not repeat matching or optimization.
Non-lattice studies do not need a lattice receipt.

Use `with run.reporting(checkpoint=checkpoint):` only around optional formatting, plotting and
`finish`, never around new physics calculations. A reporting exception publishes a normal partial
app result from the verified checkpoint, preserves the traceback, and stops that program without
rematching or accepting the design. Cancellation and scientific rejection still propagate.
`brilliance.numerical_checkpoints.read_numerical_checkpoint(descriptor, output_directory=...)`
verifies immutable files for disk-only recovery. In a new worker, set
`source_output_directory=run.workspace / 'artifacts/programs/EXACT_PRIOR_RUN'` to read
an earlier program's checkpoint in this project; keep `output_directory` current.
Use `api.restore_candidate_evaluation(candidate, saved_receipt, source_output_directory=...)`
to hydrate its original JuTrack evaluation before checkpointing or publishing it in the new run.
This verifies hashes and exact candidate identity, retains provenance, and never runs the
simulation. Another project's results, unverified metrics and arbitrary source folders are rejected.
Its `result` holds the exact original data; `checks_passed` is not full-request acceptance.

Scope is explicit: `api.lattice(periodicity=P, elements=rows)` does **not** repeat `rows`.
Here P counts periods already present in the stored sequence. For a single cell use
`api.periodic_cell(elements=cell_rows, intended_repetitions=P, ...)`; this stores one cell with
`periodicity=1`. For an expanded ring use `api.cell_ring(cell=cell_rows, periods=P, ...)`.
Both constructors own their structural metadata even when other metadata is copied from a parent.
`api.one_period_elements` checks that the complete sequence can actually be partitioned as declared;
it never guesses from the title, names, bend angle, or a hoped-for repetition count. Old malformed
artifacts remain readable through `load_lattice` for diagnosis, but require an explicit corrected
revision based on their saved construction source before use. Do not modify the archived original.

`run.finish` saves the final result but does **not** exit Python. At a top-level early failure
branch, follow it with `raise SystemExit(0)`; inside a stage function, return immediately.
Never fall through into success-only calculations or a second finish that overwrites the
failure. Use `run.progress` for intermediate results, not repeated final commits.

All other artifacts must be created with `run.artifact_path(...)`. For long loops,
call `run.cancellation_requested()` at safe boundaries and publish replaceable progress. Every
material stage must call `run.progress(...)` with a scientist-readable detail; it appears live in
the chat and remains in the conversation history. Repeated calls with the same stage update one
message; a new stage creates a new message. Publish intermediate evidence as soon as it exists:

```python
run.progress(
    "Compared complete-cell seed hypotheses",
    kind="finding",
    scientific_role="milestone",
    stage_id="complete-cell-feasibility", hypothesis_id="matching-B",
    candidate_id=candidate.lattice_id, parent_candidate_id=parent.lattice_id,
    outcome="feasible",
    detail="The revised candidate improved the measured objective while preserving the constraints.",
    metrics={"best_emittance_pm": 184.2, "feasible_hypotheses": 2},
    tables=[{"title": "Seed comparison", "rows": comparison_rows}],
    plots=[{"path": comparison_plot, "title": "One-cell seed comparison"}],
)
```

Scientific milestones should state the hypothesis, measured result, comparison with its parent,
and next decision. Stage names belong to the selected expert method, not the application.
`kind` controls display: `plan`, `progress`, `finding`, `decision`, `repair`, or `next`.
`kind="activity"` and `kind="milestone"` are accepted aliases for `progress` and `finding`.
Other labels become neutral progress with a recorded warning; labels never establish acceptance.
Use `scientific_role="activity"` for execution starts, retries, generation counters and API details;
these remain accessible in the activity view without burying the scientific story. The optional
`interpretation` and `next_decision` fields on `progress` are also supported. A final
`scientific_stage` carries these same identities and explanations; candidate identity is taken
from the actual archived lattice. An infeasible attempt is not labelled a successful result.
For `finish`, use `status="partial"` for unfinished work (`"intermediate"` is an alias).
Partial results leave the request pending even if their individual calculations are valid.

Progress plots must already exist below the program output directory. `run.finish(...)` publishes
its final summary, real tables, and declared plots directly into the conversation. Raw source,
arguments, stdout, and stderr remain archived and are not shown as the answer. Do not write
API/schema probes: the exact contracts below are authoritative.

In managed programs, high-level JuTrack evaluations retain a bounded cache of already-computed
presentation evidence. When `run.finish(candidate_lattice=...)` commits that exact candidate,
Brilliance automatically shows its available one-cell beta/dispersion/lattice figure and a
per-bend table/figure if `api.bend_radiation_contributions` was evaluated and passed consistency.
This does not run JuTrack again. `api.evaluate` retains its internally computed profile even for
a compact `include_profile=False` return; fast `evaluate_periodic_cell` must actually have
computed a profile (`include_profile=True`) to supply one. Explicit plots are respected; use
`kind="bend_radiation"` on an explicitly reformatted contribution table to prevent duplication.
`data.presentation` records provenance or explains missing data. Missing optics or contributions
are never fabricated. Supply the missing calculation only if it belongs to the current request.

For a design or optimization result, also put explicit goal semantics in `data`, for example
`design_status="FEASIBLE_SEED"`, `aspiration_met=False`, and a relevant reference ratio. A true
`scientific_passed` says that the stated calculation and its declared physical gates are valid. It
is independent of `request_satisfied`: set the latter false for contracts, diagnostics, seed
searches, and other intermediate evidence, and true only when the current scientist request is
actually answered. A valid calculation is not necessarily a successful design.
must never be paraphrased as “the design succeeded” when the requested performance was missed.

## Lattice construction and evaluation

```python
d = api.drift("D1", 0.2)
q = api.quadrupole("QF", 0.3, 2.1, family="QF", design_role="structural_quadrupole")
b = api.sector_bend("B1", 0.8, 0.03, family="B1")
s = api.sextupole("SF", 0.2, 120.0, family="SF")

# Constructors return plain dictionaries, not objects with element attributes.
# Their exact `kind` values are DRIFT, KQUAD, SBEND, KSEXT, and MARKER.
# A physical bend is most simply identified by a finite nonzero `angle_rad`.
total_length = sum(element["length_m"] for element in [d, q, b, s])
total_angle = sum(float(element.get("angle_rad") or 0.0) for element in [d, q, b, s])
names = [element["name"] for element in [d, q, b, s]]

# Map a dimensionless seed to an exact machine length while holding selected
# elements (for example insertion-straight halves) fixed.
elements = api.scale_element_lengths_to_total(
    [d, q, b, s],
    target_total_length_m=2.0,
    fixed_element_names=("D1",),
)
# Repeated section literals must become distinct physical instances. This
# changes duplicate instance names only; family labels and physics are kept.
elements = api.uniquify_element_names(elements)
geometry = api.lattice_geometry_summary(elements)  # also accepts a LatticeIR or saved path
total_length_m = geometry["total_length_m"]
total_angle_rad = geometry["total_angle_rad"]
# Geometry-summary keys differ from audit_lattice_constraints' length_m/angle_rad.
# Field-derived limits for an ultra-relativistic unit-charge beam. Override
# pole-tip fields and radii only when the scientist supplies the technology.
limits = api.magnet_strength_limits(energy_gev=3.5)
k1_max = limits["max_abs_k1_m2"]

# Generic reflection: left_half is ordered boundary -> symmetry plane. Angles
# and families are preserved; reflected physical instances get unique names.
right_half = api.mirror_elements(left_half, name_prefix="R:")
odd_symmetric_cell = left_half + [central_element] + right_half

cell = api.periodic_cell(
    title="one cell", energy_gev=3.0, elements=[d, q, b],
    source="workspace/programs/my_study.py", intended_repetitions=32,
)
ring = api.cell_ring(
    title="full ring", energy_gev=3.0, cell=[d, q, b], periods=32,
    source="workspace/programs/my_study.py",
    design_constraints=[{"quantity": "circumference_m", "target": 800.0}],
)
api.save_lattice(path, ring)
ring = api.load_lattice(path)
result = api.evaluate(
    ring, include_radiation=True, include_profile=True, include_chromaticity=True,
)
snapshot = api.evaluation_snapshot(result)  # same reader for evaluate_periodic_cell results
if not snapshot["checks"]["linear_stable"]:
    # Treat this as physics/numerical evidence and change the candidate.
    ...
actual_circumference_m = snapshot["geometry"]["ring_circumference_m"]
radiation = snapshot["radiation"]
if radiation["computed"] and snapshot["checks"]["radiation_physical_validity"]:
    emittance_pm = radiation["emittance_pm"]
# Keep result itself for finish(candidate_evaluation=result), not snapshot.
```

`api.evaluate` is the final authority for periodic optics and radiation. Radiation claims require a
stable, closed full ring. `api.natural_chromaticity(...)` and `api.dynamic_aperture(...)` are direct fast paths.
Never import `pyJuTrack` or call its element constructors directly in a research program. Use
`api.jutrack()` only when a documented JuTrack operation is absent from this API.

## Reading an actual evaluation

Use **`api.evaluation_snapshot(evaluation)`** for both `api.evaluate(...)`'s typed result and
`api.evaluate_periodic_cell(...)`'s dictionary. It also reads their saved JSON dictionaries for
reporting-only recovery. It never starts JuTrack, expands a ring, or supplies a hoped-for geometry.
Its fixed top-level keys are `schema`, `engine`, `candidate_lattice_id`, `scope`, `geometry`,
`linear_optics`, `radiation`, `chromaticity`, `checks`, `acceptance`, and `evaluation_receipt`.

| Section | Exact keys and meaning |
| --- | --- |
| `geometry` | `stored_length_m` sums the actual stored elements; `cell_length_m` is the common declared cell length; `ring_circumference_m` exists only for an explicitly stored full ring; `reference_ring_circumference_m` describes the periodic evaluation's identical reference ring, not a stored machine. Missing quantities are `None`. |
| `geometry` scope | `stored_scope`, `stored_periods`, `evaluated_periods`, `candidate_sha256`, `row_geometry_available`. The evaluated repetition count is the actual argument used, not a copied design target. |
| `linear_optics` | `stable`, `trace_x/y`, `tune_x/y`, entrance `beta_x_m`, `beta_y_m`, `alpha_x`, `alpha_y`, `Dx_m`, `Dpx`, `Dy_m`, `Dpy`; `maximum_beta_m` is sampled, not a certified continuous extremum. Missing values are `None`. |
| `linear_optics` profile | `twiss_profile` is a list; `profile_available` says whether it was returned. Profile rows use `beta_x`, `beta_y`, `alpha_x`, `alpha_y`, `Dx`, `Dpx`, `Dy`, `Dpy`, `name`, `s_m`. Pass these rows directly to the optics/lattice plotter. |
| `radiation` | `computed`, `reason`, `physical_validity`, `I1`–`I5`, `Jx`, `Jy`, `Jz`, `emittance_m`, `emittance_pm`, `energy_spread`, `energy_loss_per_turn_eV`. Original extra native fields remain intact. Unavailable numerical values are `None`, never fabricated zeros. |
| `chromaticity` | `computed`, `horizontal`, `vertical`, plus original native fields if calculated. |
| `checks` | `linear_stable`, `ring_bend_closure_required`, `ring_bend_closure_passed`, `passed_baseline_gate`, `radiation_computed`, `radiation_physical_validity`, `full_ring_evaluated`. Baseline means only stability and required bend closure; check radiation validity separately. |
| `acceptance` | `accepted=False`, `machine_requirements="not_assessed"`, `scientific_request="not_assessed"`, `publication_required=True`, `full_ring_validation_required`. An evaluation is not project publication or expert scientific acceptance. |

New evaluations capture `evaluation_geometry` from exact rows **before** recording the numerical
receipt. `evaluation_snapshot` is a detached reporting view, not a substitute receipt payload:
pass the original evaluation to `finish(candidate_evaluation=...)`. Preserve full precision in
calculations; format only for display. There is no `result.geometry_summary` attribute. Existing
`.geometry` and `.as_dict()` interfaces remain supported, but their legacy geometry key names
differ between full-ring and periodic evaluations; use the common reader above for new code.
Legacy results containing saved candidate rows can recover actual geometry without simulation.
Old periodic dictionaries without rows are labelled `row_geometry_available=False`; their old
geometry is retained under `legacy_reported_geometry`, not invented as a new row-bound measurement.

`api.evaluate(...)` returns a typed `CandidateEvaluationResult`. Its
`result.linear_optics` properties are `stable`, `trace_x`, `trace_y`, `tune_x`, `tune_y`,
`periodic_closure_max`, `periodic_residuals`, `twiss_profile`, and `one_turn_matrix`. Beta functions
and dispersion are available through `evaluation_snapshot` or profile rows, not
top-level `LinearOpticsResult.beta_x_m` attributes. Full-ring `tune_x` and `tune_y` are the
fractional one-turn tunes returned by JuTrack; the integer part cannot be reconstructed by dividing
them by the number of cells. For a repeated cell, `evaluate_periodic_cell` reports the per-cell
tunes and also `predicted_ring_fractional_tune_x/y = (periods * tune_cell) mod 1`, which may be
compared with the explicit full-ring gate.

`api.evaluate_periodic_cell(...)` returns the
top-level keys `engine`, `candidate_lattice_id`, `evaluation_scope`, `periods`, `geometry`,
`linear_optics`, `radiation`, and `full_ring_validation_required`. Its `linear_optics` mapping uses
`stable`, `trace_x`, `trace_y`, `tune_x`, `tune_y`, `Dx_m`, `Dpx`, `beta_x_m`, and `beta_y_m` when
available. These scalar beta/dispersion values are at the CELL ENTRANCE, never at a bend centre.
With `include_profile=True`, `linear_optics["twiss_profile"]` contains START and element-exit
rows with `name`, `s_m`, `beta_x`, `beta_y`, `alpha_x`, `alpha_y`, `Dx`, `Dpx`; it can be passed
directly to `save_optics_lattice_plot`. An unsplit bend has no sampled centre. Use
`build_jutrack_profile_sampler(...).linear_metrics(sample_fractions={bend_name: [0.5]})`
to measure its centre with native partial-element propagation; do not split the physical design.
Its `radiation` mapping always contains `computed`; when true it also contains `I1`–`I5`,
`Jx`, `Jy`, `Jz`, `emittance_m`, `emittance_pm`, `energy_spread`, and
`energy_loss_per_turn_eV`. Do not inspect runtime objects to obtain these names.

## Symmetric construction and engineering audits (any lattice)

`api.symmetric_cell_elements(left, *, center=None, name_prefix="R:")` returns
`left + [center] + mirror(left)`, or `left + mirror(left)` when center is omitted.
The centre argument is the **full** physical element with its full length and angle; left
excludes it. This is not a 7BA builder and imposes no bend count or solved magnet strengths.

`api.audit_lattice_constraints(elements, *, energy_gev, expected_length_m,
expected_angle_rad, expected_bends=None, require_mirror_symmetry=False,
minimum_magnet_length_m=0.0, minimum_drift_length_m=0.0, limits=None)` returns
`passed`, `issues`, `length_m`, `angle_rad`, `bends`, `magnet_margins`, `limits`, and `scope`.
Supply the original `api.magnet_strength_limits(energy_gev, ...)` envelope as `limits` so all
searches and diagnostic scans use the same hardware bounds. The audit checks names, length,
bend sum/count, mirror symmetry, and dipole/K1/K2 engineering limits **without starting Julia**.
Failed construction is not a candidate to simulate. Passing is not optical feasibility.

`api.bend_radiation_contributions(full_ring)` returns `computed`, `engine`, `method`,
`lattice_id`, `grouping`, `rows`, `totals`, `ringpara`, and `consistency`. Each row uses the exact
keys `bend`, `instances`, `length_m`, `angle_rad`, `mean_H_m`, `I1`, `I2`, `I3`, `I4`, `I5`,
`I2_fraction`, `I4_fraction`, `I5_fraction`, `I5_over_I2`, and optionally `source_periods`. It does
not have a `name` key. The decomposition is usable only when `consistency["passed"]` is true.

## Mutable JuTrack calculations

Do not rebuild thousands of Python elements for every trial. Build once and mutate compiled
indices:

An editor is compiled for one ordered element list: same count, names and kinds.
Its scalar lengths, angles and strengths may change. Inserting, removing, splitting,
renaming, reordering or replacing an element requires `api.build_jutrack_editor(new_candidate)`;
do not send the changed list to the old editor. `LatticeTopologyError` is a repairable
API misuse, not an unstable physical lattice. The identity check precedes all mutations.

```python
jt = api.jutrack()
editor = api.build_jutrack_editor(ring)
# Compile indices by exact names/families once with the editor/parameterization helpers.
metrics = api.jutrack_linear_metrics(jt, editor.lattice, include_profile=False)
# A matching/search callback can update rows and evaluate without rebuilding the lattice:
metrics = editor.linear_metrics(updated_rows, include_profile=False)
radiation = api.jutrack_radiation_metrics(jt, editor.lattice, energy_gev=3.0)
unit_radiation = api.jutrack_periodic_radiation_metrics(
    jt, editor.lattice, energy_gev=3.0, periods=192,
)
# Keyword is periods, not repetitions. Do not hide TypeError/AttributeError/KeyError
# in a broad except-and-return-feasible-False callback.
# If a raw lookup is genuinely needed, the exact call is keyword-only for the name and returns
# Julia-style 1-based indices: jt.findelem(editor.lattice, name="QF")
```

Both direct radiation helpers return `computed=True` when `ringpara` returns, plus
`physical_validity["passed"]`. Computation alone is not a physical pass. Test linear stability
**before** calling radiation for a trial. Do not turn an API exception into an unstable particle.
Element mappings use top-level physical fields (`length_m`, `k1_m2`, `angle_rad`) and nested
metadata: `row["fields"]["family"]`, not `row["family"]`. Prefer the family parameterization
helpers; if selecting indices manually, assert that every requested family matched at least one
element. Empty index lists silently leave a supposed optimization variable disconnected.

A minimal mutable evaluator pattern (works with your own lattice and family names; it does not
provide a design or matching target):

```python
knobs = api.compile_family_parameterization(
    elements, field="k1_m2", families=selected_families, bounds=bounds_by_family,
)
editor = api.build_jutrack_editor(candidate)

def evaluate_knobs(position):
    evaluated = editor.evaluate_candidate(
        knobs.apply(position), energy_gev=energy, periods=periods,
    )
    if not evaluated["passed"]:
        return {"objectives": {"emittance_pm": float("inf")}, "feasible": False,
                "constraint_violations": {"invalid_optics_or_radiation": 1.0}}
    rad = evaluated["radiation"]
    return {"objectives": {"emittance_pm": rad["emittance_pm"]},
            "feasible": rad["physical_validity"]["passed"]}

# Check the actual objective once before a large search. The compiled knob table
# lists every target and rejects missing families/overlapping field ownership.
baseline = evaluate_knobs(knobs.initial_position)
run.progress("Objective check", detail="Evaluated the exact initial knob vector.",
    metrics={"feasible": baseline["feasible"]},
    tables=[{"title": "Knob ownership", "rows": knobs.as_dict()["variables"]}])
# PSO receives knobs.bounds, knobs.variable_names, knobs.initial_position,
# evaluate_knobs, and **run.optimization_options(...).
```

`editor.evaluate_candidate(elements, *, energy_gev, periods, include_profile=False)`
requires an explicit proposal. It synchronizes the editor, checks stability, evaluates
radiation only if stable, and returns an independent physical `elements` snapshot alongside
`linear_optics`, `radiation` (None if unstable), and `passed`. No API exceptions are swallowed.
Use one editor per parallel worker. Do not pair cached trial strengths with radiation calculated
later on whatever happens to be in a mutable editor. Keep the returned snapshot with its metrics
and replay the selected snapshot before exporting or transferring it. This is not a substitute
for geometry/magnet constraints or the selected candidate's independent full-ring gate.

### Interior optics without changing the physical lattice

```python
sampler = api.build_jutrack_profile_sampler(selected_candidate, max_step_m=0.05)
sampled = sampler.linear_metrics(sample_fractions={bend_name: [0.5]})
# Check sampled['stable'] and sampled['sampling']['map_equivalent'] before using optics.
profile = sampled['twiss_profile']
centre = next(p for p in profile if p.get('requested_sample')
              and p['source_element'] == bend_name and p['sample_fraction'] == 0.5)
api.save_optics_lattice_plot(plot_path, profile, api.one_period_elements(selected_candidate))
```

`sample_fractions` maps physical names to fractions strictly inside (0,1). It can
request several positions per element. Physical exits, the regular grid and resolved
dispersion extrema are also retained. This is direct JuTrack partial-element propagation,
not interpolation. Requested points have `requested_sample=True`, `source_element` and
`sample_fraction`. Use the physical UNSPLIT rows for the layout and saved design.

Save the selected physical candidate before plotting. If diagnostics/plotting fail after
optimization, read `api.read_optimization_checkpoint(checkpoint_path)` and its
`best_candidate['variables']`/`['position']`, reconstruct with the original saved builder,
replay that candidate once, and complete reporting. Do not rerun the population search or
mistake a completed unit optimization for the requested complete machine.

Working-state handoff: archive components and comparison branches with
`use_as_working_lattice=False`. After checking the exact complete candidate and required
scientific gates, use `use_as_working_lattice=True, scientific_passed=True` if it is the intended
provisional starting point for subsequent research. This does not assert a finished design;
`request_satisfied` remains a separate scope decision. `complete_research` does not select a
branch or change its saved evidence. Always tell the scientist which disposition occurred.

For a periodic cell whose bend closes only after repetition, prefer the high-level fixed-schema path:

```python
seed = api.periodic_cell(
    title="seed", energy_gev=energy, elements=elements,
    source="workspace/programs/design.py", intended_repetitions=periods,
)
seed_result = api.evaluate_periodic_cell(seed, include_profile=True)
if not seed_result["radiation"]["computed"]:
    reason = seed_result["radiation"]["reason"]
```

This evaluates one cell and scales extensive integrals without expanding hundreds of identical
elements. In a mutable inner loop, use
`api.jutrack_periodic_radiation_metrics(jt, lattice, energy_gev=..., periods=...)` directly. Build an
explicit `api.cell_ring` only for the selected complete-cell full-ring gate.

## Parameterization and matching

`optimize_lattice_parameters` raises the typed `api.ResearchInfeasible` when a bounded baseline
match or all-feasible initialization attempt fails. The managed runner preserves the actual
attempt, returns `status="infeasible"`, and hands the negative evidence back to the agent. This is
not a Python failure, proof of physical impossibility, or permission to weaken a constraint. Inspect
the target residuals, geometry, and exclusive knob ownership before choosing another attempt.
Other expert tools may use the same outcome with `code` and `diagnostics`; do not catch arbitrary
exceptions and relabel them as this outcome.

Engineering envelopes distinguish local and integrated strengths. Calling
`api.magnet_strength_limits(energy, length_m=L)` also returns `max_abs_angle_rad`,
`max_abs_k1l_m1` and `max_abs_k2l_m2`. Local Kn limits depend on pole-tip field,
aperture and rigidity, not length; integrated KnL limits scale with L. For K3,
explicitly supply both `octupole_pole_tip_field_t` and `octupole_radius_m`;
the optional outputs are `max_abs_k3_m4` and (with L) `max_abs_k3l_m3`.
Use `audit_lattice_constraints(..., length_bounds_m={"SBEND": [Lmin,Lmax], ...})`
to enforce declared physical-kind length ranges on actual rows. Ranges are
task-specific allocations; core tools impose no MBA length defaults.

```python
limits = api.magnet_strength_limits(energy_gev=energy)
k1_max = limits["max_abs_k1_m2"]
design = api.infer_mirror_symmetric_parameterization(
    elements,
    fields=("length_m", "angle_rad", "k1_m2"),
    k1_bounds=(-k1_max, k1_max),
    fixed_element_names=(),  # list contractual fixed elements when needed
    excluded_roles_by_field={"k1_m2": ("matching_quadrupole",)},
    exact_sum_by_field={"length_m": cell_length, "angle_rad": cell_angle},
)
matching = api.compile_family_parameterization(
    elements, field="k1_m2", families=("QM1", "QM2", "QM3", "QM4"),
    bounds=(-k1_max, k1_max), role="in_objective_matching",
)
starts = api.search_feasible_parameter_seeds(
    elements, matching, evaluate_metrics,
    count=8, attempts=2000, seed=17,
    targets={
        "Dx_m": {"target": 0.0, "tolerance": 1.0e-4},
        "Dpx": {"target": 0.0, "tolerance": 1.0e-4},
    },
)
# A local response matrix needs a stable starting map. Try the ranked stable
# starts instead of writing an ad-hoc random loop or matching an unstable seed.
matched = api.bounded_response_match(
    elements, matching, evaluate_metrics, targets,
    initial_position=starts["seeds"][0]["position"],
    max_iterations=12, finite_difference_fraction=0.002,
)
if matched["success"]:
    matched_elements = matched["elements"]
```

`exact_sum_by_field` values are totals for the complete physical element sequence. The compiler
subtracts fixed or excluded elements automatically before projecting the tunable variables; pass
the real cell length and cell bend angle, not a hand-computed tunable-only subtotal.

There are no default engineering bounds for inferred strength knobs. Supply `k1_bounds`,
`k2_bounds`, or `k3_bounds` for each strength field actually varied; geometry-only or excluded
strength fields need none. Missing strength bounds raise a repairable `OptimizerInputError`
before any particle is evaluated. Use the SAME energy/aperture/technology envelope for the
outer design and inner matcher. A successful dispersion match alone is not full feasibility.
Your objective must also check `api.audit_lattice_constraints` with the declared field and
minimum-length limits, plus radiation physical validity; relative geometry bounds and exact
sum projection do not enforce these coupled constraints. Re-audit the selected physical rows.

For explicit per-field ownership use `included_element_names_by_field`. When supplied, ONLY
the listed fields/elements vary; include both global mirror partners by their real saved names.
For example, a strength-plus-drift-length search can use:

```python
design = api.infer_mirror_symmetric_parameterization(
    elements, fields=("length_m", "k1_m2"), k1_bounds=(-k1_max, k1_max),
    included_element_names_by_field={
        "length_m": design_drift_names,  # both partners; no quadrupole lengths
        "k1_m2": design_quadrupole_names,  # excludes inner matching families
    },
    minimum_length_m=declared_minimum_drift_length,
    exact_sum_by_field={"length_m": cell_length},
)
```

Without that inclusion map, BOTH length and strength vary on every eligible, unfixed quadrupole.
Count compiled knobs, not magnet families in prose: eight quadrupole pairs with both fields
plus eight drift pairs give 24 coordinates, not 16. Exact sum constraints project coordinates;
the population policy conservatively counts all compiled outer coordinates.

When adding bend-angle knobs to this example, include `"angle_rad": design_bend_names` in the
inclusion map AND `"angle_rad"` in `fields`. Adding only an exact angle sum does not select any
bends. Exclude both partners of a fixed straight; excluding only `ID_half` while retaining
`R:ID_half` is not a mirror-symmetric selection. Resolve partners from actual ordered rows rather
than assuming a prefix. Contradictory explicit field maps are caught by source preflight.

Every baseline value selected by a compiled parameterization must already lie inside its declared
bounds. Read exact saved values, not rounded display tables. If the exact baseline is genuinely
out of bounds, revise the proposal explicitly and re-evaluate it; do not silently clip an accepted
parent or widen its limits. The compiler rejects an impossible initial coordinate.

The callback determines the physics and residual scaling. Outer design parameters and inner
matching parameters must be disjoint. Seed tunes are reported, not matched, unless the scientist
explicitly requests a working point. `search_feasible_parameter_seeds` is a bounded, general
stable-basin search; its `success`, `attempts`, `stable_evaluations`, and ranked `seeds` must be
reported. It does not replace the subsequent achromat/physics gate.

The callback contract is exact: `evaluate_metrics(rows)` receives one `list[dict]` containing the
complete element sequence after the proposed parameter values were applied. Each row uses the same
public element keys (`kind`, `name`, `length_m`, `angle_rad`, `k1_m2`, `k2_m3`, `k3_m4`, `fields`,
and metadata). Family and design-role metadata are nested at `row["fields"]["family"]` and
`row["fields"]["design_role"]`; do not read or modify `row["family"]`. It must perform the relevant
JuTrack evaluation and return a flat mapping. The
stable-basin search accepts a trial only when that mapping contains `stable=True`; a missing or
false `stable` value is rejected. A periodic-cell matcher can therefore use:

```python
cell = api.periodic_cell(
    title="matching trial", energy_gev=energy, elements=elements,
    source="workspace/programs/study.py", intended_repetitions=periods,
)
editor = api.build_jutrack_editor(cell)

def evaluate_metrics(rows):
    return editor.linear_metrics(rows, include_profile=False)
```

This is the documented contract; do not launch a research program to probe callback argument types
or result keys. `bounded_response_match` always returns `success`, `reason`, `elements`, `position`,
`metrics`, `evaluations`, and `trace`; use `matched["elements"]` as the physically evaluated lattice.
The `position` values are in the exact order of `parameterization.knobs`. Never apply a family by
editing a guessed top-level `row["family"]` field.

SciPy is also supported for problem-specific matching:

```python
from scipy.optimize import least_squares
solution = least_squares(residual, x0, bounds=(lower, upper), x_scale="jac")
```

Evaluate the returned physical lattice, not merely the optimizer coordinates.

## Native JuTrack open-line matching

For a transfer line with a known incoming Twiss/dispersion state use
`api.JuTrackLineEvaluator` (reusable), `api.evaluate_jutrack_line` (one shot), or
`api.match_jutrack_line` (bounded multistart matching). None requires the open
line to be a stable periodic ring. All propagation is native JuTrack; analytical
reference physics must not become a substitute transport simulator.

`measurement = api.evaluate_jutrack_line(...)` returns `evaluable`, `passed`, `endpoint`,
`metrics`, `twiss_profile`, `geometry`, `engineering`, `checks`, `diagnostics` and `sampling`
(when transport succeeded). Read the eight output quantities from **`measurement["endpoint"]`**,
not `measurement.get("exit", {})`. `evaluable=True` means finite native line optics exist;
`passed=False` may still identify a finite over-beta candidate. Neither is a periodic-cell gate.
For `match_jutrack_line`, its fresh replay is in `match["measurement"]`; therefore read
`match["measurement"]["endpoint"]`. Do not interpret an absent/empty mapping as a matched boundary.

```python
# incoming contains all eight exact measured entrance fields:
# beta_x_m, beta_y_m, alpha_x, alpha_y, Dx_m, Dpx, Dy_m, Dpy.
match = api.match_jutrack_line(
    line_rows, matching_knobs, energy_gev=energy,
    initial_twiss=incoming, targets=exit_targets,
    expected_length_m=line_length, expected_angle_rad=line_angle,
    limits=limits, maximum_beta_m=declared_beta_cap,
    minimum_magnet_length_m=minimum_magnet_length,
    minimum_drift_length_m=minimum_drift_length,
)
# exit_targets maps each requested endpoint quantity to (value, absolute tolerance).
# Example: {"alpha_x": (0.0, 1e-4), "alpha_y": (0.0, 1e-4),
#           "Dx_m": (0.0, 1e-4), "Dpx": (0.0, 1e-5)}.
run.finish(summary="Measured open-line matching attempt", status="partial",
           data={"match": match},
           tables=[{"title": "Exit boundary residuals", "rows": match["residuals"]}],
           request_satisfied=False)
```

Knobs come from the general parameterization compiler. Preflight starts without
Julia using `api.plan_matching_starts(matching_knobs, limits=limits,
start_positions=starts, max_starts=8, seed=0)`. Explicit starts run first;
exact duplicates are merged, then the unchanged valid baseline and reproducible
Latin-hypercube proposals fill available slots. **`max_starts` caps the total**:
eight supplied starts fit a budget of eight, even if they include the baseline.
Overflow is preserved in `deferred_positions`, never silently dropped or added
to the workload. `budget` reports supplied, duplicate, generated and deferred
counts. With no cap, the default is `max(64, 10 * knobs, unique supplied + baseline)`.
All supplied positions (including deferred ones) must respect engineering bounds;
invalid automatic baselines appear in `skipped`, not as clipped coordinates.
The matcher returns this `start_plan`, `completed_starts`, and ordered
`remaining_start_positions`/`remaining_start_sources` for continuation; early
success does not imply every planned start was tried. Pass remaining positions
and an explicit cap to a later call to continue without expanding its budget.

`max_nfev=500` is a per-start solver budget, not a time limit or the total number
of JuTrack calls. Managed matching programs are submitted as durable jobs and automatically
publish native evaluation progress/residuals and honor cancellation, even when
the model omits callback boilerplate. Explicit callbacks are also supported. The result records
all attempt terminations, measured residual Jacobian singular values, actual
callback counts, exact returned `elements`, incoming state, and a fresh native
`measurement`. Engineering-invalid proposals are never simulated. Finite
over-beta lines remain evaluable during matching, but cannot pass the final
declared sampled-beta gate. Interior samples and physical exits use JuTrack;
sampled maxima do not certify continuous extrema. `maximum_beta_m=None`
explicitly means there is no beta limit. The current adapter covers uncoupled
hard-edge DRIFT/QUAD/KQUAD/pure SBEND/KSEXT/KOCT/MARKER without nontrivial
edges, transforms or skew fields; unsupported cases are explicit, not approximated.

Attempt diagnostics have nullable `nfev`, `njev`, `solver_status`, `final_position`
and `residual_jacobian_rank` when the local solver could not return those values;
null is not a measured zero. `initial_position` is the requested proposal;
`physical_initial_position` records the actual first trial after exact-sum
projection (or null if no physical trial was constructed).
Managed jobs save exact provisional rows and remaining starts after every local
start and at most once per 60 seconds during evaluable callbacks, before optional
reporting. Recover them with `api.read_numerical_checkpoint(...)`. Checkpoint kind
`jutrack_line_match_progress` is **not** a verified match: replay recovered rows
with JuTrack before use. The final `jutrack_line_match` checkpoint retains the
independently verified outcome. No checkpoint alone accepts a complete machine.
Continue using the saved program's original parameterization and physics contract;
this preserves numerical observations and start queues, not SciPy's internal
solver state. An interrupted in-flight start remains queued and may repeat work.

`solver_terminated` and `solver_success` are numerical diagnostics.
`match_success` requires every exit tolerance and declared line limit to pass a
fresh JuTrack replay. `complete_cell_accepted` is always false: assemble the exact
matched rows and separately validate periodic stability, achromat, radiation and
machine requirements. A failed matcher is not evidence of insufficient space.

## Geometry reporting

Use `run.progress("Assembly geometry", candidate_lattice=candidate)` to attach
an automatic table computed from the exact saved rows; this does not run JuTrack.
`run.finish(candidate_lattice=...)` also publishes these dimensions even if optics
failed. Stored length, periodic-cell length, projected repetition length and an
explicit full ring are distinct quantities, never copied from the target contract.
Use `kind="contract"` or `kind="targets"` on a requirements table. A custom
measured geometry table needs `kind="candidate_geometry"`, `lattice_id`, and
canonical metric/value rows; unbound geometry tables are retained for diagnosis,
not displayed as verified measurements. Prefer the automatic geometry table.
Numerical geometry inputs retain full precision; format only at display time.

## Population optimization

The public optimizer contract is complete; do not copy or reimplement PSO in a research program.
For optimization of an existing project lattice, load that exact lattice and compile its knobs,
mirror pairs, fields, and design roles. A baseline or diagnostic can legitimately precede the
search; it does not satisfy an optimization request by itself. Pass
`**run.optimization_options(...)` to the actual optimizer: besides durability and live progress,
it records proof that optimization started. If no legal disjoint variables can be compiled,
explain the concrete blocker instead of substituting a different lattice or claiming optimization.

```python
parent_path = run.working_lattice_file or run.latest_candidate_lattice_file
if parent_path is None:
    run.finish(
        summary="No project lattice is available to optimize.",
        status="infeasible",
        data={"optimization_blocker": "missing project lattice"},
        scientific_passed=True,
        request_satisfied=True,
    )
else:
    parent = api.load_lattice(parent_path)
    elements = api.one_period_elements(parent)
    # Compile ownership and start the optimizer below in this same program.
```

The evaluator receives one NumPy position vector and returns this mapping:

```python
{
    "objectives": {"emittance_pm": 103.2, "beta_burden": 18.4},
    "constraint_violations": {"unstable": 0.0, "field_limit": 0.0},
    "feasible": True,
    # any other compact entries are retained under best_candidate["metrics"]
}
```

Production searches default to `max(500, 10 * independent_knobs)` particles and
80 generations. Managed runs apply the scientist's `[optimization]` configuration
even if model-written code asks for less, and publish requested/effective budgets.
Only the scientist should lower those settings for a deliberately small experiment;
a callback preflight is not a reason to underfund the subsequent search. A saved
smaller swarm is resumed without resizing, labelled as a smaller prior campaign;
start a new named campaign to expand its population, retaining its best valid seeds.
Neither a large budget nor a completed budget proves convergence or feasibility.

Every violation is non-negative and zero when satisfied. A candidate is feasible only when all
objectives are finite, every violation is zero, and `feasible` is not false. Do not hide achromat
matching inside a weighted objective penalty.

```python
result = api.particle_swarm_optimize(
    evaluate,
    bounds=bounds,
    objective_names=("emittance_pm", "beta_burden"),
    initial_position=x0,
    variable_names=names,
    population_size=population,
    generations=generations,
    seed_positions=feasible_seeds,
    evaluate_batch=evaluate_batch,          # optional process-parallel batch
    **run.optimization_options("descriptive-campaign-name"),
)
```

The exact returned top-level keys are `algorithm`, `evaluated_candidates`, `variable_names`,
`bounds`, `objectives`, `population`, `pareto_candidates`, `best_candidate`, and `convergence`.
`best_candidate` contains `candidate_id`, `position`, `variables`, `objectives`, `feasible`,
`constraint_violation`, `constraint_violations`, and `metrics`. `convergence["history"]` contains one
row per generation with `generation`, `population_size`, `feasible_count`, `front_size`, `best`,
and `minimum_constraint_violation`. It also records generation-zero/final improvement,
feasible-population fractions, population per dimension, and explicitly never claims topology
saturation. Use these documented fields directly; source inspection or a home-grown PSO is not
needed.
`best_candidate["variables"]` is a **name-to-number mapping**, not an array. Each history row's
`best` is likewise an **objective-name-to-number mapping**, not a scalar: use
`row["best"]["emittance_pm"]` for a scalar emittance curve.
`best_candidate["position"]` is the exact ordered numeric vector in `result["variable_names"]`
order. Reconstruct a compiled parameterization with `selected_rows = knobs.apply(best["position"])`.
Never guess that variable keys equal bare family names: compiled names may be `k1_m2:QF`, not `QF`.
`bounds` accepts ordered numeric pairs, or a mapping such as
`{"qf": [1.0, k1_max], "qd": [-k1_max, -1.0]}`. For named bounds, `variable_names`
must contain exactly those keys; the callback vector follows that explicit order, not dictionary
insertion order. When omitted, names follow the named bounds' insertion order. `initial_position`
may likewise be a name-to-number mapping or a vector in that same order. Bound values must be
two-number pairs, not nested dictionaries. Invalid shapes are rejected before any evaluation.
Generation zero is initialization: 31 history rows ending at generation 30 mean 30 search
generations, not 31. Disclose a partly infeasible initial population as such.

For a saved research program, do not hand-build the durable controls. Use
`control = run.optimization_options("descriptive-campaign-name")` and pass `**control` to either
optimizer. It publishes candidate/generation progress, checks scientist cancellation between
batches, checkpoints every completed generation, and caches every completed candidate evaluation.
For fast callbacks, pass `cache_evaluations=False` alongside `**control` to skip per-evaluation
cache files; the supplied cache directory is ignored, generation checkpoints remain resumable,
and an interrupted partial generation may need reevaluation. Existing cache files are not deleted.
It supplies exactly `progress_callback`, `cancellation_check`, `checkpoint_path`,
`evaluation_cache_directory`, `campaign_identity`, and `resume`. Do not also pass those keywords
explicitly (Python rejects duplicate keywords). It does **not** supply `bounds`, `objective_names`,
or the required `initial_position`; supply these physics arguments yourself.
Programs that call the population optimizers or `optimization_options` are automatically submitted
to a durable background worker even if the model omits `background=true`; a chat-turn boundary must
never terminate an optimization.

To report/revalidate a completed or interrupted search without rerunning it:

```python
saved = api.read_optimization_checkpoint(
    run.workspace / "optimizations" / "descriptive-campaign-name" / "checkpoint.json"
)
best_values = saved["best_candidate"]["variables"]  # name -> value, not position array
best_position = saved["best_candidate"]["position"]  # exact original variable order
history = saved["history"]
```

The read-only view also returns `contract`, `complete`, `completed_generation`,
`evaluated_candidates`, `pareto_candidates`, `scope`, and `checkpoint_path`. A plot failure does
not invalidate or erase committed numerical generations. Rebuild the selected candidate from
the original saved source's construction function and these variables; independently validate
it with JuTrack before claiming it. This reporting read does not resume optimization and does
not relax the source/contract checks. `run.optimization_options` supplies control hooks, not
saved physics results; do not call it just to discover a checkpoint path.

On restart, a partially completed generation reuses its cached JuTrack results. The population
optimizer can extend a compatible checkpoint to a larger total generation count. Runtime controls
fingerprint the computation and immutable inputs: for a uniquely identified direct top-level
`api.particle_swarm_optimize(..., **run.optimization_options("name"))` call, edits only to later
plots/tables can safely reuse completed generations. Every preceding statement and evaluator
definition remains protected. Ambiguous/nested calls retain full-program identity protection.
Changed computation, arguments, saved input lattices, variables, bounds, objectives, population,
selection, or seed reject resume; inspect archived results or start a separately named campaign
instead of deleting a checkpoint. Selected candidates always need JuTrack revalidation.

For nested lattice matching use `api.optimize_lattice_parameters(...)`. Its evaluator owns JuTrack
metrics; the kernel owns proposal/selection mechanics. Multi-objective mode uses parent-plus-
offspring nondominated sorting and crowding. Generate a fully feasible initial population for hard
problems. The managed lattice helper applies the same configured production floors
before allocating its seeds; standalone kernel tests can use explicit small budgets.
Model-written smoke-test options cannot lower the managed policy. It also rejects overlapping
outer/matching field ownership and performs an inner match for every structural proposal.

```python
def make_worker_physics(baseline_rows):
    # Everything below is created inside each process. `candidate_from_rows`
    # is the study's explicit conversion to LatticeIR.
    jt = api.jutrack()
    lattice = api.build_jutrack_lattice(candidate_from_rows(baseline_rows))
    editor = api.JuTrackLatticeEditor(jt, lattice, baseline_rows)

    def metrics(rows):
        editor.sync_rows(rows)
        return api.jutrack_linear_metrics(jt, lattice)

    def objectives(rows):
        editor.sync_rows(rows)
        values = api.jutrack_periodic_radiation_metrics(
            jt, lattice, energy_gev=energy_gev, periods=periods
        )
        return {
            "objectives": {"emittance_pm": values["emittance_pm"]},
            "constraint_violations": declared_violations(values, rows),
        }

    return metrics, objectives

# Do this BEFORE loading JuTrack or probing callbacks. Omit population_size to
# use the configured production floor; small managed requests are raised visibly.
plan = api.plan_lattice_optimization(design, matching)
run.progress("Optimization setup", kind="decision",
    detail="Compiled field ownership and production budget; physics feasibility not yet tested.",
    metrics={"outer_knobs": plan["outer_knobs"], "matching_knobs": plan["matching_knobs"],
             "population": plan["population_size"], "generations": plan["generations"]},
    tables=[{"title": "Optimization and matching variables", "rows": plan["knob_table"]}])
optimization = api.optimize_lattice_parameters(
    elements,
    design,
    matching,
    evaluate_metrics,          # returns matching quantities such as Dx_m and Dpx
    evaluate_objectives,       # receives the rematched rows; returns the mapping above
    match_targets={"Dx_m": (0.0, 1.0e-4), "Dpx": (0.0, 1.0e-4)},
    objective_names=("emittance_pm",),
    population_size=plan["population_size"],
    generations=plan["generations"],
    minimum_population_per_knob=10.0,
    require_feasible_initial_population=True,
    # This factory runs independently in every configured worker. It must load
    # JuTrack and build its mutable lattice inside the process, then return
    # (evaluate_metrics, evaluate_objectives). Never capture a live Julia object.
    worker_physics_factory=make_worker_physics,
    **run.optimization_options("complete-cell-emittance-search"),
)
selected_rows = optimization["selected_elements"]
assert selected_rows is not None
```

`optimization` has the public PSO keys documented above plus `optimization_plan`, `parameterization`,
`matching_parameterization`, `selected_elements`, `selected_matching`, `initialization`,
`convergence_diagnostic`, `local_polish`, and `inner_matching_performance`. In particular, record
the generation-zero feasible count, improvement after generation zero, matching success/failure
counts, and actual population per outer knob in the result shown to the scientist.

When `worker_physics_factory` is present, `optimize_lattice_parameters` takes the process count from
`[compute].local_worker_processes` in `research_runtime.toml`; the model does not choose it.
Generation-zero feasibility screening and all later populations use those isolated processes. Each
worker retains its own JuTrack lattice and nearest successful inner-match state. The built-in inner
matcher supports any declared smooth scalar equality targets returned by `evaluate_metrics`, such
as `Dx_m`, `Dpx`, beta/alpha values, or a requested transfer-map quantity. Put inequalities and
non-smooth validity conditions in `constraint_violations`, not in the matcher. Discrete topology
changes belong in separate candidate construction or separate campaigns.

## Lattice editing and plots

The public insertion/removal helpers preserve total length by splitting or restoring neighboring
drifts. Use them for sextupoles, kickers, diagnostics, apertures, and similar modifications.

```python
cell_elements = api.one_period_elements(ring)
profile = result.linear_optics.twiss_profile
plot_path = run.artifact_path("one_cell_optics.svg")
api.save_optics_lattice_plot(
    plot_path, profile, cell_elements,
    title="One-cell optics and lattice", subtitle="revision and method",
)
```

This produces the standard combined beta/dispersion/lattice figure. SVG is preferred, while PNG and
PDF are also supported and retain the same combined layout. Do not replace a failed standard plot
with a beta/dispersion-only fallback: report the plotting error so the lattice is never silently
omitted. The plotting helper takes the common longitudinal interval, so the example above clips a
full-ring JuTrack profile to exactly the elements in one periodic cell. Use a full-ring survey only
for geometry. Radiation studies should also save the exact
per-bend decomposition and its standard plot:

```python
bend_evidence = api.bend_radiation_contributions(ring)
assert bend_evidence["consistency"]["passed"]
bend_plot = run.artifact_path("bend_radiation.svg")
api.save_bend_radiation_contribution_plot(bend_plot, bend_evidence)
```

The returned rows contain bend name, angle, length, mean horizontal radiation invariant, I1--I5,
I2/I4/I5 fractions, and I5/I2. They are grouped as B1...BM across identical periods using the saved
`source_element` provenance. The calculation uses JuTrack's per-element `elrad` and verifies that
the I2/I4/I5 sums reproduce the authoritative full-ring `ringpara` result.
