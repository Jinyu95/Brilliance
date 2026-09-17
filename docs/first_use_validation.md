# First-use reliability investigation — 2026-09-08

Current validation: [API contracts, worker recovery and fresh release trials](reliability_upgrade_20260915.md).
The entries below are historical investigations, not current release certification.
They distinguish fresh numerical construction, actual model-driven trials and
developer-assisted saved-candidate replays; these are different validation scopes.

## 2026-09-13: cell/ring scope handoff, not a JuTrack physics failure

Project `project-35bd9b9c0c89c2` stopped in program `program-d43ea5cc6f7371616895`
while extracting one cell from `program-a42103ae340b968408f4/cell_best.json`.
The saved source confirms that file stores ONE 35-element cell, but constructed
it with `api.lattice(periodicity=32)` and no scope metadata. The extractor tried
to divide 35 stored elements into 32 cells. The plain ValueError was not sent
through bounded coding repair. Production optimization had not started.

This was not the only error in the waiting program: its inclusion map selected
drift lengths but not bend angles despite an exact angle constraint; it also
excluded only `ID_half` and not `R:ID_half`. The former now fails source preflight
without starting Julia; incomplete mirror selection is a typed input error.
Static reconstruction of the intended selection yields 12 outer variables
(eight drift-length pairs, four bend-angle pairs), five matching variables and
the default 500-particle/30-generation plan. This verifies setup only.

The application now validates declared scope before new construction, extraction,
saving and high-level simulation. It retains legacy artifact readability and
does not infer scope from names or bend angles. Scope errors contain observed
counts and corrective constructors and enter bounded coding repair. Cell/ring
constructors own their scope metadata; copying a cell's metadata into an expanded
ring can no longer mislabel the ring as a cell. Single-cell topology edits update
the stored element count; full-ring topology changes require a consistent explicit
partition. Nonuniform rings should be represented as one explicit sequence, not
claimed to contain an integer partition that their physical rows do not support.

### Actual saved-artifact replay

The opt-in test `tests/test_jutrack_saved_handoff.py` exercised:

load the exact historical file -> explicitly confirmed cell scope -> save/reload ->
expand full ring -> extract same physical cell -> JuTrack periodic and full-ring
evaluations -> per-bend radiation reconciliation -> combined optics/lattice plot ->
progress and final result with real tables/plots.

It passed in 31.56 s using configured Python/Julia/JuTrack. No seed generation,
optimization, model call, internet search or user-project write occurred. Original
cell SHA-256 remained `24adfac9e5d7e5c12a5e4f02d35125e0530e6b40e1121daa895ee2a75bd32ca8`.
Derived files were confined to pytest's temporary workspace.

| Replay measurement | Result |
|---|---:|
| Physical elements / expanded ring elements | 35 / 1120 |
| Cell length / ring length | 25.001172202060822 m / 800.0375104659463 m |
| Full-ring natural emittance | 8410.827917105 pm rad |
| ID dispersion / slope | -5.2114778364280794e-5 m / -2.0027882279591118e-16 |
| Periodic vs full-ring emittance agreement | passed |
| Per-bend radiation-integral reconciliation | passed |

This is a reproducible **poor prototype**, not a successful low-emittance design.
Its original construction audit already reported a mismatch with the program's
exact 25 m cell allocation; that evidence was retained, not hidden by this repair.
The latest request described nominal circumference, not an approved tolerance interval.
The replay does not validate autonomous route choice, optimizer convergence or the
complete first-user design experience. Those require a separate model-driven release
trial; passing this handoff test must not be advertised as completing that trial.

To repeat this check on a reviewed saved single cell, set `BRILLIANCE_TEST_CELL_PATH`
and `BRILLIANCE_TEST_CELL_PERIODS`, then run the opt-in test with the configured
interpreter. Normal test runs skip it; their scope/serialization tests use only
synthetic rows, not a stored design template.

## 2026-09-12: conventional seed obtained, optimizer setup failed

Branch: `chatgpt/h7ba-radiation-aware-continuation`. Inspected the current project
`project-351d6c89c016d9`; no previous-project lattice was used or copied.

The latest failure was an execution contract error, not a failed population search.
Program `program-803692addde6769bbd90` described eight quadrupole-strength pairs and
eight drift-length pairs (16 outer knobs), but requested both `length_m` and `k1_m2`
on all selected elements. This additionally freed eight quad-length pairs: 24 knobs.
Its 160 particles violated its own ten-particles-per-knob minimum of 240. No production
generation ran. The callback also checked beta only at the ID, not over the cell.

Earlier program `program-b2aec96a99378e7cd26e` called `run.finish` in its unsuccessful
terminal-match branch without exiting. Later statements overwrote that evidence with
a different result. Some subsequent trials attempted hybrid shaping even though the
request was for a conventional starting design. Fixed-geometry transfer failures did
not establish that conventional matching was impossible: releasing drift allocation
subsequently yielded a feasible periodic cell.

Application repairs:

- Pure-Python `plan_lattice_optimization` reports actual compiled field ownership and
  population requirements before simulation setup. Explicit undersized budgets are
  repairable input errors, not silently increased or treated as physics infeasibility.
- `included_element_names_by_field` selects explicit, globally mirrored element/field
  pairs. Unlisted fields stay fixed; invalid names, incomplete pairs and conflicting
  inclusions fail explicitly. The old inference mode remains supported.
- Other nested-optimizer setup guards now use typed input errors. Bounded repair also
  recognizes the exact archived population/generation-policy errors and directs the
  model to verify intended ownership, retaining the saved candidate and machine limits.
- Source preflight rejects a normal path that can overwrite an earlier final result.
  `finish` retains its existing API; early exits must explicitly stop execution.
- MBA-specific guidance remains in the expert module: diagnose a failed terminal
  transfer, release provisional geometry, and allow complete-cell matching without
  preserving isolated-unit optics. Do not invoke hybrid shaping as infrastructure repair.
  The working modified-TME method and JuTrack implementation were not rewritten.

### Verification and limits

Replayed the exact saved candidate, without changing its JSON or running optimization:

`research_projects/project-351d6c89c016d9/workspace/artifacts/programs/program-b61b7af6939c18cd6f3e/candidate_lattice.json`

SHA-256: `d3546d6648e5e64bc16b6a41371064062d4208a6623ab1381c0aaf88cac52f8b`.

| Measured with configured JuTrack | Result |
|---|---:|
| Energy / intended periods | 3.5 GeV / 32 |
| Physical elements / positive bends per cell | 77 / 7 |
| Cell length / bend angle | 25 m / 0.19634954084936207 rad |
| Periodic stability / radiation-validity gate | passed / passed |
| Natural emittance (periodic radiation calculation) | 600.6480074984273 pm rad |
| Horizontal damping partition | 0.9993118465836074 |
| ID Dx / Dpx | -1.7306391808301312e-9 m / -2.6216651102693904e-16 |
| Maximum beta over element-boundary samples | 72.50888202364553 m |

Static replay of the actual saved setup reproduces 24 knobs and the population error;
explicit selection matching the program's stated intent yields 16 outer and six inner
knobs, with a valid 160-particle setup. These are setup checks, not optimization results.
The new source check detects both possible result overwrites in the archived terminal program.
Regression tests use synthetic rows/algebraic callbacks, not a solved research lattice.

No new model-driven end-to-end design, TME regeneration, production optimization, independent
full-ring revalidation, chromatic correction, tracking, or internet search was performed in
this audit. The replay establishes a reproducible conventional **seed**, not a fourth-generation
operational design or a demonstrated emittance target. User project files and the user-modified
JuTrack submodule were preserved. Earlier investigations below are historical, not validation
of this branch's current end-to-end autonomous behavior.

## Earlier investigation: duplicate results and lost completed work

The user's latest project, `research_projects/project-48d7a958155666`, did finish
its 40-generation, 2,460-evaluation unit search. Reporting then failed on a wrong
variable key (`QF` instead of `k1_m2:QF`). A reporting-only edit subsequently
invalidated the checkpoint identity, so the application could not recover the
completed search. Its many generation snapshots and repeated result attachments
made this look like repeated research without a useful result.

The repaired exact-program replay is isolated in `research_runs_recovery_probe`.
`replay-audit.json` records zero repeated objective evaluations, unchanged generation
40, unchanged 2,460 evaluations, and identical best candidate and history. JuTrack
re-evaluated the recovered candidate at 1,397.866 pm. That is recovery of a poor
unit result, **not** a successful complete MBA design. No original project or
checkpoint was edited.

Changes address application boundaries, not a stored lattice template:

- Progress has a stable campaign identity; current generation/evidence updates the
  same activity. Full optimizer history remains archived. Identical tables/images
  are shown once with references; changed candidates and requested re-displays remain
  distinct. Legacy histories are folded at read time, not rewritten.
- Reporting-only checkpoint recovery uses the exact objective/construction prefix
  and archived inputs. Changing physics invalidates reuse. Optimizer candidates expose
  an ordered position so reporting does not guess field names.
- Typed optimizer/lattice input errors enter bounded coding repair. Physics
  infeasibility and environment failures do not. Repairs and result reviews preserve
  the scientist's request and yield to newer input or an explicit pause.
- Native model sessions can inspect saved artifacts and retrieve complete expert
  methods without starting JuTrack. An inventory distinguishes a unit, complete cell,
  and full ring by exact saved identity. Overlong malformed provider output is not
  fed back into the model's context.
- Receipts include exact full-result, candidate, and construction-source paths.
  Wide dictionaries retain bounded scalar measurements before nested arrays. Missing
  JSON keys report the actual schema; repeated unchanged lookup batches are compacted
  into one recovery reminder. The agent's own growing event logs are not exposed as
  research evidence.
- Complete-cell extraction preserves nested physical element identities. The old
  helper flattened repeated interior units into duplicate names; the model's
  correctly written full-ring program now runs unchanged.
- Exact-candidate optics and per-bend evidence are retained during managed calculations
  and presented when that candidate is committed. Compact numerical returns no longer
  silently discard the data needed for a one-cell lattice/optics plot. This performs no
  additional JuTrack evaluation and does not create plots for each optimizer particle.
- Inferred magnetic-strength knobs require explicit bounds. The public parameterization
  example now derives them from energy and technology limits; its actual code is tested.
  Coupled bend-field and geometric constraints still belong in the task's objective audit.
- The constructive expert method covers fresh unit geometry/strength exploration
  and one-pass matching of terminal sections before full-cell evaluation. It supplies
  equations and numerical procedures, not facility lattices or solved strength arrays.

## Real model and JuTrack trials

The trials use the unchanged welcome prompt, configured `deepseek-chat`, configured
Python/Julia/JuTrack, and disabled research internet access. Model API inference is
permitted. They are isolated from user projects. No facility seed was supplied.

Trial `research_runs_default_validation/project-22cd3389314064` independently
constructed and matched a complete 32-cell 7BA. Re-evaluating its exact saved full
ring, `lattice-44a974b9578d7dc2`, with JuTrack produced:

| Check | Result |
|---|---:|
| Energy / cells / bends per cell | 3 GeV / 32 / 7 |
| Circumference | 896 m |
| Natural emittance, original ringpara | 27.5064 pm rad |
| Straight-boundary dispersion | approximately zero |
| Transverse stability, angle closure, default magnet limits | passed |
| Positive damping | passed |
| Fractional vertical tune | approximately 0.00186 |

This establishes a valid **linear draft**, not an operational design: the near-integer
vertical tune, chromatic correction, nonlinear acceptance, and robustness need further
work. Per-bend radiation integrals reconcile with the full-ring calculation. The audit
is `research_runs_default_validation/project-22cd3389314064-audit.json`.

The model nevertheless paused without presenting a plot. Another trial,
`project-e7a3d1991e9362`, finished a complete-cell optimizer but repeated the same
missing diagnostic lookup until the model-step boundary. These are application/model
delivery failures, not evidence that another optimizer generation is needed.

The latter optimizer completed 530 particles over 40 generations (21,730 evaluations)
in about 2.5 minutes with eight workers. Its reported 44.18 pm agrees with independent
JuTrack ringpara, but 12 quadrupoles exceed the declared field limit and 26 drift rows
violate the declared minimum length. The callback's `feasible` flag only described its
inner match. Do not present it as an engineering-feasible design. The earlier 120.84 pm
complete candidate already had a useful optics plot; the request for a first candidate
did not require this extra optimization. New guidance distinguishes delivering the
requested draft from optional later refinement or operational qualification.

**Release status remains unqualified.** These trials included replay of recovery
handoffs while fixes were being implemented; they are not untouched, cold-start passes.
The acceptance check deliberately requires the requested complete achromat, independent
JuTrack verification, plots and tables exposed in chat, and a finished answer. A good
unit, a successful calculation alone, or a model's success wording does not pass it.

## Follow-up default trial and closing the request

`research_runs_default_validation/project-b7172ef1408f23` was started from the same
unchanged default prompt with no user follow-up or numerical steering. It corrected
an invalid initial optimizer position and a mismatched archived transition itself.
Its complete ring, `lattice-96ed6466535b6cac`, independently passes the JuTrack gate:
3 GeV, 32 cells, seven bends per cell, 1,051.2579 m circumference, 26.88898 pm natural
emittance, stable transverse optics, achromatic insertion boundary, and the declared
magnet envelope. Circumference was not a hard requirement in this default prompt.
The conversation contains real tables, optics/lattice figures, and interpretation.

However, after saying the first candidate was delivered, the model continued into
optional chromatic/sextupole work and stopped on malformed output. The strict audit
therefore remains failed despite the independently valid candidate. Do not advertise
that trajectory as a clean unattended pass.

The controller now separates immutable calculation receipts from semantic completion
of the current scientist request. `complete_research` can finish using an exact known,
validated saved run, without changing its numerical data or rerunning an optimizer
merely to change an intermediate program flag. Its completion proof is bound to the
current request and saved evidence; new input or pause takes precedence. Descriptions
such as "preliminary" and unmet soft aspirations no longer act as hidden vetoes on a
request explicitly asking for a preliminary candidate. This is a task-boundary fix,
not an assertion that the machine has passed nonlinear or operational validation.

After that fix, the saved-result review was explicitly replayed by the developer
against the same project and original request. The configured model closed the request
using `complete_research`, without a new program execution or numerical change. The
resulting saved-state audit now **passes all checks**, including independent JuTrack
evaluation, final answer, tables and plot exposure. The prior program's intermediate
`request_satisfied=false` remains unchanged. The final task-completion proof references
`program-8485227293f8a08b16bc` and the original scientist request.

This validates recovery and delivery; it does **not** erase the earlier failures or
establish a cold unattended pass after every change. The model also did not produce a
per-bend report in this particular trajectory. That presentation path was separately
verified by replaying the unchanged archived full-ring program in
`research_runs_default_validation/presentation_replays/project-767982977fcd51`:
two figures and two real tables, seven finite bend rows with 32 instances per group,
and a 28 m one-cell optics view. The browser test confirms that these appear once.

The latest full offline suite passed 480 tests with two optional/environment skips;
Ruff passed. The final browser check of b717 shows two loaded images (unit and complete
cell), ten real tables, eleven compact previous-evidence references, a visible final
answer, and no browser errors or external requests. Byte-identical images saved under
different artifact paths are now deduplicated by bounded, cached, read-only hashes;
explicit user re-show and genuinely different image content remain supported. Equal
table values from distinct calculation receipts remain preserved rather than silently
discarding their provenance.

Completion can also resolve earlier validated project runs through the archived
source hash, not only the latest diagnostic. The model receives exact eligible IDs
and result paths. Loose result files, foreign projects, and tampered source snapshots
cannot stand in for a committed numerical execution.

No user project or history was deleted or rewritten. Research internet access remained
disabled throughout the real physics/model trials.

## Earlier baseline trial (before this repair)

Status: **not release-qualified for unattended MBA design**.

The test used the unmodified welcome-screen prompt through `ResearchChatService.create`,
the configured `deepseek-chat` backend, the configured Python/Julia/JuTrack runtime, and
`internet_access=false`. No facility lattice, validation seed, or additional user instruction
was provided. User projects were left untouched. Test project:
`research_runs_default_validation/project-bd3976032b5f98`.

## What the test established

The agent solved a modified-TME unit to approximately 30.42 pm on its repeated-unit reference
ring. It then attempted complete 7BA construction and matching. The final complete-cell
optimizer stopped because its baseline matcher did not reach the dispersion targets. This
was not proof that JuTrack could not run or that the physical requirements were impossible.

An independent JuTrack evaluation of the last archived complete-cell candidate gave:

| Check | Result |
|---|---:|
| Energy | 3 GeV |
| Periods / bends per period | 32 / 7 |
| Circumference | 640 m |
| Transverse stability | passed |
| Bend closure, default field envelope, positive damping | passed |
| Natural emittance | 4531.82 pm rad |
| Dispersion at the straight boundary | -0.114635 m |
| Dispersion slope at the straight boundary | approximately zero |
| Achromatic boundary, 0.1 mm tolerance | failed |
| Plot exposed in chat | absent |
| Default-task acceptance | failed |

A good unit did not transfer into a useful complete achromat. Do not present its 30.42 pm
as the ring emittance, or call this a successful default-task design. The machine-readable
report is `research_runs_default_validation/latest-audit.json`.

## Corrected application boundaries

- Initial prompts are persisted as real user messages. Recovery and result review therefore
  retain the original request identity; the chat no longer needs to fabricate that message.
- Public-call preflight checks known optimizer control expansions for missing required
  arguments and duplicate keywords. Documentation no longer mixes manual control keywords
  with `**run.optimization_options(...)` or demands optimization before legitimate diagnostics.
- Both direct and high-level radiation helpers report `computed`; physical validity remains
  a separate condition. Previously, a missing status field caused every particle to be rejected.
- Family metadata is documented consistently as `row['fields']['family']`. The compiled-family
  example is regression-tested to edit both intended magnets and reject missing families.
- Python API/OS errors in optics helpers are no longer silently labelled physical instability.
  Named broad exception handlers cannot simply hide those errors as infeasible particles.
- Durable results return to the agent for interpretation or a next justified calculation.
  New requests and pauses take precedence; result review and coding repair have separate budgets.
- `ResearchInfeasible` distinguishes an unsuccessful bounded scientific attempt from a broken
  program. Baseline-matching and feasible-population failures preserve attempted settings and
  diagnostics, do not start production PSO, do not advance the working lattice, and return to
  the agent. Ordinary execution errors remain errors; infeasibility is never reported as success.

## Verification and limits

The offline suite passed 316 tests; the optional real-program replay is skipped by default.
It was also run separately and passed with the exact failed matching program unchanged:
JuTrack produced a structured negative result instead of an application crash. This did not
repair its physics, and it was not counted as an unattended default-task success.

The real `--check-optimization` test passed with two isolated processes and two-coordinate inner
dispersion matching: 11 successful inner matches, zero failures, about 38.7 seconds including
startup. This is a small optimizer integration test, not proof of production search convergence.

The complete default trial preceded the final negative-outcome boundary change. No fresh
end-to-end pass after all these changes has been established. Release qualification still needs
an unattended complete achromatic MBA, its independent numerical check, and useful chat plots
and tables. The release smoke test deliberately fails rather than substituting a unit/reference
ring, ignoring missing presentation, or accepting the model's completion wording.
