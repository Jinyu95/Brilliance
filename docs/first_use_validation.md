# First-use reliability investigation — 2026-09-08

## Current investigation: duplicate results and lost completed work

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
