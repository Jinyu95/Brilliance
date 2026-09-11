# Progress publication and free-optics conversion — 2026-09-10

## What failed

The diagnostic in project `project-6b915dedeac14a`, job
`job-c960cd5f6332920052ca`, ends at `ResearchRun.progress` / `os.replace`.
It was publishing “Accepted 51 of 139 required seeds” during feasible-population
screening, before production generations. Windows denied replacement of the
progress JSON. The traceback does not identify which process held the file.
This is an I/O execution failure, not JuTrack rejecting the physics.

The generated program also imposed baseline-derived core Dx/H/I5 ceilings and
an emittance ceiling of 56 pm. Those are additional restrictions, not inherent
requirements for first-stage hybrid shaping.

## Changes

- Generic atomic writes use unique temporary files and bounded retries for access/
  sharing errors. Optimizer checkpoints and evaluation-cache writes use the same
  primitive and still propagate persistent failures.
- UI progress channels publish independently. Persistent progress-only I/O failure
  is logged, then retried after a five-second cooldown, without aborting numerical
  work. Result/checkpoint persistence and schema errors are not silently ignored.
- MBA expert module v1.12 makes source beta/dispersion preservation opt-in.
  The outer matcher can vary symmetry-plane betax/betay/Dx while propagating the
  core half and matching the new outer section. Global symmetry still imposes
  zero alpha_x/alpha_y/Dpx at the reflection plane; old tunes are not targets.
- Resulting usable drift bumps, core separation, achromat, stability, positive
  damping and engineering bounds remain independent checks. Final low-H quality
  remains separate from the initial feasibility milestone.
- `core_limits=None` and no emittance ceiling are supported by the shape objective.
  Explicit later-stage protection remains enforceable. Archived reports with an
  optical-preservation gate retain their original stricter meaning.
- Conventional TME/MBA seed generation was not changed. Saved research programs,
  working lattices and project histories were not rewritten or restarted.

## Verification

238 focused tests passed: runtime/optimizer persistence, a real Windows reader
lock, concurrent atomic writers, independent UI channels, scientific acceptance,
expert-module boundaries, and free versus explicitly fixed optical contracts.
The broader suite had 748 passes and two skips before one additional regression
test was added. Its only failure was the pre-existing local-configuration assertion
expecting `candidate_wall_time_s=600`; the configured value is 3600 and was not changed.

An offline JuTrack scout used this project's exact working conventional cell:
`program-3cb59e28d1fa218ec68c/candidate_lattice.json`, boundaries `P002:Dend`
through `P004:R:Dend`. Four new outer geometries, three starts each, 180 local
solver iterations per start completed 19,890 residual evaluations in 28.68 seconds
(including initialization). Actual centre beta and dispersion changed; no API or
runtime failure occurred. None passed all line residual tolerances. This small
scout does **not** demonstrate a successful H7BA or prove physical infeasibility.
No candidate was published. Source hash was unchanged. Diagnostics are under
`research_runs_hybrid_free_optics_20260910/`.

## Continuing the user's research

Load the updated application/module in a new worker. Ask for a new hybrid attempt
with source beta/dispersion free and no implicit baseline H/I5/emittance ceilings.
Simply rerunning the old saved program retains its explicitly coded restrictions.
Changing objectives is a new campaign: do not reuse incompatible optimization
checkpoints as though they evaluated the new problem. The old results remain useful
as diagnostics, not as automatically valid evaluations of a changed objective.
