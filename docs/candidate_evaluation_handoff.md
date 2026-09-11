# Candidate evaluation and working-state handoff

The September 10 seed-search regression was local to one new project. A generated
program scanned quadrupole settings into an in-memory list, then ranked that list
using radiation calculated on the mutable editor left by the last scan point.
No other project's lattice is needed for this bug to occur.

## Candidate-bound calculations

Use `editor.evaluate_candidate(rows, energy_gev=E, periods=P)` for fast periodic
optics/radiation trials. It applies explicit rows and returns an independent element
snapshot with both calculations. Unstable trials skip radiation; API errors propagate.
Retain a result's own snapshot, then replay it before transferring the selected
candidate. Parallel workers must own separate editors. The low-level direct JuTrack
helpers remain available, but callers are responsible for synchronizing their state.

This local pass does not establish geometry compliance, achromat, full-ring validity,
or target attainment. Those checks remain separate. The MBA module requires the
optimized unit evidence checkpoint before assembly; the conventional and hybrid
construction routes are otherwise unchanged.

## Handoff

`run.finish(candidate_lattice=...)` preserves a candidate, but does not implicitly
select it. For a validated complete candidate intended for subsequent research,
explicitly request `use_as_working_lattice=True, scientific_passed=True`. Existing
deterministic and expert checks still control promotion. Component seeds and failed
or comparison-only candidates remain branches.

Saved results expose `candidate_handoff`. Completing a chat from a receipt explicitly
reports whether that execution selected a working candidate or only archived a branch.
Completion itself never promotes a candidate or rewrites numerical history.

## Validation

Regression coverage includes candidate A → B → A snapshot/radiation consistency,
unstable-trial skipping, API error propagation, explicit handoff, and rejection of
artifact reads outside the current project's workspace.

An offline, read-only JuTrack check on the current project's archived conventional
cell returned 57.92187760366427 pm, then 57.92195925546263 pm after a quadrupole
perturbation of 0.00001 m^-2, and exactly the original value after snapshot replay.
The input file's SHA-256 was unchanged. This validates evaluator replay, not a fresh
autonomous design campaign or the ring's nonlinear performance.
