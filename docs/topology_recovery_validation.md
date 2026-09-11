# First-chat topology failure: cause, recovery and verification

Date: 2026-09-11. Project: `project-4831fee92506e4`.
Failed job: `job-07448b0d3dc8c07a4115`.

## Exact cause

The archived `program-3a219a47fecaf41f93ec/program.py` finished its unit PSO and
replayed the selected unit successfully. At line 159, its diagnostic code passed
12 rows to an editor constructed for 11: physical bend `B` had been replaced by
`B:a` and `B:b` to inspect the bend centre. Reusing that compiled editor was invalid.
The safety check itself was justified; deleting it would risk editing the wrong
elements. The error message did not explain this, and the repair classifier did
not recognize the old `ValueError`, so no repair was scheduled.

The checkpoint is complete: 30 generations, 1,860 optimizer evaluations. Failure
occurred during reporting, not because optimization failed or the seed was unstable.

## Application changes

- `LatticeTopologyError` identifies editor/sampler lifetime misuse, with expected/
  received counts, first differing element and a repair instruction. Count, order,
  names and kinds are checked for the entire list before mutating any field.
- This typed error, and the exact archived untyped equivalents, enter the existing
  bounded repair mechanism. Physical infeasibility, infrastructure failures, user
  cancellation and superseding requests retain their separate handling. Unchanged
  repeated failures and exhausted repair budgets still stop.
- Chat explains the API mismatch and gives the actual reason when repair cannot
  start, rather than only “Automatic code repair was not started.”
- A general sampler option, `linear_metrics(sample_fractions={name: [.5]})`, measures
  exact element interiors through JuTrack without splitting the physical lattice.
  It retains map checks, physical exits, dense samples and extrema. Plotting uses
  the original physical element list.
- General API guidance and MBA unit-reporting knowledge now use that sampler.
  Conventional seed construction and optimization physics were not rewritten.
- Repair handoffs explicitly direct postprocessing recovery from the saved best
  candidate, not a repeated search. The saved-evidence summary now maps actual
  checkpoint `objective_values` to `objective_names` instead of showing no values.

## Validation

202 focused tests passed, one skipped. Tests include first-chat managed execution,
automatic repair for typed/legacy errors, whole-list validation before mutation,
renaming/reordering/splitting/type changes, exact requested interior samples, and
unchanged physical candidates. The broader suite had 765 passes and two skips
before the final context-summary regression was added. Its only failure was the
existing configuration-dependent assertion expecting a 600-second candidate
timeout when the user's configuration specifies 3600 seconds. No timeout was changed.

An offline JuTrack replay used ONLY this project's original saved builder and
its selected checkpoint variables. It performed zero new optimizer evaluations.
The stored and replayed repeated-unit emittance both equal **41.420422720742366 pm**.
The exact centre sample is at s=1.5 m, beta_x=0.1527385553 m, Dx=0.0027775109 m.
The physical model remains 11 elements and one bend, and the diagnostic maps agree.
Replay, sampling and plotting completed in 24.33 seconds including initialization.

This is component evidence, not a finished 7BA or proof that the final machine
requirements are met. No result was published into the application and no working
lattice was selected. The original source and checkpoint hashes are unchanged.
Verification artifacts are in `research_runs_topology_reporting_20260911/`:
`validate.py`, `validation.json`, `recovered_unit.json`, `recovered_unit_optics.png`.

## Resume

Restart Brilliance to load the new runtime and knowledge, then continue the same
project with: “Recover the completed unit optimization checkpoint, finish its
diagnostics and plot without rerunning PSO, then continue the 7BA design.”
The already-failed job was not silently restarted or rewritten by this code update.
