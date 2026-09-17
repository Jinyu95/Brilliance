# Reliability upgrade — 15 September 2026

## Why the latest user run failed

`project-ece0a20f7db2ea` failed before JuTrack was reached: an invented magnet-limit
key, an invalid numerical output directory, then a request to resume a nonexistent
checkpoint. The three failed executions took about 2.43 seconds combined; most of
the six-minute wait was model editing/repair. This was not evidence that TME physics
or population optimization failed. Latent errors included interpreting an absolute
trace margin as a normalized trace threshold, suppressing progress, and collapsing
distinct finalists because they share total length and bend angle.

## Application changes

- Installed typed input/result contracts feed API help and source preflight. Literal
  invalid keys in known closed mappings fail before execution, with the real keys
  supplied. Dynamic/extended mappings remain supported; this is not a full Python
  type checker. Alias mutation and shadowed imports must not trigger false
  rejections. Result contracts are delivered once per editing turn, not repeatedly.
  Missing optics/radiation values are explicitly nullable, never numeric zeros.
  The reporting-stage declaration now has one shared input schema for help,
  preflight and runtime validation.
- Numerical storage and persistent optimizer checkpoints are distinct. TME tools
  accept managed child storage and default to starting or resuming an intact,
  compatible checkpoint pair. Corrupt, partial or incompatible pairs are not erased.
- Explicit same-project evidence reuse supports a new worker without numerical
  reruns. Original hashes, lattice identity and calculation identity are retained;
  another project's results are rejected. Reuse does not upgrade acceptance.
- Managed progress and cancellation remain active with custom observer hooks.
  Failure diagnostics show bounded available mapping keys rather than large values.
  Already-decoded Windows paths/JSON are no longer incorrectly unescaped a second
  time when returning an execution error to the model.
- Canonical one-cell optics/lattice plotting reads the exact selected verified
  archive, even after cache eviction or a worker restart. Invalid selected
  evidence cannot fall back to an older canonical plot. Optional presentation
  write failures preserve the numerical receipt; no simulation is rerun.
- Open-line matching returns JSON-native booleans and discoverable typed fields.
  Its recovery advice preserves line measurements/checkpoints, instead of asking
  for an inappropriate periodic calculation. It still cannot certify a ring.
- Release auditing distinguishes a live process from stale "running" metadata,
  without mutating job state, relaunching a worker or cancelling calculations.
- Claimed workers now have an outer operational-error boundary, including imports
  and agent startup. Unexpected errors archive a traceback and mark that worker
  failed; independent jobs are untouched. A failed follow-up cannot downgrade an
  already committed completed increment. Interpreter/claim checks remain outside
  this boundary so a wrong interpreter cannot seize or fail somebody else's job.
- The expert guide specifies trace-margin units, exact returned candidates and a
  short call/publication example. Physics objectives, engineering constraints and
  the unit → terminal → complete-cell route were not replaced by a template.

## Verification scopes

The initial offline suite passed **956 tests**, with **14 opt-in tests skipped**.
Follow-up regressions were added from the real trial; final totals are recorded
here: **1,026 passed, 15 opt-in tests skipped in 59.35 s**. Ruff and the scoped
whitespace diff check passed on changed files. Skipped numerical tests are not
counted as physics passes; the explicitly executed native checks are listed below.

The real configured JuTrack worker test completed in **213.44 s**, including a
second managed process which recovered and published the same calculation while
native calls and optimization were explicitly disabled. It searched geometry and
strengths with **500 feasible initial particles and 80 generations**, then verified
four units at **37.918–37.966 pm** on their repeated-unit reference rings.
Source: `research_runs_reliability_native/project-tme-geometry-ed725991ccab`.
No saved geometry or strengths initialized that search; historical input supplied
energy, total length, angle and intact bend fields only. The 100 m beta cap and
0.01 trace margins are test hypotheses, not newly approved machine requirements.
These are unit/tool and recovery tests, **not a fresh complete-MBA design pass**.

A further real managed worker recovered that same native calculation and
automatically rendered its combined one-cell PNG, with all numerical entry points
patched to fail if called. Normal application result validation passed; calculation
identity was unchanged. This test completed in **15.59 s**. Output:
`research_runs_reliability_native/project-tme-geometry-ed725991ccab/workspace/artifacts/programs/program-769ed3f0a846a8e7cd73`.
The plotted line joins retained JuTrack samples; it does not establish unsampled
interior extrema or a new bend-centre measurement.

The separate untouched welcome-prompt trial is
`research_runs_fresh_release_20260915/project-c9771461ff74e8`, using configured
`deepseek-flash`, local JuTrack and disabled research internet access. Its source,
configuration, request, code provenance, failures and repairs are recorded in the
trial directory. It **did not establish default-task completion**. At
2026-09-15 06:00:41 UTC, its pre-follow-up audit was preserved and the source freeze
was deliberately lifted to fix observed application defects. Subsequent work is
diagnostic, not an untouched release trial. Source-freeze checks remain strict;
the observer correctly rejects it for release qualification. No model instructions,
candidate files or saved research programs were manually edited, and no numerical
job was cancelled.

The later automatic-repair worker exited during the hot edit with an import error
for a newly introduced helper; that is a mixed-code-version deployment failure,
not a JuTrack or cold-start test result. The read-only observer now identifies its
dead process and reports **failed**, rather than waiting indefinitely on stale
"running" metadata. For normal upgrades, let active calculations finish and restart
the application to load one consistent code version; do not hot-edit a qualification
run. The archived pre-follow-up audit preserves what was observed before that change.

### What the conversation test actually established

- The model generated and executed a real 500-particle, 80-generation unit search:
  40,500 optimizer evaluations, with a verified best unit-reference emittance of
  about 113.49 pm for its proposed 3 GeV, 1.8 m unit. This is not an MBA emittance.
- Its fixed terminal hypothesis did not match: 64 starts, roughly 151,000 callbacks,
  ID dispersion 55.95 mm, and failed alpha/slope tolerances. Every recorded numerical
  Jacobian had rank 4. This does **not** demonstrate rank deficiency, insufficient
  space, or failure of the conventional design route.
- It then assembled using the originally published terminal rows, not the returned
  attempted match, and called the terminal "matched" despite failure. The full
  cell was vertically unstable. Acceptance remained blocked. This is an unresolved
  scientific orchestration/handoff failure, not a successful design.
- The run exposed a missing plot after cache eviction, an unsupported reporting
  field, unavailable-value conversion, and a missing checkpoint filename obscured
  by error-text unescaping. The application fixes above address these interface
  classes; they do not establish that the next autonomous design will succeed.
- The observer now distinguishes synchronous execution failures, internal recovery,
  preflight rejection, optional reporting failure, model output-token limits and
  physical rejection. It recognizes flat/nested optimizer checkpoints without
  double-counting sidecars. Initialization screening and finalist refinement are
  not silently included in the PSO evaluation count.

The release blocker remains a **fresh, unattended, accepted complete-cell and
explicit-ring result**, not another isolated tool test. A changed-source diagnostic
or a historical matched witness cannot substitute for that proof.

## Repeating the release check

Use a new empty root and the configured interpreter:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/validate_default_task.py start --root research_runs_new_release_trial
.\.venv\Scripts\python.exe -X utf8 scripts/validate_default_task.py watch '<project path>' --seconds 60
```

The observer never cancels jobs. Do not edit application/expert/native source,
inject instructions, copy candidates or change models during a fresh trial.
Ordinary internal automatic repair is allowed and counted.

After the selected result exists, use the already installed local Playwright
package with `tests/scientific_history_ui.cjs PROJECT_ID SCREENSHOT.png PROJECT_ROOT`.
The script writes a screenshot and `.png.json` receipt with selected result identity,
loaded images, real tables, duplicate/error/network checks. Pass that receipt to:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/validate_default_task.py audit '<project path>' --verify-jutrack --browser-receipt '<screenshot.png.json>' --output '<audit.json>'
```

Keep browser-only environment overrides scoped to that browser process; do not
change the model/runtime configuration of the trial. A missing browser check is
pending, not a rendering pass. Inspect the figure and scientific narrative manually
as well: loaded pixels do not establish physical meaning or useful live-update latency.

Release claims must separate a first linear candidate, optimization quality,
chromatic/nonlinear performance, and operational robustness. This test does not
establish new accelerator physics, global optimality, or broad autonomous capability.
