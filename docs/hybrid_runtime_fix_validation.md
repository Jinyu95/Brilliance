# H7BA runtime fixes from the saved pilot investigation

Base: PR #2 at `e8c2431d4e82f9e0c87aee9230a952fda76f1999`.
This follow-up changes executable application behavior, not the scientist's prompt.
No source lattice, magnetic limit, optical requirement, or working selection is changed.

## Controller behavior

- An unaccepted evaluation cutoff returns `evaluation_budget_exhausted`; a solver
  iteration cutoff returns `iteration_budget_exhausted`. Neither halves the target
  into `continuation_stalled`. No automatic increase in compute budget is hidden.
- A verified feasible point may be retained despite a resource cutoff. Its next
  target step is not enlarged on the basis of an unfinished solve.
- `has_accepted_progress`, `accepted_step_count`, and a canonical `summary` are
  independent of campaign completion. `attempt_budget_exhausted` may include a
  useful accepted candidate. It never means "no point accepted" by itself.
- Records include callback/major-iteration counts, solver status and reported
  iterations/evaluations, variable count, finite-difference stencil size, full best
  trial coordinates/measurements/slacks, and independently replayed accepted data.
- Direction checks use the declared numerical replay tolerances and name the
  offending slot or core quantity. The saved left/right SD values differing by
  approximately 1e-15 m no longer cause a false direction rejection. This does
  not relax achromat, field, or other scientific acceptance bounds.
- Identity-only calls do not impose a future deformation direction. They still
  require a complete physical contract and satisfy the current parent gates.

## Automatic app-owned evidence

A normal `mba.match_periodic_hybrid(...)` call inside a managed Brilliance program
now creates an invocation-specific `numerical_runs/periodic_hybrid-.../` directory.
No additional user callback, reporting program, or configuration file is required
for this capture. Existing explicit checkpoint/progress callbacks remain usable.

Before caller-owned callbacks run, the backend saves the complete state and exact
periodic cell (including metadata) in a new snapshot. It also saves the immutable
input contract and parent. Successful return adds the raw result and a separately
labelled explicit repeated-ring IR. Constructing that IR is NOT a new full-ring
JuTrack validation. Runtime/adapter errors retain the latest valid snapshot and
an execution-error record. A failed snapshot write cannot publish a new checkpoint
with an old candidate. File replacement uses the existing atomic I/O primitive.

`ResearchRun.finish()` validates the archive hashes and automatically adds the
numerical outcome table and downloads to the normal application result. This
works even when generated code truncates its own `data`, ignores the returned
candidate, or submits the old parent. The raw numerical branch remains separately
visible; the application does not silently override a submitted candidate or
promote any branch. Multiple invocations retain distinct candidates.

Local SLSQP execution now qualifies as optimizer execution evidence in the generic
finish check. An identity-only call does not. Existing scientific acceptance
validators and working-lattice promotion requirements are unchanged.

## Tests actually executed

Linux, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2:

```text
python -m pytest -q tests/test_periodic_hybrid.py tests/test_periodic_hybrid_profile.py tests/test_hybrid_resource_outcomes.py tests/test_numerical_evidence.py tests/test_periodic_hybrid_runtime.py
76 passed
```

The pre-existing 45 tests are retained (one budget-status expectation updated).
The 31 new cases exercise controller outcomes, the recorded SD roundoff case,
lossless arrays/deep structures, crash-safe evidence, hash/path protection,
canonical finish tables/downloads, and the actual expert-adapter-to-runtime path.
That path uses explicit accelerator API doubles, not actual JuTrack. The generic
finish tests isolate registry validation with a test double. Tests of ordinary
SLSQP use toy functions, not accelerator models.

Original production sources copied for local execution were checked against their
Git blob hashes. All changed Python files also pass a Python 3.10 grammar AST
check; that is not a Python 3.10 runtime test.

## Limits and next validation

No new JuTrack numerical campaign, full browser/application session, Windows run,
chromaticity correction, or successful H7BA conversion is claimed. Physical
feasibility of the requested hybrid optics remains to be established. No budget,
constraint, slot, emittance cap or geometry is silently changed to obtain a pass.

This patch does not implement checkpoint resume, geometry release, a two-plane
sextupole solver, or automatic parent selection. It also does not turn diagnostic
shape data into final acceptance evidence. Final full-ring, sampling, chromatic,
and nonlinear validation still use the established separate gates.

Pull this same PR branch and restart the app to exercise these changes. The
registered method name and its existing input signature are unchanged. Raw
checkpoints from past runs that were already truncated cannot be recovered by
this patch; it prevents that loss in new runs.
