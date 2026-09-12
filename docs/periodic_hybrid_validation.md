# Radiation-aware periodic H-MBA pilot: implementation and validation status

Base reviewed: `9edf4d5564c9880756d1fac0539d5f1b211f0532`.

## What changed

Added `expert_modules/mba_design/periodic_hybrid.py` and exposed
`mba.match_periodic_hybrid` through the existing expert Python API. The function
starts from an exact accepted periodic cell, varies global mirror-paired KQUAD
strengths including the core, and keeps topology, element lengths and bend angles
fixed. It reserves four explicit finite sextupole installation slots.

The search minimizes actual JuTrack natural emittance while applying separate
slot/core dispersion, core H, achromat, stability, beta, energy-spread and absolute
emittance constraints. Shape requirements are interpolated from measured parent
values. Failed target steps roll back and halve the step; selected positions are
independently replayed. No quadrupoles are owned by a second overwriting matcher.

Updated injected expert instructions and the objective-name advisory: an emittance
objective WITH active shape constraints is legitimate during shaping. Legacy
outer-DBA replacement is an alternative hypothesis, not a mandatory or validated
remedy. Legacy functions, the final shape validator and TME/conventional seed
implementation are retained. The new reference describes the distinction from
older EXPERT.md H3/H4 instructions; that historical route remains available.

## Checks actually run

Local Linux environment: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2.

```text
python -m pytest -q tests/test_periodic_hybrid.py tests/test_periodic_hybrid_profile.py
45 passed
```

The suite runs the real new Python continuation code with **toy measurement
callbacks**, not a beam-dynamics substitute. Cases include:

- zero-deformation identity without starting an optimizer;
- modest constrained continuation with an emittance objective;
- an intentionally infeasible absolute radiation cap;
- rollback to the last accepted point and non-compounding budgets;
- rejection of a good bump with enormous emittance;
- independent replay of ID and radiation measurements;
- propagation of actual API/schema errors instead of false physical infeasibility;
- bounded evaluations, cancellation and immutable callback snapshots;
- finite, disjoint global-mirror slot layouts;
- a stationary drift plateau and its analytic H invariant;
- rejection of NaN inside a profile rather than silently hiding it with max();
- guidance, registration and request-classifier compatibility checks.

All changed/new Python files were parsed using the Python 3.10 grammar. This is
not a Python 3.10 runtime test. Original copied tools.py and acceptance.py were
checked against their Git blob hashes before editing. Pushed content hashes were
checked against the locally tested files.

## Not performed / remaining gates

No Julia/JuTrack numerical run, full application or browser session, expert-registry
execution, Windows test, or Ruff check was performed. The local environment used
for these tests did not have Julia/JuTrack. No exact saved accepted lattice fixture
was included in this patch. An HTML table's rounded values are not such a fixture.

Therefore this patch does NOT establish a successful H7BA conversion or acceptable
chromatic/nonlinear performance. `continuation_targets_met` is an intermediate
numerical endpoint and always keeps `request_satisfied=False`.

The headless `scripts/validate_periodic_hybrid.py` accepts an exact saved parent
and an explicit JSON physical contract. First run --identity-only in the configured
Brilliance environment. Then run a bounded pilot with the same source and caps.
Retain failed evidence; do not weaken requirements to manufacture a pass. Check the
selected cell independently with the existing full_hybrid validator, a finer
sampling-convergence study, full-ring radiation/optics and chromatic/nonlinear
acceptance as appropriate.

This initial implementation does not automatically release lengths/angles, insert
sextupoles, select slots, perform multi-start global search, or resume checkpoints.
Those are separate capabilities; a fixed-topology stall is not topology exhaustion.
The absolute radiation budget must remain referenced to the original accepted
parent in any explicitly authorized subsequent campaign.

No main-branch changes, automatic merges, saved research-state modifications or
working-lattice promotions are part of this draft.
