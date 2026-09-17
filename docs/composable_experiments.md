# Executable experiments: implementation and validation

Brilliance now provides a reusable experiment layer rather than requiring each
research program to reimplement matching handoffs and candidate bookkeeping.
The existing TME/M-TME designer, configured JuTrack runtime, population optimizer,
expert acceptance system and saved projects remain in place.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| Expert module / project | Physical hypotheses, optional components, scientific measurements and requirements |
| `ExperimentDefinition` | Named objectives, hard constraints, residual scales and provenance |
| `prepare_length_budget` | Exact fixed/free length accounting with mirrored multiplicities and bounded projection |
| `MatchingOperation` | Exclusive inner-knob ownership and use of exact returned rows, not solver-success shorthand |
| `JuTrackCellEvaluator` | Native cell measurements; warm scalar edits and explicit rebuilds after physical constructor/topology changes |
| `LatticeExperiment` | Proposal, dependent matching, measurement and durable recording before return |
| `CandidateLedger` | Transactional exact history, feasibility-first selection, Pareto alternatives and disk-only recovery |

The public signatures and example live in
[the accelerator API reference](../brilliance/resources/accelerator_api_reference.md#composable-experiments-and-exact-candidate-selection).
The agent can discover them through the existing `api_help` tool. No new model
provider, environment discovery, internet access or external framework is needed.

This is not a fixed TME → MBA → hybrid controller. TME remains the conventional
initialization hypothesis; a complete-cell experiment may vary geometry and
matching or test optional mirrored components. The initial structural operation
replaces an explicitly named mirrored quadrupole pair with equal-length drifts.
It preserves bend count/angle and physical spans; it neither rematches nor claims
optical feasibility. Compile new knobs for changed topology and evaluate it.
An unsuccessful removal must leave the feasible parent available.

The candidate archive is `candidates.sqlite3` under the chosen project experiment
directory. Every completed trial retains exact rows, measurements, checks and
provenance before selection/reporting. Unknown quantities remain null, not zero;
optimizer infinity is internal, not emitted into scientific JSON checkpoints.
Selection reads all history, including late improvements. If no candidate is
feasible it returns `selected=None`, not the least-bad failure. Multi-objective
selection retains nondominated alternatives and explicitly reports its
lexicographic representative rule; it does not establish a unique global optimum.

`CandidateLedger.open(...)` provides read-only, hash-checked recovery without
loading the original callback or starting JuTrack. Changing requirements,
measurement settings or code requires a new contracted branch, not relabeling
old history. The LLM may still write short new measurements and experiments;
installed primitives handle the repetitive numerical bookkeeping.

## Native integration result (2026-09-15)

The real managed-worker and normal result-publication interfaces were exercised:

1. Load and evaluate a SHA-pinned historical unit (integration witness only).
2. Perturb and rematch its terminal with JuTrack.
3. Assemble the complete periodic cell at the ID boundary.
4. Record the parent and a scalar-perturbation diagnostic.
5. Remove the actual nonzero `T:Q4` / `R:T:Q4` pair, rematch and re-evaluate.
6. Select from the persistent history and independently evaluate the exact cell
   and explicit 32-cell ring; save both receipts before reporting.
7. Publish one-cell combined lattice/optics and seven per-bend radiation rows.
8. Repeat with an intentional reporting exception and recover using disk only,
   with numerical API calls disabled in the recovery test.

The selected historical-input ring had circumference 800 m, emittance
87.70887134732989 pm, ID dispersion approximately -1.13e-15 m and sampled maximum
beta 14.71575255 m. Fast cell, high-level periodic-cell and explicit-ring radiation
agreed. All eleven integration checks passed, including per-bend reconciliation.
The historical repeated-unit reference emittance was 95.34 pm; it is a different
physical object, not the emittance of the complete MBA.

The nonzero pair removal did **not** pass matching; the proposed cell was unstable.
Its exact rejected trial remained archived and the valid parent was selected.
This is a negative scientific result demonstrating safe subtraction, not a claim
that pruning necessarily improves a lattice. Both normal publication and injected
reporting-failure recovery passed (three native test cases including inspection,
59.20 seconds total).

The first integration attempt exposed JSON infinity in the adapter result. All
verified numerical checkpoints survived; disk-only recovery succeeded. The public
adapter was corrected to use nullable unavailable values, then the unchanged
native fixture passed. The runtime was frozen during each native validation.

These results validate composition and recovery, **not a fresh autonomous MBA
design, hybrid conversion, global optimum, nonlinear acceptance, or publication
readiness**. The witness is not installed as a design seed or exposed across user
projects. A fresh-user benchmark remains necessary before making those claims.

## Reproducing checks

Run the ordinary suite with the configured interpreter:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
```

The optional native regression requires its explicitly documented historical
inputs in `tests/test_experiment_native_worker.py` and the configured local
JuTrack installation. It asserts that internet research is disabled:

```powershell
$env:BRILLIANCE_RUN_EXPERIMENT_WORKER_TESTS = "1"
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_experiment_native_worker.py -q -s
```

Standalone optimizer plumbing tests use small test budgets. Production population
and generation defaults have not been reduced. A storage-only benchmark of 1,000
77-element records took 1.81 seconds to durably append and 0.61 seconds to select
on this workstation; this is not a JuTrack or optimizer performance claim.
