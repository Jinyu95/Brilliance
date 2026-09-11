# Hybrid-conversion repair and offline validation

Validated on 2026-09-09 against the saved conventional candidate in
`project-0f385ae794211d`, artifact `program-fec036b22d22c78a0637/candidate_lattice.json`.
The project history and working lattice were not changed. No internet search or model API
was used for these diagnostic calculations. The configured `.venv` Python and JuTrack were used.

## What failed in the recorded run

The approximately 22-minute hybrid attempt did not record an optimization generation.
It is not evidence that JuTrack linear calculations or PSO generations take ten minutes.

- The module linked `references/hybrid_shape_api.md`, but the agent's `read_expert` tool
  could not retrieve it. This was an application integration regression.
- The baseline dense profile worked. Four response attempts failed only after a perturbation
  made the cell unstable; generated code ignored `stable`/`failure` and diagnosed its empty
  profile as a broken API instead.
- A later response calculation omitted reflected quadrupoles, breaking full-cell symmetry,
  and divided a negative perturbation by a positive step. Its inferred direction was invalid.
- The worker factory created a local editor but called a global helper that captured the
  parent process's live JuTrack sampler. No worker/feasibility/generation evidence followed.
- The generated long job selected a 600-second timeout and was terminated. Its proposed
  finalization also omitted the full candidate-bound hybrid acceptance report.

## Changes

Core changes are topic-independent: discoverable/pageable module references, module API
examples in `api_help`, process-factory capture checks before execution, worker and initialization
progress, unbuffered diagnostic output, and non-destructive time estimates for durable jobs.
An explicitly cancelled job can still be stopped. Missing final acceptance on an intermediate
diagnostic is shown as pending; this does not relax final-result or promotion gates.

MBA-specific additions remain under `expert_modules/mba_design/`: physical mirror-pair
enumeration, a worker-local shape evaluator, adaptive signed response with optional independently
owned inner matching, and concrete callback documentation. They contain no seed lattice and do
not choose the optimization route. The conventional unit and assembly method was not rewritten.

## Numerical evidence and limits

A two-process test on the original candidate produced independent worker PIDs and stable
baseline/trial optics. Physical-map and original-element-exit comparison errors were zero.
The first batch took about 22.7 seconds including two Julia startups and JIT; warm dense
evaluations took 28–29 milliseconds. This measures the optics kernel, not a full nested
optimization objective or a production population campaign.

The complete nested-optimization path was also exercised: 20 fully feasible particles,
one PSO generation, two independent outer mirror-pair knobs, and disjoint QM1/QM2 inner
matching pairs on two JuTrack workers. It completed in 28.21 seconds including worker
startup, with 41 successful inner matches (zero failures), 40 recorded optimizer evaluations,
and a completed generation-1 checkpoint. Final ID dispersion was -2.51e-11 m. This deliberately
small integration test is not a production H-MBA search or a convergence claim.

A separate, bounded symmetric quadrupole continuation tested controllability on the saved
cell, without modifying bend geometry or using a stored lattice template. Across two roughly
153-second experiments it produced:

| Observable | Original conventional cell | Diagnostic shaped candidate |
| --- | ---: | ---: |
| Left/right outer peak | 23.012 / 23.012 mm | 35.574 / 35.574 mm |
| Maximum core dispersion | 23.012 mm | 23.497 mm |
| Mean outer/core contrast | 1.000 | 1.514 |
| Maximum core H | 6.49e-5 m | 1.182e-4 m |
| Natural emittance | about 27.3 pm | 65.0 pm |
| Maximum beta (both planes) | 49.44 m | 149.88 m |
| Maximum absolute ID dispersion | approximately zero | 1.02e-8 m |

The final candidate has two resolved interior maxima and passes stability, symmetry, exact
length/angle, field, positive-damping and achromatic-boundary checks. It passes the experiment's
**provisional** gate (35 mm peaks, contrast 1.5, core dispersion <=25 mm, core H <=1.2e-4 m).

It **does not meet the failed run's stricter hypothesis**: contrast >=2, core dispersion <=20 mm,
core H <=1e-4 m. Core H and emittance worsened, and vertical beta rose to about 150 m. This is a
controllability witness, not a completed chromatically useful H7BA or a validated nonlinear design.
The active quadrupole bound motivates examining outer geometry and matching leverage; a single
bounded local search does not prove conventional-geometry exhaustion. Do not replace the user's
thresholds with the weaker diagnostic thresholds or promote this candidate as the requested result.

The stricter case was independently re-evaluated using all eight original threshold values,
including 2-cm sampling and peak-slope tolerance 1e-3. The four failures (both contrasts, core
dispersion, core H) were reproduced and the report validated against the exact candidate.

## Reproduce and inspect

The saved diagnostic data are outside the research project:

- `research_runs_hybrid_conversion/`: first bounded response/continuation experiment.
- `research_runs_hybrid_conversion_refined/`: refined candidate, shape report and one-cell
  combined lattice/optics plot.
- `research_runs_hybrid_conversion_helper/`: measurements through the newly exported evaluator,
  twelve physical-mirror response columns with independent inner achromat matching, and
  before/after plots explicitly labeling the failed stricter request. The warm matched-response
  test took about 1.77 seconds.
- `research_runs_hmba_parallel_validation/result.json`: independent worker measurements.
- `research_runs_hmba_parallel_validation/nested-1788926216648571900/`: complete nested PSO
  integration test, including its checkpoint and progress records.
- `research_runs_hybrid_conversion_helper/original_run_acceptance.json`: full verification
  against the original run's exact thresholds.

Use a fresh output directory when rerunning the diagnostic:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\validate_hybrid_conversion.py --candidate research_projects\project-0f385ae794211d\workspace\artifacts\programs\program-fec036b22d22c78a0637\candidate_lattice.json --output research_runs_hybrid_check --evaluations 80
```

This is a numerical diagnostic, not an end-to-end replay of the LLM conversation. It does not
certify that every provider will compose a successful hybrid search. Restart Brilliance to load
the repaired runtime; a new follow-up should use the preserved conventional candidate and the
updated expert API, not resubmit the archived defective worker program unchanged.

Regression validation after integration: 682 Python tests passed, two skipped; Ruff and
JavaScript syntax checks passed. Browser end-to-end smoke was not run because Playwright
was unavailable on the Node module path. No dependency was installed during this repair.
