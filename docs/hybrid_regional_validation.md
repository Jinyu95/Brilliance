# Global H-MBA regions and grouped geometry: offline validation

Date: 2026-09-09. Source: project `project-5469713d825bee`, original conventional
`workspace/artifacts/programs/program-385b9dd9ced314d70cb0/full_ring.json` (3.5 GeV, 32 cells).
The source JSON and project history were not modified. No internet or model API was used.

## Recorded failure versus actual physics evidence

The latest 23-coordinate geometry script stopped because population 60 was below the configured
minimum 230. It did not test geometry optimization. The main construction audit already compares
complete-cell mirror partners; unit-name-based selection/freezing was the remaining orchestration
error. The new tools inventory physical regions and explicitly allow former TME units to lose
their internal symmetry while retaining whole-cell reflection.

All new scientific implementations live in the MBA expert module, not the application core.
The original TME and conventional-MBA construction route remains unchanged.

## Tested experiment

The test used three independent structural coordinates:

| Coordinate | Initial | Selected |
| --- | ---: | ---: |
| B1–B2 inter-bend length, one side | 3.975754 m | 3.974929 m |
| B2–B3 inter-bend length, one side | 2.130905 m | 2.102485 m |
| B1/(B1+B2) bend-angle fraction | 0.333333 | 0.339143 |

The length changes were funded by named non-ID terminal drifts `LEFT:d4`, `LEFT:d3` and their
whole-cell mirrors. Neither the ID straight nor the core supplied length. The angle transfer
included the corresponding outer bend-length allocation, preserving the total pair length and
angle. Core magnet geometry and strengths were frozen. Two independently owned bump-region
quadrupole pairs restored ID dispersion with the existing JuTrack response matcher.

Core dispersion, maximum core H and absolute one-cell core I5 could not exceed the original
baseline, apart from a 1e-8 relative numerical allowance. MOPSO minimized separate bump deficit,
core dispersion and core I5 objectives; no compensating weighted sum was used. Per-bend I5
comes from JuTrack `elrad`, with integral sums checked against `ringpara`, not sampled H estimates.

150 proposed seeds were screened to obtain 30 feasible production seeds. Two JuTrack workers
completed six generations, keeping 30 feasible survivors at every generation. Total elapsed
time was **66.96 seconds**, including startup and screening. This is a short commissioning
experiment, not a convergence or exhaustion test.

## Same-candidate comparison

| Observable | Conventional baseline | Selected geometry candidate |
| --- | ---: | ---: |
| Left/right peak dispersion | 21.6372 / 21.6372 mm | 22.0663 / 22.0663 mm |
| Maximum core dispersion | 21.2321 mm | 21.1721 mm |
| Maximum core H | 3.83676e-5 m | 3.64627e-5 m |
| One-cell core I5 | 1.54598e-8 m⁻¹ | 1.51711e-8 m⁻¹ |
| Outer/core contrast | 1.01908 | 1.04223 |
| Natural emittance | 36.2714 pm | 41.7873 pm |
| Maximum beta | 254.475 m | 129.738 m |

This demonstrates a functioning grouped-geometry search with improving bumps and protected
core H/I5. **It does not produce a completed H-MBA**: the requested shape gate still fails.
Total emittance increased even though core I5 decreased, so outer-region radiation remains
important. No candidate was promoted or substituted for the saved working lattice.

Artifacts are in `research_runs_hybrid_regions_check/`: `contract.json`, `baseline.json`,
`optimization.json`, `checkpoint.json`, `final_metrics.json`, `final_shape_gate.json`,
`candidate.json`, `optics.png` and `summary.json`.

## Reproduce

Use the configured interpreter and a new output directory:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\validate_hybrid_regional_search.py --candidate research_projects\project-5469713d825bee\workspace\artifacts\programs\program-385b9dd9ced314d70cb0\full_ring.json --output research_runs_hybrid_regions_repeat --generations 6 --workers 2
```

The reusable interfaces and worker contracts are documented in
`expert_modules/mba_design/references/hybrid_regional_api.md`. Population size is determined
from the three independent outer coordinates (minimum 30), not the number of changed elements
or inner matching variables. Geometry and objective tools do not choose an autonomous route
or contain a saved lattice template.

Verification: 69 focused tests passed; Ruff passed. The full suite returned 691 passed,
two skipped and one configuration-dependent failure: an existing assertion expects
`candidate_wall_time_s == 600`, while the current user configuration specifies 3600.
Neither the runtime setting nor that unrelated assertion was changed by this work.
