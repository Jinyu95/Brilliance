# Hybrid exploration update — 2026-09-11

## Scope and findings

Inspected `project-09163d58984359`. Its working conventional cell has 7 bends, 27 m length,
32 intended repetitions and a 5 m insertion straight. The conventional seed/assembly code
was not changed. No saved project, source lattice, working pointer or existing calculation
was edited, replaced, resumed automatically or published by this validation.

The hybrid programs propagated a fixed physical half-core while varying outer K1 and centre
Twiss parameters. Core K1 was still frozen. Several later programs also failed a raw dictionary
core comparison: full-ring extraction changed provenance fields without changing the magnets.
Other attempts confused optional source optical references with mandatory design requirements.

This update is confined to the MBA expert package, its knowledge, tests and offline validation
scripts. Brilliance's general application and the working TME/conventional generation route
remain unchanged. The package is version 1.13.0.

## Implemented

- `match_outer_dba_section(..., rematch_core_quadrupoles=True)` adds every half-core KQUAD
  independently. `core_from_half` restores the unsplit physical core and its GLOBAL mirrored
  partners; no old within-TME-unit family ties are imposed. All K1 are inner-matcher owned.
- Physical core comparison ignores known provenance labels but retains geometry, bend fields,
  other multipoles and unknown physical options. Evaluation allows symmetric core K1 rematching.
- `explore_hybrid_mba` executes bounded numerical multistarts and warm continuation locally.
  Release order is strengths/centre optics, drift allocation, outer angle split, then grouped
  outer magnet lengths. There are at most eight structural coordinates. Exact cell length,
  total angle and ID length remain fixed. Unfunded drifts/dipole field violations are screened.
- Heights and intermediate contrasts are explicit, independent continuation inputs. A smaller
  initial bump does not silently tighten the core-dispersion envelope. Warm starts do not repeat
  achromat-only initialization and erase the shape already obtained.
- Callbacks provide compact progress, named matching residuals, active limits, physical-region
  knob inventories and checkpoint data. Resume validates the source and contract. Numerical
  API exceptions remain errors, not physical infeasibility or optimizer saturation.
- A line match is never sufficient acceptance. The independent complete-cell gate still checks
  actual usable drift bumps, contrast, core dispersion, beta, achromat, stability, engineering
  and damping. Emittance recovery/full hybrid quality/chromatic correction remain later stages.

## Offline JuTrack validation

All experiments used `.venv/Scripts/python.exe -X utf8`, the configured local Julia/JuTrack,
and internet disabled. There were no LLM calls in the numerical campaigns. No imported facility
lattice or other project's results were used. Trial outputs are diagnostics, not expert assets.

### Fresh outer-section search on the unchanged 27 m conventional cell

The final pilot used explicit unsolved outer connectivity with 8 outer quadrupoles per side,
6 independent core pairs and free centre beta_x/beta_y/Dx. Initial alternating strengths were
unmatched fractions of the engineering limit, not archived solutions. Geometry bounds were
declared in the experiment; the original 27 m / 5 m ID budget was retained.

Command:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/validate_hybrid_continuation.py `
  --source research_projects/project-09163d58984359/workspace/artifacts/programs/program-999dfd22a01c8f55656b/candidate_lattice.json `
  --core-start P002:Dend --core-end P004:R:Dend `
  --output <new-output-directory> --alternating-start --trials-per-stage 3 --starts 3 --max-nfev 600
```

Recorded in `research_runs_hybrid_continuation_20260911_final/summary.json` and `checkpoint.json`:

| Measurement | Result |
| --- | ---: |
| Geometry/start trials across all four release stages | 12 |
| JuTrack one-pass residual evaluations | 111,463 |
| Wall time, including runtime startup | 118.14 s |
| Search execution errors | 0 |
| Source lattice hash unchanged | true |
| Final usable-bump candidate | **not found** |

This establishes executable, low-model-overhead exploration, NOT reliable autonomous H7BA
construction. Earlier zero-strength, alternating and larger multistart scouts also missed
the declared shape within their budgets. There is no evidence to claim topology exhaustion.
Changing all K1 alone is not a guarantee of a useful hybrid solution.

### Direct core-rematch/export regression on a saved trial from the SAME project

Separately used that project's already-existing **30 m rejected hybrid trial**
`program-8cd8daf35aa1179acaf9/candidate_lattice.json`. This is not a substitute for the 27 m
source, and is not permission to enlarge its circumference. It tests real rematching/export
and the independent shape gate without requiring a fresh outer solve for the regression.

`scripts/validate_hybrid_core_rematch.py` requested a small change to the measured drift
station. Output: `research_runs_hybrid_core_rematch_20260911/`.

| Check / metric | Result |
| --- | ---: |
| One-pass match | passed |
| Core quadrupoles that actually changed | 12 (6 global pairs) |
| Core geometry, full reflection, cell budget, ring closure | passed |
| Complete-cell stability and positive damping | passed |
| Maximum absolute ID dispersion | 3.86e-11 m |
| Beta maximum | 65.79 m |
| Left / right outer dispersion | 38.11 / 38.11 mm |
| Core dispersion maximum | 25.13 mm |
| Bump/core contrast | 1.516 |
| Final requested usable-bump gate | **failed** |

The combined one-cell optics/layout plot was generated and visually inspected. A separate
60 mm shaping request on this trial also failed the independent gate; outputs remain in
`research_runs_hybrid_core_shaping_20260911/`. Neither result was accepted or published.

## Tests and remaining limitation

98 focused tests pass, including global mirror reconstruction for 5/6/7/9 bends, real ownership
of core K1, provenance-only differences, preserved nonquadrupole fields, continuation/release
order, checkpoint resume, cancellation, independent inequality checks and rejection of false
hybrid success. The broader suite has one pre-existing configuration assertion: it hardcodes
600 s while the user's runtime configuration specifies 3600 s (780 passed, 2 skipped, 1 failed).
This update does not change
that configuration or impose a new wall-time limit.

The numerical tools are more controllable and less dependent on model-written debugging loops.
The tested fresh geometry hypothesis is **still not a demonstrated successful H7BA route**.
The next scientific investigation must compare declared outer connectivities/length budgets
and independent matching basins using the recorded residuals, not merely repeat the same
campaign or claim that a stable conventional lattice satisfies hybrid shaping.
