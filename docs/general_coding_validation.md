# Scientific coding and MBA cleanup validation

Validated locally on 2026-09-14 using the configured Python and JuTrack.
This is a developer validation record, not additional agent guidance or a
claim that autonomous first-use reliability is solved.

## Current boundary

The agent writes project-specific builders, residuals and objectives. The
application provides generic numerical tools, fixed-runtime execution, progress,
plots and exact-state evidence. Expert modules provide physics and measured checks.

Eight obsolete MBA design controllers, their compatibility exports and associated
guides/validation files were removed: 41 text files, 9,173 lines, plus 19 obsolete
bytecode files. Exact recovery copies are outside the repository at
`C:/Users/Jinyu/Desktop/brilliance_cleanup_backups/mba_legacy_removed_20260914`.
The active MBA module has three references and ten numerical/measurement exports.
Saved research projects and their generated source were retained; programs using
removed controllers need explicit adaptation before replay.

The default fresh conventional route is measured TME/M-TME initialization,
independent terminal matching, then complete-cell diagnosis. Seven bends and stable
tunes alone do not establish that route. Independent acceptance measures the actual
unit, its transfer into the initial assembly, and the exact final periodic/full-ring
candidate. Unit emittance and tune are not frozen during subsequent optimization.
This validator covers the unit-and-terminal route, not every possible construction.

## Native evidence replay

The saved developer-authored 177-line experiment was replayed without regenerating
its seed or rerunning optimization. It uses general tools, not a stored facility
lattice or deleted controller. This is separate from the configured-model chat test.

| Quantity | JuTrack result |
| --- | ---: |
| Energy / cells / bends per cell | 3.5 GeV / 32 / 7 |
| Circumference / cell length / straight | 800 m / 25 m / 6 m |
| Analytical homogeneous-bend TME reference | 13.5556168031 pm rad |
| Measured repeated-unit emittance | 42.8202032551 pm rad |
| Measured full-cell/full-ring emittance | 37.8395802432 pm rad |
| Sampled maximum beta | 143.0168 m |

Candidate `lattice-f1ebcffb02a4d5ac` passed the independent unit/assembly/final
evidence checks. Per-bend I2/I4/I5 sums reproduce native JuTrack radiation.
The periodic-cell per-bend API now expands only explicitly declared, validated
repetitions; it does not guess a ring from its name.

No beta ceiling was specified. High beta remains a limitation; neither chromatic
correction, nonlinear acceptance, robustness nor optimality is certified.
See [exact replay evidence](../research_runs_conventional_route_cleanup/evidence_replay/validation.json)
and [original project-written experiment](../research_runs_general_coding_validation/workspace/artifacts/experiment/program.py).
These are local developer artifacts, not expert seeds.

## Configured-model chat test: not a pass

An isolated project used the original plain-language 3.5 GeV/32-cell/7BA prototype
request, with internet research disabled and the configured DeepSeek backend.
It read the TME guide and wrote a real unit-solving program rather than a guessed
complete seven-bend layout. However, it did not complete the requested prototype.

The trial exposed API-help bloat, a generated geometry-key typo, an obsolete
optimizer-progress guard on partial programs, and a generated postprocessing
`KeyError: 'tune_x'`. Trailing native warnings hid that last exception from
automatic recovery. Application fixes reduce irrelevant API help, allow partial
calculations without optimizer evidence, and retain the actual Python exception
for bounded repair. Raw diagnostics remain archived.

One operator resume followed an application fix; no generated science program was
manually patched. This is a recovery investigation, not an unattended cold-start
success. The last failed model program has not been rerun after the final
error-extraction fix. No new physics result from that failed trial was accepted.
See [trial record](../research_runs_conventional_route_cleanup/trial.json).

## Checks

- Full Python suite: **524 passed, 3 skipped**.
- Opt-in conventional and chromatic JuTrack tests: **27 passed**.
- Knowledge surface: three references, ten exports, no legacy aliases or stored seeds.
- Ruff on changed components and `git diff --check`: passed.
- No browser end-to-end success is claimed.

Restart Brilliance to load the changed code and knowledge. Do not interpret these
component and numerical checks as proof that the configured model can already
complete every conventional, hybrid or nonlinear research request.

## 2026-09-14: native matching boundary review

The conventional TME-to-MBA route is retained. The reusable
`api.match_jutrack_line` now supplies native JuTrack open-line propagation,
explicit eight-component incoming optics, per-observable exit tolerances,
engineering/geometry checks and sampled beta limits. Normalized numerical
coordinates and residuals belong to the solver; no application transport maps
replace JuTrack. A fresh native replay determines boundary-match success.
Solver termination, boundary matching and complete-cell acceptance remain
separate outcomes. Failed checks override affirmative result prose and advice
to advance to dependent stages; the author's original wording is archived.

Read-only replay of the latest project's **later v4 candidate**, distinct from
its rejected 24.14-mm-dispersion v2 attempt:

| Quantity | Native JuTrack measurement |
| --- | ---: |
| Candidate | `lattice-6ec94260c115a8cb` |
| Circumference | 800.0 m |
| Full-ring emittance | 4257.666374530333 pm rad |
| ID horizontal dispersion | 1.5561354234285518e-7 m |
| Sampled maximum beta | 62.7979941012377 m |
| Periodic-cell and full-ring transverse stability | Passed |

Source: `research_projects/project-f8658559cdc85b/workspace/artifacts/programs/program-7b4d2a36b647f836daee/cell_7ba_v4_prototype.json`.
The new native line evaluator also closes this exact saved cell back onto its
measured periodic incoming optics. No seed was regenerated, optimization run,
or research state modified by this replay. It demonstrates achromatic linear
transport, **not** a successful low-emittance design or complete conventional
acceptance. Poor emittance persists; insufficient terminal space is not a
demonstrated cause. The v4 evidence-subdirectory archiving failure was repaired
without relaxing same-run provenance or cross-project isolation.

Updated checks: **767 passed, 8 skipped** in the default Python suite. The
opt-in line-evaluator/matcher suite additionally passed **18 tests**, including
five live-JuTrack regressions: both quadrupole representations/sign behavior,
bend-generated dispersion, explicit incoming state, and recovery of a known
native boundary after perturbing three quadrupoles. Boundary recovery took
33 native callbacks; it is not an autonomous MBA design demonstration.
Ruff on changed components and `git diff --check` passed. A fresh end-to-end
model-generated MBA run has not been certified by these component tests.

## 2026-09-15: saved unit through the real worker and app publication

This test closes the earlier component-only validation gap. It launches the
configured Python/Julia environment through `run_research_program(..., durable=True)`
and `brilliance.program_runner`, then validates the normal saved result through
the application's `validate_research_program` interface. Internet research is off.

The [integration program](../tests/fixtures/native_mba_worker_program.py) and
[test](../tests/test_native_mba_worker_handoff.py) execute:

1. Load the exact saved 95.339891-pm unit from `project-ab41b8de97b4c4` and measure
   its eight-component exit state with JuTrack.
2. Recover that project's actual terminal as a regression witness, perturb its
   four quadrupoles by 1%, and rematch using JuTrack. This is not a fresh broad
   design campaign or an expert-module seed. Exact terminal results checkpoint
   before final progress formatting.
3. Assemble the exact returned terminal, its mirror, and five unchanged units.
   Save the 77-element cell and checkpoint its periodic evaluation.
4. Construct and independently evaluate the explicit 32-cell, 2,464-element ring;
   checkpoint the candidate and native receipt before presentation.
5. Check the ID achromat, all unit interfaces, engineering bounds, sampled beta,
   positive damping and per-bend radiation reconciliation.
6. Publish tables, a combined one-cell optics/lattice plot, and a normal app result.
   A second worker performs the same numerical path and intentionally raises an
   `AttributeError` inside optional reporting. It returns a validated **partial**
   app result; exact numerical artifacts remain readable and receipt-verified
   after the process exits, with no matching or evaluation during recovery.

The first fixture run exposed a test setup mistake: two temporary projects reused
one project ID and collided at the configured Julia junction. The fixture now
uses distinct project identities, as the application does. No runtime paths,
physics limits or original research artifacts were changed to make it pass.

### Measured results of this integration replay

| Quantity | JuTrack result |
| --- | ---: |
| Saved unit length / reference repetitions | 2.72 m / 192 |
| Unit reference-ring circumference | 522.24 m, not the requested ring |
| Unit reference-ring emittance | 95.33989106199819 pm rad |
| Complete MBA cell / explicit ring length | 25 m / 800 m |
| Explicit ring emittance | 87.7088713473077 pm rad |
| ID Dx / Dpx | 9.62e-16 m / 5.01e-18 |
| ID beta x / y | 6.233889 m / 3.135435 m |
| Sampled maximum beta | 14.715753 m |
| Full-ring fractional tunes x / y | 0.35761931 / 0.29842067 |
| Terminal recovery callbacks | 365 |

All 16 declared integration checks pass, including the original 0.1-mm ID
dispersion tolerance (not the failed script's unnecessary 1-mm relaxation).
The beta ceiling of 100 m and minimum lengths of 0.15 m for magnets / 0.05 m for
drifts are the saved prototype hypotheses, not newly attributed user requirements.
Element-prefix samples are native JuTrack calculations; beta extrema are sampled,
not claimed to be an exact continuous bound.

Per-bend contributions below are grouped over the 32 identical periods and listed
in the saved cell's order. That cell starts at a unit/terminal interface, not the ID.

| Saved bend | Angle (mrad) | Mean H (m) | I2 fraction | I5 fraction |
| --- | ---: | ---: | ---: | ---: |
| T:B | 16.362462 | 7.89426e-5 | 6.13595% | 2.32141% |
| R:T:B | 16.362462 | 7.89426e-5 | 6.13595% | 2.32141% |
| C001:U:B | 32.724923 | 1.58635e-4 | 17.54562% | 19.07144% |
| C002:U:B | 32.724923 | 1.58635e-4 | 17.54562% | 19.07144% |
| C003:U:B | 32.724923 | 1.58635e-4 | 17.54562% | 19.07144% |
| C004:U:B | 32.724923 | 1.58635e-4 | 17.54562% | 19.07144% |
| C005:U:B | 32.724923 | 1.58635e-4 | 17.54562% | 19.07144% |

Native `elrad` sums for I2/I4/I5 agree with `ringpara`. The complete precision,
I4 terms, boundary residuals, gate values and receipts are in the
[numerical checkpoint](../research_runs_native_matching_handoff/project-native-handoff-7e56d05a1b46/workspace/artifacts/programs/program-8abc20456ace441a6e17/numerical_checkpoints/79a185a5d2c04d598b88e5ee6b3ab225.json).
See the [normal app result](../research_runs_native_matching_handoff/project-native-handoff-7e56d05a1b46/workspace/artifacts/programs/program-8abc20456ace441a6e17/result.json),
[combined cell plot](../research_runs_native_matching_handoff/project-native-handoff-7e56d05a1b46/workspace/artifacts/programs/program-8abc20456ace441a6e17/complete_cell_optics.svg), and
[intentional-reporting-failure result](../research_runs_native_matching_handoff/project-native-handoff-23d0f1c0cb4b/workspace/artifacts/programs/program-6013c4212136f7fb6ed1/result.json).

The saved unit still does **not** attain exact TME: its bend-centre beta x and
dispersion are 4.81 and 6.90 times their analytic references. Several generated
geometry bounds are active. It was preserved, not reoptimized in this validation.
The complete ring's lower emittance is a separately measured result, not inferred
from the unit reference. No fresh model-authored design, chromatic correction,
nonlinear acceptance, robustness, or general design reliability is certified here.

Default suite: **792 passed, 10 skipped**. Native worker integration:
**3 passed** (source preflight, normal publication, reporting failure/recovery),
63.8 seconds for the two native workers and parent validations together.
To reproduce with the configured interpreter and historical inputs present:

```powershell
$env:BRILLIANCE_RUN_MBA_WORKER_TESTS = '1'
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -s tests/test_native_mba_worker_handoff.py
```

`BRILLIANCE_MBA_HANDOFF_OUTPUT` optionally selects a persistent validation output
directory; otherwise pytest's isolated temporary directory is used. The exact
historical input hashes are checked before and after execution.

## Deterministic intact-bend unit designer — 2026-09-15

This revision changes the unit solve, not the conventional unit → independent
terminal match → complete MBA route. `laboratory.mba_design.design_tme_unit`
constructs the complete eleven-element mirror-symmetric unit from explicit
geometry and one supplied intact bend. Only K1near/K1far vary within each solve.
Both opposite-sign quadrants are measured with JuTrack; connected regions of each
finite grid receive local starts before additional separated starts. Grid regions
are not claimed to be continuous feasible basins. Refinements, local termination,
unexplored regions and resolution changes are recorded. PSO remains a comparison
route with the same native objective and physical constraints.

There is no unit-boundary achromat, tune band, alpha/Dpx target or analytical-TME
equality in the objective. Native `ringpara` on the intact physical elements
supplies radiation. Returned units are independently reevaluated, including their
explicit 192-unit reference rings. Exact elements, eight-component boundary optics
and receipts are saved before optional reporting. Completed scan levels are also
checkpointed; presentation failure cannot erase their numerical results.

### Declared validation envelope, not newly approved machine requirements

The latest project's exact 2.72 m geometry was loaded read-only from
`project-ab41b8de97b4c4`. E=3.5 GeV, unit angle=2π/192, dipole field≤1.2 T,
quadrupole aperture radius=0.02 m and pole-tip field≤0.8 T were retained. The
resulting exact K1 limit is 3.4261995199999995 m⁻². The original saved strengths
were not used as optimizer starts. No historical input was changed.

**Provisional test safeguards:** sampled βmax≤100 m, |Tr(Mx,y)|≤1.99,
minimum magnet length 0.15 m and minimum drift 0.05 m. The beta cap and margins
are not represented as scientist-approved requirements. The tool requires explicit
inputs; a sampled beta maximum is not a continuous mathematical bound.

### Measured geometry and resolution comparisons

The unchanged saved geometry improves only from 95.33989106199819 pm to
91.97558694324549 pm, identically at the 25- and 49-point polished resolutions.
Thus removing the old tune cut does not by itself repair the poor geometry.
Five separately declared geometry hypotheses were then tested, each with exactly
the same total unit length and angle. No geometry coordinates varied inside the
two-strength solve, and no hypothesis was installed as a default lattice/template.

| Geometry hypothesis | Lb (m) | Lnear/Lfar (m) | d1/d2/d3 (m) | Best reference emittance, 25→49 scan (pm) |
| --- | ---: | --- | --- | ---: |
| A | 0.6 | 0.4 / 0.4 | 0.10 / 0.11 / 0.05 | 42.15230383245256 |
| B | 0.7 | 0.45 / 0.35 | 0.05 / 0.11 / 0.05 | 43.36453427691297 |
| C | 0.9 | 0.25 / 0.4 | 0.08 / 0.13 / 0.05 | 47.25929020884281 |
| D | 1.1 | 0.32 / 0.28 | 0.05 / 0.11 / 0.05 | 161.2677954868591 |
| E | 0.8 | 0.30 / 0.36 | 0.20 / 0.05 / 0.05 | 76.16947063456348 |

Geometry A was rerun through the public expert API and managed worker with
25→49→97 grids, 12 separated local starts per level and 240 proposals per start.
Its polished minima were 42.15274784670058, 42.15230383245256 and
42.06695976639191 pm. The last refinement improved another 0.2025%; this is a
resolution study, **not proof of convergence or a global minimum**. Fourteen
local refinements reached the declared strength resolution; 22 exhausted their
bounded proposal budgets. The complete scan used 19,847 unique native evaluations.
The final worker/publication regression took 40.35 s, including runtime startup,
four explicit-reference-ring evaluations, dense unit plotting and result checks.

Three independently rechecked low-emittance alternatives from that run:

| Candidate | K1near (m⁻²) | K1far (m⁻²) | Reference emittance (pm) |
| --- | ---: | ---: | ---: |
| A1 | -2.0644190467187453 | 2.967811498281248 | 42.06695976639191 |
| A2 | -2.083379135208333 | 2.9711573962499993 | 42.10383649696784 |
| A3 | -2.167026584427078 | 2.994578682031248 | 42.150940680305865 |

The opposite polarity is retained as a comparison (363.870314469745 pm), not
described as an equally good low-emittance seed. A1 has sampled βmax=22.243849 m,
unit tunes 0.4577864291/0.0302076141 and Jx=0.9998041942. The old horizontal
0.45 ceiling would exclude it. Its measured bend-centre βx=0.1455683208110493 m
and Dx=0.002772041889222161 m remain above the analytic 0.07745966692414834 m
and 0.0008181230868723418 m references. Its emittance is about 3.10× the
13.555616803141715 pm homogeneous TME scale: practical M-TME, not exact TME.

Full-precision boundaries, rows, profiles, all scan samples and native receipts:
[search evidence](../research_runs_tme_validation/project-tme-test-b4b5184d867d/workspace/artifacts/programs/program-a1dfe54cdb04fe3bb441/search.json).
Also preserved: [normal app result](../research_runs_tme_validation/project-tme-test-b4b5184d867d/workspace/artifacts/programs/program-a1dfe54cdb04fe3bb441/result.json)
and [combined unit optics/lattice plot](../research_runs_tme_validation/project-tme-test-b4b5184d867d/workspace/artifacts/programs/program-a1dfe54cdb04fe3bb441/best_unit_optics.svg).
These 522.24 m repeated-unit rings are **not the requested 800 m 7BA ring**.
The new units have not yet undergone terminal matching or complete-MBA acceptance.

### Intact-map audit and regressions

The adapter previously dropped explicitly saved bend edges/integration settings.
Version 1.3.0 passes supported native values through unchanged and rejects ambiguous
physical settings. The latest 95 pm unit has no explicit edge fields, so this bug
does **not** explain its large emittance. Scalar editors cannot silently change
constructor-only settings. No native JuTrack source or radiation formula changed.

Internal samples use private diagnostics, never the radiation lattice. Requested
centre fractions now require native suffix-map × prefix-map and eight-component
Twiss propagation to reproduce the intact bend. A1's centre check gave maximum
map error 6.72e-12 and exit-state error 1.24e-11. Unvalidated or unsupported
interior-edge handling is reported as unavailable; it cannot supply a centre
target or dense-beta certificate. Intact native emittance remains usable.

The unchanged historical unit → terminal match → complete cell → explicit 32-cell
ring → normal app result integration still passes, including intentional reporting
failure/recovery (3 tests, 54.56 s). This replay still measures 87.7088713473077 pm
for its complete ring; it is distinct from the newly optimized units above.
Default suite: **839 passed, 13 opt-in tests skipped**. Synthetic unit contracts
also cover both polarities, isolated grid regions, insufficient-start disclosure,
negative radiation rejection, full-precision checkpoint recovery and quarantined
centre diagnostics. Ruff passes on changed code.

To run the exact-geometry native regression using the configured environment:

```powershell
$env:BRILLIANCE_RUN_TME_WORKER_TESTS = '1'
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -s tests/test_native_tme_unit_worker.py
```

`BRILLIANCE_TME_TEST_OPTIONS` can provide an explicit JSON `geometry` and
`resolutions` for developer comparisons; `BRILLIANCE_TME_TEST_OUTPUT` preserves
outputs outside pytest temporary storage. These fixtures are validation-only,
not agent seed assets. No fresh-model chat, new full-MBA design, nonlinear
acceptance or general autonomous reliability is certified by this test.

## Free unit geometry — 2026-09-15 follow-up

The default unit operation is now `mba.optimize_tme_unit_geometry`, replacing
manual selection of a fixed geometry. The fixed-geometry operation above remains
a diagnostic and finalist-refinement tool; the terminal matcher and complete-MBA
route were not rewritten.

Six length families are free: Lb, Lnear, Lfar, d1, d2 and d3. Five independent
length-allocation coordinates enforce
`Lb + 2*(Lnear + Lfar + d1 + d2 + d3) = U`. Every mirror partner shares its exact
length and strength. Only U and the bend angle are fixed geometric requirements;
field-derived minimum bend length, declared clearances and magnet minima remain
physical constraints. There are no preset length fractions or arbitrary upper
length caps. The supplied bend's edge/fringe/integration fields remain intact.

The production search jointly varies five geometry coordinates and two opposite-
sign K1 values using the existing PSO, then rescans three distinct finalist
geometries deterministically. Each measured PSO winner is re-evaluated and retained
as a rescan start; a finite grid cannot silently discard it. Full native unit and
explicit reference-ring validation remains separate from optimizer termination.
All polarity comparisons remain in the detailed studies; the returned candidate
list contains the best requested number of verified units.

Native validation used the same 3.5 GeV, 2.72 m unit, angle 2π/192 and physical
limits as above. The beta=100 m and trace-margin=0.01 safeguards remain explicitly
provisional. No stored strength or saved geometry allocation initialized the
search: it screened random length-simplex/strength proposals and local variants.
All **500 initial particles were feasible**, obtained from 1,132 tested proposals;
92 had positive K1near. The 500-particle, 80-generation campaign recorded 40,500
PSO evaluations and improved its best reference emittance from **196.34055 pm at
generation zero to 37.91828 pm**. Final native strength refinement gave
**37.91815324692756 pm**, versus 42.06696 pm for the earlier manual geometry.

| Best verified geometry | Length (m) |
| --- | ---: |
| Intact bend | 0.3183824482188526 |
| Each near quadrupole | 0.2558522741288826 |
| Each far quadrupole | 0.3269197627201373 |
| Each d1 | 0.33165690854795615 |
| Each d2 | 0.236154855229133 |
| Each d3 | 0.05022497526446432 |

Its strengths are K1near=-2.9621106447101258 and K1far=3.426004616998528 m⁻²;
sampled βmax=40.49415 m. The vertical trace margin, dipole-field envelope and
far-quadrupole limit are nearly active. These are measured limitations of this
campaign, **not proof of a global minimum**. The new unit still needs terminal
matching and complete-MBA acceptance; its 192-unit reference is not the requested
800 m ring.

Durability tests now cover completed-result restoration after clearing every
process capture, no additional PSO/native evaluations on completed resume,
resuming only unfinished finalist studies, preserving population/history during
explicit extension, and rejecting changed physics/seed or corrupted archives.
`api.restore_candidate_evaluation` validates archived native receipts before
making them usable by normal checkpoint/publication interfaces; it does not
invent a new simulation result. Fast linear PSO calls use `cache_evaluations=False`
to avoid per-particle files while retaining generation checkpoints.

Default suite: **893 passed, 14 opt-in tests skipped**. Native geometry validation
uses the real managed worker/public API and normal app result validator:

```powershell
$env:BRILLIANCE_RUN_TME_GEOMETRY_TESTS = '1'
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -s tests/test_native_tme_geometry_worker.py
```

`BRILLIANCE_TME_GEOMETRY_OUTPUT` optionally preserves these validation outputs.
The regression hashes its historical input before and after running; the history
supplies only energy, total length, bend angle and native fields—not geometry or
strength seeds. It also clears in-memory native captures and forbids simulation
calls during completed-result restoration before normal publication.

Final native result: **2 tests passed in 191.88 s**, including the full production
search, finalist validation, disk-only completed-result restoration and normal
app publication. The four returned reference emittances are 37.91815324692756,
37.94017579757251, 37.9483735264247 and 37.9657953119365 pm.
See [exact geometry-search evidence](../research_runs_tme_geometry_validation/project-tme-geometry-f4e09b2aeec1/workspace/artifacts/programs/program-60ffe3f413983b387a96/geometry_search.json),
[normal app result](../research_runs_tme_geometry_validation/project-tme-geometry-f4e09b2aeec1/workspace/artifacts/programs/program-60ffe3f413983b387a96/result.json),
and [combined unit optics/lattice plot](../research_runs_tme_geometry_validation/project-tme-geometry-f4e09b2aeec1/workspace/artifacts/programs/program-60ffe3f413983b387a96/geometry_optimized_unit_optics.svg).
