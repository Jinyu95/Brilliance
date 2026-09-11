# Modified-TME construction validation

The September 8, 2026 investigation found that the agent skipped an actual unit solve, used
the full achromat angle in its reference call, hand-filled the length budget, and searched
diagnostic strengths outside its own engineering limits. More commentary did not establish
scientific progress.

The updated MBA expert note supplies a constructive unit method, not a prebuilt lattice.
Two physics-neutral API additions support it: `symmetric_cell_elements` (one full central
element, or a between-element symmetry plane) and `audit_lattice_constraints` (geometry,
symmetry, and energy-dependent hard-edge field bounds). They do not design a 7BA or impose a
state machine on other topics.

## Offline numerical method experiments

Both experiments sampled a fresh feasible population, then searched five geometry/strength
variables with 64 particles and 40 PSO generations. No saved lattice or solved magnet
setting initialized either experiment. The field envelope used 1.2 T dipoles, 0.8 T
quadrupole pole-tip field and 20 mm quadrupole radius. No tune target was imposed.

| Quantity | 7BA allocation experiment | 6BA allocation experiment |
|---|---:|---:|
| Energy | 3.5 GeV | 3.0 GeV |
| Intended MBA periods | 32 | 32 |
| Intended circumference | 800 m | 800 m |
| Reserved straight / both matching regions | 5 m / 4 m | 5 m / 4 m |
| Interior-unit budget | 3.2 m | 4.0 m |
| Unit angle | 32.724923 mrad | 39.269908 mrad |
| Analytical homogeneous TME scale (Jx=1) | 13.5556 pm | 17.2095 pm |
| Best initial sampled unit emittance | 53.1892 pm | 53.8204 pm |
| Final unit-reference emittance | 41.2894 pm | 52.1868 pm |
| Final / analytical scale | 3.0459 | 3.0324 |

The selected lattices passed explicit repeated-unit full-ring optics/radiation checks, positive
damping checks, engineering limits, and periodic closure. A separate half-bend diagnostic
samples the physical bend centre and checks map/radiation agreement with the unsplit definition.
One-cell combined optics/lattice plots include this centre sample rather than drawing a flat
line between the two bend ends.

These numbers are **unit-reference results, not 800 m MBA designs**. The repeated-unit ring
exists to close the bend sum for radiation validation. Neither complete-MBA achromat matching,
hybridization, chromatic correction, nor nonlinear acceptance is validated by these experiments.
The ratio near three is an observed result in these search boxes, not a universal quality gate.

## Reproduce without external research access

From the repository root, with the configured environment and `internet_access=false`:

```powershell
.\.venv\Scripts\python.exe scripts/validate_tme_method.py --output C:\temp\tme-validation
.\.venv\Scripts\python.exe scripts/validate_tme_method.py --output C:\temp\tme-validation-6ba --energy 3.0 --bends 6 --seed 29
```

The experiment preserves the actual optimizer history, reference/achieved centre-optics table,
geometry and magnet audit, selected unit JSON, closed reference-ring JSON, source snapshot,
combined optics plot, and convergence plot. Output belongs to validation, not expert-module
assets; the research agent must construct and solve its own candidate for the scientist's
contract. Do not load these output lattices as hidden seeds.

Tests cover geometry/angle errors, centre duplication, parity, symmetry, stale energy limits,
out-of-limit diagnostic scans, and retrieval of the constructive note without registering a
fixed MBA protocol. Numerical method validation and autonomous model reliability are separate
claims; a model trial must be inspected independently.

## Live model trial and integration findings

The configured DeepSeek backend was also exercised in an isolated temporary research project,
with research internet access disabled and no validation lattice supplied. The agent authored
its own unit and optimizer: the initial 3.0 m unit gave 3548.94 pm; its committed 30-generation
PSO history reached 42.0123 pm. Only 11/50 generation-zero particles were feasible: that departure from the expert
initialization method must be disclosed, not reported as compliance.

After the platform fixes and explicit recovery prompts, the agent completed a read-only
checkpoint-validation program: the explicit 192-unit, 576 m reference ring reproduced 42.0123 pm
with stable optics and positive Jx/Jy/Jz. The diagnostic bend centre gave beta_x=0.1627 m and
Dx=2.96 mm; the analytic references were 0.0816 m and 0.86 mm. Four tables, a combined unit PNG,
the selected unit, source, and a final narrative were published. This was **guided recovery**,
not a clean unattended pass. The agent repaired a `beta`/`beta_x` table typo itself.

Remaining presentation/method-adherence issues in that run: the figure used only unsplit exit
samples (flattening the bend interior), the narrative counted 31 history rows as 31 generations
instead of generation 0 plus 30 search generations, and the no-departure statement described
recovery but failed to disclose the original partly infeasible population. The note/reference
now clarifies the first two issues; they are not grounds for claiming universal model reliability.
The printed zero tracking-based U0 is also not evidence of zero physical radiation loss; this
unit campaign did not validate radiation-enabled tracking or damping times. Local JuTrack's
`ringparams.jl` obtains U0 through `tracking_U0`, while the emittance calculation uses integrals.

The live trials exposed integration defects independently of the numerical method:

- A reference snippet used `repetitions` where the installed radiation API requires `periods`.
  The documentation is corrected; saved-program preflight and documentation tests now check
  explicit public calls against installed signatures.
- An anonymous broad exception hid the signature error as thousands of infeasible trials.
  Preflight rejects this pattern; programming errors must remain visible.
- The fast optics profile lacked `s_m` and element names, so the combined plot rejected it.
  Fast profiles now use the same START/element-exit positional format as ordinary evaluation,
  with a direct plotting regression test.
- Follow-up turns repeated help lookups instead of recovering the calculation. Duplicate
  help results are bounded, exact installed signatures are supplied, and unavailable tool
  calls are rejected. This is a guard, not proof that every backend recovers intelligently.
- `read_optimization_checkpoint` provides a read-only public result view for postprocessing
  after a plotting failure, without restarting PSO or guessing private checkpoint structure.
- The earlier keyword-driven optimizer-first execution gate classified “do not rerun PSO”
  as an optimization command and blocked the validation program. That gate and its keyword
  classifier are removed. Numerical optimizer-evidence checks now apply to programs that
  actually invoke an optimizer, not every program in a conversation mentioning optimization.

Do not describe these trials as validation of complete MBA construction, or as proof of
unattended end-to-end model reliability. Keep numerical correctness, method adherence, and
successful presentation as separate acceptance checks.

## Follow-up: first-use background failure

Project `project-569762dab2f0b3` failed before PSO evaluated a particle: the model supplied a
name-to-bound mapping but the numerical API expected a pair sequence. Named bounds and named
initial positions are now normalized against explicit variable order, with exact-key checks.
Static malformed container shapes are rejected in source preflight; dynamic values remain subject
to runtime validation rather than speculative linting.

The exact archived program was run without source changes in an isolated validation project.
It completed 25 PSO generations and 780 evaluations, producing a feasible but poor approximately
819.6 pm fixed-geometry unit. This proves execution of that program, not design quality.

The separate background-failure lifecycle was also tested with the configured model: an injected
dictionary-instead-of-scalar argument failed, queued a repair turn automatically, and was corrected
by the model. The repaired program published the requested engineering-envelope values, table,
and final narrative with no additional user message. No Julia simulation was needed or claimed
for that engineering calculation. Full tracebacks stayed in diagnostic files.

A fresh test of the app's unchanged default prompt completed without an execution crash and
presented its first computed unit (approximately 2897 pm), explicitly calling it a poor seed and
not a completed MBA. It did not proceed through complete-ring optimization. This remains a
method-execution/adherence limitation, separate from the repaired optimizer input and background
recovery mechanisms.

Recovery is bounded per recorded scientist request, ignores infrastructure/physics failures,
checks cancellation and superseding requests, and refuses repeated unchanged failing source.
`BRILLIANCE_AUTO_REPAIR_ATTEMPTS` configures the budget (default 2, maximum 3, 0 disables).
