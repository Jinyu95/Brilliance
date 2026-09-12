# Radiation-aware periodic hybrid continuation — experimental

## Status and purpose

This is a new **fixed-topology numerical hypothesis**, not a demonstrated H7BA
conversion and not a replacement for full-ring/nonlinear acceptance. Its Python
control and constrained-optimization tests use explicit toy callbacks, not JuTrack.
The JuTrack adapter still requires numerical validation on an exact saved parent.
Do not claim a successful accelerator design from these software tests.

The legacy `explore_hybrid_mba` / `HybridMBAOuterReplacement` route reconstructs
outer sections and initially treats emittance as diagnostic. It remains available
for archived programs and explicit alternative experiments. The new route instead
starts from the unchanged accepted periodic cell and includes radiation limits and
an emittance objective throughout the shape continuation. It never guesses input
Twiss, requires a dipole-to-dipole -I map, or preserves a parent tune by equality.

## Registered entry point

```python
mba = api.expert_module("laboratory.mba_design")
result = mba.match_periodic_hybrid(parent, **configuration)
```

`parent` is the exact `LatticeIR` loaded from its saved JSON, not parameters copied
from an HTML report. No TME generation is run. No working-lattice pointer is changed.

Required configuration keys:

| Key | Meaning |
| --- | --- |
| `periods` | Positive integer; actual cell angle times periods must be 2*pi |
| `slots` | Exactly SF_L, SD_L, SF_R, SD_R, each an ordered list of existing physical drift/marker names |
| `minimum_slot_length_m` | Minimum finite installation length for each slot, including declared clearances |
| `limits` | Same engineering envelope used to accept the parent, including max_abs_k1_m2 |
| `minimum_magnet_length_m`, `minimum_drift_length_m` | Same construction minimum lengths as the parent contract |
| `max_sample_step_m` | Declared dense-profile sampling step in m |
| `contract` | Physical requirements below, all explicit |

Slots must be disjoint, contiguous field-free intervals strictly between the two
outer bends, and globally mirrored. A slope-zero drift plateau is valid. There is
no artificial isolated-peak condition inside a field-free interval. This first
implementation does NOT split drifts or add magnets automatically. Missing physical
space is a preflight failure; reserve it explicitly in a separately audited edit.

`contract` contains:

| Key | Units / definition |
| --- | --- |
| `slot_dispersion_goals_m` | Mapping from all four slot names to positive final minimum Dx, m |
| `core_dispersion_goal_m` | Final cap on max(abs(Dx)) from B3 entrance through B5 exit, m |
| `core_h_goal_m` | Final sampled max-H cap over that core, m |
| `max_emittance_pm` | Absolute natural-emittance ceiling in pm rad, active at ALL steps |
| `max_energy_spread` | Relative rms energy-spread ceiling |
| `max_beta_m` | Both-plane beta ceiling in m, active at ALL steps |
| `max_slot_slope` | Final maximum absolute Dpx within each slot |
| `id_dispersion_tolerance_m`, `id_slope_tolerance` | Both physical ID-boundary tolerances |
| `stability_trace_margin` | Require abs(trace(M_plane)) <= 2 - margin, 0 < margin < 2 |
| `dispersion_scale_m`, `h_scale_m` | Positive normalization scales for shape inequalities; not validity limits |

This uncoupled positive homogeneous-bend route uses the existing physical role
model. LGBs, reverse bends and combined-function bend architectures are not added.
Use values from the scientist's accepted contract, not invented numerical defaults.
If the parent already exceeds a machine cap, the routine stops; it does not silently
loosen the cap. Core/slot shape requirements start at the measured parent values.

## Numerical operation

1. Check construction, global mirror symmetry, ring angle and actual slot geometry.
2. Build one K1 variable per GLOBAL mirror pair, including core quadrupoles. No
   within-unit ties, duplicate ownership or secondary overwriting matcher.
3. Measure and independently replay the parent. `end_fraction=0` stops after this
   identity gate, without calling an optimizer.
4. Minimize `epsilon_x / epsilon_parent` with SLSQP subject to separate signed
   inequality slacks. Emittance is calculated by existing JuTrack periodic radiation
   helpers, not by a surrogate or a contrast ratio.
5. For lambda in [0,1], interpolate each slot's minimum dispersion and the core Dx/H
   caps from measured parent values to the declared goals. A slot slope that is
   initially too large is continued to its final bound too. Other machine limits
   remain fixed. A relative emittance budget must be converted ONCE from the original
   parent to the absolute `max_emittance_pm`; never compound it step after step.
6. Re-evaluate selected positions with a NEW editor/sampler. Solver success alone is
   insufficient; named physics slacks and replay consistency must pass.
7. On failure, return to the last accepted position and halve the target step. On
   success, advance and optionally grow the step to its configured maximum. Failed
   candidates never become warm-start parents.

`settings` optionally configures the documented `HybridContinuationSettings`
fields: initial/minimum/maximum step, trust fraction of each physical bound range,
max_attempts, max_iterations, max_evaluations_per_attempt, finite_difference_step,
optimizer_ftol, feasibility_tolerance, replay_rtol, replay_atol. These are explicit
numerical settings with defaults visible in source. No default shape/field/energy
or emittance limits are supplied. `end_fraction` defaults to 1.

Callbacks `progress(record)`, `checkpoint(state)` and `cancelled()` are optional.
Checkpoints contain the accepted coordinates, immutable parent measurement and
absolute physical contract. This first version does not automatically resume from
a checkpoint or release geometry. Never restart a failed campaign by rebasing its
emittance budget to a worse candidate. A stalled fixed-topology experiment is NOT
proof of physical impossibility; later geometry changes need their own contract.

## Result and acceptance

Results have schema `radiation_aware_hybrid_continuation/v1`, status, fraction,
position, parent_measurement, measurement, accepted_steps, trial records and a
candidate_lattice dictionary. All statuses keep `request_satisfied=False`.

- `identity_passed`: unchanged parent reproduced, no shaping attempted.
- `continuation_targets_met`: requested intermediate endpoint reached under this
  radiation/shape contract. This is NOT final H7BA validation.
- `continuation_stalled` / `attempt_budget_exhausted`: last accepted point retained;
  report the achieved lambda and named failed constraints. Do not call it saturation.
- `cancelled`: last accepted point retained.
- API/schema/replay failures raise errors. They are not silently labelled infeasible.

Before a final hybrid claim, evaluate the exact candidate using the existing
`evaluate_hybrid_mba_shape(..., stage="full_hybrid")`, full-ring JuTrack radiation
and optics, and a finer sampling-convergence check with declared tolerances.
Then correct chromaticity at the reserved slots, check the response conditioning
and required strengths, and test sextupole phase/nonlinear acceptance. High slots
and low core dispersion alone do not establish chromatic or injection performance.

## Offline reproduction

Use the configured Brilliance Python environment and an exact source lattice:

```text
python scripts/validate_periodic_hybrid.py --source <accepted_lattice.json> --config <declared_config.json> --output <new_directory> --identity-only
python scripts/validate_periodic_hybrid.py --source <same_accepted_lattice.json> --config <same_declared_config.json> --output <another_new_directory>
```

The script records source integrity, configuration, accepted checkpoints and the
candidate, without model calls or promotion. Supply actual drift names; placeholder
names or rounded values from an HTML table cannot serve as a numerical fixture.

## Sources and engineering status

Yi Jiao, *Improving the nonlinear performance of the HEPS baseline design with
genetic algorithm*, Chinese Physics C 40, 077002 (2016), DOI
10.1088/1674-1137/40/7/077002, https://arxiv.org/abs/1509.09275:
supports considering emittance together with chromatic-correction/nonlinear
performance. It does not validate this solver or establish these continuation goals.

SciPy official SLSQP documentation,
https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html:
supports the optimizer interface and stopping/finite-difference options. The route,
trust bounds, immutable caps, continuation and replay policy here are engineering
choices to test, not results claimed by that documentation or the HEPS paper.
