# Machine-correction Python API

Resolve once inside the configured research worker (no interpreter discovery):

```python
import brilliance.accelerator_api as api
loco = api.expert_module("laboratory.machine_correction")
jt = api.jutrack()
```

The functions return ordinary dictionaries/arrays, never machine commands. Use
the normal saved-program/durable-job/result interfaces. The fit is marked durable.

A minimal experiment needs only a nominal ring, ORM and dispersion. This API does
not require a user-written configuration or old machine-settings export. The
research program records chosen numerical settings as outputs; see the
[minimal benchmark](loco.md#minimal-offline-benchmark). A native ring with unique
device identities needs no external index map.

Load a supplied native Julia ring without rebuilding its elements:

```python
ring = jt.Lattice(jt.include(str(ring_path)))
```

This wraps the native vector without changing its fields or topology. The
response helpers also accept that vector directly. Reuse saved validated
responses before starting the fit.

## Measurements and identity

```python
data = loco.read_loco_measurements(
    response_path, horizontal_sign=selected_h_sign,
    vertical_sign=selected_v_sign, dispersion_path=rf_scan_path,
)
data = loco.align_observations(data, model_bpm_names, model_corrector_names)
snapshot = loco.read_named_settings(settings_path)
```

The legacy JSON reader is specifically for the supplied full-span `response1`
(mm), `KickAngle` (mrad), `BPM_names`, `ResolX0/ResolY0` (mm) format. Do not use it
for an arbitrary matrix whose units only look similar. `response`, `sigma`, `mask`
have shape (2*N_BPM, N_corrector), rows all x then all y; values are m/rad.
`sigma = resolution / abs(full kick span)` is the source resolution weighting,
not a claim about the statistical error of the difference. Mandatory signs are
hardware-command to model-kick polarities. Selected signs may be provisional
offline hypotheses supported by response comparisons; human approval is not an
API prerequisite, and selection does not certify hardware polarity.
Device alignment is case-insensitive,
rejects duplicates, and masks missing devices; examine `alignment` diagnostics.
The source `GoodIndex` is a zero-based measurement whitelist, NOT lattice indices.

For already normalized matrices or direct measured Dx/Dy, use ordinary file
reading and pass numerical vectors to `fit_response_model`; do not fabricate
legacy fields or convert measured dispersion back into an RF scan to use this
reader. In either path, preserve names, units, order and exclusions.

Unavailable values are `None` with a false mask. Nonstandard NaNs in measurements
are recorded/masked, never filled with zero. `sources` holds paths and SHA256.
`dispersion.slope` is all-x-then-all-y **m/Hz**, with its own mask and raw work point.
No dispersion uncertainty is invented. `read_named_settings` preserves raw
`Time/Comments/Data`; it does not infer units or whether values are deltas.
The legacy reader conservatively masks entries with unavailable uncertainty;
this is not a prohibition on an explicitly nonstatistical benchmark. If deliberately
using numerical residual scales instead, construct a separate fit mask from valid
observations and actual device exclusions, and save that choice. Do not resurrect
bad-device/invalid measurements or relabel chosen scales as measured uncertainty.
`read_named_settings` is optional and is not needed for a nominal-ring fit.

`rf_slope_to_dispersion(slopes, rf_frequency_hz=..., slip_factor=...)` returns
`dispersion_m` and `slope_to_dispersion_hz = -f_RF*eta`. Convert only active finite
slopes; multiply uncertainties by the absolute factor. Alternatively fit RF slopes
directly by dividing predicted calibrated dispersion by this factor.

## Native JuTrack response

```python
response = loco.jutrack_response(
    jt, complete_native_ring,
    bpm_indices=bpm_exit_indices, corrector_indices=corrector_indices,
    corrector_planes=planes, energy_ev=energy_ev,
    kick_step=kick_step_rad, coordinate_step=coordinate_steps,
    orbit_tolerance=orbit_tolerance, mode="fixed_momentum",
)
check = loco.validate_jutrack_response(
    jt, complete_native_ring, response=response, columns=validation_columns,
    dispersion_step=momentum_difference_step,
)
```

All element indices here are **zero-based**, BPMs are **element exits**, corrector
planes are `"x"`/`"y"`; one native corrector can have both plane channels. JuTrack
`findelem` currently returns one-based indices: subtract one exactly once. Native
coordinates are `[x, px, y, py, z, delta]`. `coordinate_step` is a scalar or six
absolute coordinate steps; numerical defaults are starting points, not validated
tolerances for every machine. Energy is eV, unlike lattice-IR energy in GeV.

Transport is native tracking of baseline and positive/negative coordinate rays.
The helper derives periodic boundary responses algebraically from those measured
maps and finite-length native corrector kicks; it does not implement magnet maps.
It verifies closed-orbit closure and rejects losses, singular boundaries, unstable
or nonconservative maps, active radiation, RF and collective elements. It does not
remove cavities or disable physics behind the caller's back.

`mode="fixed_path_length"` adjusts delta to preserve the baseline one-turn z
increment. This is a static high-energy fixed-path approximation, **not** full RF
equilibrium. `dispersion_m` is fixed-momentum dx/d(delta),dy/d(delta) in either mode.
Native corrector coordinates are normalized canonical kicks; at nonzero dp check
the hardware angular normalization rather than assuming identity.

Key outputs: `response_m_per_rad`, `cross_response_m_per_rad` (orthogonal kick at
the same device), `dispersion_m`, `delta_per_rad`, `closure`, `one_turn_map`,
condition numbers, named/indexed devices, `metadata`. `computed=True` means a
calculation was returned, not that it matches a measured machine. The independent
validator returns maximum and relative response differences and dispersion
differences from actual perturbed closed orbits. Repeat with smaller steps; choose
acceptance tolerances from noise and the required accuracy. Bind both paths to the
same saved lattice/settings hash yourself, not just an element count.

## Calibration and finite-difference fit

`calibrate_response(response, cross_response, bpm_matrices=..., corrector_gain=...,
corrector_cross_gain=..., dispersion_m=None, energy_shift_per_kick=None)` returns
calibrated `response_m_per_rad` and, if supplied, `dispersion_m`. BPM matrices have
shape (N,2,2) with output-coordinate rows; gains have one entry per column. Cross
gain acts on the orthogonal kick at that same site. An optional extra energy
response is d(delta)/d(commanded kick); do not double-count fixed-path dynamics.

```python
# model(p) assigns ALL physical/calibration parameters on every call, then returns
# a fixed-order flattened prediction. It must use the native response above.
fit = loco.fit_response_model(
    model, initial, lower, upper, parameter_scale, difference_step,
    observed, sigma, parameter_names=names, mask=active_mask,
    max_nfev=recorded_evaluation_budget,
    progress=lambda entry: publish_progress(entry),
)
```

Vectors are one-dimensional. Flatten ORM in NumPy C order (row-major) consistently
in measurement, prediction and mask; append dispersion entries in a saved order.
`sigma`, positive `parameter_scale` and absolute `difference_step` may be scalar or
vector. Bounds/initial/names have one entry per parameter. Optional `prior_mean`
and positive `prior_sigma` must be supplied together; they add Gaussian residuals.
Remove a fixed parameter from the vector; equal lower/upper bounds are not free
coordinates. Freeze the active mask during a fit.

Outputs include exact `parameter_values`, named `parameters`, final `prediction`,
`physical_residuals`, `weighted_residuals`, `chi_square`, `prior_chi_square`,
`active_bounds`, `solver`, `callback_evaluations`, bounded `history`, and
`identifiability` (scaled data-only Jacobian, singular values, rank and condition
for small fits; explicitly not computed for large fits).
`solver.success` means numerical termination only. `max_nfev` counts solver
objective calls; each Jacobian and the independently repeated final prediction
also call the model. Use `callback_evaluations` for actual work, not just nfev.
The final model call reapplies the exact fitted parameters after derivative probes
and checks that the prediction is reproduced. A mutating callback must assign all
parameters on every invocation. On failure, the fitter attempts to restore the best
evaluated objective state (or the solver result if available), preserves the original
exception, and adds a diagnostic note if restoration also fails. A callback that
internally resets its own ring retains that behavior; the fitter cannot discover
or export hidden model state.

When `sigma` contains chosen numerical scales rather than measurement errors,
the field named `chi_square` is only a weighted sum of squared residuals, not a
statistically calibrated chi-square test.
An invalid model raises an exception; no fake penalty/zero measurements are made.
Save current fit/candidate checkpoints in the program and final results before
formatting plots; plotting failure must not trigger a new fit.

These are in-process numerical outputs, not chat payloads. Save large response
matrices and Jacobians as numerical artifacts; show block RMS, selected residual
plots, singular values and a parameter-change table in chat. A full machine's
finite-difference Jacobian can be large: estimate memory/evaluation cost from the
active observations and free parameters before fitting. The callback progress
should show completed evaluations and current residual, not repeatedly print the
whole matrix or every finite-difference probe.

### Reusable acceleration and numerical recovery

Optional `analytic_parameters=["bpm_gain", ...]` and `analytic_jacobian(p)` supply
unweighted prediction derivatives in a dense or SciPy sparse matrix with shape
`(all_observations, len(analytic_parameters))`. Columns follow the supplied names,
not the whole fit-vector order. The fitter applies masks and weights itself;
only the remaining parameters use finite differences. The derivative callback
must use its explicit parameters, not assume a preceding probe left a ring at p.

`batch_model(vectors)` supplies ordered full prediction vectors from private workers.
Pass the existing `api.parallel_evaluator(factory, workers=...)` context manager
directly; each factory builds its own model/ring and returns the same numerical
prediction function. Never distribute a shared mutable JuTrack ring to threads.
An initial worker prediction is checked against the parent model. `batch_size=16`
limits the number of finite-difference columns per batch (up to 32 probes).
Use the pool's `on_result` for live within-batch progress. Worker exceptions propagate.

Above `dense_jacobian_max_bytes` (64 MiB by default), the fitter uses disk-backed
physical derivative blocks, sparse analytic blocks and an LSMR linear solve.
`jacobian_directory` optionally locates temporary storage. Scratch files are owned
by their Jacobian operators and released when those operators are discarded; they
are not checkpoints. Large fits omit full SVD and nested-list Jacobian export:
`identifiability.computed=False`, `rank=None`, `rank_deficient=None`. This is **not**
a full-rank claim. Small-fit rank is floating-point numerical rank, not noise-based
identifiability. Increase the memory threshold only with sufficient resources.

Provide `checkpoint_path=".../fit_checkpoint.npz"` and a JSON-compatible
`checkpoint_context` containing exact model/lattice/data/code hashes. The fitter
atomically saves improving objective states and the final parameter/prediction
pair before reporting callbacks or final diagnostics. A reporting exception
therefore leaves numerical results recoverable. The checkpoint does not replace
the supplied ring, input measurements or the program defining the parameterization.

```python
saved = loco.load_fit_checkpoint(checkpoint_path)  # no simulation or state mutation
parameters = saved["parameter_values"]
# To continue fitting, pass the same problem/context and checkpoint_path with
# resume=True to fit_response_model. A changed fingerprint fails before simulation.
```

Resume is a **warm restart with a fresh solver budget**, not an exact trust-region
continuation or an additional fixed number of accepted updates. An intermediate
snapshot is the best evaluated objective, not necessarily an accepted solver
iterate; the metadata distinguishes it from a final solver result. No checkpoint
claims scientific acceptance. Exact parameter/prediction arrays use NPZ without
pickle; metadata records masks, names, objective and problem fingerprint.
Returned `analytic_jacobian_calls`, `batch_calls` and `callback_evaluations` make
the costs explicit (analytic callbacks' own hidden simulation work is not counted).

## Offline correction proposal

`bounded_correction_step(J, error, lower_delta, upper_delta, actuator_names=...,
actuator_scale=..., sigma=..., regularization=0, step_fraction=1)` solves
`J*delta + error = 0` approximately, where error = current - target.
Use `api.measure_response` on fitted-model JuTrack observables to obtain J;
the model-authored callback maps approved hardware coordinates to native fields.
Bounds are **increments** containing zero. Scales and uncertainties are explicit.
The ridge penalty is regularization times squared normalized increment.

Returns named `proposal`, `delta`, data rank/singular values, weighted RMS before
and **predicted** after, solver termination and `machine_applied=False`. This is
not a machine-setting export. Recalculate the exact proposal and obtain operator
approval and calibrated hardware conversion separately. Preserve settings for
rollback and remeasure to test real improvement.
