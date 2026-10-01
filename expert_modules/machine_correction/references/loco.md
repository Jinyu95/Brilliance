# LOCO: identification is different from correction

LOCO fits lattice and diagnostic parameters to closed-orbit response R = du/dc,
with orbit u ordered all x then all y and commanded kicks c. Retain all four
response blocks. An inferred lattice is not an instruction to change the machine.

## Minimal offline benchmark

Start with only **the nominal ring, measured ORM and measured dispersion**.
Use the ring as the initial model, preserving its geometry, sextupoles, edges and
existing offsets. No historical snapshot, mover file, control conversion, prior
correction export, old GUI settings or additional user configuration is required.

Default: basic uncoupled fit of normal-quadrupole gradients and scalar
BPM/corrector gains, initially nominal. Fit Rxx, Ryy and horizontal dispersion
(or measured RF slope); report Rxy, Ryx and vertical dispersion as diagnostics,
not corrected coupling. Do not silently add skew fields, rolls, movers, arbitrary
Fudge fields or AD. Group numerical slices of each physical magnet; equal nominal
strengths do not establish a family. Resolve identities from supplied data.
Check gain identifiability and document any gauge or prior.

A quadrupole-only fit with unity gains is an optional reduced baseline. Label its
calibrated-instrument assumption and use it in BOTH implementations; it is not
interchangeable with the default gain-fitting comparison.

The agent chooses and checks steps, scaling, bounds, regularization and stopping
settings. These are numerical choices, not hardware limits or user-prepared files.
Use measured errors when supplied. Otherwise choose fixed block-normalized residual
scales from the measured data BEFORE fitting. Save exact scales, masks and order;
call this a relative-error objective, not statistical chi-square, and do not infer
confidence intervals. Keep dispersion even without error bars. For zero-scale
blocks, document a finite numerical scale; never divide by zero or hide discrepancies.

Generate `fit_setup.json` with input hashes, device/parameter order, parameterization,
calibration/gauge, masks, weights, orbit/RF mode and numerical settings. This is an
OUTPUT, not user configuration. Save initial/fitted ORM, dispersion, parameters,
residuals and optics without consulting another application's fitted answers.

Use identical models, observations, parameter groups, calibration conventions and
weights in both implementations. Check initial native responses before optimizers:
disagreement may be a model/convention error. Compare final residuals, optics and
identifiable combinations, not identical histories or weak parameters. Add physics
only in separately labelled tests justified by residuals.

## Autonomous completion from one request

Reuse validated responses, run the fit, and publish the outcome without routine
approval questions. A nominal-response report alone does not complete LOCO.
Honor supplied metadata; otherwise save **provisional** hypotheses in `fit_setup.json`:

- Initialize signs from named same-plane response comparisons; check per-device
  exceptions. Compare ambiguous hypotheses, never certify inferred hardware signs.
- For unspecified stored-beam RF conditions, start with fixed path length and
  compare fixed momentum; report both if inference changes materially.
- Fit raw RF slopes using candidate JuTrack dispersion/slip factor and verified
  coordinate signs, not an unexplained stored compaction value. Recompute the
  candidate-dependent conversion in the objective and test step sensitivity.
- Choose numerical settings and disclose nonstatistical weights when needed.

Large cross-plane residuals do not veto the initial uncoupled fit. Report them
as held-out discrepancies and warn of biased inferred optics; do not certify a
machine correction. Evaluate requested model-space K1 corrections on private
rings; missing hardware calibration blocks operational settings, not that study.
Withhold trusted correction when identification is inadequate. Finish with actual
fit results or an attempted-fit failure, assumptions and before/after diagnostics.
Unknown units, unresolved identities or unsupported maps remain real blockers;
do not fabricate inputs to eliminate questions.

## Measurement contract

Bind rows/columns to unique device names, element occurrences and planes. Save
available acquisition/RF/orbit metadata, kick spans, errors and masks. Resolve aliases;
reject duplicates; mask missing data, never zero-fill. The helper samples element
exits: do not relocate BPMs to match array lengths. Handle missing conventions
using provisional hypotheses above; do not silently certify them.

Use response in m/rad, orbit in m and delta = dp/p. Divide a high-minus-low orbit
difference by its FULL high-minus-low kick span. Check positive/negative probes
for nonlinearity. BPM resolution is not automatically the uncertainty of an
averaged difference; use repeats when available and state the weighting model.

RF scans measure du/df_RF. To first order,
delta = -df_RF/(f_RF*eta), eta = alpha_c - 1/gamma^2, hence
D = -f_RF*eta*du/df_RF. Keep RF slopes as slopes unless this conversion is justified.
Use the actual RF frequency and momentum-compaction convention. A horizontal kick
can change path length and equilibrium energy, so fixed-momentum and fixed-RF
responses differ. The helper's fixed-path approximation is not full cavity/radiation
equilibrium or a low-energy time-of-flight model. Compare like conventions and do
not add an energy-response correction twice.

## Model extensions and computation

For coupling, BPM output = B @ true orbit with 2x2 B; corrector main/cross gains
use native responses at the SAME magnet. BPM roll changes coordinates, skew
fields change coupling. Correlated parameters need gauges/priors.

Bind supported fields/units to exact elements. Preserve sextupoles and supplied
orbit/offsets; omitted feed-down can misattribute errors. Restore mutations even
on exceptions or use private lattices. Reuse response only for instrument-only
trials. Parallel workers need independent lattices. Verify unsupported pass methods;
never substitute silently.

## Fit and diagnosis

Concatenate active ORM and dispersion/RF entries in saved order. Minimize
(prediction - measurement)/sigma, with declared priors if justified; for unknown
noise, sigma is only the numerical scale described above. Scale parameters in
their own units. This module uses bounded least squares and finite differences,
not AD. Check step sensitivity, closed-orbit closure and direct +/- kick probes.

Inspect the weighted, parameter-scaled DATA Jacobian apart from regularization.
Common BPM gain and inverse corrector gain can be degenerate. Priors do not
independently measure every parameter. Different regularization may change fitted
values; never copy a GUI's SVD cutoff without matching scaling, order and masks.

Report physical/weighted residual RMS by block, dispersion, spatial patterns,
outliers, saturated bounds and weak modes. Use repeat/held-out scans when available.
Low training error with implausible gains is unreliable. Record total JuTrack
calls/time, including differences. Never replace exceptions by zero residuals.
Solver termination is not scientific acceptance.

## Optional offline correction

Correction is a separate task. Target physical reference optics/coupling/dispersion,
not generally raw ORM: do not correct diagnostic gain errors with magnets.
Calculate observable derivatives on the fitted native model and propose bounded
actuator deltas with explicit units/regularization. Fitted K1 errors are not currents.
Transfer functions, shared supplies, polarity, hysteresis, slew and mover limits
are required only for this hardware-specific extension.

Reevaluate exact proposals for stability, orbit and optics; preserve the starting
state and undo proposal. No control connection exists here. Application requires
operator approval/interlocks; new measurements must establish actual improvement.

## Sources and scope

- Safranek, *Experimental determination of storage ring optics using orbit response
  measurements*, NIM A 388 (1997), [DOI](https://doi.org/10.1016/S0168-9002(97)00309-4).
- Safranek et al., *MATLAB-Based LOCO*, EPAC 2002,
  [paper](https://proceedings.jacow.org/e02/PAPERS/WEPLE003.pdf).
- [Simulated Commissioning documentation](https://sc.lbl.gov/main.html).

Sources support the method, not validation of this implementation or parity with
custom HEPS elements. Numerical settings and application limits remain explicit.
