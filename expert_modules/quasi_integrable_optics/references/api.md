# Octupole strengths and invariant diagnostics

```python
import brilliance.accelerator_api as api
expert = api.expert_module("laboratory.quasi_integrable_optics")
profile = expert.octupole_strength_profile(
    [1.0, 2.0, 4.0], beta_reference_m=1.0, k3_reference_m4=64.0
)
assert profile["k3_m4"] == [64.0, 8.0, 1.0]
variation = expert.invariant_variation(
    [0.002, 0.00201, 0.002], normalization_scale=0.002
)
```

`octupole_strength_profile(beta_m, *, beta_reference_m, k3_reference_m4)` takes
a positive finite common-beta vector, positive reference beta, and finite
signed reference strength. It returns `beta_m`, `k3_m4` and the constant
`k3_beta_cubed_per_m`. This assumes a matched common beta in both planes and
zero dispersion; it does not check the actual ring or provide a match.

In the checked JuTrack convention, `KOCT.k3` is the normalized third derivative
in m^-4. The cubic polynomial coefficient is `k3/6`, and the integrated derivative
strength is `k3*length` in m^-3. Pass the derivative strength to `KOCT`:

```python
jt = api.jutrack()
octupole = jt.KOCT("O1", 0.10, k3=profile["k3_m4"][0])
```

The numbers are illustrative and do not specify an engineered magnet. Average
the target longitudinal profile over physical segments when appropriate, then
check both physical segmentation and numerical integration convergence. Avoid
assuming that the same polynomial coefficient convention holds in another code.

`invariant_variation(values, *, normalization_scale)` takes at least two finite
samples for one particle at the same observation plane. It returns sample count,
initial value, mean, population standard deviation divided by the positive scale,
maximum departure from the initial sample divided by that scale, and signed
endpoint change divided by the scale. The explicitly chosen scale has the same
units as the invariant and stays fixed across compared settings.

Compute the candidate invariant with the conventions in the design guide and
archive those conventions. Track with JuTrack, retain every launch's survival,
and only claim a full-duration statistic for trajectories that reach the required
turn count. Small invariant variation is not a full-integrability test. A noisy
frequency estimate is not automatically diffusion; retain tune estimator,
observation-window length and alias handling in the evidence.
