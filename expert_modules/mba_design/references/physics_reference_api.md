# MBA module: analytical TME reference tools

Load the module from an ordinary research program:

```python
from brilliance import accelerator_api as api

mba = api.expert_module("laboratory.mba_design")
reference = mba.tme_targets(
    energy_gev=energy, bend_length_m=individual_bend_length,
    bend_angle_rad=individual_bend_angle, damping_partition_x=1.0,
)
```

`mba.tme_targets(...)` supplies analytical homogeneous-bend reference scales, never
JuTrack evidence. Pass the **individual bend** angle and length, not the complete
MBA angle or length. It returns `TMECellTargets`, with typed attributes
`energy_gev`, `bend_length_m`, `bend_angle_rad`, `beta_x_center_m`, `alpha_x_center`,
`dispersion_x_center_m`, `dispersion_px_center`, `minimum_emittance_m`,
`minimum_emittance_pm`, and `model`. `reference.as_dict()` contains all values.

Evaluate the actual unit separately using JuTrack. Do not impose zero dispersion
at its boundary: achromat matching belongs to the complete MBA. An analytical
reference is not proof that a practical unit or the complete cell attains TME.

For a complete ring with unequal bend angles or lengths,
`mba.homogeneous_bend_tme_reference(ring, damping_partition_x=1.0)` combines the
individual homogeneous-bend reference integrals into one geometry-weighted
emittance scale. It is a normalization for reporting the actual/reference ratio,
not a simulated lattice result or a guaranteed attainable lower bound. It returns
a mapping with `model`, `lattice_id`, `damping_partition_x`, `I2_reference`,
`I5_reference`, `reference_emittance_m`, `reference_emittance_pm`, `rows`, and
`interpretation`; use `reference["reference_emittance_pm"]`, never `float(reference)`.

The conventional method explains the centre conditions and physical seed geometry.
Source boundaries: [homogeneous-bend TME](https://lss.fnal.gov/archive/test-tm/1000/fermilab-tm-1269.pdf)
is an analytical scale; [modified-TME](https://doi.org/10.1103/PhysRevSTAB.14.054002)
trades theoretical emittance for practical optics, not a universal fixed factor.
Neither source supplies stored strengths for this module.
