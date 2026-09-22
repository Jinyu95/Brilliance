# Homogeneous-bend TME references

```python
from brilliance import accelerator_api as api
mba = api.expert_module("laboratory.mba_design")
reference = mba.tme_targets(
    energy_gev=E, bend_length_m=Lb, bend_angle_rad=theta,
    damping_partition_x=1.0,
)
```

Use the individual physical bend's length and angle. For the small-angle,
homogeneous-bend approximation, the bend-centre reference is

```text
βx = Lb/(2√15),  αx = 0,  Dx = Lb*theta/24,  Dpx = 0
εTME = Cq*γ²*|theta|³/(12√15*Jx),  Cq = 3.8319e-13 m
```

`reference.as_dict()` exposes the centre values and `minimum_emittance_pm`.
Jx=1 is an analytical assumption, not measured damping. These values guide
initialization/interpretation; they are not JuTrack results or mandatory targets.
Modified TME trades emittance against practical focusing and chromaticity, not
a universal fixed multiple of this reference.

For unequal bend angles/lengths,
`mba.homogeneous_bend_tme_reference(candidate, damping_partition_x=Jx)` returns
`reference_emittance_pm` from the ratio of summed reference I5 and I2.
Identical-bend multiplicity cancels: never multiply a single-bend emittance by
the number of bends. This is a geometry-based normalization, not a guaranteed
attainable complete-lattice minimum.

Use `mba.mba_radiation_report(candidate, evaluation)` to compare exact scoped
JuTrack emittance against this normalization. A unit's ratio does not describe
its assembled MBA. Actual radiation uses εx=Cq*γ²*I5/(Jx*I2), with physical
damping and finite positive equilibrium quantities checked separately.

Background: [homogeneous-bend TME](https://lss.fnal.gov/archive/test-tm/1000/fermilab-tm-1269.pdf)
and [modified TME](https://doi.org/10.1103/PhysRevSTAB.14.054002).
The links are optional reading; this knowledge does not require internet access.
