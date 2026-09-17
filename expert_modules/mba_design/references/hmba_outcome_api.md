# Hybrid-MBA outcome evidence

Hybrid requests require measured physics evidence. Write construction, matching,
residuals and objectives in the project's Python program using the general tools.
The old complete design controllers have been removed, not hidden or aliased.

## Physical hypothesis

Start a conversion from this project's measured conventional MBA, not a new
arbitrary seven-bend line. Identify bends in longitudinal order. For the outer
DBA-like hypothesis, B1–B2 and B(M−1)–BM create the chromatic regions; B3–B(M−2)
form the low-H core. Both distinct SF and SD drift slots on the left belong
between B1 and B2, with global mirror partners in the right outer pair. The
suppressor is the transition between that chromatic region and the core, not
another name for the SD site. Count real finite installation space, not a high
dispersion value inside a bend. Recompute periodic optics after each change.

Large useful dispersion at sextupoles can reduce chromatic strength burden.
Appropriate phase and endpoint optics between paired sextupoles can suppress
geometric aberrations; equal beta or a nominal phase alone is not proof of
nonlinear cancellation. State the chosen pair hypothesis and evaluate its real
transfer matrices, then measured chromatic response and tracking. Those later
conditions are not implied by passing the linear-shape check below. Preserve
machine requirements, not inherited seed tunes or arbitrary core-optics targets.

## Final measurement, independent of the design program

Use `mba.HybridMeasurement(parent, machine_contract=machine,
shape_requirements=shape_targets, maximum_emittance_pm=approved_search_ceiling,
budget_provenance=approval_record, max_step_m=step)` as the general match/PSO
callback. It accepts exact proposed rows, not a stored facility seed. Its
`D_bump_L_m`, `D_bump_R_m`, `D_core_m` and `H_core_max_m` come from the same
measurement as final checking. `shape` records positions, regions, map-checked
sampling and available drift intervals. `global_max_dispersion_m` is a separate
whole-cell diagnostic. Never replace these regions with unit-name filters.
Optional `machine["installation_slots"]` maps SF_L/SD_L/SF_R/SD_R to ordered
real drift-name lists. Both left slots must be disjoint spans in B1–B2, with
global mirror partners in the right region; adjacent drifts are allowed.
Search and checking report identical native slot extrema/positions. Omitting
the mapping reports `not_declared`, never an implied sextupole-pair design.
Keep shape objectives separate from radiation feasibility: the callback retains
actual emittance/I2/I5 but sets `feasible=False` outside the declared radiation
budget. A budget must be approved before production; no multiplier of the parent
or aspiration is silently made a requirement. Original caps still apply.
Pass the observer's `machine` contract to final verification: it records the
separate `hybrid_radiation_budget` ceiling/provenance, which the final ring
checker also enforces. A conflicting declared budget is an input error.

Geometry changes require a matched feasible initialization before production.
Use a separate feasibility experiment when needed; never start production from
an unstable sentinel-valued point or promote a radiation-rejected continuation.
A failed construction, failed match, bound saturation or exhausted finite budget
does not prove that a topology cannot work. Report exact candidate IDs and count
independent knobs separately from physical magnets. No new campaign is warranted
until optimizer/checker observations agree for the same rows and sampling.

```python
mba = api.expert_module("laboratory.mba_design")
report = mba.verify_hmba_candidate(
    candidate,
    machine_contract=machine,
    shape_requirements=shape_targets,
    sampling_steps_m=[coarse_step_m, fine_step_m],
    relative_tolerance=approved_relative_tolerance,
)
```

This function does **no construction, matching or optimization**. It measures
the exact selected candidate with JuTrack at two resolutions and independently
evaluates its full repeated ring. It archives measurements using the existing
candidate-evaluation receipt mechanism. The application rechecks the receipt,
physical rows and numerical outcomes; setting `passed=True` is not evidence.
Standalone tests may pass `output_directory`; managed programs must use their
configured output directory. The exact candidate must also be passed with its
high-level `candidate_evaluation` to `run.finish`.

## Original requirements versus proposed targets

`machine_contract` is a mapping with the following fields. Recover original
values from this project's saved contract; disclose and obtain approval for
missing requirements. Do not infer limits from an existing facility or change
them to make a parent or trial pass.

| Group | Required fields |
| --- | --- |
| Machine | `energy_gev`, `periods`, `expected_cell_length_m`, `expected_cell_angle_rad`, `expected_bends` |
| Physical space | `minimum_magnet_length_m`, `minimum_drift_length_m`, `id_drift_names`, `id_length_m` |
| Engineering | `limits`: the explicit energy-dependent envelope returned by `api.magnet_strength_limits` with the approved technology parameters |
| Achromat | `achromat_dx_tolerance_m`, `achromat_dpx_tolerance` |
| Optional original caps | `maximum_beta_m`, `maximum_emittance_pm`, `maximum_energy_spread`; explicitly `None` when unspecified |
| Provenance | `limit_provenance`: a nonempty mapping identifying original values and any approved changes |

Preserve optional `id_beta_x_bounds_m` and `id_beta_y_bounds_m` when specified.
An emittance aspiration is not automatically a hard validity cap.

`shape_requirements` separately declares the proposed linear-shape targets:
`min_bump_m`, `min_contrast` (>1), `max_core_dispersion_m`, `max_core_h_m`,
`min_usable_drift_m`, `min_total_usable_drift_m`, `max_drift_slope`, `max_beta_m`,
`peak_slope_tolerance`, `id_dispersion_tolerance_m`, `id_slope_tolerance`,
`max_sample_step_m`. State which are new approved shape targets rather than
original machine limits. Shape tolerances cannot relax original achromat limits.
Choose a positive fine step smaller than the coarse step; convergence tolerance
is explicit and must not be silently loosened.

The current shape diagnostic covers homogeneous positive-bend MBAs with two
outer inter-bend chromatic regions and a middle core. It measures actual interior
maxima or usable drift plateaus, drift installation lengths, core dispersion/H,
stability, achromat, symmetry, engineering limits and damped radiation. Bends are
identified by physical order, not old unit names. Other architectures need an
appropriate expert measurement adapter, not a fabricated passing report.

## Completion and optimization

Use `acceptance={"hybrid_mba_linear": report}` for a verified **linear hybrid
shape**. No 3π/π phase condition, sextupole correction or DA/MA is implicit in
this claim. Additional requested performance needs its own measurements; do not
call a linear shape an operational or nonlinear-validated design.

When emittance optimization is requested, additionally publish the same report
under `mba_emittance_optimization`, with:

```python
report["optimization"] = {
    "method": "describe the actual algorithm and objective",
    "termination_reason": "describe its observed stopping condition",
    "selection_reason": "explain the measured baseline comparison",
    "selected_index": selected_index,
    "trials": [
        {"candidate": baseline.as_dict(),
         "measurement_receipt": baseline_report["measurement_receipt"]},
        # Actual measured candidates, including the selected candidate.
    ],
}
```

The first trial is the baseline. Preserve the actual search history as a separate
artifact as well. At least two distinct measured physical snapshots are needed;
all comparisons keep the same machine contract. The selected trial must be the
exact final candidate and receipt, with lower independently measured emittance
than the baseline. A local optimizer or a callback with inner matching is valid;
PSO generations and a specific result schema are not required. A baseline may
fail hybrid-shape targets but must have a physical radiation equilibrium.

Failed targets, unconverged sampling, or no improvement remain useful partial
research: retain receipts, tables, plots and code with `request_satisfied=False`.
Do not rerun seed generation merely to obtain a missing final measurement.
