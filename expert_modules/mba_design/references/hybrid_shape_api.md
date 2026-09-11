# MBA module: hybrid dispersion-shape tools and acceptance

For initial conventional-to-hybrid conversion read `references/outer_dba_api.md` first.
Replace the outer sections and solve the new periodic optics; source beta/dispersion need not remain fixed.
`references/hybrid_regional_api.md` is optional later refinement with explicitly adopted limits.

`evaluate_hybrid_mba_shape(..., stage="bump_feasibility")` omits only max_core_h_m.
Both stages require min_usable_drift_m, min_total_usable_drift_m, max_drift_slope, max_beta_m
and max_core_dispersion_m in addition to min_bump_m/min_contrast and numerical tolerances.
Drift transport, usable installation lengths, core separation and beta envelope are independent
checks on the RESULTING cell, not equality to source optics. A quadrupole peak alone is insufficient.
replacement.evaluate defaults to preserve_core_optics=False. Only an explicitly requested
preservation contract requires measure_reference(parent) and preserve_core_optics=True.
Its scope is complete_cell_bump_feasibility_gate: save it as intermediate data, not as final
hybrid_mba_shape acceptance. The default stage="full_hybrid" retains all original shape gates.

## Worker-local evaluator and physical mirror knobs

For repeated measurements, `mba.HybridMBAShapeEvaluator` combines the checked sampler and
construction audit. It does not choose a lattice, matcher, objective, or search route.

```python
mba = api.expert_module("laboratory.mba_design")
limits = api.magnet_strength_limits(candidate.energy_gev)
pairs = mba.hybrid_mba_quadrupole_knobs(api.one_period_elements(candidate),
    max_abs_k1_m2=limits["max_abs_k1_m2"])
# Each spec: name, field='k1_m2', element_names=[physical mirror partners], bounds, initial.
# Select disjoint matching/design subsets from measured response, not unit-name substrings.
evaluator = mba.HybridMBAShapeEvaluator(candidate, thresholds=thresholds,
    expected_cell_length_m=required_cell_length, expected_cell_angle_rad=required_cell_angle,
    limits=limits, max_step_m=profile_step,
    minimum_magnet_length_m=min_magnet_length, minimum_drift_length_m=min_drift_length)
measured = evaluator.measure(include_radiation=True)
# Always inspect stable/feasible BEFORE reading numeric fields.
# Stable results expose D_bump_L_m, D_bump_R_m, D_core_m, H_core_max_m,
# dispersion_contrast, continuous_shape_penalty, signed Dx_m/Dpx, traces, shape_report.
# Rejected proposals expose failure/construction_audit or the unstable map traces.
# shape_passed is separate from feasible; final shape is not generation-zero feasibility.
response = evaluator.response(design_specs, inner_match=match_proposed_rows,
    finite_difference_fraction=1e-4, max_halvings=10)
# match_proposed_rows(rows) -> {success: bool, elements: actual_matched_rows, ...}
# It must not overwrite any outer design coordinate. Use independently owned matching knobs.
# response: success, baseline, columns[{name,element_names,derivative,scheme,
# accepted_deltas,attempts}], metric_names, jacobian (None if any column unresolved).
# Without inner_match this is explicitly an UNMATCHED response, not an achromatic direction.
```

Use signed residuals from `evaluator.linear_metrics(rows)` for the fast inner matcher.
Audit its response rank first; not every quadrupole called QM controls dispersion. A
failed trial is not zero response. Scale response rows by their physical tolerances when
analyzing rank, and validate a bounded step numerically before increasing its size.

Use `continuous_shape_penalty` as the search score. It combines scaled height/core/contrast
deficits with a continuous outer-peak prominence penalty; it avoids the Boolean interior-peak
jump in the diagnostic `shape_penalty` near a flat conventional baseline. Its numerical
prominence scale is reported separately and is not an acceptance tolerance. The surrogate
does NOT replace `shape_passed`: inspect the strict interior/slope checks at each continuation
checkpoint and use the complete final acceptance report before claiming success.

For process-parallel optimization create the complete evaluator INSIDE the worker factory:

```python
core_limits = None  # Initial shaping: no baseline H/I5/Dx no-worsening constraint.
emittance_ceiling_pm = None  # Add only if explicitly adopted, not copied from the seed.

def make_worker_physics(baseline_rows):
    from brilliance import accelerator_api as api
    mba = api.expert_module("laboratory.mba_design")
    # Only plain rows/scalars/dictionaries are captured from the parent process.
    cell = api.periodic_cell(title="shape-worker", energy_gev=energy, elements=baseline_rows,
        source="Current saved candidate; worker-local calculation", intended_repetitions=periods)
    local = mba.HybridMBAShapeEvaluator(cell, thresholds=thresholds,
        expected_cell_length_m=required_cell_length, expected_cell_angle_rad=required_cell_angle,
        limits=limits, max_step_m=profile_step,
        minimum_magnet_length_m=min_magnet_length, minimum_drift_length_m=min_drift_length)

    def objective(rows):
        return local.objective(rows, core_limits=core_limits,
                               emittance_ceiling_pm=emittance_ceiling_pm)

    return local.linear_metrics, objective
```

`energy`, `periods`, `thresholds`, `limits`, `core_limits`, lengths/angles and the emittance ceiling above
are plain values from the declared machine contract; core_limits and the emittance ceiling
default to None. They are not automatically obtained from baseline measurements. Do not refer to a GLOBAL `shape_metrics`
function that uses a parent-process sampler. Instantiate the factory locally for a baseline
pilot, then test the process-worker path before a production campaign. This evaluator's
`feasible` includes damping only when `include_radiation=True`; the example also enforces
the separate emittance ceiling only when supplied. It does not certify nonlinear acceptance.

## Dense diagnostics and final acceptance

These tools modify no seed-generation method. Start a hybrid follow-up from the exact saved
complete MBA, not a rebuilt unit. Bend roles follow physical order: first/last two positive
bends form DBA-like regions; the middle M-4 bends are the low-H core. For the unit-derived
7BA, B2=U001:B, B3=U002:B, B4=U003:B, B5=U004:B, B6=U005:B. The bends U002:B/U004:B are
CORE; a quadrupole before B3 may instead lie in the suppressor. Never classify a whole unit by name.

```python
from brilliance import accelerator_api as api

mba = api.expert_module("laboratory.mba_design")
physical_rows = api.one_period_elements(candidate)
roles = mba.hybrid_mba_bend_roles(physical_rows)
# roles: bends[{bend,name,element_index,role,mirror_bend,s_start_m,s_end_m,...}],
# regions[left_bump/right_bump/core/left_suppressor/right_suppressor],
# outer_bend_indices, core_bend_indices, core_element_indices, central_bend.

# Declare/disclose these values from the user's contract and measured baseline,
# not from universal H-MBA constants. This is a key-name example, not a threshold prescription.
thresholds = dict(min_bump_m=minimum_outer_peak, min_contrast=required_contrast,
    min_usable_drift_m=minimum_contiguous_drift, min_total_usable_drift_m=minimum_total_drift,
    max_drift_slope=allowed_drift_slope, max_beta_m=beta_envelope,
    max_core_dispersion_m=allowed_core_dispersion, max_core_h_m=allowed_core_H,
    peak_slope_tolerance=1e-4, id_dispersion_tolerance_m=1e-4,
    id_slope_tolerance=1e-4, max_sample_step_m=profile_step)

# Build once per worker. Proposed rows can change lengths, angles and strengths,
# but not names/order/types; a topology change needs a new sampler.
sampler = api.build_jutrack_profile_sampler(candidate, max_step_m=profile_step)
linear = sampler.linear_metrics(proposed_rows)
if linear["stable"]:
    shape = mba.hybrid_mba_shape_diagnostics(proposed_rows, linear["twiss_profile"],
        thresholds=thresholds, sampling=linear["sampling"])
    metrics = shape["metrics"]
    # D_bump_L_m, D_bump_R_m, D_core_m, H_core_max_m, H_core_mean_m,
    # dispersion_contrast, contrast_L, contrast_R, ID_Dx_max_abs_m,
    # ID_Dpx_max_abs, shape_penalty. No emittance objective is implied.
    # shape['peaks']['left'/'right']: s_m, Dx_m, Dpx, interior_maximum, stationary.
```

The sampler uses JuTrack propagation through a physical element's partial length from that
element's ORIGINAL entrance Twiss. It verifies full-length prefix/physical map equivalence,
retains original physical exits, and locates dispersion extrema with JuTrack + bounded root
finding. It never chains subdivided KQUADs into a different periodic lattice: changing the
number of integration steps that way changes the map. Original physical magnets, not samples,
define roles and the plotted lattice. `sampling.method` is `physical_element_prefix_map_checked`.
Supported elements: DRIFT, KQUAD/QUAD, pure SBEND, KSEXT, KOCT, MARKER. Unsupported edge,
transform, combined-function or other element physics fails explicitly.

`hybrid_mba_shape_diagnostics` requires the FULL one-cell profile, including both ID boundaries,
with `s_m,Dx,Dpx,beta_x,alpha_x` (beta_y/alpha_y retained when supplied). It checks outer maxima
strictly BETWEEN B1–B2 and B(M-1)–BM, small slope with rising/falling brackets, per-side height
and contrast, whole-core |Dx|/H and achromatic ID. Element-exit-only or unsourced profiles cannot
pass its sampling gate. `min_contrast` must exceed 1. Its shape penalty is usable for continuation
from a physically feasible conventional baseline; do not demand final shape success from all
initial particles. Audit hard engineering/symmetry/length/angle/stability/ID constraints separately.
Optimize shape primarily; emittance is a disclosed ceiling/secondary objective. Keep inner
matcher and outer design knobs disjoint. Geometry release must preserve exact cell budgets.

For nested matching, use a FAST ordinary editor for `evaluate_metrics` (ID Dx/Dpx residuals),
and the dense sampler only for the outer shape objective after matching. An example callback
with explicitly supplied machine limits (optional recovery ceilings only if adopted) is:

```python
def evaluate_shape(rows):
    return evaluator.objective(rows)  # No baseline beta/Dx/H/I5/emittance preservation.

# design and matching are YOUR declared disjoint parameterizations.
# Keep all desired mirror partners tied; freeze core bend geometry initially.
result = api.optimize_lattice_parameters(physical_rows, design, matching,
    evaluate_metrics, evaluate_shape,
    match_targets={"Dx_m": (0.0, 1e-4), "Dpx": (0.0, 1e-4)},
    objective_names=("bump_deficit", "core_dispersion_m", "core_I5_m1"),
    population_size=at_least_ten_per_outer_knob, generations=declared_generations,
    require_feasible_initial_population=True,
    **run.optimization_options("hybrid-shape"))
```

This is an evaluator recipe, not a stored lattice or fixed optimizer route. Do not use
`measured['passed']` as initial feasibility: a conventional seed can satisfy the hard machine
constraints while still having a nonzero shape penalty. Report each failure reason and the
baseline callback result before starting a large production population.

Final shape evidence must be tied to the exact selected candidate:

```python
report = mba.evaluate_hybrid_mba_shape(candidate, thresholds=thresholds,
    expected_cell_length_m=required_cell_length,
    expected_cell_angle_rad=required_cell_angle, limits=magnet_limits,
    max_step_m=profile_step, minimum_magnet_length_m=min_magnet_length,
    minimum_drift_length_m=min_drift_length)
run.finish(summary=measured_result, data={"engine": "JuTrack", **report.get("metrics", {})},
    candidate_lattice=candidate, acceptance={"hybrid_mba_shape": report},
    scientific_passed=True, request_satisfied=(report["passed"] and selected_core_objective["feasible"]),
    methods=["JuTrack physical-element profiles; H-MBA shape gate"], limitations=[])
```

The full `hybrid_mba_shape/v2` report retains measured profiles, usable drifts, physical rows/fingerprint,
matrix, radiation/damping, thresholds and construction audit. Validation recomputes its numerical
checks and binds candidate identity; setting `passed=True` manually is not sufficient. A missing
or failed hybrid report keeps the valid conventional result as intermediate evidence and does
not promote it as an H-MBA. After shape success, emittance recovery is a SEPARATE search with
shape constraints retained, followed by chromatic-sextupole and nonlinear optimization.
