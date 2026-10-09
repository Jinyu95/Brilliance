"""Shared HMBA observables from supplied native results; no matching or targets."""

from __future__ import annotations

import math

from brilliance.region_measurements import region_optics_summary


def hmba_optics_observables(
    profile, elements, *, correction_regions, core_region, bend_contributions=None,
):
    """Measure the SAME named physical regions in objectives and final reports.

    Pass one exact period's JuTrack profile and rows. ``correction_regions`` maps
    labels to contiguous lists of real element names in non-bending correction
    regions; ``core_region`` lists ALL elements spanning the low-emittance core,
    including intervening quadrupoles/drifts and every bend segment. No bend
    count, naming convention, reflection symmetry or target is assumed.

    Returns flat ``measurements`` for an ExperimentDefinition/SciPy/PSO callback,
    the detailed ``regions`` report, and sextupole-site optics with lengths. Extrema
    are sampled, not proven interior maxima. This performs no simulation or
    interpolation. Caller must bind the native profile to these exact rows.
    Optional radiation is supplied by api.bend_radiation_contributions on that
    candidate, never estimated from sparse optics or multiplied by periods.
    Undefined/nonfinite contrast is None, never an infinite score. Thin sextupole
    sites are labelled explicitly; they do not demonstrate installation space.
    No returned flag certifies an HMBA or nonlinear acceptance.
    """
    rows = list(elements)
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError("Use one physical period with unique element names")
    if not correction_regions or "core" in correction_regions:
        raise ValueError("Name the correction regions; 'core' is reserved")
    regions = {**correction_regions, "core": core_region}
    occupied = set()
    for label, selected in regions.items():
        if isinstance(selected, (str, bytes)) or not selected:
            raise ValueError(f"{label}: supply a contiguous list of actual element names")
        if len(set(selected)) != len(selected) or set(selected) - set(names):
            raise ValueError(f"{label}: element names must be distinct and exist in this period")
        indices = sorted(names.index(name) for name in selected)
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"{label}: include intervening elements, not selected bend exits alone")
        if occupied.intersection(selected):
            raise ValueError("Core and correction regions must be disjoint")
        occupied.update(selected)
        if label != "core" and any(float(rows[i].get("angle_rad") or 0) for i in indices):
            raise ValueError(f"{label}: correction region must be non-bending; use the actual installation region")
        if label == "core" and not any(float(rows[i].get("angle_rad") or 0) for i in indices):
            raise ValueError("The core region must contain bending, not just its dispersion minimum")
        if not any(float(rows[i]["length_m"]) > 0 for i in indices):
            raise ValueError(f"{label}: region must have finite physical length")
    sites = [r["name"] for r in rows if r["kind"] == "KSEXT" and r["name"] in occupied - set(core_region)]
    site_keys = {name: f"sextupole:{name}" for name in sites}
    if set(site_keys.values()) & set(regions):
        raise ValueError("Correction labels cannot use the reserved sextupole: prefix")
    report = region_optics_summary(profile, rows,
        {**regions, **{key: [name] for name, key in site_keys.items()}},
        bend_contributions=bend_contributions)
    finite_names = {row["name"] for row in rows if float(row["length_m"]) > 0}
    for label in regions:
        missing = set(report["regions"][label]["elements_without_named_samples"]) & finite_names
        if missing:
            raise ValueError(f"{label}: native samples are missing from {sorted(missing)}")
    core = report["regions"]["core"]
    peaks = [report["regions"][name]["max_Dx_m"] for name in correction_regions]
    mean_peak = math.fsum(peaks) / len(peaks)
    contrast = mean_peak / core["max_abs_Dx_m"] if core["max_abs_Dx_m"] else None
    if contrast is not None and not math.isfinite(contrast):
        contrast = None
    metrics = {
        "weakest_bump_Dx_m": min(peaks),
        "mean_bump_Dx_m": mean_peak,
        "core_max_abs_Dx_m": core["max_abs_Dx_m"],
        "core_max_H_x_m": core["max_H_x_m"],
        "dispersion_contrast": contrast,
        "global_max_Dx_m": report["global"]["max_Dx_m"],
    }
    if core["radiation"] is not None:
        metrics.update({f"core_{key}": core["radiation"][key] for key in ("I2", "I4", "I5", "I5_fraction")})
    return {"schema": "hmba_optics_observables/v1", "measurements": metrics,
            "regions": report,
            "sextupole_sites": {name: report["regions"][key] for name, key in site_keys.items()},
            "thin_sextupole_sites": [name for name in sites if name not in finite_names],
            "limitations": report["limitations"]}
