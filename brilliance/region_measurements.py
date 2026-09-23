"""Named-region summaries of supplied optics and measured radiation, not a solver."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .lattice_fingerprint import cell_fingerprint


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def region_optics_summary(
    profile: Sequence[Mapping[str, Any]],
    elements: Sequence[Mapping[str, Any]],
    regions: Mapping[str, Sequence[str]],
    *,
    bend_contributions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize explicit named-element regions using ONE supplied optics profile.

    ``elements`` are one ordered physical period (api.one_period_elements).
    ``regions`` maps arbitrary labels to exact element names, not family names
    or presumed MBA roles. Regions may be disjoint or overlap; their fractions
    must not be added unless their element sets are disjoint. ``profile`` uses
    JuTrack-style s_m, beta_x, alpha_x, beta_y, Dx, Dpx keys, in this same period.
    Native exit profiles and caller-validated interior profiles are both usable;
    maxima are SAMPLED maxima, not certified interior extrema. No interpolation,
    transport, radiation integration, field edits, or simulation is performed.

    Optional ``api.bend_radiation_contributions(candidate)`` must be consistent
    and contain every nonzero-angle bend in these rows. Its full-ring summed
    integrals are retained, never multiplied by period count again. New reports
    carry a cell fingerprint; legacy evidence without it is labelled unbound.
    The caller must bind the optics profile to the same candidate and energy.

    Returns global and per-region sampled Dx, |Dx|, Hx, betas and their actual
    locations/source elements, plus regional I2/I4/I5 and fractions if supplied.
    Hx is computed from native Twiss/dispersion, not a hand-written optics map.
    There are no design targets or acceptance decisions in this measurement.
    """
    rows = [dict(row) for row in elements]
    fingerprint = cell_fingerprint(rows)
    spans, cursor = {}, 0.0
    for row in rows:
        length = _finite(row["length_m"], "element length")
        if length < 0:
            raise ValueError("Element lengths cannot be negative")
        spans[row["name"]] = (cursor, cursor + length)
        cursor += length
    if cursor <= 0:
        raise ValueError("The measured period must have positive length")
    if not isinstance(regions, Mapping) or not regions:
        raise ValueError("regions must map labels to nonempty lists of exact element names")
    selected = {}
    for label, supplied in regions.items():
        if not isinstance(label, str) or not label or isinstance(supplied, (str, bytes)):
            raise ValueError("Each region needs a label and an element-name sequence")
        names = list(supplied)
        if not names or len(set(names)) != len(names) or set(names) - spans.keys():
            raise ValueError(f"{label}: use distinct, existing element names")
        selected[label] = set(names)

    # Floating-point boundary allowance only, not a physical tolerance.
    position_atol = 1e-10 * max(1.0, cursor)
    points = []
    for item in profile:
        s = _finite(item["s_m"], "profile s_m")
        if s < -position_atol or s > cursor + position_atol:
            raise ValueError("Profile must describe the same single period as elements; clip a ring profile explicitly")
        beta = _finite(item["beta_x"], "beta_x")
        beta_y = _finite(item["beta_y"], "beta_y")
        if beta <= 0 or beta_y <= 0:
            raise ValueError("Profile beta functions must be positive")
        alpha = _finite(item["alpha_x"], "alpha_x")
        dx, dpx = _finite(item["Dx"], "Dx"), _finite(item["Dpx"], "Dpx")
        # Equivalent to gamma*D^2 + 2*alpha*D*D' + beta*D'^2,
        # but nonnegative and less cancellation-prone.
        hx = dx * dx / beta + (alpha * dx + beta * dpx) ** 2 / beta
        source = item.get("source_element")
        if source is None and item.get("name") in spans:
            source = item["name"]
        if source is None:
            # Position uniquely identifies an interior sample without guessing
            # naming conventions. Leave unlabelled boundary sides ambiguous.
            interior = [name for name, (lo, hi) in spans.items()
                        if lo + position_atol < s < hi - position_atol]
            if len(interior) == 1:
                source = interior[0]
        if source is not None:
            if source not in spans:
                raise ValueError(f"Profile source_element {source!r} is not in this period")
            lo, hi = spans[source]
            if not lo - position_atol <= s <= hi + position_atol:
                raise ValueError(f"Profile position does not belong to {source!r}")
        possible_sources = ([source] if source is not None else
                            [name for name, (lo, hi) in spans.items()
                             if lo - position_atol <= s <= hi + position_atol])
        points.append({"s_m": s, "source_element": source,
                       "possible_source_elements": possible_sources,
                       "name": item.get("name"), "Dx_m": dx, "Dpx": dpx,
                       "abs_Dx_m": abs(dx), "H_x_m": _finite(hx, "H_x"),
                       "beta_x_m": beta, "beta_y_m": beta_y})
    if not points:
        raise ValueError("A nonempty measured optics profile is required")

    radiation, totals, binding = {}, None, None
    if bend_contributions is not None:
        evidence = bend_contributions
        if evidence.get("computed") is not True or evidence.get("consistency", {}).get("passed") is not True:
            raise ValueError("Per-bend radiation must be computed and consistent with ringpara")
        declared = evidence.get("cell_fingerprint")
        if declared is not None and declared != fingerprint:
            raise ValueError("Per-bend radiation and physical period fingerprints differ")
        binding = "cell_fingerprint" if declared is not None else "legacy_unbound; caller must verify candidate identity"
        bends = {row["name"]: row for row in rows
                 if row["kind"] in {"SBEND", "RBEND"} and float(row.get("angle_rad") or 0) != 0}
        for row in evidence["rows"]:
            name = row["bend"]
            if name in radiation or name not in bends:
                raise ValueError("Radiation must identify each bend once, by exact period element name")
            for key in ("length_m", "angle_rad"):
                if not math.isclose(_finite(row[key], key), float(bends[name][key]), rel_tol=1e-10, abs_tol=1e-13):
                    raise ValueError(f"Radiation geometry differs for {name!r}")
            radiation[name] = {key: _finite(row[key], key) for key in ("I2", "I4", "I5")}
            if radiation[name]["I2"] < 0 or radiation[name]["I5"] < 0:
                raise ValueError("Radiation I2 and I5 cannot be negative")
        if set(radiation) != set(bends):
            raise ValueError("Radiation must cover every nonzero-angle bend in this period")
        totals = {key: math.fsum(row[key] for row in radiation.values()) for key in ("I2", "I4", "I5")}
        for key, total in totals.items():
            if not math.isclose(total, _finite(evidence["totals"][key], key), rel_tol=1e-10, abs_tol=1e-13):
                raise ValueError(f"Per-bend {key} rows do not reproduce supplied totals")

    def summarize(names):
        def belongs(point):
            if point["source_element"] is not None:
                return point["source_element"] in names
            # An anonymous boundary value cannot be assigned to one side of a
            # hard edge. Use it only when ALL possible sides lie in the region.
            return bool(point["possible_source_elements"]) and set(point["possible_source_elements"]) <= names

        measured = [point for point in points if belongs(point)]
        if not measured:
            raise ValueError(f"No optics samples in region {sorted(names)!r}; provide native samples there")
        peaks = {key: dict(max(measured, key=lambda point: point[key]))
                 for key in ("Dx_m", "abs_Dx_m", "H_x_m", "beta_x_m", "beta_y_m")}
        ordered = [row["name"] for row in rows if row["name"] in names]
        result = {"element_names": ordered,
                  "length_m": math.fsum(spans[name][1] - spans[name][0] for name in ordered),
                  "element_spans_m": {name: list(spans[name]) for name in ordered},
                  "sample_count": len(measured),
                  "excluded_ambiguous_boundary_samples": sum(
                      bool(set(point["possible_source_elements"]) & names) and not belongs(point)
                      for point in points),
                  "elements_without_named_samples": [name for name in ordered
                      if not any(point["source_element"] == name for point in measured)],
                  **{f"max_{key}": point[key] for key, point in peaks.items()},
                  "peaks": peaks, "radiation": None}
        if totals is not None:
            values = {key: math.fsum(radiation.get(name, {}).get(key, 0.) for name in names)
                      for key in totals}
            result["radiation"] = {**values, **{key + "_fraction": values[key] / total if total else None
                                                for key, total in totals.items()}}
        return result

    return {"schema": "region_optics_summary/v1", "cell_fingerprint": fingerprint,
            "period_length_m": cursor, "measurement": "supplied_samples; no interpolation",
            "radiation_scope": "unchanged from supplied per-bend evidence" if totals is not None else None,
            "radiation_binding": binding,
            "global": summarize(set(spans)),
            "regions": {label: summarize(names) for label, names in selected.items()},
            "limitations": ["Sampled extrema are not certified interior maxima or minima.",
                            "Unlabelled boundary samples spanning different regions are excluded from those regions.",
                            "Caller must bind the optics profile and radiation to the same candidate and energy.",
                            "Overlapping region integrals must not be added; this is not a design acceptance."]}
