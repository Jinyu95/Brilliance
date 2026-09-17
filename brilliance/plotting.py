"""Physics-facing transformations for lattice and periodic-optics plots.

The functions in this module are deliberately independent of the web UI and
Altair.  They turn simulator evidence into small tabular records that the GUI
can plot and that tests can validate without a browser.
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping

_ELEMENT_STYLE = {
    # Keep the same visual language as pyJuTrack.plot_lattice. Drifts are
    # represented by the black reference line rather than filled grey boxes.
    "DRIFT": ("Drift", "#6b7280"),
    "KQUAD": ("Quadrupole", "#8ecae6"),
    "QUAD": ("Quadrupole", "#8ecae6"),
    "QUADRUPOLE": ("Quadrupole", "#8ecae6"),
    "KSEXT": ("Sextupole", "#2563eb"),
    "SEXT": ("Sextupole", "#2563eb"),
    "SEXTUPOLE": ("Sextupole", "#2563eb"),
    "KOCT": ("Octupole", "#16a34a"),
    "OCT": ("Octupole", "#16a34a"),
    "SBEND": ("Dipole", "#f59e0b"),
    "RBEND": ("Dipole", "#f59e0b"),
    "BEND": ("Dipole", "#f59e0b"),
    "RFCAVITY": ("RF cavity", "#dc2626"),
    "RFCA": ("RF cavity", "#dc2626"),
}


def lattice_plot_rows(elements: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return JuTrack-style rectangular glyphs for a straight lattice view."""
    rows: list[dict[str, Any]] = []
    cursor = 0.0
    for index, element in enumerate(elements):
        start = _finite(element.get("s_start_m"), cursor)
        length = max(0.0, _finite(element.get("length_m"), 0.0))
        end = _finite(element.get("s_end_m"), start + length)
        if end < start:
            start, end = end, start
        kind = str(
            element.get(
                "type", element.get("element_type", element.get("kind", "OTHER"))
            )
        ).upper()
        family, color = _ELEMENT_STYLE.get(kind, (kind.title(), "#969696"))
        y0, y1 = _element_height(kind, element)
        rows.append({
            "order": index + 1,
            "name": str(element.get("name", f"E{index + 1}")),
            "type": kind,
            "family": family,
            "s_start_m": start,
            "s_end_m": end,
            "length_m": end - start,
            "y0": y0,
            "y1": y1,
            "color": color,
            "k1_m-2": element.get("k1", element.get("k1_m2")),
            "angle_rad": element.get("angle_rad"),
        })
        cursor = end
    return rows


def ring_plot_rows(
    elements: Iterable[Mapping[str, Any]],
    *,
    bend_samples: int = 4,
) -> list[dict[str, Any]]:
    """Trace the physical floor plan implied by element lengths and bend angles.

    This is the framework-neutral counterpart of ``pyJuTrack.plot_lattice``.
    Straight elements advance along the current reference direction and dipoles
    follow circular arcs.  Every returned segment retains its JuTrack element
    identity so a UI can colour and inspect the actual accepted lattice.
    """
    if bend_samples < 1:
        raise ValueError("bend_samples must be at least one")
    rows: list[dict[str, Any]] = []
    x_m = y_m = heading = 0.0
    for index, element in enumerate(elements):
        kind = str(
            element.get("type", element.get("element_type", element.get("kind", "OTHER")))
        ).upper()
        family, color = _ELEMENT_STYLE.get(kind, (kind.title(), "#969696"))
        length_m = max(0.0, _finite(element.get("length_m"), 0.0))
        angle_rad = _finite(element.get("angle_rad", element.get("angle")), 0.0)
        sample_count = bend_samples if abs(angle_rad) > 1.0e-14 and length_m > 0.0 else 1
        segment = f"{index + 1}:{element.get('name', f'E{index + 1}')}"
        for point_index in range(sample_count + 1):
            fraction = point_index / sample_count
            if abs(angle_rad) > 1.0e-14 and length_m > 0.0:
                radius_m = length_m / angle_rad
                local_angle = fraction * angle_rad
                x_point = x_m + radius_m * (
                    math.sin(heading + local_angle) - math.sin(heading)
                )
                y_point = y_m - radius_m * (
                    math.cos(heading + local_angle) - math.cos(heading)
                )
            else:
                distance = fraction * length_m
                x_point = x_m + distance * math.cos(heading)
                y_point = y_m + distance * math.sin(heading)
            rows.append({
                "segment": segment,
                "element_order": index + 1,
                "point_order": point_index,
                "name": str(element.get("name", f"E{index + 1}")),
                "type": kind,
                "family": family,
                "color": color,
                "x_m": x_point,
                "y_m": y_point,
                "length_m": length_m,
                "angle_rad": angle_rad,
            })
        x_m, y_m = rows[-1]["x_m"], rows[-1]["y_m"]
        heading += angle_rad
    if rows:
        # A centred plot is easier to compare across revisions while retaining
        # exact distances and closure errors.
        x_mid = (min(row["x_m"] for row in rows) + max(row["x_m"] for row in rows)) / 2.0
        y_mid = (min(row["y_m"] for row in rows) + max(row["y_m"] for row in rows)) / 2.0
        for row in rows:
            row["x_m"] -= x_mid
            row["y_m"] -= y_mid
    return rows


def optics_plot_rows(
    profile: Iterable[Mapping[str, Any]],
    *,
    max_points: int = 1400,
) -> list[dict[str, Any]]:
    """Return down-sampled long-form beta and dispersion records."""
    if max_points < 2:
        raise ValueError("max_points must be at least two")
    points = [dict(point) for point in profile if _is_finite(point.get("s_m"))]
    if not points:
        return []
    points.sort(key=lambda point: float(point["s_m"]))
    stride = max(1, math.ceil(len(points) / max_points))
    selected = points[::stride]
    if selected[-1] is not points[-1]:
        selected.append(points[-1])
    definitions = (
        ("beta_x", "βx", "beta", "m"),
        ("beta_y", "βy", "beta", "m"),
        ("Dx", "Dx", "dispersion", "m"),
    )
    rows: list[dict[str, Any]] = []
    for point in selected:
        for key, quantity, group, unit in definitions:
            if _is_finite(point.get(key)):
                rows.append({
                    "s_m": float(point["s_m"]),
                    "name": str(point.get("name", "")),
                    "series": key,
                    "quantity": quantity,
                    "group": group,
                    "value": float(point[key]),
                    "unit": unit,
                })
    return rows


def optics_lattice_plot_data(
    profile: Iterable[Mapping[str, Any]],
    elements: Iterable[Mapping[str, Any]],
    *,
    max_points: int = 1400,
) -> dict[str, Any]:
    """Build one shared-coordinate payload for optics and the lattice strip.

    The plotted domain is the common longitudinal interval of the Twiss profile
    and lattice rows.  This handles both supported one-cell call patterns:

    * a one-cell profile with full-ring elements; and
    * a full-ring profile with ``one_period_elements(...)``.

    Using the common interval is important because JuTrack normally returns a
    full-ring profile.  A figure labelled "one cell" must not silently display
    all repeated periods merely because the caller supplied that profile.
    Boundary elements are clipped to the selected interval, and the Twiss rows
    are filtered before down-sampling so a short cell remains well resolved.
    """
    profile_rows = [dict(point) for point in profile if isinstance(point, Mapping)]
    lattice_rows = lattice_plot_rows(elements)
    profile_positions = [
        float(point["s_m"])
        for point in profile_rows
        if _is_finite(point.get("s_m"))
    ]
    lattice_start = (
        min(float(row["s_start_m"]) for row in lattice_rows)
        if lattice_rows
        else None
    )
    lattice_end = (
        max(float(row["s_end_m"]) for row in lattice_rows)
        if lattice_rows
        else None
    )
    if profile_positions and lattice_start is not None and lattice_end is not None:
        profile_start = min(profile_positions)
        profile_end = max(profile_positions)
        common_start = max(profile_start, lattice_start)
        common_end = min(profile_end, lattice_end)
        common_tolerance = max(
            1.0e-12,
            max(profile_end - profile_start, lattice_end - lattice_start) * 1.0e-12,
        )
        if common_end < common_start - common_tolerance:
            raise ValueError("optics profile and lattice have no common longitudinal interval")
        s_start = common_start
        s_end = common_end
    elif profile_positions:
        s_start = min(profile_positions)
        s_end = max(profile_positions)
    elif lattice_start is not None and lattice_end is not None:
        s_start = lattice_start
        s_end = lattice_end
    else:
        return {"rows": [], "lattice_rows": [], "s_domain_m": [0.0, 1.0]}

    if s_end <= s_start:
        s_end = s_start + 1.0
    tolerance = max(1.0e-12, (s_end - s_start) * 1.0e-12)
    scoped_profile = [
        point
        for point in profile_rows
        if _is_finite(point.get("s_m"))
        and s_start - tolerance <= float(point["s_m"]) <= s_end + tolerance
    ]
    optics_rows = optics_plot_rows(scoped_profile, max_points=max_points)
    aligned_lattice: list[dict[str, Any]] = []
    for row in lattice_rows:
        start = float(row["s_start_m"])
        end = float(row["s_end_m"])
        if end < s_start - tolerance or start > s_end + tolerance:
            continue
        aligned_lattice.append({
            **row,
            "s_start_m": max(s_start, start),
            "s_end_m": min(s_end, end),
        })
    return {
        "rows": optics_rows,
        "lattice_rows": aligned_lattice,
        "s_domain_m": [s_start, s_end],
        "render_contract": {
            "kind": "combined_one_cell_optics_lattice",
            "required_series": ["beta_x", "beta_y", "Dx"],
            "beta_axis": "left",
            "dispersion_axis": "right",
            "lattice_strip": "bottom",
            "shared_longitudinal_axis": True,
        },
    }


def render_optics_lattice_svg(
    plot_data: Mapping[str, Any],
    path: Path | str,
    *,
    title: str = "One-cell optics and lattice",
    subtitle: str = "",
    width: int = 1200,
    height: int = 720,
) -> Path:
    """Render beta functions, dispersion, and the lattice in one dependency-free SVG.

    This is the canonical file renderer for research-agent optics artifacts. It
    deliberately keys series by their ASCII ``series`` identifiers rather than
    localized display labels, so Windows code pages cannot silently erase the
    beta functions. The detailed optics and the magnet strip always share one
    longitudinal coordinate.
    """

    destination = Path(path)
    if destination.suffix.lower() != ".svg":
        raise ValueError(
            "render_optics_lattice_svg requires a .svg output path; the extension "
            "must match the generated image bytes"
        )
    if width < 640 or height < 420:
        raise ValueError("combined optics SVG must be at least 640 by 420 pixels")
    rows = [dict(row) for row in plot_data.get("rows", ()) if isinstance(row, Mapping)]
    lattice_rows = [
        dict(row) for row in plot_data.get("lattice_rows", ())
        if isinstance(row, Mapping)
    ]
    series: dict[str, list[tuple[float, float]]] = {
        "beta_x": [], "beta_y": [], "Dx": [],
    }
    display_to_series = {"βx": "beta_x", "βy": "beta_y", "Dx": "Dx"}
    for row in rows:
        key = str(row.get("series", "")) or display_to_series.get(
            str(row.get("quantity", "")), ""
        )
        if key in series and _is_finite(row.get("s_m")) and _is_finite(row.get("value")):
            series[key].append((float(row["s_m"]), float(row["value"])))
    missing = [key for key, values in series.items() if not values]
    if missing:
        raise ValueError(
            "combined optics plot is missing required series: " + ", ".join(missing)
        )
    domain = plot_data.get("s_domain_m", ())
    if (
        not isinstance(domain, (list, tuple))
        or len(domain) != 2
        or not all(_is_finite(value) for value in domain)
    ):
        all_s = [s for values in series.values() for s, _ in values]
        s_start, s_end = min(all_s), max(all_s)
    else:
        s_start, s_end = float(domain[0]), float(domain[1])
    if s_end <= s_start:
        raise ValueError("combined optics plot needs a positive longitudinal span")

    left, right, top = 92.0, float(width - 96), 92.0
    optics_bottom = float(height - 150)
    strip_top, strip_bottom = float(height - 118), float(height - 58)
    plot_width = right - left
    optics_height = optics_bottom - top
    beta_max = max(value for key in ("beta_x", "beta_y") for _, value in series[key])
    beta_ticks = _nice_ticks(0.0, max(1.0, 1.05 * beta_max), target_count=6)
    beta_min, beta_max = beta_ticks[0], beta_ticks[-1]
    dispersion_values = [value for _, value in series["Dx"]]
    dispersion_ticks = _nice_ticks(
        min(0.0, min(dispersion_values)),
        max(0.0, max(dispersion_values)),
        target_count=6,
        include_zero=True,
        pad_fraction=0.06,
    )
    dispersion_min, dispersion_max = dispersion_ticks[0], dispersion_ticks[-1]

    def x_position(value: float) -> float:
        return left + (value - s_start) / (s_end - s_start) * plot_width

    def beta_y(value: float) -> float:
        return optics_bottom - (
            (value - beta_min) / (beta_max - beta_min)
        ) * optics_height

    def dispersion_y(value: float) -> float:
        return optics_bottom - (
            (value - dispersion_min) / (dispersion_max - dispersion_min)
        ) * optics_height

    def polyline(values: list[tuple[float, float]], transform: Any) -> str:
        return " ".join(
            f"{x_position(s):.2f},{transform(value):.2f}" for s, value in values
        )

    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.axis{stroke:#303030;stroke-width:1.2}'
        '.grid{stroke:#e5e7eb;stroke-width:1}.dispersion-zero{stroke:#2b8c4b;stroke-width:1.35;stroke-dasharray:3 3;opacity:.75}'
        '.tick{font-size:13px}.label{font-size:15px}'
        '.legend{font-size:14px}.title{font-size:21px;font-weight:600}.subtitle{font-size:14px;fill:#555}</style>',
        f'<text class="title" x="{left:.0f}" y="32">{escape(title)}</text>',
    ]
    if subtitle:
        svg.append(
            f'<text class="subtitle" x="{left:.0f}" y="56">{escape(subtitle)}</text>'
        )
    for tick in range(6):
        fraction = tick / 5.0
        x = left + fraction * plot_width
        s_value = s_start + fraction * (s_end - s_start)
        svg.extend([
            f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{strip_bottom:.2f}"/>',
            f'<text class="tick" x="{x:.2f}" y="{height - 28}" text-anchor="middle">{s_value:.2f}</text>',
        ])
    for beta_value in beta_ticks:
        y = beta_y(beta_value)
        svg.extend([
            f'<line class="grid" x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{left - 10:.2f}" y="{y + 4:.2f}" text-anchor="end">{beta_value:.2f}</text>',
        ])
    for dispersion_value in dispersion_ticks:
        y = dispersion_y(dispersion_value)
        if abs(dispersion_value) <= 1.0e-15:
            svg.append(
                f'<line class="dispersion-zero" x1="{left:.2f}" y1="{y:.2f}" '
                f'x2="{right:.2f}" y2="{y:.2f}"/>'
            )
        svg.extend([
            f'<line x1="{right:.2f}" y1="{y:.2f}" x2="{right + 6:.2f}" '
            f'y2="{y:.2f}" stroke="#2b8c4b"/>',
            f'<text class="tick" x="{right + 10:.2f}" y="{y + 4:.2f}" '
            f'fill="#2b8c4b">{dispersion_value:.4g}</text>',
        ])
    svg.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{optics_bottom}"/>',
        f'<line class="axis" x1="{right}" y1="{top}" x2="{right}" y2="{optics_bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{optics_bottom}" x2="{right}" y2="{optics_bottom}"/>',
        f'<polyline fill="none" stroke="#1769aa" stroke-width="2.4" points="{polyline(series["beta_x"], beta_y)}"/>',
        f'<polyline fill="none" stroke="#d95f02" stroke-width="2.4" points="{polyline(series["beta_y"], beta_y)}"/>',
        f'<polyline fill="none" stroke="#2b8c4b" stroke-width="2.2" stroke-dasharray="8 4" points="{polyline(series["Dx"], dispersion_y)}"/>',
        f'<text class="label" x="18" y="{(top + optics_bottom) / 2:.2f}" transform="rotate(-90 18 {(top + optics_bottom) / 2:.2f})" text-anchor="middle">beta x, beta y [m]</text>',
        f'<text class="label" x="{width - 18}" y="{(top + optics_bottom) / 2:.2f}" transform="rotate(90 {width - 18} {(top + optics_bottom) / 2:.2f})" text-anchor="middle" fill="#2b8c4b">Dx [m]</text>',
        f'<text class="label" x="{(left + right) / 2:.2f}" y="{height - 8}" text-anchor="middle">s [m]</text>',
        f'<line x1="{left}" y1="{strip_top + 30}" x2="{right}" y2="{strip_top + 30}" stroke="#111827" stroke-width="1.25"/>',
    ])
    glyph_scale = 27.0
    for row in lattice_rows:
        if not (_is_finite(row.get("s_start_m")) and _is_finite(row.get("s_end_m"))):
            continue
        # As in pyJuTrack, drifts are the black reference orbit; only magnets
        # receive filled glyphs. This keeps a dense MBA strip legible.
        if str(row.get("type", "")).upper() == "DRIFT":
            continue
        x0 = x_position(float(row["s_start_m"]))
        x1 = x_position(float(row["s_end_m"]))
        y0 = strip_top + 30.0 - _finite(row.get("y1"), 0.0) * glyph_scale
        y1 = strip_top + 30.0 - _finite(row.get("y0"), 0.0) * glyph_scale
        color = escape(str(row.get("color", "#969696")))
        name = escape(str(row.get("name", "")))
        kind = escape(str(row.get("type", "")))
        svg.append(
            f'<rect x="{x0:.2f}" y="{min(y0, y1):.2f}" width="{max(0.8, x1 - x0):.2f}" '
            f'height="{max(1.2, abs(y1 - y0)):.2f}" fill="{color}" stroke="#444" stroke-width="0.45">'
            f'<title>{name} ({kind})</title></rect>'
        )
    legend_x = right - 360.0
    for offset, color, label, dashed in (
        (0.0, "#1769aa", "beta x", False),
        (118.0, "#d95f02", "beta y", False),
        (236.0, "#2b8c4b", "Dx", True),
    ):
        dash = ' stroke-dasharray="8 4"' if dashed else ""
        svg.extend([
            f'<line x1="{legend_x + offset:.2f}" y1="72" x2="{legend_x + offset + 34:.2f}" y2="72" stroke="{color}" stroke-width="2.5"{dash}/>',
            f'<text class="legend" x="{legend_x + offset + 40:.2f}" y="77">{label}</text>',
        ])
    svg.append('</svg>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8", newline="\n")
    return destination


def render_optics_lattice_matplotlib(
    plot_data: Mapping[str, Any],
    path: Path | str,
    *,
    title: str = "One-cell optics and lattice",
    subtitle: str = "",
    width_inches: float = 12.0,
    height_inches: float = 7.2,
    dpi: int = 140,
) -> Path:
    """Render the canonical combined optics figure with Matplotlib.

    This is the raster/PDF companion to :func:`render_optics_lattice_svg`.
    Both renderers consume the same validated plot payload, use separate beta
    and dispersion axes, and place the physical lattice on the shared
    longitudinal axis.  Keeping this path in the public plotting layer avoids
    the common and misleading fallback where generated programs plot beta and
    dispersion but silently omit the lattice.
    """

    destination = Path(path)
    if destination.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError(
            "Matplotlib optics/lattice output must use a .png or .pdf path"
        )
    if width_inches <= 0.0 or height_inches <= 0.0 or dpi < 72:
        raise ValueError("combined optics figure dimensions and dpi must be positive")

    rows = [dict(row) for row in plot_data.get("rows", ()) if isinstance(row, Mapping)]
    lattice_rows = [
        dict(row)
        for row in plot_data.get("lattice_rows", ())
        if isinstance(row, Mapping)
    ]
    series: dict[str, list[tuple[float, float]]] = {
        "beta_x": [],
        "beta_y": [],
        "Dx": [],
    }
    display_to_series = {"尾x": "beta_x", "尾y": "beta_y", "Dx": "Dx"}
    for row in rows:
        key = str(row.get("series", "")) or display_to_series.get(
            str(row.get("quantity", "")), ""
        )
        if key in series and _is_finite(row.get("s_m")) and _is_finite(row.get("value")):
            series[key].append((float(row["s_m"]), float(row["value"])))
    missing = [key for key, values in series.items() if not values]
    if missing:
        raise ValueError(
            "combined optics plot is missing required series: " + ", ".join(missing)
        )

    domain = plot_data.get("s_domain_m", ())
    if (
        not isinstance(domain, (list, tuple))
        or len(domain) != 2
        or not all(_is_finite(value) for value in domain)
    ):
        all_s = [s for values in series.values() for s, _ in values]
        s_start, s_end = min(all_s), max(all_s)
    else:
        s_start, s_end = float(domain[0]), float(domain[1])
    if s_end <= s_start:
        raise ValueError("combined optics plot needs a positive longitudinal span")

    # Imported lazily so data extraction and SVG output stay dependency-light.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure = plt.figure(figsize=(width_inches, height_inches), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(5.0, 1.05), hspace=0.03)
    optics_axis = figure.add_subplot(grid[0])
    dispersion_axis = optics_axis.twinx()
    lattice_axis = figure.add_subplot(grid[1], sharex=optics_axis)

    beta_x_line, = optics_axis.plot(
        *zip(*series["beta_x"]), color="#1769aa", linewidth=2.0, label=r"$\beta_x$"
    )
    beta_y_line, = optics_axis.plot(
        *zip(*series["beta_y"]), color="#d95f02", linewidth=2.0, label=r"$\beta_y$"
    )
    dispersion_line, = dispersion_axis.plot(
        *zip(*series["Dx"]),
        color="#2b8c4b",
        linewidth=1.9,
        linestyle="--",
        label=r"$D_x$",
    )

    beta_maximum = max(value for key in ("beta_x", "beta_y") for _, value in series[key])
    beta_ticks = _nice_ticks(0.0, max(1.0, 1.05 * beta_maximum), target_count=6)
    optics_axis.set_ylim(beta_ticks[0], beta_ticks[-1])
    optics_axis.set_yticks(beta_ticks)
    dispersion_values = [value for _, value in series["Dx"]]
    dispersion_ticks = _nice_ticks(
        min(0.0, min(dispersion_values)),
        max(0.0, max(dispersion_values)),
        target_count=6,
        include_zero=True,
        pad_fraction=0.06,
    )
    dispersion_axis.set_ylim(dispersion_ticks[0], dispersion_ticks[-1])
    dispersion_axis.set_yticks(dispersion_ticks)
    dispersion_axis.axhline(0.0, color="#2b8c4b", linewidth=0.9, linestyle=":", alpha=0.8)

    optics_axis.set_ylabel(r"$\beta_x,\ \beta_y$ [m]")
    dispersion_axis.set_ylabel(r"$D_x$ [m]", color="#2b8c4b")
    dispersion_axis.tick_params(axis="y", colors="#2b8c4b")
    optics_axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.75)
    optics_axis.tick_params(axis="x", labelbottom=False)
    optics_axis.legend(
        handles=(beta_x_line, beta_y_line, dispersion_line),
        loc="upper right",
        ncol=3,
        frameon=True,
    )

    lattice_axis.axhline(0.0, color="#111827", linewidth=1.0)
    for row in lattice_rows:
        if not (_is_finite(row.get("s_start_m")) and _is_finite(row.get("s_end_m"))):
            continue
        if str(row.get("type", "")).upper() == "DRIFT":
            continue
        start = float(row["s_start_m"])
        end = float(row["s_end_m"])
        y0 = _finite(row.get("y0"), -0.15)
        y1 = _finite(row.get("y1"), 0.15)
        lattice_axis.add_patch(
            Rectangle(
                (start, min(y0, y1)),
                max(1.0e-9, end - start),
                max(1.0e-6, abs(y1 - y0)),
                facecolor=str(row.get("color", "#969696")),
                edgecolor="#444444",
                linewidth=0.45,
            )
        )
    lattice_axis.set_ylim(-1.05, 1.05)
    lattice_axis.set_yticks([])
    lattice_axis.set_ylabel("Lattice", rotation=0, labelpad=32, va="center")
    lattice_axis.set_xlabel("s [m]")
    lattice_axis.set_xlim(s_start, s_end)
    lattice_axis.spines[["left", "right", "top"]].set_visible(False)

    if subtitle:
        figure.suptitle(f"{title}\n{subtitle}", fontsize=15)
    else:
        figure.suptitle(title, fontsize=15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination


def element_span_m(elements: Iterable[Mapping[str, Any]]) -> float:
    """Return the longitudinal span of a contiguous element sequence."""
    materialized = [dict(element) for element in elements]
    if not materialized:
        return 0.0
    try:
        start = float(materialized[0].get("s_start_m", 0.0))
        end = float(materialized[-1]["s_end_m"])
        if math.isfinite(start) and math.isfinite(end) and end >= start:
            return end - start
    except (KeyError, TypeError, ValueError):
        pass
    return sum(
        max(0.0, _finite(element.get("length_m"), 0.0))
        for element in materialized
    )


def profile_through(
    profile: Iterable[Mapping[str, Any]], length_m: float
) -> list[dict[str, Any]]:
    """Extract and rebase a Twiss profile through a requested length."""
    materialized = [dict(point) for point in profile]
    if not materialized or length_m <= 0.0:
        return materialized
    origin = _finite(materialized[0].get("s_m"), 0.0)
    tolerance = max(1.0e-9, abs(length_m) * 1.0e-10)
    selected: list[dict[str, Any]] = []
    for point in materialized:
        if not _is_finite(point.get("s_m")):
            continue
        relative_s = float(point["s_m"]) - origin
        if relative_s > length_m + tolerance:
            break
        selected.append({**point, "s_m": relative_s})
    return selected


def chromaticity_plot_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare natural, requested, and achieved chromaticities by plane."""
    rows: list[dict[str, Any]] = []
    for label, key in (
        ("Natural", "natural_chromaticities"),
        ("Target", "target_chromaticities"),
        ("Corrected", "corrected_chromaticities"),
    ):
        values = result.get(key)
        if not isinstance(values, (list, tuple)):
            continue
        for index, plane in enumerate(("x", "y")):
            if index < len(values) and _is_finite(values[index]):
                rows.append({
                    "state": label,
                    "plane": plane,
                    "chromaticity": float(values[index]),
                })
    return rows


def render_dynamic_aperture_svg(
    scans: Iterable[Mapping[str, Any]],
    path: Path | str,
    *,
    title: str = "JuTrack dynamic aperture",
    width: int = 900,
    height: int = 680,
) -> Path:
    """Render on/off-momentum dynamic-aperture boundaries as an SVG."""

    destination = Path(path)
    if destination.suffix.lower() != ".svg":
        raise ValueError("dynamic-aperture renderer requires a .svg output path")
    materialized = [dict(scan) for scan in scans if isinstance(scan, Mapping)]
    series = []
    for scan in materialized:
        points = [
            (float(point[0]) * 1.0e3, float(point[1]) * 1.0e3)
            for point in scan.get("da_boundary", ())
            if isinstance(point, (list, tuple)) and len(point) >= 2
            and _is_finite(point[0]) and _is_finite(point[1])
        ]
        if points:
            if any(x < -1.0e-12 for x, _ in points):
                displayed = points
            else:
                displayed = points + [(-x, y) for x, y in reversed(points[1:])]
            series.append((float(scan.get("dp", 0.0)), displayed))
    if not series:
        raise ValueError("dynamic-aperture plot has no finite boundary points")
    max_x = 1.12 * (max(abs(x) for _, points in series for x, _ in points) or 1.0)
    max_y = 1.12 * (max(abs(y) for _, points in series for _, y in points) or 1.0)
    left, right, top, bottom = 84.0, width - 42.0, 72.0, height - 70.0

    def xp(x: float) -> float:
        return left + (x + max_x) / (2.0 * max_x) * (right - left)

    def yp(y: float) -> float:
        return bottom - y / max_y * (bottom - top)

    colors = ("#1769aa", "#d95f02", "#2b8c4b", "#7b3294", "#666666")
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.grid{stroke:#dde1e6;stroke-width:1}.axis{stroke:#333;stroke-width:1.3}.tick{font-size:12px}.title{font-size:20px;font-weight:600}.legend{font-size:13px}</style>',
        f'<text class="title" x="{left}" y="32">{escape(title)}</text>',
    ]
    for index in range(6):
        fraction = index / 5.0
        x = left + fraction * (right - left)
        value = -max_x + 2.0 * max_x * fraction
        svg.extend([
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}"/>',
            f'<text class="tick" x="{x:.2f}" y="{bottom + 22}" text-anchor="middle">{value:.1f}</text>',
        ])
        y = bottom - fraction * (bottom - top)
        yvalue = max_y * fraction
        svg.extend([
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{left - 9}" y="{y + 4:.2f}" text-anchor="end">{yvalue:.1f}</text>',
        ])
    svg.extend([
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<text x="{(left + right) / 2}" y="{height - 18}" text-anchor="middle">x [mm]</text>',
        f'<text x="18" y="{(top + bottom) / 2}" transform="rotate(-90 18 {(top + bottom) / 2})" text-anchor="middle">y [mm]</text>',
    ])
    for index, (dp, points) in enumerate(series):
        color = colors[index % len(colors)]
        rendered = " ".join(f"{xp(x):.2f},{yp(y):.2f}" for x, y in points)
        svg.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{rendered}"/>'
        )
        lx = right - 150.0
        ly = 50.0 + 20.0 * index
        svg.extend([
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 30}" y2="{ly}" stroke="{color}" stroke-width="2.2"/>',
            f'<text class="legend" x="{lx + 38}" y="{ly + 4}">dp={dp:+.1%}</text>',
        ])
    svg.append("</svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8", newline="\n")
    return destination


def render_momentum_acceptance_svg(
    rows: Iterable[Mapping[str, Any]],
    path: Path | str,
    *,
    title: str = "JuTrack local momentum acceptance",
    width: int = 1000,
    height: int = 560,
) -> Path:
    """Render positive and negative tracked momentum acceptance versus s."""

    destination = Path(path)
    if destination.suffix.lower() != ".svg":
        raise ValueError("momentum-acceptance renderer requires a .svg output path")
    points = [
        (float(row["s_m"]), float(row["negative_dp"]), float(row["positive_dp"]))
        for row in rows
        if isinstance(row, Mapping)
        and all(_is_finite(row.get(key)) for key in ("s_m", "negative_dp", "positive_dp"))
    ]
    if not points:
        raise ValueError("momentum-acceptance plot has no finite rows")
    s_min, s_max = min(p[0] for p in points), max(p[0] for p in points)
    if s_max <= s_min:
        s_max = s_min + 1.0
    dp_max = max(abs(value) for _, negative, positive in points for value in (negative, positive)) or 0.01
    left, right, top, bottom = 86.0, width - 36.0, 64.0, height - 62.0

    def xp(s: float) -> float:
        return left + (s - s_min) / (s_max - s_min) * (right - left)

    def yp(dp: float) -> float:
        return top + (dp_max - dp) / (2.0 * dp_max) * (bottom - top)

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.grid{stroke:#dde1e6}.axis{stroke:#333;stroke-width:1.3}.tick{font-size:12px}.title{font-size:20px;font-weight:600}</style>',
        f'<text class="title" x="{left}" y="30">{escape(title)}</text>',
    ]
    for index in range(6):
        fraction = index / 5.0
        x = left + fraction * (right - left)
        s = s_min + fraction * (s_max - s_min)
        svg.extend([
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}"/>',
            f'<text class="tick" x="{x:.2f}" y="{bottom + 21}" text-anchor="middle">{s:.2f}</text>',
        ])
        y = top + fraction * (bottom - top)
        dp = dp_max * (1.0 - 2.0 * fraction)
        svg.extend([
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{dp:+.1%}</text>',
        ])
    negative = " ".join(f"{xp(s):.2f},{yp(value):.2f}" for s, value, _ in points)
    positive = " ".join(f"{xp(s):.2f},{yp(value):.2f}" for s, _, value in points)
    svg.extend([
        f'<line class="axis" x1="{left}" y1="{yp(0.0):.2f}" x2="{right}" y2="{yp(0.0):.2f}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<polyline fill="none" stroke="#1769aa" stroke-width="2.4" points="{positive}"/>',
        f'<polyline fill="none" stroke="#d95f02" stroke-width="2.4" points="{negative}"/>',
        f'<text x="{(left + right) / 2}" y="{height - 13}" text-anchor="middle">s [m]</text>',
        f'<text x="18" y="{(top + bottom) / 2}" transform="rotate(-90 18 {(top + bottom) / 2})" text-anchor="middle">dp/p</text>',
        f'<text x="{right - 145}" y="45" fill="#1769aa">positive</text>',
        f'<text x="{right - 70}" y="45" fill="#d95f02">negative</text>',
        "</svg>",
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8", newline="\n")
    return destination


def dynamic_aperture_plot_rows(scans: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert JuTrack dynamic-aperture boundaries from metres to millimetres."""
    rows: list[dict[str, Any]] = []
    for scan_index, scan in enumerate(scans):
        boundary = scan.get("da_boundary")
        if not isinstance(boundary, (list, tuple)):
            continue
        dp = _finite(scan.get("dp"), 0.0)
        points: list[tuple[float, float]] = []
        for point in boundary:
            if (
                isinstance(point, (list, tuple))
                and len(point) >= 2
                and _is_finite(point[0])
                and _is_finite(point[1])
            ):
                points.append((1000.0 * float(point[0]), 1000.0 * float(point[1])))
        if len(points) > 2 and points[0] != points[-1]:
            points.append(points[0])
        for point_index, (x_mm, y_mm) in enumerate(points):
            rows.append({
                "scan": scan_index + 1,
                "point_order": point_index,
                "momentum_offset": f"{100.0 * dp:+.1f}%",
                "dp": dp,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "area_mm2": _finite(scan.get("da_area_mm2"), 0.0),
                "nturns": int(_finite(scan.get("nturns"), 0.0)),
            })
    return rows


def robustness_plot_rows(seeds: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return before/after closed-orbit RMS values in micrometres."""
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        seed_value = seed.get("seed", index + 1)
        for state, key in (("Before correction", "rms_before_m"), ("After correction", "rms_after_m")):
            if _is_finite(seed.get(key)):
                rows.append({
                    "seed": str(seed_value),
                    "state": state,
                    "rms_um": 1.0e6 * float(seed[key]),
                    "passed": bool(seed.get("passed", False)),
                })
    return rows


def radiation_integral_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract I1-I5 without erasing their different physical dimensions."""
    rows: list[dict[str, Any]] = []
    units = {"I1": "m", "I2": "m⁻¹", "I3": "m⁻²", "I4": "m⁻¹", "I5": "m⁻¹"}
    for key in ("I1", "I2", "I3", "I4", "I5"):
        if _is_finite(result.get(key)):
            rows.append({"integral": key, "value": float(result[key]), "unit": units[key]})
    return rows


def damping_partition_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract transverse and longitudinal damping partitions."""
    rows: list[dict[str, Any]] = []
    for key, plane in (("Jx", "Horizontal"), ("Jy", "Vertical"), ("Jz", "Longitudinal")):
        if _is_finite(result.get(key)):
            rows.append({"plane": plane, "partition": float(result[key])})
    return rows


def repeat_periodic_profile(
    profile: Iterable[Mapping[str, Any]],
    *,
    repeats: int = 2,
) -> list[dict[str, Any]]:
    """Repeat one closed-cell profile while retaining cell provenance."""
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    base = [dict(point) for point in profile if _is_finite(point.get("s_m"))]
    if not base:
        return []
    base.sort(key=lambda point: float(point["s_m"]))
    origin = float(base[0]["s_m"])
    cell_length = float(base[-1]["s_m"]) - origin
    if cell_length <= 0.0:
        return []

    repeated: list[dict[str, Any]] = []
    for cell_index in range(repeats):
        for point_index, point in enumerate(base):
            if cell_index and point_index == 0:
                # The preceding exit already represents the shared boundary.
                continue
            row = dict(point)
            row["s_m"] = float(point["s_m"]) - origin + cell_index * cell_length
            row["cell"] = cell_index + 1
            repeated.append(row)
    return repeated


def periodic_closure_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare entrance and exit optics using normalized beta residuals."""
    endpoint = raw.get("periodic_endpoint")
    if isinstance(endpoint, Mapping):
        entrance = endpoint.get("entrance")
        exit_point = endpoint.get("exit")
    else:
        entrance = exit_point = None
    if not isinstance(entrance, Mapping) or not isinstance(exit_point, Mapping):
        profile = raw.get("twiss_profile") or raw.get("uc_twiss_profile") or []
        if len(profile) < 2:
            return []
        entrance, exit_point = profile[0], profile[-1]

    residuals = raw.get("periodic_residuals", {})
    rows: list[dict[str, Any]] = []
    labels = {
        "beta_x": "beta x", "alpha_x": "alpha x",
        "beta_y": "beta y", "alpha_y": "alpha y",
        "Dx": "Dx", "Dpx": "Dpx",
    }
    for key, label in labels.items():
        if not (_is_finite(entrance.get(key)) and _is_finite(exit_point.get(key))):
            continue
        start, end = float(entrance[key]), float(exit_point[key])
        residual = residuals.get(key)
        if not _is_finite(residual):
            residual = (end - start) / start if key.startswith("beta") and start else end - start
        rows.append({
            "quantity": label,
            "entrance": start,
            "exit": end,
            "closure_residual": float(residual),
            "residual_definition": "relative" if key.startswith("beta") else "absolute",
        })
    return rows


def _bend_radiation_plot_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in evidence.get("rows", ())
            if isinstance(row, Mapping) and _is_finite(row.get("I2_fraction"))
            and _is_finite(row.get("I5_fraction"))]
    if not rows:
        raise ValueError("bend-radiation plot has no finite I2/I5 contribution rows")
    return rows


def render_bend_radiation_contributions_matplotlib(
    evidence: Mapping[str, Any], path: Path | str, *,
    title: str = "Per-bend radiation-integral contributions",
    subtitle: str = "I2 and I5 fractions of the full ring",
) -> Path:
    """Headless PNG/PDF companion to the SVG chart, using the same measured rows."""
    destination = Path(path)
    if destination.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("Matplotlib bend-radiation output must use a .png or .pdf path")
    rows = _bend_radiation_plot_rows(evidence)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axis = plt.subplots(figsize=(12.0, 6.5), constrained_layout=True)
    try:
        for offset, key, color, label in ((-.18, "I2_fraction", "#4c78a8", r"$I_2$"),
                                         (.18, "I5_fraction", "#e45756", r"$I_5$")):
            axis.bar([index + offset for index in range(len(rows))],
                     [float(row[key]) for row in rows], width=.36, color=color, label=label)
        labels = [str(row.get("bend") or row.get("name") or f"B{i + 1}") for i, row in enumerate(rows)]
        rotate = len(rows) > 12 or max(map(len, labels)) > 16
        axis.set_xticks(range(len(rows)), labels, rotation=35 if rotate else 0,
                        ha="right" if rotate else "center")
        maximum = max(float(row[key]) for row in rows for key in ("I2_fraction", "I5_fraction"))
        ticks = _nice_ticks(0.0, max(.05, 1.08 * maximum), target_count=6)
        axis.set_ylim(ticks[0], ticks[-1])
        axis.set_yticks(ticks)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.set_ylabel("Ring contribution")
        axis.set_xlabel("Bend (saved element order)")
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e5e7eb", linewidth=.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper right", ncol=2)
        figure.suptitle(f"{title}\n{subtitle}" if subtitle else title, fontsize=14)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=140, format=destination.suffix[1:].lower())
    finally:
        plt.close(figure)
    return destination


def render_bend_radiation_contributions_svg(
    evidence: Mapping[str, Any],
    path: Path | str,
    *,
    title: str = "Per-bend radiation-integral contributions",
    subtitle: str = "I2 and I5 fractions of the full ring",
    width: int = 1040,
    height: int = 620,
) -> Path:
    """Render a grouped per-bend I2/I5 contribution chart.

    The fractions are expected to come from
    a saved-program bend-attribution calculation, whose
    totals are anchored to the exact JuTrack full-ring radiation integrals.
    Keeping both integrals on a shared percentage axis makes excess I5 in a
    weakly bending or matching magnet immediately visible.
    """

    destination = Path(path)
    if destination.suffix.lower() != ".svg":
        raise ValueError("bend-radiation renderer requires a .svg output path")
    rows = _bend_radiation_plot_rows(evidence)

    left, right, top, bottom = 86.0, float(width - 44), 112.0, float(height - 86)
    chart_width = right - left
    chart_height = bottom - top
    maximum_fraction = max(
        max(float(row["I2_fraction"]), float(row["I5_fraction"])) for row in rows
    )
    y_ticks = _nice_ticks(0.0, max(0.05, 1.08 * maximum_fraction), target_count=6)
    y_max = y_ticks[-1]
    group_width = chart_width / len(rows)
    bar_width = min(26.0, 0.30 * group_width)

    def yp(value: float) -> float:
        return bottom - value / y_max * chart_height

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.grid{stroke:#e5e7eb;stroke-width:1}'
        '.axis{stroke:#303030;stroke-width:1.2}.tick{font-size:12px}.title{font-size:21px;font-weight:600}'
        '.subtitle{font-size:13px;fill:#5f6368}.legend{font-size:13px}</style>',
        f'<text class="title" x="{left}" y="34">{escape(title)}</text>',
        f'<text class="subtitle" x="{left}" y="58">{escape(subtitle)}</text>',
    ]
    for value in y_ticks:
        y = yp(value)
        svg.extend([
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{100.0 * value:.1f}%</text>',
        ])
    for index, row in enumerate(rows):
        center = left + (index + 0.5) * group_width
        label = escape(str(row.get("bend") or row.get("name") or f"B{index + 1}"))
        i2 = float(row["I2_fraction"])
        i5 = float(row["I5_fraction"])
        ratio = float(row.get("I5_over_I2", math.nan))
        for offset, value, color, integral in (
            (-bar_width, i2, "#4c78a8", "I2"),
            (0.0, i5, "#e45756", "I5"),
        ):
            x = center + offset
            y = yp(value)
            tooltip = (
                f"{label}: {integral} {100.0 * value:.3f}%"
                + (f"; I5/I2={ratio:.6g}" if math.isfinite(ratio) else "")
            )
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bottom - y:.2f}" fill="{color}" opacity="0.9">'
                f'<title>{tooltip}</title></rect>'
            )
        svg.append(
            f'<text class="tick" x="{center:.2f}" y="{bottom + 22:.2f}" '
            f'text-anchor="middle">{label}</text>'
        )
    svg.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<text x="20" y="{(top + bottom) / 2:.2f}" '
        f'transform="rotate(-90 20 {(top + bottom) / 2:.2f})" text-anchor="middle">ring contribution</text>',
        f'<rect x="{right - 150}" y="76" width="14" height="14" fill="#4c78a8"/>',
        f'<text class="legend" x="{right - 130}" y="88">I2</text>',
        f'<rect x="{right - 82}" y="76" width="14" height="14" fill="#e45756"/>',
        f'<text class="legend" x="{right - 62}" y="88">I5</text>',
        '</svg>',
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8", newline="\n")
    return destination


def _nice_ticks(
    lower: float,
    upper: float,
    *,
    target_count: int = 6,
    include_zero: bool = False,
    pad_fraction: float = 0.0,
) -> list[float]:
    """Return stable human-scale ticks, optionally guaranteeing an exact zero."""

    low, high = float(lower), float(upper)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("tick bounds must be finite")
    if low > high:
        low, high = high, low
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low
    if span <= 0.0:
        scale = max(abs(low), abs(high), 1.0)
        low, high = low - 0.5 * scale, high + 0.5 * scale
        span = high - low
    padding = max(0.0, float(pad_fraction)) * span
    low -= padding
    high += padding
    rough_step = (high - low) / max(2, int(target_count) - 1)
    magnitude = 10.0 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    factor = next(item for item in (1.0, 2.0, 2.5, 5.0, 10.0) if item >= normalized)
    step = factor * magnitude
    first = math.floor(low / step + 1.0e-12) * step
    last = math.ceil(high / step - 1.0e-12) * step
    count = int(round((last - first) / step)) + 1
    ticks = [first + index * step for index in range(max(2, count))]
    ticks = [0.0 if abs(value) <= step * 1.0e-12 else value for value in ticks]
    if include_zero and not any(value == 0.0 for value in ticks):
        ticks.append(0.0)
        ticks.sort()
    return ticks


def _element_height(kind: str, element: Mapping[str, Any]) -> tuple[float, float]:
    if kind in {"KQUAD", "QUAD", "QUADRUPOLE"}:
        sign = 1.0 if _finite(element.get("k1", element.get("k1_m2")), 1.0) >= 0.0 else -1.0
        return (0.0, 0.78 * sign) if sign > 0 else (0.78 * sign, 0.0)
    if kind in {"KSEXT", "SEXT", "SEXTUPOLE"}:
        sign = 1.0 if _finite(element.get("k2", element.get("k2_m3")), 1.0) >= 0.0 else -1.0
        return (0.0, 0.52 * sign) if sign > 0 else (0.52 * sign, 0.0)
    if kind in {"KOCT", "OCT"}:
        return -0.42, 0.42
    if kind in {"SBEND", "RBEND", "BEND"}:
        return -0.30, 0.30
    if kind in {"RFCAVITY", "RFCA"}:
        return -0.60, 0.60
    return -0.035, 0.035


def _finite(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
