"""Publication-quality figures for the BRILLIANCE discovery experiments.

Uses plotly (NOT matplotlib) to avoid the numpy BLAS / DLL crash on Windows.
All functions return ``plotly.graph_objects.Figure`` objects that can be
displayed inline, saved as PNG/SVG/HTML, or embedded in LaTeX via tikz.

Dependencies: plotly (``pip install plotly kaleido`` for static export).

All plot functions lazily import plotly at call time — the module can be
imported without plotly installed as long as no plot function is called.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from physics_core import _to_float


# Lazy plotly import — deferred to first call of _go()
_go_module = None


def _go():
    """Lazy-import and cache plotly.graph_objects."""
    global _go_module
    if _go_module is None:
        import plotly.graph_objects as _go_mod
        _go_module = _go_mod
    return _go_module


# ── Colour palette (colourblind-friendly, journal-safe) ─────────────────────

FAMILY_COLORS: dict[str, str] = {
    "FODO": "#0072B2",   # blue
    "DBA":  "#D55E00",   # vermillion
    "TBA":  "#009E73",   # bluish green
    "5BA":  "#CC79A7",   # reddish purple
    "6BA":  "#56B4E9",   # sky blue
    "7BA":  "#E69F00",   # orange
    "8BA":  "#F0E442",   # yellow
    "9BA":  "#000000",   # black
    "H7BA": "#8B4513",   # saddle brown
    "H6BA": "#4682B4",   # steel blue
    "MBA":  "#A0522D",   # sienna
}

FAMILY_MARKERS: dict[str, str] = {
    "FODO": "circle",
    "DBA":  "diamond",
    "TBA":  "triangle-up",
    "5BA":  "cross",
    "6BA":  "square",
    "7BA":  "star",
    "8BA":  "x",
    "9BA":  "hexagon",
    "H7BA": "diamond-open",
    "H6BA": "triangle-down",
    "MBA":  "star-triangle-up",
}

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
    linecolor="rgba(0,0,0,0.5)",
    title_font=dict(size=13, family="serif"),
)

_LAYOUT_BASE = dict(
    font=dict(family="serif", size=11, color="#333"),
    plot_bgcolor="#fafbfc",
    paper_bgcolor="white",
    margin=dict(l=60, r=40, t=50, b=50),
    legend=dict(
        font=dict(family="serif", size=10),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1,
    ),
)


def _family_color(family: str) -> str:
    return FAMILY_COLORS.get(family.upper(), "#999999")


def _family_marker(family: str) -> str:
    return FAMILY_MARKERS.get(family.upper(), "circle")


# ── Main plot functions ─────────────────────────────────────────────────────


def plot_pareto_frontier(
    results: list[dict[str, Any]],
    *,
    x_objective: str = "emittance",
    y_objective: str = "circumference",
    x_label: str = "Natural emittance εₓ (pm·rad)",
    y_label: str = "Circumference (m)",
    x_scale_pm: bool = True,
    title: str = "Pareto Frontier — Emittance vs. Circumference",
    highlight_anomalies: list[dict[str, Any]] | None = None,
):
    """Plot the Pareto frontier of a set of design results.

    All stable designs are shown as faint scatter points; the Pareto-optimal
    subset is highlighted as larger markers with family-coloured symbols.
    """
    # Separate stable vs. unstable
    stable = [
        r for r in results
        if r.get("result") and r["result"].get("stable") is True
        and not r.get("error")
    ]
    if not stable:
        fig = _go().Figure()
        fig.add_annotation(text="No stable designs", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Extract Pareto-optimal subset
    front = [
        r for r in stable
        if r.get("result")
    ]

    def _x(r: dict) -> float | None:
        val = _to_float(r["result"].get(x_objective)) if r.get("result") else None
        if val is None:
            return None
        return val * 1e12 if x_scale_pm else val

    def _y(r: dict) -> float | None:
        val = _to_float(r["result"].get(y_objective)) if r.get("result") else None
        if val is None:
            return None
        return val  # circumference already in m

    fig = _go().Figure()

    # All stable designs (faint background)
    families_in_data = sorted(set(
        r.get("family", "") for r in front if r.get("family")
    ))
    for family in families_in_data:
        family_runs = [
            r for r in front if r.get("family", "") == family
        ]
        xs = [_x(r) for r in family_runs]
        ys = [_y(r) for r in family_runs]
        valid = [
            (x, y) for x, y in zip(xs, ys)
            if x is not None and y is not None
            and math.isfinite(x) and math.isfinite(y)
        ]
        if not valid:
            continue
        vx, vy = zip(*valid)
        fig.add_trace(_go().Scatter(
            x=vx, y=vy,
            mode="markers",
            name=f"{family} (all)",
            marker=dict(
                symbol=_family_marker(family),
                size=7,
                color=_family_color(family),
                opacity=0.3,
                line=dict(width=0),
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{family}</b><br>"
                f"εₓ: {{x:.1f}} pm·rad<br>"
                f"C: {{y:.0f}} m"
            ),
        ))

    # Pareto front computation: for each design, check domination by any other
    pareto = compute_pareto_front_plot(front, x_objective, y_objective, x_scale_pm)

    for family in families_in_data:
        family_pareto = [
            r for r in pareto if r.get("family", "") == family
        ]
        if not family_pareto:
            continue
        xs = [_x(r) for r in family_pareto]
        ys = [_y(r) for r in family_pareto]
        valid = [
            (x, y) for x, y in zip(xs, ys)
            if x is not None and y is not None
            and math.isfinite(x) and math.isfinite(y)
        ]
        if not valid:
            continue
        # Sort by x to draw the frontier line
        valid.sort(key=lambda p: p[0])
        vx, vy = zip(*valid)

        fig.add_trace(_go().Scatter(
            x=vx, y=vy,
            mode="lines+markers",
            name=f"{family} (Pareto)",
            marker=dict(
                symbol=_family_marker(family),
                size=11,
                color=_family_color(family),
                line=dict(width=1.5, color="white"),
            ),
            line=dict(width=2.5, color=_family_color(family)),
            hovertemplate=(
                f"<b>{family}</b> — Pareto optimal<br>"
                f"εₓ: {{x:.1f}} pm·rad<br>"
                f"C: {{y:.0f}} m"
            ),
        ))

    # Highlight anomalies if provided
    if highlight_anomalies:
        anom_labels = set(a.get("label", "") for a in highlight_anomalies)
        anom_runs = [r for r in front if r.get("label") in anom_labels]
        if anom_runs:
            axs = [_x(r) for r in anom_runs]
            ays = [_y(r) for r in anom_runs]
            valid = [
                (x, y) for x, y in zip(axs, ays)
                if x is not None and y is not None
            ]
            if valid:
                avx, avy = zip(*valid)
                fig.add_trace(_go().Scatter(
                    x=avx, y=avy,
                    mode="markers",
                    name="Anomaly",
                    marker=dict(
                        symbol="circle-open",
                        size=16,
                        color="red",
                        line=dict(width=2.5),
                    ),
                    hovertemplate="<b>ANOMALY</b><br>εₓ: %{x:.1f} pm·rad<br>C: %{y:.0f} m",
                ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(family="serif", size=15)),
        xaxis=dict(**_AXIS_STYLE, title=x_label, type="log"),
        yaxis=dict(**_AXIS_STYLE, title=y_label),
    )
    return fig


def compute_pareto_front_plot(
    results: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_scale_pm: bool,
) -> list[dict[str, Any]]:
    """Compute the two-objective Pareto front for plotting.

    Returns results that are NOT dominated by any other result.
    Both lower-is-better (min direction).
    """
    def _xval(r):
        v = _to_float(r["result"].get(x_key)) if r.get("result") else None
        return v * 1e12 if (x_scale_pm and v is not None) else v

    def _yval(r):
        return _to_float(r["result"].get(y_key)) if r.get("result") else None

    pareto: list[dict[str, Any]] = []
    for r in results:
        xr = _xval(r)
        yr = _yval(r)
        if xr is None or yr is None:
            continue
        if not math.isfinite(xr) or not math.isfinite(yr):
            continue

        dominated = False
        for existing in pareto:
            xe = _xval(existing)
            ye = _yval(existing)
            if xe is None or ye is None:
                continue
            # existing dominates r if it's strictly better in at least one dimension
            if xe <= xr and ye <= yr and (xe < xr or ye < yr):
                dominated = True
                break
            # r dominates existing
            if xr <= xe and yr <= ye and (xr < xe or yr < ye):
                pareto = [p for p in pareto if p is not existing]

        if not dominated:
            pareto.append(r)

    return pareto


def plot_scaling_law(
    results: list[dict[str, Any]],
    *,
    independent_var: str = "n_cells",
    x_label: str = "Number of cells N_cells",
    title: str = "Emittance Scaling — ε ∝ N_cells<sup>−3</sup>",
):
    """Plot emittance vs. a design parameter on log-log axes with a theory line.

    Overlays the analytical scaling law ε ∝ n_cells⁻³ for comparison.
    """
    from physics_core import fit_scaling_exponent

    stable = [
        r for r in results
        if r.get("result") and r["result"].get("stable") is True
    ]

    if not stable:
        fig = _go().Figure()
        fig.add_annotation(text="No stable designs", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    families = sorted(set(r.get("family", "") for r in stable if r.get("family")))

    fig = _go().Figure()

    for family in families:
        family_runs = [r for r in stable if r.get("family") == family]
        xs: list[float] = []
        ys: list[float] = []
        for r in family_runs:
            x = _to_float(r.get(independent_var))
            y = _to_float(r["result"].get("emittance")) if r.get("result") else None
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0:
                xs.append(x)
                ys.append(y * 1e12)  # → pm·rad

        if not xs:
            continue

        fig.add_trace(_go().Scatter(
            x=xs, y=ys,
            mode="markers",
            name=f"{family} (data)",
            marker=dict(
                symbol=_family_marker(family),
                size=9,
                color=_family_color(family),
                opacity=0.7,
            ),
            hovertemplate=(
                f"<b>{family}</b><br>"
                f"N_cells: {{x}}<br>"
                f"εₓ: {{y:.1f}} pm·rad"
            ),
        ))

        # Fit scaling exponent and overplot the fitted line
        fit = fit_scaling_exponent(
            [r for r in family_runs if r.get("result")],
            independent_var=independent_var,
            dependent_var="emittance",
        )
        if fit.get("exponent_beta") is not None and fit.get("r_squared", 0) > 0.3:
            beta = fit["exponent_beta"]
            alpha = fit["intercept_alpha"]
            # Overplot the fitted line on the log-log space
            x_min, x_max = min(xs), max(xs)
            x_fit = [x_min * (x_max / x_min) ** (i / 200) for i in range(201)]
            # In log-space: log(y) = alpha + beta * log(x)
            y_fit_pm = [math.exp(alpha + beta * math.log(x)) for x in x_fit]
            fig.add_trace(_go().Scatter(
                x=x_fit, y=y_fit_pm,
                mode="lines",
                name=f"{family} (fit β={beta:.2f})",
                line=dict(
                    color=_family_color(family),
                    width=2,
                    dash="dash",
                ),
                showlegend=True,
            ))

    # Draw the theoretical θ³ reference line
    # Normalize to the data: ε ∝ n_cells^{-3}
    if stable:
        all_xs = []
        all_ys = []
        for r in stable:
            x = _to_float(r.get(independent_var))
            y = _to_float(r["result"].get("emittance")) if r.get("result") else None
            if x and y and math.isfinite(x) and math.isfinite(y):
                all_xs.append(x)
                all_ys.append(y * 1e12)

        if all_xs:
            # Find the median design to anchor the theory line
            idx = len(all_xs) // 2
            x0 = sorted(all_xs)[idx]
            y_anchors = sorted(all_ys)
            y0 = y_anchors[idx]
            # Theory: ε ∝ n_cells^{-3}
            # y_theory(x) = y0 * (x / x0)^{-3}
            x_range = [min(all_xs) * 0.8, max(all_xs) * 1.2]
            x_theory = [x_range[0] * (x_range[1] / x_range[0]) ** (i / 200) for i in range(201)]
            y_theory = [y0 * (x / x0) ** (-3.0) for x in x_theory]

            fig.add_trace(_go().Scatter(
                x=x_theory, y=y_theory,
                mode="lines",
                name="Theory: ε ∝ N⁻³",
                line=dict(color="#333", width=1.5, dash="dot"),
                showlegend=True,
            ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(family="serif", size=15)),
        xaxis=dict(**_AXIS_STYLE, title=x_label, type="log"),
        yaxis=dict(**_AXIS_STYLE, title="Natural emittance εₓ (pm·rad)", type="log"),
    )
    return fig


def plot_emittance_vs_energy(
    results: list[dict[str, Any]],
    *,
    title: str = "Emittance vs. Beam Energy — Lattice Family Comparison",
):
    """Plot emittance vs. beam energy, coloured by lattice family.

    Overlays the theoretical ε ∝ γ² (≈E²) scaling line.
    """
    stable = [
        r for r in results
        if r.get("result") and r["result"].get("stable") is True
    ]

    families = sorted(set(r.get("family", "") for r in stable if r.get("family")))

    fig = _go().Figure()

    for family in families:
        family_runs = [r for r in stable if r.get("family") == family]
        es: list[float] = []
        emits: list[float] = []
        for r in family_runs:
            e = r.get("energy_gev")
            emit = _to_float(r["result"].get("emittance")) if r.get("result") else None
            if e is not None and emit is not None and math.isfinite(e) and math.isfinite(emit):
                es.append(e)
                emits.append(emit * 1e12)

        if not es:
            continue

        fig.add_trace(_go().Scatter(
            x=es, y=emits,
            mode="markers",
            name=family,
            marker=dict(
                symbol=_family_marker(family),
                size=10,
                color=_family_color(family),
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                f"<b>{family}</b><br>"
                f"E: {{x:.1f}} GeV<br>"
                f"εₓ: {{y:.1f}} pm·rad"
            ),
        ))

    # Theory reference: ε ∝ γ² ∝ E²
    if stable:
        all_e = [r.get("energy_gev") for r in stable if r.get("energy_gev")]
        all_emit = [
            _to_float(r["result"].get("emittance")) * 1e12
            for r in stable
            if r.get("result") and _to_float(r["result"].get("emittance"))
        ]
        if all_e and all_emit:
            e0 = sorted(all_e)[len(all_e) // 2]
            y0_vals = sorted(all_emit)
            y0 = y0_vals[len(y0_vals) // 2]
            e_range = [min(all_e) * 0.9, max(all_e) * 1.1]
            e_theory = [e_range[0] + (e_range[1] - e_range[0]) * i / 200 for i in range(201)]
            emit_theory = [y0 * (e / e0) ** 2.0 for e in e_theory]

            fig.add_trace(_go().Scatter(
                x=e_theory, y=emit_theory,
                mode="lines",
                name="Theory: ε ∝ γ² (∝ E²)",
                line=dict(color="#333", width=1.5, dash="dot"),
                showlegend=True,
            ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(family="serif", size=15)),
        xaxis=dict(**_AXIS_STYLE, title="Beam energy (GeV)"),
        yaxis=dict(**_AXIS_STYLE, title="Natural emittance εₓ (pm·rad)", type="log"),
    )
    return fig


def plot_dipole_grading_pattern(
    results: list[dict[str, Any]],
    *,
    title: str = "Optimal Dipole Angle Grading vs. Energy (7BA)",
):
    """Visualize the optimal dipole bending-angle ratios as a function of energy.

    This is the key figure for PRL Candidate 1 — if non-monotonic patterns
    emerge at certain energies, that's the discovery.

    Expects results to contain "dipole_angles" in the result dict (list of
    floats, one per dipole, in radians).  If the agent-generated script
    doesn't explicitly report these, they can be reconstructed from the
    code text.

    For now: plots what it can; shows a note if dipole angle data is missing.
    """
    # Try to locate dipole angle data in results
    pattern_data: dict[float, list[dict[str, Any]]] = {}

    for r in results:
        if not r.get("result") or r["result"].get("stable") is not True:
            continue
        e = r.get("energy_gev")
        if e is None:
            continue

        angles = r.get("dipole_angles")  # explicit
        if angles is None and r.get("result"):
            angles = r["result"].get("dipole_angles")
        if angles is None and r.get("result"):
            # Try alternative key names the agent might use
            for key in ("theta_per_dipole", "bend_angles", "dipole_angles_rad"):
                angles = r["result"].get(key)
                if angles is not None:
                    break

        if angles is None:
            continue

        if not isinstance(angles, (list, tuple)) or len(angles) < 2:
            continue

        try:
            angles_float = [float(a) for a in angles]
        except (TypeError, ValueError):
            continue

        e_round = round(e, 1)
        if e_round not in pattern_data:
            pattern_data[e_round] = []
        pattern_data[e_round].append({
            "angles": angles_float,
            "label": r.get("label", ""),
        })

    if not pattern_data:
        fig = _go().Figure()
        fig.add_annotation(
            text=(
                "No dipole angle data found in results.<br>"
                "Ensure the agent-generated script reports dipole angles,<br>"
                "e.g. <i>dipole_angles_rad = [θ₁, θ₂, ...]</i> in the JSON block."
            ),
            showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(size=12, color="#666"),
        )
        fig.update_layout(**_LAYOUT_BASE, title=dict(
            text=title, font=dict(family="serif", size=15),
        ))
        return fig

    # Compute mean angle per dipole position for each energy
    energies = sorted(pattern_data.keys())
    max_dipoles = max(
        len(entry["angles"])
        for entries in pattern_data.values()
        for entry in entries
    )

    fig = _go().Figure()

    for pos in range(max_dipoles):
        xs: list[float] = []
        ys: list[float] = []
        errs: list[float] = []
        for e in energies:
            angles_at_pos = [
                entry["angles"][pos]
                for entry in pattern_data[e]
                if pos < len(entry["angles"])
            ]
            if not angles_at_pos:
                continue
            mean_angle = sum(angles_at_pos) / len(angles_at_pos)
            xs.append(e)
            ys.append(mean_angle * 1e3)  # rad → mrad
            if len(angles_at_pos) > 1:
                var = sum((a - mean_angle) ** 2 for a in angles_at_pos) / (len(angles_at_pos) - 1)
                errs.append(math.sqrt(var) * 1e3)
            else:
                errs.append(0.0)

        if not xs:
            continue

        fig.add_trace(_go().Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=f"Dipole {pos + 1}",
            marker=dict(size=8),
            line=dict(width=2),
            error_y=dict(
                type="data", array=errs, visible=True,
                thickness=1.5, width=0,
            ),
            hovertemplate=(
                f"<b>Dipole {pos + 1}</b><br>"
                f"E: {{x:.1f}} GeV<br>"
                f"θ: {{y:.3f}} mrad"
            ),
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(family="serif", size=15)),
        xaxis=dict(**_AXIS_STYLE, title="Beam energy (GeV)", dtick=1.0),
        yaxis=dict(**_AXIS_STYLE, title="Mean bending angle per dipole (mrad)"),
    )
    return fig


def plot_family_comparison(
    db_summary: list[dict[str, Any]],
    *,
    title: str = "Lattice Family Performance Comparison",
):
    """Bar chart comparing emittance across lattice families.

    ``db_summary`` should come from ``summarize_by_family()``.
    """
    families = [r["family"] for r in db_summary]
    emit_min = [r.get("emittance_min_pm") or 0 for r in db_summary]
    emit_mean = [r.get("emittance_mean_pm") or 0 for r in db_summary]
    emit_max = [r.get("emittance_max_pm") or 0 for r in db_summary]
    colors = [_family_color(f) for f in families]

    fig = _go().Figure()

    # Mean emittance bar with min-max range
    for i, fam in enumerate(families):
        fig.add_trace(_go().Bar(
            x=[fam],
            y=[emit_mean[i]],
            name=fam,
            marker_color=colors[i],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[emit_max[i] - emit_mean[i]],
                arrayminus=[emit_mean[i] - emit_min[i]],
                visible=True,
            ),
            hovertemplate=(
                f"<b>{fam}</b><br>"
                f"Mean: {{y:.1f}} pm·rad<br>"
                f"Range: {emit_min[i]:.1f} – {emit_max[i]:.1f} pm·rad"
            ),
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        showlegend=False,
        title=dict(text=title, font=dict(family="serif", size=15)),
        xaxis=dict(**_AXIS_STYLE, title="Lattice family"),
        yaxis=dict(**_AXIS_STYLE, title="Natural emittance εₓ (pm·rad)", type="log"),
    )
    return fig


# ── Convenience: save figure ─────────────────────────────────────────────────


def save_figure(fig: _go().Figure, path: Path | str, *, width: int = 900, height: int = 600) -> None:
    """Save a plotly figure to disk as PNG (requires kaleido).

    Install:  pip install kaleido

    Falls back to HTML if kaleido is unavailable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in (".png", ".svg", ".pdf"):
        try:
            fig.write_image(str(path), width=width, height=height)
            return
        except (ImportError, ValueError, OSError) as exc:
            print(f"kaleido export failed ({exc}); saving as HTML instead.")

    html_path = path.with_suffix(".html")
    fig.write_html(str(html_path))
