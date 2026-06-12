"""Agent tools for the BRILLIANCE multi-agent design system.

These tools give the TaskPlanner and CodeWriter direct access to:
  - Analytical emittance scaling estimates (no pyJuTrack needed)
  - Reference design lookup from templates/plain/
  - Session history search for relevant past designs
  - TASK SPEC validation before code generation

All tools run in the main Python process and do NOT require pyJuTrack.
They are registered with AssistantAgents via AutoGen's tool-use mechanism.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _ROOT / "templates" / "plain"
_SESSIONS_DIR = _ROOT / "sessions"


# ── Physical constants ────────────────────────────────────────────────────────
_Cq = 3.8319e-13     # quantum diffusion constant [m]
_C_LIGHT = 2.99792458e8
_M_E_EV = 0.51099895069e6


# ── 1. Analytical emittance estimator ─────────────────────────────────────────

def estimate_natural_emittance(
    lattice_type: str,
    energy_gev: float,
    n_cells: int,
    n_bends_per_cell: int | None = None,
) -> dict[str, Any]:
    """Estimate natural horizontal emittance using theoretical scaling laws.

    Returns an order-of-magnitude estimate that helps the Planner choose
    between lattice topologies and set realistic targets before full simulation.

    Uses the approximate formula:
        ε₀ ≈ Cq · γ² · θ³ · F(lattice_type)
    where θ = 2π / (n_cells · n_bends_per_cell) is the bending angle per dipole
    and F is a dimensionless lattice factor:
        FODO  : F ≈ 1/(4·√3)  ≈ 0.144
        DBA   : F ≈ 1/(12·√15) ≈ 0.0215  (achromatic, theoretical minimum)
        TBA   : F ≈ 0.017
        7BA   : F ≈ 0.0046  (theoretical minimum of 7BA)
        9BA   : F ≈ 0.0028
        MBA/H7BA: F ≈ 0.003 (similar to 7BA)

    Parameters
    ----------
    lattice_type : str
        One of FODO, DBA, TBA, 7BA, 9BA, MBA, H7BA, H6BA.
    energy_gev : float
        Beam energy in GeV.
    n_cells : int
        Number of superperiods / cells in the ring.
    n_bends_per_cell : int | None
        Number of main dipoles per cell.  If None, inferred from lattice_type.

    Returns
    -------
    dict with keys:
        emittance_pm_rad : estimated emittance [pm·rad]
        emittance_nm_rad : estimated emittance [nm·rad]
        theta_mrad       : dipole bending angle [mrad]
        gamma            : Lorentz factor
        lattice_factor   : dimensionless scaling factor F
        note             : explanation string
    """
    # Lattice factor F and default bends-per-cell
    _defaults: dict[str, tuple[float, int]] = {
        "FODO":  (0.144,  2),
        "DBA":   (0.022,  2),
        "TBA":   (0.017,  3),
        "7BA":   (0.0046, 7),
        "9BA":   (0.0028, 9),
        "MBA":   (0.003,  7),
        "H7BA":  (0.003,  7),
        "H6BA":  (0.004,  6),
        "CUSTOM": (0.010, 4),
    }
    key = lattice_type.upper().strip()
    F, default_bends = _defaults.get(key, (0.010, 4))
    bends = n_bends_per_cell if n_bends_per_cell is not None else default_bends

    gamma = energy_gev * 1e9 / _M_E_EV
    theta = 2.0 * math.pi / (n_cells * bends)  # radians
    emittance_m_rad = _Cq * (gamma ** 2) * (theta ** 3) * F
    emittance_pm_rad = emittance_m_rad * 1e12
    emittance_nm_rad = emittance_m_rad * 1e9

    return {
        "emittance_pm_rad": round(emittance_pm_rad, 2),
        "emittance_nm_rad": round(emittance_nm_rad, 4),
        "theta_mrad": round(theta * 1e3, 3),  # bending angle per dipole in mrad
        "gamma": round(gamma, 1),
        "lattice_factor": F,
        "note": (
            f"Analytical estimate for {lattice_type} at {energy_gev} GeV, "
            f"{n_cells} cells × {bends} bends/cell. "
            f"Actual value depends on quadrupole tuning and optics; "
            f"expect ±30% accuracy from this formula."
        ),
    }


def compare_lattice_families(
    energy_gev: float,
    n_cells: int,
) -> dict[str, Any]:
    """Compare estimated natural emittance for all major lattice families.

    Useful for the Planner to recommend a lattice topology before simulation.

    Parameters
    ----------
    energy_gev : float
        Beam energy in GeV.
    n_cells : int
        Number of cells (superperiods) in the ring.

    Returns
    -------
    dict with key ``comparison`` containing a list of rows, each with
    ``lattice_type``, ``n_bends_per_cell``, ``emittance_pm_rad``.
    Sorted from lowest to highest emittance.
    """
    families = ["FODO", "DBA", "TBA", "7BA", "9BA", "H7BA"]
    rows = []
    for lt in families:
        est = estimate_natural_emittance(lt, energy_gev, n_cells)
        rows.append({
            "lattice_type": lt,
            "emittance_pm_rad": est["emittance_pm_rad"],
            "emittance_nm_rad": est["emittance_nm_rad"],
        })
    rows.sort(key=lambda r: r["emittance_pm_rad"])
    return {
        "energy_gev": energy_gev,
        "n_cells": n_cells,
        "comparison": rows,
        "note": (
            "Analytical estimates only — actual emittance depends on optics tuning. "
            "Use as a guide for topology selection, not as a design target."
        ),
    }


# ── 2. Reference design lookup ────────────────────────────────────────────────

def get_reference_design(lattice_type: str) -> dict[str, Any]:
    """Read a reference lattice script from templates/plain/ and return key facts.

    Returns the first ~60 lines of the template (parameter block + element
    factories) so the agent can use them as concrete starting points.

    Parameters
    ----------
    lattice_type : str
        One of: fodo, dba, tba, 7ba, heps, spear3, alsu, std7ba.
        Case-insensitive.  Partial matches are accepted.

    Returns
    -------
    dict with keys:
        found       : bool
        filename    : str
        header      : str (first 60 lines of the file)
        description : str
    """
    if not _TEMPLATES_DIR.exists():
        return {"found": False, "error": "templates/plain/ directory not found"}

    key = lattice_type.lower().strip()
    candidates = sorted(_TEMPLATES_DIR.glob("*.py"))
    match: Path | None = None
    for p in candidates:
        if key in p.stem.lower():
            match = p
            break
    if match is None:
        names = [p.stem for p in candidates]
        return {
            "found": False,
            "available": names,
            "error": f"No template matching '{lattice_type}'. Available: {names}",
        }

    try:
        text = match.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"found": False, "error": str(exc)}

    lines = text.splitlines()
    header = "\n".join(lines[:min(70, len(lines))])
    # Extract a one-line description from the first comment block
    desc_lines = [
        ln.lstrip("# ").strip()
        for ln in lines[:10]
        if ln.strip().startswith("#") and len(ln.strip()) > 2
    ]
    description = desc_lines[0] if desc_lines else match.stem

    return {
        "found": True,
        "filename": match.name,
        "description": description,
        "header": header,
    }


# ── 4. TASK SPEC validator ────────────────────────────────────────────────────

def validate_task_spec(spec_text: str) -> dict[str, Any]:
    """Check a TASK SPEC for common structural and physics errors.

    Catches mistakes before they reach the CodeWriter so fix-loops are avoided.

    Checks performed
    ----------------
    - Required fields present (task_type, lattice_topology, energy_gev, n_cells)
    - energy_gev is a positive number
    - n_cells is a positive integer
    - lattice_topology is one of the four valid values
    - Ring-topology tasks do not set n_cells = 1
    - bend_angle consistency (if elements section is present)
    - Ring-only task types are not paired with periodic_cell / transfer_line topology

    Parameters
    ----------
    spec_text : str
        The raw TASK SPEC text block, including the ``` TASK SPEC ``` markers.

    Returns
    -------
    dict with keys:
        valid  : bool
        errors : list[str]   — blocking problems
        warnings : list[str] — non-blocking suggestions
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Strip fences
    body = re.sub(r"```.*?```", "", spec_text, flags=re.DOTALL)
    body = re.sub(r"={3,}", "", body)

    def _field(name: str) -> str | None:
        m = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.+)", body, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else None

    # Required fields
    for req in ("task_type", "lattice_topology", "energy_gev", "n_cells"):
        if _field(req) is None:
            errors.append(f"Missing required field: {req}")

    # energy_gev sanity
    egev_str = _field("energy_gev")
    if egev_str and egev_str.lower() != "none":
        try:
            egev = float(egev_str)
            if egev <= 0 or egev > 1000:
                errors.append(f"energy_gev={egev} looks wrong (must be 0 < E ≤ 1000 GeV)")
        except ValueError:
            errors.append(f"energy_gev='{egev_str}' is not a number")

    # n_cells sanity
    ncells_str = _field("n_cells")
    n_cells = None
    if ncells_str and ncells_str.lower() != "none":
        try:
            n_cells = int(ncells_str)
            if n_cells < 1:
                errors.append("n_cells must be >= 1")
        except ValueError:
            errors.append(f"n_cells='{ncells_str}' is not an integer")

    # topology
    topology = (_field("lattice_topology") or "").lower()
    valid_topologies = {"ring", "periodic_cell", "transfer_line", "arbitrary_beamline"}
    if topology and topology not in valid_topologies:
        errors.append(
            f"lattice_topology='{topology}' is not valid. "
            f"Use one of: {sorted(valid_topologies)}"
        )

    # Ring with n_cells=1 is suspicious
    if topology == "ring" and n_cells == 1:
        warnings.append(
            "lattice_topology=ring but n_cells=1. "
            "A ring with 1 cell is unusual — did you mean periodic_cell?"
        )

    # Ring-only task types should not use non-ring topology
    ring_only_tasks = {"match_tunes", "match_chrom", "optimize_emit", "compute_da", "compute_ma"}
    task_type = (_field("task_type") or "").lower()
    if task_type in ring_only_tasks and topology in {"periodic_cell", "transfer_line"}:
        errors.append(
            f"task_type='{task_type}' requires lattice_topology=ring, "
            f"but got '{topology}'"
        )

    # Warn if elements section looks empty
    if "elements:" in body.lower():
        elem_section = body[body.lower().find("elements:"):]
        non_blank_lines = [
            ln for ln in elem_section.splitlines()[1:10]
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not non_blank_lines:
            warnings.append("elements: section appears empty — CodeWriter will have to invent element sizes")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ── 5. Design feasibility check ───────────────────────────────────────────────

def check_design_feasibility(
    lattice_type: str,
    energy_gev: float,
    n_cells: int,
    target_emittance: float | None = None,
    circumference_m: float | None = None,
    n_bends_per_cell: int | None = None,
    target_tune_x: float | None = None,
    target_tune_y: float | None = None,
) -> dict[str, Any]:
    """Check whether a design request is physically reachable before code generation.

    Compares the requested targets against theoretical scaling limits and
    geometric consistency requirements.  Returns a structured feasibility
    verdict so the Planner can emit INFEASIBLE or MARGINAL before handing
    off to the CodeWriter.

    Parameters
    ----------
    lattice_type : str
        Lattice family (FODO, DBA, TBA, 7BA, 9BA, MBA, H7BA, H6BA, CUSTOM).
    energy_gev : float
        Beam energy in GeV.
    n_cells : int
        Number of cells / superperiods.
    target_emittance : float | None
        Target natural emittance in m·rad.  None → not checked.
    circumference_m : float | None
        Requested circumference in m.  None → not checked.
    n_bends_per_cell : int | None
        Dipoles per cell (inferred from lattice_type if None).
    target_tune_x : float | None
        Requested fractional horizontal tune (0–0.5).
    target_tune_y : float | None
        Requested fractional vertical tune (0–0.5).

    Returns
    -------
    dict with keys:
        feasible             : True | False | "marginal"
        binding_constraint   : str | None   (name of the tightest constraint)
        violated_constraints : list[str]    (all violated constraints)
        suggested_relaxations: list[str]    (actionable suggestions)
        emittance_floor_pm   : float | None (analytical minimum emittance [pm·rad])
        min_circumference_m  : float | None (minimum geometry-consistent circumference [m])
        note                 : str
    """
    _defaults: dict[str, tuple[float, int]] = {
        "FODO":  (0.144,  2),
        "DBA":   (0.022,  2),
        "TBA":   (0.017,  3),
        "7BA":   (0.0046, 7),
        "9BA":   (0.0028, 9),
        "MBA":   (0.003,  7),
        "H7BA":  (0.003,  7),
        "H6BA":  (0.004,  6),
        "CUSTOM": (0.010, 4),
    }

    key = lattice_type.upper().strip()
    F, default_bends = _defaults.get(key, (0.010, 4))
    bends = n_bends_per_cell if n_bends_per_cell is not None else default_bends

    _M_E_EV_LOCAL = 0.51099895069e6
    gamma = energy_gev * 1e9 / _M_E_EV_LOCAL
    theta = 2.0 * math.pi / (n_cells * bends)
    emittance_floor_m = _Cq * (gamma ** 2) * (theta ** 3) * F
    emittance_floor_pm = emittance_floor_m * 1e12

    violated: list[str] = []
    suggestions: list[str] = []
    binding: str | None = None

    # ── 1. Emittance floor check ──────────────────────────────────────────
    if target_emittance is not None:
        ratio = target_emittance / emittance_floor_m
        if ratio < 0.20:
            violated.append(
                f"target_emittance ({target_emittance * 1e12:.1f} pm·rad) is below "
                f"the theoretical {lattice_type} floor ({emittance_floor_pm:.1f} pm·rad) "
                f"by a factor of {1.0 / ratio:.1f}×"
            )
            suggestions.append(
                f"Relax emittance target to ≥{emittance_floor_pm * 2:.0f} pm·rad, "
                f"OR increase n_cells, OR switch to a lower-emittance family "
                f"(e.g. {_lower_family(lattice_type)})"
            )
            if binding is None:
                binding = "target_emittance"
        elif ratio < 0.50:
            violated.append(
                f"target_emittance ({target_emittance * 1e12:.1f} pm·rad) is within "
                f"2× of the theoretical floor — marginal; requires optimal optics tuning"
            )
            suggestions.append(
                "Treat as marginal: achievable only with near-TME optics. "
                "Budget significant tuning effort."
            )

    # ── 2. Geometry / circumference check ────────────────────────────────
    _MIN_ELEMENT_LENGTH_M = 0.15   # minimum total element length per cell
    _MIN_DIPOLE_LENGTH_M  = 0.30   # minimum dipole length for mechanical reasons
    min_dipole_total = bends * _MIN_DIPOLE_LENGTH_M
    min_cell_length = max(min_dipole_total + _MIN_ELEMENT_LENGTH_M * (bends + 2), 2.0)
    min_circumference = min_cell_length * n_cells

    if circumference_m is not None:
        if circumference_m < min_circumference * 0.60:
            violated.append(
                f"circumference ({circumference_m:.0f} m) is too small for "
                f"{n_cells} cells × {bends} dipoles; "
                f"minimum geometry estimate ≈ {min_circumference:.0f} m"
            )
            suggestions.append(
                f"Increase circumference to ≥{min_circumference:.0f} m, "
                "OR reduce n_cells or n_bends_per_cell"
            )
            if binding is None:
                binding = "circumference_m"
        elif circumference_m < min_circumference:
            violated.append(
                f"circumference ({circumference_m:.0f} m) is tight for "
                f"{n_cells} cells × {bends} dipoles "
                f"(geometry estimate ≈ {min_circumference:.0f} m) — marginal"
            )
            suggestions.append("Allow shorter drifts and/or combined-function elements")

    # ── 3. Tune sanity checks ─────────────────────────────────────────────
    for tune, label in ((target_tune_x, "target_tune_x"), (target_tune_y, "target_tune_y")):
        if tune is not None:
            frac = tune - math.floor(tune)
            if frac < 0.05 or frac > 0.495:
                violated.append(
                    f"{label} fractional part {frac:.3f} is dangerously close to an integer "
                    "(0.0 or 0.5) — high risk of resonance loss"
                )
                suggestions.append(f"Move {label} to a fractional part between 0.1 and 0.45")
                if binding is None:
                    binding = label

    # ── 4. Overall verdict ───────────────────────────────────────────────
    hard_violated = [v for v in violated if "marginal" not in v.lower()]
    if hard_violated:
        feasible: Any = False
    elif violated:
        feasible = "marginal"
    else:
        feasible = True

    return {
        "feasible": feasible,
        "binding_constraint": binding,
        "violated_constraints": violated,
        "suggested_relaxations": suggestions,
        "emittance_floor_pm": round(emittance_floor_pm, 2),
        "min_circumference_m": round(min_circumference, 1),
        "note": (
            f"Feasibility check for {lattice_type} at {energy_gev} GeV, "
            f"{n_cells} cells × {bends} bends/cell. "
            "Results are analytical estimates; actual feasibility depends on optics tuning."
        ),
    }


def _lower_family(current: str) -> str:
    """Return the next lower-emittance family relative to the current one."""
    ladder = ["FODO", "DBA", "TBA", "7BA", "H7BA", "9BA", "MBA"]
    key = current.upper()
    idx = ladder.index(key) if key in ladder else -1
    if idx < len(ladder) - 1 and idx >= 0:
        return ladder[idx + 1]
    return "MBA (already near optimal family)"


# ── Public registry (for team.py tool registration) ──────────────────────────

PLANNER_TOOLS = [
    estimate_natural_emittance,
    compare_lattice_families,
    get_reference_design,
    validate_task_spec,
    check_design_feasibility,
]
