"""MBA lattice design physics tools for the BRILLIANCE agent.

These tools encode the deterministic MBA construction methodology from
"Deterministic Construction of a Distributed-Sextupole MBA Lattice" (2024).
Each tool maps to a specific step in the paper's 10-step construction workflow.

All tools are pure Python — no pyJuTrack dependency — so they can be called by
the TaskPlanner and CodeWriter without sandboxed execution.

Physical constants and sign conventions follow:
  - Coordinate order: [x, px, y, py, z, δ]  (AT/pyJuTrack style)
  - Cq   = 3.8319e-13 m   (quantum diffusion constant)
  - Cγ   = 8.85e-5 m·GeV⁻³ (radiation constant)
  - me   = 0.51099895069 MeV
  - θ > 0 bends toward negative x (inward for outward-bending arc)
  - KQUAD k1 > 0 → horizontal focusing, vertical defocusing
"""

from __future__ import annotations

import math
from typing import Any

# ── Physical constants ────────────────────────────────────────────────────────
_Cq    = 3.8319e-13       # quantum diffusion constant [m]
_Cgamma = 8.85e-5          # radiation constant [m·GeV⁻³]
_me_eV = 0.51099895069e6   # electron rest mass [eV]
_CLIGHT = 2.99792458e8     # speed of light [m/s]

# ── Lattice factor F per family (dimensionless) ───────────────────────────────
# These are empirical/approximate values for the TME emittance formula
# ε₀ = Cq·γ²·θ³·F  where θ = 2π/(n_cells·n_bends) per-dipole bending angle.
#   FODO  : 0.144   (no dispersion suppression; large emittance)
#   DBA   : 0.022   (achromatic, 2-bend, theoretical ≈ 1/(12·√15))
#   TBA   : 0.017   (achromatic, 3-bend)
#   7BA   : 0.0046  (achromatic, 7-bend, near TME)
#   9BA   : 0.0028  (achromatic, 9-bend)
#   H7BA  : 0.003   (hybrid 7BA; reverse bends help, but not fully TME)
#   H6BA  : 0.004   (hybrid 6BA)
#   MBA   : 0.003   (generic MBA, similar to H7BA)

_LATTICE_FACTORS: dict[str, float] = {
    "FODO": 0.144,  "DBA": 0.022,  "TBA": 0.017,
    "7BA":  0.0046, "9BA": 0.0028, "H7BA": 0.003,
    "H6BA": 0.004,  "MBA": 0.003,
}

# Family → default number of main bends per cell
_N_BENDS: dict[str, int] = {
    "FODO": 2, "DBA": 2, "TBA": 3, "7BA": 7,
    "9BA": 9, "H7BA": 7, "H6BA": 6, "MBA": 7,
}

# Canonical unit cell topologies: (prefix, element_sequence)
# Each entry is a list of (type, role, suggested_length_m) tuples.
# "RB" = reverse bend (often implemented as transversely offset QF).
_UNIT_CELL_TOPOLOGIES: dict[int, list[tuple[str, str, float]]] = {
    # Minimal 2-bend (FODO-like): QF - B - QD - B
    2: [
        ("DRIFT", "entrance_drift", 0.50),
        ("KQUAD", "QF_hfoc", 0.50),
        ("DRIFT", "drift_to_bend", 0.50),
        ("SBEND", "main_bend", 1.00),
        ("DRIFT", "drift_mid", 0.50),
        ("KQUAD", "QD_hdefoc", 0.50),
        ("DRIFT", "drift_to_bend", 0.50),
        ("SBEND", "main_bend", 1.00),
        ("DRIFT", "exit_drift", 0.50),
    ],
    # 3-bend (TBA): QF - B - QD - B - QF - B
    3: [
        ("DRIFT", "entrance_drift", 0.50),
        ("KQUAD", "QF1_hfoc", 0.50),
        ("DRIFT", "drift_to_bend", 0.50),
        ("SBEND", "main_bend", 1.00),
        ("DRIFT", "drift_mid", 0.30),
        ("KQUAD", "QD_hdefoc", 0.50),
        ("DRIFT", "drift_mid", 0.30),
        ("SBEND", "main_bend", 1.00),
        ("DRIFT", "drift_mid", 0.30),
        ("KQUAD", "QF2_hfoc", 0.50),
        ("DRIFT", "drift_to_bend", 0.50),
        ("SBEND", "main_bend", 1.00),
        ("DRIFT", "exit_drift", 0.50),
    ],
    # 7-bend (7BA): canonical layout from the paper
    # SF - QF/RB - QD - SD - B1 - Q3 - SD1 - Q4 - SF1 - B2 - Q5 - SD2 - Q6 - SF2 - B3 - Q7 - BC - [mirror]
    7: [
        # Left half-cell
        ("DRIFT", "entrance_drift_DS", 2.50),       # DS (long straight-side drift)
        ("KQUAD", "Q1_hfoc_RB_candidate", 0.25),    # Q1 (QF, possible RB offset)
        ("DRIFT", "DM1", 0.30),
        ("KQUAD", "Q2_hdefoc", 0.20),               # Q2 (QD)
        ("DRIFT", "DM2", 0.25),
        ("SBEND", "B1_main_bend", 0.58),            # B1
        ("DRIFT", "D1", 0.25),
        ("KQUAD", "Q3_hfoc", 0.20),                 # Q3
        ("DRIFT", "D2", 0.12),
        ("KSEXT", "SD1_defoc_sext", 0.15),          # SD1
        ("DRIFT", "D3", 0.12),
        ("KQUAD", "Q4_hdefoc", 0.20),               # Q4
        ("DRIFT", "D4a", 0.15),
        ("KSEXT", "SF1_foc_sext", 0.15),            # SF1
        ("DRIFT", "D4b", 0.15),
        ("SBEND", "B2_main_bend", 0.78),            # B2
        ("DRIFT", "D1b", 0.25),
        ("KQUAD", "Q5_hfoc", 0.20),                 # Q5
        ("DRIFT", "D2b", 0.12),
        ("KSEXT", "SD2_defoc_sext", 0.15),          # SD2
        ("DRIFT", "D3b", 0.12),
        ("KQUAD", "Q6_hdefoc", 0.20),               # Q6
        ("DRIFT", "D4c", 0.15),
        ("KSEXT", "SF2_foc_sext", 0.15),            # SF2
        ("DRIFT", "D4d", 0.15),
        ("SBEND", "B3_main_bend", 0.97),            # B3
        ("DRIFT", "D5a", 0.20),
        ("KQUAD", "Q7_hfoc", 0.25),                 # Q7
        ("DRIFT", "D5b", 0.20),
        ("SBEND", "BC_centre_bend", 1.46),          # BC (centre bend — mirror axis)
        # Mirror axis at BC centre
    ],
    # 9-bend (9BA): extension of 7BA with two more dipoles
    9: [
        ("DRIFT", "entrance_drift_DS", 2.50),
        ("KQUAD", "Q1_hfoc", 0.25),
        ("DRIFT", "DM1", 0.25),
        ("KQUAD", "Q2_hdefoc", 0.20),
        ("DRIFT", "DM2", 0.25),
        ("SBEND", "B1_main_bend", 0.50),
        ("DRIFT", "D1", 0.20),
        ("KQUAD", "Q3_hfoc", 0.20),
        ("DRIFT", "D2", 0.10),
        ("KSEXT", "SD1", 0.15),
        ("DRIFT", "D3", 0.10),
        ("KQUAD", "Q4_hdefoc", 0.20),
        ("DRIFT", "D4a", 0.12),
        ("KSEXT", "SF1", 0.15),
        ("DRIFT", "D4b", 0.12),
        ("SBEND", "B2_main_bend", 0.65),
        ("DRIFT", "D5", 0.20),
        ("KQUAD", "Q5_hfoc", 0.20),
        ("DRIFT", "D6", 0.10),
        ("KSEXT", "SD2", 0.15),
        ("DRIFT", "D7", 0.10),
        ("KQUAD", "Q6_hdefoc", 0.20),
        ("DRIFT", "D8a", 0.12),
        ("KSEXT", "SF2", 0.15),
        ("DRIFT", "D8b", 0.12),
        ("SBEND", "B3_main_bend", 0.80),
        ("DRIFT", "D9", 0.20),
        ("KQUAD", "Q7_hfoc", 0.20),
        ("DRIFT", "D10a", 0.15),
        ("SBEND", "B4_main_bend", 0.90),
        ("DRIFT", "D10b", 0.15),
        ("KQUAD", "Q8_hdefoc", 0.20),
        ("DRIFT", "D11a", 0.15),
        ("SBEND", "BC_centre_bend", 1.20),
    ],
}

# ── HOA (Higher-Order Achromat) resonance table ───────────────────────────────
# Integer resonance strengths up to order 5 — used to flag dangerous working points.
# Format: (order, type) → score (0=no resonance, 1=weak, 2=strong, 3=systematic)
_STRUCTURAL_RESONANCES: list[tuple[int, int, int]] = [
    # (n_x, n_y, order)  — systematic resonances driven by periodicity
    (1, 0, 1),  # integer
    (0, 1, 1),
    (2, 0, 2),  # half-integer
    (0, 2, 2),
    (1, 1, 2),  # difference resonance
    (1, -1, 2), # sum resonance
    (3, 0, 3),  # third-order
    (0, 3, 3),
    (2, 1, 3),
    (1, 2, 3),
    (4, 0, 4),  # fourth-order
    (0, 4, 4),
    (3, 1, 4),
    (1, 3, 4),
    (2, 2, 4),
    (5, 0, 5),  # fifth-order
    (0, 5, 5),
    (3, 2, 5),
    (2, 3, 5),
    (4, 1, 5),
    (1, 4, 5),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1: compute_hoa_phase_advance
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hoa_phase_advance(
    n_cells: int,
    n_bends_per_cell: int,
    tune_x: float | None = None,
    tune_y: float | None = None,
) -> dict[str, Any]:
    """Compute valid HOA (Higher-Order Achromat) phase advances.

    For an N-bend achromat with M superperiods (n_cells), the cancellation of
    geometric aberrations requires that the total phase advance through the
    achromat satisfies:

        n * mu_x / (2π) = N_x      n * mu_y / (2π) = N_y

    where n = n_cells * n_bends_per_cell (total dipoles in the ring) and
    N_x, N_y are small integers.

    This tool:
      1. Enumerates all valid (N_x, N_y) integer pairs consistent with
         realistic phase-advance ranges per cell.
      2. Computes the corresponding ring working points (nux, nuy).
      3. Flags working points near dangerous structural resonances.
      4. Returns the recommended set with a rationale.

    Parameters
    ----------
    n_cells : int
        Number of superperiods.
    n_bends_per_cell : int
        Number of main bends per cell.
    tune_x : float | None
        Requested horizontal tune (if None, all HOA-compatible options returned).
    tune_y : float | None
        Requested vertical tune (if None, all HOA-compatible options returned).

    Returns
    -------
    dict with keys:
        n_total_dipoles : int              — total dipoles in ring = n_cells * n_bends
        valid_working_points : list[dict]  — each with nux, nuy, mu_x_cell_deg,
                                             mu_y_cell_deg, Nx, Ny, resonance_warnings
        best_working_points : list[dict]   — top 3 recommended working points
        requested_match : dict | None      — if tune_x/tune_y specified, the best match
        note : str
    """
    n_total = n_cells * n_bends_per_cell  # total dipoles = total unit-cell equivalents

    # Phase-advance ranges per dipole:
    #  mu_x per dipole typically 20°–90° for MBA
    #  mu_y per dipole typically 15°–60° for MBA (usually smaller)
    # So per cell: mu_x_cell in [20*n_bends, 90*n_bends] degrees
    #       mu_y_cell in [15*n_bends, 60*n_bends] degrees
    # Total ring phase = n_cells * mu_cell = N * 360°
    mu_x_min_cell = max(10.0, 15.0 * n_bends_per_cell)   # deg, lower bound
    mu_x_max_cell = 90.0 * n_bends_per_cell               # deg, upper bound
    mu_y_min_cell = max(5.0, 10.0 * n_bends_per_cell)
    mu_y_max_cell = 60.0 * n_bends_per_cell

    Nx_min = max(1, math.floor(n_cells * mu_x_min_cell / 360.0))
    Nx_max = math.ceil(n_cells * mu_x_max_cell / 360.0)
    Ny_min = max(1, math.floor(n_cells * mu_y_min_cell / 360.0))
    Ny_max = math.ceil(n_cells * mu_y_max_cell / 360.0)

    valid_points: list[dict] = []

    for Nx in range(Nx_min, Nx_max + 1):
        # mu_x per cell = Nx * 360 / n_cells  [deg]
        mu_x_cell_deg = Nx * 360.0 / n_cells
        mu_x_cell_rad = mu_x_cell_deg * math.pi / 180.0

        # Per-dipole (unit-cell-equivalent) phase advance:
        mu_x_dipole_deg = mu_x_cell_deg / n_bends_per_cell

        # Skip unphysically small or large per-dipole phase advances
        if mu_x_dipole_deg < 8.0 or mu_x_dipole_deg > 95.0:
            continue

        for Ny in range(Ny_min, Ny_max + 1):
            mu_y_cell_deg = Ny * 360.0 / n_cells
            mu_y_cell_rad = mu_y_cell_deg * math.pi / 180.0
            mu_y_dipole_deg = mu_y_cell_deg / n_bends_per_cell

            if mu_y_dipole_deg < 6.0 or mu_y_dipole_deg > 65.0:
                continue

            # HOA condition: ring tune = integer Nx, Ny.
            # Per-cell phase = Nx * 360 / n_cells.
            # Per-cell fractional tune = Nx / n_cells (fractional part).
            ring_nux = float(Nx)
            ring_nuy = float(Ny)
            cell_nux_frac = (Nx / n_cells) % 1.0
            cell_nuy_frac = (Ny / n_cells) % 1.0

            # Resonance check uses the DETUNED working point.
            # The HOA gives integer ring tune, but the working point is
            # detuned ~0.1-0.3 away from integer conditions (Paper Section 8).
            # Use different detune amounts for x and y to avoid sum/difference
            # resonance hits (νx ≈ νy drives coupling resonances).
            detune_x = 0.15  # typical detuning from integer condition
            detune_y = 0.22  # slightly different to avoid νx ≈ νy
            detuned_nux = ring_nux - detune_x
            detuned_nuy = ring_nuy - detune_y

            warnings: list[str] = []
            for r_nx, r_ny, order in _STRUCTURAL_RESONANCES:
                line = detuned_nux * r_nx + detuned_nuy * r_ny
                frac = abs(line - round(line))
                width = {1: 0.005, 2: 0.01, 3: 0.015, 4: 0.02, 5: 0.025}.get(order, 0.03)
                if frac < width:
                    resonance_name = f"order-{order} ({r_nx},{'+' if r_ny>=0 else ''}{r_ny})"
                    severity = "strong" if order <= 3 else "moderate"
                    if r_nx == 0 or r_ny == 0:
                        severity = "systematic" if order <= 2 else severity
                    warnings.append(
                        f"Near {severity} {resonance_name} resonance: "
                        f"|{r_nx}*νx + {r_ny}*νy| = {abs(line - round(line)):.4f} from integer "
                        f"(detuned νx≈{detuned_nux:.2f}, νy≈{detuned_nuy:.2f})"
                    )

            valid_points.append({
                "nux_ring": float(ring_nux),
                "nuy_ring": float(ring_nuy),
                "detuned_fractional_nux": f"{(detuned_nux % 1):.6f}",
                "detuned_fractional_nuy": f"{(detuned_nuy % 1):.6f}",
                "mu_x_cell_deg": round(mu_x_cell_deg, 2),
                "mu_y_cell_deg": round(mu_y_cell_deg, 2),
                "mu_x_dipole_deg": round(mu_x_dipole_deg, 2),
                "mu_y_dipole_deg": round(mu_y_dipole_deg, 2),
                "Nx": Nx,
                "Ny": Ny,
                "detune_note": "Working point should be detuned ~0.1-0.2 from integer to avoid structural resonances",
                "resonance_warnings": warnings,
                "resonance_count": len(warnings),
            })

    # Sort by resonance count (fewest warnings first), then by per-dipole phase
    # reasonableness (mu_x around 40-70 deg is typical for MBA)
    valid_points.sort(key=lambda p: (
        p["resonance_count"],
        abs(p["mu_x_dipole_deg"] - 55.0),  # prefer ~55°/dipole
    ))

    # Best: resonance-free and near 55°/dipole for x, 25-40°/dipole for y
    best = [p for p in valid_points if p["resonance_count"] == 0][:3]
    if len(best) < 3:
        best += valid_points[len(best):3]

    # Match against requested tunes
    requested_match = None
    if tune_x is not None or tune_y is not None:
        best_distance = float("inf")
        for p in valid_points:
            dist = 0.0
            if tune_x is not None:
                detuned_nux = float(p["detuned_fractional_nux"])
                dist += abs(detuned_nux - (tune_x % 1))
            if tune_y is not None:
                detuned_nuy = float(p["detuned_fractional_nuy"])
                dist += abs(detuned_nuy - (tune_y % 1))
            if dist < best_distance:
                best_distance = dist
                requested_match = dict(p)

    return {
        "n_total_dipoles": n_total,
        "total_valid_working_points": len(valid_points),
        "valid_working_points": valid_points[:30],  # top 30
        "best_working_points": best,
        "requested_match": requested_match,
        "note": (
            f"HOA phase analysis for {n_cells} superperiods × "
            f"{n_bends_per_cell} bends/cell ({n_total} total dipoles). "
            f"Phase advances must satisfy n·mu/(2π) = integer. "
            f"Nx∈[{Nx_min},{Nx_max}], Ny∈[{Ny_min},{Ny_max}]. "
            f"{len(valid_points)} valid working points found. "
            "Use the best_working_points entries with fewest resonance warnings."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2: build_unit_cell_topology
# ═══════════════════════════════════════════════════════════════════════════════

def build_unit_cell_topology(
    n_bends: int,
    *,
    include_reverse_bend: bool = False,
) -> dict[str, Any]:
    """Build the canonical generic symmetric unit-cell topology for an MBA.

    Start from the paper's minimal, realistic, symmetric topology:
        SF - QF/RB - QD - SD - MB - SD - QD - QF/RB - SF
    where:
      SF = focusing chromatic sextupole (near outer symmetry point)
      QF = horizontally focusing quadrupole
      RB = weak reverse bend, optionally generated by transversely offsetting QF
      QD = horizontally defocusing quadrupole
      SD = defocusing chromatic sextupole (near main bend)
      MB = main bend dipole

    This is the 'generic symmetric unit cell' from Step 3 of the paper.
    Sextupole placement follows correction-leverage principles (Step 6):
      SF near the outer symmetry point where β_x is favorable
      SD near the main bend where β_y leverage is favorable

    Parameters
    ----------
    n_bends : int
        Number of main bends in the unit cell (2, 3, 7, or 9).
    include_reverse_bend : bool
        If True, tag Q1/QF as a candidate for reverse-bend offset.

    Returns
    -------
    dict with keys:
        element_sequence     : list[dict]  — each with type, role, suggested_length_m
        n_quad_families      : int
        sextupole_placements : list[dict]  — where SF/SD pairs go and why
        n_sextupole_pairs    : int
        mirror_axis_offset   : int         — index where cell mirrors (center of centre bend)
        total_length_est_m   : float
        is_symmetric         : bool        — True (the topology is mirror-symmetric)
        note                 : str
    """
    if n_bends not in _UNIT_CELL_TOPOLOGIES:
        available = sorted(_UNIT_CELL_TOPOLOGIES.keys())
        return {
            "error": True,
            "message": f"No canonical topology for {n_bends}-bend cell. "
                       f"Available: {available}",
            "closest": _derive_closest_topology(n_bends, available),
        }

    raw = _UNIT_CELL_TOPOLOGIES[n_bends]

    # Convert to structured records
    sequence: list[dict[str, Any]] = []
    for i, (typ, role, length) in enumerate(raw):
        seq_entry: dict[str, Any] = {
            "index": i,
            "type": typ,
            "role": role,
            "suggested_length_m": length,
        }
        if include_reverse_bend and "RB" in role:
            seq_entry["reverse_bend_candidate"] = True
            seq_entry["rb_note"] = (
                "This quadrupole can be transversely offset to generate a weak "
                "reverse bend. RB increases Jx, lowers αc, and may reduce "
                "emittance — but must not violate the αc lower bound."
            )
        if "sext" in role.lower():
            seq_entry["sextupole_placement_rationale"] = (
                "SF near outer symmetry point: strong β_x leverage for "
                "horizontal chromatic correction"
                if role.startswith("SF") else
                "SD near main bend: strong β_y leverage for vertical "
                "chromatic correction"
            )
        sequence.append(seq_entry)

    # Count independent quad families (unique roles containing "Q" and "foc"/"defoc"
    # excluding center Q that appears twice in the mirror)
    quad_roles: set[str] = set()
    for entry in sequence:
        if entry["type"] == "KQUAD":
            quad_roles.add(entry["role"])
    n_quad_families = len(quad_roles)

    # Identify sextupole placements
    sext_placements = [s for s in sequence if s["type"] == "KSEXT"]
    n_sext_pairs = len([s for s in sext_placements if "SF" in s["role"]]) + \
                   len([s for s in sext_placements if "SD" in s["role"]])
    # Each pair has a mirror counterpart
    n_sext_pairs = n_sext_pairs // 2 if n_sext_pairs % 2 == 0 else n_sext_pairs

    # Mirror axis is at the centre bend (BC)
    mirror_idx = None
    for s in sequence:
        if "centre_bend" in s["role"].lower() or "BC" in s["role"]:
            mirror_idx = s["index"]
            break

    # Total length estimate (half-cell; full cell is ~2x this minus centre bend double-count)
    total_len_est = sum(s["suggested_length_m"] for s in sequence)
    # Full cell: half + mirror (exclude centre bend from mirror since it IS the axis)
    if mirror_idx is not None:
        centre_len = sequence[mirror_idx]["suggested_length_m"]
        full_cell_len = 2 * total_len_est - centre_len
    else:
        full_cell_len = 2 * total_len_est

    return {
        "element_sequence": sequence,
        "n_quad_families": n_quad_families,
        "sextupole_placements": sext_placements,
        "n_sextupole_pairs": n_sext_pairs,
        "mirror_axis_index": mirror_idx,
        "total_half_cell_length_est_m": round(total_len_est, 2),
        "total_full_cell_length_est_m": round(full_cell_len, 2),
        "is_symmetric": True,
        "reverse_bend_enabled": include_reverse_bend,
        "note": (
            f"Canonical {n_bends}-bend unit cell (mirror-symmetric about centre bend). "
            f"{n_quad_families} independent quadrupole families. "
            f"Sextupole pairs: {n_sext_pairs} distributed (SF near outer symmetry, "
            f"SD near main bends — maximum correction leverage). "
            "Total full-cell length estimate: {:.0f} m. ".format(full_cell_len) +
            ("Reverse-bend candidate identified at QF — scan RB angle for αc/emittance tradeoff. "
             if include_reverse_bend else
             "No reverse bend tagged. Consider RB scan for emittance reduction if αc≥target.")
        ),
    }


def _derive_closest_topology(n_bends: int, available: list[int]) -> dict:
    """Suggest the closest available topology and how to adapt it."""
    best = min(available, key=lambda b: abs(b - n_bends))
    return {
        "closest_available_bends": best,
        "adaptation": (
            f"Use the {best}-bend topology as base. "
            f"{'Add' if n_bends > best else 'Remove'} {abs(n_bends - best)} bends "
            "by inserting or removing QF-Bend-QD blocks. "
            "Maintain mirror symmetry and sextupole lever arm lengths."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3: decompose_superperiod
# ═══════════════════════════════════════════════════════════════════════════════

def decompose_superperiod(
    n_unit_cells: int | None = None,
    straight_length_m: float = 5.0,
    n_bends: int = 7,
    total_superperiod_length_m: float | None = None,
) -> dict[str, Any]:
    """Decompose one superperiod into its MBA substructures.

    One superperiod = StraightMatcher + DSC + N×UnitCell + mirror DSC + StraightMatcher

    This is Step 2 of the paper's methodology.  The DSC (Dispersion Suppression
    Cell) transitions from the periodic dispersive arc (unit cells) to the
    zero-dispersion straight section.

    Parameters
    ----------
    n_unit_cells : int | None
        Number of identical unit cells per superperiod. If None, estimated.
    straight_length_m : float
        Length of each straight section (matcher + drift).
    n_bends : int
        Number of bends per unit cell.
    total_superperiod_length_m : float | None
        If known, used to validate the decomposition.

    Returns
    -------
    dict with keys:
        components            : list[dict]  — ordered sub-cells with roles
        n_unit_cells          : int
        n_bends_per_period    : int
        estimated_length_m    : float
        dsc_structure_suggestion : str
        note                  : str
    """
    # Estimate unit cell length from topology
    topo = build_unit_cell_topology(n_bends)
    uc_length = topo.get("total_full_cell_length_est_m", 10.0)

    # DSC is roughly half a unit cell in length
    dsc_length = uc_length * 0.55  # DSC has fewer bends

    # Straight matcher: ~4 quadrupoles + drifts
    straight_matcher_length = straight_length_m

    if n_unit_cells is None:
        # Guess: a typical superperiod is 15-35 m for MBA.
        # 7BA cell ~20m → 2-4 UC per period.
        if total_superperiod_length_m is not None:
            available = total_superperiod_length_m - 2 * straight_matcher_length - 2 * dsc_length
            n_unit_cells = max(1, round(available / uc_length))
        else:
            n_unit_cells = max(1, min(4, n_bends // 2))  # heuristic

    # Estimated total
    total_est = (
        2 * straight_matcher_length +
        2 * dsc_length +
        n_unit_cells * uc_length
    )

    components = [
        {
            "order": 1,
            "name": "straight_matcher_entrance",
            "role": (
                "Matches target beta and alpha functions at the straight centre. "
                "Decouples insertion/injection optics from the arc. "
                "Typically 3-4 quadrupoles near the straight."
            ),
            "approximate_length_m": round(straight_matcher_length, 1),
            "required_outputs": {
                "betax_m": "target (e.g. 10-15)", "betay_m": "target (e.g. 3-5)",
                "alphax": 0.0, "alphay": 0.0,
                "Dx": 0.0, "Dpx": 0.0,
            },
        },
        {
            "order": 2,
            "name": "dispersion_suppression_cell_entrance",
            "role": (
                "Transitions from periodic dispersive arc (unit cell exit Twiss) "
                "to zero-dispersion straight section.  Keep as similar as possible "
                "to half the unit cell to preserve HOA phase structure."
            ),
            "approximate_length_m": round(dsc_length, 1),
            "matching_inputs": "unit_cell_exit_twiss (beta, alpha, Dx, Dpx)",
            "matching_targets": {"Dx": 0.0, "Dpx": 0.0, "alphay_approx": 0.0},
            "dsc_tuning_knobs": [
                "outer-bend angle or length",
                "offset QD/QF for additional bending",
                "QF and QD gradients",
                "drift lengths near the bend",
                "local sextupole location or strength",
                "optional octupole space",
            ],
        },
        {
            "order": 3,
            "name": f"unit_cells_×{n_unit_cells}",
            "role": (
                "Repeated arc building blocks. Control emittance, phase advance, "
                "chromaticity, and sextupole leverage."
            ),
            "approximate_length_m": round(n_unit_cells * uc_length, 1),
            "n_repeats": n_unit_cells,
            "unit_cell_length_est_m": round(uc_length, 1),
        },
        {
            "order": 4,
            "name": "dispersion_suppression_cell_exit",
            "role": "Mirror-symmetric counterpart of the entrance DSC.",
            "approximate_length_m": round(dsc_length, 1),
            "note": "Mirror-symmetric about the superperiod centre (between the two DSCs).",
        },
        {
            "order": 5,
            "name": "straight_matcher_exit",
            "role": "Mirror-symmetric counterpart of the entrance straight matcher.",
            "approximate_length_m": round(straight_matcher_length, 1),
        },
    ]

    return {
        "components": components,
        "n_unit_cells": n_unit_cells,
        "n_bends_per_period": n_unit_cells * n_bends,
        "estimated_length_m": round(total_est, 1),
        "dsc_structure_suggestion": (
            "Build the DSC as a constrained transfer line. Start from half the "
            "unit cell, replace the centre bend with a straight connection, and "
            "adjust QF/QD gradients and drift lengths to match eta_x=0, eta_x'=0 "
            "at the straight boundary while keeping alpha_y approximately zero."
        ),
        "note": (
            f"One superperiod decomposes as: straight_matcher → DSC → {n_unit_cells}×UC "
            f"→ mirror DSC → straight_matcher. "
            f"Estimated total length {total_est:.1f} m. "
            "Design objects in order: UnitCell → DSC → StraightMatcher → Full Ring. "
            "Do NOT optimize all objects simultaneously."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 4: compute_correction_leverage
# ═══════════════════════════════════════════════════════════════════════════════

def compute_correction_leverage(
    sextupole_position_name: str,
    beta_x_m: float,
    beta_y_m: float,
    Dx_m: float,
    energy_gev: float = 6.0,
    k2_hint: float = 10.0,
) -> dict[str, Any]:
    """Compute chromatic correction leverage at a sextupole position.

    Paper Step 6: "Place sextupoles using correction leverage, not chromaticity alone."

    The chromaticity correction from a sextupole depends on:
      ξ_x ∝ -k2 · β_x · Dx · L_sext / (4π · Bρ)
      ξ_y ∝ +k2 · β_y · Dx · L_sext / (4π · Bρ)   (opposite sign)

    where Bρ = p/e = E/(e·c) is the magnetic rigidity.

    Effective correction requires:
      - Large Dx (dispersion) at the sextupole location
      - Large β_x for horizontal, large β_y for vertical correction
      - Adequate separation between β_x and β_y if independent correction desired

    A sextupole at Dx ≈ 0 provides essentially NO chromatic correction,
    regardless of chromaticity.

    Parameters
    ----------
    sextupole_position_name : str
        Descriptive name (e.g. "SF1_near_outer_symmetry", "SD1_near_B2").
    beta_x_m : float
        Horizontal beta function at the sextupole [m].
    beta_y_m : float
        Vertical beta function at the sextupole [m].
    Dx_m : float
        Horizontal dispersion at the sextupole [m].
    energy_gev : float
        Beam energy [GeV].
    k2_hint : float
        Trial sextupole strength [m⁻³] for leverage estimation.

    Returns
    -------
    dict with keys:
        leverage_x          : float   — dξx/d(k2) [1/(m⁻³)]
        leverage_y          : float   — dξy/d(k2) [1/(m⁻³)]
        chromatic_ratio     : float   — |ξy_shift / ξx_shift| for same k2
        is_effective_x      : bool    — provides meaningful X correction
        is_effective_y      : bool    — provides meaningful Y correction
        recommended_k2_range : list[float]
        beta_separation_ratio : float — β_y/β_x (useful for independent correction)
        note                : str
    """
    # Magnetic rigidity Bρ = p/e = E/(ec)
    # Bρ [T·m] = E [eV] / c = energy_eV / 299792458
    E_eV = energy_gev * 1e9
    B_rho = E_eV / _CLIGHT  # [T·m]

    # Chromaticity shifts per unit k2 (per sextupole of 0.15 m length):
    L_sext = 0.15  # m (typical sextupole length)
    # dξx = -k2 · β_x · Dx · L / (4π · Bρ)
    # So leverage_x = dξx/dk2 = -β_x · Dx · L / (4π · Bρ)
    dxi_x_per_k2 = -beta_x_m * Dx_m * L_sext / (4.0 * math.pi * B_rho)
    dxi_y_per_k2 = +beta_y_m * Dx_m * L_sext / (4.0 * math.pi * B_rho)

    # Effectiveness thresholds
    # A single 0.15m sextupole at typical 6 GeV, 7BA optics gives:
    #   |dξ/dk2| ≈ β·Dx·L/(4π·Bρ) ≈ 15·0.3·0.15/(4π·20) ≈ 0.0027 per k2=1
    # With 4-6 sextupole pairs, total |dξ| per |k2|=1 ≈ 0.01-0.02.
    # A sextupole is "effective" if |dξ/dk2| > 0.0005 (per unit k2).
    min_leverage = 0.0005   # |dξ| per |k2|=1 for single sextupole

    is_eff_x = abs(dxi_x_per_k2) > min_leverage
    is_eff_y = abs(dxi_y_per_k2) > min_leverage

    # Recommended k2 range: total chromaticity correction split across all
    # sextupole families. For a ring with total |ξ| ~ 200, each of ~4 SF/SD
    # pairs contributes ~50 units. But leverage is per-unit-k2, so:
    #   k2_needed = 50 / |leverage| ≈ 50/0.003 ≈ 16667 — way too high.
    # In reality, leverage is small per sextupole but the MULTIPLE pairs
    # together contribute. The k2 range should be physically sensible:
    # typical MBA sextupole |k2| = 50-500 m⁻³.
    # Use the per-pair leverage to assess feasibility, but cap the range.
    per_pair_xi_contribution = 50.0  # typical contribution per pair
    k2_central = per_pair_xi_contribution / max(abs(dxi_x_per_k2), abs(dxi_y_per_k2), 0.0001)
    # Clamp: if k2_central > 500, this position would need very strong sextupoles
    lo = max(10.0, k2_central * 0.3)
    hi = min(800.0, k2_central * 3.0)
    if lo > hi:
        lo, hi = hi, lo
    if hi > 800:
        hi = 800.0
        lo = min(lo, 500.0)

    return {
        "position": sextupole_position_name,
        "energy_gev": energy_gev,
        "B_rho_Tm": round(B_rho, 2),
        "beta_x_m": beta_x_m,
        "beta_y_m": beta_y_m,
        "Dx_m": Dx_m,
        "leverage_x": round(dxi_x_per_k2, 6),
        "leverage_y": round(dxi_y_per_k2, 6),
        "chromatic_ratio": round(abs(dxi_y_per_k2 / dxi_x_per_k2) if dxi_x_per_k2 != 0 else float('inf'), 2),
        "is_effective_x": is_eff_x,
        "is_effective_y": is_eff_y,
        "recommended_k2_range": [round(lo, 1), round(hi, 1)],
        "beta_separation_ratio": round(beta_y_m / beta_x_m if beta_x_m > 0 else float('inf'), 3),
        "note": (
            f"At {sextupole_position_name}: "
            f"dξx/dk2 = {dxi_x_per_k2:.4f}, dξy/dk2 = {dxi_y_per_k2:.4f}. "
            f"The sextupole is {'effective' if is_eff_x else 'INEFFECTIVE'} for X, "
            f"{'effective' if is_eff_y else 'INEFFECTIVE'} for Y chromatic correction. "
            f"{'Good β-separation — independent X/Y correction possible. ' if abs(beta_y_m/beta_x_m - 1.0) > 0.3 else 'Poor β-separation — X/Y correction coupled. ' if beta_x_m > 0 else ''}"
            f"Recommended |k2| range: [{lo:.0f}, {hi:.0f}] m⁻³. "
            f"{'WARNING: Dx ≈ 0 — sextupole provides NO chromatic correction!' if abs(Dx_m) < 0.001 else ''}"
            if beta_x_m > 0 else
            f"Sextupole at zero/horizon β — no chromatic leverage."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 5: check_dsc_matching_conditions
# ═══════════════════════════════════════════════════════════════════════════════

def check_dsc_matching_conditions(
    beta_x_entry: float, alpha_x_entry: float,
    beta_y_entry: float, alpha_y_entry: float,
    Dx_entry: float, Dpx_entry: float,
    beta_x_target: float = 10.0, alpha_x_target: float = 0.0,
    beta_y_target: float = 3.0, alpha_y_target: float = 0.0,
    Dx_target: float = 0.0, Dpx_target: float = 0.0,
    n_dsc_quad_families: int | None = None,
    n_dsc_dipoles: int = 2,
) -> dict[str, Any]:
    """Verify that DSC matching is satisfiable given available degrees of freedom.

    Paper Step 7: "Treat the DSC as a constrained transfer line, not as an
    independently optimized mini-cell."

    The DSC must match:
      - UC exit Twiss → straight-section entrance Twiss
      - 6 constraints: (β_x, α_x, β_y, α_y, D_x, D'_x) at the straight boundary

    Available DOF:
      - Each quadrupole family gives 1 DOF (gradient)
      - Each dipole bend gives 1 DOF (angle, or length at fixed total angle)
      - Drift lengths give additional DOF
      - Total: n_quads + n_dipoles + n_drifts

    The matching is overconstrained if n_constraints > n_DOF.

    Parameters
    ----------
    beta_x_entry, alpha_x_entry, beta_y_entry, alpha_y_entry : float
        Unit cell exit Twiss parameters.
    Dx_entry, Dpx_entry : float
        Unit cell exit dispersion and dispersion derivative [m] and [rad].
    beta_x_target, alpha_x_target, beta_y_target, alpha_y_target : float
        Target Twiss at the straight section entrance.
    Dx_target, Dpx_target : float
        Target dispersion (usually 0, 0 for straight section).
    n_dsc_quad_families : int | None
        Number of quadrupole families available in the DSC. If None, estimated.
    n_dsc_dipoles : int
        Number of dipoles available in the DSC for bend-angle tuning.

    Returns
    -------
    dict with keys:
        n_constraints      : int
        n_dof_estimated    : int
        is_overconstrained : bool
        is_underconstrained : bool
        feasibility        : str  — "feasible" | "overconstrained" | "underconstrained"
        n_quads_recommended : int
        suggested_dsc_quad_count : int
        matching_difficulty : str  — "easy" | "moderate" | "hard"
        note               : str
    """
    # Constraints to match:
    constraints = []
    constraints.append(("beta_x", beta_x_entry, beta_x_target))
    constraints.append(("alpha_x", alpha_x_entry, alpha_x_target))
    constraints.append(("beta_y", beta_y_entry, beta_y_target))
    constraints.append(("alpha_y", alpha_y_entry, alpha_y_target))
    constraints.append(("Dx", Dx_entry, Dx_target))
    constraints.append(("Dpx", Dpx_entry, Dpx_target))

    # Count significant constraints (where entry ≠ target by > 10%)
    significant = []
    for name, val, tgt in constraints:
        if tgt != 0 and abs(val - tgt) / max(abs(tgt), 1e-6) > 0.05:
            significant.append(name)
        elif tgt == 0 and abs(val) > 1e-6:
            significant.append(f"{name}→0")

    n_constraints = len(significant)

    # DOF estimation
    if n_dsc_quad_families is None:
        # Typical DSC has 2-4 quadrupole families
        n_dsc_quad_families = max(2, min(4, len(significant)))

    n_dof = n_dsc_quad_families + n_dsc_dipoles

    # In practice, drift lengths provide extra DOF
    # But matching is truly overconstrained if constraints > total DOF
    is_over = n_constraints > n_dof + 1  # +1 for drift-length tuning
    is_under = n_constraints <= n_dof

    # Dispersion matching (η=0, η'=0) is the hard part — usually needs 2+ bends
    dispersion_constraints = [c for c in significant if "D" in c.upper() or "η" in c]
    n_dispersion = len(dispersion_constraints)

    feasibility = "feasible"
    if is_over:
        feasibility = "overconstrained"
    elif n_dof > n_constraints + 2:
        feasibility = "underconstrained"

    # Difficulty assessment
    if n_dispersion >= 2 and n_dsc_dipoles < 2:
        difficulty = "hard"
        feasibility = "overconstrained"
    elif n_dispersion >= 2 and n_constraints <= n_dof:
        difficulty = "moderate"
    elif n_constraints <= 3:
        difficulty = "easy"
    else:
        difficulty = "moderate"

    n_quads_needed = max(2, n_constraints - n_dsc_dipoles)

    return {
        "n_constraints": n_constraints,
        "significant_constraints": significant,
        "n_dof_estimated": n_dof,
        "n_dispersion_constraints": n_dispersion,
        "is_overconstrained": is_over,
        "is_underconstrained": is_under,
        "feasibility": feasibility,
        "matching_difficulty": difficulty,
        "n_quads_recommended": n_quads_needed,
        "suggested_dsc_quad_count": max(2, n_quads_needed),
        "note": (
            f"DSC matching: {n_constraints} significant constraints vs "
            f"~{n_dof} DOF ({n_dsc_quad_families} quad families + {n_dsc_dipoles} dipoles). "
            f"Status: {feasibility.upper()}. Difficulty: {difficulty}. "
            + (
                "Add quadrupole families or bend-angle DOF to satisfy constraints."
                if is_over else
                "May need additional constraints for a unique solution."
                if is_under else
                "Matching appears satisfiable with provided DOF."
            )
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 6: validate_mba_construction_step
# ═══════════════════════════════════════════════════════════════════════════════

def validate_mba_construction_step(
    step_name: str,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Validate that a construction-step output satisfies physics criteria.

    This is the gate-keeper that prevents physically invalid designs from
    propagating to the next construction stage.  Each step has specific
    acceptance criteria from the paper's validation checklist.

    Parameters
    ----------
    step_name : str
        One of: "design_contract", "decompose_superperiod", "unit_cell_topology",
        "hoa_phase_selection", "unit_cell_optimization", "sextupole_placement",
        "dsc_construction", "straight_matching", "ring_assembly_linear",
        "nonlinear_check".
    results : dict
        The outputs produced by the current stage.  Keys vary by stage.

    Returns
    -------
    dict with keys:
        passed       : bool
        step         : str
        checks       : list[dict]  — each check: name, passed, detail
        failures     : list[str]   — human-readable failure reasons
        warnings     : list[str]   — non-blocking concerns
        can_proceed  : bool
        note         : str
    """
    checks: list[dict] = []
    failures: list[str] = []
    warnings: list[str] = []

    def _check(name: str, condition: bool, detail: str):
        checks.append({"name": name, "passed": condition, "detail": detail})
        if not condition:
            failures.append(f"[{name}] {detail}")

    # ── design_contract ───────────────────────────────────────────────────
    if step_name == "design_contract":
        _check("has_energy", "energy_gev" in results, "Beam energy must be specified")
        _check("has_n_cells", "n_cells" in results, "Number of superperiods must be specified")
        _check("has_n_bends", "n_bends" in results, "Number of bends per cell must be specified")
        _check("has_ranked_goals",
               "ranked_goals" in results or "goals" in results,
               "Ranked design goals must be listed")
        energy = results.get("energy_gev", 0)
        if isinstance(energy, (int, float)) and energy > 0:
            _check("energy_positive", energy > 0, f"Energy {energy} GeV is positive")
            _check("energy_range", 0.5 <= energy <= 20,
                   f"Energy {energy} GeV in typical storage-ring range (0.5-20 GeV)")
        n_cells = results.get("n_cells", 0)
        if isinstance(n_cells, int):
            _check("n_cells_range", 4 <= n_cells <= 100,
                   f"n_cells={n_cells} in typical range (4-100)")
            _check("n_cells_even_for_mba", n_cells % 2 == 0,
                   "For symmetric MBA, n_cells should be even")
        n_bends = results.get("n_bends", results.get("n_bends_per_cell", 0))
        if isinstance(n_bends, int):
            _check("n_bends_range", 2 <= n_bends <= 12,
                   f"n_bends={n_bends} in typical range (2-12)")

    # ── decompose_superperiod ─────────────────────────────────────────────
    elif step_name == "decompose_superperiod":
        _check("has_components",
               "components" in results,
               "Superperiod decomposition must list components")
        if "components" in results:
            comps = results["components"]
            names = [c.get("name", "") for c in comps]
            _check("has_straight", any("straight" in n.lower() for n in names),
                   "Must include straight-matcher sections")
            _check("has_dsc", any("dispersion_suppression" in n.lower() or "dsc" in n.lower() for n in names),
                   "Must include dispersion-suppression cells")
            _check("has_unit_cells", any("unit_cell" in n.lower() or "unit" in n.lower() for n in names),
                   "Must include unit cells")
            _check("mirror_symmetric",
                   any("mirror" in str(c).lower() for c in comps) or len(comps) >= 5,
                   "Structure should be mirror-symmetric about superperiod centre")

    # ── unit_cell_topology ────────────────────────────────────────────────
    elif step_name == "unit_cell_topology":
        n_bends_raw = results.get("n_bends", results.get("n_bends_per_cell", 0))
        n_quads = results.get("n_quad_families", results.get("n_quadrupole_families", 0))
        _check("has_n_quads", isinstance(n_quads, int) and n_quads > 0,
               f"Must have at least 1 quad family, got {n_quads}")
        _check("quads_match_bends",
               isinstance(n_quads, int) and isinstance(n_bends_raw, int) and n_quads >= n_bends_raw - 1,
               f"Number of quad families ({n_quads}) should be ~number of bends ({n_bends_raw})")
        _check("has_sextupoles",
               results.get("n_sextupole_pairs", 0) >= 2,
               "Must have at least 2 sextupole pairs for chromatic correction")
        _check("is_symmetric",
               results.get("is_symmetric", False),
               "Unit cell must be mirror-symmetric about its centre")

    # ── hoa_phase_selection ──────────────────────────────────────────────
    elif step_name == "hoa_phase_selection":
        detuned_nux_str = results.get("detuned_fractional_nux", str(results.get("nux", "0.0")))
        detuned_nuy_str = results.get("detuned_fractional_nuy", str(results.get("nuy", "0.0")))
        try:
            nux_float = float(str(detuned_nux_str).split(",")[0].strip("(){}[] \""))
        except (ValueError, IndexError):
            nux_float = float(str(detuned_nux_str).strip("(){}[] \"") or 0)
        try:
            nuy_float = float(str(detuned_nuy_str).split(",")[0].strip("(){}[] \""))
        except (ValueError, IndexError):
            nuy_float = float(str(detuned_nuy_str).strip("(){}[] \"") or 0)

        _check("hoa_integers_provided",
               "Nx" in results and "Ny" in results,
               "Nx, Ny integers must be reported")
        _check("nux_not_integer",
               abs(nux_float) > 0.01 and abs(nux_float - 0.5) > 0.01,
               f"Fractional tune nux={nux_float:.4f} should avoid integers and half-integers")
        _check("nuy_not_integer",
               abs(nuy_float) > 0.01 and abs(nuy_float - 0.5) > 0.01,
               f"Fractional tune nuy={nuy_float:.4f} should avoid integers and half-integers")
        res_count = results.get("resonance_count", results.get("n_resonance_warnings", 0))
        _check("resonance_free",
               res_count == 0 or res_count is None,
               f"Should be resonance-free, got {res_count} resonance warnings")

    # ── unit_cell_optimization ───────────────────────────────────────────
    elif step_name == "unit_cell_optimization":
        stable = results.get("stable", results.get("is_stable", False))
        _check("cell_stable", bool(stable), "Unit cell must have |trace| < 2 in both planes")

        tx = results.get("trace_x", None)
        ty = results.get("trace_y", None)
        if tx is not None:
            _check("trace_x_bounds", abs(float(str(tx))) < 2, f"|trace_x|={tx} must be < 2")
        if ty is not None:
            _check("trace_y_bounds", abs(float(str(ty))) < 2, f"|trace_y|={ty} must be < 2")

        D_exit = results.get("D_exit", results.get("dx_exit", None))
        Dp_exit = results.get("Dp_exit", results.get("dpx_exit", None))
        if D_exit is not None:
            _check("cell_achromatic",
                   abs(float(str(D_exit))) < 0.01,
                   f"D_exit={D_exit} m should be near zero (achromatic cell)")
        if Dp_exit is not None:
            _check("cell_dispersion_derivative",
                   abs(float(str(Dp_exit))) < 0.01,
                   f"Dp_exit={Dp_exit} should be near zero")

        mu_x = results.get("mu_x_cell_deg", results.get("phase_advance_x_deg", None))
        mu_y = results.get("mu_y_cell_deg", results.get("phase_advance_y_deg", None))
        target_mu_x = results.get("target_mu_x_deg", None)
        target_mu_y = results.get("target_mu_y_deg", None)
        if mu_x is not None and target_mu_x is not None:
            delta_x = abs(float(str(mu_x)) - float(str(target_mu_x)))
            _check("phase_advance_x_on_target",
                   delta_x < 5.0,
                   f"mu_x={mu_x}° off target {target_mu_x}° by {delta_x:.1f}°")
        if mu_y is not None and target_mu_y is not None:
            delta_y = abs(float(str(mu_y)) - float(str(target_mu_y)))
            _check("phase_advance_y_on_target",
                   delta_y < 5.0,
                   f"mu_y={mu_y}° off target {target_mu_y}° by {delta_y:.1f}°")

    # ── sextupole_placement ──────────────────────────────────────────────
    elif step_name == "sextupole_placement":
        _check("has_leverage_report",
               "correction_leverage_report" in results or any("leverage" in str(k).lower() for k in results),
               "Correction leverage must be computed for each sextupole position")
        k2_vals = results.get("k2_initial_values", results.get("k2_values", []))
        if isinstance(k2_vals, list) and k2_vals:
            all_ok = all(abs(v) < 1000 for v in k2_vals if isinstance(v, (int, float)))
            _check("k2_not_extreme",
                   all_ok,
                   "All |k2| should be < 1000 m⁻³ to avoid extreme resonance driving")
            sf_at_dx = results.get("sf_at_high_dispersion", True)
            sd_at_dx = results.get("sd_at_high_dispersion", True)
            if not sf_at_dx:
                warnings.append("SF sextupole not at high-Dx location — may be ineffective")
            if not sd_at_dx:
                warnings.append("SD sextupole not at high-Dx location — may be ineffective")

    # ── dsc_construction ─────────────────────────────────────────────────
    elif step_name == "dsc_construction":
        _check("dsc_matched",
               results.get("dsc_matched", results.get("matched", False)),
               "DSC must match UC exit optics to straight-section entrance")
        Dx_exit = results.get("Dx_exit", results.get("Dx_straight", None))
        Dpx_exit = results.get("Dpx_exit", results.get("Dpx_straight", None))
        if Dx_exit is not None:
            _check("dsc_dispersion_zero",
                   abs(float(str(Dx_exit))) < 0.001,
                   f"Dx at straight={Dx_exit} m must be near zero")
        if Dpx_exit is not None:
            _check("dsc_dispersion_derivative_zero",
                   abs(float(str(Dpx_exit))) < 0.001,
                   f"Dp_x at straight={Dpx_exit} must be near zero")
        residuals = results.get("dsc_matching_residuals", results.get("residuals", {}))
        if residuals:
            max_res = max(abs(float(str(v))) for v in residuals.values()
                         if v is not None)
            _check("residuals_small",
                   max_res < 0.05,
                   f"Max matching residual {max_res:.4f} should be < 5%")

    # ── straight_matching ────────────────────────────────────────────────
    elif step_name == "straight_matching":
        _check("superperiod_matched",
               results.get("superperiod_matched", results.get("matched", False)),
               "Superperiod must be matched end-to-end")
        betax = results.get("straight_betax", results.get("betax_centre", None))
        betay = results.get("straight_betay", results.get("betay_centre", None))
        if betax is not None:
            _check("beta_x_positive",
                   float(str(betax)) > 1,
                   f"β_x={betax} m at straight centre must be > 1 m")
        if betay is not None:
            _check("beta_y_positive",
                   float(str(betay)) > 1,
                   f"β_y={betay} m at straight centre must be > 1 m")

    # ── ring_assembly_linear ─────────────────────────────────────────────
    elif step_name == "ring_assembly_linear":
        _check("ring_linear_ok",
               results.get("ring_linear_ok", results.get("stable", False)),
               "Ring linear optics must be stable")
        robinson = results.get("robinson_sum", results.get("Jx_Jy_Je_sum", None))
        if robinson is not None:
            _check("robinson_sum",
                   abs(float(str(robinson)) - 4.0) < 0.05,
                   f"Robinson sum Jx+Jy+Je = {robinson} should be ≈ 4.0")
        emittance = results.get("emittance", results.get("emittance_pm", None))
        target_emit = results.get("target_emittance_pm", None)
        if emittance is not None and target_emit is not None:
            ratio = float(str(emittance)) / float(str(target_emit))
            _check("emittance_in_budget",
                   ratio < 5.0,
                   f"Emittance {emittance} pm·rad = {ratio:.1f}× target (should be < 5×)")

    # ── nonlinear_check ──────────────────────────────────────────────────
    elif step_name == "nonlinear_check":
        _check("nonlinear_ok",
               results.get("nonlinear_ok", results.get("tolerable", False)),
               "Nonlinear optics must be acceptable")
        da_est = results.get("da_estimate", results.get("dynamic_aperture_mm", None))
        if da_est is not None:
            _check("da_reasonable",
                   float(str(da_est)) > 1.0,
                   f"Dynamic aperture estimate {da_est} mm — should be > 1 mm")

    else:
        warnings.append(f"Unknown step '{step_name}' — performing only basic checks")
        _check("has_results", bool(results), "Results dict must not be empty")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "step": step_name,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "can_proceed": passed,
        "note": (
            f"Step '{step_name}': {'PASSED' if passed else 'FAILED'}. "
            f"{len(checks)} checks: {sum(1 for c in checks if c['passed'])}/{len(checks)} passed. "
            + (f" Failures: {'; '.join(failures)}" if failures else " All checks passed.")
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 7: estimate_mba_emittance
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_mba_emittance(
    energy_gev: float,
    n_cells: int,
    n_bends_per_cell: int,
    n_unit_cells_per_period: int | None = None,
    reverse_bend_fraction: float = 0.0,
    lattice_type: str = "MBA",
) -> dict[str, Any]:
    """Estimate natural horizontal emittance for an MBA lattice.

    More accurate than the generic Sands scaling because it accounts for:
      - Reverse-bend contribution (RB increases I5 and Jx)
      - Finite dipole length and dispersion suppression
      - Achromatic cell structure

    Base formula: ε₀ = Cq · γ² · θ³ · F_eff
    where F_eff = F(type) · f_RB_correction · f_achromaticity

    The reverse bend increases emittance via:
      ε_RB ≈ ε₀ · (1 + c_RB · f_RB · N_RB/N_total)
    where c_RB ≈ 2-5 is the RB emittance penalty coefficient.

    Parameters
    ----------
    energy_gev : float
        Beam energy in GeV.
    n_cells : int
        Number of superperiods.
    n_bends_per_cell : int
        Number of main bends per unit cell.
    n_unit_cells_per_period : int | None
        Number of unit cells per superperiod. If None, assumed = 1.
    reverse_bend_fraction : float
        RB bending angle as fraction of total bend angle (0.0 = no RB).
    lattice_type : str
        MBA variant for the lattice factor.

    Returns
    -------
    dict with keys:
        emittance_pm_rad       : float
        emittance_nm_rad       : float
        emittance_tme_floor_pm : float  — TME lower bound
        theta_per_dipole_mrad  : float
        lattice_factor_F       : float
        effective_F            : float  — after RB/achromaticity corrections
        gamma                  : float
        reverse_bend_penalty_pct : float
        note                   : str
    """
    F = _LATTICE_FACTORS.get(lattice_type.upper(), 0.010)
    uc_per_period = n_unit_cells_per_period if n_unit_cells_per_period else 1
    total_dipoles = n_cells * n_bends_per_cell * uc_per_period
    gamma = energy_gev * 1e9 / _me_eV
    theta = 2.0 * math.pi / total_dipoles  # per-dipole bending angle [rad]

    # TME floor: ε_min = Cq · γ² · θ³ · F_TME
    # F_TME depends on bend count: F_TME(7BA) ≈ 0.002
    F_TME = {2: 0.050, 3: 0.020, 7: 0.004, 9: 0.0025}.get(n_bends_per_cell, 0.005)
    emittance_tme_m = _Cq * (gamma ** 2) * (theta ** 3) * F_TME * uc_per_period

    # Reverse-bend correction
    # RB increases effective F by adding to I5 while also increasing Jx
    if reverse_bend_fraction > 0:
        # RB fraction of total bending: f_RB = reverse_bend_fraction
        # Emittance penalty: ε ≈ ε₀ · (1 + α_RB · f_RB)
        # α_RB ≈ (n_bends_per_cell / uc_per_period) * 3  (empirical)
        alpha_rb = (n_bends_per_cell / uc_per_period) * 3.0
        rb_penalty = alpha_rb * reverse_bend_fraction
        F_eff = F * (1.0 + rb_penalty)
    else:
        rb_penalty = 0.0
        F_eff = F

    # Achromatic cell correction: achromatic cells (DBA, TBA, 7BA, etc.)
    # have Dx≈0, Dpx≈0 at the cell boundary.  This reduces the effective F
    # compared to non-achromatic cells.  The F values above already account
    # for this for the standard families.

    emittance_m = _Cq * (gamma ** 2) * (theta ** 3) * F_eff
    emittance_pm = emittance_m * 1e12
    emittance_nm = emittance_m * 1e9

    return {
        "emittance_pm_rad": round(emittance_pm, 2),
        "emittance_nm_rad": round(emittance_nm, 4),
        "emittance_tme_floor_pm": round(emittance_tme_m * 1e12, 2),
        "ratio_to_tme_floor": round(emittance_pm / (emittance_tme_m * 1e12 + 1e-20), 2),
        "theta_per_dipole_mrad": round(theta * 1e3, 3),
        "total_dipoles": total_dipoles,
        "gamma": round(gamma, 1),
        "lattice_factor_F_uncorrected": F,
        "effective_F": round(F_eff, 6),
        "reverse_bend_penalty_pct": round(rb_penalty * 100, 1),
        "note": (
            f"MBA emittance estimate for {lattice_type} at {energy_gev} GeV: "
            f"{emittance_pm:.1f} pm·rad "
            f"({emittance_pm / (emittance_tme_m * 1e12 + 1e-20):.1f}× the TME floor of "
            f"{emittance_tme_m * 1e12:.1f} pm·rad). "
            f"Theta per dipole = {theta * 1e3:.2f} mrad across {total_dipoles} dipoles. "
            + (
                f"Reverse bend ({reverse_bend_fraction*100:.1f}% of total bend angle) "
                f"adds {rb_penalty*100:.1f}% emittance penalty. "
                if reverse_bend_fraction > 0 else
                "No reverse bend — standard achromatic emittance."
            )
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Public tool registry
# ═══════════════════════════════════════════════════════════════════════════════

MBA_TOOLS = [
    compute_hoa_phase_advance,
    build_unit_cell_topology,
    decompose_superperiod,
    compute_correction_leverage,
    check_dsc_matching_conditions,
    validate_mba_construction_step,
    estimate_mba_emittance,
]
