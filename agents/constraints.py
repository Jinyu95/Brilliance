"""Parse accelerator design constraints from user requests and notebook text."""

from __future__ import annotations

import re
from typing import Any


_CELL_TYPES = ("H7BA", "H6BA", "7BA", "9BA", "DBA", "FODO", "TBA", "MBA")

_EXPLICIT_CELL_TYPE_PATTERNS = [
    re.compile(r"\bcell_type\s*[:=]\s*(H7BA|H6BA|7BA|9BA|DBA|FODO|TBA|MBA)\b", re.IGNORECASE),
    re.compile(r"\blattice_type\s*[:=]\s*(H7BA|H6BA|7BA|9BA|DBA|FODO|TBA|MBA)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:chosen_direction|recommended_direction)\s*:\s*(?:[^:\n]*?\b)?(H7BA|H6BA|7BA|9BA|DBA|FODO|TBA|MBA)\b",
        re.IGNORECASE,
    ),
]

_CELL_TYPE_PATTERNS = {
    "H7BA": (
        re.compile(r"\bH7BA\b", re.IGNORECASE),
        re.compile(r"\bhybrid[-\s]*7[-\s]*bend\b", re.IGNORECASE),
        re.compile(r"\breverse[-\s]*bend[-\s]*7BA\b", re.IGNORECASE),
    ),
    "H6BA": (
        re.compile(r"\bH6BA\b", re.IGNORECASE),
        re.compile(r"\bhybrid[-\s]*6[-\s]*bend\b", re.IGNORECASE),
    ),
    "7BA": (
        re.compile(r"\b7BA\b", re.IGNORECASE),
        re.compile(r"\bSEVEN[-\s]*BEND ACHROMAT\b", re.IGNORECASE),
    ),
    "9BA": (
        re.compile(r"\b9BA\b", re.IGNORECASE),
        re.compile(r"\bNINE[-\s]*BEND ACHROMAT\b", re.IGNORECASE),
    ),
    "DBA": (
        re.compile(r"\bDBA\b", re.IGNORECASE),
        re.compile(r"\bDOUBLE[-\s]*BEND ACHROMAT\b", re.IGNORECASE),
    ),
    "FODO": (re.compile(r"\bFODO\b", re.IGNORECASE),),
    "TBA": (
        re.compile(r"\bTBA\b", re.IGNORECASE),
        re.compile(r"\bTRIPLE[-\s]*BEND ACHROMAT\b", re.IGNORECASE),
    ),
    "MBA": (
        re.compile(r"\bMBA\b", re.IGNORECASE),
        re.compile(r"\bMULTI[-\s]*BEND ACHROMAT\b", re.IGNORECASE),
    ),
}

_MACHINE_CLASS_PATTERNS = [
    ("booster", re.compile(r"\bbooster(?:\s+ring)?\b", re.IGNORECASE)),
    ("storage_ring", re.compile(r"\bstorage\s+ring\b", re.IGNORECASE)),
    ("transfer_line", re.compile(r"\btransfer\s+line\b", re.IGNORECASE)),
    ("linac", re.compile(r"\blinac\b|\blinear\s+accelerator\b", re.IGNORECASE)),
    ("bunch_compressor", re.compile(
        r"\bbunch[-\s]*compressor\b|\bchicane\b|\bdog[-\s]*leg\b|\bR56\b",
        re.IGNORECASE,
    )),
    ("insertion", re.compile(
        r"\binsertion\s+section\b|\bstraight\s+section\b|\bIP\s+region\b"
        r"|\blow[-\s]*beta\s+section\b|\bfree[-\s]*electron\s+laser\b|\bFEL\b",
        re.IGNORECASE,
    )),
    # "cell" matches phrases like "TBA cell", "FODO cell", "unit cell"
    # but NOT inside a ring description.
    ("cell", re.compile(
        r"\b(?:unit[-\s]*cell|single[-\s]*cell|cell[-\s]*design|cell[-\s]*study)\b"
        r"|\b(?:TBA|DBA|FODO|MBA|H7BA|H6BA|7BA|9BA|FODO)[-\s]*cell\b"
        r"|\bdesign\s+(?:an?\s+)?(?:TBA|DBA|FODO|MBA|H7BA|H6BA|7BA|9BA)(?:\s+cell)?\b",
        re.IGNORECASE,
    )),
]

# Patterns that explicitly describe the task as a ring/full-machine design
# (used to suppress the cell classification when ring context is clear)
_RING_CONTEXT_PATTERNS = re.compile(
    r"\b(?:full\s+ring|storage\s+ring|booster\s+ring|lattice\s+ring|ring\s+design|design\s+a\s+ring)\b",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    normalized = text or ""
    for broken in ("鐠?", "閳?", "鍗?", "鈥?", "卤", "±"):
        normalized = normalized.replace(broken, " ")
    return normalized


def _extract_machine_class(text: str) -> str | None:
    normalized = _normalize_text(text)

    # 1. Check for explicit notebook field first — this is authoritative.
    explicit = re.search(
        r"machine_class\s*:\s*"
        r"(storage_ring|booster|transfer_line|linac|cell"
        r"|bunch_compressor|insertion|chicane|unknown)\b",
        normalized,
        re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).lower()

    # 2. Fall back to keyword matching, but disambiguate:
    #    If "storage ring" appears, don't let an incidental mention of
    #    "booster" (e.g. "injection from separate booster") override it.
    found: list[str] = []
    for machine_class, pattern in _MACHINE_CLASS_PATTERNS:
        if pattern.search(normalized):
            found.append(machine_class)
    if not found:
        return None
    if len(found) == 1:
        return found[0]

    # "cell" is suppressed if the text also clearly describes a full ring.
    if "cell" in found and _RING_CONTEXT_PATTERNS.search(normalized):
        found = [f for f in found if f != "cell"]
    if len(found) == 1:
        return found[0]

        # bunch_compressor and insertion are specific — don't suppress them.
        if "bunch_compressor" in found:
            return "bunch_compressor"
        if "insertion" in found:
            return "insertion"
        booster_design = re.search(
            r"\b(?:design|build|initiate|create)\b[^.]{0,40}\bbooster\b",
            normalized,
            re.IGNORECASE,
        )
        if not booster_design:
            return "storage_ring"
    return found[0]


def _find_cell_types(text: str) -> list[str]:
    normalized = _normalize_text(text)
    found: list[str] = []
    for cell_type, patterns in _CELL_TYPE_PATTERNS.items():
        if any(pattern.search(normalized) for pattern in patterns):
            found.append(cell_type)
    return found


def _extract_cell_type(text: str) -> str | None:
    normalized = _normalize_text(text)

    for pattern in _EXPLICIT_CELL_TYPE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return str(match.group(1)).upper()

    found = _find_cell_types(normalized)
    if len(found) == 1:
        return found[0]
    return None


def _extract_candidate_cell_types(text: str) -> list[str] | None:
    found = _find_cell_types(text)
    if len(found) <= 1:
        return None
    return found


def _is_range_like_match(text: str, start: int) -> bool:
    idx = start - 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    if idx >= 0 and text[idx] in {"-", "–", "—", "~"}:
        return True
    prefix = text[max(0, start - 8):start].lower()
    return prefix.rstrip().endswith("to")


def _last_clean_numeric_match(
    text: str,
    patterns: list[str],
    *,
    minimum: float | None = None,
) -> float | None:
    candidates: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if _is_range_like_match(text, match.start(1)):
                continue
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                continue
            if minimum is not None and value < minimum:
                continue
            candidates.append(value)
    if candidates:
        return candidates[-1]
    return None


def _extract_energy_gev(text: str) -> float | None:
    patterns = [
        r"(?:beam\s+)?energy(?:\s*\[gev\])?\s*[:=]\s*(\d+(?:\.\d+)?)\s*gev\b",
        r"energy_gev\s*[:=]\s*(\d+(?:\.\d+)?)\b",
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*gev\b",
    ]
    return _last_clean_numeric_match(text, patterns, minimum=0.05)


def _extract_injection_energy_gev(text: str) -> float | None:
    normalized = _normalize_text(text)
    patterns = [
        r"injection(?:\s+beam)?\s+energy(?:\s*\[gev\])?\s*[:=]\s*(\d+(?:\.\d+)?)\s*gev\b",
        r"injection(?:\s+beam)?\s+energy(?:\s*\[mev\])?\s*[:=]\s*(\d+(?:\.\d+)?)\s*mev\b",
        r"from\s+(\d+(?:\.\d+)?)\s*gev\b",
        r"from\s+(\d+(?:\.\d+)?)\s*mev\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                continue
            if "mev" in match.group(0).lower():
                value /= 1000.0
            if value > 0:
                return value
    return None


def _extract_count(text: str) -> int | None:
    patterns = [
        r"Number of cells\s*:\s*(\d+)",
        r"(\d+)\s+cells\b",
        r"(\d+)[-\s]*cell\b",
        r"(\d+)[-\s]*fold symmetry\b",
        r"(\d+)\s+superperiods\b",
        r"(\d+)\s+periods\b",
    ]
    matches: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                matches.append(int(match.group(1)))
            except (TypeError, ValueError):
                continue
    if matches:
        return matches[-1]
    return None


def _extract_circumference_m(text: str) -> float | None:
    patterns = [
        r"Circumference\s*:\s*(\d+(?:\.\d+)?)\s*m",
        r"circumference\s*(?:around|~|approximately|approx\.?|=|targets?)?\s*(\d+(?:\.\d+)?)\s*m",
        r"C\s*~\s*(\d+(?:\.\d+)?)\s*m",
        r"(?<![\d.])([1-9]\d{2,})\s+m\b",
    ]
    return _last_clean_numeric_match(text, patterns, minimum=50.0)


def _unit_scale(unit_prefix: str | None) -> float:
    prefix = (unit_prefix or "m").lower()
    scales = {
        "pm": 1e-12,
        "nm": 1e-9,
        "um": 1e-6,
        "mm": 1e-3,
        "m": 1.0,
    }
    return scales.get(prefix, 1.0)


def _extract_target_emittance(text: str) -> float | None:
    normalized = _normalize_text(text)
    patterns = [
        r"Target emittance\s*:\s*(\d+(?:\.\d+)?)\s*(pm|nm|um|mm|m)?(?:[-\s]*rad|\.rad)",
        r"target(?:\s+horizontal)?\s+emittance\s*(?:of|=|:|~|approximately)?\s*(\d+(?:\.\d+)?)\s*(pm|nm|um|mm|m)?(?:[-\s]*rad|\.rad)",
        r"emittance\s*(?:below|under|less than|<)\s*(\d+(?:\.\d+)?)\s*(pm|nm|um|mm|m)?(?:[-\s]*rad|\.rad)",
        r"target natural emittance\s*(?:below|under|less than|<)?\s*(\d+(?:\.\d+)?)\s*(pm|nm|um|mm|m)?(?:[-\s]*rad|\.rad)",
        r"(?<![\d.])([\d]+(?:\.\d+)?)\s*(pm|nm|um)[-\s]*rad\b",
    ]
    candidates: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            if _is_range_like_match(normalized, match.start(1)):
                continue
            value = float(match.group(1))
            prefix = match.group(2)
            candidates.append(value * _unit_scale(prefix or "m"))
    if candidates:
        return candidates[-1]
    return None


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = _normalize_text(text).lower()
    return any(phrase in lowered for phrase in phrases)


def parse_design_constraints(text: str | None) -> dict[str, Any]:
    """Infer structured design constraints from user prompts or notebook text."""
    if not text:
        return {}

    normalized = _normalize_text(text)
    machine_class = _extract_machine_class(normalized)
    cell_type = _extract_cell_type(normalized)
    candidate_cell_types = _extract_candidate_cell_types(normalized)

    constraints = {
        "machine_class": machine_class,
        "cell_type": cell_type,
        "candidate_cell_types": candidate_cell_types,
        "energy_gev": _extract_energy_gev(normalized),
        "injection_energy_gev": _extract_injection_energy_gev(normalized),
        "n_cells": _extract_count(normalized),
        "circumference_m": _extract_circumference_m(normalized),
        "target_emittance": _extract_target_emittance(normalized),
        "requires_injection_straight": _contains_any(
            normalized,
            (
                "injection straight",
                "off-axis injection",
                "injection section",
                "injection kicker",
                "injection constraint",
                "injection constraints",
                "dedicated injection",
                "injection and extraction",
                "septum",
            ),
        ),
        "requires_extraction_straight": _contains_any(
            normalized,
            (
                "extraction straight",
                "extraction section",
                "extraction kicker",
                "extraction constraint",
                "extraction constraints",
                "dedicated extraction",
                "injection and extraction",
                "extraction septum",
            ),
        ),
        "requires_energy_ramp": _contains_any(
            normalized,
            ("ramp energy", "ramp from", "accelerating from", "ramping"),
        ),
        # Cell-level physics conditions
        "requires_isochronous": _contains_any(
            normalized,
            ("isochronous", "isochronicity", "zero momentum compaction", "alphap = 0", "alpha_p = 0"),
        ),
        "requires_achromatic": _contains_any(
            normalized,
            ("achromatic", "achromat", "zero dispersion", "dispersion-free"),
        ),
        # Other cell/beamline task types
        "requires_phase_advance": bool(
            re.search(r"\b(?:phase\s+advance|betatron\s+phase|phase\s+slip)\b", normalized, re.IGNORECASE)
        ),
        # General computation task flags
        "requires_transfer_matrix": _contains_any(
            normalized,
            ("transfer matrix", "m66", "findm66", "6x6 matrix", "6×6 matrix",
             "matrix element", "t56", "r56", "coupling matrix"),
        ),
        "requires_twiss_propagation": _contains_any(
            normalized,
            ("twiss propagation", "twissline", "propagate twiss", "beta function along",
             "beam envelope", "optics along", "envelope along"),
        ),
        "requires_tracking": _contains_any(
            normalized,
            ("particle tracking", "track particles", "track beam", "multi-particle",
             "single-particle tracking", "trackParticles", "turn-by-turn"),
        ),
        "requires_dynamic_aperture": _contains_any(
            normalized,
            ("dynamic aperture", "da study", "on-momentum da", "off-momentum da",
             "frequency map", "fma"),
        ),
        "requires_momentum_aperture": _contains_any(
            normalized,
            ("momentum aperture", "ma study", "touschek", "beam lifetime"),
        ),
        "requires_bunch_compression": _contains_any(
            normalized,
            ("bunch compressor", "bunch compression", "r56", "chicane",
             "longitudinal phase space", "chirp"),
        ),
        "concept_only": _contains_any(
            normalized,
            ("do not write code", "don't write code", "no code yet", "first discuss", "discussion only", "concept only"),
        ),
    }

    # Avoid reporting a weak generic MBA cell type when more specific candidates are still open.
    if cell_type == "MBA" and candidate_cell_types:
        constraints["cell_type"] = None

    return {
        key: value
        for key, value in constraints.items()
        if value is not None and value is not False and value != []
    }
