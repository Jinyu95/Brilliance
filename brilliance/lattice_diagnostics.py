"""Non-mutating structural observations about explicitly declared lattice metadata."""
from __future__ import annotations

import math
from typing import Any

from .lattice import LatticeIR


def declared_mirror_diagnostics(candidate: LatticeIR) -> dict[str, Any] | None:
    """Compare declared mirror partners without imposing a symmetry requirement.

    ``mirror_source_element`` is explicit construction provenance, not proof that
    a later asymmetric study is invalid. Report discrepancies for interpretation;
    never modify fields, infer new partners by position, or reject the candidate.
    Repeated cells prepend namespaces but preserve the original source marker.
    """

    by_name = {element.name: element for element in candidate.elements}
    declarations = [element for element in candidate.elements
                    if str(element.fields.get("mirror_source_element", "")).strip()]
    if not declarations:
        return None
    checked: set[tuple[str, str]] = set()
    unresolved: list[dict[str, str]] = []
    differences: list[dict[str, Any]] = []
    compared = 0
    for element in declarations:
        source = str(element.fields["mirror_source_element"]).strip()
        # Prefer the nearest enclosing instance namespace. For example,
        # P003:P002:R:QF -> P003:P002:QF, not a different cell's bare QF.
        parts = element.name.split(":")
        targets = [":".join(parts[:size] + [source])
                   for size in range(len(parts) - 1, -1, -1)]
        if "source_period" not in element.fields:
            # Without repeat-instance provenance the literal declaration wins;
            # a second reflection may retain an older source marker deliberately.
            targets.insert(0, source)
        partner = next((by_name[name] for name in targets
                        if name != element.name and name in by_name), None)
        if partner is None:
            unresolved.append({"element": element.name, "declared_source": source})
            continue
        key = tuple(sorted((element.name, partner.name)))
        if key in checked:
            continue
        checked.add(key)
        compared += 1
        fields: list[dict[str, Any]] = []
        if element.kind != partner.kind:
            fields.append({"field": "kind", "source_value": partner.kind,
                           "mirror_value": element.kind})
        for field in ("length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4"):
            left = float(getattr(partner, field) or 0.0)
            right = float(getattr(element, field) or 0.0)
            if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
                fields.append({"field": field, "source_value": left, "mirror_value": right})
        if fields:
            differences.append({"source_element": partner.name, "mirror_element": element.name,
                                "differences": fields})
    return {
        "title": "declared mirror partners differ" if differences else "declared mirror partners checked",
        "severity": "advisory" if differences else "info",
        "scope": "explicit mirror_source_element metadata; scalar physical fields only",
        "lattice_id": candidate.lattice_id,
        "declared_partner_count": len(declarations),
        "compared_partner_count": compared,
        "mismatch_count": len(differences),
        "mismatches": differences,
        "unresolved_declarations": unresolved,
        "interpretation": (
            "Explicit mirror provenance no longer describes equal physical fields. "
            "These declarations may describe old local unit pairs, not the complete lattice's "
            "global mirror partners; releasing local unit symmetry can preserve global symmetry. "
            "Explain intentional changes or correct unintended field changes, and check the "
            "claimed symmetry scope separately. "
            "This advisory does not reject or modify the candidate."
        ) if differences else "No differing physical scalar fields were found among resolved declared partners.",
    }
