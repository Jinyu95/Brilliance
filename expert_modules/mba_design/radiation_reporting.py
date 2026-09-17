"""Scope-bound interpretation of measured radiation; no optics or design solver."""
from __future__ import annotations

import math

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import geometry_snapshot
from brilliance.lattice import LatticeIR
from brilliance.lattice_fingerprint import cell_fingerprint
from brilliance.lattice_optimization import radiation_physical_validity

from .hybrid_shape import hybrid_mba_bend_roles
from .physics_references import homogeneous_bend_tme_reference


def mba_radiation_report(candidate, evaluation, *, bend_contributions=None):
    """Interpret ONE exact candidate's high-level JuTrack evaluation, without rerun.

    Emittance is a ratio of summed integrals, never a sum of bend emittances.
    Report Jx=1 reference and a separately labelled measured-Jx normalization.
    Optional api.bend_radiation_contributions must bind to the same lattice and
    supply each physical bend; radiation fractions use order-derived roles.
    This is interpretation, not an acceptance receipt or an attainable optimum.
    """
    candidate = candidate if isinstance(candidate, LatticeIR) else api.load_lattice(candidate)
    raw = evaluation.as_dict() if hasattr(evaluation, "as_dict") else evaluation
    view = api.evaluation_snapshot(raw)
    binding = geometry_snapshot(candidate)["binding"]
    if (view["geometry"].get("candidate_sha256") != binding["candidate_sha256"]
            or view["candidate_lattice_id"] != candidate.lattice_id):
        raise ValueError("Evaluation and reported candidate differ")
    radiation = view["radiation"]
    if not radiation_physical_validity(radiation)["passed"]:
        raise ValueError("No physical radiation equilibrium to interpret")
    reference = homogeneous_bend_tme_reference(candidate)
    adjusted = homogeneous_bend_tme_reference(candidate, damping_partition_x=radiation["Jx"])
    report = {"schema": "mba_radiation_interpretation/v1", "lattice_id": candidate.lattice_id,
        "physical_fingerprint": candidate.physical_fingerprint,
        "cell_fingerprint": cell_fingerprint(api.one_period_elements(candidate)),
        "scope": view["scope"], "geometry": view["geometry"],
        "measured_emittance_pm": radiation["emittance_pm"], "measured_Jx": radiation["Jx"],
        "reference_emittance_pm_Jx1": reference["reference_emittance_pm"],
        "measured_over_reference_Jx1": radiation["emittance_pm"] / reference["reference_emittance_pm"],
        "reference_emittance_pm_measured_Jx": adjusted["reference_emittance_pm"],
        "measured_over_reference_measured_Jx": radiation["emittance_pm"] / adjusted["reference_emittance_pm"],
        "interpretation": "Independent homogeneous-bend normalization; not an attainable full-lattice minimum. Bend count cancels for identical bends.",
        "bend_roles": [], "region_fractions": None}
    if bend_contributions is not None:
        if bend_contributions.get("lattice_id") != candidate.lattice_id:
            raise ValueError("Per-bend radiation belongs to another candidate")
        roles = hybrid_mba_bend_roles(api.one_period_elements(candidate))
        by_name = {row["bend"]: row for row in bend_contributions["rows"]}
        if set(by_name) != {b["name"] for b in roles["bends"]}:
            raise ValueError("Per-bend report must identify every physical bend, not merged families")
        totals = {key: math.fsum(row[key] for row in by_name.values()) for key in ("I2", "I4", "I5")}
        if any(not math.isclose(totals[key], radiation[key], rel_tol=1e-7, abs_tol=1e-13) for key in totals):
            raise ValueError("Per-bend integrals and evaluation have different radiation scopes")
        report["bend_roles"] = [{**bend, **{key: by_name[bend["name"]][key] for key in ("I2", "I4", "I5")},
            "I5_fraction": by_name[bend["name"]]["I5"] / totals["I5"]} for bend in roles["bends"]]
        report["region_fractions"] = {region: math.fsum(b["I5_fraction"] for b in report["bend_roles"]
            if (b["role"] == "low_H_core") == (region == "core")) for region in ("core", "outer_pairs")}
    return report
