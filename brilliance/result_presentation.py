"""Present retained JuTrack evidence, never run or infer a physics calculation.

The managed program may use compact API returns while the chat still needs the
computed profile. Keep a small run-scoped cache; render only the exact candidate
committed by ResearchRun.finish, not every optimizer trial.
"""
from __future__ import annotations

import copy
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

from .artifacts import json_safe

_scope = ""
_evidence: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_CANDIDATES = 8


def _active_scope() -> str:
    global _scope
    configured = str(os.environ.get("BRILLIANCE_RUN_OUTPUT", "")).strip()
    scope = str(Path(configured).resolve()) if configured else ""
    if scope != _scope:
        _evidence.clear()
        _scope = scope
    return scope


def remember_candidate_evidence(candidate: Any, *, linear: Mapping[str, Any] | None = None,
                                bends: Mapping[str, Any] | None = None) -> None:
    """Retain already-computed evidence in managed runs; do no filesystem I/O."""
    if not _active_scope():
        return
    from .accelerator_api import one_period_elements
    from .plotting import element_span_m, profile_through

    profile = linear.get("twiss_profile", []) if isinstance(linear, Mapping) else []
    if not profile and bends is None:
        return
    key = candidate.lattice_id
    record = _evidence.setdefault(key, {"lattice_id": key, "candidate_title": str(candidate.title)})
    _evidence.move_to_end(key)
    while len(_evidence) > _MAX_CANDIDATES:
        _evidence.popitem(last=False)
    if profile and linear.get("stable") is True:
        try:
            if len(profile) < 2:
                raise ValueError("computed optics profile has fewer than two points")
            for index, point in enumerate(profile):
                for field in ("s_m", "beta_x", "beta_y", "Dx"):
                    if not isinstance(point, Mapping) or not math.isfinite(float(point.get(field))):
                        raise ValueError(f"computed optics profile point {index} has missing/non-finite {field}")
            elements = one_period_elements(candidate)
            length = element_span_m(elements)
            selected = profile_through(profile, length)
            if len(selected) < 2 or abs(float(selected[-1]["s_m"]) - length) > max(1e-8, length * 1e-9):
                raise ValueError("computed optics profile does not cover the declared complete cell")
            record.pop("optics_unavailable", None)
            record.pop("unplotted_profile", None)
            record.update({
                "profile": json_safe(selected),
                "elements": json_safe(elements),
                "cell_length_m": length,
                "lattice_length_m": candidate.total_length_m,
                "source": "JuTrack computed periodic optics",
            })
        except Exception as exc:
            # Plot scope/data validation is not a new physics gate. Keep the
            # completed evaluation and exact unavailable-data diagnostic;
            # never silently join a curve across NaN/missing samples.
            record.pop("profile", None)
            record.pop("elements", None)
            record["optics_unavailable"] = f"{type(exc).__name__}: {exc}"
            record["unplotted_profile"] = json_safe(profile)
    if bends is not None:
        record["bend_radiation"] = copy.deepcopy(json_safe(bends))


def _has_bend_table(tables: list[dict[str, Any]]) -> bool:
    return any(table.get("kind") == "bend_radiation" or any(
        isinstance(row, Mapping) and {"bend", "I2", "I5"} <= row.keys()
        for row in table.get("rows", [])
    ) for table in tables)


def present_candidate_evidence(candidate: Any, output: Path, *,
                               tables: list[dict[str, Any]], plots: list[dict[str, Any]],
                               candidate_receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Attach canonical views only from evidence for this exact lattice id.

    Explicit plots are respected. Callers can identify an explicit per-bend
    table using ``kind='bend_radiation'`` or the raw bend/I2/I5 columns. Missing
    evidence is reported, not manufactured by recalculation or interpolation.
    The selected verified archive, not the transient optimizer cache, owns its
    optics. Reading that archive checks its hash, scope and exact candidate rows.
    """
    if _active_scope() != str(output.resolve()):
        return {"status": "unavailable", "reason": "No evidence captured in this program."}
    if candidate_receipt is not None:
        from .candidate_evidence import read_numerical_evidence

        try:
            measured = read_numerical_evidence(candidate, candidate_receipt, output_directory=output)
            linear = measured.get("linear_optics")
            remember_candidate_evidence(candidate, linear=linear)
            retained = _evidence.get(candidate.lattice_id)
            if retained is not None and isinstance(linear, Mapping) and linear.get("twiss_profile"):
                retained["optics_receipt"] = copy.deepcopy(json_safe(candidate_receipt))
        except Exception as exc:
            # Keep the numerical checkpoint. Never fall back to caller-provided
            # curves or another simulation when the selected archive cannot be read.
            error = f"Archived optics presentation unavailable: {type(exc).__name__}: {exc}"
            return {"status": "unavailable", "reason": error, "errors": [error],
                    "simulation_rerun": False}
    record = _evidence.get(candidate.lattice_id)
    if not record:
        return {"status": "unavailable", "reason": "This exact candidate has no retained optics or per-bend evidence."}
    from .accelerator_api import save_bend_radiation_contribution_plot, save_optics_lattice_plot

    evidence_path = output / "computed_presentation.json"
    try:
        evidence_path.write_text(json.dumps(json_safe(record), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        error = f"Presentation archive unavailable: {type(exc).__name__}: {exc}"
        return {"status": "unavailable", "reason": error, "errors": [error],
                "simulation_rerun": False}
    generated: list[str] = []
    failures: list[str] = ([f"Optics presentation unavailable: {record['optics_unavailable']}"]
                           if record.get("optics_unavailable") else [])
    explicit_plots = bool(plots)
    profile = record.get("profile", [])
    if len(profile) >= 2 and not explicit_plots:
        name = "one_cell_optics_lattice.png"
        try:
            save_optics_lattice_plot(output / name, profile, record["elements"],
                title="One-cell optics and lattice",
                subtitle=f"{candidate.title}; cell length {record['cell_length_m']:.6g} m\n{candidate.lattice_id}")
            plots.append({"path": name, "title": f"{candidate.title} — optics and lattice of one {record['cell_length_m']:.6g} m cell",
                          "kind": "optics_lattice", "lattice_id": candidate.lattice_id})
            generated.append(name)
        except Exception as exc:
            # A presentation error must not discard the numerical result.
            failures.append(f"Optics presentation failed: {type(exc).__name__}: {exc}")
    bends = record.get("bend_radiation", {})
    rows = bends.get("rows", [])
    if rows and bends.get("consistency", {}).get("passed") is not True:
        failures.append("Per-bend presentation unavailable: the JuTrack elrad decomposition did not pass its ringpara consistency check.")
    if rows and bends.get("consistency", {}).get("passed") is True:
        if not _has_bend_table(tables):
            tables.append({"title": "Per-bend radiation contributions (JuTrack)",
                "kind": "bend_radiation", "lattice_id": candidate.lattice_id,
                "generated_from_evidence": True,
                "description": "Rows group repeated physical instances by source bend. θ and L are per-instance means; ⟨H⟩ is the curvature-weighted group mean. I₂, I₄ and I₅ are summed over the listed instances; fractions refer to the full ring.",
                "rows": [{"Bend": row.get("bend"), "Instances": row.get("instances"), "θ (rad)": row.get("angle_rad"),
                          "L (m)": row.get("length_m"), "⟨H⟩ (m)": row.get("mean_H_m"),
                          "I₂ (m⁻¹)": row.get("I2"), "I₄ (m⁻¹)": row.get("I4"),
                          "I₅ (m⁻¹)": row.get("I5"), "I₅ / I₂": row.get("I5_over_I2"),
                          "I₂ fraction": row.get("I2_fraction"), "I₅ fraction": row.get("I5_fraction")}
                         for row in rows if isinstance(row, Mapping)]})
        if not explicit_plots:
            name = "bend_radiation_contributions.svg"
            try:
                save_bend_radiation_contribution_plot(output / name, bends,
                    subtitle=f"JuTrack elrad checked against ringpara; {candidate.lattice_id}")
                plots.append({"path": name, "title": f"{candidate.title} — per-bend I₂ and I₅ contributions",
                              "kind": "bend_radiation", "lattice_id": candidate.lattice_id})
                generated.append(name)
            except Exception as exc:
                failures.append(f"Bend presentation failed: {type(exc).__name__}: {exc}")
    return {"status": "partial" if failures else "available", "evidence": evidence_path.name,
            "lattice_id": candidate.lattice_id, "generated_figures": generated,
            "optics_available": len(profile) >= 2, "bend_consistency": bends.get("consistency"),
            "errors": failures, "simulation_rerun": False}
