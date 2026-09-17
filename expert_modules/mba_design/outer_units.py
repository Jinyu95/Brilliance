"""TME repetition and outer-unit ownership; no facility lattice or stored strengths."""
from __future__ import annotations

import copy
import math
import os
import time
from typing import Any, Callable, Mapping, Sequence

from brilliance import accelerator_api as api
from brilliance.experiment_geometry import _verify_global_mirror, project_weighted_sum
from brilliance.lattice_optimization import LatticeParameterization

from .result_types import OuterUnitAssembly, OuterUnitMatch


def assemble_mba_outer_units(unit_elements: Sequence[Mapping[str, Any]], *, energy_gev: float,
                             periods: int, bends_per_cell: int, cell_length_m: float,
                             straight_length_m: float, limits: Mapping[str, float],
                             minimum_magnet_length_m: float, minimum_drift_length_m: float,
                             end_bend_fraction: float = 1.0) -> OuterUnitAssembly:
    """Assemble M complete TME-derived units before matching the outer two pairs.

    Requires M>=5, a globally symmetric single-positive-bend source, and its
    angle theta=2*pi/[P*(M-2+2*end_bend_fraction)]. By default repeat equal bends;
    end_bend_fraction=.5 explicitly chooses half-angle end bends. This closes
    the cell, not an optical match. The M-4
    central copies retain every physical field. Reserve the full straight;
    allocate the remaining arc length equally to the four outer units by
    projecting their drift lengths, keeping source magnet lengths. Infeasible
    allocations raise before numerical work, never shrink requirements.
    Returns explicit rows, all M provenance groups, and geometric audit.
    Repetition is not an achromat; no stability or low-emittance claim is made.
    """
    if type(periods) is not int or periods < 1 or type(bends_per_cell) is not int or bends_per_cell < 5:
        raise ValueError("positive integer periods and bends_per_cell >= 5 are required")
    if not (math.isfinite(cell_length_m) and math.isfinite(straight_length_m)
            and cell_length_m > straight_length_m > 0):
        raise ValueError("finite cell_length_m > straight_length_m > 0 required")
    if isinstance(end_bend_fraction, bool) or not math.isfinite(end_bend_fraction) or end_bend_fraction <= 0:
        raise ValueError("end_bend_fraction must be finite and positive")
    unit = api.repeat_cell(unit_elements, 1, period_prefix="SOURCE")
    # Restore names, retaining source field values and native integration options.
    for row, original in zip(unit, unit_elements):
        row["name"] = original["name"]
        row["fields"] = copy.deepcopy(original.get("fields", {}))
    _verify_global_mirror(unit)
    bends = [row for row in unit if float(row.get("angle_rad") or 0) != 0]
    theta = 2 * math.pi / (periods * (bends_per_cell - 2 + 2 * end_bend_fraction))
    if len(bends) != 1 or not math.isclose(float(bends[0]["angle_rad"]), theta, rel_tol=0, abs_tol=1e-12):
        raise ValueError("source angle does not close the declared MBA bend distribution; re-evaluate a different unit explicitly")
    if bends[0]["kind"] != "SBEND" or any(float(bends[0].get(key) or 0) != 0 for key in ("k1_m2", "k2_m3", "k3_m4")) or any(
            float(value) != 0 for value in bends[0].get("fields", {}).get("PolynomB", [])):
        raise ValueError("this conventional route requires intact pure positive sector bends")
    unit_length = math.fsum(row["length_m"] for row in unit)
    outer_length = (cell_length_m - straight_length_m - (bends_per_cell - 4) * unit_length) / 4
    source_audit = api.audit_lattice_constraints(unit, energy_gev=energy_gev, limits=limits,
        expected_length_m=unit_length, expected_angle_rad=theta, expected_bends=1,
        require_mirror_symmetry=True, minimum_magnet_length_m=minimum_magnet_length_m,
        minimum_drift_length_m=minimum_drift_length_m)
    if not source_audit["passed"]:
        raise ValueError("source unit engineering: " + "; ".join(source_audit["issues"]))
    rows = [api.drift("ID:L", straight_length_m / 2)]
    groups = []
    outer_indices = {1, 2, bends_per_cell - 1, bends_per_cell}
    for index in range(1, bends_per_cell + 1):
        part = api.repeat_cell(unit, 1, period_prefix="TMP")
        for row, original in zip(part, unit):
            row["name"] = f"U{index:03d}:{original['name']}"
            row["fields"] = {**copy.deepcopy(original["fields"]),
                             "source_element": original["name"], "source_period": index}
        if index in outer_indices:
            drifts = [row for row in part if row["kind"] == "DRIFT"]
            fixed = math.fsum(row["length_m"] for row in part if row["kind"] != "DRIFT")
            lengths = project_weighted_sum([row["length_m"] for row in drifts],
                [minimum_drift_length_m] * len(drifts), [cell_length_m] * len(drifts),
                [1.] * len(drifts), outer_length - fixed)
            for row, length in zip(drifts, lengths):
                row["length_m"] = length
        if index in {1, bends_per_cell}:
            next(row for row in part if row["kind"] == "SBEND")["angle_rad"] = theta * end_bend_fraction
        groups.append([row["name"] for row in part])
        rows.extend(part)
    rows.append(api.drift("ID:R", straight_length_m / 2))
    _verify_global_mirror(rows)
    audit = api.audit_lattice_constraints(rows, energy_gev=energy_gev, limits=limits,
        expected_length_m=cell_length_m, expected_angle_rad=2 * math.pi / periods,
        expected_bends=bends_per_cell, require_mirror_symmetry=True,
        minimum_magnet_length_m=minimum_magnet_length_m, minimum_drift_length_m=minimum_drift_length_m)
    if not audit["passed"]:
        raise ValueError("assembled MBA engineering: " + "; ".join(audit["issues"]))
    return {"elements": rows, "unit_groups": groups, "core_unit_groups": groups[2:-2],
            "id_drift_names": [rows[0]["name"], rows[-1]["name"]], "id_length_m": straight_length_m,
            "outer_unit_numbers": sorted(outer_indices), "outer_unit_length_m": outer_length,
            "source_unit_length_m": unit_length, "geometry": audit, "physics_accepted": False,
            "end_bend_fraction": end_bend_fraction,
            "route": "tme_repetition_outer_pairs"}


def mba_outer_unit_knobs(elements: Sequence[Mapping[str, Any]], *,
                         unit_groups: Sequence[Sequence[str]], limits: Mapping[str, float]) -> LatticeParameterization:
    """Select globally mirror-paired K1 knobs in first/last TWO whole units.

    Uses declared ordered physical groups, not QF/QD or inherited TME family
    names. Quadrupoles inside an original unit are NOT tied internally.
    Core, bends and lengths are untouched. Returns a general parameterization.
    """
    _verify_global_mirror(elements)
    if len(unit_groups) < 5:
        raise ValueError("at least five ordered unit groups required")
    lookup = {row["name"]: i for i, row in enumerate(elements)}
    indices = [[lookup[name] for name in group] for group in unit_groups]
    for group in indices:
        if not group or group != list(range(group[0], group[-1] + 1)):
            raise ValueError("unit groups must be nonempty contiguous ordered intervals")
    flat = [i for group in indices for i in group]
    if flat != list(range(flat[0], flat[-1] + 1)):
        raise ValueError("unit groups overlap or have missing/out-of-order elements")
    outer = set(indices[0] + indices[1] + indices[-2] + indices[-1])
    specs = []
    cap = float(limits["max_abs_k1_m2"])
    for index in indices[0] + indices[1]:
        row, partner = elements[index], elements[-1-index]
        if row["kind"] not in {"KQUAD", "QUAD"}:
            continue
        if len(elements) - 1 - index not in outer:
            raise ValueError("declared outer groups are not globally mirrored")
        specs.append({"name": row["name"] + ":mirror_K1", "field": "k1_m2",
                      "element_names": [row["name"], partner["name"]], "bounds": [-cap, cap]})
    if not specs:
        raise ValueError("outer units contain no independently controllable quadrupoles")
    return api.compile_lattice_parameterization(elements, specs)


def match_mba_outer_units(assembly: OuterUnitAssembly, *, energy_gev: float, periods: int,
                          limits: Mapping[str, float], maximum_beta_m: float | None,
                          minimum_magnet_length_m: float, minimum_drift_length_m: float,
                          trace_margin_x: float = .01, trace_margin_y: float = .01,
                          dx_tolerance_m: float = 1e-4, dpx_tolerance: float = 1e-5,
                          max_starts: int = 64, max_iterations: int = 80, seed: int = 0,
                          progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
                          cancellation_check: Callable[[], None] | None = None) -> OuterUnitMatch:
    """Native response match of the existing outer two units; no added terminal.

    Match periodic Dx/Dpx of the COMPLETE assembled cell at the ID boundary.
    Core magnets stay unchanged, but core optics are NOT frozen to isolated-unit
    optics. No tune, alpha, or fixed ID-beta target. Solver uses the general
    bounded_response_match and JuTrackLatticeEditor; every response is native
    periodic optics, never a line with an imposed source Twiss state. A passing
    response is replayed densely and radiation checked. Caller MUST evaluate the explicit ring,
    archive receipts and run conventional acceptance before a machine claim.
    Budgets are explicit search hypotheses, not a proof of infeasibility.
    Unstable starts first restore native trace inequalities, not a tune target.
    Dense interior-beta refinement is applied only to shortlisted matches.
    Managed progress/checkpoints/cancellation are supplied automatically.
    """
    for key, value in (("max_iterations", max_iterations), ("periods", periods)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, value in (("dx_tolerance_m", dx_tolerance_m), ("dpx_tolerance", dpx_tolerance),
                       ("trace_margin_x", trace_margin_x), ("trace_margin_y", trace_margin_y)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if max(trace_margin_x, trace_margin_y) >= 1:
        raise ValueError("trace margins must be less than one")
    if maximum_beta_m is not None and (isinstance(maximum_beta_m, bool) or not math.isfinite(maximum_beta_m) or maximum_beta_m <= .05):
        raise ValueError("maximum_beta_m must exceed the 0.05 m numerical refinement headroom or be None")
    rows = copy.deepcopy(assembly["elements"])
    groups = assembly["unit_groups"]
    full_knobs = mba_outer_unit_knobs(rows, unit_groups=groups, limits=limits)
    audit = api.audit_lattice_constraints(rows, energy_gev=energy_gev, limits=limits,
        expected_length_m=assembly["geometry"]["length_m"], expected_angle_rad=2 * math.pi / periods,
        expected_bends=len(groups), require_mirror_symmetry=True,
        minimum_magnet_length_m=minimum_magnet_length_m, minimum_drift_length_m=minimum_drift_length_m)
    if not audit["passed"]:
        raise ValueError("outer-unit matching contract: " + "; ".join(audit["issues"]))
    plan = api.plan_matching_starts(full_knobs, limits=limits, max_starts=max_starts, seed=seed)
    run = None
    if os.environ.get("BRILLIANCE_RUN_OUTPUT"):
        from brilliance.research_api import open_research_run
        run = open_research_run()
        if cancellation_check is None:
            cancellation_check = run.raise_if_cancelled
        if progress_callback is None:
            def progress_callback(update):
                run.progress("Complete MBA outer-unit matching", kind="progress", event_key="mba-outer-unit-match",
                    detail="Only the first two units and their global mirror partners vary; the complete periodic cell supplies every residual.",
                    metrics=update)
    base_cell = api.periodic_cell(title="Assembled MBA before matching", energy_gev=energy_gev,
        elements=rows, intended_repetitions=periods, source="tme_repetition_outer_pairs")
    editor = api.build_jutrack_editor(base_cell)
    targets = {"Dx_m": (0., dx_tolerance_m), "Dpx": (0., dpx_tolerance)}
    if maximum_beta_m is not None:
        targets["exit_beta_excess"] = (0., 1e-3)
    attempts, best, evaluations = [], None, 0
    started = time.monotonic()

    def measure(proposed):
        nonlocal evaluations
        if cancellation_check:
            cancellation_check()
        editor.sync_rows(proposed)
        measured = editor.linear_metrics(include_profile=True)
        evaluations += 1
        if measured["stable"]:
            # Interior extrema can exceed exit samples; search with numerical headroom,
            # then enforce the ORIGINAL cap on the dense independent profile.
            measured["exit_beta_excess"] = (max(0., measured["maximum_beta_m"] - (maximum_beta_m - .01))
                                             if maximum_beta_m is not None else 0.)
        return measured

    def checkpoint(result, kind="mba_outer_unit_matching_progress"):
        if run is not None:
            from brilliance.numerical_checkpoints import write_numerical_checkpoint
            return write_numerical_checkpoint(kind, result, checks_passed=None)
        return None

    for index, position in enumerate(plan["positions"]):
        # Recover a periodic stability basin using only measured native traces.
        # This is an inequality, NOT a target tune: every interior trace is free.
        initial = measure(full_knobs.apply(position))
        if not initial.get("stable"):
            def stability_metrics(proposed):
                observed = measure(proposed)
                finite = all(math.isfinite(observed.get(key, math.nan)) for key in ("trace_x", "trace_y"))
                return {"feasible": finite, "periodic_stable": observed.get("stable", False),
                        **{key + "_violation": max(0., abs(observed[key]) - (2 - 2 * margin))
                           for key, margin in (("trace_x", trace_margin_x), ("trace_y", trace_margin_y)) if finite}}
            restored = api.bounded_response_match(rows, full_knobs, stability_metrics,
                {"trace_x_violation": (0., 1e-3), "trace_y_violation": (0., 1e-3)},
                initial_position=position, max_iterations=30, finite_difference_fraction=1e-4)
            if restored["success"]:
                position = restored["position"]
        solved = api.bounded_response_match(rows, full_knobs, measure, targets,
            initial_position=position, max_iterations=max_iterations, finite_difference_fraction=1e-4)
        metrics = solved["metrics"]
        norm = max(abs((float(metrics[key]) - target) / tol) for key, (target, tol) in targets.items()) if metrics.get("stable") else None
        attempts.append({"start": index, "response_matched": solved["success"],
                         "evaluations": solved["evaluations"], "maximum_scaled_residual": norm,
                         "reason": solved["reason"]})
        if norm is not None and (best is None or norm < best["maximum_scaled_residual"]):
            best = {"maximum_scaled_residual": norm, "periodic_match": solved}
            checkpoint({"best": best, "attempts": attempts, "knobs": full_knobs.as_dict(),
                        "remaining_starts": plan["positions"][index + 1:], "complete_ring_accepted": False})
        if progress_callback:
            progress_callback({"completed_starts": index + 1, "evaluations": evaluations,
                               "best_maximum_scaled_residual": best["maximum_scaled_residual"] if best else None,
                               "elapsed_s": time.monotonic() - started})
        if not solved["success"]:
            continue
        candidate_rows = full_knobs.apply(solved["position"])
        candidate = api.periodic_cell(title="Outer-pair matched TME-derived MBA", energy_gev=energy_gev,
            elements=candidate_rows, intended_repetitions=periods, source="tme_repetition_outer_pairs")
        sampler = api.build_jutrack_profile_sampler(candidate, max_step_m=.05)
        replay = sampler.linear_metrics()
        if maximum_beta_m is not None and replay.get("stable") and replay["maximum_beta_m"] > maximum_beta_m:
            def dense_measure(proposed):
                nonlocal evaluations
                if cancellation_check:
                    cancellation_check()
                measured = sampler.linear_metrics(proposed)
                evaluations += 1
                if measured.get("stable"):
                    measured["exit_beta_excess"] = max(0., measured["maximum_beta_m"] - maximum_beta_m + .05)
                return measured
            dense_targets = {**targets, "exit_beta_excess": (0., .01)}
            polished = api.bounded_response_match(rows, full_knobs, dense_measure, dense_targets,
                initial_position=solved["position"], max_iterations=max_iterations, finite_difference_fraction=1e-4)
            attempts[-1]["dense_polish_evaluations"] = polished["evaluations"]
            if not polished["success"]:
                attempts[-1]["reason"] = "interior_beta_polish_not_matched"
                attempts[-1]["dense_diagnostics"] = {key: value for key, value in polished["metrics"].items()
                                                     if key != "twiss_profile"}
                continue
            candidate_rows = polished["elements"]
            candidate = api.periodic_cell(title=candidate.title, energy_gev=energy_gev,
                elements=candidate_rows, intended_repetitions=periods, source="tme_repetition_outer_pairs")
            replay = sampler.linear_metrics(candidate_rows)
        evaluation = api.evaluate_periodic_cell(candidate, include_profile=True)
        optical = evaluation["linear_optics"]
        passed = (optical.get("stable") is True and evaluation.get("radiation", {}).get("physical_validity", {}).get("passed") is True
                  and abs(optical["Dx_m"]) <= dx_tolerance_m and abs(optical["Dpx"]) <= dpx_tolerance
                  and abs(optical["trace_x"]) <= 2 - trace_margin_x and abs(optical["trace_y"]) <= 2 - trace_margin_y
                  and (maximum_beta_m is None or max(max(p["beta_x"], p["beta_y"]) for p in replay["twiss_profile"]) <= maximum_beta_m))
        if not passed:
            attempts[-1]["reason"] = "complete_cell_verification_failed"
            attempts[-1]["verification"] = {"stable": optical.get("stable"), "Dx_m": optical.get("Dx_m"),
                "Dpx": optical.get("Dpx"), "maximum_beta_m": max(max(p["beta_x"], p["beta_y"]) for p in replay["twiss_profile"]),
                "radiation": evaluation.get("radiation")}
            continue
        result = {"matched": True, "elements": candidate_rows, "candidate": candidate.as_dict(),
                "evaluation": evaluation, "profile_verification": replay, "attempts": attempts,
                "evaluations": evaluations, "elapsed_s": time.monotonic() - started,
                "complete_ring_accepted": False, "knobs": full_knobs.as_dict()}
        saved = checkpoint(result, "mba_outer_unit_matched_cell")
        if saved:
            result["numerical_checkpoint"] = saved
        return result
    result = {"matched": False, "attempts": attempts, "best": best, "evaluations": evaluations,
              "elapsed_s": time.monotonic() - started, "complete_ring_accepted": False}
    saved = checkpoint(result)
    if saved:
        result["numerical_checkpoint"] = saved
    return result
