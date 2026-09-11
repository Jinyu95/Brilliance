"""Offline, bounded H-MBA response/continuation experiment on a selected lattice.

This script is a diagnostic, not a lattice template or a production protocol.
It never changes a research project or its working candidate. All physical
quadrupole pairs are obtained from their mirrored positions, not unit names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from brilliance import accelerator_api as api
from brilliance.research_environment import ResearchEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluations", type=int, default=0)
    parser.add_argument("--comparison-candidate", type=Path)
    args = parser.parse_args()
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="offline hybrid conversion validation")
    if environment.internet_access:
        raise ValueError("This validation requires internet research to be disabled")
    before = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    candidate = api.load_lattice(args.candidate)
    original = api.one_period_elements(candidate)
    mba = api.expert_module("laboratory.mba_design")
    roles = mba.hybrid_mba_bend_roles(original)
    thresholds = {"min_bump_m": 0.035, "min_contrast": 1.5,
                  "max_core_dispersion_m": 0.025, "max_core_h_m": 0.00012,
                  "max_sample_step_m": 0.05}
    pairs = [(i, len(original) - 1 - i) for i in range((len(original) + 1) // 2)
             if original[i]["kind"] in {"QUAD", "KQUAD"}]
    for left, right in pairs:
        if original[right]["kind"] != original[left]["kind"]:
            raise ValueError("Source topology is not mirror symmetric")
        if abs(original[left]["k1_m2"] - original[right]["k1_m2"]) > 1e-9:
            raise ValueError("Source quadrupole strengths are not mirror symmetric")
    initial = np.array([original[left]["k1_m2"] for left, _ in pairs])
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    limits = api.magnet_strength_limits(candidate.energy_gev)
    k1max = limits["max_abs_k1_m2"]
    evaluator = mba.HybridMBAShapeEvaluator(candidate, thresholds=thresholds,
        expected_cell_length_m=roles["cell_length_m"],
        expected_cell_angle_rad=sum(float(row.get("angle_rad") or 0) for row in original), limits=limits)
    sampler = evaluator.sampler

    def rows_at(position):
        rows = deepcopy(original)
        for value, (left, right) in zip(position, pairs):
            rows[left]["k1_m2"] = rows[right]["k1_m2"] = float(value)
        return rows

    def measure(position):
        rows = rows_at(position)
        linear = sampler.linear_metrics(rows)
        if not linear["stable"]:
            return None, linear
        report = mba.hybrid_mba_shape_diagnostics(rows, linear["twiss_profile"],
            thresholds=thresholds, sampling=linear["sampling"])
        return report, linear

    baseline, baseline_linear = measure(initial)
    if baseline is None:
        raise ValueError("Selected baseline is not linearly stable")
    print(json.dumps({"phase": "baseline", "elapsed_s": time.monotonic() - started,
                      "metrics": baseline["metrics"]}), flush=True)
    (args.output / "baseline_shape.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    api.save_optics_lattice_plot(args.output / "baseline_optics.png", baseline["profile"], original,
        title="Selected source: measured one-cell optics", subtitle="Read-only source for the bounded shaping experiment")
    helper_baseline = evaluator.measure(include_radiation=True)
    all_knobs = mba.hybrid_mba_quadrupole_knobs(original, max_abs_k1_m2=k1max)
    matching_indices = [index for index in roles["regions"]["left_bump"]["element_indices"]
                        if original[index]["kind"] == "KQUAD"][:2]
    matching_names = {original[index]["name"] for index in matching_indices}
    matching_specs = [knob for knob in all_knobs if matching_names.intersection(knob["element_names"])]
    outer_specs = [knob for knob in all_knobs if not matching_names.intersection(knob["element_names"])]
    matcher = api.compile_lattice_parameterization(original, matching_specs)

    def inner_match(proposed_rows):
        return api.bounded_fast_response_match(proposed_rows, matcher, evaluator.linear_metrics,
            {"Dx_m": (0., 1e-7), "Dpx": (0., 1e-7)}, max_evaluations=24,
            finite_difference_fraction=1e-4)

    helper_started = time.monotonic()
    matched_response = evaluator.response(outer_specs, inner_match=inner_match)
    (args.output / "helper_matched_response.json").write_text(json.dumps(matched_response, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "reusable_helper", "stable": helper_baseline["stable"],
        "feasible": helper_baseline["feasible"], "baseline_emittance_pm": helper_baseline["emittance_pm"],
        "matched_response_success": matched_response["success"],
        "matched_response_columns": len(matched_response["columns"]),
        "matching_knobs": [knob["name"] for knob in matching_specs],
        "duration_s": time.monotonic() - helper_started}), flush=True)
    response = []
    step = 1e-3
    base = baseline["metrics"]
    observable_names = ["D_bump_L_m", "D_bump_R_m", "D_core_m", "H_core_max_m",
                        "dispersion_contrast", "ID_Dx_max_abs_m"]
    for index, (left, right) in enumerate(pairs):
        position = initial.copy()
        position[index] += step
        report, linear = measure(position)
        response.append({"pair": [original[left]["name"], original[right]["name"]],
                         "indices": [left, right], "stable": linear["stable"],
                         "derivative": ({name: (report["metrics"][name] - base[name]) / step
                                         for name in observable_names} if report else None),
                         "signed_ID_Dx": linear.get("Dx_m")})
    (args.output / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "symmetric_response", "pairs": len(pairs),
                      "elapsed_s": time.monotonic() - started}), flush=True)
    final_position = initial.copy()
    history = []
    if args.evaluations:
        # Joint local feasibility/shaping continuation is used here to establish
        # numerical controllability. Production nested optimizers must allocate
        # separate owners to outer structural and inner matching coordinates.
        for target_contrast in (1.1, 1.25, 1.5):
            anchor = final_position.copy()
            count = 0
            best = {"score": float("inf"), "position": anchor.copy()}

            def residual(position):
                nonlocal count
                count += 1
                report, linear = measure(position)
                if report is None:
                    instability = max(abs(linear.get("trace_x", 10)) - 1.999, 0)
                    instability += max(abs(linear.get("trace_y", 10)) - 1.999, 0)
                    return np.r_[np.ones(8) * (20 + instability), (position - anchor) * .001]
                m = report["metrics"]
                mean_bump = (m["D_bump_L_m"] + m["D_bump_R_m"]) / 2
                r = np.array([
                    linear["Dx_m"] / 5e-5,
                    linear["Dpx"] / 5e-5,
                    max(0, (target_contrast + .025) * m["D_core_m"] - m["D_bump_L_m"]) / .01,
                    max(0, (target_contrast + .025) * m["D_core_m"] - m["D_bump_R_m"]) / .01,
                    max(0, .0355 - mean_bump) / .01,
                    max(0, m["D_core_m"] - .025) / .01,
                    max(0, m["H_core_max_m"] - .000114) / .00012,
                    max(0, linear.get("maximum_beta_m", 0) - 150) / 150,
                ])
                score = float(r @ r)
                if score < best["score"]:
                    best.update(score=score, position=position.copy())
                return np.r_[r, (position - anchor) * .001]

            optimized = least_squares(residual, anchor, bounds=(-k1max, k1max),
                max_nfev=args.evaluations, diff_step=1e-5, xtol=1e-8, ftol=1e-7, gtol=1e-7,
                x_scale="jac")
            final_position = best["position"]
            report, linear = measure(final_position)
            item = {"target_contrast": target_contrast, "callback_evaluations": count,
                    "solver_nfev": optimized.nfev, "message": optimized.message,
                    "duration_s": time.monotonic() - started, "metrics": report["metrics"],
                    "position": final_position.tolist(), "checks": report["checks"]}
            history.append(item)
            print(json.dumps({"phase": "continuation", **item}), flush=True)
            (args.output / "continuation.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    final_rows = rows_at(final_position)
    final = api.cell_ring(title="Offline symmetric hybrid-shaping experiment", energy_gev=candidate.energy_gev,
        cell=final_rows, periods=candidate.periodicity, source="offline_validation",
        parent_lattice_id=candidate.lattice_id)
    report = mba.evaluate_hybrid_mba_shape(final, thresholds=thresholds,
        expected_cell_length_m=roles["cell_length_m"],
        expected_cell_angle_rad=sum(float(row.get("angle_rad") or 0) for row in original))
    (args.output / "candidate.json").write_text(json.dumps(final.as_dict(), indent=2), encoding="utf-8")
    (args.output / "final_shape.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    requested_thresholds = {"min_bump_m": .02, "min_contrast": 2.,
        "max_core_dispersion_m": .02, "max_core_h_m": 1e-4, "max_sample_step_m": .05}
    requested_shape = mba.hybrid_mba_shape_diagnostics(final_rows, report["profile"],
        thresholds=requested_thresholds, sampling=report["sampling"])
    (args.output / "latest_run_thresholds.json").write_text(json.dumps(requested_shape, indent=2), encoding="utf-8")
    api.save_optics_lattice_plot(args.output / "final_optics.png", report["profile"], final_rows,
        title="Offline H-MBA shaping diagnostic",
        subtitle=f"Provisional shape gate: {'passed' if report['passed'] else 'failed'}; original stricter request: {'passed' if requested_shape['passed'] else 'failed'}")
    summary = {"elapsed_s": time.monotonic() - started, "metrics": report["metrics"],
               "passed": report["passed"], "checks": report["checks"],
               "emittance_pm": report["radiation"].get("emittance_pm"),
               "threshold_scope": "Provisional diagnostic experiment, not the scientist's adopted contract",
               "latest_run_thresholds_passed": requested_shape["passed"],
               "latest_run_thresholds_failed_checks": [name for name, passed in requested_shape["checks"].items() if not passed],
               "pairs": [[original[i]["name"], original[j]["name"]] for i, j in pairs],
               "source_preserved": hashlib.sha256(args.candidate.read_bytes()).hexdigest() == before}
    print(json.dumps({"phase": "final", **summary}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.comparison_candidate:
        comparison_hash = hashlib.sha256(args.comparison_candidate.read_bytes()).hexdigest()
        compared = api.load_lattice(args.comparison_candidate)
        compared_rows = api.one_period_elements(compared)
        compared_report = mba.evaluate_hybrid_mba_shape(compared, thresholds=thresholds,
            expected_cell_length_m=roles["cell_length_m"],
            expected_cell_angle_rad=sum(float(row.get("angle_rad") or 0) for row in original))
        compared_strict = mba.hybrid_mba_shape_diagnostics(compared_rows, compared_report["profile"],
            thresholds=requested_thresholds, sampling=compared_report["sampling"])
        compared_report["strict_request_shape_gate"] = {"thresholds": requested_thresholds,
            "passed": compared_strict["passed"], "checks": compared_strict["checks"]}
        api.save_optics_lattice_plot(args.output / "comparison_optics.png", compared_report["profile"], compared_rows,
            title="Bounded symmetric hybrid-shaping experiment",
            subtitle=f"Provisional shape gate: {'passed' if compared_report['passed'] else 'failed'}; original stricter request: {'passed' if compared_strict['passed'] else 'failed'}")
        (args.output / "comparison_shape.json").write_text(json.dumps(compared_report, indent=2), encoding="utf-8")
        comparison = []
        for label, evidence, emittance in (("original conventional", baseline, helper_baseline["emittance_pm"]),
                                          ("bounded shape experiment", compared_report, compared_report["radiation"]["emittance_pm"])):
            comparison.append({"candidate": label, "metrics": evidence["metrics"], "emittance_pm": emittance,
                "maximum_beta_m": max(max(row["beta_x"], row["beta_y"]) for row in evidence["profile"]),
                "provisional_gate_passed": evidence["passed"]})
        (args.output / "comparison_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        if hashlib.sha256(args.comparison_candidate.read_bytes()).hexdigest() != comparison_hash:
            raise RuntimeError("Comparison source unexpectedly changed")
        print(json.dumps({"phase": "comparison", "rows": comparison,
            "strict_request_passed": compared_strict["passed"], "source_preserved": True}), flush=True)


if __name__ == "__main__":
    main()
