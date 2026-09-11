"""Offline diagnostics for scientist-selected saved cells; never a seed source.

Compare existing candidates with real JuTrack interior optics and preserve all
original files. Explicit thresholds here define the validation experiment only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from brilliance import accelerator_api as api
from brilliance.research_environment import ResearchEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-bump-m", type=float, required=True)
    parser.add_argument("--min-contrast", type=float, required=True)
    parser.add_argument("--max-core-dispersion-m", type=float, required=True)
    parser.add_argument("--max-core-h-m", type=float, required=True)
    parser.add_argument("--step-m", type=float, default=.05)
    args = parser.parse_args()
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="H-MBA shape validation")
    if environment.internet_access:
        raise ValueError("Run this validation with research internet access disabled.")
    mba = api.expert_module("laboratory.mba_design")
    args.output.mkdir(parents=True, exist_ok=True)
    thresholds = {"min_bump_m": args.min_bump_m, "min_contrast": args.min_contrast,
                  "max_core_dispersion_m": args.max_core_dispersion_m,
                  "max_core_h_m": args.max_core_h_m, "max_sample_step_m": args.step_m}
    summaries = []
    for number, path in enumerate(args.candidate):
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        candidate = api.load_lattice(path)
        rows = api.one_period_elements(candidate)
        started = time.monotonic()
        report = mba.evaluate_hybrid_mba_shape(candidate, thresholds=thresholds,
            expected_cell_length_m=sum(row["length_m"] for row in rows),
            expected_cell_angle_rad=sum(float(row.get("angle_rad") or 0) for row in rows),
            max_step_m=args.step_m)
        issues = mba.validate_hybrid_mba_shape_report(report, candidate)
        name = f"{number + 1:02d}-{candidate.lattice_id}"
        (args.output / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if report.get("profile"):
            api.save_optics_lattice_plot(args.output / f"{name}.png", report["profile"], rows,
                title=f"Measured one-cell optics — {'hybrid shape passed' if report['passed'] else 'hybrid shape NOT met'}",
                subtitle=f"{candidate.lattice_id}; {candidate.title}")
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if after != before:
            raise RuntimeError("Input lattice changed during a read-only validation")
        summary = {"candidate": str(path), "lattice_id": candidate.lattice_id,
                   "duration_s": time.monotonic() - started, "source_sha256": before,
                   "input_preserved": True, "shape_passed": report["passed"],
                   "emittance_pm": report.get("radiation", {}).get("emittance_pm"),
                   "metrics": report.get("metrics", {}), "peaks": report.get("peaks", {}),
                   "failed_checks": [key for key, passed in report["checks"].items() if not passed],
                   "validation_issues": issues, "sampling": report.get("sampling")}
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    (args.output / "comparison.json").write_text(json.dumps({"thresholds": thresholds, "candidates": summaries},
                                                           ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
