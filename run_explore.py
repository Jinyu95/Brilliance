#!/usr/bin/env python
"""BRILLIANCE discovery harness — unified CLI for PRL experiments.

Usage (all modes work with chat OR batch execution):

  # ── Mode 1: Generate prompts to copy-paste into Streamlit chat ──
  python run_explore.py prompts candidate1 --energy 3,6 --cells 16,20,24
  python run_explore.py prompts candidate2 --energy 1,3,6 --cells 12,20,32

  # ── Mode 2: Single interactive run (debug/test one design) ──
  python run_explore.py single "Design a 6 GeV 7BA ring with 24 cells..."
  python run_explore.py single --explore "Design a 6 GeV 7BA ring..."

  # ── Mode 3: Batch sweep (run overnight, resumes on restart) ──
  python run_explore.py batch candidate1 --energy 2,3,4,5,6,7,8 --cells 16,20,24,28,32,36 --runs 5
  python run_explore.py batch candidate3 --family 7BA --energy 6 --cells 8,12,16,20,24,28,32,36,40,48

  # ── Mode 4: Ingest saved Streamlit sessions into analysis format ──
  python run_explore.py ingest sessions/
  python run_explore.py ingest sessions/20260624_005526_Design/  (single session)

  # ── Mode 5: Analyze results and produce figures ──
  python run_explore.py analyze science/results/20260624_120000/results.jsonl --figures figures/
  python run_explore.py analyze results.jsonl --anomalies

  # ── Mode 6: Dry-run (print prompts, no API calls) ──
  python run_explore.py batch candidate1 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv
load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BRILLIANCE discovery harness — PRL experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── prompts ───────────────────────────────────────────────────────────
    p_prompts = sub.add_parser("prompts", help="Generate prompts for copy-paste into chat")
    p_prompts.add_argument("experiment", choices=["candidate1", "candidate2", "candidate3", "candidate5"],
                           help="Which PRL experiment to generate prompts for")
    p_prompts.add_argument("--energy", type=_parse_float_list, help="Comma-separated energies in GeV")
    p_prompts.add_argument("--cells", type=_parse_int_list, help="Comma-separated cell counts")
    p_prompts.add_argument("--family", type=str, help="Single lattice family (for candidate3)")

    # ── single ─────────────────────────────────────────────────────────────
    p_single = sub.add_parser("single", help="Run one prompt interactively (debug)")
    p_single.add_argument("prompt", nargs="+", help="The design prompt (quote it)")
    p_single.add_argument("--explore", action="store_true", default=False,
                          help="Use exploration mode (no human priors)")

    # ── batch ──────────────────────────────────────────────────────────────
    p_batch = sub.add_parser("batch", help="Run a full parameter sweep")
    p_batch.add_argument("experiment", choices=["candidate1", "candidate2", "candidate3", "candidate5"],
                         help="Which PRL experiment to run")
    p_batch.add_argument("--energy", type=_parse_float_list, help="Energies in GeV (comma-sep)")
    p_batch.add_argument("--cells", type=_parse_int_list, help="Cell counts (comma-sep)")
    p_batch.add_argument("--family", type=str, help="Single family (candidate3)")
    p_batch.add_argument("--runs", type=int, default=3, help="Repeat count per grid point")
    p_batch.add_argument("--max-messages", type=int, default=30, help="Message budget per run")
    p_batch.add_argument("--dry-run", action="store_true", default=False, help="Print prompts without running")
    p_batch.add_argument("--no-resume", action="store_true", default=False,
                         help="Don't skip existing results (re-run everything)")
    p_batch.add_argument("--output-dir", type=str, default="", help="Override output directory")

    # ── ingest ─────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Convert saved chat sessions to analysis format")
    p_ingest.add_argument("sessions_path", type=str, help="Path to sessions/ dir or a single session dir")
    p_ingest.add_argument("--output", type=str, default="", help="Output JSONL path (default: results.jsonl)")

    # ── analyze ────────────────────────────────────────────────────────────
    p_analyze = sub.add_parser("analyze", help="Analyze results and produce figures")
    p_analyze.add_argument("results_path", type=str, help="Path to results.jsonl or a directory of JSONL files")
    p_analyze.add_argument("--figures", type=str, default="", help="Output directory for figures")
    p_analyze.add_argument("--anomalies", action="store_true", default=False, help="Print anomaly detection")
    p_analyze.add_argument("--summary", action="store_true", default=False, help="Print per-family summary table")
    p_analyze.add_argument("--pareto", action="store_true", default=False, help="Print Pareto front")
    return parser


# ══════════════════════════════════════════════════════════════════════════════
# Command handlers
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prompts(args):
    """Print prompts for copy-paste into Streamlit chat."""
    explorer = _build_explorer(args)
    prompts = explorer.prompts_only()

    print(f"# {len(prompts)} prompts for {args.experiment}")
    print(f"# Copy each one into the Streamlit chat input.\n")
    for i, p in enumerate(prompts):
        print(f"{'─' * 60}")
        print(f"# [{i+1}/{len(prompts)}]  {p['family']}  {p['energy_gev']:.0f} GeV  {p['n_cells']} cells")
        print(f"# Analytical epsilon_0 ≈ {p['analytical_emittance_pm']:.1f} pm-rad")
        print(f"{'─' * 60}")
        print(p["prompt"])
        print()
    print(f"{'─' * 60}")
    print(f"# Total: {len(prompts)} prompts")
    print(f"# After running in chat, ingest results with:")
    print(f"#   python run_explore.py ingest sessions/")


def cmd_single(args):
    """Run one prompt through the full pipeline."""
    prompt = " ".join(args.prompt)
    exploration_mode = args.explore

    print(f"{'='*60}")
    print(f"Running single design{' (exploration mode)' if exploration_mode else ''}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"{'='*60}")

    from agents.team import run_design
    import asyncio

    result = asyncio.run(
        run_design(prompt, use_docker=False, max_messages=28,
                    exploration_mode=exploration_mode)
    )

    messages = list(getattr(result, "messages", []))
    print(f"\n{'='*60}")
    print(f"Pipeline complete. {len(messages)} messages.")

    # Show the last message content
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "")
        source = getattr(last, "source", "unknown")
        print(f"\n[Final: {source}]")
        print(content[:3000])


def cmd_batch(args):
    """Run a full parameter sweep."""
    explorer = _build_explorer(args)
    explorer.config.n_runs_per_point = args.runs
    explorer.config.max_messages = args.max_messages
    explorer.config.dry_run = args.dry_run
    explorer.config.resume = not args.no_resume

    if args.output_dir:
        explorer.config.output_dir = Path(args.output_dir)

    if args.dry_run:
        print(f"[DRY RUN] {len(explorer.points)} grid points x {args.runs} repeats = "
              f"{len(explorer.points) * args.runs} total")
        print(f"[DRY RUN] Output would go to: {explorer.config.output_dir}\n")
        prompts = explorer.prompts_only()
        for p in prompts[:5]:
            print(f"  [{p['family']} {p['energy_gev']:.0f}GeV {p['n_cells']}c] "
                  f"{p['prompt'][:80]}...")
        if len(prompts) > 5:
            print(f"  ... and {len(prompts) - 5} more")
        return

    print(f"Output directory: {explorer.config.output_dir}")
    print(f"Grid: {len(explorer.points)} points x {args.runs} repeats = "
          f"{len(explorer.points) * args.runs} total runs")
    print(f"Resume: {explorer.config.resume}  "
          f"Max messages: {explorer.config.max_messages}")
    print()

    summary = explorer.run()
    print(f"\n{'='*60}")
    print(f"Sweep complete!")
    print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
    print(f"  Stable rate:  {summary.get('stable_rate', 0):.1%}")
    print(f"  Output:       {summary.get('output_dir')}")
    print(f"\nAnalyze with:")
    print(f"  python run_explore.py analyze {summary.get('output_dir')}/results.jsonl --figures figures/")


def cmd_ingest(args):
    """Convert saved sessions to analysis JSONL."""
    import json
    from datetime import datetime

    sessions_path = Path(args.sessions_path)
    output_path = Path(args.output or "ingested_results.jsonl")

    if not sessions_path.exists():
        print(f"Error: path '{sessions_path}' does not exist.")
        return 1

    # Determine whether this is a single session or the sessions/ directory
    conv_files: list[Path] = []
    if (sessions_path / "conversation.json").exists():
        conv_files = [sessions_path / "conversation.json"]
        session_dirs = [sessions_path]
    else:
        conv_files = sorted(sessions_path.glob("*/conversation.json"))
        session_dirs = [cf.parent for cf in conv_files]

    if not conv_files:
        print(f"No conversation.json files found in {sessions_path}")
        return 1

    print(f"Found {len(conv_files)} session(s)")

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for conv_file, sess_dir in zip(conv_files, session_dirs):
            try:
                conv = json.loads(conv_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  SKIP {sess_dir.name}: {exc}")
                continue

            # Extract the user request
            user_request = ""
            for msg in conv:
                if msg.get("source") == "user" and isinstance(msg.get("content"), str):
                    user_request = msg["content"].strip()
                    break

            # Extract the final structured result from any message
            from physics_core import parse_result_output
            parsed_result = None
            for msg in reversed(conv):
                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue
                parsed = parse_result_output(content)
                if parsed:
                    parsed_result = parsed
                    break

            if parsed_result is None:
                print(f"  SKIP {sess_dir.name}: no structured result found")
                continue

            # Extract code blocks as evidence
            import re
            code_blocks = []
            for msg in conv:
                if msg.get("source") == "CodeWriter":
                    for match in re.finditer(r"```python\s*\n(.*?)```", msg.get("content", ""), re.DOTALL):
                        code_blocks.append(match.group(1)[:500])

            # Convert to exploration format
            row = {
                "label": sess_dir.name,
                "family": "from_chat",
                "energy_gev": parsed_result.get("energy_gev"),
                "n_cells": parsed_result.get("n_cells", 0),
                "n_bends_per_cell": parsed_result.get("n_bends_per_cell", 0),
                "analytical_emittance_pm": None,
                "prompt": user_request[:500],
                "success": True,
                "error": None,
                "elapsed_s": 0,
                "message_count": len(conv),
                "result": parsed_result,
                "score": 0,
                "status": "from_session",
                "brightness_A_per_m_rad": None,
                "F_empirical": None,
                "timestamp": datetime.now().isoformat(),
                "source_path": str(sess_dir),
                "code_preview": code_blocks[-1][:200] if code_blocks else "",
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1
            print(f"  OK {sess_dir.name}")

    print(f"\nIngested {rows_written} sessions → {output_path}")
    print(f"Analyze with:")
    print(f"  python run_explore.py analyze {output_path} --figures figures/")


def cmd_analyze(args):
    """Analyze results and produce figures."""
    from science.analyze import (
        ResultDatabase, detect_anomalies, compare_to_reference, summarize_by_family,
    )

    results_path = Path(args.results_path)
    if not results_path.exists():
        # Try as a directory
        jsonl_files = list(Path(args.results_path).glob("*.jsonl")) if results_path.is_dir() else []
        if jsonl_files:
            results_path = jsonl_files[0]
        else:
            print(f"Error: {args.results_path} not found")
            return 1

    db = ResultDatabase(results_path)
    stats = db.stats()
    family_stats = db.family_stats()

    print(f"{'='*60}")
    print(f"Analysis of: {results_path.name}")
    print(f"{'='*60}")
    print(f"\nTotal runs:      {stats['n_total']}")
    print(f"Successful:      {stats['n_successful']} ({stats['success_rate']:.1%})")
    print(f"Stable designs:  {stats['n_stable']} ({stats['stable_rate']:.1%})")
    print(f"Mean score:      {stats.get('score_mean', 0):.1f}")
    if stats.get("emittance_min_pm"):
        print(f"Emittance range: {stats['emittance_min_pm']:.1f} – {stats['emittance_max_pm']:.1f} pm-rad "
              f"(median {stats['emittance_median_pm']:.1f})")
    if stats.get("F_empirical_mean"):
        print(f"Mean F_empirical: {stats['F_empirical_mean']:.5f}")

    print(f"\nFamilies found:")
    for fam, fstat in sorted(family_stats.items()):
        print(f"  {fam:6s}: {fstat['n_runs']:3d} runs, {fstat['n_stable']:3d} stable "
              f"({fstat['stable_rate']:.0%}), "
              f"emit_min={fstat.get('emittance_min_pm','?'):.0f} pm-rad")

    # ── Anomalies ─────────────────────────────────────────────────────────
    if args.anomalies:
        print(f"\n{'─'*60}")
        print("Anomaly Detection")
        print(f"{'─'*60}")
        anomalies = detect_anomalies(db, z_threshold=2.0)
        if anomalies:
            for a in anomalies:
                print(f"\n  {a['type'].upper()}: {a['label']}")
                print(f"    emit/floor = {a['emittance_ratio']:.1f}x ({a['z_score']:+.1f} sigma)")
                print(f"    {a['note']}")
        else:
            print("  No anomalies detected (all designs within 2 sigma of theoretical expectations)")

    # ── Summary table ─────────────────────────────────────────────────────
    if args.summary:
        print(f"\n{'─'*60}")
        print("Per-Family Summary")
        print(f"{'─'*60}")
        summary_rows = summarize_by_family(db)
        if summary_rows:
            header = f"{'Family':6s} {'E(GeV)':>6s} {'N':>4s} {'emit_min':>10s} {'emit_mean':>10s} {'emit_std':>10s} {'C(m)':>8s}"
            print(header)
            print("-" * len(header))
            for r in summary_rows:
                print(
                    f"{r['family']:6s} {r['energy_gev']:6.1f} {r['n_runs_stable']:4d} "
                    f"{r.get('emittance_min_pm','n/a'):>10.1f} "
                    f"{r.get('emittance_mean_pm','n/a'):>10.1f} "
                    f"{r.get('emittance_std_pm','n/a'):>10.1f} "
                    f"{r.get('circumference_mean_m','n/a'):>8.0f}"
                )
        else:
            print("  No stable designs available for summary")

    # ── Pareto front ──────────────────────────────────────────────────────
    if args.pareto:
        print(f"\n{'─'*60}")
        print("Pareto Frontier (emittance vs. circumference)")
        print(f"{'─'*60}")
        from physics_core import compute_pareto_front
        front = compute_pareto_front(db.rows, ["emittance", "circumference"])
        if front:
            for r in front:
                emit = r.get("result", {}).get("emittance", 0) * 1e12
                circ = r.get("result", {}).get("circumference", 0)
                fam = r.get("family", "?")
                n_c = r.get("n_cells", "?")
                print(f"  {fam:6s}  {n_c:>3s}c  emit={emit:.1f} pm-rad  C={circ:.0f} m")
        else:
            print("  No stable designs for Pareto analysis")

    # ── Figures ───────────────────────────────────────────────────────────
    if args.figures:
        figures_dir = Path(args.figures)
        figures_dir.mkdir(parents=True, exist_ok=True)

        from science.plots import (
            plot_pareto_frontier, plot_scaling_law,
            plot_emittance_vs_energy, plot_dipole_grading_pattern,
            plot_family_comparison, save_figure,
        )

        print(f"\n{'─'*60}")
        print(f"Generating figures → {figures_dir}/")
        print(f"{'─'*60}")

        # Figure 1: Pareto frontier
        try:
            fig1 = plot_pareto_frontier(db.rows)
            save_figure(fig1, figures_dir / "fig1_pareto.png")
            print("  fig1_pareto.png")
        except Exception as exc:
            print(f"  fig1_pareto.png FAILED: {exc}")

        # Figure 2: Scaling law
        try:
            fig2 = plot_scaling_law(db.rows)
            save_figure(fig2, figures_dir / "fig2_scaling.png")
            print("  fig2_scaling.png")
        except Exception as exc:
            print(f"  fig2_scaling.png FAILED: {exc}")

        # Figure 3: Emittance vs energy
        try:
            fig3 = plot_emittance_vs_energy(db.rows)
            save_figure(fig3, figures_dir / "fig3_emittance_vs_energy.png")
            print("  fig3_emittance_vs_energy.png")
        except Exception as exc:
            print(f"  fig3_emittance_vs_energy.png FAILED: {exc}")

        # Figure 4: Dipole grading (Candidate 1 specific)
        try:
            fig4 = plot_dipole_grading_pattern(db.rows)
            save_figure(fig4, figures_dir / "fig4_dipole_grading.png")
            print("  fig4_dipole_grading.png")
        except Exception as exc:
            print(f"  fig4_dipole_grading.png FAILED: {exc}")

        # Figure S1: Family comparison
        try:
            summary_rows = summarize_by_family(db)
            if summary_rows:
                figS1 = plot_family_comparison(summary_rows)
                save_figure(figS1, figures_dir / "figS1_families.png")
                print("  figS1_families.png")
        except Exception as exc:
            print(f"  figS1_families.png FAILED: {exc}")

        # Write anomaly report
        try:
            anomalies = detect_anomalies(db, z_threshold=2.0)
            if anomalies:
                import json
                (figures_dir / "anomalies.json").write_text(
                    json.dumps(anomalies, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print("  anomalies.json")
        except Exception as exc:
            print(f"  anomalies.json FAILED: {exc}")


def _build_explorer(args):
    """Build the appropriate DesignSpaceExplorer from CLI arguments."""
    from science.explore import (
        candidate1_dipole_grading_sweep,
        candidate2_nbend_sweep,
        candidate3_scaling_sweep,
        candidate5_asymmetry_sweep,
    )

    energy = getattr(args, "energy", None)
    cells = getattr(args, "cells", None)
    family = getattr(args, "family", None)

    if args.experiment == "candidate1":
        return candidate1_dipole_grading_sweep(
            energies_gev=energy, n_cells_values=cells,
        )
    elif args.experiment == "candidate2":
        return candidate2_nbend_sweep(
            energies_gev=energy, n_cells_values=cells,
        )
    elif args.experiment == "candidate3":
        return candidate3_scaling_sweep(
            family=family or "7BA",
            energy_gev=float(energy[0]) if energy else 6.0,
        )
    elif args.experiment == "candidate5":
        return candidate5_asymmetry_sweep(
            energies_gev=energy,
        )
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "prompts": cmd_prompts,
        "single": cmd_single,
        "batch": cmd_batch,
        "ingest": cmd_ingest,
        "analyze": cmd_analyze,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    result = handler(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
