#!/usr/bin/env python
"""CLI entry-point for the BRILLIANCE JuTrack assistant.

Usage
-----
    # Interactive prompt
    python run_design.py

    # One-shot (quote the request)
    python run_design.py "Build a FODO ring with 20 cells, QF k1=1.5, QD k1=-1.4, cell length 5m"

Environment
-----------
Copy ``.env.example`` → ``.env`` and fill in your API key.
See ``.env.example`` for all configurable knobs.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Force UTF-8 I/O in child Python processes (avoids GBK decode errors on
# Chinese Windows when AutoGen's LocalCommandLineCodeExecutor reads stdout).
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Ensure the project root is on sys.path so ``import agents`` works
# regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()

from agents.team import run_design, run_mba_design  # noqa: E402  (after dotenv)


def main():
    parser = argparse.ArgumentParser(
        description="BRILLIANCE — AI-accelerated lattice design"
    )
    parser.add_argument(
        "task", nargs="*",
        help="Design request. If omitted, interactive prompt opens."
    )
    parser.add_argument(
        "--mba", action="store_true",
        help="Use the 10-stage MBA construction methodology "
             "(Design Contract → Decompose → UC Topology → HOA Phase → "
             "UC Optimization → Sextupole Placement → DSC → Straight → "
             "Ring Assembly → Nonlinear Check)."
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Use exploration-mode CodeWriter (no human priors)."
    )
    parser.add_argument(
        "--messages", type=int, default=None,
        help="Max messages per stage (overrides MAX_MESSAGES env var)."
    )
    args = parser.parse_args()

    # ── task from CLI args or interactive prompt ──────────────────
    if args.task:
        task = " ".join(args.task)
    else:
        print("=" * 60)
        print("  BRILLIANCE — JuTrack Assistant")
        print("=" * 60)
        print()
        print("Describe what you want to build or simulate.")
        print("Examples:")
        print('  "Build a FODO ring with 20 cells, QF k1=1.5, QD k1=-1.4, cell length 5m"')
        print('  "Design a 6 GeV 7BA storage ring with 24 superperiods"')
        print('  python run_design.py --mba "Design a 6 GeV 7BA ring..."')
        print()
        task = input("> ").strip()
        if not task:
            print("No request provided — exiting.")
            return

    # ── executor mode ─────────────────────────────────────────────
    executor_env = os.environ.get("EXECUTOR", "docker").lower()
    use_docker = executor_env == "docker"
    max_messages = args.messages or int(os.environ.get("MAX_MESSAGES", "30"))

    if use_docker:
        print("\n[executor] Docker (image: jutrack-sandbox)")
    else:
        print("\n[executor] Local Python — make sure pyJuTrack env is active")

    print(f"[model]    {os.environ.get('LLM_MODEL', 'deepseek-chat')}")
    print(f"[limit]    {max_messages} messages")
    if args.mba:
        print("[mode]     MBA 10-stage construction")
    elif args.explore:
        print("[mode]     exploration (no human priors)")
    print()

    # ── run the agent team ────────────────────────────────────────
    if args.mba:
        result = asyncio.run(
            run_mba_design(
                task,
                use_docker=use_docker,
                max_messages_per_stage=max_messages,
            )
        )
        print("\n" + "=" * 60)
        print("MBA Construction workflow complete.")
        if result:
            for key in sorted(result.keys()):
                val = result[key]
                if isinstance(val, (list, dict)):
                    continue  # skip large values in summary
                print(f"  {key}: {val}")
        print("=" * 60)
    else:
        result = asyncio.run(
            run_design(
                task,
                use_docker=use_docker,
                max_messages=max_messages,
                exploration_mode=args.explore,
            )
        )

        # ── summary ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        if result and hasattr(result, "messages"):
            last = result.messages[-1]
            content = getattr(last, "content", "")
            if "TASK_COMPLETE" in content:
                print("Task completed successfully.")
            elif "TASK_FAILED" in content:
                print("Task failed after maximum fix attempts.")
            else:
                print("Reached message limit. Review the conversation for progress.")
        print("=" * 60)


if __name__ == "__main__":
    main()
