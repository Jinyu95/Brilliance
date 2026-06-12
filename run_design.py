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

from agents.team import run_design  # noqa: E402  (after dotenv)


def main():
    # ── task from CLI args or interactive prompt ──────────────────
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("=" * 60)
        print("  BRILLIANCE — JuTrack Assistant")
        print("=" * 60)
        print()
        print("Describe what you want to build or simulate.")
        print("Examples:")
        print('  "Build a FODO ring with 20 cells, QF k1=1.5, QD k1=-1.4, cell length 5m"')
        print('  "Compute Twiss functions and plot beta functions for the ring"')
        print('  "Compute dynamic aperture for 1024 turns"')
        print()
        task = input("> ").strip()
        if not task:
            print("No request provided — exiting.")
            return

    # ── executor mode ─────────────────────────────────────────────
    executor_env = os.environ.get("EXECUTOR", "docker").lower()
    use_docker = executor_env == "docker"
    max_messages = int(os.environ.get("MAX_MESSAGES", "30"))

    if use_docker:
        print("\n[executor] Docker (image: jutrack-sandbox)")
    else:
        print("\n[executor] Local Python — make sure pyJuTrack env is active")

    print(f"[model]    {os.environ.get('LLM_MODEL', 'deepseek-chat')}")
    print(f"[limit]    {max_messages} messages\n")

    # ── run the agent team ────────────────────────────────────────
    result = asyncio.run(
        run_design(task, use_docker=use_docker, max_messages=max_messages)
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
