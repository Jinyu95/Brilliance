#!/usr/bin/env python
"""CLI entry point for the lattice-design benchmark suite."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()

from agents.benchmark import DEFAULT_TASK_FILE, run_benchmarks  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lattice-design benchmarks")
    parser.add_argument(
        "--tasks",
        default=str(DEFAULT_TASK_FILE),
        help="Path to benchmark task JSON",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the JSON summary",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=int(os.environ.get("MAX_MESSAGES", "30")),
        help="Fallback message limit for tasks that do not specify one",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    use_docker = os.environ.get("EXECUTOR", "docker").lower() == "docker"
    summary = asyncio.run(
        run_benchmarks(
            task_file=Path(args.tasks),
            use_docker=use_docker,
            default_max_messages=args.max_messages,
        )
    )

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.output:
        Path(args.output).write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    return 0 if summary.get("passed_count", 0) == summary.get("task_count", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())