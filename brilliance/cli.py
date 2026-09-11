"""Private worker entry point used by durable Brilliance jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifacts import json_safe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m brilliance")
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("project-worker", help=argparse.SUPPRESS)
    worker.add_argument("project_id")
    worker.add_argument("job_id")
    worker.add_argument("--root", type=Path, default=Path("research_projects"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "project-worker":
        raise ValueError(f"Unknown command: {args.command}")
    from .jobs import JobState, run_research_job

    job = run_research_job(args.root, args.project_id, args.job_id)
    print(json.dumps(json_safe(job), ensure_ascii=True, indent=2, sort_keys=True))
    return 2 if job.state is JobState.FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
