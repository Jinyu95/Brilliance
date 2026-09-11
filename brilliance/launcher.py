"""Console launcher for the chat-native research interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from .research_environment import ResearchEnvironment


def _load_project_dotenv(path: Path | str | None = None) -> Path:
    """Load Brilliance configuration before runtime or worker activation.

    Console entry points are commonly launched from a shortcut or another
    directory, so relying on python-dotenv's current-directory search is not
    deterministic. Existing process variables retain precedence over `.env`.
    """

    configured = str(os.environ.get("BRILLIANCE_ENV_FILE", "")).strip()
    dotenv_path = Path(
        path
        or configured
        or (Path(__file__).resolve().parent.parent / ".env")
    ).resolve()
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    return dotenv_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the BRILLIANCE research chat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the configured Python/Julia/JuTrack application boundary and exit",
    )
    parser.add_argument(
        "--check-optimization",
        action="store_true",
        help=(
            "run a real process-parallel JuTrack optimization with inner dispersion "
            "matching and exit"
        ),
    )
    return parser


def _activate_configured_runtime() -> ResearchEnvironment:
    """Enter the configured runtime before application modules are imported."""

    _load_project_dotenv()
    environment = ResearchEnvironment.load()
    if not environment.is_active_python():
        # Replace a global/Conda console-script process with the configured
        # runtime before importing the web application or starting workers.
        # The path comes only from research_runtime.toml, never from the model.
        os.execve(
            str(environment.python_executable),
            [
                str(environment.python_executable),
                "-m",
                "brilliance.launcher",
                *sys.argv[1:],
            ],
            environment.process_environment(),
        )
        raise RuntimeError("Could not restart Brilliance with configured Python")
    os.environ.update(environment.process_environment())
    return environment


def main() -> int:
    environment = _activate_configured_runtime()
    args = _parser().parse_args()
    from .runtime_health import ensure_runtime_ready

    try:
        print("Checking the configured JuTrack application boundary ...")
        readiness = ensure_runtime_ready(
            environment, force=args.check or args.check_optimization
        )
    except Exception as exc:
        print(f"BRILLIANCE STARTUP FAILED\n{exc}", file=sys.stderr)
        return 2
    if args.check:
        print(json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.check_optimization:
        from .optimization_readiness import probe_optimization_runtime

        try:
            optimization = probe_optimization_runtime(environment)
        except Exception as exc:
            print(f"BRILLIANCE OPTIMIZATION CHECK FAILED\n{exc}", file=sys.stderr)
            return 3
        print(json.dumps(optimization, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(
        "JuTrack application boundary ready: "
        f"Julia {readiness.get('julia_version', 'configured')} · "
        f"{readiness.get('checks', {}).get('lattice_elements_built', 0)} probe elements"
    )
    from .web_app import run_web_app

    if not args.no_browser:
        display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://{display_host}:{args.port}")
        ).start()
    run_web_app(host=args.host, port=args.port, project_root=args.project_root)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
