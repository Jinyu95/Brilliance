#!/usr/bin/env python3
"""
BRILLIANCE environment setup — run once after cloning.

What this does:
  1. Creates .venv/ using the current Python (>=3.10 required)
  2. Installs all Python dependencies
  3. Installs pyJuTrack from the bundled JuTrack.jl/python_integration/ source
  4. Checks that Julia >=1.10 is available (or tells you where to get it)
  5. Writes .env from .env.example if no .env exists yet
  6. Prints the exact command to launch the app

Usage:
  python scripts/setup_env.py
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # scripts/
ROOT = HERE.parent                           # project root
VENV = ROOT / ".venv"
PYTRACK_SRC = ROOT / "JuTrack.jl" / "python_integration"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\nERROR: command failed (exit {result.returncode})")
        sys.exit(result.returncode)
    return result


def _pip(venv_python: Path, *args: str) -> None:
    _run([str(venv_python), "-m", "pip", "--quiet", *args])


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ required; you have {sys.version}")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} OK")


def create_venv() -> Path:
    if VENV.exists():
        print(f".venv already exists at {VENV}")
    else:
        print(f"Creating virtual environment at {VENV} ...")
        _run([sys.executable, "-m", "venv", str(VENV)])
    # Return path to the venv Python
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def install_dependencies(venv_python: Path) -> None:
    print("\nInstalling Python dependencies ...")
    _pip(venv_python, "install", "--upgrade", "pip", "setuptools", "wheel")

    # Install the project itself (pulls in autogen, streamlit, juliacall, etc.)
    _pip(venv_python, "install", "-e", str(ROOT))

    # Install pyJuTrack from the bundled source (not on PyPI yet)
    if PYTRACK_SRC.exists():
        print(f"\nInstalling pyJuTrack from {PYTRACK_SRC} ...")
        _pip(venv_python, "install", "-e", str(PYTRACK_SRC))
    else:
        print(
            f"\nWARNING: {PYTRACK_SRC} not found. "
            "Make sure JuTrack.jl/ is present in the project root."
        )


def check_julia() -> None:
    print("\nChecking Julia installation ...")
    result = subprocess.run(
        ["julia", "--version"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(
            "\nWARNING: 'julia' not found on PATH.\n"
            "juliacall can download Julia automatically on first import, OR\n"
            "install it yourself from https://julialang.org/downloads/\n"
            "Recommended: Julia 1.10 LTS"
        )
        return

    ver_line = result.stdout.strip()
    print(f"Found {ver_line}")

    # Sanity-check version >= 1.10
    try:
        parts = ver_line.split()[2].split(".")
        major, minor = int(parts[0]), int(parts[1])
        if (major, minor) < (1, 10):
            print(
                f"WARNING: Julia {major}.{minor} is older than the required 1.10. "
                "Consider upgrading."
            )
    except (IndexError, ValueError):
        pass


def write_env_file() -> None:
    env_file = ROOT / ".env"
    example = ROOT / ".env.example"
    if env_file.exists():
        print("\n.env already exists — skipping.")
        return
    if not example.exists():
        return
    env_file.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nCreated .env from .env.example — edit it to set LLM_API_KEY.")


def print_next_steps(venv_python: Path) -> None:
    if sys.platform == "win32":
        activate = str(VENV / "Scripts" / "activate.bat")
        run_cmd = f"{VENV / 'Scripts' / 'streamlit'} run app.py"
    else:
        activate = f"source {VENV / 'bin' / 'activate'}"
        run_cmd = f"{VENV / 'bin' / 'streamlit'} run app.py"

    print(
        f"""
=== Setup complete! ===

Next steps:
  1. Edit .env and set your LLM_API_KEY (and LLM_MODEL / LLM_BASE_URL if needed).

  2. On first launch Julia will install JuTrack dependencies — this takes 1-3 min.
     Subsequent launches are instant (sysimage cached in julia_depot/).

  3. Launch the app:
       {activate}
       {run_cmd}

     Or with Docker (no Julia installation required on the host):
       docker compose up

Tip: to pre-warm the Julia sysimage (eliminates the 1-3 min wait on first run):
  {str(venv_python)} -c "import pyJuTrack"
"""
    )


def main() -> None:
    os.chdir(ROOT)
    print(f"BRILLIANCE setup — project root: {ROOT}\n")

    check_python_version()
    venv_python = create_venv()
    install_dependencies(venv_python)
    check_julia()
    write_env_file()
    print_next_steps(venv_python)


if __name__ == "__main__":
    main()
