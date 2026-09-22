#!/usr/bin/env python3
"""Install Brilliance and prepare the exact Julia/JuTrack runtime it will use."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENV = ROOT / ".venv"
PYTRACK_SRC = ROOT / "JuTrack.jl" / "python_integration"
JULIA_ENV = VENV / "julia_env"
JULIA_DEPOT = Path.home() / ".julia"
# Keep the fragile Python/Julia bridge deterministic. These are the versions
# recorded in uv.lock and exercised by the setup probe and test suite.
JULIACALL_VERSION = "0.9.35"
JULIAPKG_VERSION = "0.1.26"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install Brilliance and verify its Python/Julia/JuTrack boundary"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify and repair the existing runtime without installing packages",
    )
    return parser


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(item) for item in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


def _pip(venv_python: Path, *args: str) -> None:
    _run([str(venv_python), "-m", "pip", "--quiet", *args])


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ is required; found {sys.version}")
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} OK")


def venv_python(*, create: bool) -> Path:
    executable = (
        VENV / "Scripts" / "python.exe" if sys.platform == "win32" else VENV / "bin" / "python"
    )
    if executable.is_file():
        print(f"Using existing virtual environment: {VENV}")
        return executable
    if not create:
        raise RuntimeError(f"{executable} does not exist; run `python scripts/setup_env.py` first")
    print(f"Creating virtual environment at {VENV} ...")
    _run([sys.executable, "-m", "venv", str(VENV)])
    if not executable.is_file():
        raise RuntimeError(f"virtual environment did not create {executable}")
    return executable


def check_venv_python_version(python: Path) -> None:
    result = _run(
        [
            str(python),
            "-c",
            (
                "import json, sys; "
                "print(json.dumps({'major': sys.version_info.major, "
                "'minor': sys.version_info.minor, 'executable': sys.executable}))"
            ),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        payload = json.loads(result.stdout.strip())
        version = (int(payload["major"]), int(payload["minor"]))
        executable = str(payload["executable"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"could not inspect virtual-environment Python: {result.stdout!r}"
        ) from exc
    if version < (3, 10):
        raise RuntimeError(
            "The existing .venv uses Python "
            f"{version[0]}.{version[1]}, but Brilliance requires Python 3.10+. "
            "Move that .venv aside and rerun setup with a supported Python."
        )
    print(f"Research Python {version[0]}.{version[1]} OK: {executable}")


def check_scientific_plotting(python: Path) -> None:
    """Import the supported toolbox and render with headless Matplotlib."""

    code = (
        "import io, json, numpy, scipy, pandas, sympy, h5py, joblib; "
        "import matplotlib; matplotlib.use('Agg'); "
        "import matplotlib.pyplot as plt; "
        "fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1]); "
        "buffer = io.BytesIO(); fig.savefig(buffer, format='png'); plt.close(fig); "
        "print(json.dumps({'packages': {'numpy': numpy.__version__, "
        "'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__, "
        "'pandas': pandas.__version__, 'sympy': sympy.__version__, "
        "'h5py': h5py.__version__, 'joblib': joblib.__version__}, "
        "'png_bytes': len(buffer.getvalue())}))"
    )
    result = _run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        record = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"scientific plotting probe returned invalid output: {result.stdout!r}"
        ) from exc
    if int(record.get("png_bytes", 0)) < 100:
        raise RuntimeError("Matplotlib plotting probe produced no usable PNG data")
    print(
        "Scientific Python ready: "
        + ", ".join(
            f"{name} {version}" for name, version in record["packages"].items()
        )
        + " (Matplotlib Agg)"
    )


def install_dependencies(python: Path) -> None:
    print("\nInstalling Python dependencies ...")
    _pip(python, "install", "--upgrade", "pip", "setuptools", "wheel")
    _pip(python, "install", "-e", str(ROOT))
    # Repair partial or stale Julia bridge installs as well as installing them.
    # A package directory can exist while its compiled extension or metadata is
    # missing, in which case a normal dependency check incorrectly skips it.
    _pip(
        python,
        "install",
        "--upgrade",
        "--force-reinstall",
        f"juliacall=={JULIACALL_VERSION}",
        f"juliapkg=={JULIAPKG_VERSION}",
    )
    if not (PYTRACK_SRC / "pyJuTrack.py").is_file():
        raise RuntimeError(
            f"Bundled pyJuTrack source is missing: {PYTRACK_SRC}. "
            "Clone the repository including JuTrack.jl."
        )
    print(f"\nInstalling pyJuTrack from {PYTRACK_SRC} ...")
    _pip(python, "install", "-e", str(PYTRACK_SRC))


def _runtime_environment(*, offline: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "JULIA_DEPOT_PATH": str(JULIA_DEPOT),
            "JULIA_PROJECT": str(ROOT / "JuTrack.jl"),
            "JUTRACK_JL_PROJECT": str(ROOT / "JuTrack.jl"),
            "PYTHON_JULIAPKG_PROJECT": str(JULIA_ENV),
            "PYJUTRACK_DISABLE_PYCALL": "1",
            "BRILLIANCE_JULIA_RESOLVE": "false" if offline else "true",
            "JULIA_PKG_OFFLINE": "true" if offline else "false",
            "PYTHON_JULIAPKG_OFFLINE": "yes" if offline else "no",
        }
    )
    return env


def _run_probe(python: Path, *, offline: bool) -> subprocess.CompletedProcess:
    if offline:
        command = [str(python), "-m", "brilliance.runtime_health", "--probe"]
    else:
        code = (
            "import json, pyJuTrack as jt; "
            "print(json.dumps({'status':'provisioned','pyjutrack':str(jt.__file__),"
            "'julia_version':str(jt._jl.seval('VERSION'))}))"
        )
        command = [str(python), "-c", code]
    print(
        "  $ "
        + str(python)
        + (" <offline Brilliance/JuTrack probe>" if offline else " <Julia/JuTrack provision>")
    )
    return subprocess.run(
        command,
        cwd=ROOT,
        env=_runtime_environment(offline=offline),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )


def _cache_boundary_failure(output: str) -> bool:
    normalized = output.casefold()
    return "pythoncall" in normalized and any(
        token in normalized
        for token in (
            "longpath",
            "access denied",
            "access is denied",
            "拒绝访问",
            "did not start properly",
        )
    )


def _quarantine_pythoncall_cache() -> list[Path]:
    """Move only PythonCall compiled caches aside; never delete the Julia depot."""

    compiled = JULIA_DEPOT / "compiled"
    if not compiled.is_dir():
        return []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = JULIA_DEPOT / "brilliance_cache_backups" / stamp
    moved: list[Path] = []
    depot = JULIA_DEPOT.resolve()
    for source in sorted(compiled.glob("v*/PythonCall")):
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(depot)
        except ValueError as exc:
            raise RuntimeError(f"refusing to move cache outside {depot}: {resolved}") from exc
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append(destination)
    return moved


def _require_probe(result: subprocess.CompletedProcess, *, phase: str) -> dict[str, object]:
    output = "\n".join((result.stdout or "", result.stderr or "")).strip()
    if result.returncode != 0:
        raise RuntimeError(f"{phase} failed with exit code {result.returncode}.\n\n{output}")
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("status"):
            return {str(key): item for key, item in value.items()}
    raise RuntimeError(f"{phase} returned no readiness record.\n\n{output}")


def prepare_jutrack(python: Path, *, provision: bool) -> dict[str, object]:
    JULIA_DEPOT.mkdir(parents=True, exist_ok=True)
    if provision:
        print("\nProvisioning JuliaCall and JuTrack (the first run can take several minutes) ...")
        provisioned = _run_probe(python, offline=False)
        _require_probe(provisioned, phase="Julia/JuTrack provisioning")

    print("\nChecking the exact offline runtime used by Brilliance research ...")
    checked = _run_probe(python, offline=True)
    combined = "\n".join((checked.stdout or "", checked.stderr or ""))
    backups: list[Path] = []
    if checked.returncode != 0 and _cache_boundary_failure(combined):
        backups = _quarantine_pythoncall_cache()
        print("PythonCall cache failed its Windows path check; moved it aside and retrying:")
        for path in backups:
            print(f"  {path}")
        checked = _run_probe(python, offline=True)
    record = _require_probe(checked, phase="Offline Brilliance/JuTrack readiness check")
    record.update(
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "cache_backups": os.pathsep.join(str(path) for path in backups),
        }
    )
    readiness = JULIA_ENV / "brilliance-runtime.json"
    readiness.parent.mkdir(parents=True, exist_ok=True)
    readiness.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("JuTrack runtime ready:")
    print(f"  Python:  {record.get('python', python)}")
    print(f"  Julia:   {record.get('julia')}")
    print(f"  Depot:   {record.get('depot')}")
    print(f"  JuTrack: {record.get('pyjutrack')}")
    return record


def write_env_file() -> None:
    destination = ROOT / ".env"
    source = ROOT / ".env.example"
    if destination.exists() or not source.is_file():
        return
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    print("\nCreated .env from .env.example; configure the selected model backend.")


def print_next_steps(python: Path) -> None:
    executable = (
        VENV / "Scripts" / "brilliance.exe"
        if sys.platform == "win32"
        else VENV / "bin" / "brilliance"
    )
    print(
        "\n=== Setup complete ===\n\n"
        "1. Configure the model backend in .env.\n"
        f"2. Launch Brilliance with:\n   {executable}\n\n"
        "Recheck or safely rebuild only a failed PythonCall cache with:\n"
        f"   {python} scripts/setup_env.py --check-only\n"
    )


def main() -> int:
    args = _parser().parse_args()
    os.chdir(ROOT)
    print(f"Brilliance setup - project root: {ROOT}\n")
    try:
        check_python_version()
        python = venv_python(create=not args.check_only)
        check_venv_python_version(python)
        if not args.check_only:
            install_dependencies(python)
            write_env_file()
        check_scientific_plotting(python)
        prepare_jutrack(python, provision=not args.check_only)
        print_next_steps(python)
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"\nSETUP FAILED\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
