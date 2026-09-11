"""Deterministic Julia/JuTrack runtime configuration."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, MutableMapping

_JULIACALL_DISCOVERY = (
    'import Libdl; print(abspath(Libdl.dlpath("libjulia")), "\\0", Sys.BINDIR)'
)
_IMPORT_LOCK = threading.Lock()


def _python_julia_environment(env: MutableMapping[str, str]) -> Path | None:
    configured = str(env.get("PYTHON_JULIAPKG_PROJECT", "")).strip()
    if configured:
        # Keep a configured Windows junction spelling intact. ``resolve()``
        # expands it back to a sandbox-restricted Desktop path before Julia's
        # long-path checks run.
        candidate = Path(configured).absolute()
        if (candidate / "pyjuliapkg" / "meta.json").is_file():
            return candidate
    configured_python = str(
        env.get("BRILLIANCE_PYTHON", "")
        or env.get("PYTHON_EXECUTABLE", "")
    ).strip()
    if not configured_python:
        try:
            from .research_environment import ResearchEnvironment

            configured_python = str(ResearchEnvironment.load().python_executable)
        except Exception:
            configured_python = ""
    virtual_environment = (
        Path(configured_python).resolve().parent.parent / "julia_env"
        if configured_python
        else Path("__missing_configured_python__")
    )
    if (virtual_environment / "pyjuliapkg" / "meta.json").is_file():
        return virtual_environment
    return None


def _juliacall_metadata(env: MutableMapping[str, str]) -> dict[str, str]:
    project = _python_julia_environment(env)
    if project is None:
        return {}
    try:
        payload = json.loads(
            (project / "pyjuliapkg" / "meta.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    executable = Path(str(payload.get("executable", "")))
    libjulia = Path(str(payload.get("libjulia", "")))
    if not executable.is_file() or not libjulia.is_file():
        return {}
    return {
        "project": str(project),
        # Preserve Windows junction spellings. Resolving them expands the paths
        # back into sandbox-restricted Desktop/AppData prefixes.
        "executable": str(executable.absolute()),
        "libjulia": str(libjulia.absolute()),
        "bindir": str(executable.absolute().parent),
    }


def configure_julia_runtime(
    environment: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Configure the pinned local JuTrack environment without runtime updates.

    pyJuTrack historically called ``Pkg.resolve`` and ``Pkg.instantiate`` on
    every import unless the configured offline flag is supplied.
    Local GUI/CLI runs therefore stalled or mutated their dependency graph.
    BRILLIANCE now defaults to the repository-pinned environment everywhere.
    Set ``BRILLIANCE_JULIA_RESOLVE=true`` only during explicit environment
    maintenance.
    """
    env = environment if environment is not None else os.environ
    project_root = Path(__file__).resolve().parent.parent
    jutrack_project = env.get("JUTRACK_JL_PROJECT", "").strip()
    if not jutrack_project:
        bundled = project_root / "JuTrack.jl"
        if bundled.is_dir():
            jutrack_project = str(bundled)
    if jutrack_project:
        env.setdefault("JULIA_PROJECT", jutrack_project)

    allow_resolve = env.get("BRILLIANCE_JULIA_RESOLVE", "").lower() in {
        "1", "true", "yes", "on",
    }
    if allow_resolve:
        env["JULIA_PKG_OFFLINE"] = "false"
        env["PYTHON_JULIAPKG_OFFLINE"] = "no"
    else:
        env.setdefault("JULIA_PKG_OFFLINE", "true")
        env.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")

    # Pin JuliaCall to the prepared PythonCall environment. This avoids package
    # discovery and guarantees that the process uses the tested Julia binary.
    metadata = _juliacall_metadata(env)
    if metadata:
        env.setdefault("PYTHON_JULIACALL_EXE", metadata["executable"])
        env.setdefault("PYTHON_JULIACALL_PROJECT", metadata["project"])
        env.setdefault("PYTHON_JULIACALL_BINDIR", metadata["bindir"])
    return env


def load_pyjutrack() -> Any:
    env = configure_julia_runtime()
    metadata = _juliacall_metadata(env)
    if os.name != "nt" or not metadata:
        return importlib.import_module("pyJuTrack")

    # JuliaCall 0.9.32 always starts `julia -e ...` once to rediscover paths
    # that are already present in pyjuliapkg/meta.json. The Codex Windows
    # workspace sandbox can corrupt quoting in that probe. Intercept only this
    # exact read-only discovery call during import and return the pinned,
    # validated paths; all other subprocess calls retain their normal behavior.
    with _IMPORT_LOCK:
        original_run = subprocess.run

        def run_with_pinned_discovery(command: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                isinstance(command, (list, tuple))
                and _JULIACALL_DISCOVERY in command
                and str(command[0]).casefold()
                == metadata["executable"].casefold()
            ):
                stdout = metadata["libjulia"] + "\0" + metadata["bindir"]
                if not kwargs.get("text") and kwargs.get("encoding") is None:
                    stdout = stdout.encode("utf-8")
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
            return original_run(command, *args, **kwargs)

        subprocess.run = run_with_pinned_discovery
        try:
            return importlib.import_module("pyJuTrack")
        finally:
            subprocess.run = original_run
