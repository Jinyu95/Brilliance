"""Application-owned readiness gate for the configured JuTrack boundary.

The language model never performs this check.  Brilliance verifies its own
Python/Julia/JuTrack installation before it accepts research work and caches a
versioned result for the web health endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import json_safe
from .jutrack_lint import JUTRACK_API_CONTRACT_VERSION
from .jutrack_service import JUTRACK_SERVICE_VERSION
from .research_environment import ResearchEnvironment, ResearchEnvironmentError

RUNTIME_READINESS_SCHEMA_VERSION = "1.0"
DEFAULT_READINESS_MAX_AGE_S = 86_400
REQUIRED_JUTRACK_CALLABLES = (
    "Lattice",
    "MARKER",
    "DRIFT",
    "KQUAD",
    "QUAD",
    "KSEXT",
    "KOCT",
    "SBEND",
    "RBEND",
    "RFCA",
    "CORRECTOR",
    "findelem",
    "findm66",
    "periodicEdwardsTengTwiss",
    "twissline",
    "twissring",
    "gettune",
    "getchrom",
    "ringpara",
    "dynamic_aperture",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def readiness_path(environment: ResearchEnvironment) -> Path:
    return environment.python_julia_environment / "brilliance-runtime.json"


def _contract_files(environment: ResearchEnvironment) -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        environment.accelerator_api.resolve(),
        (Path(__file__).resolve().parent / "jutrack_service.py").resolve(),
        (Path(__file__).resolve().parent / "jutrack_lint.py").resolve(),
        environment.pyjutrack_module.resolve(),
        (environment.julia_project / "Project.toml").resolve(),
    )


def runtime_contract_fingerprint(environment: ResearchEnvironment) -> str:
    digest = hashlib.sha256()
    for path in _contract_files(environment):
        digest.update(str(path).casefold().encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            digest.update(f"missing:{type(exc).__name__}".encode("utf-8"))
    digest.update(JUTRACK_SERVICE_VERSION.encode("ascii"))
    digest.update(JUTRACK_API_CONTRACT_VERSION.encode("ascii"))
    return digest.hexdigest()[:24]


def _write_readiness(environment: ResearchEnvironment, record: Mapping[str, Any]) -> Path:
    path = readiness_path(environment)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{time.time_ns()}.tmp"
    temporary.write_text(
        json.dumps(json_safe(record), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return path


def _parse_checked_at(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def cached_runtime_readiness(
    environment: ResearchEnvironment,
    *,
    max_age_s: int = DEFAULT_READINESS_MAX_AGE_S,
) -> dict[str, Any]:
    """Return a bounded readiness status without importing Julia."""

    path = readiness_path(environment)
    if not path.is_file():
        return {
            "status": "missing",
            "ready": False,
            "detail": "No application-owned JuTrack readiness record exists.",
            "path": str(path),
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "ready": False,
            "detail": f"Readiness record is unreadable: {type(exc).__name__}: {exc}",
            "path": str(path),
        }
    if not isinstance(value, Mapping):
        return {
            "status": "invalid",
            "ready": False,
            "detail": "Readiness record is not a JSON object.",
            "path": str(path),
        }
    record = dict(value)
    reasons: list[str] = []
    if record.get("status") != "ready":
        reasons.append(str(record.get("detail") or "the last JuTrack probe failed"))
    if record.get("schema_version") != RUNTIME_READINESS_SCHEMA_VERSION:
        reasons.append("the readiness schema changed")
    if record.get("contract_fingerprint") != runtime_contract_fingerprint(environment):
        reasons.append("the Brilliance/JuTrack API contract changed")
    expected_paths = {
        "python": environment.python_executable,
        "julia": environment.julia_executable,
        "libjulia": environment.libjulia,
        "pyjutrack": environment.pyjutrack_module,
    }
    for name, expected in expected_paths.items():
        recorded = str(record.get(name, "")).casefold()
        if recorded != str(expected.resolve()).casefold():
            reasons.append(f"configured {name} changed")
    checked_at = _parse_checked_at(record.get("checked_at"))
    if checked_at is None or time.time() - checked_at > max_age_s:
        reasons.append("the runtime check is stale")
    if reasons:
        return {
            **record,
            "status": "stale" if record.get("status") == "ready" else "failed",
            "ready": False,
            "detail": "; ".join(dict.fromkeys(reasons)),
            "path": str(path),
        }
    return {**record, "ready": True, "path": str(path)}


def probe_runtime(environment: ResearchEnvironment | None = None) -> dict[str, Any]:
    """Exercise the actual public lattice builder and JuTrack map boundary."""

    runtime = environment or ResearchEnvironment.load()
    runtime.require_active_python(operation="JuTrack application readiness check")
    started = time.monotonic()

    from . import accelerator_api as api

    jt = api.jutrack()
    loaded_pyjutrack = Path(str(getattr(jt, "__file__", ""))).resolve()
    if loaded_pyjutrack != runtime.pyjutrack_module.resolve():
        raise RuntimeError(
            f"loaded pyJuTrack from {loaded_pyjutrack}, expected {runtime.pyjutrack_module.resolve()}"
        )
    missing = [name for name in REQUIRED_JUTRACK_CALLABLES if not callable(getattr(jt, name, None))]
    if missing:
        raise RuntimeError(f"configured pyJuTrack is missing required callables: {missing}")

    cell = api.periodic_cell(
        title="Brilliance JuTrack API contract probe",
        energy_gev=3.0,
        intended_repetitions=16,
        source="brilliance.runtime_health",
        elements=[
            api.drift("BRILLIANCE_CHECK:D1", 0.2),
            api.quadrupole("BRILLIANCE_CHECK:QF", 0.1, 0.2),
            api.sector_bend("BRILLIANCE_CHECK:B", 0.5, 2.0 * math.pi / 16.0),
            api.sextupole("BRILLIANCE_CHECK:S", 0.1, 0.0),
            api.quadrupole("BRILLIANCE_CHECK:QD", 0.1, -0.2),
            api.drift("BRILLIANCE_CHECK:D2", 0.2),
        ],
    )
    ring = api.build_jutrack_lattice(cell)
    matrix = np.asarray(jt.findm66(ring, dp=0.0), dtype=float)
    if matrix.shape != (6, 6) or not bool(np.all(np.isfinite(matrix))):
        raise RuntimeError(
            f"JuTrack findm66 contract returned shape {matrix.shape} or non-finite values"
        )
    qf_indices = [int(value) for value in jt.findelem(ring, name="BRILLIANCE_CHECK:QF")]
    if qf_indices != [2]:
        raise RuntimeError(f"JuTrack findelem(name=...) contract returned {qf_indices}, expected [2]")

    record = {
        "schema_version": RUNTIME_READINESS_SCHEMA_VERSION,
        "status": "ready",
        "ready": True,
        "checked_at": _now(),
        "duration_s": time.monotonic() - started,
        "contract_fingerprint": runtime_contract_fingerprint(runtime),
        "jutrack_service_version": JUTRACK_SERVICE_VERSION,
        "jutrack_api_contract_version": JUTRACK_API_CONTRACT_VERSION,
        "python": str(runtime.python_executable.resolve()),
        "julia": str(runtime.julia_executable.resolve()),
        "libjulia": str(runtime.libjulia.resolve()),
        "depot": str(runtime.julia_depot.resolve()),
        "pyjutrack": str(runtime.pyjutrack_module.resolve()),
        "julia_version": str(jt._jl.seval("VERSION")),
        "checks": {
            "configured_python": True,
            "required_callables": list(REQUIRED_JUTRACK_CALLABLES),
            "lattice_elements_built": len(ring),
            "findm66_shape": list(matrix.shape),
            "findm66_finite": True,
            "findelem_name_indices": qf_indices,
        },
        "detail": (
            "Configured Python loaded pyJuTrack, built Brilliance lattice elements, and passed "
            "the JuTrack findm66 and findelem(name=...) contracts."
        ),
    }
    _write_readiness(runtime, record)
    return record


def _last_json_object(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return None


def ensure_runtime_ready(
    environment: ResearchEnvironment,
    *,
    force: bool = False,
    max_age_s: int = DEFAULT_READINESS_MAX_AGE_S,
    timeout_s: int = 300,
) -> dict[str, Any]:
    """Require a successful isolated JuTrack probe before the app starts."""

    cached = cached_runtime_readiness(environment, max_age_s=max_age_s)
    if cached.get("ready") is True and not force:
        return cached
    completed = subprocess.run(
        [str(environment.python_executable), "-m", "brilliance.runtime_health", "--probe"],
        cwd=str(environment.source_root),
        env=environment.process_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    combined = "\n".join((completed.stdout or "", completed.stderr or "")).strip()
    record = _last_json_object(completed.stdout or "")
    if completed.returncode != 0 or not record or record.get("status") != "ready":
        detail = str((record or {}).get("detail") or combined[-2_000:] or "unknown failure")
        raise ResearchEnvironmentError(
            "JuTrack application readiness gate failed: "
            + detail
            + ". Run the configured `scripts/setup_env.py --check-only` repair command."
        )
    return {**record, "path": str(readiness_path(environment))}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Brilliance's configured JuTrack boundary")
    parser.add_argument("--probe", action="store_true", help="run the full JuTrack API probe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    environment = ResearchEnvironment.load()
    try:
        record = probe_runtime(environment) if args.probe else cached_runtime_readiness(environment)
    except Exception as exc:
        record = {
            "schema_version": RUNTIME_READINESS_SCHEMA_VERSION,
            "status": "failed",
            "ready": False,
            "checked_at": _now(),
            "detail": f"{type(exc).__name__}: {exc}",
        }
        _write_readiness(environment, record)
    print(json.dumps(json_safe(record), ensure_ascii=True, sort_keys=True))
    return 0 if record.get("ready") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
