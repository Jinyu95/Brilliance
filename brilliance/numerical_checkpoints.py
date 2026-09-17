"""Durable numerical output before optional presentation; never design acceptance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .atomic_io import atomic_write_text

_SCHEMA = "brilliance_numerical_checkpoint/v1"


class ReportingStopped(RuntimeError):
    """Optional reporting failed after a verified checkpoint was published."""


def _root(output_directory: Path | str | None) -> Path:
    managed = os.environ.get("BRILLIANCE_RUN_OUTPUT")
    if not managed and output_directory is None:
        raise ValueError("Numerical checkpoints require an explicit or managed output directory")
    root = Path(managed or output_directory).resolve()
    if output_directory is not None and Path(output_directory).resolve() != root:
        raise ValueError("Numerical checkpoints cannot change the current run's output directory")
    return root


def _json_default(value):
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported numerical checkpoint value: {type(value).__name__}")


def write_numerical_checkpoint(kind: str, result: Mapping[str, Any], *,
                               output_directory: Path | str | None = None,
                               checks_passed: bool | None = None) -> dict[str, Any]:
    """Save detached exact numerical output, then atomically advance its pointer.

    No display rounding, permissive nonfinite conversion, simulation or machine
    acceptance occurs. ``checks_passed`` describes only this operation's checks.
    Earlier checkpoints are immutable and remain available after later failures.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("Numerical checkpoint kind must be a nonempty string")
    if not isinstance(result, Mapping):
        raise TypeError("Numerical checkpoint result must be a mapping")
    if checks_passed is not None and not isinstance(checks_passed, bool):
        raise TypeError("Numerical checkpoint checks_passed must be true, false or null")
    root = _root(output_directory)
    relative = f"numerical_checkpoints/{uuid4().hex}.json"
    payload = {"schema": _SCHEMA, "kind": kind, "result": dict(result),
               "checks_passed": checks_passed, "request_satisfied": False,
               "use_as_working_lattice": False}
    text = json.dumps(payload, default=_json_default, allow_nan=False, ensure_ascii=False,
                      sort_keys=True, indent=2) + "\n"
    descriptor = {"schema": _SCHEMA, "path": relative,
                  "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  "kind": kind, "checks_passed": checks_passed}
    atomic_write_text(root / relative, text)
    atomic_write_text(root / "numerical_checkpoints" / "latest.json",
                      json.dumps(descriptor, sort_keys=True, indent=2) + "\n")
    return descriptor


def read_numerical_checkpoint(descriptor: Mapping[str, Any] | None = None, *,
                              output_directory: Path | str | None = None,
                              source_output_directory: Path | str | None = None) -> dict[str, Any]:
    """Read exact saved output after containment and SHA verification.

    Explicit source_output_directory may refer to an earlier managed worker in
    this project. Reading a checkpoint does not validate its physics, import a
    different project's results, or change current output ownership.
    """
    from .evidence_scope import evidence_source_root
    root = evidence_source_root(_root(output_directory), source_output_directory)
    if descriptor is None:
        descriptor = json.loads((root / "numerical_checkpoints" / "latest.json").read_text(encoding="utf-8"))
    relative = Path(str(descriptor.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Numerical checkpoint path leaves the current run")
    path = (root / relative).resolve()
    path.relative_to(root)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor.get("sha256"):
        raise ValueError("Numerical checkpoint failed its SHA-256 check")
    payload = json.loads(raw.decode("utf-8"))
    if (payload.get("schema") != _SCHEMA or descriptor.get("schema") != _SCHEMA
            or payload.get("kind") != descriptor.get("kind")
            or payload.get("checks_passed") != descriptor.get("checks_passed")):
        raise ValueError("Numerical checkpoint identity does not match its descriptor")
    return {**payload, "descriptor": dict(descriptor)}
