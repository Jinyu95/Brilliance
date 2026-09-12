"""Runtime-owned, lossless evidence for numerical expert methods.

Snapshots are committed before caller-owned reporting callbacks. They are not
working-lattice selections or scientific acceptance reports. ResearchRun.finish
collects their manifests independently of model-written summaries/serializers.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .atomic_io import atomic_write_text

SCHEMA = "brilliance_numerical_evidence/v1"


def _default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported numerical evidence value: {type(value).__name__}")


def _render(value: Any) -> str:
    # Deliberately no display truncation, depth limit, or nonfinite-to-string repair.
    return json.dumps(value, default=_default, allow_nan=False, ensure_ascii=False,
                      sort_keys=True, indent=2) + "\n"


def _relative_file(root: Path, relative: str) -> Path:
    supplied = Path(relative)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError("Numerical evidence path must remain inside its archive")
    path = (root / supplied).resolve()
    path.relative_to(root.resolve())
    return path


class NumericalEvidenceArchive:
    """One invocation's inputs, snapshots and exact candidate, never a promotion."""

    def __init__(self, output: Path, method: str, inputs: Mapping[str, Any], *,
                 parent_lattice: Mapping[str, Any], run: Any = None):
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", method).strip("-")[:80]
        if not slug:
            raise ValueError("Numerical method needs a nonempty identifier")
        self.output = Path(output).resolve()
        root = self.output / "numerical_runs"
        root.resolve().relative_to(self.output)
        self.directory = root / f"{slug}-{uuid4().hex}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.run = run
        self.sequence = 0
        self.manifest = {"schema": SCHEMA, "method": method, "status": "initialized",
                         "summary": {}, "files": {}, "has_accepted_progress": False,
                         "request_satisfied": False, "use_as_working_lattice": False}
        self._file("inputs", "inputs.json", inputs)
        self._file("parent_lattice", "parent_lattice.json", parent_lattice)
        self._commit()

    @classmethod
    def for_environment(cls, method: str, inputs: Mapping[str, Any], *,
                        parent_lattice: Mapping[str, Any]):
        if not os.environ.get("BRILLIANCE_RUN_OUTPUT", "").strip():
            return None  # Standalone callers retain their explicit callbacks.
        from .research_api import open_research_run
        run = open_research_run()  # Validates workspace/output containment.
        return cls(run.output_directory, method, inputs, parent_lattice=parent_lattice, run=run)

    def _file(self, role: str, name: str, value: Any) -> None:
        path = _relative_file(self.directory, name)
        text = _render(value)
        atomic_write_text(path, text)
        self.manifest["files"][role] = {
            "path": name, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}

    def _commit(self) -> None:
        atomic_write_text(self.directory / "manifest.json", _render(self.manifest))

    def snapshot(self, state: Mapping[str, Any], candidate_lattice: Mapping[str, Any], *,
                 summary: Mapping[str, Any]) -> None:
        """Write a new immutable snapshot, then atomically advance the manifest."""
        prefix = f"snapshots/{self.sequence:06d}"
        self.sequence += 1
        previous = deepcopy(self.manifest)
        try:
            self._file("checkpoint", f"{prefix}/checkpoint.json", state)
            self._file("candidate_lattice", f"{prefix}/candidate_lattice.json", candidate_lattice)
            self.manifest.update(status=str(state["status"]), summary=deepcopy(dict(summary)),
                                 has_accepted_progress=bool(summary.get("has_accepted_progress")))
            self._commit()
        except Exception:
            self.manifest = previous
            raise
        if self.run is not None:
            self.run.progress(
                f"Numerical method: {self.manifest['method']}",
                detail=(f"{summary['status']}; accepted steps={summary.get('accepted_step_count', 0)}; "
                        f"achieved fraction={summary.get('fraction')}. Raw evidence archived."),
                metrics={k: v for k, v in summary.items() if isinstance(v, (str, int, float, bool))},
                tables=[{"title": "Runtime numerical outcome", "rows": [dict(summary)]}],
                event_key=f"numerical:{self.directory.name}",
                kind="finding" if summary.get("has_accepted_progress") else "progress")

    def complete(self, result: Mapping[str, Any], *, ring_lattice: Mapping[str, Any] | None = None) -> None:
        self._file("result", "result.json", result)
        if ring_lattice is not None:
            self._file("candidate_ring", "candidate_ring.json", ring_lattice)
        self.manifest["completed"] = True
        self._commit()

    def failed(self, error: Exception) -> None:
        """An execution error does not delete the last accepted snapshot."""
        self._file("error", "execution_error.json", {"type": type(error).__name__, "message": str(error)})
        self.manifest["status"] = "execution_error"
        self.manifest["summary"]["status"] = "execution_error"
        self._commit()


def collect_numerical_evidence(output: Path) -> dict[str, Any]:
    """Validate runtime manifests and make them visible to the normal result UI.

    No candidate is selected, replaced, or declared scientifically accepted here.
    File hashes and containment are checked even when a wrapper discarded result
    fields. Each invocation remains separate; multiple runs cannot overwrite one
    another's candidates. An interrupted write retains the previous valid manifest.
    """
    output = Path(output).resolve()
    runs, downloads, rows = [], [], []
    for path in sorted((output / "numerical_runs").glob("*/manifest.json")):
        directory = path.parent.resolve()
        directory.relative_to(output)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"Unknown runtime numerical evidence schema: {path}")
        files = {}
        for role, item in manifest["files"].items():
            file_path = _relative_file(directory, item["path"])
            if hashlib.sha256(file_path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Numerical evidence integrity failure: {file_path}")
            relative = file_path.relative_to(output).as_posix()
            files[role] = {**item, "path": relative}
            downloads.append({"path": relative, "title": f"{manifest['method']}: {role.replace('_', ' ')}"})
        summary = dict(manifest["summary"])
        rows.append({"method": manifest["method"], **summary,
                     "accepted_candidate": files.get("candidate_lattice", {}).get("path")
                     if manifest["has_accepted_progress"] else None})
        runs.append({"method": manifest["method"], "status": manifest["status"],
                     "has_accepted_progress": manifest["has_accepted_progress"],
                     "summary": summary, "files": files})
        downloads.append({"path": path.relative_to(output).as_posix(),
                          "title": f"{manifest['method']}: numerical evidence manifest"})
    return {"runs": runs, "downloads": downloads, "rows": rows,
            "optimization_started": any(r["summary"].get("optimization_started") for r in runs)}
