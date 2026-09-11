"""Content-addressed, immutable artifacts for scientific provenance."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def json_safe(value: Any) -> Any:
    """Convert scientific Python values into deterministic JSON values."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(value: Any, *, length: int = 20) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


class ArtifactConflictError(RuntimeError):
    """Raised when an immutable artifact path is reused with different data."""


@dataclass(frozen=True, slots=True)
class RunRecord:
    schema_version: str
    run_id: str
    study_id: str
    experiment_id: str
    status: str
    backend: str
    backend_version: str
    started_at: str
    completed_at: str
    seed: int
    simulator_calls: int
    request: Mapping[str, Any]
    metrics: Mapping[str, Any]
    validation: Mapping[str, Any]
    raw_result: Mapping[str, Any]
    provenance: Mapping[str, Any]
    error: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunRecord":
        return cls(**dict(data))


class ArtifactStore:
    """Write immutable study and run artifacts beneath one root directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _write_once(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
        except FileExistsError:
            existing = path.read_text(encoding="utf-8")
            if canonical_json(json.loads(existing)) != canonical_json(payload):
                raise ArtifactConflictError(
                    f"immutable artifact already exists with different content: {path}"
                )
        return path

    def study_dir(self, study_id: str) -> Path:
        return self.root / study_id

    def save_study(self, spec: Any) -> Path:
        return self._write_once(self.study_dir(spec.id) / "study.json", spec)

    def list_study_ids(self) -> list[str]:
        """Return studies that contain a readable study contract."""
        paths = sorted(
            self.root.glob("*/study.json"),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )
        return [
            path.parent.name
            for path in paths
            if path.is_file()
        ]

    def load_study_data(self, study_id: str) -> dict[str, Any] | None:
        path = self.study_dir(study_id) / "study.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_run(self, record: RunRecord) -> Path:
        return self._write_once(
            self.study_dir(record.study_id) / "runs" / f"{record.run_id}.json",
            record,
        )

    def load_run(self, study_id: str, run_id: str) -> RunRecord | None:
        path = self.study_dir(study_id) / "runs" / f"{run_id}.json"
        if not path.exists():
            return None
        return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_runs(self, study_id: str) -> list[RunRecord]:
        directory = self.study_dir(study_id) / "runs"
        if not directory.exists():
            return []
        return [
            RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(directory.glob("*.json"))
        ]

    def save_outcome(self, study_id: str, outcome: Any) -> Path:
        digest = content_hash(outcome)
        return self._write_once(
            self.study_dir(study_id) / "outcomes" / f"{digest}.json", outcome
        )

    def load_outcome_data(self, study_id: str) -> list[dict[str, Any]]:
        directory = self.study_dir(study_id) / "outcomes"
        if not directory.exists():
            return []
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        ]

    def save_campaign(self, study_id: str, campaign: Any) -> Path:
        """Save an immutable multi-stage evidence-ladder outcome."""
        digest = content_hash(campaign)
        return self._write_once(
            self.study_dir(study_id) / "campaigns" / f"{digest}.json", campaign
        )

    def load_campaign_data(self, study_id: str) -> list[dict[str, Any]]:
        directory = self.study_dir(study_id) / "campaigns"
        if not directory.exists():
            return []
        paths = sorted(
            directory.glob("*.json"),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in paths
        ]

    def save_report(self, study_id: str, report: str, outcome_id: str) -> Path:
        path = self.study_dir(study_id) / "reports" / f"{outcome_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(report.rstrip() + "\n")
        except FileExistsError:
            if path.read_text(encoding="utf-8") != report.rstrip() + "\n":
                raise ArtifactConflictError(
                    f"immutable report already exists with different content: {path}"
                )
        return path
