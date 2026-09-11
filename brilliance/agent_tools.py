"""Read-only tools for inspecting archived research evidence.

Scientific calculations are deliberately absent from this catalog.  The coding
agent performs new work in a saved research program through
``brilliance.accelerator_api``; these two tools only let it discover and query
results that already exist.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifacts import canonical_json, content_hash, json_safe
from .evidence_data import EvidenceCatalog, validate_query_result

AGENT_TOOL_VERSION = "3.0.0"


@dataclass(frozen=True, slots=True)
class AgentToolSpec:
    id: str
    description: str
    parameters: Mapping[str, Any]
    output: str
    operation_class: str = "inspect"

    def as_dict(self, *, available: bool = True) -> dict[str, Any]:
        return {**json_safe(asdict(self)), "requires_lattice": False, "available": available}


_TOOLS = (
    AgentToolSpec(
        "inspect_evidence",
        (
            "List archived datasets with schema, units, scope, acceptance state, and "
            "provenance. This performs no accelerator calculation."
        ),
        {"dataset_id": "optional exact dataset ID"},
        "a compact catalog or one dataset descriptor",
    ),
    AgentToolSpec(
        "query_evidence",
        (
            "Select, filter, sort, derive, tabulate, or plot rows from an archived "
            "dataset. This is a read-only transformation and never reruns JuTrack."
        ),
        {
            "dataset_id": "exact ID returned by inspect_evidence",
            "columns": "optional columns to return",
            "filters": "optional declarative filters",
            "sort_by": "optional declarative sort",
            "derived": "optional declarative derived columns",
            "offset": "non-negative row offset",
            "limit": "1..5000 rows",
            "view": "table, line, scatter, or bar view declaration",
        },
        "a provenance-linked table or plot over immutable evidence",
        "transform",
    ),
)


def agent_tool_catalog(_project: Any) -> list[dict[str, Any]]:
    """Return the intentionally small model-visible archive tool surface."""

    return [tool.as_dict() for tool in _TOOLS]


class AgentEvidenceStore:
    """Immutable archive retained for both new and historical projects."""

    def __init__(self, project_root: Path | str):
        self.root = Path(project_root).resolve()

    def save(self, project_id: str, evidence: Mapping[str, Any]) -> Path:
        payload = json_safe(evidence)
        artifact_id = str(payload["artifact_id"])
        path = self.root / project_id / "agent_evidence" / f"{artifact_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json(_evidence_identity(existing)) != canonical_json(
                _evidence_identity(payload)
            ):
                raise RuntimeError(f"immutable agent-evidence conflict: {path}")
        return path

    def load(self, project_id: str, artifact_id: str) -> Mapping[str, Any]:
        if not artifact_id.startswith("agent-"):
            raise LookupError(f"Unknown agent artifact: {artifact_id}")
        path = self.root / project_id / "agent_evidence" / f"{artifact_id}.json"
        if not path.is_file():
            raise LookupError(f"Agent evidence not found: {artifact_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid agent evidence: {artifact_id}")
        return payload


def _evidence_identity(evidence: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "project_id",
        "project_revision",
        "lattice_id",
        "tool_id",
        "tool_version",
        "arguments",
        "request",
        "generated_code",
        "result",
        "validation",
    )
    return {key: evidence.get(key) for key in keys}


def execute_agent_tool(
    project: Any,
    tool_id: str,
    arguments: Mapping[str, Any],
    *,
    project_root: Path | str,
) -> Mapping[str, Any]:
    """Execute one read-only evidence operation and archive its result."""

    if tool_id == "inspect_evidence":
        catalog = EvidenceCatalog(project, project_root)
        dataset_id = str(arguments.get("dataset_id", "")).strip()
        result = {
            "engine": "archived_evidence_query",
            "datasets": [catalog.describe(dataset_id)] if dataset_id else catalog.descriptors(),
            "read_only": True,
        }
        validation = {
            "passed": True,
            "validator": "evidence_catalog_contract",
            "version": "1.0.0",
            "issues": [],
            "checked_values": {"dataset_count": len(result["datasets"])},
            "claim_authority": "metadata_only",
        }
    elif tool_id == "query_evidence":
        result = EvidenceCatalog(project, project_root).query(arguments)
        validation = validate_query_result(result)
    else:
        raise ValueError(
            f"Unknown read-only tool {tool_id!r}. New calculations belong in a saved "
            "research program, not a scenario-specific registered tool."
        )

    identity = {
        "project_id": project.id,
        "project_revision": project.revision,
        "lattice_id": getattr(project, "active_lattice_id", None),
        "tool_id": tool_id,
        "tool_version": AGENT_TOOL_VERSION,
        "arguments": json_safe(arguments),
        "result": result,
        "validation": validation,
    }
    evidence = {
        "schema_version": "1.0",
        "artifact_id": "agent-" + content_hash(identity, length=20),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **identity,
        "source": "registered_read_only_evidence_query",
        "structured_result": json_safe(result),
        "deterministic_validation": json_safe(validation),
    }
    AgentEvidenceStore(project_root).save(project.id, evidence)
    return evidence


def render_agent_artifact(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    """Render current archive queries and retain readable historical artifacts."""

    tool_id = str(evidence.get("tool_id", ""))
    value = evidence.get("structured_result", evidence.get("result", {}))
    result = value if isinstance(value, Mapping) else {}
    validation = evidence.get("deterministic_validation", evidence.get("validation", {}))
    common = {
        "tool": tool_id,
        "engine": result.get("engine"),
        "passed": validation.get("passed") if isinstance(validation, Mapping) else None,
        "lattice_id": evidence.get("lattice_id"),
    }
    if tool_id == "inspect_evidence":
        return {
            "kind": "dataset_catalog",
            "title": "Available archived evidence datasets",
            "rows": result.get("datasets", []),
            "meta": common,
        }
    if tool_id == "query_evidence":
        dataset = result.get("dataset")
        dataset = dataset if isinstance(dataset, Mapping) else {}
        view = result.get("view")
        view = view if isinstance(view, Mapping) else {"kind": "table"}
        return {
            "kind": "data_view",
            "title": str(view.get("title") or dataset.get("title") or "Evidence view"),
            "rows": result.get("rows", []),
            "columns": result.get("columns", []),
            "view": view,
            "dataset": dataset,
            "meta": {
                **common,
                "shown_rows": result.get("row_count"),
                "matching_rows": result.get("total_matches"),
                "truncated": result.get("truncated"),
            },
        }
    # Historical fixed-tool artifacts remain viewable after their executors
    # have been retired. Preserve their already-computed tables and plots.
    return {
        "kind": "archived_research_result",
        "title": str(evidence.get("title") or tool_id.replace("_", " ").title()),
        "summary": result.get("summary", {}),
        "rows": result.get("rows", result.get("comparison_table", [])),
        "tables": result.get("tables", []),
        "plots": result.get("plots", []),
        "result": result,
        "meta": {**common, "historical_executor_retired": True},
    }


def compact_agent_evidence(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    """Bound an archived result before it is shown to a language model."""

    value = evidence.get("structured_result", evidence.get("result", {}))
    result = value if isinstance(value, Mapping) else {}
    tool_id = str(evidence.get("tool_id", ""))
    if tool_id == "inspect_evidence":
        datasets = result.get("datasets", [])
        rows = datasets if isinstance(datasets, (list, tuple)) else []
        compact: Any = {
            "dataset_count": len(rows),
            "datasets": [
                {
                    "id": item.get("id"),
                    "title": str(item.get("title", ""))[:160],
                    "row_count": item.get("row_count"),
                    "columns": list(item.get("columns", []))[:24],
                    "scope": item.get("scope"),
                    "accepted": item.get("accepted"),
                }
                for item in rows[:50]
                if isinstance(item, Mapping)
            ],
        }
    elif tool_id == "query_evidence":
        rows = result.get("rows", [])
        rows = rows if isinstance(rows, (list, tuple)) else []
        compact = {
            "dataset": result.get("dataset"),
            "columns": result.get("columns"),
            "row_count": result.get("row_count"),
            "total_matches": result.get("total_matches"),
            "truncated": result.get("truncated"),
            "view": result.get("view"),
            "sample_rows": [_compact_value(row) for row in rows[:12]],
        }
    else:
        compact = {
            str(key): _compact_value(item)
            for key, item in list(result.items())[:40]
            if key not in {"trajectory", "twiss_profile", "scans", "elements"}
        }
    return {
        "artifact_id": evidence.get("artifact_id"),
        "tool_id": tool_id,
        "arguments": evidence.get("arguments"),
        "result": compact,
        "validation": evidence.get("deterministic_validation", evidence.get("validation")),
    }


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return "[nested data omitted]"
    if isinstance(value, Mapping):
        return {
            str(key): _compact_value(item, depth=depth + 1)
            for key, item in list(value.items())[:24]
        }
    if isinstance(value, (list, tuple)):
        return [_compact_value(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, str):
        return value[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]
