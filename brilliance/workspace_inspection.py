"""Bounded, read-only access to research evidence without starting a worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".log", ".csv", ".md", ".py"}
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_CHARS = 12_000
_INTERNAL_DIRECTORIES = {"agent_sessions", "runtime"}


def saved_lattice_context(workspace: Path) -> str:
    """Index exact archived inputs so a unit is not confused with its assembled system."""
    from .lattice import LatticeIR

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    paths = sorted((workspace / "artifacts" / "programs").glob("*/result.json"),
                   key=lambda item: item.stat().st_mtime_ns, reverse=True)
    for path in paths[:100]:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            relative = report.get("candidate_lattice")
            if not relative:
                continue
            candidate_path = (path.parent / relative).resolve()
            candidate_path.relative_to(path.parent.resolve())
            if candidate_path.stat().st_size > _MAX_JSON_BYTES:
                continue
            candidate = LatticeIR.from_dict(json.loads(candidate_path.read_text(encoding="utf-8")))
            if candidate.lattice_id in seen:
                continue
            seen.add(candidate.lattice_id)
            entries.append({
                "lattice_id": candidate.lattice_id, "title": candidate.title,
                "stage": report.get("data", {}).get("design_status"),
                "path": candidate_path.relative_to(workspace.resolve()).as_posix(),
                "energy_gev": candidate.energy_gev, "length_m": candidate.total_length_m,
                "bend_sum_rad": candidate.bend_sum_rad, "element_count": len(candidate.elements),
                "periodicity": candidate.periodicity,
                "intended_repetitions": candidate.metadata.get("intended_repetitions"),
                "evaluation_scope": candidate.metadata.get("evaluation_scope"),
            })
            if len(entries) == 12:
                break
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            continue
    if not entries:
        return ""
    return ("Archived lattice input inventory (includes rejected hypotheses; not a validity certificate):\n"
            + json.dumps(entries, ensure_ascii=False)
            + "\nSelect the exact saved object by scope and identity. A latest candidate may be an assembled ring, not its earlier unit or matching section. Load the chosen path with api.load_lattice(run.workspace / path); inspect older result/source artifacts with read_artifact if needed. Do not reconstruct an archived input from remembered constants.")


def saved_recovery_context(workspace: Path) -> str:
    """Small evidence index for recovery; never treat a checkpoint as a design gate."""
    lines = ["Saved evidence can be inspected with read_artifact; no new calculation is needed to read it."]
    failures = sorted((workspace / "runtime_failures").glob("job-*.json"),
                      key=lambda item: item.stat().st_mtime_ns, reverse=True)
    diagnostics = [path for path in failures if not path.name.endswith(".recovery.json")]
    if diagnostics:
        lines.append("Latest execution diagnostic: " + diagnostics[0].relative_to(workspace).as_posix())
    checkpoints = sorted((workspace / "optimizations").glob("*/checkpoint.json"),
                         key=lambda item: item.stat().st_mtime_ns, reverse=True)
    for path in checkpoints[:4]:
        try:
            if path.stat().st_size > _MAX_JSON_BYTES:
                continue
            saved = json.loads(path.read_text(encoding="utf-8"))
            contract = saved.get("contract", {})
            best = saved.get("global_best") or {}
            entry = {"path": path.relative_to(workspace).as_posix(),
                     "complete": saved.get("complete"),
                     "completed_generation": saved.get("completed_generation"),
                     "evaluation_count": saved.get("evaluation_count"),
                     "variable_names": contract.get("variable_names"),
                     "best_objectives": best.get("objectives") or dict(zip(
                         contract.get("objective_names", []), best.get("objective_values", []))),
                     "best_feasible": best.get("feasible")}
            lines.append("Saved optimization (not a validated lattice): " + json.dumps(entry, ensure_ascii=False))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    if checkpoints:
        lines.append("Inspect /contract, /global_best, or /history as needed. A complete checkpoint means the search finished, not that the scientist's request passed. If only reporting failed, preserve and recover this computation; do not create a new campaign or repeat the search. Changed objectives/geometry require a distinct campaign, never forced reuse.")
    return "\n".join(lines)[:8_000]


def _descriptor(value: Any) -> Any:
    if isinstance(value, dict):
        return {"type": "object", "length": len(value), "keys": list(value)[:20]}
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    if isinstance(value, str) and len(value) > 500:
        return {"type": "string", "length": len(value), "preview": value[:500]}
    return value


def read_workspace_artifact(
    workspace: Path,
    path: str = "",
    *,
    pointer: str = "",
    offset: int = 0,
    limit: int = 40,
) -> dict[str, Any]:
    """Inspect a directory, JSON pointer, or text lines within this workspace.

    Offsets are zero-based. JSON arrays/objects and directory entries are paged;
    text uses line offsets. Large JSON values return descriptors, never silently
    truncated numerical data. Follow a JSON pointer to inspect the exact value.
    """

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    root = workspace.resolve()
    relative = str(path).strip().replace("\\", "/")
    if relative.startswith("workspace/"):
        relative = relative[len("workspace/"):]
    target = (root / relative).resolve()
    try:
        local = target.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path must stay inside the project workspace") from exc
    if any(part.startswith(".") for part in local.parts):
        raise ValueError("hidden files are not research artifacts")
    if local.parts and local.parts[0] in _INTERNAL_DIRECTORIES:
        raise ValueError(
            "Agent event logs and managed runtime files are not scientific artifacts. "
            "Read the execution receipt's full_result path, or list artifacts/programs "
            "for saved result.json and candidate lattices. Coding failures are in runtime_failures."
        )
    result: dict[str, Any] = {"path": local.as_posix(), "offset": offset}
    if target.is_dir():
        if pointer:
            raise ValueError("a JSON pointer requires a JSON file")
        entries = sorted(
            (entry for entry in target.iterdir() if not entry.name.startswith(".")
             and not (target == root and entry.name in _INTERNAL_DIRECTORIES)),
            key=lambda entry: entry.name,
        )
        selected = entries[offset:offset + limit]
        result.update(kind="directory", total=len(entries), entries=[
            {"name": entry.name, "kind": "directory" if entry.is_dir() else "file"}
            for entry in selected
        ])
        result["next_offset"] = offset + len(selected) if offset + len(selected) < len(entries) else None
        return result
    if target.suffix.lower() not in _TEXT_SUFFIXES:
        raise ValueError("Only saved research JSON, text, CSV, Markdown, and Python files can be inspected")
    if target.suffix.lower() == ".json":
        if target.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError("JSON exceeds the 32 MiB inspection limit; use its compact result or a targeted analysis program")
        value = json.loads(target.read_text(encoding="utf-8-sig"))
        if pointer and not pointer.startswith("/"):
            raise ValueError("JSON pointer must be empty or start with '/' (for example /global_best/variables)")
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict) and token in value:
                value = value[token]
            elif isinstance(value, list) and token.isdecimal() and int(token) < len(value):
                value = value[int(token)]
            else:
                if isinstance(value, dict):
                    available = ", ".join(str(key) for key in list(value)[:30])
                    hint = f"Available keys at this level: {available or '(none)'}."
                elif isinstance(value, list):
                    hint = f"This array has {len(value)} items; indices are zero-based."
                else:
                    hint = "The containing value is a scalar, not an object or array."
                raise ValueError(f"JSON pointer does not exist: {pointer}. {hint} Read the parent pointer to inspect its actual schema.")
        total = len(value) if isinstance(value, (list, dict)) else 1
        if isinstance(value, dict):
            data = dict(list(value.items())[offset:offset + limit])
        elif isinstance(value, list):
            data = value[offset:offset + limit]
        else:
            data = value if offset == 0 else None
        result.update(kind="json", pointer=pointer, total=total, data=data,
                      next_offset=offset + limit if offset + limit < total else None,
                      summarized=False)
        if len(json.dumps(data, ensure_ascii=False)) > _MAX_CHARS:
            result["data"] = (
                {key: _descriptor(item) for key, item in data.items()} if isinstance(data, dict)
                else [_descriptor(item) for item in data] if isinstance(data, list)
                else _descriptor(data)
            )
            # A hundred large descriptors can also exceed the response budget.
            if len(json.dumps(result["data"], ensure_ascii=False)) > _MAX_CHARS:
                result["data"] = {"type": type(value).__name__, "length": total}
            result["summarized"] = True
            result["instruction"] = "Values are descriptors, not numerical evidence. Read a narrower JSON pointer or a smaller page for exact values."
        return result
    if pointer:
        raise ValueError("a JSON pointer requires a .json file")
    lines: list[str] = []
    chars = 0
    clipped_line = False
    more = False
    with target.open(encoding="utf-8-sig", errors="replace") as stream:
        for index, line in enumerate(stream):
            if index < offset:
                continue
            if len(lines) >= limit or chars >= _MAX_CHARS:
                more = True
                break
            remaining = _MAX_CHARS - chars
            if len(line) > remaining:
                line = line[:remaining]
                clipped_line = True
            lines.append(line)
            chars += len(line)
    result.update(kind="text", text="".join(lines), line_count=len(lines),
                  next_offset=offset + len(lines) if more else None,
                  truncated=more or clipped_line, clipped_line=clipped_line)
    return result
