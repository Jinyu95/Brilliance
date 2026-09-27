"""Identify inherited research evidence from a branch's immutable checkpoint."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


def collect_scope(project_dir: Path, project_meta: Mapping[str, Any],
                  origin: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read only a sibling project's checkpoint; never infer scope from mtimes."""
    directory = Path(project_dir).resolve()
    origin = origin if isinstance(origin, Mapping) else {}
    branch = bool(origin or project_meta.get("checkpoint_origin") or
                  (directory / "workspace" / "RESTORED_CHECKPOINT.json").exists())
    result: dict[str, Any] = {
        "branch": branch, "exact": not branch,
        "inherited_conversation_files": set(), "inherited_run_ids": set(),
        "source_manifest": None, "warnings": [],
    }
    if not branch:
        return result

    def unavailable(reason: str) -> dict[str, Any]:
        result["warnings"].append(
            "Exact branch inheritance is unavailable: " + reason +
            ". Saved workspace counts may include inherited evidence.")
        return result

    source_id = origin.get("source_project")
    checkpoint_id = origin.get("source_checkpoint")
    if (not isinstance(source_id, str) or
            not re.fullmatch(r"project-[A-Za-z0-9_-]+", source_id) or
            not isinstance(checkpoint_id, str) or
            not re.fullmatch(r"[0-9]+-[0-9a-f]+", checkpoint_id)):
        return unavailable("the saved source project or checkpoint identifier is invalid")
    root = directory.parent
    source = (root / source_id).resolve()
    manifest = (source / "chat_checkpoints" / f"{checkpoint_id}.json").resolve()
    if source.parent != root or source == directory or not manifest.is_relative_to(source):
        return unavailable("the source checkpoint is outside its sibling project")
    if not manifest.is_file():
        return unavailable("the source checkpoint is not present under the current project root; "
                           "external source_directory paths are not followed")
    try:
        if manifest.stat().st_size > 64 * 1024 * 1024:
            return unavailable("the source checkpoint exceeds the 64 MiB read limit")
        raw = manifest.read_bytes()
        payload = json.loads(raw)
        if (not isinstance(payload, dict) or
                payload.get("schema") != "brilliance-conversation-checkpoint/v1" or
                payload.get("project_id") != source_id or payload.get("id") != checkpoint_id or
                not isinstance(payload.get("files"), dict) or
                not isinstance(payload.get("conversation"), list)):
            return unavailable("the source checkpoint schema or identity is invalid")
        inherited_runs: set[str] = set()
        inherited_messages: set[str] = set()
        for relative, digest in payload["files"].items():
            path = PurePosixPath(relative)
            if (path.is_absolute() or ".." in path.parts or "\\" in relative or
                    not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                return unavailable("the source checkpoint file inventory is malformed")
            if path.parts[:3] == ("workspace", "artifacts", "programs") and len(path.parts) >= 5:
                run_id = path.parts[3]
                if not re.fullmatch(r"program-[A-Za-z0-9_-]+", run_id):
                    return unavailable("the source checkpoint contains an invalid program run identifier")
                inherited_runs.add(run_id)
        for exchange in payload["conversation"]:
            filename = exchange.get("filename") if isinstance(exchange, dict) else None
            if not isinstance(filename, str) or not re.fullmatch(r"[0-9]+-[0-9a-f]+\.json", filename):
                return unavailable("the source checkpoint conversation inventory is malformed")
            inherited_messages.add(filename)
    except (OSError, ValueError, TypeError) as exc:
        return unavailable(f"the source checkpoint could not be read ({type(exc).__name__})")
    result.update(exact=True, inherited_conversation_files=inherited_messages,
                  inherited_run_ids=inherited_runs, source_manifest=str(manifest),
                  source_manifest_record={"path": str(manifest), "bytes": len(raw),
                                          "sha256": hashlib.sha256(raw).hexdigest()})
    return result
