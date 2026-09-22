"""Serve explicitly referenced local figures without changing scientific evidence."""
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote

from .codex_runtime import _resolve_completion_artifact, _workspace_artifact_record

_IMAGE_SUFFIXES = {".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp"}
_LINK = re.compile(r"(!?)\[([^\]\n]*)\]\((<[^>\n]+>|[^\n]+?)\)")


def present_local_artifacts(root: Path, project_id: str, message: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only repair of saved chat, scoped strictly to this project's artifacts.

    Explicit references can show older figures. Undeclared figures beside a
    result are displayed as program output, never as accepted physics evidence.
    """
    if message.get("role") != "assistant":
        return dict(message)
    artifact_root = (Path(root) / project_id / "workspace" / "artifacts").resolve()
    attachments = [dict(item) for item in message.get("attachments", []) if isinstance(item, Mapping)]
    seen = {str(item.get("path")) for item in attachments if item.get("type") == "image"}

    def resolve(reference: str) -> Path | None:
        raw = unquote(reference.strip().strip("<>"))
        prefix = f"/api/projects/{project_id}/workspace-artifacts/"
        if raw.startswith(prefix):
            raw = raw[len(prefix):].split("?", 1)[0]
        elif raw.startswith("file:///"):
            raw = raw[8:]
        elif re.match(r"^[a-zA-Z][\w+.-]*://", raw) or raw.startswith("/api/"):
            return None
        try:
            path = _resolve_completion_artifact(artifact_root, raw.replace("\\", "/"))
            return path if path.is_file() else None
        except (OSError, ValueError):
            return None

    def attach(path: Path, title: str, role: str) -> dict[str, Any]:
        record = _workspace_artifact_record(path, artifact_root, project_id)
        if record["path"] not in seen:
            attachments.append({**record, "type": "image", "title": title or record["title"],
                                "presentation_role": role})
            seen.add(record["path"])
        return record

    def link(match: re.Match) -> str:
        path = resolve(match[3])
        if path is None:
            return match[0]
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            record = attach(path, match[2], "referenced_output")
            # The attachment renderer owns the image, avoiding two copies.
            return f"[{match[2] or record['title']}]({record['url']})"
        record = _workspace_artifact_record(path, artifact_root, project_id)
        return f"[{match[2]}]({record['url']})"

    content = _LINK.sub(link, str(message.get("content", "")))
    for match in re.finditer(r"`([^`\n]+)`", content):
        path = resolve(match[1])
        if path and path.suffix.lower() in _IMAGE_SUFFIXES:
            attach(path, "", "referenced_output")

    for item in list(attachments):
        if item.get("type") != "evidence_summary" or item.get("presentation_role") == "selected_evidence":
            continue
        reference = item.get("result_path") or (
            f"programs/{item['run_id']}/result.json" if item.get("run_id") else "")
        result = resolve(str(reference)) if reference else None
        if result is None or result.name != "result.json":
            continue
        # Do not search unrelated runs or nested candidate branches.
        for path in sorted(result.parent.iterdir()):
            if path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file():
                checked = resolve(str(path))
                if checked:
                    attach(checked, path.stem.replace("_", " ") + " — program output (not independently validated)",
                           "unverified_artifact")
    return {**message, "content": content, "attachments": attachments}
