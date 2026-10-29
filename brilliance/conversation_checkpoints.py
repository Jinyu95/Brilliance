"""File-backed conversation checkpoints; restore by forking, never by rewind.

These capture saved state, not a process memory image. Live jobs are deliberately
not cloned. Concurrently changing files make a checkpoint unavailable rather
than silently associating later data with an earlier conversation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import json_safe
from .atomic_io import atomic_write_text
from .projects import ProjectStore, ResearchProject

_ROOTS = ("workspace", "state", "agent_evidence", "evidence")
_EXCLUDED = {"agent_sessions", "runtime", "progress", "__pycache__", ".git", ".venv",
             "node_modules", ".env", "auth.json", "credentials.json"}
_SCHEMA = "brilliance-conversation-checkpoint/v1"


def _restore_io_path(path: Path) -> Path:
    """Allow Windows staging to exceed MAX_PATH without a system setting change."""
    if os.name != "nt":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _publish_branch(staging: Path, destination: Path) -> None:
    """Publish atomically, allowing brief Windows scanner/sharing locks to clear."""
    # A large restored tree can stay locked for several seconds after its last
    # write. Allow about 14 seconds, without extending the research/job budget.
    for attempt in range(32):
        # Never replace or merge with another project, including one created
        # while a transient lock was being retried.
        if destination.exists():
            raise FileExistsError(f"The destination project already exists: {destination}")
        try:
            staging.rename(destination)
            return
        except OSError as exc:
            transient = (
                isinstance(exc, PermissionError)
                or getattr(exc, "winerror", None) in (5, 32, 33)
            )
            if not transient:
                raise
            if attempt == 31:
                raise PermissionError(
                    "Checkpoint restore could not publish its new project folder "
                    "because access remained denied after bounded retries. "
                    "The original project and saved checkpoint are unchanged; "
                    "no research job was started. Close applications holding the "
                    f"folder open and retry Continue. Destination: {destination}"
                ) from exc
            time.sleep(min(0.05 * 2**attempt, 0.5))


def _project(root, project_id):
    if not re.fullmatch(r"project-[A-Za-z0-9_-]+", project_id):
        raise ValueError("Invalid project identifier")
    base = Path(root).resolve()
    directory = base / project_id
    if directory.resolve().parent != base:
        raise ValueError("Project path escapes research storage")
    return directory


def _inventory(directory):
    files = {}
    for folder in _ROOTS:
        root = directory / folder
        if not root.exists():
            continue
        if root.is_symlink() or getattr(root.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            continue
        for parent, dirs, names in os.walk(root, followlinks=False):
            # Windows junctions are reparse points but aren't always symlinks.
            def allowed(path):
                info = path.lstat()
                return not (path.is_symlink() or
                    getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)
            dirs[:] = [name for name in dirs if name not in _EXCLUDED
                       and not name.startswith(".") and allowed(Path(parent) / name)]
            for name in names:
                path = Path(parent) / name
                if (name in _EXCLUDED or name.startswith(".") or
                        name.endswith((".tmp", ".lock", ".pyc", ".request")) or not allowed(path)):
                    continue
                info = path.stat()
                if stat.S_ISREG(info.st_mode):
                    files[path.relative_to(directory).as_posix()] = (
                        info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return files


def _write_blob(source, destination):
    digest = hashlib.sha256()
    destination.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination, suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as incoming:
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                digest.update(chunk)
                output.write(chunk)
        identity = digest.hexdigest()
        target = destination / identity
        if not target.exists():
            os.replace(temporary, target)
        return identity
    finally:
        temporary.unlink(missing_ok=True)


class ConversationCheckpoints:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def capture(self, project_id, conversation_path):
        directory = _project(self.root, project_id)
        checkpoint_id = Path(conversation_path).stem
        storage = directory / "chat_checkpoints"
        storage.mkdir(parents=True, exist_ok=True)
        manifest_path = storage / f"{checkpoint_id}.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        project = ProjectStore(self.root).load(project_id)
        if project is None:
            raise ValueError("Project state was not saved before this conversation")
        messages = []
        for path in sorted((directory / "conversation").glob("*.json")):
            if path.name > Path(conversation_path).name:
                break
            messages.append({"filename": path.name,
                             "payload": json.loads(path.read_text(encoding="utf-8"))})
        # Reuse hashes only for unchanged file identities; manifests still point
        # to immutable bytes, not to the mutable source files.
        cache_path = storage / "file_index.json"
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        before = _inventory(directory)
        files = {}
        index = {}
        for relative, signature in before.items():
            old = cache.get(relative, {})
            digest = old.get("sha256", "")
            if (old.get("signature") != list(signature) or
                    not re.fullmatch(r"[0-9a-f]{64}", digest) or
                    not (storage / "blobs" / digest).is_file()):
                digest = _write_blob(directory / relative, storage / "blobs")
            files[relative] = digest
            index[relative] = {"signature": signature, "sha256": digest}
        if before != _inventory(directory) or ProjectStore(self.root).load(project_id) != project:
            raise RuntimeError("Saved research files changed during checkpoint capture; no restorable checkpoint created")
        payload = {"schema": _SCHEMA, "id": checkpoint_id, "project_id": project_id,
            "captured_at": datetime.now(timezone.utc).isoformat(), "status": "available",
            "project": json_safe(project), "conversation": messages, "files": files,
            "file_mtime_ns": {relative: signature[1] for relative, signature in before.items()},
            "scope": "saved scientific files and conversation; no process memory, jobs, or provider sessions",
            "source_directory": str(directory)}
        atomic_write_text(manifest_path, json.dumps(payload, ensure_ascii=False, indent=2))
        atomic_write_text(cache_path, json.dumps(index))
        return payload

    def record_failure(self, project_id, conversation_path, exc):
        storage = _project(self.root, project_id) / "chat_checkpoints"
        atomic_write_text(storage / f"{Path(conversation_path).stem}.unavailable.json",
                          json.dumps({"status": "unavailable", "reason": str(exc)}))

    def fork(self, project_id, checkpoint_id, *, continuation=False, new_id=None):
        directory = _project(self.root, project_id)
        if not re.fullmatch(r"[0-9]+-[0-9a-f]+", checkpoint_id):
            raise ValueError("Invalid conversation checkpoint identifier")
        storage = directory / "chat_checkpoints"
        path = storage / f"{checkpoint_id}.json"
        if not path.is_file():
            raise LookupError("This conversation has no saved file checkpoint; older states cannot be reconstructed retroactively")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != _SCHEMA or payload.get("project_id") != project_id:
            raise ValueError("Checkpoint identity mismatch")
        if continuation and not any(
            message.get("role") == "user" and str(message.get("content") or "").strip()
            for exchange in payload.get("conversation", ())
            for message in exchange.get("payload", {}).get("messages", ())
        ):
            raise ValueError("This checkpoint has no saved scientist request to continue")
        new_id = new_id or "project-" + uuid4().hex[:14]
        destination = _project(self.root, new_id)
        if destination.exists():
            raise ValueError("The destination project already exists")
        # A staging directory prevents incomplete branches appearing as projects.
        # Its extra path layer can exceed Windows MAX_PATH even when every
        # original file is readable. Extend only private I/O paths; published
        # project paths and restore provenance retain their ordinary spelling.
        temporary_root = _restore_io_path(self.root / (".restore-" + uuid4().hex))
        staging = temporary_root / new_id
        staging.mkdir(parents=True)
        try:
            regenerated = []
            reset = []
            receipts = {}
            for relative, digest in payload["files"].items():
                target = staging / relative
                if not target.resolve().is_relative_to(staging.resolve()):
                    raise ValueError("Checkpoint file escapes branch workspace")
                if Path(relative).parts[0] not in _ROOTS or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Invalid checkpoint file entry")
                parts = Path(relative).parts
                if continuation and (
                    parts[:2] in {("workspace", "request_budgets"), ("workspace", "result_reviews")}
                    or (parts[:2] == ("workspace", "runtime_failures")
                        and parts[-1].endswith(".recovery.json"))
                ):
                    # This click authorizes a fresh allowance in the new branch.
                    # Raw diagnostics and all numerical checkpoints remain saved.
                    reset.append(relative)
                    continue
                if (len(parts) == 4 and parts[:2] == ("state", "lattices")
                        and parts[-1] in {"reproduce.py", "provenance.json"}):
                    # Archive sidecars embed the owner project ID. Regenerate
                    # these only; canonical lattice bytes and identity stay exact.
                    regenerated.append(relative)
                    continue
                source = storage / "blobs" / digest
                target.parent.mkdir(parents=True, exist_ok=True)
                actual = hashlib.sha256()
                with source.open("rb") as incoming, target.open("xb") as output:
                    for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                        actual.update(chunk)
                        output.write(chunk)
                if actual.hexdigest() != digest:
                    raise ValueError(f"Checkpoint checksum mismatch: {relative}")
                if (parts[:2] == ("workspace", "artifacts") and len(parts) == 3
                        and parts[-1].startswith("expert-execution-") and parts[-1].endswith(".json")):
                    receipts[Path(relative).as_posix()] = digest
                # Preserve ordering used to select the latest saved result.
                modified = payload.get("file_mtime_ns", {}).get(relative, source.stat().st_mtime_ns)
                os.utime(target, ns=(modified, modified))
            conversation = staging / "conversation"
            conversation.mkdir()
            for exchange_record in payload["conversation"]:
                exchange = exchange_record["payload"]
                item = {**exchange, "project_id": new_id}
                if exchange_record["filename"] == f"{checkpoint_id}.json" and item.get("messages"):
                    # Keep this useful point actionable if the next provider
                    # connection fails before any new scientific progress.
                    messages = list(item["messages"])
                    last = messages[-1]
                    messages[-1] = {**last, "meta": {**last.get("meta", {}),
                        "conversation_checkpoint": checkpoint_id, "checkpoint_project_id": project_id}}
                    item["messages"] = messages
                # Old absolute download URLs are display references, not science data.
                rendered = json.dumps(item, ensure_ascii=False)
                rendered = rendered.replace(f"/api/projects/{project_id}/", f"/api/projects/{new_id}/")
                rendered = rendered.replace(json.dumps(str(directory))[1:-1],
                                            json.dumps(str(destination))[1:-1])
                rendered = rendered.replace(directory.as_posix(), destination.as_posix())
                filename = exchange_record["filename"]
                if Path(filename).name != filename or not re.fullmatch(r"[0-9]+-[0-9a-f]+\.json", filename):
                    raise ValueError("Invalid saved conversation filename")
                atomic_write_text(conversation / filename, rendered)
            previous_origin = staging / "workspace" / "RESTORED_CHECKPOINT.json"
            inherited = json.loads(previous_origin.read_text(encoding="utf-8")) if previous_origin.is_file() else {}
            chain = [{"project_id": project_id, "directory": str(directory)}]
            chain.extend(inherited.get("source_chain") or ([{
                "project_id": inherited["source_project"], "directory": inherited["source_directory"]
            }] if inherited.get("source_project") and inherited.get("source_directory") else []))
            origin = {"source_project": project_id, "source_checkpoint": checkpoint_id,
                "source_directory": str(directory), "branch_directory": str(destination),
                "saved_at": payload["captured_at"], "running_jobs_restored": False,
                "continuation": bool(continuation), "reset_operational_files": reset,
                "source_chain": chain, "restored_file_sha256": receipts,
                "regenerated_project_archive_sidecars": regenerated,
                "note": "Scientific files are byte-preserved. Old absolute paths in saved programs must be adapted to this workspace before execution. Resume numerical campaigns only from their saved optimizer checkpoints; no running process or provider thread is restored."}
            atomic_write_text(staging / "workspace" / "RESTORED_CHECKPOINT.json", json.dumps(origin, indent=2))
            now = datetime.now(timezone.utc).isoformat()
            project = replace(ResearchProject.from_dict(payload["project"]), id=new_id,
                revision=1, created_at=now, updated_at=now, status="needs_input",
                title=str(payload["project"]["title"]) + ("" if continuation else " (checkpoint branch)"),
                next_action=("Continue the authorized research from the selected saved conversation."
                             if continuation else "Restored saved checkpoint. Send an instruction to resume; no numerical jobs were restarted."),
                context=str(payload["project"].get("context", "")) + "\n" + json.dumps({
                    "source_project": project_id, "source_checkpoint": checkpoint_id,
                    "provenance": "workspace/RESTORED_CHECKPOINT.json"}))
            # Publish only after the branch can be loaded, including its revision
            # and regenerated archive sidecars. A crash cannot expose half a project.
            ProjectStore(temporary_root).save(project)
            _publish_branch(staging, destination)
            return project
        finally:
            # Only our explicit, freshly-created staging directory is removed.
            # Cleanup must not turn a published branch into a reported failure,
            # or obscure the original restore error.
            try:
                if temporary_root.exists():
                    shutil.rmtree(temporary_root)
            except OSError:
                logging.getLogger(__name__).warning(
                    "Could not clean checkpoint staging folder %s",
                    temporary_root, exc_info=True,
                )
