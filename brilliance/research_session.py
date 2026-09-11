"""Client-independent session views and safe, inspectable research programs.

Physics workers and model providers do not depend on the web interface. A client
can reconnect from a snapshot; subsequent events contain only changed messages.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


def conversation_context(rows: Sequence[Mapping[str, Any]], *, limit: int = 24000) -> list[dict[str, str]]:
    """Bounded public conversation only; never reinsert tool traces or private reasoning."""
    result: list[dict[str, str]] = []
    remaining = limit
    for row in reversed(rows):
        role = row.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(row.get("content") or "")[:6000]
        if not content or remaining <= 0:
            continue
        content = content[:remaining]
        result.append({"role": str(role), "content": content})
        remaining -= len(content)
    return list(reversed(result))


def snapshot_delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    old = {item["id"]: item for item in previous.get("messages", [])}
    rows = current.get("messages", [])
    return {
        **{key: value for key, value in current.items() if key != "messages"},
        "upsert": [row for row in rows if old.get(row["id"]) != row],
        "message_order": [row["id"] for row in rows],
    }


class SessionWorkspace:
    """Only research programs are exposed; no credentials, traces, or arbitrary files."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.programs = self.directory / "workspace" / "programs"
        self.pause_file = self.directory / "agent_pause.request"

    def _program(self, name: str) -> Path:
        if not name or Path(name).name != name or not name.endswith(".py") or ":" in name or "\\" in name:
            raise ValueError("Choose a Python filename inside workspace/programs")
        path = (self.programs / name).resolve()
        if not path.is_relative_to(self.directory) or path.parent != self.programs.resolve():
            raise ValueError("Program path escapes the research workspace")
        return path

    def list_programs(self) -> list[dict[str, Any]]:
        rows = []
        for candidate in sorted(self.programs.glob("*.py")):
            try:
                path = self._program(candidate.name)
                rows.append({"name": path.name, "size": path.stat().st_size})
            except (ValueError, OSError):
                continue
        return rows

    def read_program(self, name: str) -> dict[str, Any]:
        path = self._program(name)
        if not path.is_file():
            raise LookupError("Program not found")
        if path.stat().st_size > 1_000_000:
            raise ValueError("Program exceeds the editor's 1 MB limit")
        data = path.read_bytes()
        return {"name": name, "source": data.decode("utf-8"), "sha256": hashlib.sha256(data).hexdigest()}

    def save_draft(self, name: str, source: str) -> dict[str, Any]:
        path = self._program(name)
        if len(source.encode("utf-8")) > 1_000_000:
            raise ValueError("Program exceeds the editor's 1 MB limit")
        self.programs.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(source)
        except FileExistsError as exc:
            raise ValueError("This name already exists. Save a new draft; executed source is not overwritten.") from exc
        return self.read_program(name)
