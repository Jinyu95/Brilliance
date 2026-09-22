"""Traceable source records and a deterministic local research corpus."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import content_hash, json_safe

SOURCE_SCHEMA_VERSION = "1.0"
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    title: str
    locator: str
    source_type: str
    value_class: str
    excerpt: str
    content_sha256: str
    line_start: int | None = None
    line_end: int | None = None
    doi: str | None = None
    url: str | None = None
    schema_version: str = SOURCE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


class LocalResearchCorpus:
    """Search local Markdown/text evidence with exact excerpt provenance.

    This is intentionally small and deterministic.  It provides honest local
    retrieval now and the same ``SourceRecord`` contract can later be populated
    by DOI, arXiv, laboratory-report, or web connectors.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"research corpus directory does not exist: {self.root}")

    def search(self, query: str, *, limit: int = 5) -> list[SourceRecord]:
        if limit < 1:
            raise ValueError("source search limit must be positive")
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        candidates: list[tuple[float, SourceRecord]] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            for start, end, excerpt in _paragraphs(lines):
                paragraph_tokens = _tokens(excerpt)
                overlap = query_tokens & paragraph_tokens
                if not overlap:
                    continue
                coverage = len(overlap) / len(query_tokens)
                density = len(overlap) / max(math.sqrt(len(paragraph_tokens)), 1.0)
                heading_bonus = 0.2 if excerpt.lstrip().startswith("#") else 0.0
                score = coverage + density + heading_bonus
                relative = path.relative_to(self.root).as_posix()
                record_data = {
                    "locator": relative,
                    "line_start": start,
                    "line_end": end,
                    "excerpt": excerpt,
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
                candidates.append((score, SourceRecord(
                    source_id="source-" + content_hash(record_data, length=16),
                    title=_document_title(lines, path.stem),
                    locator=f"{relative}:{start}-{end}",
                    source_type="local_summary",
                    value_class="secondary_summary",
                    excerpt=excerpt,
                    content_sha256=record_data["sha256"],
                    line_start=start,
                    line_end=end,
                )))
        candidates.sort(key=lambda item: (-item[0], item[1].locator))
        unique: list[SourceRecord] = []
        seen_files: set[str] = set()
        for _, record in candidates:
            filename = record.locator.split(":", 1)[0]
            if filename in seen_files and len(unique) >= max(1, limit // 2):
                continue
            seen_files.add(filename)
            unique.append(record)
            if len(unique) >= limit:
                break
        return unique

    def verify(self, record: Mapping[str, Any]) -> tuple[bool, str]:
        """Verify a local source record against the current corpus bytes."""
        if record.get("source_type") != "local_summary":
            return True, "Non-local source requires its connector-specific verifier."
        locator = str(record.get("locator", ""))
        relative = locator.split(":", 1)[0]
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return False, "Source locator escapes the corpus root."
        if not path.is_file():
            return False, "Source file does not exist."
        text = path.read_text(encoding="utf-8", errors="replace")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != record.get("content_sha256"):
            return False, "Source content hash changed."
        start, end = record.get("line_start"), record.get("line_end")
        if not isinstance(start, int) or not isinstance(end, int):
            return False, "Source line span is missing."
        actual = "\n".join(text.splitlines()[start - 1:end]).strip()
        if actual != str(record.get("excerpt", "")).strip():
            return False, "Source excerpt does not match its line span."
        return True, "Local source record verified."




def _tokens(text: str) -> set[str]:
    return {item.lower() for item in _TOKEN.findall(text)}


def _paragraphs(lines: list[str]) -> Iterable[tuple[int, int, str]]:
    start: int | None = None
    buffer: list[str] = []
    for index, line in enumerate(lines, start=1):
        if line.strip():
            if start is None:
                start = index
            buffer.append(line)
        elif buffer and start is not None:
            yield start, index - 1, "\n".join(buffer).strip()
            start, buffer = None, []
    if buffer and start is not None:
        yield start, len(lines), "\n".join(buffer).strip()


def _document_title(lines: list[str], fallback: str) -> str:
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return fallback.replace("_", " ").strip().title()
