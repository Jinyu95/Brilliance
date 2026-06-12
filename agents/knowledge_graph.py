"""Lightweight conversation-parsing utilities."""

from __future__ import annotations

import re
from typing import Any


def slugify(text: str, *, limit: int = 64) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (normalized or "item")[:limit]


def latest_fenced_block(messages: list[dict[str, Any]], block_name: str) -> str | None:
    pattern = re.compile(rf"```{re.escape(block_name)}\s*\n(.*?)```", re.DOTALL)
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
    return None


def parse_colon_block(block_text: str | None) -> dict[str, str]:
    if not block_text:
        return {}
    parsed: dict[str, str] = {}
    for raw_line in block_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def parse_list_field(value: str | None) -> list[str]:
    if not value:
        return []
    text = value.strip()
    if text.lower() in {"none", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip(" '\"") for part in re.split(r"\s*,\s*", text) if part.strip()]
    return [part for part in parts if part.lower() != "none"]
