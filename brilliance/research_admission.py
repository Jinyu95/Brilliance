"""Conversation entry point; scientific routes are chosen by the model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ResearchAdmission:
    allowed: bool
    mode: str
    code: str
    detail: str


def assess_research_admission(
    project: Any,
    project_root: Path | str,
    message: str,
    *,
    internet_access: bool | None = None,
) -> ResearchAdmission:
    """Do not turn keywords into a scientific route or required design basis.

    Network permission belongs to the configured tools/worker, not a natural-
    language classifier. An offline model can explain a limitation or use local
    evidence without the entire conversation being rejected before it starts.
    Runtime readiness is checked when the configured worker is used.
    """
    return ResearchAdmission(True, "research", "MODEL_LED_RESEARCH", "")
