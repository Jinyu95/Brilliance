"""Deterministic admission control before an expensive autonomous research turn."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .research_environment import ResearchEnvironment, ResearchEnvironmentError

_DESIGN_ACTION = re.compile(
    r"\b(design|build|construct|create|develop|optimi[sz]e|match)\w*\b",
    re.IGNORECASE,
)
_COMPLEX_LATTICE = re.compile(
    r"\b(?:(?:[2-9]|[1-9]\d+)\s*ba|dba|tba|mba|multi[- ]?bend|hybrid[- ]?achromat|"
    r"reverse[- ]?bend|anti[- ]?bend|dlsr|diffraction[- ]?limited)\b",
    re.IGNORECASE,
)
_ONLINE_REFERENCE_ACQUISITION = re.compile(
    r"\b(search|browse|download|fetch)\w*\b.{0,100}"
    r"\b(reference|seed|lattice|model|file|github|gitlab|sourceforge|web|internet|online)\b|"
    r"\b(reference|seed|lattice|model|file|github|gitlab|sourceforge|web|internet|online)\b"
    r".{0,100}\b(search|browse|download|fetch)\w*\b|"
    r"\b(find|locate|acquire)\w*\b.{0,100}\b(on(?:line)?|internet|web|github|gitlab|sourceforge)\b",
    re.IGNORECASE | re.DOTALL,
)
_CONTINUATION_ONLY = re.compile(
    r"^\s*(?:go\s+ahead|continue|proceed|start|begin|do\s+it|yes|okay|ok)[.!\s]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ResearchAdmission:
    allowed: bool
    mode: str
    code: str
    detail: str


def complex_greenfield_request(text: str) -> bool:
    normalized = " ".join(str(text).split())
    return bool(_DESIGN_ACTION.search(normalized) and _COMPLEX_LATTICE.search(normalized))


def reference_acquisition_request(text: str) -> bool:
    """Return true only for an explicit online-source request.

    Facility names, "MAX-IV-like", "use this lattice", and local imports are design
    instructions, not permission to access the internet.
    """

    return bool(_ONLINE_REFERENCE_ACQUISITION.search(" ".join(str(text).split())))


def assess_research_admission(
    project: Any,
    project_root: Path | str,
    message: str,
    *,
    internet_access: bool | None = None,
) -> ResearchAdmission:
    """Choose a safe entry mode for the *current* research request.

    Admission is deliberately narrow.  A project's original goal may describe a
    complex lattice, but that must not turn every later plot, analysis, tracking,
    or hypothesis request into another greenfield-design turn.
    """

    project_directory = Path(project_root).resolve() / str(project.id)
    goal = str(getattr(project, "goal", ""))
    explicit = reference_acquisition_request(message)
    try:
        environment = ResearchEnvironment.load()
    except ResearchEnvironmentError:
        environment = None
    if internet_access is None:
        internet_access = environment.internet_access if environment is not None else False
    if explicit:
        if not internet_access:
            return ResearchAdmission(
                False,
                "needs_input",
                "RESEARCH_NETWORK_DISABLED",
                (
                    "Internet access is disabled in research_runtime.toml, so I did not search, "
                    "download, call an online connector, or run a network command. Enable "
                    "execution.internet_access or place the reference file in the project."
                ),
            )
        return ResearchAdmission(
            True,
            "reference_acquisition",
            "REFERENCE_ACQUISITION",
            "Acquire and validate a traceable executable reference; do not invent a lattice.",
        )
    if (
        getattr(project, "active_lattice_id", None)
        or getattr(project, "evidence", ())
        or _has_working_research_state(project_directory)
    ):
        return ResearchAdmission(True, "research", "ESTABLISHED_PROJECT_EVIDENCE", "")
    effective_request = goal if _CONTINUATION_ONLY.fullmatch(message) else message
    if not complex_greenfield_request(effective_request):
        return ResearchAdmission(True, "research", "GENERAL_RESEARCH", "")

    # A coding agent does not need a pre-registered workflow for every complex
    # accelerator question.  It needs relevant knowledge, a programmable
    # JuTrack boundary, and honest evidence.  Design bases remain optional
    # comparison/validation assets; their absence must not prevent the model
    # from constructing and testing a provisional hypothesis offline.
    return ResearchAdmission(
        True,
        "research",
        "OPEN_ENDED_SCIENTIFIC_PROGRAM",
        (
            "Use retrieved accelerator knowledge and a saved task-specific JuTrack program. "
            "Treat every new design as provisional until its declared numerical and physics "
            "checks pass; no fixed construction workflow is implied."
        ),
    )


def _has_working_research_state(project_directory: Path) -> bool:
    """Recognize preserved provisional work without interpreting its domain."""

    workspace = project_directory / "workspace"
    if (workspace / "WORKING_LATTICE.json").is_file():
        return True
    state = workspace / "research_state" / "records"
    return state.is_dir() and any(state.glob("*.json"))
