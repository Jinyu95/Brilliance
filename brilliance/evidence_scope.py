"""Explicit read-only reuse of earlier numerical work in the same project."""
from __future__ import annotations

import os
from pathlib import Path


def evidence_source_root(current_root: Path | str,
                         source_output_directory: Path | str | None = None) -> Path:
    """Resolve an archive without allowing another project or arbitrary files.

    Outputs never move: callers verify the original descriptors and retain their
    calculation identity when publishing into the current run. This is an
    accidental-provenance guard, not a sandbox against malicious Python code.
    """
    current = Path(current_root).resolve()
    if source_output_directory is None:
        return current
    source = Path(source_output_directory).resolve()
    if source == current:
        return current
    workspace_value = os.environ.get("BRILLIANCE_WORKSPACE")
    if not workspace_value:
        raise ValueError("Cross-run evidence reuse requires the configured current project workspace")
    programs = (Path(workspace_value).resolve() / "artifacts" / "programs").resolve()
    for label, path in (("current", current), ("source", source)):
        try:
            relative = path.relative_to(programs)
        except ValueError as exc:
            raise ValueError("Evidence reuse cannot leave this project's managed program archives") from exc
        if len(relative.parts) != 1:
            raise ValueError(f"The {label} evidence root must be one managed program output directory")
    if not (source / "program.py").is_file():
        raise ValueError("The source numerical archive has no preserved research program")
    return source
