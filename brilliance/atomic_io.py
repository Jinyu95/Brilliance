"""Small atomic-file primitive, including transient Windows sharing locks."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a file completely or raise, leaving the previous file intact.

    Each writer owns a unique sibling temporary file. Windows readers/indexers
    may briefly deny replacement; retry only access/sharing errors, not arbitrary
    I/O failures. This primitive never treats lost scientific data as success.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                return
            except OSError as exc:
                transient = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)
                if not transient or attempt == 4:
                    raise
                time.sleep(.01 * 2**attempt)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # Cleanup must not conceal the original failure (or undo success).
            pass
