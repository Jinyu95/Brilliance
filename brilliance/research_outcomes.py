"""Typed negative scientific outcomes, distinct from execution/API failures."""
from __future__ import annotations

from typing import Any, Mapping


class ResearchInfeasible(ValueError):
    """A bounded scientific attempt found no admissible result.

    Does not mean the machine requirements are impossible or a topology exhausted.
    The program runner archives the evidence and returns it to the research agent;
    ordinary TypeError/ValueError/infrastructure failures remain execution errors.
    Experts can use this for other accelerator topics without a design state machine.
    """

    def __init__(self, message: str, *, code: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.code = code
        self.diagnostics = dict(diagnostics)
