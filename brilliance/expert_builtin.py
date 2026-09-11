"""General saved-program tools shipped by the built-in research kernel."""

from __future__ import annotations

from typing import Any, Mapping

from .research_program import run_research_program, validate_research_program


def execute_research_program(context: Any, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """Execute one short saved research program in the configured runtime."""

    return run_research_program(context, arguments)


def execute_long_research_program(context: Any, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """Execute one durable saved research program in a managed worker."""

    return run_research_program(context, arguments, durable=True)


def validate_saved_research_program(
    context: Any,
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    return validate_research_program(context, arguments, result)
