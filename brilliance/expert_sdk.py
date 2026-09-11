"""Small public authoring API for Brilliance expert tools.

Experts write normally typed Python functions.  The decorators adapt them to
the internal execution boundary and retain enough metadata for module discovery
to generate the tool contract.  No registry or manifest knowledge is required.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Mapping,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)


@dataclass(frozen=True, slots=True)
class Field:
    """Optional physical and numerical constraints for one tool argument."""

    description: str = ""
    unit: str = ""
    minimum: float | int | None = None
    maximum: float | int | None = None
    choices: tuple[Any, ...] = ()


def tool(
    *,
    id: str | None = None,
    title: str | None = None,
    authority: str = "provisional",
    output: str | None = None,
    requires_lattice: bool = False,
    timeout_s: int = 600,
    parameters: Mapping[str, Field | Mapping[str, Any]] | None = None,
    state_scope: str = "",
    stateful_parameters: tuple[str, ...] = (),
    inherit_from: Mapping[str, str] | None = None,
) -> Callable[[Callable[..., Mapping[str, Any]]], Callable[..., Mapping[str, Any]]]:
    """Declare a normally typed function as an expert tool.

    A function may optionally receive ``context`` as its first parameter.  All
    remaining parameters become the public JSON argument schema.
    """

    def decorate(function: Callable[..., Mapping[str, Any]]) -> Callable[..., Mapping[str, Any]]:
        if inspect.iscoroutinefunction(function):
            raise TypeError("expert tools must be synchronous functions")
        signature = inspect.signature(function)
        parameters_by_name = list(signature.parameters.values())
        uses_context = bool(
            parameters_by_name and parameters_by_name[0].name in {"context", "ctx"}
        )
        public_parameters = parameters_by_name[1:] if uses_context else parameters_by_name
        schema = _schema_from_parameters(
            function,
            public_parameters,
            overrides=dict(parameters or {}),
        )

        def adapter(context: Any, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            keyword_arguments = dict(arguments)
            if uses_context:
                return function(context, **keyword_arguments)
            return function(**keyword_arguments)

        adapter.__name__ = function.__name__
        adapter.__qualname__ = function.__qualname__
        adapter.__doc__ = function.__doc__
        adapter.__module__ = function.__module__
        adapter.__brilliance_tool__ = {  # type: ignore[attr-defined]
            "id": id,
            "title": title,
            "authority": str(authority),
            "output": output,
            "requires_lattice": bool(requires_lattice),
            "timeout_s": int(timeout_s),
            "parameter_schema": schema,
            "state_scope": str(state_scope).strip().lower(),
            "stateful_parameters": tuple(str(item) for item in stateful_parameters),
            "inherit_from": {
                str(name): str(scope).strip().lower()
                for name, scope in dict(inherit_from or {}).items()
            },
            "description": inspect.getdoc(function) or "",
        }
        adapter.__brilliance_original__ = function  # type: ignore[attr-defined]
        return adapter

    return decorate


def validator(
    *,
    for_tool: str,
) -> Callable[[Callable[..., Mapping[str, Any]]], Callable[..., Mapping[str, Any]]]:
    """Register a separate validator for a decorated tool.

    The expert may write ``validate(arguments, result)`` or
    ``validate(context, arguments, result)``.  Brilliance adapts either form to
    its internal validator boundary.
    """

    if not str(for_tool).strip():
        raise ValueError("validator for_tool must not be empty")

    def decorate(function: Callable[..., Mapping[str, Any]]) -> Callable[..., Mapping[str, Any]]:
        if inspect.iscoroutinefunction(function):
            raise TypeError("expert validators must be synchronous functions")
        count = len(inspect.signature(function).parameters)
        if count not in {2, 3}:
            raise TypeError(
                "expert validators accept (arguments, result) or "
                "(context, arguments, result)"
            )

        def adapter(
            context: Any,
            arguments: Mapping[str, Any],
            result: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            if count == 3:
                return function(context, arguments, result)
            return function(arguments, result)

        adapter.__name__ = function.__name__
        adapter.__qualname__ = function.__qualname__
        adapter.__doc__ = function.__doc__
        adapter.__module__ = function.__module__
        adapter.__brilliance_validator__ = {  # type: ignore[attr-defined]
            "for_tool": str(for_tool).strip(),
        }
        adapter.__brilliance_original__ = function  # type: ignore[attr-defined]
        return adapter

    return decorate


def python_tool(
    *, legacy_names: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Export an ordinary Python function in this expert's in-process library.

    Unlike ``tool``, this does not adapt arguments to JSON or change the callable.
    This is suitable for optimizer callbacks, arrays, and numerical objects. The
    optional legacy aliases keep archived programs callable while a module is
    installed; new programs should use ``api.expert_module(module_id).function``.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(function) or inspect.iscoroutinefunction(function):
            raise TypeError("expert Python tools must be synchronous callables")
        aliases = tuple(str(item).strip() for item in legacy_names)
        if any(not item.isidentifier() or item.startswith("_") for item in aliases):
            raise ValueError("legacy Python tool names must be public Python identifiers")
        if len(set(aliases)) != len(aliases):
            raise ValueError("duplicate legacy Python tool names")
        function.__brilliance_python_tool__ = {  # type: ignore[attr-defined]
            "legacy_names": aliases,
        }
        return function

    return decorate


def acceptance(
    *,
    id: str,
    applies_to: Callable[[str], bool],
    instructions: str,
    preflight: Callable[[str], list[str]] | None = None,
    requires_lattice: bool = True,
) -> Callable[[Callable[..., list[str]]], Callable[..., list[str]]]:
    """Declare a module-owned scientific request/evidence acceptance check.

    Decorate ``validate(report, lattice) -> list[str]``. An empty list means the
    evidence passes; otherwise return concrete failures. ``applies_to`` selects
    requests and ``instructions`` explains the required evidence to the agent.
    ``preflight`` may return advisory warnings for a saved Python program. Neither
    selection nor validation runs the simulation or prescribes a core workflow.
    """

    if not str(id).strip() or not str(instructions).strip():
        raise ValueError("expert acceptance needs an id and instructions")
    for label, function in (("applies_to", applies_to), ("preflight", preflight)):
        if function is not None and (
            not callable(function) or inspect.iscoroutinefunction(function)
        ):
            raise TypeError(f"expert acceptance {label} must be a synchronous callable")
    if not callable(applies_to):
        raise TypeError("expert acceptance applies_to is required")

    def decorate(function: Callable[..., list[str]]) -> Callable[..., list[str]]:
        if not callable(function) or inspect.iscoroutinefunction(function):
            raise TypeError("expert acceptance validators must be synchronous callables")
        try:
            inspect.signature(function).bind({}, None)
        except TypeError as exc:
            raise TypeError("expert acceptance validators accept (report, lattice)") from exc
        function.__brilliance_acceptance__ = {  # type: ignore[attr-defined]
            "id": str(id).strip(),
            "applies_to": applies_to,
            "instructions": str(instructions).strip(),
            "preflight": preflight,
            "requires_lattice": bool(requires_lattice),
        }
        return function

    return decorate


def _schema_from_parameters(
    function: Callable[..., Any],
    parameters: list[inspect.Parameter],
    *,
    overrides: Mapping[str, Field | Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        hints = get_type_hints(function, include_extras=True)
    except (NameError, TypeError) as exc:
        raise TypeError(
            f"cannot resolve type hints for expert tool {function.__name__!r}: {exc}"
        ) from exc
    properties: dict[str, Any] = {}
    required: list[str] = []
    names = {item.name for item in parameters}
    unknown_overrides = sorted(set(overrides) - names)
    if unknown_overrides:
        raise TypeError(
            f"expert tool {function.__name__!r} has constraints for unknown parameters: "
            f"{unknown_overrides}"
        )
    for parameter in parameters:
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }:
            raise TypeError(
                f"expert tool {function.__name__!r} uses unsupported parameter "
                f"{parameter.name!r}; use named parameters"
            )
        if parameter.name not in hints:
            raise TypeError(
                f"expert tool {function.__name__!r} parameter {parameter.name!r} "
                "needs a type annotation"
            )
        annotation = hints[parameter.name]
        field_metadata: Field | Mapping[str, Any] | None = overrides.get(parameter.name)
        if get_origin(annotation) is Annotated:
            annotation, *extras = get_args(annotation)
            annotated_field = next((item for item in extras if isinstance(item, Field)), None)
            if field_metadata is None:
                field_metadata = annotated_field
        rule = _annotation_schema(annotation)
        if field_metadata is not None:
            rule.update(_field_schema(field_metadata))
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
        else:
            rule["default"] = parameter.default
        properties[parameter.name] = rule
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in {Union, types.UnionType}:
        non_null = [item for item in arguments if item is not type(None)]
        if len(non_null) == 1:
            rule = _annotation_schema(non_null[0])
            rule["nullable"] = True
            return rule
        raise TypeError("expert tool union annotations must only express Optional[T]")
    if origin is Literal:
        values = list(arguments)
        if not values:
            raise TypeError("empty Literal is not a valid expert parameter")
        schema = _annotation_schema(type(values[0]))
        schema["enum"] = values
        return schema
    primitive = {
        float: "number",
        int: "integer",
        str: "string",
        bool: "boolean",
        dict: "object",
        Mapping: "object",
        Any: "object",
    }
    if annotation in primitive:
        return {"type": primitive[annotation]}
    if is_typeddict(annotation):
        try:
            hints = get_type_hints(annotation, include_extras=True)
        except (NameError, TypeError) as exc:
            raise TypeError(
                f"cannot resolve TypedDict {annotation.__name__!r}: {exc}"
            ) from exc
        required_keys = set(getattr(annotation, "__required_keys__", hints))
        return {
            "type": "object",
            "properties": {
                name: _annotation_schema(value) for name, value in hints.items()
            },
            "required": sorted(str(name) for name in required_keys),
            "additionalProperties": False,
        }
    if origin in {dict, Mapping, ABCMapping}:
        return {"type": "object"}
    if origin in {list, tuple, ABCSequence}:
        rule: dict[str, Any] = {"type": "array"}
        if arguments and arguments[0] is not Ellipsis:
            rule["items"] = _annotation_schema(arguments[0])
        return rule
    raise TypeError(
        f"unsupported expert parameter annotation {annotation!r}; use float, int, str, bool, "
        "list[T], dict, Literal, or Optional[T]"
    )


def _field_schema(value: Field | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    rendered: dict[str, Any] = {}
    if value.description:
        rendered["description"] = value.description
    if value.unit:
        rendered["unit"] = value.unit
    if value.minimum is not None:
        rendered["minimum"] = value.minimum
    if value.maximum is not None:
        rendered["maximum"] = value.maximum
    if value.choices:
        rendered["enum"] = list(value.choices)
    return rendered


__all__ = ["Field", "acceptance", "python_tool", "tool", "validator"]
