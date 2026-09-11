"""Generic contracts for module-owned Python libraries and evidence checks."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .loading import load_expert_source
from .models import ExpertModule, _identifier


@dataclass(frozen=True, slots=True)
class ExpertAcceptanceCheck:
    id: str
    module_id: str
    validator: Callable[..., list[str]]
    applies_to: Callable[[str], bool]
    instructions: str
    preflight: Callable[[str], list[str]] | None = None
    requires_lattice: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "module_id": self.module_id,
            "instructions": self.instructions, "requires_lattice": self.requires_lattice,
            "has_preflight": self.preflight is not None,
        }


@dataclass(frozen=True, slots=True)
class ExpertPythonTool:
    name: str
    function: Callable[..., Any]
    legacy_names: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signature": str(inspect.signature(self.function)),
            "description": inspect.getdoc(self.function) or "",
            "legacy_names": list(self.legacy_names),
        }


class ExpertPythonLibrary:
    """Read-only namespace exposing only explicitly registered module functions."""

    __slots__ = ("_functions",)

    def __init__(self, tools: tuple[ExpertPythonTool, ...]):
        object.__setattr__(self, "_functions", MappingProxyType({item.name: item.function for item in tools}))

    def __getattr__(self, name: str) -> Callable[..., Any]:
        try:
            return self._functions[name]
        except KeyError as exc:
            raise AttributeError(f"expert library has no Python tool {name!r}") from exc

    def __setattr__(self, name, value):
        raise AttributeError("expert Python libraries are read-only")

    def __dir__(self):
        return sorted(self._functions)


def discover_extensions(module: ExpertModule) -> tuple[tuple[ExpertAcceptanceCheck, ...], tuple[ExpertPythonTool, ...]]:
    source = module.root / "tools.py"
    if not module.enabled or module.status == "deprecated" or not source.is_file():
        return (), ()
    imported = load_expert_source(module.root, source)
    checks, functions = [], []
    for name, function in vars(imported).items():
        if name.startswith("_"):
            continue
        metadata = getattr(function, "__brilliance_acceptance__", None)
        if isinstance(metadata, Mapping):
            checks.append(ExpertAcceptanceCheck(
                id=_identifier(metadata.get("id"), "acceptance check id"), module_id=module.id,
                validator=function, applies_to=metadata["applies_to"],
                instructions=metadata["instructions"], preflight=metadata.get("preflight"),
                requires_lattice=bool(metadata.get("requires_lattice", True)),
            ))
        metadata = getattr(function, "__brilliance_python_tool__", None)
        if isinstance(metadata, Mapping):
            functions.append(ExpertPythonTool(name, function, tuple(metadata.get("legacy_names", ()))))
    return tuple(checks), tuple(functions)
