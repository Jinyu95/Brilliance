"""Reject accidental capture of live simulation state by process recipes.

This is a call-contract check, not a Python sandbox. Numeric parameters and
module imports are portable; a Julia interpreter or mutable lattice is not.
"""

from __future__ import annotations

import ast
import functools
import types
from dataclasses import fields, is_dataclass
from typing import Any

_CONSTRUCTORS = frozenset({
    "jutrack", "build_jutrack_editor", "build_jutrack_lattice",
    "build_jutrack_profile_sampler", "JuTrackLatticeEditor", "JuTrackProfileSampler",
})


def lint_worker_factory_captures(tree: ast.AST) -> list[str]:
    """Find direct and helper-mediated module-global simulation captures."""
    if not isinstance(tree, ast.Module):
        return []
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value

    def dependencies(node: ast.AST) -> set[str]:
        loaded = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        bound = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        bound.update(n.arg for n in ast.walk(node) if isinstance(n, ast.arg))
        return loaded - bound

    def capture(name: str, visited: set[str]) -> list[str] | None:
        if name in visited:
            return None
        visited = visited | {name}
        expression = assignments.get(name)
        if expression is not None:
            for item in ast.walk(expression):
                if isinstance(item, ast.Call):
                    called = item.func.attr if isinstance(item.func, ast.Attribute) else getattr(item.func, "id", "")
                    if called in _CONSTRUCTORS:
                        return [name]
        node = definitions.get(name, expression)
        if node is not None:
            for dependency in sorted(dependencies(node)):
                path = capture(dependency, visited)
                if path:
                    return [name, *path]
        return None

    issues: list[str] = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        factories = [kw.value for kw in call.keywords
                     if kw.arg in {"worker_physics_factory", "evaluator_factory"}]
        name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
        if name in {"ProcessBatchEvaluator", "ParallelEvaluator", "parallel_evaluator", "process_batch_evaluator"} and call.args:
            factories.append(call.args[0])
        for factory in factories:
            if isinstance(factory, ast.Name) and (path := capture(factory.id, set())):
                issues.append(
                    f"line {call.lineno}: process factory captures parent-owned JuTrack state via "
                    + " -> ".join(path)
                    + "; construct the editor/sampler inside the worker factory and define its "
                    "callbacks there. Only plain lattice rows and numeric settings may cross processes."
                )
    return issues


def _code_globals(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            names.update(_code_globals(value))
    return names


def validate_worker_factory(factory: Any) -> None:
    """Check live references without importing Julia, invoking, or pickling them."""
    from .population_optimization import OptimizerInputError

    seen: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        cls = type(value)
        module = cls.__module__.casefold()
        if (module.startswith(("juliacall", "julia.", "_juliacall"))
                or cls.__name__ in {"JuTrackLatticeEditor", "JuTrackProfileSampler"}):
            raise OptimizerInputError(
                f"process factory captures parent-owned simulation state at {path} "
                f"({cls.__name__}); create JuTrack editors/samplers inside the worker factory, "
                "including those used by objective helper functions"
            )
        if value is None or isinstance(value, (str, bytes, int, float, bool, types.ModuleType, type)):
            return
        if isinstance(value, types.FunctionType):
            for name in _code_globals(value.__code__):
                if name in value.__globals__:
                    visit(value.__globals__[name], f"{path}.{name}")
            for index, cell in enumerate(value.__closure__ or ()):
                try:
                    contents = cell.cell_contents
                except ValueError:  # empty closure cell
                    continue
                visit(contents, f"{path}.closure[{index}]")
            visit(value.__defaults__, f"{path}.defaults")
            visit(value.__kwdefaults__, f"{path}.kwdefaults")
        elif isinstance(value, types.MethodType):
            visit(value.__self__, f"{path}.self")
            visit(value.__func__, f"{path}.method")
        elif isinstance(value, functools.partial):
            visit(value.func, f"{path}.func")
            visit(value.args, f"{path}.args")
            visit(value.keywords, f"{path}.keywords")
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}[{str(key)[:60]}]")
        elif isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}")
        elif hasattr(value, "__dict__"):
            visit(vars(value), f"{path}.state")

    visit(factory, "worker_factory")
