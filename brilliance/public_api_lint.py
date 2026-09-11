"""Check explicit public API calls without executing a research program or Julia."""
from __future__ import annotations

import ast
import inspect
import math
from typing import Any


def _expanded_keyword_keys(value: ast.AST, assignments: dict[str, ast.AST]) -> set[str] | None:
    """Known runtime kwargs must not hide missing required physics arguments."""
    if isinstance(value, ast.Name):
        value = assignments.get(value.id, value)
    if isinstance(value, ast.Dict) and all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in value.keys):
        return {key.value for key in value.keys}
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "optimization_options"):
        from .research_api import OptimizationControls
        return set(OptimizationControls.__required_keys__)
    return None


def _optimizer_shapes(node: ast.Call, assignments: dict[str, ast.AST]) -> list[str]:
    def resolve(value):
        if isinstance(value, ast.Name):
            return assignments.get(value.id, value)
        return value

    values = {item.arg: resolve(item.value) for item in node.keywords if item.arg}
    bounds, names = values.get("bounds"), values.get("variable_names")
    issues = []
    if isinstance(bounds, ast.Dict):
        pairs = bounds.values
        keys = [item.value for item in bounds.keys if isinstance(item, ast.Constant)]
        if isinstance(names, (ast.List, ast.Tuple)) and all(isinstance(item, ast.Constant) for item in names.elts):
            order = [item.value for item in names.elts]
            if len(keys) == len(bounds.keys) and (set(keys) != set(order) or len(order) != len(set(order))):
                issues.append("named bounds must have exactly the unique variable_names keys")
    elif isinstance(bounds, (ast.List, ast.Tuple)):
        pairs = bounds.elts
    elif isinstance(bounds, (ast.Constant, ast.Set)):
        return ["bounds must be [lower, upper] pairs or a name-to-pair mapping"]
    else:
        return []  # Dynamic values are checked by the optimizer, never guessed here.
    for pair in pairs:
        if isinstance(pair, (ast.List, ast.Tuple)):
            if len(pair.elts) != 2:
                issues.append("each bound must contain exactly [lower, upper]")
            else:
                # Literal signed numbers are safe to check without executing
                # agent expressions or guessing dynamically computed limits.
                try:
                    lower, upper = (ast.literal_eval(item) for item in pair.elts)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    continue
                if all(isinstance(item, (int, float)) and not isinstance(item, bool)
                       for item in (lower, upper)):
                    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                        issues.append(f"bound [{lower}, {upper}] must be finite and satisfy lower < upper")
        elif isinstance(pair, (ast.Constant, ast.Dict, ast.Set)):
            issues.append("each bound must be a numeric [lower, upper] pair")
    return issues


def lint_public_api_calls(tree: ast.AST, *, registry: Any = None, project_directory: Any = None) -> list[str]:
    from . import accelerator_api

    modules: set[str] = set()
    functions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "brilliance":
            modules.update(item.asname or item.name for item in node.names if item.name == "accelerator_api")
        elif isinstance(node, ast.ImportFrom) and node.module == "brilliance.accelerator_api":
            functions.update({item.asname or item.name: item.name for item in node.names if item.name != "*"})
        elif isinstance(node, ast.Import):
            modules.update(item.asname for item in node.names if item.name == "brilliance.accelerator_api" and item.asname)
    issues = []
    assignments = {}
    nested_calls = {id(child) for scope in ast.walk(tree)
                    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))
                    for child in ast.walk(scope) if isinstance(child, ast.Call)}
    if isinstance(tree, ast.Module):
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                assignments[statement.targets[0].id] = statement.value

    def installed():
        nonlocal registry
        if registry is None:
            from .experts import ExpertRegistry
            registry = ExpertRegistry.current(project_directory=project_directory)
        return registry

    def library_id(value):
        if isinstance(value, ast.Name):
            value = assignments.get(value.id)
        if not isinstance(value, ast.Call):
            return None
        function = value.func
        factory = ((isinstance(function, ast.Attribute) and function.attr == "expert_module"
                    and isinstance(function.value, ast.Name) and function.value.id in modules)
                   or (isinstance(function, ast.Name) and functions.get(function.id) == "expert_module"))
        if not factory:
            return None
        argument = value.args[0] if value.args else next((item.value for item in value.keywords if item.arg == "module_id"), None)
        return argument.value if isinstance(argument, ast.Constant) and isinstance(argument.value, str) else None

    # Do not mistake a local parameter/rebinding for the global expert namespace.
    shadowed: dict[int, set[str]] = {}
    for scope in ast.walk(tree):
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            locals_ = {item.arg for item in ast.walk(scope.args) if isinstance(item, ast.arg)}
            locals_.update(item.id for item in ast.walk(scope) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store))
            for call in ast.walk(scope):
                if isinstance(call, ast.Call):
                    shadowed.setdefault(id(call), set()).update(locals_)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            owner = "api"
            module_id = None
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in modules:
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = functions.get(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                value = node.func.value
                if not (isinstance(value, ast.Name) and value.id in shadowed.get(id(node), ())):
                    module_id = library_id(value)
                if module_id is not None:
                    name = node.func.attr
                    owner = f"api.expert_module({module_id!r})"
            if name is None:
                continue
            target = None
            try:
                if module_id is not None:
                    target = getattr(installed().python_library(module_id), name, None)
                else:
                    target = vars(accelerator_api).get(name)
                    if target is None and not name.startswith("_"):
                        target = installed().legacy_python_tool(name)
            except LookupError:
                pass
            if target is None or name.startswith("_"):
                issues.append(f"line {node.lineno}: {owner} has no installed public function {name!r}; use api_help")
                continue
            if not callable(target):
                continue
            if name == "expert_module" and module_id is None and library_id(node) is not None:
                try:
                    installed().module(library_id(node))
                except LookupError:
                    issues.append(f"line {node.lineno}: expert module {library_id(node)!r} is not installed/enabled")
            if name == "particle_swarm_optimize":
                issues.extend(f"line {node.lineno}: {issue}" for issue in
                              _optimizer_shapes(node, {} if id(node) in nested_calls else assignments))
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            positional = [None for item in node.args if not isinstance(item, ast.Starred)]
            keywords = {item.arg: None for item in node.keywords if item.arg is not None}
            expanded = any(isinstance(item, ast.Starred) for item in node.args) or any(item.arg is None for item in node.keywords)
            try:
                unknown_expansion = any(isinstance(item, ast.Starred) for item in node.args)
                for item in node.keywords:
                    if item.arg is not None:
                        continue
                    keys = _expanded_keyword_keys(item.value, {} if id(node) in nested_calls else assignments)
                    if keys is None:
                        unknown_expansion = True
                    else:
                        duplicates = keys & keywords.keys()
                        if duplicates:
                            raise TypeError(f"duplicate expanded keyword(s): {', '.join(sorted(duplicates))}")
                        keywords.update({key: None for key in keys})
                if expanded and unknown_expansion:
                    # Expansions may supply required arguments. Do not guess them.
                    signature.bind_partial(**keywords)
                else:
                    signature.bind_partial(*positional, **keywords)
                    signature.bind(*positional, **keywords)
            except TypeError as exc:
                issues.append(f"line {node.lineno}: {owner}.{name}{signature}: {exc}")
        elif isinstance(node, ast.ExceptHandler):
            broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"})
            # Naming the exception does not make swallowing a TypeError safe.
            # An explicit re-raise branch can protect programming errors while
            # retaining documented physical rejection handling.
            if not broad or any(isinstance(statement, ast.Raise) for statement in ast.walk(node)):
                continue
            for statement in ast.walk(node):
                if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Dict):
                    keys = {key.value for key in statement.value.keys if isinstance(key, ast.Constant)}
                    if keys & {"feasible", "constraint_violations", "objectives"}:
                        issues.append(f"line {node.lineno}: broad exception converts an API/coding error into physics infeasibility. Catch documented physical failures only; let TypeError/AttributeError/KeyError surface, and preserve exact error diagnostics.")
                        break
    return issues
