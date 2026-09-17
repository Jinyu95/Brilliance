"""Check explicit public API calls without executing a research program or Julia."""
from __future__ import annotations

import ast
import collections.abc
import inspect
import math
import types
import typing
from typing import Any


def _implicit_binding(node):
    """Python bindings which are not represented by Name(Store)."""
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return node.name
    if isinstance(node, ast.MatchMapping):
        return node.rest
    return None


def _public_imports(tree):
    """Return only lexically proven public imports at each call site."""
    calls = {}
    boundaries = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    cross_scope = {name for node in ast.walk(tree) if isinstance(node, (ast.Global, ast.Nonlocal)) for name in node.names}

    def inspect_scope(scope, inherited):
        nodes, children = [], []
        def collect(node):
            nodes.append(node)
            if isinstance(node, boundaries):
                children.append(node)
            else:
                for child in ast.iter_child_nodes(node):
                    collect(child)
        for statement in scope.body if isinstance(scope.body, list) else [scope.body]:
            collect(statement)
        writes, imports = {}, {}
        def write(name):
            writes[name] = writes.get(name, 0) + 1
        if hasattr(scope, "args"):
            for arg in ast.walk(scope.args):
                if isinstance(arg, ast.arg):
                    write(arg.arg)
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                write(node.id)
            elif isinstance(node, boundaries) and hasattr(node, "name"):
                write(node.name)
            elif _implicit_binding(node):
                write(_implicit_binding(node))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                    write(name)
                    if (isinstance(node, ast.ImportFrom) and node.module == "brilliance"
                            and alias.name == "accelerator_api"):
                        imports[name] = ("module", None)
                    elif isinstance(node, ast.ImportFrom) and node.module == "brilliance.accelerator_api" and alias.name != "*":
                        imports[name] = ("function", alias.name)
                    elif isinstance(node, ast.Import) and alias.name == "brilliance.accelerator_api" and alias.asname:
                        imports[name] = ("module", None)
        known = {name: value for name, value in inherited.items() if name not in writes and name not in cross_scope}
        known.update({name: value for name, value in imports.items() if writes[name] == 1 and name not in cross_scope})
        for node in nodes:
            if isinstance(node, ast.Call):
                calls[id(node)] = known
        for child in children:
            inspect_scope(child, known if not isinstance(child, ast.ClassDef) else {})
    if isinstance(tree, ast.Module):
        inspect_scope(tree, {})
    return calls


def _readonly_arguments(node, target, signature):
    """Installed Mapping parameters promise a read-only consumer interface."""
    try:
        hints = typing.get_type_hints(inspect.unwrap(target), include_extras=True)
    except (NameError, TypeError, AttributeError, ValueError):
        return set()

    def readonly(annotation):
        origin = typing.get_origin(annotation)
        if origin in (typing.Union, types.UnionType):
            return all(readonly(item) for item in typing.get_args(annotation) if item is not type(None))
        return origin is collections.abc.Mapping

    try:
        bound = signature.bind_partial(*node.args, **{item.arg: item.value for item in node.keywords if item.arg})
    except TypeError:
        return set()
    return {id(value) for name, value in bound.arguments.items() if readonly(hints.get(name))}


def _literal_argument_issues(node, target, signature, owner):
    """Check literal values under explicitly closed input records, never infer code."""
    from .result_contracts import parameter_contracts

    contracts = parameter_contracts(target)
    try:
        positional = [] if any(isinstance(arg, ast.Starred) for arg in node.args) else node.args
        bound = signature.bind_partial(*positional, **{kw.arg: kw.value for kw in node.keywords if kw.arg})
    except TypeError:
        return []  # The ordinary signature checker reports this independently.
    issues = []

    def check(value, schema, label, active=False):
        kind = schema.get("type")
        active = active or bool(schema.get("closed"))
        if isinstance(value, ast.Constant) and value.value is None and schema.get("nullable"):
            return
        if kind == "object" and isinstance(value, ast.Dict):
            properties = schema.get("properties", {})
            literal_keys = {key.value for key in value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            if schema.get("closed"):
                for key in sorted(literal_keys - properties.keys()):
                    issues.append(f"line {value.lineno}: {label} has no installed input key {key!r}; "
                                  f"exact keys: {', '.join(properties)}")
                if len(literal_keys) == len(value.keys):
                    missing = set(schema.get("required", ())) - literal_keys
                    if missing:
                        issues.append(f"line {value.lineno}: {label} is missing required input keys: {', '.join(sorted(missing))}")
            for key, child in zip(value.keys, value.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in properties:
                    check(child, properties[key.value], f"{label}[{key.value!r}]", active)
        elif kind == "array" and isinstance(value, (ast.List, ast.Tuple)):
            for child in value.elts:
                check(child, schema.get("items", {}), label + "[i]", active)
        elif active and kind not in (None, "unknown"):
            if isinstance(value, ast.Constant):
                actual = ("null" if value.value is None else "boolean" if isinstance(value.value, bool)
                          else "number" if isinstance(value.value, (int, float))
                          else "string" if isinstance(value.value, str) else "other")
            elif isinstance(value, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                actual = "object" if isinstance(value, ast.Dict) else "array" if not isinstance(value, ast.Set) else "set"
            else:
                return  # Calls, names, operators and expansions are runtime values.
            if actual != kind:
                expected = kind + (" or null" if schema.get("nullable") else "")
                issues.append(f"line {value.lineno}: {label} requires {expected}; literal {actual} supplied")

    for name, value in bound.arguments.items():
        if name in contracts and isinstance(value, ast.AST):
            check(value, contracts[name], f"{owner} argument {name}")
    return issues


def _result_key_issues(tree, call_contracts, readonly_arguments):
    """Track only unmodified, unescaped installed return records and literal keys."""
    issues = []
    boundaries = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    cross_scope = {name for node in ast.walk(tree) if isinstance(node, (ast.Global, ast.Nonlocal)) for name in node.names}

    def root_name(node):
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            return root_name(node.func.value)
        return node.id if isinstance(node, ast.Name) else None

    def inspect_scope(scope, inherited, enclosing_bindings):
        nodes, children = [], []
        def collect(node):
            nodes.append(node)
            if isinstance(node, boundaries):
                children.append(node)
            else:
                for child in ast.iter_child_nodes(node):
                    collect(child)
        for statement in scope.body if isinstance(scope.body, list) else [scope.body]:
            collect(statement)
        writes, bindings, unsafe_at, aliases = {}, {}, {name: 0 for name in cross_scope}, []
        def invalidate(name, node):
            if name is not None:
                unsafe_at[name] = min(unsafe_at.get(name, math.inf), node.lineno)
        arguments = {arg.arg for arg in ast.walk(scope.args) if isinstance(arg, ast.arg)} if hasattr(scope, "args") else set()
        for name in arguments:
            writes[name] = 1
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                writes[node.id] = writes.get(node.id, 0) + 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    writes[name] = writes.get(name, 0) + 1
            elif isinstance(node, boundaries):
                if hasattr(node, "name"):
                    writes[node.name] = writes.get(node.name, 0) + 1
            elif _implicit_binding(node):
                name = _implicit_binding(node)
                writes[name] = writes.get(name, 0) + 1
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and node.value is not None:
                        bindings[target.id] = (node.value, False)
                        owner = root_name(node.value)
                        if owner is not None:
                            aliases.append((target.id, owner))
            if isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
                bindings[node.target.id] = (node.iter, True)
                owner = root_name(node.iter)
                if owner is not None:
                    aliases.append((node.target.id, owner))
            if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                invalidate(root_name(node), node)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in {
                        "update", "setdefault", "pop", "popitem", "clear", "append", "extend", "insert", "__setitem__", "__delitem__"}:
                    invalidate(root_name(node.func.value), node)
                # Arbitrary consumers may mutate a returned mapping. Do not
                # manufacture a closed shape after it escapes our analysis.
                for value in [*node.args, *(kw.value for kw in node.keywords)]:
                    if id(value) not in readonly_arguments.get(id(node), ()):
                        invalidate(root_name(value), node)
        unsafe_at.update({name: 0 for name, count in writes.items() if count > 1})
        for _ in aliases:
            for left, right in aliases:
                if left in unsafe_at or right in unsafe_at:
                    at = min(unsafe_at.get(left, math.inf), unsafe_at.get(right, math.inf))
                    unsafe_at[left] = unsafe_at[right] = at
        known = {name: shape for name, shape in inherited.items() if name not in writes and unsafe_at.get(name) != 0}
        shadowed_call_roots = arguments | (set(writes) & enclosing_bindings)

        def resolve(node):
            if isinstance(node, ast.Name):
                return known.get(node.id) if node.lineno < unsafe_at.get(node.id, math.inf) else None
            if isinstance(node, ast.Call) and id(node) in call_contracts:
                name = root_name(node.func)
                if name not in shadowed_call_roots and node.lineno < unsafe_at.get(name, math.inf):
                    return call_contracts[id(node)]
            owner, key = None, None
            if isinstance(node, ast.Subscript):
                owner, key = resolve(node.value), node.slice
                if owner and owner[0].get("type") == "array":
                    return owner[0].get("items", {}), owner[1] + "[i]"
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "get" and node.args):
                owner, key = resolve(node.func.value), node.args[0]
            if owner and isinstance(key, ast.Constant) and isinstance(key.value, str):
                child = owner[0].get("properties", {}).get(key.value)
                if child is not None:
                    return child, owner[1] + f"[{key.value!r}]"
            return None

        for _ in range(len(bindings) + 1):
            changed = False
            for name, (expression, iterate) in bindings.items():
                if name in known or unsafe_at.get(name) == 0 or name in arguments or writes.get(name) != 1:
                    continue
                shape = resolve(expression)
                if iterate:
                    shape = (shape[0]["items"], shape[1] + "[i]") if shape and shape[0].get("type") == "array" else None
                if shape:
                    known[name] = shape
                    changed = True
            if not changed:
                break
        for node in nodes:
            base, key = None, None
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                base, key = node.value, node.slice
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "get" and node.args):
                base, key = node.func.value, node.args[0]
            if base is None or not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            shape = resolve(base)
            if shape and shape[0].get("closed") and key.value not in shape[0].get("properties", {}):
                keys = ", ".join(shape[0].get("properties", {}))
                issues.append(f"line {node.lineno}: {shape[1]} has no installed result key {key.value!r}; "
                              f"exact keys: {keys}. Read the installed return contract; do not guess or replace missing values.")
        for child in children:
            # A closure may execute after any mutation in the enclosing scope.
            captured = {name: shape for name, shape in known.items() if name not in unsafe_at}
            inspect_scope(child, captured if not isinstance(child, ast.ClassDef) else {}, enclosing_bindings | set(writes))
    if isinstance(tree, ast.Module):
        inspect_scope(tree, {}, set())
    return issues


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


def _parameterization_shapes(node: ast.Call, assignments: dict[str, ast.AST]) -> list[str]:
    """Catch explicit contradictory selection maps before loading any lattice/Julia."""
    values = {item.arg: assignments.get(item.value.id, item.value)
              if isinstance(item.value, ast.Name) else item.value
              for item in node.keywords if item.arg}

    def dict_keys(value):
        if isinstance(value, ast.Dict) and all(isinstance(key, ast.Constant)
                                              and isinstance(key.value, str) for key in value.keys):
            return {key.value for key in value.keys}
        return None

    included = dict_keys(values.get("included_element_names_by_field"))
    sums = dict_keys(values.get("exact_sum_by_field"))
    if included is not None and sums is not None and sums - included:
        return [f"exact sums {sorted(sums - included)} have no fields in included_element_names_by_field; "
                "an inclusion map varies ONLY its listed fields. Select the intended element pairs "
                "for those fields, or explicitly keep them fixed and audit their totals separately."]
    return []


def _research_run_methods(tree: ast.AST) -> dict[int, str]:
    """Identify proven handles only; any rebinding makes that name unknown.

    This deliberately conservative lexical analysis does not infer arbitrary
    objects from variable names, annotations, parameter names or dynamic calls.
    """
    calls: dict[int, str] = {}
    boundaries = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    cross_scope = {name for node in ast.walk(tree) if isinstance(node, (ast.Global, ast.Nonlocal))
                   for name in node.names}

    def inspect_scope(scope, inherited):
        nodes, children = [], []

        def collect(node):
            nodes.append(node)
            if isinstance(node, boundaries):
                children.append(node)
            else:
                for child in ast.iter_child_nodes(node):
                    collect(child)

        body = scope.body if isinstance(scope.body, list) else [scope.body]
        for statement in body:
            collect(statement)
        writes, imports, values = {}, {}, {}

        def write(name):
            writes[name] = writes.get(name, 0) + 1

        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for argument in ast.walk(scope.args):
                if isinstance(argument, ast.arg):
                    write(argument.arg)
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                write(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                write(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                # Cross-scope mutation is not statically owned here.
                for name in node.names:
                    writes[name] = 2
            elif isinstance(node, ast.ExceptHandler) and node.name:
                write(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                    write(name)
                    kind = None
                    if isinstance(node, ast.Import):
                        if alias.name == "brilliance.research_api":
                            kind = "module" if alias.asname else "package"
                        elif alias.name == "brilliance":
                            kind = "package"
                    elif node.module == "brilliance.research_api" and alias.name == "open_research_run":
                        kind = "factory"
                    elif node.module == "brilliance" and alias.name == "research_api":
                        kind = "module"
                    if kind:
                        imports[name] = kind
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                values[node.target.id] = node.value
        known = {name: kind for name, kind in inherited.items() if name not in writes and name not in cross_scope}
        known.update({name: kind for name, kind in imports.items() if writes[name] == 1 and name not in cross_scope})

        def kind(value):
            if isinstance(value, ast.Name):
                return known.get(value.id)
            if isinstance(value, ast.Attribute):
                owner = kind(value.value)
                if owner == "package" and value.attr == "research_api":
                    return "module"
                if owner == "module" and value.attr == "open_research_run":
                    return "factory"
            if isinstance(value, ast.Call) and kind(value.func) == "factory":
                return "run"
            return None

        # Resolve simple aliases without executing expressions or guessing types.
        for _ in range(len(values) + 1):
            additions = {name: resolved for name, value in values.items()
                         if writes.get(name) == 1 and name not in known and name not in cross_scope and (resolved := kind(value))}
            if not additions:
                break
            known.update(additions)
        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and kind(node.func.value) == "run":
                calls[id(node)] = node.func.attr
        for child in children:
            inspect_scope(child, known if not isinstance(child, ast.ClassDef) else {})

    if isinstance(tree, ast.Module):
        inspect_scope(tree, {})
    return calls


def lint_public_api_calls(tree: ast.AST, *, registry: Any = None, project_directory: Any = None,
                          return_contracts: dict[str, Any] | None = None) -> list[str]:
    from . import accelerator_api
    from .research_api import ResearchRun

    run_methods = _research_run_methods(tree)
    imports = _public_imports(tree)
    issues = []
    call_contracts = {}
    readonly_arguments = {}
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
        available = imports.get(id(value), {})
        factory = ((isinstance(function, ast.Attribute) and function.attr == "expert_module"
                    and isinstance(function.value, ast.Name) and available.get(function.value.id) == ("module", None))
                   or (isinstance(function, ast.Name) and available.get(function.id) == ("function", "expert_module")))
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
            available = imports.get(id(node), {})
            name = None
            owner = "api"
            module_id = None
            run_method = run_methods.get(id(node))
            if run_method is not None:
                name, owner = run_method, "ResearchRun"
            elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                  and available.get(node.func.value.id) == ("module", None)):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                binding = available.get(node.func.id)
                name = binding[1] if binding and binding[0] == "function" else None
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
                if run_method is not None:
                    target = getattr(ResearchRun, name, None)
                elif module_id is not None:
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
            from .result_contracts import return_contract
            schema = return_contract(target)
            if schema:
                call_contracts[id(node)] = (schema, f"{owner}.{name}(...)")
                if return_contracts is not None and (schema.get("properties") or schema.get("items")):
                    return_contracts[f"{owner}.{name}"] = schema
            if name == "expert_module" and module_id is None and library_id(node) is not None:
                try:
                    installed().module(library_id(node))
                except LookupError:
                    issues.append(f"line {node.lineno}: expert module {library_id(node)!r} is not installed/enabled")
            if name == "particle_swarm_optimize":
                issues.extend(f"line {node.lineno}: {issue}" for issue in
                              _optimizer_shapes(node, {} if id(node) in nested_calls else assignments))
            if name == "infer_mirror_symmetric_parameterization":
                issues.extend(f"line {node.lineno}: {issue}" for issue in
                              _parameterization_shapes(node, {} if id(node) in nested_calls else assignments))
            try:
                signature = inspect.signature(target)
                if run_method is not None and next(iter(signature.parameters), None) == "self":
                    signature = signature.replace(parameters=list(signature.parameters.values())[1:])
            except (TypeError, ValueError):
                continue
            readonly_arguments[id(node)] = _readonly_arguments(node, target, signature)
            issues.extend(_literal_argument_issues(node, target, signature, f"{owner}.{name}"))
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
    return [*issues, *_result_key_issues(tree, call_contracts, readonly_arguments)]
