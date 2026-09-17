"""Read-only native JuTrack contracts from the configured Python wrapper.

AST inspection never imports the wrapper, evaluates its defaults or starts Julia.
Only the exact path supplied by managed runtime configuration is read.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


def _returns(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    descriptions: list[str] = []
    generator = False

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            pass

        def visit_Lambda(self, node):
            pass

        def visit_Yield(self, node):
            nonlocal generator
            generator = True

        visit_YieldFrom = visit_Yield

        def visit_Return(self, node):
            value = node.value
            if isinstance(value, ast.Dict):
                keys = [key.value for key in value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)]
                label = "literal dictionary keys: " + ", ".join(keys)
                if len(keys) != len(value.keys):
                    label += "; other keys are dynamic"
            elif isinstance(value, ast.Tuple):
                label = f"tuple with {len(value.elts)} entries: " + ast.unparse(value)
            else:
                label = ast.unparse(value) if value is not None else "None"
            if label not in descriptions:
                descriptions.append(label)

    Visitor().visit(function)
    if not descriptions and not generator:
        descriptions.append("implicit None on normal completion (no return statement)")
    return descriptions


def _function_entry(node, *, name=None, bound=False):
    arguments = copy.deepcopy(node.args)
    if bound:
        positional = arguments.posonlyargs or arguments.args
        if positional and positional[0].arg in {"self", "cls"}:
            positional.pop(0)
    annotation = " -> " + ast.unparse(node.returns) if node.returns else ""
    return {
        "name": name or node.name,
        "signature": f"{name or node.name}({ast.unparse(arguments)}){annotation}",
        "line": node.lineno,
        "docstring": ast.get_docstring(node, clean=True) or "",
        "return_annotation": ast.unparse(node.returns) if node.returns else None,
        "return_expressions": _returns(node),
    }


@lru_cache(maxsize=8)
def _read_source(path: str, modified_ns: int, size: int) -> dict[str, Any]:
    del modified_ns
    if size > 8_000_000:
        raise ValueError("Configured JuTrack wrapper exceeds the bounded source-reference size")
    source_path = Path(path)
    raw = source_path.read_bytes()
    source = raw.decode("utf-8-sig")
    module = ast.parse(source, filename=path)
    entries = []
    version = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (str, int, float)):
                version = str(node.value.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            entries.append(_function_entry(node))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            constructor = next((item for item in node.body
                                if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
            if constructor is not None:
                entry = _function_entry(constructor, name=node.name, bound=True)
                entry["docstring"] = "\n\n".join(filter(None, (
                    ast.get_docstring(node, clean=True), entry["docstring"])))
                if entry["return_annotation"]:
                    entry["signature"] = entry["signature"].rsplit(" -> ", 1)[0]
                    entry["return_annotation"] = None
                entry["return_expressions"] = [f"instance of {node.name}; native runtime properties are not inferred"]
                entries.append(entry)
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and not method.name.startswith("_"):
                    decorators = {ast.unparse(item) for item in method.decorator_list}
                    if any(item.endswith((".setter", ".deleter")) for item in decorators):
                        continue
                    entry = _function_entry(method, name=f"{node.name}.{method.name}", bound=True)
                    if "property" in decorators:
                        entry["signature"] = f"{node.name}.{method.name}  # property, not a method call"
                    entries.append(entry)
    return {
        "source": path, "sha256": hashlib.sha256(raw).hexdigest(),
        "declared_wrapper_version": version, "entries": entries,
        "authority": "Configured Python wrapper source; not a Julia runtime compatibility test",
    }


def source_contract(path: Path | str) -> dict[str, Any]:
    """Inspect this one managed source; never search paths or import its code."""
    source = Path(path).resolve()
    stat = source.stat()
    if not source.is_file():
        raise ValueError("Configured JuTrack Python module is not a file")
    return copy.deepcopy(_read_source(str(source), stat.st_mtime_ns, stat.st_size))


def _entry_text(entry: dict[str, Any]) -> str:
    blocks = [f"## {entry['name']}", "", "```python", entry["signature"], "```",
              f"Configured source line: {entry['line']}."]
    if entry["docstring"]:
        blocks.extend(["", "Wrapper documentation:", entry["docstring"]])
    blocks.extend(["", "Static return information (not executed):"])
    if entry["return_annotation"]:
        blocks.append("Declared annotation: " + entry["return_annotation"])
    blocks.extend("- " + item for item in entry["return_expressions"])
    if not entry["return_expressions"] and not entry["return_annotation"]:
        blocks.append("Not declared by this wrapper; do not invent a return schema.")
    blocks.append("Expressions describe source syntax only; documentation is not independently runtime-validated.")
    return "\n".join(blocks)


def render_reference(path: Path | str, *, query: str | None = None, max_chars: int | None = None) -> str:
    """Render the full staged reference or exact-symbol lookup for local api_help."""
    contract = source_contract(path)
    entries = contract["entries"]
    if query is not None:
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query.casefold()))
        qualified = query.casefold()
        entries = [entry for entry in entries if entry["name"].casefold() in tokens
                   or ("." in entry["name"] and entry["name"].casefold() in qualified)]
        if not entries:
            if not re.search(r"\b(?:jt\.|pyjutrack|native jutrack|jutrack native)\b", qualified):
                return ""
            names = ", ".join(entry["name"] for entry in contract["entries"])
            body = "No exact requested symbol was found. Available configured-source names:\n" + names
        else:
            body = "\n\n".join(_entry_text(entry) for entry in entries)
    else:
        body = "\n\n".join(_entry_text(entry) for entry in entries)
    header = (
        "# Configured native JuTrack API — read-only source reference\n\n"
        f"Source: {contract['source']}\nSHA256: {contract['sha256']}\n"
        f"Declared Python-wrapper version: {contract['declared_wrapper_version'] or 'not declared'}\n"
        "Generated without importing pyJuTrack or starting Julia. This identifies the Python wrapper, "
        "not the installed Julia package version or proven numerical behavior.\n"
        "Use jt = api.jutrack() in configured research programs. Canonical lattice construction and "
        "evidence requirements still apply. Do not copy native code or infer undocumented return fields.\n\n"
    )
    result = header + body
    if max_chars is not None and len(result) > max_chars:
        notice = "\n[Native reference truncated; narrow api_help to one symbol or read the staged JUTRACK_API_REFERENCE.md section.]"
        result = result[:max(0, max_chars-len(notice))] + notice[:max_chars]
    return result
