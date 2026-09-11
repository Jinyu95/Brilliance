"""Shared, content-addressed local expert package loading.

An expert directory is a private Python package, so tools may use normal relative
imports without sys.path edits or collisions with another expert's helpers. Its
identity includes the resolved directory and all Python source content. Reading
source directly also avoids stale timestamp-based pyc caches during expert edits.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType

from .models import ExpertModuleError

_LOCK = threading.RLock()
_ROOTS: dict[str, Path] = {}


def package_source_paths(root: Path) -> tuple[Path, ...]:
    """All local Python dependencies, including helpers outside tools.py."""

    resolved = root.resolve()
    paths = []
    for path in sorted(resolved.rglob("*.py")):
        if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(resolved).parts):
            continue
        if not path.resolve().is_relative_to(resolved):
            raise ExpertModuleError(f"expert source escapes its module: {path}")
        paths.append(path)
    return tuple(paths)


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, source: Path):
        self.source = source

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        code = compile(self.source.read_bytes(), str(self.source), "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _PackageFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix, _, suffix = fullname.partition(".")
        root = _ROOTS.get(prefix)
        if root is None or not suffix:
            return None
        stem = root.joinpath(*suffix.split("."))
        package_source = stem / "__init__.py"
        source = package_source if package_source.is_file() else stem.with_suffix(".py")
        if source.is_file():
            if not source.resolve().is_relative_to(root):
                raise ExpertModuleError(f"expert source escapes its module: {source}")
            return importlib.util.spec_from_file_location(
                fullname, source, loader=_SourceLoader(source),
                submodule_search_locations=[str(stem)] if source == package_source else None,
            )
        if stem.is_dir():
            spec = importlib.util.spec_from_loader(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(stem)]
            return spec
        return None


_FINDER = _PackageFinder()


def load_expert_source(root: Path, source: Path) -> ModuleType:
    """Load a file within a module's isolated, dependency-aware Python package."""

    root, source = root.resolve(), source.resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ExpertModuleError(f"expert source is missing or outside its module: {source}")
    relative = source.relative_to(root).with_suffix("")
    if any(not item.isidentifier() for item in relative.parts):
        raise ExpertModuleError(f"expert Python source needs importable path names: {relative}")
    digest = hashlib.sha256(str(root).encode("utf-8"))
    for path in package_source_paths(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    prefix = "_brilliance_expert_" + digest.hexdigest()[:24]
    suffix = ".".join(relative.parts)
    if suffix.endswith(".__init__"):
        suffix = suffix.removesuffix(".__init__")
    with _LOCK:
        if _FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _FINDER)
        if prefix not in sys.modules:
            package = ModuleType(prefix)
            package.__path__ = [str(root)]
            package.__package__ = prefix
            package.__spec__ = importlib.util.spec_from_loader(prefix, loader=None, is_package=True)
            package.__spec__.submodule_search_locations = [str(root)]
            _ROOTS[prefix] = root
            sys.modules[prefix] = package
        try:
            return importlib.import_module(f"{prefix}.{suffix}")
        except Exception as exc:
            raise ExpertModuleError(
                f"cannot import expert source {source}: {type(exc).__name__}: {exc}"
            ) from exc
