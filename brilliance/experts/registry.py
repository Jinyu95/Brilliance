"""Discovery and bounded retrieval for local expert modules."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .extensions import ExpertAcceptanceCheck, ExpertPythonLibrary, discover_extensions
from .loading import package_source_paths
from .models import ExpertModule, ExpertModuleError, ExpertTool, KnowledgeAsset

_TOKEN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+-]{1,}")
_HEADING = re.compile(r"(?m)^(#{1,4})\s+(.+?)\s*$")
_EXPERT_FRONT_MATTER = re.compile(
    r"\A---\s*\r?\n.*?\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL
)
_COHERENT_KNOWLEDGE_KINDS = {"method", "protocol"}
_REFERENCE_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    module_id: str
    module_version: str
    asset_id: str
    title: str
    heading: str
    status: str
    kind: str
    source: Path
    text: str
    score: float


class ExpertRegistry:
    """Immutable view of reviewed local expertise available to one project."""

    def __init__(
        self,
        modules: Iterable[ExpertModule] = (),
        *,
        issues: Iterable[str] = (),
    ):
        enabled = tuple(
            item for item in modules if item.enabled and item.status != "deprecated"
        )
        module_ids = [item.id for item in enabled]
        duplicates = sorted({item for item in module_ids if module_ids.count(item) > 1})
        if duplicates:
            raise ExpertModuleError(f"duplicate expert module ids: {duplicates}")
        tool_ids = [tool.id for module in enabled for tool in module.tools]
        duplicate_tools = sorted({item for item in tool_ids if tool_ids.count(item) > 1})
        if duplicate_tools:
            raise ExpertModuleError(f"duplicate expert tool ids: {duplicate_tools}")
        self._modules = tuple(sorted(enabled, key=lambda item: item.id))
        self._issues = tuple(str(item) for item in issues)
        checks: dict[str, ExpertAcceptanceCheck] = {}
        legacy_tools = {}
        self._python_tools = {}
        self._python_libraries = {}
        for module in self._modules:
            module_checks, python_tools = discover_extensions(module)
            for check in module_checks:
                if check.id in checks:
                    raise ExpertModuleError(f"duplicate expert acceptance check id: {check.id!r}")
                checks[check.id] = check
            for tool in python_tools:
                for alias in tool.legacy_names:
                    if alias in legacy_tools:
                        raise ExpertModuleError(f"duplicate expert Python legacy alias: {alias!r}")
                    legacy_tools[alias] = tool.function
            self._python_tools[module.id] = python_tools
            self._python_libraries[module.id] = ExpertPythonLibrary(python_tools)
        self._acceptance_checks = tuple(checks.values())
        self._acceptance_by_id = checks
        self._legacy_python_tools = legacy_tools

    @classmethod
    def discover(
        cls,
        *,
        source_root: Path | str | None = None,
        configured_roots: Sequence[Path | str] = (),
        project_directory: Path | str | None = None,
        strict: bool = True,
    ) -> "ExpertRegistry":
        root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
        sources: list[Path] = []
        builtin = root / "brilliance" / "resources" / "expert.toml"
        if builtin.is_file():
            sources.append(builtin)
        roots = [Path(item).expanduser() for item in configured_roots]
        if not roots:
            roots.append(root / "expert_modules")
        environment_paths = str(os.environ.get("BRILLIANCE_EXPERT_PATH", "")).strip()
        if environment_paths:
            roots.extend(Path(item).expanduser() for item in environment_paths.split(os.pathsep))
        if project_directory is not None:
            project = Path(project_directory).resolve()
            roots.extend([project / "experts", project / "workspace" / "experts"])
        for candidate in roots:
            resolved = candidate if candidate.is_absolute() else root / candidate
            sources.extend(_expert_paths(resolved.resolve()))
        modules: list[ExpertModule] = []
        issues: list[str] = []
        seen_paths: set[Path] = set()
        from .simple import load_module_path

        for source in sources:
            if source in seen_paths:
                continue
            seen_paths.add(source)
            try:
                modules.append(load_module_path(source))
            except ExpertModuleError as exc:
                if strict:
                    raise
                issues.append(str(exc))
        return cls(modules, issues=issues)

    @classmethod
    def from_environment(
        cls,
        environment: Any,
        *,
        project_directory: Path | str | None = None,
    ) -> "ExpertRegistry":
        return cls.discover(
            source_root=environment.source_root,
            configured_roots=getattr(environment, "expert_paths", ()),
            project_directory=(
                project_directory
                if bool(getattr(environment, "project_expert_modules_enabled", True))
                else None
            ),
            strict=bool(getattr(environment, "expert_strict", True)),
        )

    @classmethod
    def current(cls, project_directory: Path | str | None = None) -> "ExpertRegistry":
        """Resolve configured modules and the current managed project, if any."""

        from ..research_environment import ResearchEnvironment

        environment = ResearchEnvironment.load()
        if project_directory is None:
            workspace = os.environ.get("BRILLIANCE_WORKSPACE", "").strip()
            root = os.environ.get("BRILLIANCE_PROJECT_ROOT", "").strip()
            project_id = os.environ.get("BRILLIANCE_PROJECT_ID", "").strip()
            if workspace:
                project_directory = Path(workspace).resolve().parent
            elif root and project_id:
                project_directory = Path(root) / project_id
        return cls.from_environment(environment, project_directory=project_directory)

    @property
    def modules(self) -> tuple[ExpertModule, ...]:
        return self._modules

    @property
    def issues(self) -> tuple[str, ...]:
        return self._issues

    @property
    def acceptance_checks(self) -> tuple[ExpertAcceptanceCheck, ...]:
        return self._acceptance_checks

    def acceptance_check(self, name: str) -> ExpertAcceptanceCheck:
        try:
            return self._acceptance_by_id[str(name).strip().lower()]
        except KeyError as exc:
            raise LookupError(f"expert acceptance check not installed: {name}") from exc

    def python_library(self, module_id: str) -> ExpertPythonLibrary:
        module = self.module(module_id)
        return self._python_libraries[module.id]

    @property
    def durable_python_calls(self) -> frozenset[str]:
        """Installed long numerical operations; execution policy is topic-neutral."""
        return frozenset(name for group in self._python_tools.values() for tool in group
                         if tool.durable for name in (tool.name, *tool.legacy_names))

    def legacy_python_tool(self, name: str):
        try:
            return self._legacy_python_tools[name]
        except KeyError as exc:
            raise LookupError(f"expert Python legacy alias not installed: {name}") from exc

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for module in self._modules:
            for path in (
                module.manifest_path,
                *(item.path for item in module.knowledge),
                *_tool_source_paths(module),
                *package_source_paths(module.root),
                *self.reference_assets(module.id),
            ):
                digest.update(str(path).encode("utf-8"))
                digest.update(path.read_bytes())
            for tool in module.tools:
                digest.update(
                    json.dumps(tool.as_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
                )
        return digest.hexdigest()[:24]

    def module(self, module_id: str) -> ExpertModule:
        normalized = str(module_id).strip().lower()
        try:
            return next(item for item in self._modules if item.id == normalized)
        except StopIteration as exc:
            raise LookupError(f"expert module not installed: {module_id}") from exc

    def tool(self, tool_id: str) -> tuple[ExpertModule, ExpertTool]:
        normalized = str(tool_id).strip().lower()
        for module in self._modules:
            for tool in module.tools:
                if tool.id == normalized:
                    return module, tool
        raise LookupError(f"expert tool not installed: {tool_id}")

    def catalog(self, *, goal: str = "") -> dict[str, Any]:
        ranked = sorted(
            self._modules,
            key=lambda item: (-_module_score(item, goal), item.id),
        )
        return {
            "schema_version": "1.0",
            "fingerprint": self.fingerprint,
            "goal": goal,
            "issues": list(self._issues),
            "modules": [
                {
                    **module.as_dict(include_assets=True),
                    "python_tools": [item.as_dict() for item in self._python_tools[module.id]
                                     if item.discoverable],
                    "acceptance_checks": [
                        item.as_dict() for item in self._acceptance_checks if item.module_id == module.id
                    ],
                    "reference_assets": [
                        path.relative_to(module.root).as_posix()
                        for path in self.reference_assets(module.id)
                    ],
                    "relevance_score": _module_score(module, goal),
                }
                for module in ranked
            ],
            "tools": self.tool_catalog(),
            "development_contract": {
                "primary_source": "EXPERT.md",
                "knowledge": "one Markdown protocol with explicit draft|reviewed|validated status",
                "tools": "optional tools.py functions decorated through brilliance.expert_sdk",
                "python_libraries": "@python_tool functions via api.expert_module(module_id)",
                "acceptance": "optional @acceptance request/evidence checks in tools.py",
                "generated": "tool schemas, runtime adapters, catalog records, and fingerprints",
                "legacy": "expert.toml remains readable but is not required for new modules",
                "rule": (
                    "Expert notes are optional resources, not mandatory routes. The model may "
                    "develop and test new methods, but must not present draft knowledge or "
                    "untested hypotheses as validated evidence."
                ),
            },
        }

    def reference_assets(self, module_id: str | None = None) -> tuple[Path, ...]:
        """Return bounded, non-executable-by-default assets shipped with expert modules."""

        modules = (self.module(module_id),) if module_id else self._modules
        paths: list[Path] = []
        for module in modules:
            # Explicit context files may live elsewhere within their module. Include
            # them in staging, discovery and the registry's content fingerprint too.
            paths.extend(path for asset in module.knowledge
                         for path in asset.context_references)
            for folder in ("assets", "references"):
                directory = module.root / folder
                if not directory.is_dir():
                    continue
                for path in directory.rglob("*"):
                    resolved = path.resolve()
                    if (resolved.is_relative_to(module.root) and resolved.is_file()
                            and resolved.suffix.casefold() in _REFERENCE_SUFFIXES):
                        paths.append(resolved)
        return tuple(sorted(set(paths), key=lambda path: path.as_posix()))

    def knowledge_assets(self, module_id: str) -> tuple[KnowledgeAsset, ...]:
        """Discover local references without requiring experts to maintain a second index.

        Explicit manifest entries retain their identity/status (including deprecated
        entries). Other shipped reference files use their module-relative path as ID
        and inherit the module's status, never a stronger validation claim.
        """
        module = self.module(module_id)
        declared = {item.path.resolve() for item in module.knowledge}
        return (*module.knowledge, *(
            KnowledgeAsset(
                id=path.relative_to(module.root).as_posix(),
                title=path.stem.replace("_", " "), path=path, kind="reference",
                status=module.status, summary="Module-owned local reference; read as data, not executable code.",
                priority=20,
            )
            for path in self.reference_assets(module.id) if path not in declared
        ))

    def tool_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                **tool.as_dict(),
                "module_id": module.id,
                "module_version": module.version,
                "module_status": module.status,
                "effective_authority": (
                    tool.authority
                    if module.status in {"reviewed", "validated"}
                    else "provisional"
                ),
                "operation_class": "expert_tool",
            }
            for module in self._modules
            for tool in module.tools
        ]

    def read_knowledge(
        self,
        module_id: str,
        asset_id: str | None = None,
        *,
        offset: int = 0,
        limit: int = 12_000,
    ) -> dict[str, Any]:
        """Read an installed knowledge asset in bounded, source-ordered pages.

        Offsets and limits count Unicode characters, not bytes or lines. Omitting
        ``asset_id`` returns the module's small asset index. Callers can follow
        ``next_offset`` until ``eof`` without shell access or executing a program.
        ``complete`` means this response itself contains the entire source; a
        final page alone must not be mistaken for a complete method.
        """

        if type(offset) is not int or offset < 0:
            raise ValueError("knowledge offset must be a nonnegative integer character offset")
        if type(limit) is not int or not 1 <= limit <= 20_000:
            raise ValueError("knowledge limit must be between 1 and 20000 characters")
        module = self.module(module_id)
        assets = [item for item in self.knowledge_assets(module.id) if item.status != "deprecated"]
        common = {
            "module_id": module.id,
            "module_version": module.version,
            "module_status": module.status,
            "registry_fingerprint": self.fingerprint,
        }
        if asset_id is None:
            return {
                **common,
                "assets": [
                    {
                        "asset_id": item.id,
                        "title": item.title,
                        "summary": item.summary,
                        "kind": item.kind,
                        "status": item.status,
                        "source": str(item.path),
                        "total_chars": len(_read_knowledge_asset(module, item)),
                    }
                    for item in assets
                ],
            }
        # Match only indexed aliases, not an arbitrary user-supplied filesystem path.
        alias = str(asset_id).replace("\\", "/")
        asset = next((item for item in assets if item.id == asset_id
                      or item.path.relative_to(module.root).as_posix() == alias), None)
        if asset is None:
            raise ExpertModuleError(
                f"unknown active knowledge asset {asset_id!r} in module {module_id!r}; "
                "omit asset_id to list installed assets"
            )
        source = _read_knowledge_asset(module, asset)
        if offset > len(source):
            raise ValueError(f"knowledge offset exceeds source length {len(source)}")
        end = min(len(source), offset + limit)
        return {
            **common,
            "asset_id": asset.id,
            "title": asset.title,
            "kind": asset.kind,
            "status": asset.status,
            "source": str(asset.path),
            "offset": offset,
            "next_offset": end if end < len(source) else None,
            "total_chars": len(source),
            "text": source[offset:end],
            "eof": end == len(source),
            "complete": offset == 0 and end == len(source),
        }

    def retrieve(
        self,
        query: str,
        *,
        max_chars: int = 12_000,
        max_chunks: int = 16,
        module_id: str | None = None,
    ) -> tuple[KnowledgeChunk, ...]:
        if max_chars < 1 or max_chunks < 1:
            return ()
        selected_modules = (
            (self.module(module_id),)
            if module_id
            else tuple(
                module
                for module in self._modules
                if _module_matches_query(module, query)
            )
        )
        candidates: list[KnowledgeChunk] = []
        for module in selected_modules:
            for asset in module.knowledge:
                if asset.status == "deprecated":
                    continue
                text = _read_knowledge_asset(module, asset)
                asset_chunks = list(_chunks(text, fallback_heading=asset.title))
                scored_chunks = [
                    (
                        heading,
                        chunk_text,
                        _knowledge_score(module, asset, heading, chunk_text, query),
                    )
                    for heading, chunk_text in asset_chunks
                ]
                # A method is a connected decision procedure: its route selector, seed
                # construction, gates, and recovery rules are not independent search hits.
                # Splitting a compact method and ranking its headings separately can produce
                # a dangerous "route A -> route A.2 -> route A.4" prompt with the decisive
                # A.3 obligations missing. Keep the complete method together whenever it fits
                # the caller's bounded context. Large handbooks and foundations remain
                # section-retrieved.
                best_score = max((item[2] for item in scored_chunks), default=0.0)
                coherent_cost = len(text) + 300
                if (asset.kind in _COHERENT_KNOWLEDGE_KINDS or asset.context_references) and best_score > 0.0:
                    if coherent_cost <= max_chars:
                        heading = "Complete method (coherent route context)"
                        coherent_text = text
                    else:
                        # A budget is not permission to silently remove prerequisites
                        # or reorder an expert's connected method. Expose discovery
                        # metadata and require a source-ordered read instead.
                        heading = "Incomplete method context — read the complete source"
                        outline = [name for name, _ in asset_chunks]
                        coherent_text = (
                            "INCOMPLETE EXPERT METHOD: the complete source exceeds this "
                            "context budget. No method subsections have been selected or "
                            "reordered. Do not construct or execute a specialist method "
                            "from this index alone. Read the complete source in order with "
                            "read_expert, following next_offset until eof, before using it.\n\n"
                            f"read_expert(module_id={module.id!r}, asset_id={asset.id!r}, "
                            "offset=0, limit=12000)\n"
                            f"Complete source: {asset.path}\n"
                            f"Source length: {len(text)} characters.\n"
                            f"Summary: {asset.summary}\n\n"
                            "Source-ordered outline:\n"
                            + "\n".join(f"- {name}" for name in dict.fromkeys(outline))
                        )
                    candidates.append(KnowledgeChunk(
                        module_id=module.id,
                        module_version=module.version,
                        asset_id=asset.id,
                        title=asset.title,
                        heading=heading,
                        status=asset.status,
                        kind=asset.kind,
                        source=asset.path,
                        text=coherent_text,
                        score=best_score + 0.5,
                    ))
                    continue
                for heading, chunk_text, score in scored_chunks:
                    candidates.append(KnowledgeChunk(
                        module_id=module.id,
                        module_version=module.version,
                        asset_id=asset.id,
                        title=asset.title,
                        heading=heading,
                        status=asset.status,
                        kind=asset.kind,
                        source=asset.path,
                        text=chunk_text,
                        score=score,
                    ))
        candidates.sort(key=lambda item: (-item.score, item.module_id, item.asset_id, item.heading))
        chosen: list[KnowledgeChunk] = []
        consumed = 0
        for item in candidates:
            if len(chosen) >= max_chunks:
                break
            foundational = item.kind == "foundation" and item.score >= 9.0
            if item.score <= 0.0 and not foundational:
                continue
            cost = len(item.text) + 300
            if chosen and consumed + cost > max_chars:
                continue
            if not chosen and cost > max_chars:
                item = KnowledgeChunk(
                    module_id=item.module_id,
                    module_version=item.module_version,
                    asset_id=item.asset_id,
                    title=item.title,
                    heading=item.heading,
                    status=item.status,
                    kind=item.kind,
                    source=item.source,
                    text=item.text[:max_chars],
                    score=item.score,
                )
                cost = max_chars
            chosen.append(item)
            consumed += cost
        return tuple(chosen)

    def render_context(
        self,
        query: str,
        *,
        max_chars: int = 12_000,
        max_chunks: int = 16,
        module_id: str | None = None,
    ) -> str:
        chunks = self.retrieve(
            query,
            max_chars=max_chars,
            max_chunks=max_chunks,
            module_id=module_id,
        )
        lines = [
            "# Installed expert context",
            "",
            f"Query: {query}",
            f"Registry fingerprint: `{self.fingerprint}`",
            "",
            "Knowledge status is explicit. Draft material may guide a provisional hypothesis; "
            "`reviewed` is expert method knowledge and `validated` is accepted within its stated "
            "scope. Knowledge never selects a fixed software workflow. The research agent composes "
            "the calculation and numerical claims still require recorded method evidence.",
            "",
        ]
        authoritative = [
            item for item in chunks if item.status in {"reviewed", "validated"}
        ]
        if chunks and not authoritative:
            lines.extend([
                "## Expert coverage decision",
                "",
                "Only draft knowledge matched. Use it to formulate and test a provisional "
                "hypothesis with a saved research program. State the uncertainty and do not "
                "upgrade the result to expert-reviewed evidence.",
                "",
            ])
        if not chunks:
            lines.extend([
                "No installed expert knowledge matched this request.",
                "General calculations and inspections may still proceed with the documented "
                "JuTrack API. Disclose the missing specialist knowledge before making a "
                "domain-specific interpretation or design claim.",
                "",
            ])
        for item in chunks:
            lines.extend([
                f"## {item.title}: {item.heading}",
                "",
                f"Module: `{item.module_id}` v{item.module_version}; asset: `{item.asset_id}`; "
                f"status: `{item.status}`; kind: `{item.kind}`; source: `{item.source}`",
                "",
                item.text.strip(),
                "",
            ])
        return "\n".join(lines).rstrip() + "\n"


def _expert_paths(root: Path) -> list[Path]:
    if root.is_file() and root.name in {"expert.toml", "EXPERT.md"}:
        return [root]
    if not root.is_dir():
        return []
    paths: list[Path] = []

    def append_module(directory: Path) -> None:
        legacy = directory / "expert.toml"
        simple = directory / "EXPERT.md"
        if legacy.is_file():
            paths.append(legacy)
        elif simple.is_file():
            paths.append(simple)

    append_module(root)
    for directory in sorted(item for item in root.iterdir() if item.is_dir()):
        append_module(directory)
    return paths


def _tool_source_paths(module: ExpertModule) -> tuple[Path, ...]:
    paths: list[Path] = []
    for tool in module.tools:
        for entrypoint in (tool.entrypoint, tool.validator_entrypoint):
            if entrypoint is None:
                continue
            if entrypoint.startswith("file:"):
                _, relative, _ = entrypoint.split(":", 2)
                paths.append((module.root / relative).resolve())
                continue
            module_name, _ = entrypoint.split(":", 1)
            specification = importlib.util.find_spec(module_name)
            origin = str(getattr(specification, "origin", "") or "")
            if origin and origin not in {"built-in", "frozen"}:
                path = Path(origin).resolve()
                if path.is_file():
                    paths.append(path)
    return tuple(dict.fromkeys(paths))


def _tokens(value: str) -> set[str]:
    return {
        item.lower()
        for item in _TOKEN.findall(str(value))
        if any(character.isalpha() for character in item)
    }


_GENERIC_ROUTING_TOKENS = {
    "accelerator",
    "physics",
    "beam",
    "dynamics",
    "lattice",
    "design",
    "electron",
    "storage",
    "ring",
    "rings",
    "synchrotron",
    "light",
    "source",
    "jutrack",
    "matching",
    "radiation",
    "optimization",
}


def _module_matches_query(module: ExpertModule, query: str) -> bool:
    """Include specialist knowledge only for distinguishing request terms."""

    if any(item.kind == "foundation" for item in module.knowledge):
        return True
    query_tokens = _tokens(query)
    routing_tokens = _tokens(
        " ".join(
            (
                module.id,
                module.title,
                *module.domains,
                *(tag for item in module.knowledge for tag in item.tags),
            )
        )
    ) - _GENERIC_ROUTING_TOKENS
    return bool(query_tokens & routing_tokens)


def _module_score(module: ExpertModule, query: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    fields = _tokens(" ".join((module.title, module.description, *module.domains)))
    knowledge = _tokens(" ".join(
        value for item in module.knowledge for value in (item.title, item.summary, *item.tags)
    ))
    return float(4 * len(query_tokens & fields) + 2 * len(query_tokens & knowledge))


def _knowledge_score(
    module: ExpertModule,
    asset: KnowledgeAsset,
    heading: str,
    text: str,
    query: str,
) -> float:
    query_tokens = _tokens(query)
    metadata_tokens = _tokens(" ".join((
        module.title, module.description, *module.domains,
        asset.title, asset.summary, *asset.tags, heading,
    )))
    content_tokens = _tokens(text)
    score = float(5 * len(query_tokens & metadata_tokens) + len(query_tokens & content_tokens))
    score += asset.priority / 20.0
    if asset.kind == "foundation" and asset.priority >= 90:
        score += 5.0
    if asset.status == "validated":
        score += 2.0
    elif asset.status == "reviewed":
        score += 1.0
    elif asset.status == "draft":
        score -= 1.0
    return score


def _read_knowledge_asset(module: ExpertModule, asset: KnowledgeAsset) -> str:
    """Read one ordered context bundle; includes never expand recursively.

    Retrieval and paged reads use this same text, so an incomplete-context
    locator resolves to the complete method rather than just its short index.
    Assets without explicit references retain their original representation.
    """
    if not asset.context_references:
        return _read_asset(asset.path)
    parts = []
    for source in (asset.path, *asset.context_references):
        relative = source.relative_to(module.root).as_posix()
        parts.append(
            f"## Context source: `{relative}`\n\n"
            f"Module: `{module.id}` v{module.version}; status: `{asset.status}`.\n\n"
            + _read_asset(source)
        )
    return "\n\n".join(parts)


def _read_asset(path: Path) -> str:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    text = path.read_text(encoding="utf-8-sig")
    if path.name.casefold() == "expert.md":
        text = _EXPERT_FRONT_MATTER.sub("", text, count=1)
    return text


def _chunks(text: str, *, fallback_heading: str) -> Iterable[tuple[str, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        yield from _split_large(fallback_heading, text)
        return
    prefix = text[:matches[0].start()].strip()
    if prefix:
        yield from _split_large(fallback_heading, prefix)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = text[match.end():end].strip()
        if body:
            yield from _split_large(heading, body)


def _split_large(heading: str, text: str, *, limit: int = 6_000) -> Iterable[tuple[str, str]]:
    remaining = text.strip()
    part = 1
    while remaining:
        if len(remaining) <= limit:
            yield heading if part == 1 else f"{heading} (part {part})", remaining
            return
        boundary = remaining.rfind("\n\n", 0, limit)
        if boundary < limit // 2:
            boundary = limit
        yield heading if part == 1 else f"{heading} (part {part})", remaining[:boundary]
        remaining = remaining[boundary:].strip()
        part += 1
