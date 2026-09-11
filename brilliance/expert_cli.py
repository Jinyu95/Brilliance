"""Simple developer CLI for local expert knowledge and tools."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

from .experts import (
    ExpertRegistry,
    doctor_module,
    load_module_path,
    package_module,
    test_module,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brilliance-expert")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("new", "init"):
        initialize = commands.add_parser(name, help="Create one simple EXPERT.md module")
        initialize.add_argument("directory")
        initialize.add_argument("--id")
        initialize.add_argument("--title")
        initialize.add_argument("--author", default="")
        initialize.add_argument("--with-tools", action="store_true")
    for name in ("check", "validate", "doctor"):
        check = commands.add_parser(name, help="Check a module without running calculations")
        check.add_argument("module")
    test = commands.add_parser("test", help="Run examples/*.json through the real tool boundary")
    test.add_argument("module")
    package = commands.add_parser("pack", help="Create a portable expert-module ZIP")
    package.add_argument("module")
    package.add_argument("--output")
    install = commands.add_parser("install", help="Install an EXPERT.md directory or ZIP")
    install.add_argument("module")
    install.add_argument("--root", default="expert_modules")
    listing = commands.add_parser("list", help="Discover installed expert modules")
    listing.add_argument("--root", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"new", "init"}:
        directory = Path(args.directory).resolve()
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"expert module directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        module_id = str(args.id or f"local.{_slug(directory.name)}")
        title = str(args.title or directory.name.replace("_", " ").replace("-", " ").title())
        (directory / "EXPERT.md").write_text(
            _starter_expert(module_id, title, str(args.author)),
            encoding="utf-8",
            newline="\n",
        )
        if args.with_tools:
            (directory / "tools.py").write_text(
                _starter_tool(), encoding="utf-8", newline="\n"
            )
        _print({
            "created": True,
            "directory": str(directory),
            "files_to_edit": [
                "EXPERT.md",
                *( ["tools.py"] if args.with_tools else [] ),
            ],
            "next": f"brilliance-expert doctor {directory}",
        })
        return 0
    if args.command in {"check", "validate", "doctor"}:
        report = doctor_module(args.module)
        _print(report)
        return 0 if report["passed"] else 2
    if args.command == "test":
        report = test_module(args.module)
        _print(report)
        return 0 if report["passed"] else 2
    if args.command == "pack":
        path = package_module(args.module, args.output)
        _print({"packed": True, "path": str(path)})
        return 0
    if args.command == "install":
        installed = _install_module(Path(args.module), Path(args.root))
        report = doctor_module(installed)
        _print({"installed": True, "path": str(installed), "doctor": report})
        return 0 if report["passed"] else 2
    if args.command == "list":
        registry = ExpertRegistry.discover(configured_roots=args.root, strict=False)
        _print(registry.catalog())
        return 0 if not registry.issues else 2
    raise ValueError(f"unknown command: {args.command}")


def _install_module(source: Path, root: Path) -> Path:
    source = source.resolve()
    destination_root = root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="brilliance-expert-install-") as temporary:
        if source.suffix.lower() == ".zip":
            extracted = Path(temporary)
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    member_path = Path(member.filename)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError(f"unsafe path in expert ZIP: {member.filename}")
                archive.extractall(extracted)
            candidates = sorted({path.parent for path in extracted.rglob("EXPERT.md")})
            if len(candidates) != 1:
                raise ValueError("expert ZIP must contain exactly one EXPERT.md module")
            module_source = candidates[0]
        else:
            module_source = source
        module = load_module_path(module_source)
        destination = destination_root / module.root.name
        if destination.exists():
            raise ValueError(f"expert module is already installed: {destination}")
        shutil.copytree(module.root, destination)
    return destination


def _starter_expert(module_id: str, title: str, author: str) -> str:
    authors = f"[{author}]" if author else "[]"
    return f"""---
id: {module_id}
title: {title}
summary: Replace this with the method's scope, machine classes, and scientific purpose.
domains: [accelerator physics]
authors: {authors}
tags: [replace-me]
---

# {title}

## Applicability

State where this method is valid, its approximations, and its exclusions.

## Required inputs

List every physical input and its unit.

## Method

Give the ordered expert procedure, initialization choices, alternatives, and checkpoints.

## Gates and tolerances

Define quantitative criteria. Distinguish feasibility from optimization objectives.

## Failure and recovery

Separate invalid physics, infeasibility, numerical failure, and software failure.

## Required evidence

Define the tables, plots, provenance, comparison cases, and stopping conditions.

## Sources

List primary sources and identify which recommendations are expert inference.
"""


def _starter_tool() -> str:
    return '''"""Optional scientific tools for this expert module."""

from typing import Annotated

from brilliance.expert_sdk import Field, tool, validator


@tool(
    authority="deterministic",
    output="A documented, JSON-compatible calculation result.",
)
def example_scale(
    value: Annotated[float, Field(description="Input scale", unit="m", minimum=0.0)],
) -> dict:
    """Replace this example with one bounded scientific calculation."""
    return {
        "value": value,
        "claim_boundary": "Example only; replace before scientific use.",
    }


@validator(for_tool="example_scale")
def validate_example(arguments, result):
    passed = result.get("value") == arguments.get("value")
    return {
        "passed": passed,
        "validator": "replace-example-validator-v1",
        "checks": {"value_preserved": passed},
        "issues": [] if passed else ["The tool did not preserve its input value."],
    }
'''


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower() or "module"


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
