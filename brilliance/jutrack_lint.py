"""Deterministic checks for the pyJuTrack API used by generated studies."""

from __future__ import annotations

import ast

JUTRACK_API_CONTRACT_VERSION = "2026.09.1"


def lint_jutrack_api(tree: ast.AST) -> list[str]:
    """Find common, high-impact pyJuTrack contract mistakes.

    The checks intentionally cover only contracts that can be established from
    syntax and simple assignment flow.  They complement execution; they do not
    attempt to prove accelerator-physics correctness.
    """
    twiss_results: set[str] = set()
    ringpara_results: set[str] = set()
    module_aliases = {"jt", "pyJuTrack"}
    errors: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pyJuTrack":
                    module_aliases.add(alias.asname or alias.name)
                    errors.append(
                        "do not import pyJuTrack directly; import brilliance.accelerator_api and "
                        "use api.jutrack() only for operations absent from the public API"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module == "pyJuTrack":
            errors.append(
                "do not import from pyJuTrack directly; use brilliance.accelerator_api so the "
                "configured Julia/JuTrack boundary remains reproducible"
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Call) and _is_public_jutrack_loader(value):
                module_aliases.update(_assigned_names(node))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        name = _jutrack_call_name(value, module_aliases)
        targets = _assigned_names(node)
        if name == "twissring":
            twiss_results.update(targets)
        elif name == "ringpara":
            ringpara_results.update(targets)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _jutrack_call_name(node, module_aliases)
            if name == "twissring":
                if len(node.args) > 1:
                    errors.append(
                        "jt.twissring accepts only lattice positionally in generated studies; "
                        "use dp=0.0 and E0=energy_eV (the second positional argument is dp, not energy)"
                    )
                keywords = {item.arg: item.value for item in node.keywords if item.arg}
                if "energy" in keywords:
                    errors.append(
                        "jt.twissring energy keyword is E0, not energy"
                    )
                refpts = keywords.get("refpts")
                if (
                    isinstance(refpts, ast.Call)
                    and isinstance(refpts.func, ast.Name)
                    and refpts.func.id == "range"
                ):
                    errors.append(
                        "jt.twissring refpts must be a concrete integer list in generated code; "
                        "use refpts=list(range(...))"
                    )
            elif name in {"gettune", "getchrom"} and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Name) and argument.id in twiss_results:
                    errors.append(
                        f"jt.{name} expects a lattice, not the result of jt.twissring"
                    )
            elif name == "findelem" and len(node.args) > 1:
                errors.append(
                    "jt.findelem accepts the lattice positionally, but element names must use "
                    "the name= keyword (for example jt.findelem(ring, name='QF'))"
                )
            elif name in {
                "MARKER",
                "DRIFT",
                "KQUAD",
                "QUAD",
                "KSEXT",
                "KOCT",
                "SBEND",
                "RBEND",
                "RFCA",
                "CORRECTOR",
            }:
                errors.append(
                    f"do not call jt.{name} directly in a research program; construct canonical "
                    "elements with brilliance.accelerator_api"
                )

        elif isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in ringpara_results
                and _literal_subscript(node) == "emittance"
            ):
                errors.append(
                    "jt.ringpara returns horizontal emittance under 'emittx', not 'emittance'"
                )

        elif isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in twiss_results
                and node.attr in {"spos", "beta", "alpha", "D"}
            ):
                errors.append(
                    "jt.twissring returns a sequence of Twiss points; iterate it and use "
                    ".s, .betax, .betay, .alphax, .alphay, .dx, and .dpx"
                )

    return list(dict.fromkeys(errors))


def _jutrack_call_name(node: ast.Call, aliases: set[str] | None = None) -> str | None:
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in (aliases or {"jt", "pyJuTrack"})
    ):
        return func.attr
    return None


def _is_public_jutrack_loader(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "jutrack"
        and isinstance(func.value, ast.Name)
    )


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    raw_targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: set[str] = set()
    for target in raw_targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _literal_subscript(node: ast.Subscript) -> object:
    value = node.slice
    if isinstance(value, ast.Constant):
        return value.value
    return None
