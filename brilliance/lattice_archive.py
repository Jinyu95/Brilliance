"""Portable, immutable JuTrack reproduction packages for canonical lattices."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import canonical_json, json_safe
from .jutrack_service import JUTRACK_SERVICE_VERSION
from .lattice import LatticeIR

LATTICE_ARCHIVE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class LatticeArchivePackage:
    directory: Path
    definition_path: Path
    reproduction_path: Path
    provenance_path: Path
    created: bool


def archive_lattice_package(
    project_dir: Path | str,
    project_id: str,
    definition: Mapping[str, Any],
) -> LatticeArchivePackage:
    """Save one canonical JSON definition and its standalone JuTrack builder."""
    lattice = LatticeIR.from_dict(definition)
    if not re.fullmatch(r"lattice-[0-9a-f]+", lattice.lattice_id):
        raise ValueError(f"unsafe canonical lattice id: {lattice.lattice_id!r}")

    directory = Path(project_dir) / "state" / "lattices" / lattice.lattice_id
    definition_path = directory / "lattice.json"
    reproduction_path = directory / "reproduce.py"
    provenance_path = directory / "provenance.json"

    payload = lattice.as_dict()
    rendered_json = _render_json(payload)
    definition_sha256 = _sha256_bytes(rendered_json.encode("utf-8"))
    rendered_script = _reproduction_script(
        project_id=project_id,
        lattice=lattice,
        definition_sha256=definition_sha256,
    )
    script_sha256 = _sha256_bytes(rendered_script.encode("utf-8"))
    bends = [item for item in lattice.elements if item.kind in {"SBEND", "RBEND"}]
    implicit_zero_sextupoles = [
        item.name for item in lattice.elements
        if item.kind == "KSEXT" and item.k2_m3 is None
    ]
    provenance = {
        "schema_version": "1.0",
        "archive_version": LATTICE_ARCHIVE_VERSION,
        "project_id": project_id,
        "lattice_id": lattice.lattice_id,
        "parent_lattice_id": lattice.parent_lattice_id,
        "title": lattice.title,
        "source": lattice.source,
        "physical_fingerprint": lattice.physical_fingerprint,
        "element_count": len(lattice.elements),
        "total_length_m": lattice.total_length_m,
        "bend_sum_rad": lattice.bend_sum_rad,
        "negative_angle_bend_count": sum(
            (item.angle_rad or 0.0) < 0.0 for item in bends
        ),
        "zero_angle_bend_count": sum(
            abs(item.angle_rad or 0.0) == 0.0 for item in bends
        ),
        "legacy_execution_defaults": ({
            "KSEXT.k2_m3": 0.0,
            "affected_elements": implicit_zero_sextupoles,
            "reason": (
                "Legacy MBA artifacts serialized an implicit JuTrack zero strength as null. "
                "The reproduction script applies the original execution convention."
            ),
        } if implicit_zero_sextupoles else {}),
        "definition": {
            "path": "lattice.json",
            "sha256": definition_sha256,
            "authority": "canonical executable element sequence",
        },
        "reproduction": {
            "path": "reproduce.py",
            "sha256": script_sha256,
            "entry_point": "build_lattice",
            "jutrack_service_version": JUTRACK_SERVICE_VERSION,
        },
    }

    existed = all(path.is_file() for path in (
        definition_path, reproduction_path, provenance_path
    ))
    _write_text_once(definition_path, rendered_json)
    _write_text_once(reproduction_path, rendered_script)
    _write_json_once(provenance_path, provenance)
    return LatticeArchivePackage(
        directory=directory,
        definition_path=definition_path,
        reproduction_path=reproduction_path,
        provenance_path=provenance_path,
        created=not existed,
    )


def _reproduction_script(
    *, project_id: str, lattice: LatticeIR, definition_sha256: str
) -> str:
    """Render code frozen against the JuTrack constructor contract used at save time."""
    return f'''"""Recreate JuTrack lattice {lattice.lattice_id} from its canonical JSON.

This file and lattice.json are an immutable reproduction package archived by
BRILLIANCE. Run this file in the project's Python environment, or import
``build_lattice`` from it. Do not edit lattice.json without creating a new
lattice artifact: its SHA-256 is checked before any JuTrack object is built.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyJuTrack as jt

PROJECT_ID = {project_id!r}
LATTICE_ID = {lattice.lattice_id!r}
PARENT_LATTICE_ID = {lattice.parent_lattice_id!r}
ENERGY_EV = {lattice.energy_gev * 1.0e9!r}
EXPECTED_DEFINITION_SHA256 = {definition_sha256!r}
DEFINITION_PATH = Path(__file__).with_name("lattice.json")


def load_definition():
    raw = DEFINITION_PATH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_DEFINITION_SHA256:
        raise RuntimeError(
            f"lattice.json integrity failure: expected {{EXPECTED_DEFINITION_SHA256}}, got {{actual}}"
        )
    definition = json.loads(raw.decode("utf-8"))
    if definition.get("lattice_id") != LATTICE_ID:
        raise RuntimeError("lattice.json identity does not match this reproduction script")
    return definition


def _build_element(row):
    kind = str(row["kind"]).upper()
    name = str(row["name"])
    length = float(row["length_m"])
    fields = row.get("fields") or {{}}
    if kind == "MARKER":
        return jt.MARKER(name)
    if kind == "DRIFT":
        return jt.DRIFT(name, length)
    if kind in {{"KQUAD", "QUAD"}}:
        return getattr(jt, kind)(name, length, k1=float(row["k1_m2"]))
    if kind == "KSEXT":
        # Legacy MBA archives used null to mean the zero strength that JuTrack
        # actually executed. New archives store 0.0 explicitly.
        strength = row.get("k2_m3")
        return jt.KSEXT(name, length, k2=0.0 if strength is None else float(strength))
    if kind == "KOCT":
        return jt.KOCT(name, length, k3=float(row["k3_m4"]))
    if kind in {{"SBEND", "RBEND"}}:
        kwargs = {{}}
        if row.get("k1_m2") is not None:
            kwargs["PolynomB"] = [0.0, float(row["k1_m2"]), 0.0, 0.0]
            kwargs["MaxOrder"] = 1
        return getattr(jt, kind)(name, length, float(row["angle_rad"]), **kwargs)
    if kind == "RFCA":
        voltage = fields.get("voltage_v", fields.get("volt"))
        frequency = fields.get("frequency_hz", fields.get("freq"))
        return jt.RFCA(name, length, float(voltage), float(frequency))
    if kind == "CORRECTOR":
        return jt.CORRECTOR(
            name,
            length,
            xkick=float(fields.get("xkick_rad", 0.0)),
            ykick=float(fields.get("ykick_rad", 0.0)),
        )
    raise ValueError(f"unsupported archived JuTrack element kind: {{kind}}")


def build_lattice():
    definition = load_definition()
    return jt.Lattice([_build_element(row) for row in definition["elements"]])


if __name__ == "__main__":
    definition = load_definition()
    lattice = build_lattice()
    print(json.dumps({{
        "project_id": PROJECT_ID,
        "lattice_id": LATTICE_ID,
        "parent_lattice_id": PARENT_LATTICE_ID,
        "energy_eV": ENERGY_EV,
        "element_count": len(definition["elements"]),
        "total_length_m": sum(float(row["length_m"]) for row in definition["elements"]),
        "bend_sum_rad": sum(float(row.get("angle_rad") or 0.0) for row in definition["elements"]),
    }}, indent=2))
'''


def _render_json(payload: Any) -> str:
    return json.dumps(
        json_safe(payload), ensure_ascii=False, allow_nan=False,
        indent=2, sort_keys=True,
    ) + "\n"


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = _render_json(payload)
    try:
        _write_text_once(path, rendered)
    except RuntimeError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(payload):
            raise


def _write_text_once(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"immutable lattice archive conflict: {path}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
