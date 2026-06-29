"""Convert saved Streamlit chat sessions into the exploration analysis format.

This bridges the gap between the chat UI workflow and the batch analysis
pipeline: you design lattices in the chat, save sessions, then run analysis
on them.

Usage:
    from science.session_ingest import ingest_sessions
    ingest_sessions("sessions/", "results.jsonl")
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

from physics_core import (
    parse_result_output,
    compute_brightness_fom,
    measure_lattice_factor_F,
    compute_robinson_sum,
)


def find_sessions(base_dir: Path | str) -> list[Path]:
    """Find all session directories that contain a conversation.json."""
    base = Path(base_dir)
    if not base.exists():
        return []
    # Single session directory?
    if (base / "conversation.json").exists():
        return [base]
    # Directory of sessions
    return sorted(
        d for d in base.iterdir()
        if d.is_dir() and (d / "conversation.json").exists()
    )


def extract_latest_code(messages: list[dict]) -> str:
    """Get the last CodeWriter python code block."""
    for msg in reversed(messages):
        if msg.get("source") != "CodeWriter":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for match in re.finditer(r"```python\s*\n(.*?)```", content, re.DOTALL):
            return match.group(1)
    return ""


def extract_dipole_angles(code: str) -> list[float] | None:
    """Try to extract dipole_angles_rad from the generated code."""
    # Look for explicit list assignment
    for pat in [
        r"dipole_angles_rad\s*=\s*\[([^\]]+)\]",
        r"theta_per_dipole\s*=\s*\[([^\]]+)\]",
        r"bend_angles\s*=\s*\[([^\]]+)\]",
    ]:
        m = re.search(pat, code)
        if m:
            try:
                return [float(x.strip()) for x in m.group(1).split(",")]
            except ValueError:
                continue
    return None


def ingest_session(session_dir: Path) -> dict | None:
    """Ingest one session directory into the exploration result format.

    Returns a dict compatible with the exploration JSONL schema, or None
    if no structured result could be extracted.
    """
    conv_path = session_dir / "conversation.json"
    if not conv_path.exists():
        return None

    try:
        messages = json.loads(conv_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # Extract user request
    user_request = ""
    for msg in messages:
        if msg.get("source") == "user" and isinstance(msg.get("content"), str):
            user_request = msg["content"].strip()
            break

    # Extract constraints from the request
    from agents.constraints import parse_design_constraints
    constraints = parse_design_constraints(user_request)

    # Extract final structured result
    parsed_result = None
    for msg in reversed(messages):
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        parsed = parse_result_output(content)
        if parsed and parsed.get("stable") is not None:
            parsed_result = parsed
            break

    if parsed_result is None:
        return None

    # Extract code
    code = extract_latest_code(messages)

    # Infer family from constraints or result
    family = constraints.get("cell_type") or constraints.get("machine_class") or "unknown"

    # Extract dipole angles if present
    dipole_angles = extract_dipole_angles(code)
    if dipole_angles:
        parsed_result["dipole_angles_rad"] = dipole_angles

    return {
        "label": session_dir.name,
        "family": family,
        "energy_gev": parsed_result.get("energy_gev", constraints.get("energy_gev")),
        "n_cells": parsed_result.get("n_cells", constraints.get("n_cells", 0)),
        "n_bends_per_cell": parsed_result.get("n_bends_per_cell", 2),
        "analytical_emittance_pm": None,
        "prompt": user_request[:500],
        "success": parsed_result.get("stable") is True,
        "error": parsed_result.get("error"),
        "elapsed_s": 0,
        "message_count": len(messages),
        "result": parsed_result,
        "score": 0,
        "status": "from_session",
        "brightness_A_per_m_rad": compute_brightness_fom(parsed_result),
        "F_empirical": measure_lattice_factor_F(parsed_result),
        "timestamp": datetime.now().isoformat(),
        "source_path": str(session_dir),
        "code_preview": code[:300] if code else "",
    }


def ingest_sessions(
    base_dir: Path | str,
    output_path: Path | str,
    *,
    verbose: bool = True,
) -> int:
    """Walks session directories and writes an exploration-compatible JSONL file.

    Parameters
    ----------
    base_dir : Path | str
        Either a single session directory or the ``sessions/`` parent.
    output_path : Path | str
        Where to write the JSONL database.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    int
        Number of sessions successfully ingested.
    """
    session_dirs = find_sessions(base_dir)
    if not session_dirs:
        if verbose:
            print(f"No sessions found in {base_dir}")
        return 0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as out:
        for sd in session_dirs:
            row = ingest_session(sd)
            if row is None:
                if verbose:
                    print(f"  SKIP {sd.name}: no structured result")
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if verbose:
                stable = row["result"].get("stable")
                emit = row["result"].get("emittance", 0) * 1e12
                print(f"  OK  {sd.name}  stable={stable}  emit={emit:.1f} pm-rad")

    if verbose:
        print(f"\nIngested {count}/{len(session_dirs)} sessions → {output}")
        print(f"Analyze with:")
        print(f"  python run_explore.py analyze {output} --figures figures/")

    return count
