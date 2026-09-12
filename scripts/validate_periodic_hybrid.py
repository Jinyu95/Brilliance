"""Run the experimental periodic hybrid route on an explicitly supplied parent.

Uses the configured JuTrack environment. Does not publish a lattice, call a model,
change machine requirements, or substitute a facility/rounded HTML seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON keyword arguments documented in periodic_hybrid_api.md")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--identity-only", action="store_true")
    args = parser.parse_args()
    from brilliance import accelerator_api as api
    from brilliance.atomic_io import atomic_write_text
    from brilliance.research_environment import ResearchEnvironment

    env = ResearchEnvironment.load()
    env.require_active_python(operation="offline periodic H-MBA continuation validation")
    if env.internet_access:
        raise ValueError("Disable research internet access for this numerical validation")
    supplied = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be one JSON object")
    if set(supplied) & {"parent", "progress", "checkpoint", "cancelled"}:
        raise ValueError("Source and runtime callbacks are owned by this validation script")
    if args.identity_only:
        supplied["end_fraction"] = 0.0
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, data):
        atomic_write_text(args.output / name, json.dumps(data, indent=2, allow_nan=False) + "\n")

    save("input.json", {"source": str(args.source.resolve()), "source_sha256": source_hash,
                        "settings": supplied})
    try:
        parent = api.load_lattice(args.source)
        mba = api.expert_module("laboratory.mba_design")
        result = mba.match_periodic_hybrid(parent, **supplied,
            progress=lambda row: print(json.dumps(row, allow_nan=False), flush=True),
            checkpoint=lambda state: save("checkpoint.json", state))
        save("result.json", result)
        save("candidate_lattice.json", result["candidate_lattice"])
        print(json.dumps({"status": result["status"], "fraction": result["fraction"],
                          "emittance_pm": result["measurement"]["emittance_pm"],
                          "request_satisfied": False}, allow_nan=False), flush=True)
    except Exception:
        save("execution_error.json", {"status": "execution_error", "traceback": traceback.format_exc()})
        raise
    finally:
        unchanged = hashlib.sha256(args.source.read_bytes()).hexdigest() == source_hash
        save("source_integrity.json", {"source_unchanged": unchanged, "source_sha256": source_hash})
        if not unchanged:
            raise RuntimeError("Source lattice was modified during validation")


if __name__ == "__main__":
    main()
