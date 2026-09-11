"""Replay archived model source unchanged in a NEW isolated presentation test project."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from brilliance.jobs import JobState, ResearchJobStore, run_research_job
from brilliance.projects import ProjectStore, create_planning_project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--root", type=Path, default=Path("research_runs_default_validation/presentation_replays"))
    options = parser.parse_args()
    root = options.root.resolve()
    project = create_planning_project("Offline display validation: replay unchanged saved JuTrack program and candidate.")
    ProjectStore(root).save(project)
    workspace = root / project.id / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "replay.py"
    shutil.copyfile(options.source, source)
    archived_input = workspace / "artifacts" / "programs" / "replay-input"
    archived_input.mkdir(parents=True)
    shutil.copyfile(options.candidate, archived_input / "candidate.json")
    (archived_input / "result.json").write_text(json.dumps({
        "candidate_lattice": "candidate.json", "status": "validation_input",
        "summary": "Unchanged archived candidate copied only for presentation validation.",
    }), encoding="utf-8")
    jobs = ResearchJobStore(root)
    job = jobs.submit(project.id, project.revision, action="expert_tool", payload={
        "tool_id": "accelerator.run_long_research_program",
        "arguments": {"script": "replay.py", "arguments": {}, "timeout_s": 600},
        # No new agent turn: this is deterministic validation, not a new research request.
    })
    outcome = run_research_job(root, project.id, job.job_id)
    assert outcome.state is JobState.COMPLETE, outcome.detail
    reports = sorted((workspace / "artifacts" / "programs").glob("*/result.json"),
                     key=lambda path: path.stat().st_mtime)
    result_path = reports[-1]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    presentation = result["data"]["presentation"]
    assert presentation["status"] == "available", presentation
    assert len(result["plots"]) == 2, result["plots"]
    assert all((result_path.parent / row["path"]).is_file() for row in result["plots"])
    table = next(row for row in result["tables"] if row.get("kind") == "bend_radiation")
    assert table["rows"]
    for row in table["rows"]:
        assert all(isinstance(row[key], (int, float)) and math.isfinite(row[key])
                   for key in ("θ (rad)", "L (m)", "⟨H⟩ (m)", "I₂ (m⁻¹)", "I₄ (m⁻¹)", "I₅ (m⁻¹)", "I₅ / I₂")), row
    evidence = json.loads((result_path.parent / presentation["evidence"]).read_text(encoding="utf-8"))
    assert max(row["s_m"] for row in evidence["profile"]) <= evidence["cell_length_m"] + 1e-8
    assert evidence["cell_length_m"] < evidence["lattice_length_m"]
    assert source.read_bytes() == options.source.read_bytes()
    assert (archived_input / "candidate.json").read_bytes() == options.candidate.read_bytes()
    audit = {"passed": True, "project_id": project.id, "project_root": str(root),
             "output_directory": str(result_path.parent), "unchanged_program": str(options.source.resolve()),
             "unchanged_input": str(options.candidate.resolve()),
             "emittance_pm": result["data"].get("emittance_pm"),
             "plot_count": len(result["plots"]), "bend_rows": len(table["rows"]),
             "cell_length_m": evidence["cell_length_m"], "ring_length_m": evidence["lattice_length_m"],
             "presentation_simulation_rerun": presentation["simulation_rerun"]}
    (root / "latest-presentation-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
