"""Opt-in real-runtime handoff check on a caller-confirmed saved single cell.

No stored design fixture, seed generation, optimization or model call. Set
BRILLIANCE_TEST_CELL_PATH and BRILLIANCE_TEST_CELL_PERIODS only after inspecting
that artifact's source. All derived files go to pytest's isolated temp workspace.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from brilliance import accelerator_api as api
from brilliance.research_api import ResearchRun
from brilliance.research_environment import ResearchEnvironment


@pytest.mark.skipif(not os.environ.get("BRILLIANCE_TEST_CELL_PATH"), reason="opt-in configured JuTrack replay")
def test_saved_cell_to_ring_to_chat_result(tmp_path, monkeypatch):
    original_path = Path(os.environ["BRILLIANCE_TEST_CELL_PATH"]).resolve()
    periods = int(os.environ["BRILLIANCE_TEST_CELL_PERIODS"])
    fingerprint = hashlib.sha256(original_path.read_bytes()).hexdigest()
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Saved-cell handoff validation")
    for key, value in environment.process_environment().items():
        monkeypatch.setenv(key, value)

    original = api.load_lattice(original_path)
    # Scope confirmation is external and explicit; never inferred by the library.
    cell = api.periodic_cell(
        title=original.title, energy_gev=original.energy_gev, elements=original.elements,
        source="explicit saved-cell scope verification", intended_repetitions=periods,
        parent_lattice_id=original.lattice_id, metadata=original.metadata,
    )
    output = tmp_path / "artifacts"
    output.mkdir()
    api.save_lattice(output / "cell.json", cell)
    restored = api.load_lattice(output / "cell.json")
    rows = api.one_period_elements(restored)
    assert rows == [asdict(item) for item in original.elements]

    ring = api.cell_ring(
        title="Saved cell expanded for independent validation", energy_gev=cell.energy_gev,
        cell=rows, periods=periods, source="saved-cell handoff validation",
        parent_lattice_id=cell.lattice_id, metadata=cell.metadata,
    )
    api.save_lattice(output / "ring.json", ring)
    assert api.one_period_elements(api.load_lattice(output / "ring.json")) == rows
    assert len(ring.elements) == periods * len(rows)

    periodic = api.evaluate_periodic_cell(cell, include_profile=True)
    full = api.evaluate(ring, include_radiation=True, include_profile=False)
    assert periodic["linear_optics"]["stable"]
    assert full.linear_optics.stable
    radiation = full.require_radiation()
    assert radiation["emittance_pm"] > 0
    assert periodic["radiation"]["emittance_pm"] == pytest.approx(radiation["emittance_pm"], rel=1e-4)
    bend_evidence = api.bend_radiation_contributions(ring)
    assert bend_evidence["consistency"]["passed"]

    plot = api.save_optics_lattice_plot(
        output / "saved_cell_optics.svg", periodic["linear_optics"]["twiss_profile"], rows,
        title="Unchanged saved cell: optics and lattice",
    )
    bends = api.save_bend_radiation_contribution_plot(output / "bend_radiation.svg", bend_evidence)
    run = ResearchRun(
        workspace=tmp_path, output_directory=output, arguments_file=tmp_path / "arguments.json",
        result_file=output / "result.json", progress_file=tmp_path / "progress.json",
        cancellation_file=None, working_lattice_file=None, latest_candidate_lattice_file=None,
        compute_job_id="saved-cell-handoff-validation",
    )
    summary = {
        "source_sha256": fingerprint, "physical_elements_per_cell": len(rows),
        "ring_elements": len(ring.elements), "cell_length_m": cell.total_length_m,
        "ring_length_m": ring.total_length_m, "emittance_pm": radiation["emittance_pm"],
        "Dx_m": periodic["linear_optics"]["Dx_m"],
        "Dpx": periodic["linear_optics"]["Dpx"],
        "bend_integrals_reconcile": bend_evidence["consistency"]["passed"],
    }
    run.progress("Saved-state replay", kind="finding", detail="Same physical rows; explicit scope only.",
                 metrics=summary, plots=[str(plot), str(bends)])
    run.finish(
        summary="Unchanged saved-cell replay completed; no design optimization was performed.",
        data=summary, tables=[{"title": "Replay results", "rows": [summary]}],
        plots=[str(plot), str(bends)], candidate_lattice=cell, request_satisfied=False,
        scientific_passed=True,
        limitations=["Scope/serialization/computation/reporting validation only, not a design acceptance gate."],
    )
    receipt = json.loads(run.result_file.read_text(encoding="utf-8"))
    assert receipt["tables"] and len(receipt["plots"]) >= 2
    assert receipt["request_satisfied"] is False
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == fingerprint
    print("HANDOFF_REPLAY " + json.dumps(summary, sort_keys=True))
    print("HANDOFF_RESULT " + str(run.result_file))
