"""Installed experiment contracts must be discoverable without source spelunking."""
from pathlib import Path

from brilliance.native_research_agent import api_help


def test_experiment_and_recovery_help_are_available():
    reference = (Path(__file__).resolve().parents[1] / "brilliance/resources/accelerator_api_reference.md").read_text(encoding="utf-8")
    for symbol in ("LatticeExperiment", "MatchingOperation", "prepare_length_budget",
                   "remove_mirrored_quadrupole_pair", "JuTrackCellEvaluator", "CandidateLedger"):
        help_text = api_help(reference, symbol, max_chars=12000)
        assert f"api.{symbol}(" in help_text
    recovered = api_help(reference, "CandidateLedger open", max_chars=12000)
    assert "CandidateLedger.open(" in recovered
    assert "read-only" in recovered
