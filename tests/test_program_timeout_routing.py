from __future__ import annotations

import time
import tomllib
from pathlib import Path

from brilliance.codex_tools import _managed_expert_job_required


ROOT = Path(__file__).resolve().parents[1]


def _tool_by_id(tool_id: str) -> dict:
    manifest = tomllib.loads((ROOT / "brilliance/resources/expert.toml").read_text(encoding="utf-8"))
    return next(item for item in manifest["tools"] if item["id"] == tool_id)


def test_short_program_keeps_600_second_default_but_accepts_longer_estimates():
    tool = _tool_by_id("accelerator.run_research_program")
    timeout = tool["parameters"]["properties"]["timeout_s"]
    assert tool["timeout_s"] == 600
    assert timeout["default"] == 600
    assert timeout["minimum"] == 1
    assert timeout["maximum"] == 604800


def test_durable_program_retains_seven_day_timeout_contract():
    tool = _tool_by_id("accelerator.run_long_research_program")
    timeout = tool["parameters"]["properties"]["timeout_s"]
    assert tool["timeout_s"] == 604800
    assert timeout["default"] == 604800
    assert timeout["maximum"] == 604800


def test_requested_timeout_that_will_not_fit_turn_is_managed(monkeypatch):
    # The native research runtime supplies this deadline to brilliance-tool.
    # A 900 s request cannot fit a turn with roughly five minutes remaining,
    # so the existing host router must submit it as a managed calculation.
    monkeypatch.setenv("BRILLIANCE_MODEL_TURN_DEADLINE_EPOCH", str(time.time() + 300.0))
    assert _managed_expert_job_required(
        600,
        "accelerator.run_research_program",
        {"timeout_s": 900},
        turn_wall_time_s=600,
    ) is True


def test_short_request_can_still_run_foreground_when_it_fits(monkeypatch):
    monkeypatch.delenv("BRILLIANCE_MODEL_TURN_DEADLINE_EPOCH", raising=False)
    assert _managed_expert_job_required(
        600,
        "accelerator.run_research_program",
        {"timeout_s": 300},
        turn_wall_time_s=600,
    ) is False
