"""Experts declare long operations without topic names in the execution core."""
from types import SimpleNamespace

import pytest

from brilliance.expert_sdk import python_tool
from brilliance.experts.extensions import ExpertPythonTool
from brilliance.native_research_agent import _requires_durable_execution
from brilliance.research_program import inspect_research_program


def test_python_tool_declares_durable_without_becoming_an_optimizer():
    @python_tool(durable=True)
    def scan():
        return 1

    assert scan() == 1
    assert scan.__brilliance_python_tool__["durable"] is True
    assert ExpertPythonTool("scan", scan, durable=True).as_dict()["durable"] is True
    assert _requires_durable_execution({"detected_calls": ["scan"], "durable_required": True})
    assert not _requires_durable_execution({"detected_calls": ["scan"], "durable_required": False})
    with pytest.raises(TypeError, match="durable must be a boolean"):
        python_tool(durable="true")(lambda: None)


def test_source_preflight_reads_expert_execution_metadata_not_a_topic_list(tmp_path):
    source = tmp_path / "scan.py"
    source.write_text(
        "from brilliance.research_api import open_research_run\n"
        "run = open_research_run()\n"
        "library = object()\n"
        "library.long_response_scan()\n"
        "run.finish(summary='A diagnostic result', scientific_passed=None)\n", encoding="utf-8")
    registry = SimpleNamespace(durable_python_calls={"long_response_scan"})
    result = inspect_research_program(tmp_path, source.name, expert_registry=registry)
    assert result["passed"], result["issues"]
    assert result["durable_required"]
    assert result["durable_expert_calls"] == ["long_response_scan"]
    assert _requires_durable_execution(result)

