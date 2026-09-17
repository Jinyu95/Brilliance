"""Declarative mapping inputs should be checked before managed execution."""

import ast
from typing import TypedDict, Unpack

from brilliance import accelerator_api as api
from brilliance.native_research_agent import api_help
from brilliance.public_api_lint import lint_public_api_calls
from brilliance.result_contracts import (
    closed_result,
    parameter_contracts,
    render_parameter_contracts,
)


def lint_finish(value):
    return lint_public_api_calls(ast.parse(
        "from brilliance.research_api import open_research_run\n"
        "run = open_research_run()\n"
        f"run.finish(summary='Measured candidate', scientific_stage={value})\n"))


def test_actual_unknown_scientific_stage_key_is_rejected_before_execution():
    issues = lint_finish("{'stage': 'first_complete_candidate', 'outcome': 'cell_gate_failed'}")
    assert len(issues) == 1
    assert "scientific_stage" in issues[0] and "'stage'" in issues[0]
    assert "stage_id" in issues[0]


def test_literal_wrong_scientific_stage_value_type_is_preflight_error():
    issues = lint_finish("{'stage_id': 3}")
    assert len(issues) == 1 and "stage_id" in issues[0]


def test_valid_scientific_stage_mapping_is_not_rejected():
    assert lint_finish("{'stage_id': 'first_complete_candidate', 'outcome': 'cell_gate_failed', "
                       "'interpretation': None, 'next_decision': 'Inspect saved residuals'}") == []


def test_dynamic_scientific_stage_values_are_not_invented_by_static_analysis():
    assert lint_finish("load_scientific_stage_from_saved_file()") == []


def test_api_help_exposes_nested_input_fields_without_a_topic_specific_guide():
    text = api_help("", "run.finish scientific_stage", max_chars=20000)
    assert "scientific_stage" in text
    for name in ("stage_id", "hypothesis_id", "candidate_id", "parent_candidate_id",
                 "outcome", "interpretation", "next_decision"):
        assert name in text


@closed_result
class ArbitraryInput(TypedDict, total=False):
    measurement_id: str
    explanation: str | None


class ExtensibleInput(TypedDict, total=False):
    measurement_id: str


def test_input_contract_check_is_generic_not_scientific_stage_special_case(monkeypatch):
    def local_input_probe(*, options: ArbitraryInput):
        raise AssertionError("Preflight must not call the function")
    monkeypatch.setattr(api, "local_input_probe", local_input_probe, raising=False)
    issues = lint_public_api_calls(ast.parse(
        "from brilliance import accelerator_api as api\n"
        "api.local_input_probe(options={'guessed_identifier': 'x'})\n"))
    assert len(issues) == 1
    assert "guessed_identifier" in issues[0] and "measurement_id" in issues[0]


def test_open_mapping_annotations_do_not_reject_extensions(monkeypatch):
    def local_input_probe(*, options: ExtensibleInput):
        raise AssertionError("Preflight must not call the function")
    monkeypatch.setattr(api, "local_input_probe", local_input_probe, raising=False)
    assert lint_public_api_calls(ast.parse(
        "from brilliance import accelerator_api as api\n"
        "api.local_input_probe(options={'extension_value': 'x'})\n")) == []


def test_parameter_schema_reuses_declared_required_optional_and_nullability():
    from brilliance.research_api import ResearchRun
    schema = parameter_contracts(ResearchRun.finish)["scientific_stage"]
    assert schema["closed"] and schema["nullable"]
    assert schema["required"] == []
    assert schema["properties"]["stage_id"] == {"type": "string", "nullable": True}
    rendered = render_parameter_contracts(ResearchRun.finish)
    assert "scientific_stage" in rendered and "stage_id" in rendered


def test_unpack_declaration_is_discoverable_without_executing_target():
    def accepts_keywords(**kwargs: Unpack[ArbitraryInput]):
        raise AssertionError("Do not run during schema discovery")
    schema = parameter_contracts(accepts_keywords)["kwargs"]
    assert schema["unpack"] is True and schema["closed"] is True
    assert set(schema["properties"]) == {"measurement_id", "explanation"}
