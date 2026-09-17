"""Installed read contracts prevent guessed keys without running any science."""
import ast
import json
from typing import TypedDict

import pytest

from brilliance import accelerator_api as api
from brilliance.public_api_lint import lint_public_api_calls
from brilliance.result_contracts import (
    closed_result,
    compact_return_contracts,
    render_return_contract,
    return_contract,
)


@closed_result
class _Entry(TypedDict):
    value: float
    extensions: dict[str, float]


@closed_result
class _Bank(TypedDict):
    candidates: list[_Entry]


def _probe() -> _Bank:
    pytest.fail("Preflight must not execute an installed numerical function")


def lint(source, **kwargs):
    return lint_public_api_calls(ast.parse("from brilliance import accelerator_api as api\n" + source), **kwargs)


def test_exact_limit_return_contract_matches_installed_pure_result():
    schema = return_contract(api.magnet_strength_limits)
    actual = api.magnet_strength_limits(3.5)
    assert schema["closed"]
    assert set(actual) <= set(schema["properties"])
    complete = api.magnet_strength_limits(3.5, length_m=.3,
        octupole_pole_tip_field_t=.5, octupole_radius_m=.02)
    assert set(schema["properties"]) == set(complete)
    assert "minimum_bend_radius_m" in schema["properties"]
    assert "brho_tm" in schema["properties"]
    assert "min_bend_radius_m" not in schema["properties"]


@pytest.mark.parametrize("access", [
    "api.magnet_strength_limits(3.5)['min_bend_radius_m']",
    "api.magnet_strength_limits(3.5).get('brho_t_m', None)",
])
def test_direct_guessed_literal_key_rejected_with_exact_installed_keys(access):
    issues = lint(access)
    assert len(issues) == 1
    assert "has no installed result key" in issues[0]
    assert "minimum_bend_radius_m" in issues[0] and "brho_tm" in issues[0]


def test_prior_readonly_mapping_consumer_does_not_hide_real_typo():
    issues = lint('''
limits = api.magnet_strength_limits(3.5)
api.audit_lattice_constraints([], energy_gev=3.5, expected_length_m=3, expected_angle_rad=0.1, limits=limits)
rho = limits['min_bend_radius_m']
publish(limits)
''')
    assert len(issues) == 1 and "min_bend_radius_m" in issues[0]


def test_valid_periodic_keywords_and_snapshot_nested_checks_are_not_rejected():
    assert lint('''
result = api.evaluate_periodic_cell(cell, periods=32, include_profile=True)
snapshot = api.evaluation_snapshot(result)
stable = snapshot['checks']['linear_stable']
length = snapshot['geometry']['cell_length_m']
future = snapshot['linear_optics'].get('future_extension', None)
''') == []


def test_closed_nested_checks_and_geometry_catch_guessed_fields():
    issues = lint('''
snapshot = api.evaluation_snapshot(result)
checks = snapshot['checks']
bad = checks['stability_confirmed']
geometry = snapshot['geometry']
also_bad = geometry['circumference']
''')
    assert len(issues) == 2
    assert any("linear_stable" in message for message in issues)
    assert any("cell_length_m" in message for message in issues)


def test_annotations_read_without_simulation_and_nested_list_contract(monkeypatch):
    monkeypatch.setattr(api, "contract_probe", _probe, raising=False)
    contracts = {}
    issues = lint('''
bank = api.contract_probe()
first = bank['candidates'][0]
missing = first['made_up']
for candidate in bank['candidates']:
    other_missing = candidate['also_made_up']
    allowed_extension = candidate['extensions'].get('user_value')
''', return_contracts=contracts)
    assert len(issues) == 2
    assert all("value, extensions" in message for message in issues)
    assert contracts['api.contract_probe']['properties']['candidates']['items']['closed']


def test_expert_result_uses_its_own_annotation_not_topic_names():
    class Registry:
        def python_library(self, module_id):
            assert module_id == "local.arbitrary_topic"
            return type("Library", (), {"probe": staticmethod(_probe)})

        def module(self, module_id):
            assert module_id == "local.arbitrary_topic"

    issues = lint('''
library = api.expert_module('local.arbitrary_topic')
result = library.probe()
bad = result['candidates'][0]['guessed']
''', registry=Registry())
    assert len(issues) == 1
    assert "local.arbitrary_topic" in issues[0] and "value, extensions" in issues[0]


@pytest.mark.parametrize("source", [
    "limits = arbitrary_factory()\nx = limits['unknown']",
    "limits = api.magnet_strength_limits(3.5)\nx = limits[user_key]",
    "limits = api.magnet_strength_limits(3.5)\nlimits = other\nx = limits['unknown']",
    "limits = api.magnet_strength_limits(3.5)\nlimits['unknown'] = 3\nx = limits['unknown']",
    "limits = api.magnet_strength_limits(3.5)\nlimits.update(additions)\nx = limits['unknown']",
    "limits = api.magnet_strength_limits(3.5)\nmutate(limits)\nx = limits['unknown']",
    "limits = api.magnet_strength_limits(3.5)\nalias = limits\nmutate(alias)\nx = limits['unknown']",
    "limits = api.magnet_strength_limits(3.5)\ndef fn(limits):\n    return limits['unknown']",
    "limits = api.magnet_strength_limits(3.5)\ndef fn():\n    limits = other\n    return limits['unknown']",
    "limits = api.magnet_strength_limits(3.5)\ndef fn():\n    return limits['unknown']\nmutate(limits)",
    "limits = api.magnet_strength_limits(3.5)\ndef fn():\n    global limits\n    limits = other\n    return limits['unknown']",
    "api = other\nlimits = api.magnet_strength_limits(3.5)\nx = limits['unknown']",
    "def fn(api):\n    limits = api.magnet_strength_limits(3.5)\n    return limits['unknown']",
])
def test_unknown_dynamic_shadowed_or_mutated_shapes_are_not_guessed(source):
    assert lint(source) == []


def test_typo_before_later_escape_is_still_reported_and_aliases_inherit_contract():
    issues = lint('''
limits = api.magnet_strength_limits(3.5)
alias = limits
x = alias['unknown']
mutate(limits)
y = alias['extension_added_by_consumer']
''')
    assert len(issues) == 1 and "'unknown'" in issues[0]


def test_unmodified_enclosing_binding_is_available_inside_callback():
    issues = lint('''
limits = api.magnet_strength_limits(3.5)
def callback():
    return limits['unknown']
''')
    assert len(issues) == 1


def test_help_contains_real_geometry_and_checks_without_executing_evaluation():
    help_text = render_return_contract(api.evaluation_snapshot)
    assert "result['geometry']" in help_text
    assert "cell_length_m" in help_text and "reference_ring_circumference_m" in help_text
    assert "result['checks']" in help_text and "linear_stable" in help_text
    assert not return_contract(api.evaluate)  # A dataclass is not a dict of invented keys.


def test_snapshot_declared_fields_cover_actual_legacy_output_and_nullable_geometry():
    actual = api.evaluation_snapshot({"engine": "JuTrack", "evaluation_scope": "periodic_cell",
                                     "geometry": {}, "linear_optics": {"stable": True}})
    schema = return_contract(api.evaluation_snapshot)
    assert set(actual) == set(schema['properties'])
    assert set(actual['geometry']) == set(schema['properties']['geometry']['properties'])
    assert set(actual['checks']) == set(schema['properties']['checks']['properties'])
    assert actual['geometry']['cell_length_m'] is None


def test_compact_schema_budget_never_samples_key_names():
    contract = return_contract(_probe)
    contracts, omitted = compact_return_contracts({f"api.probe_{i}": contract for i in range(100)}, max_bytes=1000)
    assert len(json.dumps(contracts, ensure_ascii=False, separators=(",", ":")).encode('utf-8')) <= 1000
    assert omitted
    assert all(value['keys'] == ['candidates'] for value in contracts.values())


def test_source_inspection_exposes_contracts_alongside_actionable_issues(tmp_path, monkeypatch):
    from brilliance import research_api
    from brilliance.research_program import inspect_research_program

    monkeypatch.setattr(research_api, "open_research_run", lambda: pytest.fail("Opened a run"))
    source = tmp_path / "program.py"
    source.write_text('''
from brilliance import accelerator_api as api
from brilliance.research_api import open_research_run
run = open_research_run()
limits = api.magnet_strength_limits(3.5)
rho = limits['min_bend_radius_m']
run.finish(summary="Report", status="partial", request_satisfied=False)
''', encoding="utf-8")
    result = inspect_research_program(tmp_path, "program.py")
    assert not result['passed']
    assert any("min_bend_radius_m" in issue for issue in result['issues'])
    assert "minimum_bend_radius_m" in result['installed_return_contracts']['api.magnet_strength_limits']['keys']
    assert result['omitted_return_contracts'] == []
