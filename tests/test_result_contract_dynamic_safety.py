"""Conservative checks must not reject records legitimately changed by Python."""
import ast

import pytest

from brilliance.public_api_lint import lint_public_api_calls


def lint(source):
    return lint_public_api_calls(ast.parse("from brilliance import accelerator_api as api\n" + source))


@pytest.mark.parametrize("source", [
    "view = api.evaluation_snapshot(raw)\ngeometry = view['geometry']\n"
    "geometry['custom_label'] = 'draft'\nlabel = view['geometry']['custom_label']",
    "view = api.evaluation_snapshot(raw)\ngeometry = view['geometry']\n"
    "view['geometry']['custom_label'] = 'draft'\nlabel = geometry['custom_label']",
    "view = api.evaluation_snapshot(raw)\ngeometry = view.get('geometry')\n"
    "geometry.update(extra_fields)\nlabel = view['geometry']['custom_label']",
    "view = api.evaluation_snapshot(raw)\ngeometry = view['geometry']\n"
    "customize(geometry)\nlabel = view['geometry']['custom_label']",
    "view = api.evaluation_snapshot(raw)\ngeometry = view['geometry']\n"
    "customize(view)\nlabel = geometry['custom_label']",
])
def test_nested_alias_mutation_or_escape_invalidates_prior_closed_record(source):
    assert lint(source) == []


@pytest.mark.parametrize("source", [
    "def callback(api):\n    return api.project_specific_method()",
    "api = local_adapter\nvalue = api.project_specific_method()",
    "def callback():\n    api = local_adapter\n    return api.project_specific_method()",
])
def test_shadowed_import_is_not_the_installed_accelerator_module(source):
    assert lint(source) == []


@pytest.mark.parametrize("source", [
    "limits = api.magnet_strength_limits(3.5)\ntry:\n    check_request()\n"
    "except MappingException as limits:\n    reason = limits['error_reason']",
    "limits = api.magnet_strength_limits(3.5)\nmatch request:\n"
    "    case {'custom_limits': limits}:\n        label = limits['custom_label']",
])
def test_exception_and_pattern_bindings_do_not_retain_an_old_result_shape(source):
    assert lint(source) == []


@pytest.mark.parametrize("source", [
    "view = api.evaluation_snapshot(raw)\nvalue = view['checks']['linear_stable']\n"
    "if value is not None:\n    result = bool(value)",
    "view = api.evaluation_snapshot(raw)\ngeometry = view.get('geometry') or {}\n"
    "value = geometry.get('cell_length_m')",
    "def callback(dict):\n    limits = api.magnet_strength_limits(3.5)\n"
    "    custom = dict(limits)\n    return custom['custom_label']",
    "view = api.evaluation_snapshot(raw)\ngeometry = view['geometry']\n"
    "if isinstance(geometry, dict):\n    value = geometry.get('cell_length_m')",
])
def test_nullable_narrowing_and_shadowed_builtin_consumers_remain_allowed(source):
    assert lint(source) == []
