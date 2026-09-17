"""Compatibility callables remain usable without crowding the default tool surface."""
import pytest

from brilliance.expert_sdk import python_tool
from brilliance.experts.extensions import ExpertPythonLibrary, ExpertPythonTool


def test_hidden_recipe_still_callable_but_not_recommended():
    def primitive(value):
        return value + 1

    @python_tool(discoverable=False, legacy_names=("old_recipe",))
    def recipe(value):
        return primitive(value)

    metadata = recipe.__brilliance_python_tool__
    library = ExpertPythonLibrary((ExpertPythonTool("primitive", primitive),
        ExpertPythonTool("recipe", recipe, metadata["legacy_names"], metadata["discoverable"])))
    assert dir(library) == ["primitive"]
    assert library.recipe(2) == 3
    assert library.primitive(2) == 3
    with pytest.raises(AttributeError):
        library.recipe = primitive


def test_discovery_flag_is_explicit_boolean():
    with pytest.raises(TypeError, match="boolean"):
        python_tool(discoverable="false")(lambda: None)
