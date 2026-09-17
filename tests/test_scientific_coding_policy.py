"""Authorship and configured-source help tests; no provider or Julia is started."""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from brilliance import codex_runtime, native_research_agent
from brilliance.codex_backends import CodexBackend
from brilliance.coding_policy import SCIENTIFIC_CODING_POLICY
from brilliance.experts import ExpertRegistry
from brilliance.local_jutrack_help import render_reference, source_contract

WRAPPER = '''"""Configured fake wrapper. It must never be imported."""
raise RuntimeError("Help imported the native wrapper")
__version__ = "test-wrapper-1"
def unsafe_default():
    raise RuntimeError("Help evaluated a default")
def linepass(lattice, beam, refpts=None) -> tuple:
    """Track through a line. Returns beam coordinates and loss flags."""
    def internal():
        return {"not_a_public_return": 1}
    return (beam.coordinates, beam.loss_flags)
def twissring(lattice, dp=0.0, *, refpts=None, options=unsafe_default()):
    """Return wrapper Twiss points; coordinates are documented by the wrapper."""
    return _jl.twissring(lattice, dp, refpts, options)
def getchrom(lattice, dp=0.0) -> tuple[float, float]:
    return (1.0, 2.0)
def dictionary_result(lattice):
    return {"actual_key": 1, "second": 2}
def undocumented(lattice):
    ...
def ringpass(lattice, beam, num_turns):
    _jl.ringpass_b(lattice, beam.julia_object, num_turns)
def generator_result(lattice):
    yield lattice
def nested_generator(lattice):
    def inner():
        yield lattice
    _jl.operation(lattice)
def dynamic_aperture(lattice, amp_max=0.1):
    return _jl.dynamic_aperture(lattice, amp_max)
class Beam:
    """A native beam wrapper."""
    def __init__(self, coordinates, energy=3e9) -> None:
        self.coordinates = coordinates
    def copy(self):
        """Copy the native beam."""
        return Beam(self.coordinates)
    @property
    def count(self) -> int:
        return len(self.coordinates)
    @count.setter
    def count(self, value):
        self._count = value
'''


@pytest.fixture
def wrapper(tmp_path):
    source = tmp_path / "configured_wrapper.py"
    source.write_text(WRAPPER, encoding="utf-8")
    return source


def test_native_contract_is_static_and_honest(wrapper, monkeypatch):
    import subprocess

    before = {key for key in sys.modules if key.lower().startswith(("julia", "pyjutrack"))}
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Help launched a process"))
    contract = source_contract(wrapper)
    assert contract["sha256"] == hashlib.sha256(wrapper.read_bytes()).hexdigest()
    assert contract["declared_wrapper_version"] == "test-wrapper-1"
    entries = {entry["name"]: entry for entry in contract["entries"]}
    assert entries["linepass"]["signature"] == "linepass(lattice, beam, refpts=None) -> tuple"
    assert "not_a_public_return" not in str(entries["linepass"])
    assert "unsafe_default()" in entries["twissring"]["signature"]
    assert "literal dictionary keys: actual_key, second" in entries["dictionary_result"]["return_expressions"]
    assert entries["Beam"]["signature"] == "Beam(coordinates, energy=3000000000.0)"
    assert "self" not in entries["Beam.copy"]["signature"]
    assert entries["Beam.count"]["signature"] == "Beam.count  # property, not a method call"
    assert entries["Beam.count"]["return_annotation"] == "int"
    assert "implicit None on normal completion" in render_reference(wrapper, query="undocumented")
    assert "not the installed Julia package version" in render_reference(wrapper, query="jt.linepass")
    assert before == {key for key in sys.modules if key.lower().startswith(("julia", "pyjutrack"))}


def test_tracking_no_return_is_none_without_guessing_generator_contract(wrapper):
    entries = {entry["name"]: entry for entry in source_contract(wrapper)["entries"]}
    implicit_none = ["implicit None on normal completion (no return statement)"]
    assert entries["ringpass"]["return_expressions"] == implicit_none
    assert entries["nested_generator"]["return_expressions"] == implicit_none
    assert entries["generator_result"]["return_expressions"] == []
    text = native_research_agent.api_help("", "jt.ringpass", jutrack_source=wrapper)
    assert "implicit None on normal completion" in text
    assert "Not declared by this wrapper" not in text
    assert "ringpass(lattice, beam, num_turns)" in text


def test_native_help_has_exact_configured_signature_without_hijacking_public_helper(wrapper):
    native = native_research_agent.api_help("", "jt.twissring", jutrack_source=wrapper)
    assert "twissring(lattice, dp=0.0, *, refpts=None, options=unsafe_default())" in native
    assert str(wrapper.resolve()) in native
    assert "SHA256:" in native
    assert "_jl.twissring" in native and "not executed" in native
    plain = native_research_agent.api_help("", "linepass", jutrack_source=wrapper)
    assert "linepass(lattice, beam, refpts=None)" in plain
    public = native_research_agent.api_help("", "api.dynamic_aperture", jutrack_source=wrapper)
    assert "api.dynamic_aperture(" in public
    assert "Configured native JuTrack" not in public
    unknown = native_research_agent.api_help("", "jt.imagined_operation", jutrack_source=wrapper)
    assert "No exact requested symbol" in unknown
    assert "imagined_operation(" not in unknown


def test_missing_source_and_missing_version_do_not_trigger_discovery(wrapper, tmp_path):
    text = native_research_agent.api_help("", "jt.linepass", jutrack_source=tmp_path/"missing.py")
    assert "unavailable" in text and "do not discover" in text
    wrapper.write_text("def linepass(ring):\n    return run(ring)\n", encoding="utf-8")
    text = render_reference(wrapper, query="linepass")
    assert "version: not declared" in text
    assert "run(ring)" in text
    assert len(render_reference(wrapper, query="linepass", max_chars=350)) <= 350


def test_hidden_expert_function_help_is_explicit_only():
    def archived_recipe(value, *, tolerance=1e-4):
        """Compatibility method."""
    class Library:
        def __dir__(self):
            return []
        def __getattr__(self, name):
            if name == "archived_recipe":
                return archived_recipe
            raise AttributeError(name)
    registry = SimpleNamespace(modules=[SimpleNamespace(id="example.physics")],
        python_library=lambda _: Library(), knowledge_assets=lambda _: ())
    broad = native_research_agent.api_help("", "example.physics", expert_registry=registry)
    explicit = native_research_agent.api_help("", "archived_recipe", expert_registry=registry)
    assert "archived_recipe" not in broad
    assert ".archived_recipe(value, *, tolerance=0.0001)" in explicit


def test_authorship_keeps_physical_initialization_without_domain_rules_in_core():
    assert "Freedom to choose numerical" in SCIENTIFIC_CODING_POLICY
    assert "not permission to ignore an applicable expert's physical starting point" in SCIENTIFIC_CODING_POLICY
    assert "measured evidence justifies a clearly explained branch" in SCIENTIFIC_CODING_POLICY
    assert "Generic stability or valid" in SCIENTIFIC_CODING_POLICY
    assert "TME" not in SCIENTIFIC_CODING_POLICY and "7BA" not in SCIENTIFIC_CODING_POLICY


def _environment(tmp_path, wrapper):
    resource = tmp_path / "package/resources"
    resource.mkdir(parents=True, exist_ok=True)
    (resource / "accelerator_api_reference.md").write_text("# Public API\n", encoding="utf-8")
    handbook = resource / "handbook.md"
    handbook.write_text("# Shared handbook\n", encoding="utf-8")
    return SimpleNamespace(python_executable=Path(sys.executable), jutrack_root=wrapper.parent,
        pyjutrack_module=wrapper, accelerator_api=resource.parent/"accelerator_api.py",
        handbook=handbook, expert_max_context_chars=6000, internet_access=False,
        prepare_project_julia_environment=lambda _: None, render_manifest=lambda **_: "{}",
        manifest=lambda **_: {"python": {"command": "configured-python"}})


def _project():
    return SimpleNamespace(id="project-policy", goal="Study a measured beam response",
                           title="Beam response", context="", active_lattice_id=None)


def test_codex_prompts_share_authorship_policy_and_allow_contract_reads(tmp_path, wrapper, monkeypatch):
    environment, project = _environment(tmp_path, wrapper), _project()
    monkeypatch.setattr(ExpertRegistry, "from_environment", lambda *a, **k: ExpertRegistry([]))
    monkeypatch.setattr(codex_runtime, "ResearchConversationStore",
                        lambda _: SimpleNamespace(messages=lambda _: []))
    developer = codex_runtime._developer_instructions(project, tmp_path, environment)
    agents = codex_runtime._workspace_agents_md(project, tmp_path, environment)
    turn = codex_runtime._turn_prompt(project, "Fit a response model", initial=True)
    for text in (developer, agents):
        assert SCIENTIFIC_CODING_POLICY in text
        assert "NumPy/SciPy/Matplotlib" in text
        assert "JUTRACK_API_REFERENCE.md" in text
        assert "EXECUTE BEFORE" not in text and "read documentation before that source exists" not in text
    assert "Inspect the necessary" in turn and "focused API/method contract" in turn
    assert "Historical examples are seed context" not in developer


def test_codex_workspace_stages_same_configured_source_reference(tmp_path, wrapper, monkeypatch):
    from brilliance import agent_tools
    environment, project = _environment(tmp_path, wrapper), _project()
    monkeypatch.setattr(ExpertRegistry, "from_environment", lambda *a, **k: ExpertRegistry([]))
    monkeypatch.setattr(agent_tools, "agent_tool_catalog", lambda _: [])
    directory = codex_runtime.ensure_codex_workspace(tmp_path/"projects", project, environment)
    text = (directory/"workspace/JUTRACK_API_REFERENCE.md").read_text(encoding="utf-8")
    assert "linepass(lattice, beam, refpts=None) -> tuple" in text
    assert hashlib.sha256(wrapper.read_bytes()).hexdigest() in text


def test_native_system_uses_shared_authorship_policy_and_configured_native_help(tmp_path, wrapper):
    class PromptCaptured(Exception):
        pass
    captured = []
    async def completion(messages, tools):
        captured.extend(messages)
        raise PromptCaptured
    backend = CodexBackend(key="custom", name="test", provider_id="test", model="test",
                           mode="chat", config_overrides=(), base_url="https://unused.invalid")
    with pytest.raises(PromptCaptured):
        asyncio.run(native_research_agent.run_native_research_agent(
            backend=backend, project_root=tmp_path, project_id="project-policy",
            workspace=tmp_path/"project-policy/workspace", python_executable=Path(sys.executable),
            process_environment={"JUTRACK_PYTHON": str(wrapper), "BRILLIANCE_INTERNET_ACCESS": "false"},
            turn_wall_time_s=60, project_goal="Study a beam", scientist_message="Use jt.linepass",
            expert_context="", project_state_context="", api_reference="",
            expert_registry=ExpertRegistry([]), completion=completion))
    assert SCIENTIFIC_CODING_POLICY in captured[0]["content"]
    assert "use SciPy directly only" not in captured[0]["content"]
    assert "Reuse installed executable expert methods when available" not in captured[0]["content"]
    assert "linepass(lattice, beam, refpts=None) -> tuple" in captured[-1]["content"]
