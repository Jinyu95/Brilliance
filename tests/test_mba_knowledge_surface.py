"""Regression coverage for the installed MBA knowledge surface, not its physics."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from brilliance.codex_runtime import _stage_expert_reference_assets
from brilliance.experts import ExpertRegistry
from brilliance.experts.models import ExpertModuleError
from brilliance.experts.simple import load_module_path
from brilliance.native_research_agent import api_help
from brilliance.research_api import ResearchRun

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "expert_modules/mba_design"
MODULE_ID = "laboratory.mba_design"
ACTIVE_REFERENCES = {
    "references/conventional_api.md",
    "references/physics_reference_api.md",
    "references/hmba_outcome_api.md",
}
REMOVED_REFERENCES = {
    "references/classical_hmba_api.md",
    "references/classical_hmba_sources.md",
    "references/experimental_hybrid_routes.md",
    "references/hybrid_shape_api.md",
    "references/hybrid_continuation_api.md",
    "references/hybrid_regional_api.md",
    "references/outer_dba_api.md",
    "references/periodic_hybrid_api.md",
    "references/conventional_validation.md",
    "references/source_claim_matrix.json",
    "references/deferred_lgb_reverse_bend_mba.json",
}
REMOVED_SOURCES = {
    "conventional.py", "hmba_classical.py", "hmba_pair_optics.py",
    "hybrid_response.py", "hybrid_regions.py", "hybrid_continuation.py",
    "outer_dba.py", "periodic_hybrid.py",
}
REMOVED_TOOLS = {
    "ConventionalMBAContract", "conventional_unit_rows", "conventional_terminal_rows",
    "design_conventional_unit", "match_conventional_terminals", "assemble_conventional_mba",
    "refine_conventional_achromat", "conventional_stage_report", "publish_conventional_stage",
    "ClassicalHMBAContract", "ClassicalWithinCellEvaluator", "SextupolePairEvaluator",
    "HybridMBAShapeEvaluator", "HybridMBARegionalGeometry", "HybridMBAOuterReplacement",
    "build_outer_dba_section", "match_outer_dba_section", "explore_hybrid_mba",
    "match_periodic_hybrid",
}


@pytest.fixture
def registry():
    # Use only this installed module; no .env, other projects or numerical calls.
    return ExpertRegistry([load_module_path(MODULE)])


def test_only_current_guides_are_indexed_and_readable(registry):
    paths = {path.relative_to(MODULE).as_posix()
             for path in registry.reference_assets(MODULE_ID)}
    assert paths == ACTIVE_REFERENCES
    index = registry.read_knowledge(MODULE_ID)
    indexed = {Path(item["source"]).relative_to(MODULE).as_posix()
               for item in index["assets"]}
    assert indexed == ACTIVE_REFERENCES | {"EXPERT.md"}
    catalog = json.dumps(registry.catalog(goal="Design a hybrid MBA"))
    for relative in REMOVED_REFERENCES:
        assert relative not in catalog
    for relative in ACTIVE_REFERENCES:
        page = registry.read_knowledge(MODULE_ID, relative)
        assert page["text"].strip()
        assert page["status"] == "draft"
    for relative in REMOVED_REFERENCES:
        with pytest.raises(ExpertModuleError, match="unknown active knowledge"):
            registry.read_knowledge(MODULE_ID, relative)


def test_entry_point_is_short_and_supports_project_authored_methods(registry):
    raw = (MODULE / "EXPERT.md").read_text(encoding="utf-8")
    body = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", raw, count=1, flags=re.S)
    assert len(body.split()) <= 500, "Keep routing in the entry point and method details in references"
    for reference in ACTIVE_REFERENCES:
        assert reference in body
    chunks = registry.retrieve("Design a conventional 7BA storage ring", max_chars=6000, max_chunks=2)
    entry = next(item for item in chunks if item.source == MODULE / "EXPERT.md")
    assert entry.heading == "Complete method (coherent route context)"
    assert "INCOMPLETE EXPERT METHOD" not in entry.text
    assert "references/conventional_api.md" in entry.text
    conventional = (MODULE / "references/conventional_api.md").read_text(encoding="utf-8")
    conventional_text = " ".join(conventional.split())
    assert "instead of rewriting the same construction" not in conventional
    assert "project-local code" in conventional
    assert "agent defines" in body
    assert "LatticeExperiment" in body and "CandidateLedger" in body
    assert "fresh, measured TME/M-TME unit" in conventional_text
    assert "rematching of the first two units" in conventional_text
    assert "Do not invent two short" in conventional_text
    assert "match_mba_outer_units" in conventional_text
    assert "scientist request/approval" in conventional_text
    assert "measured evidence" in conventional_text
    assert "Exact equality is not" in conventional_text
    for removed in REMOVED_TOOLS:
        assert removed not in conventional


def test_default_discovery_contains_measurements_not_whole_design_procedures(registry):
    library = registry.python_library(MODULE_ID)
    advertised = set(dir(library))
    assert {"tme_targets", "design_tme_unit", "optimize_tme_unit_geometry",
            "verify_conventional_candidate", "verify_hmba_candidate",
            "solve_chromatic_correction"} <= advertised
    assert not advertised & REMOVED_TOOLS
    for name in REMOVED_TOOLS:
        with pytest.raises(AttributeError):
            getattr(library, name)
        with pytest.raises(LookupError):
            registry.legacy_python_tool(name)
    catalog = registry.catalog(goal="Design a hybrid MBA")
    assert not REMOVED_TOOLS & {item["name"] for item in catalog["modules"][0]["python_tools"]}


def test_default_unit_search_frees_geometry_and_retains_deterministic_reference(registry):
    entry = registry.read_knowledge(MODULE_ID, "EXPERT.md")["text"]
    text = " ".join(registry.read_knowledge(MODULE_ID, "references/conventional_api.md")["text"].split())
    assert "optimize_tme_unit_geometry" in entry
    assert "Default to `mba.optimize_tme_unit_geometry`" in text
    assert "five geometry coordinates plus two strengths" in text
    assert "not a full two-dimensional grid nested inside every particle" in text.replace("—", " ")
    assert "new candidates; retain both the original PSO result and exact refined rows" in text
    assert "complete eleven-element" in text
    assert "both opposite-sign assignments" in text
    assert "intact-map equivalence" in text
    assert "an unavailable centre diagnostic does not block the native emittance solve" in text
    assert "Sample the actual bend centre" not in text
    assert "no unit Dx=0 constraint, implicit low-tune band, or alpha/Dpx objective" in text
    assert "same-objective, same-limits comparisons" in text
    assert "fixed-geometry diagnostic, not the normal design default" in " ".join(entry.split())
    assert "scientist approval" in text
    assert "provisional experiment" in text
    assert "QF has positive K1; QD negative" not in text
    assert "Initial fraction boxes" not in text
    function = registry.python_library(MODULE_ID).design_tme_unit
    parameters = inspect.signature(function).parameters
    for required in ("energy_gev", "bend", "d1_m", "d2_m", "d3_m",
                     "quadrupole_near_length_m", "quadrupole_far_length_m", "expected_length_m",
                     "reference_repetitions", "limits", "maximum_beta_m", "trace_margin_x",
                     "trace_margin_y", "minimum_magnet_length_m", "minimum_drift_length_m"):
        assert required in parameters
        assert f"`{required}`" in text
    assert "target_tune" not in parameters
    assert "target_dx" not in parameters
    assert function.__brilliance_python_tool__["durable"] is True
    geometry_function = registry.python_library(MODULE_ID).optimize_tme_unit_geometry
    geometry_parameters = inspect.signature(geometry_function).parameters
    for name in ("energy_gev", "bend", "expected_length_m", "reference_repetitions", "limits",
                 "maximum_beta_m", "trace_margin_x", "trace_margin_y", "minimum_magnet_length_m",
                 "minimum_drift_length_m"):
        assert geometry_parameters[name].default is inspect.Parameter.empty
        assert f"`{name}`" in text
    assert geometry_parameters["initial_geometry"].default is None
    assert geometry_parameters["resume"].default == "auto"
    assert "abs(trace(M_plane)) <= 2 - trace_margin_plane" in text
    assert "1.99" in text and "1.01" in text
    assert 'result["candidates"]' in text
    assert 'candidate_evaluation=selected["evaluation"]' in text
    assert "partial/corrupt/incompatible pair requires inspection" in text
    assert geometry_function.__brilliance_python_tool__["durable"] is True


def test_obsolete_controllers_are_removed_not_hidden():
    for relative in REMOVED_SOURCES | REMOVED_REFERENCES:
        assert not (MODULE / relative).exists(), f"Obsolete source still ships: {relative}"


def _local_references(text):
    # Explicit resource IDs and Markdown links; never inspect arbitrary quoted physics prose.
    resource_ids = re.findall(r"(?:references|assets)/[\w./-]+\.(?:md|json|csv|py)", text)
    links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
    for value in (*resource_ids, *links):
        target = value.strip().strip("<>").split("#", 1)[0]
        if target and not re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target):
            yield target


def test_active_guides_and_exported_docstrings_have_no_dangling_resources(registry):
    sources = [(MODULE / "EXPERT.md", (MODULE / "EXPERT.md").read_text(encoding="utf-8"))]
    sources.extend((path, path.read_text(encoding="utf-8")) for path in registry.reference_assets(MODULE_ID))
    library = registry.python_library(MODULE_ID)
    sources.extend((MODULE / "tools.py", inspect.getdoc(getattr(library, name)) or "")
                   for name in dir(library))
    sources.extend((MODULE / "acceptance.py", item.instructions)
                   for item in registry.acceptance_checks)
    for source, text in sources:
        for removed in REMOVED_REFERENCES:
            assert Path(removed).name not in text, f"{source} still directs the model to {removed}"
        for relative in _local_references(text):
            target = MODULE / relative if relative.startswith(("references/", "assets/")) else source.parent / relative
            assert target.is_file() or (ROOT / relative).is_file(), f"{source}: missing reference {relative}"


def test_current_api_help_uses_current_route_guide_only(registry):
    help_text = api_help("", "verify_conventional_candidate", max_chars=12000, expert_registry=registry)
    assert "references/conventional_api.md" in help_text
    assert "verify_conventional_candidate" in help_text
    for removed in REMOVED_REFERENCES | REMOVED_TOOLS:
        assert removed not in help_text


def test_restaging_does_not_resurface_old_history_assets(registry, tmp_path):
    workspace = tmp_path / "workspace"
    old_relative = "references/hybrid_shape_api.md"
    historical = workspace / "expert_assets" / MODULE_ID / old_relative
    historical.parent.mkdir(parents=True)
    historical.write_text("Preserved old experiment documentation", encoding="utf-8")
    old_bytes = historical.read_bytes()
    manifest = _stage_expert_reference_assets(registry, workspace, query="Design a hybrid MBA")
    assert {item["relative"] for item in manifest["assets"]} == ACTIVE_REFERENCES
    assert historical.read_bytes() == old_bytes  # Cleanup must not erase saved project history.
    (workspace / "EXPERT_ASSETS.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = workspace / "artifacts"
    run = ResearchRun(workspace=workspace, output_directory=output,
        arguments_file=output/"arguments.json", result_file=output/"result.json",
        progress_file=workspace/"progress.json", cancellation_file=None,
        working_lattice_file=None, latest_candidate_lattice_file=None, compute_job_id=None)
    with pytest.raises(LookupError, match="not registered"):
        run.expert_asset(MODULE_ID, old_relative)
    for relative in ACTIVE_REFERENCES:
        assert run.expert_asset(MODULE_ID, relative).read_bytes() == (MODULE / relative).read_bytes()
