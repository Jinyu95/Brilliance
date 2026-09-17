"""Focused contract retrieval; no inference request or numerical execution."""
from pathlib import Path

from brilliance import accelerator_api
from brilliance.experts import ExpertRegistry
from brilliance.experts.simple import load_module_path
from brilliance.native_research_agent import _new_contract_hints, api_help


def test_exact_symbols_beat_repeated_incidental_keywords_and_keep_examples():
    reference = ("## Saved program\n" + "periodic_cell cell_ring result periodic " * 400
                 + "\n## Cell construction\n```python\n"
                 + "# Preserve the actual cell before repetition.\n"
                 + "cell = api.periodic_cell(title=title, energy_gev=energy, elements=rows, source=source, intended_repetitions=32)\n"
                 + "ring = api.cell_ring(title=title, energy_gev=energy, cell=rows, periods=32, source=source)\n```\n"
                 + "## Unrelated tracking\nRun a long particle tracking campaign.\n")
    text = api_help(reference, "periodic_cell cell_ring result periodic")
    assert "api.periodic_cell(" in text and "api.cell_ring(" in text
    assert "## Cell construction" in text and "Preserve the actual cell" in text
    assert "## Saved program" not in text and "Unrelated tracking" not in text


def test_whole_symbol_matching_does_not_expand_drift_to_unrelated_words():
    reference = ("## Unrelated matching\n" + "drifting quadrupoles elements " * 400
                 + "\n## Constructors\napi.drift(name, length_m)\napi.quadrupole(name, length_m, k1_m2)\n")
    text = api_help(reference, "drift quadrupole")
    assert "## Constructors" in text
    assert "## Unrelated matching" not in text


def test_all_requested_declarations_precede_long_docstrings(monkeypatch):
    def first_operation(value, *, tolerance=1e-4):
        pass

    def second_operation(value, *, iterations=10):
        pass

    first_operation.__module__ = second_operation.__module__ = "brilliance.accelerator_api"
    first_operation.__doc__ = "Documented first-operation detail. " * 500
    monkeypatch.setattr(accelerator_api, "first_operation", first_operation, raising=False)
    monkeypatch.setattr(accelerator_api, "second_operation", second_operation, raising=False)
    text = api_help("", "first_operation second_operation", max_chars=1600)
    assert "api.first_operation(value, *, tolerance=0.0001)" in text
    assert "api.second_operation(value, *, iterations=10)" in text
    assert "truncated" in text.lower()
    assert len(text) <= 1600


def test_imported_typing_classes_and_unmatched_sections_are_not_public_help():
    text = api_help("## Unrelated\nNo contract here.\n", "any drift")
    assert "api.drift(" in text
    assert "api.Any" not in text and "## Unrelated" not in text
    missing = api_help("## Unrelated\nNo contract here.\n", "imagined_api_name")
    assert "No matching installed API signature" in missing
    assert "## Unrelated" not in missing
    mixed = api_help("## Constructors\napi.drift(name, length_m)\n", "drift imagined_api_name")
    assert "api.drift(" in mixed
    assert "No installed signature or reference match for: imagined_api_name" in mixed


def test_research_run_method_help_uses_the_actual_installed_method():
    text = api_help("", "run.raise_if_cancelled")
    assert "run.raise_if_cancelled(self)" in text
    assert "No contract was inferred" not in text


def test_actual_return_keys_are_available_without_a_numerical_discovery_run():
    limits = api_help("", "magnet_strength_limits", max_chars=12000)
    assert "Installed return contract" in limits
    assert "minimum_bend_radius_m" in limits
    assert "brho_tm" in limits
    snapshot = api_help("", "evaluation_snapshot", max_chars=12000)
    assert "result['checks']" in snapshot and "linear_stable" in snapshot
    assert "result['linear_optics']" in snapshot and "twiss_profile" in snapshot
    assert "result['geometry']" in snapshot and "cell_length_m" in snapshot


def test_edit_responses_do_not_repeat_unchanged_contract_payloads():
    schema = {"type": "object", "properties": {"measured": {"type": "number"}}}
    inspection = {"passed": False, "issues": ["invented field"],
                  "installed_return_contracts": {"api.operation": schema}}
    seen = {}
    assert _new_contract_hints(inspection, seen)["installed_return_contracts"] == {"api.operation": schema}
    repeated = _new_contract_hints(inspection, seen)
    assert repeated["installed_return_contracts"] == {}
    assert repeated["issues"] == inspection["issues"]
    assert repeated["unchanged_return_contracts"] == ["api.operation"]
    changed = {**inspection, "installed_return_contracts": {"api.operation": {"type": "array"}}}
    assert _new_contract_hints(changed, seen)["installed_return_contracts"]
    assert inspection["installed_return_contracts"] == {"api.operation": schema}


def test_explicit_field_requests_retain_the_field_table(monkeypatch):
    def measure_state(candidate, *, machine_contract):
        pass

    measure_state.__module__ = "brilliance.accelerator_api"
    monkeypatch.setattr(accelerator_api, "measure_state", measure_state, raising=False)
    reference = """## Measurement
api.measure_state(candidate, machine_contract=machine)
## Machine fields
| Field | Meaning |
| --- | --- |
| id_drift_names | Names from the exact candidate |
## Other calculations
Tracking and optimization are separate requests.
"""
    text = api_help(reference, "measure_state machine_contract id_drift_names")
    assert "## Measurement" in text and "## Machine fields" in text
    assert "Names from the exact candidate" in text
    assert "Other calculations" not in text


def test_recorded_query_shapes_keep_contracts_with_lower_average_payload():
    root = Path(__file__).resolve().parents[1]
    reference = (root / "brilliance/resources/accelerator_api_reference.md").read_text(encoding="utf-8")
    registry = ExpertRegistry([load_module_path(root / "expert_modules/mba_design")])
    # Query texts only, not a dependency on user projects or their result files.
    queries = [
        "verify_conventional_candidate",
        "symmetric_cell_elements build_jutrack_editor",
        "magnet_strength_limits audit_lattice_constraints",
        "periodic_cell evaluate_periodic_cell cell_ring mirror_elements repeat_cell one_period_elements",
        "jutrack_linear_metrics profile sampler save_optics_lattice_plot",
        "run.finish candidate_lattice candidate_evaluation acceptance",
        "particle_swarm_optimize optimization_options checkpoint",
        "evaluate_candidate editor linear_metrics return keys radiation periodic",
        "drift quadrupole sector_bend element constructor helpers",
        "verify_hmba_candidate machine_contract fields limits id_drift_names achromat",
        "jutrack_periodic_radiation_metrics radiation emittance keys physical_validity",
    ]
    texts = [api_help(reference, query, expert_registry=registry) for query in queries]
    assert sum(map(len, texts)) / len(texts) < 9000
    for query, text in zip(queries, texts):
        for token in query.split():
            if callable(vars(accelerator_api).get(token)):
                assert "api." + token + "(" in text or "Signature help truncated" in text
    assert "## Saved program" not in texts[3] or texts[3].index("Lattice construction") < texts[3].index("Saved program")
    assert "## Parameterization and matching" not in texts[8]
    assert "id_drift_names" in texts[9]


def test_evaluation_help_identifies_unavailable_values_instead_of_promising_numbers():
    from brilliance.result_contracts import compact_return_contracts, return_contract

    schema = return_contract(accelerator_api.evaluation_snapshot)
    optics = schema["properties"]["linear_optics"]["properties"]
    radiation = schema["properties"]["radiation"]["properties"]
    assert optics["Dx_m"]["nullable"]
    assert radiation["emittance_pm"]["nullable"]
    text = api_help("", "evaluation_snapshot", max_chars=20000)
    assert "nullable values (not zero):" in text
    compact, _ = compact_return_contracts({"api.evaluation_snapshot": schema})
    assert "emittance_pm" in compact["api.evaluation_snapshot"]["nested"]["radiation"]["nullable_values"]
