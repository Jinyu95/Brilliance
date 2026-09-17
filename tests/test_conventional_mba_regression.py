"""Pin a real conventional route before optional HMBA changes.

Historical files remain in their original project and are never copied into an
expert module. The fixture is regression-only. Real physics is opt-in:
BRILLIANCE_RUN_JUTRACK_TESTS=1 with the configured interpreter. No optimizer,
seed generation, model call, or complete saved program is executed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from brilliance import accelerator_api as api
from brilliance.research_environment import ResearchEnvironment

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "conventional_mba_regression.json"


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved_conventional():
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    workspace = ROOT / expected["project_workspace"]
    missing = [name for name in expected["artifact_sha256"] if not (workspace / name).is_file()]
    if missing:
        message = "Exact conventional regression history unavailable: " + ", ".join(missing)
        if os.environ.get("BRILLIANCE_RUN_JUTRACK_TESTS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    for relative, fingerprint in expected["artifact_sha256"].items():
        assert _hash(workspace / relative) == fingerprint, relative
    selected_folder = workspace / expected["selected_program"]
    parent_folder = workspace / expected["parent_program"]
    selected = api.load_lattice(selected_folder / "candidate_lattice.json")
    parent = api.load_lattice(parent_folder / "candidate_lattice.json")
    assert selected.lattice_id == expected["lattice_id"]
    assert parent.lattice_id == expected["parent_lattice_id"]
    assert selected.parent_lattice_id == parent.lattice_id
    assert json.loads((selected_folder / "arguments.json").read_text(encoding="utf-8")) == {}
    yield expected, workspace, parent, selected
    # This fixture also proves computations never mutate historical evidence.
    for relative, fingerprint in expected["artifact_sha256"].items():
        assert _hash(workspace / relative) == fingerprint, relative


def _reconstructed_rows(expected, workspace, parent):
    baseline = api.one_period_elements(parent)
    indices = {row["name"]: index for index, row in enumerate(baseline)}
    selected_inputs = expected["reconstruction"]
    k1_max = expected["magnet_limits_from_source_defaults"]["max_abs_k1_m2"]
    design = api.infer_mirror_symmetric_parameterization(
        baseline,
        fields=("k1_m2",),
        k1_bounds=(-0.999 * k1_max, 0.999 * k1_max),
        fixed_element_names=tuple(r["name"] for r in baseline if r["kind"] != "KQUAD"),
    )
    assert len(design.knobs) == len(selected_inputs["quadrupole_pairs"]) == 14
    strength_by_pair = {
        frozenset((row["left"], row["right"])): row["k1_m2"]
        for row in selected_inputs["quadrupole_pairs"]
    }
    strengths = [strength_by_pair[frozenset(knob.target_names)] for knob in design.knobs]
    groups, deltas = [], []
    for item in selected_inputs["drift_deltas"]:
        i, j = indices[item["left"]], indices[item["right"]]
        assert j == len(baseline) - 1 - i
        assert baseline[i]["kind"] == baseline[j]["kind"] == "DRIFT"
        groups.append(
            {"i": i, "j": j, "li": baseline[i]["length_m"], "lj": baseline[j]["length_m"]}
        )
        deltas.append(item["delta_length_m"])
    assert len(groups) == 20
    assert sum(2 * value for value in deltas) == pytest.approx(0.0, abs=1e-12)
    source = workspace / expected["selected_program"] / "program.py"
    # Only the pinned, pure construction function runs, not the archived search.
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_rows"
    ]
    assert len(functions) == 1
    executable = ast.Module(body=functions, type_ignores=[])
    namespace = {"design": design, "drift_groups": groups}
    exec(compile(executable, str(source), "exec"), namespace)
    return namespace["build_rows"](strengths, deltas)


def _assert_same_physical_rows(first, second, tolerance):
    assert len(first) == len(second)
    for before, after in zip(first, second):
        assert before.keys() == after.keys()
        for key, value in before.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert after[key] == pytest.approx(value, rel=0, abs=tolerance), (
                    before["name"],
                    key,
                )
            else:
                assert after[key] == value, (before["name"], key)


def test_selected_conventional_reconstructed_from_pinned_source_inputs(saved_conventional):
    expected, workspace, parent, selected = saved_conventional
    rebuilt = _reconstructed_rows(expected, workspace, parent)
    _assert_same_physical_rows(
        api.one_period_elements(selected),
        rebuilt,
        expected["regression_tolerances"]["row_field_absolute"],
    )


def test_conventional_geometry_and_global_symmetry_unchanged(saved_conventional):
    expected, _, parent, selected = saved_conventional
    rows = api.one_period_elements(selected)
    original = api.one_period_elements(parent)
    geometry = expected["geometry"]
    assert len(rows) == geometry["elements_per_cell"]
    assert selected.energy_gev == geometry["energy_gev"]
    assert selected.metadata["intended_repetitions"] == geometry["periods"]
    assert rows[0] == original[0] and rows[-1] == original[-1]
    assert rows[0]["length_m"] + rows[-1]["length_m"] == geometry["insertion_straight_m"]
    limits = dict(expected["magnet_limits_from_source_defaults"])
    maximum_k1 = limits.pop("max_abs_k1_m2")
    limits = api.magnet_strength_limits(energy_gev=geometry["energy_gev"], **limits)
    assert limits["max_abs_k1_m2"] == pytest.approx(maximum_k1, rel=1e-14)
    audit = api.audit_lattice_constraints(
        rows,
        energy_gev=geometry["energy_gev"],
        expected_length_m=geometry["cell_length_m"],
        expected_angle_rad=geometry["cell_bend_angle_rad"],
        expected_bends=geometry["positive_bends_per_cell"],
        require_mirror_symmetry=True,
        limits=limits,
    )
    assert audit["passed"], audit["issues"]
    assert geometry["periods"] * audit["angle_rad"] == pytest.approx(2 * math.pi, abs=1e-10)


@pytest.mark.skipif(
    os.environ.get("BRILLIANCE_RUN_JUTRACK_TESTS") != "1",
    reason="Opt-in real configured JuTrack conventional regression",
)
def test_real_jutrack_conventional_periodic_and_full_ring_regression(
    saved_conventional, monkeypatch
):
    expected, workspace, parent, selected = saved_conventional
    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Conventional MBA regression")
    for key, value in environment.process_environment().items():
        monkeypatch.setenv(key, value)
    settings = expected["geometry"]
    tolerance = expected["regression_tolerances"]
    references = expected["measured_baseline"]
    reconstructed = api.periodic_cell(
        title="Regression-only conventional source replay",
        energy_gev=settings["energy_gev"],
        elements=_reconstructed_rows(expected, workspace, parent),
        intended_repetitions=settings["periods"],
        source="test_conventional_mba_regression.py",
        parent_lattice_id=parent.lattice_id,
    )
    for candidate in (selected, reconstructed):
        measured = api.evaluate_periodic_cell(candidate, include_profile=True)
        linear, radiation = measured["linear_optics"], measured["radiation"]
        assert linear["stable"] and radiation["computed"]
        assert radiation["physical_validity"]["passed"]
        for key, reference in references["periodic_optics"].items():
            absolute = (
                tolerance["dispersion_reproduction_absolute_m"]
                if key == "Dx_m"
                else tolerance["slope_reproduction_absolute"]
                if key == "Dpx"
                else tolerance["optics_absolute"]
            )
            assert linear[key] == pytest.approx(
                reference, rel=tolerance["optics_relative"], abs=absolute
            ), key
        for endpoint in (linear["twiss_profile"][0], linear["twiss_profile"][-1]):
            assert (
                abs(endpoint["Dx"])
                < expected["original_achromat_requirements"]["id_dispersion_tolerance_m"]
            )
            assert (
                abs(endpoint["Dpx"])
                < expected["original_achromat_requirements"]["id_slope_tolerance"]
            )
        for key, reference in references["radiation"].items():
            absolute = (
                tolerance["emittance_absolute_pm"]
                if key == "emittance_pm"
                else tolerance["radiation_absolute"]
            )
            assert radiation[key] == pytest.approx(
                reference, rel=tolerance["radiation_relative"], abs=absolute
            ), key
        assert measured["geometry"]["ring_bend_closure_passed"]
    ring = api.cell_ring(
        title="Independent conventional regression ring",
        energy_gev=settings["energy_gev"],
        cell=api.one_period_elements(reconstructed),
        periods=settings["periods"],
        source="test_conventional_mba_regression.py",
        parent_lattice_id=selected.lattice_id,
    )
    full = api.evaluate(ring, include_radiation=True, include_profile=False)
    assert full.linear_optics.stable and full.geometry["ring_bend_closure_passed"]
    assert full.geometry["circumference_m"] == pytest.approx(
        settings["circumference_m"], abs=tolerance["geometry_absolute"]
    )
    radiation = full.require_radiation()
    assert radiation["physical_validity"]["passed"]
    for key, reference in references["radiation"].items():
        absolute = (
            tolerance["emittance_absolute_pm"]
            if key == "emittance_pm"
            else tolerance["radiation_absolute"]
        )
        assert radiation[key] == pytest.approx(
            reference, rel=tolerance["radiation_relative"], abs=absolute
        ), key
    for key, reference in references["full_ring_optics"].items():
        assert getattr(full.linear_optics, key) == pytest.approx(
            reference, rel=tolerance["optics_relative"], abs=tolerance["optics_absolute"]
        )
    assert full.linear_optics.periodic_closure_max < tolerance["optics_absolute"]
    print(
        "CONVENTIONAL_REGRESSION "
        + json.dumps(
            {
                "lattice_id": selected.lattice_id,
                "emittance_pm": radiation["emittance_pm"],
                "I2": radiation["I2"],
                "I4": radiation["I4"],
                "I5": radiation["I5"],
                "circumference_m": full.geometry["circumference_m"],
                "passed": True,
                "source_reconstruction": True,
                "new_optimization_performed": False,
            },
            sort_keys=True,
        )
    )
