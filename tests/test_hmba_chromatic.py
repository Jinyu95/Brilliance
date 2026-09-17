"""Synthetic algebra and adapter-contract tests; NOT accelerator-physics validation."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from brilliance.lattice import LatticeInputError
from expert_modules.mba_design.hmba_chromatic import (
    measure_and_solve_chromaticity,
    solve_chromatic_correction,
)


def fixture_rows():
    return [
        {"name": "D0", "kind": "DRIFT", "length_m": 0.4},
        {"name": "SF_L", "kind": "KSEXT", "length_m": 0.2, "k2_m3": 0.0},
        {"name": "SD_L", "kind": "KSEXT", "length_m": 0.3, "k2_m3": 0.0},
        {"name": "SD_R", "kind": "KSEXT", "length_m": 0.3, "k2_m3": 0.0},
        {"name": "SF_R", "kind": "KSEXT", "length_m": 0.2, "k2_m3": 0.0},
    ]


def families(bounds=(-100.0, 100.0)):
    return {
        name: {
            "element_names": [name + "_L", name + "_R"],
            "k2_bounds_m3": list(bounds),
            "response_step_m3": 0.5,
        }
        for name in ("SF", "SD")
    }


def coordinates(rows):
    by_name = {row["name"]: row for row in rows}
    for name in ("SF", "SD"):
        assert by_name[name + "_L"]["k2_m3"] == by_name[name + "_R"]["k2_m3"]
    return np.array([by_name["SF_L"]["k2_m3"], by_name["SD_L"]["k2_m3"]])


def model(matrix=None):
    response = np.asarray(matrix if matrix is not None else [[0.5, -0.2], [-0.1, 0.4]])
    return lambda rows: np.array([-10.0, -8.0]) + response @ coordinates(rows)


def solve(rows=None, specification=None, measure=None, verify=None, **options):
    measurement = model() if measure is None else measure
    verification = model() if verify is None else verify
    return measure_and_solve_chromaticity(
        fixture_rows() if rows is None else rows,
        families() if specification is None else specification,
        measurement,
        verification,
        **{
            "target_chromaticities": (2.0, 2.0),
            "chromaticity_tolerance": (1e-8, 1e-8),
            "periods": 32,
            "max_response_condition": 1e6,
            "evaluation_budget": 32,
            **options,
        },
    )


def test_synthetic_two_plane_response_and_independent_verification():
    source = fixture_rows()
    unchanged = deepcopy(source)
    verified = []
    progress = []

    def verify(rows):
        verified.append(deepcopy(rows))
        return model()(rows)

    result = solve(source, verify=verify, progress=progress.append)
    assert result["passed"] and result["status"] == "corrected"
    assert result["response"]["rank"] == 2
    assert np.asarray(result["response"]["matrix"]) == pytest.approx(
        np.array([[0.5, -0.2], [-0.1, 0.4]])
    )
    assert result["corrected_chromaticities"] == pytest.approx({"horizontal": 2, "vertical": 2})
    assert len(verified) == 1 and verified[0] == result["corrected_rows"]
    assert source == unchanged
    assert result["evaluations"] == len(progress) == 7
    assert progress[-1]["purpose"] == "independent_verification"
    assert all(result["checks"].values())
    assert result["full_ring_validation_required"] is True


def test_lengths_and_multiplicities_are_not_conflated_with_strength_coordinate():
    rows = fixture_rows()
    rows[-1]["length_m"] = 0.25
    result = solve(rows)
    table = {row["element"]: row for row in result["strengths"]}
    assert table["SF_L"]["K2_m3"] == table["SF_R"]["K2_m3"]
    assert table["SF_R"]["K2L_m2"] / table["SF_L"]["K2L_m2"] == pytest.approx(1.25)
    assert table["SF_L"]["family_multiplicity_per_cell"] == 2
    assert table["SF_L"]["family_multiplicity_full_ring"] == 64
    assert "k2 / 2" in result["normalization"]["tracking_polynomial"]
    assert result["normalization"]["jutrack_field"] == "KSEXT.k2"


def test_synthetic_rank_deficiency_is_reported_without_fake_correction():
    result = solve(measure=model([[1, 2], [2, 4]]))
    assert result["status"] == "rank_deficient"
    assert result["response"]["rank"] == 1
    assert not result["passed"] and result["corrected_rows"] is None
    assert result["candidate_rows"] == fixture_rows()
    assert result["evaluations"] == 5


def test_synthetic_ill_conditioned_response_is_explicit():
    result = solve(measure=model([[1, 0], [0, 1e-8]]), max_response_condition=1e5)
    assert result["response"]["rank"] == 2
    assert result["response"]["condition_number"] > 1e7
    assert result["status"] == "ill_conditioned"
    assert result["passed"] is False


def test_bounded_solution_never_evaluates_overstrength_magnets():
    measured = []

    def measure(rows):
        values = coordinates(rows)
        assert np.all(np.abs(values) <= 1.0)
        measured.append(values.copy())
        return model()(rows)

    result = solve(specification=families((-1.0, 1.0)), measure=measure, max_iterations=2)
    assert result["status"] == "strength_limit"
    assert result["unconstrained_strength_violations"]
    assert result["passed"] is False and result["corrected_rows"] is None
    assert all(row["within_limit"] for row in result["strengths"])
    assert result["max_strength_fraction"] <= 1.0
    assert measured


def test_initial_strength_violation_does_not_start_numerical_work():
    rows = fixture_rows()
    for row in rows:
        if row["kind"] == "KSEXT":
            row["k2_m3"] = 2.0

    def forbidden(rows):
        pytest.fail("Over-limit initial strengths must not be silently clamped or evaluated")

    result = solve(rows, families((-1, 1)), measure=forbidden)
    assert result["status"] == "strength_limit" and result["evaluations"] == 0
    assert result["candidate_rows"] == rows


def test_budget_exhaustion_preserves_partial_response_and_baseline():
    result = solve(evaluation_budget=3)
    assert result["status"] == "resource_limit" and result["passed"] is False
    assert result["evaluations"] == 3
    assert len(result["response"]["columns"]) == 1
    assert result["response"]["complete"] is False
    assert result["candidate_rows"] == fixture_rows()
    assert result["baseline_chromaticities"] == {"horizontal": -10.0, "vertical": -8.0}


def test_budget_must_include_independent_verification():
    result = solve(evaluation_budget=6)
    assert result["status"] == "resource_limit"
    assert result["candidate_chromaticities"] == pytest.approx({"horizontal": 2.0, "vertical": 2.0})
    assert result["corrected_rows"] is None and result["passed"] is False
    assert result["checks"]["independent_verification"] is False


def test_independent_disagreement_cannot_be_accepted_as_affine_prediction():
    result = solve(verify=lambda rows: model()(rows) + np.array([0.1, -0.2]))
    assert result["status"] == "verification_failed"
    assert result["corrected_chromaticities"] == pytest.approx({"horizontal": 2.1, "vertical": 1.8})
    assert result["residuals"] == pytest.approx({"horizontal": 0.1, "vertical": -0.2})
    assert result["passed"] is False


def test_response_probes_respect_boundaries_with_one_sided_difference():
    rows = fixture_rows()
    for row in rows:
        if row["name"].startswith("SF"):
            row["k2_m3"] = 1.0
    result = solve(rows, families((-1, 1)), max_iterations=1)
    column = result["response"]["columns"][0]
    assert column["upper_probe_K2_m3"] == 1.0
    assert column["lower_probe_K2_m3"] == 0.5
    assert not any(
        record["purpose"] == "response_plus:SF" for record in result["evaluation_history"]
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r, f: r[1].update(kind="DRIFT"), "actual KSEXT"),
        (lambda r, f: r[1].update(length_m=0), "length_m"),
        (lambda r, f: r[1].update(k2_m3=None), "must be declared"),
        (lambda r, f: r[1].update(k2_m3=1), "share one physical"),
        (lambda r, f: f["SF"].update(element_names=["missing"]), "unknown sextupole"),
        (lambda r, f: f["SD"].update(element_names=["SF_L"]), "two correction families"),
        (lambda r, f: f["SF"].update(k2_bounds_m3=[1, 1]), "increasing"),
        (lambda r, f: f["SF"].update(response_step_m3=0), "response_step"),
        (lambda r, f: f.pop("SD"), "at least two"),
    ],
)
def test_invalid_physical_family_contract_fails_before_measurement(mutate, message):
    rows, specification = fixture_rows(), families()
    mutate(rows, specification)

    def forbidden(rows):
        pytest.fail("Invalid contracts must be detected before JuTrack")

    with pytest.raises(LatticeInputError, match=message):
        solve(rows, specification, measure=forbidden)


def test_nonfinite_measurement_is_a_visible_api_failure_not_physics_infeasibility():
    with pytest.raises(LatticeInputError, match="two finite"):
        solve(measure=lambda rows: [np.nan, 1])


def test_runtime_errors_propagate_instead_of_becoming_rank_or_limit_failures():
    def unavailable(rows):
        raise RuntimeError("configured Julia worker unavailable")

    with pytest.raises(RuntimeError, match="Julia worker"):
        solve(measure=unavailable)


class FakeJuTrack:
    """Algebraic fake for API ownership/scaling tests, not a tracking model."""

    def __init__(self):
        self.calls = []

    def DRIFT(self, name, length):
        return SimpleNamespace(name=name, len=length, kind="DRIFT")

    def KSEXT(self, name, length, k2):
        return SimpleNamespace(name=name, len=length, kind="KSEXT", k2=k2)

    def Lattice(self, rows):
        return list(rows)

    def getchrom(self, lattice, *, dp, energy):
        self.calls.append((lattice, dp, energy))
        sf = sum(item.k2 * item.len for item in lattice if item.name.startswith("SF"))
        sd = sum(item.k2 * item.len for item in lattice if item.name.startswith("SD"))
        return np.array([-2 + sf - 0.25 * sd, -1 - 0.2 * sf + sd])


class FakeEditor:
    def __init__(self):
        self.jt = FakeJuTrack()
        self.rows = fixture_rows()
        self.lattice = [
            self.jt.DRIFT(r["name"], r["length_m"])
            if r["kind"] == "DRIFT"
            else self.jt.KSEXT(r["name"], r["length_m"], r["k2_m3"])
            for r in self.rows
        ]

    def export_rows(self):
        rows = deepcopy(self.rows)
        for row, item in zip(rows, self.lattice):
            if row["kind"] == "KSEXT":
                row["k2_m3"] = item.k2
        return rows

    def sync_rows(self, rows):
        for row, item in zip(rows, self.lattice):
            if row["kind"] == "KSEXT":
                item.k2 = row["k2_m3"]


def wrapper(editor, **options):
    return solve_chromatic_correction(
        editor,
        fixture_rows(),
        energy_gev=3.5,
        periods=32,
        families=families(),
        target_chromaticities=(2, 2),
        chromaticity_tolerance=(1e-8, 1e-8),
        max_response_condition=1e6,
        evaluation_budget=32,
        **options,
    )


def test_synthetic_adapter_uses_actual_K2_lengths_ring_scaling_and_fresh_verifier():
    editor = FakeEditor()
    unchanged = editor.export_rows()
    result = wrapper(editor)
    assert result["passed"]
    expected = 32 * np.asarray([[0.4, -0.15], [-0.08, 0.6]])
    assert np.asarray(result["response"]["matrix"]) == pytest.approx(expected)
    assert result["baseline_chromaticities"] == {"horizontal": -64, "vertical": -32}
    assert all(call[0] is editor.lattice for call in editor.jt.calls[:-1])
    assert editor.jt.calls[-1][0] is not editor.lattice
    assert all(dp == 0 and energy == 3.5e9 for _, dp, energy in editor.jt.calls)
    assert editor.export_rows() == unchanged
    assert "supplied origin" in result["limitations"][-1]


def test_synthetic_adapter_restores_original_editor_even_when_measurement_raises():
    editor = FakeEditor()
    original_getchrom = editor.jt.getchrom
    unchanged = editor.export_rows()

    def fail_during_probe(*args, **kwargs):
        if len(editor.jt.calls) == 1:
            raise RuntimeError("simulated Julia bridge failure")
        return original_getchrom(*args, **kwargs)

    editor.jt.getchrom = fail_during_probe
    with pytest.raises(RuntimeError, match="bridge failure"):
        wrapper(editor)
    assert editor.export_rows() == unchanged


@pytest.mark.skipif(
    os.environ.get("BRILLIANCE_RUN_JUTRACK_TESTS") != "1",
    reason="Opt-in real configured JuTrack chromatic-response integration",
)
def test_real_jutrack_response_and_fresh_correction_on_copied_conventional_cell(monkeypatch):
    """Real finite-sextupole response, not HMBA phase/nonlinear-design validation.

    Original machine limits come from the pinned conventional regression. The
    explicit new test layout has four 0.1-m sextupoles, 0.02-m clearances. A
    nearby diagnostic target is measured from a tiny known bounded perturbation;
    it does NOT replace the separate proposed +2,+2 correction objective.
    """
    from brilliance import accelerator_api as api
    from brilliance.research_environment import ResearchEnvironment

    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="HMBA chromatic-response integration")
    for key, value in environment.process_environment().items():
        monkeypatch.setenv(key, value)
    fixture = json.loads(
        (Path(__file__).with_name("fixtures") / "conventional_mba_regression.json").read_text(
            encoding="utf-8"
        )
    )
    root = Path(__file__).resolve().parents[1]
    relative = fixture["selected_program"] + "/candidate_lattice.json"
    parent_path = root / fixture["project_workspace"] / relative
    original_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    assert original_hash == fixture["artifact_sha256"][relative]
    parent = api.load_lattice(parent_path)
    energy = fixture["geometry"]["energy_gev"]
    periods = fixture["geometry"]["periods"]
    original_limits = dict(fixture["magnet_limits_from_source_defaults"])
    original_limits.pop("max_abs_k1_m2")
    maximum = api.magnet_strength_limits(energy_gev=energy, **original_limits)["max_abs_k2_m3"]
    hosts = {"SF_L": "P001:Dend", "SF_R": "P005:R:Dend", "SD_L": "LEFT:T:D1", "SD_R": "T:D1"}
    # This tests the reusable chromatic response, not a prescribed HMBA layout.
    rows = api.one_period_elements(parent)
    for name, host in hosts.items():
        rows = api.insert_element_in_drift(
            rows, drift_name=host,
            inserted_element=api.sextupole("HMBA:" + name, 0.1, 0.0, family=name[:2]),
            split_ratio=0.5, minimum_drift_segment_m=0.02,
        )
    candidate = api.periodic_cell(
        title="Chromatic response regression", energy_gev=energy, elements=rows,
        intended_repetitions=periods, source="test_hmba_chromatic.py",
        parent_lattice_id=parent.lattice_id,
    )
    rows = api.one_period_elements(candidate)
    source_rows = deepcopy(rows)
    editor = api.build_jutrack_editor(candidate)
    specification = {
        name: {
            "element_names": ["HMBA:" + name + "_L", "HMBA:" + name + "_R"],
            "k2_bounds_m3": [-maximum, maximum],
            "response_step_m3": 1.0,
        }
        for name in ("SF", "SD")
    }
    options = {
        "energy_gev": energy,
        "periods": periods,
        "families": specification,
        "chromaticity_tolerance": (0.01, 0.01),
        "max_response_condition": 1e6,
        "evaluation_budget": 32,
        "max_iterations": 3,
    }
    requested = solve_chromatic_correction(
        editor, rows, target_chromaticities=(2.0, 2.0), **options
    )
    assert requested["status"] in {"corrected", "strength_limit"}, requested
    assert requested["response"]["rank"] == 2
    assert requested["max_strength_fraction"] <= 1.0
    assert editor.export_rows(base_elements=rows) == source_rows

    # This is a separate explicitly diagnostic target, not a loosened design requirement.
    probe = deepcopy(rows)
    for row in probe:
        if row["name"].startswith("HMBA:SF"):
            row["k2_m3"] = 1.0
        elif row["name"].startswith("HMBA:SD"):
            row["k2_m3"] = -1.0
    try:
        editor.sync_rows(probe)
        diagnostic_target = periods * np.asarray(
            editor.jt.getchrom(editor.lattice, dp=0.0, energy=energy * 1e9)
        )
    finally:
        editor.sync_rows(source_rows)
    diagnostic = solve_chromatic_correction(
        editor, rows, target_chromaticities=diagnostic_target, **options
    )
    assert diagnostic["passed"], diagnostic
    assert all(diagnostic["checks"].values())
    assert diagnostic["max_strength_fraction"] <= 1.0
    assert rows == source_rows and editor.export_rows(base_elements=rows) == source_rows
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == original_hash
    print(
        "REAL_CHROMATIC_RESPONSE "
        + json.dumps(
            {
                "parent_sha256": original_hash,
                "K2_max_m3": maximum,
                "requested_target": requested["target_chromaticities"],
                "requested_status": requested["status"],
                "requested_strengths": requested["family_coordinates_m3"],
                "requested_xi": requested["candidate_chromaticities"],
                "response": requested["response"],
                "baseline_xi": requested["baseline_chromaticities"],
                "diagnostic_only_target": diagnostic["target_chromaticities"],
                "diagnostic_status": diagnostic["status"],
                "diagnostic_strengths": diagnostic["family_coordinates_m3"],
                "diagnostic_corrected_xi": diagnostic["corrected_chromaticities"],
            },
            sort_keys=True,
        )
    )
