"""A shared evaluation reader must not substitute targets or lose native evidence."""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence
from brilliance.jutrack_service import JuTrackService
from brilliance.lattice_optimization import jutrack_linear_metrics


def cell():
    return api.periodic_cell(title="Result schema regression", energy_gev=3.5,
        elements=[api.drift("D", 23.94), api.sector_bend("B", 1.0, 2 * math.pi / 32)],
        source="synthetic reporting test", intended_repetitions=32,
        metadata={"design_constraints": [{"quantity": "circumference_m", "target": 800.0}]})


def ring():
    return api.cell_ring(title="Result schema full ring", energy_gev=3.5,
        cell=api.one_period_elements(cell()), periods=32, source="synthetic reporting test",
        design_constraints=[{"quantity": "circumference_m", "target": 800.0}])


def linear():
    return {"stable": True, "trace_x": 0.0, "trace_y": 0.1, "tune_x": 0.2, "tune_y": 0.3,
            "beta_x_m": 3.0, "beta_y_m": 4.0, "alpha_x": 0.2, "alpha_y": -0.3,
            "Dx_m": 0.012, "Dpx": -0.001, "Dy_m": 0.002, "Dpy": -0.0001,
            "twiss_profile": [{"name": "START", "s_m": 0.0, "beta_x": 3.0, "beta_y": 4.0,
                                "alpha_x": 0.2, "alpha_y": -0.3, "Dx": 0.012, "Dpx": -0.001,
                                "Dy": 0.002, "Dpy": -0.0001}],
            "solver": "test native adapter"}


def radiation(*, valid=True):
    return {"computed": True, "emittance_pm": 38.123456789 if valid else -20000.0,
            "I1": 0.03, "I2": 0.2, "I3": 0.004, "I4": 0.001, "I5": 5e-7,
            "Jx": 0.995, "Jy": 1.0, "Jz": 2.005, "energy_spread": 5e-4,
            "energy_loss_per_turn_eV": 700000.0,
            "physical_validity": {"passed": valid, "checks": {"positive_natural_emittance": valid}},
            "jutrack_ring_parameters": {"extra_native_field": [1.1, 2.2]}}


@pytest.fixture
def simulated(monkeypatch):
    monkeypatch.setattr(api, "load_pyjutrack", lambda: object())
    monkeypatch.setattr(api, "_linear_evaluation", lambda *a, **k: linear())
    monkeypatch.setattr(api, "_radiation_evaluation", lambda *a, **k: radiation())
    monkeypatch.setattr(api, "natural_chromaticity", lambda *a, **k: {"horizontal": -20.0, "vertical": -10.0})
    monkeypatch.setattr(api, "jutrack_linear_metrics", lambda *a, **k: linear())
    monkeypatch.setattr(api, "jutrack_periodic_radiation_metrics", lambda *a, **k: radiation())
    monkeypatch.setattr(JuTrackService, "build_lattice", lambda *a, **k: object())
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)


def test_one_reader_for_typed_ring_and_periodic_dictionary_preserves_actual_scope(simulated):
    explicit = api.evaluate(ring(), include_profile=True, include_chromaticity=True)
    unit = api.evaluate_periodic_cell(cell(), periods=32, include_profile=True)
    full, reference = api.evaluation_snapshot(explicit), api.evaluation_snapshot(unit)
    assert set(full) == set(reference)
    assert set(full["geometry"]) == set(reference["geometry"])
    assert full["geometry"]["stored_length_m"] == pytest.approx(798.08)
    assert full["geometry"]["ring_circumference_m"] == pytest.approx(798.08)
    assert full["geometry"]["cell_length_m"] == pytest.approx(24.94)
    assert full["geometry"]["reference_ring_circumference_m"] is None
    assert reference["geometry"]["stored_length_m"] == pytest.approx(24.94)
    assert reference["geometry"]["ring_circumference_m"] is None
    assert reference["geometry"]["reference_ring_circumference_m"] == pytest.approx(798.08)
    assert full["checks"]["full_ring_evaluated"]
    assert not reference["checks"]["full_ring_evaluated"]
    for snapshot in (full, reference):
        assert snapshot["linear_optics"]["alpha_x"] == 0.2
        assert snapshot["linear_optics"]["Dy_m"] == 0.002
        assert snapshot["radiation"]["emittance_pm"] == 38.123456789
        assert snapshot["radiation"]["jutrack_ring_parameters"]["extra_native_field"] == [1.1, 2.2]
        assert not snapshot["acceptance"]["accepted"]
        assert snapshot["acceptance"]["machine_requirements"] == "not_assessed"
    assert full["chromaticity"]["computed"] and full["chromaticity"]["horizontal"] == -20.0
    assert not reference["chromaticity"]["computed"]
    assert explicit.geometry["circumference_m"] == pytest.approx(798.08)  # unchanged public API
    with pytest.raises(AttributeError):
        getattr(explicit, "geometry_summary")  # no dummy alias for a fabricated target summary


def test_reference_geometry_uses_actual_evaluation_periods_not_stored_intent(simulated):
    # Doubling the reference count fails bend closure. It must neither mutate
    # the cell's declared intent nor silently call the 64-fold reference 32 cells.
    candidate = cell()
    result = api.evaluate_periodic_cell(candidate, periods=64)
    snapshot = api.evaluation_snapshot(result)
    assert snapshot["geometry"]["evaluated_periods"] == 64
    assert snapshot["geometry"]["reference_ring_circumference_m"] == pytest.approx(24.94 * 64)
    assert not snapshot["checks"]["ring_bend_closure_passed"]
    assert not snapshot["radiation"]["computed"]
    assert candidate.metadata["intended_repetitions"] == 32


def test_row_geometry_is_inside_original_runtime_receipt_not_an_unbound_view(simulated, tmp_path):
    candidate = ring()
    result = api.evaluate(candidate)
    verified = candidate_evidence.verify_candidate_evaluation(candidate, result, tmp_path)
    assert verified["passed"]
    archive = json.loads((tmp_path / verified["receipt"]["path"]).read_text(encoding="utf-8"))
    assert archive["calculation"]["evaluation_geometry"]["candidate_sha256"] == archive["binding"]["candidate_sha256"]
    # Reporting-only recovery reads the saved calculation, not a new simulation.
    restored = archive["calculation"] | {"evaluation_receipt": result.raw["evaluation_receipt"]}
    assert api.evaluation_snapshot(restored) == api.evaluation_snapshot(result)
    changed = copy.deepcopy(restored)
    changed["evaluation_geometry"]["ring_circumference_m"] = 800.0
    assert not candidate_evidence.verify_candidate_evaluation(candidate, changed, tmp_path)["passed"]


def test_reader_is_detached_does_not_start_julia_or_alter_original_receipt(simulated, monkeypatch):
    result = api.evaluate(ring(), include_profile=True)
    original = result.as_dict()
    monkeypatch.setattr(api, "load_pyjutrack", lambda: pytest.fail("Reader must not initialize Julia"))
    snapshot = api.evaluation_snapshot(result)
    assert api.evaluation_snapshot(json.loads(json.dumps(original))) == snapshot
    snapshot["radiation"]["emittance_pm"] = 0.0
    snapshot["linear_optics"]["twiss_profile"][0]["beta_x"] = 10000.0
    assert result.as_dict() == original


def test_computed_negative_emittance_does_not_masquerade_as_physical_or_request_acceptance(simulated, monkeypatch):
    monkeypatch.setattr(api, "_radiation_evaluation", lambda *a, **k: radiation(valid=False))
    snapshot = api.evaluation_snapshot(api.evaluate(ring()))
    assert snapshot["checks"]["passed_baseline_gate"]  # stability and bend closure only
    assert snapshot["checks"]["radiation_computed"]
    assert not snapshot["checks"]["radiation_physical_validity"]
    assert snapshot["radiation"]["emittance_pm"] == -20000.0
    assert not snapshot["acceptance"]["accepted"]


def test_missing_radiation_is_unavailable_not_dummy_zero(simulated):
    snapshot = api.evaluation_snapshot(api.evaluate(ring(), include_radiation=False))
    assert snapshot["radiation"]["emittance_pm"] is None
    assert snapshot["radiation"]["I2"] is None
    assert not snapshot["radiation"]["computed"]
    assert snapshot["checks"]["radiation_physical_validity"] is None
    assert "not requested" in snapshot["radiation"]["reason"]
    assert not snapshot["linear_optics"]["profile_available"]
    assert snapshot["linear_optics"]["beta_x_m"] == 3.0


def test_legacy_geometry_is_read_from_saved_rows_or_marked_unknown_not_guessed(simulated):
    explicit = api.evaluate(ring()).as_dict()
    explicit.pop("evaluation_geometry")
    snapshot = api.evaluation_snapshot(explicit)
    assert snapshot["geometry"]["source"] == "saved_evaluation_candidate_rows"
    assert snapshot["geometry"]["ring_circumference_m"] == pytest.approx(798.08)
    legacy = api.evaluate_periodic_cell(cell())
    legacy.pop("evaluation_geometry")
    snapshot = api.evaluation_snapshot(legacy)
    assert snapshot["geometry"]["source"] == "legacy_evaluation_without_rows"
    assert not snapshot["geometry"]["row_geometry_available"]
    assert snapshot["geometry"]["ring_circumference_m"] is None
    assert snapshot["geometry"]["evaluated_periods"] is None
    assert snapshot["geometry"]["legacy_reported_geometry"]["predicted_ring_length_m"] == pytest.approx(798.08)


def test_compact_native_linear_metrics_keep_full_incoming_state_without_another_call():
    point = SimpleNamespace(mode=1, betax=3.0, betay=4.0, alphax=0.2, alphay=-0.3,
                            dx=0.01, dpx=0.002, dy=0.003, dpy=-0.004)
    matrix = np.eye(6)
    matrix[:4, :4] = [[0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., -1., 0.]]
    native = SimpleNamespace(findm66=lambda *a, **k: matrix,
                             periodicEdwardsTengTwiss=lambda *a, **k: point,
                             gettune=lambda *a, **k: [0.25, 0.25],
                             twissline=lambda *a, **k: pytest.fail("Compact reading must not request a profile"))
    result = jutrack_linear_metrics(native, [])
    expected = {"beta_x_m": 3.0, "beta_y_m": 4.0, "alpha_x": 0.2, "alpha_y": -0.3,
                "Dx_m": 0.01, "Dpx": 0.002, "Dy_m": 0.003, "Dpy": -0.004}
    assert {key: result[key] for key in expected} == expected


@pytest.mark.parametrize("value", [None, {}, {"engine": "JuTrack", "endpoint": {"beta_x_m": 2.0}},
                                     {"engine": "JuTrack", "geometry": {}, "linear_optics": None}])
def test_wrong_result_type_never_silently_becomes_an_empty_optics_mapping(value):
    with pytest.raises(TypeError, match="evaluation_snapshot requires"):
        api.evaluation_snapshot(value)
