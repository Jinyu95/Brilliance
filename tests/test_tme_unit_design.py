"""Synthetic contract/search regressions, not claims of accelerator feasibility."""

import copy
import json
import math
from types import SimpleNamespace

import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence
from brilliance.numerical_checkpoints import read_numerical_checkpoint
from brilliance.optics_sampling import UnsupportedOpticsSampling
from expert_modules.mba_design import tme_unit as unit


@pytest.fixture(autouse=True)
def independent_environment(monkeypatch):
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    candidate_evidence._captured.clear()


def arguments(**overrides):
    return dict(energy_gev=3.5, bend=api.sector_bend("B", .8, 2 * math.pi / 192),
        d1_m=.05, d2_m=.35, d3_m=.1, quadrupole_near_length_m=.3,
        quadrupole_far_length_m=.2, expected_length_m=2.8, reference_repetitions=192,
        limits=api.magnet_strength_limits(3.5), maximum_beta_m=100., trace_margin_x=.01,
        trace_margin_y=.01, minimum_magnet_length_m=.1, minimum_drift_length_m=.01,
        scan_resolutions=(5, 9), local_starts=4, local_max_evaluations=16, **overrides)


def synthetic(position):
    near, far = position
    return {"position": list(position), "passed": True, "checks": {"synthetic": True},
            "emittance_pm": 10 + (abs(near) - 1.4)**2 + (abs(far) - 2.)**2 + (10 if near < 0 else 0)}


def synthetic_evaluation(candidate, **kwargs):
    rows = api.one_period_elements(candidate)
    near = next(r["k1_m2"] for r in rows if "QNEAR_L" in r["name"])
    far = next(r["k1_m2"] for r in rows if "QFAR_L" in r["name"])
    result = {"engine": "JuTrack", "linear_optics": {"stable": True, "trace_x": .1, "trace_y": -.2,
              "maximum_beta_m": 10., "beta_x_m": 3.123456789123456, "beta_y_m": 7.,
              "alpha_x": .01, "alpha_y": -.02, "Dx_m": .03214567891234567, "Dpx": .001,
              "Dy_m": 0., "Dpy": 0.},
              "radiation": {"computed": True, "physical_validity": {"passed": True},
                            "emittance_pm": synthetic((near, far))["emittance_pm"]}}
    result["evaluation_receipt"] = candidate_evidence.record_evaluation(candidate, result,
        effective_periods=candidate.metadata.get("intended_repetitions", candidate.periodicity))
    return result


@pytest.fixture
def native_doubles(monkeypatch):
    monkeypatch.setattr(unit, "_NativeUnitEvaluator", lambda *args: synthetic)
    monkeypatch.setattr(api, "evaluate_periodic_cell", synthetic_evaluation)
    monkeypatch.setattr(api, "evaluate", synthetic_evaluation)
    def sampler(candidate, **kwargs):
        bend = next(r for r in api.one_period_elements(candidate) if r["kind"] == "SBEND")
        return SimpleNamespace(linear_metrics=lambda **kw: {"stable": True,
            "sampling": {"map_equivalent": True, "validated": True}, "twiss_profile": [
                {"requested_sample": True, "source_element": bend["name"], "sample_fraction": .5,
                 "beta_x": 2., "beta_y": 4., "Dx": .01, "Dpx": 0., "alpha_x": 0., "alpha_y": 0.}]})
    monkeypatch.setattr(api, "build_jutrack_profile_sampler", sampler)


def test_complete_eleven_rows_preserve_real_bend_and_global_pairs():
    bend = api.sector_bend("ACTUAL_B", .8, 2 * math.pi / 192, e1=.003, e2=.003,
                           fint1=.1, fint2=.1, NumIntSteps=13)
    before = copy.deepcopy(bend)
    rows = unit._unit_rows(bend, (.05, .35, .1, .3, .2), -3.123456789123456, 2.654321987654321)
    assert len(rows) == 11
    assert [r["kind"] for r in rows] == ["DRIFT", "KQUAD", "DRIFT", "KQUAD", "DRIFT", "SBEND",
                                        "DRIFT", "KQUAD", "DRIFT", "KQUAD", "DRIFT"]
    assert rows[5] == before == bend
    assert rows[3]["k1_m2"] == rows[7]["k1_m2"] == -3.123456789123456
    assert rows[1]["k1_m2"] == rows[9]["k1_m2"] == 2.654321987654321
    assert sum(r["length_m"] for r in rows) == pytest.approx(2.8)
    rows[5]["fields"]["e1"] = .99
    assert bend == before


def test_managed_child_output_archives_receipts_and_root_checkpoints(native_doubles, monkeypatch, tmp_path):
    from brilliance import research_api
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(tmp_path))
    monkeypatch.setattr(research_api, "open_research_run", lambda: SimpleNamespace(
        raise_if_cancelled=lambda: None, progress=lambda *args, **kwargs: None))
    child = tmp_path / "unit_stage"
    result = unit.design_tme_unit(**arguments(output_directory=child))
    assert result["passed"]
    assert result["numerical_checkpoint"]["source_output_directory"] == str(tmp_path)
    assert read_numerical_checkpoint(result["numerical_checkpoint"], output_directory=tmp_path)["checks_passed"]
    for candidate in result["candidates"]:
        assert candidate_evidence.verify_candidate_evaluation(candidate["periodic_cell"], candidate["evaluation"], tmp_path)["passed"]
        assert candidate["reference_ring_receipt"]["path"].startswith("unit_stage/")
    with pytest.raises(ValueError, match="managed run directory or its child"):
        unit.design_tme_unit(**arguments(output_directory=tmp_path.parent / "sibling"))


@pytest.mark.parametrize("changes,match", [({"expected_length_m": 2.7}, "declared contract"),
    ({"trace_margin_y": 2.}, "trace_margin"), ({"maximum_beta_m": -1}, "maximum_beta"),
    ({"scan_resolutions": (9,)}, "at least two"), ({"reference_repetitions": 191}, "2\\*pi"),
    ({"profile_step_m": 0}, "profile_step"), ({"d1_m": 0}, "positive and finite")])
def test_contract_errors_fail_before_loading_native_runtime(monkeypatch, changes, match):
    monkeypatch.setattr(unit, "_NativeUnitEvaluator", lambda *a: pytest.fail("Native must not load"))
    inputs = arguments()
    inputs.update(changes)
    with pytest.raises(ValueError, match=match):
        unit.design_tme_unit(**inputs)


@pytest.mark.parametrize("fields", [{"e1": .01, "e2": -.01}, {"fint1": .2, "fint2": .1},
                                   {"FringeBendEntrance": 0}, {"PolynomB": [0, .1, 0, 0]}])
def test_central_physical_asymmetry_or_combined_fields_never_erased(fields):
    bend = arguments()["bend"]
    bend["fields"].update(fields)
    before = copy.deepcopy(bend)
    with pytest.raises(ValueError, match="self-mirror|self-mirrored"):
        unit._check_central_bend(bend)
    assert bend == before


def test_each_resolution_refines_multiple_distinct_points_in_both_quadrants():
    calls, progress = [], []
    def callback(point):
        assert point[0] * point[1] < 0
        assert max(map(abs, point)) <= 4
        calls.append(point)
        return synthetic(point)
    samples, levels, attempts = unit._strength_search(callback, cap=4., resolutions=(9, 17),
        local_starts=8, local_budget=48, progress=progress.append, cancel=None)
    assert len(samples) == len(calls) == len(set(calls))
    assert {item["resolution"] for item in attempts} == {9, 17}
    for level in levels:
        assert all(branch["distinct_local_starts"] == 4 for branch in level["polarities"].values())
        assert level["polished_best_emittance_pm"] <= level["grid_best_emittance_pm"]
        for sign in (-1, 1):
            starts = [tuple(r["start_position"]) for r in attempts if r["resolution"] == level["resolution"] and r["polarity"] == sign]
            assert len(set(starts)) == 4
    assert len(progress) == 2
    assert progress[0]["phase"] == "unit_strength_scan"
    assert min(r["emittance_pm"] for r in samples if r["position"][0] > 0) < 10.001
    assert levels[1]["relative_improvement_from_previous_resolution"] is not None
    assert all(item["termination"] in {"strength_resolution_reached", "proposal_budget_exhausted"} for item in attempts)


def test_scan_numeric_level_is_durable_before_optional_progress(tmp_path):
    delivered = []
    def failing_presentation(update):
        saved = read_numerical_checkpoint(output_directory=tmp_path)
        assert saved["kind"] == "tme_unit_scan_level"
        assert len(saved["result"]["samples"]) == update["evaluations"]
        assert saved["result"]["resolution_levels"][-1]["resolution"] == update["level"]["resolution"]
        delivered.append(saved["descriptor"])
        raise ValueError("Optional progress formatting")
    unit._strength_search(synthetic, cap=4., resolutions=(3, 5), local_starts=2, local_budget=8,
                          progress=failing_presentation, cancel=None, checkpoint_directory=tmp_path)
    assert len(delivered) == 2
    assert delivered[0]["path"] != delivered[1]["path"]


@pytest.mark.parametrize("local_starts,unrefined", [(4, 0), (2, 1)])
def test_disconnected_measured_regions_get_separate_polish_seeds(local_starts, unrefined):
    def islands(position):
        x, y = map(abs, position)
        low = .5 <= x <= 1.5 and .5 <= y <= 1.5
        high = 3. <= x <= 4. and 3. <= y <= 4.
        return {"position": list(position), "passed": low or high, "checks": {"synthetic": low or high},
                "emittance_pm": (1 + (x-1)**2 + (y-1)**2) if low else (10 + (x-3.5)**2 + (y-3.5)**2)}
    _, levels, attempts = unit._strength_search(islands, cap=4., resolutions=(9, 17),
        local_starts=local_starts, local_budget=16, progress=None, cancel=None)
    for level in levels:
        for sign in (-1, 1):
            regions = level["polarities"][str(sign)]["feasible_grid_regions"]
            assert len(regions) == 2
            assert level["polarities"][str(sign)]["unrefined_grid_regions"] == unrefined
            assert sum(r["locally_refined"] for r in regions) == 2 - unrefined
            if not unrefined:
                starts = [r["start_position"] for r in attempts if r["resolution"] == level["resolution"] and r["polarity"] == sign]
                assert any(abs(p[0]) >= 3. for p in starts)  # Not only the lower-emittance island.
                assert any(abs(p[0]) <= 1.5 for p in starts)


def test_rejected_nonfinite_native_values_do_not_break_scan_checkpoint(tmp_path):
    def invalid(position):
        return {"position": list(position), "passed": False, "checks": {"physical_radiation": False},
                "emittance_pm": math.nan, "radiation": {"I5": math.nan}}
    unit._strength_search(invalid, cap=4., resolutions=(3, 5), local_starts=2, local_budget=8,
                          progress=None, cancel=None, checkpoint_directory=tmp_path)
    saved = read_numerical_checkpoint(output_directory=tmp_path)
    assert all(not r["passed"] and r["emittance_pm"] is None for r in saved["result"]["samples"])
    assert all(not r["checks"]["physical_radiation"] for r in saved["result"]["samples"])


def test_native_tracking_radiation_flag_not_allowed_for_conservative_unit():
    bend = arguments()["bend"]
    bend["fields"]["rad"] = 1
    with pytest.raises(ValueError, match="conservative periodic optics"):
        unit._check_central_bend(bend)
    assert bend["fields"]["rad"] == 1


def test_native_callback_mutates_two_families_and_gates_before_radiation(monkeypatch):
    editor = SimpleNamespace(jt=object(), lattice=object(), apply_parameterization=lambda knobs, point: updates.append(point))
    updates, calls = [], []
    monkeypatch.setattr(api, "build_jutrack_editor", lambda candidate: editor)
    metrics = {"stable": True, "trace_x": .1, "trace_y": -.1, "maximum_beta_m": 10.}
    monkeypatch.setattr(api, "jutrack_linear_metrics", lambda *a, **kw: copy.deepcopy(metrics))
    def radiation(jt, lattice, **kw):
        assert jt is editor.jt and lattice is editor.lattice
        calls.append(kw)
        return {"computed": True, "physical_validity": {"passed": True}, "emittance_pm": 12.}
    monkeypatch.setattr(api, "jutrack_periodic_radiation_metrics", radiation)
    rows = unit._unit_rows(arguments()["bend"], (.05, .35, .1, .3, .2), 2., -2.)
    evaluator = unit._NativeUnitEvaluator(rows, 3.5, 192, .01, .02, 100., 4.)
    assert evaluator((3.4261995199999995, -2.))["passed"]
    assert updates[-1][0] == 3.4261995199999995
    for key, value in (("trace_x", 1.995), ("trace_y", -1.985), ("maximum_beta_m", 101.)):
        before = metrics[key]
        metrics[key] = value
        assert not evaluator((2., -2.))["passed"]
        metrics[key] = before
    assert len(calls) == 1
    monkeypatch.setattr(api, "jutrack_periodic_radiation_metrics", lambda *a, **kw: {
        "computed": True, "physical_validity": {"passed": False}, "emittance_pm": -20000.})
    assert evaluator((2., -2.))["passed"] is False


def test_distinct_native_verified_units_keep_both_polarities_and_exact_evidence(native_doubles, tmp_path):
    result = unit.design_tme_unit(**arguments(output_directory=tmp_path))
    assert result["passed"] and not result["complete_mba_accepted"]
    candidates = result["candidates"]
    assert len(candidates) == 4
    assert {row["polarity"] for row in candidates} == {"near_positive", "near_negative"}
    assert len({tuple(row["position"]) for row in candidates}) == 4
    assert not result["resolution_convergence_demonstrated"]
    for candidate in candidates:
        assert all(candidate["checks"].values())
        assert set(candidate["boundary_twiss"]) == set(unit._BOUNDARY)
        assert candidate["boundary_twiss"]["Dx_m"] == .03214567891234567  # Unit is not an achromat.
        assert candidate["beta_sampling_scope"] == "native_map_checked_profile"
        assert candidate["reference_ring_receipt"]["path"]
        saved = read_numerical_checkpoint(candidate["numerical_checkpoint"], output_directory=tmp_path)
        assert saved["result"]["boundary_twiss"] == candidate["boundary_twiss"]
        assert saved["checks_passed"] is None  # Initial capture does not certify final unit checks.
        verified = read_numerical_checkpoint(candidate["verified_unit_checkpoint"], output_directory=tmp_path)
        assert verified["checks_passed"] is True and not verified["request_satisfied"]


def test_unsupported_true_edge_diagnostics_preserve_intact_native_candidate(native_doubles, monkeypatch):
    def unsupported(*a, **kw):
        raise UnsupportedOpticsSampling("True edge centre is unavailable")
    monkeypatch.setattr(api, "build_jutrack_profile_sampler", unsupported)
    inputs = arguments()
    inputs["bend"]["fields"].update(e1=.001, e2=.001)
    result = unit.design_tme_unit(**inputs)
    assert result["passed"]
    for candidate in result["candidates"]:
        assert candidate["elements"][5] == inputs["bend"]
        assert candidate["bend_centre_diagnostic"]["status"] == "unavailable"
        assert candidate["beta_sampling_scope"] == "native_element_boundaries"


def test_unvalidated_optional_profile_is_quarantined_not_used_to_reject_native_unit(native_doubles, monkeypatch):
    monkeypatch.setattr(api, "build_jutrack_profile_sampler", lambda *a, **kw: SimpleNamespace(
        linear_metrics=lambda **kw: {"stable": False, "sampling": {"validated": False, "map_equivalent": False},
            "failure": "Private map was not equivalent", "twiss_profile": [{"beta_x": 999., "beta_y": 999.}]}))
    result = unit.design_tme_unit(**arguments())
    assert result["passed"] and result["candidates"]
    for candidate in result["candidates"]:
        assert all(candidate["checks"].values())
        assert candidate["beta_sampling_scope"] == "native_element_boundaries"
        diagnostic = candidate["bend_centre_diagnostic"]
        assert diagnostic["status"] == "unavailable" and not diagnostic["diagnostic_validation_passed"]
        assert "centre" not in diagnostic and "twiss_profile" not in diagnostic


def test_fresh_dense_beta_and_reference_radiation_are_hard_final_checks(native_doubles, monkeypatch):
    monkeypatch.setattr(api, "build_jutrack_profile_sampler", lambda *a, **kw: SimpleNamespace(
        linear_metrics=lambda **kw: {"stable": True, "sampling": {"validated": True}, "twiss_profile": [
            {"requested_sample": True, "source_element": "B", "sample_fraction": .5, "beta_x": 101., "beta_y": 1.}]}))
    result = unit.design_tme_unit(**arguments())
    assert not result["passed"] and not result["candidates"]
    assert all(not r["checks"]["sampled_beta_limit"] for r in result["rejected_final_candidates"])
    def bad_ring(candidate, **kwargs):
        result = synthetic_evaluation(candidate, **kwargs)
        result["radiation"]["emittance_pm"] *= 2
        return result
    monkeypatch.setattr(api, "evaluate", bad_ring)
    result = unit.design_tme_unit(**arguments())
    assert all(not r["checks"]["reference_emittance_agrees"] for r in result["rejected_final_candidates"])


def test_native_receipt_and_exact_boundary_survive_optional_diagnostic_error(native_doubles, monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise RuntimeError("diagnostic API bug")
    monkeypatch.setattr(api, "build_jutrack_profile_sampler", broken)
    with pytest.raises(RuntimeError, match="diagnostic API bug"):
        unit.design_tme_unit(**arguments(output_directory=tmp_path))
    checkpoint = read_numerical_checkpoint(output_directory=tmp_path)
    assert checkpoint["kind"] == "candidate_evaluation"
    assert set(checkpoint["result"]["boundary_twiss"]) == set(unit._BOUNDARY)
    receipt = checkpoint["result"]["candidate_evaluation"]
    candidate_evidence._captured.clear()
    assert candidate_evidence.validate_saved_candidate_evaluation(checkpoint["result"]["candidate_lattice"],
        receipt, tmp_path)["passed"]
    assert len(list((tmp_path / "candidate_evaluations").glob("*.json"))) == 2


def test_empty_search_persists_negative_evidence_without_acceptance(native_doubles, monkeypatch, tmp_path):
    monkeypatch.setattr(unit, "_NativeUnitEvaluator", lambda *a: lambda p: {
        "position": list(p), "passed": False, "emittance_pm": None, "checks": {"stable": False}})
    result = unit.design_tme_unit(**arguments(output_directory=tmp_path))
    assert not result["passed"] and not result["candidates"]
    saved = read_numerical_checkpoint(output_directory=tmp_path)
    assert saved["checks_passed"] is False
    json.dumps(result, allow_nan=False)


def test_programming_errors_and_cancellation_are_not_called_infeasible():
    def broken(point):
        raise TypeError("wrong native API")
    with pytest.raises(TypeError, match="wrong native API"):
        unit._strength_search(broken, cap=4., resolutions=(3, 5), local_starts=2,
                              local_budget=8, progress=None, cancel=None)
    def cancelled():
        raise InterruptedError("scientist cancelled")
    with pytest.raises(InterruptedError):
        unit._strength_search(synthetic, cap=4., resolutions=(3, 5), local_starts=2,
                              local_budget=8, progress=None, cancel=cancelled)


def test_exact_geometry_champion_survives_when_no_grid_point_is_feasible():
    anchor = (1.123456789123456, -2.234567891234567)
    calls = []
    def narrow_region(point):
        calls.append(point)
        return {"position": list(point), "passed": point == anchor,
                "emittance_pm": 7.0 if point == anchor else None,
                "checks": {"inside_narrow_region": point == anchor}}
    samples, levels, attempts = unit._strength_search(
        narrow_region, cap=4., resolutions=(3, 5), local_starts=2, local_budget=8,
        progress=None, cancel=None, initial_strengths=[anchor])
    assert calls.count(anchor) == 1
    assert min(r["emittance_pm"] for r in samples if r["passed"]) == 7.0
    assert all(level["feasible_grid_points"] == 0 for level in levels)
    assert all(level["polished_best_emittance_pm"] == 7.0 for level in levels)
    assert len(attempts) == 2
    assert all(r["start_position"] == list(anchor) for r in attempts)


@pytest.mark.parametrize("pair", [(1., 2.), (99., -2.), (float("nan"), -2.), (1.,)])
def test_invalid_supplied_strengths_fail_before_native(monkeypatch, pair):
    monkeypatch.setattr(unit, "_NativeUnitEvaluator", lambda *a: pytest.fail("Native must not load"))
    with pytest.raises(ValueError, match="initial_strengths"):
        unit.design_tme_unit(**arguments(initial_strengths=[pair]))


def test_supplied_strengths_are_remeasured_not_assumed_feasible(native_doubles):
    result = unit.design_tme_unit(**arguments(initial_strengths=[(1.4, -2.)]))
    assert result["candidates"][0]["emittance_pm"] == 10.
    assert result["supplied_initial_strengths"] == [[1.4, -2.]]
    assert result["candidates"][0]["position"] == [1.4, -2.]
