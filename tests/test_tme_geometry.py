"""Geometry-search plumbing with synthetic physics; native validation is separate."""

import copy
import math

import numpy as np
import pytest

from brilliance import accelerator_api as api
from brilliance import candidate_evidence
from brilliance.numerical_checkpoints import read_numerical_checkpoint, write_numerical_checkpoint
from expert_modules.mba_design import tme_geometry as geometry


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)
    monkeypatch.setattr(geometry, "_implementation_identity", lambda: {"synthetic_implementation": 1})
    candidate_evidence._captured.clear()


def arguments(**changes):
    result = dict(energy_gev=.3, bend=api.sector_bend("B", .8, 2 * math.pi / 16), expected_length_m=2.72,
        reference_repetitions=16, limits=api.magnet_strength_limits(.3), maximum_beta_m=100.,
        trace_margin_x=.01, trace_margin_y=.01, minimum_magnet_length_m=.1, minimum_drift_length_m=.02,
        length_bounds_m={}, population_size=4, generations=1, geometry_candidates=2, scan_resolutions=(3, 5),
        local_starts=2, local_max_evaluations=8, candidates_to_return=2, max_initialization_evaluations=80)
    result.update(changes)
    return result


class FakeEvaluator:
    calls = 0
    instances = []

    def __init__(self, decoder, bend, energy, periods, limits, beta, mx, my):
        self.decoder, self.cache = decoder, {}
        self.__class__.instances.append(self)

    def __call__(self, position):
        key = tuple(map(float, position))
        if key not in self.cache:
            self.__class__.calls += 1
            lengths = self.decoder.decode(key[:5])
            feasible = key[-2] * key[-1] < 0
            self.cache[key] = {"position": list(key), "geometry": lengths, "feasible": feasible,
                "checks": {"opposite_polarities": feasible},
                "constraint_violations": {} if feasible else {"opposite_polarities": 1.},
                "objectives": {"emittance_pm": 20. + (lengths["bend_length_m"] - .8)**2 + .001 * sum(k*k for k in key[-2:])}
                    if feasible else {}}
        return copy.deepcopy(self.cache[key])


def fake_study(**inputs):
    point = inputs["initial_strengths"][0]
    rows = geometry._rows(inputs["bend"], {key: inputs["bend"]["length_m"] if key == "bend_length_m" else inputs[key]
                                           for key in geometry._KEYS}, point)
    cell = geometry.tme_unit._cell(rows, inputs["energy_gev"], inputs["reference_repetitions"])
    ring = api.cell_ring(title="Synthetic reference", energy_gev=inputs["energy_gev"], cell=rows,
                        periods=inputs["reference_repetitions"], source="geometry test double")
    def measured(candidate):
        result = {"engine": "JuTrack", "candidate": candidate.as_dict(), "linear_optics": {"stable": True},
                  "radiation": {"computed": True, "emittance_pm": 12., "physical_validity": {"passed": True}}}
        result["geometry"] = ({"cell_length_m": candidate.total_length_m,
            "predicted_ring_length_m": candidate.total_length_m * inputs["reference_repetitions"],
            "predicted_ring_bend_sum_rad": candidate.bend_sum_rad * inputs["reference_repetitions"]}
            if candidate.metadata["evaluation_scope"] == "periodic_cell" else
            {"circumference_m": candidate.total_length_m, "bend_sum_rad": candidate.bend_sum_rad})
        result["evaluation_receipt"] = candidate_evidence.record_evaluation(candidate, result,
            effective_periods=inputs["reference_repetitions"])
        return result
    evaluation, ring_result = measured(cell), measured(ring)
    descriptor, ring_receipt = None, None
    if inputs["output_directory"]:
        output = inputs["output_directory"]
        receipt = candidate_evidence.verify_candidate_evaluation(cell, evaluation, output)["receipt"]
        ring_receipt = candidate_evidence.verify_candidate_evaluation(ring, ring_result, output)["receipt"]
        descriptor = write_numerical_checkpoint("candidate_evaluation", {
            "candidate_lattice": cell.as_dict(), "candidate_evaluation": receipt}, output_directory=output)
    return {"passed": True, "candidates": [{"elements": rows, "periodic_cell": cell.as_dict(),
        "evaluation": evaluation, "reference_ring_evaluation": ring_result, "reference_ring_receipt": ring_receipt,
        "emittance_pm": 12., "position": list(point), "numerical_checkpoint": descriptor,
        "boundary_twiss": {key: .12345678912345678 for key in geometry.tme_unit._BOUNDARY},
        "complete_mba_accepted": False}]}


@pytest.fixture
def doubles(monkeypatch):
    FakeEvaluator.calls, FakeEvaluator.instances = 0, []
    monkeypatch.setattr(geometry, "_Evaluator", FakeEvaluator)
    calls = []
    def study(**inputs):
        calls.append(copy.deepcopy(inputs))
        return fake_study(**inputs)
    monkeypatch.setattr(geometry.tme_unit, "design_tme_unit", study)
    return calls


def test_five_coordinates_release_all_six_lengths_with_exact_sum_and_engineering():
    args = arguments()
    decoder = geometry._Geometry(args["expected_length_m"], args["energy_gev"], args["bend"]["angle_rad"],
        args["limits"], args["minimum_magnet_length_m"], args["minimum_drift_length_m"])
    rng = np.random.default_rng(91)
    rows = [decoder.decode(decoder.random(rng)) for _ in range(50)]
    for lengths in rows:
        vector = np.asarray([lengths[key] for key in geometry._KEYS])
        assert vector @ geometry._WEIGHTS == pytest.approx(2.72, abs=1e-14)
        assert np.all(vector >= decoder.minimum)
        assert decoder.decode(decoder.encode(lengths)) == pytest.approx(lengths)
        field = .3 / .299792458 * args["bend"]["angle_rad"] / lengths["bend_length_m"]
        assert field <= args["limits"]["dipole_field_t"] * (1 + 1e-12)
    assert all(len({row[key] for row in rows}) == 50 for key in geometry._KEYS)
    # The interior map has rank five; the sixth length is not frozen or independent.
    x = np.asarray([.2] * 5)
    baseline = np.asarray(list(decoder.decode(x).values()))
    matrix = np.column_stack([(np.asarray(list(decoder.decode(x + np.eye(5)[i] * 1e-6).values())) - baseline) / 1e-6 for i in range(5)])
    assert np.linalg.matrix_rank(matrix) == 5


def test_geometry_keeps_native_bend_fields_angle_and_global_mirror():
    bend = arguments()["bend"]
    bend["fields"].update(e1=.02, e2=.02, NumIntSteps=17)
    original = copy.deepcopy(bend)
    lengths = dict(zip(geometry._KEYS, [.8, .2, .25, .05, .2, .26]))
    rows = geometry._rows(bend, lengths, [.1] * 5 + [-3.1234567891234567, 2.134567891234567])
    assert bend == original and rows[5]["fields"] == original["fields"]
    assert rows[5]["angle_rad"] == bend["angle_rad"]
    assert len(rows) == 11
    assert rows[3]["k1_m2"] == rows[7]["k1_m2"] == -3.1234567891234567
    assert rows[1]["k1_m2"] == rows[9]["k1_m2"] == 2.134567891234567


@pytest.mark.parametrize("changes,error", [({"expected_length_m": .1}, "below the engineering minimum"),
    ({"local_starts": 1}, "local_starts"), ({"scan_resolutions": (5,)}, "scan_resolutions"),
    ({"profile_step_m": 0.}, "profile_step_m"), ({"maximum_beta_m": -1}, "maximum_beta_m"),
    ({"trace_margin_x": 3.}, "Trace margins"), ({"minimum_drift_length_m": 0}, "physical minimum"),
    ({"reference_repetitions": 15}, "bend closure")])
def test_invalid_arguments_fail_before_jutrack_or_optimization(monkeypatch, changes, error):
    monkeypatch.setattr(geometry, "_Evaluator", lambda *a: pytest.fail("Native evaluator must not load"))
    with pytest.raises(ValueError, match=error):
        geometry.optimize_tme_unit_geometry(**arguments(**changes))


def test_generic_population_is_all_feasible_and_final_scans_are_not_inner_objectives(doubles, monkeypatch, tmp_path):
    original = api.particle_swarm_optimize
    def checked(evaluate, **kwargs):
        assert len(doubles) == 0
        assert kwargs["cache_evaluations"] is False
        assert len(kwargs["bounds"]) == 7
        points = [kwargs["initial_position"], *kwargs["seed_positions"]]
        assert len(points) == len({tuple(p) for p in points}) == 4
        assert all(evaluate(p)["feasible"] for p in points)
        return original(evaluate, **kwargs)
    monkeypatch.setattr(api, "particle_swarm_optimize", checked)
    result = geometry.optimize_tme_unit_geometry(**arguments(output_directory=tmp_path))
    assert result["passed"] and result["optimizer_started"]
    assert result["initialization"]["all_feasible"]
    assert len(doubles) == len(result["geometry_finalists"]) == 2
    for inputs, finalist in zip(doubles, result["geometry_finalists"]):
        assert inputs["initial_strengths"] == [finalist["outer_position"][-2:]]
        assert inputs["expected_length_m"] == 2.72
    assert not result["complete_mba_accepted"]
    assert not list(tmp_path.rglob("*.evaluations"))
    assert len(result["optimizer"]["convergence"]["history"]) == 2
    assert read_numerical_checkpoint(result["numerical_checkpoint"], output_directory=tmp_path)["checks_passed"]


def test_feasible_initialization_failure_never_starts_optimizer(doubles, monkeypatch, tmp_path):
    def impossible(self, point):
        return {"position": list(point), "feasible": False}
    monkeypatch.setattr(FakeEvaluator, "__call__", impossible)
    monkeypatch.setattr(api, "particle_swarm_optimize", lambda *a, **kw: pytest.fail("No incomplete initial population"))
    result = geometry.optimize_tme_unit_geometry(**arguments(output_directory=tmp_path, max_initialization_evaluations=10))
    assert not result["passed"] and not result["optimizer_started"] and not doubles
    assert result["initialization"]["attempted"] == 10
    assert result["numerical_checkpoint"]["path"]


def test_managed_population_floor_is_screened_before_initialization_can_finish(doubles, monkeypatch, tmp_path):
    # No native work: scientist runtime requires 8, model asks 4, only 4 screens allowed.
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(tmp_path))
    from types import SimpleNamespace

    from brilliance import research_api
    monkeypatch.setattr(research_api, "open_research_run", lambda: SimpleNamespace(
        raise_if_cancelled=lambda: None, progress=lambda *args, **kwargs: None))
    original = geometry.resolve_population_budget
    def floor(*args, **kwargs):
        result = original(*args, **kwargs)
        result["effective"] = {"population_size": 8, "generations": 1}
        return result
    monkeypatch.setattr(geometry, "resolve_population_budget", floor)
    result = geometry.optimize_tme_unit_geometry(**arguments(output_directory=tmp_path,
        max_initialization_evaluations=4, cancellation_check=lambda: None, progress_callback=lambda p: None))
    assert not result["optimizer_started"]
    assert result["initialization"]["required_particles"] == 8


def test_completed_resume_verifies_receipts_without_optimizer_or_rescans(doubles, monkeypatch, tmp_path):
    inputs = arguments(output_directory=tmp_path)
    first = geometry.optimize_tme_unit_geometry(**inputs)
    before = (tmp_path / "tme_geometry_swarm.json").read_bytes()
    candidate_evidence._captured.clear()
    monkeypatch.setattr(geometry, "_Evaluator", lambda *a: pytest.fail("Completed resume must not load JuTrack"))
    monkeypatch.setattr(geometry.tme_unit, "design_tme_unit", lambda **kw: pytest.fail("Do not repeat scans"))
    resumed = geometry.optimize_tme_unit_geometry(**{**inputs, "resume": True, "generations": None})
    assert resumed["resumed_completed_result"]
    assert resumed["optimizer"]["convergence"]["history"] == first["optimizer"]["convergence"]["history"]
    assert (tmp_path / "tme_geometry_swarm.json").read_bytes() == before


def test_resume_contract_change_is_rejected_without_touching_history(doubles, tmp_path):
    inputs = arguments(output_directory=tmp_path)
    geometry.optimize_tme_unit_geometry(**inputs)
    paths = [tmp_path / "tme_geometry_swarm.json", tmp_path / "tme_geometry_swarm.geometry.json"]
    originals = {p: p.read_bytes() for p in paths}
    with pytest.raises(ValueError, match="physics contract changed"):
        geometry.optimize_tme_unit_geometry(**{**inputs, "maximum_beta_m": 101., "resume": True})
    assert all(p.read_bytes() == content for p, content in originals.items())
    with pytest.raises(ValueError, match="already exists"):
        geometry.optimize_tme_unit_geometry(**{**inputs, "resume": False})
    with pytest.raises(ValueError, match="physics contract changed"):
        geometry.optimize_tme_unit_geometry(**{**inputs, "resume": True, "seed": 1})
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_completed_resume_reports_preserved_population_without_resizing(doubles, tmp_path):
    inputs = arguments(output_directory=tmp_path)
    geometry.optimize_tme_unit_geometry(**inputs)
    calls = FakeEvaluator.calls
    restored = geometry.optimize_tme_unit_geometry(**{**inputs, "resume": True, "population_size": 8})
    assert FakeEvaluator.calls == calls
    assert restored["budget"]["requested"]["population_size"] == 8
    assert restored["budget"]["effective"]["population_size"] == 4
    assert restored["budget"]["notes"]
    for candidate in restored["candidates"]:
        assert candidate_evidence.verify_candidate_evaluation(candidate["periodic_cell"], candidate["evaluation"], tmp_path)["passed"]


def test_partial_finalist_verification_resumes_without_repeating_completed_work(doubles, monkeypatch, tmp_path):
    inputs = arguments(output_directory=tmp_path)
    dispatched = []
    def interrupted_study(**kwargs):
        dispatched.append(kwargs["bend"]["length_m"])
        if len(dispatched) == 2:
            raise RuntimeError("Interrupted during second geometry verification")
        return fake_study(**kwargs)
    monkeypatch.setattr(geometry.tme_unit, "design_tme_unit", interrupted_study)
    with pytest.raises(RuntimeError, match="second geometry"):
        geometry.optimize_tme_unit_geometry(**inputs)
    calls = FakeEvaluator.calls
    candidate_evidence._captured.clear()
    restored = geometry.optimize_tme_unit_geometry(**{**inputs, "resume": True})
    assert FakeEvaluator.calls == calls  # No repeated PSO generation.
    assert len(dispatched) == 3 and dispatched[1] == dispatched[2]
    assert dispatched[0] != dispatched[2]  # First successful finalist was not rescanned.
    assert len(restored["geometry_finalists"]) == 2
    for candidate in restored["candidates"]:
        assert candidate_evidence.verify_candidate_evaluation(candidate["periodic_cell"], candidate["evaluation"], tmp_path)["passed"]


def test_explicit_generation_extension_preserves_history(doubles, tmp_path):
    inputs = arguments(output_directory=tmp_path)
    first = geometry.optimize_tme_unit_geometry(**inputs)
    second = geometry.optimize_tme_unit_geometry(**{**inputs, "resume": True, "generations": 2})
    assert second["optimizer"]["convergence"]["history"][:2] == first["optimizer"]["convergence"]["history"]
    assert len(second["optimizer"]["convergence"]["history"]) == 3
    assert len(FakeEvaluator.instances) == 2


def test_resume_detects_modified_archived_native_evidence(doubles, tmp_path):
    inputs = arguments(output_directory=tmp_path)
    first = geometry.optimize_tme_unit_geometry(**inputs)
    receipt = first["candidates"][0]["reference_ring_receipt"]
    (tmp_path / receipt["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot restore candidate evaluation"):
        geometry.optimize_tme_unit_geometry(**{**inputs, "resume": True})


def test_native_callback_reuses_editor_and_does_not_swallow_programming_errors(monkeypatch):
    args = arguments()
    decoder = geometry._Geometry(2.72, .3, args["bend"]["angle_rad"], args["limits"], .1, .02)
    updated, radiation_calls = [], []
    class Editor:
        jt, lattice = object(), object()
        def sync_rows(self, rows):
            updated.append(rows)
    editor = Editor()
    monkeypatch.setattr(api, "build_jutrack_editor", lambda cell: editor)
    monkeypatch.setattr(api, "jutrack_linear_metrics", lambda jt, lattice, **kw: {
        "stable": True, "trace_x": 0., "trace_y": 0., "maximum_beta_m": 10.})
    def radiation(jt, lattice, **kw):
        assert jt is editor.jt and lattice is editor.lattice
        radiation_calls.append(kw)
        return {"computed": True, "physical_validity": {"passed": True}, "emittance_pm": 12.}
    monkeypatch.setattr(api, "jutrack_periodic_radiation_metrics", radiation)
    evaluator = geometry._Evaluator(decoder, args["bend"], .3, 16, args["limits"], 100., .01, .01)
    assert evaluator([.2] * 5 + [2., -2.])["feasible"]
    assert evaluator([.3] * 5 + [-2., 2.])["feasible"]
    assert len(updated) == len(radiation_calls) == 2
    assert updated[0][5]["length_m"] != updated[1][5]["length_m"]
    assert not evaluator([.2] * 5 + [2., 2.])["feasible"]
    assert len(updated) == 2
    def broken(*a, **kw):
        raise TypeError("Native API mismatch")
    monkeypatch.setattr(api, "jutrack_linear_metrics", broken)
    with pytest.raises(TypeError, match="Native API mismatch"):
        evaluator([.1] * 5 + [2., -2.])


def test_cancellation_does_not_become_scientific_rejection(doubles):
    def cancelled():
        raise InterruptedError("Scientist cancelled")
    with pytest.raises(InterruptedError):
        geometry.optimize_tme_unit_geometry(**arguments(cancellation_check=cancelled))


@pytest.mark.parametrize("resume", [True, "auto"])
def test_partial_resume_pair_is_actionable_and_does_not_load_native(monkeypatch, tmp_path, resume):
    (tmp_path / "tme_geometry_swarm.geometry.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(geometry, "_Evaluator", lambda *a: pytest.fail("No native work for partial pair"))
    with pytest.raises(ValueError, match="partial pair requires inspection"):
        geometry.optimize_tme_unit_geometry(**arguments(output_directory=tmp_path, resume=resume))


def test_default_auto_restores_completed_work_and_explicit_true_missing_is_actionable(doubles, tmp_path):
    with pytest.raises(ValueError, match="For a fresh campaign omit resume"):
        geometry.optimize_tme_unit_geometry(**arguments(output_directory=tmp_path, resume=True))
    geometry.optimize_tme_unit_geometry(**arguments(output_directory=tmp_path))
    calls = FakeEvaluator.calls
    assert geometry.optimize_tme_unit_geometry(**arguments(output_directory=tmp_path))["resumed_completed_result"]
    assert FakeEvaluator.calls == calls


def test_boolean_cancellation_and_resolved_trace_contract_precede_native(monkeypatch):
    monkeypatch.setattr(geometry, "_Evaluator", lambda *a: pytest.fail("Cancelled before native"))
    events = []
    with pytest.raises(InterruptedError):
        geometry.optimize_tme_unit_geometry(**arguments(progress_callback=events.append,
            cancellation_check=lambda: True, trace_margin_x=.99))
    assert events[0]["trace_limits"]["maximum_abs_trace_x"] == pytest.approx(1.01)
    assert events[0]["trace_limits"]["maximum_abs_trace_y"] == pytest.approx(1.99)


def test_completed_same_project_cross_worker_restores_exact_receipts(doubles, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from brilliance import research_api
    workspace = tmp_path / "workspace"
    source, target = [workspace / "artifacts" / "programs" / name for name in ("old", "new")]
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    (source / "program.py").write_text("# preserved generated program", encoding="utf-8")
    first = geometry.optimize_tme_unit_geometry(**arguments(output_directory=source))
    monkeypatch.setenv("BRILLIANCE_WORKSPACE", str(workspace))
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(target))
    monkeypatch.setattr(research_api, "open_research_run", lambda: SimpleNamespace(
        raise_if_cancelled=lambda: None, progress=lambda *a, **kw: None))
    monkeypatch.setattr(geometry, "_Evaluator", lambda *a: pytest.fail("No repeated native work"))
    candidate_evidence._captured.clear()
    restored = geometry.optimize_tme_unit_geometry(**arguments(output_directory=target,
        checkpoint_path=source / "tme_geometry_swarm.json"))
    assert restored["resumed_completed_result"]
    assert restored["candidates"][0]["elements"] == first["candidates"][0]["elements"]
    for candidate in restored["candidates"]:
        assert candidate_evidence.verify_candidate_evaluation(candidate["periodic_cell"], candidate["evaluation"], target)["passed"]
        assert candidate["numerical_checkpoint"]["source_output_directory"] == str(target)


def test_managed_progress_is_not_suppressed_by_optional_hook(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from brilliance import research_api
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(tmp_path))
    host_events, cancellations = [], []
    monkeypatch.setattr(research_api, "open_research_run", lambda: SimpleNamespace(
        raise_if_cancelled=lambda: cancellations.append(True),
        progress=lambda *args, **kw: host_events.append(kw)))
    def broken_hook(update):
        raise ValueError("custom presentation failure")
    publish, cancel = geometry.tme_unit._managed_callbacks(broken_hook, lambda: False, title="Unit", event_key="unit")
    publish({"phase": "generation", "generation": 3, "best": {"emittance_pm": 12.}})
    cancel()
    assert host_events[0]["metrics"]["best.emittance_pm"] == 12.
    assert cancellations == [True]


def test_partial_finalist_resume_preserves_original_source_root(doubles, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from brilliance import research_api
    workspace = tmp_path / "workspace"
    source, target = [workspace / "artifacts" / "programs" / name for name in ("old", "new")]
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    (source / "program.py").write_text("# preserved original program", encoding="utf-8")
    calls = []
    def interrupted(**kwargs):
        calls.append(kwargs["bend"]["length_m"])
        if len(calls) == 2:
            raise InterruptedError("second finalist interrupted")
        return fake_study(**kwargs)
    monkeypatch.setattr(geometry.tme_unit, "design_tme_unit", interrupted)
    with pytest.raises(InterruptedError):
        geometry.optimize_tme_unit_geometry(**arguments(output_directory=source))
    monkeypatch.setenv("BRILLIANCE_WORKSPACE", str(workspace))
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(target))
    monkeypatch.setattr(research_api, "open_research_run", lambda: SimpleNamespace(
        raise_if_cancelled=lambda: None, progress=lambda *a, **kw: None))
    candidate_evidence._captured.clear()
    measured_calls = FakeEvaluator.calls
    restored = geometry.optimize_tme_unit_geometry(**arguments(output_directory=target,
        checkpoint_path=source / "tme_geometry_swarm.json"))
    assert restored["passed"] and len(calls) == 3
    assert FakeEvaluator.calls == measured_calls
    assert calls[0] != calls[1] == calls[2]
    for candidate in restored["candidates"]:
        assert candidate_evidence.verify_candidate_evaluation(candidate["periodic_cell"], candidate["evaluation"], target)["passed"]


def test_managed_checkpoint_cannot_leave_current_project(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "artifacts" / "programs" / "current"
    monkeypatch.setenv("BRILLIANCE_WORKSPACE", str(workspace))
    monkeypatch.setenv("BRILLIANCE_RUN_OUTPUT", str(output))
    monkeypatch.setattr(geometry, "_Evaluator", lambda *a: pytest.fail("No native work"))
    with pytest.raises(ValueError, match="must remain inside this project's workspace"):
        geometry.optimize_tme_unit_geometry(**arguments(checkpoint_path=tmp_path / "other_project" / "swarm.json"))
