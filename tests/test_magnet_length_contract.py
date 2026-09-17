"""Length allocations, local/integrated strength conventions and geometry bounds."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from brilliance import accelerator_api as api
from expert_modules.mba_design import tme_geometry as tme


def test_local_strengths_are_length_independent_but_integrated_limits_scale():
    options = dict(octupole_pole_tip_field_t=.5, octupole_radius_m=.02)
    short = api.magnet_strength_limits(3.5, length_m=.25, **options)
    long = api.magnet_strength_limits(3.5, length_m=.5, **options)
    for n in (1, 2, 3):
        assert short[f"max_abs_k{n}_m{n+1}"] == long[f"max_abs_k{n}_m{n+1}"]
        assert long[f"max_abs_k{n}l_m{n}"] == 2 * short[f"max_abs_k{n}l_m{n}"]
    assert long["max_abs_angle_rad"] == 2 * short["max_abs_angle_rad"]
    assert short["max_abs_k3_m4"] == pytest.approx(6*.5/(short["brho_tm"]*.02**3))
    assert "max_abs_k3_m4" not in api.magnet_strength_limits(3.5)
    with pytest.raises(ValueError, match="both octupole"):
        api.magnet_strength_limits(3.5, octupole_radius_m=.02)


def test_audit_checks_declared_lengths_and_octupole_limits():
    limits = api.magnet_strength_limits(3.5, octupole_pole_tip_field_t=.5, octupole_radius_m=.02)
    rows = [{"name": "O", "kind": "KOCT", "length_m": .3, "k3_m4": limits["max_abs_k3_m4"]}]
    def audit(envelope, lengths=None):
        return api.audit_lattice_constraints(rows, energy_gev=3.5, expected_length_m=.3,
            expected_angle_rad=0., limits=envelope, length_bounds_m=lengths)
    assert audit(limits, {"KOCT": [.25, .5]})["passed"]
    assert not audit(limits, {"KOCT": [.4, .5]})["passed"]
    assert not audit(api.magnet_strength_limits(3.5))["passed"]
    rows[0]["k3_m4"] *= 1.01
    assert "O: octupole exceeds field limit" in audit(limits)["issues"]


def test_bounded_geometry_keeps_exact_length_without_clipping_after_decode():
    bounds = {**tme._DEFAULT_LENGTH_BOUNDS, "d1_m": [.06, .3], "d2_m": [.06, .4], "d3_m": [.06, .5]}
    decoder = tme._Geometry(3.2, 3.5, 2*math.pi/192, api.magnet_strength_limits(3.5), .15, .06, bounds)
    for position in [np.zeros(5), np.ones(5), *np.random.default_rng(40).uniform(size=(1000, 5))]:
        lengths = decoder.decode(position)
        assert sum(lengths[k]*w for k, w in zip(tme._KEYS, tme._WEIGHTS)) == pytest.approx(3.2, abs=1e-12)
        for key, (low, high) in bounds.items():
            assert low-1e-12 <= lengths[key] <= high+1e-12
        assert decoder.decode(decoder.encode(lengths)) == pytest.approx(lengths, abs=1e-12)


def test_fixed_magnet_lengths_leave_drift_allocation_free():
    bounds = {"bend_length_m": [1., 1.], "quadrupole_near_length_m": [.3, .3], "quadrupole_far_length_m": [.4, .4]}
    decoder = tme._Geometry(3.2, 3.5, 2*math.pi/192, api.magnet_strength_limits(3.5), .15, .06, bounds)
    for position in ([0.]*5, [.5]*5, [1.]*5):
        lengths = decoder.decode(position)
        for key, pair in bounds.items():
            assert lengths[key] == pytest.approx(pair[0])


def test_mba_default_ranges_fail_incompatible_budget_before_native_work(monkeypatch):
    monkeypatch.setattr(tme, "_Evaluator", lambda *a: pytest.fail("Launched native work with impossible geometry"))
    with pytest.raises(ValueError, match="engineering minimum|Length bounds"):
        tme.optimize_tme_unit_geometry(energy_gev=3.5, bend=api.sector_bend("B", 1., 2*math.pi/192),
            expected_length_m=2., reference_repetitions=192, limits=api.magnet_strength_limits(3.5),
            maximum_beta_m=30., trace_margin_x=.01, trace_margin_y=.01,
            minimum_magnet_length_m=.15, minimum_drift_length_m=.06)


def test_octupole_matching_knobs_respect_explicit_envelope():
    rows = [{"name": "O", "kind": "KOCT", "length_m": .3, "k3_m4": 0.}]
    knobs = api.compile_lattice_parameterization(rows, [{"name": "oct", "field": "k3_m4",
        "element_names": ["O"], "bounds": [-1e8, 1e8]}])
    with pytest.raises(ValueError, match="max_abs_k3_m4"):
        api.plan_matching_starts(knobs, limits=api.magnet_strength_limits(3.5))
    limits = api.magnet_strength_limits(3.5, octupole_pole_tip_field_t=.5, octupole_radius_m=.02)
    plan = api.plan_matching_starts(knobs, limits=limits, max_starts=10)
    assert all(abs(row[0]) <= limits["max_abs_k3_m4"] for row in plan["positions"])


@pytest.mark.skipif(os.environ.get("BRILLIANCE_RUN_LENGTH_TESTS") != "1", reason="Opt-in real JuTrack bounded geometry search")
def test_native_unit_search_enforces_new_length_defaults(tmp_path):
    import json

    from brilliance.research_environment import ResearchEnvironment

    environment = ResearchEnvironment.load()
    environment.require_active_python(operation="Magnet length regression")
    assert not environment.internet_access
    os.environ.update(environment.process_environment())
    output = Path(os.environ.get("BRILLIANCE_LENGTH_TEST_OUTPUT", str(tmp_path)))
    result = tme.optimize_tme_unit_geometry(energy_gev=3.5,
        bend=api.sector_bend("B", 1., 2*math.pi/224), expected_length_m=20/7,
        reference_repetitions=224, limits=api.magnet_strength_limits(3.5), maximum_beta_m=25.,
        trace_margin_x=.02, trace_margin_y=.02, minimum_magnet_length_m=.12, minimum_drift_length_m=.05,
        population_size=70, generations=10, max_initialization_evaluations=10000, seed=41,
        scan_resolutions=(17, 33), local_starts=4, local_max_evaluations=80,
        geometry_candidates=1, candidates_to_return=2, output_directory=output)
    assert result["passed"], result.get("initialization")
    assert result["contract"]["length_bounds_source"] == "MBA proposed defaults"
    for candidate in result["candidates"]:
        rows = candidate["elements"]
        for row in rows:
            if row["kind"] == "SBEND":
                assert .8-1e-12 <= row["length_m"] <= 1.5+1e-12
            elif row["kind"] == "KQUAD":
                assert .25-1e-12 <= row["length_m"] <= .5+1e-12
        assert math.fsum(r["length_m"] for r in rows) == pytest.approx(20/7, abs=1e-12)
        assert all(candidate["checks"].values())
    print("NATIVE_LENGTH_VALIDATION " + json.dumps({"output": str(output),
        "candidates": [{"emittance_pm": c["emittance_pm"], "bend_m": c["elements"][5]["length_m"],
            "near_quad_m": c["elements"][3]["length_m"], "far_quad_m": c["elements"][1]["length_m"]}
            for c in result["candidates"]]}))
