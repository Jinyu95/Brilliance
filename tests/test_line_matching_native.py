"""Known-reachable boundary recovery using configured JuTrack, never an MBA seed."""
import copy
import os

import pytest

from brilliance import accelerator_api as api

pytestmark = pytest.mark.skipif(os.environ.get("BRILLIANCE_RUN_JUTRACK_TESTS") != "1",
                                reason="Opt-in configured native JuTrack matching")


def test_native_match_recovers_native_measured_boundary_with_bends_and_both_quad_signs(monkeypatch):
    from brilliance.research_environment import ResearchEnvironment
    env = ResearchEnvironment.load()
    env.require_active_python(operation="native JuTrack matching regression")
    for key, value in env.process_environment().items():
        monkeypatch.setenv(key, value)
    # Deliberately small transfer-line fixture. Not a periodic light-source design.
    known = [api.drift("D0", 0.2), api.quadrupole("QF", 0.2, 1.2), api.drift("D1", 0.3),
             api.sector_bend("B", 0.8, 0.02), api.drift("D2", 0.2),
             api.quadrupole("QD", 0.2, -1.0), api.drift("D3", 0.25),
             api.quadrupole("QF2", 0.2, 0.6), api.drift("D4", 0.3)]
    initial = dict(beta_x_m=3.0, beta_y_m=4.0, alpha_x=0.3, alpha_y=-0.2,
                   Dx_m=0.01, Dpx=-0.002, Dy_m=0.0, Dpy=0.0)
    configuration = dict(energy_gev=3.5, initial_twiss=initial,
                         maximum_beta_m=100.0, limits=api.magnet_strength_limits(3.5),
                         expected_length_m=sum(row["length_m"] for row in known),
                         expected_angle_rad=0.02, profile_step_m=0.5)
    measured = api.evaluate_jutrack_line(known, **configuration)
    assert measured["passed"]
    proposed = copy.deepcopy(known)
    for row in proposed:
        if row["name"] in {"QF", "QD", "QF2"}:
            row["k1_m2"] *= 0.8
    knobs = api.compile_lattice_parameterization(proposed, [
        {"name": name, "field": "k1_m2", "element_names": [name], "bounds": [-3, 3]}
        for name in ("QF", "QD", "QF2")])
    tolerances = {"beta_x_m": 1e-4, "beta_y_m": 1e-4, "alpha_x": 1e-5,
                  "alpha_y": 1e-5, "Dx_m": 1e-7, "Dpx": 1e-7}
    targets = {key: (measured["endpoint"][key], tolerance) for key, tolerance in tolerances.items()}
    matched = api.match_jutrack_line(proposed, knobs, targets=targets,
                                    max_starts=2, max_nfev=100, **configuration)
    assert matched["match_success"], matched
    assert all(row["passed"] for row in matched["residuals"])
    assert matched["measurement"]["passed"]
    assert not matched["complete_cell_accepted"]
    replay = api.evaluate_jutrack_line(matched["elements"], **configuration)
    for key, (target, tolerance) in targets.items():
        assert replay["endpoint"][key] == pytest.approx(target, abs=tolerance)
    assert matched["measurement"]["endpoint"] == pytest.approx(replay["endpoint"], rel=1e-10, abs=1e-12)
    for row, original in zip(matched["elements"], known):
        assert row["length_m"] == original["length_m"]
        assert row.get("angle_rad") == original.get("angle_rad")
    print("JUTRACK_LINE_MATCH", {"matched": matched["match_success"],
          "starts": len(matched["attempts"]), "callback_evaluations": matched["callback_evaluations"],
          "maximum_scaled_residual": max(abs(row["scaled_residual"]) for row in matched["residuals"]),
          "maximum_beta_m": replay["metrics"]["maximum_beta_m"], "complete_cell_accepted": False})
