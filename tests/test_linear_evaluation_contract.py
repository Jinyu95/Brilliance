"""Final evaluation rejects physical optics, not programming/runtime faults."""
from types import SimpleNamespace

import numpy as np
import pytest

from brilliance import accelerator_api as api
from brilliance.research_outcomes import ResearchInfeasible


def point(**overrides):
    values = dict(mode=1, betax=2.0, betay=3.0, alphax=0.0,
                  alphay=0.0, dx=0.0, dpx=0.0, dy=0.0, dpy=0.0)
    return SimpleNamespace(**(values | overrides))


@pytest.fixture
def optics(monkeypatch):
    elements = [SimpleNamespace(name="D1", length_m=1.0),
                SimpleNamespace(name="D2", length_m=2.0)]
    lattice = SimpleNamespace(elements=elements)
    matrix = np.eye(6)
    matrix[:4, :4] = [[0., 1., 0., 0.], [-1., 0., 0., 0.],
                      [0., 0., 0., 1.], [0., 0., -1., 0.]]
    native = SimpleNamespace(
        findm66=lambda *a, **k: matrix,
        periodicEdwardsTengTwiss=lambda *a, **k: point(),
        twissline=lambda *a, **k: [point(), point()] if "refpts" in k else point(),
        gettune=lambda *a, **k: [0.25, 0.25],
    )
    monkeypatch.setattr(api, "build_jutrack_lattice", lambda _: elements)
    return native, lattice


def test_final_valid_optics_keeps_residuals_and_saved_positions(optics):
    native, lattice = optics
    result = api._linear_evaluation(native, lattice)
    assert result["stable"] is True
    assert result["tune_x"] == result["tune_y"] == 0.25
    assert result["periodic_closure_max"] == 0.0
    assert all(value == 0.0 for value in result["periodic_residuals"].values())
    assert [(row["name"], row["s_m"]) for row in result["twiss_profile"]] == [
        ("START", 0.0), ("D1", 1.0), ("D2", 3.0)]


@pytest.mark.parametrize("operation", ["findm66", "periodicEdwardsTengTwiss", "twissline", "gettune"])
@pytest.mark.parametrize("error", [TypeError, KeyError, ValueError, RuntimeError, OSError, AttributeError])
def test_final_optics_propagates_api_and_runtime_errors(optics, operation, error):
    native, lattice = optics

    def broken(*args, **kwargs):
        raise error("native bridge failure")

    setattr(native, operation, broken)
    with pytest.raises(error, match="native bridge failure"):
        api._linear_evaluation(native, lattice)


def test_final_invalid_mode_rejects_before_propagation(optics):
    native, lattice = optics
    native.periodicEdwardsTengTwiss = lambda *a, **k: SimpleNamespace(mode=0)
    native.twissline = lambda *a, **k: pytest.fail("Do not propagate an invalid periodic solution")
    native.gettune = lambda *a, **k: pytest.fail("Do not calculate invalid tunes")
    result = api._linear_evaluation(native, lattice)
    assert result["stable"] is False
    assert "mode=0" in result["model_limits"][0]


@pytest.mark.parametrize("location", ["periodic", "endpoint", "profile"])
@pytest.mark.parametrize("changes", [{"betax": 0.0}, {"betay": -1.0},
                                     {"dx": float("nan")}, {"alphax": float("inf")}])
def test_final_nonphysical_optics_are_explicit_rejections(optics, location, changes):
    native, lattice = optics
    if location == "periodic":
        native.periodicEdwardsTengTwiss = lambda *a, **k: point(**changes)
    else:
        def propagate(*args, **kwargs):
            if "refpts" in kwargs:
                return [point(), point(**changes) if location == "profile" else point()]
            return point(**changes) if location == "endpoint" else point()
        native.twissline = propagate
    result = api._linear_evaluation(native, lattice)
    assert result["stable"] is False
    assert result["trace_stable"] is True
    assert "nonfinite or nonpositive" in result["model_limits"][0]


@pytest.mark.parametrize("tunes", [[0.25], [], [0.25, 0.25, 0.25]])
def test_final_tune_wrong_arity_is_a_schema_error(optics, tunes):
    native, lattice = optics
    native.gettune = lambda *a, **k: tunes
    with pytest.raises(TypeError, match="must return two tunes"):
        api._linear_evaluation(native, lattice)


@pytest.mark.parametrize("tunes", [[float("nan"), 0.25], [0.25, float("inf")]])
def test_final_nonfinite_tunes_are_physical_rejection(optics, tunes):
    native, lattice = optics
    native.gettune = lambda *a, **k: tunes
    assert api._linear_evaluation(native, lattice)["stable"] is False


@pytest.mark.parametrize("count", [0, 1, 3])
def test_final_profile_must_cover_every_saved_element(optics, count):
    native, lattice = optics
    native.twissline = lambda *a, **k: [point()] * count if "refpts" in k else point()
    with pytest.raises(ValueError, match="does not cover every element exit"):
        api._linear_evaluation(native, lattice)


def test_final_profile_schema_error_is_not_physics(optics):
    native, lattice = optics
    native.twissline = lambda *a, **k: [{}, {}] if "refpts" in k else point()
    with pytest.raises(AttributeError, match="betax"):
        api._linear_evaluation(native, lattice)


@pytest.mark.parametrize("message,physical", [
    ("DomainError with -1: negative sqrt argument in twissPropagate(mode=1)", True),
    ("MethodError: no method matching twissPropagate", False),
    ("DomainError: an unrelated invalid API argument", False),
])
def test_final_julia_error_classification_is_narrow(optics, message, physical):
    native, lattice = optics
    JuliaError = type("JuliaError", (Exception,), {"__module__": "juliacall"})

    def fail(*args, **kwargs):
        raise JuliaError(message)

    native.periodicEdwardsTengTwiss = fail
    if physical:
        assert api._linear_evaluation(native, lattice)["stable"] is False
    else:
        with pytest.raises(JuliaError, match=message):
            api._linear_evaluation(native, lattice)


@pytest.mark.parametrize("operation", ["findm66", "periodicEdwardsTengTwiss", "twissline", "gettune"])
def test_final_typed_infeasible_outcome_remains_physical(optics, operation):
    native, lattice = optics

    def reject(*args, **kwargs):
        raise ResearchInfeasible("no admissible solution", code="unstable", diagnostics={})

    setattr(native, operation, reject)
    assert api._linear_evaluation(native, lattice)["stable"] is False


def test_final_saved_and_native_element_counts_must_agree(optics, monkeypatch):
    native, lattice = optics
    monkeypatch.setattr(api, "build_jutrack_lattice", lambda _: [])
    with pytest.raises(ValueError, match="saved element count"):
        api._linear_evaluation(native, lattice)


def test_final_matrix_shape_is_not_a_physics_rejection(optics):
    native, lattice = optics
    native.findm66 = lambda *a, **k: np.eye(4)
    with pytest.raises(TypeError, match="6x6"):
        api._linear_evaluation(native, lattice)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_final_unstable_or_nonfinite_map_is_physical_rejection(optics, nonfinite):
    native, lattice = optics
    matrix = np.eye(6)
    if nonfinite:
        matrix[0, 0] = float("nan")
    native.findm66 = lambda *a, **k: matrix
    native.periodicEdwardsTengTwiss = lambda *a, **k: pytest.fail("Unstable maps stop before Twiss")
    assert api._linear_evaluation(native, lattice)["stable"] is False
