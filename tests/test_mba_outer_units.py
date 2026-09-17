"""Structural/ownership tests; native complete-ring validation is separate."""
import copy
import math

import pytest

from brilliance import accelerator_api as api
from expert_modules.mba_design import outer_units as mba


def source(m=7, fraction=1.0):
    angle = 2 * math.pi / (32 * (m - 2 + 2 * fraction))
    left = [api.drift("D3:L", .2), api.quadrupole("F:L", .3, 1.5), api.drift("D2:L", .2),
            api.quadrupole("N:L", .2, -1.5), api.drift("D1:L", .2)]
    return left + [api.sector_bend("B", .5, angle)] + api.mirror_elements(left)


def assemble(m=7, fraction=1.0, **changes):
    rows = source(m, fraction)
    args = dict(energy_gev=3.5, periods=32, bends_per_cell=m, cell_length_m=40., straight_length_m=6.,
                limits=api.magnet_strength_limits(3.5), minimum_magnet_length_m=.15,
                minimum_drift_length_m=.06, end_bend_fraction=fraction)
    return rows, mba.assemble_mba_outer_units(rows, **{**args, **changes})


@pytest.mark.parametrize("m", [5, 6, 7, 9])
@pytest.mark.parametrize("fraction", [1., .5])
def test_complete_cell_geometry_and_only_outer_two_pairs_are_adjustable(m, fraction):
    unit, assembled = assemble(m, fraction)
    rows = assembled["elements"]
    assert assembled["geometry"]["passed"]
    assert assembled["id_drift_names"] == [rows[0]["name"], rows[-1]["name"]]
    assert sum(row["length_m"] for row in rows if row["name"] in assembled["id_drift_names"]) == assembled["id_length_m"] == 6.
    assert math.fsum(row["length_m"] for row in rows) == pytest.approx(40., abs=1e-12)
    assert math.fsum(row.get("angle_rad") or 0 for row in rows) == pytest.approx(2 * math.pi / 32, abs=1e-14)
    assert sum(row["kind"] == "SBEND" for row in rows) == m
    assert len(assembled["unit_groups"]) == m and len(assembled["core_unit_groups"]) == m-4
    by_name = {row["name"]: row for row in rows}
    for group in assembled["core_unit_groups"]:
        for name, original in zip(group, unit):
            for key in ("kind", "length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4"):
                assert by_name[name][key] == original[key]
    before = copy.deepcopy(rows)
    params = mba.mba_outer_unit_knobs(rows, unit_groups=assembled["unit_groups"], limits=api.magnet_strength_limits(3.5))
    assert len(params.knobs) == 8  # not four internally tied TME families
    outer_names = set(sum(assembled["unit_groups"][:2] + assembled["unit_groups"][-2:], []))
    for knob in params.knobs:
        assert len(knob.target_indices) == 2
        assert sum(knob.target_indices) == len(rows) - 1
        assert set(knob.target_names) <= outer_names
    position = params.initial_position
    position[0] += .1
    changed = params.apply(position)
    assert [i for i, (a, b) in enumerate(zip(rows, changed)) if a["k1_m2"] != b["k1_m2"]] == sorted(params.knobs[0].target_indices)
    assert rows == before
    assert assembled["physics_accepted"] is False


def test_incompatible_budget_fails_without_shrinking_straight():
    with pytest.raises(ValueError):
        assemble(cell_length_m=8.)


def test_wrong_angle_is_not_silently_changed_in_core():
    unit = source()
    unit[5]["angle_rad"] *= 1.1
    with pytest.raises(ValueError, match="source angle"):
        mba.assemble_mba_outer_units(unit, energy_gev=3.5, periods=32, bends_per_cell=7,
            cell_length_m=40., straight_length_m=6., limits=api.magnet_strength_limits(3.5),
            minimum_magnet_length_m=.15, minimum_drift_length_m=.06)


def test_bad_trace_or_budget_preflight_does_not_load_julia(monkeypatch):
    _, assembly = assemble()
    monkeypatch.setattr(api, "build_jutrack_editor", lambda *a, **kw: pytest.fail("invalid input launched Julia"))
    for change in ({"max_starts": 0}, {"max_iterations": False}, {"trace_margin_x": -1.}, {"maximum_beta_m": 0.}):
        with pytest.raises(ValueError):
            mba.match_mba_outer_units(assembly, **{**dict(energy_gev=3.5, periods=32,
                limits=api.magnet_strength_limits(3.5), maximum_beta_m=30., minimum_magnet_length_m=.15,
                minimum_drift_length_m=.06), **change})


@pytest.mark.parametrize("change_core", [False, True])
def test_independent_transfer_verifier_distinguishes_outer_edits_from_core(change_core):
    from expert_modules.mba_design.conventional_claims import _transfer

    unit_rows, assembly = assemble(fraction=.5)
    rows = copy.deepcopy(assembly["elements"])
    # A globally mirrored perturbation is still forbidden in the initial core
    # transfer; mirror/engineering checks alone cannot establish provenance.
    group_index = 2 if change_core else 0
    name = assembly["unit_groups"][group_index][1]
    index = next(i for i, row in enumerate(rows) if row["name"] == name)
    rows[index]["k1_m2"] += .1
    rows[-1-index]["k1_m2"] += .1
    unit = api.periodic_cell(title="Test source", energy_gev=3.5, elements=unit_rows,
                            intended_repetitions=192, source="test")
    cell = api.periodic_cell(title="Test transfer", energy_gev=3.5, elements=rows,
                            intended_repetitions=32, source="test")
    machine = dict(energy_gev=3.5, periods=32, expected_cell_length_m=40.,
        expected_cell_angle_rad=2*math.pi/32, expected_bends=7,
        minimum_magnet_length_m=.15, minimum_drift_length_m=.06,
        limits=api.magnet_strength_limits(3.5), id_drift_names=["ID:L", "ID:R"],
        id_length_m=6., achromat_dx_tolerance_m=1e-4, achromat_dpx_tolerance=1e-5,
        maximum_beta_m=None, maximum_emittance_pm=None, maximum_energy_spread=None,
        limit_provenance={"source": "synthetic test"})
    issues, _, _ = _transfer(unit, cell, cell, assembly["unit_groups"], machine)
    if change_core:
        assert "assembly changes the supplied unit's physical fields outside the declared outer-unit edits" in issues
    else:
        assert not issues, issues
