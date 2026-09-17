"""Synthetic evidence-contract tests; no numerical design or JuTrack simulation."""
from __future__ import annotations

import copy
import math

import pytest

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import numerical_evidence_directory, record_numerical_evidence
from brilliance.lattice_fingerprint import cell_fingerprint
from expert_modules.mba_design import conventional_claims as claims


@pytest.fixture
def example():
    theta = 2 * math.pi / 192
    rows = [api.drift("D_L", .1), api.quadrupole("QF_L", .2, 1.), api.drift("G_L", .1),
            api.quadrupole("QD_L", .2, -1.), api.sector_bend("B", .8, theta),
            api.quadrupole("QD_R", .2, -1.), api.drift("G_R", .1),
            api.quadrupole("QF_R", .2, 1.), api.drift("D_R", .1)]
    unit = api.periodic_cell(title="Synthetic measured unit", energy_gev=3.5, elements=rows,
                             intended_repetitions=192, source="unit test")
    copies = api.repeat_cell(rows, 5, period_prefix="U")
    assembly_rows = [api.drift("ID_L", 2.), api.sector_bend("TERM_L", .4, theta/2),
                     *copies, api.sector_bend("TERM_R", .4, theta/2), api.drift("ID_R", 2.)]
    assembly = api.periodic_cell(title="Synthetic assembly", energy_gev=3.5, elements=assembly_rows,
                                 intended_repetitions=32, source="unit test")
    final_rows = copy.deepcopy(assembly_rows)
    # A real final matcher may change the transferred seed's strengths. Check the
    # transfer against assembly, not these final values.
    final_rows[3]["k1_m2"] = final_rows[-4]["k1_m2"] = 1.1
    candidate = api.periodic_cell(title="Synthetic rematched candidate", energy_gev=3.5, elements=final_rows,
                                  intended_repetitions=32, source="unit test")
    machine = {"energy_gev": 3.5, "periods": 32, "expected_cell_length_m": 14.8,
               "expected_cell_angle_rad": 2*math.pi/32, "expected_bends": 7,
               "minimum_magnet_length_m": .01, "minimum_drift_length_m": .01,
               "limits": api.magnet_strength_limits(3.5), "id_drift_names": ["ID_L", "ID_R"],
               "id_length_m": 4., "achromat_dx_tolerance_m": 1e-4, "achromat_dpx_tolerance": 1e-4,
               "maximum_beta_m": None, "maximum_emittance_pm": None, "maximum_energy_spread": None,
               "limit_provenance": {"original": "synthetic test contract, not machine defaults"}}
    groups = [[row["name"] for row in copies[i:i+len(rows)]] for i in range(0, len(copies), len(rows))]
    radiation = {"I2": .2, "I3": .03, "I4": 0., "I5": 1e-7, "Jx": 1., "Jy": 1., "Jz": 2.,
                 "emittance_pm": 80., "energy_spread": .001}

    def point(s):
        return {"s_m": s, "Dx": 0., "Dpx": 0., "beta_x": 2., "beta_y": 3., "alpha_x": 0., "alpha_y": 0.}

    def profile(subject, positions):
        return {"sampling": {"engine": "JuTrack", "method": "physical_element_prefix_map_checked",
                             "cell_fingerprint": cell_fingerprint(api.one_period_elements(subject)), "map_equivalent": True},
                "twiss_profile": [point(s) for s in positions]}

    def evaluation(periods):
        return {"periods": periods, "linear_optics": {"trace_x": 0., "trace_y": 0., "stable": True,
                    "twiss_profile": [point(0.)]}, "radiation": copy.deepcopy(radiation)}

    ring = api.cell_ring(title="Synthetic exact measurement ring", energy_gev=3.5, cell=final_rows,
                        periods=32, source="unit test")
    raw = {"schema": claims.SCHEMA, "unit": unit.as_dict(), "assembled_candidate": assembly.as_dict(),
           "unit_groups": groups, "machine_contract": machine, "unit_evaluation": evaluation(192),
           "assembly_evaluation": evaluation(32), "candidate_evaluation": evaluation(32),
           "unit_profile": profile(unit, [0., 1., 2.]), "candidate_profile": profile(candidate, [0., 7.4, 14.8]),
           "full_ring_evaluation": {**evaluation(32), "candidate": ring.as_dict()},
           "per_bend_radiation": {"rows": [{key: radiation[key]/7 for key in ("I2", "I4", "I5")} for _ in range(7)]}}
    return candidate, unit, assembly, raw


def checked(candidate, raw, directory):
    with numerical_evidence_directory(directory):
        receipt = record_numerical_evidence(candidate, raw)
        return claims.validate_conventional_linear({"passed": True, "measurement_receipt": receipt}, candidate)


def test_project_authored_transfer_and_final_scalar_rematching_pass(example, tmp_path):
    candidate, _, _, raw = example
    issues, metrics = claims._checks(candidate, raw)
    assert not issues, issues
    assert metrics["transferred_units"] == 5
    assert metrics["unit_reference_repetitions"] == 192
    assert metrics["unit_to_tme_ratio"] > 1
    assert metrics["measured_bend_centre"]["s_m"] == 1.
    assert checked(candidate, raw, tmp_path) == []


def test_assembly_may_need_matching_and_unit_ratio_is_not_an_invented_gate(example, tmp_path):
    candidate, _, _, raw = example
    raw["assembly_evaluation"] = {"linear_optics": {"stable": False}}
    raw["unit_evaluation"]["radiation"]["emittance_pm"] = 100000.
    assert checked(candidate, raw, tmp_path) == []


@pytest.mark.parametrize("near_sign", [-1.0, 1.0])
def test_complete_eleven_element_unit_accepts_both_opposite_polarity_assignments(example, near_sign):
    """Physical near/far roles do not prescribe which family focuses horizontally."""
    _, _, _, raw = example
    theta = 2 * math.pi / 192
    left = [api.drift("END.L", .1), api.quadrupole("FAR.L", .2, -near_sign),
            api.drift("G1.L", .1), api.quadrupole("NEAR.L", .2, near_sign),
            api.drift("G2.L", .1)]
    unit_rows = api.symmetric_cell_elements(left, center=api.sector_bend("B", .8, theta))
    assert len(unit_rows) == 11
    unit = api.periodic_cell(title="Synthetic complete two-family unit", energy_gev=3.5,
        elements=unit_rows, intended_repetitions=192, source="unit test")
    copies = api.repeat_cell(unit_rows, 5, period_prefix="U")
    assembled_rows = [api.drift("ID_L", 2.), api.sector_bend("TERM_L", .4, theta / 2),
                      *copies, api.sector_bend("TERM_R", .4, theta / 2), api.drift("ID_R", 2.)]
    assembled = api.periodic_cell(title="Synthetic opposite-polarity transfer", energy_gev=3.5,
        elements=assembled_rows, intended_repetitions=32, source="unit test")
    groups = [[row["name"] for row in copies[i:i + len(unit_rows)]]
              for i in range(0, len(copies), len(unit_rows))]
    machine = copy.deepcopy(raw["machine_contract"])
    machine["expected_cell_length_m"] = sum(row["length_m"] for row in assembled_rows)
    issues, _, repetitions = claims._transfer(unit, assembled, assembled, groups, machine)
    assert not issues, issues
    assert repetitions == 192


@pytest.mark.parametrize("mutation", [None, "strength", "multipole", "integration"])
def test_uniquified_unit_copies_ignore_provenance_but_preserve_physics(example, mutation):
    _, unit, _, raw = example
    unit_rows = api.one_period_elements(unit)
    theta = sum(row.get("angle_rad") or 0.0 for row in unit_rows)
    # Exercise the same public path as project-written assembly: duplicate
    # rows first, then give each occurrence a unique saved element name.
    rows = api.uniquify_element_names([
        api.drift("ID_L", 2.), api.sector_bend("TERM_L", .4, theta / 2),
        *[copy.deepcopy(row) for _ in range(5) for row in unit_rows],
        api.sector_bend("TERM_R", .4, theta / 2), api.drift("ID_R", 2.),
    ])
    groups = [[row["name"] for row in rows[2 + i * len(unit_rows):2 + (i + 1) * len(unit_rows)]]
              for i in range(5)]
    copied_quad = rows[2 + len(unit_rows) + 1]
    assert copied_quad["fields"]["source_element_name"] == "QF_L"
    assert copied_quad["name"] == "QF_L#2"
    if mutation == "strength":
        copied_quad["k1_m2"] += .1
    elif mutation == "multipole":
        copied_quad["fields"]["PolynomB"] = [0., 1., .05]
    elif mutation == "integration":
        copied_quad["fields"]["NumIntSteps"] = 20
    assembled = api.periodic_cell(title="Uniquified unit transfer", energy_gev=unit.energy_gev,
        elements=rows, intended_repetitions=32, source="unit test")
    issues, _, _ = claims._transfer(unit, assembled, assembled, groups, raw["machine_contract"])
    if mutation is None:
        assert not issues, issues
    else:
        assert "assembly changes the supplied unit's physical fields before the declared transfer" in issues


@pytest.mark.parametrize("mutation", ["missing_transfer", "changed_unit_fields", "overlapping_groups", "unit_count",
    "wrong_closure", "unstable_unit", "negative_damping", "missing_centre", "wrong_profile_rows", "fake_sampler",
    "wrong_full_ring", "radiation_sum", "achromat", "nan_beta", "nan_spread", "wrong_scope", "machine_field"])
def test_bad_measured_evidence_cannot_be_overridden_by_passed_flag(example, tmp_path, mutation):
    candidate, _, _, raw = example
    if mutation == "missing_transfer":
        del raw["unit_groups"]
    elif mutation == "changed_unit_fields":
        raw["assembled_candidate"]["elements"][3]["k1_m2"] = 1.2
    elif mutation == "overlapping_groups":
        raw["unit_groups"][1] = raw["unit_groups"][0]
    elif mutation == "unit_count":
        raw["unit_groups"].pop()
    elif mutation == "wrong_closure":
        raw["unit"]["elements"][4]["angle_rad"] *= 1.01
    elif mutation == "unstable_unit":
        raw["unit_evaluation"]["linear_optics"]["trace_x"] = 2.1
    elif mutation == "negative_damping":
        raw["unit_evaluation"]["radiation"]["Jx"] = -1.
    elif mutation == "missing_centre":
        raw["unit_profile"]["twiss_profile"].pop(1)
    elif mutation == "wrong_profile_rows":
        raw["candidate_profile"]["sampling"]["cell_fingerprint"] = "different"
    elif mutation == "fake_sampler":
        raw["unit_profile"]["sampling"]["engine"] = "model approximation"
    elif mutation == "wrong_full_ring":
        raw["full_ring_evaluation"]["candidate"]["elements"][3]["k1_m2"] = 1.9
    elif mutation == "radiation_sum":
        raw["per_bend_radiation"]["rows"][0]["I5"] *= 4
    elif mutation == "achromat":
        raw["full_ring_evaluation"]["linear_optics"]["twiss_profile"][0]["Dx"] = .01
    elif mutation == "nan_beta":
        raw["candidate_profile"]["twiss_profile"][1]["beta_y"] = float("nan")
    elif mutation == "nan_spread":
        raw["full_ring_evaluation"]["radiation"]["energy_spread"] = float("nan")
    elif mutation == "wrong_scope":
        raw["candidate_evaluation"]["periods"] = 192
    else:
        raw["machine_contract"]["limits"]["max_abs_k1_m2"] = .5
    assert checked(candidate, raw, tmp_path)


def test_scalar_tme_reference_is_not_a_measured_unit_or_transfer(example, tmp_path):
    candidate, _, _, _ = example
    assert checked(candidate, {"schema": claims.SCHEMA, "passed": True, "tme_emittance_pm": 10.}, tmp_path)
    with numerical_evidence_directory(tmp_path):
        assert claims.validate_conventional_linear({"passed": True}, candidate)


def test_receipt_from_other_candidate_cannot_certify_archived_rows(example, tmp_path):
    candidate, _, assembly, raw = example
    with numerical_evidence_directory(tmp_path):
        receipt = record_numerical_evidence(candidate, raw)
        assert claims.validate_conventional_linear({"measurement_receipt": receipt}, assembly)


def test_measurement_helper_calls_native_adapters_without_a_search(example, monkeypatch, tmp_path):
    candidate, unit, assembly, raw = example
    calls = []
    values = iter(raw[key] for key in ("unit_evaluation", "assembly_evaluation", "candidate_evaluation"))

    def periodic(subject, **kwargs):
        calls.append(("periodic", subject.metadata["intended_repetitions"]))
        return next(values)

    def sampler(subject, **kwargs):
        class Sampler:
            def linear_metrics(self, **sampling):
                is_unit = len(subject.elements) == len(unit.elements)
                if is_unit:
                    assert sampling == {"sample_fractions": {"B": [.5]}}
                return raw["unit_profile" if is_unit else "candidate_profile"]
        return Sampler()

    def evaluate(ring, **kwargs):
        class Result:
            def as_dict(self):
                return {**raw["full_ring_evaluation"], "candidate": ring.as_dict()}
        calls.append(("full_ring", ring.periodicity))
        return Result()

    monkeypatch.setattr(claims.api, "evaluate_periodic_cell", periodic)
    monkeypatch.setattr(claims.api, "build_jutrack_profile_sampler", sampler)
    monkeypatch.setattr(claims.api, "evaluate", evaluate)
    monkeypatch.setattr(claims.api, "bend_radiation_contributions", lambda ring: raw["per_bend_radiation"])
    result = claims.verify_conventional_candidate(candidate, unit=unit, assembled_candidate=assembly,
        unit_groups=raw["unit_groups"], machine_contract=raw["machine_contract"], output_directory=tmp_path)
    assert result["passed"], result["issues"]
    assert calls == [("periodic", 192), ("periodic", 32), ("periodic", 32), ("full_ring", 32)]
    assert "nonlinear" in result["limitations"][1].lower()
    with numerical_evidence_directory(tmp_path):
        assert claims.validate_conventional_linear(result, candidate) == []
