"""Independent evidence for a project-written modified-TME to MBA prototype.

This module measures and checks supplied objects; it never designs or matches.
"""
from __future__ import annotations

import copy
import math

from brilliance import accelerator_api as api
from brilliance.candidate_evidence import read_numerical_evidence, record_numerical_evidence
from brilliance.lattice import LatticeIR
from brilliance.lattice_fingerprint import cell_fingerprint
from brilliance.lattice_optimization import radiation_physical_validity

from .outcome_claims import _machine
from .physics_references import tme_targets

SCHEMA = "conventional_mba_outcome/v1"


def _load(value):
    return value if isinstance(value, LatticeIR) else api.load_lattice(value)


def _physical(row):
    # Public copy/rename helpers attach origin names without changing a JuTrack
    # element. Keep all other fields, including integration and multipole data.
    provenance = {"family", "source_element", "source_element_name", "source_period",
                  "mirror_source_element", "design_role"}
    return {"kind": row["kind"], **{key: float(row.get(key) or 0) for key in
        ("length_m", "angle_rad", "k1_m2", "k2_m3", "k3_m4")},
        "fields": {key: value for key, value in row.get("fields", {}).items()
                   if key not in provenance}}


def _transfer(unit, assembled, candidate, groups, machine):
    unit_rows, assembled_rows, final_rows = (api.one_period_elements(value) for value in (unit, assembled, candidate))
    issues = _machine(assembled, machine) + _machine(candidate, machine)
    if unit.energy_gev != candidate.energy_gev:
        issues.append("unit and complete cell have different beam energies")
    bends = [i for i, row in enumerate(unit_rows) if float(row.get("angle_rad") or 0) != 0]
    if len(bends) != 1:
        raise ValueError("measured modified-TME unit must contain one physical positive bend")
    bend = unit_rows[bends[0]]
    for rows in (unit_rows, assembled_rows, final_rows):
        if any(row["kind"] not in {"SBEND"} or float(row["angle_rad"]) <= 0
               or any(_physical(row)[key] != 0 for key in ("k1_m2", "k2_m3", "k3_m4"))
               or any(row.get("fields", {}).get("PolynomB", []))
               for row in rows if float(row.get("angle_rad") or 0) != 0):
            issues.append("this conventional prototype requires positive homogeneous bends")
    strengths = [float(row.get("k1_m2") or 0) for row in unit_rows if row["kind"] in {"KQUAD", "QUAD"}]
    if not any(value > 0 for value in strengths) or not any(value < 0 for value in strengths):
        issues.append("the unit lacks both focusing and defocusing quadrupoles")
    count = round(2 * math.pi / float(bend["angle_rad"])) if float(bend["angle_rad"]) > 0 else 0
    if count < 1 or abs(count * float(bend["angle_rad"]) - 2 * math.pi) > 1e-9:
        issues.append("unit reference repetitions do not close to 2*pi")
    declared = unit.metadata.get("intended_repetitions", unit.metadata.get("repetitions", unit.periodicity))
    if declared != count:
        issues.append("unit's declared reference-ring repetitions are inconsistent with its actual bend")
    audit = api.audit_lattice_constraints(unit_rows, energy_gev=unit.energy_gev,
        expected_length_m=sum(row["length_m"] for row in unit_rows), expected_angle_rad=float(bend["angle_rad"]),
        expected_bends=1, require_mirror_symmetry=True, limits=machine["limits"],
        minimum_magnet_length_m=machine["minimum_magnet_length_m"], minimum_drift_length_m=machine["minimum_drift_length_m"])
    issues.extend("unit: " + issue for issue in audit["issues"])
    if [(row["name"], row["kind"]) for row in final_rows] != [(row["name"], row["kind"]) for row in assembled_rows]:
        issues.append("final candidate does not retain the declared assembled cell's named topology")
    if not isinstance(groups, (list, tuple)) or not groups or len(groups) not in {
            machine["expected_bends"], machine["expected_bends"] - 2}:
        raise ValueError("unit_groups must identify M repeated units (or M-2 archived legacy core copies)")
    outer_pair_route = len(groups) == machine["expected_bends"]
    lookup = {row["name"]: i for i, row in enumerate(assembled_rows)}
    covered = []
    for group_number, group in enumerate(groups):
        if not isinstance(group, (list, tuple)) or len(group) != len(unit_rows) or any(name not in lookup for name in group):
            raise ValueError("each unit group must name every source-unit element in assembly order")
        indices = [lookup[name] for name in group]
        if indices != list(range(indices[0], indices[0] + len(unit_rows))):
            issues.append("transferred unit elements are not contiguous and ordered")
        actual, source = [_physical(assembled_rows[i]) for i in indices], [_physical(row) for row in unit_rows]
        outer = outer_pair_route and group_number in {0, 1, len(groups)-2, len(groups)-1}
        if outer:
            # Outer units are derived from the unit topology, not asserted to
            # retain its periodic optics. Account explicitly for geometry and
            # matching-field edits; no changed kinds/native edge conventions.
            for supplied, original in zip(actual, source):
                for key in ("length_m", "k1_m2", "angle_rad"):
                    supplied[key] = original[key]
        if actual != source:
            issues.append("assembly changes the supplied unit's physical fields outside the declared outer-unit edits"
                          if outer_pair_route else "assembly changes the supplied unit's physical fields before the declared transfer")
        covered.extend(indices)
    if covered != list(range(covered[0], covered[-1] + 1)):
        issues.append("unit copies overlap, are out of order, or have gaps between them")
    cell_bends = [i for i, row in enumerate(assembled_rows) if float(row.get("angle_rad") or 0) != 0]
    if outer_pair_route:
        if len(cell_bends) != len(groups) or not set(cell_bends) <= set(covered):
            issues.append("each repeated unit must contain one of the complete MBA's physical bends")
    elif len(cell_bends) != machine["expected_bends"] or not set(cell_bends[1:-1]) <= set(covered) or cell_bends[0] in covered or cell_bends[-1] in covered:
        issues.append("legacy unit copies must lie between two distinct terminal bends")
    return issues, bend, count


def _stable(measured):
    linear = measured.get("linear_optics", measured)
    return all(isinstance(linear.get(key), (int, float)) and math.isfinite(linear[key])
               and abs(linear[key]) < 2 for key in ("trace_x", "trace_y"))


def _checks(candidate, raw):
    if raw.get("schema") != SCHEMA:
        raise ValueError("a measured unit, assembled cell and final candidate are required")
    unit, assembled = LatticeIR.from_dict(raw["unit"]), LatticeIR.from_dict(raw["assembled_candidate"])
    machine = raw["machine_contract"]
    issues, bend, repetitions = _transfer(unit, assembled, candidate, raw["unit_groups"], machine)
    for label, value in (("unit", raw["unit_evaluation"]), ("final cell", raw["candidate_evaluation"]), ("full ring", raw["full_ring_evaluation"])):
        if not _stable(value) or not radiation_physical_validity(value.get("radiation", {}))["passed"]:
            issues.append(label + " does not have stable optics and a physical radiation equilibrium")
    if raw["unit_evaluation"].get("periods") != repetitions or raw["candidate_evaluation"].get("periods") != machine["periods"]:
        issues.append("measured radiation scopes do not match the unit/full-cell repetition counts")
    sampled = raw["unit_profile"]
    for label, value, source in (("unit", sampled, unit), ("final cell", raw["candidate_profile"], candidate)):
        sampling = value["sampling"]
        if (sampling.get("cell_fingerprint") != cell_fingerprint(api.one_period_elements(source))
                or sampling.get("map_equivalent") is not True or sampling.get("engine") != "JuTrack"
                or sampling.get("method") != "physical_element_prefix_map_checked"):
            issues.append(label + " interior optics are not map-checked against the supplied physical rows")
    centre_s = sum(row["length_m"] for row in api.one_period_elements(unit)[:next(i for i, row in enumerate(api.one_period_elements(unit)) if row["name"] == bend["name"])]) + bend["length_m"] / 2
    centre = next((point for point in sampled.get("twiss_profile", []) if abs(point["s_m"] - centre_s) < 1e-9), None)
    if centre is None or any(not math.isfinite(centre[key]) for key in ("beta_x", "beta_y", "alpha_x", "Dx", "Dpx")) or min(centre["beta_x"], centre["beta_y"]) <= 0:
        raise ValueError("a real finite JuTrack sample at the physical unit bend centre is missing")
    target = tme_targets(energy_gev=unit.energy_gev, bend_length_m=bend["length_m"], bend_angle_rad=bend["angle_rad"]).as_dict()
    final = raw["full_ring_evaluation"]
    measured_ring = LatticeIR.from_dict(final["candidate"])
    if (measured_ring.energy_gev != candidate.energy_gev or measured_ring.periodicity != machine["periods"]
            or [_physical(row) for row in api.one_period_elements(measured_ring)] != [_physical(row) for row in api.one_period_elements(candidate)]):
        issues.append("independent full-ring measurement does not repeat the exact selected cell")
    boundary = final["linear_optics"]["twiss_profile"][0]
    if any(not math.isfinite(boundary[key]) for key in ("Dx", "Dpx", "beta_x", "beta_y")):
        issues.append("final complete ring has a nonfinite ID boundary")
    if abs(boundary["Dx"]) > machine["achromat_dx_tolerance_m"] or abs(boundary["Dpx"]) > machine["achromat_dpx_tolerance"]:
        issues.append("final complete ring is not achromatic at the ID boundary")
    for axis in ("x", "y"):
        bounds = machine.get(f"id_beta_{axis}_bounds_m")
        if bounds is not None and not bounds[0] <= boundary[f"beta_{axis}"] <= bounds[1]:
            issues.append(f"original ID beta_{axis} bounds exceeded")
    radiation = final["radiation"]
    for key in ("I2", "I4", "I5", "emittance_pm", "energy_spread"):
        if not math.isclose(radiation[key], raw["candidate_evaluation"]["radiation"][key], rel_tol=1e-7, abs_tol=1e-13):
            issues.append("periodic/full-ring numerical disagreement: " + key)
    contributions = raw["per_bend_radiation"]
    for key in ("I2", "I4", "I5"):
        if not math.isclose(sum(row[key] for row in contributions["rows"]), radiation[key], rel_tol=2e-10, abs_tol=1e-13):
            issues.append("per-bend radiation sum disagrees with JuTrack ringpara: " + key)
    profile = raw["candidate_profile"]["twiss_profile"]
    if not profile or any(not math.isfinite(point[key]) or point[key] <= 0 for point in profile for key in ("beta_x", "beta_y")):
        raise ValueError("final sampled beta functions must be finite and positive")
    beta_max = max(max(point["beta_x"], point["beta_y"]) for point in profile)
    if not math.isfinite(radiation["energy_spread"]) or radiation["energy_spread"] <= 0:
        issues.append("equilibrium energy spread is not finite and positive")
    for value, cap in ((beta_max, "maximum_beta_m"), (radiation["emittance_pm"], "maximum_emittance_pm"), (radiation["energy_spread"], "maximum_energy_spread")):
        if machine[cap] is not None and value > machine[cap]:
            issues.append("original machine cap exceeded: " + cap)
    return issues, {"analytic_tme": target, "measured_bend_centre": centre,
        "unit_reference_repetitions": repetitions, "unit_emittance_pm": raw["unit_evaluation"]["radiation"]["emittance_pm"],
        "unit_to_tme_ratio": raw["unit_evaluation"]["radiation"]["emittance_pm"] / target["minimum_emittance_pm"],
        "assembled_emittance_pm": raw["assembly_evaluation"].get("radiation", {}).get("emittance_pm"),
        "final_emittance_pm": radiation["emittance_pm"], "sampled_beta_max_m": beta_max,
        "transferred_units": len(raw["unit_groups"])}


def verify_conventional_candidate(candidate, *, unit, assembled_candidate, unit_groups, machine_contract, output_directory=None):
    """Measure project-authored unit/assembly/final objects; never design or optimize.

    See references/conventional_api.md. unit_groups lists the real saved assembly
    names of all M complete repeated units, in physical order (M-2 is supported
    only for archived legacy unit/terminal evidence). machine_contract
    uses the original energy/periods/geometry/ID/field/cap/provenance fields of
    verify_hmba_candidate. Optional profile_step_m (default 0.05) controls
    diagnostic sampling, not the physical element integration. No tune target or
    universal TME-quality ceiling is imposed. All raw evidence is archived.
    """
    candidate, unit, assembled = map(_load, (candidate, unit, assembled_candidate))
    machine = copy.deepcopy(dict(machine_contract))
    _, bend, count = _transfer(unit, assembled, candidate, unit_groups, machine)
    step = machine.get("profile_step_m", .05)

    def periodic(value, periods):
        return api.periodic_cell(title=value.title + " — independent measurement", energy_gev=value.energy_gev,
            elements=api.one_period_elements(value), intended_repetitions=periods, source="independent conventional verification")

    unit_cell, assembly_cell, final_cell = periodic(unit, count), periodic(assembled, machine["periods"]), periodic(candidate, machine["periods"])
    ring = api.cell_ring(title="Independent complete conventional ring", energy_gev=candidate.energy_gev,
                        cell=api.one_period_elements(candidate), periods=machine["periods"], source="exact selected candidate replay")
    raw = {"schema": SCHEMA, "unit": unit.as_dict(), "assembled_candidate": assembled.as_dict(),
        "unit_groups": copy.deepcopy(unit_groups), "machine_contract": machine,
        "unit_evaluation": api.evaluate_periodic_cell(unit_cell, include_profile=True),
        "assembly_evaluation": api.evaluate_periodic_cell(assembly_cell, include_profile=True),
        "candidate_evaluation": api.evaluate_periodic_cell(final_cell, include_profile=True),
        "unit_profile": api.build_jutrack_profile_sampler(unit_cell, max_step_m=step).linear_metrics(sample_fractions={bend["name"]: [.5]}),
        "candidate_profile": api.build_jutrack_profile_sampler(final_cell, max_step_m=step).linear_metrics(),
        "full_ring_evaluation": api.evaluate(ring, include_profile=True, include_radiation=True).as_dict(),
        "per_bend_radiation": api.bend_radiation_contributions(ring)}
    receipt = record_numerical_evidence(candidate, raw, output_directory=output_directory)
    try:
        issues, metrics = _checks(candidate, raw)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        issues, metrics = ["Incomplete numerical prototype evidence: " + str(exc)], {}
    return {"schema": SCHEMA, "scope": "linear_conventional_mba_prototype", "measurement_receipt": receipt,
            "passed": not issues, "issues": issues, "metrics": metrics,
            "limitations": ["Measured TME conditions are a comparison, not an exact-TME or optimality claim.",
                            "Chromatic correction, nonlinear acceptance and robustness are not certified."]}


def validate_conventional_linear(report, candidate):
    try:
        return _checks(candidate, read_numerical_evidence(candidate, report["measurement_receipt"]))[0]
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError) as exc:
        return ["Unverified conventional prototype: " + str(exc)]
