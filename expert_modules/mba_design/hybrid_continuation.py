"""Bounded, resumable hybrid hypothesis exploration, not a lattice design recipe.

The scientist/agent supplies connectivity, geometry bounds and shape requirements.
Numerical trials run locally, not through repeated model-written matcher scripts.
No conventional seed creation, facility assets, or automatic publication here.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Callable, Mapping

import numpy as np

from .hybrid_regions import hybrid_mba_regions
from .hybrid_shape import _thresholds
from .outer_dba import build_outer_dba_section, match_outer_dba_section

RELEASE_GROUPS = (
    ("all_mirrored_quadrupoles", ()),
    ("drift_allocation", ("suppressor_length_m", "bump_length_m")),
    ("outer_bend_angles", ("inner_bend_fraction",)),
    ("grouped_magnet_lengths", ("suppressor_quad_length_m", "bump_quad_length_m",
                                "terminal_quad_length_m", "inner_bend_length_m", "terminal_bend_length_m")),
)


def _coordinates(geometry):
    counts = list(geometry.get("quadrupoles_per_region", (2, 2, 2)))
    lengths = geometry.get("quadrupole_lengths_m", [geometry["quadrupole_length_m"]]*sum(counts))
    groups, offset = [], 0
    for count in counts:
        section = lengths[offset:offset+count]
        if len(section) != count or not all(math.isclose(v, section[0], abs_tol=1e-12) for v in section):
            raise ValueError("Grouped continuation requires one initial quadrupole length per physical region")
        groups.append(section[0])
        offset += count
    return dict(suppressor_length_m=geometry["suppressor_length_m"], bump_length_m=geometry["bump_length_m"],
        inner_bend_fraction=geometry["inner_bend_fraction"],
        suppressor_quad_length_m=groups[0], bump_quad_length_m=groups[1], terminal_quad_length_m=groups[2],
        inner_bend_length_m=geometry["bend_lengths_m"][0], terminal_bend_length_m=geometry["bend_lengths_m"][1])


def _geometry(base, coordinates):
    result = deepcopy(base)
    counts = result.get("quadrupoles_per_region", (2, 2, 2))
    for key in ("suppressor_length_m", "bump_length_m", "inner_bend_fraction"):
        result[key] = coordinates[key]
    result["bend_lengths_m"] = [coordinates["inner_bend_length_m"], coordinates["terminal_bend_length_m"]]
    result["quadrupole_lengths_m"] = [coordinates[key] for key, count in
        zip(("suppressor_quad_length_m", "bump_quad_length_m", "terminal_quad_length_m"), counts) for _ in range(count)]
    return result


def explore_hybrid_mba(replacement, *, energy_gev: float, geometry: Mapping[str, Any],
        input_twiss: Mapping[str, float], center_optics_bounds: Mapping[str, Any],
        thresholds: Mapping[str, float], limits: Mapping[str, float],
        continuation_targets_m, geometry_bounds: Mapping[str, Any],
        continuation_contrasts=None,
        trials_per_stage: int = 4, starts_per_trial: int = 2, max_nfev: int = 300,
        initialize_achromat: bool = True,
        random_seed: int = 0, max_trials: int | None = None,
        parent_lattice_id: str | None = None,
        progress: Callable[[dict], None] | None = None,
        checkpoint: Callable[[dict], None] | None = None,
        cancelled: Callable[[], bool] | None = None, resume: Mapping | None = None) -> dict:
    """Try all mirrored K1 first, then cumulatively release declared geometry.

    Exact outer length/angle budgets and ID length are never relaxed. Every K1
    (including core pairs) belongs to the inner matcher; outer coordinates are
    geometry only. After stalls: drift allocation -> outer angle split -> grouped
    magnet lengths. Geometry trial budgets are explicit, NOT optimizer saturation
    evidence. No H/I5/source-optics equality is imposed in this feasibility stage.

    Return status, trial records and a checkpoint containing the best candidate
    and independent shape report. ``bump_feasible`` is NOT a finished hybrid
    design; emittance recovery and full_hybrid acceptance remain separate.
    callbacks receive JSON-compatible data. A cancellation or max_trials pauses
    between trials; it never kills an independent job or discards its checkpoint.
    """
    from brilliance import accelerator_api as api
    targets = list(map(float, continuation_targets_m))
    shape = _thresholds(thresholds, stage="bump_feasibility")
    if (not targets or any(not math.isfinite(v) or v <= 0 for v in targets)
            or any(b <= a for a,b in zip(targets,targets[1:])) or targets[-1] < shape["min_bump_m"]):
        raise ValueError("Use increasing positive continuation targets ending at or above min_bump_m")
    contrasts = ([shape['min_contrast']]*len(targets) if continuation_contrasts is None
                 else list(map(float,continuation_contrasts)))
    if (len(contrasts)!=len(targets) or any(not math.isfinite(v) or v<=1 for v in contrasts)
            or any(b<a for a,b in zip(contrasts,contrasts[1:])) or contrasts[-1]<shape['min_contrast']):
        raise ValueError("Continuation contrasts must be nondecreasing, exceed 1, and end at or above min_contrast")
    for label, value in (("trials_per_stage",trials_per_stage),("starts_per_trial",starts_per_trial),("max_nfev",max_nfev)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if max_trials is not None and (type(max_trials) is not int or max_trials < 1):
        raise ValueError("max_trials must be a positive integer or None")
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    if type(initialize_achromat) is not bool:
        raise ValueError("initialize_achromat must be boolean")
    if parent_lattice_id is not None and (not isinstance(parent_lattice_id,str) or not parent_lattice_id.strip()):
        raise ValueError("parent_lattice_id must be a nonempty source identity or None")
    base = deepcopy(dict(geometry))
    if "k1_values" not in base:
        raise ValueError("Declare initial outer k1_values explicitly (they need not be solved)")
    initial_rows = build_outer_dba_section(replacement, **base)  # geometry schema preflight
    initial = _coordinates(base)
    unknown = set(geometry_bounds)-set(initial)
    if unknown:
        raise ValueError("Unknown grouped geometry bounds: " + ", ".join(sorted(unknown)))
    bounds = {}
    for key, pair in geometry_bounds.items():
        pair = list(map(float,pair))
        if (len(pair)!=2 or not all(math.isfinite(v) for v in pair) or
                not 0 < pair[0] <= initial[key] <= pair[1] or pair[0] == pair[1]
                or (key == "inner_bend_fraction" and pair[1] >= 1)):
            raise ValueError(f"Invalid bounds for {key}; include the initial geometry, positive lengths and fraction in (0,1)")
        bounds[key] = pair
    kmax, radius = float(limits["max_abs_k1_m2"]), float(limits["minimum_bend_radius_m"])
    if not all(math.isfinite(v) and v > 0 for v in (energy_gev,kmax,radius)):
        raise ValueError("Positive energy, K1 limit and minimum bend radius are required")
    core_half = replacement.right_core_half()
    initial_k = [float(r["k1_m2"] or 0) for r in initial_rows+core_half if r["kind"] == "KQUAD"]
    if any(abs(k)>kmax for k in initial_k):
        raise ValueError("Initial outer/core strengths exceed the declared K1 envelope")
    contract = dict(schema="hybrid_continuation/v1", replacement=replacement.describe(),
        source=replacement.source, energy_gev=energy_gev,geometry=base,input_twiss=dict(input_twiss),
        center_optics_bounds=dict(center_optics_bounds),thresholds=shape,limits=dict(limits),
        targets=targets,contrasts=contrasts,bounds=bounds,trials_per_stage=trials_per_stage,starts_per_trial=starts_per_trial,
        max_nfev=max_nfev,initialize_achromat=initialize_achromat,random_seed=random_seed,
        parent_lattice_id=parent_lattice_id)
    identity = hashlib.sha256(json.dumps(contract,sort_keys=True,allow_nan=False).encode()).hexdigest()
    state = deepcopy(dict(resume)) if resume is not None else dict(
        schema="hybrid_continuation/v1",contract_id=identity,target_index=0,release_index=0,trial_index=0,
        records=[],warm_position=initial_k+[input_twiss[k] for k in ("betax","betay","dx")],
        warm_geometry=initial,warm_score=None,best=None,status="running")
    if state.get("schema") != "hybrid_continuation/v1" or state.get("contract_id") != identity:
        raise ValueError("Continuation checkpoint belongs to a different source, bounds, targets or matching contract")
    if (any(type(state.get(k)) is not int or state[k] < 0 for k in ('target_index','release_index','trial_index'))
            or not isinstance(state.get('records'),list) or not isinstance(state.get('warm_geometry'),dict)
            or set(state['warm_geometry']) != set(initial)
            or len(state.get('warm_position',[])) != len(initial_k)+3
            or state['target_index'] > len(targets) or state['release_index'] > len(RELEASE_GROUPS)):
        raise ValueError("Incomplete or invalid hybrid continuation checkpoint; restore the last complete atomic checkpoint")
    if state.get("status") == "bump_feasible":
        return state
    state["status"] = "running"
    performed = 0
    def emit(record):
        if progress:
            progress(deepcopy(record))
    def save():
        if checkpoint:
            checkpoint(deepcopy(state))
    while state["target_index"] < len(targets) and state["release_index"] < len(RELEASE_GROUPS):
        if (cancelled and cancelled()) or (max_trials is not None and performed >= max_trials):
            state["status"] = "paused"
            save()
            return state
        ti, ri, trial = (state[k] for k in ("target_index","release_index","trial_index"))
        stage = RELEASE_GROUPS[ri][0]
        active = [key for _,group in RELEASE_GROUPS[:ri+1] for key in group if key in bounds]
        target = targets[ti]
        rng = np.random.default_rng(np.random.SeedSequence([random_seed,ti,ri,trial]))
        coords = deepcopy(state["warm_geometry"])
        if trial > 0 or ri > 0:
            for key in active:
                lo,hi = bounds[key]
                # Alternate neighbourhood and full-bound exploration, not just local polish.
                coords[key] = float(np.clip(rng.normal(coords[key],.2*(hi-lo)),lo,hi) if trial%2==0 else rng.uniform(lo,hi))
        contrast = contrasts[ti]
        record = dict(trial=len(state["records"])+1,stage=stage,target_bump_m=target,target_contrast=contrast,
            active_geometry_variables=active,geometry_coordinates=coords)
        emit({**record,"status":"matching","message":"All global mirror-paired quadrupoles and free centre optics rematch; no tune target."})
        full_gate = None
        # Only known geometric infeasibility is screened. API/runtime exceptions
        # propagate; they must not be mistaken for negative physics evidence.
        try:
            outer = build_outer_dba_section(replacement,**_geometry(base,coords))
        except ValueError as exc:
            if "insufficient physical drift budget" not in str(exc):
                raise
            record.update(status="geometry_rejected",reason=str(exc))
            outer = None
        if outer is not None and any(r.get("angle_rad") and r["length_m"] < radius*abs(r["angle_rad"])-1e-12 for r in outer):
            record.update(status="geometry_rejected",reason="Dipole field exceeds declared limit")
            outer = None
        if outer is not None:
            inventory = hybrid_mba_regions(replacement.assemble(outer))['elements']
            record['knob_regions'] = [{key:r[key] for key in ('name','mirror_name','region','s_start_m','s_end_m')}
                for r in inventory if r['kind']=='KQUAD' and r['index'] >= r['mirror_index']]
            position = state["warm_position"]
            starts = [position]
            for restart in range(starts_per_trial-1):
                # Include independent optical starts, not the same centre point
                # with tiny strength perturbations that keep returning one basin.
                broad = restart % 2 == 1
                strengths = ([float(v) for v in rng.uniform(-kmax,kmax,len(position)-3)] if broad else
                    [float(np.clip(v+rng.normal(0,.25*kmax),-kmax,kmax)) for v in position[:-3]])
                optics = []
                for key,v in zip(('betax','betay','dx'),position[-3:]):
                    lo,hi = center_optics_bounds[key]
                    if key == 'dx':
                        value = rng.uniform(lo,hi) if broad else v+rng.normal(0,.1*(hi-lo))
                    else:
                        value = math.exp(rng.uniform(math.log(lo),math.log(hi))) if broad else v*math.exp(rng.normal(0,.5))
                    optics.append(float(np.clip(value,lo,hi)))
                starts.append(strengths+optics)
            fit = match_outer_dba_section(outer,energy_gev=energy_gev,input_twiss=input_twiss,
                core_prefix=core_half,center_optics_bounds=center_optics_bounds,rematch_core_quadrupoles=True,
                bump_station_name="DBA_R:BUMP",target_bump_m=target,starts=starts,max_abs_k1_m2=kmax,
                max_nfev=max_nfev,id_dispersion_tolerance_m=shape["id_dispersion_tolerance_m"],
                # A warm start already contains shape information. Repeating an
                # achromat-only prefit can erase it and return the same poor basin.
                initialize_achromat=initialize_achromat and not any('evaluations' in r for r in state['records']),
                slope_tolerance=min(shape["id_slope_tolerance"],shape["max_drift_slope"]),
                max_core_dispersion_m=shape["max_core_dispersion_m"],
                max_beta_m=shape["max_beta_m"])
            score = max(abs(v) for v in fit["scaled_residuals"])
            record.update(status="line_matched" if fit["matched"] else "line_unmatched",evaluations=fit["evaluations"],
                matching_residuals=dict(zip(fit["residual_names"],fit["scaled_residuals"])),
                saturated_strengths=fit["diagnostics"]["saturated_strengths"],
                quadrupole_ownership=fit["quadrupole_ownership"])
            # Best line state is a warm start, never an accepted periodic candidate.
            if state["warm_score"] is None or score < state["warm_score"]:
                state.update(warm_position=fit["position"],warm_geometry=coords,warm_score=score)
            if fit["matched"]:
                cell = replacement.assemble(fit["elements"], core_elements=replacement.core_from_half(fit["core_prefix"]))
                candidate = api.periodic_cell(title="Hybrid bump continuation trial",energy_gev=energy_gev,
                    elements=cell,source="expert hybrid continuation",parent_lattice_id=parent_lattice_id,
                    intended_repetitions=round(2*math.pi/replacement.cell_angle))
                current_shape = {**shape,"min_bump_m":min(target,shape["min_bump_m"]),
                                 "min_contrast":min(contrast,shape['min_contrast'])}
                gate = replacement.evaluate(candidate,thresholds=current_shape,limits=limits,
                    max_step_m=min(.02,shape["max_sample_step_m"]))
                full_gate = gate if current_shape == shape else None
                record.update(status="continuation_passed" if gate["passed"] else "periodic_gate_rejected",
                    metrics=gate.get("metrics",{}),failed_checks=[k for k,v in gate["checks"].items() if not v])
                # Save separate candidates/diagnostics; never hide H, beta or damping
                # behind an improved contrast or label an incomplete stage 'hybrid'.
                best = state["best"]
                rank = (bool(gate["passed"]),target,-len(record["failed_checks"]),-score)
                if best is None or rank > tuple(best["rank"]):
                    state["best"] = dict(rank=list(rank),candidate_lattice=candidate.as_dict(),report=gate,
                        geometry=coords,position=fit["position"],target_bump_m=target)
                if gate["passed"]:
                    state.update(warm_position=fit["position"],warm_geometry=coords,warm_score=None,
                        target_index=ti+1,trial_index=0)
        state["records"].append(record)
        performed += 1
        if full_gate is not None and full_gate["passed"]:
            state["status"] = "bump_feasible"
        elif state["target_index"] == ti:
            state["trial_index"] += 1
            if state["trial_index"] >= trials_per_stage:
                state.update(release_index=ri+1,trial_index=0,warm_score=None)
                # Skip undeclared release groups, never invent geometry bounds.
                while state["release_index"] < len(RELEASE_GROUPS) and not any(k in bounds for k in RELEASE_GROUPS[state["release_index"]][1]):
                    state["release_index"] += 1
                record["transition"] = "bounded_stage_stalled; release_next_declared_group"
        emit(record)
        save()
        if state["status"] == "bump_feasible":
            return state
    state["status"] = "search_budget_exhausted"
    state["limitation"] = "This bounded hypothesis search did not establish usable hybrid bumps. It does not prove physical impossibility or topology saturation. Inspect independent failed checks before choosing new bounds/connectivity."
    save()
    return state
