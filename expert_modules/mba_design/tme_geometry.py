"""Joint geometry/strength unit search; native callbacks, existing PSO and gates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from brilliance import accelerator_api as api
from brilliance.artifacts import json_safe
from brilliance.atomic_io import atomic_write_text
from brilliance.numerical_checkpoints import read_numerical_checkpoint, write_numerical_checkpoint
from brilliance.optimization_budget import resolve_population_budget
from brilliance.population_optimization import _publish_progress

from . import tme_unit
from .result_types import TmeGeometryResult

_KEYS = ("bend_length_m", "quadrupole_near_length_m", "quadrupole_far_length_m", "d1_m", "d2_m", "d3_m")
_WEIGHTS = np.asarray((1., 2., 2., 2., 2., 2.))
_DEFAULT_LENGTH_BOUNDS = {"bend_length_m": (.8, 1.5),
    "quadrupole_near_length_m": (.25, .5), "quadrupole_far_length_m": (.25, .5)}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _implementation_identity():
    from brilliance import (
        jutrack_service,
        lattice_optimization,
        optics_sampling,
        population_optimization,
    )
    from brilliance.research_environment import ResearchEnvironment

    from . import physics_references

    environment = ResearchEnvironment.load()
    files = [Path(__file__), Path(tme_unit.__file__), Path(api.__file__), Path(jutrack_service.__file__),
             Path(lattice_optimization.__file__), Path(optics_sampling.__file__), Path(population_optimization.__file__),
             Path(physics_references.__file__), environment.pyjutrack_module,
             *sorted((environment.jutrack_root / "src").rglob("*.jl"))]
    return {"version": "joint-tme-simplex-v1", "numpy": np.__version__,
            "sources": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def _checkpoint_source(descriptor, default):
    return descriptor.get("source_output_directory") or default


def _tagged_checkpoint(kind, result, root, *, checks_passed):
    return {**write_numerical_checkpoint(kind, result, output_directory=root, checks_passed=checks_passed),
            "source_output_directory": str(root)}


def _restore_candidates(candidates, storage, *, source_output_directory=None):
    for candidate in candidates:
        descriptor = candidate["numerical_checkpoint"]
        source = _checkpoint_source(descriptor, source_output_directory)
        saved = read_numerical_checkpoint(descriptor, output_directory=storage,
                                         source_output_directory=source)["result"]
        candidate["evaluation"] = api.restore_candidate_evaluation(candidate["periodic_cell"],
            saved["candidate_evaluation"], output_directory=storage, source_output_directory=source)
        candidate["reference_ring_evaluation"] = api.restore_candidate_evaluation(
            candidate["reference_ring_evaluation"]["candidate"], candidate["reference_ring_receipt"],
            output_directory=storage, source_output_directory=source)
        if source is not None and Path(source).resolve() != Path(storage).resolve():
            from brilliance.candidate_evidence import verify_candidate_evaluation
            cell_receipt = verify_candidate_evaluation(candidate["periodic_cell"], candidate["evaluation"], storage)
            ring_receipt = verify_candidate_evaluation(candidate["reference_ring_evaluation"]["candidate"],
                                                       candidate["reference_ring_evaluation"], storage)
            if not cell_receipt["passed"] or not ring_receipt["passed"]:
                raise ValueError("Restored unit evidence could not be rebound to this managed run")
            candidate["reference_ring_receipt"] = ring_receipt["receipt"]
            candidate["numerical_checkpoint"] = _tagged_checkpoint("candidate_evaluation", {
                **saved, "candidate_evaluation": cell_receipt["receipt"]}, storage, checks_passed=True)
            candidate["evidence_reused_from"] = str(source)


def _checkpoint_location(checkpoint_path, storage, evidence_root):
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else storage / "tme_geometry_swarm.json" if storage else None
    if checkpoint is not None and os.environ.get("BRILLIANCE_RUN_OUTPUT"):
        workspace = Path(os.environ.get("BRILLIANCE_WORKSPACE") or evidence_root).resolve()
        if not Path(evidence_root).is_relative_to(workspace) or not checkpoint.is_relative_to(workspace):
            raise ValueError("checkpoint_path must remain inside this project's workspace; another project cannot supply its seed or checkpoint")
    return checkpoint


def _resume_requested(resume, checkpoint, sidecar):
    if not (isinstance(resume, bool) or resume == "auto"):
        raise ValueError("resume must be 'auto', True, or False")
    present = (checkpoint is not None and checkpoint.exists(), sidecar is not None and sidecar.exists())
    if resume == "auto" and not any(present):
        return False
    if resume is False:
        if any(present):
            raise ValueError("This campaign already exists; use resume='auto' or True, or choose a new checkpoint without discarding history")
        return False
    if not all(present) or not checkpoint.is_file() or not sidecar.is_file():
        raise ValueError("Resume requires both the optimizer checkpoint and its exact geometry-contract sidecar. "
                         "For a fresh campaign omit resume or use resume='auto'; a partial pair requires inspection, not deletion.")
    return True


class _Geometry:
    def __init__(self, total, energy, angle, limits, minimum_magnet, minimum_drift, length_bounds=None):
        self.total = float(total)
        if any(not math.isfinite(v) or v <= 0 for v in (total, energy, angle, minimum_magnet, minimum_drift)):
            raise ValueError("Unit length, energy, angle and physical minimum lengths must be positive and finite")
        field = float(limits["dipole_field_t"])
        if not math.isfinite(field) or field <= 0:
            raise ValueError("dipole_field_t must be positive and finite")
        self.minimum = np.asarray((max(minimum_magnet, energy / .299792458 * angle / field),
            minimum_magnet, minimum_magnet, minimum_drift, minimum_drift, minimum_drift), dtype=float)
        self.maximum = total / _WEIGHTS
        for key, pair in (length_bounds or {}).items():
            if key not in _KEYS or len(pair) != 2 or not all(math.isfinite(x) and x > 0 for x in pair) or pair[0] > pair[1]:
                raise ValueError(f"Invalid length_bounds_m entry: {key}; use a known geometry key and positive [lower, upper]")
            index = _KEYS.index(key)
            self.minimum[index] = max(self.minimum[index], pair[0])
            self.maximum[index] = min(self.maximum[index], pair[1])
        self.available = self.total - float(self.minimum @ _WEIGHTS)
        if self.available < 0:
            raise ValueError(f"Unit length {self.total:g} m is below the engineering minimum {self.total-self.available:g} m")
        if np.any(self.minimum > self.maximum) or float(self.maximum @ _WEIGHTS) < total - 1e-12:
            raise ValueError("Length bounds conflict with the exact unit length or dipole field limit; revise the declared geometry, not the requirements")
        if self.available == 0:
            raise ValueError("The exact unit length has no geometry freedom after engineering minima; use the fixed-geometry tool")

    def decode(self, coordinates):
        vector = np.asarray(coordinates, dtype=float)
        if vector.shape != (5,) or not np.isfinite(vector).all() or np.any(vector < 0) or np.any(vector > 1):
            raise ValueError("Geometry requires five finite simplex coordinates in [0,1]")
        remaining, allocations = self.available, []
        capacities = (self.maximum - self.minimum) * _WEIGHTS
        for i, fraction in enumerate(vector):
            low = max(0., remaining - float(sum(capacities[i+1:])))
            high = min(remaining, capacities[i])
            amount = low + fraction * max(0., high - low)
            allocations.append(amount)
            remaining = max(0., remaining - amount)
        allocations.append(remaining)
        return dict(zip(_KEYS, map(float, self.minimum + np.asarray(allocations) / _WEIGHTS)))

    def encode(self, geometry):
        lengths = np.asarray([geometry[key] for key in _KEYS], dtype=float)
        if (not np.isfinite(lengths).all() or np.any(lengths < self.minimum - 1e-12)
                or np.any(lengths > self.maximum + 1e-12)
                or not math.isclose(float(lengths @ _WEIGHTS), self.total, abs_tol=1e-9, rel_tol=0)):
            raise ValueError("Initial geometry must meet all minima, length bounds and the exact weighted unit length")
        allocation = np.maximum(0., (lengths - self.minimum) * _WEIGHTS)
        remaining, result = self.available, []
        capacities = (self.maximum - self.minimum) * _WEIGHTS
        for i, amount in enumerate(allocation[:-1]):
            low = max(0., remaining - float(sum(capacities[i+1:])))
            high = min(remaining, capacities[i])
            result.append(float(np.clip((amount - low) / (high - low), 0., 1.)) if high > low else 0.)
            remaining = max(0., remaining - amount)
        return result

    def random(self, rng):
        return list(rng.uniform(0., 1., 5))


def _rows(bend, geometry, position):
    actual_bend = {**copy.deepcopy(dict(bend)), "length_m": geometry["bend_length_m"]}
    return tme_unit._unit_rows(actual_bend, tuple(geometry[key] for key in
        ("d1_m", "d2_m", "d3_m", "quadrupole_near_length_m", "quadrupole_far_length_m")), *position[-2:])


class _Evaluator:
    def __init__(self, decoder, bend, energy, periods, limits, beta_cap, margin_x, margin_y):
        self.decoder, self.bend, self.energy, self.periods = decoder, bend, energy, periods
        self.beta_cap, self.margin_x, self.margin_y = beta_cap, margin_x, margin_y
        self.cap = float(limits["max_abs_k1_m2"])
        self.editor = api.build_jutrack_editor(tme_unit._cell(
            _rows(bend, decoder.decode([.5] * 5), [self.cap / 2, -self.cap / 2]), energy, periods))
        self.cache = {}

    def __call__(self, position):
        key = tuple(map(float, position))
        if key in self.cache:
            return copy.deepcopy(self.cache[key])
        geometry = self.decoder.decode(key[:5])
        checks = {"opposite_polarities": key[-2] * key[-1] < 0,
                  "quadrupole_limits": max(abs(key[-2]), abs(key[-1])) <= self.cap}
        radiation, linear = None, {}
        if all(checks.values()):
            self.editor.sync_rows(_rows(self.bend, geometry, key))
            linear = api.jutrack_linear_metrics(self.editor.jt, self.editor.lattice, include_profile=True)
            checks.update(tme_unit._linear_checks(linear, self.margin_x, self.margin_y, self.beta_cap))
            if all(checks.values()):
                radiation = api.jutrack_periodic_radiation_metrics(self.editor.jt, self.editor.lattice,
                    energy_gev=self.energy, periods=self.periods)
        checks["physical_radiation"] = bool(radiation and radiation.get("computed")
                                            and radiation.get("physical_validity", {}).get("passed"))
        feasible = all(checks.values())
        result = {"position": list(key), "geometry": geometry, "feasible": feasible, "checks": checks,
            "objectives": {"emittance_pm": float(radiation["emittance_pm"])} if feasible else {},
            "constraint_violations": {name: 1. for name, passed in checks.items() if not passed},
            "linear_optics": {name: value for name, value in linear.items() if name != "twiss_profile"},
            "radiation": radiation if feasible else None}
        self.cache[key] = json_safe(result)
        return copy.deepcopy(self.cache[key])


def optimize_tme_unit_geometry(*, energy_gev: float, bend: Mapping[str, Any], expected_length_m: float,
        reference_repetitions: int, limits: Mapping[str, float], maximum_beta_m: float | None,
        trace_margin_x: float, trace_margin_y: float, minimum_magnet_length_m: float,
        minimum_drift_length_m: float, initial_geometry: Mapping[str, float] | None = None,
        length_bounds_m: Mapping[str, Sequence[float]] | None = None,
        population_size: int | None = None, generations: int | None = None, seed: int = 0,
        geometry_candidates: int = 3, scan_resolutions: Sequence[int] = (25, 49), local_starts: int = 12,
        local_max_evaluations: int = 240, candidates_to_return: int = 4, profile_step_m: float = .05,
        output_directory: Path | str | None = None, checkpoint_path: Path | str | None = None,
        resume: bool | str = "auto", progress_callback: Callable | None = None, cancellation_check: Callable | None = None,
        max_initialization_evaluations: int | None = None) -> TmeGeometryResult:
    """Search five exact-length geometry DOFs plus two K1s, then verify finalists.

    No full two-dimensional scan runs inside a PSO objective. Initial particles
    must all pass the actual native callback. Geometry/stability/engineering
    constraints are unchanged; no unit-boundary dispersion or tune is targeted.
    Initial geometry is optional and never taken from another project.
    Default length_bounds_m: bend 0.8–1.5 m, each quadrupole 0.25–0.5 m.
    These are disclosed design allocations, not universal technology limits.
    A supplied mapping replaces these defaults; {} explicitly selects only the
    engineering minima. Keys are bend_length_m, quadrupole_near_length_m,
    quadrupole_far_length_m, d1_m, d2_m, d3_m; pairs may fix a length with [L,L].
    Incompatible geometry fails before Julia; never silently shrink magnets.
    """
    if bend.get("kind") != "SBEND" or any(float(bend.get(key) or 0) != 0 for key in ("k1_m2", "k2_m3", "k3_m4")):
        raise ValueError("The geometry search needs one pure positive SBEND")
    tme_unit._check_central_bend(bend)
    angle = float(bend["angle_rad"])
    if (isinstance(reference_repetitions, bool) or not isinstance(reference_repetitions, int)
            or reference_repetitions < 1 or abs(reference_repetitions * angle - 2 * math.pi) > 1e-9):
        raise ValueError("Reference repetitions must give exact 2*pi bend closure")
    for value in (trace_margin_x, trace_margin_y):
        if not math.isfinite(value) or not 0 <= value < 2:
            raise ValueError("Trace margins must be finite in [0,2)")
    if maximum_beta_m is not None and (not math.isfinite(maximum_beta_m) or maximum_beta_m <= 0):
        raise ValueError("maximum_beta_m must be positive or explicitly None")
    for name, value, minimum in (("seed", seed, 0), ("geometry_candidates", geometry_candidates, 1),
                                 ("population_size", population_size, 4), ("generations", generations, 1),
                                 ("max_initialization_evaluations", max_initialization_evaluations, 1),
                                 ("local_starts", local_starts, 2), ("local_max_evaluations", local_max_evaluations, 8),
                                 ("candidates_to_return", candidates_to_return, 2)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    resolutions = tuple(scan_resolutions)
    if (len(resolutions) < 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in resolutions)
            or any(b <= a for a, b in zip(resolutions, resolutions[1:]))):
        raise ValueError("scan_resolutions needs at least two increasing integers >=3")
    if not math.isfinite(profile_step_m) or profile_step_m <= 0:
        raise ValueError("profile_step_m must be positive and finite")
    resolved_bounds = copy.deepcopy(_DEFAULT_LENGTH_BOUNDS if length_bounds_m is None else dict(length_bounds_m))
    decoder = _Geometry(expected_length_m, energy_gev, angle, limits, minimum_magnet_length_m, minimum_drift_length_m, resolved_bounds)
    initial = decoder.encode(initial_geometry) if initial_geometry is not None else None
    cap = float(limits["max_abs_k1_m2"])
    audit = api.audit_lattice_constraints(_rows(bend, decoder.decode([.5] * 5), [cap / 2, -cap / 2]),
        energy_gev=energy_gev, expected_length_m=expected_length_m, expected_angle_rad=angle,
        expected_bends=1, require_mirror_symmetry=True, limits=limits,
        minimum_magnet_length_m=minimum_magnet_length_m, minimum_drift_length_m=minimum_drift_length_m)
    if not audit["passed"]:
        raise ValueError("Geometry contract failed: " + "; ".join(audit["issues"]))
    storage, evidence_root = tme_unit._output_locations(output_directory)
    checkpoint = _checkpoint_location(checkpoint_path, storage, evidence_root)
    sidecar = checkpoint.with_suffix(".geometry.json") if checkpoint else None
    if checkpoint is not None and sidecar == checkpoint:
        raise ValueError("checkpoint_path cannot be the geometry-contract sidecar itself; choose an optimizer filename")
    resume = _resume_requested(resume, checkpoint, sidecar)
    contract = {"energy_gev": energy_gev, "bend": copy.deepcopy(dict(bend)), "expected_length_m": expected_length_m,
        "reference_repetitions": reference_repetitions, "limits": dict(limits), "maximum_beta_m": maximum_beta_m,
        "trace_margin_x": trace_margin_x, "trace_margin_y": trace_margin_y,
        "minimum_magnet_length_m": minimum_magnet_length_m, "minimum_drift_length_m": minimum_drift_length_m,
        "length_bounds_m": resolved_bounds, "length_bounds_source": "MBA proposed defaults" if length_bounds_m is None else "caller",
        "decoder": "five_stick_coordinates_six_weighted_lengths", "weights": _WEIGHTS.tolist(),
        "optimizer": {"name": "particle_swarm_optimize", "seed": seed,
                      "initial_velocity_fraction": .10, "selection": "first_objective"},
        "opposite_signs_required": True, "implementation": _implementation_identity()}
    identity = _digest(contract)
    verification_contract = {"geometry_candidates": geometry_candidates, "scan_resolutions": list(resolutions),
        "local_starts": local_starts, "local_max_evaluations": local_max_evaluations,
        "candidates_to_return": candidates_to_return, "profile_step_m": profile_step_m}
    progress_callback, cancellation_check = tme_unit._managed_callbacks(progress_callback, cancellation_check,
        title="TME geometry and strength search", event_key="tme-geometry-search")
    trace_limits = tme_unit._trace_limits(trace_margin_x, trace_margin_y)
    _publish_progress(progress_callback, "resolved_unit_constraints", {"trace_limits": trace_limits,
        "maximum_beta_m": maximum_beta_m, "expected_length_m": expected_length_m,
        "length_bounds_m": resolved_bounds})
    cancellation_check()
    state, metadata = None, {"evidence_root": str(evidence_root) if evidence_root is not None else None}
    if resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("contract_sha256") != identity or state["contract"].get("campaign_identity") != identity:
            raise ValueError("Resume physics contract changed; retain this history and start a separate campaign")
        finished = metadata.get("finished_result")
        if (finished and metadata.get("verification_contract") == verification_contract
                and (generations is None or generations <= state["completed_generation"])):
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata.get("finished_swarm_sha256"):
                raise ValueError("Completed optimizer bytes changed after final verification")
            source = _checkpoint_source(finished, metadata.get("evidence_root") or evidence_root)
            restored = read_numerical_checkpoint(finished, output_directory=evidence_root,
                                                source_output_directory=source)["result"]
            _restore_candidates(restored["candidates"], evidence_root, source_output_directory=source)
            for study in restored["geometry_finalists"]:
                _restore_candidates(study["verification"]["candidates"], evidence_root, source_output_directory=source)
            retained_budget = resolve_population_budget(7, population_size, generations,
                resumed_population_size=state["contract"]["population_size"], completed_generation=state["completed_generation"],
                saved_generation_target=state.get("budget", {}).get("effective", {}).get("generations"),
                checkpoint_complete=bool(state.get("complete")))
            if source is not None and Path(source).resolve() != Path(evidence_root).resolve():
                finished = _tagged_checkpoint("tme_geometry_search", restored, evidence_root, checks_passed=bool(restored["passed"]))
            return {**restored, "budget": retained_budget, "optimizer": {**restored["optimizer"], "budget": retained_budget},
                    "resumed_completed_result": True, "numerical_checkpoint": finished}
    budget = resolve_population_budget(7, population_size, generations,
        resumed_population_size=state["contract"]["population_size"] if state else None,
        completed_generation=state["completed_generation"] if state else 0,
        saved_generation_target=state.get("budget", {}).get("effective", {}).get("generations") if state else None,
        checkpoint_complete=bool(state.get("complete")) if state else False)
    size = budget["effective"]["population_size"]
    evaluate = _Evaluator(decoder, bend, energy_gev, reference_repetitions, limits, maximum_beta_m, trace_margin_x, trace_margin_y)
    bank = metadata.get("initial_bank", [])
    screening = metadata.get("initialization", {})
    if not resume:
        rng = np.random.default_rng(seed)
        trial_limit = max_initialization_evaluations or max(10000, size * 200)
        seen, attempted, broad = set(), 0, 0
        while len(bank) < size and attempted < trial_limit:
            if cancellation_check:
                cancellation_check()
            if bank and rng.random() < .5:
                parent = np.asarray(bank[int(rng.integers(len(bank)))]["position"])
                scale = np.exp(rng.uniform(math.log(.005), math.log(.1)))
                point = np.clip(parent + rng.normal(size=7) * scale * np.asarray([1.] * 5 + [2*cap] * 2),
                                [0.] * 5 + [-cap] * 2, [1.] * 5 + [cap] * 2).tolist()
            else:
                broad += 1
                sign = 1 if broad % 2 else -1
                geometry = initial if initial is not None and attempted < 64 else decoder.random(rng)
                point = [*geometry, sign * cap * rng.uniform(1e-12, 1), -sign * cap * rng.uniform(1e-12, 1)]
            if tuple(point) in seen:
                continue
            seen.add(tuple(point))
            measured = evaluate(point)
            attempted += 1
            if measured["feasible"]:
                bank.append(measured)
            if attempted % 1000 == 0 or len(bank) == size:
                _publish_progress(progress_callback, "feasible_population_screening", {
                    "attempted": attempted, "feasible_particles": len(bank), "required_particles": size})
        screening = {"attempted": attempted, "maximum_evaluations": trial_limit, "broad_proposals": broad,
            "feasible_particles": len(bank), "required_particles": size, "all_feasible": len(bank) == size,
            "positive_near_particles": sum(r["position"][-2] > 0 for r in bank), "production_generations": 0}
        initialization_checkpoint = None
        if storage:
            initialization_checkpoint = _tagged_checkpoint("tme_geometry_initialization", {"contract": contract, "budget": budget,
                "trace_limits": trace_limits, "initialization": screening, "feasible_bank": bank}, evidence_root,
                checks_passed=len(bank) == size)
        if len(bank) < size:
            return {"passed": False, "candidates": [], "initialization": screening, "budget": budget,
                    "optimizer_started": False, "complete_mba_accepted": False,
                    "contract": contract, "contract_sha256": identity, "trace_limits": trace_limits,
                    "numerical_checkpoint": initialization_checkpoint,
                    "limitation": "The bounded native screening did not fill an all-feasible population; this is not proof of infeasibility."}
        metadata = {"contract": contract, "contract_sha256": identity, "initial_bank": bank, "initialization": screening,
                    "evidence_root": str(evidence_root) if evidence_root is not None else None}
        if sidecar:
            atomic_write_text(sidecar, json.dumps(metadata, sort_keys=True, allow_nan=False))
    optimizer = api.particle_swarm_optimize(evaluate, bounds=[[0., 1.]] * 5 + [[-cap, cap]] * 2,
        objective_names=["emittance_pm"], variable_names=[f"length_simplex_{i+1}" for i in range(5)] + ["K1_near", "K1_far"],
        initial_position=bank[0]["position"], seed_positions=[r["position"] for r in bank[1:]],
        population_size=population_size, generations=generations, seed=seed, selection="first_objective",
        initial_velocity_fraction=.10,
        progress_callback=progress_callback, cancellation_check=cancellation_check, checkpoint_path=checkpoint,
        campaign_identity=identity, resume=resume, cache_evaluations=False)
    optimizer = json_safe(optimizer)
    pool = [*bank, *evaluate.cache.values()]
    pool += [{**r["metrics"], "objectives": r["objectives"], "position": r["position"], "feasible": r["feasible"]}
             for r in [optimizer["best_candidate"], *optimizer["population"]]]
    finalists = []
    for row in sorted((r for r in pool if r["feasible"]), key=lambda r: r["objectives"]["emittance_pm"]):
        if all(np.linalg.norm(np.asarray([row["geometry"][k] - old["geometry"][k] for k in _KEYS])
                              * _WEIGHTS / expected_length_m) >= .01 for old in finalists):
            finalists.append(row)
        if len(finalists) >= geometry_candidates:
            break
    verification_key, campaign = None, {}
    if sidecar:
        verification_key = _digest({"verification": verification_contract,
            "swarm_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        campaigns = metadata.setdefault("verification_campaigns", {})
        campaign = campaigns.setdefault(verification_key, {"finalists": finalists, "completed_studies": []})
        finalists = campaign["finalists"]
        atomic_write_text(sidecar, json.dumps(metadata, sort_keys=True, allow_nan=False))
    verified, studies = [], []
    for descriptor in campaign.get("completed_studies", []):
        source = _checkpoint_source(descriptor, metadata.get("evidence_root") or evidence_root)
        study = read_numerical_checkpoint(descriptor, output_directory=evidence_root,
                                         source_output_directory=source)["result"]
        if study["geometry"] != finalists[len(studies)]["geometry"] or study["outer_position"] != finalists[len(studies)]["position"]:
            raise ValueError("Saved finalist verification does not match the ordered campaign")
        _restore_candidates(study["verification"]["candidates"], evidence_root, source_output_directory=source)
        studies.append(study)
        verified.extend(study["verification"]["candidates"])
    for finalist_index, finalist in enumerate(finalists, 1):
        if finalist_index <= len(studies):
            continue
        def finalist_progress(update):
            _publish_progress(progress_callback, update["phase"], {**update, "geometry_finalist_index": finalist_index,
                                                                 "geometry_finalist_count": len(finalists)})
        finalist_progress._managed_numerical_observer = getattr(progress_callback, "_managed_numerical_observer", False)
        finalist_progress({"phase": "verify_geometry_finalist", "outer_emittance_pm": finalist["objectives"]["emittance_pm"]})
        geometry = dict(finalist["geometry"])
        actual_bend = {**copy.deepcopy(dict(bend)), "length_m": geometry.pop("bend_length_m")}
        study = tme_unit.design_tme_unit(energy_gev=energy_gev, bend=actual_bend, **geometry,
            expected_length_m=expected_length_m, reference_repetitions=reference_repetitions, limits=limits,
            maximum_beta_m=maximum_beta_m, trace_margin_x=trace_margin_x, trace_margin_y=trace_margin_y,
            minimum_magnet_length_m=minimum_magnet_length_m, minimum_drift_length_m=minimum_drift_length_m,
            initial_strengths=[finalist["position"][-2:]], scan_resolutions=scan_resolutions, local_starts=local_starts,
            local_max_evaluations=local_max_evaluations, candidates_to_return=candidates_to_return,
            profile_step_m=profile_step_m, output_directory=storage, progress_callback=finalist_progress,
            cancellation_check=cancellation_check)
        studies.append({"geometry": finalist["geometry"], "outer_position": finalist["position"],
            "outer_emittance_pm": finalist["objectives"]["emittance_pm"], "verification": study})
        verified.extend(study["candidates"])
        if storage and verification_key is not None:
            campaign["completed_studies"].append(_tagged_checkpoint("tme_geometry_finalist",
                studies[-1], evidence_root, checks_passed=bool(study["candidates"])))
            atomic_write_text(sidecar, json.dumps(metadata, sort_keys=True, allow_nan=False))
    result = {"method": "joint_tme_geometry_strength_pso_then_native_rescan", "engine": "JuTrack",
        "passed": bool(verified), "candidates": sorted(verified, key=lambda c: c["emittance_pm"])[:candidates_to_return],
        "contract": contract, "contract_sha256": identity, "initialization": screening, "budget": budget,
        "trace_limits": trace_limits,
        "provided_constraints": {key: contract[key] for key in ("energy_gev", "expected_length_m", "reference_repetitions",
            "maximum_beta_m", "trace_margin_x", "trace_margin_y", "minimum_magnet_length_m", "minimum_drift_length_m", "limits")},
        "optimizer": optimizer, "optimizer_started": True, "geometry_finalists": studies,
        "complete_mba_accepted": False, "optimality_demonstrated": False,
        "limitations": ["No fixed-geometry grid is nested inside a PSO evaluation.",
                        "Geometry finalists differ by at least 1% in the normalized weighted-length vector.",
                        "Verified units are seeds, not an accepted complete MBA or a global-optimality certificate."]}
    if storage:
        result["numerical_checkpoint"] = _tagged_checkpoint("tme_geometry_search", result, evidence_root,
                                                           checks_passed=bool(verified))
        if sidecar:
            metadata.update(finished_result=result["numerical_checkpoint"],
                verification_contract=verification_contract,
                finished_swarm_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            atomic_write_text(sidecar, json.dumps(metadata, sort_keys=True, allow_nan=False))
    return result
