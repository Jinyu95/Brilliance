"""Construct new outer sections around explicitly retained core connectivity.

No solved lattice, facility geometry or conventional seed-generation dependency.
One-pass matching is deliberately separate from complete-periodic-cell acceptance.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

from brilliance.lattice_fingerprint import cell_fingerprint

from .hybrid_regions import hybrid_mba_regions
from .hybrid_shape import _rows, hybrid_mba_bend_roles


def _physical(row, *, allow_k1=False):
    """Ignore only known labels, never arbitrary magnetic fields or map options."""
    result = deepcopy(row)
    labels = {"family", "source_element", "source_period", "source_element_name",
              "mirror_source_element", "design_role"}
    result["fields"] = {k: v for k, v in result["fields"].items() if k not in labels}
    if allow_k1 and result["kind"] == "KQUAD":
        result["k1_m2"] = None
    return result


class HybridMBAOuterReplacement:
    """Retain M-4-bend core geometry; replace BOTH two-bend outer sections.

    Select full optical core boundaries (including its quadrupoles/drifts), not
    just B3 entrance/B(M-2) exit. Boundary names are explicit, never inferred from
    Uxxx labels. Input must already be globally mirror symmetric.
    """
    def __init__(self, elements: Sequence[Any], *, core_start_name: str, core_end_name: str):
        self.source = _rows(elements)
        inventory = hybrid_mba_regions(self.source)
        if not inventory["mirror_passed"]:
            raise ValueError("Source must have complete-cell mirror symmetry")
        names = [row["name"] for row in self.source]
        self.start, self.end = names.index(core_start_name), names.index(core_end_name)
        if self.start > self.end or self.start != len(names) - 1 - self.end:
            raise ValueError("Core boundaries must be ordered complete-cell mirror partners")
        roles = inventory["bend_roles"]
        inside = [b["bend"] for b in roles["bends"] if self.start <= b["element_index"] <= self.end]
        if inside != roles["regions"]["core"]["bends"]:
            raise ValueError("Core must contain exactly the middle M-4 physical bends")
        self.core = deepcopy(self.source[self.start:self.end + 1])
        self.bend_count = roles["bend_count"]
        self.cell_length = sum(r["length_m"] for r in self.source)
        self.cell_angle = sum(r.get("angle_rad") or 0 for r in self.source)
        self.outer_length = (self.cell_length - sum(r["length_m"] for r in self.core)) / 2
        self.outer_angle = (self.cell_angle - sum(r.get("angle_rad") or 0 for r in self.core)) / 2
        self.core_fingerprint = cell_fingerprint(self.core)
        self.minimum_half_straight = self.source[-1]["length_m"] if self.source[-1]["kind"] == "DRIFT" else 0.
        self.core_optics_reference = None

    def measure_reference(self, candidate: Any, *, max_step_m: float = .02) -> dict[str, Any]:
        """Bind measured periodic core optics from THIS contract's source lattice."""
        from brilliance import accelerator_api as api

        from .hybrid_optics import measure_core_optics
        if cell_fingerprint(api.one_period_elements(candidate)) != cell_fingerprint(self.source):
            raise ValueError("Optical reference must be the replacement contract's source cell")
        self.core_optics_reference = measure_core_optics(candidate,
            core_start_name=self.core[0]["name"], core_end_name=self.core[-1]["name"], max_step_m=max_step_m)
        return deepcopy(self.core_optics_reference)

    def describe(self) -> dict[str, Any]:
        return {"route": "outer_DBA_replacement", "core_start_name": self.core[0]["name"],
            "core_end_name": self.core[-1]["name"], "core_element_names": [r["name"] for r in self.core],
            "core_fingerprint": self.core_fingerprint, "preserved_core_bends": self.bend_count - 4,
            "outer_length_m": self.outer_length, "outer_angle_rad": self.outer_angle,
            "expected_cell_length_m": self.cell_length, "expected_cell_angle_rad": self.cell_angle,
            "minimum_half_straight_m": self.minimum_half_straight,
            "orientation": "right section runs core exit -> inner outer bend -> bump -> terminal bend -> ID centre",
            "stage": "bump_feasibility", "core_optics_reference_required": False,
            "default_optical_policy": "free_periodic_optics",
            "core_quadrupole_policy": "global_mirror_pairs_may_rematch",
            "deferred": ["emittance_recovery", "chromatic_correction"]}

    def assemble(self, right_outward: Sequence[Any], *, core_elements=None) -> list[dict[str, Any]]:
        from brilliance import accelerator_api as api
        right = _rows(right_outward)
        bends = [r for r in right if r.get("angle_rad")]
        if len(bends) != 2 or any(r["kind"] != "SBEND" or r["angle_rad"] <= 0 for r in bends):
            raise ValueError("Replacement must contain exactly two positive homogeneous sector bends")
        if not math.isclose(sum(r["length_m"] for r in right), self.outer_length, abs_tol=1e-10, rel_tol=0):
            raise ValueError("Replacement does not fit the complete-cell outer length budget")
        if not math.isclose(sum(r.get("angle_rad") or 0 for r in right), self.outer_angle, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Replacement does not preserve the outer bend-angle budget")
        if right[-1]["kind"] != "DRIFT" or right[-1]["length_m"] < self.minimum_half_straight - 1e-12:
            raise ValueError("Replacement must retain the required ID half-straight")
        core = deepcopy(self.core) if core_elements is None else _rows(core_elements)
        if not self.core_preserved(core, allow_quadrupole_rematch=True):
            raise ValueError("Replacement changes preserved physical core geometry or nonquadrupole fields")
        rows = _rows(api.mirror_elements(right, name_prefix="DBA_L:") + core + right)
        hybrid_mba_bend_roles(rows)  # rejects combined-function fields/reverse bends
        if not hybrid_mba_regions(rows)["mirror_passed"]:
            raise ValueError("Replacement broke global reflection")
        return rows

    def core_preserved(self, elements: Sequence[Any], *, allow_quadrupole_rematch=False) -> bool:
        rows = _rows(elements)
        names = [r["name"] for r in rows]
        if self.core[0]["name"] not in names:
            return False
        start = names.index(self.core[0]["name"])
        actual = rows[start:start + len(self.core)]
        return (len(actual) == len(self.core) and
                [_physical(r, allow_k1=allow_quadrupole_rematch) for r in actual] ==
                [_physical(r, allow_k1=allow_quadrupole_rematch) for r in self.core])

    def core_from_half(self, matched_half: Sequence[Any]) -> list[dict[str, Any]]:
        """Export half-line K1 to GLOBAL mirror partners, restoring the unsplit centre.

        No within-TME-unit family ties. Reject topology/geometry edits and keep
        the original unsplit physical core and source object untouched.
        """
        half, expected = _rows(matched_half), self.right_core_half()
        if [_physical(r, allow_k1=True) for r in half] != [_physical(r, allow_k1=True) for r in expected]:
            raise ValueError("Matched core half may change only quadrupole K1, not topology or geometry")
        core = deepcopy(self.core)
        offset = len(core)//2
        for j, row in enumerate(half):
            if row["kind"] == "KQUAD":
                i = offset+j
                core[i]["k1_m2"] = core[len(core)-1-i]["k1_m2"] = row["k1_m2"]
        return core

    def right_core_half(self) -> list[dict[str, Any]]:
        """Diagnostic centre-to-core-exit line; never changes the archived full core.

        For an odd element count, bisect the central homogeneous element. Bend
        edges are unsupported here: splitting edge focusing needs a map-aware tool.
        """
        mid = len(self.core)//2
        if len(self.core)%2 == 0:
            return deepcopy(self.core[mid:])
        half = deepcopy(self.core[mid:])
        center = half[0]
        if center["kind"] not in {"DRIFT","KQUAD","SBEND","MARKER"}:
            raise ValueError("Diagnostic core half requires a homogeneous central element")
        if center["kind"] == "SBEND" and any(float(v or 0) for k,v in center["fields"].items()
                if k.casefold() in {"e1","e2","e1_rad","e2_rad","entranceangle","exitangle"}):
            raise ValueError("Core-half splitting with bend edge focusing is not supported")
        center["length_m"] /= 2
        if center.get("angle_rad"):
            center["angle_rad"] /= 2
        return half

    def evaluate(self, candidate: Any, *, thresholds: Mapping[str, float],
            limits: Mapping[str, float], max_step_m: float = .02,
            minimum_magnet_length_m: float = 0., minimum_drift_length_m: float = .01,
            core_optics_tolerances: Mapping[str, float] | None = None,
            preserve_core_optics: bool = False,
            stage: str = "bump_feasibility") -> dict[str, Any]:
        """Gate the resulting periodic shape; source-optics preservation is opt-in."""
        from brilliance import accelerator_api as api

        from .hybrid_shape import evaluate_hybrid_mba_shape
        rows = api.one_period_elements(candidate)
        if not self.core_preserved(rows, allow_quadrupole_rematch=True):
            raise ValueError("Candidate changes the preserved physical core")
        if type(preserve_core_optics) is not bool:
            raise ValueError("preserve_core_optics must be an explicit boolean")
        if preserve_core_optics and self.core_optics_reference is None:
            raise ValueError("Call measure_reference(source_candidate) before requesting optical preservation")
        if core_optics_tolerances is not None and not preserve_core_optics:
            raise ValueError("core_optics_tolerances requires preserve_core_optics=True")
        if self.core_optics_reference is not None:
            if not math.isclose(candidate.energy_gev, self.core_optics_reference["energy_gev"], rel_tol=1e-12):
                raise ValueError("Candidate and source optical reference have different energy")
            # Identical sampling anchors permit diagnostic comparison as well.
            max_step_m = self.core_optics_reference["max_step_m"]
        report = evaluate_hybrid_mba_shape(candidate,stage=stage,thresholds=thresholds,
            expected_cell_length_m=self.cell_length,expected_cell_angle_rad=self.cell_angle,
            limits=limits,max_step_m=max_step_m,minimum_magnet_length_m=minimum_magnet_length_m,
            minimum_drift_length_m=minimum_drift_length_m)
        report["replacement_contract"] = self.describe()
        report["core_quadrupoles_rematched"] = not self.core_preserved(rows)
        report["preserve_core_optics"] = preserve_core_optics
        report["replacement_contract"]["core_optics_reference_required"] = preserve_core_optics
        from .hybrid_optics import compare_core_optics
        if self.core_optics_reference is not None:
            report["core_optics_reference"] = deepcopy(self.core_optics_reference)
            if report.get("stable"):
                report["core_optics"] = compare_core_optics(self.core_optics_reference, report["profile"],
                    tolerances=core_optics_tolerances)
        if preserve_core_optics:
            report["checks"]["core_optics_preserved"] = report.get("core_optics", {}).get("passed", False)
        report["passed"] = all(report["checks"].values())
        report["design_status"] = ("HYBRID_SHAPE_VALIDATED" if stage == "full_hybrid" else "USABLE_BUMPS_FEASIBLE") if report["passed"] else "HYBRID_CANDIDATE_REJECTED"
        return report


def build_outer_dba_section(contract: HybridMBAOuterReplacement, *, suppressor_length_m: float,
        bump_length_m: float, quadrupole_length_m: float, bend_lengths_m: Sequence[float],
        inner_bend_fraction: float, half_straight_m: float, k1_values: Sequence[float],
        minimum_drift_length_m: float = .01,
        quadrupoles_per_region: Sequence[int] = (2, 2, 2),
        quadrupole_lengths_m: Sequence[float] | None = None) -> list[dict[str, Any]]:
    """Unsolved outer connectivity; explicit suppressor/bump/terminal quad counts.

    Core -> (2 suppressor quads) -> Binner -> (2 bump quads + station) ->
    Bterminal -> (2 ID-matching quads) -> half straight. Lengths include quads
    in each named region, exclude bends. Remaining terminal length is computed
    from the exact budget, never obtained by shortening the ID straight.
    """
    from brilliance import accelerator_api as api
    q, s, b, straight, minimum = map(float, (quadrupole_length_m, suppressor_length_m,
        bump_length_m, half_straight_m, minimum_drift_length_m))
    lens, ks = list(map(float, bend_lengths_m)), list(map(float, k1_values))
    counts = list(quadrupoles_per_region)
    if len(counts) != 3 or any(type(n) is not int or n < 1 for n in counts) or counts[1] % 2:
        raise ValueError("Use three positive quadrupole counts; the bump count must be even")
    fraction = float(inner_bend_fraction)
    if len(lens) != 2 or len(ks) != sum(counts) or not all(math.isfinite(v) for v in [q,s,b,straight,minimum,fraction,*lens,*ks]):
        raise ValueError("Supply two bend lengths, one K1 per quadrupole and finite geometry")
    if min(q,minimum,*lens) <= 0 or not 0 < fraction < 1 or straight < contract.minimum_half_straight:
        raise ValueError("Invalid magnet lengths, angle fraction or ID straight")
    terminal = contract.outer_length - s - b - sum(lens) - straight
    ns, nb, nt = counts
    qlens = [q]*sum(counts) if quadrupole_lengths_m is None else list(map(float,quadrupole_lengths_m))
    if len(qlens) != sum(counts) or any(not math.isfinite(v) or v <= 0 for v in qlens):
        raise ValueError("quadrupole_lengths_m needs one finite positive length per outer quadrupole")
    gaps = [(s - sum(qlens[:ns]))/(ns+1), (b - sum(qlens[ns:ns+nb]))/(nb+2),
            (terminal - sum(qlens[ns+nb:]))/nt]
    if min(gaps) < minimum:
        raise ValueError("Outer DBA geometry has insufficient physical drift budget")
    def d(name, length):
        return api.drift("DBA_R:"+name, length)
    def quad(i, role):
        return api.quadrupole(f"DBA_R:Q{i+1}", qlens[i], ks[i], family=f"DBA:Q{i+1}", design_role=role)
    a, c, e = gaps
    result = [d("S0", a)]
    for i in range(ns):
        result += [quad(i, "dispersion_suppressor"), d(f"S{i+1}", a)]
    result += [api.sector_bend("DBA_R:INNER", lens[0], contract.outer_angle*fraction), d("B0", c)]
    gap = 1
    for i in range(nb):
        result += [quad(ns+i, "bump_generator"), d(f"B{gap}", c)]
        gap += 1
        if i == nb//2-1:
            result += [api.marker("DBA_R:BUMP"), d(f"B{gap}", c)]
            gap += 1
    result += [api.sector_bend("DBA_R:TERMINAL", lens[1], contract.outer_angle*(1-fraction))]
    for i in range(nt):
        result += [d(f"T{i}", e), quad(ns+nb+i, "terminal_matching")]
    return result + [d("ID", straight)]


def match_outer_dba_section(elements: Sequence[Any], *, energy_gev: float,
        input_twiss: Mapping[str, float], bump_station_name: str, target_bump_m: float,
        starts: Sequence[Sequence[float]], max_abs_k1_m2: float, max_nfev: int = 300,
        id_dispersion_tolerance_m: float = 1e-4, slope_tolerance: float = 1e-4,
        alpha_tolerance: float = 1e-4, bump_tolerance_m: float = 1e-3,
        core_prefix: Sequence[Any] | None = None,
        center_optics_bounds: Mapping[str, Sequence[float]] | None = None,
        bump_station_fraction: float = 1.0,
        initialize_achromat: bool = False,
        rematch_core_quadrupoles: bool = False,
        max_core_dispersion_m: float | None = None,
        max_beta_m: float | None = None) -> dict[str, Any]:
    """Match a NEW outward line, even when the unclosed periodic cell is unstable.

    All K1 belong exclusively to this matcher. Supply core_prefix (centre to
    core exit) and center_optics_bounds to solve new symmetry-plane beta_x,
    beta_y and Dx alongside the outer strengths. Alpha_x/alpha_y/Dpx are zero
    at the reflection plane, NOT targets copied from the conventional lattice.
    Without a prefix this is a fixed-input one-pass line solve, still available
    for explicitly requested optical preservation. At least six independent
    quadrupoles; no tune, core H, I5 or emittance
    residual. A stationary station is NOT proof of an interior maximum: assemble
    and run the dense complete-cell bump gate before reporting stage success.

    rematch_core_quadrupoles=True adds every half-core KQUAD as an independent
    coordinate; export via replacement.core_from_half(result['core_prefix']).
    Coordinates are outer K1, half-core K1, then centre betax/betay/dx. Legacy
    default False keeps archived fixed-core programs reproducible. Explicit
    envelope bounds add non-compensating inequality residuals, not source-optics
    targets. Dense periodic checks remain authoritative.
    """
    import numpy as np
    from scipy.optimize import least_squares

    from brilliance import accelerator_api as api
    rows = _rows(elements)
    if (core_prefix is None) != (center_optics_bounds is None):
        raise ValueError("Supply core_prefix and center_optics_bounds together to solve free periodic optics")
    free_optics = core_prefix is not None
    prefix = _rows(core_prefix) if free_optics else []
    optical_names = ("betax", "betay", "dx") if free_optics else ()
    optical_bounds = []
    if free_optics:
        if not prefix or set(center_optics_bounds) != set(optical_names):
            raise ValueError("Nonempty core_prefix and center_optics_bounds for betax, betay, dx are required")
        for key in optical_names:
            pair = list(map(float, center_optics_bounds[key]))
            if (len(pair) != 2 or not all(math.isfinite(v) for v in pair) or pair[0] >= pair[1]
                    or (key != "dx" and pair[0] <= 0)):
                raise ValueError(f"Invalid centre optics bounds for {key}")
            optical_bounds.append(pair)
    if type(rematch_core_quadrupoles) is not bool or (rematch_core_quadrupoles and not free_optics):
        raise ValueError("Core quadrupole rematching requires a core_prefix and free centre optics")
    if max_core_dispersion_m is not None and not free_optics:
        raise ValueError("Core dispersion envelope requires core_prefix")
    for label, value in (("max_core_dispersion_m",max_core_dispersion_m),("max_beta_m",max_beta_m)):
        if value is not None and (not math.isfinite(float(value)) or value <= 0):
            raise ValueError(f"{label} must be positive and finite")
    outer_quads = [r for r in rows if r["kind"] == "KQUAD"]
    core_quads = [r for r in prefix if r["kind"] == "KQUAD"] if rematch_core_quadrupoles else []
    quads = outer_quads + core_quads
    nq = len(quads)
    if nq < 6:
        raise ValueError("This six-residual matcher needs at least six independently owned quadrupoles")
    scales = [alpha_tolerance,alpha_tolerance,id_dispersion_tolerance_m,slope_tolerance,bump_tolerance_m,slope_tolerance]
    if not all(math.isfinite(float(v)) and float(v)>0 for v in [*scales,target_bump_m,max_abs_k1_m2]):
        raise ValueError("Matching tolerances, target and K1 bound must be positive and finite")
    if type(max_nfev) is not int or max_nfev < 1 or not len(starts):
        raise ValueError("Declare at least one matching start and positive max_nfev")
    # Explicit full coordinate vectors avoid ambiguous legacy outer+3 starts.
    for pos in starts:
        if len(pos) not in {nq, nq+len(optical_names)} or any(not math.isfinite(float(k)) or abs(k)>max_abs_k1_m2 for k in pos[:nq]):
            raise ValueError(f"Matching starts need {nq} in-bound K1 values (outer then half-core), optionally followed by {len(optical_names)} centre coordinates")
        if len(pos) > nq and any(not math.isfinite(float(v)) or not lo <= float(v) <= hi
                               for v, (lo, hi) in zip(pos[nq:], optical_bounds)):
            raise ValueError("Centre optics in a matching start must be finite and in bounds")
    station = [r["name"] for r in rows].index(bump_station_name) + 1  # JuTrack refpts are 1-based
    bend_indices = [i+1 for i,r in enumerate(rows) if r.get("angle_rad")]
    if len(bend_indices)!=2 or not bend_indices[0]<station<bend_indices[1]:
        raise ValueError("Bump station must lie strictly between the two replacement bends")
    fraction = float(bump_station_fraction)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("bump_station_fraction must lie in (0,1]")
    station_row = rows[station-1]
    in_drift = station_row["kind"] == "DRIFT" or (station_row["kind"] == "MARKER"
        and station >= 2 and station < len(rows) and rows[station-2]["kind"] == rows[station]["kind"] == "DRIFT")
    if not in_drift or (fraction < 1 and station_row["kind"] != "DRIFT"):
        raise ValueError("Select a physical DRIFT or marker between drifts for the chromatic plateau, not a quadrupole/bend peak")
    keys = ("betax","betay","alphax","alphay","dx","dpx")
    tin = {key: float(input_twiss[key]) for key in keys}
    if not all(math.isfinite(v) for v in tin.values()) or min(tin["betax"],tin["betay"])<=0:
        raise ValueError("Finite input Twiss with positive betas is required")
    if free_optics:
        if any(abs(tin[key]) > 1e-12 for key in ("alphax", "alphay", "dpx")):
            raise ValueError("Free centre optics starts at a reflection plane: alphax, alphay and dpx must be zero")
        if any(not lo <= tin[key] <= hi for key, (lo, hi) in zip(optical_names, optical_bounds)):
            raise ValueError("Initial centre optics must lie inside center_optics_bounds")
    line_rows = prefix + rows
    line = api.lattice(title="Outer DBA boundary-value problem", energy_gev=energy_gev,
                       periodicity=1, elements=line_rows, source="expert outer DBA matching")
    editor = api.build_jutrack_editor(line)
    jt = editor.jt
    partial_editor = None
    if fraction<1:
        partial_row = deepcopy(station_row)
        partial_row["length_m"] *= fraction
        partial_editor = api.build_jutrack_editor(api.lattice(title="Bump interior diagnostic",
            energy_gev=energy_gev,periodicity=1,elements=[partial_row],source="outer DBA prefix map"))
    def initial_values(pos):
        return {**tin, **dict(zip(optical_names, map(float, pos[nq:])))}
    knobs = api.compile_lattice_parameterization(line_rows, [{"name":r["name"], "element_names":[r["name"]],
        "field":"k1_m2", "bounds":[-max_abs_k1_m2,max_abs_k1_m2], "role":"in_objective_matching"} for r in quads])
    count, best, trials = 0, None, []
    envelopes = max_core_dispersion_m is not None or max_beta_m is not None
    core_bend_end = max((i+1 for i,r in enumerate(prefix) if r.get("angle_rad")), default=len(prefix))
    residual_names = ["ID_alpha_x","ID_alpha_y","ID_Dx","ID_Dpx","bump_Dx","bump_Dpx"]
    if max_core_dispersion_m is not None:
        residual_names.append("core_dispersion_envelope")
    if max_beta_m is not None:
        residual_names.append("beta_envelope")
    def acceptance_residuals(values):
        return np.r_[values[:6]*np.asarray([1.,1.,.01,.01,.01,.01])/np.asarray(scales),
                     values[6:]/1e-4]
    def residual(pos):
        nonlocal count
        count += 1
        editor.apply_parameterization(knobs,pos[:nq])
        initial = jt.EdwardsTengTwiss(**initial_values(pos))
        point_index = len(prefix)+station-(1 if partial_editor is not None else 0)
        refpts = list(range(1,len(line_rows)+1)) if envelopes else [point_index,len(line_rows)]
        values = jt.twissline(initial,editor.lattice,dp=0.,order=0,refpts=refpts)
        peak, end = (values[point_index-1],values[-1]) if envelopes else (values[0],values[1])
        if partial_editor is not None:
            partial_editor.sync_rows([partial_row])
            peak = jt.twissline(peak,partial_editor.lattice,dp=0.,order=0)
        raw = np.asarray([float(end.alphax),float(end.alphay),float(end.dx),float(end.dpx),
                          float(peak.dx)-target_bump_m,float(peak.dpx)])
        if not np.all(np.isfinite(raw)):
            raise ValueError("Nonfinite one-pass optics; inspect geometry/input, not a hidden infeasible particle")
        # Physical scales condition the solve; acceptance tolerances are checked
        # independently afterwards. Tiny alpha tolerances must not drown out Dx.
        extra = []
        if max_core_dispersion_m is not None:
            dmax = max(abs(float(p.dx)) for p in [initial,*values[:core_bend_end]])
            extra.append(max(0.,dmax/max_core_dispersion_m-1))
        if max_beta_m is not None:
            bmax = max(max(float(p.betax),float(p.betay)) for p in [initial,*values])
            extra.append(max(0.,bmax/max_beta_m-1))
        result = np.r_[raw / np.asarray([1.,1.,.01,.01,.01,.01]),extra]
        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite one-pass envelope")
        return result
    lower = [-max_abs_k1_m2]*nq + [pair[0] for pair in optical_bounds]
    upper = [max_abs_k1_m2]*nq + [pair[1] for pair in optical_bounds]
    for start in starts:
        position = list(start)
        if len(position) == nq:
            position += [tin[key] for key in optical_names]
        achromat = None
        if initialize_achromat:
            achromat = least_squares(lambda pos: residual(pos)[:4],position,
                                bounds=(lower,upper),max_nfev=max_nfev,
                                x_scale="jac",xtol=1e-10,ftol=1e-10,gtol=1e-10)
            position = achromat.x
        sol = least_squares(residual,position,bounds=(lower,upper),max_nfev=max_nfev,
                            x_scale="jac",xtol=1e-10,ftol=1e-10,gtol=1e-10)
        checked = acceptance_residuals(sol.fun)
        score = float(np.max(np.abs(checked)))
        trials.append({"max_scaled_residual":score,"nfev":sol.nfev,"achromat_nfev":achromat.nfev if achromat else 0,
                       "scaled_residuals": checked.tolist(),
                       "solver_success":bool(sol.success)})
        if best is None or score<best[0]:
            best = (score,sol.x,checked,sol.jac)
        if score<=1:
            break
    # Reapply/re-evaluate the winning start, not the last attempted mutable state.
    replay = residual(best[1])
    checked = acceptance_residuals(replay)
    score = float(np.max(np.abs(checked)))
    singular = np.linalg.svd(best[3], compute_uv=False)
    return {"matched":score<=1, "elements":knobs.apply(best[1][:nq])[len(prefix):], "position":best[1].tolist(),
        "core_prefix":knobs.apply(best[1][:nq])[:len(prefix)],
        "quadrupole_ownership":{"outer":[r['name'] for r in outer_quads],
            "core_half":[r['name'] for r in core_quads],"owner":"inner_matcher",
            "symmetry":"complete_cell_reflection"},
        "variable_names":[r["name"] for r in quads] + ["center:"+key for key in optical_names],
        "matched_input_twiss":initial_values(best[1]),
        "center_optics_bounds":dict(center_optics_bounds) if free_optics else None,
        "boundary_mode":"free_symmetry_plane_optics" if free_optics else "fixed_source_core_exit",
        "bump_station":{"name":bump_station_name,"fraction":fraction},
        "scaled_residuals":checked.tolist(), "residual_names":residual_names,
        "input_twiss":tin, "evaluations":count, "starts":trials,
        "diagnostics": {"jacobian_singular_values": singular.tolist(),
            "jacobian_rank": int(np.linalg.matrix_rank(best[3])),
            "saturated_strengths": [r["name"] for r,k in zip(quads,best[1]) if abs(k) >= .999*max_abs_k1_m2],
            "initialize_achromat": bool(initialize_achromat)},
        "stage":"outer_line_matched" if score<=1 else "outer_line_unmatched",
        "limitations":["One-pass match only. Validate global symmetry, magnet limits, complete-cell stability, usable drift bumps and damping before accepting bump feasibility. Source beta/dispersion preservation is not a default requirement."]}
