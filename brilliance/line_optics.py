"""Reusable JuTrack open-line transport; no periodic solve or hand-written maps."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .lattice import require_same_lattice_topology
from .lattice_optimization import JuTrackLatticeEditor, _is_jutrack_twiss_domain_error
from .optics_sampling import _physical_rows

_FIELDS = {"beta_x_m": "betax", "beta_y_m": "betay", "alpha_x": "alphax", "alpha_y": "alphay",
           "Dx_m": "dx", "Dpx": "dpx", "Dy_m": "dy", "Dpy": "dpy"}


class _InvalidLineOptics(ArithmeticError):
    pass


def _state(point: Any) -> dict[str, float]:
    values = {key: float(getattr(point, attribute)) for key, attribute in _FIELDS.items()}
    if (not all(math.isfinite(value) for value in values.values())
            or values["beta_x_m"] <= 0 or values["beta_y_m"] <= 0 or int(point.mode) == 0):
        raise _InvalidLineOptics("JuTrack returned nonfinite/nonpositive open-line optics or mode=0")
    return values


def _profile(point: Any, name: str, position: float, **extra: Any) -> dict[str, Any]:
    state = _state(point)
    return {"name": name, "s_m": position, "beta_x": state["beta_x_m"],
            "beta_y": state["beta_y_m"], "alpha_x": state["alpha_x"], "alpha_y": state["alpha_y"],
            "Dx": state["Dx_m"], "Dpx": state["Dpx"], "Dy": state["Dy_m"], "Dpy": state["Dpy"], **extra}


class JuTrackLineEvaluator:
    """Transport a complete incoming state through an uncoupled hard-edge line.

    All eight initial_twiss keys are required: beta_x_m, beta_y_m, alpha_x,
    alpha_y, Dx_m, Dpx, Dy_m, Dpy. Required machine bounds never come from a
    guessed match target. ``maximum_beta_m=None`` explicitly means no beta cap.
    Supported elements follow the native prefix sampler: DRIFT, KQUAD, QUAD,
    pure SBEND, KSEXT, KOCT and MARKER without nontrivial edges/transforms.

    Scalar proposals reuse one physical editor and private native prefixes.
    Every prefix starts from the original element entrance, never from the
    previous subdivision. Thus interior diagnostics do not change NumIntSteps
    of the physical line. Topology changes require a new evaluator.
    ``passed`` checks this line only, never periodic-cell or ring acceptance.
    """

    def __init__(self, elements: Sequence[Any], *, energy_gev: float,
                 initial_twiss: Mapping[str, float], maximum_beta_m: float | None,
                 limits: Mapping[str, float], expected_length_m: float, expected_angle_rad: float,
                 minimum_magnet_length_m: float = 0.0, minimum_drift_length_m: float = 0.0,
                 profile_step_m: float = 0.25):
        from . import accelerator_api as api

        if not isinstance(initial_twiss, Mapping) or set(initial_twiss) != set(_FIELDS):
            raise ValueError("initial_twiss requires exactly: " + ", ".join(_FIELDS))
        for key, value in initial_twiss.items():
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"initial_twiss.{key} must be finite")
        self.initial = {key: float(value) for key, value in initial_twiss.items()}
        if self.initial["beta_x_m"] <= 0 or self.initial["beta_y_m"] <= 0:
            raise ValueError("incoming beta functions must be positive")
        for key, value in (("profile_step_m", profile_step_m), ("maximum_beta_m", maximum_beta_m)):
            if value is not None and (isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{key} must be positive and finite")
        if profile_step_m is None or not isinstance(limits, Mapping):
            raise TypeError("profile_step_m and an explicit limits mapping are required")
        self.api, self.rows = api, _physical_rows(elements)
        self.energy_gev, self.step = float(energy_gev), float(profile_step_m)
        self.beta_limit = None if maximum_beta_m is None else float(maximum_beta_m)
        self.audit_options = dict(energy_gev=self.energy_gev, limits=dict(limits),
                                  expected_length_m=expected_length_m, expected_angle_rad=expected_angle_rad,
                                  minimum_magnet_length_m=minimum_magnet_length_m,
                                  minimum_drift_length_m=minimum_drift_length_m)
        api.audit_lattice_constraints(self.rows, **self.audit_options)  # validate configuration, no Julia
        self.editor = self.prefix_editor = None

    def _initialize(self, rows: Sequence[Mapping[str, Any]]) -> None:
        candidate = self.api.lattice(title="Private JuTrack transfer line", energy_gev=self.energy_gev,
                                     periodicity=1, elements=rows, source="brilliance.line_optics",
                                     metadata={"evaluation_scope": "transfer_line"})
        self.editor = self.api.build_jutrack_editor(candidate)
        self.jt = self.editor.jt
        copies = self.jt.Lattice([self.jt.deepcopy_element(element) for element in self.editor.lattice])
        self.prefix_editor = JuTrackLatticeEditor(self.jt, copies, rows)
        self.prefix_lines = [self.jt.Lattice([element]) for element in copies]

    def evaluate(self, elements: Sequence[Any] | None = None) -> dict[str, Any]:
        """Return finite line transport and sampled beta burden for one proposal.

        Engineering-infeasible rows return evaluable=False before any native
        call. Finite optics above the beta cap remain evaluable=True, passed=False
        so a matcher can navigate. API/runtime exceptions are not physics rejects.
        """
        rows = _physical_rows(self.rows if elements is None else elements)
        require_same_lattice_topology(self.rows, rows, operation="JuTrack line evaluator")
        audit = self.api.audit_lattice_constraints(rows, **self.audit_options)
        result = {"engine": "JuTrack", "scope": "uncoupled_hard_edge_transfer_line",
                  "evaluable": False, "passed": False, "endpoint": {}, "metrics": {}, "twiss_profile": [],
                  "geometry": {"length_m": audit["length_m"], "angle_rad": audit["angle_rad"]},
                  "engineering": audit, "checks": {"engineering": audit["passed"], "finite_optics": False,
                                                   "maximum_beta": False}, "diagnostics": list(audit["issues"])}
        if not audit["passed"]:
            return result
        if self.editor is None:
            self._initialize(rows)
        self.editor.sync_rows(rows)
        self.prefix_editor.sync_rows(rows)
        tin = self.jt.EdwardsTengTwiss(self.initial["beta_x_m"], self.initial["beta_y_m"],
              **{_FIELDS[key]: value for key, value in self.initial.items() if not key.startswith("beta_")})
        native_options = {"dp": 0.0, "order": 0, "E0": self.energy_gev * 1e9}
        try:
            points = list(self.jt.twissline(tin, self.editor.lattice,
                                           refpts=list(range(1, len(rows) + 1)), **native_options))
            if len(points) != len(rows):
                raise ValueError("JuTrack line profile must return every requested physical element exit")
            profile, entrance, position = [_profile(tin, "START", 0.0)], tin, 0.0
            for index, (row, endpoint) in enumerate(zip(rows, points)):
                expected = _state(endpoint)
                prefix, singleton = self.prefix_editor.lattice[index], self.prefix_lines[index]
                length, angle = float(row["length_m"]), row["angle_rad"]
                # A full private prefix must reproduce the original finite-step element.
                full = _state(self.jt.twissline(entrance, singleton, **native_options))
                if any(not math.isclose(full[key], expected[key], rel_tol=1e-7, abs_tol=1e-9) for key in _FIELDS):
                    raise ValueError("JuTrack private prefix did not reproduce the original element exit")
                count = max(1, math.ceil(length / self.step))
                fractions = sorted({0.5, *(part / count for part in range(1, count))}) if length else []
                try:
                    for fraction in fractions:
                        prefix.len = float(length * fraction)
                        if angle is not None:
                            prefix.angle = float(angle * fraction)
                        sampled = self.jt.twissline(entrance, singleton, **native_options)
                        profile.append(_profile(sampled, f"{row['name']}@{fraction:.6g}", position + length * fraction,
                                                source_element=row["name"], sample_fraction=fraction,
                                                physical_element_exit=False))
                finally:
                    prefix.len = length
                    if angle is not None:
                        prefix.angle = float(angle)
                position += length
                profile.append(_profile(endpoint, row["name"], position, source_element=row["name"],
                                        sample_fraction=1.0, physical_element_exit=True))
                entrance = endpoint
        except _InvalidLineOptics as exc:
            result["diagnostics"].append(str(exc))
            return result
        except Exception as exc:
            if not _is_jutrack_twiss_domain_error(exc):
                raise
            result["diagnostics"].append(str(exc))
            return result
        maximum = max(max(point["beta_x"], point["beta_y"]) for point in profile)
        beta_ok = self.beta_limit is None or maximum <= self.beta_limit
        final = _state(points[-1])
        result.update(evaluable=True, passed=beta_ok, endpoint=final,
                      metrics={**final, "maximum_beta_m": maximum}, twiss_profile=profile,
                      sampling={"method": "JuTrack_physical_element_prefixes", "profile_step_m": self.step,
                                "includes_midpoints": True, "maximum_is_sampled": True,
                                "physical_element_count": len(rows), "sample_count": len(profile)})
        result["checks"].update(finite_optics=True, maximum_beta=beta_ok)
        if not beta_ok:
            result["diagnostics"].append(f"sampled maximum beta {maximum:g} m exceeds limit {self.beta_limit:g} m")
        return result


def evaluate_jutrack_line(elements: Sequence[Any], *, energy_gev: float,
                         initial_twiss: Mapping[str, float], maximum_beta_m: float | None,
                         limits: Mapping[str, float], expected_length_m: float, expected_angle_rad: float,
                         minimum_magnet_length_m: float = 0.0, minimum_drift_length_m: float = 0.0,
                         profile_step_m: float = 0.25) -> dict[str, Any]:
    """One-shot native open-line transport; use JuTrackLineEvaluator in match callbacks."""
    return JuTrackLineEvaluator(elements, energy_gev=energy_gev, initial_twiss=initial_twiss,
        maximum_beta_m=maximum_beta_m, limits=limits, expected_length_m=expected_length_m,
        expected_angle_rad=expected_angle_rad, minimum_magnet_length_m=minimum_magnet_length_m,
        minimum_drift_length_m=minimum_drift_length_m, profile_step_m=profile_step_m).evaluate()
