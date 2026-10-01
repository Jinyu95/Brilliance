"""Independent, read-only import of named LOCO measurements and settings.

No control-system clients or reference LOCO implementation are imported. Units
below describe the supported legacy JSON schema, not arbitrary machine files.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def _load(path, *, mask_nonfinite=False):
    source = Path(path).resolve()
    raw = source.read_bytes()
    invalid = []

    def constant(value):
        if not mask_nonfinite:
            return _invalid_constant(value)
        invalid.append(value)
        return None

    data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                      parse_constant=constant)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data, {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest(),
                  "nonfinite_json_values_masked": len(invalid)}


def _names(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty list")
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{label} must contain nonempty names")
    if len({v.casefold() for v in values}) != len(values):
        raise ValueError(f"Duplicate case-insensitive {label}")
    return values


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must have length {length}")
    return [_number(v) for v in value]


def _plane(name):
    name = name.casefold()
    if name.endswith("h") or name.endswith(":x"):
        return "x"
    if name.endswith("v") or name.endswith(":y"):
        return "y"
    return None


def _read_dispersion(path, bpm_names, good):
    raw, source = _load(path, mask_nonfinite=True)
    fits = raw.get("fit parameters", raw.get("fitParameters"))
    if not isinstance(fits, dict):
        raise ValueError("Dispersion JSON requires named fit parameters")
    _names(list(fits), "dispersion BPM names")
    lookup = {name.casefold(): value for name, value in fits.items()}
    bad = raw.get("Not used BPMs", [])
    if not isinstance(bad, list) or any(not isinstance(v, str) for v in bad):
        raise ValueError("Not used BPMs must be a list of names")
    bad = {name.casefold() for name in bad}
    slope, mask = [], []
    for plane in ("x", "y"):
        for index, name in enumerate(bpm_names):
            entry = lookup.get(name.casefold(), {})
            value = None
            if isinstance(entry, dict):
                value = entry.get("k_D" + plane, entry.get(plane + "_fit_popt"))
                if isinstance(value, list):
                    value = value[0] if value else None
            value = _number(value)
            slope.append(value)
            mask.append(value is not None and good[index] and name.casefold() not in bad)
    return {"bpm_names": list(bpm_names), "row_order": "all_x_then_all_y",
            "slope": slope, "mask": mask, "unit": "m/Hz",
            "work_point": raw.get("work-point", raw.get("workPoint")),
            "not_used_bpms": raw.get("Not used BPMs", [])}, source


def read_loco_measurements(response_path, *, horizontal_sign, vertical_sign,
                           dispersion_path=None):
    """Read legacy full-span mm/mrad ORM data without touching machine state.

    Both signs must explicitly be +1 or -1. Rows are all X then all Y; columns
    follow corrector dictionary insertion order. Missing/invalid observations
    use None, never fictitious zero measurements. Positive resolution-derived
    sigma is required for an active mask entry; no unit-weight fallback occurs.
    GoodIndex is an optional, zero-based whitelist, not a lattice index list.
    """
    for sign in (horizontal_sign, vertical_sign):
        if isinstance(sign, bool) or sign not in (-1, 1):
            raise ValueError("Explicit horizontal_sign and vertical_sign must be +1 or -1")
    raw, source = _load(response_path, mask_nonfinite=True)
    bpms = _names(raw.get("BPM_names"), "BPM names")
    n = len(bpms)
    warnings = []
    good = [True] * n
    if "GoodIndex" in raw:
        indices = raw["GoodIndex"]
        if (not isinstance(indices, list)
                or any(type(i) is not int or not 0 <= i < n for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("GoodIndex must contain unique zero-based BPM indices")
        good = [i in indices for i in range(n)]
    resolutions = []
    for key in ("ResolX0", "ResolY0"):
        if key not in raw:
            resolutions.extend([None] * n)
            warnings.append(f"{key} absent: plane masked because uncertainty is unknown")
        else:
            resolutions.extend(_vector(raw[key], n, key))
    row_mask = [good[i % n] and r is not None and r > 0
                for i, r in enumerate(resolutions)]
    if any(r is None or r <= 0 for r in resolutions):
        warnings.append("Missing/nonpositive BPM resolutions are masked; no weights inferred")
    columns = [(name, value) for name, value in raw.items()
               if isinstance(value, dict) and ("response1" in value or "KickAngle" in value)]
    names = _names([name for name, _ in columns], "corrector names")
    response = [[None] * len(columns) for _ in range(2 * n)]
    sigma = [[None] * len(columns) for _ in range(2 * n)]
    mask = [[False] * len(columns) for _ in range(2 * n)]
    planes, spans = [], []
    for col, (name, entry) in enumerate(columns):
        plane = _plane(name)
        if plane is None:
            raise ValueError(f"Corrector plane is not explicit in H/V or :x/:y name: {name}")
        planes.append(plane)
        kick = _number(entry.get("KickAngle"))
        spans.append(abs(kick) * 1e-3 if kick is not None else None)
        if kick is None or kick == 0:
            warnings.append(f"{name}: invalid/zero KickAngle; entire column masked")
            continue
        vector = _vector(entry.get("response1"), 2 * n, name + ".response1")
        sign = horizontal_sign if plane == "x" else vertical_sign
        for row, value in enumerate(vector):
            response[row][col] = _number(value / kick * sign) if value is not None else None
            resolution = resolutions[row]
            if resolution is not None and resolution > 0:
                uncertainty = _number(resolution / abs(kick))
                sigma[row][col] = uncertainty if uncertainty and uncertainty > 0 else None
            mask[row][col] = (response[row][col] is not None
                              and sigma[row][col] is not None and row_mask[row])
    result = {"schema": "brilliance.loco_measurements.v1", "sources": {"response": source},
              "bpm_names": bpms, "corrector_names": names, "corrector_planes": planes,
              "row_order": "all_x_then_all_y", "response_unit": "m/rad",
              "sigma_unit": "m/rad", "response": response, "sigma": sigma, "mask": mask,
              "good_bpm_mask": good, "row_mask": row_mask, "kick_spans_rad": spans,
              "coordinate_signs": {"horizontal": horizontal_sign, "vertical": vertical_sign},
              "dispersion": None, "warnings": warnings}
    if dispersion_path is not None:
        result["dispersion"], result["sources"]["dispersion"] = _read_dispersion(
            dispersion_path, bpms, good)
    for kind, provenance in result["sources"].items():
        if provenance["nonfinite_json_values_masked"]:
            warnings.append(f"{kind}: nonfinite JSON constants converted to unavailable values")
    return result


def align_observations(data, bpm_names, corrector_names):
    """Align an imported bundle by unique case-insensitive name, never position.

    Missing targets remain None/False; diagnostics list unused and missing
    identities. This does not alter the supplied measurement bundle.
    """
    bpms = _names(list(bpm_names), "target BPM names")
    cors = _names(list(corrector_names), "target corrector names")
    source_bpms = _names(data["bpm_names"], "source BPM names")
    source_cors = _names(data["corrector_names"], "source corrector names")
    bmap = {name.casefold(): i for i, name in enumerate(source_bpms)}
    cmap = {name.casefold(): i for i, name in enumerate(source_cors)}
    bi, ci = [bmap.get(v.casefold()) for v in bpms], [cmap.get(v.casefold()) for v in cors]
    rows = [None if i is None else i + plane * len(source_bpms)
            for plane in (0, 1) for i in bi]
    result = dict(data, bpm_names=bpms, corrector_names=cors)
    for key in ("response", "sigma", "mask"):
        default = False if key == "mask" else None
        result[key] = [[default if r is None or c is None else data[key][r][c]
                        for c in ci] for r in rows]
    result["good_bpm_mask"] = [False if i is None else data["good_bpm_mask"][i] for i in bi]
    result["row_mask"] = [False if r is None else data["row_mask"][r] for r in rows]
    for key in ("corrector_planes", "kick_spans_rad"):
        result[key] = [None if i is None else data[key][i] for i in ci]
    if data.get("dispersion") is not None:
        dispersion = dict(data["dispersion"], bpm_names=bpms)
        for key in ("slope", "mask"):
            dispersion[key] = [(False if key == "mask" else None) if r is None
                               else data["dispersion"][key][r] for r in rows]
        result["dispersion"] = dispersion
    result["alignment"] = {
        "missing_bpms": [name for name, i in zip(bpms, bi) if i is None],
        "missing_correctors": [name for name, i in zip(cors, ci) if i is None],
        "extra_source_bpms": [name for name in source_bpms
                              if name.casefold() not in {v.casefold() for v in bpms}],
        "extra_source_correctors": [name for name in source_cors
                                    if name.casefold() not in {v.casefold() for v in cors}]}
    return result


def read_named_settings(path):
    """Read a Time/Comments/Data snapshot; infer neither units nor correction meaning."""
    raw, source = _load(path)
    values = raw.get("Data")
    if not isinstance(values, dict):
        raise ValueError("Named settings JSON requires a Data object")
    return {"schema": "brilliance.named_settings.v1", "source": source,
            "time": raw.get("Time"), "comments": raw.get("Comments"), "values": values,
            "unit_contract": "unspecified", "setting_semantics": "unspecified"}
