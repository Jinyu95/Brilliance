"""Typed, read-only access to archived scientific datasets.

The research agent must be able to answer unanticipated inspection and
presentation requests without rerunning JuTrack or receiving entire numerical
artifacts in its prompt.  This module exposes compact dataset descriptors and a
small declarative query language over immutable project evidence.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import content_hash, json_safe
from .lattice import LatticeIR

EVIDENCE_QUERY_VERSION = "1.0.0"
MAX_CATALOG_DATASETS = 120
MAX_QUERY_ROWS = 5_000
MAX_QUERY_COLUMNS = 40


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    id: str
    title: str
    source: str
    evidence_id: str | None
    path: str
    row_count: int
    columns: tuple[Mapping[str, Any], ...]
    scope: Mapping[str, Any]
    accepted: bool
    immutable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class _Dataset:
    descriptor: DatasetDescriptor
    rows: tuple[Mapping[str, Any], ...]


class EvidenceCatalog:
    """Discover and query tabular arrays already archived by a project."""

    def __init__(self, project: Any, project_root: Path | str):
        self.project = project
        self.project_root = Path(project_root).resolve()
        self._datasets: dict[str, _Dataset] = {}
        self._register_lattice_datasets()
        self._register_project_evidence()
        self._register_agent_evidence()

    def descriptors(self) -> list[dict[str, Any]]:
        return [dataset.descriptor.as_dict() for dataset in self._datasets.values()]

    def describe(self, dataset_id: str) -> dict[str, Any]:
        return self._get(dataset_id).descriptor.as_dict()

    def query(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        dataset_id = str(arguments.get("dataset_id", "")).strip()
        dataset = self._get(dataset_id)
        rows = [dict(row) for row in dataset.rows]
        source_columns = {
            str(column.get("name")) for column in dataset.descriptor.columns
            if isinstance(column, Mapping) and column.get("name")
        }
        filters = arguments.get("filters", [])
        if filters is None:
            filters = []
        if not isinstance(filters, (list, tuple)):
            raise ValueError("filters must be a list")
        for expression in filters:
            if not isinstance(expression, Mapping):
                raise ValueError("every filter must be an object")
            column = str(expression.get("column", ""))
            if column not in source_columns:
                raise ValueError(f"unknown filter column: {column!r}")
            rows = [row for row in rows if _matches_filter(row, expression)]

        derived = arguments.get("derived", [])
        if derived is None:
            derived = []
        if not isinstance(derived, (list, tuple)):
            raise ValueError("derived must be a list")
        for specification in derived:
            if not isinstance(specification, Mapping):
                raise ValueError("every derived column must be an object")
            rows = [_with_derived(row, specification) for row in rows]
        derived_columns = {
            str(item.get("name")) for item in derived
            if isinstance(item, Mapping) and item.get("name")
        }
        query_columns = source_columns | derived_columns

        sort_by = arguments.get("sort_by", [])
        if isinstance(sort_by, str):
            sort_by = [{"column": sort_by}]
        if not isinstance(sort_by, (list, tuple)):
            raise ValueError("sort_by must be a list")
        for specification in reversed(sort_by):
            if isinstance(specification, str):
                column, descending = specification, False
            elif isinstance(specification, Mapping):
                column = str(specification.get("column", ""))
                descending = bool(specification.get("descending", False))
            else:
                raise ValueError("every sort specification must be a string or object")
            if not column:
                raise ValueError("sort column must not be empty")
            if column not in query_columns:
                raise ValueError(f"unknown sort column: {column!r}")
            rows.sort(key=lambda row: _sort_key(row.get(column)), reverse=descending)

        available_columns = tuple(dict.fromkeys((
            *(
                str(column.get("name")) for column in dataset.descriptor.columns
                if isinstance(column, Mapping) and column.get("name")
            ),
            *(
                str(item.get("name")) for item in derived
                if isinstance(item, Mapping) and item.get("name")
            ),
            *_column_names(rows),
        )))
        requested_columns = arguments.get("columns", [])
        if requested_columns is None:
            requested_columns = []
        if not isinstance(requested_columns, (list, tuple)):
            raise ValueError("columns must be a list")
        columns = tuple(dict.fromkeys(str(column) for column in requested_columns if str(column)))
        if len(columns) > MAX_QUERY_COLUMNS:
            raise ValueError(f"a query may select at most {MAX_QUERY_COLUMNS} columns")
        if columns:
            unknown = set(columns) - set(available_columns)
            if unknown:
                raise ValueError(f"unknown dataset columns: {sorted(unknown)}")
        else:
            columns = available_columns[:MAX_QUERY_COLUMNS]

        try:
            offset = max(0, int(arguments.get("offset", 0)))
            limit = int(arguments.get("limit", min(500, MAX_QUERY_ROWS)))
        except (TypeError, ValueError) as exc:
            raise ValueError("offset and limit must be integers") from exc
        limit = max(1, min(limit, MAX_QUERY_ROWS))
        total_matches = len(rows)
        selected = rows[offset:offset + limit]
        projected = [
            {column: json_safe(row.get(column)) for column in columns}
            for row in selected
        ]
        view = _validated_view(arguments.get("view"), columns)
        return {
            "engine": "archived_evidence_query",
            "query_version": EVIDENCE_QUERY_VERSION,
            "dataset": dataset.descriptor.as_dict(),
            "columns": list(columns),
            "rows": projected,
            "row_count": len(projected),
            "total_matches": total_matches,
            "offset": offset,
            "truncated": offset + len(projected) < total_matches,
            "view": view,
            "provenance": {
                "project_id": getattr(self.project, "id", None),
                "project_revision": getattr(self.project, "revision", None),
                "dataset_id": dataset_id,
                "source_evidence_id": dataset.descriptor.evidence_id,
                "read_only": True,
            },
        }

    def _get(self, dataset_id: str) -> _Dataset:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            available = list(self._datasets)[:20]
            raise LookupError(
                f"unknown evidence dataset {dataset_id!r}; available datasets include {available}"
            ) from exc

    def _register_lattice_datasets(self) -> None:
        lattice = _active_lattice_mapping(self.project)
        if lattice is None:
            return
        try:
            rows = LatticeIR.from_dict(lattice).element_rows()
        except (KeyError, TypeError, ValueError):
            rows = _normalize_rows(lattice.get("elements", []))
        if not rows:
            return
        lattice_id = str(lattice.get("lattice_id", "active-lattice"))
        self._add(
            "lattice.active.elements",
            "Active lattice elements",
            rows,
            source=str(lattice.get("source", "accepted lattice")),
            evidence_id=lattice_id,
            path="active_lattice.elements",
            scope={"kind": "full_ring", "lattice_id": lattice_id},
            accepted=True,
        )
        source_rows, source_scope, recurring_rows, recurring_scope = _source_cell_rows(
            rows, lattice, self.project
        )
        if source_rows:
            self._add(
                "lattice.source_cell.elements",
                "One accepted periodic lattice cell",
                source_rows,
                source=str(lattice.get("source", "accepted lattice")),
                evidence_id=lattice_id,
                path="active_lattice.elements[source_cell]",
                scope=source_scope,
                accepted=True,
            )
            bending = [
                row for row in source_rows
                if str(row.get("type", row.get("kind", ""))).upper() in {"SBEND", "RBEND"}
                and abs(_float_or_zero(row.get("angle_rad"))) > 0.0
            ]
            if bending:
                self._add(
                    "lattice.source_cell.bending_magnets",
                    "All physical bending magnets in one periodic cell",
                    bending,
                    source=str(lattice.get("source", "accepted lattice")),
                    evidence_id=lattice_id,
                    path="active_lattice.elements[source_cell,bending_magnets]",
                    scope={
                        **source_scope,
                        "physical_bend_count": len(bending),
                        "definition": "every nonzero-angle SBEND or RBEND",
                    },
                    accepted=True,
                )
                primary = [row for row in bending if _is_primary_bend(row)]
                if primary:
                    self._add(
                        "lattice.source_cell.primary_bends",
                        "Primary positive bends in one periodic cell",
                        primary,
                        source=str(lattice.get("source", "accepted lattice")),
                        evidence_id=lattice_id,
                        path="active_lattice.elements[source_cell,primary_bends]",
                        scope={
                            **source_scope,
                            "primary_bend_count": len(primary),
                            "definition": (
                                "positive-angle main bends; auxiliary reverse bends are in "
                                "lattice.source_cell.bending_magnets"
                            ),
                        },
                        accepted=True,
                    )
        if recurring_rows and len(recurring_rows) != len(source_rows):
            self._add(
                "lattice.recurring_unit.elements",
                "Elementary recurring optimizer unit",
                recurring_rows,
                source=str(lattice.get("source", "accepted lattice")),
                evidence_id=lattice_id,
                path="active_lattice.elements[recurring_unit]",
                scope=recurring_scope,
                accepted=True,
            )

    def _register_project_evidence(self) -> None:
        for index, evidence in enumerate(getattr(self.project, "evidence", ())):
            if not isinstance(evidence, Mapping):
                continue
            evidence_id = str(
                evidence.get("evidence_id") or f"project-evidence-{index + 1}"
            )
            validation = evidence.get("deterministic_validation")
            accepted = not (
                isinstance(validation, Mapping) and validation.get("passed") is False
            )
            result = evidence.get("structured_result")
            if isinstance(result, Mapping):
                self._discover(
                    result,
                    evidence_id=evidence_id,
                    source=str(evidence.get("source", "project evidence")),
                    accepted=accepted,
                    prefix="structured_result",
                )

    def _register_agent_evidence(self) -> None:
        directory = (
            self.project_root / str(getattr(self.project, "id", "")) / "agent_evidence"
        )
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("agent-*.json"))[-40:]:
            try:
                evidence = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(evidence, Mapping):
                continue
            result = evidence.get("structured_result")
            if not isinstance(result, Mapping):
                continue
            validation = evidence.get("deterministic_validation")
            accepted = not (
                isinstance(validation, Mapping) and validation.get("passed") is False
            )
            self._discover(
                result,
                evidence_id=str(evidence.get("artifact_id", path.stem)),
                source=str(evidence.get("source", "agent evidence")),
                accepted=accepted,
                prefix="structured_result",
            )

    def _discover(
        self,
        value: Mapping[str, Any],
        *,
        evidence_id: str,
        source: str,
        accepted: bool,
        prefix: str,
        depth: int = 0,
    ) -> None:
        if depth > 6 or len(self._datasets) >= MAX_CATALOG_DATASETS:
            return
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if isinstance(child, (list, tuple)) and child:
                if all(isinstance(row, Mapping) for row in child):
                    if not _redundant_lattice_path(path):
                        dataset_id = _dataset_id(evidence_id, path)
                        self._add(
                            dataset_id,
                            _dataset_title(key, evidence_id),
                            child,
                            source=source,
                            evidence_id=evidence_id,
                            path=path,
                            scope={"kind": _scope_from_path(path)},
                            accepted=accepted,
                        )
                elif all(_scalar(item) for item in child):
                    self._add(
                        _dataset_id(evidence_id, path),
                        _dataset_title(key, evidence_id),
                        ({"index": index, "value": item} for index, item in enumerate(child)),
                        source=source,
                        evidence_id=evidence_id,
                        path=path,
                        scope={"kind": _scope_from_path(path)},
                        accepted=accepted,
                    )
            elif isinstance(child, Mapping):
                self._discover(
                    child,
                    evidence_id=evidence_id,
                    source=source,
                    accepted=accepted,
                    prefix=path,
                    depth=depth + 1,
                )

    def _add(
        self,
        dataset_id: str,
        title: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        source: str,
        evidence_id: str | None,
        path: str,
        scope: Mapping[str, Any],
        accepted: bool,
    ) -> None:
        if dataset_id in self._datasets or len(self._datasets) >= MAX_CATALOG_DATASETS:
            return
        normalized = tuple(_normalize_rows(rows))
        if not normalized:
            return
        descriptor = DatasetDescriptor(
            id=dataset_id,
            title=title,
            source=source,
            evidence_id=evidence_id,
            path=path,
            row_count=len(normalized),
            columns=_infer_columns(normalized),
            scope=dict(scope),
            accepted=accepted,
        )
        self._datasets[dataset_id] = _Dataset(descriptor, normalized)




def validate_query_result(result: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    rows = result.get("rows")
    if not isinstance(rows, list):
        issues.append("query result rows are not a list")
        rows = []
    if len(rows) > MAX_QUERY_ROWS:
        issues.append(f"query returned more than {MAX_QUERY_ROWS} rows")
    dataset = result.get("dataset")
    if not isinstance(dataset, Mapping) or not dataset.get("id"):
        issues.append("query result has no source dataset descriptor")
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("read_only") is not True:
        issues.append("query result is missing read-only provenance")
    return {
        "passed": not issues,
        "validator": "evidence_query_contract",
        "version": EVIDENCE_QUERY_VERSION,
        "criterion_results": [{
            "criterion": "The result is a bounded, provenance-linked view of immutable evidence.",
            "passed": not issues,
        }],
        "issues": issues,
        "checked_values": {
            "dataset_id": dataset.get("id") if isinstance(dataset, Mapping) else None,
            "row_count": len(rows),
            "total_matches": result.get("total_matches"),
            "read_only": provenance.get("read_only") if isinstance(provenance, Mapping) else None,
        },
        "claim_authority": "accepted_evidence_presentation",
    }


def _active_lattice_mapping(project: Any) -> Mapping[str, Any] | None:
    active_id = getattr(project, "active_lattice_id", None)
    for item in reversed(getattr(project, "lattice_history", ())):
        if not isinstance(item, Mapping):
            continue
        if active_id is None or item.get("lattice_id") == active_id:
            return item
    # Older projects archived construction evidence before lattice-history was
    # introduced.  Treat the latest accepted lattice artifact as the active
    # read-only dataset so inspection remains available after an upgrade.
    for evidence in reversed(getattr(project, "evidence", ())):
        if not isinstance(evidence, Mapping):
            continue
        validation = evidence.get("deterministic_validation")
        if isinstance(validation, Mapping) and validation.get("passed") is False:
            continue
        result = evidence.get("structured_result")
        if not isinstance(result, Mapping):
            continue
        artifact = result.get("lattice_artifact")
        if isinstance(artifact, Mapping) and artifact.get("elements"):
            return artifact
        elements = result.get("lattice_elements")
        if isinstance(elements, (list, tuple)) and elements:
            return {
                "lattice_id": evidence.get("lattice_id") or evidence.get("evidence_id"),
                "source": evidence.get("source", "accepted construction evidence"),
                "elements": elements,
                "metadata": {
                    "repetitions": result.get("repetitions", result.get("periodicity", 1)),
                },
            }
    return None


def _source_cell_rows(
    rows: list[dict[str, Any]], lattice: Mapping[str, Any], project: Any
) -> tuple[
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]
]:
    periods = [row.get("source_period") for row in rows if row.get("source_period") is not None]
    if periods:
        first = periods[0]
        recurring = [dict(row) for row in rows if row.get("source_period") == first]
        equivalents = _source_cell_equivalents(project, lattice)
        ordered_periods = list(dict.fromkeys(periods))
        selected_periods = ordered_periods[:equivalents]
        selected = [
            dict(row) for row in rows if row.get("source_period") in selected_periods
        ]
        cell_scope = {
            "kind": "periodic_lattice_cell",
            "source_periods": selected_periods,
            "recurring_units": equivalents,
            "lattice_id": lattice.get("lattice_id"),
            "construction_source": _construction_source(lattice),
        }
        recurring_scope = {
            "kind": "elementary_recurring_unit",
            "source_period": first,
            "lattice_id": lattice.get("lattice_id"),
            "use_boundary": (
                "This is the optimizer's elementary repeat, not the complete periodic cell."
            ),
        }
        return (
            _rebase_positions(selected), cell_scope,
            _rebase_positions(recurring), recurring_scope,
        )
    metadata = lattice.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    repetitions = metadata.get("repetitions", metadata.get("source_periodicity"))
    try:
        count = int(repetitions)
    except (TypeError, ValueError):
        count = 1
    if count > 1 and len(rows) % count == 0:
        selected = rows[: len(rows) // count]
    else:
        selected = rows
    scope = {
        "kind": "periodic_lattice_cell" if count > 1 else "accepted_lattice",
        "inferred_repetitions": count,
        "lattice_id": lattice.get("lattice_id"),
    }
    return _rebase_positions(selected), scope, [], {}


def _source_cell_equivalents(project: Any, lattice: Mapping[str, Any]) -> int:
    metadata = lattice.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    candidates: list[Any] = [
        metadata.get("source_cell_repetitions"),
        metadata.get("n_cell_equivalents"),
    ]
    parent_id = lattice.get("parent_lattice_id")
    if parent_id:
        parent = next(
            (
                item for item in reversed(getattr(project, "lattice_history", ()))
                if isinstance(item, Mapping) and item.get("lattice_id") == parent_id
            ),
            None,
        )
        if isinstance(parent, Mapping):
            parent_metadata = parent.get("metadata")
            parent_metadata = parent_metadata if isinstance(parent_metadata, Mapping) else {}
            candidates.extend((
                parent_metadata.get("source_cell_repetitions"),
                parent_metadata.get("n_cell_equivalents"),
            ))
    for value in candidates:
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return 1


def _is_primary_bend(row: Mapping[str, Any]) -> bool:
    name = " ".join((
        str(row.get("name", "")),
        str(row.get("source_element", "")),
        str(row.get("family", "")),
    )).upper()
    angle = _float_or_zero(row.get("angle_rad"))
    if any(token in name for token in ("MAIN", ":MB", " MB", "MB_")):
        return angle > 0.0
    return angle > 0.0


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _construction_source(lattice: Mapping[str, Any]) -> str:
    metadata = lattice.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    explicit = str(metadata.get("construction_source", "")).strip()
    if explicit:
        return explicit
    if str(lattice.get("source", "")) == "native-full-ring-construction":
        return "brilliance.physics_services.construct_full_ring"
    return str(lattice.get("source", "accepted lattice artifact"))


def _rebase_positions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    try:
        origin = float(rows[0].get("s_start_m", 0.0))
    except (TypeError, ValueError):
        origin = 0.0
    rebased: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ("s_start_m", "s_end_m", "s_m"):
            try:
                if key in item:
                    item[key] = float(item[key]) - origin
            except (TypeError, ValueError):
                pass
        rebased.append(item)
    return rebased


def _normalize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_flatten_row(row) for row in rows if isinstance(row, Mapping)]


def _flatten_row(row: Mapping[str, Any], *, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key, value in row.items():
        key = f"{prefix}.{raw_key}" if prefix else str(raw_key)
        if isinstance(value, Mapping) and depth < 2:
            result.update(_flatten_row(value, prefix=key, depth=depth + 1))
        elif _scalar(value):
            result[key] = value
        elif value is not None:
            result[key] = json_safe(value)
    return result


def _infer_columns(rows: tuple[Mapping[str, Any], ...]) -> tuple[Mapping[str, Any], ...]:
    names = _column_names(rows)
    columns: list[Mapping[str, Any]] = []
    for name in names:
        values = [row.get(name) for row in rows[:200] if row.get(name) is not None]
        columns.append({
            "name": name,
            "type": _value_type(values),
            "unit": _unit_for_column(name),
            "nullable": len(values) < min(len(rows), 200),
        })
    return tuple(columns)


def _column_names(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(str(name))
        if len(names) >= MAX_QUERY_COLUMNS * 2:
            break
    return tuple(names)


def _value_type(values: list[Any]) -> str:
    if not values:
        return "unknown"
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "integer"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return "number"
    if all(isinstance(value, str) for value in values):
        return "string"
    return "json"


def _unit_for_column(name: str) -> str | None:
    lowered = name.lower()
    exact = {
        "k1": "m^-2", "k1_m2": "m^-2", "k1_m-2": "m^-2",
        "k2": "m^-3", "k2_m3": "m^-3", "k3_m4": "m^-4",
        "angle_rad": "rad", "dp": "1", "dx": "m", "dpx": "1",
        "beta_x": "m", "beta_y": "m", "alpha_x": "1", "alpha_y": "1",
    }
    if lowered in exact:
        return exact[lowered]
    for suffix, unit in (
        ("_mrad", "mrad"), ("_rad", "rad"), ("_mm", "mm"),
        ("_um", "um"), ("_pm", "pm"), ("_gev", "GeV"),
        ("_ev", "eV"), ("_m", "m"), ("_s", "s"),
    ):
        if lowered.endswith(suffix):
            return unit
    return None


def _matches_filter(row: Mapping[str, Any], expression: Mapping[str, Any]) -> bool:
    column = str(expression.get("column", ""))
    operation = str(expression.get("op", "eq")).lower()
    if not column:
        raise ValueError("filter column must not be empty")
    actual = row.get(column)
    expected = expression.get("value")
    if operation == "eq":
        return actual == expected
    if operation == "ne":
        return actual != expected
    if operation == "contains":
        return str(expected).lower() in str(actual).lower()
    if operation == "in":
        if not isinstance(expected, (list, tuple)):
            raise ValueError("the 'in' filter requires a list value")
        return actual in expected
    if operation == "between":
        if not isinstance(expected, (list, tuple)) or len(expected) != 2:
            raise ValueError("the 'between' filter requires [lower, upper]")
        return _compare(actual, expected[0], "gte") and _compare(actual, expected[1], "lte")
    if operation in {"lt", "lte", "gt", "gte"}:
        return _compare(actual, expected, operation)
    raise ValueError(f"unsupported filter operation: {operation}")


def _compare(actual: Any, expected: Any, operation: str) -> bool:
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError):
        left, right = str(actual), str(expected)
    return {
        "lt": left < right,
        "lte": left <= right,
        "gt": left > right,
        "gte": left >= right,
    }[operation]


def _with_derived(row: Mapping[str, Any], specification: Mapping[str, Any]) -> dict[str, Any]:
    name = str(specification.get("name", "")).strip()
    operation = str(specification.get("op", "")).strip().lower()
    column = str(specification.get("column", "")).strip()
    if not name or not operation or not column:
        raise ValueError("derived columns require name, op, and column")
    value = _number(row.get(column), column)
    if operation == "abs":
        derived = abs(value)
    elif operation == "scale":
        derived = value * _number(specification.get("factor", 1.0), "factor")
    else:
        other = _number(row.get(str(specification.get("other_column", ""))), "other_column")
        if operation == "difference":
            derived = value - other
        elif operation == "sum":
            derived = value + other
        elif operation == "ratio":
            if other == 0.0:
                raise ValueError("derived ratio denominator is zero")
            derived = value / other
        else:
            raise ValueError(f"unsupported derived operation: {operation}")
    return {**dict(row), name: derived}


def _validated_view(value: Any, columns: tuple[str, ...]) -> dict[str, Any]:
    if value is None:
        return {"kind": "table", "title": "Evidence table"}
    if not isinstance(value, Mapping):
        raise ValueError("view must be an object")
    kind = str(value.get("kind", "table")).lower()
    if kind not in {"table", "line", "scatter", "bar"}:
        raise ValueError("view kind must be table, line, scatter, or bar")
    view = {
        "kind": kind,
        "title": str(value.get("title", "Evidence view"))[:200],
    }
    if kind != "table":
        x = str(value.get("x", ""))
        y_value = value.get("y", [])
        y = [str(item) for item in y_value] if isinstance(y_value, (list, tuple)) else [str(y_value)]
        y = [item for item in y if item]
        unknown = ({x, *y} - set(columns)) - {""}
        if not x or not y or unknown:
            raise ValueError(
                f"plot view requires existing x and y columns; unknown={sorted(unknown)}"
            )
        series = str(value.get("series", "")) or None
        if series is not None and series not in columns:
            raise ValueError(f"plot series column does not exist: {series!r}")
        view.update({
            "x": x,
            "y": y,
            "series": series,
            "x_label": str(value.get("x_label", x)),
            "y_label": str(value.get("y_label", ", ".join(y))),
        })
    return view


def _number(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _sort_key(value: Any) -> tuple[int, Any]:
    if value is None:
        return 2, ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0, float(value)
    return 1, str(value).lower()


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _dataset_id(evidence_id: str, path: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", path.lower()).strip(".")[-72:]
    return f"evidence.{content_hash(evidence_id, length=8)}.{slug}"


def _dataset_title(key: Any, evidence_id: str) -> str:
    words = re.sub(r"[_-]+", " ", str(key)).strip().title()
    return f"{words} ({evidence_id})"


def _scope_from_path(path: str) -> str:
    lowered = path.lower()
    if "twiss" in lowered or "optics" in lowered:
        return "optics_samples"
    if "trajectory" in lowered:
        return "particle_trajectory"
    if "scan" in lowered or "population" in lowered or "candidate" in lowered:
        return "parameter_scan"
    if "seed" in lowered:
        return "ensemble"
    return "archived_rows"


def _redundant_lattice_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith("lattice_elements") or lowered.endswith("lattice_artifact.elements")
