"""Durable exact-candidate history and feasibility-first experiment selection.

SQLite transactions avoid a file per objective evaluation. This ledger records
what a callback measured; scalar checks are not accelerator-physics acceptance.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .candidate_evidence import validate_saved_candidate_evaluation

SCHEMA = "brilliance_experiment_candidates/v1"


def _plain(value):
    """Preserve finite JSON values exactly; do not turn NaN into a measurement."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Ledger numbers must be finite; use None for unavailable measurements")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Ledger mapping keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict"):
        return _plain(value.as_dict())
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return _plain(value.item())
    raise TypeError(f"Unsupported ledger value: {type(value).__name__}")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


class CandidateLedger:
    """Append-only measurements, scoped to one workspace and exact contract.

    ``objectives`` explicitly maps measurement names to ``'min'`` or ``'max'``.
    Managed callers use BRILLIANCE_WORKSPACE; standalone callers supply workspace.
    Resume never silently changes the contract, objective order/direction or scope.
    One connection per writer thread is reused; independent processes can append.
    """

    def __init__(self, directory, *, contract, objectives, workspace=None, resume=True):
        if not isinstance(resume, bool):
            raise TypeError("resume must be boolean")
        self._configure_storage(directory, workspace, read_only=False)
        if not isinstance(contract, Mapping) or not isinstance(objectives, Mapping) or not objectives:
            raise ValueError("Supply a contract mapping and explicit nonempty objectives")
        if any(not isinstance(name, str) or not name.strip() or direction not in {"min", "max"}
               for name, direction in objectives.items()):
            raise ValueError("Objective names must be nonempty strings with direction min or max")
        self.objectives = dict(objectives)
        manifest = _plain({"schema": SCHEMA, "workspace": str(self.workspace), "directory": str(self.directory),
                           "contract": contract, "objectives": list(self.objectives.items())})
        self.fingerprint = _hash(manifest)
        self._manifest = {**manifest, "fingerprint": self.fingerprint}
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            # SQLite's initial WAL transition can return BUSY without invoking
            # busy_timeout when independent workers create the ledger together.
            deadline = time.monotonic() + 30.0
            while True:
                try:
                    connection.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if getattr(exc, "sqlite_errorcode", 0) & 255 not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("CREATE TABLE IF NOT EXISTS manifest (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload TEXT NOT NULL)")
                connection.execute("CREATE TABLE IF NOT EXISTS candidates (sequence INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)")
                for table in ("manifest", "candidates"):
                    for operation in ("UPDATE", "DELETE"):
                        connection.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable candidate ledger'); END")
                previous = connection.execute("SELECT payload FROM manifest WHERE singleton=1").fetchone()
                if previous is None:
                    connection.execute("INSERT INTO manifest(singleton,payload) VALUES(1,?)", (_json(self._manifest),))
                elif not resume:
                    raise ValueError("Candidate ledger already exists; resume explicitly or choose a new experiment directory")
                else:
                    self._validate_manifest(previous[0])
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _configure_storage(self, directory, workspace, *, read_only):
        managed = os.environ.get("BRILLIANCE_WORKSPACE", "").strip()
        if not managed and workspace is None:
            raise ValueError("An explicit workspace is required outside managed execution")
        self.workspace = Path(managed or workspace).resolve()
        if workspace is not None and Path(workspace).resolve() != self.workspace:
            raise ValueError("Cannot replace the configured experiment workspace")
        self.directory = Path(directory).resolve()
        if not self.directory.is_relative_to(self.workspace):
            raise ValueError("Candidate ledger must stay inside the current workspace")
        self.path = self.directory / "candidates.sqlite3"
        self._read_only = read_only
        self._local = threading.local()
        self._connections = []
        self._connection_lock = threading.Lock()

    @classmethod
    def open(cls, directory, *, workspace=None):
        """Recover saved selection read-only, without importing a callback."""
        ledger = cls.__new__(cls)
        ledger._configure_storage(directory, workspace, read_only=True)
        if not ledger.path.is_file():
            raise FileNotFoundError("No candidate ledger exists in this experiment directory")
        with ledger._connection() as connection:
            row = connection.execute("SELECT payload FROM manifest WHERE singleton=1").fetchone()
            if row is None:
                raise ValueError("The candidate ledger manifest is missing")
            ledger._manifest = json.loads(row[0])
            ledger.fingerprint = ledger._manifest.get("fingerprint")
            ledger.objectives = dict(ledger._manifest["objectives"])
            ledger._validate_manifest(row[0])
        return ledger

    @contextmanager
    def _connection(self):
        if self.path.resolve().parent != self.directory or not self.path.resolve().is_relative_to(self.workspace):
            raise ValueError("Candidate ledger storage no longer belongs to this workspace")
        managed = os.environ.get("BRILLIANCE_WORKSPACE", "").strip()
        if managed and Path(managed).resolve() != self.workspace:
            raise ValueError("Candidate ledger cannot cross the current managed workspace")
        connection = getattr(self._local, "connection", None)
        if connection is None:
            target = self.path.as_uri() + "?mode=ro" if self._read_only else self.path
            connection = sqlite3.connect(target, uri=self._read_only, timeout=30.0,
                                         isolation_level=None, check_same_thread=False)
            if not self._read_only:
                connection.execute("PRAGMA synchronous=FULL")
            self._local.connection = connection
            with self._connection_lock:
                self._connections.append(connection)
        yield connection

    def close(self):
        """Close owned connections after all writers have finished."""
        with self._connection_lock:
            for connection in self._connections:
                connection.close()
            self._connections.clear()
        self._local = threading.local()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items()
                if key not in {"_local", "_connections", "_connection_lock"}}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()
        self._connections = []
        self._connection_lock = threading.Lock()

    def _validate_manifest(self, raw):
        stored = json.loads(raw)
        fingerprint = stored.pop("fingerprint", None)
        if (_hash(stored) != fingerprint or {**stored, "fingerprint": fingerprint} != self._manifest
                or stored.get("schema") != SCHEMA or stored.get("workspace") != str(self.workspace)
                or stored.get("directory") != str(self.directory)
                or stored.get("objectives") != _plain(list(self.objectives.items()))):
            raise ValueError("Candidate ledger contract, objectives, scope or manifest hash does not match")

    def _evidence(self, candidate, supplied):
        if supplied is None:
            return {"kind": "scalar_measurements_only", "candidate_binding_verified": False,
                    "measurements_verified": False, "physics_accepted": False}
        if not isinstance(supplied, Mapping):
            raise TypeError("evaluation_receipt must be a saved receipt mapping")
        receipt = supplied.get("receipt", supplied)
        output = Path(supplied.get("output_directory") or os.environ.get("BRILLIANCE_RUN_OUTPUT") or self.workspace).resolve()
        if not output.is_relative_to(self.workspace):
            raise ValueError("An evaluation receipt cannot refer to another workspace")
        verified = validate_saved_candidate_evaluation(candidate, receipt, output)
        if not verified["passed"]:
            raise ValueError("Unverified archived candidate evaluation: " + "; ".join(verified["issues"]))
        return {"kind": "verified_archived_evaluation_receipt", "candidate_binding_verified": True,
                "receipt": _plain(receipt), "output_directory": str(output),
                "measurements_verified": False, "physics_accepted": False}

    def record(self, *, candidate, measurements, checks, provenance=None, evaluation_receipt=None):
        """Commit one exact attempt before returning it to selection/reporting.

        Checks are booleans or mappings with passed=bool/None and a nonnegative
        violation (None means unknown). Measurements may be None, never NaN/Inf.
        A verified archived receipt requires a complete serialized lattice, not
        only element rows, and an optional {receipt, output_directory} envelope.
        Receipt binding does not certify arbitrary summary metrics or acceptance.
        """
        if self._read_only:
            raise ValueError("A ledger opened for read-only recovery cannot record new evaluations")
        snapshot, measured = _plain(candidate), _plain(measurements)
        if not isinstance(snapshot, dict) or not snapshot:
            raise ValueError("candidate must be a nonempty exact parameter or element mapping")
        if "elements" in snapshot:
            if not isinstance(snapshot["elements"], list) or not snapshot["elements"]:
                raise ValueError("candidate elements must be a nonempty exact list")
            if any(not isinstance(row, dict) for row in snapshot["elements"]):
                raise ValueError("candidate elements must be mappings")
        if not isinstance(measured, dict) or not isinstance(checks, Mapping):
            raise TypeError("measurements and checks must be mappings")
        normalized = {}
        for name, check in _plain(checks).items():
            if isinstance(check, bool):
                check = {"passed": check, "violation": 0.0 if check else 1.0}
            if not isinstance(check, dict) or check.get("passed") not in (True, False, None):
                raise ValueError("Each check must declare boolean/nullable passed and its violation")
            passed, violation = check.get("passed"), check.get("violation")
            if passed is not None and not isinstance(passed, bool):
                raise ValueError("Check passed must be boolean or None")
            if violation is not None and (not _number(violation) or violation < 0):
                raise ValueError("Constraint violation must be finite, nonnegative or None")
            if passed is True:
                if violation not in (None, 0):
                    raise ValueError("A passed check cannot have a positive violation")
                violation = 0.0
            normalized[name] = {**check, "passed": passed, "violation": violation}
        objective_values = {name: measured.get(name) if _number(measured.get(name)) else None for name in self.objectives}
        violations = [item["violation"] for item in normalized.values()]
        total = sum(violations) if all(value is not None for value in violations) else None
        if total is not None and not math.isfinite(total):
            total = None
        entry = {"schema": SCHEMA, "ledger_fingerprint": self.fingerprint, "attempt_id": uuid4().hex,
                 "created_at": datetime.now(timezone.utc).isoformat(), "candidate": snapshot,
                 "candidate_fingerprint": _hash(snapshot), "measurements": measured, "checks": normalized,
                 "objective_values": objective_values, "constraint_violation": total,
                 "feasible": all(item["passed"] is True for item in normalized.values()) and all(value is not None for value in objective_values.values()),
                 "provenance": _plain(provenance or {}), "evidence": self._evidence(snapshot, evaluation_receipt)}
        content_hash = _hash(entry)
        entry.update(candidate_id="candidate-" + content_hash, content_sha256=content_hash)
        encoded = _json(entry)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._validate_manifest(connection.execute("SELECT payload FROM manifest WHERE singleton=1").fetchone()[0])
                connection.execute("INSERT INTO candidates(candidate_id,payload) VALUES(?,?)", (entry["candidate_id"], encoded))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return json.loads(encoded)

    def _validated_entry(self, candidate_id, raw):
        entry = json.loads(raw)
        identity, digest = entry.pop("candidate_id", None), entry.pop("content_sha256", None)
        if (identity != candidate_id or identity != "candidate-" + str(digest) or _hash(entry) != digest
                or entry.get("ledger_fingerprint") != self.fingerprint
                or _hash(entry.get("candidate")) != entry.get("candidate_fingerprint")):
            raise ValueError("Candidate ledger entry failed its immutable content hash")
        evidence = entry["evidence"]
        if evidence["candidate_binding_verified"]:
            self._evidence(entry["candidate"], evidence)
        return {**entry, "candidate_id": identity, "content_sha256": digest}

    def _iter_entries(self, *, recent_limit=None):
        with self._connection() as connection:
            self._validate_manifest(connection.execute("SELECT payload FROM manifest WHERE singleton=1").fetchone()[0])
            cursor = (connection.execute("SELECT candidate_id,payload FROM candidates ORDER BY sequence")
                      if recent_limit is None else connection.execute(
                          "SELECT candidate_id,payload FROM candidates ORDER BY sequence DESC LIMIT ?", (recent_limit,)))
            try:
                for candidate_id, raw in cursor:
                    yield self._validated_entry(candidate_id, raw)
            finally:
                cursor.close()

    def entries(self):
        """Read every immutable entry, checking content and receipt integrity."""
        return list(self._iter_entries())

    def recent_entries(self, limit=12):
        """Return at most limit exact entries, newest first, using a bounded SQL query.

        These are display alternatives, not a selection pool. ``select()`` still
        verifies and considers all history. Every returned entry is verified.
        """
        if type(limit) is not int or limit < 1:
            raise ValueError("recent entries limit must be a positive integer")
        return list(self._iter_entries(recent_limit=limit))

    def select(self, *, mode="pareto"):
        """Select from all committed history, never a caller's possibly stale pool.

        Pareto mode respects each min/max direction. Its selected representative
        is lexicographically first in declared objective order, not a unique
        multi-objective optimum. Equal objective vectors share one front entry;
        all original attempts remain in entries(). Minimum mode needs one objective.
        No feasible entry means selected=None; least_violation is diagnostic only.
        """
        if mode not in {"pareto", "minimum"} or (mode == "minimum" and len(self.objectives) != 1):
            raise ValueError("Use pareto, or minimum with exactly one declared objective")
        def vector(entry):
            return tuple((1 if direction == "min" else -1) * entry["objective_values"][name]
                         for name, direction in self.objectives.items())
        front, least = [], None
        evaluated = feasible = 0
        def violation_rank(entry):
            return (entry["constraint_violation"] is None, entry["constraint_violation"] or 0.0)
        for entry in self._iter_entries():
            evaluated += 1
            if not entry["feasible"]:
                if least is None or violation_rank(entry) < violation_rank(least):
                    least = entry
                continue
            feasible += 1
            values = vector(entry)
            if len(self.objectives) == 1:
                if not front or values < vector(front[0]):
                    front = [entry]
                continue
            if any(all(left <= right for left, right in zip(vector(other), values)) for other in front):
                continue
            front = [other for other in front if not all(left <= right for left, right in zip(values, vector(other)))]
            front.append(entry)
        front.sort(key=vector)
        return {"status": "feasible_selection" if front else "no_feasible_candidate", "selected": front[0] if front else None,
                "pareto_front": front, "least_violation": least,
                "counts": {"evaluated": evaluated, "feasible": feasible, "infeasible": evaluated - feasible},
                "selection_rule": "feasibility first; min/max Pareto dominance; lexicographic representative in declared objective order",
                "physics_accepted": False}
