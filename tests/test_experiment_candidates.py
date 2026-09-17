"""Exact candidate history and disk-only selection, without numerical execution."""
from __future__ import annotations

import copy
import json
import math
import pickle
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import perf_counter

import pytest

from brilliance.experiment_candidates import CandidateLedger


@pytest.fixture(autouse=True)
def isolated_workspace(monkeypatch):
    monkeypatch.delenv("BRILLIANCE_WORKSPACE", raising=False)
    monkeypatch.delenv("BRILLIANCE_RUN_OUTPUT", raising=False)


def make_ledger(root, **kwargs):
    return CandidateLedger(root / "experiment", workspace=root,
                           contract=kwargs.pop("contract", {"energy": 3.5, "length": 25.0}),
                           objectives=kwargs.pop("objectives", {"score": "min"}), **kwargs)


def rows(strength=.7346199519999999):
    return {"elements": [{"name": "Q", "length": .23123456789123,
                          "k1": strength, "fields": {"PolynomB": [0., strength, 0., 0.]}}]}


def record(ledger, score, *, passed=True, candidate=None, **extra):
    return ledger.record(candidate=candidate or rows(), measurements={"score": score},
                         checks={"limit": passed}, **extra)


def test_exact_snapshots_and_reevaluations_are_immutable_distinct_attempts(tmp_path):
    with make_ledger(tmp_path) as ledger:
        original = rows()
        measured = {"score": 3.12345678912345, "unavailable": None}
        first = ledger.record(candidate=original, measurements=measured, checks={"stable": True})
        duplicate = ledger.record(candidate=original, measurements=measured, checks={"stable": True})
        assert first["candidate_id"] != duplicate["candidate_id"]
        assert first["candidate_fingerprint"] == duplicate["candidate_fingerprint"]
        original["elements"][0]["k1"] = 88
        measured["score"] = 99
        first["candidate"]["elements"][0]["length"] = 100
        archived = ledger.entries()[0]
        assert archived["candidate"] == rows()
        assert archived["measurements"] == {"score": 3.12345678912345, "unavailable": None}
        assert archived["evidence"] == {"kind": "scalar_measurements_only", "candidate_binding_verified": False,
                                       "measurements_verified": False, "physics_accepted": False}
        assert len(ledger._connections) == 1  # Not a new SQLite connection per trial.


def test_all_history_selection_keeps_early_parent_after_later_failure_and_resume(tmp_path):
    with make_ledger(tmp_path) as ledger:
        best = record(ledger, 5.)
        record(ledger, 100.)
        record(ledger, .1, passed={"passed": False, "violation": .01})
        assert ledger.select()["selected"]["candidate_id"] == best["candidate_id"]
    with make_ledger(tmp_path) as resumed:
        assert resumed.select(mode="minimum")["selected"]["candidate_id"] == best["candidate_id"]
        assert resumed.select()["counts"] == {"evaluated": 3, "feasible": 2, "infeasible": 1}
        assert resumed.select()["physics_accepted"] is False


def test_no_feasible_result_is_never_selected_or_falsely_certified(tmp_path):
    with make_ledger(tmp_path) as ledger:
        record(ledger, None)
        least = record(ledger, 8, passed={"passed": False, "violation": .1})
        record(ledger, 1, passed={"passed": None, "violation": None})
        selected = ledger.select()
        assert selected["selected"] is None and selected["pareto_front"] == []
        assert selected["least_violation"]["candidate_id"] != least["candidate_id"]  # Missing score is diagnostic-only too.
        assert selected["status"] == "no_feasible_candidate"


def test_multiobjective_directions_dominance_and_explicit_representative(tmp_path):
    with make_ledger(tmp_path, objectives={"cost": "min", "acceptance": "max"}) as ledger:
        def add(cost, acceptance):
            return ledger.record(candidate=rows(), measurements={"cost": cost, "acceptance": acceptance}, checks={})
        a, b = add(1., 4.), add(2., 6.)
        add(2., 3.)
        add(1., 4.)
        selection = ledger.select()
        assert [e["candidate_id"] for e in selection["pareto_front"]] == [a["candidate_id"], b["candidate_id"]]
        assert selection["selected"]["candidate_id"] == a["candidate_id"]
        assert selection["counts"]["evaluated"] == 4
        with pytest.raises(ValueError, match="exactly one"):
            ledger.select(mode="minimum")
    with CandidateLedger(tmp_path / "maximum", workspace=tmp_path, contract={}, objectives={"score": "max"}) as ledger:
        record(ledger, 2.)
        best = record(ledger, 7.)
        assert ledger.select(mode="minimum")["selected"]["candidate_id"] == best["candidate_id"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("place", ["measurement", "candidate", "check"])
def test_nonfinite_numbers_never_become_measurements(tmp_path, value, place):
    with make_ledger(tmp_path) as ledger:
        kwargs = {"candidate": rows(), "measurements": {"score": 1.}, "checks": {}}
        if place == "measurement":
            kwargs["measurements"]["score"] = value
        elif place == "candidate":
            kwargs["candidate"]["elements"][0]["k1"] = value
        else:
            kwargs["checks"] = {"limit": {"passed": False, "violation": value}}
        with pytest.raises(ValueError, match="finite"):
            ledger.record(**kwargs)
        assert ledger.entries() == []


@pytest.mark.parametrize("check", [{"passed": True, "violation": 1.}, {"passed": False, "violation": -1.},
                                  {"passed": 1}, {"passed": "yes"}])
def test_inconsistent_checks_are_rejected(tmp_path, check):
    with make_ledger(tmp_path) as ledger, pytest.raises(ValueError):
        record(ledger, 1., passed=check)


def test_nullable_missing_or_boolean_objectives_cannot_pass(tmp_path):
    with make_ledger(tmp_path) as ledger:
        for value in (None, True, "1.2"):
            assert not record(ledger, value)["feasible"]
        assert not ledger.record(candidate=rows(), measurements={}, checks={})["feasible"]


def test_nonlattice_experiments_preserve_their_own_parameters(tmp_path):
    with make_ledger(tmp_path) as ledger:
        parameters = {"frequency_hz": 499999999.123456, "phase_rad": -.01723456789012345,
                      "calibration": {"coefficients": [0., .12, -.0003]}}
        entry = record(ledger, .5, candidate=parameters)
        assert ledger.select()["selected"]["candidate"] == parameters
        assert not entry["evidence"]["physics_accepted"]


@pytest.mark.parametrize("changed", [{"contract": {"energy": 4}}, {"objectives": {"score": "max"}}, {"resume": False}])
def test_resume_cannot_change_contract_or_objectives(tmp_path, changed):
    make_ledger(tmp_path).close()
    with pytest.raises(ValueError):
        make_ledger(tmp_path, **changed)


def test_objective_order_is_part_of_resume_contract(tmp_path):
    make_ledger(tmp_path, objectives={"a": "min", "b": "max"}).close()
    with pytest.raises(ValueError, match="contract"):
        make_ledger(tmp_path, objectives={"b": "max", "a": "min"})


def test_scope_and_copied_ledger_cannot_cross_projects(tmp_path, monkeypatch):
    current, other = tmp_path / "current", tmp_path / "other"
    with make_ledger(current) as ledger:
        record(ledger, 1.)
        monkeypatch.setenv("BRILLIANCE_WORKSPACE", str(other))
        with pytest.raises(ValueError, match="managed workspace"):
            ledger.select()
        with pytest.raises(ValueError, match="configured"):
            make_ledger(current)
    monkeypatch.delenv("BRILLIANCE_WORKSPACE")
    other.mkdir()
    shutil.copytree(current / "experiment", other / "experiment")
    with pytest.raises(ValueError, match="scope"):
        CandidateLedger.open(other / "experiment", workspace=other)
    with pytest.raises(ValueError, match="inside"):
        CandidateLedger(current / "experiment", contract={}, objectives={"a": "min"}, workspace=other)
    with pytest.raises(ValueError, match="explicit workspace"):
        CandidateLedger.open(current / "experiment")


def test_read_only_disk_recovery_never_runs_or_reconstructs_callback(tmp_path, monkeypatch):
    with make_ledger(tmp_path) as ledger:
        selected = record(ledger, 2.)
    path = tmp_path / "experiment" / "candidates.sqlite3"
    before = path.read_bytes()
    import brilliance.accelerator_api as api
    monkeypatch.setattr(api, "evaluate", lambda *a, **k: pytest.fail("Disk-only recovery must not simulate"))
    with CandidateLedger.open(path.parent, workspace=tmp_path) as reader:
        assert reader.select()["selected"] == selected
        with pytest.raises(ValueError, match="read-only"):
            record(reader, 1.)
    assert path.read_bytes() == before


def test_recent_entries_query_and_verification_are_bounded_not_a_selection_pool(tmp_path, monkeypatch):
    with make_ledger(tmp_path) as ledger:
        first = record(ledger, -1)
        later = [record(ledger, i) for i in range(20)]
        validated, queries = [], []
        original_validate = ledger._validated_entry
        def observe_validation(candidate_id, raw):
            validated.append(candidate_id)
            return original_validate(candidate_id, raw)
        monkeypatch.setattr(ledger, "_validated_entry", observe_validation)
        ledger._local.connection.set_trace_callback(queries.append)
        recent = ledger.recent_entries(3)
        assert [entry["candidate_id"] for entry in recent] == [entry["candidate_id"] for entry in reversed(later[-3:])]
        assert len(validated) == 3
        assert any("ORDER BY sequence DESC LIMIT 3" in query for query in queries)
        assert ledger.select()["selected"]["candidate_id"] == first["candidate_id"]
        assert len(ledger.recent_entries(100)) == 21
    with CandidateLedger.open(tmp_path / "experiment", workspace=tmp_path) as reader:
        assert reader.recent_entries(1) == [later[-1]]


@pytest.mark.parametrize("limit", [0, -1, True, 1.0, "2", None])
def test_recent_entries_rejects_unbounded_or_noninteger_limits(tmp_path, limit):
    with make_ledger(tmp_path) as ledger, pytest.raises(ValueError, match="positive integer"):
        ledger.recent_entries(limit)


def test_sql_update_delete_and_payload_tamper_are_detected(tmp_path):
    with make_ledger(tmp_path) as ledger:
        record(ledger, 2.)
        with sqlite3.connect(ledger.path) as connection:
            for sql in ("DELETE FROM candidates", "UPDATE candidates SET payload='{}'", "DELETE FROM manifest"):
                with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                    connection.execute(sql)
            connection.execute("DROP TRIGGER immutable_candidates_UPDATE")
            entry = json.loads(connection.execute("SELECT payload FROM candidates").fetchone()[0])
            entry["measurements"]["score"] = 0.
            connection.execute("UPDATE candidates SET payload=?", (json.dumps(entry),))
        with pytest.raises(ValueError, match="content hash"):
            ledger.select()
        with pytest.raises(ValueError, match="content hash"):
            ledger.recent_entries(1)


def test_manifest_tampering_detected_on_recovery(tmp_path):
    with make_ledger(tmp_path) as ledger:
        with sqlite3.connect(ledger.path) as connection:
            connection.execute("DROP TRIGGER immutable_manifest_UPDATE")
            manifest = json.loads(connection.execute("SELECT payload FROM manifest").fetchone()[0])
            manifest["contract"]["energy"] = 99
            connection.execute("UPDATE manifest SET payload=?", (json.dumps(manifest),))
    with pytest.raises(ValueError, match="manifest hash"):
        CandidateLedger.open(tmp_path / "experiment", workspace=tmp_path)


def test_concurrent_initialization_and_appends_are_durable(tmp_path):
    barrier = Barrier(4)
    def writer(index):
        barrier.wait()
        with make_ledger(tmp_path) as ledger:
            return [record(ledger, index * 20 + attempt)["candidate_id"] for attempt in range(20)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        identifiers = [item for batch in pool.map(writer, range(4)) for item in batch]
    with CandidateLedger.open(tmp_path / "experiment", workspace=tmp_path) as reader:
        assert len(set(identifiers)) == len(reader.entries()) == 80
        assert reader.select()["selected"]["measurements"]["score"] == 0


def test_shared_instance_reuses_thread_connections_and_sees_all_external_writes(tmp_path):
    with make_ledger(tmp_path) as ledger:
        record(ledger, 100)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda value: record(ledger, value), range(20)))
        assert len(ledger._connections) <= 5
        clone = pickle.loads(pickle.dumps(ledger))
        with clone:
            best = record(clone, -1.)
        assert ledger.select()["counts"]["evaluated"] == 22
        assert ledger.select()["selected"]["candidate_id"] == best["candidate_id"]


def test_archived_receipt_binding_is_distinct_from_unverified_summary(tmp_path, monkeypatch):
    from brilliance import accelerator_api as api
    from brilliance import candidate_evidence as evidence
    subject = api.periodic_cell(title="Synthetic ledger test, not physics", source="Synthetic test only", energy_gev=3., intended_repetitions=8,
        elements=[api.drift("D", .3), api.sector_bend("B", .8, 2 * math.pi / 8)])
    calculation = {"engine": "JuTrack", "candidate_lattice_id": subject.lattice_id,
                   "linear_optics": {"stable": True}, "radiation": {"computed": True, "emittance_pm": 42.}}
    calculation["evaluation_receipt"] = evidence.record_evaluation(subject, calculation, effective_periods=8)
    archived = evidence.verify_candidate_evaluation(subject, calculation, tmp_path)
    assert archived["passed"]
    monkeypatch.setattr(api, "evaluate", lambda *a, **k: pytest.fail("Receipt verification must not simulate"))
    envelope = {"receipt": archived["receipt"], "output_directory": str(tmp_path)}
    with make_ledger(tmp_path) as ledger:
        entry = record(ledger, 1.0, candidate=subject.as_dict(), evaluation_receipt=envelope)
        assert entry["evidence"]["candidate_binding_verified"]
        assert not entry["evidence"]["measurements_verified"]
        assert not entry["evidence"]["physics_accepted"]
        changed = copy.deepcopy(subject.as_dict())
        changed["elements"][0]["length"] = .4
        with pytest.raises(ValueError, match="Unverified"):
            record(ledger, 1., candidate=changed, evaluation_receipt=envelope)
        with pytest.raises(ValueError, match="another workspace"):
            record(ledger, 1., candidate=subject.as_dict(), evaluation_receipt={**envelope, "output_directory": str(tmp_path.parent)})
        receipt_path = tmp_path / archived["receipt"]["path"]
        receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
        with pytest.raises(ValueError, match="Unverified"):
            ledger.select()
        with pytest.raises(ValueError, match="Unverified"):
            ledger.recent_entries(1)


def test_thousand_77_element_snapshots_use_one_database_not_per_trial_files(tmp_path):
    """Storage-only benchmark; timings are observational, never a physics claim."""
    from brilliance.atomic_io import atomic_write_text
    candidate = {"elements": [{"name": f"E{i:03}", "length": .123456789012345,
                              "k1": .7346199519999999, "angle": .0031234567890123,
                              "fields": {"PolynomB": [0., .7346199519999999, 0., 0.],
                                         "NumIntSteps": 17}} for i in range(77)]}
    payloads = []
    with make_ledger(tmp_path) as ledger:
        started = perf_counter()
        for index in range(1000):
            candidate["elements"][0]["k1"] = .73 + index / 10000.
            entry = record(ledger, 1000 - index, candidate=candidate)
            payloads.append(json.dumps(entry, sort_keys=True, separators=(",", ":")))
        ledger_seconds = perf_counter() - started
        started = perf_counter()
        selection = ledger.select()
        selection_seconds = perf_counter() - started
        assert selection["counts"]["evaluated"] == 1000
        assert selection["selected"]["measurements"]["score"] == 1
        assert len(ledger._connections) == 1
        assert not list(ledger.directory.glob("*.json"))
        assert len(list(ledger.directory.iterdir())) <= 3  # database, WAL, shared-memory
    baseline = tmp_path / "per_evaluation_files"
    baseline.mkdir()
    started = perf_counter()
    for index, payload in enumerate(payloads):
        atomic_write_text(baseline / f"{index:04}.json", payload)
    file_seconds = perf_counter() - started
    print(json.dumps({"storage_only_benchmark": {"records": 1000, "elements_per_record": 77,
        "sqlite_append_seconds": round(ledger_seconds, 3), "all_history_select_seconds": round(selection_seconds, 3),
        "preencoded_atomic_file_seconds": round(file_seconds, 3), "database_bytes": ledger.path.stat().st_size,
        "atomic_file_count": len(list(baseline.iterdir()))}}))
