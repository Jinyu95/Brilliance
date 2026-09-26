"""A durable model-turn allowance shared by every handoff of a scientist request.

Numerical jobs do not consume model turns. Within a project, internal notifications
never replenish a request's budget. A new scientist request or an explicit
continuation into a new checkpoint branch starts a fresh allowance.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .artifacts import content_hash
from .atomic_io import atomic_write_text
from .research_environment import ResearchEnvironment


@contextmanager
def _locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "continuation.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 5
        while True:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.02)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _allowance(root: Path | str, project_id: str, origin: Mapping[str, Any], *,
               automatic: bool, job_id: str | None = None, report_only: bool = False) -> dict[str, Any]:
    from .research_recovery import scientist_request

    directory = Path(root).resolve() / project_id
    request = dict(origin)
    latest = scientist_request(Path(root), project_id)
    if not request.get("id") or request.get("id") != latest.get("id"):
        return {"allowed": False, "reason": "request_changed_or_not_recorded"}
    if (directory / "agent_pause.request").exists():
        return {"allowed": False, "reason": "scientist_paused_or_cancelled"}
    policy = ResearchEnvironment.load()
    with _locked(directory / "codex"):
        if request.get("id") != scientist_request(Path(root), project_id).get("id"):
            return {"allowed": False, "reason": "request_changed_or_not_recorded"}
        if (directory / "agent_pause.request").exists():
            return {"allowed": False, "reason": "scientist_paused_or_cancelled"}
        path = directory / "workspace" / "request_budgets" / (content_hash(request["id"]) + ".json")
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
        else:
            state = {"schema_version": "1.0", "scientist_request_id": request["id"],
                     "max_increments": max(1, int(policy.max_autonomous_increments)), "turns": []}
        turns = state["turns"]
        limit = int(state["max_increments"])
        if automatic and not policy.auto_continue:
            return {"allowed": False, "reason": "auto_continue_disabled",
                    "increment": len(turns) + 1, "max_increments": limit}
        if report_only:
            if len(turns) < limit:
                return {"allowed": False, "reason": "research_budget_remaining"}
            reserved = state.get("report_turn")
            if reserved and reserved != job_id:
                return {"allowed": False, "reason": "report_already_reserved"}
            if job_id is not None and not reserved:
                state["report_turn"] = job_id
                atomic_write_text(path, json.dumps(state, indent=2))
            return {"allowed": True, "reason": "", "increment": limit + 1,
                    "max_increments": limit + 1, "scientist_request": request}
        # Retrying the same claimed job never consumes a second turn.
        if job_id is not None and job_id in turns:
            number = turns.index(job_id) + 1
        else:
            number = len(turns) + 1
            if number > limit:
                return {"allowed": False, "reason": "request_budget_exhausted",
                        "increment": len(turns), "max_increments": limit}
            if job_id is not None:
                turns.append(job_id)
                atomic_write_text(path, json.dumps(state, indent=2))
        return {"allowed": True, "reason": "", "increment": number,
                "max_increments": limit, "scientist_request": request}


def turn_allowance(root: Path | str, project_id: str, origin: Mapping[str, Any], *,
                   automatic: bool = True) -> dict[str, Any]:
    """Inspect the remaining budget without charging for a notification."""
    return _allowance(root, project_id, origin, automatic=automatic)


def request_started(root: Path | str, project_id: str, origin: Mapping[str, Any]) -> bool:
    """Whether a model turn has already been charged to this conversation entry."""
    return (Path(root) / project_id / "workspace" / "request_budgets" /
            (content_hash(origin.get("id", "")) + ".json")).is_file()


def reserve_turn(root: Path | str, project_id: str, origin: Mapping[str, Any], job_id: str, *,
                 automatic: bool = True) -> dict[str, Any]:
    """Charge one actual model invocation, idempotently by durable job identity."""
    return _allowance(root, project_id, origin, automatic=automatic, job_id=job_id)


def report_allowance(root: Path | str, project_id: str, origin: Mapping[str, Any],
                     job_id: str | None = None) -> dict[str, Any]:
    """One protected closing report, never another scientific work allowance."""
    return _allowance(root, project_id, origin, automatic=True, job_id=job_id, report_only=True)


def handoff_payload(allowance: Mapping[str, Any]) -> dict[str, Any]:
    if not allowance.get("allowed"):
        raise ValueError("A stopped research request cannot schedule another model turn")
    return {key: allowance[key] for key in ("increment", "max_increments", "scientist_request")}


def pause_explanation(reason: str) -> str:
    return {
        "auto_continue_disabled": "Automatic continuation is disabled. Send a follow-up to resume the unfinished request.",
        "request_budget_exhausted": "The model-turn budget for this request has been reached. Send a follow-up to grant another research budget or change direction.",
        "scientist_paused_or_cancelled": "Research continuation is paused at your request.",
        "request_changed_or_not_recorded": "A newer scientist request superseded this automatic continuation.",
    }.get(reason, "Research continuation could not be started.") + " Saved results and independent numerical jobs are preserved."
