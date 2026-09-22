"""Read-only process telemetry. Activity is not scientific progress or acceptance."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

_samples: dict = {}
_lock = Lock()


def process_activity(pid: int) -> dict:
    try:
        import psutil
    except ImportError:
        return {"status": "unavailable", "detail": "Process telemetry dependency unavailable"}
    try:
        parent = psutil.Process(pid)
        identity = (pid, parent.create_time())
        members = [parent, *parent.children(recursive=True)]
        cpu, memory = 0.0, 0
        for member in members:
            try:
                times = member.cpu_times()
                cpu += times.user + times.system
                memory += member.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        now = time.monotonic()
        with _lock:
            previous = _samples.get(identity)
            rate = None
            if previous and now - previous[0] >= 1:
                rate = max(0, cpu - previous[1]) / (now - previous[0]) * 100
            if not previous or now - previous[0] >= 1:
                _samples[identity] = (now, cpu, rate)
            else:
                rate = previous[2]
            if len(_samples) > 256:
                del _samples[next(iter(_samples))]
        return {"status": "alive", "pid": pid, "process_count": len(members),
                "cpu_seconds": round(cpu, 2), "cpu_percent": round(rate, 1) if rate is not None else None,
                "memory_mb": round(memory / 1024**2, 1)}
    except psutil.NoSuchProcess:
        return {"status": "exited", "pid": pid}
    except (psutil.AccessDenied, OSError):
        return {"status": "unavailable", "pid": pid}


def age_seconds(value, now):
    try:
        return max(0, now - datetime.fromisoformat(str(value)).timestamp())
    except (ValueError, TypeError):
        return None


def observe_calculation(workspace: Path, progress_path: Path | None, progress: dict) -> dict:
    now = time.time()
    result = {"progress_age_s": age_seconds(progress.get("updated_at") or progress.get("timestamp"), now),
              "status": "unknown", "files": []}
    if progress_path is None:
        result["message"] = "No numerical progress report has been received."
        return result
    output = workspace / "artifacts" / "programs" / progress_path.stem
    try:
        runtime = json.loads((output / "runtime_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        runtime = {}
    result["heartbeat_age_s"] = age_seconds(runtime.get("updated_at"), now)
    pid = runtime.get("pid")
    result["process"] = process_activity(pid) if isinstance(pid, int) and pid > 0 else {"status": "unknown"}
    result["program"] = progress.get("metrics", {}).get("program")
    # Metadata only: never promote arbitrary checkpoint contents to accepted results.
    for path in output.glob("*"):
        if path.name in {"runtime_status.json", "program.py", "arguments.json"} or not path.is_file():
            continue
        try:
            stat = path.stat()
            if stat.st_size:
                result["files"].append({"name": path.name, "bytes": stat.st_size, "age_s": max(0, now-stat.st_mtime)})
        except OSError:
            continue
    result["files"] = sorted(result["files"], key=lambda row: row["age_s"])[:6]
    stale = result["progress_age_s"] is None or result["progress_age_s"] > 120
    if result["process"]["status"] == "exited":
        result.update(status="process_exited", message="Calculation process exited; waiting for the job's final receipt. No success is inferred.")
    elif result["heartbeat_age_s"] is None or result["heartbeat_age_s"] > 90:
        result.update(status="heartbeat_stale", message="Runner heartbeat is missing or stale. Job health is uncertain; no automatic cancellation was requested.")
    elif stale:
        result.update(status="progress_stale", message="Runner is responding, but numerical progress has not been reported recently. CPU activity does not prove convergence; a stall cannot be ruled out.")
    else:
        result.update(status="reporting", message="Recent numerical status received. Process activity and scientific acceptance are separate.")
    return result
