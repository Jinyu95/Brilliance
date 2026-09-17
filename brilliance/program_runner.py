"""Execute a saved research program with bounded failure diagnostics."""

from __future__ import annotations

import ipaddress
import json
import linecache
import os
import re
import runpy
import sys
import traceback
from pathlib import Path
from typing import Any

from .numerical_checkpoints import ReportingStopped
from .research_outcomes import ResearchInfeasible

_SENSITIVE = ("key", "token", "secret", "password", "credential")


def _offline_network_audit(event: str, args: tuple[Any, ...]) -> None:
    """Block Python internet clients before DNS/connect; retain local worker IPC.

    This is an accidental-access guard, not an OS sandbox for hostile/native code.
    The model-provider connection is in a different process and is unaffected.
    """
    host = None
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        host = args[0]
    elif event in {"socket.connect", "socket.sendto"}:
        address = args[-1]
        if isinstance(address, tuple):
            host = address[0]
    if host is None:
        return
    text = host.decode("ascii", errors="replace") if isinstance(host, bytes) else str(host)
    if text.lower() == "localhost":
        return
    try:
        if ipaddress.ip_address(text).is_loopback:
            return
    except ValueError:
        pass
    raise PermissionError("Internet access is disabled for research programs; only loopback connections are allowed")


def _safe_local(name: str, value: Any) -> Any:
    lowered = name.casefold()
    if any(term in lowered for term in _SENSITIVE):
        return "<redacted>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:300]
    if type(value) is dict:
        # Key names resolve a missing-field error without dumping numerical
        # archives, credentials, table rows or private values into model context.
        keys = [key[:80] for key in value if isinstance(key, str)
                and not any(term in key.casefold() for term in _SENSITIVE)]
        return {"type": "dict", "length": len(value), "keys": keys[:24],
                "keys_truncated": len(keys) > 24}
    if isinstance(value, (list, tuple, set, dict)):
        return f"<{type(value).__name__} length={len(value)}>"
    return f"<{type(value).__name__}>"


def _failure_locals(frame: Any, *, max_chars: int = 2400) -> dict[str, Any]:
    """Keep the failing expression's locals first, within the excerpt budget."""
    source = linecache.getline(frame.tb_frame.f_code.co_filename, frame.tb_lineno)
    relevant = set(re.findall(r"\b[A-Za-z_]\w*\b", source))
    values = frame.tb_frame.f_locals
    result: dict[str, Any] = {}
    for name in sorted(values, key=lambda item: (item not in relevant, item)):
        if name.startswith("__"):
            continue
        value = _safe_local(name, values[name])
        proposed = {**result, name: value}
        if len(json.dumps(proposed, ensure_ascii=True)) <= max_chars:
            result = proposed
    return result


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        print("usage: python -m brilliance.program_runner PROGRAM.py", file=sys.stderr)
        return 2
    path = Path(values[0]).resolve()
    if os.environ.get("BRILLIANCE_INTERNET_ACCESS", "false").lower() not in {"true", "1", "yes"}:
        sys.addaudithook(_offline_network_audit)
    try:
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except ReportingStopped:
        # The explicit reporting-only scope already committed a verified partial
        # result and original traceback. Do not rerun numerical work or continue
        # success-dependent statements after the broken presentation block.
        return 0
    except ResearchInfeasible as exc:
        # A negative scientific result is evidence, not a broken Python program.
        # Stop this program; do not execute its success-only postprocessing or
        # silently relax the bounds. Preserve the complete failed attempt.
        from .research_api import _write_json, open_research_run

        run = open_research_run()
        diagnostic = run.artifact_path("infeasible_attempt.json")
        _write_json(diagnostic, {"code": exc.code, "summary": str(exc), "diagnostics": exc.diagnostics})
        rows = exc.diagnostics.get("target_observations", [])
        run.finish(summary=str(exc), status="infeasible",
            data={"outcome": exc.code, "optimization_blocker": str(exc),
                  "diagnostics": exc.diagnostics},
            tables=[{"title": "Unreached matching targets", "rows": rows}] if rows else [],
            downloads=[{"path": diagnostic, "title": "Negative-result diagnostics and attempted elements"}],
            limitations=["This attempt did not meet its constraints; the original research request remains incomplete."],
            scientific_passed=False, request_satisfied=False)
        return 0
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    except BaseException as exc:  # noqa: BLE001 - boundary must diagnose generated code
        traceback.print_exc()
        frame = exc.__traceback__
        while frame is not None and frame.tb_next is not None:
            frame = frame.tb_next
        if frame is not None:
            locals_payload = _failure_locals(frame)
            print(
                "BRILLIANCE_FAILURE_LOCALS="
                + json.dumps(locals_payload, ensure_ascii=True, sort_keys=True),
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
