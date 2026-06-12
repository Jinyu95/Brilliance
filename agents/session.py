"""Session management — saves all code, plots and conversation artifacts."""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from .constraints import parse_design_constraints
from physics_core import parse_result_output, score_design_result

SESSION_DIR = Path(__file__).resolve().parent.parent / "sessions"
SESSION_DIR.mkdir(exist_ok=True)


class Session:
    """Manages a single design session's artifacts."""

    def __init__(self, request: str):
        self._start_time = datetime.now()
        ts = self._start_time.strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^\w\s-]", "", request[:40]).strip().replace(" ", "_")
        self.dir = SESSION_DIR / f"{ts}_{safe}"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.code_dir = self.dir / "code"
        self.code_dir.mkdir(exist_ok=True)
        self.plots_dir = self.dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.request = request
        self.messages: list[dict] = []
        self._code_count = 0

    # ── message recording ─────────────────────────────────────────

    def add_message(self, source: str, content: str, msg_type: str = "text"):
        """Record a message and auto-extract any ``python`` code blocks."""
        entry = {
            "source": source,
            "content": content,
            "type": msg_type,
            "timestamp": datetime.now().isoformat(),
        }
        self.messages.append(entry)

        for code in re.findall(r"```python\s*\n(.*?)```", content, re.DOTALL):
            self._code_count += 1
            path = self.code_dir / f"lattice_v{self._code_count}.py"
            path.write_text(code.strip(), encoding="utf-8")

    # ── artifact collection ───────────────────────────────────────

    def collect_plots(self, workspace_dir: Path):
        """Copy image files generated during this session from workspace_dir."""
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.pdf", "*.svg"):
            for img in workspace_dir.glob(ext):
                try:
                    mtime = datetime.fromtimestamp(img.stat().st_mtime)
                    if mtime >= self._start_time:
                        shutil.copy2(img, self.plots_dir / img.name)
                except OSError:
                    pass

    # ── persistence ───────────────────────────────────────────────

    def save(self) -> Path:
        """Write conversation log and a human-readable summary."""
        (self.dir / "conversation.json").write_text(
            json.dumps(self.messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        lines = [
            "# Design Session\n",
            f"**Request:** {self.request}",
            f"**Time:** {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Messages:** {len(self.messages)}",
            f"**Code versions:** {self._code_count}",
            f"**Plots:** {len(list(self.plots_dir.glob('*')))}",
            "",
        ]

        final_result = latest_result_snapshot(
            self.messages,
            fallback_context=latest_user_request(self.messages, fallback=self.request),
        )
        if final_result:
            lines += [
                "## Final Results",
                "```json",
                json.dumps(final_result["result"], indent=2, ensure_ascii=False),
                "```",
                f"Score: {final_result['assessment']['score']} ({final_result['assessment']['status']})",
                "",
            ]

        (self.dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        return self.dir


# ── helpers ───────────────────────────────────────────────────────


def list_sessions() -> list[dict]:
    """Return metadata for all saved sessions, newest first."""
    result = []
    if not SESSION_DIR.exists():
        return result
    for d in sorted(SESSION_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        code_dir = d / "code"
        plots_dir = d / "plots"
        request_text = ""
        summary = d / "summary.md"
        if summary.exists():
            for line in summary.read_text(encoding="utf-8").splitlines():
                if line.startswith("**Request:**"):
                    request_text = line.replace("**Request:**", "").strip()
                    break
        result.append(
            {
                "path": str(d),
                "name": d.name,
                "request": request_text,
                "code_count": len(list(code_dir.glob("*.py"))) if code_dir.exists() else 0,
                "plot_count": len(list(plots_dir.glob("*"))) if plots_dir.exists() else 0,
            }
        )
    return result


def latest_user_request(messages: list[dict], *, fallback: str = "") -> str:
    """Return the most recent user request captured in the session."""
    for message in reversed(messages):
        if message.get("source") == "user" and isinstance(message.get("content"), str):
            text = message["content"].strip()
            if text:
                return text
    return fallback


def latest_result_snapshot(
    messages: list[dict],
    *,
    fallback_context: str | None = None,
) -> dict | None:
    """Return the most recent parsed result/assessment pair."""
    for msg in reversed(messages):
        parsed = parse_lattice_results(msg.get("content", ""))
        if parsed:
            assessment = assess_lattice_results(parsed, context_text=fallback_context)
            return {"result": parsed, "assessment": assessment}
    return None


def parse_lattice_results(text: str) -> dict | None:
    """Extract key metrics from structured or legacy lattice result output."""
    parsed = parse_result_output(text)
    if not parsed:
        return None

    aliases = {
        "nux": parsed.get("tune_nux", parsed.get("nux")),
        "nuy": parsed.get("tune_nuy", parsed.get("nuy")),
        "xix": parsed.get("chromaticity_xix", parsed.get("xix")),
        "xiy": parsed.get("chromaticity_xiy", parsed.get("xiy")),
    }
    parsed.update({key: value for key, value in aliases.items() if value is not None})
    return parsed


def assess_lattice_results(
    result: dict | None,
    *,
    context_text: str | None = None,
    constraints: dict | None = None,
) -> dict:
    """Score parsed lattice results using deterministic review rules."""
    resolved_constraints = constraints or parse_design_constraints(context_text)
    # Forward the FULL constraint dict — not just 4 keys — so every
    # parsed field (phase_advance, DA, MA, chrom target, etc.) reaches
    # the scorer and can influence the score and issues list.
    assessment = score_design_result(
        result,
        target_emittance=resolved_constraints.get("target_emittance"),
        target_circumference=resolved_constraints.get("circumference_m"),
        target_energy_gev=resolved_constraints.get("energy_gev"),
        machine_class=resolved_constraints.get("machine_class"),
        full_constraints=resolved_constraints,
    )
    assessment["constraints"] = resolved_constraints
    return assessment
