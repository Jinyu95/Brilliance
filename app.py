#!/usr/bin/env python
"""Streamlit web UI — JuTrack assistant with a 4-agent pipeline.

Pipeline: TaskPlanner → CodeWriter → CodeRunner → CodeReviewer
Each agent step requires explicit user action.

Launch with:
    streamlit run app.py
"""

import asyncio
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# UTF-8 fix for Chinese Windows (must be before other imports)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).resolve().parent))
ASSETS = Path(__file__).resolve().parent / "assets"

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.config import WORKSPACE_DIR
from agents.knowledge_graph import parse_colon_block
from agents.session import (
    Session,
    assess_lattice_results,
    latest_result_snapshot,
    latest_user_request,
    list_sessions,
    parse_lattice_results,
)
from agents.team import (
    call_agent,
    PLANNER,
    CODEWRITER,
    RUNNER,
    REVIEWER,
)

# ── page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="BRILLIANCE — Accelerator Physics & Design",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────
AGENT_CFG = {
    PLANNER:    {"label": "Task Planner",  "icon": "🎯", "color": "#6c3ce0"},
    CODEWRITER: {"label": "Code Writer",   "icon": "✍️", "color": "#0088cc"},
    RUNNER:     {"label": "Code Runner",   "icon": "▶️",  "color": "#00966e"},
    REVIEWER:   {"label": "Code Reviewer", "icon": "🔬", "color": "#d14050"},
    "user":     {"label": "You",           "icon": "👤", "color": "#3d4f6f"},
}

PIPELINE_STEPS = [
    (PLANNER,    "Plan"),
    (CODEWRITER, "Code"),
    (RUNNER,     "Run"),
    (REVIEWER,   "Review"),
]


def _img_base64(filename: str) -> str:
    """Return a data-URI for an image asset (svg/png/jpg/webp)."""
    img_path = ASSETS / filename
    if not img_path.exists():
        return ""
    suffix = img_path.suffix.lower()
    mime = {
        ".svg": "image/svg+xml", ".png": "image/png",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


# ── CSS (light high-tech theme) ───────────────────────────────────
st.markdown("""
<style>
/* ── Import fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ───────────────────────────────────────── */
:root {
    --bg-primary:    #f5f7fb;
    --bg-secondary:  #ffffff;
    --bg-card:       #ffffff;
    --bg-glass:      rgba(255, 255, 255, 0.85);
    --border-subtle: rgba(60, 80, 130, 0.12);
    --border-accent: rgba(0, 140, 210, 0.25);
    --accent-cyan:   #0088cc;
    --accent-purple: #6c3ce0;
    --accent-green:  #00b894;
    --accent-red:    #e74c5e;
    --accent-orange: #f5a623;
    --text-primary:  #1a2332;
    --text-secondary:#3d4f6f;
    --text-muted:    #7a8ba8;
    --gradient-main: linear-gradient(135deg, #0088cc 0%, #6c3ce0 50%, #00b894 100%);
    --shadow-soft:   0 2px 12px rgba(0, 0, 0, 0.06);
    --shadow-card:   0 4px 20px rgba(60, 80, 130, 0.08);
    --shadow-accent: 0 4px 16px rgba(0, 136, 204, 0.12);
}

/* ── Global overrides ─────────────────────────────────────── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #b0bec5; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fafbfe 0%, #f0f2f8 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.3px;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* ── Header / hero ────────────────────────────────────────── */
.hero-container {
    position: relative;
    padding: 28px 36px 24px;
    margin: -1rem -1rem 24px -1rem;
    background-color: #e8edf5;
    background-size: cover;
    background-position: center;
    border-bottom: 1px solid var(--border-subtle);
    overflow: hidden;
    min-height: 120px;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(
        135deg,
        rgba(245,247,251,0.88) 0%,
        rgba(240,242,248,0.75) 40%,
        rgba(238,242,250,0.70) 100%
    );
    pointer-events: none;
    z-index: 0;
}
.hero-flex {
    display: flex;
    align-items: center;
    gap: 18px;
    position: relative;
    z-index: 1;
}
.hero-ihep-logo {
    height: 52px;
    width: auto;
    flex-shrink: 0;
    margin-left: auto;
    filter: drop-shadow(0 1px 4px rgba(0,0,0,0.10));
}
.hero-logo {
    width: 56px; height: 56px;
    flex-shrink: 0;
    filter: drop-shadow(0 2px 8px rgba(0,136,204,0.25));
}
.hero-text {
    flex: 1;
}
.hero-text h1 {
    margin: 0;
    font-size: 1.65rem;
    font-weight: 700;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.hero-text p {
    margin: 4px 0 0;
    font-size: 0.9rem;
    color: var(--text-secondary);
    letter-spacing: 0.2px;
    font-weight: 400;
}

/* ── Pipeline progress indicator ──────────────────────────── */
.pipeline-bar {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 14px 0 18px;
    padding: 10px 18px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    box-shadow: var(--shadow-soft);
}
.pipe-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-muted);
    transition: all 0.3s ease;
    white-space: nowrap;
}
.pipe-step.active {
    background: rgba(0, 136, 204, 0.08);
    color: var(--accent-cyan);
    box-shadow: 0 0 0 1px rgba(0, 136, 204, 0.2);
    font-weight: 600;
}
.pipe-step.done {
    color: var(--accent-green);
}
.pipe-arrow {
    color: var(--text-muted);
    font-size: 0.75rem;
    margin: 0 4px;
    opacity: 0.35;
}

/* (feature cards removed) */

/* ── Chat messages ────────────────────────────────────────── */
div[data-testid="stChatMessage"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    box-shadow: var(--shadow-soft) !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s ease !important;
    padding: 8px 20px !important;
    box-shadow: var(--shadow-soft) !important;
}
.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--shadow-accent) !important;
    transform: translateY(-1px) !important;
    color: var(--accent-cyan) !important;
}
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #0088cc, #6c3ce0) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(0,136,204,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(0,136,204,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Info / warning / success boxes ───────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* ── Metrics ──────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 16px !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-card) !important;
}
div[data-testid="stMetric"] label { color: var(--text-muted) !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Code blocks ──────────────────────────────────────────── */
.stCodeBlock {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
}

/* ── Expanders ────────────────────────────────────────────── */
details {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    font-weight: 500 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

/* ── Chat input ───────────────────────────────────────────── */
div[data-testid="stChatInput"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-soft) !important;
}

/* ── Captions ─────────────────────────────────────────────── */
.agent-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-bottom: 4px;
}

/* ── Hide defaults ────────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: rgba(245, 247, 251, 0.92) !important;
    backdrop-filter: blur(8px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── session-state defaults ────────────────────────────────────────
_DEFAULTS = {
    "messages": [],       # [{"source", "content", "type", "thoughts"?}]
    "session_obj": None,  # Session instance
    "last_agent": None,   # name of the agent that spoke last (or None)
    "viewing": None,      # path to a past session being viewed
    "_inspector_resim_result": None,  # cached re-simulation output
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _run_async(coro):
    """Run an async coroutine; force ProactorEventLoop on Windows."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            # Cancel lingering tasks (e.g. httpx AsyncClient cleanup)
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass
        loop.close()


def _push_config():
    """Copy sidebar widget values into environment variables."""
    os.environ["LLM_MODEL"] = st.session_state.get("_cfg_model", os.environ.get("LLM_MODEL", "deepseek-chat"))
    os.environ["LLM_API_KEY"] = st.session_state.get("_cfg_key", os.environ.get("LLM_API_KEY", ""))
    os.environ["LLM_BASE_URL"] = st.session_state.get("_cfg_url", os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    os.environ["CODE_TIMEOUT"] = str(st.session_state.get("_cfg_timeout", int(os.environ.get("CODE_TIMEOUT", "600"))))
    # Do NOT overwrite EXECUTOR — read-only from env.  User changes go through the selectbox
    # which writes to st.session_state._cfg_exec, but _push_config should not reset it.
    _cfg_exec = st.session_state.get("_cfg_exec")
    if _cfg_exec is not None:
        os.environ["EXECUTOR"] = _cfg_exec


def _show_inline_plots():
    """Display images from workspace that were created during the current session."""
    sess = st.session_state.session_obj
    if not sess:
        return
    from datetime import datetime
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
        for img in sorted(WORKSPACE_DIR.glob(ext)):
            try:
                mtime = datetime.fromtimestamp(img.stat().st_mtime)
                if mtime >= sess._start_time:
                    st.image(str(img), caption=img.name, use_container_width=True)
            except OSError:
                pass


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#rrggbb' to 'r,g,b' for use in rgba()."""
    h = str(hex_color or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in h):
        h = "888888"
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _truncate_inline(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _format_live_metric(value, *, scientific: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if scientific:
        return f"{numeric:.3e}"
    return f"{numeric:.3f}"


def _summarize_structured_block(block_name: str, block_body: str) -> str:
    fields = parse_colon_block(block_body)
    if not fields:
        return _truncate_inline(block_body)
    return _truncate_inline(block_body)


def _summarize_runner_output(text: str) -> str:
    parsed = parse_lattice_results(text)
    if parsed:
        return (
            f"stable={parsed.get('stable', 'n/a')}, "
            f"nux={_format_live_metric(parsed.get('tune_nux'))}, "
            f"nuy={_format_live_metric(parsed.get('tune_nuy'))}, "
            f"emit={_format_live_metric(parsed.get('emittance'), scientific=True)}"
        )

    for marker in (
        "Stable configuration found!",
        "Stable seed found",
        "No stable configurations found",
        "No stable configuration found",
        "Failed to get beta functions",
        "Traceback",
        "ERROR:",
    ):
        if marker in text:
            for line in text.splitlines():
                if marker in line:
                    return _truncate_inline(line)
    return _truncate_inline(text)


def _progress_summary(source: str, content: str, msg_type: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""

    if msg_type == "thought":
        return "reasoning updated"

    if msg_type == "status":
        lines = text.splitlines()
        header = lines[0].strip() if lines else ""
        fields = parse_colon_block("\n".join(lines[1:]))
        active = fields.get("active agent")
        step = fields.get("step")
        round_id = fields.get("round")
        if active and step and round_id:
            return f"round {round_id}: {active} - {step}"
        if active and step:
            return f"{active} - {step}"
        return _truncate_inline(text)

    fenced = re.search(r"```([^\n`]+)\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        return _summarize_structured_block(fenced.group(1).strip().upper(), fenced.group(2).strip())

    if source == RUNNER:
        return _summarize_runner_output(text)

    if source == CODEWRITER and "```python" in text:
        return "generated a candidate lattice script"

    if source == PLANNER and "task_type" in text.lower():
        fields = parse_colon_block(text)
        description = fields.get("description", "")
        stage_goal = fields.get("stage_goal", "")
        return _truncate_inline(f"{description} | {stage_goal}".strip(" |"))

    return _truncate_inline(text)


def _render_pipeline(active_agent=None):
    """Render the horizontal agent pipeline progress bar."""
    done_agents = set()
    for msg in st.session_state.messages:
        if msg["source"] in {s[0] for s in PIPELINE_STEPS}:
            done_agents.add(msg["source"])

    parts = []
    for i, (agent_key, label) in enumerate(PIPELINE_STEPS):
        cfg = AGENT_CFG.get(agent_key, {})
        icon = cfg.get("icon", "")
        if agent_key == active_agent:
            cls = "active"
        elif agent_key in done_agents:
            cls = "done"
        else:
            cls = ""
        parts.append(f'<div class="pipe-step {cls}">{icon} {label}</div>')
        if i < len(PIPELINE_STEPS) - 1:
            parts.append('<div class="pipe-arrow">▸</div>')

    st.markdown(
        '<div class="pipeline-bar">' + "".join(parts) + '</div>',
        unsafe_allow_html=True,
    )


def _render_msg(msg: dict):
    """Render one chat message with agent-colored tag."""
    src = msg["source"]
    msg_type = str(msg.get("type", "text"))
    cfg = AGENT_CFG.get(src, {"label": src, "icon": "💬", "color": "#888"})
    role = "user" if src == "user" else "assistant"
    with st.chat_message(role):
        color = cfg.get("color", "#888")
        icon = cfg.get("icon", "")
        st.markdown(
            f'<span class="agent-tag" style="background:rgba({_hex_to_rgb(color)},0.15);'
            f'color:{color};border:1px solid rgba({_hex_to_rgb(color)},0.3);">'
            f'{icon} {cfg["label"]}</span>',
            unsafe_allow_html=True,
        )
        content = msg["content"]
        if msg_type == "status":
            summary = _progress_summary(src, content, msg_type) or _truncate_inline(content)
            st.caption(summary)
            if len(str(content or "").strip()) > len(summary) + 24:
                with st.expander("Progress details", expanded=False):
                    st.text(str(content).strip()[:6000])
            return
        parts = re.split(r"(```\w*\n.*?```)", content, flags=re.DOTALL)
        for part in parts:
            m = re.match(r"```(\w*)\n(.*?)```", part, re.DOTALL)
            if m:
                with st.expander("Code", expanded=False):
                    st.code(m.group(2), language=m.group(1) or "text")
            elif part.strip():
                # Runner output is code execution stdout — render as plain text
                # to prevent '====...' lines becoming Markdown setext headings.
                if src == RUNNER:
                    st.text(part.strip()[:6000])
                else:
                    st.markdown(part[:6000])
        # Show any optics plots generated in the workspace (only for Runner output)
        if src == RUNNER:
            _show_inline_plots()


def _add_msg(source: str, content: str, msg_type: str = "text"):
    """Append a message and log it to the session object."""
    st.session_state.messages.append({
        "source": source, "content": content, "type": msg_type,
    })
    sess = st.session_state.session_obj
    if sess:
        sess.add_message(source, content, msg_type)


def _invoke(agent_name: str, extra_label: str = ""):
    """Call *agent_name*, store its response, and ``st.rerun()``."""
    label = extra_label or AGENT_CFG.get(agent_name, {}).get("label", agent_name)
    use_docker = os.environ.get("EXECUTOR", "local").lower() == "docker"
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))
    retry_delay_s = float(os.environ.get("LLM_RETRY_DELAY_S", "1.5"))

    result = None
    last_error = None

    with st.status(f"{label} is working ...", expanded=False) as status:
        for attempt in range(1, max_retries + 1):
            try:
                result = _run_async(
                    call_agent(agent_name, st.session_state.messages, use_docker=use_docker)
                )
                status.update(label=f"{label} finished", state="complete")
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    status.update(
                        label=f"{label} network issue; retrying ({attempt}/{max_retries - 1}) ...",
                        state="running",
                    )
                    time.sleep(retry_delay_s)
                else:
                    status.update(label=f"{label} failed", state="error")

    if result is None:
        err_text = str(last_error) if last_error else "Unknown error"
        friendly = (
            f"{label} could not reach the LLM service. "
            "Please check network/API endpoint and try again. "
            f"Details: {err_text}"
        )
        st.error(friendly)
        _add_msg("system", friendly)
        return

    if result.get("thoughts"):
        _add_msg(agent_name, result["thoughts"], "thought")

    _add_msg(agent_name, result["content"])
    st.session_state.last_agent = agent_name

    if agent_name == RUNNER and st.session_state.session_obj:
        st.session_state.session_obj.collect_plots(WORKSPACE_DIR)

    st.rerun()


# ── Lattice Inspector helpers ─────────────────────────────────────

# Parameter names that are never physics parameters
_INSPECTOR_SKIP = frozenset({
    "i", "j", "k", "n", "m", "x", "y", "z",
    "pi", "e", "step", "idx", "flag", "verbose", "debug",
    "nturns", "nturn", "fig", "ax", "cmap", "color",
})


def _extract_latest_script(messages: list[dict]) -> str | None:
    """Return the most recent Python script produced by CodeWriter."""
    for msg in reversed(messages):
        if msg.get("source") == CODEWRITER:
            m = re.search(r"```python\n(.*?)```", msg.get("content", ""), re.DOTALL)
            if m:
                return m.group(1)
    # Fallback: any code block
    for msg in reversed(messages):
        m = re.search(r"```python\n(.*?)```", msg.get("content", ""), re.DOTALL)
        if m:
            return m.group(1)
    return None


def _parse_script_params(script: str) -> list[dict]:
    """Extract top-level numeric parameter assignments from a script.

    Only scans lines before the first function/class/loop definition so
    the table stays focused on the physics parameter block.
    """
    params: list[dict] = []
    seen: set[str] = set()
    for line in script.splitlines():
        # Stop at the first block statement
        if re.match(r"^\s*(def |class |for |while |with |if __name__)", line):
            break
        m = re.match(
            r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*(?:#\s*(.*))?$",
            line,
        )
        if not m:
            continue
        name, val_str, comment = m.group(1), m.group(2), (m.group(3) or "")
        if name in seen or name in _INSPECTOR_SKIP or name.startswith("_"):
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue
        params.append({"parameter": name, "value": val, "notes": comment.strip()})
        seen.add(name)
    return params


def _patch_script_params(script: str, params: list[dict]) -> str:
    """Substitute parameter values in the script text."""
    for row in params:
        name = re.escape(row["parameter"])
        new_val = row["value"]
        # Replace only top-level assignments (line start, no indentation)
        script = re.sub(
            rf"^({name}\s*=\s*)[-+]?[\d.]+(?:[eE][-+]?\d+)?",
            lambda m, v=new_val: m.group(1) + repr(float(v)),
            script,
            flags=re.MULTILINE,
        )
    return script


def _run_patched_script(script: str, timeout: int = 120) -> tuple[str, dict | None]:
    """Write script to workspace and execute it.  Returns (stdout, parsed_result)."""
    from agents.config import WORKSPACE_DIR
    from agents.session import parse_lattice_results

    script_path = WORKSPACE_DIR / "_inspector_resim.py"
    script_path.write_text(script, encoding="utf-8")

    jutrack_python = os.environ.get("JUTRACK_PYTHON", "").strip()
    python_exe = (
        jutrack_python
        if jutrack_python and Path(jutrack_python).is_file()
        else sys.executable
    )

    try:
        proc = subprocess.run(
            [python_exe, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKSPACE_DIR),
        )
        stdout = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        stdout = f"[Timeout after {timeout}s]"
    except Exception as exc:
        stdout = f"[Execution error: {exc}]"

    parsed = parse_lattice_results(stdout)
    return stdout, parsed


def _render_lattice_inspector():
    """Show the interactive parameter editor + re-simulation widget."""
    import pandas as pd

    script = _extract_latest_script(st.session_state.messages)
    if not script:
        st.info("No generated script found in the current conversation.")
        return

    params = _parse_script_params(script)
    if not params:
        st.info("No editable numeric parameters found at the top of the generated script.")
        return

    st.caption(
        "Edit parameter values below, then click **Re-simulate** to run the patched "
        "script directly — no new agent call needed."
    )

    df_orig = pd.DataFrame(params)
    # Show the editor; 'notes' column is read-only metadata
    edited_df = st.data_editor(
        df_orig,
        column_config={
            "parameter": st.column_config.TextColumn("Parameter", disabled=True),
            "value": st.column_config.NumberColumn("Value", format="%.6g"),
            "notes": st.column_config.TextColumn("Notes", disabled=True),
        },
        use_container_width=True,
        hide_index=True,
        key="_inspector_editor",
    )

    col_run, col_reset = st.columns([3, 1])
    run_clicked = col_run.button("Re-simulate", type="primary", use_container_width=True, key="_inspector_run")
    if col_reset.button("Reset to original", use_container_width=True, key="_inspector_reset"):
        # Clear cached result so baseline shows alone
        st.session_state.pop("_inspector_resim_result", None)
        st.rerun()

    if run_clicked:
        patched = _patch_script_params(script, edited_df.to_dict("records"))
        timeout = int(os.environ.get("CODE_TIMEOUT", "120"))
        with st.status("Re-simulating with edited parameters ...", expanded=False) as s:
            stdout, parsed = _run_patched_script(patched, timeout=timeout)
            s.update(label="Re-simulation complete", state="complete")
        st.session_state["_inspector_resim_result"] = (stdout, parsed)

    # ── side-by-side comparison ───────────────────────────────────
    if st.session_state.get("_inspector_resim_result") is not None:
        orig_stdout_lines: list[str] = []
        for msg in reversed(st.session_state.messages):
            if msg.get("source") == RUNNER:
                orig_stdout_lines = msg.get("content", "").splitlines()
                break

        resim_stdout, resim_parsed = st.session_state["_inspector_resim_result"]

        col_orig, col_new = st.columns(2)
        with col_orig:
            st.markdown("**Baseline (last run)**")
            if orig_stdout_lines:
                st.text("\n".join(orig_stdout_lines[:40]))
            else:
                st.caption("No prior run output found.")
        with col_new:
            st.markdown("**Re-simulation (edited)**")
            st.text(resim_stdout[:2000])

        # Metric comparison
        if resim_parsed:
            from agents.session import parse_lattice_results
            orig_parsed = None
            for msg in reversed(st.session_state.messages):
                if msg.get("source") == RUNNER:
                    orig_parsed = parse_lattice_results(msg.get("content", ""))
                    break

            st.markdown("**Key metrics comparison**")
            metric_keys = [
                ("stable",    "Stable",       None),
                ("tune_nux",  "νx",           ".3f"),
                ("tune_nuy",  "νy",           ".3f"),
                ("emittance", "ε (m·rad)",    ".3e"),
                ("chromaticity_xix", "ξx",    ".2f"),
                ("chromaticity_xiy", "ξy",    ".2f"),
                ("alphac",    "α_c",          ".3e"),
                ("sigma_E",   "σ_E",          ".3e"),
            ]
            metric_cols = st.columns(min(len(metric_keys), 4))
            for i, (key, label, fmt) in enumerate(metric_keys):
                new_val = resim_parsed.get(key)
                if new_val is None:
                    continue
                old_val = (orig_parsed or {}).get(key)
                col = metric_cols[i % len(metric_cols)]
                try:
                    nv = float(new_val)
                    ov = float(old_val) if old_val is not None else None
                    display = f"{nv:{fmt}}" if fmt else str(nv)
                    delta = f"{nv - ov:{fmt}}" if ov is not None and fmt else None
                    col.metric(label, display, delta=delta)
                except (TypeError, ValueError):
                    col.metric(label, str(new_val))


def _make_zip(session_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in session_path.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(session_path))
    return buf.getvalue()


def _save_session():
    """Persist the current session and show download UI."""
    sess = st.session_state.session_obj
    if not sess:
        st.warning("No active session to save.")
        return

    sess.collect_plots(WORKSPACE_DIR)
    sess_dir = sess.save()
    st.success(f"Session saved to `{sess_dir}`")

    # Final metrics
    snapshot = latest_result_snapshot(
        st.session_state.messages,
        fallback_context=latest_user_request(
            st.session_state.messages,
            fallback=st.session_state.session_obj.request if st.session_state.session_obj else "",
        ),
    )
    if snapshot:
        parsed = snapshot["result"]
        assessment = snapshot["assessment"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Stable", parsed.get("stable", "n/a"))
        c2.metric("Score", assessment.get("score", "n/a"))
        c3.metric("νx", parsed.get("nux", "n/a"))
        c4.metric("νy", parsed.get("nuy", "n/a"))
        c5.metric("Emittance (m.rad)", parsed.get("emittance", "n/a"))
        if assessment.get("issues"):
            st.caption("Evaluator notes: " + " | ".join(assessment["issues"][:3]))
    else:
        for msg in reversed(st.session_state.messages):
            parsed = parse_lattice_results(msg["content"])
            if parsed:
                context_text = latest_user_request(
                    st.session_state.messages,
                    fallback=st.session_state.session_obj.request if st.session_state.session_obj else "",
                )
                assessment = assess_lattice_results(parsed, context_text=context_text)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Stable", parsed.get("stable", "n/a"))
                c2.metric("Score", assessment.get("score", "n/a"))
                c3.metric("νx", parsed.get("nux", "n/a"))
                c4.metric("νy", parsed.get("nuy", "n/a"))
                c5.metric("Emittance (m.rad)", parsed.get("emittance", "n/a"))
                if assessment.get("issues"):
                    st.caption("Evaluator notes: " + " | ".join(assessment["issues"][:3]))
                break

    tab_names = ["Summary", "Code", "Plots", "Inspector", "Download"]
    t_summary, t_code, t_plots, t_inspect, t_dl = st.tabs(tab_names)
    with t_summary:
        msgs = st.session_state.messages
        total = len(msgs)
        user_msgs = sum(1 for m in msgs if m["source"] == "user")
        agent_turns = {
            PLANNER:    sum(1 for m in msgs if m["source"] == PLANNER    and m.get("type") != "thought"),
            CODEWRITER: sum(1 for m in msgs if m["source"] == CODEWRITER and m.get("type") != "thought"),
            RUNNER:     sum(1 for m in msgs if m["source"] == RUNNER     and m.get("type") != "thought"),
            REVIEWER:   sum(1 for m in msgs if m["source"] == REVIEWER   and m.get("type") != "thought"),
        }
        code_blocks = sum(
            1 for m in msgs
            if m["source"] == CODEWRITER and "```python" in m.get("content", "")
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total messages", total)
        c2.metric("User turns", user_msgs)
        c3.metric("Code scripts", code_blocks)
        c4.metric("Runner executions", agent_turns[RUNNER])
        st.markdown("**Agent turn breakdown**")
        for agent, count in agent_turns.items():
            label = AGENT_CFG.get(agent, {}).get("label", agent)
            st.caption(f"{label}: {count} turn(s)")
    with t_code:
        code_files = sorted((sess_dir / "code").glob("*.py"))
        if code_files:
            for cf in code_files:
                with st.expander(cf.name, expanded=(cf == code_files[-1])):
                    st.code(cf.read_text(encoding="utf-8"), language="python")
        else:
            st.info("No code generated.")
    with t_plots:
        plot_files = sorted((sess_dir / "plots").glob("*"))
        if plot_files:
            cols = st.columns(min(len(plot_files), 3))
            for i, pf in enumerate(plot_files):
                with cols[i % 3]:
                    st.image(str(pf), caption=pf.name, use_container_width=True)
        else:
            st.info("No plots generated.")
    with t_inspect:
        _render_lattice_inspector()
    with t_dl:
        st.download_button(
            "Download session (.zip)",
            data=_make_zip(sess_dir),
            file_name=f"{sess_dir.name}.zip",
            mime="application/zip",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="text-align:center;font-size:0.82rem;color:#7a8ba8;margin:-4px 0 12px;font-weight:500;">'
        'BRILLIANCE</p>',
        unsafe_allow_html=True,
    )
    # Sidebar logo
    logo_uri = _img_base64("logo.svg")
    if logo_uri:
        st.markdown(
            f'<div style="text-align:center;padding:16px 0 8px;">'
            f'<img src="{logo_uri}" width="56" style="filter:drop-shadow(0 0 8px rgba(0,212,255,0.4));"/>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("## ⚙ Configuration")
    st.text_input("LLM Model", value=os.environ.get("LLM_MODEL", "deepseek-chat"),
                  key="_cfg_model", help="deepseek-chat, gpt-4o, etc.")
    st.text_input("API Key", value=os.environ.get("LLM_API_KEY", ""),
                  type="password", key="_cfg_key")
    st.text_input("API Base URL",
                  value=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
                  key="_cfg_url")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Executor", ["local", "docker"],
                     index=0 if os.environ.get("EXECUTOR", "local").lower() == "local" else 1,
                     key="_cfg_exec")
    with c2:
        st.number_input("Max Rounds", 1, 50, 6, key="_cfg_rounds")
    st.slider("Code Timeout (s)", 60, 1200, 600, step=60, key="_cfg_timeout",
              help="Max seconds for code execution")

    st.markdown("---")
    st.markdown("## 📂 Past Sessions")
    sessions = list_sessions()
    if sessions:
        for s in sessions[:15]:
            label = s["request"][:48] or s["name"]
            with st.expander(label):
                st.caption(f"Code: {s['code_count']}  |  Plots: {s['plot_count']}")
                if st.button("View", key=f"v_{s['name']}"):
                    st.session_state.viewing = s["path"]
                    st.rerun()
    else:
        st.caption("No sessions yet.")


# ══════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════
# Build hero images
_logo_uri = _img_base64("logo.svg")
_ihep_uri = _img_base64("ihep_logo.png") or _img_base64("ihep_logo.svg") or _img_base64("ihep_logo.jpg")
_heps_uri = _img_base64("heps_background.jpg") or _img_base64("heps_background.png") or _img_base64("heps.jpg") or _img_base64("heps.png")

_ihep_tag = f'<img class="hero-ihep-logo" src="{_ihep_uri}"/>' if _ihep_uri else ''
_logo_tag = f'<img class="hero-logo" src="{_logo_uri}"/>' if _logo_uri else ''
_bg_style = f'background-image:url({_heps_uri});' if _heps_uri else ''

st.markdown(
    f'<div class="hero-container" style="{_bg_style}">'
    f'  <div class="hero-flex">'
    f'    {_logo_tag}'
    f'    <div class="hero-text">'
    f'      <h1>BRILLIANCE</h1>'
    f'      <p>Beam Research with Intelligent Learning and Integrated Accelerator Numerical Computation Engine</p>'
    f'    </div>'
    f'    {_ihep_tag}'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── view a past session ──────────────────────────────────────────
if st.session_state.viewing:
    sp = Path(st.session_state.viewing)
    if st.button("← Back to designer"):
        st.session_state.viewing = None
        st.rerun()

    st.markdown(f"### {sp.name}")
    summary_f = sp / "summary.md"
    if summary_f.exists():
        st.markdown(summary_f.read_text(encoding="utf-8"))

    tab_conv, tab_code, tab_plots, tab_dl = st.tabs(
        ["Conversation", "Code", "Plots", "Download"])
    with tab_conv:
        conv_f = sp / "conversation.json"
        if conv_f.exists():
            for msg in json.loads(conv_f.read_text(encoding="utf-8")):
                if msg["type"] == "thought":
                    with st.expander(f"{msg['source']} thinking"):
                        st.markdown(msg["content"][:4000])
                else:
                    _render_msg(msg)
    with tab_code:
        cd = sp / "code"
        if cd.exists():
            files = sorted(cd.glob("*.py"))
            if files:
                sel = st.selectbox("Version", [f.name for f in files],
                                   index=len(files) - 1)
                st.code((cd / sel).read_text(encoding="utf-8"), language="python")
    with tab_plots:
        pd_ = sp / "plots"
        if pd_.exists():
            imgs = sorted(pd_.glob("*"))
            if imgs:
                cols = st.columns(min(len(imgs), 3))
                for i, img in enumerate(imgs):
                    with cols[i % 3]:
                        st.image(str(img), caption=img.name,
                                 use_container_width=True)
    with tab_dl:
        st.download_button("Download (.zip)", data=_make_zip(sp),
                           file_name=f"{sp.name}.zip",
                           mime="application/zip",
                           use_container_width=True)
    st.stop()

# ── render conversation history ───────────────────────────────────
if any(msg["source"] in {step[0] for step in PIPELINE_STEPS} for msg in st.session_state.messages) or (
    st.session_state.last_agent in {step[0] for step in PIPELINE_STEPS}
):
    _render_pipeline(st.session_state.last_agent)

for msg in st.session_state.messages:
    if msg["type"] == "thought":
        with st.expander(f"{msg['source']} reasoning", expanded=False):
            st.markdown(msg["content"][:4000])
    else:
        _render_msg(msg)

# ── action buttons (depend on who spoke last) ─────────────────────
last = st.session_state.last_agent

if last == PLANNER:
    st.info(
        "**Task Planner** produced a structured task spec. "
        "Review it, then click **Write Code** or type modifications below."
    )
    c1, c2 = st.columns(2)
    if c1.button("Write Code", type="primary", use_container_width=True):
        _push_config()
        _invoke(CODEWRITER, "Code Writer (generating pyJuTrack script)")
    if c2.button("Accept & Finish", use_container_width=True):
        _save_session()

elif last == CODEWRITER:
    st.info(
        "**Code Writer** generated a pyJuTrack script. "
        "Click **Run Code** to execute, or type changes below."
    )
    c1, c2 = st.columns(2)
    if c1.button("Run Code", type="primary", use_container_width=True):
        _push_config()
        _invoke(RUNNER, "Code Runner (executing pyJuTrack)")
    if c2.button("Request Revision", use_container_width=True):
        st.session_state.last_agent = "_awaiting_input_planner"
        st.rerun()

elif last == RUNNER:
    st.info("Code executed. Review the output, then choose an action.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Physics Review", type="primary", use_container_width=True):
        _push_config()
        _invoke(REVIEWER)
    if c2.button("Regenerate Code", use_container_width=True):
        st.session_state.last_agent = "_awaiting_input_planner"
        st.rerun()
    if c3.button("Accept & Finish", use_container_width=True):
        _save_session()

    with st.expander("🔬 Lattice Inspector — tweak parameters & re-simulate", expanded=False):
        _render_lattice_inspector()

elif last == REVIEWER:
    content_last = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
    if "TASK_COMPLETE" in content_last:
        st.success("Task completed successfully.")
        c1, c2 = st.columns(2)
        if c1.button("Save & Finish", type="primary", use_container_width=True):
            _save_session()
        if c2.button("Continue (new request)", use_container_width=True):
            st.session_state.last_agent = None

        with st.expander("🔬 Lattice Inspector — tweak parameters & re-simulate", expanded=False):
            _render_lattice_inspector()
    elif "NEEDS_FIX" in content_last:
        st.warning("Code needs a fix. Click **Retry** to have the Code Writer correct it.")
        c1, c2 = st.columns(2)
        if c1.button("Retry", type="primary", use_container_width=True):
            _push_config()
            _invoke(CODEWRITER)
        if c2.button("Accept anyway", use_container_width=True):
            _save_session()
    else:
        st.info("**Reviewer** finished. You can continue or save.")
        c1, c2 = st.columns(2)
        if c1.button("Save & Finish", type="primary", use_container_width=True):
            _save_session()
        if c2.button("Retry Code", use_container_width=True):
            _push_config()
            _invoke(CODEWRITER)

elif last is not None and last.startswith("_awaiting_input"):
    st.warning("Type your instructions below, then press Enter.")

# ── chat input (always available) ─────────────────────────────────
# Keep input above suggestions while still allowing suggestions to prefill it.
chat_input_container = st.container()

# ── suggested prompts: loaded from benchmark tasks, simple → advanced ────
def _load_benchmark_prompts() -> dict[str, str]:
    """Return display-title → full prompt from benchmarks/tasks.json."""
    import json as _json
    tasks_file = Path(__file__).resolve().parent / "benchmarks" / "tasks.json"
    try:
        raw = _json.loads(tasks_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    _TITLE_MAP = {
        "stage1_single_element_matrices":       "Stage 1 — Single-element transfer matrices",
        "stage2_periodic_fodo_cell_60deg":      "Stage 2 — Periodic FODO cell (60° phase advance)",
        "stage3_fodo_arc_dispersion":           "Stage 3 — FODO arc: tunes, dispersion & radiation integrals",
        "dba_3gev_12_cells":                    "Stage 3b — DBA storage ring, 3 GeV, 12 cells",
        "fodo_25gev_20_cells":                  "Stage 3c — FODO booster ring, 2.5 GeV, 20 cells",
        "stage6_mba_6gev_radiation_integrals":  "Stage 6 — 7BA 6 GeV: I₁–5, emittance & Robinson sum",
        "std7ba_6gev_24_cells":                 "Stage 6b — 7BA 6 GeV free design (~1360 m)",
        "stage10_infeasible_dba_refusal":       "Stage 10 — Impossible design (constraint conflict)",
    }
    by_id = {t["id"]: t["prompt"] for t in raw}
    return {title: by_id[tid] for tid, title in _TITLE_MAP.items() if tid in by_id}

_SUGGESTED_PROMPTS = _load_benchmark_prompts()



if not st.session_state.messages:
    st.markdown("---")
    st.caption("Suggested prompts (click to paste):")
    if "_pending_chat_prefill" not in st.session_state:
        st.session_state["_pending_chat_prefill"] = None

    selected_prompt_title = st.pills(
        "Suggestions",
        list(_SUGGESTED_PROMPTS.keys()),
        key="_suggested_prompt_picker",
        label_visibility="collapsed",
    )
    if selected_prompt_title and st.session_state.get("_last_suggested_prompt") != selected_prompt_title:
        st.session_state["_pending_chat_prefill"] = _SUGGESTED_PROMPTS[selected_prompt_title]
        st.session_state["_last_suggested_prompt"] = selected_prompt_title
        st.rerun()

# Apply pending prefill BEFORE creating st.chat_input widget.
if st.session_state.get("_pending_chat_prefill") is not None:
    st.session_state["_chat_input"] = st.session_state["_pending_chat_prefill"]
    st.session_state["_pending_chat_prefill"] = None

with chat_input_container:
    prompt = st.chat_input(
        "Type your design request or instructions...",
        key="_chat_input",
    )

if prompt:
    _push_config()
    if st.session_state.session_obj is None:
        st.session_state.session_obj = Session(prompt)
    _add_msg("user", prompt)
    _invoke(PLANNER)

# ── new-session button ────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("---")
    if st.button("🔄  New Session"):
        if st.session_state.session_obj:
            try:
                st.session_state.session_obj.collect_plots(WORKSPACE_DIR)
                st.session_state.session_obj.save()
            except Exception:
                pass
        for k in _DEFAULTS:
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()
