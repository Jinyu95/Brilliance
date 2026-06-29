"""LLM client and code-executor factories — model-agnostic via env vars."""

import os
import re
import shutil
import types
from pathlib import Path

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)


def _copy_workspace_support_files():
    """Keep shared helper modules available to generated scripts."""
    for filename in ("physics_core.py",):
        src = PROJECT_ROOT / filename
        if src.exists():
            shutil.copy2(src, WORKSPACE_DIR / filename)


def _role_env_key(role: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", role.strip()).upper().strip("_")
    return normalized or "DEFAULT"


def _resolve_model_name(role: str | None = None, model: str | None = None) -> str:
    if model:
        return model

    if role:
        role_key = _role_env_key(role)
        for env_name in (f"LLM_MODEL_{role_key}", f"{role_key}_MODEL"):
            value = os.environ.get(env_name)
            if value:
                return value

    return os.environ.get("LLM_MODEL", "deepseek-chat")


def create_model_client(
    *,
    role: str | None = None,
    model: str | None = None,
) -> OpenAIChatCompletionClient:
    """Return an LLM client configured from environment variables.

    Works with any OpenAI-compatible endpoint (DeepSeek, OpenAI,
    Together, OpenRouter, Ollama, vLLM …).
    """
    model_name = _resolve_model_name(role=role, model=model)
    api_key = os.environ["LLM_API_KEY"]
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")

    return OpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        # model_info is required for models not in AutoGen's built-in registry.
        # Adjust "vision" / "function_calling" if your model differs.
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": False,
            "family": "unknown",
        },
    )


def create_code_executor(use_docker: bool = True):
    """Return a code executor (Docker or local).

    Docker  – runs inside the ``jutrack-sandbox`` image (build first).
    Local   – uses the active Python; activate the pyJuTrack
              conda env **before** launching run_design.py.
    """
    timeout = int(os.environ.get("CODE_TIMEOUT", "600"))

    _copy_workspace_support_files()

    if use_docker:
        from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

        return DockerCommandLineCodeExecutor(
            image="jutrack-sandbox",
            work_dir=WORKSPACE_DIR,
            timeout=timeout,
        )
    else:
        from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

        # Resolve which Python to use for code execution:
        #   1. JUTRACK_PYTHON env var (explicit conda / venv path)
        #   2. .venv/ inside the project root  (created by setup_env.py)
        #   3. The current interpreter (fallback)
        jutrack_python = os.environ.get("JUTRACK_PYTHON", "").strip()
        if not jutrack_python:
            _venv_candidates = [
                PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows
                PROJECT_ROOT / ".venv" / "bin" / "python",           # Unix
            ]
            for c in _venv_candidates:
                if c.is_file():
                    jutrack_python = str(c)
                    break
        venv_ctx = None
        if jutrack_python:
            exe = Path(jutrack_python)
            if exe.is_file():
                # Verify the executable is actually runnable on this platform.
                # A Windows .exe in a Linux container causes "Exec format error".
                import platform as _platform
                _is_win = _platform.system() == "Windows"
                _is_win_exe = exe.suffix.lower() == ".exe"
                if _is_win_exe and not _is_win:
                    # Linux container with a Windows binary — skip, fall through to system python
                    import sys as _sys
                    jutrack_python = _sys.executable
                else:
                    venv_ctx = types.SimpleNamespace(
                        env_exe=str(exe),
                        bin_path=str(exe.parent),
                    )

        # Set JULIA_PROJECT so Julia activates JuTrack.jl regardless of cwd.
        # Without this, running scripts from WORKSPACE_DIR causes PyCallExt
        # to fail (Julia looks for a project in workspace/ and can't find it).
        jutrack_jl = os.environ.get("JUTRACK_JL_PROJECT", "").strip()
        if not jutrack_jl:
            # Search order:
            #   1. JuTrack.jl/ bundled inside the project root  (new default)
            #   2. JuTrack.jl/ as a sibling of the project root (legacy location)
            for candidate in (
                PROJECT_ROOT / "JuTrack.jl",
                PROJECT_ROOT.parent / "JuTrack.jl",
            ):
                if candidate.exists():
                    jutrack_jl = str(candidate)
                    break
        if jutrack_jl:
            os.environ.setdefault("JULIA_PROJECT", jutrack_jl)

        return LocalCommandLineCodeExecutor(
            work_dir=WORKSPACE_DIR,
            timeout=timeout,
            virtual_env_context=venv_ctx,
        )
