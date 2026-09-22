"""Named, non-secret model profiles. Existing backend/runtime adapters remain owners."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from threading import RLock

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

from .atomic_io import atomic_write_text

DEFAULT_PATH = Path(__file__).resolve().parent / "config" / "models.toml"
_LOCK = RLock()
PROFILE_KEYS = {
    "provider": "BRILLIANCE_CODEX_PROVIDER", "name": "BRILLIANCE_CODEX_PROVIDER_NAME",
    "provider_id": "BRILLIANCE_CODEX_PROVIDER_ID", "model": "BRILLIANCE_CODEX_MODEL",
    "base_url": "BRILLIANCE_CODEX_BASE_URL", "api_key_env": "BRILLIANCE_CODEX_API_KEY_ENV",
    "wire_api": "BRILLIANCE_CODEX_WIRE_API", "auth": "BRILLIANCE_OPENAI_AUTH",
    "reasoning_effort": "BRILLIANCE_CODEX_EFFORT", "context_window": "BRILLIANCE_CODEX_CONTEXT_WINDOW",
}
OPTION_KEYS = {
    "agent": {"native": "BRILLIANCE_NATIVE_RESEARCH_AGENT", "wire_api": "BRILLIANCE_NATIVE_AGENT_WIRE_API",
              "model_timeout_s": "BRILLIANCE_NATIVE_AGENT_MODEL_TIMEOUT_S", "steps": "BRILLIANCE_NATIVE_AGENT_STEPS",
              "max_tokens": "BRILLIANCE_NATIVE_AGENT_MAX_TOKENS", "max_tokens_ceiling": "BRILLIANCE_NATIVE_AGENT_MAX_TOKENS_CEILING",
              "stream": "BRILLIANCE_NATIVE_AGENT_STREAM", "stream_usage": "BRILLIANCE_NATIVE_AGENT_STREAM_USAGE",
              "context_rollover_ratio": "BRILLIANCE_CONTEXT_ROLLOVER_RATIO"},
    "transport": {"request_retries": "BRILLIANCE_MODEL_REQUEST_RETRIES", "stream_retries": "BRILLIANCE_MODEL_STREAM_RETRIES",
                  "stream_idle_timeout_s": "BRILLIANCE_MODEL_STREAM_IDLE_TIMEOUT_S"},
    "interface": {"heartbeat_s": "BRILLIANCE_PROGRESS_HEARTBEAT", "chat_update_interval_s": "BRILLIANCE_CHAT_UPDATE_INTERVAL_S"},
}


def config_path() -> Path:
    return Path(os.environ.get("BRILLIANCE_MODEL_CONFIG") or DEFAULT_PATH).resolve()


def read_settings() -> dict:
    path = config_path()
    if not path.is_file():
        if os.environ.get("BRILLIANCE_MODEL_CONFIG"):
            raise ValueError(f"Model configuration not found: {path}")
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", {})
    selected = data.get("selection", {}).get("profile")
    if selected not in profiles:
        raise ValueError("models.toml selection.profile must name an installed profile")
    for name, profile in profiles.items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or not isinstance(profile, dict):
            raise ValueError("Invalid model profile name or table")
        if set(profile) - PROFILE_KEYS.keys():
            raise ValueError(f"Unknown model profile fields in {name}; keep credentials in .env")
        if profile.get("provider") not in {"chatgpt", "openai", "custom", "deepseek", "ollama", "lmstudio"}:
            raise ValueError(f"Invalid provider in profile {name}")
    return data


def apply_model_settings() -> dict:
    """Translate TOML into existing internal configuration names, not a new adapter.

    TOML wins for explicitly owned options. Credentials are never read/copied here.
    Legacy installations without a model file retain their environment configuration.
    """
    with _LOCK:
        data = read_settings()
        if not data:
            return {}
        profile = data["profiles"][data["selection"]["profile"]]
        def value(item):
            return str(item).lower() if isinstance(item, bool) else str(item)
        for field, key in PROFILE_KEYS.items():
            os.environ[key] = value(profile.get(field, ""))
        for section, mapping in OPTION_KEYS.items():
            for field, item in data.get(section, {}).items():
                if field not in mapping:
                    raise ValueError(f"Unknown model option {section}.{field}")
                os.environ[mapping[field]] = value(item)
        # A model-specific reasoning override must not leak across profile changes.
        os.environ["BRILLIANCE_NATIVE_AGENT_REASONING_EFFORT"] = str(profile.get("reasoning_effort", ""))
        return data


def public_settings() -> dict:
    data = read_settings()
    return {"source": str(config_path()), "selected": data.get("selection", {}).get("profile"),
            "profiles": [{"id": name, "provider": p["provider"], "name": p.get("name", name),
                          "model": p.get("model", ""), "reasoning_effort": p.get("reasoning_effort", "")}
                         for name, p in data.get("profiles", {}).items()]}


def select_profile(name: str, model: str | None = None) -> dict:
    """Update only the selected profile/model, preserving comments and other settings."""
    with _LOCK:
        data = read_settings()
        if name not in data.get("profiles", {}):
            raise ValueError("Unknown model profile")
        if model is not None and (not isinstance(model, str) or len(model) > 200 or any(ord(c) < 32 for c in model)):
            raise ValueError("Invalid model identifier")
        text = config_path().read_text(encoding="utf-8")
        updates = [("selection", "profile", name)]
        if model is not None:
            updates.append((f"profiles.{name}", "model", model))
        for section, field, value in updates:
            pattern = rf"(?ms)(^\[{re.escape(section)}\]\s*\n)(.*?)(?=^\[|\Z)"
            def edit(match):
                body = match[2]
                line = f"{field} = {json.dumps(value)}"
                if re.search(rf"(?m)^{field}\s*=", body):
                    body = re.sub(rf"(?m)^{field}\s*=.*$", lambda _: line, body)
                else:
                    body = line + "\n" + body
                return match[1] + body
            text, count = re.subn(pattern, edit, text)
            if count != 1:
                raise ValueError(f"Expected one [{section}] table")
        tomllib.loads(text)
        atomic_write_text(config_path(), text)
        apply_model_settings()
        return public_settings()
