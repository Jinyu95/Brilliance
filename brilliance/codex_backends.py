"""Inference-backend configuration for Brilliance's coding-agent runtimes.

OpenAI uses Codex; compatible third-party and local models may use Brilliance's
compact native tool loop. This module resolves either route and also produces
Codex config overrides for the fallback runtime. Provider credentials are
referenced by environment-variable name; their values are never persisted in
project state or command-line arguments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Mapping


class CodexBackendError(RuntimeError):
    """Raised when the selected inference backend is not usable."""


def _value(env: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = str(env.get(name, "")).strip()
        if value:
            return value
    return ""


def _toml_string(value: str) -> str:
    # A JSON string is also a valid TOML basic string for these scalar values.
    return json.dumps(value, ensure_ascii=True)


def _provider_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return f"brilliance_{normalized or 'provider'}"


def _responses_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if not base:
        raise CodexBackendError("A Responses-compatible provider base URL is required.")
    if not base.lower().endswith("/v1"):
        base += "/v1"
    return base


def _chat_completions_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if not base:
        raise CodexBackendError("A chat-completions-compatible provider base URL is required.")
    # A pasted full endpoint (e.g. ".../chat/completions") is a common mistake;
    # Codex's chat wire_api appends that suffix itself.
    if base.lower().endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _wire_api(values: Mapping[str, str]) -> str:
    selected = (_value(values, "BRILLIANCE_CODEX_WIRE_API") or "responses").lower()
    if selected not in {"responses", "chat"}:
        raise CodexBackendError(
            f"BRILLIANCE_CODEX_WIRE_API must be 'responses' or 'chat', not {selected!r}."
        )
    return selected


def _remote_base_url(values: Mapping[str, str], wire_api: str, *names: str, default: str = "") -> str:
    raw = _value(values, *names) or default
    return _chat_completions_base_url(raw) if wire_api == "chat" else _responses_base_url(raw)


def _positive_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _bounded_nonnegative_int(value: str, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(maximum, parsed))


@dataclass(frozen=True, slots=True)
class CodexBackend:
    """Resolved model backend for one Codex app-server process."""

    key: str
    name: str
    provider_id: str
    model: str
    mode: str
    config_overrides: tuple[str, ...]
    base_url: str = ""
    api_key_env: str = ""
    supports_reasoning_effort: bool = False
    setup_hint: str = ""
    wire_api: str = ""

    @property
    def fingerprint(self) -> str:
        identity = "\n".join((self.key, self.provider_id, self.model, self.base_url))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]

    def public(self) -> dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "provider_id": self.provider_id,
            "model": self.model or "Codex default",
            "mode": self.mode,
            "base_url": self.base_url,
            "wire_api": self.wire_api,
            "fingerprint": self.fingerprint,
            "supports_reasoning_effort": self.supports_reasoning_effort,
        }


def resolve_codex_backend(
    env: Mapping[str, str] | None = None,
    *,
    validate_credentials: bool = True,
) -> CodexBackend:
    """Resolve the selected backend without falling back to an OpenAI model."""

    if env is None:
        from .model_settings import apply_model_settings
        apply_model_settings()
    values = os.environ if env is None else env
    selected = _value(values, "BRILLIANCE_CODEX_PROVIDER").lower() or "deepseek"
    aliases = {
        "local": "ollama",
        "offline": "ollama",
        "lm-studio": "lmstudio",
        "lm_studio": "lmstudio",
        "gpt": "openai",
        "codex": "openai",
    }
    selected = aliases.get(selected, selected)

    if selected == "chatgpt":
        model = _value(values, "BRILLIANCE_CODEX_MODEL")
        overrides = ('model_provider="openai"',) + ((f"model={_toml_string(model)}",) if model else ())
        return CodexBackend(key="chatgpt", name="ChatGPT subscription", provider_id="openai",
                            model=model, mode="subscription", config_overrides=overrides,
                            supports_reasoning_effort=True,
                            setup_hint="Sign in with ChatGPT in Models & account. API keys are not used in this mode.")

    context_window = _positive_int(
        _value(values, "BRILLIANCE_CODEX_CONTEXT_WINDOW"),
        128_000 if selected in {"deepseek", "custom"} else 32_768,
    )

    if selected == "deepseek":
        model = _value(
            values,
            "BRILLIANCE_CODEX_MODEL",
            "BRILLIANCE_DEEPSEEK_MODEL",
            "LLM_MODEL",
        ) or "deepseek-chat"
        wire_api = _wire_api(values)
        base_url = _remote_base_url(
            values,
            wire_api,
            "BRILLIANCE_CODEX_BASE_URL",
            "BRILLIANCE_DEEPSEEK_BASE_URL",
            "LLM_BASE_URL",
            default="https://api.deepseek.com",
        )
        key_env = _value(values, "BRILLIANCE_CODEX_API_KEY_ENV") or "LLM_API_KEY"
        if validate_credentials and not _value(values, key_env):
            raise CodexBackendError(
                f"DeepSeek is selected but {key_env} is not configured. "
                "Set it in .env or select BRILLIANCE_CODEX_PROVIDER=ollama/lmstudio."
            )
        provider_id = _provider_id("deepseek")
        overrides = _remote_provider_overrides(
            provider_id=provider_id,
            name="DeepSeek",
            model=model,
            base_url=base_url,
            key_env=key_env,
            context_window=context_window,
            wire_api=wire_api,
            values=values,
        )
        return CodexBackend(
            key="deepseek",
            name="DeepSeek",
            provider_id=provider_id,
            model=model,
            mode="remote",
            config_overrides=overrides,
            base_url=base_url,
            api_key_env=key_env,
            setup_hint=(
                f"Verify {key_env}, the DeepSeek {wire_api} endpoint, and model {model!r}. "
                "A chat-completions-only gateway needs BRILLIANCE_CODEX_WIRE_API=chat."
            ),
            wire_api=wire_api,
        )

    if selected in {"ollama", "lmstudio"}:
        model = _value(
            values,
            "BRILLIANCE_CODEX_MODEL",
            "OLLAMA_MODEL" if selected == "ollama" else "LMSTUDIO_MODEL",
        )
        if not model:
            variable = "OLLAMA_MODEL" if selected == "ollama" else "LMSTUDIO_MODEL"
            raise CodexBackendError(
                f"{selected} is selected but no model is configured. "
                f"Set BRILLIANCE_CODEX_MODEL or {variable}."
            )
        display = "Ollama" if selected == "ollama" else "LM Studio"
        local_default = (
            "http://127.0.0.1:11434/v1"
            if selected == "ollama"
            else "http://127.0.0.1:1234/v1"
        )
        base_url = _chat_completions_base_url(
            _value(
                values,
                "BRILLIANCE_CODEX_BASE_URL",
                "OLLAMA_BASE_URL" if selected == "ollama" else "LMSTUDIO_BASE_URL",
            )
            or local_default
        )
        overrides = (
            f"model_provider={_toml_string(selected)}",
            f"oss_provider={_toml_string(selected)}",
            f"model={_toml_string(model)}",
            f"model_context_window={context_window}",
        )
        return CodexBackend(
            key=selected,
            name=display,
            provider_id=selected,
            model=model,
            mode="offline",
            config_overrides=overrides,
            base_url=base_url,
            wire_api="chat",
            setup_hint=(
                f"Start {display}, load model {model!r}, and make its local API available."
            ),
        )

    if selected == "custom":
        model = _value(values, "BRILLIANCE_CODEX_MODEL")
        if not model:
            raise CodexBackendError(
                "A custom provider requires BRILLIANCE_CODEX_MODEL."
            )
        wire_api = _wire_api(values)
        base_url = _remote_base_url(values, wire_api, "BRILLIANCE_CODEX_BASE_URL")
        key_env = _value(values, "BRILLIANCE_CODEX_API_KEY_ENV")
        if key_env and validate_credentials and not _value(values, key_env):
            raise CodexBackendError(
                f"The custom provider expects credentials in {key_env}, but it is unset."
            )
        provider_id = _provider_id(_value(values, "BRILLIANCE_CODEX_PROVIDER_ID") or "custom")
        overrides = _remote_provider_overrides(
            provider_id=provider_id,
            name=_value(values, "BRILLIANCE_CODEX_PROVIDER_NAME") or "Custom provider",
            model=model,
            base_url=base_url,
            key_env=key_env,
            context_window=context_window,
            wire_api=wire_api,
            values=values,
        )
        return CodexBackend(
            key="custom",
            name=_value(values, "BRILLIANCE_CODEX_PROVIDER_NAME") or "Custom provider",
            provider_id=provider_id,
            model=model,
            mode="remote",
            config_overrides=overrides,
            base_url=base_url,
            api_key_env=key_env,
            setup_hint=(
                f"Verify the provider's OpenAI {wire_api} compatibility and credentials."
            ),
            wire_api=wire_api,
        )

    if selected == "openai":
        model = _value(values, "BRILLIANCE_CODEX_MODEL")
        auth = (_value(values, "BRILLIANCE_OPENAI_AUTH") or "auto").lower()
        if auth not in {"auto", "api_key", "codex"}:
            raise CodexBackendError("BRILLIANCE_OPENAI_AUTH must be auto, api_key, or codex.")
        if auth == "api_key" or (auth == "auto" and _value(values, "OPENAI_API_KEY")):
            if validate_credentials and not _value(values, "OPENAI_API_KEY"):
                raise CodexBackendError("OpenAI API authentication requires OPENAI_API_KEY in .env.")
            if not model:
                raise CodexBackendError("OpenAI API authentication requires BRILLIANCE_CODEX_MODEL.")
            return CodexBackend(
                key="openai", name="OpenAI", provider_id="openai", model=model,
                mode="remote", config_overrides=(), base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY", supports_reasoning_effort=True,
                wire_api="responses", setup_hint="Configure OPENAI_API_KEY and BRILLIANCE_CODEX_MODEL.",
            )
        overrides = tuple(
            item
            for item in (
                'model_provider="openai"',
                f"model={_toml_string(model)}" if model else "",
            )
            if item
        )
        return CodexBackend(
            key="openai",
            name="OpenAI",
            provider_id="openai",
            model=model,
            mode="remote",
            config_overrides=overrides,
            supports_reasoning_effort=True,
            setup_hint="Run `codex login` or configure OpenAI authentication.",
        )

    raise CodexBackendError(
        f"Unknown BRILLIANCE_CODEX_PROVIDER={selected!r}. "
        "Use deepseek, ollama, lmstudio, custom, or openai."
    )


def describe_codex_backend(env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Return non-secret backend status suitable for the web interface."""

    try:
        backend = resolve_codex_backend(env, validate_credentials=False)
    except (CodexBackendError, ValueError, OSError) as exc:
        return {
            "key": "invalid",
            "name": "Model backend not configured",
            "provider_id": "",
            "model": "",
            "mode": "unavailable",
            "configured": False,
            "error": str(exc),
        }
    public = backend.public()
    public["configured"] = True
    if backend.api_key_env:
        values = os.environ if env is None else env
        public["credentials_configured"] = bool(_value(values, backend.api_key_env))
    return public


def _remote_provider_overrides(
    *,
    provider_id: str,
    name: str,
    model: str,
    base_url: str,
    key_env: str,
    context_window: int,
    wire_api: str = "responses",
    values: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    configured = os.environ if values is None else values
    request_retries = _bounded_nonnegative_int(
        _value(configured, "BRILLIANCE_MODEL_REQUEST_RETRIES"), 1, 3
    )
    stream_retries = _bounded_nonnegative_int(
        _value(configured, "BRILLIANCE_MODEL_STREAM_RETRIES"), 1, 3
    )
    stream_idle_seconds = max(
        30,
        min(
            600,
            _positive_int(
                _value(configured, "BRILLIANCE_MODEL_STREAM_IDLE_TIMEOUT_S"),
                120,
            ),
        ),
    )
    prefix = f"model_providers.{provider_id}"
    values = [
        f"model_provider={_toml_string(provider_id)}",
        f"model={_toml_string(model)}",
        f"{prefix}.name={_toml_string(name)}",
        f"{prefix}.base_url={_toml_string(base_url)}",
        f"{prefix}.wire_api={_toml_string(wire_api)}",
        f"{prefix}.request_max_retries={request_retries}",
        f"{prefix}.stream_max_retries={stream_retries}",
        f"{prefix}.stream_idle_timeout_ms={stream_idle_seconds * 1000}",
        f"model_context_window={context_window}",
    ]
    if key_env:
        values.append(f"{prefix}.env_key={_toml_string(key_env)}")
    return tuple(values)
