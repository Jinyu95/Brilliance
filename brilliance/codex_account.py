"""Account/model controls using the SAME official Codex app-server SDK as research.

No browser cookies, auth-file reads, OAuth token extraction or direct ChatGPT HTTP.
The long-lived control connection keeps Codex's managed sign-in alive until completion.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlsplit


def subscription_environment(existing: dict[str, str]) -> dict[str, str]:
    # SDK merges its env with the parent: blank explicit overrides prevent API
    # credentials or a custom API endpoint leaking into the subscription process.
    return {**existing, "OPENAI_API_KEY": "", "CODEX_API_KEY": "",
            "OPENAI_BASE_URL": "https://api.openai.com/v1"}


def public_account(response: Any) -> dict:
    data = response.model_dump(by_alias=True, mode="json") if hasattr(response, "model_dump") else response
    account = data.get("account") or {}
    return {"authenticated": account.get("type") == "chatgpt", "type": account.get("type"),
            "email": account.get("email"), "plan": account.get("planType"),
            "requires_auth": data.get("requiresOpenaiAuth", True)}


def runtime_versions() -> dict:
    result = {}
    for key, package in (("sdk", "openai-codex"), ("runtime", "openai-codex-cli-bin")):
        try:
            result[key] = version(package)
        except PackageNotFoundError:
            result[key] = "unknown"
    return result


async def require_subscription(codex: Any) -> dict:
    account = public_account(await codex.account())
    if not account["authenticated"]:
        raise ValueError("ChatGPT subscription mode requires Codex-managed ChatGPT sign-in. "
                         "Open Models & account and sign in; API-key authentication is not used.")
    return account


class CodexAccountService:
    def __init__(self, client_factory=None):
        self.client_factory = client_factory
        self.client = None
        self.lock = asyncio.Lock()
        self.login: dict = {"status": "idle"}
        self.login_task = None

    async def _connect(self):
        if self.client is None:
            from openai_codex import CodexConfig
            from openai_codex.async_client import AsyncCodexClient
            config = CodexConfig(config_overrides=('model_provider="openai"',),
                                 env=subscription_environment(dict(os.environ)),
                                 client_name="brilliance_account", client_title="Brilliance account")
            self.client = (self.client_factory or AsyncCodexClient)(config)
            try:
                await asyncio.wait_for(self.client.start(), 20)
                await asyncio.wait_for(self.client.initialize(), 20)
            except BaseException:
                await self.close()
                raise
        return self.client

    async def _request(self, method, params=None):
        from pydantic import RootModel
        client = await self._connect()
        return (await asyncio.wait_for(client.request(method, params, response_model=RootModel[dict]), 20)).root

    async def models(self):
        result, cursor = [], None
        for _ in range(20):
            page = await self._request("model/list", {"includeHidden": False, "limit": 100, "cursor": cursor})
            for item in page.get("data", []):
                if not item.get("hidden"):
                    result.append({key: item.get(key) for key in (
                        "id", "model", "displayName", "isDefault", "defaultReasoningEffort", "supportedReasoningEfforts")})
            following = page.get("nextCursor")
            if not following:
                return result
            if following == cursor:
                raise RuntimeError("Codex model pagination did not advance")
            cursor = following
        raise RuntimeError("Codex model catalog exceeded pagination bound")

    async def status(self):
        async with self.lock:
            client = await self._connect()
            account = public_account(await asyncio.wait_for(client.account_read({"refreshToken": False}), 20))
            result = {"account": account, "login": dict(self.login), "models": [],
                      "runtime_versions": runtime_versions(),
                      "usage": None, "usage_status": "not_authenticated", "warnings": []}
            if account["authenticated"]:
                try:
                    result["models"] = await self.models()
                except Exception as exc:
                    result["warnings"].append(f"Model discovery unavailable ({type(exc).__name__}); no model names were guessed.")
                try:
                    limits = await self._request("account/rateLimits/read")
                    result["usage"] = {key: limits.get(key) for key in ("rateLimits", "rateLimitsByLimitId")}
                    result["usage_status"] = "available"
                except Exception as exc:
                    result["usage_status"] = "unavailable"
                    result["warnings"].append(f"Codex usage information unavailable ({type(exc).__name__}); remaining quota is unknown.")
            return result

    async def start_login(self, method="browser"):
        if method not in {"browser", "device"}:
            raise ValueError("Sign-in method must be browser or device")
        async with self.lock:
            if self.login_task and not self.login_task.done():
                return dict(self.login)
            client = await self._connect()
            reply = await asyncio.wait_for(client.account_login_start(
                {"type": "chatgpt" if method == "browser" else "chatgptDeviceCode"}), 30)
            data = reply.model_dump(by_alias=True, mode="json")
            url = data.get("authUrl") or data.get("verificationUrl")
            if not isinstance(url, str) or urlsplit(url).scheme != "https":
                raise ValueError("Codex returned no HTTPS sign-in URL")
            self.login = {"status": "pending", "login_id": data["loginId"],
                          "url": url, "user_code": data.get("userCode"), "method": method}
            self.login_task = asyncio.create_task(self._wait_login(client, data["loginId"]))
            return dict(self.login)

    async def _wait_login(self, client, login_id):
        try:
            completed = await asyncio.wait_for(client.wait_for_login_completed(login_id), 600)
            data = completed.model_dump(by_alias=True, mode="json")
            self.login = {"status": "completed" if data.get("success") else "failed",
                          "message": None if data.get("success") else "Codex sign-in did not complete; retry sign-in."}
        except asyncio.CancelledError:
            raise
        except Exception:
            with suppress(Exception):
                await asyncio.wait_for(client.account_login_cancel(login_id), 5)
            self.login = {"status": "failed", "message": "Codex sign-in timed out or disconnected; retry sign-in."}
        finally:
            client.unregister_login_notifications(login_id)

    async def cancel_login(self):
        async with self.lock:
            if self.client and self.login.get("status") == "pending":
                await asyncio.wait_for(self.client.account_login_cancel(self.login["login_id"]), 10)
            if self.login_task and not self.login_task.done():
                self.login_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.login_task
            self.login = {"status": "cancelled"}
            return dict(self.login)

    async def logout(self):
        await self.cancel_login()
        async with self.lock:
            client = await self._connect()
            await asyncio.wait_for(client.account_logout(), 20)
            return {"authenticated": False}

    async def close(self):
        if self.login_task and not self.login_task.done():
            self.login_task.cancel()
        client, self.client = self.client, None
        if client:
            await client.close()
        if self.login_task:
            with suppress(asyncio.CancelledError):
                await self.login_task
