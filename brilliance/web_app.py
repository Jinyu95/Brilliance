"""FastAPI application for the chat-native BRILLIANCE interface."""

import asyncio
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncIterator

from .artifacts import json_safe
from .chat_service import ResearchChatService
from .codex_backends import describe_codex_backend

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = Path(__file__).resolve().parent / "web" / "static"


def create_web_app(project_root: Path | str | None = None) -> Any:
    """Create the web application lazily so scientific CLI use needs no web imports."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import (
            FileResponse,
            HTMLResponse,
            PlainTextResponse,
            StreamingResponse,
        )
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised by launcher error path
        raise RuntimeError(
            "The chat UI requires FastAPI and Uvicorn. Run `pip install -e .` to install them."
        ) from exc

    root = Path(
        project_root
        or os.environ.get("BRILLIANCE_PROJECT_ROOT", "")
        or (_ROOT / "research_projects")
    ).resolve()
    service = ResearchChatService(root)
    from .codex_account import CodexAccountService
    from .model_settings import public_settings, read_settings, select_profile
    accounts = CodexAccountService()
    async def supervise_jobs() -> None:
        while True:
            await asyncio.to_thread(service.recover_jobs)
            await asyncio.sleep(5.0)

    @asynccontextmanager
    async def lifespan(app: Any):
        app.state.job_supervisor = asyncio.create_task(supervise_jobs())
        try:
            yield
        finally:
            await accounts.close()
            task = app.state.job_supervisor
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Brilliance Research Workbench", version="2.0", lifespan=lifespan)
    app.state.chat_service = service
    app.state.job_supervisor = None
    app.state.codex_accounts = accounts

    def settings_write_guard(request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(status_code=403, detail="Account and model controls require a same-origin request")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(status_code=403, detail="Cross-site account requests are not allowed")
        if any(service.jobs.active(pid, action="codex_turn") for pid in service.store.list_project_ids()):
            raise HTTPException(status_code=409, detail="Wait for or pause the active model turn before changing models/account. Numerical jobs need not be stopped.")

    @app.get("/api/settings/models")
    async def model_settings() -> MappingResponse:
        return {**public_settings(), "backend": describe_codex_backend()}

    @app.post("/api/settings/models")
    async def set_model(request: Request) -> MappingResponse:
        settings_write_guard(request)
        body = await request.json()
        profile = str(body.get("profile", ""))
        model = body.get("model")
        try:
            data = read_settings()
            if data.get("profiles", {}).get(profile, {}).get("provider") == "chatgpt":
                status = await accounts.status()
                if not status["account"]["authenticated"]:
                    raise ValueError("Sign in with ChatGPT before selecting subscription mode")
                chosen = model if model is not None else data["profiles"][profile].get("model", "")
                if chosen and chosen not in {m["model"] for m in status["models"]}:
                    raise ValueError("Choose a model reported by Codex model discovery")
            settings_write_guard(request)
            return select_profile(profile, model)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/account/codex")
    async def codex_account() -> MappingResponse:
        try:
            return await accounts.status()
        except Exception as exc:
            return {"account": {"authenticated": False}, "models": [], "usage": None,
                    "usage_status": "unavailable", "error": f"Codex account service unavailable ({type(exc).__name__}). Check the installed official Codex runtime."}

    @app.post("/api/account/codex/{action}")
    async def codex_account_action(action: str, request: Request) -> MappingResponse:
        settings_write_guard(request)
        try:
            if action == "login":
                body = await request.json()
                return await accounts.start_login(body.get("method", "browser"))
            if action == "cancel":
                return await accounts.cancel_login()
            if action == "logout":
                return await accounts.logout()
            raise ValueError("Unknown account action")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Codex {action} unavailable ({type(exc).__name__}). Check the official Codex runtime.") from exc

    def api_error(exc: Exception) -> HTTPException:
        if isinstance(exc, LookupError):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, ValueError):
            return HTTPException(status_code=400, detail=str(exc))
        return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    @app.get("/api/health")
    async def health() -> MappingResponse:
        try:
            from .research_environment import ResearchEnvironment
            from .runtime_health import cached_runtime_readiness

            runtime = cached_runtime_readiness(ResearchEnvironment.load())
        except Exception as exc:
            runtime = {
                "status": "failed",
                "ready": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        return {
            "status": "ok" if runtime.get("ready") is True else "degraded",
            "project_root": str(service.root),
            "codex_backend": describe_codex_backend(),
            "jutrack_runtime": runtime,
        }

    @app.get("/api/projects")
    async def projects() -> MappingResponse:
        return {"projects": await asyncio.to_thread(service.list_projects)}

    @app.post("/api/projects", status_code=201)
    async def new_project(request: dict[str, Any]) -> MappingResponse:
        try:
            goal = str(request.get("goal", "")).strip()
            context = str(request.get("context", "")).strip()
            if len(goal) < 12:
                raise ValueError("Describe the research goal in at least 12 characters")
            if len(goal) > 20_000 or len(context) > 100_000:
                raise ValueError("The research request is too long")
            return json_safe(await asyncio.to_thread(service.create, goal, context))
        except Exception as exc:
            raise api_error(exc) from exc

    @app.get("/api/projects/{project_id}")
    async def project_snapshot(project_id: str) -> MappingResponse:
        try:
            return json_safe(await asyncio.to_thread(service.snapshot, project_id))
        except Exception as exc:
            raise api_error(exc) from exc

    @app.post("/api/projects/{project_id}/checkpoints/{checkpoint_id}/branch", status_code=201)
    async def branch_checkpoint(project_id: str, checkpoint_id: str) -> MappingResponse:
        try:
            return json_safe(await asyncio.to_thread(service.branch_checkpoint, project_id, checkpoint_id))
        except Exception as exc:
            raise api_error(exc) from exc

    @app.post("/api/projects/{project_id}/messages")
    async def project_message(project_id: str, request: dict[str, Any]) -> MappingResponse:
        try:
            content = str(request.get("content", "")).strip()
            if len(content) > 20_000:
                raise ValueError("The message is too long")
            return json_safe(
                await asyncio.to_thread(service.send_message, project_id, content)
            )
        except Exception as exc:
            raise api_error(exc) from exc

    @app.post("/api/projects/{project_id}/computations/{job_id}/cancel")
    async def cancel_project_computation(
        project_id: str, job_id: str
    ) -> MappingResponse:
        try:
            return json_safe(
                await asyncio.to_thread(
                    service.cancel_computation, project_id, job_id
                )
            )
        except Exception as exc:
            raise api_error(exc) from exc

    @app.get("/api/projects/{project_id}/artifacts/{artifact_id}")
    async def project_artifact(project_id: str, artifact_id: str) -> MappingResponse:
        try:
            return json_safe(
                await asyncio.to_thread(service.artifact, project_id, artifact_id)
            )
        except Exception as exc:
            raise api_error(exc) from exc

    @app.get("/api/projects/{project_id}/lattices/{lattice_id}/{artifact}")
    async def lattice_package(
        project_id: str, lattice_id: str, artifact: str
    ) -> FileResponse:
        try:
            path = await asyncio.to_thread(
                service.lattice_file, project_id, lattice_id, artifact
            )
        except Exception as exc:
            raise api_error(exc) from exc
        suffix = {"definition": ".json", "reproduction": ".py", "provenance": ".json"}.get(
            artifact, path.suffix
        )
        media_type = "text/x-python" if suffix == ".py" else "application/json"
        return FileResponse(
            path,
            filename=f"{lattice_id}-{artifact}{suffix}",
            media_type=media_type,
        )

    @app.get("/api/projects/{project_id}/workspace-artifacts/{relative_path:path}")
    async def workspace_artifact(project_id: str, relative_path: str) -> FileResponse:
        try:
            path = await asyncio.to_thread(
                service.workspace_artifact_file, project_id, relative_path
            )
        except Exception as exc:
            raise api_error(exc) from exc
        from .codex_runtime import _workspace_artifact_media_type

        media_type = _workspace_artifact_media_type(path)
        return FileResponse(
            path,
            filename=path.name,
            media_type=media_type,
            content_disposition_type=(
                "inline" if media_type.startswith("image/") else "attachment"
            ),
        )

    @app.get("/api/projects/{project_id}/report", response_class=PlainTextResponse)
    async def project_report(project_id: str) -> PlainTextResponse:
        try:
            report = await asyncio.to_thread(service.report, project_id)
        except Exception as exc:
            raise api_error(exc) from exc
        return PlainTextResponse(
            report,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{project_id}-scientific-report.md"'
            },
        )

    @app.post("/api/projects/{project_id}/pause")
    async def pause_agent(project_id: str) -> MappingResponse:
        try:
            return dict(await asyncio.to_thread(service.pause_agent, project_id))
        except Exception as exc:
            raise api_error(exc) from exc

    @app.post("/api/projects/{project_id}/protocol-recovery")
    async def recover_protocol(project_id: str, request: dict[str, Any]) -> MappingResponse:
        try:
            return json_safe(await asyncio.to_thread(
                service.recover_protocol, project_id,
                action=str(request.get("action", "")),
                failure_id=str(request.get("failure_id", "")),
                request_id=str(request.get("request_id", "")),
            ))
        except Exception as exc:
            raise api_error(exc) from exc

    @app.get("/api/projects/{project_id}/programs/{name}")
    async def read_program(project_id: str, name: str) -> MappingResponse:
        try:
            workspace = await asyncio.to_thread(service.session_workspace, project_id)
            return await asyncio.to_thread(workspace.read_program, name)
        except Exception as exc:
            raise api_error(exc) from exc

    @app.post("/api/projects/{project_id}/programs", status_code=201)
    async def save_draft(project_id: str, request: dict[str, Any]) -> MappingResponse:
        try:
            workspace = await asyncio.to_thread(service.session_workspace, project_id)
            return await asyncio.to_thread(workspace.save_draft, str(request.get("name", "")), str(request.get("source", "")))
        except Exception as exc:
            raise api_error(exc) from exc

    @app.get("/api/projects/{project_id}/events")
    async def project_events(
        project_id: str, request: Request, version: str = ""
    ) -> StreamingResponse:
        try:
            await asyncio.to_thread(service.version, project_id)
        except Exception as exc:
            raise api_error(exc) from exc

        async def stream() -> AsyncIterator[str]:
            from .research_session import snapshot_delta

            # A fresh connection always receives a complete snapshot, including
            # reconnects after a missed event. No volatile in-memory replay buffer.
            last_version = None
            previous: dict[str, Any] | None = None
            keepalive = 0
            while not await request.is_disconnected():
                try:
                    current_version = await asyncio.to_thread(service.version, project_id)
                    if current_version != last_version:
                        current = json_safe(await asyncio.to_thread(service.snapshot, project_id))
                        event = "snapshot" if previous is None else "delta"
                        payload = json.dumps(
                            current if previous is None else snapshot_delta(previous, current),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        yield f"id: {current_version}\nevent: {event}\ndata: {payload}\n\n"
                        previous = current
                        last_version = current_version
                        keepalive = 0
                    else:
                        keepalive += 1
                        if keepalive >= 15:
                            yield ": keep-alive\n\n"
                            keepalive = 0
                except (GeneratorExit, asyncio.CancelledError):
                    break
                except Exception as exc:
                    error = json.dumps(
                        {"detail": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
                    )
                    yield f"event: error\ndata: {error}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    class RevalidatedStaticFiles(StaticFiles):
        async def get_response(self, path, scope):
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
            return response

    app.mount("/static", RevalidatedStaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/p/{project_id}", include_in_schema=False)
    async def frontend(project_id: str | None = None) -> HTMLResponse:
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        # Bind HTML to the actual installed assets, including after local upgrades.
        for name in ("styles.css", "workbench.css", "app.js"):
            digest = hashlib.sha256((_STATIC / name).read_bytes()).hexdigest()[:16]
            html = re.sub(rf'/static/{re.escape(name)}(?:\?[^"\s]*)?',
                          f"/static/{name}?v={digest}", html)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    return app


# FastAPI resolves annotations after the factory's local imports. A simple
# JSON-compatible alias keeps route declarations clear without importing its
# response model into the scientific runtime.
MappingResponse = dict[str, Any]


def run_web_app(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    project_root: Path | str | None = None,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Uvicorn is required for the BRILLIANCE chat UI") from exc
    uvicorn.run(
        create_web_app(project_root),
        host=host,
        port=port,
        log_level=os.environ.get("BRILLIANCE_WEB_LOG_LEVEL", "info"),
    )
