"""Configured accelerator-research runtime exposed to the Codex harness."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_RUNTIME_CONFIG = Path(__file__).resolve().parent / "config" / "research_runtime.toml"
REQUIRED_SCIENTIFIC_IMPORTS = (
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "sympy",
    "h5py",
    "joblib",
)


class ResearchEnvironmentError(RuntimeError):
    """Raised when the configured scientific runtime is incomplete."""


@dataclass(frozen=True, slots=True)
class ResearchEnvironment:
    config_path: Path
    source_root: Path
    python_executable: Path
    jutrack_root: Path
    pyjutrack_module: Path
    pyjutrack_import: str
    julia_project: Path
    python_julia_environment: Path
    julia_depot: Path
    julia_executable: Path
    libjulia: Path
    sandbox_alias_root: Path
    project_store: Path
    handbook: Path
    accelerator_api: Path
    minimum_julia: str
    julia_packages_offline: bool
    allow_runtime_dependency_updates: bool
    candidate_wall_time_s: int
    codex_sandbox: str
    internet_access: bool
    local_mcp_enabled: bool
    remote_mcp_enabled: bool
    expert_paths: tuple[Path, ...]
    expert_strict: bool
    project_expert_modules_enabled: bool
    expert_max_context_chars: int
    auto_continue: bool
    max_autonomous_increments: int
    turn_wall_time_s: int
    require_scientist_checkpoint: bool
    compute_backend: str
    local_worker_processes: int
    threads_per_worker: int
    repair_max_attempts: int = 12
    repair_max_same_error: int = 3
    max_consecutive_timeouts: int = 2
    expert_modules_enabled: bool = True

    @classmethod
    def load(
        cls,
        config_path: Path | str | None = None,
        *,
        source_root: Path | str | None = None,
    ) -> "ResearchEnvironment":
        root = Path(source_root or Path(__file__).resolve().parent.parent).resolve()
        configured_value = Path(
            config_path or os.environ.get("BRILLIANCE_RUNTIME_CONFIG", "") or DEFAULT_RUNTIME_CONFIG
        )
        configured_path = (
            configured_value if configured_value.is_absolute() else root / configured_value
        ).resolve()
        if not configured_path.is_file():
            raise ResearchEnvironmentError(
                f"Research runtime config does not exist: {configured_path}"
            )
        with configured_path.open("rb") as handle:
            payload = tomllib.load(handle)
        if not isinstance(payload, Mapping):
            raise ResearchEnvironmentError("Research runtime config must be a TOML table.")

        python_config = _table(payload, "python")
        platform_key = "windows" if os.name == "nt" else "posix"
        configured_python = str(python_config.get(platform_key, "")).strip()
        if not configured_python:
            raise ResearchEnvironmentError(
                f"Research runtime config must define [python].{platform_key}; "
                "Brilliance never selects a Python interpreter at runtime."
            )
        python_path = _resolve(root, configured_python)
        if not python_path.is_file():
            raise ResearchEnvironmentError(
                "Configured Python executable does not exist: "
                f"{python_path}. Run scripts/setup_env.py or correct "
                f"[python].{platform_key}; the current interpreter will not be used "
                "as a fallback."
            )

        jutrack = _table(payload, "jutrack")
        paths = _table(payload, "paths")
        execution = _table(payload, "execution")
        mcp = _table(payload, "mcp")
        experts = _table(payload, "experts")
        interaction = _table(payload, "interaction")
        compute = _table(payload, "compute")
        python_julia_environment = _resolve(
            root,
            str(jutrack.get("python_julia_environment", "")),
        )
        julia_metadata = _read_juliacall_metadata(python_julia_environment)
        environment = cls(
            config_path=configured_path,
            source_root=root,
            python_executable=python_path,
            jutrack_root=_resolve(root, str(jutrack.get("root", ""))),
            pyjutrack_module=_resolve(root, str(jutrack.get("python_module", ""))),
            pyjutrack_import=str(jutrack.get("python_import", "pyJuTrack")),
            julia_project=_resolve(root, str(jutrack.get("julia_project", ""))),
            python_julia_environment=python_julia_environment,
            julia_depot=_resolve(root, str(jutrack.get("julia_depot", ""))),
            julia_executable=_resolve_runtime_binary(
                root,
                str(jutrack.get("julia_executable", "auto")),
                julia_metadata,
                "executable",
            ),
            libjulia=_resolve_runtime_binary(
                root,
                str(jutrack.get("libjulia", "auto")),
                julia_metadata,
                "libjulia",
            ),
            sandbox_alias_root=_resolve(
                root,
                str(jutrack.get("sandbox_alias_root", "~/.julia/brilliance_runtime")),
            ),
            project_store=_resolve(root, str(paths.get("project_store", "research_projects"))),
            handbook=_resolve(root, str(paths.get("handbook", ""))),
            accelerator_api=_resolve(root, str(paths.get("accelerator_api", ""))),
            minimum_julia=str(jutrack.get("minimum_julia", "1.10")),
            julia_packages_offline=bool(execution.get("julia_packages_offline", True)),
            allow_runtime_dependency_updates=bool(
                execution.get("allow_runtime_dependency_updates", False)
            ),
            candidate_wall_time_s=int(execution.get("candidate_wall_time_s", 600)),
            codex_sandbox=str(execution.get("codex_sandbox", "workspace-write")).strip().lower(),
            internet_access=bool(execution.get("internet_access", True)),
            local_mcp_enabled=bool(mcp.get("local_stdio_enabled", True)),
            remote_mcp_enabled=bool(mcp.get("remote_http_enabled", False)),
            expert_paths=tuple(
                _resolve(root, str(item)) for item in experts.get("paths", ("expert_modules",))
            ),
            expert_strict=bool(experts.get("strict", True)),
            expert_modules_enabled=bool(experts.get("enabled", True)),
            project_expert_modules_enabled=bool(experts.get("project_modules_enabled", True)),
            expert_max_context_chars=int(experts.get("max_context_chars", 12_000)),
            auto_continue=bool(interaction.get("auto_continue", False)),
            max_autonomous_increments=int(interaction.get("max_autonomous_increments", 1)),
            turn_wall_time_s=int(interaction.get("turn_wall_time_s", 1800)),
            require_scientist_checkpoint=bool(
                interaction.get("require_scientist_checkpoint", True)
            ),
            compute_backend=str(compute.get("backend", "local_process")).strip().lower(),
            local_worker_processes=int(compute.get("local_worker_processes", 1)),
            threads_per_worker=int(compute.get("threads_per_worker", 1)),
            repair_max_attempts=int(_table(payload, "recovery").get("max_attempts_per_request", 12)),
            repair_max_same_error=int(_table(payload, "recovery").get("max_attempts_per_error", 3)),
            max_consecutive_timeouts=int(interaction.get("max_consecutive_timeouts", 2)),
        )
        environment.validate()
        return environment

    def is_active_python(self) -> bool:
        """Return whether this process is running under the configured Python."""

        active = Path(sys.executable)
        try:
            return os.path.samefile(active, self.python_executable)
        except OSError:
            active_text = str(active.resolve()).casefold()
            configured_text = str(self.python_executable.resolve()).casefold()
            return active_text == configured_text

    def require_active_python(self, *, operation: str = "Brilliance") -> None:
        """Reject scientific execution under an unconfigured interpreter."""

        if self.is_active_python():
            return
        raise ResearchEnvironmentError(
            f"{operation} is running under {Path(sys.executable).resolve()}, but the "
            f"configured interpreter is {self.python_executable}. Restart through the "
            "configured interpreter; Brilliance will not substitute the active Python."
        )

    def validate(self) -> None:
        required_files = {
            "Python executable": self.python_executable,
            "pyJuTrack module": self.pyjutrack_module,
            "JuTrack Project.toml": self.julia_project / "Project.toml",
            "Python Julia environment Project.toml": (
                self.python_julia_environment / "Project.toml"
            ),
            "Python Julia environment Manifest.toml": (
                self.python_julia_environment / "Manifest.toml"
            ),
            "Python Julia environment metadata": (
                self.python_julia_environment / "pyjuliapkg" / "meta.json"
            ),
            "Julia executable": self.julia_executable,
            "libjulia": self.libjulia,
            "accelerator API": self.accelerator_api,
            "accelerator API reference": (
                self.accelerator_api.parent / "resources" / "accelerator_api_reference.md"
            ),
            "accelerator handbook": self.handbook,
        }
        missing = [
            f"{label}: {path}" for label, path in required_files.items() if not path.is_file()
        ]
        if not self.jutrack_root.is_dir():
            missing.append(f"JuTrack root: {self.jutrack_root}")
        if not self.julia_depot.is_dir():
            missing.append(f"Julia depot: {self.julia_depot}")
        if self.is_active_python():
            failed_imports: list[str] = []
            for name in REQUIRED_SCIENTIFIC_IMPORTS:
                if importlib_util.find_spec(name) is None:
                    failed_imports.append(name)
            if failed_imports:
                missing.append(
                    "Configured Python is missing required scientific packages: "
                    f"{failed_imports}; run `python scripts/setup_env.py` outside the research agent"
                )
        if self.codex_sandbox not in {"workspace-write", "full-access"}:
            missing.append(
                f"Codex sandbox must be 'workspace-write' or 'full-access': {self.codex_sandbox!r}"
            )
        if not self.internet_access and self.codex_sandbox != "workspace-write":
            missing.append(
                "internet_access=false requires codex_sandbox='workspace-write' so shell "
                "networking is denied below the model"
            )
        if self.remote_mcp_enabled and not self.internet_access:
            missing.append("mcp.remote_http_enabled=true requires execution.internet_access=true")
        if not 1 <= self.max_autonomous_increments <= 20:
            missing.append("interaction.max_autonomous_increments must be in [1, 20]")
        if self.turn_wall_time_s < 30:
            missing.append("interaction.turn_wall_time_s must be at least 30 seconds")
        if self.repair_max_attempts < 0 or self.repair_max_same_error < 1:
            missing.append("recovery requires max_attempts_per_request >= 0 and max_attempts_per_error >= 1")
        if self.max_consecutive_timeouts < 1:
            missing.append("interaction.max_consecutive_timeouts must be positive")
        if self.candidate_wall_time_s < 1:
            missing.append("execution.candidate_wall_time_s must be positive")
        elif self.candidate_wall_time_s > self.turn_wall_time_s:
            missing.append(
                "execution.candidate_wall_time_s cannot exceed interaction.turn_wall_time_s"
            )
        if self.compute_backend not in {"serial", "local_process"}:
            missing.append(
                f"compute.backend must be 'serial' or 'local_process': {self.compute_backend!r}"
            )
        if not 1 <= self.local_worker_processes <= 64:
            missing.append("compute.local_worker_processes must be in [1, 64]")
        if not 1 <= self.threads_per_worker <= 64:
            missing.append("compute.threads_per_worker must be in [1, 64]")
        if not 4_000 <= self.expert_max_context_chars <= 500_000:
            missing.append("experts.max_context_chars must be in [4000, 500000]")
        if missing:
            raise ResearchEnvironmentError(
                "Configured accelerator runtime is incomplete:\n- " + "\n- ".join(missing)
            )

    def process_environment(self, existing: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return the exact subprocess environment Codex research commands receive."""

        base = dict(os.environ if existing is None else existing)
        path_entries = [str(self.python_executable.parent)]
        current_path = str(base.get("PATH", "")).strip()
        if current_path:
            path_entries.append(current_path)
        python_paths = [str(self.source_root), str(self.pyjutrack_module.parent)]
        current_pythonpath = str(base.get("PYTHONPATH", "")).strip()
        if current_pythonpath:
            python_paths.append(current_pythonpath)
        base.update(
            {
                "PATH": os.pathsep.join(path_entries),
                "PYTHONPATH": os.pathsep.join(python_paths),
                # Windows otherwise inherits the active ANSI code page (GBK on
                # many laboratory machines). Research reports intentionally use
                # symbols such as beta, epsilon, subscripts, and superscripts, so
                # every configured Python child must use UTF-8 for stdio as well
                # as files. Override conflicting parent values deliberately.
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONLEGACYWINDOWSSTDIO": "0",
                "MPLBACKEND": "Agg",
                "BRILLIANCE_PYTHON": str(self.python_executable),
                "PYTHON_EXECUTABLE": str(self.python_executable),
                "BRILLIANCE_SOURCE_ROOT": str(self.source_root),
                "BRILLIANCE_RUNTIME_CONFIG": str(self.config_path),
                "JUTRACK_ROOT": str(self.jutrack_root),
                "JUTRACK_PYTHON": str(self.pyjutrack_module),
                "JUTRACK_JL_PROJECT": str(self.julia_project),
                "JULIA_PROJECT": str(self.julia_project),
                "PYTHON_JULIACALL_EXE": str(self.julia_executable),
                "PYTHON_JULIACALL_BINDIR": str(self.julia_executable.parent),
                "JULIA_PKG_OFFLINE": "true" if self.julia_packages_offline else "false",
                "PYTHON_JULIAPKG_OFFLINE": "yes" if self.julia_packages_offline else "no",
                "BRILLIANCE_JULIA_RESOLVE": (
                    "true" if self.allow_runtime_dependency_updates else "false"
                ),
                "BRILLIANCE_INTERNET_ACCESS": ("true" if self.internet_access else "false"),
                "BRILLIANCE_LOCAL_MCP": ("true" if self.local_mcp_enabled else "false"),
                "BRILLIANCE_REMOTE_MCP": ("true" if self.remote_mcp_enabled else "false"),
                "BRILLIANCE_EXPERT_PATH": (
                    os.pathsep.join(str(item) for item in self.expert_paths)
                    if self.expert_modules_enabled else ""
                ),
                "JULIA_NUM_THREADS": str(self.threads_per_worker),
                "OPENBLAS_NUM_THREADS": str(self.threads_per_worker),
                "OMP_NUM_THREADS": str(self.threads_per_worker),
                "MKL_NUM_THREADS": str(self.threads_per_worker),
            }
        )
        return base

    def project_process_environment(
        self,
        workspace: Path | str,
        existing: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return a configured environment safe for one project worker.

        A managed worker may be created by a command already running inside the
        Codex workspace sandbox. On Windows that child inherits the restricted
        token, so sending it back to the original Desktop/AppData spellings can
        make Julia's long-path probe fail even though the configured Python is
        correct. Keep every Julia/JuTrack path on the aliases prepared outside
        the sandbox instead.
        """

        inherited = os.environ if existing is None else existing
        workspace_path = Path(workspace).resolve()
        alias_directory = self.sandbox_alias_root / workspace_path.parent.name
        expected_project = alias_directory / "julia_env"
        expected_jutrack = alias_directory / "JuTrack.jl"
        inherited_project = Path(
            str(inherited.get("PYTHON_JULIACALL_PROJECT", "") or "__missing__")
        ).absolute()
        inherited_jutrack = Path(str(inherited.get("JUTRACK_ROOT", "") or "__missing__")).absolute()
        aliases_already_active = (
            os.name == "nt"
            and self.codex_sandbox == "workspace-write"
            and str(inherited_project).casefold() == str(expected_project).casefold()
            and str(inherited_jutrack).casefold() == str(expected_jutrack).casefold()
        )
        if aliases_already_active:
            # Do not revalidate junction targets from inside the sandbox. The
            # unrestricted parent created these aliases before Codex started;
            # Win32 target-resolution calls are exactly what the aliases avoid.
            aliases = {
                "jutrack_root": expected_jutrack,
                "pyjutrack_module": Path(str(inherited["JUTRACK_PYTHON"])),
                "python_julia_environment": expected_project,
                "julia_executable": Path(str(inherited["PYTHON_JULIACALL_EXE"])),
            }
        else:
            aliases = self.prepare_sandbox_runtime_aliases(workspace_path)
        base = self.process_environment(existing)
        matplotlib_config = workspace_path / "runtime" / "matplotlib"
        matplotlib_config.mkdir(parents=True, exist_ok=True)
        project_environment = aliases["python_julia_environment"]
        jutrack_root = aliases["jutrack_root"]
        julia_executable = aliases.get("julia_executable", self.julia_executable)
        base.update(
            {
                "JULIA_DEPOT_PATH": str(self.julia_depot),
                "JUTRACK_ROOT": str(jutrack_root),
                "JUTRACK_PYTHON": str(aliases["pyjutrack_module"]),
                "JUTRACK_JL_PROJECT": str(jutrack_root),
                "JULIA_PROJECT": str(jutrack_root),
                "PYTHON_JULIACALL_EXE": str(julia_executable),
                "PYTHON_JULIACALL_BINDIR": str(julia_executable.parent),
                "MPLCONFIGDIR": str(matplotlib_config),
            }
        )
        if self.codex_sandbox == "workspace-write":
            base["PYTHON_JULIAPKG_PROJECT"] = str(project_environment)
            base["PYTHON_JULIACALL_PROJECT"] = str(project_environment)
        else:
            base.pop("PYTHON_JULIAPKG_PROJECT", None)
            base.pop("PYTHON_JULIACALL_PROJECT", None)
        return base

    def prepare_project_julia_environment(self, workspace: Path | str) -> Path:
        """Clone the small Juliapkg environment into the sandbox-writable workspace."""

        destination = Path(workspace).resolve() / "runtime" / "julia_env"
        for relative in (
            Path("Project.toml"),
            Path("Manifest.toml"),
            Path("pyjuliapkg") / "meta.json",
        ):
            source = self.python_julia_environment / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or source.read_bytes() != target.read_bytes():
                shutil.copy2(source, target)
        metadata_path = destination / "pyjuliapkg" / "meta.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "executable": str(self.julia_executable),
                "libjulia": str(self.libjulia),
                "override_executable": str(self.julia_executable),
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return destination

    def prepare_sandbox_runtime_aliases(self, workspace: Path | str) -> Mapping[str, Path]:
        """Create stable Windows junctions before the Codex sandbox starts.

        Julia calls ``GetLongPathNameW`` while initializing. The Windows Codex
        workspace sandbox permits ordinary reads of the configured source tree
        but rejects that Win32 call for Desktop/AppData prefixes. Preparing the
        aliases here keeps infrastructure repair outside the research agent.
        """

        workspace_path = Path(workspace).resolve()
        project_environment = self.prepare_project_julia_environment(workspace_path)
        if os.name != "nt" or self.codex_sandbox != "workspace-write":
            return {
                "jutrack_root": self.jutrack_root,
                "pyjutrack_module": self.pyjutrack_module,
                "python_julia_environment": project_environment,
                "julia_executable": self.julia_executable,
                "libjulia": self.libjulia,
            }
        project_id = workspace_path.parent.name
        alias_directory = self.sandbox_alias_root / project_id
        alias_directory.mkdir(parents=True, exist_ok=True)
        jutrack_alias = _ensure_windows_junction(self.jutrack_root, alias_directory / "JuTrack.jl")
        environment_alias = _ensure_windows_junction(
            project_environment, alias_directory / "julia_env"
        )
        julia_installation = self.julia_executable.parent.parent
        try:
            executable_relative = self.julia_executable.relative_to(julia_installation)
            libjulia_relative = self.libjulia.relative_to(julia_installation)
        except ValueError as exc:
            raise ResearchEnvironmentError(
                "Julia executable and libjulia must share one installation root"
            ) from exc
        julia_alias = _ensure_windows_junction(julia_installation, alias_directory / "julia")
        julia_executable = julia_alias / executable_relative
        libjulia = julia_alias / libjulia_relative
        _rewrite_project_runtime_paths(
            project_environment,
            jutrack_alias=jutrack_alias,
            julia_executable=julia_executable,
            libjulia=libjulia,
        )
        return {
            "jutrack_root": jutrack_alias,
            "pyjutrack_module": jutrack_alias / "python_integration" / "pyJuTrack.py",
            "python_julia_environment": environment_alias,
            "julia_executable": julia_executable,
            "libjulia": libjulia,
        }

    def manifest(self, *, project_root: Path | str | None = None) -> dict[str, Any]:
        """Return a non-secret manifest suitable for the agent and chat UI."""

        root = Path(project_root).resolve() if project_root else self.project_store
        return {
            "schema_version": "1.0",
            "config_file": str(self.config_path),
            "python": {
                "executable": str(self.python_executable),
                "command": _shell_command(self.python_executable),
                "selection": "configured_only",
                "active_process_matches": self.is_active_python(),
                "required_scientific_imports": list(REQUIRED_SCIENTIFIC_IMPORTS),
                "available_research_packages": _package_versions(
                    REQUIRED_SCIENTIFIC_IMPORTS
                ),
                "plotting_backend": "matplotlib:Agg",
            },
            "jutrack": {
                "root": str(self.jutrack_root),
                "python_module": str(self.pyjutrack_module),
                "python_import": self.pyjutrack_import,
                "julia_project": str(self.julia_project),
                "python_julia_environment_template": str(self.python_julia_environment),
                "julia_depot": str(self.julia_depot),
                "julia_executable": str(self.julia_executable),
                "libjulia": str(self.libjulia),
                "sandbox_alias_root": str(self.sandbox_alias_root),
                "minimum_julia": self.minimum_julia,
                "dependency_updates_allowed": self.allow_runtime_dependency_updates,
            },
            "brilliance": {
                "source_root": str(self.source_root),
                "project_store": str(root),
                "accelerator_api": str(self.accelerator_api),
                "accelerator_api_reference": str(
                    self.accelerator_api.parent / "resources" / "accelerator_api_reference.md"
                ),
                "handbook": str(self.handbook),
            },
            "execution": {
                "julia_packages_offline": self.julia_packages_offline,
                "candidate_wall_time_s": self.candidate_wall_time_s,
                "codex_sandbox": self.codex_sandbox,
                "internet_access": self.internet_access,
            },
            "mcp": {
                "local_stdio_enabled": self.local_mcp_enabled,
                "remote_http_enabled": self.remote_mcp_enabled,
            },
            "experts": {
                "enabled": self.expert_modules_enabled,
                "paths": [str(item) for item in self.expert_paths],
                "strict": self.expert_strict,
                "project_modules_enabled": self.project_expert_modules_enabled,
                "max_context_chars": self.expert_max_context_chars,
            },
            "interaction": {
                "source": str(self.config_path),
                "auto_continue": self.auto_continue,
                "max_autonomous_increments": self.max_autonomous_increments,
                "turn_wall_time_s": self.turn_wall_time_s,
                "require_scientist_checkpoint": self.require_scientist_checkpoint,
                "max_consecutive_timeouts": self.max_consecutive_timeouts,
            },
            "recovery": {
                "source": str(self.config_path),
                "max_attempts_per_request": self.repair_max_attempts,
                "max_attempts_per_error": self.repair_max_same_error,
                "unchanged_failure": "stop",
            },
            "compute": {
                "backend": self.compute_backend,
                "local_worker_processes": self.local_worker_processes,
                "threads_per_worker": self.threads_per_worker,
                "candidate_isolation": "one mutable JuTrack lattice per worker process",
            },
        }

    def render_manifest(self, *, project_root: Path | str | None = None) -> str:
        return (
            json.dumps(
                self.manifest(project_root=project_root),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    """Return installed versions without importing heavy scientific modules."""

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _table(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ResearchEnvironmentError(f"Runtime config section [{key}] must be a table.")
    return value


def _resolve(root: Path, value: str) -> Path:
    if not value.strip():
        return root / "__missing_configuration__"
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _read_juliacall_metadata(environment: Path) -> Mapping[str, Any]:
    path = environment / "pyjuliapkg" / "meta.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _resolve_runtime_binary(
    root: Path,
    configured: str,
    metadata: Mapping[str, Any],
    key: str,
) -> Path:
    value = str(configured).strip()
    if value.lower() == "auto":
        value = str(metadata.get(key, "")).strip()
    return _resolve(root, value)


def _shell_command(path: Path) -> str:
    return f"& '{str(path).replace(chr(39), chr(39) * 2)}'" if os.name == "nt" else str(path)


def _ensure_windows_junction(target: Path, link: Path) -> Path:
    """Create or reuse one exact directory junction without replacing data."""

    target = target.resolve()
    if not target.is_dir():
        raise ResearchEnvironmentError(f"Junction target is not a directory: {target}")
    if link.exists():
        try:
            if os.path.samefile(target, link):
                return link
        except OSError:
            pass
        raise ResearchEnvironmentError(
            f"Sandbox runtime alias exists but targets another directory: {link}"
        )
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, OSError) as exc:
        raise ResearchEnvironmentError(
            f"Could not create sandbox runtime alias {link} -> {target}: {exc}"
        ) from exc
    return link


def _rewrite_project_runtime_paths(
    environment: Path,
    *,
    jutrack_alias: Path,
    julia_executable: Path,
    libjulia: Path,
) -> None:
    """Point sandbox JuliaPkg state only at prepared non-Desktop aliases."""

    alias = jutrack_alias.as_posix()
    manifest_path = environment / "Manifest.toml"
    manifest = manifest_path.read_text(encoding="utf-8")
    pattern = r'(\[\[deps\.JuTrack\]\][\s\S]*?\npath\s*=\s*)"[^"]*"'
    rewritten, count = re.subn(pattern, rf'\1"{alias}"', manifest, count=1)
    # Current JuliaCall environments contain PythonCall only; JuTrack is loaded
    # from the separately configured JULIA_PROJECT/JUTRACK_ROOT. Older lock
    # environments also listed JuTrack as a development dependency. Rewrite
    # that optional legacy entry when present, but never require or synthesize
    # it in the PythonCall environment.
    if count == 1:
        manifest_path.write_text(rewritten, encoding="utf-8", newline="\n")

    metadata_path = environment / "pyjuliapkg" / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "executable": str(julia_executable),
            "libjulia": str(libjulia),
            "override_executable": str(julia_executable),
        }
    )
    packages = metadata.get("pkgs", [])
    if isinstance(packages, list):
        for package in packages:
            if isinstance(package, dict) and str(package.get("name")) == "JuTrack":
                package["path"] = alias
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
