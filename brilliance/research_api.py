"""Small runtime API for model-authored accelerator research programs.

The program, rather than a scenario-specific Brilliance function, owns the
scientific parameterization and calculation.  Brilliance supplies only a
workspace, immutable inputs, progress publication, and a typed result file.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypedDict

from .api_result_types import ScientificStage
from .artifacts import json_safe
from .atomic_io import atomic_write_text

_log = logging.getLogger(__name__)


class ProgressInputError(ValueError):
    """A malformed presentation payload, never a numerical or physics failure."""


_PROGRESS_KINDS = frozenset({"plan", "progress", "finding", "decision", "repair", "next"})
_PROGRESS_KIND_ALIASES = {"activity": "progress", "milestone": "finding"}


class OptimizationControls(TypedDict):
    """Exact kwargs supplied by the runtime, never physics inputs or an initial seed."""
    progress_callback: Callable[[Mapping[str, Any]], None]
    cancellation_check: Callable[[], None]
    checkpoint_path: Path
    evaluation_cache_directory: Path
    campaign_identity: str
    resume: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_path(name: str) -> Path:
    value = str(os.environ.get(name, "")).strip()
    if not value:
        raise RuntimeError(f"{name} is available only inside a Brilliance research program")
    return Path(value).resolve()


def _render_json(value: Any) -> str:
    return json.dumps(
        json_safe(value), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"


def _write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, _render_json(value))


def _optimization_source_prefix(source: str, name: str) -> str | None:
    """Conservatively isolate computation before a direct top-level PSO call.

    Every preceding statement, including evaluator/helper definitions and
    geometry, belongs to the identity. Only later reporting is excluded. Calls
    hidden inside loops/functions, aliased controls, or repeated campaign names
    are ambiguous and deliberately retain the full-program identity.
    """
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    controls = [node for node in ast.walk(module) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "optimization_options"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == name]
    if len(controls) != 1:
        return None
    control = controls[0]
    for index, statement in enumerate(module.body):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
            continue
        call = statement.value
        if not isinstance(call, ast.Call):
            continue
        function = call.func
        function_name = (function.attr if isinstance(function, ast.Attribute)
                         else function.id if isinstance(function, ast.Name) else "")
        if function_name != "particle_swarm_optimize":
            continue
        if not any(item.arg is None and item.value is control for item in call.keywords):
            continue
        prefix = ast.Module(body=module.body[:index + 1], type_ignores=[])
        later_bindings: set[str] = set()
        for later in module.body[index + 1:]:
            for node in ast.walk(later):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    later_bindings.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    later_bindings.add(node.name)
                elif isinstance(node, ast.alias):
                    later_bindings.add(node.asname or node.name.split(".")[0])
        prior_references = {node.id for node in ast.walk(prefix)
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        if prior_references & later_bindings:
            # A deferred evaluator could consult a global defined/rebound later.
            # Scope analysis cannot prove that harmless: keep the full source.
            return None
        return hashlib.sha256(ast.dump(prefix, include_attributes=False).encode("utf-8")).hexdigest()
    return None


def _input_digests(directory: Path) -> dict[str, str | None]:
    # These files are immutable snapshots supplied to the calculation by the
    # managed runner. Paths/run IDs themselves are intentionally not identities.
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if (directory / name).is_file() else None
            for name in ("arguments.json", "active-lattice.json", "latest-candidate.json")}


def _optimization_identity(
    workspace: Path, output: Path, name: str, normalized: str, checkpoint: Path,
) -> tuple[str, dict[str, Any]]:
    snapshot = output / "program.py"
    source = snapshot.read_bytes() if snapshot.is_file() else b""
    program_sha = hashlib.sha256(source).hexdigest() if source else "unavailable"
    prefix = _optimization_source_prefix(source.decode("utf-8-sig"), name) if source else None
    inputs = _input_digests(output)
    contract = {"program": prefix or program_sha, "inputs": inputs}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
    identity = f"{normalized}:computation-v1:{digest}"
    provenance: dict[str, Any] = {
        "program_sha256": program_sha,
        "identity_policy": "top_level_computation_prefix" if prefix else "full_program",
        "computation_prefix_sha256": prefix,
        "input_sha256": inputs,
    }

    # Old versions bound the checkpoint to the entire source file. Recover its
    # identity ONLY when the exact archived original proves that changes occur
    # after the optimizer and all managed input snapshots remain identical.
    if prefix and checkpoint.is_file():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            previous = str(saved.get("contract", {}).get("campaign_identity", ""))
        except (OSError, ValueError, AttributeError):
            previous = ""  # The optimizer still validates/rejects malformed checkpoints.
        legacy = re.fullmatch(re.escape(normalized) + r":([0-9a-f]{64})", previous)
        if legacy:
            for archived in sorted((workspace / "artifacts" / "programs").glob("*/program.py")):
                archived_source = archived.read_bytes()
                if hashlib.sha256(archived_source).hexdigest() != legacy.group(1):
                    continue
                if (_optimization_source_prefix(archived_source.decode("utf-8-sig"), name) == prefix
                        and _input_digests(archived.parent) == inputs):
                    identity = previous
                    provenance["compatible_legacy_source"] = str(archived.relative_to(workspace))
                break
    provenance["campaign_identity"] = identity
    return identity, provenance


@dataclass(frozen=True, slots=True)
class ResearchRun:
    """Paths and reporting operations supplied to one saved research program."""

    workspace: Path
    output_directory: Path
    arguments_file: Path
    result_file: Path
    progress_file: Path | None
    cancellation_file: Path | None
    working_lattice_file: Path | None
    latest_candidate_lattice_file: Path | None
    compute_job_id: str | None
    _progress_retry_after: dict[Path, float] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _publish_progress(self, path: Path, payload: Mapping[str, Any]) -> None:
        """UI telemetry is recoverable; checkpoint/result writes are not.

        Try each channel independently and avoid stalling every optimizer seed
        on a persistently locked file. A later update repairs the live view.
        Schema/serialization errors still propagate instead of being hidden.
        """
        rendered = _render_json(payload)  # Validate even during a telemetry cooldown.
        if time.monotonic() < self._progress_retry_after.get(path, 0.):
            return
        try:
            atomic_write_text(path, rendered)
        except OSError as exc:
            if path not in self._progress_retry_after:
                _log.warning("Progress publication delayed for %s: %s. Calculation continues; "
                             "scientific result/checkpoint writes remain mandatory.", path, exc)
            self._progress_retry_after[path] = time.monotonic() + 5.
        else:
            if self._progress_retry_after.pop(path, None) is not None:
                _log.warning("Progress publication recovered for %s.", path)

    @property
    def arguments(self) -> dict[str, Any]:
        value = json.loads(self.arguments_file.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("research program arguments must contain one JSON object")
        return dict(value)

    def artifact_path(self, relative: str) -> Path:
        """Return a writable path confined to this program's artifact directory."""

        destination = (self.output_directory / str(relative)).resolve()
        try:
            destination.relative_to(self.output_directory)
        except ValueError as exc:
            raise ValueError("research program artifact path escapes its output directory") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def checkpoint_candidate(self, candidate_lattice: Any, candidate_evaluation: Any, *,
                             name: str = "candidate") -> dict[str, Any]:
        """Persist the exact selected native evaluation before plot/table formatting.

        Pass the original result returned by api.evaluate/evaluate_periodic_cell,
        not a formatted summary. No matching or simulation is repeated. This is
        provenance only; selection and scientific acceptance remain separate.
        """
        from .candidate_evidence import verify_candidate_evaluation
        from .lattice import LatticeIR
        from .numerical_checkpoints import write_numerical_checkpoint

        candidate = (candidate_lattice if isinstance(candidate_lattice, LatticeIR)
                     else LatticeIR.from_dict(candidate_lattice))
        verified = verify_candidate_evaluation(candidate, candidate_evaluation, self.output_directory)
        if not verified["passed"]:
            raise ValueError("Cannot checkpoint an unverified candidate evaluation: " + "; ".join(verified["issues"]))
        return write_numerical_checkpoint("candidate_evaluation", {
            "name": str(name), "candidate_lattice": candidate.as_dict(),
            "candidate_evaluation": verified["receipt"],
        }, output_directory=self.output_directory)

    @contextmanager
    def reporting(self, *, checkpoint: Mapping[str, Any] | None = None):
        """Optional formatting only, after numerical work has been checkpointed.

        Put plots, table formatting and finish inside this block, not physics
        calculations. A formatting exception publishes the exact saved numerical
        checkpoint as partial and stops this program; no rematching, acceptance
        or selection is invented. Physics rejection and cancellation propagate.
        """
        from .numerical_checkpoints import ReportingStopped, read_numerical_checkpoint
        from .research_outcomes import ResearchInfeasible

        saved = read_numerical_checkpoint(checkpoint, output_directory=self.output_directory)
        try:
            yield
        except (ResearchInfeasible, ReportingStopped, InterruptedError):
            raise
        except Exception as exc:
            # Explicit presentation scope, not a catch-all around scientific work.
            diagnostic = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            self._publish_checkpoint_after_reporting_failure(saved["descriptor"], diagnostic)
            raise ReportingStopped(f"Optional reporting failed: {type(exc).__name__}: {exc}") from exc

    def _publish_checkpoint_after_reporting_failure(self, checkpoint, diagnostic) -> None:
        from .candidate_evidence import validate_saved_candidate_evaluation
        from .numerical_checkpoints import read_numerical_checkpoint
        from .scientific_reporting import normalize_scientific_outcome

        saved = read_numerical_checkpoint(checkpoint, output_directory=self.output_directory)
        raw = saved["result"]
        diagnostic_path = self.artifact_path("reporting_failure.json")
        _write_json(diagnostic_path, diagnostic)
        candidate_path = receipt = None
        tables = []
        data = {"numerical_checkpoint": saved["descriptor"], "reporting_failure": diagnostic}
        if saved["kind"] == "candidate_evaluation":
            receipt = raw["candidate_evaluation"]
            verified = validate_saved_candidate_evaluation(raw["candidate_lattice"], receipt, self.output_directory)
            if not verified["passed"]:
                raise ValueError("Cannot recover altered numerical evidence: " + "; ".join(verified["issues"]))
            candidate_path = "checkpoint_candidate_lattice.json"
            _write_json(self.artifact_path(candidate_path), raw["candidate_lattice"])
            tables.append(verified["table"])
            data["verified_candidate_metrics"] = verified["canonical_metrics"]
        elif saved["kind"] == "jutrack_line_match":
            data["match"] = raw
            tables.append({"title": "Saved JuTrack boundary residuals", "rows": raw.get("residuals", [])})
        payload = {
            "schema_version": "1.0", "status": "partial",
            "summary": ("Numerical work was checkpointed, but optional reporting failed. "
                        "The saved measurements are shown below; repair reporting without repeating matching or optimization."),
            "data": data, "tables": tables, "plots": [],
            "downloads": [{"path": saved["descriptor"]["path"], "title": "Exact numerical checkpoint"},
                          {"path": "reporting_failure.json", "title": "Optional reporting diagnostic"}],
            "methods": [], "limitations": [f"Reporting failure: {diagnostic['type']}: {diagnostic['message']}",
                                              "No new scientific acceptance or working-candidate selection was made."],
            "candidate_lattice": candidate_path, "candidate_evaluation": receipt,
            "candidate_handoff": {"disposition": "archived_branch" if candidate_path else "no_candidate"},
            "scientific_stage": {"outcome": "reporting_pending", "next_decision": "Repair presentation using the saved checkpoint."},
            "scientific_passed": saved["checks_passed"], "request_satisfied": False,
            "use_as_working_lattice": False,
            "scientific_acceptance": None,
            "completed_at": _now(),
        }
        payload = normalize_scientific_outcome(payload)
        _write_json(self.result_file, payload)  # Mandatory; cannot be swallowed as a UI failure.
        try:
            self.progress("Numerical checkpoint preserved; reporting needs repair", kind="finding",
                          event_key="result", detail=payload["summary"], tables=payload["tables"],
                          scientific_role="milestone", **payload["scientific_stage"])
        except Exception:
            _log.exception("Optional checkpoint presentation failed; the partial result is already saved")

    def expert_asset(self, module_id: str, relative: str) -> Path:
        """Resolve one registered expert reference asset without repository discovery."""

        manifest_path = self.workspace / "EXPERT_ASSETS.json"
        if not manifest_path.is_file():
            raise LookupError("no expert reference assets were staged for this project")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assets = manifest.get("assets", []) if isinstance(manifest, Mapping) else []
        requested_module = str(module_id).strip().lower()
        requested_relative = Path(str(relative).replace("\\", "/")).as_posix().lstrip("./")
        match = next(
            (
                item
                for item in assets
                if isinstance(item, Mapping)
                and str(item.get("module_id", "")).strip().lower() == requested_module
                and str(item.get("relative", "")) == requested_relative
            ),
            None,
        )
        if match is None:
            raise LookupError(
                f"expert asset is not registered: {module_id}/{requested_relative}"
            )
        path = (self.workspace / str(match["workspace_path"])).resolve()
        try:
            path.relative_to((self.workspace / "expert_assets").resolve())
        except ValueError as exc:
            raise ValueError("registered expert asset escapes the expert asset directory") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(match.get("sha256", "")):
            raise ValueError(f"expert asset failed its SHA-256 identity: {requested_relative}")
        return path

    def cancellation_requested(self) -> bool:
        return bool(self.cancellation_file and self.cancellation_file.is_file())

    def raise_if_cancelled(self) -> None:
        """Stop a numerical loop at its next candidate/batch boundary."""

        if self.cancellation_requested():
            raise InterruptedError("the scientist requested cancellation")

    def parallel_evaluator(self, evaluator_factory, *, workers=None, on_result=None,
                           stage="Parallel numerical evaluation"):
        """Configured process pool with per-completion progress in this research job."""
        from .parallel_evaluation import ParallelEvaluator

        def completed(index, result, status):
            if on_result is not None:
                on_result(index, result, status)
            self.progress(stage, completed=status["completed"], total=status["total"],
                          metrics={"workers": status["workers"],
                                   "batch_elapsed_s": status["elapsed_s"]})

        return ParallelEvaluator(evaluator_factory, workers=workers, on_result=completed,
                                 cancellation_check=self.cancellation_requested)

    def optimization_options(self, name: str) -> OptimizationControls:
        """Return durable control hooks for a PSO/MOPSO campaign.

        The directory is stable across saved-program attempts.  A compatible
        checkpoint resumes automatically, while the optimizer rejects a stale
        checkpoint whose variables, bounds, objectives, population, or seed
        changed. A direct top-level optimizer call is fingerprinted together
        with its preceding computation and immutable inputs; repairing later
        plots/tables does not discard or repeat that computation. Ambiguous
        program shapes remain bound to the entire source file.
        """

        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name).strip()).strip("-.")
        if not normalized:
            raise ValueError("optimization name must contain a letter or number")
        directory = (self.workspace / "optimizations" / normalized).resolve()
        try:
            directory.relative_to(self.workspace)
        except ValueError as exc:  # pragma: no cover - normalization is restrictive
            raise ValueError("optimization directory escapes the workspace") from exc
        checkpoint = directory / "checkpoint.json"
        started_marker = self.output_directory / "optimizer_evidence.json"
        campaign_identity, identity_evidence = _optimization_identity(
            self.workspace, self.output_directory, str(name), normalized, checkpoint,
        )

        def publish(update: Mapping[str, Any]) -> None:
            phase = str(update.get("phase", "optimization"))
            preparing = phase in {"budget", "baseline_matching", "baseline_objective_validation"}
            if not preparing and not started_marker.is_file():
                _write_json(
                    started_marker,
                    {
                        "schema_version": "1.0",
                        "campaign": normalized,
                        **identity_evidence,
                        "first_progress": dict(update),
                        "started_at": _now(),
                    },
                )
            generation = update.get("generation")
            population = update.get("population_size")
            evaluated = update.get("evaluated_in_generation")
            metrics = {
                str(key): value
                for key, value in update.items()
                if key
                not in {
                    "phase",
                    "history",
                }
                and isinstance(value, (str, int, float, bool))
            }
            best = update.get("best")
            if isinstance(best, Mapping):
                for key, value in best.items():
                    if isinstance(value, (int, float)):
                        metrics[f"best.{key}"] = value
            if phase == "budget":
                stage = f"Optimization budget: {normalized}"
                detail = str(update.get("summary", "Production search budget resolved."))
                completed = total = None
            elif phase == "baseline_matching":
                stage = f"Checking baseline match: {normalized}"
                detail = (f"Validating {update.get('matching_knobs', 0)} inner matching knobs "
                          f"before searching {update.get('outer_knobs', 0)} outer design knobs.")
                completed = total = None
            elif phase == "baseline_objective_validation":
                stage = f"Checking objective on the matched baseline: {normalized}"
                detail = "Production optimization has not started; verifying the evaluator contract first."
                completed = total = None
            elif phase == "feasible_population_screening":
                stage = f"Preparing feasible population: {normalized}"
                detail = (
                    f"Accepted {update.get('accepted_nonbaseline_seeds', 0)} of "
                    f"{update.get('required_nonbaseline_seeds', 0)} required seeds."
                )
                if update.get("pending_batch_size"):
                    detail += (f" Evaluating the next {update['pending_batch_size']} candidates "
                               f"with {update.get('parallel_workers', 1)} worker(s); "
                               "the first batch may include runtime initialization.")
                completed = update.get("accepted_nonbaseline_seeds")
                total = update.get("required_nonbaseline_seeds")
            elif phase == "population_evaluation":
                stage = f"Evaluating generation {generation}: {normalized}"
                detail = f"Completed {evaluated} of {population} candidate evaluations."
                completed = evaluated
                total = population
            elif phase == "generation":
                stage = f"Generation {generation} complete: {normalized}"
                feasible_count = update.get("feasible_count", 0)
                front_size = update.get("front_size", 0)
                detail = (
                    f"Feasible candidates: {feasible_count}; feasible Pareto front: {front_size}."
                    if feasible_count else
                    f"No feasible candidate yet; {front_size} non-dominated infeasible archive entries. "
                    "These are rejected trials, not usable designs."
                )
                completed = generation
                total = update.get("total_generations")
            elif phase == "resume":
                stage = f"Resuming optimization: {normalized}"
                detail = (
                    f"Restored generation {update.get('completed_generation', 0)}; "
                    f"{update.get('remaining_generations', 0)} remain."
                )
                completed = update.get("completed_generation")
                total = None
            else:
                stage = f"Optimization: {normalized}"
                detail = phase
                completed = None
                total = None
            self.progress(
                stage,
                detail=detail,
                completed=int(completed) if isinstance(completed, (int, float)) else None,
                total=int(total) if isinstance(total, (int, float)) else None,
                metrics=metrics,
                event_key=(f"optimization-budget:{normalized}" if phase == "budget"
                           else f"optimization:{normalized}"),
                kind="decision" if phase == "budget" else "progress",
            )

        return {
            "progress_callback": publish,
            "cancellation_check": self.raise_if_cancelled,
            "checkpoint_path": checkpoint,
            "evaluation_cache_directory": directory / "evaluations",
            "campaign_identity": campaign_identity,
            "resume": checkpoint.is_file(),
        }

    def progress(
        self,
        stage: str,
        *,
        detail: str = "",
        completed: int | None = None,
        total: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        tables: Sequence[Mapping[str, Any]] = (),
        plots: Sequence[Mapping[str, Any] | str | Path] = (),
        kind: str = "progress",
        event_key: str | None = None,
        stage_id: str | None = None,
        hypothesis_id: str | None = None,
        candidate_id: str | None = None,
        parent_candidate_id: str | None = None,
        candidate_lattice: Any | str | Path | None = None,
        outcome: str | None = None,
        scientific_role: str | None = None,
        interpretation: str | None = None,
        next_decision: str | None = None,
    ) -> None:
        """Publish live progress and retain one chat milestone per distinct stage.

        Repeated updates to the same stage replace that stage's timeline entry. A
        new stage becomes a durable chat message. This keeps population counters
        live without producing one message per particle, while preserving actual
        scientific decisions, tables, and figures across a long calculation.
        Set ``event_key`` for a live activity whose display title changes (for
        example an optimizer generation counter). Its latest state updates one
        message; numerical campaign history remains in its checkpoint.
        Use ``scientific_role='milestone'`` for a hypothesis, measured outcome,
        comparison or decision. Operational counters belong to ``'activity'``.
        ``kind`` selects the display category: plan, progress, finding, decision,
        repair or next. ``kind='activity'`` aliases progress; ``'milestone'``
        aliases finding. Unknown categories are published as neutral progress
        with their requested label and a presentation warning retained. A display
        category never certifies a scientific result or completes a request.
        Malformed tables, plot references or explicit scientific roles raise
        ``ProgressInputError``, a repairable reporting error, not infeasibility.
        Optional stage and candidate identities link independent program attempts
        without imposing a topic-specific workflow on the application.
        Optional ``candidate_lattice`` attaches its identity only. Tables, metrics
        and prose are published as authored; this method does not evaluate physics
        or infer acceptance requirements from column names.
        """

        if self.progress_file is None:
            return
        requested_kind = str(kind).strip().lower() or "progress"
        normalized_kind = _PROGRESS_KIND_ALIASES.get(requested_kind, requested_kind)
        presentation_warnings = []
        if normalized_kind not in _PROGRESS_KINDS:
            normalized_kind = "progress"
            presentation_warnings.append(
                f"Unknown progress kind {requested_kind!r}; published as neutral progress."
            )
        role = str(scientific_role).strip().lower() if scientific_role else (
            "milestone" if normalized_kind in {"plan", "finding", "decision", "next"} else "activity"
        )
        if role not in {"milestone", "activity"}:
            raise ProgressInputError("scientific_role must be milestone or activity")
        normalized_tables: list[dict[str, Any]] = []
        for table in tables:
            if not isinstance(table, Mapping):
                raise ProgressInputError("each progress table must be a mapping with a rows list")
            row = dict(table)
            table_rows = row.get("rows", [])
            if not isinstance(table_rows, list):
                raise ProgressInputError("progress table must contain a rows list")
            if any(not isinstance(item, Mapping) for item in table_rows):
                raise ProgressInputError("every progress table row must be a mapping of column names")
            normalized_tables.append(row)
        normalized_plots: list[dict[str, Any]] = []
        artifact_root = (self.workspace / "artifacts").resolve()
        for plot in plots:
            if not isinstance(plot, (str, Path, Mapping)):
                raise ProgressInputError("each progress plot must be a path or a mapping with a path")
            row = {"path": str(plot)} if isinstance(plot, (str, Path)) else dict(plot)
            path = Path(str(row.get("path", "")))
            resolved = (
                path.resolve()
                if path.is_absolute()
                else (self.output_directory / path).resolve()
            )
            try:
                relative = resolved.relative_to(artifact_root)
            except ValueError as exc:
                raise ProgressInputError(
                    "progress plot must be stored below workspace/artifacts"
                ) from exc
            if not resolved.is_file():
                raise ProgressInputError(f"progress plot does not exist: {path}")
            row["path"] = relative.as_posix()
            normalized_plots.append(row)
        if candidate_lattice is not None:
            from .lattice import LatticeIR
            supplied = candidate_lattice
            if isinstance(supplied, (str, Path)):
                path = Path(supplied)
                path = path.resolve() if path.is_absolute() else (self.output_directory / path).resolve()
                if not path.is_relative_to(self.workspace.resolve()):
                    raise ProgressInputError("progress candidate must belong to this workspace")
                supplied = json.loads(path.read_text(encoding="utf-8"))
            candidate = supplied if isinstance(supplied, LatticeIR) else LatticeIR.from_dict(supplied)
            candidate_id = candidate.lattice_id
        updated_at = _now()
        stage_text = str(stage).strip() or "Research progress"
        event_identity = str(event_key).strip() if event_key is not None else stage_text
        if not event_identity:
            raise ProgressInputError("progress event_key must not be empty")
        event_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", event_identity).strip("-.")[:80]
        event_digest = hashlib.sha256(event_identity.encode("utf-8")).hexdigest()[:12]
        event_id = f"research-program:{self.progress_file.stem}:{event_digest}"
        timeline = self.progress_file.parent / "events" / self.progress_file.stem
        timeline_name = f"{event_slug or 'stage'}-{event_digest}.json"
        timeline_path = timeline / timeline_name
        if timeline_path.is_file() and (not normalized_tables or not normalized_plots):
            try:
                previous = json.loads(timeline_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
            if isinstance(previous, Mapping):
                if not normalized_tables and isinstance(previous.get("tables"), list):
                    normalized_tables = [
                        dict(item) for item in previous["tables"] if isinstance(item, Mapping)
                    ]
                if not normalized_plots and isinstance(previous.get("plots"), list):
                    normalized_plots = [
                        dict(item) for item in previous["plots"] if isinstance(item, Mapping)
                    ]
        payload = {
            "schema_version": "1.0",
            "id": event_id,
            "event_key": event_identity,
            "kind": normalized_kind,
            "scientific_role": role,
            "stage_id": stage_id,
            "hypothesis_id": hypothesis_id,
            "candidate_id": candidate_id,
            "parent_candidate_id": parent_candidate_id,
            "outcome": outcome,
            "interpretation": interpretation,
            "next_decision": next_decision,
            "compute_job_id": self.compute_job_id,
            "timestamp": updated_at,
            "updated_at": updated_at,
            "stage": stage_text,
            "detail": str(detail),
            "completed": completed,
            "total": total,
            "metrics": dict(metrics or {}),
            "tables": normalized_tables,
            "plots": normalized_plots,
        }
        if requested_kind != normalized_kind:
            payload["requested_kind"] = requested_kind
        if presentation_warnings:
            payload["presentation_warnings"] = presentation_warnings
        self._publish_progress(self.progress_file, payload)
        self._publish_progress(timeline_path, payload)

    def finish(
        self, *, summary: str, status: str = "completed",
        data: Mapping[str, Any] | None = None,
        tables: Sequence[Mapping[str, Any]] = (),
        plots: Sequence[Mapping[str, Any] | str | Path] = (),
        downloads: Sequence[Mapping[str, Any] | str | Path] = (),
        methods: Sequence[str] = (), limitations: Sequence[str] = (),
        candidate_lattice: Any | str | Path | None = None,
        candidate_evaluation: Any | None = None,
        use_as_working_lattice: bool = False,
        scientific_passed: bool | None = None,
        request_satisfied: bool | None = None,
        acceptance: Mapping[str, Any] | None = None,
        scientific_stage: ScientificStage | None = None,
    ) -> Path:
        """Optionally publish results, without imposing a scientific workflow.

        Plain Python needs no finish call. Scientific flags are author declarations,
        not app certification. An optional candidate_evaluation archives its exact
        receipt; an absent receipt does not prevent publication or explicit selection.
        No simulation, plots, geometry policing or inferred acceptance runs here.
        Numerical checkpoints and result.json survive optional live-reporting errors.
        Publish measured quantities in data/tables and explain their significance in
        summary. File paths belong in downloads; naming selected.json does not publish
        its contents. Read saved results for reporting rather than repeating a calculation.
        """
        normalized_status = {"complete": "completed", "intermediate": "partial"}.get(
            str(status).strip().lower(), str(status).strip().lower())
        if normalized_status not in {"completed", "infeasible", "partial"}:
            raise ValueError("status must be completed, infeasible, or partial")
        for name, value in (("scientific_passed", scientific_passed),
                            ("request_satisfied", request_satisfied)):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be true, false, or null")
        result_data = dict(data or {})
        rendered_tables = [dict(table) for table in tables]
        for table in rendered_tables:
            if not isinstance(table.get("rows", []), list) or any(
                    not isinstance(row, Mapping) for row in table.get("rows", [])):
                raise ValueError("table rows must be a list of column-name mappings")

        def artifact_entries(items):
            rendered = []
            for item in items:
                entry = {"path": str(item)} if isinstance(item, (str, Path)) else dict(item)
                path = Path(str(entry.get("path", "")))
                resolved = path.resolve() if path.is_absolute() else (self.output_directory / path).resolve()
                try:
                    entry["path"] = resolved.relative_to(self.output_directory).as_posix()
                except ValueError as exc:
                    raise ValueError("published artifact must be inside the program output directory") from exc
                rendered.append(entry)
            return rendered

        rendered_plots = artifact_entries(plots)
        rendered_downloads = artifact_entries(downloads)
        candidate = None
        lattice_path = None
        verification = None
        if candidate_lattice is not None:
            from .lattice import LatticeIR
            if isinstance(candidate_lattice, (str, Path)):
                path = Path(candidate_lattice)
                resolved = path.resolve() if path.is_absolute() else (self.output_directory / path).resolve()
                if not resolved.is_relative_to(self.output_directory):
                    raise ValueError("candidate lattice must be inside the program output directory")
                candidate = LatticeIR.from_dict(json.loads(resolved.read_text(encoding="utf-8")))
            else:
                candidate = (candidate_lattice if isinstance(candidate_lattice, LatticeIR)
                             else LatticeIR.from_dict(candidate_lattice))
                resolved = self.artifact_path("candidate_lattice.json")
                _write_json(resolved, candidate.as_dict())
            lattice_path = resolved.relative_to(self.output_directory).as_posix()
            if candidate_evaluation is not None:
                from .candidate_evidence import verify_candidate_evaluation

                verification = verify_candidate_evaluation(
                    candidate, candidate_evaluation, self.output_directory)
                result_data["candidate_verification"] = verification

        acceptance_result = None
        if acceptance:
            from .scientific_acceptance import assess_scientific_acceptance, load_expert_registry

            acceptance_result = assess_scientific_acceptance(
                acceptance, required=(), lattice=candidate, evidence_directory=self.output_directory,
                registry=load_expert_registry(project_directory=self.workspace.parent))
        payload = {
            "schema_version": "1.0",
            "status": normalized_status,
            "summary": str(summary),
            "data": result_data,
            "tables": rendered_tables,
            "plots": rendered_plots,
            "downloads": rendered_downloads,
            "methods": list(methods),
            "limitations": list(limitations),
            "candidate_lattice": lattice_path,
            "candidate_handoff": {
                "disposition": ("working_selection_requested" if candidate is not None and use_as_working_lattice
                                else "archived_branch" if candidate is not None else "no_candidate"),
                "selection_requires_validated_execution": True,
                "request_completion_selects_lattice": False,
            },
            "candidate_evaluation": verification.get("receipt") if verification else None,
            "scientific_stage": dict(scientific_stage or {}),
            "use_as_working_lattice": bool(use_as_working_lattice),
            "scientific_passed": scientific_passed,
            "request_satisfied": request_satisfied,
            "scientific_acceptance": acceptance_result,
            "completed_at": _now(),
        }
        from .scientific_reporting import normalize_scientific_outcome

        payload = normalize_scientific_outcome(payload)
        _write_json(self.result_file, payload)
        if self.progress_file is not None:
            try:
                self.progress("Result published", kind="finding", event_key="result",
                              detail=payload["summary"], completed=1, total=1,
                              tables=rendered_tables, plots=rendered_plots,
                              scientific_role="milestone", **{
                                  key: value for key, value in payload["scientific_stage"].items()
                                  if key in {"stage_id", "hypothesis_id", "candidate_id", "parent_candidate_id",
                                             "outcome", "interpretation", "next_decision"}
                                  and (isinstance(value, str) or value is None)})
            except Exception as exc:
                # Presentation is optional. Never discard completed work or
                # restart a numerical campaign because a progress card failed.
                payload["presentation_warnings"] = [f"{type(exc).__name__}: {exc}"]
                _write_json(self.result_file, payload)
                _log.warning("Result preserved despite optional progress failure: %s", exc)
        return self.result_file


def open_research_run() -> ResearchRun:
    """Open the environment injected by ``run-program`` or ``submit-program``."""

    workspace = _required_path("BRILLIANCE_WORKSPACE")
    output = _required_path("BRILLIANCE_RUN_OUTPUT")
    try:
        output.relative_to(workspace)
    except ValueError as exc:
        raise RuntimeError("Brilliance run output is outside the project workspace") from exc
    optional = lambda name: (  # noqa: E731 - compact environment adapter
        Path(os.environ[name]).resolve() if str(os.environ.get(name, "")).strip() else None
    )
    return ResearchRun(
        workspace=workspace,
        output_directory=output,
        arguments_file=_required_path("BRILLIANCE_RUN_ARGUMENTS"),
        result_file=_required_path("BRILLIANCE_RUN_RESULT"),
        progress_file=optional("BRILLIANCE_RUN_PROGRESS"),
        cancellation_file=optional("BRILLIANCE_RUN_CANCEL"),
        working_lattice_file=optional("BRILLIANCE_WORKING_LATTICE"),
        latest_candidate_lattice_file=optional("BRILLIANCE_LATEST_CANDIDATE_LATTICE"),
        compute_job_id=str(os.environ.get("BRILLIANCE_COMPUTE_JOB_ID", "")).strip() or None,
    )
