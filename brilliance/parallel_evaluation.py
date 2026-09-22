"""Ordered, process-isolated evaluation for optimizers and parameter scans.

Workers own mutable simulation state; the coordinator owns reporting and files.
Resource leases cover pools using this API, not arbitrary third-party processes.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any, Callable


class _Slots:
    def __init__(self, directory: Path, capacity: int, requested: int):
        directory.mkdir(parents=True, exist_ok=True)
        self.handles = []
        for index in range(capacity):
            handle = (directory / f"worker-{index}.lock").open("a+b")
            try:
                if os.name == "nt":
                    import msvcrt
                    if handle.seek(0, 2) == 0:
                        handle.write(b"0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                continue
            self.handles.append(handle)
            if len(self.handles) == requested:
                return
        if not self.handles:
            raise RuntimeError("All configured parallel worker slots are busy; retry when another pool closes")

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles.clear()


_evaluator = None


def _initialize(factory):
    global _evaluator
    _evaluator = factory()
    if not callable(_evaluator):
        raise TypeError("evaluator_factory must return a callable")


def _evaluate(value):
    return _evaluator(value)


class ParallelEvaluator:
    """Persistent factory per worker; ordered results; progress on each completion.

    Use as a context manager. Inputs/results must be serializable, never live
    JuTrack objects. Each call must reset *all* candidate-dependent worker state.
    Exceptions propagate without converting software failures into physics scores.
    """

    def __init__(self, evaluator_factory: Callable, *, workers: int | None = None,
                 on_result: Callable | None = None, cancellation_check: Callable | None = None,
                 environment=None):
        from .research_environment import ResearchEnvironment
        from .parallel_safety import validate_worker_factory
        from .population_optimization import _integer_option
        if os.environ.get("BRILLIANCE_EVALUATOR_WORKER") == "1":
            raise RuntimeError("Nested parallel evaluator pools are not allowed")
        validate_worker_factory(evaluator_factory)
        if not callable(evaluator_factory):
            raise TypeError("evaluator_factory must be callable")
        env = environment or ResearchEnvironment.load()
        if environment is None and Path(sys.executable).resolve() != env.python_executable.resolve():
            raise RuntimeError("Parallel evaluation must run with the configured Python interpreter")
        capacity = env.local_worker_processes if env.compute_backend == "local_process" else 1
        requested = capacity if workers is None else _integer_option(workers, "workers", 1, capacity)
        # Every pool using the same runtime configuration shares this allocation.
        identity = hashlib.sha256(str(env.config_path.resolve()).encode()).hexdigest()[:20]
        self._slots = _Slots(Path(tempfile.gettempdir()) / "brilliance-compute" / identity,
                             capacity, requested)
        self.workers = len(self._slots.handles)
        self.threads_per_worker = env.threads_per_worker
        self._closed = False
        self._on_result = on_result
        self._cancel = cancellation_check
        self._busy = False
        thread_env = {key: str(env.threads_per_worker) for key in
                      ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                       "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        thread_env["BRILLIANCE_EVALUATOR_WORKER"] = "1"
        try:
            from joblib.externals.loky import ProcessPoolExecutor
            # Even a one-worker allocation stays out of the coordinator's Julia runtime.
            self._pool = ProcessPoolExecutor(max_workers=self.workers, initializer=_initialize,
                                            initargs=(evaluator_factory,), env=thread_env)
        except BaseException:
            self._slots.close()
            raise

    def __call__(self, values):
        if self._closed or self._busy:
            raise RuntimeError("Evaluator is closed or already evaluating a batch")
        self._busy = True
        pending = {}
        started = time.monotonic()
        try:
            items = list(values)
            if self._cancel and self._cancel():
                raise InterruptedError("Evaluation cancelled before batch submission")
            results = [None] * len(items)
            pending = {self._pool.submit(_evaluate, item): i for i, item in enumerate(items)}
            for completed, future in enumerate(as_completed(pending), 1):
                index = pending[future]
                result = future.result()
                results[index] = result
                if self._on_result:
                    self._on_result(index, result, {"completed": completed, "total": len(items),
                        "workers": self.workers, "elapsed_s": time.monotonic() - started})
                if self._cancel and self._cancel():
                    raise InterruptedError("Evaluation cancelled at a completed evaluation boundary")
            return results
        finally:
            for future in pending:
                future.cancel()  # Does not interrupt already executing native calculations.
            self._busy = False

    def scipy_objective(self, _position):
        """Objective marker for SciPy; actual evaluation belongs to worker factories."""
        raise RuntimeError("Use pool.scipy_objective with workers=pool.scipy_map and polish=False")

    def scipy_map(self, function, iterable):
        """SciPy workers adapter bound to this pool's single scalar objective.

        The explicit marker prevents silently ignoring a different objective
        supplied to SciPy. No live parent-owned simulation closure is serialized.
        """
        objective = getattr(function, "f", function)  # SciPy's _FunctionWrapper
        if (getattr(objective, "__self__", None) is not self or
                getattr(objective, "__func__", None) is not ParallelEvaluator.scipy_objective):
            raise TypeError("Pass pool.scipy_objective as SciPy's func; define the scalar objective in the worker factory")
        return [float(value) for value in self(iterable)]

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._pool.shutdown(wait=True)
            finally:
                self._slots.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
