"""Callback-based weighted fitting with explicit, bound-aware finite differences.

No automatic differentiation or machine-specific model is used here. Numerical
termination and local Jacobian diagnostics do not establish scientific validity.
"""
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import weakref

import numpy as np
from scipy.optimize import least_squares
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


def _save_checkpoint(path, parameters, prediction, metadata):
    """Atomic numerical snapshot, including retry for Windows reader locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, parameters=parameters, prediction=prediction,
                     metadata=np.array(json.dumps(metadata, allow_nan=False)))
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if attempt == 4 or not (isinstance(exc, PermissionError)
                                       or getattr(exc, "winerror", None) in (5, 32, 33)):
                    raise
                time.sleep(.01 * 2 ** attempt)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def load_fit_checkpoint(path):
    """Read saved numerical state without running JuTrack or applying settings.

    A best-evaluated checkpoint is NOT a solver iteration checkpoint. Resume
    starts a fresh solver from its parameters, after problem-identity checking.
    Predictions can contain inactive NaNs; use the saved active mask.
    """
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        parameters = np.array(saved["parameters"], dtype=float, copy=True)
        prediction = np.array(saved["prediction"], dtype=float, copy=True)
    if metadata.get("schema") != "brilliance.loco.fit.v1":
        raise ValueError("Unsupported fit checkpoint schema")
    if parameters.ndim != 1 or not np.isfinite(parameters).all() or prediction.ndim != 1:
        raise ValueError("Invalid numerical fit checkpoint")
    return {"parameter_values": parameters, "prediction": prediction, "metadata": metadata}


class _MappedArray:
    """One unique scratch file per Jacobian, owned as long as its operator lives."""
    def __init__(self, shape, directory):
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="loco-jacobian-", suffix=".dat", dir=directory)
        os.close(fd)
        self.array = np.memmap(path, mode="w+", dtype=float, shape=shape)
        self._cleanup = weakref.finalize(self, self._close, self.array._mmap, path)

    @staticmethod
    def _close(mapping, path):
        mapping.close()
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass  # Never hide the original fit failure during scratch cleanup.


def _block_operator(physical, physical_indices, analytic, analytic_indices, count, owner=None):
    """Dense/mapped physical block plus sparse analytic instrument derivatives."""
    rows = physical.shape[0]

    def matvec(value):
        _ = owner  # Keep mapped storage alive while SciPy retains the operator.
        value = np.asarray(value).reshape(count)
        return np.asarray(physical @ value[physical_indices] + analytic @ value[analytic_indices]).ravel()

    def rmatvec(value):
        _ = owner
        value = np.asarray(value).reshape(rows)
        result = np.empty(count)
        result[physical_indices] = np.asarray(physical.T @ value).ravel()
        result[analytic_indices] = np.asarray(analytic.T @ value).ravel()
        return result

    return LinearOperator((rows, count), matvec=matvec, rmatvec=rmatvec, dtype=float)


def fit_response_model(
    model, initial, lower, upper, parameter_scale, difference_step, observed, sigma,
    *, parameter_names, mask=None, prior_mean=None, prior_sigma=None, max_nfev,
    ftol=1e-8, xtol=1e-8, gtol=1e-8, progress=None,
    analytic_jacobian=None, analytic_parameters=(), batch_model=None, batch_size=16,
    dense_jacobian_max_bytes=64 * 1024 * 1024, jacobian_directory=None,
    checkpoint_path=None, checkpoint_context=None, resume=False,
):
    """Fit a deterministic ``model(parameter_vector) -> observation_vector``.

    Scales, absolute finite-difference steps and uncertainties may be positive
    scalars or vectors. Bounds and prior means are parameter-length vectors;
    infinite bounds are permitted. The observation mask is copied once: invalid
    active data/predictions fail, while inactive values never enter the fit.

    Central differences use the requested step where both sides fit in bounds;
    otherwise a one-sided step is shortened only as needed to stay in bounds.
    Priors add ``(parameters - prior_mean) / prior_sigma`` to the residual but
    are excluded from identifiability diagnostics. ``max_nfev`` limits SciPy's
    objective evaluations, NOT the total number of underlying model calls.

    Progress receives a fresh history record after each successful model call.
    Exceptions propagate; they are never converted into large penalty residuals.
    Final prediction and data Jacobian are independently recalculated and counted.
    Optional analytic_jacobian(p) supplies UNWEIGHTED full-observation derivatives
    for analytic_parameters (ordered names); only remaining columns use differences.
    Dense or SciPy sparse analytic blocks are accepted. batch_model(list_of_vectors)
    returns ordered predictions, e.g. api.parallel_evaluator with PRIVATE worker
    lattices. It is never invoked on a shared parent model in concurrent threads.

    Large Jacobians use mapped physical blocks, sparse analytic blocks and LSMR;
    no full SVD/list export is attempted beyond dense_jacobian_max_bytes. Such
    fits explicitly report identifiability as not computed, not full rank.
    Checkpoints preserve best evaluated objective states and final predictions
    BEFORE diagnostics/progress callbacks. Resume is a fingerprint-checked warm
    restart, not restoration of a trust region or an iteration counter. Supply
    checkpoint_context with hashes of the ring, measurements and model code.
    The parent model is restored to the returned fitted parameters on success,
    or the best evaluated objective state on failure (original errors propagate).

    Returned inactive residuals/nonfinite inactive predictions are JSON ``null``;
    a rank-deficient condition number is also ``null``, not infinity.
    """
    def vector(value, name, size=None, *, broadcast=False, finite=True):
        array = np.array(value, dtype=float, copy=True)
        if broadcast and array.ndim == 0:
            array = np.full(size, array.item())
        if array.ndim != 1 or not array.size or (size is not None and array.size != size):
            raise ValueError(f"{name} must be a nonempty vector of length {size}")
        if np.isnan(array).any() or (finite and not np.isfinite(array).all()):
            raise ValueError(f"{name} must contain valid {'finite ' if finite else ''}numbers")
        return array

    x0 = vector(initial, "initial")
    count = x0.size
    lo = vector(lower, "lower", count, finite=False)
    hi = vector(upper, "upper", count, finite=False)
    scale = vector(parameter_scale, "parameter_scale", count, broadcast=True)
    steps = vector(difference_step, "difference_step", count, broadcast=True)
    if np.any(lo >= hi) or np.any(x0 < lo) or np.any(x0 > hi):
        raise ValueError("Bounds must be ordered strictly and contain initial")
    if np.any(scale <= 0) or np.any(steps <= 0):
        raise ValueError("parameter_scale and difference_step must be strictly positive")
    if isinstance(parameter_names, (str, bytes)):
        raise ValueError("parameter_names must be a sequence, not a single string")
    names = list(parameter_names)
    if (len(names) != count or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != count):
        raise ValueError("parameter_names must be distinct nonempty strings, one per parameter")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)) or max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")
    for name, value in (("ftol", ftol), ("xtol", xtol), ("gtol", gtol)):
        if value is not None and (not np.isfinite(value) or value <= np.finfo(float).eps):
            raise ValueError(f"{name} must be None or finite and greater than machine epsilon")
    if all(value is None for value in (ftol, xtol, gtol)):
        raise ValueError("At least one termination tolerance must be enabled")
    if not callable(model) or (progress is not None and not callable(progress)):
        raise ValueError("model and optional progress must be callable")

    data = np.array(observed, dtype=float, copy=True)
    if data.ndim != 1 or not data.size:
        raise ValueError("observed must be a nonempty vector")
    active = np.ones(data.size, dtype=bool) if mask is None else np.array(mask, copy=True)
    if active.shape != data.shape or active.dtype != np.dtype(bool) or not active.any():
        raise ValueError("mask must be a boolean observation-length vector with active data")
    uncertainty = np.array(sigma, dtype=float, copy=True)
    if uncertainty.ndim == 0:
        uncertainty = np.full(data.size, uncertainty.item())
    if (uncertainty.shape != data.shape or not np.isfinite(data[active]).all()
            or not np.isfinite(uncertainty[active]).all() or np.any(uncertainty[active] <= 0)):
        raise ValueError("Active observed values must be finite and sigma finite and positive")
    if (prior_mean is None) != (prior_sigma is None):
        raise ValueError("prior_mean and prior_sigma must be supplied together")
    mean = None if prior_mean is None else vector(prior_mean, "prior_mean", count)
    prior = None if prior_sigma is None else vector(prior_sigma, "prior_sigma", count, broadcast=True)
    if prior is not None and np.any(prior <= 0):
        raise ValueError("prior_sigma must be strictly positive")

    if isinstance(analytic_parameters, (str, bytes)):
        raise ValueError("analytic_parameters must be ordered parameter names")
    analytic_names = list(analytic_parameters)
    if (any(not isinstance(name, str) for name in analytic_names)
            or len(set(analytic_names)) != len(analytic_names)
            or any(name not in names for name in analytic_names)
            or bool(analytic_names) != (analytic_jacobian is not None)
            or (analytic_jacobian is not None and not callable(analytic_jacobian))):
        raise ValueError("Supply a callable analytic_jacobian and distinct known analytic_parameters together")
    if batch_model is not None and not callable(batch_model):
        raise ValueError("batch_model must be an ordered batch prediction callable")
    for name, value in (("batch_size", batch_size), ("dense_jacobian_max_bytes", dense_jacobian_max_bytes)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(resume, bool) or (resume and checkpoint_path is None):
        raise ValueError("resume requires a checkpoint_path")
    # Callables are not stable identities; the caller supplies model/data hashes.
    context = json.loads(json.dumps(checkpoint_context, sort_keys=True, allow_nan=False))
    if resume and not context:
        raise ValueError("resume requires checkpoint_context identifying the model and inputs")
    fingerprint = hashlib.sha256(json.dumps(
        {"names": names, "analytic_names": analytic_names, "context": context},
        sort_keys=True, allow_nan=False).encode())
    for value in (lo, hi, scale, steps, data[active], uncertainty[active], active,
                  np.array([]) if mean is None else mean, np.array([]) if prior is None else prior):
        array = np.ascontiguousarray(value)
        fingerprint.update(str((array.shape, array.dtype.str)).encode())
        fingerprint.update(array.tobytes())
    identity = fingerprint.hexdigest()
    resumed = None
    if resume:
        resumed = load_fit_checkpoint(checkpoint_path)
        if resumed["metadata"].get("problem_fingerprint") != identity:
            raise ValueError("Fit checkpoint does not match this problem/context")
        x0 = resumed["parameter_values"]
        if x0.shape != lo.shape or np.any(x0 < lo) or np.any(x0 > hi):
            raise ValueError("Fit checkpoint parameters do not match current bounds")
    analytic_indices = np.array([names.index(name) for name in analytic_names], dtype=int)
    physical_indices = np.array([i for i in range(count) if names[i] not in analytic_names], dtype=int)
    rows = int(active.sum())
    large = (rows + (count if prior is not None else 0)) * count * np.dtype(float).itemsize > dense_jacobian_max_bytes

    evaluations = 0
    history = deque(maxlen=128)
    cache_x = cache_prediction = None
    best_x, best_cost = x0.copy(), float("inf")
    restore_x = None
    analytic_calls = 0
    batch_calls = 0
    batch_checked = False

    def checkpoint(parameters, prediction, kind, cost):
        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, parameters, prediction, {
                "schema": "brilliance.loco.fit.v1", "kind": kind,
                "problem_fingerprint": identity, "context": context,
                "parameter_names": names, "active_mask": active.tolist(),
                "objective_sum_squares": cost, "callback_evaluations": evaluations,
                "resume_semantics": "fresh solver from saved parameters; no optimizer state",
                "scientifically_accepted": False,
            })

    def weighted(prediction):
        residual = (prediction[active] - data[active]) / uncertainty[active]
        if not np.isfinite(residual).all():
            raise ValueError("Active weighted residuals must be finite")
        return residual

    def record(parameters, raw_prediction, phase, *, notify=True):
        nonlocal cache_x, cache_prediction, best_x, best_cost
        prediction = np.array(raw_prediction, dtype=float, copy=True)
        if prediction.shape != data.shape or not np.isfinite(prediction[active]).all():
            raise ValueError("Model prediction must match observed shape and be finite on the active mask")
        residual = weighted(prediction)
        chi_square = float(residual @ residual)
        if not np.isfinite(chi_square):
            raise ValueError("Weighted chi-square overflowed")
        record = {"evaluation": evaluations, "phase": phase, "chi_square": chi_square}
        history.append(record)
        cache_x, cache_prediction = parameters.copy(), prediction.copy()
        total = chi_square + (0 if prior is None else float(np.sum(((parameters - mean) / prior) ** 2)))
        if not np.isfinite(total):
            raise ValueError("Combined data and prior chi-square overflowed")
        if phase == "objective" and total < best_cost:
            best_x, best_cost = parameters.copy(), total
            checkpoint(parameters, prediction, "best_evaluated_objective", total)
        elif phase == "final_prediction":
            checkpoint(parameters, prediction, "solver_result_before_diagnostics", total)
        if notify and progress is not None:
            progress(dict(record))
        return prediction

    def evaluate(parameters, phase, *, reuse=False, notify=True):
        nonlocal evaluations
        if not np.isfinite(parameters).all():
            raise ValueError("Model parameter trials must be finite")
        if reuse and cache_x is not None and np.array_equal(parameters, cache_x):
            return cache_prediction.copy()
        evaluations += 1
        return record(parameters, model(parameters.copy()), phase, notify=notify)

    def differences(parameters, baseline, phase):
        nonlocal evaluations, analytic_calls, batch_calls, batch_checked
        if batch_model is not None and not batch_checked and len(physical_indices):
            batch_calls += 1
            evaluations += 1
            check = list(batch_model([parameters.copy()]))
            if len(check) != 1:
                raise ValueError("batch_model must return one ordered prediction per probe")
            prediction = record(parameters, check[0], "batch_model_check")
            if not np.allclose(prediction[active], baseline[active], rtol=1e-10, atol=1e-12):
                raise ValueError("batch_model does not reproduce the parent model at the baseline")
            batch_checked = True
        owner = _MappedArray((rows, len(physical_indices)), jacobian_directory) if large and len(physical_indices) else None
        physical = owner.array if owner is not None else np.empty((rows, len(physical_indices)))
        # A batch contains complete finite-difference columns, bounded in memory.
        tasks = []

        def flush():
            nonlocal evaluations, batch_calls
            if not tasks:
                return
            probes = [p for _, points, _ in tasks for p in points]
            if batch_model is None:
                values = [evaluate(p, phase)[active] for p in probes]
            else:
                batch_calls += 1
                evaluations += len(probes)
                raw = list(batch_model([p.copy() for p in probes]))
                if len(raw) != len(probes):
                    raise ValueError("batch_model must return one ordered prediction per probe")
                values = []
                # Count each completed prediction separately in public history.
                evaluations -= len(probes)
                for p, value in zip(probes, raw):
                    evaluations += 1
                    values.append(record(p, value, phase)[active])
            cursor = 0
            for column, points, denominator in tasks:
                delta = values[cursor] - (values[cursor + 1] if len(points) == 2 else baseline[active])
                physical[:, column] = delta / denominator / uncertainty[active]
                if not np.isfinite(physical[:, column]).all():
                    raise ValueError("Finite-difference weighted Jacobian must be finite")
                cursor += len(points)
            tasks.clear()

        for column, index in enumerate(physical_indices):
            requested = steps[index]
            left, right = parameters[index] - lo[index], hi[index] - parameters[index]
            plus, minus = parameters.copy(), parameters.copy()
            if left >= requested and right >= requested:
                plus[index] += requested
                minus[index] -= requested
                if plus[index] == parameters[index] or minus[index] == parameters[index]:
                    raise ValueError("difference_step is not representable at the parameter value")
                denominator = plus[index] - minus[index]
                points = [plus, minus]
            else:
                direction = 1 if right >= left else -1
                available = right if direction == 1 else left
                plus[index] = np.clip(parameters[index] + direction * min(requested, available),
                                     lo[index], hi[index])
                denominator = plus[index] - parameters[index]
                if denominator == 0 or not np.isfinite(denominator):
                    raise ValueError("difference_step is not representable within the parameter bounds")
                points = [plus]
            tasks.append((column, points, denominator))
            if len(tasks) >= batch_size:
                flush()
        flush()
        analytic = sparse.csr_matrix((rows, 0))
        if analytic_jacobian is not None:
            analytic_calls += 1
            raw = analytic_jacobian(parameters.copy())
            if sparse.issparse(raw):
                raw = sparse.csr_matrix(raw, dtype=float)
            else:
                raw = np.asarray(raw, dtype=float)
            if raw.shape != (data.size, len(analytic_indices)):
                raise ValueError("analytic_jacobian shape must be (all observations, analytic parameters)")
            analytic = raw[active]
            if not np.isfinite(analytic.data if sparse.issparse(analytic) else analytic).all():
                raise ValueError("Active analytic_jacobian must be finite")
            analytic = sparse.csr_matrix(analytic).multiply(1 / uncertainty[active, None]).tocsr()
            if not np.isfinite(analytic.data).all():
                raise ValueError("Weighted analytic_jacobian must be finite")
        if large:
            return _block_operator(physical, physical_indices, analytic, analytic_indices, count, owner)
        matrix = np.empty((rows, count))
        matrix[:, physical_indices] = physical
        matrix[:, analytic_indices] = analytic.toarray()
        return matrix

    def residual(parameters):
        values = weighted(evaluate(parameters, "objective"))
        if prior is not None:
            values = np.concatenate((values, (parameters - mean) / prior))
        if not np.isfinite(values @ values):
            raise ValueError("Combined data and prior chi-square must be finite")
        return values

    def jacobian(parameters):
        nonlocal cache_x, cache_prediction
        baseline = evaluate(parameters, "jacobian_base", reuse=True)
        values = differences(parameters, baseline, "jacobian")
        cache_x, cache_prediction = parameters.copy(), baseline.copy()
        if prior is None:
            return values
        if not large:
            return np.vstack((values, np.diag(1 / prior)))
        return LinearOperator((rows + count, count), dtype=float,
            matvec=lambda x: np.r_[values.matvec(np.asarray(x).ravel()), np.asarray(x).ravel() / prior],
            rmatvec=lambda y: values.rmatvec(np.asarray(y).ravel()[:rows]) + np.asarray(y).ravel()[rows:] / prior)

    def json_vector(values):
        return [float(value) if np.isfinite(value) else None for value in values]

    @contextmanager
    def preserve_state():
        try:
            yield
        except BaseException as error:
            try:
                evaluate(best_x if restore_x is None else restore_x, "restore_after_failure", notify=False)
            except BaseException as restore_error:
                error.add_note(f"Parent model restoration also failed: {type(restore_error).__name__}: {restore_error}")
            raise

    with preserve_state():
        result = least_squares(residual, x0, jac=jacobian, bounds=(lo, hi), x_scale=scale,
                               max_nfev=int(max_nfev), ftol=ftol, xtol=xtol, gtol=gtol, method="trf",
                               **({"tr_solver": "lsmr"} if large else {}))
        restore_x = result.x.copy()
        prediction = evaluate(result.x, "final_prediction")
        diagnostics = {
            "computed": False, "singular_values": [], "rank": None,
            "parameter_count": count, "rank_deficient": None,
            "condition_number": None, "rank_threshold": None, "jacobian": None,
            "jacobian_definition": "data residual derivative times parameter_scale; no priors",
            "rank_scope": "floating-point numerical rank, not measurement-noise identifiability",
        }
        if not large:
            final_jacobian = differences(result.x, prediction, "final_jacobian") * scale
            if not np.isfinite(final_jacobian).all():
                raise ValueError("Scaled data Jacobian must be finite")
            singular = np.linalg.svd(final_jacobian, compute_uv=False)
            if not np.isfinite(singular).all():
                raise ValueError("Data Jacobian singular values must be finite")
            threshold = float(np.finfo(float).eps * max(final_jacobian.shape) * singular[0])
            rank = int(np.count_nonzero(singular > threshold))
            diagnostics.update(computed=True, singular_values=singular.tolist(), rank=rank,
                               rank_deficient=rank < count, rank_threshold=threshold,
                               condition_number=float(singular[0] / singular[-1]) if rank == count else None,
                               jacobian=final_jacobian.tolist())
        else:
            diagnostics["reason"] = "Dense SVD/export exceeds dense_jacobian_max_bytes; identifiability not established"
        # Derivative callbacks may mutate the parent lattice. Always reapply the
        # exact returned parameters, even when all probes ran in private workers.
        restored = evaluate(result.x, "restore_fitted_state")
        if not np.allclose(restored[active], prediction[active], rtol=1e-10, atol=1e-12):
            raise ValueError("Restored fitted state does not reproduce the final prediction")
        final_weighted = weighted(prediction)
        prior_residual = np.array([]) if prior is None else (result.x - mean) / prior
        if not np.isfinite([result.cost, result.optimality, prior_residual @ prior_residual]).all():
            raise ValueError("Final solver and prior diagnostics must be finite")
        physical = np.full(data.size, np.nan)
        normalized = physical.copy()
        physical[active] = prediction[active] - data[active]
        normalized[active] = final_weighted
        return {
            "parameter_names": names, "parameter_values": result.x.tolist(),
            "parameter_scale": scale.tolist(), "difference_step": steps.tolist(),
            "parameters": dict(zip(names, result.x.tolist())), "prediction": json_vector(prediction),
            "physical_residuals": json_vector(physical), "weighted_residuals": json_vector(normalized),
            "active_mask": active.tolist(), "chi_square": float(final_weighted @ final_weighted),
            "prior_chi_square": float(prior_residual @ prior_residual),
            "solver": {"status": int(result.status), "message": str(result.message),
                       "success": bool(result.success), "nfev": int(result.nfev),
                       "njev": int(result.njev), "optimality": float(result.optimality),
                       "cost": float(result.cost), "success_scope": "numerical termination only",
                       "linear_solver": "lsmr" if large else "exact"},
            "callback_evaluations": evaluations, "history": list(history), "history_limit": 128,
            "analytic_jacobian_calls": analytic_calls, "batch_calls": batch_calls,
            "analytic_parameters": analytic_names, "parent_state_restored": True,
            "checkpoint": {"path": str(checkpoint_path) if checkpoint_path is not None else None,
                           "resumed": resume, "resume_semantics": "warm restart; fresh solver budget",
                           "previous_evaluations": resumed["metadata"]["callback_evaluations"] if resumed else 0},
            "identifiability": diagnostics,
            "active_bounds": {"mask": result.active_mask.tolist(),
                              "lower": [name for name, flag in zip(names, result.active_mask) if flag == -1],
                              "upper": [name for name, flag in zip(names, result.active_mask) if flag == 1]},
        }
