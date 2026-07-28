"""Multi-objective hyperparameter optimisation with Optuna.

The study optimises two objectives at once: **maximise PR-AUC** and **minimise
median single-row inference latency**. That is not decoration. A random forest
with 800 deep trees will often edge out a small LightGBM on PR-AUC while costing
twenty times as much per decision, and a single-objective study will pick it
every time without ever surfacing the trade.

Optuna returns a Pareto front rather than one winner. The front is collapsed
with explicit preference weights from configuration, so the trade-off is a
recorded decision rather than an accident of the sampler:

    score = pr_auc_weight * PR-AUC - latency_weight * (latency / latency_budget)

Search spaces come from ``configs/models/*.yaml``, so tuning a new parameter
needs no code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import NSGAIISampler, TPESampler
from sklearn.metrics import average_precision_score

from domain.exceptions import ConfigurationError, ModelTrainingError
from domain.value_objects import CostMatrix
from infrastructure.config.schemas import ModelSpec, TuningConfig
from infrastructure.logging import get_logger
from infrastructure.models.trainers import build_model

__all__ = ["TuningResult", "sample_search_space", "tune_model"]

_log = get_logger(__name__)

# Optuna's own logger is noisy and does not go through structlog.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Outcome of a tuning study.

    Attributes:
        model_name: Model that was tuned.
        best_params: Winning hyperparameters.
        best_pr_auc: Cross-validated PR-AUC of the winner.
        best_latency_ms: Median single-row latency of the winner.
        n_trials: Completed trials.
        pareto_front: ``(pr_auc, latency_ms, params)`` for each front member.
        tuned: Whether tuning actually ran.
    """

    model_name: str
    best_params: dict[str, Any] = field(default_factory=dict)
    best_pr_auc: float = 0.0
    best_latency_ms: float = 0.0
    n_trials: int = 0
    pareto_front: list[tuple[float, float, dict[str, Any]]] = field(default_factory=list)
    tuned: bool = False


def sample_search_space(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Sample one point from a declarative search space.

    Args:
        trial: The Optuna trial doing the sampling.
        search_space: Parameter name mapped to a spec such as
            ``{"type": "float", "low": 0.01, "high": 0.3, "log": True}``.

    Returns:
        The sampled hyperparameters.

    Raises:
        ConfigurationError: If a spec is malformed or names an unknown type.
    """
    sampled: dict[str, Any] = {}

    for name, raw_spec in search_space.items():
        if not isinstance(raw_spec, dict):
            raise ConfigurationError(
                "Search space entry must be a mapping", parameter=name, got=type(raw_spec).__name__
            )
        spec = dict(raw_spec)
        kind = str(spec.get("type", "")).lower()

        if kind == "categorical":
            choices = spec.get("choices")
            if not choices:
                raise ConfigurationError(
                    "Categorical search space needs a non-empty 'choices'", parameter=name
                )
            sampled[name] = trial.suggest_categorical(name, list(choices))
            continue

        if kind not in ("float", "int"):
            raise ConfigurationError(
                "Unknown search space type",
                parameter=name,
                type=spec.get("type"),
                known=["float", "int", "categorical"],
            )

        if "low" not in spec or "high" not in spec:
            raise ConfigurationError(
                "Numeric search space needs 'low' and 'high'", parameter=name, spec=spec
            )
        low, high = spec["low"], spec["high"]
        log = bool(spec.get("log", False))
        step = spec.get("step")

        if kind == "float":
            # Optuna rejects step and log together.
            if step is not None and not log:
                sampled[name] = trial.suggest_float(name, float(low), float(high), step=float(step))
            else:
                sampled[name] = trial.suggest_float(name, float(low), float(high), log=log)
        elif step is not None and not log:
            sampled[name] = trial.suggest_int(name, int(low), int(high), step=int(step))
        else:
            sampled[name] = trial.suggest_int(name, int(low), int(high), log=log)

    return sampled


def _cross_validated_pr_auc(
    model: Any,
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Average PR-AUC across the supplied folds.

    Args:
        model: The unfitted adapter to evaluate.
        x: Feature matrix.
        y: Binary labels.
        splits: ``(train_idx, val_idx)`` pairs.

    Returns:
        The mean PR-AUC over usable folds, or ``0.0`` when none were usable.
    """
    from sklearn.base import clone

    scores: list[float] = []
    for train_idx, val_idx in splits:
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        y_train, y_val = y[train_idx], y[val_idx]
        if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
            continue
        fitted = clone(model).fit(x.iloc[train_idx], y_train)
        probability = fitted.predict_proba(x.iloc[val_idx])[:, 1]
        scores.append(float(average_precision_score(y_val, probability)))
    return float(np.mean(scores)) if scores else 0.0


def tune_model(
    model_name: str,
    spec: ModelSpec,
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: TuningConfig,
    *,
    imbalance_strategy: str = "class_weight",
    cost_matrix: CostMatrix | None = None,
    n_jobs: int = -1,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> TuningResult:
    """Run a multi-objective study for one model.

    Args:
        model_name: Model to tune.
        spec: Configuration supplying static params and the search space.
        x: Feature matrix.
        y: Binary labels.
        splits: Time-aware ``(train_idx, val_idx)`` pairs.
        config: Tuning configuration.
        imbalance_strategy: Imbalance handling mode.
        cost_matrix: Business costs.
        n_jobs: Worker count for the estimator.
        focal_gamma: Focusing parameter for the focal objective.
        focal_alpha: Positive-class weight for the focal objective.

    Returns:
        The tuning result. When tuning is disabled or no search space is
        declared, a result with ``tuned=False`` and empty params is returned so
        the caller falls back to the static configuration.

    Raises:
        ModelTrainingError: If every trial fails.
    """
    if not config.enabled or not spec.search_space:
        _log.info(
            "tuning.skipped",
            model=model_name,
            reason="disabled" if not config.enabled else "no search space declared",
        )
        return TuningResult(model_name=model_name, tuned=False)

    labels = np.asarray(y).ravel().astype(int)
    failures: list[str] = []

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        """Evaluate one hyperparameter configuration.

        Args:
            trial: The Optuna trial.

        Returns:
            The cross-validated PR-AUC and the median single-row latency in ms.
        """
        params = sample_search_space(trial, spec.search_space)
        model = build_model(
            model_name,
            spec,
            random_state=config.seed,
            n_jobs=n_jobs,
            imbalance_strategy=imbalance_strategy,
            cost_matrix=cost_matrix,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            param_overrides=params,
        )
        try:
            pr_auc = _cross_validated_pr_auc(model, x, labels, splits)
            fitted = model.fit(x, labels)
            latency = fitted.measure_latency_ms(x, n_repeats=10)
        except Exception as exc:
            # One bad corner of the space must not abort the study; record it
            # and let Optuna steer away via the poor objective value.
            failures.append(f"{type(exc).__name__}: {exc}"[:200])
            _log.warning(
                "tuning.trial_failed",
                model=model_name,
                trial=trial.number,
                error_type=type(exc).__name__,
                reason=str(exc)[:200],
            )
            return 0.0, float(config.latency_budget_ms * 100)
        return pr_auc, latency

    # NSGA-II is designed for multi-objective search; TPE degrades to scalarised
    # behaviour with several objectives.
    sampler = (
        NSGAIISampler(seed=config.seed)
        if len(config.directions) > 1
        else TPESampler(seed=config.seed, n_startup_trials=config.n_startup_trials)
    )
    study = optuna.create_study(directions=list(config.directions), sampler=sampler)

    if config.pruning_enabled and len(config.directions) > 1:
        _log.info(
            "tuning.pruning_unavailable",
            model=model_name,
            detail="Optuna does not support pruners for multi-objective studies",
        )

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout_s or None,
        n_jobs=config.n_jobs,
        show_progress_bar=False,
        catch=(),
    )

    front = [
        (float(t.values[0]), float(t.values[1]), dict(t.params))
        for t in study.best_trials
        if t.values is not None
    ]
    if not front:
        raise ModelTrainingError(
            "Every tuning trial failed",
            model=model_name,
            n_trials=config.n_trials,
            sample_failures=failures[:3],
        )

    budget = max(config.latency_budget_ms, 1e-6)
    best_pr_auc, best_latency, best_params = max(
        front,
        key=lambda item: (
            config.pr_auc_weight * item[0] - config.latency_weight * (item[1] / budget)
        ),
    )

    _log.info(
        "tuning.completed",
        model=model_name,
        n_trials=len(study.trials),
        n_failed=len(failures),
        pareto_size=len(front),
        best_pr_auc=round(best_pr_auc, 4),
        best_latency_ms=round(best_latency, 3),
        best_params=best_params,
        pr_auc_weight=config.pr_auc_weight,
        latency_weight=config.latency_weight,
    )
    return TuningResult(
        model_name=model_name,
        best_params=best_params,
        best_pr_auc=best_pr_auc,
        best_latency_ms=best_latency,
        n_trials=len(study.trials),
        pareto_front=front,
        tuned=True,
    )
