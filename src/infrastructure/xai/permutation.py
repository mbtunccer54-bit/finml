"""Permutation feature importance.

Native tree importances (split counts, impurity gain) are biased towards
high-cardinality and continuous features, which in a credit dataset means they
systematically flatter the ratios over the categorical and topic features.
Permutation importance measures something a reviewer actually cares about
instead: how much does *held-out* performance degrade when this feature is
shuffled.

Scored on PR-AUC by default rather than accuracy — at an 8% base rate, accuracy
barely moves when a genuinely predictive feature is destroyed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, make_scorer

from domain.exceptions import ExplainerError
from infrastructure.logging import get_logger

__all__ = ["compare_models", "permutation_importances"]

_log = get_logger(__name__)


def _pr_auc_scorer() -> Any:
    """Build a PR-AUC scorer for permutation importance.

    Returns:
        A scorer evaluating average precision on the positive class.
    """
    return make_scorer(average_precision_score, response_method="predict_proba")


def permutation_importances(
    model: Any,
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = 1,
    scoring: Any = None,
) -> pd.DataFrame:
    """Compute permutation importance for one model.

    Args:
        model: A fitted model exposing ``predict_proba``.
        x: Held-out feature matrix. Must not be the training split: importance
            measured on data the model memorised is not informative.
        y: Held-out labels.
        n_repeats: Shuffles per feature.
        random_state: Seed for reproducibility.
        n_jobs: Parallel workers.
        scoring: Scorer to use; PR-AUC when omitted.

    Returns:
        A frame with ``feature``, ``importance_mean``, ``importance_std`` and
        ``rank``, most important first.

    Raises:
        ExplainerError: If the computation fails.
    """
    labels = np.asarray(y).ravel().astype(int)
    try:
        result = permutation_importance(
            model,
            x,
            labels,
            scoring=scoring or _pr_auc_scorer(),
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    except Exception as exc:
        raise ExplainerError(
            "Permutation importance failed",
            model=type(model).__name__,
            n_rows=len(x),
            reason=str(exc)[:300],
        ) from exc

    frame = (
        pd.DataFrame(
            {
                "feature": list(x.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    frame["rank"] = np.arange(1, len(frame) + 1)

    _log.info(
        "xai.permutation_importance",
        model=type(model).__name__,
        n_features=len(frame),
        n_repeats=n_repeats,
        top_features=frame.head(5)["feature"].tolist(),
    )
    return frame


def compare_models(
    models: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    top_k: int = 15,
) -> pd.DataFrame:
    """Compare permutation importance across several models.

    Agreement between models on the top drivers is evidence the signal is in the
    data rather than in one model's inductive bias — which is precisely the
    question a model validation function asks.

    Args:
        models: Model name mapped to a fitted model.
        x: Held-out feature matrix.
        y: Held-out labels.
        n_repeats: Shuffles per feature.
        random_state: Seed for reproducibility.
        top_k: Features retained in the comparison.

    Returns:
        A frame indexed by feature with one importance column per model, plus
        ``mean_rank`` and ``rank_spread``. Empty if every model failed.
    """
    ranks: dict[str, pd.Series] = {}
    importances: dict[str, pd.Series] = {}

    for name, model in models.items():
        try:
            frame = permutation_importances(
                model, x, y, n_repeats=n_repeats, random_state=random_state
            )
        except ExplainerError as exc:
            # One model failing should not lose the comparison for the rest.
            _log.warning("xai.permutation_model_failed", model=name, reason=exc.message)
            continue
        indexed = frame.set_index("feature")
        importances[name] = indexed["importance_mean"]
        ranks[name] = indexed["rank"]

    if not importances:
        _log.warning("xai.permutation_comparison_empty", n_models=len(models))
        return pd.DataFrame()

    combined = pd.DataFrame(importances)
    rank_frame = pd.DataFrame(ranks)
    combined["mean_rank"] = rank_frame.mean(axis=1)
    # A wide spread means the models disagree about why they agree.
    combined["rank_spread"] = rank_frame.max(axis=1) - rank_frame.min(axis=1)

    return combined.sort_values("mean_rank").head(top_k)
