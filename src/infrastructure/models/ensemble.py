"""Stacked generalisation over the base learners.

sklearn's ``StackingClassifier`` is not usable here. It builds its meta-features
with ``cross_val_predict``, which requires the folds to form a *partition* —
every row in exactly one validation fold. The time-aware splitters deliberately
violate that: purging drops rows near a fold boundary and the walk-forward
strategies never validate the earliest block at all. Passing them in raises
``cross_val_predict only works for partitions``.

So the stack is assembled directly. The mechanics are standard — out-of-fold
base predictions become the meta-learner's features — with two adjustments the
panel setting requires:

* Rows covered by no validation fold are **excluded** from meta-training rather
  than filled in. An imputed meta-feature is an invented model opinion.
* Rows covered by several folds are **averaged**, so no row gets extra weight
  purely because of where the fold boundaries fell.

Base models are then refitted on the full training split for inference, which is
the usual stacking convention: the out-of-fold predictions exist to fit the
meta-learner honestly, not to be served.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from domain.exceptions import ModelTrainingError
from infrastructure.logging import get_logger
from infrastructure.models.base import BaseModelAdapter, ensure_frame

__all__ = ["StackedEnsemble", "build_meta_learner"]

_log = get_logger(__name__)


def build_meta_learner(kind: str, *, random_state: int = 42) -> Any:
    """Construct the meta-learner.

    Logistic regression is the default and the right one for most credit stacks:
    with four correlated base probabilities as inputs there is very little signal
    left to extract, and a linear blend keeps the stack's own contribution
    inspectable — the coefficients are readable as model weights.

    Args:
        kind: ``logistic_regression`` or ``lightgbm``.
        random_state: Seed for reproducibility.

    Returns:
        The unfitted meta-learner.

    Raises:
        ModelTrainingError: If ``kind`` names no known meta-learner.
    """
    key = kind.strip().lower()
    if key in ("logistic_regression", "logistic", "lr"):
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    if key in ("lightgbm", "lgbm"):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            num_leaves=7,
            learning_rate=0.05,
            min_child_samples=20,
            verbosity=-1,
            random_state=random_state,
        )
    raise ModelTrainingError(
        "Unknown meta-learner", kind=kind, known=["logistic_regression", "lightgbm"]
    )


class StackedEnsemble(ClassifierMixin, BaseEstimator):
    """Stacking over base learners using leakage-aware out-of-fold predictions.

    Note:
        ``ClassifierMixin`` must precede ``BaseEstimator`` so ``is_classifier()``
        holds; see :class:`~infrastructure.models.base.BaseModelAdapter`.

    Attributes:
        base_models: ``(name, adapter)`` pairs for the base learners.
        meta_learner: The unfitted meta-learner.
        passthrough: Append the original features to the meta-features.
        n_folds_required: Minimum folds a row must appear in to be used.
    """

    def __init__(
        self,
        base_models: list[tuple[str, BaseModelAdapter]],
        meta_learner: Any | None = None,
        *,
        passthrough: bool = False,
        n_folds_required: int = 1,
    ) -> None:
        """Initialise the ensemble.

        Args:
            base_models: ``(name, adapter)`` pairs for the base learners.
            meta_learner: Unfitted meta-learner; logistic regression when omitted.
            passthrough: Append the original features to the meta-features.
            n_folds_required: Minimum folds a row must appear in to be used.
        """
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.passthrough = passthrough
        self.n_folds_required = n_folds_required

    @property
    def name(self) -> str:
        """Model name used in logs and the registry.

        Returns:
            ``"ensemble"``.
        """
        return "ensemble"

    @property
    def supports_tree_shap(self) -> bool:
        """Whether TreeSHAP applies.

        Returns:
            ``False``; a stack is not a single tree model, so explanations must
            use the model-agnostic KernelSHAP path.
        """
        return False

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        *,
        cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> Self:
        """Fit the stack.

        Args:
            x: Feature matrix.
            y: Binary labels.
            cv_splits: Time-aware ``(train_idx, val_idx)`` pairs used to build
                the out-of-fold meta-features. A single 70/30 chronological
                split is used when omitted.

        Returns:
            This ensemble, fitted.

        Raises:
            ModelTrainingError: If there are no base models, or too few rows end
                up covered by a validation fold to train the meta-learner.
        """
        if not self.base_models:
            raise ModelTrainingError("Stacked ensemble requires at least one base model")

        frame = ensure_frame(x)
        labels = np.asarray(y).ravel().astype(int)
        n_rows, n_models = len(frame), len(self.base_models)

        splits = cv_splits or self._default_split(n_rows)
        oof = np.zeros((n_rows, n_models), dtype=float)
        coverage = np.zeros(n_rows, dtype=int)

        for fold, (train_idx, val_idx) in enumerate(splits):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            fold_labels = labels[train_idx]
            if np.unique(fold_labels).size < 2:
                _log.warning("ensemble.fold_single_class", fold=fold, action="skipped")
                continue
            for j, (model_name, model) in enumerate(self.base_models):
                fitted = clone(model).fit(frame.iloc[train_idx], fold_labels)
                oof[val_idx, j] += fitted.predict_proba(frame.iloc[val_idx])[:, 1]
                _log.debug("ensemble.oof_fold", fold=fold, model=model_name)
            coverage[val_idx] += 1

        covered = coverage >= max(self.n_folds_required, 1)
        n_covered = int(covered.sum())
        if n_covered < 10 or np.unique(labels[covered]).size < 2:
            raise ModelTrainingError(
                "Too few out-of-fold rows to train the meta-learner",
                n_covered=n_covered,
                n_rows=n_rows,
                n_classes=int(np.unique(labels[covered]).size) if n_covered else 0,
                hint="Reduce n_splits or widen the training window",
            )

        # Average across folds so multiply-validated rows are not over-weighted.
        oof[covered] /= coverage[covered][:, None]

        meta_features = self._assemble_meta_features(oof[covered], frame.iloc[covered])
        self.meta_learner_ = clone(self.meta_learner or build_meta_learner("logistic_regression"))
        self.meta_learner_.fit(meta_features, labels[covered])

        # Refit each base model on everything for inference-time scoring.
        self.fitted_base_: list[tuple[str, Any]] = [
            (model_name, clone(model).fit(frame, labels)) for model_name, model in self.base_models
        ]

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(list(frame.columns), dtype=object)
        self._expected_feature_names = list(frame.columns)
        self.oof_coverage_ = float(n_covered / n_rows)

        _log.info(
            "ensemble.fitted",
            n_base_models=n_models,
            base_models=[n for n, _ in self.base_models],
            n_folds=len(splits),
            oof_coverage=round(self.oof_coverage_, 4),
            n_meta_rows=n_covered,
            passthrough=self.passthrough,
            meta_learner=type(self.meta_learner_).__name__,
        )
        return self

    @staticmethod
    def _default_split(n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Build a single chronological split as a fallback.

        Args:
            n_rows: Number of rows.

        Returns:
            One ``(train_idx, val_idx)`` pair splitting 70/30 in row order.
        """
        cut = max(int(n_rows * 0.7), 1)
        return [(np.arange(cut), np.arange(cut, n_rows))]

    def _assemble_meta_features(self, oof: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        """Combine base predictions with optional passthrough features.

        Args:
            oof: Base-model probabilities.
            frame: The original feature matrix for those rows.

        Returns:
            The meta-learner's input matrix.
        """
        if not self.passthrough:
            return oof
        return np.hstack([oof, frame.to_numpy(dtype=float)])

    def _check_fitted(self) -> None:
        """Guard prediction methods.

        Raises:
            ModelTrainingError: If the ensemble has not been fitted.
        """
        if not hasattr(self, "fitted_base_"):
            raise ModelTrainingError("Stacked ensemble is not fitted")

    def base_predictions(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return each base model's probability for a matrix.

        Useful for diagnosing disagreement between the base learners, which is
        where a stack's value comes from in the first place.

        Args:
            x: Feature matrix.

        Returns:
            One column per base model.
        """
        self._check_fitted()
        frame = ensure_frame(x, getattr(self, "_expected_feature_names", None))
        return pd.DataFrame(
            {name: model.predict_proba(frame)[:, 1] for name, model in self.fitted_base_},
            index=frame.index,
        )

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities from the stack.

        Args:
            x: Feature matrix.

        Returns:
            An ``(n_samples, 2)`` array of probabilities.
        """
        self._check_fitted()
        frame = ensure_frame(x, getattr(self, "_expected_feature_names", None))
        base = np.column_stack([model.predict_proba(frame)[:, 1] for _, model in self.fitted_base_])
        meta_features = self._assemble_meta_features(base, frame)
        return np.asarray(self.meta_learner_.predict_proba(meta_features))

    def predict(self, x: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict hard labels.

        Args:
            x: Feature matrix.
            threshold: Decision cut-off.

        Returns:
            Predicted labels as integers.
        """
        return (self.predict_proba(x)[:, 1] >= threshold).astype(int)

    @property
    def meta_weights(self) -> dict[str, float]:
        """Meta-learner coefficient per base model.

        Reads as the weight the stack assigned to each learner, which is the
        one genuinely interpretable output of a stacked model.

        Returns:
            Base model name mapped to its coefficient; empty for a non-linear
            meta-learner.
        """
        self._check_fitted()
        coefficients = getattr(self.meta_learner_, "coef_", None)
        if coefficients is None:
            return {}
        flat = np.asarray(coefficients).ravel()
        return {
            name: float(flat[i]) for i, (name, _) in enumerate(self.fitted_base_) if i < flat.size
        }

    def measure_latency_ms(self, x: pd.DataFrame, *, n_repeats: int = 20) -> float:
        """Measure median single-row scoring latency for the whole stack.

        Args:
            x: Feature matrix to sample a row from.
            n_repeats: Number of timed calls.

        Returns:
            Median latency in milliseconds.
        """
        import time

        self._check_fitted()
        if len(x) == 0 or n_repeats < 1:
            return 0.0

        row = ensure_frame(x, getattr(self, "_expected_feature_names", None)).iloc[[0]]
        self.predict_proba(row)

        timings: list[float] = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            self.predict_proba(row)
            timings.append((time.perf_counter() - start) * 1000.0)
        return float(np.median(timings))
