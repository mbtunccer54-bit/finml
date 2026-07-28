"""Model adapter foundation.

Every learner in the platform is wrapped in a :class:`BaseModelAdapter` so that
the rest of the system depends on one shape rather than on four vendor APIs.
The adapters are sklearn-compatible estimators, which is what lets them drop
into ``StackingClassifier``, ``CalibratedClassifierCV`` and
``permutation_importance`` unmodified.

The wrapper is not ceremony. It absorbs three real incompatibilities:

* **Probability output shape.** Under a custom objective LightGBM's
  ``predict_proba`` returns 1-D *raw logits* while XGBoost returns proper
  ``(n, 2)`` probabilities. Unnormalised, that difference silently turns a
  log-odds score into a "probability" of 1.58.
* **Imbalance handling.** Each library spells class weighting differently
  (``scale_pos_weight``, ``class_weight``, per-row weights).
* **Latency measurement.** Multi-objective tuning needs a comparable
  single-row latency number for every model.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from domain.exceptions import ModelTrainingError
from domain.value_objects import CostMatrix
from infrastructure.logging import get_logger

__all__ = [
    "BaseModelAdapter",
    "ModelBundle",
    "SupportsPredictProba",
    "ensure_frame",
    "make_focal_objective",
    "sigmoid",
]

_log = get_logger(__name__)

#: Clamp applied to logits before the logistic transform, avoiding overflow.
_LOGIT_CLAMP = 50.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform.

    Args:
        z: Raw scores.

    Returns:
        Probabilities in ``(0, 1)``.
    """
    clipped = np.clip(np.asarray(z, dtype=float), -_LOGIT_CLAMP, _LOGIT_CLAMP)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=float)


def ensure_frame(x: Any, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Coerce model input to a named dataframe.

    sklearn utilities routinely strip a dataframe down to a bare array before
    handing it to an inner estimator. The adapters need column names — for
    LightGBM's categorical handling and for every explainer downstream — so the
    names are restored rather than the loss being tolerated.

    Args:
        x: A dataframe, array or anything array-like.
        feature_names: Names to apply when ``x`` carries none.

    Returns:
        A dataframe with named columns.

    Raises:
        ModelTrainingError: If the supplied names do not match the width.
    """
    if isinstance(x, pd.DataFrame):
        return x
    array = np.asarray(x)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if feature_names is None:
        return pd.DataFrame(array, columns=[f"f{i}" for i in range(array.shape[1])])
    if len(feature_names) != array.shape[1]:
        raise ModelTrainingError(
            "Feature name count does not match the matrix width",
            n_names=len(feature_names),
            n_columns=array.shape[1],
        )
    return pd.DataFrame(array, columns=feature_names)


@runtime_checkable
class SupportsPredictProba(Protocol):
    """The contract every model in the platform satisfies."""

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Any:
        """Fit the model.

        Args:
            x: Feature matrix.
            y: Binary labels.

        Returns:
            The fitted estimator.
        """
        ...

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict hard labels.

        Args:
            x: Feature matrix.

        Returns:
            Predicted labels.
        """
        ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            x: Feature matrix.

        Returns:
            An ``(n_samples, 2)`` array of probabilities.
        """
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the estimator's hyperparameters.

        Args:
            deep: Include nested estimator parameters.

        Returns:
            The parameter mapping.
        """
        ...


def make_focal_objective(gamma: float = 2.0, alpha: float = 0.25) -> Any:
    """Build a binary focal-loss objective for gradient boosting.

    Focal loss down-weights easy examples so training capacity is spent on the
    hard minority cases. For ``L = -alpha_t (1 - p_t)^gamma log(p_t)`` the exact
    first and second derivatives with respect to the raw score are returned;
    they are verified against finite differences in the test suite, because a
    silently wrong Hessian degrades a booster rather than failing it.

    Args:
        gamma: Focusing parameter; ``0`` recovers weighted cross-entropy.
        alpha: Weight on the positive class.

    Returns:
        A callable ``(y_true, y_pred) -> (grad, hess)`` for LightGBM/XGBoost.

    Raises:
        ModelTrainingError: If ``gamma`` is negative or ``alpha`` is outside ``(0, 1)``.
    """
    if gamma < 0.0:
        raise ModelTrainingError("focal_gamma must not be negative", gamma=gamma)
    if not 0.0 < alpha < 1.0:
        raise ModelTrainingError("focal_alpha must lie in (0, 1)", alpha=alpha)

    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the focal-loss gradient and Hessian.

        Args:
            y_true: Binary labels.
            y_pred: Raw scores.

        Returns:
            The gradient and Hessian with respect to the raw scores.
        """
        labels = np.asarray(y_true, dtype=float).ravel()
        p = sigmoid(np.asarray(y_pred, dtype=float).ravel())

        is_positive = labels == 1.0
        # p_t is the probability assigned to the *true* class.
        t = np.clip(np.where(is_positive, p, 1.0 - p), 1e-9, 1.0 - 1e-9)
        u = 1.0 - t
        s = np.where(is_positive, 1.0, -1.0)
        a_t = np.where(is_positive, alpha, 1.0 - alpha)
        log_t = np.log(t)

        # dL/dt and d2L/dt2, then chained through dt/dz = s * t * u.
        d1 = a_t * gamma * u ** (gamma - 1.0) * log_t - a_t * u**gamma / t
        d2 = (
            -a_t * gamma * (gamma - 1.0) * u ** (gamma - 2.0) * log_t
            + 2.0 * a_t * gamma * u ** (gamma - 1.0) / t
            + a_t * u**gamma / t**2
        )
        grad = d1 * s * t * u
        hess = d2 * (t * u) ** 2 + d1 * t * u * (1.0 - 2.0 * t)
        # Boosters require a positive Hessian to form a valid leaf value.
        return grad, np.maximum(hess, 1e-6)

    return objective


@dataclass(slots=True)
class ModelBundle:
    """A trained model together with everything needed to reproduce its scores.

    Attributes:
        model: The fitted estimator.
        model_id: Identifier used in logs, the registry and API responses.
        feature_names: Matrix columns in the order the model expects.
        trained_at: Training timestamp.
        params: Hyperparameters the model was fitted with.
        metrics: Evaluation metrics keyed by ``"{split}.{metric}"``.
        is_calibrated: Whether probabilities passed through a calibrator.
        calibration_method: Calibration method used, if any.
        dataset_version: Dataset lineage tag.
        extra: Free-form additional metadata.
    """

    model: Any
    model_id: str
    feature_names: list[str] = field(default_factory=list)
    trained_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    is_calibrated: bool = False
    calibration_method: str = ""
    dataset_version: str = "v1"
    extra: dict[str, Any] = field(default_factory=dict)

    def predict_pd(self, x: pd.DataFrame) -> np.ndarray:
        """Score a feature matrix, returning the positive-class probability.

        Args:
            x: Feature matrix.

        Returns:
            A one-dimensional array of default probabilities.

        Raises:
            ModelTrainingError: If the matrix columns do not match the model.
        """
        aligned = self.align(x)
        proba = self.model.predict_proba(aligned)
        return np.asarray(proba)[:, 1]

    def align(self, x: pd.DataFrame) -> pd.DataFrame:
        """Reorder and check a matrix against the model's expected columns.

        Args:
            x: Feature matrix.

        Returns:
            The matrix with columns in training order.

        Raises:
            ModelTrainingError: If an expected column is absent.
        """
        if not self.feature_names:
            return x
        missing = [c for c in self.feature_names if c not in x.columns]
        if missing:
            raise ModelTrainingError(
                "Feature matrix is missing columns the model was trained on",
                model_id=self.model_id,
                missing=missing[:20],
                n_missing=len(missing),
            )
        return x[self.feature_names]


class BaseModelAdapter(ClassifierMixin, BaseEstimator, ABC):
    """Common behaviour for the four base learners.

    Note:
        ``ClassifierMixin`` must precede ``BaseEstimator``. With the reverse
        order sklearn's ``__sklearn_tags__`` resolves to the base implementation,
        ``is_classifier()`` returns ``False``, and ``CalibratedClassifierCV``
        then reads the *negative* class column — silently inverting every
        calibrated probability instead of raising.

    Attributes:
        params: Vendor-specific hyperparameters.
        random_state: Seed for reproducibility.
        n_jobs: Worker count where the library supports it.
        imbalance_strategy: ``class_weight``, ``focal``, ``smote`` or ``none``.
        cost_matrix: Business costs; supplies the positive-class weight.
        focal_gamma: Focusing parameter for the focal objective.
        focal_alpha: Positive-class weight for the focal objective.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        random_state: int = 42,
        n_jobs: int = -1,
        imbalance_strategy: str = "class_weight",
        cost_matrix: CostMatrix | None = None,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        """Initialise the adapter.

        Args:
            params: Vendor-specific hyperparameters.
            random_state: Seed for reproducibility.
            n_jobs: Worker count where supported.
            imbalance_strategy: Imbalance handling mode.
            cost_matrix: Business costs supplying the positive-class weight.
            focal_gamma: Focusing parameter for the focal objective.
            focal_alpha: Positive-class weight for the focal objective.
        """
        # Stored verbatim: sklearn's get_params/clone contract requires that
        # every constructor argument survives unmodified as an attribute.
        self.params = params
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.imbalance_strategy = imbalance_strategy
        self.cost_matrix = cost_matrix
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    # -- Subclass contract ---------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Short model name used in logs, metrics and the registry.

        Returns:
            The model name.
        """

    @property
    def supports_tree_shap(self) -> bool:
        """Whether TreeSHAP applies to this model.

        Returns:
            ``True`` for tree ensembles.
        """
        return True

    @abstractmethod
    def _build_estimator(self, positive_weight: float) -> Any:
        """Construct the underlying estimator.

        Args:
            positive_weight: Weight to apply to the positive class.

        Returns:
            The unfitted vendor estimator.
        """

    # -- Fitting -------------------------------------------------------------
    @property
    def _effective_params(self) -> dict[str, Any]:
        """Hyperparameters with ``None`` normalised to an empty mapping.

        Returns:
            The effective parameter mapping.
        """
        return dict(self.params or {})

    @property
    def _effective_cost_matrix(self) -> CostMatrix:
        """Cost matrix, defaulting to the platform standard.

        Returns:
            The cost matrix in force.
        """
        return self.cost_matrix or CostMatrix()

    def _positive_weight(self, y: np.ndarray) -> float:
        """Compute the positive-class weight.

        Two forces are combined: the empirical class imbalance and the business
        cost asymmetry. Using prevalence alone would ignore that a missed
        default costs ten times a false alarm.

        Args:
            y: Binary labels.

        Returns:
            The weight, or ``1.0`` when class weighting is not in use.
        """
        if self.imbalance_strategy != "class_weight":
            return 1.0
        n_positive = float(np.sum(y == 1))
        n_negative = float(np.sum(y == 0))
        if n_positive <= 0.0 or n_negative <= 0.0:
            return 1.0
        prevalence_weight = n_negative / n_positive
        return float(np.sqrt(prevalence_weight * self._effective_cost_matrix.imbalance_ratio))

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        **fit_kwargs: Any,
    ) -> Self:
        """Fit the model.

        Args:
            x: Feature matrix.
            y: Binary labels.
            sample_weight: Optional per-row weights.
            **fit_kwargs: Extra keyword arguments forwarded to the estimator.

        Returns:
            This adapter, fitted.

        Raises:
            ModelTrainingError: If the input is degenerate or the library raises.
        """
        x = ensure_frame(x, getattr(self, "_expected_feature_names", None))
        labels = np.asarray(y).ravel()
        if len(x) != len(labels):
            raise ModelTrainingError(
                "Feature and label counts differ",
                model=self.name,
                n_x=len(x),
                n_y=len(labels),
            )
        unique = np.unique(labels)
        if unique.size < 2:
            raise ModelTrainingError(
                "Training split contains a single class",
                model=self.name,
                classes=unique.tolist(),
                hint="Widen the fold or reduce n_splits",
            )

        positive_weight = self._positive_weight(labels)
        estimator = self._build_estimator(positive_weight)

        try:
            if sample_weight is not None:
                estimator.fit(x, labels, sample_weight=sample_weight, **fit_kwargs)
            else:
                estimator.fit(x, labels, **fit_kwargs)
        except Exception as exc:
            raise ModelTrainingError(
                "Underlying estimator failed to fit",
                model=self.name,
                error_type=type(exc).__name__,
                reason=str(exc)[:400],
            ) from exc

        self.estimator_ = estimator
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = x.shape[1]
        self.feature_names_in_ = np.asarray(list(x.columns), dtype=object)
        self._expected_feature_names = list(x.columns)
        self._positive_weight_used = positive_weight

        _log.debug(
            "model.fitted",
            model=self.name,
            n_rows=len(x),
            n_features=x.shape[1],
            positive_rate=round(float(labels.mean()), 4),
            positive_weight=round(positive_weight, 3),
            imbalance_strategy=self.imbalance_strategy,
        )
        return self

    # -- Prediction ----------------------------------------------------------
    def _check_fitted(self) -> None:
        """Guard prediction methods.

        Raises:
            ModelTrainingError: If the adapter has not been fitted.
        """
        if not hasattr(self, "estimator_"):
            raise ModelTrainingError("Model is not fitted", model=self.name)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            x: Feature matrix.

        Returns:
            An ``(n_samples, 2)`` array whose second column is the default
            probability.

        Raises:
            ModelTrainingError: If the model is not fitted or the backend
                returns an unusable shape.
        """
        self._check_fitted()
        frame = ensure_frame(x, getattr(self, "_expected_feature_names", None))
        raw = np.asarray(self.estimator_.predict_proba(frame))
        return self._normalise_proba(raw, n_rows=len(frame))

    def _normalise_proba(self, raw: np.ndarray, *, n_rows: int) -> np.ndarray:
        """Coerce a backend's output into an ``(n, 2)`` probability array.

        Under a custom objective LightGBM returns 1-D raw logits here rather
        than probabilities. Passing that through untouched would produce
        "probabilities" outside ``[0, 1]`` that then flow into PD reporting, so
        the shape and range are checked rather than assumed.

        Args:
            raw: Whatever the backend returned.
            n_rows: Expected row count.

        Returns:
            An ``(n_rows, 2)`` probability array.

        Raises:
            ModelTrainingError: If the output cannot be interpreted.
        """
        if raw.ndim == 2 and raw.shape == (n_rows, 2):
            return raw
        if raw.ndim == 1 and raw.shape[0] == n_rows:
            # 1-D: either probabilities of the positive class, or raw logits.
            positive = raw if (raw.min() >= 0.0 and raw.max() <= 1.0) else sigmoid(raw)
            return np.column_stack([1.0 - positive, positive])
        if raw.ndim == 2 and raw.shape == (n_rows, 1):
            positive = raw.ravel()
            positive = (
                positive if (positive.min() >= 0.0 and positive.max() <= 1.0) else sigmoid(positive)
            )
            return np.column_stack([1.0 - positive, positive])
        raise ModelTrainingError(
            "Backend returned an uninterpretable probability array",
            model=self.name,
            shape=tuple(raw.shape),
            expected_rows=n_rows,
        )

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
    def feature_importances_(self) -> np.ndarray:
        """Native feature importances.

        Returns:
            One importance per feature; zeros when the backend exposes none.
        """
        self._check_fitted()
        importances = getattr(self.estimator_, "feature_importances_", None)
        if importances is None:
            _log.warning("model.no_native_importances", model=self.name)
            return np.zeros(self.n_features_in_)
        return np.asarray(importances, dtype=float)

    def measure_latency_ms(self, x: pd.DataFrame, *, n_repeats: int = 20) -> float:
        """Measure median single-row scoring latency.

        Single-row rather than batch, because that is what an interactive
        credit decision actually pays, and it is the figure the multi-objective
        tuner minimises.

        Args:
            x: Feature matrix to sample a row from.
            n_repeats: Number of timed calls.

        Returns:
            Median latency in milliseconds.
        """
        self._check_fitted()
        if len(x) == 0 or n_repeats < 1:
            return 0.0

        row = x.iloc[[0]]
        self.predict_proba(row)  # warm any lazy initialisation

        timings: list[float] = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            self.predict_proba(row)
            timings.append((time.perf_counter() - start) * 1000.0)
        return float(np.median(timings))

    def __sklearn_tags__(self) -> Any:
        """Declare sklearn estimator tags.

        Returns:
            The tag object with binary-classification requirements relaxed for
            the dataframe input this platform uses throughout.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
