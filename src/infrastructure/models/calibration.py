"""Probability calibration.

A gradient-boosted classifier trained with class weights is a good *ranker* and
a poor *probability estimator* — weighting deliberately distorts the output
scale. That distinction matters here: the number this platform reports is a
probability of default that feeds pricing, provisioning and IFRS 9 staging, so
it has to mean what it says.

Two methods are fitted and compared:

* **Platt scaling** (``sigmoid``) — a one-parameter logistic fit. Robust on small
  validation sets, but it cannot fix a non-monotone distortion.
* **Isotonic regression** — non-parametric and strictly monotone. Strictly more
  flexible, and correspondingly happy to overfit a few hundred rows.

The benchmark picks between them on expected calibration error rather than
assuming either.

A compatibility note: sklearn 1.6 deprecated ``cv="prefit"`` in favour of
``FrozenEstimator`` and 1.9 removed it. :func:`calibrate_prefit` handles both so
the platform is not pinned to one sklearn minor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV

from domain.exceptions import ModelTrainingError
from infrastructure.logging import get_logger
from infrastructure.models.evaluation import ReliabilityCurve, reliability_curve

__all__ = [
    "CalibrationReport",
    "PrefitCalibratedClassifier",
    "calibrate_prefit",
    "fit_best_calibrator",
]

_log = get_logger(__name__)


def _freeze(estimator: Any) -> Any:
    """Wrap a fitted estimator so calibration will not refit it.

    Args:
        estimator: A fitted estimator.

    Returns:
        The estimator wrapped in ``FrozenEstimator`` when available, otherwise
        the estimator unchanged (older sklearn takes ``cv="prefit"`` instead).
    """
    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:  # pragma: no cover - sklearn < 1.6
        return estimator
    return FrozenEstimator(estimator)


class PrefitCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """A fitted model paired with a calibrator, kept sklearn-compatible.

    Note:
        ``ClassifierMixin`` must precede ``BaseEstimator``; see
        :class:`~infrastructure.models.base.BaseModelAdapter`.

    Exists so a calibrated model presents exactly the same surface as an
    uncalibrated one. Everything downstream — the benchmark, SHAP, the API,
    the dashboard — then treats them identically.

    Attributes:
        base_estimator: The fitted, uncalibrated model.
        calibrator: The fitted calibration wrapper.
        method: Calibration method name.
    """

    def __init__(self, base_estimator: Any, calibrator: Any, method: str) -> None:
        """Initialise the wrapper.

        Args:
            base_estimator: The fitted, uncalibrated model.
            calibrator: The fitted calibration wrapper.
            method: Calibration method name.
        """
        self.base_estimator = base_estimator
        self.calibrator = calibrator
        self.method = method
        self.classes_ = np.array([0, 1])

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> PrefitCalibratedClassifier:  # noqa: ARG002
        """Return self; both components are already fitted.

        Args:
            x: Unused; present for interface compatibility.
            y: Unused; present for interface compatibility.

        Returns:
            This wrapper, unchanged.
        """
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict calibrated class probabilities.

        Args:
            x: Feature matrix.

        Returns:
            An ``(n_samples, 2)`` array of calibrated probabilities.
        """
        return np.asarray(self.calibrator.predict_proba(x))

    def predict(self, x: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict hard labels from the calibrated probabilities.

        Args:
            x: Feature matrix.
            threshold: Decision cut-off.

        Returns:
            Predicted labels as integers.
        """
        return (self.predict_proba(x)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Importances delegated to the underlying model.

        Returns:
            The base model's importances; calibration does not alter them.
        """
        return np.asarray(getattr(self.base_estimator, "feature_importances_", []))


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Outcome of comparing calibration methods.

    Attributes:
        method: Winning method, or ``"none"`` when calibration did not help.
        model: The calibrated (or original) model.
        before: Reliability curve of the uncalibrated model.
        after: Reliability curve of the winning model.
        candidates: Expected calibration error per method tried.
    """

    method: str
    model: Any
    before: ReliabilityCurve
    after: ReliabilityCurve
    candidates: dict[str, float]

    @property
    def improvement(self) -> float:
        """Reduction in expected calibration error.

        Returns:
            ``ece_before - ece_after``; positive means calibration helped.
        """
        return self.before.expected_calibration_error - self.after.expected_calibration_error


def calibrate_prefit(
    estimator: Any,
    x_calib: pd.DataFrame,
    y_calib: np.ndarray,
    *,
    method: str = "sigmoid",
) -> PrefitCalibratedClassifier:
    """Calibrate an already-fitted model on a held-out split.

    The calibration split must be data the base model has not seen. Calibrating
    on the training split would fit the calibrator to the model's memorised
    outputs and report a calibration quality that does not exist.

    Args:
        estimator: A fitted model exposing ``predict_proba``.
        x_calib: Held-out feature matrix.
        y_calib: Held-out labels.
        method: ``sigmoid`` for Platt scaling, ``isotonic`` for isotonic regression.

    Returns:
        The calibrated model.

    Raises:
        ModelTrainingError: If the method is unknown, the split has one class,
            or calibration fails.
    """
    if method not in ("sigmoid", "isotonic"):
        raise ModelTrainingError(
            "Unknown calibration method", method=method, known=["sigmoid", "isotonic"]
        )
    labels = np.asarray(y_calib).ravel().astype(int)
    if np.unique(labels).size < 2:
        raise ModelTrainingError(
            "Calibration split contains a single class", method=method, n_rows=len(labels)
        )

    frozen = _freeze(estimator)
    try:
        if frozen is estimator:  # pragma: no cover - sklearn < 1.6
            calibrator = CalibratedClassifierCV(estimator, method=method, cv="prefit")
        else:
            calibrator = CalibratedClassifierCV(frozen, method=method)
        calibrator.fit(x_calib, labels)
    except Exception as exc:
        raise ModelTrainingError(
            "Calibration failed",
            method=method,
            error_type=type(exc).__name__,
            reason=str(exc)[:300],
        ) from exc

    return PrefitCalibratedClassifier(estimator, calibrator, method)


def fit_best_calibrator(
    estimator: Any,
    x_calib: pd.DataFrame,
    y_calib: np.ndarray,
    *,
    methods: list[str] | None = None,
    n_bins: int = 10,
) -> CalibrationReport:
    """Fit each calibration method and keep the best.

    The uncalibrated model competes as a candidate in its own right. If neither
    method improves the expected calibration error, the original is kept —
    calibration is not free, and a wrapper that makes things worse is worse.

    Args:
        estimator: A fitted model exposing ``predict_proba``.
        x_calib: Held-out feature matrix.
        y_calib: Held-out labels.
        methods: Methods to try; defaults to both.
        n_bins: Bins used for the calibration error.

    Returns:
        The comparison report, carrying the winning model.
    """
    labels = np.asarray(y_calib).ravel().astype(int)
    tried = methods or ["sigmoid", "isotonic"]

    baseline_prob = np.asarray(estimator.predict_proba(x_calib))[:, 1]
    before = reliability_curve(labels, baseline_prob, n_bins=n_bins)

    candidates: dict[str, float] = {"none": before.expected_calibration_error}
    best_method = "none"
    best_model: Any = estimator
    best_curve = before

    for method in tried:
        try:
            calibrated = calibrate_prefit(estimator, x_calib, labels, method=method)
        except ModelTrainingError as exc:
            # One method failing (isotonic on a tiny split, say) must not stop
            # the other from being tried.
            _log.warning("calibration.method_failed", method=method, reason=exc.message)
            continue

        prob = calibrated.predict_proba(x_calib)[:, 1]
        curve = reliability_curve(labels, prob, n_bins=n_bins)
        candidates[method] = curve.expected_calibration_error

        if curve.expected_calibration_error < best_curve.expected_calibration_error:
            best_method, best_model, best_curve = method, calibrated, curve

    _log.info(
        "calibration.completed",
        selected=best_method,
        ece_before=round(before.expected_calibration_error, 5),
        ece_after=round(best_curve.expected_calibration_error, 5),
        candidates={k: round(v, 5) for k, v in candidates.items()},
        n_calibration_rows=len(labels),
    )
    return CalibrationReport(
        method=best_method,
        model=best_model,
        before=before,
        after=best_curve,
        candidates=candidates,
    )


def cross_validated_calibrator(
    estimator: Any,
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    method: str = "sigmoid",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Fit a calibrated model using internal cross-validation.

    Preferred over :func:`calibrate_prefit` when no separate calibration split
    can be spared: every row is used for both fitting and calibration, at
    different times.

    Args:
        estimator: An *unfitted* estimator; it is cloned and refitted per fold.
        x: Feature matrix.
        y: Binary labels.
        method: ``sigmoid`` or ``isotonic``.
        cv: Number of internal folds.

    Returns:
        The fitted calibrated classifier.

    Raises:
        ModelTrainingError: If calibration fails.
    """
    try:
        calibrator = CalibratedClassifierCV(clone(estimator), method=method, cv=cv)
        calibrator.fit(x, np.asarray(y).ravel().astype(int))
    except Exception as exc:
        raise ModelTrainingError(
            "Cross-validated calibration failed",
            method=method,
            cv=cv,
            error_type=type(exc).__name__,
            reason=str(exc)[:300],
        ) from exc
    return calibrator
