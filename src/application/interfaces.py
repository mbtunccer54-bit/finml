"""Ports for the application layer.

:mod:`domain.repositories` states what the *business* needs. This module states
what the *use cases* need — and unlike the domain ports, these signatures may
speak ``pandas``, because the application layer is exactly the seam where
tabular data meets domain types.

Structural protocols again, so no adapter has to import from here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from domain.entities import DriftReport, ModelPerformance, ScoreExplanation
from domain.value_objects import SentimentScore

__all__ = [
    "DriftMonitor",
    "ExplainerPort",
    "FeatureBuilder",
    "ModelRegistryPort",
    "PanelSource",
    "SentimentPort",
    "SplitStrategy",
    "TrainableModel",
]


@runtime_checkable
class PanelSource(Protocol):
    """Supplies the modelling dataset."""

    def fetch(self) -> Any:
        """Retrieve the dataset.

        Returns:
            A :class:`~infrastructure.data.fetchers.PanelDataset`.
        """
        ...


@runtime_checkable
class FeatureBuilder(Protocol):
    """Turns a contract-shaped frame into a numeric model matrix."""

    def fit(self, df: pd.DataFrame) -> Any:
        """Learn preprocessing statistics from the training split.

        Args:
            df: Training frame.

        Returns:
            The fitted builder.
        """
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project a frame onto the learned feature space.

        Args:
            df: Frame to transform.

        Returns:
            The numeric matrix.
        """
        ...

    @property
    def feature_names(self) -> list[str]:
        """Output column names in matrix order.

        Returns:
            The feature names.
        """
        ...


@runtime_checkable
class SplitStrategy(Protocol):
    """Produces leakage-aware cross-validation folds."""

    def split(self, x: Any, y: Any = None, groups: Any = None) -> Any:
        """Yield train/validation index pairs.

        Args:
            x: Feature matrix.
            y: Labels.
            groups: Per-row timestamps.

        Returns:
            An iterator of index pairs.
        """
        ...

    def get_n_splits(self, x: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the fold count.

        Args:
            x: Feature matrix.
            y: Labels.
            groups: Per-row timestamps.

        Returns:
            The number of folds.
        """
        ...


@runtime_checkable
class TrainableModel(Protocol):
    """A model the training pipeline can fit and score with."""

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Any:
        """Fit the model.

        Args:
            x: Feature matrix.
            y: Binary labels.

        Returns:
            The fitted model.
        """
        ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            x: Feature matrix.

        Returns:
            An ``(n_samples, 2)`` probability array.
        """
        ...


@runtime_checkable
class SentimentPort(Protocol):
    """Produces sentiment scores from text."""

    def aggregate(self, texts: Sequence[str]) -> SentimentScore:
        """Score a document set and average into one score.

        Args:
            texts: Document texts.

        Returns:
            The aggregated sentiment.
        """
        ...

    @property
    def is_fallback(self) -> bool:
        """Whether a degraded backend produced the scores.

        Returns:
            ``True`` when the fallback tier was used.
        """
        ...


@runtime_checkable
class DriftMonitor(Protocol):
    """Compares a scoring batch against a reference distribution."""

    def fit(self, reference: pd.DataFrame) -> Any:
        """Record the reference distribution.

        Args:
            reference: The training feature frame.

        Returns:
            The fitted monitor.
        """
        ...

    def detect(self, current: pd.DataFrame) -> DriftReport:
        """Assess a batch for drift.

        Args:
            current: The scoring batch.

        Returns:
            The drift report.
        """
        ...

    def enforce(self, report: DriftReport) -> DriftReport:
        """Apply the fail-safe policy to a report.

        Args:
            report: The drift report.

        Returns:
            The report, when scoring may continue.
        """
        ...


@runtime_checkable
class ExplainerPort(Protocol):
    """Produces local explanations."""

    def explain_instance(
        self, row: pd.DataFrame, *, entity_id: str, model_id: str | None = None
    ) -> ScoreExplanation:
        """Explain a single obligor.

        Args:
            row: A one-row feature matrix.
            entity_id: Obligor identifier.
            model_id: Model identifier.

        Returns:
            The explanation.
        """
        ...


@runtime_checkable
class ModelRegistryPort(Protocol):
    """Persists and retrieves trained models."""

    def register_model(self, bundle: Any, *, promote: bool = True) -> str | None:
        """Persist and register a model bundle.

        Args:
            bundle: The bundle to register.
            promote: Mark it as the champion.

        Returns:
            The registry URI, or ``None`` when only stored locally.
        """
        ...

    def load_champion(self) -> Any:
        """Load the promoted champion.

        Returns:
            The champion bundle.
        """
        ...

    def log_performance(self, performance: ModelPerformance) -> None:
        """Record an evaluation result.

        Args:
            performance: The record to log.
        """
        ...
