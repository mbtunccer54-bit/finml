"""Repository ports.

These :class:`~typing.Protocol` definitions are the outbound boundary of the
hexagon. The domain states what it needs; :mod:`infrastructure` supplies the
adapters. Because the protocols are structural, no adapter has to import or
subclass anything from this module — the dependency arrow points inwards only.

Every signature here is expressed in domain types (or plain builtins), never in
``pandas``/``numpy`` types, which is what keeps the domain substitutable.

Standard library only — see :mod:`domain`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Protocol, runtime_checkable

from domain.entities import (
    DriftReport,
    FinancialEntity,
    MacroScenario,
    ModelPerformance,
    RiskScore,
    ScoreExplanation,
)
from domain.value_objects import EntityId, ModelId, SentimentScore

__all__ = [
    "DriftRepository",
    "EntityRepository",
    "ExplanationRepository",
    "MacroRepository",
    "ModelRepository",
    "NewsRepository",
    "ScenarioRepository",
    "ScoreRepository",
]


@runtime_checkable
class EntityRepository(Protocol):
    """Access to obligor observations."""

    def get(self, entity_id: EntityId, as_of: date | None = None) -> FinancialEntity:
        """Fetch a single observation.

        Args:
            entity_id: Obligor identifier.
            as_of: Observation date; the latest available is used when omitted.

        Returns:
            The matching entity.

        Raises:
            InfrastructureError: If no such observation exists.
        """
        ...

    def list_ids(self) -> Sequence[EntityId]:
        """List every known obligor identifier.

        Returns:
            The available identifiers.
        """
        ...

    def find_by_period(self, start: date, end: date) -> Sequence[FinancialEntity]:
        """Fetch every observation in a closed date range.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            The matching observations.
        """
        ...


@runtime_checkable
class MacroRepository(Protocol):
    """Access to macroeconomic time series."""

    def latest(self) -> Mapping[str, float]:
        """Fetch the most recent macro state.

        Returns:
            Macro variable name mapped to its latest value.
        """
        ...

    def history(self, variable: str, start: date, end: date) -> Sequence[tuple[date, float]]:
        """Fetch one macro series over a date range.

        Args:
            variable: Macro variable name.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Chronologically ordered ``(date, value)`` observations.
        """
        ...

    def forecast(self, horizon_quarters: int) -> Sequence[Mapping[str, float]]:
        """Project the macro state forward with the satellite model.

        Args:
            horizon_quarters: Number of quarters to project.

        Returns:
            One macro state per projected quarter.
        """
        ...


@runtime_checkable
class NewsRepository(Protocol):
    """Access to news text and derived sentiment."""

    def sentiment_for(self, entity_id: EntityId, as_of: date) -> SentimentScore:
        """Fetch aggregated sentiment for an obligor.

        Args:
            entity_id: Obligor identifier.
            as_of: Observation date terminating the lookback window.

        Returns:
            The aggregated sentiment, neutral when no documents were found.
        """
        ...

    def documents_for(self, entity_id: EntityId, as_of: date, lookback_days: int) -> Sequence[str]:
        """Fetch the raw documents behind a sentiment score.

        Args:
            entity_id: Obligor identifier.
            as_of: End of the lookback window.
            lookback_days: Window length in days.

        Returns:
            The document texts.
        """
        ...


@runtime_checkable
class ScenarioRepository(Protocol):
    """Access to the configured stress-testing scenarios."""

    def get(self, name: str) -> MacroScenario:
        """Fetch a scenario by name.

        Args:
            name: Scenario identifier.

        Returns:
            The matching scenario.

        Raises:
            ScenarioError: If no scenario carries that name.
        """
        ...

    def list_all(self) -> Sequence[MacroScenario]:
        """List every configured scenario.

        Returns:
            The available scenarios.
        """
        ...


@runtime_checkable
class ModelRepository(Protocol):
    """Access to trained model artifacts and their recorded performance."""

    def load(self, model_id: ModelId) -> object:
        """Load a model artifact.

        The return type is deliberately opaque: the domain never calls the model
        directly, it only passes the handle back to an infrastructure adapter.

        Args:
            model_id: Model identifier.

        Returns:
            The deserialised model object.

        Raises:
            ModelNotFoundError: If no artifact exists for the identifier.
        """
        ...

    def load_champion(self) -> tuple[ModelId, object]:
        """Load the model currently promoted to production.

        Returns:
            The champion's identifier and artifact.

        Raises:
            ModelNotFoundError: If no champion has been promoted.
        """
        ...

    def performance(self, model_id: ModelId) -> Sequence[ModelPerformance]:
        """Fetch recorded evaluation metrics for a model.

        Args:
            model_id: Model identifier.

        Returns:
            One record per evaluated split.
        """
        ...

    def list_models(self) -> Sequence[ModelId]:
        """List every registered model.

        Returns:
            The registered model identifiers.
        """
        ...


@runtime_checkable
class ScoreRepository(Protocol):
    """Persistence for scoring results."""

    def save(self, score: RiskScore) -> None:
        """Persist one score.

        Args:
            score: The score to store.
        """
        ...

    def save_many(self, scores: Sequence[RiskScore]) -> None:
        """Persist a batch of scores.

        Args:
            scores: The scores to store.
        """
        ...

    def latest_for(self, entity_id: EntityId) -> RiskScore | None:
        """Fetch the most recent score for an obligor.

        Args:
            entity_id: Obligor identifier.

        Returns:
            The latest score, or ``None`` when the obligor was never scored.
        """
        ...


@runtime_checkable
class ExplanationRepository(Protocol):
    """Persistence for local explanations, required for audit replay."""

    def save(self, explanation: ScoreExplanation) -> None:
        """Persist one explanation.

        Args:
            explanation: The explanation to store.
        """
        ...

    def get(self, entity_id: EntityId, method: str) -> ScoreExplanation | None:
        """Fetch a stored explanation.

        Args:
            entity_id: Obligor identifier.
            method: Explainer name.

        Returns:
            The stored explanation, or ``None`` when absent.
        """
        ...


@runtime_checkable
class DriftRepository(Protocol):
    """Persistence for drift monitoring history."""

    def save(self, report: DriftReport) -> None:
        """Persist one drift report.

        Args:
            report: The report to store.
        """
        ...

    def latest(self) -> DriftReport | None:
        """Fetch the most recent drift report.

        Returns:
            The latest report, or ``None`` when none has been recorded.
        """
        ...
