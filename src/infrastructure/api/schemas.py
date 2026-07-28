"""Pydantic v2 request and response models.

The scoring request model is *generated* from
:data:`domain.entities.FEATURE_CONTRACT` rather than hand-written. A feature
added to the domain contract immediately gains API validation, bounds and an
OpenAPI description, and it becomes impossible for the API to accept a payload
the model cannot score — the two cannot drift apart because there is only one
declaration.

The cost is that the generated model's fields are invisible to a static type
checker. That is the right trade here: the alternative is a 27-field literal
duplicate of the contract that a reviewer must diff by eye.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from domain.entities import FEATURE_CONTRACT
from domain.value_objects import FieldSpec, RiskBand, Sector

__all__ = [
    "BatchPredictRequest",
    "CounterfactualResponse",
    "EntityFeatures",
    "ExplainRequest",
    "ExplainResponse",
    "FeatureContributionModel",
    "HealthResponse",
    "PredictResponse",
    "PredictionItem",
    "ScenarioRequest",
    "ScenarioResponse",
    "ScenarioSectorImpact",
]


def _field_for(spec: FieldSpec) -> tuple[Any, Any]:
    """Build the annotation and ``Field`` for one contract column.

    Args:
        spec: The contract field specification.

    Returns:
        An ``(annotation, FieldInfo)`` pair for :func:`pydantic.create_model`.
    """
    constraints: dict[str, Any] = {"description": spec.description}
    if spec.unit:
        constraints["json_schema_extra"] = {"unit": spec.unit}

    if spec.allowed is not None:
        annotation: Any = Literal[tuple(spec.allowed)]  # type: ignore[valid-type]
    elif spec.dtype in ("float", "int"):
        annotation = float if spec.dtype == "float" else int
        if spec.ge is not None:
            constraints["ge"] = spec.ge
        if spec.le is not None:
            constraints["le"] = spec.le
    else:
        annotation = str

    # Every feature is optional. A partially observed obligor still gets scored,
    # with absent values falling back to the contract's typical value during
    # matrix construction rather than being rejected at the edge.
    return (annotation | None, Field(default=None, **constraints))


def _build_entity_model() -> type[BaseModel]:
    """Generate the scoring request model from the domain contract.

    Returns:
        A Pydantic model with one optional field per contract feature.
    """
    fields: dict[str, Any] = {
        spec.name: _field_for(spec) for spec in FEATURE_CONTRACT.fields if spec.is_feature
    }
    fields["entity_id"] = (
        str | None,
        Field(default=None, description="Caller's identifier for this obligor"),
    )
    # The config must be passed to create_model, not assigned afterwards:
    # assigning `model_config` post-creation leaves the compiled validator built
    # from the default config, so `extra="forbid"` would silently not apply and
    # a typo'd field name would be accepted and ignored.
    return create_model(
        "EntityFeatures",
        __config__=ConfigDict(
            extra="forbid",
            json_schema_extra={"description": "Generated from domain.entities.FEATURE_CONTRACT"},
        ),
        **fields,
    )


#: One obligor's features. Generated from the domain contract.
EntityFeatures: type[BaseModel] = _build_entity_model()


class BatchPredictRequest(BaseModel):
    """A scoring request.

    Attributes:
        entities: Obligors to score.
        explain: Attach a SHAP explanation to every prediction.
        check_drift: Run the drift check before scoring.
        overlay_log_odds: Expert judgement shift applied to every PD.
    """

    model_config = ConfigDict(extra="forbid")

    entities: list[EntityFeatures] = Field(  # type: ignore[valid-type]
        ..., min_length=1, description="Obligors to score"
    )
    explain: bool = Field(default=False, description="Attach a SHAP explanation")
    check_drift: bool = Field(default=True, description="Run the drift check first")
    overlay_log_odds: float = Field(
        default=0.0,
        ge=-5.0,
        le=5.0,
        description="Expert overlay in log-odds space; positive is more conservative",
    )


class FeatureContributionModel(BaseModel):
    """One feature's contribution to a prediction.

    Attributes:
        feature: Feature name.
        value: The feature's value for this obligor.
        contribution: Signed attribution.
        direction: Whether the feature raised or lowered risk.
    """

    feature: str
    value: float | str
    contribution: float
    direction: str


class PredictionItem(BaseModel):
    """One obligor's score.

    Attributes:
        entity_id: Obligor identifier.
        probability_of_default: Reported PD after floor, cap and overlay.
        risk_band: Bucketed rating.
        is_flagged: Whether the PD breaches the decision threshold.
        threshold: Threshold applied.
        credit_score: Scorecard-style points, higher is safer.
        top_drivers: Attributions, when explanation was requested.
    """

    entity_id: str
    probability_of_default: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    is_flagged: bool
    threshold: float
    credit_score: float
    top_drivers: list[FeatureContributionModel] = Field(default_factory=list)


class DriftSummary(BaseModel):
    """Drift status attached to a scoring response.

    Attributes:
        severity: Worst severity across monitored features.
        max_psi: Largest PSI observed.
        n_drifted: Number of drifting features.
        drifted_features: The drifting feature names, worst first.
        policy: Fail-safe policy in force.
    """

    severity: str
    max_psi: float
    n_drifted: int
    drifted_features: list[str] = Field(default_factory=list)
    policy: str = "block"


class PredictResponse(BaseModel):
    """The result of a scoring request.

    Attributes:
        predictions: One item per obligor.
        model_id: Model that produced the scores.
        model_version: Training timestamp of the model.
        scored_at: Server timestamp.
        n_scored: Number of obligors scored.
        n_flagged: Number breaching the threshold.
        drift: Drift status, when the check ran.
        latency_ms: Server-side handling time.
    """

    predictions: list[PredictionItem]
    model_id: str
    model_version: str
    scored_at: datetime
    n_scored: int
    n_flagged: int
    drift: DriftSummary | None = None
    latency_ms: float = 0.0


class ExplainRequest(BaseModel):
    """A request for a local explanation.

    Attributes:
        entity: The obligor to explain.
        method: Explainer to use.
        top_k: Number of drivers to return.
        target_pd: Target PD for the counterfactual method.
    """

    model_config = ConfigDict(extra="forbid")

    entity: EntityFeatures  # type: ignore[valid-type]
    method: Literal["shap", "lime", "dice"] = "shap"
    top_k: int = Field(default=10, ge=1, le=50)
    target_pd: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Only used when method is 'dice'"
    )


class CounterfactualChange(BaseModel):
    """One feature move in a recourse plan.

    Attributes:
        feature: Feature to change.
        description: Business meaning of the feature.
        current: Current value.
        required: Value needed.
        change: Signed difference.
    """

    feature: str
    description: str
    current: float
    required: float
    change: float


class CounterfactualResponse(BaseModel):
    """A single recourse option.

    Attributes:
        option: Option number.
        original_pd: PD before the changes.
        counterfactual_pd: PD after the changes.
        pd_reduction: Improvement delivered.
        n_changes: Number of features that must move.
        changes: The required moves.
    """

    option: int
    original_pd: float
    counterfactual_pd: float
    pd_reduction: float
    n_changes: int
    changes: list[CounterfactualChange] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    """The result of an explanation request.

    Attributes:
        entity_id: Obligor explained.
        method: Explainer used.
        model_id: Model explained.
        probability_of_default: The score being explained.
        base_value: Explainer baseline.
        contributions: Ranked attributions.
        counterfactuals: Recourse options, for the ``dice`` method.
        latency_ms: Server-side handling time.
    """

    entity_id: str
    method: str
    model_id: str
    probability_of_default: float
    base_value: float = 0.0
    contributions: list[FeatureContributionModel] = Field(default_factory=list)
    counterfactuals: list[CounterfactualResponse] = Field(default_factory=list)
    latency_ms: float = 0.0


class ScenarioRequest(BaseModel):
    """A macro scenario simulation request.

    Attributes:
        entities: Obligors to stress. Uses the demo portfolio when omitted.
        scenario: Scenario name; every configured scenario when omitted.
        sectors: Sector per obligor, when not carried on the entities.
    """

    model_config = ConfigDict(extra="forbid")

    entities: list[EntityFeatures] | None = None  # type: ignore[valid-type]
    scenario: str | None = Field(
        default=None, description="Scenario name; omit to run all configured scenarios"
    )
    sectors: list[Sector] | None = None

    @field_validator("scenario")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        """Normalise the scenario name.

        Args:
            value: Raw scenario name.

        Returns:
            The trimmed name, or ``None``.
        """
        return value.strip() if value else None


class ScenarioSectorImpact(BaseModel):
    """A scenario's effect on one sector.

    Attributes:
        sector: The sector.
        odds_multiplier: Multiplier applied to default odds.
        stressed_pd: Mean stressed PD for the sector.
    """

    sector: str
    odds_multiplier: float
    stressed_pd: float = 0.0


class ScenarioOutcome(BaseModel):
    """Portfolio outcome under one scenario.

    Attributes:
        scenario: Scenario name.
        severity: Regulatory severity class.
        description: Narrative description.
        baseline_pd: Mean PD before the shock.
        stressed_pd: Mean PD after the shock.
        pd_uplift: Absolute increase.
        relative_uplift: Increase relative to baseline.
        n_entities: Obligors included.
        by_sector: Per-sector impacts, riskiest first.
    """

    scenario: str
    severity: str
    description: str = ""
    baseline_pd: float
    stressed_pd: float
    pd_uplift: float
    relative_uplift: float
    n_entities: int
    by_sector: list[ScenarioSectorImpact] = Field(default_factory=list)


class ScenarioResponse(BaseModel):
    """The result of a scenario request.

    Attributes:
        outcomes: One outcome per scenario run, least severe first.
        model_id: Model that produced the baseline PDs.
        latency_ms: Server-side handling time.
    """

    outcomes: list[ScenarioOutcome]
    model_id: str
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    """Service readiness.

    Attributes:
        status: ``ok`` or ``degraded``.
        model_loaded: Whether a champion is loaded.
        model_id: Loaded model identifier.
        model_trained_at: Training timestamp of the loaded model.
        is_calibrated: Whether the loaded model is calibrated.
        n_features: Feature count the model expects.
        drift_monitoring: Whether a drift reference is available.
        drift_policy: Fail-safe policy in force.
        version: API version.
        checks: Individual readiness checks.
    """

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_id: str = ""
    model_trained_at: str = ""
    is_calibrated: bool = False
    n_features: int = 0
    drift_monitoring: bool = False
    drift_policy: str = ""
    version: str = ""
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """A structured error body.

    Attributes:
        error_type: Exception class name.
        message: Human-readable description.
        context: Structured detail.
    """

    error_type: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


#: Convenience alias for the annotated batch size limit used in routes.
BatchSize = Annotated[int, Field(ge=1)]
