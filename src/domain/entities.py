"""Domain entities and the canonical dataset contract.

Entities have identity and a lifecycle, unlike the interchangeable
:mod:`domain.value_objects`. They are still modelled as frozen dataclasses:
state transitions return new instances, which keeps a scored obligor auditable
after the fact.

:data:`FEATURE_CONTRACT` is the single source of truth for the modelling
dataset. The pandera schemas, the Pydantic API models and the dashboard slider
ranges are all compiled from it, so adding a feature here propagates through the
stack without duplicated literals.

Standard library only — see :mod:`domain`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Final, Self

from domain.exceptions import InvariantViolationError, ScenarioError, ValidationError
from domain.value_objects import (
    CostMatrix,
    DriftSeverity,
    EntityId,
    FeatureContract,
    FieldSpec,
    ModelId,
    MonetaryAmount,
    ProbabilityOfDefault,
    RiskBand,
    ScenarioSeverity,
    Sector,
    SentimentScore,
)

__all__ = [
    "DATE_COLUMN",
    "ENTITY_ID_COLUMN",
    "FEATURE_CONTRACT",
    "MACRO_VARIABLES",
    "TARGET_COLUMN",
    "CounterfactualExample",
    "DriftReport",
    "FeatureContribution",
    "FeatureDrift",
    "FinancialEntity",
    "MacroScenario",
    "ModelPerformance",
    "PortfolioExposure",
    "RiskScore",
    "ScenarioResult",
    "ScoreExplanation",
]

# ---------------------------------------------------------------------------
# Reserved column names
# ---------------------------------------------------------------------------
ENTITY_ID_COLUMN: Final[str] = "entity_id"
DATE_COLUMN: Final[str] = "as_of_date"
TARGET_COLUMN: Final[str] = "default_flag"

#: Macro drivers produced by the satellite models and consumed by the bridge.
MACRO_VARIABLES: Final[tuple[str, ...]] = (
    "gdp_growth",
    "unemployment_rate",
    "interest_rate",
    "inflation_rate",
    "credit_spread",
)


# ---------------------------------------------------------------------------
# The canonical dataset contract
# ---------------------------------------------------------------------------
def _build_feature_contract() -> FeatureContract:
    """Assemble the modelling dataset contract.

    Bounds are set to reject data-quality accidents rather than to censor
    genuine tail behaviour, so they sit well outside the observed range of each
    ratio.

    Returns:
        The fully populated :class:`FeatureContract`.
    """
    identity = (
        FieldSpec(ENTITY_ID_COLUMN, "str", "Stable obligor identifier", is_feature=False),
        FieldSpec(DATE_COLUMN, "datetime", "Observation date", is_feature=False),
        FieldSpec(
            TARGET_COLUMN,
            "int",
            "1 if the obligor defaulted within the horizon",
            ge=0,
            le=1,
            is_feature=False,
        ),
    )

    liquidity = (
        FieldSpec(
            "current_ratio",
            "float",
            "Current assets / current liabilities",
            ge=0.0,
            le=25.0,
            unit="x",
            typical=1.6,
        ),
        FieldSpec(
            "quick_ratio",
            "float",
            "(Current assets - inventory) / current liabilities",
            ge=0.0,
            le=20.0,
            unit="x",
            typical=1.1,
        ),
        FieldSpec(
            "working_capital_ratio",
            "float",
            "Working capital / total assets",
            ge=-2.0,
            le=2.0,
            unit="x",
            typical=0.18,
        ),
        FieldSpec(
            "cash_flow_to_debt",
            "float",
            "Operating cash flow / total debt",
            ge=-5.0,
            le=10.0,
            unit="x",
            typical=0.28,
        ),
    )

    leverage = (
        FieldSpec(
            "debt_to_equity",
            "float",
            "Total debt / shareholders' equity",
            ge=0.0,
            le=50.0,
            unit="x",
            typical=1.3,
        ),
        FieldSpec(
            "debt_to_assets",
            "float",
            "Total debt / total assets",
            ge=0.0,
            le=3.0,
            unit="x",
            typical=0.42,
        ),
        FieldSpec(
            "interest_coverage",
            "float",
            "EBIT / interest expense",
            ge=-50.0,
            le=100.0,
            unit="x",
            typical=4.5,
        ),
    )

    profitability = (
        FieldSpec(
            "return_on_assets",
            "float",
            "Net income / total assets",
            ge=-2.0,
            le=1.0,
            unit="x",
            typical=0.05,
        ),
        FieldSpec(
            "return_on_equity",
            "float",
            "Net income / shareholders' equity",
            ge=-5.0,
            le=3.0,
            unit="x",
            typical=0.12,
        ),
        FieldSpec(
            "net_profit_margin",
            "float",
            "Net income / revenue",
            ge=-5.0,
            le=1.0,
            unit="x",
            typical=0.07,
        ),
        FieldSpec(
            "operating_margin",
            "float",
            "Operating income / revenue",
            ge=-5.0,
            le=1.0,
            unit="x",
            typical=0.11,
        ),
    )

    efficiency = (
        FieldSpec(
            "asset_turnover",
            "float",
            "Revenue / total assets",
            ge=0.0,
            le=10.0,
            unit="x",
            typical=0.85,
        ),
        FieldSpec(
            "revenue_growth",
            "float",
            "Year-on-year revenue growth",
            ge=-1.0,
            le=5.0,
            unit="x",
            typical=0.04,
        ),
        FieldSpec(
            "altman_z_score",
            "float",
            "Altman Z-score composite",
            ge=-10.0,
            le=20.0,
            unit="",
            typical=2.9,
        ),
    )

    nlp = (
        FieldSpec(
            "sentiment_compound",
            "float",
            "FinBERT net sentiment (positive - negative)",
            ge=-1.0,
            le=1.0,
            unit="",
            typical=0.05,
        ),
        FieldSpec(
            "sentiment_uncertainty",
            "float",
            "Normalised entropy of the sentiment posterior",
            ge=0.0,
            le=1.0,
            unit="",
            typical=0.55,
        ),
        FieldSpec(
            "sentiment_negative_prob",
            "float",
            "FinBERT negative-class probability",
            ge=0.0,
            le=1.0,
            unit="",
            typical=0.25,
        ),
        FieldSpec(
            "topic_credit_risk_score",
            "float",
            "BERTopic mass on credit-risk topics",
            ge=0.0,
            le=1.0,
            unit="",
            typical=0.2,
        ),
        FieldSpec(
            "topic_liquidity_risk_score",
            "float",
            "BERTopic mass on liquidity-risk topics",
            ge=0.0,
            le=1.0,
            unit="",
            typical=0.15,
        ),
        FieldSpec(
            "topic_operational_risk_score",
            "float",
            "BERTopic mass on operational-risk topics",
            ge=0.0,
            le=1.0,
            unit="",
            typical=0.12,
        ),
        FieldSpec(
            "news_volume",
            "float",
            "Documents observed in the lookback window",
            ge=0.0,
            le=5000.0,
            unit="docs",
            typical=12.0,
        ),
    )

    macro = (
        FieldSpec(
            "gdp_growth",
            "float",
            "Real GDP growth, annualised",
            ge=-0.25,
            le=0.25,
            unit="%",
            typical=0.021,
        ),
        FieldSpec(
            "unemployment_rate",
            "float",
            "Unemployment rate",
            ge=0.0,
            le=0.40,
            unit="%",
            typical=0.052,
        ),
        FieldSpec(
            "interest_rate", "float", "Policy rate", ge=-0.05, le=0.30, unit="%", typical=0.031
        ),
        FieldSpec(
            "inflation_rate",
            "float",
            "Headline CPI inflation",
            ge=-0.10,
            le=0.50,
            unit="%",
            typical=0.026,
        ),
        FieldSpec(
            "credit_spread",
            "float",
            "Corporate credit spread over risk-free",
            ge=0.0,
            le=0.30,
            unit="%",
            typical=0.018,
        ),
    )

    categorical = (
        FieldSpec(
            "sector",
            "category",
            "GICS-like sector",
            allowed=tuple(s.value for s in Sector),
            typical=0.0,
        ),
    )

    return FeatureContract(
        identity + liquidity + leverage + profitability + efficiency + nlp + macro + categorical
    )


#: The canonical modelling dataset contract.
FEATURE_CONTRACT: Final[FeatureContract] = _build_feature_contract()


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FinancialEntity:
    """An obligor observed at a point in time.

    Identity is the ``(entity_id, as_of_date)`` pair: the same company observed
    in two quarters is two entities, which is what keeps the panel structure
    explicit and prevents look-ahead leakage.

    Attributes:
        entity_id: Stable obligor identifier.
        name: Human-readable company name.
        sector: Sector classification.
        country: ISO-3166 alpha-2 country code.
        as_of_date: Observation date.
        ratios: Financial ratio values keyed by feature name.
        macro: Macro variable values prevailing at ``as_of_date``.
        sentiment: Aggregated news sentiment, if any was available.
        exposure: Exposure at default, used for portfolio aggregation.
        default_flag: Realised outcome, where known.
    """

    entity_id: EntityId
    name: str
    sector: Sector
    as_of_date: date
    ratios: Mapping[str, float] = field(default_factory=dict)
    macro: Mapping[str, float] = field(default_factory=dict)
    sentiment: SentimentScore | None = None
    country: str = "US"
    exposure: MonetaryAmount | None = None
    default_flag: int | None = None

    def __post_init__(self) -> None:
        """Validate identity and outcome invariants.

        Raises:
            ValidationError: If the identifier or name is blank, the country
                code is malformed, or ``default_flag`` is not ``0``/``1``/``None``.
        """
        if not str(self.entity_id).strip():
            raise ValidationError("entity_id must not be blank")
        if not self.name.strip():
            raise ValidationError("name must not be blank", entity_id=self.entity_id)
        code = self.country.strip().upper()
        if len(code) != 2 or not code.isalpha():
            raise ValidationError(
                "country must be an ISO-3166 alpha-2 code",
                entity_id=self.entity_id,
                country=self.country,
            )
        object.__setattr__(self, "country", code)
        if self.default_flag not in (None, 0, 1):
            raise ValidationError(
                "default_flag must be 0, 1 or None",
                entity_id=self.entity_id,
                value=self.default_flag,
            )
        object.__setattr__(self, "ratios", dict(self.ratios))
        object.__setattr__(self, "macro", dict(self.macro))

    @property
    def key(self) -> tuple[str, date]:
        """Composite identity of this observation.

        Returns:
            The ``(entity_id, as_of_date)`` pair.
        """
        return (str(self.entity_id), self.as_of_date)

    def feature_values(
        self, contract: FeatureContract = FEATURE_CONTRACT
    ) -> dict[str, float | str]:
        """Project the entity onto the model's feature space.

        Missing numeric features fall back to the contract's ``typical`` value so
        a partially observed obligor can still be scored; the caller is expected
        to log the imputation.

        Args:
            contract: Contract describing the expected feature space.

        Returns:
            A mapping from feature name to value, in canonical contract order.
        """
        derived = self._derived_sentiment_features()
        out: dict[str, float | str] = {}
        for spec in contract.fields:
            if not spec.is_feature:
                continue
            if spec.name == "sector":
                out[spec.name] = self.sector.value
                continue
            if spec.name in self.ratios:
                out[spec.name] = float(self.ratios[spec.name])
            elif spec.name in self.macro:
                out[spec.name] = float(self.macro[spec.name])
            elif spec.name in derived:
                out[spec.name] = derived[spec.name]
            else:
                out[spec.name] = spec.typical
        return out

    def _derived_sentiment_features(self) -> dict[str, float]:
        """Expand the sentiment value object into flat model features.

        Returns:
            Sentiment-derived feature values, empty when no sentiment is set.
        """
        if self.sentiment is None:
            return {}
        return {
            "sentiment_compound": self.sentiment.compound,
            "sentiment_uncertainty": self.sentiment.uncertainty,
            "sentiment_negative_prob": self.sentiment.negative_prob,
            "news_volume": float(self.sentiment.n_documents),
        }

    def with_overrides(self, overrides: Mapping[str, float]) -> Self:
        """Return a copy with selected ratio or macro values replaced.

        This backs the dashboard's what-if sliders: the original observation is
        never mutated, so the baseline stays available for comparison.

        Args:
            overrides: Feature name to replacement value.

        Returns:
            A new entity carrying the overrides.

        Raises:
            ValidationError: If a name matches no known feature.
        """
        known = set(FEATURE_CONTRACT.feature_names)
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValidationError(
                "Unknown feature in override", unknown=unknown, entity_id=self.entity_id
            )
        new_ratios = dict(self.ratios)
        new_macro = dict(self.macro)
        for name, value in overrides.items():
            spec = FEATURE_CONTRACT.get(name)
            clamped = spec.clamp(float(value))
            if name in MACRO_VARIABLES:
                new_macro[name] = clamped
            else:
                new_ratios[name] = clamped
        return replace(self, ratios=new_ratios, macro=new_macro)

    def with_macro(self, macro: Mapping[str, float]) -> Self:
        """Return a copy observed under a different macro state.

        Args:
            macro: Replacement macro variable values.

        Returns:
            A new entity carrying the merged macro state.
        """
        return replace(self, macro={**self.macro, **dict(macro)})


@dataclass(frozen=True, slots=True)
class MacroScenario:
    """A named macroeconomic scenario used for stress testing.

    Shocks are expressed as *absolute deltas* in the natural unit of each
    variable, matching how CCAR and EBA publish their scenario tables. A
    ``gdp_growth`` shock of ``-0.04`` moves growth from +2% to -2%.

    Attributes:
        name: Scenario identifier.
        severity: Regulatory severity class.
        shocks: Absolute deltas keyed by macro variable name.
        horizon_quarters: Projection horizon.
        description: Narrative used in the dashboard and reports.
        probability_weight: Weight when aggregating scenarios into an expectation.
    """

    name: str
    severity: ScenarioSeverity
    shocks: Mapping[str, float] = field(default_factory=dict)
    horizon_quarters: int = 8
    description: str = ""
    probability_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the scenario definition.

        Raises:
            ScenarioError: If the name is blank, the horizon is not positive,
                the weight is negative, or a shock names an unknown variable.
        """
        if not self.name.strip():
            raise ScenarioError("Scenario name must not be blank")
        if self.horizon_quarters <= 0:
            raise ScenarioError(
                "Scenario horizon must be positive",
                name=self.name,
                horizon_quarters=self.horizon_quarters,
            )
        if self.probability_weight < 0.0:
            raise ScenarioError(
                "Scenario weight must not be negative",
                name=self.name,
                weight=self.probability_weight,
            )
        unknown = sorted(set(self.shocks) - set(MACRO_VARIABLES))
        if unknown:
            raise ScenarioError(
                "Scenario shocks reference unknown macro variables",
                name=self.name,
                unknown=unknown,
                known=list(MACRO_VARIABLES),
            )
        object.__setattr__(self, "shocks", dict(self.shocks))

    @property
    def is_stress(self) -> bool:
        """Whether the scenario is adverse rather than the baseline.

        Returns:
            ``True`` for adverse and severely adverse scenarios.
        """
        return self.severity is not ScenarioSeverity.BASELINE

    def shock_for(self, variable: str) -> float:
        """Look up the shock applied to one variable.

        Args:
            variable: Macro variable name.

        Returns:
            The absolute delta, or ``0.0`` when the scenario leaves it untouched.
        """
        return float(self.shocks.get(variable, 0.0))

    def apply_to(self, macro: Mapping[str, float]) -> dict[str, float]:
        """Project a macro state through the scenario's shocks.

        Args:
            macro: Baseline macro state.

        Returns:
            The shocked macro state.
        """
        return {name: float(value) + self.shock_for(name) for name, value in macro.items()}


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One feature's signed contribution to a single prediction.

    Attributes:
        feature: Feature name.
        value: The feature's value for this observation.
        contribution: Signed contribution in the explainer's output space.
    """

    feature: str
    value: float | str
    contribution: float

    @property
    def direction(self) -> str:
        """Whether the feature pushed the prediction up or down.

        Returns:
            ``"increases_risk"``, ``"decreases_risk"`` or ``"neutral"``.
        """
        if self.contribution > 0:
            return "increases_risk"
        if self.contribution < 0:
            return "decreases_risk"
        return "neutral"


@dataclass(frozen=True, slots=True)
class ScoreExplanation:
    """Local attribution for one scored obligor.

    Attributes:
        entity_id: The obligor explained.
        method: Explainer that produced the attributions.
        base_value: Explainer baseline, in the same space as the contributions.
        contributions: Per-feature attributions.
        model_id: Model the explanation refers to.
    """

    entity_id: EntityId
    method: str
    base_value: float
    contributions: tuple[FeatureContribution, ...] = field(default_factory=tuple)
    model_id: ModelId | None = None

    def top_drivers(self, k: int = 5) -> tuple[FeatureContribution, ...]:
        """Return the ``k`` features with the largest absolute contribution.

        Args:
            k: Number of drivers to return.

        Returns:
            The most influential contributions, strongest first.
        """
        ranked = sorted(self.contributions, key=lambda c: abs(c.contribution), reverse=True)
        return tuple(ranked[: max(k, 0)])

    @property
    def total_contribution(self) -> float:
        """Sum of every attribution.

        Returns:
            The additive total, which for SHAP reconstructs the margin when
            added to :attr:`base_value`.
        """
        return sum(c.contribution for c in self.contributions)


@dataclass(frozen=True, slots=True)
class RiskScore:
    """The output of scoring one obligor with one model.

    Attributes:
        entity_id: Obligor scored.
        probability_of_default: Model output.
        model_id: Model that produced the score.
        scored_at: Timestamp of scoring.
        threshold: Decision cut-off applied.
        band: Bucketed rating.
        explanation: Attached local explanation, if requested.
        drift_severity: Drift status of the batch this score came from.
    """

    entity_id: EntityId
    probability_of_default: ProbabilityOfDefault
    model_id: ModelId
    scored_at: datetime
    threshold: float = 0.5
    band: RiskBand = RiskBand.MEDIUM
    explanation: ScoreExplanation | None = None
    drift_severity: DriftSeverity = DriftSeverity.NONE

    def __post_init__(self) -> None:
        """Validate the decision threshold.

        Raises:
            InvariantViolationError: If the threshold is outside ``(0, 1)``.
        """
        if not 0.0 < self.threshold < 1.0:
            raise InvariantViolationError(
                "Decision threshold must lie strictly inside (0, 1)",
                entity_id=self.entity_id,
                threshold=self.threshold,
            )

    @property
    def is_flagged(self) -> bool:
        """Whether the obligor breaches the decision threshold.

        Returns:
            ``True`` when the PD is at or above the threshold.
        """
        return self.probability_of_default.value >= self.threshold

    @property
    def pd_value(self) -> float:
        """Convenience accessor for the raw probability.

        Returns:
            The probability of default as a float.
        """
        return self.probability_of_default.value

    def expected_loss(
        self, *, loss_given_default: float, exposure: MonetaryAmount
    ) -> MonetaryAmount:
        """Compute expected loss for this obligor.

        Args:
            loss_given_default: Fraction of exposure lost on default, in ``[0, 1]``.
            exposure: Exposure at default.

        Returns:
            ``PD * LGD * EAD`` as a monetary amount.

        Raises:
            InvariantViolationError: If ``loss_given_default`` is outside ``[0, 1]``.
        """
        if not 0.0 <= loss_given_default <= 1.0:
            raise InvariantViolationError(
                "LGD must lie in [0, 1]", lgd=loss_given_default, entity_id=self.entity_id
            )
        return exposure.scaled(self.pd_value * loss_given_default)


# ---------------------------------------------------------------------------
# Monitoring and evaluation entities
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """Population-stability result for a single feature.

    Attributes:
        feature: Feature name.
        psi: Population Stability Index against the reference window.
        ks_statistic: Two-sample Kolmogorov-Smirnov statistic.
        ks_p_value: KS test p-value.
        severity: Bucketed severity derived from the PSI thresholds.
    """

    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    severity: DriftSeverity


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Drift assessment for one scoring batch.

    Attributes:
        computed_at: When the comparison ran.
        features: Per-feature results.
        reference_size: Row count of the reference window.
        current_size: Row count of the scored batch.
    """

    computed_at: datetime
    features: tuple[FeatureDrift, ...] = field(default_factory=tuple)
    reference_size: int = 0
    current_size: int = 0

    @property
    def severity(self) -> DriftSeverity:
        """Worst severity observed across features.

        Returns:
            The maximum per-feature severity, or ``NONE`` when empty.
        """
        if not self.features:
            return DriftSeverity.NONE
        return max((f.severity for f in self.features), key=lambda s: s.value)

    @property
    def drifted_features(self) -> tuple[FeatureDrift, ...]:
        """Features whose severity is above ``NONE``, worst first.

        Returns:
            The drifting features ordered by descending PSI.
        """
        drifted = [f for f in self.features if f.severity is not DriftSeverity.NONE]
        return tuple(sorted(drifted, key=lambda f: f.psi, reverse=True))

    @property
    def max_psi(self) -> float:
        """Largest PSI observed.

        Returns:
            The maximum PSI, or ``0.0`` when no features were compared.
        """
        return max((f.psi for f in self.features), default=0.0)


@dataclass(frozen=True, slots=True)
class ModelPerformance:
    """Evaluation metrics for one model on one dataset split.

    Attributes:
        model_id: Model evaluated.
        split: Dataset split name, for example ``"validation"``.
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the precision-recall curve.
        f1: F1 score at the decision threshold.
        precision: Precision at the decision threshold.
        recall: Recall at the decision threshold.
        brier_score: Mean squared error of the probabilities.
        expected_calibration_error: Binned calibration gap.
        business_cost: Cost-weighted total from the cost matrix.
        inference_latency_ms: Median single-row scoring latency.
        threshold: Decision threshold the point metrics were computed at.
        n_samples: Rows evaluated.
    """

    model_id: ModelId
    split: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    brier_score: float
    expected_calibration_error: float
    business_cost: float
    inference_latency_ms: float
    threshold: float = 0.5
    n_samples: int = 0

    def meets_sla(
        self,
        *,
        max_calibration_error: float,
        max_latency_ms: float,
        min_pr_auc: float,
    ) -> bool:
        """Check the model against the deployment gate.

        Args:
            max_calibration_error: Largest tolerated calibration error.
            max_latency_ms: Largest tolerated median latency.
            min_pr_auc: Smallest acceptable PR-AUC.

        Returns:
            ``True`` when every gate is satisfied.
        """
        return (
            self.expected_calibration_error <= max_calibration_error
            and self.inference_latency_ms <= max_latency_ms
            and self.pr_auc >= min_pr_auc
        )

    def as_dict(self) -> dict[str, float | str | int]:
        """Flatten the metrics for tabular benchmarking.

        Returns:
            A mapping suitable for constructing a benchmark dataframe row.
        """
        return {
            "model_id": str(self.model_id),
            "split": self.split,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "business_cost": self.business_cost,
            "inference_latency_ms": self.inference_latency_ms,
            "threshold": self.threshold,
            "n_samples": self.n_samples,
        }


@dataclass(frozen=True, slots=True)
class CounterfactualExample:
    """A minimally perturbed input that flips the model's decision.

    Attributes:
        entity_id: Obligor the counterfactual belongs to.
        original_pd: PD before the change.
        counterfactual_pd: PD after the change.
        changes: Feature name mapped to its ``(from, to)`` pair.
    """

    entity_id: EntityId
    original_pd: float
    counterfactual_pd: float
    changes: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise the change mapping to a plain dict."""
        object.__setattr__(self, "changes", dict(self.changes))

    @property
    def pd_reduction(self) -> float:
        """Absolute PD improvement delivered by the changes.

        Returns:
            ``original_pd - counterfactual_pd``.
        """
        return self.original_pd - self.counterfactual_pd

    @property
    def n_changes(self) -> int:
        """Number of features that had to move.

        Returns:
            The count of changed features — the sparsity of the recourse.
        """
        return len(self.changes)

    def required_deltas(self) -> dict[str, float]:
        """Signed change required for each feature.

        Returns:
            Feature name mapped to ``to - from``.
        """
        return {name: to - frm for name, (frm, to) in self.changes.items()}


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Portfolio outcome under one macro scenario.

    Attributes:
        scenario: Scenario applied.
        baseline_pd: Mean PD before the shock.
        stressed_pd: Mean PD after the shock.
        pd_by_sector: Stressed mean PD per sector.
        expected_loss: Portfolio expected loss under the scenario.
        n_entities: Obligors included.
    """

    scenario: MacroScenario
    baseline_pd: float
    stressed_pd: float
    pd_by_sector: Mapping[str, float] = field(default_factory=dict)
    expected_loss: MonetaryAmount | None = None
    n_entities: int = 0

    def __post_init__(self) -> None:
        """Normalise the sector mapping to a plain dict."""
        object.__setattr__(self, "pd_by_sector", dict(self.pd_by_sector))

    @property
    def pd_uplift(self) -> float:
        """Absolute increase in mean PD caused by the scenario.

        Returns:
            ``stressed_pd - baseline_pd``.
        """
        return self.stressed_pd - self.baseline_pd

    @property
    def relative_uplift(self) -> float:
        """Increase in mean PD relative to the baseline.

        Returns:
            The relative uplift, or ``0.0`` when the baseline PD is zero.
        """
        if self.baseline_pd <= 0.0:
            return 0.0
        return self.pd_uplift / self.baseline_pd

    def worst_sectors(self, k: int = 3) -> tuple[tuple[str, float], ...]:
        """Return the sectors with the highest stressed PD.

        Args:
            k: Number of sectors to return.

        Returns:
            ``(sector, pd)`` pairs, riskiest first.
        """
        ranked = sorted(self.pd_by_sector.items(), key=lambda kv: kv[1], reverse=True)
        return tuple(ranked[: max(k, 0)])


@dataclass(frozen=True, slots=True)
class PortfolioExposure:
    """An aggregate view over a set of scored obligors.

    Attributes:
        scores: The scores in the portfolio.
        loss_given_default: Portfolio-wide LGD assumption.
        cost_matrix: Cost matrix used for decision reporting.
    """

    scores: Sequence[RiskScore] = field(default_factory=tuple)
    loss_given_default: float = 0.45
    cost_matrix: CostMatrix = field(default_factory=CostMatrix)

    def __post_init__(self) -> None:
        """Validate the LGD assumption.

        Raises:
            InvariantViolationError: If LGD is outside ``[0, 1]``.
        """
        if not 0.0 <= self.loss_given_default <= 1.0:
            raise InvariantViolationError("LGD must lie in [0, 1]", lgd=self.loss_given_default)
        object.__setattr__(self, "scores", tuple(self.scores))

    @property
    def mean_pd(self) -> float:
        """Unweighted mean PD across the portfolio.

        Returns:
            The mean PD, or ``0.0`` when the portfolio is empty.
        """
        if not self.scores:
            return 0.0
        return sum(s.pd_value for s in self.scores) / len(self.scores)

    @property
    def flagged_count(self) -> int:
        """Number of obligors above their decision threshold.

        Returns:
            The flagged obligor count.
        """
        return sum(1 for s in self.scores if s.is_flagged)

    def band_distribution(self) -> dict[str, int]:
        """Count obligors per risk band.

        Returns:
            Band name mapped to obligor count, covering every band.
        """
        counts = {band.value: 0 for band in RiskBand}
        for score in self.scores:
            counts[score.band.value] += 1
        return counts
