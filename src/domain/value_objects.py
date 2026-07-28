"""Immutable value objects.

Value objects have no identity: two instances holding the same data are
interchangeable. Every one of them validates its invariants in ``__post_init__``
and raises :class:`~domain.exceptions.ValidationError` on breach, so an invalid
instance can never exist.

Standard library only — see :mod:`domain`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, StrEnum, unique
from itertools import pairwise
from typing import Annotated, ClassVar, Final, Literal, NewType, Self, TypeAlias

from domain.exceptions import CalculationError, ValidationError

__all__ = [
    "EPSILON",
    "CostMatrix",
    "DriftSeverity",
    "EntityId",
    "FeatureContract",
    "FieldSpec",
    "ModelId",
    "MonetaryAmount",
    "Percentage",
    "Probability",
    "ProbabilityOfDefault",
    "Ratio",
    "RiskBand",
    "ScalarDType",
    "ScenarioSeverity",
    "Sector",
    "SentimentScore",
]

# ---------------------------------------------------------------------------
# Type aliases and domain-wide constants
# ---------------------------------------------------------------------------
EntityId = NewType("EntityId", str)
ModelId = NewType("ModelId", str)

Probability: TypeAlias = Annotated[float, "closed unit interval [0, 1]"]
Percentage: TypeAlias = Annotated[float, "percentage points, e.g. -2.5 for -2.5%"]
ScalarDType: TypeAlias = Literal["float", "int", "str", "bool", "datetime", "category"]

#: Numerical guard used when moving between probability and log-odds space.
EPSILON: Final[float] = 1e-9


def _require(condition: bool, message: str, /, **context: object) -> None:
    """Raise a :class:`ValidationError` when ``condition`` is false.

    Args:
        condition: Invariant that must hold.
        message: Message attached to the raised error.
        **context: Structured detail attached to the raised error.

    Raises:
        ValidationError: If ``condition`` is false.
    """
    if not condition:
        raise ValidationError(message, **context)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
@unique
class Sector(StrEnum):
    """Coarse GICS-like sector taxonomy used for segmentation and fairness."""

    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    TECHNOLOGY = "technology"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"

    @classmethod
    def parse(cls, raw: str) -> Sector:
        """Parse a sector from a loosely formatted string.

        Args:
            raw: Sector name in any case, using spaces or hyphens as separators.

        Returns:
            The matching :class:`Sector` member.

        Raises:
            ValidationError: If ``raw`` matches no known sector.
        """
        normalised = raw.strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return cls(normalised)
        except ValueError as exc:
            raise ValidationError(
                "Unknown sector",
                value=raw,
                allowed=[s.value for s in cls],
            ) from exc


@unique
class RiskBand(StrEnum):
    """Internal rating buckets ordered from the safest to the riskiest."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

    @property
    def rank(self) -> int:
        """Ordinal position, ``0`` for the safest band.

        Returns:
            The zero-based index of the band in declaration order.
        """
        return list(RiskBand).index(self)

    @property
    def is_investment_grade(self) -> bool:
        """Whether the band is treated as investment grade.

        Returns:
            ``True`` for the two safest bands.
        """
        return self in (RiskBand.VERY_LOW, RiskBand.LOW)


@unique
class ScenarioSeverity(StrEnum):
    """CCAR / EBA stress-testing scenario severities."""

    BASELINE = "baseline"
    ADVERSE = "adverse"
    SEVERELY_ADVERSE = "severely_adverse"


@unique
class DriftSeverity(Enum):
    """Outcome of a population-stability comparison, ordered by seriousness."""

    NONE = 0
    MODERATE = 1
    SEVERE = 2

    def __lt__(self, other: DriftSeverity) -> bool:
        """Order severities by their underlying value.

        Args:
            other: Severity to compare against.

        Returns:
            ``True`` when this severity is less serious than ``other``.
        """
        if not isinstance(other, DriftSeverity):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: DriftSeverity) -> bool:
        """Order severities by their underlying value.

        Args:
            other: Severity to compare against.

        Returns:
            ``True`` when this severity is no more serious than ``other``.
        """
        if not isinstance(other, DriftSeverity):
            return NotImplemented
        return self.value <= other.value


# ---------------------------------------------------------------------------
# Financial primitives
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Ratio:
    """A named financial ratio constrained to a plausible range.

    Bounds are deliberately wide: they reject data-quality accidents (a leverage
    ratio of 10**6 from a unit error) without censoring genuine tail values.

    Attributes:
        name: Canonical feature name, for example ``debt_to_equity``.
        value: Observed value.
        lower_bound: Inclusive minimum considered plausible.
        upper_bound: Inclusive maximum considered plausible.
    """

    name: str
    value: float
    lower_bound: float = -1e6
    upper_bound: float = 1e6

    def __post_init__(self) -> None:
        """Validate the ratio invariants.

        Raises:
            ValidationError: If the name is blank, the value is not finite, the
                bounds are inverted, or the value falls outside the bounds.
        """
        _require(bool(self.name.strip()), "Ratio name must not be blank")
        _require(
            math.isfinite(self.value),
            "Ratio value must be finite",
            name=self.name,
            value=self.value,
        )
        _require(
            self.lower_bound <= self.upper_bound,
            "Ratio bounds are inverted",
            name=self.name,
            lower=self.lower_bound,
            upper=self.upper_bound,
        )
        _require(
            self.lower_bound <= self.value <= self.upper_bound,
            "Ratio value outside plausible bounds",
            name=self.name,
            value=self.value,
            lower=self.lower_bound,
            upper=self.upper_bound,
        )

    def shocked(self, *, pct_change: float) -> Ratio:
        """Return a copy moved by a relative shock.

        The result is clipped to the declared bounds so a large shock degrades
        gracefully rather than raising.

        Args:
            pct_change: Relative change, ``-0.2`` for a 20% reduction.

        Returns:
            A new :class:`Ratio` with the shocked, clipped value.
        """
        shocked_value = self.value * (1.0 + pct_change)
        clipped = min(max(shocked_value, self.lower_bound), self.upper_bound)
        return Ratio(self.name, clipped, self.lower_bound, self.upper_bound)

    def with_value(self, value: float) -> Ratio:
        """Return a copy carrying a different value and the same bounds.

        Args:
            value: Replacement value.

        Returns:
            A new :class:`Ratio`.
        """
        return Ratio(self.name, value, self.lower_bound, self.upper_bound)


@dataclass(frozen=True, slots=True)
class MonetaryAmount:
    """An amount of money in a single ISO-4217 currency.

    Attributes:
        amount: Numeric amount; may be negative for liabilities.
        currency: Upper-case three-letter ISO-4217 code.
    """

    amount: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        """Validate and normalise the currency code.

        Raises:
            ValidationError: If the amount is not finite or the currency code is
                not three alphabetic characters.
        """
        _require(math.isfinite(self.amount), "Amount must be finite", amount=self.amount)
        code = self.currency.strip().upper()
        _require(
            len(code) == 3 and code.isalpha(),
            "Currency must be a 3-letter ISO-4217 code",
            currency=self.currency,
        )
        object.__setattr__(self, "currency", code)

    def __add__(self, other: MonetaryAmount) -> MonetaryAmount:
        """Add two amounts of the same currency.

        Args:
            other: Amount to add.

        Returns:
            The summed amount.

        Raises:
            ValidationError: If the currencies differ.
        """
        _require(
            self.currency == other.currency,
            "Cannot add amounts in different currencies",
            left=self.currency,
            right=other.currency,
        )
        return MonetaryAmount(self.amount + other.amount, self.currency)

    def scaled(self, factor: float) -> MonetaryAmount:
        """Multiply the amount by a scalar.

        Args:
            factor: Multiplier.

        Returns:
            The scaled amount.
        """
        return MonetaryAmount(self.amount * factor, self.currency)


@dataclass(frozen=True, slots=True, order=True)
class ProbabilityOfDefault:
    """A probability of default over the model's stated horizon.

    Instances are ordered, so a collection of PDs sorts from safest to riskiest.

    Attributes:
        value: Probability in ``[0, 1]``.
        horizon_months: Forecast horizon the probability refers to.
    """

    value: Probability
    horizon_months: int = 12

    def __post_init__(self) -> None:
        """Validate the probability invariants.

        Raises:
            ValidationError: If the value is not a finite probability or the
                horizon is not positive.
        """
        _require(
            math.isfinite(self.value) and 0.0 <= self.value <= 1.0,
            "PD must lie in [0, 1]",
            value=self.value,
        )
        _require(
            self.horizon_months > 0,
            "PD horizon must be positive",
            horizon_months=self.horizon_months,
        )

    @property
    def log_odds(self) -> float:
        """Log-odds of the probability, clamped away from the asymptotes.

        Returns:
            ``log(p / (1 - p))`` evaluated on the clamped probability.
        """
        p = min(max(self.value, EPSILON), 1.0 - EPSILON)
        return math.log(p / (1.0 - p))

    @classmethod
    def from_log_odds(cls, log_odds: float, *, horizon_months: int = 12) -> Self:
        """Build a PD from a log-odds score.

        Args:
            log_odds: Score in log-odds space.
            horizon_months: Horizon carried onto the result.

        Returns:
            The corresponding :class:`ProbabilityOfDefault`.

        Raises:
            CalculationError: If ``log_odds`` is not finite.
        """
        if not math.isfinite(log_odds):
            raise CalculationError("Log-odds must be finite", log_odds=log_odds)
        # Overflow-safe logistic: exp() is only ever applied to a negative number.
        if log_odds >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-log_odds))
        else:
            exp_z = math.exp(log_odds)
            probability = exp_z / (1.0 + exp_z)
        return cls(probability, horizon_months)

    def credit_score(self, *, base: float = 600.0, pdo: float = 20.0) -> float:
        """Convert to a scorecard-style points score.

        Higher is safer, following the industry "points to double the odds"
        convention.

        Args:
            base: Score assigned at even odds.
            pdo: Points required to double the odds.

        Returns:
            The scaled score.
        """
        factor = pdo / math.log(2.0)
        return base - factor * self.log_odds

    def shocked(self, *, multiplier: float) -> ProbabilityOfDefault:
        """Apply a multiplicative stress in odds space.

        Scaling the *odds* rather than the probability keeps the result inside
        ``[0, 1]`` for any positive multiplier, which is what stress-testing
        transmission functions need.

        Args:
            multiplier: Odds multiplier; ``1.0`` leaves the PD unchanged.

        Returns:
            The stressed PD.

        Raises:
            CalculationError: If ``multiplier`` is not strictly positive.
        """
        if multiplier <= 0.0 or not math.isfinite(multiplier):
            raise CalculationError("PD shock multiplier must be positive", multiplier=multiplier)
        odds = math.exp(self.log_odds) * multiplier
        return ProbabilityOfDefault(odds / (1.0 + odds), self.horizon_months)

    def blended_with(self, other: ProbabilityOfDefault, *, weight: float = 0.5) -> Self:
        """Blend two PDs in log-odds space.

        Args:
            other: The other PD.
            weight: Weight applied to ``other``; ``0`` returns this PD unchanged.

        Returns:
            The blended PD, carrying this instance's horizon.

        Raises:
            CalculationError: If ``weight`` is outside ``[0, 1]`` or the horizons
                do not match.
        """
        if not 0.0 <= weight <= 1.0:
            raise CalculationError("Blend weight must lie in [0, 1]", weight=weight)
        if self.horizon_months != other.horizon_months:
            raise CalculationError(
                "Cannot blend PDs with different horizons",
                left=self.horizon_months,
                right=other.horizon_months,
            )
        combined = (1.0 - weight) * self.log_odds + weight * other.log_odds
        return type(self).from_log_odds(combined, horizon_months=self.horizon_months)

    def band(
        self,
        thresholds: tuple[float, float, float, float] = (0.01, 0.05, 0.15, 0.30),
    ) -> RiskBand:
        """Bucket the PD into a :class:`RiskBand`.

        Args:
            thresholds: Four ascending cut-points separating the five bands.

        Returns:
            The band the PD falls into.

        Raises:
            ValidationError: If the thresholds are not strictly ascending.
        """
        _require(
            all(a < b for a, b in pairwise(thresholds)),
            "Risk band thresholds must be strictly ascending",
            thresholds=thresholds,
        )
        bands = (RiskBand.VERY_LOW, RiskBand.LOW, RiskBand.MEDIUM, RiskBand.HIGH)
        for cut, band in zip(thresholds, bands, strict=True):
            if self.value < cut:
                return band
        return RiskBand.VERY_HIGH


@dataclass(frozen=True, slots=True)
class SentimentScore:
    """FinBERT-style sentiment over a document or a document set.

    Attributes:
        positive_prob: Probability mass on the positive class.
        negative_prob: Probability mass on the negative class.
        neutral_prob: Probability mass on the neutral class.
        n_documents: Number of documents the score aggregates.
    """

    positive_prob: Probability
    negative_prob: Probability
    neutral_prob: Probability
    n_documents: int = 1

    # ClassVar, not Final: a bare `Final` annotation would be collected as a
    # dataclass field and given a slot.
    _TOLERANCE: ClassVar[float] = 1e-6

    def __post_init__(self) -> None:
        """Validate that the class probabilities form a distribution.

        Raises:
            ValidationError: If any probability is outside ``[0, 1]``, the three
                do not sum to one, or the document count is negative.
        """
        for name, prob in (
            ("positive_prob", self.positive_prob),
            ("negative_prob", self.negative_prob),
            ("neutral_prob", self.neutral_prob),
        ):
            _require(
                math.isfinite(prob) and 0.0 <= prob <= 1.0,
                "Sentiment probability must lie in [0, 1]",
                field=name,
                value=prob,
            )
        total = self.positive_prob + self.negative_prob + self.neutral_prob
        _require(
            abs(total - 1.0) <= self._TOLERANCE,
            "Sentiment probabilities must sum to 1",
            total=total,
        )
        _require(
            self.n_documents >= 0,
            "Document count must not be negative",
            n_documents=self.n_documents,
        )

    @property
    def compound(self) -> float:
        """Net directional sentiment in ``[-1, 1]``.

        Returns:
            ``positive_prob - negative_prob``.
        """
        return self.positive_prob - self.negative_prob

    @property
    def uncertainty(self) -> float:
        """Normalised Shannon entropy of the class distribution.

        ``0`` means the classifier is fully confident in one class; ``1`` means
        it is maximally undecided. Used as a standalone model feature because
        disagreement in the news flow is itself a risk signal.

        Returns:
            Entropy scaled to ``[0, 1]``.
        """
        probs = (self.positive_prob, self.negative_prob, self.neutral_prob)
        entropy = -sum(p * math.log(p) for p in probs if p > EPSILON)
        # max(0.0, ...) collapses the -0.0 produced by a degenerate distribution.
        return max(0.0, min(entropy / math.log(len(probs)), 1.0))

    @property
    def is_negative(self) -> bool:
        """Whether negative is the modal class.

        Returns:
            ``True`` when ``negative_prob`` is the largest of the three.
        """
        return self.negative_prob > max(self.positive_prob, self.neutral_prob)

    @classmethod
    def neutral(cls) -> Self:
        """Build the fully neutral score used when no documents are available.

        Returns:
            A score with all mass on the neutral class and ``n_documents=0``.
        """
        return cls(positive_prob=0.0, negative_prob=0.0, neutral_prob=1.0, n_documents=0)


@dataclass(frozen=True, slots=True)
class CostMatrix:
    """Asymmetric misclassification costs.

    Missing a default (a false negative) writes off exposure, while a false
    positive only forgoes margin — so ``false_negative_cost`` dominates. The
    platform default encodes the 10:1 ratio mandated by the credit policy.

    Attributes:
        false_negative_cost: Cost of scoring a true defaulter as safe.
        false_positive_cost: Cost of scoring a healthy obligor as risky.
        true_positive_cost: Cost of a correctly flagged defaulter.
        true_negative_cost: Cost of a correctly cleared obligor.
    """

    false_negative_cost: float = 10.0
    false_positive_cost: float = 1.0
    true_positive_cost: float = 0.0
    true_negative_cost: float = 0.0

    def __post_init__(self) -> None:
        """Validate the cost invariants.

        Raises:
            ValidationError: If either error cost is not strictly positive, or
                any cost is not finite.
        """
        _require(
            self.false_negative_cost > 0.0 and self.false_positive_cost > 0.0,
            "Misclassification costs must be strictly positive",
            fn=self.false_negative_cost,
            fp=self.false_positive_cost,
        )
        for name, value in (
            ("true_positive_cost", self.true_positive_cost),
            ("true_negative_cost", self.true_negative_cost),
        ):
            _require(math.isfinite(value), "Cost must be finite", field=name, value=value)

    @property
    def imbalance_ratio(self) -> float:
        """How much worse a false negative is than a false positive.

        Returns:
            ``false_negative_cost / false_positive_cost``.
        """
        return self.false_negative_cost / self.false_positive_cost

    @property
    def positive_class_weight(self) -> float:
        """Weight to apply to positive-class rows during fitting.

        Returns:
            The cost imbalance ratio, usable directly as a ``sample_weight``
            multiplier or as XGBoost's ``scale_pos_weight``.
        """
        return self.imbalance_ratio

    def total_cost(self, *, tp: int, fp: int, tn: int, fn: int) -> float:
        """Total business cost of a confusion matrix.

        Args:
            tp: True positive count.
            fp: False positive count.
            tn: True negative count.
            fn: False negative count.

        Returns:
            The cost-weighted total.

        Raises:
            ValidationError: If any count is negative.
        """
        for name, count in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
            _require(
                count >= 0,
                "Confusion-matrix counts must not be negative",
                cell=name,
                value=count,
            )
        return (
            tp * self.true_positive_cost
            + fp * self.false_positive_cost
            + tn * self.true_negative_cost
            + fn * self.false_negative_cost
        )

    @property
    def break_even_threshold(self) -> float:
        """Decision threshold that equalises the expected cost of both actions.

        Returns:
            The cost-optimal cut-off under a well-calibrated model.
        """
        denominator = self.false_negative_cost + self.false_positive_cost
        return self.false_positive_cost / denominator


# ---------------------------------------------------------------------------
# Schema description — the single source of truth for data contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Declarative description of one column in the modelling dataset.

    The infrastructure layer compiles these into ``pandera`` schemas
    (:mod:`infrastructure.data.validators`), Pydantic API models
    (:mod:`infrastructure.api.schemas`) and dashboard slider ranges. Declaring
    the contract once in the domain keeps those three in lockstep — a field
    added here propagates everywhere without any duplicated literals.

    Attributes:
        name: Column name.
        dtype: Logical scalar type.
        description: Human-readable meaning, surfaced in API docs and the UI.
        nullable: Whether nulls are contractually permitted.
        ge: Inclusive lower bound, if any.
        le: Inclusive upper bound, if any.
        allowed: Closed set of permitted values, for categoricals.
        is_feature: Whether the column is fed to the model.
        unit: Presentation unit, for example ``"x"`` or ``"%"``.
        typical: Representative value used for UI defaults and imputation.
    """

    name: str
    dtype: ScalarDType
    description: str
    nullable: bool = False
    ge: float | None = None
    le: float | None = None
    allowed: tuple[str, ...] | None = None
    is_feature: bool = True
    unit: str = ""
    typical: float = 0.0

    def __post_init__(self) -> None:
        """Validate the field specification.

        Raises:
            ValidationError: If the name is blank, the bounds are inverted, or
                ``allowed`` is combined with a non-categorical dtype.
        """
        _require(bool(self.name.strip()), "Field name must not be blank")
        if self.ge is not None and self.le is not None:
            _require(
                self.ge <= self.le,
                "Field bounds are inverted",
                name=self.name,
                ge=self.ge,
                le=self.le,
            )
        if self.allowed is not None:
            _require(
                self.dtype in ("str", "category"),
                "Only string/category fields may declare an allowed set",
                name=self.name,
                dtype=self.dtype,
            )
            _require(len(self.allowed) > 0, "Allowed set must not be empty", name=self.name)

    def clamp(self, value: float) -> float:
        """Clip a value into the declared bounds.

        Args:
            value: Raw value.

        Returns:
            The value constrained to ``[ge, le]`` where those are declared.
        """
        if self.ge is not None:
            value = max(value, self.ge)
        if self.le is not None:
            value = min(value, self.le)
        return value

    def to_ratio(self, value: float) -> Ratio:
        """Build a :class:`Ratio` for this field.

        Args:
            value: Observed value.

        Returns:
            A validated :class:`Ratio` carrying this field's bounds.
        """
        return Ratio(
            name=self.name,
            value=value,
            lower_bound=self.ge if self.ge is not None else -1e6,
            upper_bound=self.le if self.le is not None else 1e6,
        )


@dataclass(frozen=True, slots=True)
class FeatureContract:
    """An ordered, queryable collection of :class:`FieldSpec` objects.

    Attributes:
        fields: The specifications, in canonical column order.
    """

    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate that field names are unique.

        Raises:
            ValidationError: If any name is declared more than once.
        """
        names = [f.name for f in self.fields]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        _require(not duplicates, "Duplicate field names in contract", duplicates=duplicates)

    def __iter__(self) -> Iterator[FieldSpec]:
        """Iterate over the field specifications.

        Returns:
            An iterator over :class:`FieldSpec` instances.
        """
        return iter(self.fields)

    def __len__(self) -> int:
        """Return the number of declared fields.

        Returns:
            The field count.
        """
        return len(self.fields)

    def get(self, name: str) -> FieldSpec:
        """Look up a field by name.

        Args:
            name: Column name.

        Returns:
            The matching specification.

        Raises:
            ValidationError: If no field carries that name.
        """
        for spec in self.fields:
            if spec.name == name:
                return spec
        raise ValidationError("Unknown field", name=name, known=[f.name for f in self.fields])

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of every column fed to the model, in canonical order.

        Returns:
            The model input column names.
        """
        return tuple(f.name for f in self.fields if f.is_feature)

    @property
    def numeric_feature_names(self) -> tuple[str, ...]:
        """Names of the numeric model features.

        Returns:
            Feature names whose dtype is ``float`` or ``int``.
        """
        return tuple(f.name for f in self.fields if f.is_feature and f.dtype in ("float", "int"))

    @property
    def categorical_feature_names(self) -> tuple[str, ...]:
        """Names of the categorical model features.

        Returns:
            Feature names whose dtype is ``str`` or ``category``.
        """
        return tuple(f.name for f in self.fields if f.is_feature and f.dtype in ("str", "category"))

    def subset(self, names: tuple[str, ...]) -> FeatureContract:
        """Restrict the contract to the named fields.

        Args:
            names: Field names to keep.

        Returns:
            A new contract containing only those fields, in this contract's order.
        """
        keep = set(names)
        return FeatureContract(tuple(f for f in self.fields if f.name in keep))
