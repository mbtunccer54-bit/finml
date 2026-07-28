"""Domain services.

Logic that belongs to the business but not to any single entity lives here:
turning model output into a regulatory PD, choosing a cost-optimal decision
threshold, and transmitting a macroeconomic shock into obligor-level default
risk.

Every service is stateless and side-effect free, and operates on plain builtins
or domain types. That is what makes them unit-testable without fixtures and
reviewable by a model-risk function that does not read ``pandas``.

Standard library only — see :mod:`domain`.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from domain.entities import (
    MacroScenario,
    PortfolioExposure,
    RiskScore,
    ScenarioResult,
)
from domain.exceptions import CalculationError, ScenarioError
from domain.value_objects import (
    EPSILON,
    CostMatrix,
    MonetaryAmount,
    ProbabilityOfDefault,
    RiskBand,
    Sector,
)

__all__ = [
    "DEFAULT_SECTOR_SENSITIVITIES",
    "AttributionStabilityAssessor",
    "CostSensitiveThresholdOptimizer",
    "MacroTransmissionService",
    "PDCalculator",
    "PortfolioRiskAggregator",
    "RiskBandClassifier",
    "SectorSensitivity",
    "StabilityVerdict",
    "ThresholdDecision",
]

#: Basel-style regulatory PD floor (3 basis points).
REGULATORY_PD_FLOOR: Final[float] = 0.0003


# ---------------------------------------------------------------------------
# PD construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PDCalculator:
    """Turns raw model output into a reportable probability of default.

    Three adjustments are applied in a fixed order, each of which a model-risk
    reviewer can inspect independently:

    1. an expert overlay expressed as a shift in log-odds space,
    2. a regulatory floor, and
    3. a conservatism cap.

    Working in log-odds keeps the overlay symmetric: the same overlay moves a
    0.01 PD and a 0.5 PD by the same amount of evidence, rather than the same
    number of percentage points.

    Attributes:
        floor: Minimum reportable PD.
        cap: Maximum reportable PD.
        horizon_months: Horizon stamped onto the produced PDs.
    """

    floor: float = REGULATORY_PD_FLOOR
    cap: float = 1.0
    horizon_months: int = 12

    def __post_init__(self) -> None:
        """Validate the floor and cap.

        Raises:
            CalculationError: If the bounds are not a valid sub-interval of
                ``[0, 1]``.
        """
        if not 0.0 <= self.floor < self.cap <= 1.0:
            raise CalculationError(
                "PD floor/cap must satisfy 0 <= floor < cap <= 1",
                floor=self.floor,
                cap=self.cap,
            )

    def from_probability(
        self, probability: float, *, overlay_log_odds: float = 0.0
    ) -> ProbabilityOfDefault:
        """Build a reportable PD from a model probability.

        Args:
            probability: Raw model output in ``[0, 1]``.
            overlay_log_odds: Expert judgement shift; positive is more conservative.

        Returns:
            The floored, capped and overlaid PD.

        Raises:
            CalculationError: If ``probability`` is not a finite value in ``[0, 1]``.
        """
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise CalculationError("Model probability must lie in [0, 1]", value=probability)
        clamped = min(max(probability, EPSILON), 1.0 - EPSILON)
        log_odds = math.log(clamped / (1.0 - clamped)) + overlay_log_odds
        raw = ProbabilityOfDefault.from_log_odds(log_odds, horizon_months=self.horizon_months)
        bounded = min(max(raw.value, self.floor), self.cap)
        return ProbabilityOfDefault(bounded, self.horizon_months)

    def blend(
        self,
        probabilities: Sequence[ProbabilityOfDefault],
        weights: Sequence[float] | None = None,
    ) -> ProbabilityOfDefault:
        """Combine several model PDs into one.

        Averaging in log-odds space rather than probability space is what keeps
        an ensemble of well-calibrated models calibrated.

        Args:
            probabilities: The PDs to combine.
            weights: Relative weights; uniform when omitted.

        Returns:
            The blended PD, floored and capped.

        Raises:
            CalculationError: If the input is empty, the lengths disagree, or
                the weights do not sum to a positive number.
        """
        if not probabilities:
            raise CalculationError("Cannot blend an empty set of PDs")
        effective = [1.0] * len(probabilities) if weights is None else list(weights)
        if len(effective) != len(probabilities):
            raise CalculationError(
                "Weight count must match PD count",
                n_pds=len(probabilities),
                n_weights=len(effective),
            )
        if any(w < 0.0 for w in effective):
            raise CalculationError("Blend weights must not be negative", weights=effective)
        total = sum(effective)
        if total <= 0.0:
            raise CalculationError("Blend weights must sum to a positive number", total=total)
        combined = (
            sum(w * p.log_odds for w, p in zip(effective, probabilities, strict=True)) / total
        )
        raw = ProbabilityOfDefault.from_log_odds(combined, horizon_months=self.horizon_months)
        return ProbabilityOfDefault(min(max(raw.value, self.floor), self.cap), self.horizon_months)


@dataclass(frozen=True, slots=True)
class RiskBandClassifier:
    """Maps PDs onto internal rating bands.

    Attributes:
        thresholds: Four ascending PD cut-points separating the five bands.
    """

    thresholds: tuple[float, float, float, float] = (0.01, 0.05, 0.15, 0.30)

    def classify(self, pd_value: ProbabilityOfDefault) -> RiskBand:
        """Assign a band to one PD.

        Args:
            pd_value: The PD to classify.

        Returns:
            The corresponding band.
        """
        return pd_value.band(self.thresholds)

    def distribution(self, pds: Iterable[ProbabilityOfDefault]) -> dict[str, int]:
        """Count PDs per band.

        Args:
            pds: The PDs to bucket.

        Returns:
            Band name mapped to count, covering every band.
        """
        counts = {band.value: 0 for band in RiskBand}
        for pd_value in pds:
            counts[self.classify(pd_value).value] += 1
        return counts


# ---------------------------------------------------------------------------
# Cost-sensitive decisioning
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """The outcome of a cost-optimal threshold search.

    Attributes:
        threshold: Selected decision cut-off.
        expected_cost: Total business cost at that cut-off.
        true_positives: Correctly flagged defaulters.
        false_positives: Healthy obligors flagged.
        true_negatives: Correctly cleared obligors.
        false_negatives: Missed defaulters.
    """

    threshold: float
    expected_cost: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def recall(self) -> float:
        """Share of true defaulters caught.

        Returns:
            Recall, or ``0.0`` when there are no positives.
        """
        actual_positives = self.true_positives + self.false_negatives
        return self.true_positives / actual_positives if actual_positives else 0.0

    @property
    def precision(self) -> float:
        """Share of flagged obligors that truly defaulted.

        Returns:
            Precision, or ``0.0`` when nothing was flagged.
        """
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def cost_per_sample(self) -> float:
        """Average business cost per scored obligor.

        Returns:
            Mean cost, or ``0.0`` when nothing was scored.
        """
        n = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        return self.expected_cost / n if n else 0.0


@dataclass(frozen=True, slots=True)
class CostSensitiveThresholdOptimizer:
    """Chooses the decision threshold that minimises expected business cost.

    The 0.5 default cut-off is only optimal when errors are symmetric. With a
    10:1 false-negative penalty the optimum sits far lower, and picking it
    explicitly is what turns a ranking model into a lending decision.

    Attributes:
        cost_matrix: The asymmetric cost structure to minimise against.
        n_candidates: Size of the fallback threshold grid.
    """

    cost_matrix: CostMatrix = field(default_factory=CostMatrix)
    n_candidates: int = 200

    def optimise(self, y_true: Sequence[int], y_prob: Sequence[float]) -> ThresholdDecision:
        """Search for the cost-minimising threshold.

        Candidate thresholds are the observed probabilities themselves when the
        sample is small, and a uniform grid otherwise — so the search is exact
        on small validation sets without becoming quadratic on large ones.

        Args:
            y_true: Binary outcomes.
            y_prob: Predicted probabilities.

        Returns:
            The best threshold and its confusion matrix.

        Raises:
            CalculationError: If the inputs are empty, differ in length, or
                contain values outside their domains.
        """
        if len(y_true) != len(y_prob):
            raise CalculationError(
                "Label and probability lengths differ",
                n_true=len(y_true),
                n_prob=len(y_prob),
            )
        if not y_true:
            raise CalculationError("Cannot optimise a threshold on an empty sample")
        if any(label not in (0, 1) for label in y_true):
            raise CalculationError("Labels must be binary (0 or 1)")
        if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in y_prob):
            raise CalculationError("Probabilities must be finite values in [0, 1]")

        candidates = self._candidate_thresholds(y_prob)
        best: ThresholdDecision | None = None
        for threshold in candidates:
            decision = self.evaluate(y_true, y_prob, threshold=threshold)
            if best is None or decision.expected_cost < best.expected_cost:
                best = decision
        if best is None:  # pragma: no cover - candidates is never empty
            raise CalculationError("Threshold search produced no candidates")
        return best

    def _candidate_thresholds(self, y_prob: Sequence[float]) -> list[float]:
        """Build the threshold grid to search.

        Args:
            y_prob: Predicted probabilities.

        Returns:
            Candidate thresholds strictly inside ``(0, 1)``.
        """
        if len(y_prob) <= self.n_candidates:
            observed = sorted({round(p, 6) for p in y_prob})
        else:
            step = 1.0 / (self.n_candidates + 1)
            observed = [step * (i + 1) for i in range(self.n_candidates)]
        inside = [t for t in observed if 0.0 < t < 1.0]
        return inside or [0.5]

    def evaluate(
        self, y_true: Sequence[int], y_prob: Sequence[float], *, threshold: float
    ) -> ThresholdDecision:
        """Score one threshold against the cost matrix.

        Args:
            y_true: Binary outcomes.
            y_prob: Predicted probabilities.
            threshold: Cut-off to evaluate.

        Returns:
            The confusion matrix and total cost at that cut-off.
        """
        tp = fp = tn = fn = 0
        for label, prob in zip(y_true, y_prob, strict=True):
            predicted_positive = prob >= threshold
            if label == 1 and predicted_positive:
                tp += 1
            elif label == 0 and predicted_positive:
                fp += 1
            elif label == 0:
                tn += 1
            else:
                fn += 1
        cost = self.cost_matrix.total_cost(tp=tp, fp=fp, tn=tn, fn=fn)
        return ThresholdDecision(
            threshold=threshold,
            expected_cost=cost,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )


# ---------------------------------------------------------------------------
# Macro to micro transmission
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SectorSensitivity:
    """How one sector transmits macro shocks into default risk.

    The bridge is deliberately two-stage so each leg can be challenged
    separately during model validation:

    * **Real-economy leg** — macro deltas move sector revenue, via the
      ``*_revenue_beta`` elasticities.
    * **Credit leg** — the revenue shock plus the direct financing channel
      (policy rate, credit spread) moves default odds.

    Attributes:
        sector: The sector described.
        gdp_revenue_beta: Revenue elasticity to a unit change in GDP growth.
        unemployment_revenue_beta: Revenue elasticity to unemployment.
        inflation_revenue_beta: Revenue elasticity to inflation.
        rate_log_odds_beta: Direct log-odds sensitivity to the policy rate.
        spread_log_odds_beta: Direct log-odds sensitivity to credit spreads.
    """

    sector: Sector
    gdp_revenue_beta: float
    unemployment_revenue_beta: float
    inflation_revenue_beta: float
    rate_log_odds_beta: float
    spread_log_odds_beta: float


def _default_sensitivities() -> dict[Sector, SectorSensitivity]:
    """Build the shipped sector sensitivity table.

    Values follow the conventional cyclicality ordering: discretionary demand
    and real estate are the most GDP-elastic, staples and utilities the least;
    financials and real estate carry the largest direct rate sensitivity.

    Returns:
        Sector mapped to its sensitivity parameters.
    """
    rows = (
        # sector, gdp_rev, unemp_rev, infl_rev, rate_lo, spread_lo
        (Sector.CONSUMER_DISCRETIONARY, 2.20, -1.10, -0.60, 3.0, 8.0),
        (Sector.REAL_ESTATE, 2.00, -0.80, -0.40, 6.5, 10.0),
        (Sector.ENERGY, 1.90, -0.40, 0.30, 2.5, 9.0),
        (Sector.INDUSTRIALS, 1.80, -0.70, -0.30, 3.0, 8.0),
        (Sector.MATERIALS, 1.80, -0.50, 0.10, 2.8, 8.5),
        (Sector.TECHNOLOGY, 1.60, -0.60, -0.35, 4.0, 7.0),
        (Sector.FINANCIALS, 1.50, -0.90, -0.20, 5.5, 11.0),
        (Sector.HEALTHCARE, 0.80, -0.20, -0.25, 2.0, 6.0),
        (Sector.CONSUMER_STAPLES, 0.60, -0.15, -0.45, 1.8, 5.5),
        (Sector.UTILITIES, 0.50, -0.10, -0.30, 4.5, 6.5),
    )
    return {
        sector: SectorSensitivity(
            sector=sector,
            gdp_revenue_beta=gdp,
            unemployment_revenue_beta=unemp,
            inflation_revenue_beta=infl,
            rate_log_odds_beta=rate,
            spread_log_odds_beta=spread,
        )
        for sector, gdp, unemp, infl, rate, spread in rows
    }


#: Sector sensitivity table used when configuration supplies no override.
DEFAULT_SECTOR_SENSITIVITIES: Final[dict[Sector, SectorSensitivity]] = _default_sensitivities()


@dataclass(frozen=True, slots=True)
class MacroTransmissionService:
    """Translates a macro scenario into obligor-level PD uplift.

    Attributes:
        sensitivities: Per-sector bridge parameters.
        revenue_to_log_odds: How strongly a relative revenue shock moves default
            odds. A ``-10%`` revenue shock at the default value of ``8.0`` lifts
            log-odds by ``0.8``, roughly doubling the odds of default.
        max_log_odds_shift: Cap on the total shift, so an extreme scenario
            saturates rather than producing a PD of 1.0.
    """

    sensitivities: Mapping[Sector, SectorSensitivity] = field(
        default_factory=lambda: dict(DEFAULT_SECTOR_SENSITIVITIES)
    )
    revenue_to_log_odds: float = 8.0
    max_log_odds_shift: float = 3.0

    def __post_init__(self) -> None:
        """Validate the transmission parameters.

        Raises:
            ScenarioError: If either coefficient is not strictly positive.
        """
        if self.revenue_to_log_odds <= 0.0:
            raise ScenarioError(
                "revenue_to_log_odds must be positive", value=self.revenue_to_log_odds
            )
        if self.max_log_odds_shift <= 0.0:
            raise ScenarioError(
                "max_log_odds_shift must be positive", value=self.max_log_odds_shift
            )

    def _sensitivity(self, sector: Sector) -> SectorSensitivity:
        """Look up a sector's bridge parameters.

        Args:
            sector: Sector to look up.

        Returns:
            The configured sensitivity.

        Raises:
            ScenarioError: If the sector has no configured sensitivity.
        """
        try:
            return self.sensitivities[sector]
        except KeyError as exc:
            raise ScenarioError(
                "No transmission sensitivity configured for sector",
                sector=sector.value,
                configured=[s.value for s in self.sensitivities],
            ) from exc

    def revenue_shock(self, scenario: MacroScenario, sector: Sector) -> float:
        """Stage one: relative revenue impact of the scenario on a sector.

        Args:
            scenario: The macro scenario.
            sector: Sector being shocked.

        Returns:
            Relative revenue change, ``-0.08`` for an 8% contraction.
        """
        s = self._sensitivity(sector)
        return (
            s.gdp_revenue_beta * scenario.shock_for("gdp_growth")
            + s.unemployment_revenue_beta * scenario.shock_for("unemployment_rate")
            + s.inflation_revenue_beta * scenario.shock_for("inflation_rate")
        )

    def log_odds_shift(self, scenario: MacroScenario, sector: Sector) -> float:
        """Stage two: total shift in default log-odds.

        Args:
            scenario: The macro scenario.
            sector: Sector being shocked.

        Returns:
            The capped log-odds shift; positive means higher default risk.
        """
        s = self._sensitivity(sector)
        real_economy = -self.revenue_to_log_odds * self.revenue_shock(scenario, sector)
        financing = s.rate_log_odds_beta * scenario.shock_for(
            "interest_rate"
        ) + s.spread_log_odds_beta * scenario.shock_for("credit_spread")
        total = real_economy + financing
        return max(-self.max_log_odds_shift, min(total, self.max_log_odds_shift))

    def odds_multiplier(self, scenario: MacroScenario, sector: Sector) -> float:
        """Multiplicative stress applied to default odds.

        Args:
            scenario: The macro scenario.
            sector: Sector being shocked.

        Returns:
            The odds multiplier; ``1.0`` means no impact.
        """
        return math.exp(self.log_odds_shift(scenario, sector))

    def transmit(
        self, base_pd: ProbabilityOfDefault, scenario: MacroScenario, sector: Sector
    ) -> ProbabilityOfDefault:
        """Apply a scenario to one obligor's PD.

        Args:
            base_pd: Unstressed PD.
            scenario: The macro scenario.
            sector: The obligor's sector.

        Returns:
            The stressed PD.
        """
        return base_pd.shocked(multiplier=self.odds_multiplier(scenario, sector))

    def transmit_portfolio(
        self,
        base_pds: Mapping[Sector, Sequence[ProbabilityOfDefault]],
        scenario: MacroScenario,
    ) -> ScenarioResult:
        """Apply a scenario across a sector-partitioned portfolio.

        Args:
            base_pds: Unstressed PDs grouped by sector.
            scenario: The macro scenario.

        Returns:
            Baseline and stressed mean PDs, plus the stressed mean per sector.
        """
        all_base: list[float] = []
        all_stressed: list[float] = []
        by_sector: dict[str, float] = {}

        for sector, pds in base_pds.items():
            if not pds:
                continue
            stressed = [self.transmit(p, scenario, sector).value for p in pds]
            by_sector[sector.value] = statistics.fmean(stressed)
            all_base.extend(p.value for p in pds)
            all_stressed.extend(stressed)

        return ScenarioResult(
            scenario=scenario,
            baseline_pd=statistics.fmean(all_base) if all_base else 0.0,
            stressed_pd=statistics.fmean(all_stressed) if all_stressed else 0.0,
            pd_by_sector=by_sector,
            n_entities=len(all_base),
        )


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PortfolioRiskAggregator:
    """Rolls obligor-level scores up to portfolio risk measures.

    Attributes:
        loss_given_default: Fraction of exposure lost on default.
        currency: Reporting currency for monetary outputs.
    """

    loss_given_default: float = 0.45
    currency: str = "USD"

    def __post_init__(self) -> None:
        """Validate the LGD assumption.

        Raises:
            CalculationError: If LGD is outside ``[0, 1]``.
        """
        if not 0.0 <= self.loss_given_default <= 1.0:
            raise CalculationError("LGD must lie in [0, 1]", lgd=self.loss_given_default)

    def expected_loss(
        self, scores: Sequence[RiskScore], exposures: Mapping[str, float]
    ) -> MonetaryAmount:
        """Compute portfolio expected loss.

        Obligors with no recorded exposure are skipped rather than defaulted to
        zero, so a missing exposure surfaces as a smaller portfolio rather than
        silently understating risk on a full one.

        Args:
            scores: Scored obligors.
            exposures: Exposure at default keyed by obligor identifier.

        Returns:
            ``sum(PD * LGD * EAD)`` in the reporting currency.
        """
        total = 0.0
        for score in scores:
            ead = exposures.get(str(score.entity_id))
            if ead is None:
                continue
            total += score.pd_value * self.loss_given_default * ead
        return MonetaryAmount(total, self.currency)

    def concentration_index(self, exposures: Mapping[str, float]) -> float:
        """Herfindahl-Hirschman concentration of the exposure distribution.

        Args:
            exposures: Exposure at default keyed by obligor identifier.

        Returns:
            HHI in ``(0, 1]``: ``1/n`` for a perfectly diversified book and
            ``1.0`` when a single name carries everything. ``0.0`` if empty.
        """
        values = [v for v in exposures.values() if v > 0.0]
        total = sum(values)
        if total <= 0.0:
            return 0.0
        return sum((v / total) ** 2 for v in values)

    def summarise(
        self, portfolio: PortfolioExposure, exposures: Mapping[str, float]
    ) -> dict[str, float | int | dict[str, int]]:
        """Produce the headline portfolio metrics.

        Args:
            portfolio: The scored portfolio.
            exposures: Exposure at default keyed by obligor identifier.

        Returns:
            A mapping of summary metrics for reporting.
        """
        el = self.expected_loss(portfolio.scores, exposures)
        return {
            "n_entities": len(portfolio.scores),
            "mean_pd": portfolio.mean_pd,
            "flagged_count": portfolio.flagged_count,
            "expected_loss": el.amount,
            "concentration_hhi": self.concentration_index(exposures),
            "band_distribution": portfolio.band_distribution(),
        }


# ---------------------------------------------------------------------------
# Explanation stability
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StabilityVerdict:
    """Result of an attribution stability assessment.

    Attributes:
        is_stable: Whether every feature stayed within the tolerance.
        max_coefficient_of_variation: Worst observed relative dispersion.
        unstable_features: Features that breached the tolerance, worst first.
        per_feature_cv: Relative dispersion per feature.
    """

    is_stable: bool
    max_coefficient_of_variation: float
    unstable_features: tuple[str, ...] = field(default_factory=tuple)
    per_feature_cv: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise the per-feature mapping to a plain dict."""
        object.__setattr__(self, "per_feature_cv", dict(self.per_feature_cv))


@dataclass(frozen=True, slots=True)
class AttributionStabilityAssessor:
    """Checks whether explanations are reproducible across random seeds.

    SR 11-7 treats an explanation that changes materially between runs as
    unusable evidence. Dispersion is measured relative to the *overall*
    attribution scale rather than per-feature, so a feature whose mean
    attribution is near zero does not register as wildly unstable purely
    because its denominator is small.

    Attributes:
        tolerance: Largest acceptable scaled dispersion.
        min_runs: Minimum number of seeds required for a verdict.
    """

    tolerance: float = 0.25
    min_runs: int = 3

    def assess(self, attributions: Mapping[str, Sequence[float]]) -> StabilityVerdict:
        """Assess dispersion of per-feature attributions across runs.

        Args:
            attributions: Feature name mapped to its attribution in each run.

        Returns:
            The stability verdict.

        Raises:
            CalculationError: If no features were supplied, the runs disagree in
                length, or fewer than :attr:`min_runs` runs are present.
        """
        if not attributions:
            raise CalculationError("No attributions supplied")
        lengths = {len(v) for v in attributions.values()}
        if len(lengths) != 1:
            raise CalculationError(
                "Every feature must have the same number of runs", lengths=sorted(lengths)
            )
        n_runs = lengths.pop()
        if n_runs < self.min_runs:
            raise CalculationError(
                "Too few runs for a stability verdict", n_runs=n_runs, required=self.min_runs
            )

        # Scale by the mean absolute attribution across all features so that
        # near-zero features are not judged against a vanishing denominator.
        scale = statistics.fmean([abs(v) for values in attributions.values() for v in values])
        denominator = max(scale, EPSILON)

        per_feature: dict[str, float] = {}
        for feature, values in attributions.items():
            dispersion = statistics.stdev(values) if n_runs > 1 else 0.0
            per_feature[feature] = dispersion / denominator

        unstable = sorted(
            (f for f, cv in per_feature.items() if cv > self.tolerance),
            key=lambda f: per_feature[f],
            reverse=True,
        )
        worst = max(per_feature.values(), default=0.0)
        return StabilityVerdict(
            is_stable=not unstable,
            max_coefficient_of_variation=worst,
            unstable_features=tuple(unstable),
            per_feature_cv=per_feature,
        )
