"""Unit tests for the domain layer.

The domain has no external dependencies, so these tests need no fixtures, no
data and no models — which is the whole point of keeping it pure.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from domain.entities import (
    FEATURE_CONTRACT,
    CounterfactualExample,
    FinancialEntity,
    MacroScenario,
    ModelPerformance,
)
from domain.exceptions import (
    CalculationError,
    DomainError,
    InfrastructureError,
    ScenarioError,
    ValidationError,
)
from domain.services import (
    AttributionStabilityAssessor,
    CostSensitiveThresholdOptimizer,
    MacroTransmissionService,
    PDCalculator,
    PortfolioRiskAggregator,
    RiskBandClassifier,
)
from domain.value_objects import (
    CostMatrix,
    DriftSeverity,
    EntityId,
    FeatureContract,
    FieldSpec,
    ModelId,
    MonetaryAmount,
    ProbabilityOfDefault,
    Ratio,
    RiskBand,
    ScenarioSeverity,
    Sector,
    SentimentScore,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class TestExceptions:
    """The exception hierarchy and its structured context."""

    def test_hierarchy_separates_domain_from_infrastructure(self) -> None:
        """A domain error must not be catchable as an infrastructure error."""
        assert issubclass(ValidationError, DomainError)
        assert not issubclass(ValidationError, InfrastructureError)

    def test_context_is_preserved_and_serialisable(self) -> None:
        """Structured context survives onto the serialised form."""
        error = ValidationError("bad value", field="pd", value=1.5)
        assert error.context == {"field": "pd", "value": 1.5}
        assert error.to_dict()["error_type"] == "ValidationError"
        assert "field='pd'" in str(error)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
class TestProbabilityOfDefault:
    """PD arithmetic, which everything downstream depends on."""

    @pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
    def test_rejects_values_outside_the_unit_interval(self, value: float) -> None:
        """Only finite probabilities in [0, 1] are constructible."""
        with pytest.raises(ValidationError):
            ProbabilityOfDefault(value)

    @pytest.mark.parametrize("value", [1e-6, 0.01, 0.25, 0.5, 0.75, 0.99])
    def test_log_odds_round_trip(self, value: float) -> None:
        """Converting to log-odds and back is lossless."""
        pd_value = ProbabilityOfDefault(value)
        assert ProbabilityOfDefault.from_log_odds(pd_value.log_odds).value == pytest.approx(
            value, abs=1e-9
        )

    def test_from_log_odds_does_not_overflow(self) -> None:
        """Extreme scores saturate instead of raising OverflowError."""
        assert ProbabilityOfDefault.from_log_odds(-1000.0).value == pytest.approx(0.0)
        assert ProbabilityOfDefault.from_log_odds(1000.0).value == pytest.approx(1.0)

    def test_shock_stays_inside_the_unit_interval(self) -> None:
        """Scaling odds keeps the result a probability for any positive factor."""
        for multiplier in (0.01, 1.0, 5.0, 1000.0):
            shocked = ProbabilityOfDefault(0.4).shocked(multiplier=multiplier)
            assert 0.0 <= shocked.value <= 1.0

    def test_shock_of_one_is_the_identity(self) -> None:
        """A neutral shock leaves the PD unchanged."""
        assert ProbabilityOfDefault(0.37).shocked(multiplier=1.0).value == pytest.approx(0.37)

    def test_shock_rejects_non_positive_multiplier(self) -> None:
        """A non-positive odds multiplier is meaningless."""
        with pytest.raises(CalculationError):
            ProbabilityOfDefault(0.4).shocked(multiplier=0.0)

    def test_symmetric_blend_is_the_log_odds_midpoint(self) -> None:
        """Blending 0.2 and 0.8 in log-odds space lands exactly on 0.5."""
        blended = ProbabilityOfDefault(0.2).blended_with(ProbabilityOfDefault(0.8))
        assert blended.value == pytest.approx(0.5, abs=1e-9)

    def test_blend_rejects_mismatched_horizons(self) -> None:
        """Blending PDs over different horizons is not meaningful."""
        with pytest.raises(CalculationError):
            ProbabilityOfDefault(0.1, 12).blended_with(ProbabilityOfDefault(0.1, 24))

    def test_credit_score_decreases_as_risk_rises(self) -> None:
        """Higher PD must map to a lower score."""
        assert ProbabilityOfDefault(0.01).credit_score() > ProbabilityOfDefault(0.5).credit_score()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.005, RiskBand.VERY_LOW),
            (0.03, RiskBand.LOW),
            (0.10, RiskBand.MEDIUM),
            (0.20, RiskBand.HIGH),
            (0.60, RiskBand.VERY_HIGH),
        ],
    )
    def test_banding(self, value: float, expected: RiskBand) -> None:
        """PDs bucket into the expected rating bands."""
        assert ProbabilityOfDefault(value).band() is expected

    def test_band_rejects_unordered_thresholds(self) -> None:
        """Thresholds must be strictly ascending."""
        with pytest.raises(ValidationError):
            ProbabilityOfDefault(0.1).band((0.2, 0.1, 0.3, 0.4))

    def test_ordering(self) -> None:
        """PDs sort from safest to riskiest."""
        values = [ProbabilityOfDefault(v) for v in (0.9, 0.1, 0.5)]
        assert [p.value for p in sorted(values)] == [0.1, 0.5, 0.9]


class TestSentimentScore:
    """Sentiment as a probability distribution."""

    def test_requires_a_valid_distribution(self) -> None:
        """Class probabilities must sum to one."""
        with pytest.raises(ValidationError):
            SentimentScore(0.5, 0.5, 0.5)

    def test_compound_and_direction(self) -> None:
        """Compound is the signed difference and the modal class is reported."""
        score = SentimentScore(0.2, 0.6, 0.2)
        assert score.compound == pytest.approx(-0.4)
        assert score.is_negative

    def test_uncertainty_bounds(self) -> None:
        """Entropy is zero when certain and near one when undecided."""
        assert SentimentScore.neutral().uncertainty == pytest.approx(0.0)
        assert SentimentScore(1 / 3, 1 / 3, 1 / 3).uncertainty == pytest.approx(1.0, abs=1e-9)

    def test_no_negative_zero(self) -> None:
        """A degenerate distribution must not produce -0.0."""
        assert not math.copysign(1.0, SentimentScore.neutral().uncertainty) < 0


class TestCostMatrix:
    """The asymmetric cost structure driving every threshold decision."""

    def test_default_encodes_the_ten_to_one_policy(self) -> None:
        """Missing a default costs ten times a false alarm."""
        assert CostMatrix().imbalance_ratio == pytest.approx(10.0)

    def test_break_even_threshold_is_well_below_half(self) -> None:
        """Asymmetric costs move the optimal cut-off away from 0.5."""
        assert CostMatrix().break_even_threshold == pytest.approx(1.0 / 11.0)

    def test_total_cost_weights_false_negatives_heavily(self) -> None:
        """A false negative contributes ten units, a false positive one."""
        assert CostMatrix().total_cost(tp=0, fp=1, tn=0, fn=1) == pytest.approx(11.0)

    def test_rejects_non_positive_costs(self) -> None:
        """Error costs must be strictly positive."""
        with pytest.raises(ValidationError):
            CostMatrix(false_negative_cost=0.0)


class TestRatioAndMoney:
    """The remaining financial primitives."""

    def test_rejects_non_finite(self) -> None:
        """A NaN ratio is a data-quality failure, not a value."""
        with pytest.raises(ValidationError):
            Ratio("x", float("nan"))

    def test_shock_clips_to_bounds(self) -> None:
        """A large shock saturates at the declared bound."""
        assert Ratio("x", 5.0, 0.0, 10.0).shocked(pct_change=10.0).value == pytest.approx(10.0)

    def test_currency_must_match_for_addition(self) -> None:
        """Adding across currencies is rejected."""
        with pytest.raises(ValidationError):
            MonetaryAmount(1.0, "USD") + MonetaryAmount(1.0, "EUR")

    def test_currency_is_normalised(self) -> None:
        """Currency codes are upper-cased on construction."""
        assert MonetaryAmount(1.0, "usd").currency == "USD"


class TestFeatureContract:
    """The contract every schema and API model is derived from."""

    def test_rejects_duplicate_names(self) -> None:
        """A duplicated column name is a contract error."""
        with pytest.raises(ValidationError):
            FeatureContract((FieldSpec("a", "float", "A"), FieldSpec("a", "float", "B")))

    def test_partitions_numeric_and_categorical(self) -> None:
        """Feature names split cleanly by dtype."""
        assert "sector" in FEATURE_CONTRACT.categorical_feature_names
        assert "current_ratio" in FEATURE_CONTRACT.numeric_feature_names
        assert "default_flag" not in FEATURE_CONTRACT.feature_names

    def test_lookup_of_unknown_field_raises(self) -> None:
        """An unknown column name is rejected rather than defaulted."""
        with pytest.raises(ValidationError):
            FEATURE_CONTRACT.get("does_not_exist")

    def test_clamp_respects_bounds(self) -> None:
        """Clamping honours the declared range."""
        spec = FEATURE_CONTRACT.get("current_ratio")
        assert spec.clamp(1e9) == spec.le
        assert spec.clamp(-1e9) == spec.ge

    def test_every_feature_has_a_description(self) -> None:
        """Descriptions surface in the OpenAPI schema and the dashboard."""
        assert all(spec.description.strip() for spec in FEATURE_CONTRACT.fields)


class TestSector:
    """Sector parsing, which sits on the API boundary."""

    @pytest.mark.parametrize(
        "raw", ["consumer_discretionary", "Consumer Discretionary", "CONSUMER-DISCRETIONARY"]
    )
    def test_parses_loose_formatting(self, raw: str) -> None:
        """Case and separator variations resolve to the same member."""
        assert Sector.parse(raw) is Sector.CONSUMER_DISCRETIONARY

    def test_rejects_unknown(self) -> None:
        """An unknown sector is rejected with the permitted set attached."""
        with pytest.raises(ValidationError) as info:
            Sector.parse("crypto")
        assert "allowed" in info.value.context


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
class TestFinancialEntity:
    """Obligor observations."""

    def _entity(self, **kwargs: object) -> FinancialEntity:
        """Build a valid entity for mutation in tests.

        Args:
            **kwargs: Field overrides.

        Returns:
            The entity.
        """
        defaults = {
            "entity_id": EntityId("E1"),
            "name": "Acme",
            "sector": Sector.TECHNOLOGY,
            "as_of_date": date(2024, 1, 1),
            "ratios": {"current_ratio": 1.5},
            "macro": {"gdp_growth": 0.02},
        }
        defaults.update(kwargs)
        return FinancialEntity(**defaults)  # type: ignore[arg-type]

    def test_rejects_blank_identity(self) -> None:
        """A blank identifier is not a valid identity."""
        with pytest.raises(ValidationError):
            self._entity(entity_id=EntityId("  "))

    def test_rejects_malformed_country(self) -> None:
        """The country must be an ISO alpha-2 code."""
        with pytest.raises(ValidationError):
            self._entity(country="USA")

    def test_feature_vector_covers_the_contract(self) -> None:
        """Every model feature is present, defaulted where unobserved."""
        values = self._entity().feature_values()
        assert set(values) == set(FEATURE_CONTRACT.feature_names)
        assert values["current_ratio"] == 1.5
        assert values["debt_to_equity"] == FEATURE_CONTRACT.get("debt_to_equity").typical

    def test_overrides_do_not_mutate_the_original(self) -> None:
        """What-if analysis must leave the baseline observation intact."""
        entity = self._entity()
        adjusted = entity.with_overrides({"current_ratio": 3.0})
        assert entity.ratios["current_ratio"] == 1.5
        assert adjusted.ratios["current_ratio"] == 3.0

    def test_overrides_are_clamped_to_contract_bounds(self) -> None:
        """A slider cannot push a ratio outside its plausible range."""
        adjusted = self._entity().with_overrides({"current_ratio": 1e9})
        assert adjusted.ratios["current_ratio"] == FEATURE_CONTRACT.get("current_ratio").le

    def test_overrides_reject_unknown_features(self) -> None:
        """An unknown feature name is a caller error."""
        with pytest.raises(ValidationError):
            self._entity().with_overrides({"nonexistent": 1.0})

    def test_sentiment_expands_into_features(self) -> None:
        """A sentiment value object becomes flat model features."""
        entity = self._entity(sentiment=SentimentScore(0.1, 0.7, 0.2, n_documents=5))
        values = entity.feature_values()
        assert values["sentiment_negative_prob"] == pytest.approx(0.7)
        assert values["news_volume"] == pytest.approx(5.0)


class TestMacroScenario:
    """Scenario definitions."""

    def test_rejects_unknown_macro_variables(self) -> None:
        """A shock must name a variable the bridge understands."""
        with pytest.raises(ScenarioError):
            MacroScenario("bad", ScenarioSeverity.ADVERSE, {"house_prices": -0.1})

    def test_apply_shifts_only_named_variables(self) -> None:
        """Unshocked variables pass through unchanged."""
        scenario = MacroScenario("s", ScenarioSeverity.ADVERSE, {"gdp_growth": -0.04})
        shocked = scenario.apply_to({"gdp_growth": 0.02, "interest_rate": 0.03})
        assert shocked["gdp_growth"] == pytest.approx(-0.02)
        assert shocked["interest_rate"] == pytest.approx(0.03)

    def test_baseline_is_not_stress(self) -> None:
        """Only adverse scenarios count as stress."""
        assert not MacroScenario("b", ScenarioSeverity.BASELINE).is_stress


class TestModelPerformance:
    """Evaluation records and the deployment gate."""

    def _performance(self, **kwargs: float) -> ModelPerformance:
        """Build a performance record.

        Args:
            **kwargs: Metric overrides.

        Returns:
            The record.
        """
        defaults = {
            "roc_auc": 0.85,
            "pr_auc": 0.40,
            "f1": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "brier_score": 0.08,
            "expected_calibration_error": 0.03,
            "business_cost": 100.0,
            "inference_latency_ms": 5.0,
        }
        defaults.update(kwargs)
        return ModelPerformance(
            model_id=ModelId("m"),
            split="validation",
            **defaults,  # type: ignore[arg-type]
        )

    def test_gate_passes_when_every_criterion_holds(self) -> None:
        """A compliant model clears the gate."""
        assert self._performance().meets_sla(
            max_calibration_error=0.05, max_latency_ms=50.0, min_pr_auc=0.1
        )

    @pytest.mark.parametrize(
        "override",
        [{"expected_calibration_error": 0.2}, {"inference_latency_ms": 500.0}, {"pr_auc": 0.01}],
    )
    def test_gate_fails_on_any_single_breach(self, override: dict[str, float]) -> None:
        """Each gate is independently binding."""
        assert not self._performance(**override).meets_sla(
            max_calibration_error=0.05, max_latency_ms=50.0, min_pr_auc=0.1
        )


class TestCounterfactual:
    """Recourse examples."""

    def test_reports_reduction_and_sparsity(self) -> None:
        """A counterfactual reports how much it helps and how much must move."""
        example = CounterfactualExample(
            entity_id=EntityId("E1"),
            original_pd=0.6,
            counterfactual_pd=0.2,
            changes={"interest_coverage": (1.0, 4.0)},
        )
        assert example.pd_reduction == pytest.approx(0.4)
        assert example.n_changes == 1
        assert example.required_deltas()["interest_coverage"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------
class TestPDCalculator:
    """Turning model output into a reportable PD."""

    def test_applies_the_regulatory_floor(self) -> None:
        """A zero probability is floored, never reported as zero."""
        assert PDCalculator(floor=0.0003).from_probability(0.0).value == pytest.approx(0.0003)

    def test_applies_the_conservatism_cap(self) -> None:
        """A capped calculator never reports above its cap."""
        assert PDCalculator(cap=0.9).from_probability(1.0).value == pytest.approx(0.9)

    def test_overlay_moves_risk_in_the_right_direction(self) -> None:
        """A positive overlay is more conservative."""
        calculator = PDCalculator()
        assert (
            calculator.from_probability(0.1, overlay_log_odds=1.0).value
            > calculator.from_probability(0.1).value
        )

    def test_rejects_out_of_range_input(self) -> None:
        """Model output outside [0, 1] is a bug worth surfacing."""
        with pytest.raises(CalculationError):
            PDCalculator().from_probability(1.5)

    def test_blend_is_order_independent(self) -> None:
        """Blending is symmetric under equal weights."""
        calculator = PDCalculator()
        left = calculator.blend([ProbabilityOfDefault(0.1), ProbabilityOfDefault(0.4)])
        right = calculator.blend([ProbabilityOfDefault(0.4), ProbabilityOfDefault(0.1)])
        assert left.value == pytest.approx(right.value)

    def test_blend_rejects_empty_and_mismatched_weights(self) -> None:
        """Degenerate blend requests are rejected."""
        calculator = PDCalculator()
        with pytest.raises(CalculationError):
            calculator.blend([])
        with pytest.raises(CalculationError):
            calculator.blend([ProbabilityOfDefault(0.1)], [1.0, 2.0])

    def test_rejects_inverted_bounds(self) -> None:
        """A floor above the cap is a configuration error."""
        with pytest.raises(CalculationError):
            PDCalculator(floor=0.9, cap=0.1)


class TestCostSensitiveThreshold:
    """The threshold search, which is where the cost matrix earns its keep."""

    def test_finds_the_cost_minimising_cut_off(self) -> None:
        """A perfectly separable sample yields zero cost."""
        truth = [0, 0, 0, 1, 1]
        probability = [0.1, 0.2, 0.3, 0.8, 0.9]
        decision = CostSensitiveThresholdOptimizer().optimise(truth, probability)
        assert decision.expected_cost == pytest.approx(0.0)
        assert decision.recall == pytest.approx(1.0)

    def test_asymmetric_costs_lower_the_threshold(self) -> None:
        """A heavy false-negative penalty buys recall with false positives."""
        truth = [0] * 90 + [1] * 10
        probability = [0.05 + i * 0.002 for i in range(90)] + [0.3 + i * 0.05 for i in range(10)]
        symmetric = CostSensitiveThresholdOptimizer(
            cost_matrix=CostMatrix(false_negative_cost=1.0, false_positive_cost=1.0)
        ).optimise(truth, probability)
        asymmetric = CostSensitiveThresholdOptimizer(
            cost_matrix=CostMatrix(false_negative_cost=20.0, false_positive_cost=1.0)
        ).optimise(truth, probability)
        assert asymmetric.threshold <= symmetric.threshold
        assert asymmetric.recall >= symmetric.recall

    def test_rejects_malformed_input(self) -> None:
        """Length mismatches, empty samples and non-binary labels are errors."""
        optimiser = CostSensitiveThresholdOptimizer()
        with pytest.raises(CalculationError):
            optimiser.optimise([0, 1], [0.5])
        with pytest.raises(CalculationError):
            optimiser.optimise([], [])
        with pytest.raises(CalculationError):
            optimiser.optimise([0, 2], [0.1, 0.9])
        with pytest.raises(CalculationError):
            optimiser.optimise([0, 1], [0.1, 1.5])


class TestRiskBandClassifier:
    """Band assignment."""

    def test_distribution_covers_every_band(self) -> None:
        """The distribution reports all bands, including empty ones."""
        classifier = RiskBandClassifier()
        counts = classifier.distribution([ProbabilityOfDefault(0.5)])
        assert set(counts) == {b.value for b in RiskBand}
        assert counts[RiskBand.VERY_HIGH.value] == 1


class TestMacroTransmission:
    """The macro-to-micro bridge."""

    def _severe(self) -> MacroScenario:
        """Build a severe scenario.

        Returns:
            The scenario.
        """
        return MacroScenario(
            "severe",
            ScenarioSeverity.SEVERELY_ADVERSE,
            {
                "gdp_growth": -0.045,
                "unemployment_rate": 0.045,
                "interest_rate": 0.02,
                "credit_spread": 0.028,
            },
        )

    def test_baseline_is_neutral(self) -> None:
        """An unshocked scenario leaves PDs untouched."""
        service = MacroTransmissionService()
        baseline = MacroScenario("base", ScenarioSeverity.BASELINE)
        assert service.odds_multiplier(baseline, Sector.TECHNOLOGY) == pytest.approx(1.0)

    def test_downturn_raises_risk(self) -> None:
        """A recession must increase default odds."""
        service = MacroTransmissionService()
        assert service.odds_multiplier(self._severe(), Sector.TECHNOLOGY) > 1.0

    def test_cyclical_sectors_are_hit_harder_than_defensive_ones(self) -> None:
        """The bridge must differentiate by sector or it adds nothing."""
        service = MacroTransmissionService()
        scenario = self._severe()
        cyclical = service.odds_multiplier(scenario, Sector.CONSUMER_DISCRETIONARY)
        defensive = service.odds_multiplier(scenario, Sector.UTILITIES)
        assert cyclical > defensive

    def test_revenue_shock_is_negative_in_a_downturn(self) -> None:
        """The real-economy leg contracts revenue."""
        assert MacroTransmissionService().revenue_shock(self._severe(), Sector.INDUSTRIALS) < 0.0

    def test_shift_saturates_at_the_cap(self) -> None:
        """An extreme scenario saturates instead of driving PD to one."""
        service = MacroTransmissionService(max_log_odds_shift=1.5)
        extreme = MacroScenario("extreme", ScenarioSeverity.SEVERELY_ADVERSE, {"gdp_growth": -0.25})
        assert service.log_odds_shift(extreme, Sector.REAL_ESTATE) == pytest.approx(1.5)

    def test_transmitted_pd_stays_a_probability(self) -> None:
        """Transmission cannot produce an invalid PD."""
        service = MacroTransmissionService()
        stressed = service.transmit(ProbabilityOfDefault(0.5), self._severe(), Sector.ENERGY)
        assert 0.0 <= stressed.value <= 1.0

    def test_portfolio_transmission_reports_per_sector(self) -> None:
        """Portfolio results break down by sector."""
        service = MacroTransmissionService()
        result = service.transmit_portfolio(
            {
                Sector.UTILITIES: [ProbabilityOfDefault(0.05)],
                Sector.CONSUMER_DISCRETIONARY: [ProbabilityOfDefault(0.05)],
            },
            self._severe(),
        )
        assert result.stressed_pd > result.baseline_pd
        assert (
            result.pd_by_sector[Sector.CONSUMER_DISCRETIONARY.value]
            > result.pd_by_sector[Sector.UTILITIES.value]
        )

    def test_rejects_unconfigured_sector(self) -> None:
        """A sector with no sensitivity is an error, not a silent default."""
        service = MacroTransmissionService(sensitivities={})
        with pytest.raises(ScenarioError):
            service.odds_multiplier(self._severe(), Sector.ENERGY)

    def test_rejects_invalid_parameters(self) -> None:
        """Non-positive coefficients are configuration errors."""
        with pytest.raises(ScenarioError):
            MacroTransmissionService(revenue_to_log_odds=0.0)


class TestPortfolioAggregation:
    """Portfolio roll-up."""

    def test_expected_loss_is_pd_times_lgd_times_ead(self) -> None:
        """Expected loss follows the standard decomposition."""
        from datetime import UTC, datetime

        from domain.entities import RiskScore

        score = RiskScore(
            entity_id=EntityId("E1"),
            probability_of_default=ProbabilityOfDefault(0.1),
            model_id=ModelId("m"),
            scored_at=datetime.now(UTC),
        )
        aggregator = PortfolioRiskAggregator(loss_given_default=0.5)
        assert aggregator.expected_loss([score], {"E1": 1000.0}).amount == pytest.approx(50.0)

    def test_concentration_index_bounds(self) -> None:
        """HHI is 1/n when even and 1 when concentrated."""
        aggregator = PortfolioRiskAggregator()
        assert aggregator.concentration_index({"a": 1.0, "b": 1.0}) == pytest.approx(0.5)
        assert aggregator.concentration_index({"a": 1.0}) == pytest.approx(1.0)
        assert aggregator.concentration_index({}) == pytest.approx(0.0)

    def test_rejects_invalid_lgd(self) -> None:
        """LGD outside [0, 1] is a configuration error."""
        with pytest.raises(CalculationError):
            PortfolioRiskAggregator(loss_given_default=1.5)


class TestAttributionStability:
    """Explanation reproducibility."""

    def test_identical_runs_are_stable(self) -> None:
        """Zero dispersion is stable."""
        verdict = AttributionStabilityAssessor(min_runs=2).assess(
            {"a": [1.0, 1.0, 1.0], "b": [0.5, 0.5, 0.5]}
        )
        assert verdict.is_stable
        assert verdict.max_coefficient_of_variation == pytest.approx(0.0)

    def test_flags_a_wildly_varying_feature(self) -> None:
        """A feature swinging between runs is reported as unstable."""
        verdict = AttributionStabilityAssessor(tolerance=0.2, min_runs=2).assess(
            {"stable": [1.0, 1.01, 0.99], "erratic": [-2.0, 2.0, 0.0]}
        )
        assert not verdict.is_stable
        assert "erratic" in verdict.unstable_features

    def test_rejects_insufficient_runs(self) -> None:
        """A verdict needs enough runs to be meaningful."""
        with pytest.raises(CalculationError):
            AttributionStabilityAssessor(min_runs=3).assess({"a": [1.0, 1.0]})

    def test_rejects_ragged_input(self) -> None:
        """Every feature must have the same number of runs."""
        with pytest.raises(CalculationError):
            AttributionStabilityAssessor(min_runs=2).assess({"a": [1.0, 1.0], "b": [1.0]})


class TestDriftSeverityOrdering:
    """Severity comparison, used to pick the worst feature."""

    def test_orders_by_seriousness(self) -> None:
        """NONE is less serious than MODERATE, which is less than SEVERE."""
        assert DriftSeverity.NONE < DriftSeverity.MODERATE < DriftSeverity.SEVERE
        assert (
            max(
                [DriftSeverity.NONE, DriftSeverity.SEVERE, DriftSeverity.MODERATE],
                key=lambda s: s.value,
            )
            is DriftSeverity.SEVERE
        )
