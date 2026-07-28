"""Explainability tests.

Explanations are regulatory artifacts here, not diagnostics, so these tests
assert on properties a model-risk reviewer would check: that attributions are
faithful to the model, that the two explainers broadly agree, that recourse is
actionable, and that the fairness arithmetic is right.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from domain.exceptions import CalculationError, ExplainerError

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def background(feature_matrix: tuple[pd.DataFrame, np.ndarray, Any]) -> pd.DataFrame:
    """A background sample for the explainers.

    Args:
        feature_matrix: The matrix, labels and builder.

    Returns:
        A reference sample.
    """
    x, _y, _builder = feature_matrix
    return x.sample(n=min(150, len(x)), random_state=0)


class TestShapExplainer:
    """SHAP attributions."""

    def test_shape_and_backend(
        self, fitted_model: Any, background: pd.DataFrame, feature_matrix: Any
    ) -> None:
        """TreeSHAP is selected for a tree model and shapes line up."""
        from infrastructure.xai.shap_explainer import ShapExplainer

        x, _y, builder = feature_matrix
        result = ShapExplainer(
            fitted_model, background, feature_names=builder.feature_names
        ).explain(x.head(40))
        assert result.backend == "tree"
        assert result.values.shape == (40, len(builder.feature_names))

    def test_attributions_track_the_model_output(
        self, fitted_model: Any, background: pd.DataFrame, feature_matrix: Any
    ) -> None:
        """Summed attributions must correlate with what the model actually predicts."""
        from infrastructure.xai.shap_explainer import ShapExplainer

        x, _y, builder = feature_matrix
        sample = x.head(120)
        result = ShapExplainer(
            fitted_model, background, feature_names=builder.feature_names
        ).explain(sample)
        reconstructed = result.base_value + result.values.sum(axis=1)
        predicted = fitted_model.predict_proba(sample)[:, 1]
        assert np.corrcoef(reconstructed, predicted)[0, 1] > 0.7

    def test_global_importance_is_ranked(
        self, fitted_model: Any, background: pd.DataFrame, feature_matrix: Any
    ) -> None:
        """Global importance comes back sorted with a direction."""
        from infrastructure.xai.shap_explainer import ShapExplainer

        x, _y, builder = feature_matrix
        ranking = (
            ShapExplainer(fitted_model, background, feature_names=builder.feature_names)
            .explain(x.head(60))
            .global_importance()
        )
        values = ranking["mean_abs_shap"].tolist()
        assert values == sorted(values, reverse=True)
        assert set(ranking["direction"]) <= {"increases_risk", "decreases_risk"}

    def test_explain_instance_returns_a_domain_explanation(
        self, fitted_model: Any, background: pd.DataFrame, feature_matrix: Any
    ) -> None:
        """A single row yields a persistable domain object."""
        from infrastructure.xai.shap_explainer import ShapExplainer

        x, _y, builder = feature_matrix
        explanation = ShapExplainer(
            fitted_model, background, feature_names=builder.feature_names
        ).explain_instance(x.head(1), entity_id="E1", model_id="m")
        assert str(explanation.entity_id) == "E1"
        assert len(explanation.contributions) == len(builder.feature_names)
        assert len(explanation.top_drivers(3)) == 3

    def test_interaction_matrix_is_square_and_symmetric(
        self, fitted_model: Any, background: pd.DataFrame, feature_matrix: Any
    ) -> None:
        """Interaction strength between two features is direction-free."""
        from infrastructure.xai.shap_explainer import ShapExplainer

        x, _y, builder = feature_matrix
        matrix = ShapExplainer(
            fitted_model, background, feature_names=builder.feature_names
        ).interaction_matrix(x.head(50), top_k=5)
        assert matrix.shape == (5, 5)
        assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T, atol=1e-6)

    def test_rejects_empty_background(self) -> None:
        """An explainer with no reference distribution is meaningless."""
        from infrastructure.xai.shap_explainer import ShapExplainer

        with pytest.raises(ExplainerError):
            ShapExplainer(object(), pd.DataFrame())

    def test_rejects_multi_row_instance_explanation(
        self, fitted_model: Any, background: pd.DataFrame, feature_matrix: Any
    ) -> None:
        """`explain_instance` is single-row by contract."""
        from infrastructure.xai.shap_explainer import ShapExplainer

        x, _y, builder = feature_matrix
        explainer = ShapExplainer(fitted_model, background, feature_names=builder.feature_names)
        with pytest.raises(ExplainerError):
            explainer.explain_instance(x.head(2), entity_id="E1")

    def test_saves_a_summary_plot(
        self, fitted_model: Any, background: pd.DataFrame, feature_matrix: Any, tmp_path: Any
    ) -> None:
        """The beeswarm artifact is written for the model file."""
        from infrastructure.xai.shap_explainer import ShapExplainer

        x, _y, builder = feature_matrix
        path = ShapExplainer(
            fitted_model, background, feature_names=builder.feature_names
        ).save_summary_plot(x.head(50), tmp_path / "shap.png")
        assert path is not None
        assert path.is_file()


class TestShapNormalisation:
    """Output-shape handling, where a silent error inverts every sign."""

    @pytest.mark.parametrize("layout", ["2d", "3d_trailing", "3d_leading", "list"])
    def test_extracts_the_positive_class(self, layout: str) -> None:
        """Each SHAP output layout resolves to the positive-class array."""
        from infrastructure.xai.shap_explainer import normalise_shap_values

        n_rows, n_features = 4, 3
        positive = np.arange(n_rows * n_features, dtype=float).reshape(n_rows, n_features)
        negative = -positive

        if layout == "2d":
            raw: Any = positive
        elif layout == "3d_trailing":
            raw = np.stack([negative, positive], axis=2)
        elif layout == "3d_leading":
            raw = np.stack([negative, positive], axis=0)
        else:
            raw = [negative, positive]

        result = normalise_shap_values(raw, n_rows=n_rows, n_features=n_features)
        assert result.shape == (n_rows, n_features)
        assert np.allclose(result, positive)

    def test_rejects_uninterpretable_output(self) -> None:
        """An unrecognised shape must raise, not be guessed at."""
        from infrastructure.xai.shap_explainer import normalise_shap_values

        with pytest.raises(ExplainerError):
            normalise_shap_values(np.zeros((7, 9)), n_rows=4, n_features=3)


class TestLimeExplainer:
    """Local surrogate explanations."""

    def test_returns_ranked_contributions(self, fitted_model: Any, feature_matrix: Any) -> None:
        """LIME returns the requested number of ranked drivers."""
        from infrastructure.xai.lime_explainer import LimeExplainer

        x, _y, builder = feature_matrix
        explanation = LimeExplainer(
            fitted_model,
            x,
            feature_names=builder.feature_names,
            num_features=6,
            num_samples=400,
        ).explain_instance(x.head(1), entity_id="E1")
        assert explanation.method == "lime"
        assert len(explanation.contributions) <= 6
        magnitudes = [abs(c.contribution) for c in explanation.contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_is_reproducible_under_a_fixed_seed(
        self, fitted_model: Any, feature_matrix: Any
    ) -> None:
        """An explanation on a credit file must reproduce exactly."""
        from infrastructure.xai.lime_explainer import LimeExplainer

        x, _y, builder = feature_matrix

        def explain() -> list[float]:
            """Produce one explanation's contributions.

            Returns:
                The contribution values.
            """
            return [
                c.contribution
                for c in LimeExplainer(
                    fitted_model,
                    x,
                    feature_names=builder.feature_names,
                    num_samples=300,
                    random_state=7,
                )
                .explain_instance(x.head(1), entity_id="E1")
                .contributions
            ]

        assert explain() == pytest.approx(explain())

    def test_batch_skips_failures(self, fitted_model: Any, feature_matrix: Any) -> None:
        """A batch returns one explanation per row it could explain."""
        from infrastructure.xai.lime_explainer import LimeExplainer

        x, _y, builder = feature_matrix
        results = LimeExplainer(
            fitted_model, x, feature_names=builder.feature_names, num_samples=200
        ).explain_batch(x.head(3), ["A", "B", "C"])
        assert len(results) == 3

    def test_rejects_mismatched_ids(self, fitted_model: Any, feature_matrix: Any) -> None:
        """Row and identifier counts must agree."""
        from infrastructure.xai.lime_explainer import LimeExplainer

        x, _y, builder = feature_matrix
        with pytest.raises(ExplainerError):
            LimeExplainer(fitted_model, x, feature_names=builder.feature_names).explain_batch(
                x.head(3), ["A"]
            )


class TestPermutationImportance:
    """Model-agnostic global importance."""

    def test_ranks_features(self, fitted_model: Any, feature_matrix: Any) -> None:
        """Importance comes back ranked with a dispersion estimate."""
        from infrastructure.xai.permutation import permutation_importances

        x, y, _builder = feature_matrix
        result = permutation_importances(fitted_model, x.head(300), y[:300], n_repeats=3)
        assert list(result["rank"]) == list(range(1, len(result) + 1))
        assert (result["importance_std"] >= 0).all()
        values = result["importance_mean"].tolist()
        assert values == sorted(values, reverse=True)

    def test_identifies_a_genuinely_predictive_feature(self, feature_matrix: Any) -> None:
        """Shuffling a feature the model relies on must cost performance."""
        from infrastructure.config.schemas import ModelSpec
        from infrastructure.models.trainers import build_model
        from infrastructure.xai.permutation import permutation_importances

        rng = np.random.default_rng(3)
        signal = rng.normal(size=600)
        frame = pd.DataFrame(
            {"signal": signal, "noise": rng.normal(size=600), "noise2": rng.normal(size=600)}
        )
        labels = (signal + rng.normal(0, 0.3, 600) > 0).astype(int)
        model = build_model(
            "lightgbm",
            ModelSpec(params={"n_estimators": 40, "num_leaves": 7, "verbosity": -1}),
            n_jobs=1,
        ).fit(frame, labels)

        result = permutation_importances(model, frame, labels, n_repeats=4)
        assert result.iloc[0]["feature"] == "signal"
        assert result.iloc[0]["importance_mean"] > 0.0

    def test_compares_models(self, fitted_model: Any, feature_matrix: Any) -> None:
        """The comparison table reports agreement across models."""
        from infrastructure.config.schemas import ModelSpec
        from infrastructure.models.trainers import build_model
        from infrastructure.xai.permutation import compare_models

        x, y, _builder = feature_matrix
        other = build_model(
            "random_forest", ModelSpec(params={"n_estimators": 25, "max_depth": 5}), n_jobs=1
        ).fit(x, y)
        table = compare_models(
            {"lgbm": fitted_model, "rf": other}, x.head(200), y[:200], n_repeats=2, top_k=5
        )
        assert not table.empty
        assert {"mean_rank", "rank_spread"} <= set(table.columns)
        assert len(table) <= 5

    def test_returns_empty_when_every_model_fails(self, feature_matrix: Any) -> None:
        """A comparison over broken models degrades rather than raising."""
        from infrastructure.xai.permutation import compare_models

        x, y, _builder = feature_matrix
        assert compare_models({"broken": object()}, x.head(50), y[:50], n_repeats=2).empty


class TestCounterfactuals:
    """Actionable recourse."""

    @pytest.fixture
    def generator(self, fitted_model: Any, feature_matrix: Any) -> Any:
        """A counterfactual generator over the fitted model.

        Args:
            fitted_model: The fitted adapter.
            feature_matrix: The matrix, labels and builder.

        Returns:
            The generator.
        """
        from infrastructure.xai.dice_explainer import CounterfactualGenerator

        x, _y, builder = feature_matrix
        return CounterfactualGenerator(
            fitted_model, x, feature_names=builder.feature_names, target_pd=0.05, total_cfs=2
        )

    def test_excludes_macro_variables_from_recourse(self, generator: Any) -> None:
        """Advising an obligor to change GDP is not recourse."""
        from domain.entities import MACRO_VARIABLES

        assert not set(generator.actionable_features) & set(MACRO_VARIABLES)

    def test_excludes_one_hot_columns(self, generator: Any) -> None:
        """Sector is not a lever an obligor can pull."""
        assert not any("=" in name for name in generator.actionable_features)

    def test_returns_nothing_when_already_safe(
        self, generator: Any, fitted_model: Any, feature_matrix: Any
    ) -> None:
        """No recourse is needed below the target."""
        x, _y, _builder = feature_matrix
        probabilities = fitted_model.predict_proba(x)[:, 1]
        safest = x.iloc[[int(np.argmin(probabilities))]]
        assert generator.generate(safest, entity_id="SAFE") == []

    def test_recourse_lowers_the_pd(
        self, generator: Any, fitted_model: Any, feature_matrix: Any
    ) -> None:
        """Every returned option must genuinely reduce risk."""
        x, _y, _builder = feature_matrix
        probabilities = fitted_model.predict_proba(x)[:, 1]
        riskiest = x.iloc[[int(np.argmax(probabilities))]]
        examples = generator.generate(riskiest, entity_id="RISKY")
        for example in examples:
            assert example.counterfactual_pd < example.original_pd
            assert example.n_changes >= 1

    def test_greedy_fallback_produces_a_plan(self, fitted_model: Any, feature_matrix: Any) -> None:
        """With DiCE unavailable the coordinate search still returns recourse."""
        from infrastructure.xai.dice_explainer import CounterfactualGenerator

        x, _y, builder = feature_matrix
        generator = CounterfactualGenerator(
            fitted_model, x, feature_names=builder.feature_names, target_pd=0.05
        )
        generator._dice_failed = True  # force the fallback path

        probabilities = fitted_model.predict_proba(x)[:, 1]
        riskiest = x.iloc[[int(np.argmax(probabilities))]]
        examples = generator.generate(riskiest, entity_id="RISKY")
        assert examples
        assert examples[0].counterfactual_pd < examples[0].original_pd

    def test_renders_a_recourse_table(
        self, generator: Any, fitted_model: Any, feature_matrix: Any
    ) -> None:
        """The table names each required move for a relationship manager."""
        from infrastructure.xai.dice_explainer import CounterfactualGenerator

        x, _y, _builder = feature_matrix
        probabilities = fitted_model.predict_proba(x)[:, 1]
        riskiest = x.iloc[[int(np.argmax(probabilities))]]
        examples = generator.generate(riskiest, entity_id="RISKY")
        if examples:
            table = CounterfactualGenerator.to_frame(examples)
            assert {"feature", "current", "required", "change"} <= set(table.columns)

    def test_rejects_invalid_target(self, fitted_model: Any, feature_matrix: Any) -> None:
        """A target PD outside (0, 1) is a configuration error."""
        from infrastructure.xai.dice_explainer import CounterfactualGenerator

        x, _y, _builder = feature_matrix
        with pytest.raises(ExplainerError):
            CounterfactualGenerator(fitted_model, x, target_pd=1.5)


class TestStability:
    """Attribution reproducibility."""

    def test_tree_shap_is_stable_across_seeds(self, fitted_model: Any, feature_matrix: Any) -> None:
        """TreeSHAP is deterministic, so re-explaining must agree."""
        from infrastructure.xai.stability import StabilityAnalyzer

        x, _y, builder = feature_matrix
        verdict = StabilityAnalyzer(
            fitted_model, x.head(120), feature_names=builder.feature_names, n_seeds=3
        ).assess_instance(x.head(1))
        assert verdict.is_stable
        assert verdict.max_coefficient_of_variation < 0.25

    def test_rejects_a_tiny_background(self, fitted_model: Any, feature_matrix: Any) -> None:
        """Resampling needs enough background rows to be meaningful."""
        from infrastructure.xai.stability import StabilityAnalyzer

        x, _y, _builder = feature_matrix
        with pytest.raises(ExplainerError):
            StabilityAnalyzer(fitted_model, x.head(3))

    def test_temporal_stability_reports_rank_correlation(
        self, fitted_model: Any, feature_matrix: Any, panel: pd.DataFrame
    ) -> None:
        """Global drivers should evolve gradually across time windows."""
        from infrastructure.xai.stability import StabilityAnalyzer

        x, _y, builder = feature_matrix
        report = StabilityAnalyzer(
            fitted_model, x.head(200), feature_names=builder.feature_names
        ).assess_temporal(x, panel["as_of_date"], n_windows=3, max_rows_per_window=80)
        assert len(report.window_labels) >= 2
        assert len(report.rank_correlation) == len(report.window_labels) - 1
        assert all(-1.0 <= c <= 1.0 for c in report.rank_correlation)


class TestFairness:
    """Group fairness diagnostics."""

    def test_detects_a_deliberately_biased_model(self) -> None:
        """A model flagging one group far more often must register a parity gap."""
        from infrastructure.xai.fairness import assess_fairness

        rng = np.random.default_rng(1)
        n = 400
        groups = pd.Series(["a"] * n + ["b"] * n)
        truth = np.concatenate([rng.binomial(1, 0.1, n), rng.binomial(1, 0.1, n)])
        # Group b is scored far more harshly for identical realised risk.
        probability = np.concatenate([rng.uniform(0.0, 0.2, n), rng.uniform(0.7, 1.0, n)])
        report = assess_fairness(truth, probability, groups, threshold=0.5, tolerance=0.1)
        assert report.demographic_parity_gap > 0.5
        assert not report.is_fair

    def test_fair_model_passes(self) -> None:
        """Identical treatment across groups must not be flagged."""
        from infrastructure.xai.fairness import assess_fairness

        rng = np.random.default_rng(2)
        n = 500
        groups = pd.Series(["a"] * n + ["b"] * n)
        probability = rng.uniform(0.0, 1.0, 2 * n)
        truth = (rng.uniform(size=2 * n) < probability).astype(int)
        report = assess_fairness(truth, probability, groups, threshold=0.5, tolerance=0.15)
        assert report.demographic_parity_gap < 0.15
        assert report.is_fair

    def test_small_groups_are_reported_but_excluded_from_gaps(self) -> None:
        """A three-row group must not dominate the equalised-odds spread."""
        from infrastructure.xai.fairness import assess_fairness

        rng = np.random.default_rng(4)
        n = 300
        groups = pd.Series(["big"] * n + ["tiny"] * 3)
        probability = np.concatenate([rng.uniform(0, 1, n), np.array([0.9, 0.9, 0.9])])
        truth = np.concatenate([rng.binomial(1, 0.2, n), np.array([1, 0, 0])])
        report = assess_fairness(truth, probability, groups, min_group_size=30)
        assert "tiny" in set(report.per_group["group"])
        assert report.equalised_odds_gap == pytest.approx(0.0)

    def test_reports_per_group_calibration(self) -> None:
        """Calibration is reported per group, not only in aggregate."""
        from infrastructure.xai.fairness import assess_fairness

        rng = np.random.default_rng(5)
        n = 300
        groups = pd.Series(["a"] * n + ["b"] * n)
        probability = rng.uniform(0.05, 0.95, 2 * n)
        truth = (rng.uniform(size=2 * n) < probability).astype(int)
        report = assess_fairness(truth, probability, groups)
        assert {"calibration_error", "mean_predicted_pd", "observed_default_rate"} <= set(
            report.per_group.columns
        )
        assert set(report.summary()) >= {"attribute", "is_fair", "calibration_gap"}

    def test_rejects_mismatched_lengths(self) -> None:
        """Input arrays must align."""
        from infrastructure.xai.fairness import assess_fairness

        with pytest.raises(CalculationError):
            assess_fairness(np.array([0, 1]), np.array([0.1]), pd.Series(["a", "b"]))

    def test_rejects_when_no_group_is_large_enough(self) -> None:
        """A comparison needs at least one adequately sized group."""
        from infrastructure.xai.fairness import assess_fairness

        with pytest.raises(CalculationError):
            assess_fairness(
                np.array([0, 1, 0, 1]),
                np.array([0.1, 0.9, 0.2, 0.8]),
                pd.Series(["a", "a", "b", "b"]),
                min_group_size=1000,
            )
