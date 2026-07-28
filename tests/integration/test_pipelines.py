"""End-to-end pipeline tests.

These run the real training pipeline on a small synthetic panel and then score,
explain and stress-test through the artifact it produced. They are slower than
the unit tests and catch a different class of defect: the ones that only appear
when the stages are wired together, such as training-serving skew or a
preprocessor that fails to travel with its model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Run the training pipeline once for the whole module.

    Args:
        tmp_path_factory: Pytest temporary directory factory.

    Returns:
        The training result.
    """
    from application.train_pipeline import TrainPipeline
    from infrastructure.config.loader import load_config, to_object

    workspace = tmp_path_factory.mktemp("finml")
    config = to_object(
        load_config(
            [
                "models@model=fast",
                "data.n_entities=250",
                "data.n_periods=10",
                "data.split.n_splits=3",
                "training.compute_shap_on_train=false",
                "tracking.enabled=false",
                f"paths.artifacts={workspace / 'artifacts'}",
                f"paths.reports={workspace / 'artifacts' / 'reports'}",
                f"paths.feature_store={workspace / 'store'}",
            ]
        )
    )
    return TrainPipeline(config).run(), config


class TestTrainingPipeline:
    """The training use case."""

    def test_produces_a_champion(self, trained: Any) -> None:
        """A run must end with a promoted, servable model."""
        result, _config = trained
        assert result.champion_id
        assert result.bundle.model is not None
        assert result.bundle.feature_names

    def test_evaluates_every_model_on_both_splits(self, trained: Any) -> None:
        """Base learners, the stack and the calibrated variant are all scored."""
        result, _config = trained
        splits = {p.split for p in result.performances}
        models = {str(p.model_id) for p in result.performances}
        assert splits == {"validation", "test"}
        assert {"catboost", "xgboost", "lightgbm", "random_forest"} <= models
        assert "ensemble" in models

    def test_benchmark_table_is_populated(self, trained: Any) -> None:
        """The comparison table is the artifact a reviewer reads first."""
        result, _config = trained
        assert not result.benchmark.empty
        assert {"pr_auc", "business_cost", "expected_calibration_error"} <= set(
            result.benchmark.columns
        )

    def test_models_beat_chance_on_the_held_out_block(self, trained: Any) -> None:
        """A pipeline that learns nothing would pass every structural test."""
        result, _config = trained
        test_records = [p for p in result.performances if p.split == "test"]
        assert test_records
        base_rate = 0.08
        for record in test_records:
            assert record.roc_auc > 0.6, record.model_id
            assert record.pr_auc > base_rate, record.model_id

    def test_validation_and_test_metrics_are_consistent(self, trained: Any) -> None:
        """A large gap means the selection split leaked into the model."""
        result, _config = trained
        by_model: dict[str, dict[str, float]] = {}
        for record in result.performances:
            by_model.setdefault(str(record.model_id), {})[record.split] = record.pr_auc
        for name, splits in by_model.items():
            if "validation" in splits and "test" in splits:
                # A calibrator fitted on its own evaluation rows previously
                # produced a validation PR-AUC three times the test figure.
                assert splits["validation"] < splits["test"] + 0.45, name

    def test_bundle_carries_its_preprocessor_and_drift_reference(self, trained: Any) -> None:
        """Without both, the artifact is neither reproducible nor monitorable."""
        from application.train_pipeline import BUNDLE_DRIFT_REFERENCE, BUNDLE_PREPROCESSOR

        result, _config = trained
        assert result.bundle.extra.get(BUNDLE_PREPROCESSOR) is not None
        reference = result.bundle.extra.get(BUNDLE_DRIFT_REFERENCE)
        assert isinstance(reference, pd.DataFrame)
        assert not reference.empty

    def test_champion_artifact_round_trips(self, trained: Any) -> None:
        """The promoted model reloads and scores identically."""
        from infrastructure.models.registry import LocalModelStore

        result, config = trained
        store = LocalModelStore(config.paths.artifacts)
        assert store.has_champion()
        reloaded = store.load_champion()
        assert reloaded.model_id == result.champion_id
        assert reloaded.feature_names == result.bundle.feature_names

    def test_writes_the_benchmark_artifact(self, trained: Any) -> None:
        """The benchmark is persisted for the dashboard and MLflow."""
        result, _config = trained
        assert Path(result.artifacts["benchmark"]).is_file()


class TestInferencePipeline:
    """The scoring use case."""

    @pytest.fixture(scope="class")
    def pipeline(self, trained: Any) -> Any:
        """Load the trained champion into an inference pipeline.

        Args:
            trained: The training result and configuration.

        Returns:
            The inference pipeline.
        """
        from application.inference_pipeline import InferencePipeline

        _result, config = trained
        return InferencePipeline.from_champion(config)

    @pytest.fixture(scope="class")
    def batch(self, trained: Any, pipeline: Any) -> pd.DataFrame:
        """The most recent period of the demo panel, enriched as training was.

        The enrichment step is not optional. Skipping it leaves the NLP columns
        as raw generator output while the model was fitted on FinBERT-derived
        values — textbook training-serving skew, and the drift detector
        correctly refuses to score it.

        Args:
            trained: The training result and configuration.
            pipeline: The inference pipeline supplying the shared enricher.

        Returns:
            The scoring batch.
        """
        from infrastructure.data.fetchers import build_fetcher

        _result, config = trained
        dataset = build_fetcher(config.data).fetch()
        panel = dataset.panel
        latest = panel[panel["as_of_date"] == panel["as_of_date"].max()].reset_index(drop=True)
        return pipeline.enrich(latest, dataset.news)

    def test_scores_a_batch(self, pipeline: Any, batch: pd.DataFrame) -> None:
        """Every row gets a valid, banded score."""
        result = pipeline.score(batch, check_drift=False)
        assert len(result.scores) == len(batch)
        assert all(0.0 <= s.pd_value <= 1.0 for s in result.scores)
        assert all(s.band is not None for s in result.scores)

    def test_applies_the_regulatory_floor(self, pipeline: Any, batch: pd.DataFrame) -> None:
        """No reported PD may fall below the configured floor."""
        result = pipeline.score(batch, check_drift=False)
        assert min(s.pd_value for s in result.scores) >= pipeline.pd_calculator.floor

    def test_overlay_shifts_every_score_upwards(self, pipeline: Any, batch: pd.DataFrame) -> None:
        """A conservative overlay must raise risk monotonically."""
        base = pipeline.score(batch, check_drift=False)
        overlaid = pipeline.score(batch, check_drift=False, overlay_log_odds=1.0)
        assert all(
            b.pd_value <= o.pd_value + 1e-12
            for b, o in zip(base.scores, overlaid.scores, strict=True)
        )

    def test_no_drift_against_the_training_distribution(
        self, pipeline: Any, batch: pd.DataFrame
    ) -> None:
        """Scoring the same generator the model was trained on must not alarm."""
        from domain.value_objects import DriftSeverity

        result = pipeline.score(batch, check_drift=True)
        assert result.drift is not None
        assert result.drift.severity is not DriftSeverity.SEVERE

    def test_blocks_a_genuinely_drifted_batch(self, pipeline: Any, batch: pd.DataFrame) -> None:
        """The fail-safe must fire when the population really has moved.

        The shift stays inside the contract's declared bounds on purpose, so the
        schema check passes and it is genuinely *drift* being detected rather
        than a malformed payload being caught one layer earlier.
        """
        from domain.entities import FEATURE_CONTRACT
        from domain.exceptions import DriftDetectedError

        corrupted = batch.copy()
        for column in ("current_ratio", "debt_to_assets", "return_on_assets"):
            spec = FEATURE_CONTRACT.get(column)
            midpoint = (float(spec.ge) + float(spec.le)) / 2.0
            corrupted[column] = float(spec.clamp(midpoint))

        with pytest.raises(DriftDetectedError):
            pipeline.score(corrupted, check_drift=True)

    def test_explains_a_single_obligor(self, pipeline: Any, batch: pd.DataFrame) -> None:
        """SHAP returns one contribution per model feature."""
        explanation = pipeline.explain(batch.head(1), method="shap")
        assert explanation.contributions
        assert len(explanation.contributions) == len(pipeline.bundle.feature_names)
        assert explanation.top_drivers(3)

    def test_lime_explains_the_same_obligor(self, pipeline: Any, batch: pd.DataFrame) -> None:
        """LIME provides an independent second opinion."""
        explanation = pipeline.explain(batch.head(1), method="lime")
        assert explanation.method == "lime"
        assert explanation.contributions

    def test_rejects_an_unknown_explanation_method(
        self, pipeline: Any, batch: pd.DataFrame
    ) -> None:
        """An unknown method is a caller error."""
        from domain.exceptions import ExplainerError

        with pytest.raises(ExplainerError):
            pipeline.explain(batch.head(1), method="magic")

    def test_counterfactuals_reduce_the_pd(self, pipeline: Any, batch: pd.DataFrame) -> None:
        """Recourse must actually lower risk for the riskiest obligor."""
        result = pipeline.score(batch, check_drift=False)
        worst = int(np.argmax([s.pd_value for s in result.scores]))
        examples = pipeline.counterfactuals(batch.iloc[[worst]], target_pd=0.05)
        if examples:
            assert all(e.counterfactual_pd < e.original_pd for e in examples)
            assert all(e.n_changes >= 1 for e in examples)


class TestScenarioSimulation:
    """The stress-testing use case."""

    @pytest.fixture(scope="class")
    def simulator(self, trained: Any) -> Any:
        """Build a scenario simulator over the trained champion.

        Args:
            trained: The training result and configuration.

        Returns:
            The simulator.
        """
        from application.inference_pipeline import InferencePipeline
        from application.scenario_simulator import ScenarioSimulator

        _result, config = trained
        pipeline = InferencePipeline.from_champion(config)
        return ScenarioSimulator(
            config, scoring_fn=lambda x: pipeline.bundle.model.predict_proba(x)[:, 1]
        )

    def test_satellite_model_projects_the_macro_block(self, simulator: Any, trained: Any) -> None:
        """The satellite model produces a forward path."""
        from infrastructure.data.fetchers import build_fetcher

        _result, config = trained
        macro = build_fetcher(config.data).fetch().macro
        forecast = simulator.project_macro(macro, horizon=4)
        assert len(forecast.path) == 4
        assert forecast.model in ("var", "ar1")

    def test_severity_ordering_holds(self, simulator: Any) -> None:
        """A severe scenario must stress harder than an adverse one."""
        pds = pd.Series([0.05] * 40)
        sectors = pd.Series(["technology"] * 40)
        outcomes = simulator.run_all_scenarios(pds, sectors).set_index("scenario")
        assert (
            outcomes.loc["baseline", "stressed_pd"]
            < outcomes.loc["adverse", "stressed_pd"]
            < outcomes.loc["severely_adverse", "stressed_pd"]
        )

    def test_baseline_leaves_pds_unchanged(self, simulator: Any) -> None:
        """An unshocked scenario is the identity."""
        pds = pd.Series([0.1] * 20)
        result = simulator.run_scenario("baseline", pds, pd.Series(["utilities"] * 20))
        assert result.stressed_pd == pytest.approx(result.baseline_pd, abs=1e-9)

    def test_sector_differentiation(self, simulator: Any) -> None:
        """Cyclical sectors must be hit harder than defensive ones."""
        matrix = simulator.sector_impact_matrix()
        assert (
            matrix.loc["consumer_discretionary", "severely_adverse"]
            > matrix.loc["utilities", "severely_adverse"]
        )

    def test_what_if_composes_micro_and_macro(self, simulator: Any, trained: Any) -> None:
        """Ratio overrides and a macro scenario apply together."""
        from application.inference_pipeline import InferencePipeline
        from infrastructure.data.fetchers import build_fetcher

        _result, config = trained
        pipeline = InferencePipeline.from_champion(config)
        panel = build_fetcher(config.data).fetch().panel
        row = panel.head(1)
        matrix = pipeline.build_matrix(row)

        improved = simulator.what_if(
            matrix,
            entity_id="E0",
            sector=str(row["sector"].iloc[0]),
            overrides={"interest_coverage": 5.0},
        )
        stressed = simulator.what_if(
            matrix,
            entity_id="E0",
            sector=str(row["sector"].iloc[0]),
            overrides={"interest_coverage": 5.0},
            scenario_name="severely_adverse",
        )
        assert stressed.odds_multiplier > 1.0
        assert stressed.adjusted_pd > improved.adjusted_pd

    def test_rejects_an_unknown_scenario(self, simulator: Any) -> None:
        """An unknown scenario name is an error, not a silent baseline."""
        from domain.exceptions import ScenarioError

        with pytest.raises(ScenarioError):
            simulator.get_scenario("apocalypse")
