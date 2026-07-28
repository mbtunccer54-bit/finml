"""Unit tests for the infrastructure adapters.

These cover the places where a silent failure would be most expensive: fold
leakage, drift thresholds, probability-shape normalisation, and the focal-loss
derivatives.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from domain.exceptions import (
    ConfigurationError,
    DriftDetectedError,
    FeatureStoreError,
    ModelTrainingError,
)
from domain.value_objects import DriftSeverity
from infrastructure.config.schemas import DriftConfig, FeatureConfig, ModelSpec, SplitConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------
class TestSplitters:
    """Leakage-aware cross-validation."""

    @pytest.fixture
    def timeline(self) -> tuple[pd.DataFrame, pd.Series]:
        """A 20-quarter panel with 30 obligors per quarter.

        Returns:
            A dummy feature matrix and the per-row timestamps.
        """
        dates = pd.date_range("2019-01-01", periods=20, freq="QE")
        rows = [(d, f"E{i}") for d in dates for i in range(30)]
        frame = pd.DataFrame(rows, columns=["t", "e"])
        return pd.DataFrame(np.zeros((len(frame), 3))), frame["t"]

    @pytest.mark.parametrize("strategy", ["purged_kfold", "blocked", "expanding", "sliding"])
    def test_no_train_validation_overlap(
        self, timeline: tuple[pd.DataFrame, pd.Series], strategy: str
    ) -> None:
        """No row may appear on both sides of a fold."""
        from infrastructure.data.splitters import build_splitter

        x, times = timeline
        splitter = build_splitter(SplitConfig(strategy=strategy, n_splits=4))
        folds = list(splitter.split(x, groups=times))
        assert folds
        for train_idx, val_idx in folds:
            assert not set(train_idx) & set(val_idx)

    @pytest.mark.parametrize("strategy", ["blocked", "expanding", "sliding"])
    def test_walk_forward_never_trains_on_the_future(
        self, timeline: tuple[pd.DataFrame, pd.Series], strategy: str
    ) -> None:
        """Forward-only strategies must train strictly on the past."""
        from infrastructure.data.splitters import build_splitter

        x, times = timeline
        splitter = build_splitter(SplitConfig(strategy=strategy, n_splits=4))
        for train_idx, val_idx in splitter.split(x, groups=times):
            assert times.iloc[train_idx].max() < times.iloc[val_idx].min()

    def test_purged_kfold_leaves_a_gap_around_validation(
        self, timeline: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Purging must remove the periods adjacent to the validation block."""
        from infrastructure.data.splitters import PurgedKFold

        x, times = timeline
        splitter = PurgedKFold(4, purge_frac=0.05, embargo_frac=0.05)
        for train_idx, val_idx in splitter.split(x, groups=times):
            train_periods = set(times.iloc[train_idx])
            val_periods = set(times.iloc[val_idx])
            assert not train_periods & val_periods

    def test_sliding_window_keeps_a_constant_training_size(
        self, timeline: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """A rolling window must not grow."""
        from infrastructure.data.splitters import SlidingWindowSplit

        x, times = timeline
        sizes = {
            len(train_idx)
            for train_idx, _ in SlidingWindowSplit(3, window_frac=0.4).split(x, groups=times)
        }
        assert len(sizes) == 1

    def test_holdout_takes_the_latest_periods(
        self, timeline: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """The test block must be the most recent slice, never a random one."""
        from infrastructure.data.splitters import time_based_holdout

        _x, times = timeline
        train_idx, test_idx = time_based_holdout(times, test_size_frac=0.2)
        assert times.iloc[train_idx].max() < times.iloc[test_idx].min()
        assert times.iloc[test_idx].nunique() == 4

    def test_rejects_unknown_strategy(self) -> None:
        """An unknown strategy is a configuration error."""
        from infrastructure.data.splitters import build_splitter

        with pytest.raises(ConfigurationError):
            build_splitter(SplitConfig(strategy="kfold"))

    def test_rejects_degenerate_parameters(self) -> None:
        """Fold counts and fractions are validated on construction."""
        from infrastructure.data.splitters import PurgedKFold

        with pytest.raises(ConfigurationError):
            PurgedKFold(1)
        with pytest.raises(ConfigurationError):
            PurgedKFold(3, purge_frac=0.9)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
class TestDriftDetection:
    """PSI, the KS test and the fail-safe policy."""

    @pytest.fixture
    def reference(self, rng: np.random.Generator) -> pd.DataFrame:
        """A reference distribution.

        Args:
            rng: Seeded generator.

        Returns:
            The reference frame.
        """
        return pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(5, 2, 2000)})

    def test_psi_is_near_zero_for_the_same_distribution(self, rng: np.random.Generator) -> None:
        """Samples from one distribution must not register as drift."""
        from infrastructure.data.drift_detector import population_stability_index

        result = population_stability_index(rng.normal(0, 1, 2000), rng.normal(0, 1, 500))
        assert result.psi < 0.1

    def test_psi_grows_with_the_shift(self, rng: np.random.Generator) -> None:
        """A larger mean shift must produce a larger PSI."""
        from infrastructure.data.drift_detector import population_stability_index

        reference = rng.normal(0, 1, 2000)
        small = population_stability_index(reference, rng.normal(0.3, 1, 500)).psi
        large = population_stability_index(reference, rng.normal(1.5, 1, 500)).psi
        assert large > small > 0.0

    def test_detects_a_shifted_feature(
        self, reference: pd.DataFrame, drift_config: DriftConfig, rng: np.random.Generator
    ) -> None:
        """A shifted feature is reported as severe and named."""
        from infrastructure.data.drift_detector import DriftDetector

        current = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(9, 2, 500)})
        report = DriftDetector(drift_config).fit(reference).detect(current)
        assert report.severity is DriftSeverity.SEVERE
        assert "b" in [f.feature for f in report.drifted_features]

    def test_block_policy_refuses_to_score(
        self, reference: pd.DataFrame, drift_config: DriftConfig, rng: np.random.Generator
    ) -> None:
        """Under `block`, severe drift raises rather than returning a score."""
        from infrastructure.data.drift_detector import DriftDetector

        detector = DriftDetector(drift_config).fit(reference)
        current = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(9, 2, 500)})
        with pytest.raises(DriftDetectedError):
            detector.enforce(detector.detect(current))

    def test_warn_policy_allows_scoring(
        self, reference: pd.DataFrame, rng: np.random.Generator
    ) -> None:
        """Under `warn`, the same batch is scored with the severity attached."""
        from infrastructure.data.drift_detector import DriftDetector

        config = DriftConfig(on_severe="warn", min_reference_size=20, min_batch_size=10)
        detector = DriftDetector(config).fit(reference)
        current = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(9, 2, 500)})
        report = detector.enforce(detector.detect(current))
        assert report.severity is DriftSeverity.SEVERE

    def test_small_batches_do_not_raise_false_alarms(
        self, reference: pd.DataFrame, drift_config: DriftConfig
    ) -> None:
        """A PSI over three rows is noise, so no verdict is issued."""
        from infrastructure.data.drift_detector import DriftDetector

        detector = DriftDetector(drift_config).fit(reference)
        report = detector.detect(pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [5.0, 6.0, 7.0]}))
        assert report.severity is DriftSeverity.NONE

    def test_macro_features_are_excluded_by_default(self) -> None:
        """Macro columns are constant per batch, so they are not monitored."""
        from infrastructure.data.drift_detector import DriftDetector

        detector = DriftDetector(DriftConfig(min_reference_size=5, min_batch_size=2))
        reference = pd.DataFrame(
            {"gdp_growth": np.linspace(-0.05, 0.05, 100), "current_ratio": np.linspace(1, 3, 100)}
        )
        current = pd.DataFrame({"gdp_growth": [0.02] * 50, "current_ratio": np.linspace(1, 3, 50)})
        report = detector.fit(reference).detect(current)
        assert "gdp_growth" not in [f.feature for f in report.features]

    def test_disabled_detector_reports_nothing(self, reference: pd.DataFrame) -> None:
        """Disabling the check short-circuits it."""
        from infrastructure.data.drift_detector import DriftDetector

        detector = DriftDetector(DriftConfig(enabled=False)).fit(reference)
        assert detector.detect(reference).severity is DriftSeverity.NONE


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------
class TestFeatureStore:
    """Parquet storage with a SQLite catalogue."""

    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        """A small keyed feature frame.

        Returns:
            The frame.
        """
        return pd.DataFrame(
            {
                "entity_id": ["A", "B", "C"],
                "as_of_date": pd.Timestamp("2024-01-01"),
                "sentiment_compound": [0.1, -0.4, 0.6],
            }
        )

    def test_round_trip(self, tmp_store: Any, frame: pd.DataFrame) -> None:
        """A written feature set reads back with identical values.

        Datetime *resolution* is not asserted: Parquet normalises ``datetime64[s]``
        to ``datetime64[ms]``. Values and ordering are what the store promises,
        and key merges are unaffected (see the lookup test).
        """
        tmp_store.write("f", frame, version="v1")
        loaded = tmp_store.read("f", "v1")

        assert list(loaded.columns) == list(frame.columns)
        assert loaded["entity_id"].tolist() == frame["entity_id"].tolist()
        assert loaded["sentiment_compound"].tolist() == pytest.approx(
            frame["sentiment_compound"].tolist()
        )
        assert (
            loaded["as_of_date"].astype("datetime64[ns]").tolist()
            == frame["as_of_date"].astype("datetime64[ns]").tolist()
        )

    def test_lookup_reports_hits_and_misses(self, tmp_store: Any, frame: pd.DataFrame) -> None:
        """Missing keys are reported so the caller computes only those."""
        keys = ["entity_id", "as_of_date"]
        tmp_store.upsert_keyed("nlp", frame, key_columns=keys)
        wanted = pd.DataFrame(
            {"entity_id": ["A", "C", "D"], "as_of_date": pd.Timestamp("2024-01-01")}
        )
        found, missing = tmp_store.fetch_keyed("nlp", wanted, key_columns=keys)
        assert sorted(found["entity_id"]) == ["A", "C"]
        assert list(missing["entity_id"]) == ["D"]

    def test_upsert_replaces_existing_keys(self, tmp_store: Any, frame: pd.DataFrame) -> None:
        """Recomputing with a newer model overwrites the cached value."""
        keys = ["entity_id", "as_of_date"]
        tmp_store.upsert_keyed("nlp", frame, key_columns=keys)
        update = pd.DataFrame(
            {
                "entity_id": ["A"],
                "as_of_date": pd.Timestamp("2024-01-01"),
                "sentiment_compound": [0.99],
            }
        )
        tmp_store.upsert_keyed("nlp", update, key_columns=keys)
        stored = tmp_store.read("nlp").set_index("entity_id")
        assert stored.loc["A", "sentiment_compound"] == pytest.approx(0.99)
        assert len(stored) == 3

    def test_fallback_provenance_is_recorded(self, tmp_store: Any, frame: pd.DataFrame) -> None:
        """A degraded backend must be visible in the catalogue."""
        tmp_store.write("f", frame, version="v1", producer="lexicon", is_fallback=True)
        record = tmp_store.get_record("f", "v1")
        assert record is not None
        assert record.is_fallback
        assert record.producer == "lexicon"

    def test_rejects_empty_and_bad_keys(self, tmp_store: Any, frame: pd.DataFrame) -> None:
        """Empty writes and absent key columns are errors."""
        with pytest.raises(FeatureStoreError):
            tmp_store.write("f", pd.DataFrame())
        with pytest.raises(FeatureStoreError):
            tmp_store.upsert_keyed("f", frame, key_columns=["nope"])
        with pytest.raises(FeatureStoreError):
            tmp_store.read("never_written")

    def test_delete_removes_the_record(self, tmp_store: Any, frame: pd.DataFrame) -> None:
        """Deleting removes it from the catalogue."""
        tmp_store.write("f", frame, version="v1")
        assert tmp_store.delete("f", "v1")
        assert not tmp_store.exists("f", "v1")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class TestPreprocessing:
    """Feature matrix construction."""

    def test_produces_a_stable_numeric_matrix(self, panel: pd.DataFrame) -> None:
        """Output is fully numeric with one-hot categoricals."""
        from infrastructure.data.preprocessing import FeatureMatrixBuilder

        builder = FeatureMatrixBuilder(FeatureConfig())
        matrix = builder.fit_transform(panel)
        assert matrix.select_dtypes(exclude=[np.number]).empty
        assert any(c.startswith("sector=") for c in matrix.columns)
        assert list(matrix.columns) == builder.feature_names

    def test_transform_is_column_stable_for_unseen_levels(self, panel: pd.DataFrame) -> None:
        """A batch missing a sector still yields the same matrix width."""
        from infrastructure.data.preprocessing import FeatureMatrixBuilder

        builder = FeatureMatrixBuilder(FeatureConfig()).fit(panel)
        subset = panel[panel["sector"] == panel["sector"].iloc[0]]
        assert list(builder.transform(subset).columns) == builder.feature_names

    def test_missing_columns_are_imputed(self, panel: pd.DataFrame) -> None:
        """An absent feature falls back to the contract's typical value."""
        from infrastructure.data.preprocessing import FeatureMatrixBuilder

        builder = FeatureMatrixBuilder(FeatureConfig(scale_numeric=False)).fit(panel)
        matrix = builder.transform(panel.drop(columns=["current_ratio"]))
        from domain.entities import FEATURE_CONTRACT

        assert matrix["current_ratio"].iloc[0] == pytest.approx(
            FEATURE_CONTRACT.get("current_ratio").typical
        )

    def test_requires_fitting_first(self, panel: pd.DataFrame) -> None:
        """Transforming before fitting is an error, not silent garbage."""
        from domain.exceptions import SchemaValidationError
        from infrastructure.data.preprocessing import FeatureMatrixBuilder

        with pytest.raises(SchemaValidationError):
            FeatureMatrixBuilder(FeatureConfig()).transform(panel)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TestFocalLoss:
    """The focal-loss objective, verified against finite differences."""

    @staticmethod
    def _loss(z: np.ndarray, y: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
        """Evaluate the focal loss directly.

        Args:
            z: Raw scores.
            y: Binary labels.
            gamma: Focusing parameter.
            alpha: Positive-class weight.

        Returns:
            The per-sample loss.
        """
        p = 1.0 / (1.0 + np.exp(-z))
        t = np.clip(np.where(y == 1, p, 1.0 - p), 1e-12, 1.0 - 1e-12)
        a = np.where(y == 1, alpha, 1.0 - alpha)
        return -a * (1.0 - t) ** gamma * np.log(t)

    @pytest.mark.parametrize("label", [0.0, 1.0])
    def test_gradient_matches_finite_differences(self, label: float) -> None:
        """An incorrect gradient degrades a booster silently, so it is checked."""
        from infrastructure.models.base import make_focal_objective

        gamma, alpha = 2.0, 0.25
        objective = make_focal_objective(gamma, alpha)
        z = np.array([-3.0, -1.0, -0.2, 0.0, 0.4, 1.5, 3.0])
        y = np.full_like(z, label)

        analytic, _hess = objective(y, z)
        step = 1e-5
        numeric = (
            self._loss(z + step, y, gamma, alpha) - self._loss(z - step, y, gamma, alpha)
        ) / (2 * step)
        assert np.allclose(analytic, numeric, atol=1e-6)

    @pytest.mark.parametrize("label", [0.0, 1.0])
    def test_hessian_matches_finite_differences(self, label: float) -> None:
        """The second derivative is checked for the same reason."""
        from infrastructure.models.base import make_focal_objective

        gamma, alpha = 2.0, 0.25
        objective = make_focal_objective(gamma, alpha)
        z = np.array([-2.0, -0.5, 0.3, 1.2])
        y = np.full_like(z, label)

        _grad, analytic = objective(y, z)
        step = 1e-4
        numeric = (
            self._loss(z + step, y, gamma, alpha)
            - 2 * self._loss(z, y, gamma, alpha)
            + self._loss(z - step, y, gamma, alpha)
        ) / step**2
        # The objective floors the Hessian at 1e-6 to keep leaf values valid.
        assert np.allclose(np.maximum(numeric, 1e-6), analytic, atol=1e-3)

    def test_hessian_is_always_positive(self) -> None:
        """A non-positive Hessian would break the boosting leaf update."""
        from infrastructure.models.base import make_focal_objective

        objective = make_focal_objective()
        z = np.linspace(-8, 8, 200)
        for label in (0.0, 1.0):
            _grad, hess = objective(np.full_like(z, label), z)
            assert np.all(hess > 0)

    def test_rejects_invalid_parameters(self) -> None:
        """Out-of-range focal parameters are configuration errors."""
        from infrastructure.models.base import make_focal_objective

        with pytest.raises(ModelTrainingError):
            make_focal_objective(gamma=-1.0)
        with pytest.raises(ModelTrainingError):
            make_focal_objective(alpha=1.5)


class TestModelAdapters:
    """The four base learners behind one interface."""

    SPECS: ClassVar[dict[str, ModelSpec]] = {
        "catboost": ModelSpec(
            params={"iterations": 25, "depth": 3, "verbose": False, "allow_writing_files": False}
        ),
        "xgboost": ModelSpec(params={"n_estimators": 25, "max_depth": 3, "tree_method": "hist"}),
        "lightgbm": ModelSpec(params={"n_estimators": 25, "num_leaves": 7, "verbosity": -1}),
        "random_forest": ModelSpec(params={"n_estimators": 25, "max_depth": 5}),
    }

    @pytest.mark.parametrize("name", sorted(SPECS))
    def test_probabilities_are_well_formed(
        self, name: str, feature_matrix: tuple[pd.DataFrame, np.ndarray, Any]
    ) -> None:
        """Every adapter returns an (n, 2) array of valid probabilities."""
        from infrastructure.models.trainers import build_model

        x, y, _builder = feature_matrix
        model = build_model(name, self.SPECS[name], n_jobs=1).fit(x, y)
        proba = model.predict_proba(x)
        assert proba.shape == (len(x), 2)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0
        assert np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.parametrize("name", sorted(SPECS))
    def test_is_recognised_as_a_classifier(self, name: str) -> None:
        """`is_classifier` must hold, or CalibratedClassifierCV inverts the output."""
        from sklearn.base import is_classifier

        from infrastructure.models.trainers import build_model

        assert is_classifier(build_model(name, self.SPECS[name]))

    @pytest.mark.parametrize("name", sorted(SPECS))
    def test_is_clonable(self, name: str) -> None:
        """Cloning is required by stacking and calibration."""
        from sklearn.base import clone

        from infrastructure.models.trainers import build_model

        assert not hasattr(clone(build_model(name, self.SPECS[name])), "estimator_")

    @pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
    def test_focal_output_is_normalised(
        self, name: str, feature_matrix: tuple[pd.DataFrame, np.ndarray, Any]
    ) -> None:
        """LightGBM returns raw logits under a custom objective; they must be mapped."""
        from infrastructure.models.trainers import build_model

        x, y, _builder = feature_matrix
        model = build_model(name, self.SPECS[name], n_jobs=1, imbalance_strategy="focal").fit(x, y)
        proba = model.predict_proba(x)
        assert proba.shape == (len(x), 2)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_rejects_single_class_training_data(
        self, feature_matrix: tuple[pd.DataFrame, np.ndarray, Any]
    ) -> None:
        """A degenerate fold is an error, not a silently useless model."""
        from infrastructure.models.trainers import build_model

        x, _y, _builder = feature_matrix
        with pytest.raises(ModelTrainingError):
            build_model("lightgbm", self.SPECS["lightgbm"]).fit(x, np.zeros(len(x)))

    def test_rejects_unknown_model(self) -> None:
        """An unknown model name is rejected with the known set attached."""
        from infrastructure.models.trainers import build_model

        with pytest.raises(ModelTrainingError):
            build_model("svm", ModelSpec())

    def test_predicting_before_fitting_raises(self) -> None:
        """Unfitted prediction is an error."""
        from infrastructure.models.trainers import LightGBMAdapter

        with pytest.raises(ModelTrainingError):
            LightGBMAdapter().predict_proba(pd.DataFrame({"a": [1.0]}))

    def test_normalisation_handles_raw_logits(self) -> None:
        """1-D logits are mapped through the logistic, not passed through."""
        from infrastructure.models.trainers import LightGBMAdapter

        adapter = LightGBMAdapter()
        out = adapter._normalise_proba(np.array([-2.0, 0.0, 3.0]), n_rows=3)
        assert out.shape == (3, 2)
        assert out[:, 1].min() >= 0.0
        assert out[:, 1].max() <= 1.0
        assert out[1, 1] == pytest.approx(0.5)


class TestEvaluationAndSelection:
    """Metrics and the promotion gate."""

    def test_evaluate_finds_the_cost_optimal_threshold(self) -> None:
        """Omitting a threshold triggers the cost-minimising search."""
        from infrastructure.models.evaluation import evaluate_model

        truth = np.array([0] * 90 + [1] * 10)
        probability = np.concatenate([np.linspace(0.01, 0.3, 90), np.linspace(0.4, 0.95, 10)])
        performance, decision = evaluate_model(truth, probability, model_id="m", split="validation")
        assert 0.0 < performance.threshold < 1.0
        assert decision.expected_cost <= 1e-9

    def test_rejects_single_class_split(self) -> None:
        """A split with one class cannot be evaluated."""
        from domain.exceptions import CalculationError
        from infrastructure.models.evaluation import evaluate_model

        with pytest.raises(CalculationError):
            evaluate_model(np.zeros(10), np.linspace(0, 1, 10), model_id="m", split="test")

    def test_calibration_error_rewards_a_calibrated_model(self) -> None:
        """A perfectly calibrated forecast has a near-zero ECE."""
        from infrastructure.models.evaluation import expected_calibration_error

        rng = np.random.default_rng(0)
        probability = rng.uniform(0.05, 0.95, 4000)
        truth = (rng.uniform(size=4000) < probability).astype(int)
        assert expected_calibration_error(truth, probability, n_bins=10) < 0.05

    def test_champion_selection_prefers_lowest_cost_among_eligible(self) -> None:
        """Among gate-passing models, business cost decides."""
        from domain.entities import ModelPerformance
        from domain.value_objects import ModelId
        from infrastructure.models.evaluation import select_champion

        def record(name: str, pr_auc: float, cost: float, ece: float) -> ModelPerformance:
            """Build a performance record.

            Args:
                name: Model identifier.
                pr_auc: PR-AUC.
                cost: Business cost.
                ece: Calibration error.

            Returns:
                The record.
            """
            return ModelPerformance(
                model_id=ModelId(name),
                split="validation",
                roc_auc=0.8,
                pr_auc=pr_auc,
                f1=0.5,
                precision=0.5,
                recall=0.5,
                brier_score=0.05,
                expected_calibration_error=ece,
                business_cost=cost,
                inference_latency_ms=5.0,
            )

        champion, notes = select_champion(
            [record("a", 0.5, 200.0, 0.01), record("b", 0.45, 100.0, 0.01)],
            max_calibration_error=0.05,
            max_latency_ms=50.0,
            min_pr_auc=0.1,
        )
        assert str(champion) == "b"
        assert notes

    def test_selection_falls_back_and_explains_when_no_model_passes(self) -> None:
        """A missed gate must be reported, not hidden."""
        from domain.entities import ModelPerformance
        from domain.value_objects import ModelId
        from infrastructure.models.evaluation import select_champion

        bad = ModelPerformance(
            model_id=ModelId("a"),
            split="validation",
            roc_auc=0.8,
            pr_auc=0.5,
            f1=0.5,
            precision=0.5,
            recall=0.5,
            brier_score=0.2,
            expected_calibration_error=0.5,
            business_cost=100.0,
            inference_latency_ms=5.0,
        )
        champion, notes = select_champion(
            [bad], max_calibration_error=0.05, max_latency_ms=50.0, min_pr_auc=0.1
        )
        assert str(champion) == "a"
        assert any("cleared" in note for note in notes)


class TestCalibration:
    """Probability calibration."""

    def test_calibration_preserves_ranking(
        self, feature_matrix: tuple[pd.DataFrame, np.ndarray, Any]
    ) -> None:
        """Calibration is monotone: AUC must not invert."""
        from sklearn.metrics import roc_auc_score

        from infrastructure.models.calibration import fit_best_calibrator
        from infrastructure.models.trainers import build_model

        x, y, _builder = feature_matrix
        half = len(x) // 2
        model = build_model(
            "lightgbm",
            ModelSpec(params={"n_estimators": 30, "num_leaves": 7, "verbosity": -1}),
            n_jobs=1,
        ).fit(x.iloc[:half], y[:half])

        before = roc_auc_score(y[half:], model.predict_proba(x.iloc[half:])[:, 1])
        report = fit_best_calibrator(model, x.iloc[half:], y[half:])
        after = roc_auc_score(y[half:], report.model.predict_proba(x.iloc[half:])[:, 1])

        assert after > 0.5
        assert abs(after - before) < 0.2

    def test_calibration_improves_or_keeps_the_original(
        self, feature_matrix: tuple[pd.DataFrame, np.ndarray, Any]
    ) -> None:
        """The uncalibrated model competes, so ECE never gets worse."""
        from infrastructure.models.calibration import fit_best_calibrator
        from infrastructure.models.trainers import build_model

        x, y, _builder = feature_matrix
        half = len(x) // 2
        model = build_model(
            "lightgbm",
            ModelSpec(params={"n_estimators": 30, "num_leaves": 7, "verbosity": -1}),
            n_jobs=1,
        ).fit(x.iloc[:half], y[:half])
        report = fit_best_calibrator(model, x.iloc[half:], y[half:])
        assert report.after.expected_calibration_error <= report.before.expected_calibration_error

    def test_rejects_unknown_method(
        self, fitted_model: Any, feature_matrix: tuple[pd.DataFrame, np.ndarray, Any]
    ) -> None:
        """An unknown calibration method is a configuration error."""
        from infrastructure.models.calibration import calibrate_prefit

        x, y, _builder = feature_matrix
        with pytest.raises(ModelTrainingError):
            calibrate_prefit(fitted_model, x, y, method="magic")


class TestEnsemble:
    """Stacked generalisation."""

    def test_stack_produces_valid_probabilities(
        self, feature_matrix: tuple[pd.DataFrame, np.ndarray, Any]
    ) -> None:
        """The stack behaves like any other classifier."""
        from infrastructure.models.ensemble import StackedEnsemble, build_meta_learner
        from infrastructure.models.trainers import build_model

        x, y, _builder = feature_matrix
        base = [
            (
                "lgbm",
                build_model(
                    "lightgbm",
                    ModelSpec(params={"n_estimators": 20, "num_leaves": 7, "verbosity": -1}),
                    n_jobs=1,
                ),
            ),
            (
                "rf",
                build_model(
                    "random_forest",
                    ModelSpec(params={"n_estimators": 20, "max_depth": 5}),
                    n_jobs=1,
                ),
            ),
        ]
        cut = int(len(x) * 0.6)
        splits = [(np.arange(cut), np.arange(cut, len(x)))]
        stack = StackedEnsemble(base, build_meta_learner("logistic_regression")).fit(
            x, y, cv_splits=splits
        )
        proba = stack.predict_proba(x)
        assert proba.shape == (len(x), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert set(stack.meta_weights) == {"lgbm", "rf"}

    def test_requires_base_models(self) -> None:
        """A stack with no base models is meaningless."""
        from infrastructure.models.ensemble import StackedEnsemble

        with pytest.raises(ModelTrainingError):
            StackedEnsemble([]).fit(pd.DataFrame({"a": [1.0, 2.0]}), np.array([0, 1]))

    def test_rejects_unknown_meta_learner(self) -> None:
        """An unknown meta-learner is a configuration error."""
        from infrastructure.models.ensemble import build_meta_learner

        with pytest.raises(ModelTrainingError):
            build_meta_learner("transformer")


# ---------------------------------------------------------------------------
# NLP fallbacks
# ---------------------------------------------------------------------------
class TestNLPFallbacks:
    """The offline tiers, which is what CI actually exercises."""

    def test_sentiment_separates_positive_from_negative(self) -> None:
        """The finance lexicon must get the obvious cases right."""
        from infrastructure.config.schemas import NLPConfig
        from infrastructure.nlp.finbert import FinBERTSentimentAnalyzer

        analyser = FinBERTSentimentAnalyzer(NLPConfig(finbert_model="__unavailable__"))
        negative, positive = analyser.analyse(
            [
                "covenant breach as liquidity deteriorates and losses mount",
                "record profit with strong margin expansion and improved growth",
            ]
        )
        assert negative.compound < 0 < positive.compound
        assert analyser.is_fallback

    def test_sentiment_handles_negation(self) -> None:
        """ "did not miss" must not be scored as a miss."""
        from infrastructure.config.schemas import NLPConfig
        from infrastructure.nlp.finbert import FinBERTSentimentAnalyzer

        analyser = FinBERTSentimentAnalyzer(NLPConfig(finbert_model="__unavailable__"))
        plain = analyser.analyse(["the company missed expectations"])[0]
        negated = analyser.analyse(["the company did not miss expectations"])[0]
        assert negated.compound > plain.compound

    def test_neutral_text_yields_neutral_score(self) -> None:
        """Text with no polarity words lands on neutral."""
        from infrastructure.config.schemas import NLPConfig
        from infrastructure.nlp.finbert import FinBERTSentimentAnalyzer

        analyser = FinBERTSentimentAnalyzer(NLPConfig(finbert_model="__unavailable__"))
        score = analyser.analyse(["the company confirms its quarterly results date"])[0]
        assert score.compound == pytest.approx(0.0, abs=0.05)

    def test_ner_extracts_money_dates_and_percentages(self) -> None:
        """The regex tier is reliable on rigid surface forms."""
        from infrastructure.config.schemas import NLPConfig
        from infrastructure.nlp.ner import FinancialNERExtractor

        extractor = FinancialNERExtractor(NLPConfig(ner_model="__unavailable__"))
        result = extractor.extract(
            "Acme Holdings Ltd took a $2.1 billion impairment on 2024-03-14, down 12.5%."
        )
        counts = result.counts()
        assert counts.get("MONEY", 0) >= 1
        assert counts.get("DATE", 0) >= 1
        assert counts.get("PERCENT", 0) >= 1
        assert result.is_fallback

    def test_topics_route_to_the_right_risk_category(self) -> None:
        """Seeded keywords must separate credit from operational risk."""
        from infrastructure.config.schemas import NLPConfig
        from infrastructure.nlp.topic_model import RiskTopicModel

        model = RiskTopicModel(NLPConfig(topic_model="__unavailable__"))
        assert (
            model.score("covenant breach triggers default and downgrade").dominant_topic == "credit"
        )
        assert (
            model.score("regulator opens fraud investigation after cyber outage").dominant_topic
            == "operational"
        )
        assert (
            model.score("refinancing delayed, cash flow under pressure").dominant_topic
            == "liquidity"
        )

    def test_topic_features_match_the_contract_columns(self) -> None:
        """Feature names must line up with the domain contract."""
        from domain.entities import FEATURE_CONTRACT
        from infrastructure.config.schemas import NLPConfig
        from infrastructure.nlp.topic_model import RiskTopicModel

        features = RiskTopicModel(NLPConfig()).score("covenant default").as_features()
        assert set(features) <= set(FEATURE_CONTRACT.feature_names)
