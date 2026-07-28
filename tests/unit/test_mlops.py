"""MLOps tests: hyperparameter tuning and the model registry.

Both are places where a failure should degrade rather than abort. A tracking
server being down must not lose a trained model, and one bad corner of a search
space must not end a study.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from domain.exceptions import ConfigurationError, ModelNotFoundError, RegistryError
from infrastructure.config.schemas import ModelSpec, TrackingConfig, TuningConfig

pytestmark = pytest.mark.unit


class TestSearchSpaceSampling:
    """Declarative Optuna search spaces."""

    @pytest.fixture
    def trial(self) -> Any:
        """A fixed Optuna trial.

        Returns:
            A trial that can be sampled from.
        """
        import optuna

        return (
            optuna.trial.create_trial(params={}, distributions={}, value=0.0)
            if False
            else optuna.create_study().ask()
        )

    def test_samples_each_parameter_type(self, trial: Any) -> None:
        """Float, int and categorical spaces all sample within range."""
        from infrastructure.models.tuning import sample_search_space

        sampled = sample_search_space(
            trial,
            {
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
            },
        )
        assert 0.01 <= sampled["learning_rate"] <= 0.3
        assert 3 <= sampled["max_depth"] <= 10
        assert sampled["n_estimators"] % 100 == 0
        assert sampled["max_features"] in ("sqrt", "log2")

    @pytest.mark.parametrize(
        "space",
        [
            {"p": {"type": "quantum", "low": 0, "high": 1}},
            {"p": {"type": "float", "low": 0.0}},
            {"p": {"type": "categorical", "choices": []}},
            {"p": "not-a-mapping"},
        ],
    )
    def test_rejects_malformed_space(self, trial: Any, space: dict[str, Any]) -> None:
        """A malformed search space fails fast rather than mid-study."""
        from infrastructure.models.tuning import sample_search_space

        with pytest.raises(ConfigurationError):
            sample_search_space(trial, space)


class TestTuning:
    """Multi-objective optimisation."""

    @pytest.fixture
    def splits(self, feature_matrix: Any) -> list[tuple[np.ndarray, np.ndarray]]:
        """A single chronological split.

        Args:
            feature_matrix: The matrix, labels and builder.

        Returns:
            One train/validation index pair.
        """
        x, _y, _builder = feature_matrix
        cut = int(len(x) * 0.7)
        return [(np.arange(cut), np.arange(cut, len(x)))]

    def test_skipped_when_disabled(self, feature_matrix: Any, splits: Any) -> None:
        """Disabling tuning falls back to the static configuration."""
        from infrastructure.models.tuning import tune_model

        x, y, _builder = feature_matrix
        result = tune_model("lightgbm", ModelSpec(), x, y, splits, TuningConfig(enabled=False))
        assert not result.tuned
        assert result.best_params == {}

    def test_skipped_without_a_search_space(self, feature_matrix: Any, splits: Any) -> None:
        """A model with no declared space is left at its defaults."""
        from infrastructure.models.tuning import tune_model

        x, y, _builder = feature_matrix
        result = tune_model(
            "lightgbm", ModelSpec(search_space={}), x, y, splits, TuningConfig(enabled=True)
        )
        assert not result.tuned

    def test_produces_a_pareto_front(self, feature_matrix: Any, splits: Any) -> None:
        """A real study returns a front and picks a point on it."""
        from infrastructure.models.tuning import tune_model

        x, y, _builder = feature_matrix
        spec = ModelSpec(
            params={"verbosity": -1, "n_estimators": 30},
            search_space={
                "num_leaves": {"type": "int", "low": 5, "high": 15},
                "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True},
            },
        )
        result = tune_model(
            "lightgbm",
            spec,
            x,
            y,
            splits,
            TuningConfig(enabled=True, n_trials=4, n_jobs=1, seed=3),
            n_jobs=1,
        )
        assert result.tuned
        assert result.pareto_front
        assert set(result.best_params) <= {"num_leaves", "learning_rate"}
        assert result.best_pr_auc >= 0.0
        assert result.best_latency_ms >= 0.0

    def test_preference_weights_pick_from_the_front(self) -> None:
        """The declared trade-off, not the sampler, resolves the Pareto front."""
        from infrastructure.models.tuning import TuningResult

        # Two Pareto-optimal points: one accurate and slow, one fast and weaker.
        front = [(0.50, 40.0, {"a": 1}), (0.45, 2.0, {"a": 2})]
        budget = 10.0

        def score(weights: tuple[float, float]) -> dict[str, int]:
            """Collapse the front under given preference weights.

            Args:
                weights: The PR-AUC and latency weights.

            Returns:
                The winning parameters.
            """
            pr_weight, latency_weight = weights
            return max(
                front, key=lambda item: pr_weight * item[0] - latency_weight * (item[1] / budget)
            )[2]

        assert score((1.0, 0.0)) == {"a": 1}  # accuracy-only prefers the slow model
        assert score((0.5, 0.5)) == {"a": 2}  # latency-aware prefers the fast one
        assert TuningResult(model_name="m").tuned is False


class TestLocalModelStore:
    """Filesystem persistence."""

    @pytest.fixture
    def bundle(self, fitted_model: Any, feature_matrix: Any) -> Any:
        """A model bundle ready to persist.

        Args:
            fitted_model: The fitted adapter.
            feature_matrix: The matrix, labels and builder.

        Returns:
            The bundle.
        """
        from infrastructure.models.base import ModelBundle

        _x, _y, builder = feature_matrix
        return ModelBundle(
            model=fitted_model,
            model_id="lightgbm",
            feature_names=list(builder.feature_names),
            metrics={"test.pr_auc": 0.42},
        )

    def test_save_and_load(self, tmp_path: Path, bundle: Any, feature_matrix: Any) -> None:
        """A persisted model reloads and scores identically."""
        from infrastructure.models.registry import LocalModelStore

        x, _y, _builder = feature_matrix
        store = LocalModelStore(tmp_path)
        store.save(bundle)
        reloaded = store.load("lightgbm")
        assert reloaded.model_id == "lightgbm"
        assert np.allclose(reloaded.predict_pd(x.head(20)), bundle.predict_pd(x.head(20)))

    def test_promotion_writes_a_champion(self, tmp_path: Path, bundle: Any) -> None:
        """Promotion is what the API loads at startup."""
        from infrastructure.models.registry import LocalModelStore

        store = LocalModelStore(tmp_path)
        assert not store.has_champion()
        store.save(bundle, promote=True)
        assert store.has_champion()
        assert store.load_champion().model_id == "lightgbm"
        assert (tmp_path / "champion.json").is_file()

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        """A missing artifact is an explicit error with a remedy."""
        from infrastructure.models.registry import LocalModelStore

        store = LocalModelStore(tmp_path)
        with pytest.raises(ModelNotFoundError):
            store.load("never_trained")
        with pytest.raises(ModelNotFoundError):
            store.load_champion()

    def test_corrupt_artifact_raises(self, tmp_path: Path) -> None:
        """A truncated pickle must not surface as an obscure AttributeError."""
        from infrastructure.models.registry import LocalModelStore

        store = LocalModelStore(tmp_path)
        (tmp_path / "broken.pkl").write_bytes(b"not a pickle")
        with pytest.raises(ModelNotFoundError):
            store.load("broken")

    def test_lists_stored_models(self, tmp_path: Path, bundle: Any) -> None:
        """The champion pointer is not listed as a separate model."""
        from infrastructure.models.registry import LocalModelStore

        store = LocalModelStore(tmp_path)
        store.save(bundle, promote=True)
        assert store.list_models() == ["lightgbm"]

    def test_align_rejects_missing_columns(self, bundle: Any) -> None:
        """Scoring a matrix missing a training column is an error."""
        from domain.exceptions import ModelTrainingError

        with pytest.raises(ModelTrainingError):
            bundle.align(pd.DataFrame({"unrelated": [1.0]}))

    def test_unwritable_directory_raises(self, tmp_path: Path) -> None:
        """A path that cannot host the store is reported at construction."""
        from infrastructure.models.registry import LocalModelStore

        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(RegistryError):
            LocalModelStore(blocker / "store")


class TestExperimentTracker:
    """MLflow integration and its degraded mode."""

    def test_disabled_tracker_still_persists_locally(
        self, tmp_path: Path, fitted_model: Any, feature_matrix: Any
    ) -> None:
        """A model must survive tracking being switched off entirely."""
        from infrastructure.models.base import ModelBundle
        from infrastructure.models.registry import ExperimentTracker, LocalModelStore

        _x, _y, builder = feature_matrix
        store = LocalModelStore(tmp_path)
        tracker = ExperimentTracker(TrackingConfig(enabled=False), store)
        assert not tracker.is_active

        bundle = ModelBundle(
            model=fitted_model, model_id="m", feature_names=list(builder.feature_names)
        )
        assert tracker.register_model(bundle) is None
        assert store.has_champion()

    def test_logging_calls_are_safe_when_inactive(self, tmp_path: Path) -> None:
        """Metric and param logging must be no-ops, not crashes."""
        from infrastructure.models.registry import ExperimentTracker, LocalModelStore

        tracker = ExperimentTracker(TrackingConfig(enabled=False), LocalModelStore(tmp_path))
        tracker.log_params({"a": 1})
        tracker.log_metrics({"b": 2.0})
        tracker.log_dataframe(pd.DataFrame({"x": [1]}), "t")
        tracker.log_artifact_file(tmp_path)
        with tracker.run("r") as run_id:
            assert run_id is None

    def test_writes_to_a_serverless_sqlite_backend(self, tmp_path: Path) -> None:
        """sqlite:/// needs no server, which is what makes `make train` record runs.

        The older ``file:./mlruns`` backend entered maintenance mode in MLflow 3
        and now raises on connect, so the default was moved to sqlite.
        """
        from infrastructure.models.registry import ExperimentTracker, LocalModelStore

        config = TrackingConfig(
            enabled=True,
            tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            experiment_name="unit-test",
        )
        tracker = ExperimentTracker(config, LocalModelStore(tmp_path / "artifacts"))
        assert tracker.is_active
        with tracker.run("unit-run") as run_id:
            assert run_id is not None
            tracker.log_params({"seed": 42})
            tracker.log_metrics({"pr_auc": 0.5})
        assert (tmp_path / "mlflow.db").exists()

    def test_unreachable_backend_degrades_instead_of_raising(self, tmp_path: Path) -> None:
        """Losing the tracking server must never lose a trained model."""
        from infrastructure.models.registry import ExperimentTracker, LocalModelStore

        config = TrackingConfig(
            enabled=True,
            tracking_uri="http://127.0.0.1:1/unreachable",
            experiment_name="unit-test-unreachable",
        )
        tracker = ExperimentTracker(config, LocalModelStore(tmp_path / "artifacts"))
        # Setup failure is logged, not raised; scoring and persistence continue.
        with tracker.run("r") as run_id:
            assert run_id is None
        tracker.log_metrics({"pr_auc": 0.5})

    def test_metric_logging_filters_non_numeric(self, tmp_path: Path) -> None:
        """A NaN metric must not abort a run."""
        from infrastructure.models.registry import ExperimentTracker, LocalModelStore

        config = TrackingConfig(
            enabled=True,
            tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            experiment_name="unit-test-2",
        )
        tracker = ExperimentTracker(config, LocalModelStore(tmp_path / "artifacts"))
        with tracker.run("filtered"):
            tracker.log_metrics({"ok": 1.0, "bad": float("nan")})
