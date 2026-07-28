"""Training use case.

Orchestrates the full path from raw data to a registered champion:

1. fetch and contract-validate the panel,
2. enrich with NLP features, caching the expensive ones in the feature store,
3. carve off a chronological test block — never a random one,
4. fit preprocessing on the training block only,
5. tune, fit and evaluate four base learners under leakage-aware folds,
6. stack them, calibrate the result, and benchmark everything,
7. select a champion against the credit-policy gates,
8. produce the model-risk evidence (SHAP, permutation, fairness, stability),
9. register the winner with its preprocessing and drift reference attached.

Step 9 matters as much as the modelling: a model artifact without its
preprocessor is not reproducible, and one without a drift reference cannot be
monitored. Both travel inside the bundle.

Run it with::

    python -m application.train_pipeline
    python -m application.train_pipeline models@model=fast data.n_entities=300
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from domain.entities import (
    DATE_COLUMN,
    ENTITY_ID_COLUMN,
    TARGET_COLUMN,
    ModelPerformance,
)
from domain.exceptions import FinMLError, ModelTrainingError
from domain.value_objects import CostMatrix, ModelId
from infrastructure.config.schemas import RootConfig, register_configs
from infrastructure.data.feature_store import FeatureStore
from infrastructure.data.fetchers import PanelDataset, build_fetcher
from infrastructure.data.preprocessing import FeatureMatrixBuilder
from infrastructure.data.splitters import build_splitter, time_based_holdout
from infrastructure.data.validators import TRAINING_SCHEMA, validate_dataframe
from infrastructure.logging import bind_run_context, configure_logging, get_logger, log_duration
from infrastructure.models.base import ModelBundle
from infrastructure.models.calibration import fit_best_calibrator
from infrastructure.models.ensemble import StackedEnsemble, build_meta_learner
from infrastructure.models.evaluation import benchmark_table, evaluate_model, select_champion
from infrastructure.models.registry import ExperimentTracker, LocalModelStore
from infrastructure.models.trainers import build_model
from infrastructure.models.tuning import tune_model

__all__ = ["TrainPipeline", "TrainingResult", "main"]

_log = get_logger(__name__)

#: Bundle keys the inference pipeline and API rely on.
BUNDLE_PREPROCESSOR = "preprocessor"
BUNDLE_DRIFT_REFERENCE = "drift_reference"
BUNDLE_THRESHOLD = "decision_threshold"
BUNDLE_RISK_BANDS = "risk_band_thresholds"


@dataclass(slots=True)
class TrainingResult:
    """Everything a training run produced.

    Attributes:
        champion_id: Identifier of the selected model.
        bundle: The champion bundle, ready to serve.
        performances: Every evaluation record across models and splits.
        benchmark: The comparison table.
        selection_notes: Why the champion was chosen.
        artifacts: Paths of files written.
        run_id: MLflow run id, when tracking was active.
    """

    champion_id: str
    bundle: ModelBundle
    performances: list[ModelPerformance] = field(default_factory=list)
    benchmark: pd.DataFrame = field(default_factory=pd.DataFrame)
    selection_notes: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    run_id: str | None = None


class TrainPipeline:
    """Runs the training use case.

    Attributes:
        config: The composed configuration.
    """

    def __init__(self, config: RootConfig) -> None:
        """Initialise the pipeline.

        Args:
            config: The composed configuration.
        """
        self.config = config
        self.cost_matrix = CostMatrix(
            false_negative_cost=config.model.cost.false_negative_cost,
            false_positive_cost=config.model.cost.false_positive_cost,
            true_positive_cost=config.model.cost.true_positive_cost,
            true_negative_cost=config.model.cost.true_negative_cost,
        )
        self.store = LocalModelStore(config.paths.artifacts)
        self.tracker = ExperimentTracker(config.tracking, self.store)
        self.reports_dir = Path(config.paths.reports)

    # -- Stages --------------------------------------------------------------
    def _load_data(self) -> PanelDataset:
        """Fetch and validate the panel.

        Returns:
            The validated dataset.
        """
        with log_duration("train.load_data", source=self.config.data.source):
            dataset = build_fetcher(self.config.data).fetch()
            if self.config.data.validate_schema:
                validated = validate_dataframe(
                    dataset.panel, TRAINING_SCHEMA, context="training_panel"
                )
                dataset = PanelDataset(panel=validated, macro=dataset.macro, news=dataset.news)
        _log.info(
            "train.data_loaded",
            n_rows=len(dataset.panel),
            n_entities=dataset.n_entities,
            default_rate=round(dataset.default_rate, 4),
            n_news=len(dataset.news),
        )
        return dataset

    def _enrich_with_nlp(self, dataset: PanelDataset) -> pd.DataFrame:
        """Apply NLP features via the shared enricher.

        The same component runs at serving time, which is what keeps the two
        feature distributions identical. Duplicating the logic here is how
        training-serving skew gets introduced.

        Args:
            dataset: The loaded dataset.

        Returns:
            The panel, with NLP columns overwritten where news was available.
        """
        from infrastructure.nlp.enrichment import NLPEnricher

        store = (
            FeatureStore(self.config.paths.feature_store)
            if self.config.features.feature_store_enabled
            else None
        )
        result = NLPEnricher(self.config.nlp, store).enrich(dataset.panel, dataset.news)
        return result.panel

    def _apply_smote(self, x: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """Oversample the minority class inside a training fold.

        Only ever called on a training fold. Applying SMOTE before the split
        would synthesise a minority point from neighbours that end up on the
        other side of it, leaking the validation distribution into training and
        inflating every metric.

        Args:
            x: Training fold features.
            y: Training fold labels.

        Returns:
            The resampled features and labels, unchanged on failure.
        """
        if self.config.features.imbalance_strategy != "smote":
            return x, y

        minority = int(min(np.sum(y == 0), np.sum(y == 1)))
        if minority <= 1:
            _log.warning(
                "train.smote_skipped",
                minority_count=minority,
                action="continuing without oversampling",
            )
            return x, y

        try:
            from imblearn.over_sampling import SMOTE

            k = min(self.config.features.smote_k_neighbors, max(minority - 1, 1))
            sampler = SMOTE(
                random_state=self.config.seed,
                k_neighbors=k,
                sampling_strategy=self.config.features.smote_sampling_strategy,
            )
            x_res, y_res = sampler.fit_resample(x, y)
        except Exception as exc:
            _log.warning(
                "train.smote_failed",
                error_type=type(exc).__name__,
                reason=str(exc)[:200],
                action="continuing without oversampling",
            )
            return x, y

        _log.info(
            "train.smote_applied",
            n_before=len(x),
            n_after=len(x_res),
            positive_rate_before=round(float(np.mean(y)), 4),
            positive_rate_after=round(float(np.mean(y_res)), 4),
        )
        return pd.DataFrame(x_res, columns=x.columns), np.asarray(y_res)

    def _enabled_models(self) -> dict[str, Any]:
        """Collect the enabled base learner specs.

        Returns:
            Model name mapped to its spec.

        Raises:
            ModelTrainingError: If every base learner is disabled.
        """
        model_cfg = self.config.model
        candidates = {
            "catboost": model_cfg.catboost,
            "xgboost": model_cfg.xgboost,
            "lightgbm": model_cfg.lightgbm,
            "random_forest": model_cfg.random_forest,
        }
        enabled = {name: spec for name, spec in candidates.items() if spec.enabled}
        if not enabled:
            raise ModelTrainingError("No base learners are enabled", candidates=sorted(candidates))
        return enabled

    def run(self) -> TrainingResult:
        """Execute the pipeline.

        Returns:
            The training result.

        Raises:
            FinMLError: If a stage fails unrecoverably.
        """
        started = datetime.now(UTC)
        run_name = self.config.run_name or f"train_{started:%Y%m%dT%H%M%SZ}"
        bind_run_context(run=run_name, seed=self.config.seed)
        np.random.seed(self.config.seed)

        dataset = self._load_data()
        panel = self._enrich_with_nlp(dataset)
        panel = panel.sort_values([DATE_COLUMN, ENTITY_ID_COLUMN]).reset_index(drop=True)

        # Chronological holdout: the test block is strictly the latest period.
        train_index, test_index = time_based_holdout(
            panel[DATE_COLUMN], test_size_frac=self.config.data.split.test_size_frac
        )
        train_df = panel.iloc[train_index].reset_index(drop=True)
        test_df = panel.iloc[test_index].reset_index(drop=True)

        builder = FeatureMatrixBuilder(self.config.features).fit(train_df)
        x_train, x_test = builder.transform(train_df), builder.transform(test_df)
        y_train = train_df[TARGET_COLUMN].to_numpy()
        y_test = test_df[TARGET_COLUMN].to_numpy()

        splitter = build_splitter(self.config.data.split)
        splits = list(splitter.split(x_train, y_train, groups=train_df[DATE_COLUMN]))
        if not splits:
            raise ModelTrainingError(
                "Cross-validation produced no usable folds",
                strategy=self.config.data.split.strategy,
                n_train_rows=len(x_train),
                hint="Reduce data.split.n_splits or widen the training window",
            )

        # The last fold's validation block is split again, chronologically, into
        # a calibration half and a scoring half.
        #
        # Both halves are unseen by the base fits, and the calibrator only ever
        # sees the first. Without this second split a calibrator fitted on the
        # same rows the calibrated model is then scored on reports near-perfect
        # validation metrics that collapse on test -- and champion selection
        # duly promotes the overfitted variant.
        fit_index, validation_index = splits[-1]
        calibration_index, evaluation_index = self._halve(validation_index)
        _log.info(
            "train.splits_ready",
            strategy=self.config.data.split.strategy,
            n_folds=len(splits),
            n_train=len(train_df),
            n_test=len(test_df),
            n_fit=len(fit_index),
            n_calibration=len(calibration_index),
            n_evaluation=len(evaluation_index),
            train_positive_rate=round(float(y_train.mean()), 4),
            test_positive_rate=round(float(y_test.mean()), 4),
        )

        tags = {
            "dataset_version": self.config.tracking.dataset_version,
            "split_strategy": self.config.data.split.strategy,
            "imbalance_strategy": self.config.features.imbalance_strategy,
            "source": self.config.data.source,
        }

        with self.tracker.run(run_name, tags=tags) as run_id:
            self.tracker.log_params(
                {
                    "seed": self.config.seed,
                    "n_train_rows": len(x_train),
                    "n_test_rows": len(x_test),
                    "n_features": len(builder.feature_names),
                    "split_strategy": self.config.data.split.strategy,
                    "n_splits": self.config.data.split.n_splits,
                    "tuning_enabled": self.config.model.tuning.enabled,
                    "fn_cost": self.cost_matrix.false_negative_cost,
                    "fp_cost": self.cost_matrix.false_positive_cost,
                }
            )

            fitted, held_out, performances = self._train_base_models(
                x_train, y_train, x_test, y_test, fit_index, evaluation_index
            )
            fitted, held_out, performances = self._train_ensemble(
                fitted,
                held_out,
                performances,
                x_train,
                y_train,
                x_test,
                y_test,
                splits,
                fit_index,
                evaluation_index,
            )
            fitted, performances = self._calibrate(
                fitted,
                held_out,
                performances,
                x_train,
                y_train,
                x_test,
                y_test,
                calibration_index,
                evaluation_index,
            )

            benchmark = benchmark_table(performances)
            champion_id, notes = select_champion(
                performances,
                split="validation",
                max_calibration_error=self.config.model.calibration.max_calibration_error,
                max_latency_ms=self.config.model.selection_max_latency_ms,
                min_pr_auc=self.config.model.selection_min_pr_auc,
            )

            bundle = self._build_bundle(
                str(champion_id), fitted, builder, x_train, validation_index, performances
            )
            artifacts = self._write_evidence(
                bundle, fitted, benchmark, x_train, x_test, y_test, test_df
            )

            for performance in performances:
                self.tracker.log_performance(performance)
            self.tracker.log_dataframe(benchmark, "benchmark")
            if self.config.training.register_best:
                self.tracker.register_model(bundle, promote=True)

        _log.info(
            "train.completed",
            champion=str(champion_id),
            n_models_evaluated=len({p.model_id for p in performances}),
            duration_s=round((datetime.now(UTC) - started).total_seconds(), 2),
            artifacts=list(artifacts),
        )
        return TrainingResult(
            champion_id=str(champion_id),
            bundle=bundle,
            performances=performances,
            benchmark=benchmark,
            selection_notes=notes,
            artifacts=artifacts,
            run_id=run_id,
        )

    @staticmethod
    def _halve(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split an index array in two, preserving order.

        Args:
            index: Positional indices, in chronological order.

        Returns:
            The earlier and later halves.
        """
        cut = max(len(index) // 2, 1)
        return index[:cut], index[cut:]

    def _train_base_models(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_test: pd.DataFrame,
        y_test: np.ndarray,
        fit_index: np.ndarray,
        evaluation_index: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any], list[ModelPerformance]]:
        """Tune, fit and evaluate every enabled base learner.

        Two variants of each model are produced and both are kept:

        * a **held-out** model fitted on ``fit_index`` only, which is what
          validation metrics and calibration are allowed to touch, and
        * a **full** model refitted on the whole training block, which is what
          gets served and what the test metrics describe.

        Args:
            x_train: Training feature matrix.
            y_train: Training labels.
            x_test: Test feature matrix.
            y_test: Test labels.
            fit_index: Rows the held-out models are fitted on.
            evaluation_index: Rows validation metrics are computed on.

        Returns:
            The full models, the held-out models, and the evaluation records.
        """
        fitted: dict[str, Any] = {}
        held_out: dict[str, Any] = {}
        performances: list[ModelPerformance] = []

        # Tuning folds are restricted to the fit block so the tuner never sees
        # the calibration or evaluation rows either.
        tuning_splits = self._inner_splits(x_train, y_train, fit_index)

        for name, spec in self._enabled_models().items():
            with log_duration("train.model", model=name):
                tuning = tune_model(
                    name,
                    spec,
                    x_train.iloc[fit_index],
                    y_train[fit_index],
                    tuning_splits,
                    self.config.model.tuning,
                    imbalance_strategy=self.config.features.imbalance_strategy,
                    cost_matrix=self.cost_matrix,
                    n_jobs=self.config.training.n_jobs,
                    focal_gamma=self.config.features.focal_gamma,
                    focal_alpha=self.config.features.focal_alpha,
                )

                x_fit, y_fit = self._apply_smote(x_train.iloc[fit_index], y_train[fit_index])
                validation_model = self._make_model(name, spec, tuning.best_params).fit(
                    x_fit, y_fit
                )
                latency = validation_model.measure_latency_ms(x_train, n_repeats=15)

                performances.append(
                    evaluate_model(
                        y_train[evaluation_index],
                        validation_model.predict_proba(x_train.iloc[evaluation_index])[:, 1],
                        model_id=name,
                        split="validation",
                        cost_matrix=self.cost_matrix,
                        latency_ms=latency,
                        n_bins=self.config.model.calibration.n_bins,
                    )[0]
                )

                x_full, y_full = self._apply_smote(x_train, y_train)
                model = self._make_model(name, spec, tuning.best_params).fit(x_full, y_full)
                performances.append(
                    evaluate_model(
                        y_test,
                        model.predict_proba(x_test)[:, 1],
                        model_id=name,
                        split="test",
                        cost_matrix=self.cost_matrix,
                        latency_ms=latency,
                        n_bins=self.config.model.calibration.n_bins,
                    )[0]
                )

                fitted[name] = model
                held_out[name] = validation_model
                if tuning.tuned:
                    self.tracker.log_params(
                        {f"{name}.{k}": v for k, v in tuning.best_params.items()}
                    )

        return fitted, held_out, performances

    def _make_model(self, name: str, spec: Any, overrides: dict[str, Any]) -> Any:
        """Build a fresh, unfitted adapter with the tuned hyperparameters.

        A method rather than a closure inside the training loop: a nested
        function would capture ``name`` and ``spec`` by reference, which is a
        standing invitation to a late-binding bug the first time the call is
        deferred.

        Args:
            name: Model name.
            spec: The model's configuration.
            overrides: Tuned hyperparameters, which win over ``spec.params``.

        Returns:
            The unfitted adapter.
        """
        return build_model(
            name,
            spec,
            random_state=self.config.seed,
            n_jobs=self.config.training.n_jobs,
            imbalance_strategy=self.config.features.imbalance_strategy,
            cost_matrix=self.cost_matrix,
            focal_gamma=self.config.features.focal_gamma,
            focal_alpha=self.config.features.focal_alpha,
            param_overrides=overrides,
        )

    def _inner_splits(
        self, x_train: pd.DataFrame, y_train: np.ndarray, fit_index: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Build cross-validation folds confined to the fit block.

        Args:
            x_train: Training feature matrix.
            y_train: Training labels.
            fit_index: Rows the folds must stay within.

        Returns:
            Folds expressed against ``fit_index`` positions.
        """
        subset = x_train.iloc[fit_index]
        splitter = build_splitter(self.config.data.split)
        groups = pd.Series(np.arange(len(subset)) // max(len(subset) // 8, 1))
        folds = list(splitter.split(subset, y_train[fit_index], groups=groups))
        return folds or [
            (np.arange(int(len(subset) * 0.7)), np.arange(int(len(subset) * 0.7), len(subset)))
        ]

    def _train_ensemble(
        self,
        fitted: dict[str, Any],
        held_out: dict[str, Any],
        performances: list[ModelPerformance],
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_test: pd.DataFrame,
        y_test: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        fit_index: np.ndarray,
        evaluation_index: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any], list[ModelPerformance]]:
        """Fit and evaluate the stacked ensemble.

        Args:
            fitted: Full models fitted on the whole training block.
            held_out: Models fitted on the fit block only.
            performances: Evaluation records so far.
            x_train: Training feature matrix.
            y_train: Training labels.
            x_test: Test feature matrix.
            y_test: Test labels.
            splits: Cross-validation folds.
            fit_index: Rows the held-out stack is fitted on.
            evaluation_index: Rows validation metrics are computed on.

        Returns:
            The full models, held-out models and records, extended with the stack.
        """
        if not self.config.model.ensemble.enabled or len(fitted) < 2:
            return fitted, held_out, performances

        ensemble_cfg = self.config.model.ensemble
        specs = self._enabled_models()

        def fresh_base() -> list[tuple[str, Any]]:
            """Build unfitted base learners for the stack.

            Returns:
                ``(name, adapter)`` pairs.
            """
            return [
                (
                    name,
                    build_model(
                        name,
                        spec,
                        random_state=self.config.seed,
                        n_jobs=self.config.training.n_jobs,
                        imbalance_strategy=self.config.features.imbalance_strategy,
                        cost_matrix=self.cost_matrix,
                    ),
                )
                for name, spec in specs.items()
            ]

        try:
            with log_duration("train.ensemble", meta_learner=ensemble_cfg.meta_learner):
                validation_ensemble = StackedEnsemble(
                    fresh_base(),
                    build_meta_learner(ensemble_cfg.meta_learner, random_state=self.config.seed),
                    passthrough=ensemble_cfg.passthrough,
                ).fit(
                    x_train.iloc[fit_index],
                    y_train[fit_index],
                    cv_splits=self._inner_splits(x_train, y_train, fit_index),
                )
                latency = validation_ensemble.measure_latency_ms(x_train, n_repeats=10)
                performances.append(
                    evaluate_model(
                        y_train[evaluation_index],
                        validation_ensemble.predict_proba(x_train.iloc[evaluation_index])[:, 1],
                        model_id="ensemble",
                        split="validation",
                        cost_matrix=self.cost_matrix,
                        latency_ms=latency,
                        n_bins=self.config.model.calibration.n_bins,
                    )[0]
                )

                ensemble = StackedEnsemble(
                    fresh_base(),
                    build_meta_learner(ensemble_cfg.meta_learner, random_state=self.config.seed),
                    passthrough=ensemble_cfg.passthrough,
                ).fit(x_train, y_train, cv_splits=splits)
                performances.append(
                    evaluate_model(
                        y_test,
                        ensemble.predict_proba(x_test)[:, 1],
                        model_id="ensemble",
                        split="test",
                        cost_matrix=self.cost_matrix,
                        latency_ms=latency,
                        n_bins=self.config.model.calibration.n_bins,
                    )[0]
                )
                fitted["ensemble"] = ensemble
                held_out["ensemble"] = validation_ensemble
                self.tracker.log_params(
                    {f"ensemble.weight.{k}": round(v, 4) for k, v in ensemble.meta_weights.items()}
                )
        except FinMLError as exc:
            # A failed stack should not cost the run its base models.
            _log.warning(
                "train.ensemble_failed", reason=exc.message, action="continuing with base models"
            )
        return fitted, held_out, performances

    def _calibrate(
        self,
        fitted: dict[str, Any],
        held_out: dict[str, Any],
        performances: list[ModelPerformance],
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_test: pd.DataFrame,
        y_test: np.ndarray,
        calibration_index: np.ndarray,
        evaluation_index: np.ndarray,
    ) -> tuple[dict[str, Any], list[ModelPerformance]]:
        """Calibrate the strongest model and evaluate the calibrated variant.

        The calibrator is fitted against the **held-out** model — the one that
        never saw the calibration rows — and scored on a disjoint evaluation
        block. Calibrating the full model on rows it was trained on produces a
        validation ECE near zero and a test ECE an order of magnitude worse,
        and champion selection then promotes the illusion.

        The calibrated model that is registered is therefore the held-out one,
        so its reported metrics describe exactly the artifact that gets served.

        Args:
            fitted: Full models fitted on the whole training block.
            held_out: Models fitted on the fit block only.
            performances: Evaluation records so far.
            x_train: Training feature matrix.
            y_train: Training labels.
            x_test: Test feature matrix.
            y_test: Test labels.
            calibration_index: Rows the calibrator is fitted on.
            evaluation_index: Rows validation metrics are computed on.

        Returns:
            The models and records, extended with the calibrated variant.
        """
        if not self.config.model.calibration.enabled or not held_out:
            return fitted, performances

        validation_records = {p.model_id: p for p in performances if p.split == "validation"}
        target = max(
            held_out,
            key=lambda name: (
                validation_records[ModelId(name)].pr_auc
                if ModelId(name) in validation_records
                else 0.0
            ),
        )
        base_latency = (
            validation_records[ModelId(target)].inference_latency_ms
            if ModelId(target) in validation_records
            else 0.0
        )

        try:
            with log_duration("train.calibration", model=target):
                report = fit_best_calibrator(
                    held_out[target],
                    x_train.iloc[calibration_index],
                    y_train[calibration_index],
                    methods=list(self.config.model.calibration.methods),
                    n_bins=self.config.model.calibration.n_bins,
                )
                if report.method == "none":
                    _log.info("train.calibration_not_beneficial", model=target)
                    return fitted, performances

                name = f"{target}_calibrated"
                fitted[name] = report.model
                for split, x_split, y_split in (
                    ("validation", x_train.iloc[evaluation_index], y_train[evaluation_index]),
                    ("test", x_test, y_test),
                ):
                    performances.append(
                        evaluate_model(
                            y_split,
                            report.model.predict_proba(x_split)[:, 1],
                            model_id=name,
                            split=split,
                            cost_matrix=self.cost_matrix,
                            latency_ms=base_latency,
                            n_bins=self.config.model.calibration.n_bins,
                        )[0]
                    )
                self.tracker.log_params(
                    {"calibration.method": report.method, "calibration.base_model": target}
                )
        except FinMLError as exc:
            _log.warning("train.calibration_failed", model=target, reason=exc.message)
        return fitted, performances

    def _build_bundle(
        self,
        champion_id: str,
        fitted: dict[str, Any],
        builder: FeatureMatrixBuilder,
        x_train: pd.DataFrame,
        reference_index: np.ndarray,
        performances: list[ModelPerformance],
    ) -> ModelBundle:
        """Assemble the champion bundle.

        The preprocessor and a drift reference sample travel with the model:
        without them the artifact can be loaded but neither reproduced nor
        monitored.

        Args:
            champion_id: Selected model name.
            fitted: Fitted models.
            builder: The fitted preprocessing builder.
            x_train: Training feature matrix.
            reference_index: Rows forming the drift reference — the most recent
                validated block, not a sample of all history. A scoring batch is
                one reporting period; comparing it against a reference pooled
                over many periods turns ordinary seasonal variation into a drift
                alert and trains operators to ignore the alarm.
            performances: Evaluation records.

        Returns:
            The bundle.

        Raises:
            ModelTrainingError: If the champion is not among the fitted models.
        """
        if champion_id not in fitted:
            raise ModelTrainingError(
                "Champion is not among the fitted models",
                champion=champion_id,
                fitted=sorted(fitted),
            )

        records = [p for p in performances if str(p.model_id) == champion_id]
        threshold = next(
            (p.threshold for p in records if p.split == "validation"),
            next((p.threshold for p in records if p.split == "test"), 0.5),
        )
        reference = x_train.iloc[reference_index]
        if len(reference) < self.config.drift.min_reference_size:
            _log.warning(
                "train.drift_reference_small",
                n_reference=len(reference),
                required=self.config.drift.min_reference_size,
                action="falling back to the full training block",
            )
            reference = x_train
        _log.info(
            "train.drift_reference",
            n_reference=len(reference),
            source="most recent validated block",
        )

        return ModelBundle(
            model=fitted[champion_id],
            model_id=champion_id,
            feature_names=list(builder.feature_names),
            params=getattr(fitted[champion_id], "params", {}) or {},
            metrics={
                f"{p.split}.{k}": v
                for p in records
                for k, v in p.as_dict().items()
                if isinstance(v, (int, float))
            },
            is_calibrated=champion_id.endswith("_calibrated"),
            calibration_method="isotonic_or_sigmoid" if champion_id.endswith("_calibrated") else "",
            dataset_version=self.config.tracking.dataset_version,
            extra={
                BUNDLE_PREPROCESSOR: builder,
                BUNDLE_DRIFT_REFERENCE: reference,
                BUNDLE_THRESHOLD: threshold,
                BUNDLE_RISK_BANDS: list(self.config.model.risk_band_thresholds),
            },
        )

    def _write_evidence(
        self,
        bundle: ModelBundle,
        fitted: dict[str, Any],
        benchmark: pd.DataFrame,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: np.ndarray,
        test_df: pd.DataFrame,
    ) -> dict[str, str]:
        """Produce the model-risk evidence pack.

        Every item here is optional to the *pipeline* but mandatory to the model
        file, so each is attempted independently and a failure is logged rather
        than propagated.

        Args:
            bundle: The champion bundle.
            fitted: Fitted models.
            benchmark: The comparison table.
            x_train: Training feature matrix.
            x_test: Test feature matrix.
            y_test: Test labels.
            test_df: The raw test frame, for the fairness attribute.

        Returns:
            Artifact name mapped to the path written.
        """
        artifacts: dict[str, str] = {}
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        benchmark_path = self.reports_dir / "benchmark.csv"
        try:
            benchmark.to_csv(benchmark_path, index=False)
            artifacts["benchmark"] = str(benchmark_path)
        except OSError as exc:
            _log.warning("train.benchmark_write_failed", reason=str(exc)[:200])

        if not self.config.training.compute_shap_on_train:
            return artifacts

        sample_size = min(self.config.training.shap_sample_size, len(x_test))
        sample = x_test.sample(n=sample_size, random_state=self.config.seed)

        try:
            from infrastructure.xai.shap_explainer import ShapExplainer

            explainer = ShapExplainer(
                bundle.model,
                x_train.sample(n=min(200, len(x_train)), random_state=self.config.seed),
                feature_names=bundle.feature_names,
                backend=self.config.xai.shap_backend,
            )
            result = explainer.explain(sample)
            importance_path = self.reports_dir / "shap_global_importance.csv"
            result.global_importance().to_csv(importance_path, index=False)
            artifacts["shap_importance"] = str(importance_path)

            plot_path = explainer.save_summary_plot(sample, self.reports_dir / "shap_summary.png")
            if plot_path is not None:
                artifacts["shap_summary_plot"] = str(plot_path)
                self.tracker.log_artifact_file(plot_path)
        except FinMLError as exc:
            _log.warning("train.shap_evidence_failed", reason=exc.message)

        try:
            from infrastructure.xai.permutation import compare_models

            comparison = compare_models(
                fitted,
                x_test,
                y_test,
                n_repeats=self.config.xai.permutation_repeats,
                random_state=self.config.seed,
            )
            if not comparison.empty:
                path = self.reports_dir / "permutation_importance.csv"
                comparison.to_csv(path)
                artifacts["permutation_importance"] = str(path)
        except FinMLError as exc:
            _log.warning("train.permutation_evidence_failed", reason=exc.message)

        try:
            from infrastructure.xai.fairness import assess_fairness

            attribute = self.config.xai.fairness_attribute
            if attribute in test_df.columns:
                report = assess_fairness(
                    y_test,
                    bundle.predict_pd(x_test),
                    test_df[attribute],
                    attribute=attribute,
                    threshold=float(bundle.extra.get(BUNDLE_THRESHOLD, 0.5)),
                    tolerance=self.config.xai.fairness_tolerance,
                )
                path = self.reports_dir / "fairness.csv"
                report.per_group.to_csv(path, index=False)
                artifacts["fairness"] = str(path)
                self.tracker.log_metrics(
                    {
                        "fairness.demographic_parity_gap": report.demographic_parity_gap,
                        "fairness.equalised_odds_gap": report.equalised_odds_gap,
                        "fairness.calibration_gap": report.calibration_gap,
                    }
                )
        except FinMLError as exc:
            _log.warning("train.fairness_evidence_failed", reason=exc.message)

        for artifact_path in artifacts.values():
            self.tracker.log_artifact_file(artifact_path)
        return artifacts


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for the training pipeline.

    Args:
        cfg: The composed configuration.
    """
    configure_logging(
        level=cfg.logging.level,
        json_output=cfg.logging.json_output,
        add_timestamp=cfg.logging.add_timestamp,
        include_caller=cfg.logging.include_caller,
    )
    config: RootConfig = OmegaConf.to_object(cfg)  # type: ignore[assignment]

    try:
        result = TrainPipeline(config).run()
    except FinMLError as exc:
        _log.error("train.failed", **exc.to_dict())
        sys.exit(1)

    _log.info(
        "train.summary",
        champion=result.champion_id,
        notes=result.selection_notes,
        artifacts=result.artifacts,
    )


register_configs()

if __name__ == "__main__":
    main()
