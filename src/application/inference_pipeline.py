"""Inference use case.

Scoring is not ``model.predict_proba``. Between a request and a reportable
probability of default sit several steps that exist because this is a regulated
decision:

1. **Contract validation** — reject a malformed batch before it reaches a model.
2. **Drift detection** — compare the batch against the training distribution and
   apply the fail-safe policy.
3. **Preprocessing** — with the *exact* builder fitted during training, carried
   inside the model bundle. A separately-constructed one would silently score on
   different scaling.
4. **Scoring** — raw model output.
5. **Domain adjustment** — the regulatory floor, the conservatism cap and the
   cost-optimal threshold, applied by :class:`~domain.services.PDCalculator`.
6. **Banding and explanation** — the outputs a credit decision is recorded with.

Run a batch over the demo dataset with::

    python -m application.inference_pipeline
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from domain.entities import (
    DATE_COLUMN,
    ENTITY_ID_COLUMN,
    DriftReport,
    RiskScore,
    ScoreExplanation,
)
from domain.exceptions import FinMLError, ModelNotFoundError
from domain.services import PDCalculator, RiskBandClassifier
from domain.value_objects import DriftSeverity, EntityId, ModelId
from infrastructure.config.schemas import RootConfig, register_configs
from infrastructure.data.drift_detector import DriftDetector
from infrastructure.data.validators import INFERENCE_SCHEMA, coerce_to_contract, validate_dataframe
from infrastructure.logging import configure_logging, get_logger, log_duration
from infrastructure.models.base import ModelBundle
from infrastructure.models.registry import LocalModelStore

__all__ = ["InferencePipeline", "ScoringResult", "main"]

_log = get_logger(__name__)


@dataclass(slots=True)
class ScoringResult:
    """The outcome of scoring one batch.

    Attributes:
        scores: One score per input row.
        drift: Drift assessment for the batch.
        model_id: Model that produced the scores.
        scored_at: Timestamp of scoring.
        threshold: Decision threshold applied.
    """

    scores: list[RiskScore] = field(default_factory=list)
    drift: DriftReport | None = None
    model_id: str = ""
    scored_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    threshold: float = 0.5

    @property
    def flagged_count(self) -> int:
        """Number of obligors above the threshold.

        Returns:
            The flagged count.
        """
        return sum(1 for s in self.scores if s.is_flagged)

    def to_frame(self) -> pd.DataFrame:
        """Render the scores as a dataframe.

        Returns:
            One row per scored obligor.
        """
        return pd.DataFrame(
            [
                {
                    "entity_id": str(s.entity_id),
                    "probability_of_default": s.pd_value,
                    "risk_band": s.band.value,
                    "is_flagged": s.is_flagged,
                    "threshold": s.threshold,
                    "model_id": str(s.model_id),
                    "drift_severity": s.drift_severity.name,
                    "scored_at": s.scored_at,
                }
                for s in self.scores
            ]
        )


class InferencePipeline:
    """Scores obligors with the promoted champion.

    Attributes:
        config: The composed configuration.
        bundle: The model bundle being served.
    """

    def __init__(self, config: RootConfig, bundle: ModelBundle) -> None:
        """Initialise the pipeline.

        Args:
            config: The composed configuration.
            bundle: The model bundle to serve.
        """
        from application.train_pipeline import (
            BUNDLE_DRIFT_REFERENCE,
            BUNDLE_PREPROCESSOR,
            BUNDLE_RISK_BANDS,
            BUNDLE_THRESHOLD,
        )

        self.config = config
        self.bundle = bundle
        self.preprocessor = bundle.extra.get(BUNDLE_PREPROCESSOR)
        self.threshold = float(bundle.extra.get(BUNDLE_THRESHOLD, 0.5))
        self.pd_calculator = PDCalculator(floor=config.model.pd_floor, cap=config.model.pd_cap)
        bands = bundle.extra.get(BUNDLE_RISK_BANDS) or config.model.risk_band_thresholds
        self.band_classifier = RiskBandClassifier(
            thresholds=(float(bands[0]), float(bands[1]), float(bands[2]), float(bands[3]))
        )

        self.drift_detector = DriftDetector(config.drift)
        reference = bundle.extra.get(BUNDLE_DRIFT_REFERENCE)
        if reference is not None and isinstance(reference, pd.DataFrame):
            self.drift_detector.fit(reference)
        else:
            _log.warning(
                "inference.no_drift_reference",
                model_id=bundle.model_id,
                detail="drift monitoring is inactive for this bundle",
            )

        if self.preprocessor is None:
            _log.warning(
                "inference.no_preprocessor",
                model_id=bundle.model_id,
                detail="inputs must already be in model feature space",
            )

    @classmethod
    def from_champion(cls, config: RootConfig) -> InferencePipeline:
        """Load the promoted champion and build a pipeline around it.

        Args:
            config: The composed configuration.

        Returns:
            The ready pipeline.

        Raises:
            ModelNotFoundError: If no champion has been promoted.
        """
        store = LocalModelStore(config.paths.artifacts)
        if not store.has_champion():
            raise ModelNotFoundError(
                "No champion model has been promoted",
                artifacts_dir=config.paths.artifacts,
                hint="Run: python -m application.train_pipeline",
            )
        bundle = store.load_champion()
        _log.info(
            "inference.champion_loaded",
            model_id=bundle.model_id,
            trained_at=bundle.trained_at.isoformat(),
            is_calibrated=bundle.is_calibrated,
            n_features=len(bundle.feature_names),
        )
        return cls(config, bundle)

    # -- Scoring -------------------------------------------------------------
    def enrich(self, df: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        """Apply NLP features using the same enricher training used.

        Cached values are preferred; ``compute_missing`` stays on so a genuinely
        new obligor is still scored on real features rather than contract
        defaults.

        Args:
            df: Raw input frame.
            news: News corpus covering the batch.

        Returns:
            The enriched frame.
        """
        from infrastructure.data.feature_store import FeatureStore
        from infrastructure.nlp.enrichment import NLPEnricher

        store = (
            FeatureStore(self.config.paths.feature_store)
            if self.config.features.feature_store_enabled
            else None
        )
        return NLPEnricher(self.config.nlp, store).enrich(df, news).panel

    def build_matrix(self, df: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
        """Turn raw input into the model's feature matrix.

        Args:
            df: Raw input frame.
            validate: Enforce the inference data contract first.

        Returns:
            The numeric feature matrix.
        """
        frame = coerce_to_contract(df, fill_missing=True)
        if validate and self.config.data.validate_schema:
            frame = validate_dataframe(frame, INFERENCE_SCHEMA, context="inference_batch")
        if self.preprocessor is None:
            return frame[self.bundle.feature_names]
        matrix: pd.DataFrame = self.preprocessor.transform(frame)
        return matrix

    def check_drift(self, matrix: pd.DataFrame) -> DriftReport:
        """Assess a matrix for drift and apply the fail-safe policy.

        Args:
            matrix: The model feature matrix for the batch.

        Returns:
            The drift report.

        Raises:
            DriftDetectedError: If the policy is ``block`` and drift is severe.
        """
        report = self.drift_detector.detect(matrix)
        return self.drift_detector.enforce(report)

    def score(
        self,
        df: pd.DataFrame,
        *,
        entity_ids: list[str] | None = None,
        check_drift: bool = True,
        overlay_log_odds: float = 0.0,
    ) -> ScoringResult:
        """Score a batch of obligors.

        Args:
            df: Raw input frame.
            entity_ids: Identifiers per row; taken from the frame when omitted.
            check_drift: Run the drift check before scoring.
            overlay_log_odds: Expert judgement shift applied to every PD.

        Returns:
            The scoring result.

        Raises:
            DriftDetectedError: If drift is severe and the policy is ``block``.
        """
        if df.empty:
            return ScoringResult(model_id=self.bundle.model_id, threshold=self.threshold)

        with log_duration("inference.score", n_rows=len(df), model=self.bundle.model_id):
            matrix = self.build_matrix(df)

            report: DriftReport | None = None
            severity = DriftSeverity.NONE
            if check_drift and self.drift_detector.is_fitted:
                report = self.check_drift(matrix)
                severity = report.severity

            probabilities = self.bundle.predict_pd(matrix)
            identifiers = self._resolve_ids(df, entity_ids)
            now = datetime.now(UTC)

            scores = [
                self._to_risk_score(
                    entity_id=identifier,
                    probability=float(probability),
                    severity=severity,
                    scored_at=now,
                    overlay_log_odds=overlay_log_odds,
                )
                for identifier, probability in zip(identifiers, probabilities, strict=True)
            ]

        result = ScoringResult(
            scores=scores,
            drift=report,
            model_id=self.bundle.model_id,
            scored_at=now,
            threshold=self.threshold,
        )
        _log.info(
            "inference.scored",
            n_rows=len(scores),
            model_id=self.bundle.model_id,
            n_flagged=result.flagged_count,
            mean_pd=round(float(np.mean([s.pd_value for s in scores])), 5),
            threshold=round(self.threshold, 4),
            drift_severity=severity.name,
        )
        return result

    def _to_risk_score(
        self,
        *,
        entity_id: str,
        probability: float,
        severity: DriftSeverity,
        scored_at: datetime,
        overlay_log_odds: float,
    ) -> RiskScore:
        """Turn a raw model probability into a reportable score.

        Args:
            entity_id: Obligor identifier.
            probability: Raw model output.
            severity: Drift severity for the batch.
            scored_at: Timestamp of scoring.
            overlay_log_odds: Expert judgement shift.

        Returns:
            The domain risk score.
        """
        pd_value = self.pd_calculator.from_probability(
            probability, overlay_log_odds=overlay_log_odds
        )
        return RiskScore(
            entity_id=EntityId(entity_id),
            probability_of_default=pd_value,
            model_id=ModelId(self.bundle.model_id),
            scored_at=scored_at,
            threshold=self.threshold,
            band=self.band_classifier.classify(pd_value),
            drift_severity=severity,
        )

    @staticmethod
    def _resolve_ids(df: pd.DataFrame, entity_ids: list[str] | None) -> list[str]:
        """Determine an identifier for each row.

        Args:
            df: The input frame.
            entity_ids: Explicit identifiers, if supplied.

        Returns:
            One identifier per row, synthesising positional ones if needed.
        """
        if entity_ids is not None:
            return [str(e) for e in entity_ids]
        if ENTITY_ID_COLUMN in df.columns:
            return [str(v) for v in df[ENTITY_ID_COLUMN]]
        return [f"row_{i}" for i in range(len(df))]

    # -- Explanation ---------------------------------------------------------
    def explain(
        self,
        df: pd.DataFrame,
        *,
        entity_id: str | None = None,
        method: str = "shap",
        background: pd.DataFrame | None = None,
    ) -> ScoreExplanation:
        """Explain a single obligor's score.

        Args:
            df: A one-row raw input frame.
            entity_id: Obligor identifier; taken from the frame when omitted.
            method: ``shap`` or ``lime``.
            background: Reference rows; the bundle's drift reference by default.

        Returns:
            The explanation.

        Raises:
            ExplainerError: If the method is unknown or explanation fails.
        """
        from application.train_pipeline import BUNDLE_DRIFT_REFERENCE
        from domain.exceptions import ExplainerError

        matrix = self.build_matrix(df)
        identifier = entity_id or self._resolve_ids(df, None)[0]
        reference = background
        if reference is None:
            reference = self.bundle.extra.get(BUNDLE_DRIFT_REFERENCE)
        if reference is None or not isinstance(reference, pd.DataFrame) or reference.empty:
            raise ExplainerError(
                "No background sample is available for explanation",
                model_id=self.bundle.model_id,
                hint="Retrain so the bundle carries a drift reference",
            )

        if method == "shap":
            from infrastructure.xai.shap_explainer import ShapExplainer

            shap_explainer = ShapExplainer(
                self.bundle.model,
                reference.sample(n=min(200, len(reference)), random_state=self.config.seed),
                feature_names=self.bundle.feature_names,
                backend=self.config.xai.shap_backend,
            )
            return shap_explainer.explain_instance(
                matrix.head(1), entity_id=identifier, model_id=self.bundle.model_id
            )

        if method == "lime":
            from infrastructure.xai.lime_explainer import LimeExplainer

            lime_explainer = LimeExplainer(
                self.bundle.model,
                reference,
                feature_names=self.bundle.feature_names,
                num_features=self.config.xai.lime_num_features,
                num_samples=self.config.xai.lime_num_samples,
                random_state=self.config.seed,
            )
            return lime_explainer.explain_instance(
                matrix.head(1), entity_id=identifier, model_id=self.bundle.model_id
            )

        raise ExplainerError("Unknown explanation method", method=method, known=["shap", "lime"])

    def counterfactuals(
        self, df: pd.DataFrame, *, entity_id: str | None = None, target_pd: float | None = None
    ) -> list[Any]:
        """Generate recourse options for one obligor.

        Args:
            df: A one-row raw input frame.
            entity_id: Obligor identifier; taken from the frame when omitted.
            target_pd: PD to reach; the configured default when omitted.

        Returns:
            The counterfactuals found.

        Raises:
            ExplainerError: If no background sample is available.
        """
        from application.train_pipeline import BUNDLE_DRIFT_REFERENCE
        from domain.exceptions import ExplainerError
        from infrastructure.xai.dice_explainer import CounterfactualGenerator

        reference = self.bundle.extra.get(BUNDLE_DRIFT_REFERENCE)
        if reference is None or not isinstance(reference, pd.DataFrame) or reference.empty:
            raise ExplainerError(
                "No background sample is available for counterfactuals",
                model_id=self.bundle.model_id,
            )

        matrix = self.build_matrix(df)
        identifier = entity_id or self._resolve_ids(df, None)[0]
        generator = CounterfactualGenerator(
            self.bundle.model,
            reference,
            feature_names=self.bundle.feature_names,
            actionable_features=list(self.config.xai.dice_features_to_vary) or None,
            target_pd=target_pd if target_pd is not None else self.config.xai.dice_desired_pd,
            total_cfs=self.config.xai.dice_total_cfs,
            method=self.config.xai.dice_method,
            random_state=self.config.seed,
        )
        return generator.generate(matrix.head(1), entity_id=identifier)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for batch inference over the demo dataset.

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
        pipeline = InferencePipeline.from_champion(config)
        from infrastructure.data.fetchers import build_fetcher

        dataset = build_fetcher(config.data).fetch()
        latest = dataset.panel[dataset.panel[DATE_COLUMN] == dataset.panel[DATE_COLUMN].max()]
        # Same enrichment as training, or the model scores a different
        # feature distribution than it was fitted on.
        enriched = pipeline.enrich(latest, dataset.news)
        result = pipeline.score(enriched)
    except FinMLError as exc:
        _log.error("inference.failed", **exc.to_dict())
        sys.exit(1)

    frame = result.to_frame()
    _log.info(
        "inference.batch_summary",
        n_scored=len(frame),
        n_flagged=result.flagged_count,
        mean_pd=round(float(frame["probability_of_default"].mean()), 5),
        band_counts=frame["risk_band"].value_counts().to_dict(),
        drift_severity=result.drift.severity.name if result.drift else "not_checked",
    )


register_configs()

if __name__ == "__main__":
    main()
