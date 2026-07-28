"""Shared NLP feature enrichment.

Both the training and inference pipelines route through this one component, and
that is the entire point. If training overwrites ``sentiment_compound`` from a
FinBERT pass while serving leaves the raw column in place, the model scores a
population it was never fitted on — training-serving skew, which shows up as a
drift alert if you are lucky and as quiet miscalibration if you are not.

The feature store makes the shared path affordable: training computes each
``(entity, date)`` score once and writes it, and inference reads it back rather
than paying for another transformer pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from domain.entities import DATE_COLUMN, ENTITY_ID_COLUMN
from infrastructure.config.schemas import NLPConfig
from infrastructure.data.feature_store import FeatureStore
from infrastructure.logging import get_logger, log_duration

__all__ = ["EnrichmentResult", "NLPEnricher"]

_log = get_logger(__name__)

#: Feature-store group holding computed NLP features.
FEATURE_SET_NAME = "nlp_features"

#: Columns this enricher owns and will overwrite where it has a value.
NLP_COLUMNS = (
    "sentiment_compound",
    "sentiment_uncertainty",
    "sentiment_negative_prob",
    "news_volume",
    "topic_credit_risk_score",
    "topic_liquidity_risk_score",
    "topic_operational_risk_score",
)


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    """Outcome of an enrichment pass.

    Attributes:
        panel: The panel with NLP columns applied.
        n_cached: Keys served from the feature store.
        n_computed: Keys that required a model pass.
        row_coverage: Share of panel rows that received values.
        used_fallback: Whether a degraded NLP backend produced any value.
    """

    panel: pd.DataFrame
    n_cached: int = 0
    n_computed: int = 0
    row_coverage: float = 0.0
    used_fallback: bool = False


class NLPEnricher:
    """Applies sentiment and topic features to a panel.

    Attributes:
        config: NLP configuration.
        store: Feature store used for caching; ``None`` disables caching.
    """

    def __init__(self, config: NLPConfig, store: FeatureStore | None = None) -> None:
        """Initialise the enricher.

        Args:
            config: NLP configuration.
            store: Feature store used for caching.
        """
        self.config = config
        self.store = store
        self._keys = [ENTITY_ID_COLUMN, DATE_COLUMN]

    def enrich(
        self, panel: pd.DataFrame, news: pd.DataFrame, *, compute_missing: bool = True
    ) -> EnrichmentResult:
        """Apply NLP features to a panel.

        Args:
            panel: The panel to enrich.
            news: News corpus, with ``published_at`` or ``as_of_date``.
            compute_missing: Run the models for uncached keys. Set false on a
                latency-sensitive serving path to use only what is cached.

        Returns:
            The enriched panel and cache statistics.
        """
        if not self.config.enabled or news.empty or panel.empty:
            _log.info(
                "nlp.enrichment_skipped",
                enabled=self.config.enabled,
                n_news=len(news),
                n_panel=len(panel),
            )
            return EnrichmentResult(panel=panel)

        with log_duration("nlp.enrichment", n_news=len(news)):
            documents = news.rename(columns={"published_at": DATE_COLUMN})
            missing_keys = [k for k in self._keys if k not in documents.columns]
            if missing_keys:
                _log.warning("nlp.news_missing_keys", missing=missing_keys)
                return EnrichmentResult(panel=panel)

            wanted = documents[self._keys].drop_duplicates()

            cached = pd.DataFrame()
            missing = wanted
            if self.store is not None:
                cached, missing = self.store.fetch_keyed(
                    FEATURE_SET_NAME, wanted, key_columns=self._keys
                )

            computed = pd.DataFrame()
            used_fallback = False
            if compute_missing and not missing.empty:
                computed, used_fallback = self._compute(documents, missing)
                if self.store is not None and not computed.empty:
                    self.store.upsert_keyed(
                        FEATURE_SET_NAME,
                        computed,
                        key_columns=self._keys,
                        producer="finbert+bertopic",
                        is_fallback=used_fallback,
                    )

            frames = [f for f in (cached, computed) if not f.empty]
            if not frames:
                return EnrichmentResult(panel=panel)

            features = pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=self._keys, keep="last"
            )
            enriched, coverage = self._apply(panel, features)

        _log.info(
            "nlp.enriched",
            n_cached=len(cached),
            n_computed=len(computed),
            row_coverage=round(coverage, 4),
            used_fallback=used_fallback,
        )
        return EnrichmentResult(
            panel=enriched,
            n_cached=len(cached),
            n_computed=len(computed),
            row_coverage=coverage,
            used_fallback=used_fallback,
        )

    def _apply(self, panel: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Overlay computed features onto the panel.

        Args:
            panel: The panel to enrich.
            features: One row of NLP features per key.

        Returns:
            The enriched panel and the share of rows that received a value.
        """
        # Align key dtypes; a datetime/string mismatch merges to all-NaN silently.
        left, right = panel.copy(), features.copy()
        for frame in (left, right):
            frame[ENTITY_ID_COLUMN] = frame[ENTITY_ID_COLUMN].astype(str)
            frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])

        merged = left.merge(right, on=self._keys, how="left", suffixes=("", "_nlp"))
        coverage = 0.0
        for column in NLP_COLUMNS:
            candidate = f"{column}_nlp"
            if candidate not in merged.columns:
                continue
            mask = merged[candidate].notna()
            coverage = max(coverage, float(mask.mean()))
            if column in merged.columns:
                merged.loc[mask, column] = merged.loc[mask, candidate]
            else:
                merged[column] = merged[candidate]
            merged = merged.drop(columns=[candidate])
        return merged, coverage

    def _compute(self, documents: pd.DataFrame, missing: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Run the NLP stack over the documents for uncached keys.

        Args:
            documents: The news corpus.
            missing: Keys with no cached features.

        Returns:
            One row of features per key, and whether a fallback backend was used.
        """
        from infrastructure.nlp.finbert import FinBERTSentimentAnalyzer
        from infrastructure.nlp.topic_model import RiskTopicModel

        subset = documents.merge(missing, on=self._keys, how="inner")
        if subset.empty:
            return pd.DataFrame(), False

        analyser = FinBERTSentimentAnalyzer(self.config)
        topic_model = RiskTopicModel(self.config)
        topic_model.fit(subset["headline"].astype(str).tolist())

        rows: list[dict[str, Any]] = []
        for key_values, group in subset.groupby(self._keys, sort=False):
            texts = group["headline"].astype(str).tolist()
            sentiment = analyser.aggregate(texts)
            topics = topic_model.score(" ".join(texts))
            record: dict[str, Any] = dict(zip(self._keys, key_values, strict=True))
            record.update(
                {
                    "sentiment_compound": sentiment.compound,
                    "sentiment_uncertainty": sentiment.uncertainty,
                    "sentiment_negative_prob": sentiment.negative_prob,
                    "news_volume": float(len(texts)),
                    **topics.as_features(),
                }
            )
            rows.append(record)

        _log.info(
            "nlp.computed",
            n_keys=len(rows),
            sentiment_backend=analyser.backend,
            topic_backend=topic_model.backend,
        )
        return pd.DataFrame(rows), analyser.is_fallback or topic_model.is_fallback
