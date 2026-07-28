"""Risk-topic modelling.

Turns a news corpus into three tabular features — ``topic_credit_risk_score``,
``topic_liquidity_risk_score``, ``topic_operational_risk_score`` — so that *what*
the coverage is about enters the model alongside how positive it sounds. The
distinction carries real signal: "covenant renegotiation" and "product recall"
are both negative, but only one is a credit event.

BERTopic is the preferred backend. Unsupervised topics do not arrive labelled,
so discovered clusters are mapped onto the three configured risk categories by
keyword overlap with their seed terms. When BERTopic (or its sentence-transformer
backbone) is unavailable, the seeded keyword scorer runs directly — the same
mapping step, without the clustering in front of it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from infrastructure.config.schemas import NLPConfig
from infrastructure.logging import get_logger

__all__ = ["RiskTopicModel", "TopicScores"]

_log = get_logger(__name__)

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z'-]+")


@dataclass(frozen=True, slots=True)
class TopicScores:
    """Risk-category intensities for one document.

    Attributes:
        scores: Category name mapped to intensity in ``[0, 1]``.
        dominant_topic: Highest-scoring category, or ``"none"``.
        is_fallback: Whether the keyword tier produced the scores.
    """

    scores: Mapping[str, float] = field(default_factory=dict)
    dominant_topic: str = "none"
    is_fallback: bool = False

    def __post_init__(self) -> None:
        """Normalise the score mapping to a plain dict."""
        object.__setattr__(self, "scores", dict(self.scores))

    def as_features(self, prefix: str = "topic_", suffix: str = "_risk_score") -> dict[str, float]:
        """Render the scores as model feature columns.

        Args:
            prefix: Feature name prefix.
            suffix: Feature name suffix.

        Returns:
            Feature name mapped to intensity, matching the domain contract's
            ``topic_*_risk_score`` columns.
        """
        return {f"{prefix}{name}{suffix}": value for name, value in self.scores.items()}


class RiskTopicModel:
    """Assigns documents to configured risk categories.

    Attributes:
        config: NLP configuration supplying the risk categories and seed terms.
    """

    def __init__(self, config: NLPConfig) -> None:
        """Initialise the model.

        Args:
            config: NLP configuration.
        """
        self.config = config
        self._model: Any = None
        self._topic_map: dict[int, str] = {}
        self._fitted = False
        self._fallback = False

    @property
    def is_fallback(self) -> bool:
        """Whether the keyword tier is in use.

        Returns:
            ``True`` once keyword scoring has run.
        """
        return self._fallback

    @property
    def backend(self) -> str:
        """Name of the backend in use.

        Returns:
            The BERTopic backbone name, or ``"keyword-fallback"``.
        """
        return "keyword-fallback" if self._fallback else f"bertopic:{self.config.topic_model}"

    @property
    def categories(self) -> tuple[str, ...]:
        """Configured risk categories.

        Returns:
            The category names.
        """
        return tuple(self.config.risk_topics.keys())

    def fit(self, documents: Sequence[str]) -> RiskTopicModel:
        """Fit BERTopic and map its clusters onto the risk categories.

        Args:
            documents: Corpus to fit on.

        Returns:
            This model, fitted or degraded to the keyword tier.

        Raises:
            InfrastructureError: If BERTopic fails and the fallback is disabled.
        """
        # BERTopic needs a corpus with enough documents to cluster; below that
        # its clusters are noise and the keyword tier is strictly better.
        if len(documents) < 20:
            self._fallback = True
            self._fitted = True
            _log.info(
                "nlp.topic_corpus_too_small",
                n_documents=len(documents),
                minimum=20,
                fallback="seeded keywords",
            )
            return self

        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(self.config.topic_model)
            seed_lists = [list(v) for v in self.config.risk_topics.values()]
            self._model = BERTopic(
                embedding_model=embedder,
                seed_topic_list=seed_lists,
                calculate_probabilities=True,
                verbose=False,
            )
            self._model.fit(list(documents))
            self._topic_map = self._map_topics_to_categories()
            self._fitted = True
            _log.info(
                "nlp.topic_model_fitted",
                backend=self.config.topic_model,
                n_documents=len(documents),
                n_topics=len(self._topic_map),
            )
        except Exception as exc:
            if not self.config.allow_fallback:
                from domain.exceptions import InfrastructureError

                raise InfrastructureError(
                    "BERTopic could not be fitted and fallback is disabled",
                    model=self.config.topic_model,
                    error_type=type(exc).__name__,
                    reason=str(exc)[:300],
                    hint="Install the optional extra: pip install -e '.[nlp]'",
                ) from exc
            self._fallback = True
            self._fitted = True
            self._model = None
            _log.warning(
                "nlp.topic_model_unavailable",
                model=self.config.topic_model,
                error_type=type(exc).__name__,
                reason=str(exc)[:200],
                fallback="seeded keywords",
            )
        return self

    def _map_topics_to_categories(self) -> dict[int, str]:
        """Label each discovered cluster with a risk category.

        Clusters come out of BERTopic unlabelled. Each is scored against the
        configured seed terms and takes the best-matching category; a cluster
        matching nothing is left unassigned rather than forced into one.

        Returns:
            Topic id mapped to category name.
        """
        mapping: dict[int, str] = {}
        if self._model is None:
            return mapping

        topic_info = self._model.get_topics()
        for topic_id in topic_info:
            if topic_id == -1:  # BERTopic's outlier cluster
                continue
            words = {str(w).lower() for w, _ in self._model.get_topic(topic_id) or []}
            best_category, best_overlap = "", 0
            for category, seeds in self.config.risk_topics.items():
                seed_tokens = {t for s in seeds for t in _TOKEN_RE.findall(str(s).lower())}
                overlap = len(words & seed_tokens)
                if overlap > best_overlap:
                    best_category, best_overlap = category, overlap
            if best_category:
                mapping[topic_id] = best_category
        return mapping

    def score(self, text: str) -> TopicScores:
        """Score one document across the risk categories.

        Args:
            text: Document text.

        Returns:
            The category intensities.
        """
        if not self._fitted:
            self.fit([text])

        if self._model is not None and not self._fallback:
            try:
                return self._bertopic_score(text)
            except Exception as exc:
                self._fallback = True
                _log.warning(
                    "nlp.topic_inference_failed",
                    error_type=type(exc).__name__,
                    reason=str(exc)[:200],
                    fallback="seeded keywords",
                )
        return self._keyword_score(text)

    def score_many(self, texts: Sequence[str]) -> list[TopicScores]:
        """Score a batch of documents.

        Args:
            texts: Document texts.

        Returns:
            One score set per document.
        """
        return [self.score(t) for t in texts]

    def _bertopic_score(self, text: str) -> TopicScores:
        """Score a document using the fitted BERTopic model.

        Args:
            text: Document text.

        Returns:
            The category intensities.
        """
        topics, probabilities = self._model.transform([text])
        scores = dict.fromkeys(self.categories, 0.0)

        probability_row = probabilities[0] if probabilities is not None else None
        if probability_row is not None and hasattr(probability_row, "__len__"):
            for topic_id, probability in enumerate(probability_row):
                category = self._topic_map.get(topic_id)
                if category:
                    scores[category] += float(probability)
        else:
            category = self._topic_map.get(int(topics[0]))
            if category:
                scores[category] = 1.0

        total = sum(scores.values())
        if total > 1.0:
            scores = {k: v / total for k, v in scores.items()}
        dominant = max(scores, key=lambda k: scores[k]) if any(scores.values()) else "none"
        return TopicScores(scores=scores, dominant_topic=dominant, is_fallback=False)

    def _keyword_score(self, text: str) -> TopicScores:
        """Score a document by seed-keyword density.

        Multi-word seeds are matched as substrings; single-word seeds are matched
        against tokens so that "funding" does not fire on "refunding".

        Args:
            text: Document text.

        Returns:
            The category intensities.
        """
        self._fallback = True
        lowered = text.lower()
        tokens = set(_TOKEN_RE.findall(lowered))

        raw: dict[str, float] = {}
        for category, seeds in self.config.risk_topics.items():
            hits = 0
            for seed in seeds:
                term = str(seed).lower()
                if " " in term:
                    hits += 1 if term in lowered else 0
                else:
                    hits += 1 if term in tokens else 0
            # Saturating transform: the third mention of "covenant" adds less
            # than the first, matching how the score is read.
            raw[category] = 1.0 - math.exp(-1.2 * hits)

        total = sum(raw.values())
        scores = {k: v / total for k, v in raw.items()} if total > 1.0 else raw
        dominant = max(scores, key=lambda k: scores[k]) if any(scores.values()) else "none"
        return TopicScores(scores=scores, dominant_topic=dominant, is_fallback=True)
