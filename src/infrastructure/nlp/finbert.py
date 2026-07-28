"""Financial sentiment extraction.

``ProsusAI/finbert`` is the intended backend. It is also a ~440MB download and a
``torch`` dependency, which is a poor thing to require before a test suite can
run. So the analyser lazily imports transformers and, when they are absent,
falls back to a Loughran-McDonald-style finance lexicon.

The fallback is deliberately visible, never silent: :attr:`FinBERTSentimentAnalyzer.is_fallback`
is exported to the feature store, stamped on the API response and logged on
every batch. A lexicon score and a transformer score are not interchangeable,
and a model card must not claim the latter when it used the former.

Note that general-purpose sentiment is actively wrong in finance. "Liability",
"restructuring" and "impairment" are neutral or positive in ordinary usage and
strongly negative on a balance sheet, which is the whole reason FinBERT exists —
and why the fallback lexicon is finance-specific rather than generic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Final

from domain.value_objects import SentimentScore
from infrastructure.config.schemas import NLPConfig
from infrastructure.logging import get_logger

__all__ = ["LEXICON_NEGATIVE", "LEXICON_POSITIVE", "FinBERTSentimentAnalyzer"]

_log = get_logger(__name__)

#: Finance-negative terms, following the spirit of the Loughran-McDonald list.
LEXICON_NEGATIVE: Final[frozenset[str]] = frozenset(
    {
        "default",
        "defaults",
        "defaulted",
        "bankruptcy",
        "bankrupt",
        "insolvency",
        "insolvent",
        "downgrade",
        "downgraded",
        "delinquent",
        "delinquency",
        "impairment",
        "impaired",
        "writedown",
        "write-down",
        "writeoff",
        "restructuring",
        "restructure",
        "covenant",
        "breach",
        "breached",
        "distress",
        "distressed",
        "liquidation",
        "receivership",
        "arrears",
        "shortfall",
        "deficit",
        "loss",
        "losses",
        "decline",
        "declined",
        "declining",
        "weak",
        "weakness",
        "weakened",
        "deteriorate",
        "deteriorating",
        "deterioration",
        "adverse",
        "concern",
        "concerns",
        "risk",
        "risks",
        "risky",
        "litigation",
        "lawsuit",
        "investigation",
        "probe",
        "fraud",
        "misconduct",
        "penalty",
        "fine",
        "fined",
        "recall",
        "layoff",
        "layoffs",
        "closure",
        "shutdown",
        "outage",
        "disruption",
        "downturn",
        "recession",
        "contraction",
        "miss",
        "missed",
        "warning",
        "warn",
        "warns",
        "warned",
        "cut",
        "cuts",
        "slump",
        "plunge",
        "plunged",
        "tumble",
        "volatility",
        "uncertainty",
        "uncertain",
        "pressure",
        "headwind",
        "headwinds",
        "dilution",
        "leverage",
        "overleveraged",
        "illiquid",
        "refinancing",
    }
)

#: Finance-positive terms.
LEXICON_POSITIVE: Final[frozenset[str]] = frozenset(
    {
        "profit",
        "profits",
        "profitable",
        "profitability",
        "growth",
        "grow",
        "growing",
        "grew",
        "gain",
        "gains",
        "increase",
        "increased",
        "increasing",
        "improve",
        "improved",
        "improving",
        "improvement",
        "strong",
        "strength",
        "strengthened",
        "robust",
        "solid",
        "upgrade",
        "upgraded",
        "outperform",
        "outperformed",
        "beat",
        "beats",
        "exceeded",
        "surpassed",
        "record",
        "expansion",
        "expand",
        "expanded",
        "recovery",
        "recovered",
        "rebound",
        "surplus",
        "dividend",
        "buyback",
        "efficient",
        "efficiency",
        "margin",
        "margins",
        "accretive",
        "deleveraging",
        "upside",
        "momentum",
        "tailwind",
        "resilient",
        "resilience",
        "stable",
        "stability",
        "confident",
        "confidence",
        "opportunity",
        "opportunities",
        "favourable",
        "favorable",
        "successful",
    }
)

#: Terms that invert the polarity of the word that follows.
_NEGATORS: Final[frozenset[str]] = frozenset(
    {"not", "no", "never", "without", "despite", "avoided", "avoids", "avoid", "less"}
)

_TOKEN_RE = re.compile(r"[a-z][a-z'-]+")


class FinBERTSentimentAnalyzer:
    """Produces :class:`SentimentScore` values from financial text.

    Attributes:
        config: NLP configuration.
    """

    def __init__(self, config: NLPConfig) -> None:
        """Initialise the analyser.

        The transformer is not loaded here — construction stays cheap so the
        object can be built during dependency wiring. The model loads on first
        use.

        Args:
            config: NLP configuration.
        """
        self.config = config
        self._pipeline: Any = None
        self._load_attempted = False
        self._fallback = False

    @property
    def is_fallback(self) -> bool:
        """Whether scores come from the lexicon rather than the transformer.

        Returns:
            ``True`` once a fallback score has been produced.
        """
        return self._fallback

    @property
    def backend(self) -> str:
        """Name of the backend actually in use.

        Returns:
            The model id, or ``"lexicon-fallback"``.
        """
        return "lexicon-fallback" if self._fallback else self.config.finbert_model

    def _ensure_pipeline(self) -> Any:
        """Load the transformer pipeline once, tolerating absence.

        Returns:
            The pipeline, or ``None`` when transformers are unavailable and the
            fallback is permitted.

        Raises:
            InfrastructureError: If loading fails and the fallback is disabled.
        """
        if self._load_attempted:
            return self._pipeline
        self._load_attempted = True

        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = self.config.device
            if device == "cuda":
                resolved = 0
            elif device == "auto":
                resolved = 0 if torch.cuda.is_available() else -1
            else:
                resolved = -1

            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.config.finbert_model,
                top_k=None,
                device=resolved,
                truncation=True,
                max_length=self.config.max_length,
            )
            _log.info(
                "nlp.finbert_loaded",
                model=self.config.finbert_model,
                device="cuda" if resolved == 0 else "cpu",
            )
        except Exception as exc:
            if not self.config.allow_fallback:
                from domain.exceptions import InfrastructureError

                raise InfrastructureError(
                    "FinBERT could not be loaded and fallback is disabled",
                    model=self.config.finbert_model,
                    error_type=type(exc).__name__,
                    reason=str(exc)[:300],
                    hint="Install the optional extra: pip install -e '.[nlp]'",
                ) from exc
            self._fallback = True
            self._pipeline = None
            _log.warning(
                "nlp.finbert_unavailable",
                model=self.config.finbert_model,
                error_type=type(exc).__name__,
                reason=str(exc)[:200],
                fallback="finance lexicon",
                detail="scores are flagged is_fallback=True downstream",
            )
        return self._pipeline

    def analyse(self, texts: Sequence[str]) -> list[SentimentScore]:
        """Score a batch of documents.

        Args:
            texts: Document texts.

        Returns:
            One :class:`SentimentScore` per document.
        """
        if not texts:
            return []

        pipeline = self._ensure_pipeline()
        if pipeline is None:
            return [self._lexicon_score(t) for t in texts]

        try:
            raw = pipeline(list(texts), batch_size=self.config.batch_size)
        except Exception as exc:
            # A runtime failure mid-batch is treated exactly like an unavailable
            # model: degrade, flag, continue.
            self._fallback = True
            _log.warning(
                "nlp.finbert_inference_failed",
                error_type=type(exc).__name__,
                reason=str(exc)[:200],
                fallback="finance lexicon",
                n_texts=len(texts),
            )
            return [self._lexicon_score(t) for t in texts]

        return [self._from_pipeline_output(item) for item in raw]

    @staticmethod
    def _from_pipeline_output(scores: Any) -> SentimentScore:
        """Convert one FinBERT output into a :class:`SentimentScore`.

        Args:
            scores: The per-label scores for a single document.

        Returns:
            The corresponding sentiment score.
        """
        mapping = {str(entry["label"]).lower(): float(entry["score"]) for entry in scores}
        positive = mapping.get("positive", 0.0)
        negative = mapping.get("negative", 0.0)
        neutral = mapping.get("neutral", 0.0)

        total = positive + negative + neutral
        if total <= 0.0:
            return SentimentScore.neutral()
        return SentimentScore(
            positive_prob=positive / total,
            negative_prob=negative / total,
            neutral_prob=neutral / total,
            n_documents=1,
        )

    def _lexicon_score(self, text: str) -> SentimentScore:
        """Score one document with the finance lexicon.

        Polarity counts are mapped through a softmax-like transform so the
        output is a genuine distribution rather than a hand-normalised ratio,
        and a document with no hits lands on neutral instead of on a coin flip.

        Args:
            text: Document text.

        Returns:
            The lexicon-derived sentiment score.
        """
        self._fallback = True
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return SentimentScore.neutral()

        positive = negative = 0.0
        for index, token in enumerate(tokens):
            polarity = (
                1.0 if token in LEXICON_POSITIVE else -1.0 if token in LEXICON_NEGATIVE else 0.0
            )
            if polarity == 0.0:
                continue
            # A negator within the preceding three tokens flips the polarity.
            window = tokens[max(0, index - 3) : index]
            if any(w in _NEGATORS for w in window):
                polarity = -polarity
            if polarity > 0:
                positive += 1.0
            else:
                negative += 1.0

        hits = positive + negative
        if hits == 0.0:
            return SentimentScore.neutral()

        # Confidence grows with evidence but saturates, so one hit in a long
        # document does not produce a 100% confident classification.
        strength = min(hits / 4.0, 1.0)
        net = (positive - negative) / hits
        exp_pos = math.exp(2.0 * net * strength)
        exp_neg = math.exp(-2.0 * net * strength)
        scale = exp_pos + exp_neg

        # Polar mass sums to `strength`; the remainder is neutral, so weak
        # evidence stays near neutral instead of asserting a confident call.
        positive_prob = strength * exp_pos / scale
        negative_prob = strength * exp_neg / scale
        return SentimentScore(
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=max(0.0, 1.0 - positive_prob - negative_prob),
            n_documents=1,
        )

    def aggregate(self, texts: Sequence[str]) -> SentimentScore:
        """Score a document set and average into one score.

        Args:
            texts: Document texts, for example every headline in the lookback
                window for one obligor.

        Returns:
            The mean sentiment across documents, with ``n_documents`` set to the
            number aggregated. Neutral when the set is empty.
        """
        scores = self.analyse(texts)
        if not scores:
            return SentimentScore.neutral()
        n = len(scores)
        return SentimentScore(
            positive_prob=sum(s.positive_prob for s in scores) / n,
            negative_prob=sum(s.negative_prob for s in scores) / n,
            neutral_prob=sum(s.neutral_prob for s in scores) / n,
            n_documents=n,
        )


@lru_cache(maxsize=4)
def _cached_analyser(model: str, allow_fallback: bool, device: str) -> FinBERTSentimentAnalyzer:
    """Build and memoise an analyser.

    Args:
        model: Hugging Face model id.
        allow_fallback: Whether the lexicon fallback is permitted.
        device: Requested device.

    Returns:
        The memoised analyser.
    """
    return FinBERTSentimentAnalyzer(
        NLPConfig(finbert_model=model, allow_fallback=allow_fallback, device=device)
    )


def get_analyser(config: NLPConfig) -> FinBERTSentimentAnalyzer:
    """Return a shared analyser for a configuration.

    Loading FinBERT costs seconds and hundreds of megabytes; sharing the
    instance keeps that to once per process.

    Args:
        config: NLP configuration.

    Returns:
        The shared analyser.
    """
    return _cached_analyser(config.finbert_model, config.allow_fallback, config.device)
