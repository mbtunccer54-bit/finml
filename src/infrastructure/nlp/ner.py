"""Financial named-entity extraction.

Pulls the concrete facts out of a news item — which company, how much money,
which date, what percentage — so a downstream alert can say *"Acme, $2.1bn
impairment, filed 2024-03-14"* rather than *"negative sentiment detected"*.

spaCy is the preferred backend. As with :mod:`infrastructure.nlp.finbert` it is
optional, and a regex extractor covers the money/percentage/date cases when it
is absent. The regex tier is genuinely good at those three (they have rigid
surface forms) and genuinely weak at organisation names, so the fallback is
reported rather than presented as equivalent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from infrastructure.config.schemas import NLPConfig
from infrastructure.logging import get_logger

__all__ = ["ExtractedEntity", "FinancialNERExtractor", "NERResult"]

_log = get_logger(__name__)

#: Money: optional currency symbol/code, digits, optional scale word.
_MONEY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:(?:US)?[$€£¥]|\b(?:USD|EUR|GBP|JPY|TRY)\b)\s?"
    r"\d[\d,]*(?:\.\d+)?\s?(?:billion|bn|million|mn|m|thousand|k|trillion|tn)?",
    re.IGNORECASE,
)
_PERCENT_RE: Final[re.Pattern[str]] = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?\s?(?:%|percent|percentage points|pp|bps|basis points)",
    re.IGNORECASE,
)
_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|Q[1-4]\s?(?:FY)?\d{2,4})\b",
    re.IGNORECASE,
)
# Capitalised runs ending in a corporate suffix; deliberately conservative.
_ORG_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:[A-Z][\w&.-]*\s){0,4}"
    r"[A-Z][\w&.-]*\s"
    r"(?:Inc|Corp|Corporation|Ltd|Limited|LLC|PLC|plc|NV|SA|AG|Group|Holdings|Bank)\b\.?"
)


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    """One entity mention.

    Attributes:
        text: The matched surface text.
        label: Entity type, for example ``ORG``, ``MONEY``, ``PERCENT``, ``DATE``.
        start: Character offset of the match start.
        end: Character offset of the match end.
        confidence: Extractor confidence in ``[0, 1]``.
    """

    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class NERResult:
    """Entities found in one document.

    Attributes:
        entities: The mentions, in order of appearance.
        is_fallback: Whether the regex tier produced them.
        backend: Name of the backend used.
    """

    entities: tuple[ExtractedEntity, ...] = field(default_factory=tuple)
    is_fallback: bool = False
    backend: str = ""

    def by_label(self, label: str) -> tuple[ExtractedEntity, ...]:
        """Filter mentions by entity type.

        Args:
            label: Entity type to keep.

        Returns:
            The matching mentions.
        """
        wanted = label.upper()
        return tuple(e for e in self.entities if e.label.upper() == wanted)

    @property
    def organisations(self) -> tuple[str, ...]:
        """Distinct organisation names, in order of first appearance.

        Returns:
            The organisation surface forms.
        """
        return tuple(dict.fromkeys(e.text for e in self.by_label("ORG")))

    @property
    def monetary_amounts(self) -> tuple[str, ...]:
        """Distinct money mentions.

        Returns:
            The money surface forms.
        """
        return tuple(dict.fromkeys(e.text for e in self.by_label("MONEY")))

    def counts(self) -> dict[str, int]:
        """Count mentions per entity type.

        Returns:
            Entity type mapped to mention count.
        """
        out: dict[str, int] = {}
        for entity in self.entities:
            out[entity.label] = out.get(entity.label, 0) + 1
        return out


class FinancialNERExtractor:
    """Extracts financial entities from text.

    Attributes:
        config: NLP configuration.
    """

    #: spaCy labels worth keeping for financial reporting.
    _KEEP_LABELS: Final[frozenset[str]] = frozenset(
        {"ORG", "MONEY", "PERCENT", "DATE", "GPE", "CARDINAL", "LAW"}
    )

    def __init__(self, config: NLPConfig) -> None:
        """Initialise the extractor.

        Args:
            config: NLP configuration.
        """
        self.config = config
        self._nlp: Any = None
        self._load_attempted = False
        self._fallback = False

    @property
    def is_fallback(self) -> bool:
        """Whether the regex tier is in use.

        Returns:
            ``True`` once a regex extraction has run.
        """
        return self._fallback

    @property
    def backend(self) -> str:
        """Name of the backend in use.

        Returns:
            The spaCy pipeline name, or ``"regex-fallback"``.
        """
        return "regex-fallback" if self._fallback else self.config.ner_model

    def _ensure_model(self) -> Any:
        """Load the spaCy pipeline once, tolerating absence.

        Returns:
            The loaded pipeline, or ``None`` when unavailable.

        Raises:
            InfrastructureError: If loading fails and the fallback is disabled.
        """
        if self._load_attempted:
            return self._nlp
        self._load_attempted = True

        try:
            import spacy

            self._nlp = spacy.load(self.config.ner_model)
            _log.info("nlp.ner_loaded", model=self.config.ner_model)
        except Exception as exc:
            if not self.config.allow_fallback:
                from domain.exceptions import InfrastructureError

                raise InfrastructureError(
                    "spaCy NER model could not be loaded and fallback is disabled",
                    model=self.config.ner_model,
                    error_type=type(exc).__name__,
                    reason=str(exc)[:300],
                    hint=f"python -m spacy download {self.config.ner_model}",
                ) from exc
            self._fallback = True
            self._nlp = None
            _log.warning(
                "nlp.ner_unavailable",
                model=self.config.ner_model,
                error_type=type(exc).__name__,
                reason=str(exc)[:200],
                fallback="regex patterns",
                detail="organisation recall is materially lower on this tier",
            )
        return self._nlp

    def extract(self, text: str) -> NERResult:
        """Extract entities from one document.

        Args:
            text: Document text.

        Returns:
            The entities found.
        """
        if not text or not text.strip():
            return NERResult(backend=self.backend)

        model = self._ensure_model()
        if model is None:
            return self._regex_extract(text)

        try:
            doc = model(text[: self.config.max_length * 8])
        except Exception as exc:
            self._fallback = True
            _log.warning(
                "nlp.ner_inference_failed",
                error_type=type(exc).__name__,
                reason=str(exc)[:200],
                fallback="regex patterns",
            )
            return self._regex_extract(text)

        entities = tuple(
            ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start=int(ent.start_char),
                end=int(ent.end_char),
            )
            for ent in doc.ents
            if ent.label_ in self._KEEP_LABELS
        )
        return NERResult(entities=entities, is_fallback=False, backend=self.config.ner_model)

    def extract_many(self, texts: Sequence[str]) -> list[NERResult]:
        """Extract entities from a batch of documents.

        Args:
            texts: Document texts.

        Returns:
            One result per document.
        """
        return [self.extract(t) for t in texts]

    def _regex_extract(self, text: str) -> NERResult:
        """Extract entities using the regex tier.

        Args:
            text: Document text.

        Returns:
            The entities found, sorted by position.
        """
        self._fallback = True
        found: list[ExtractedEntity] = []

        # Confidence reflects how rigid each surface form is: money and
        # percentages are near-unambiguous, organisation names are guesses.
        for pattern, label, confidence in (
            (_MONEY_RE, "MONEY", 0.95),
            (_PERCENT_RE, "PERCENT", 0.95),
            (_DATE_RE, "DATE", 0.90),
            (_ORG_RE, "ORG", 0.55),
        ):
            for match in pattern.finditer(text):
                found.append(
                    ExtractedEntity(
                        text=match.group(0).strip(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                    )
                )

        found.sort(key=lambda e: e.start)
        return NERResult(entities=tuple(found), is_fallback=True, backend="regex-fallback")
