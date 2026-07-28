"""Exception hierarchy for the platform.

Two roots hang off :class:`FinMLError`:

* :class:`DomainError` — a business rule or invariant was violated. These are
  deterministic and are *never* retried; they signal that the caller supplied
  something the business model considers impossible.
* :class:`InfrastructureError` — an external concern failed (network, disk,
  registry, third-party library). These may be transient and are candidates for
  retry or fallback.

Every exception carries a structured ``context`` mapping so that failures can be
emitted as JSON log events without string parsing, which is what SR 11-7 audit
trails require.
"""

from __future__ import annotations

from typing import Any, TypeAlias

__all__ = [
    "CalculationError",
    "ConfigurationError",
    "DataFetchError",
    "DomainError",
    "DriftDetectedError",
    "ExplainerError",
    "FeatureStoreError",
    "FinMLError",
    "InfrastructureError",
    "InvariantViolationError",
    "ModelNotFoundError",
    "ModelTrainingError",
    "RegistryError",
    "ScenarioError",
    "SchemaValidationError",
    "ValidationError",
]

ErrorContext: TypeAlias = dict[str, Any]


class FinMLError(Exception):
    """Root of every exception raised by this platform.

    Attributes:
        message: Human-readable description of the failure.
        context: Structured key/value detail suitable for JSON logging.
    """

    def __init__(self, message: str, /, **context: Any) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the failure.
            **context: Arbitrary structured detail attached to the event.
        """
        super().__init__(message)
        self.message: str = message
        self.context: ErrorContext = dict(context)

    def __str__(self) -> str:
        """Render the message followed by sorted context pairs.

        Returns:
            The formatted error string.
        """
        if not self.context:
            return self.message
        rendered = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({rendered})"

    def __repr__(self) -> str:
        """Return an unambiguous representation.

        Returns:
            The formatted representation string.
        """
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the error for structured logging or an API error body.

        Returns:
            A mapping with ``error_type``, ``message`` and ``context`` keys.
        """
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# Domain errors — business invariants
# ---------------------------------------------------------------------------
class DomainError(FinMLError):
    """A business rule or domain invariant was violated."""


class ValidationError(DomainError):
    """A value object rejected the value it was constructed with."""


class InvariantViolationError(DomainError):
    """An aggregate reached a state its invariants forbid."""


class CalculationError(DomainError):
    """A domain calculation could not produce a meaningful result."""


class ScenarioError(DomainError):
    """A macro scenario is undefined, incomplete or internally inconsistent."""


# ---------------------------------------------------------------------------
# Infrastructure errors — external concerns
# ---------------------------------------------------------------------------
class InfrastructureError(FinMLError):
    """An external dependency failed."""


class ConfigurationError(InfrastructureError):
    """Configuration is missing, malformed or mutually inconsistent."""


class DataFetchError(InfrastructureError):
    """A remote data source could not be read."""


class SchemaValidationError(InfrastructureError):
    """A dataframe violated its declared data contract."""


class FeatureStoreError(InfrastructureError):
    """The feature store could not be read from or written to."""


class ModelTrainingError(InfrastructureError):
    """A model failed to fit."""


class ModelNotFoundError(InfrastructureError):
    """No model artifact exists for the requested identifier or stage."""


class RegistryError(InfrastructureError):
    """The experiment tracker or model registry rejected an operation."""


class ExplainerError(InfrastructureError):
    """An explainability backend failed to produce attributions."""


class DriftDetectedError(InfrastructureError):
    """Input drift breached the configured fail-safe threshold.

    Raised by the inference pipeline only when the drift policy is set to
    ``block``; under the ``warn`` policy the same condition is logged and
    surfaced on the response instead of interrupting scoring.
    """
