"""Structured logging.

``print`` is banned platform-wide (enforced by ruff's ``T20`` rule). Every event
goes through ``structlog`` and is emitted as a JSON line, which is what makes a
scoring decision reconstructible months later from log storage alone.

Call :func:`configure_logging` once per process, then obtain loggers with
:func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any, Literal

import structlog
from structlog.types import EventDict, Processor

__all__ = ["bind_run_context", "configure_logging", "get_logger", "log_duration"]

_CONFIGURED = False


def _add_service_context(_logger: object, _method: str, event_dict: EventDict) -> EventDict:
    """Stamp every event with the emitting service name.

    Args:
        _logger: Unused; part of the structlog processor signature.
        _method: Unused; part of the structlog processor signature.
        event_dict: The event under construction.

    Returns:
        The event with a ``service`` key.
    """
    event_dict.setdefault("service", "finml-platform")
    return event_dict


def _drop_color_message(_logger: object, _method: str, event_dict: EventDict) -> EventDict:
    """Remove uvicorn's duplicated ``color_message`` key.

    Args:
        _logger: Unused; part of the structlog processor signature.
        _method: Unused; part of the structlog processor signature.
        event_dict: The event under construction.

    Returns:
        The event without the redundant key.
    """
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = True,
    add_timestamp: bool = True,
    include_caller: bool = False,
) -> None:
    """Configure structlog and the stdlib logging bridge.

    Repeat calls are no-ops so importing a module that configures logging does
    not clobber an entrypoint's earlier configuration.

    Args:
        level: Root log level name, for example ``"INFO"``.
        json_output: Emit JSON lines; when false, use the console renderer.
        add_timestamp: Include an ISO-8601 UTC timestamp.
        include_caller: Include module, function and line number.

    Raises:
        ValueError: If ``level`` is not a valid log level name.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        # ValueError, not TypeError: the argument is the right type, the *value*
        # simply names no known level. getLevelName returns a string on miss.
        raise ValueError(f"Unknown log level: {level!r}")  # noqa: TRY004

    shared: list[Processor] = [
        # filter_by_level and add_logger_name both read the underlying stdlib
        # logger, which is why the factory below must be the stdlib one.
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_service_context,
        _drop_color_message,
        structlog.processors.StackInfoRenderer(),
    ]
    if add_timestamp:
        shared.append(structlog.processors.TimeStamper(fmt="iso", utc=True))
    if include_caller:
        shared.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            )
        )

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    # basicConfig first: the stdlib logger factory below emits through these
    # handlers, and configuring them afterwards would leave the first events
    # on the root logger's default handler.
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric_level, force=True)

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    for noisy in ("uvicorn.access", "mlflow", "matplotlib", "urllib3", "git"):
        logging.getLogger(noisy).setLevel(max(numeric_level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger.

    Args:
        name: Logger name; pass ``__name__`` from the calling module.

    Returns:
        A logger that emits structured events.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_run_context(**context: Any) -> None:
    """Bind key/value pairs to every subsequent event in this context.

    Used to attach a ``run_id`` or ``request_id`` once so that every downstream
    event carries it without threading the value through call signatures.

    Args:
        **context: Values to bind.
    """
    structlog.contextvars.bind_contextvars(**context)


def clear_run_context() -> None:
    """Drop everything bound by :func:`bind_run_context`."""
    structlog.contextvars.clear_contextvars()


class log_duration:  # noqa: N801 - used as a lowercase context manager
    """Context manager that logs how long a block took.

    Emits one event on entry and one on exit, including ``duration_ms``. On
    failure it logs the exception and re-raises: timing information must never
    swallow an error.

    Example:
        >>> with log_duration("train_model", model="catboost"):  # doctest: +SKIP
        ...     model.fit(x, y)

    Attributes:
        operation: Name recorded on both events.
        context: Extra key/value pairs attached to both events.
    """

    def __init__(self, operation: str, /, **context: Any) -> None:
        """Initialise the timer.

        Args:
            operation: Name recorded on both events.
            **context: Extra key/value pairs attached to both events.
        """
        self.operation = operation
        self.context = context
        self._log = get_logger("finml.timing")
        self._start: float = 0.0

    def __enter__(self) -> log_duration:
        """Record the start time and emit the started event.

        Returns:
            This context manager.
        """
        import time

        self._start = time.perf_counter()
        self._log.debug(f"{self.operation}.started", **self.context)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: object,
    ) -> Literal[False]:
        """Emit the completion event and propagate any exception.

        Args:
            exc_type: Exception class, if one is propagating.
            exc: Exception instance, if one is propagating.
            _tb: Traceback; unused.

        Returns:
            ``False`` so that exceptions are never suppressed. Typed as
            ``Literal[False]`` rather than ``bool`` so callers can rely on
            that: a plain ``bool`` tells a type checker the block might
            swallow an exception.
        """
        import time

        duration_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is None:
            self._log.info(
                f"{self.operation}.completed",
                duration_ms=round(duration_ms, 3),
                **self.context,
            )
        else:
            self._log.error(
                f"{self.operation}.failed",
                duration_ms=round(duration_ms, 3),
                error_type=exc_type.__name__,
                error=str(exc),
                **self.context,
            )
        return False


def timed(operation: str, **context: Any) -> Callable[..., Any]:
    """Decorate a function so each call is timed and logged.

    Args:
        operation: Name recorded on the emitted events.
        **context: Extra key/value pairs attached to the events.

    Returns:
        A decorator wrapping the target function in :class:`log_duration`.
    """
    import functools

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with log_duration(operation, **context):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
