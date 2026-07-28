"""Leakage-aware cross-validation for panel data.

``KFold`` is wrong here because it shuffles the future into the past.
``TimeSeriesSplit`` is better but still insufficient: it puts the validation
block immediately after the training block, so an observation whose outcome
window straddles the boundary contributes its label to both sides. On a
quarterly panel with a twelve-month default horizon that is a guaranteed leak,
and it is exactly the kind of optimism that does not survive production.

Two guards fix it, following López de Prado's *Advances in Financial Machine
Learning*:

* **Purging** — drop training observations whose outcome window overlaps the
  validation block.
* **Embargo** — additionally drop training observations shortly *after* the
  validation block, because serial correlation leaks information backwards.

All four splitters implement the sklearn splitter interface, so they drop
straight into ``cross_val_score``, ``GridSearchCV`` or ``StackingClassifier``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from domain.exceptions import ConfigurationError
from infrastructure.config.schemas import SplitConfig
from infrastructure.logging import get_logger

__all__ = [
    "BlockedTimeSeriesSplit",
    "ExpandingWindowSplit",
    "PurgedKFold",
    "SlidingWindowSplit",
    "TimeAwareSplitter",
    "build_splitter",
    "time_based_holdout",
]

_log = get_logger(__name__)


def _time_positions(n_samples: int, groups: Any | None) -> np.ndarray:
    """Map each row to an ordinal position on the shared timeline.

    Rows sharing a timestamp must never be split apart — in a panel that would
    put the same quarter on both sides of the boundary — so positions are
    assigned per distinct timestamp, not per row.

    Args:
        n_samples: Number of rows.
        groups: Per-row timestamps. When ``None``, row order is assumed to be
            chronological.

    Returns:
        An integer array of timeline positions, one per row.

    Raises:
        ConfigurationError: If ``groups`` length does not match ``n_samples``.
    """
    if groups is None:
        _log.debug("splitter.no_groups", detail="assuming rows are time-ordered")
        return np.arange(n_samples)

    values = pd.Series(np.asarray(groups).ravel())
    if len(values) != n_samples:
        raise ConfigurationError(
            "Time group length does not match the sample count",
            n_groups=len(values),
            n_samples=n_samples,
        )
    # sort=True makes the codes themselves the chronological rank, so rows
    # keep their original order while carrying a timeline position.
    codes, _ = pd.factorize(values, sort=True)
    return np.asarray(codes)


class TimeAwareSplitter(ABC):
    """Base class for the time-aware splitters.

    Attributes:
        n_splits: Number of folds produced.
    """

    def __init__(self, n_splits: int = 5) -> None:
        """Initialise the splitter.

        Args:
            n_splits: Number of folds.

        Raises:
            ConfigurationError: If ``n_splits`` is below two.
        """
        if n_splits < 2:
            raise ConfigurationError("n_splits must be at least 2", n_splits=n_splits)
        self.n_splits = n_splits

    def get_n_splits(self, _x: Any = None, _y: Any = None, _groups: Any = None) -> int:
        """Return the number of folds.

        Args:
            _x: Unused; part of the sklearn splitter interface.
            _y: Unused; part of the sklearn splitter interface.
            _groups: Unused; part of the sklearn splitter interface.

        Returns:
            The configured fold count.
        """
        return self.n_splits

    @abstractmethod
    def split(
        self, x: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train/validation index pairs.

        Args:
            x: Feature matrix; only its length is used.
            y: Labels; unused, present for interface compatibility.
            groups: Per-row timestamps defining the timeline.

        Yields:
            ``(train_index, validation_index)`` positional index arrays.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_fold(
        train_idx: np.ndarray, val_idx: np.ndarray, fold: int, name: str
    ) -> bool:
        """Log and reject a fold that is unusable.

        Args:
            train_idx: Training indices.
            val_idx: Validation indices.
            fold: Fold number, for logging.
            name: Splitter name, for logging.

        Returns:
            ``True`` when the fold has rows on both sides.
        """
        if len(train_idx) == 0 or len(val_idx) == 0:
            _log.warning(
                "splitter.empty_fold",
                splitter=name,
                fold=fold,
                n_train=len(train_idx),
                n_val=len(val_idx),
                detail="skipped; widen the data or reduce n_splits",
            )
            return False
        return True


class PurgedKFold(TimeAwareSplitter):
    """K-fold over contiguous time blocks, with purging and an embargo.

    Each fold uses one contiguous block of the timeline for validation and
    everything else for training, minus a purge window immediately before the
    block and an embargo window immediately after it.

    Attributes:
        n_splits: Number of folds.
        purge_frac: Fraction of the timeline purged before each validation block.
        embargo_frac: Fraction of the timeline embargoed after each block.
    """

    def __init__(
        self, n_splits: int = 5, *, purge_frac: float = 0.01, embargo_frac: float = 0.01
    ) -> None:
        """Initialise the splitter.

        Args:
            n_splits: Number of folds.
            purge_frac: Fraction of the timeline purged before each block.
            embargo_frac: Fraction of the timeline embargoed after each block.

        Raises:
            ConfigurationError: If either fraction is outside ``[0, 0.5)``.
        """
        super().__init__(n_splits)
        for name, value in (("purge_frac", purge_frac), ("embargo_frac", embargo_frac)):
            if not 0.0 <= value < 0.5:
                raise ConfigurationError(
                    "Purge and embargo fractions must lie in [0, 0.5)",
                    field=name,
                    value=value,
                )
        self.purge_frac = purge_frac
        self.embargo_frac = embargo_frac

    def split(
        self, x: Any, y: Any = None, groups: Any = None  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield purged and embargoed train/validation index pairs.

        Args:
            x: Feature matrix; only its length is used.
            y: Unused; present for interface compatibility.
            groups: Per-row timestamps defining the timeline.

        Yields:
            ``(train_index, validation_index)`` positional index arrays.
        """
        n_samples = len(x)
        positions = _time_positions(n_samples, groups)
        n_periods = int(positions.max()) + 1

        bounds = np.linspace(0, n_periods, self.n_splits + 1).astype(int)
        purge = max(round(self.purge_frac * n_periods), 0)
        embargo = max(round(self.embargo_frac * n_periods), 0)

        for fold, (start, end) in enumerate(pairwise(bounds)):
            if end <= start:
                continue
            val_mask = (positions >= start) & (positions < end)
            # Exclude the purge window before and the embargo window after.
            excluded = (positions >= start - purge) & (positions < end + embargo)
            train_mask = ~excluded

            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(val_mask)
            if not self._validate_fold(train_idx, val_idx, fold, "purged_kfold"):
                continue
            yield train_idx, val_idx


class BlockedTimeSeriesSplit(TimeAwareSplitter):
    """Strictly forward-looking blocks: train on the past, validate on the next block.

    Unlike :class:`PurgedKFold` this never trains on data after the validation
    block, so it mirrors deployment exactly. It is the more conservative choice
    and the one to quote in a model validation report.

    Attributes:
        n_splits: Number of folds.
        embargo_frac: Fraction of the timeline embargoed between train and validation.
    """

    def __init__(self, n_splits: int = 5, *, embargo_frac: float = 0.01) -> None:
        """Initialise the splitter.

        Args:
            n_splits: Number of folds.
            embargo_frac: Fraction of the timeline embargoed between the blocks.

        Raises:
            ConfigurationError: If ``embargo_frac`` is outside ``[0, 0.5)``.
        """
        super().__init__(n_splits)
        if not 0.0 <= embargo_frac < 0.5:
            raise ConfigurationError(
                "embargo_frac must lie in [0, 0.5)", embargo_frac=embargo_frac
            )
        self.embargo_frac = embargo_frac

    def split(
        self, x: Any, y: Any = None, groups: Any = None  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield forward-looking train/validation index pairs.

        Args:
            x: Feature matrix; only its length is used.
            y: Unused; present for interface compatibility.
            groups: Per-row timestamps defining the timeline.

        Yields:
            ``(train_index, validation_index)`` positional index arrays.
        """
        positions = _time_positions(len(x), groups)
        n_periods = int(positions.max()) + 1
        bounds = np.linspace(0, n_periods, self.n_splits + 2).astype(int)
        embargo = max(round(self.embargo_frac * n_periods), 0)

        for fold in range(self.n_splits):
            train_end = bounds[fold + 1]
            val_start, val_end = bounds[fold + 1], bounds[fold + 2]
            train_idx = np.flatnonzero(positions < train_end - embargo)
            val_idx = np.flatnonzero((positions >= val_start) & (positions < val_end))
            if not self._validate_fold(train_idx, val_idx, fold, "blocked"):
                continue
            yield train_idx, val_idx


class ExpandingWindowSplit(TimeAwareSplitter):
    """Anchored walk-forward: the training window grows, validation rolls on.

    Matches a model retrained on all history at each cycle.

    Attributes:
        n_splits: Number of folds.
        min_train_frac: Fraction of the timeline in the first training window.
        embargo_frac: Fraction embargoed between train and validation.
    """

    def __init__(
        self, n_splits: int = 5, *, min_train_frac: float = 0.3, embargo_frac: float = 0.01
    ) -> None:
        """Initialise the splitter.

        Args:
            n_splits: Number of folds.
            min_train_frac: Fraction of the timeline in the first training window.
            embargo_frac: Fraction embargoed between train and validation.

        Raises:
            ConfigurationError: If ``min_train_frac`` is outside ``(0, 1)``.
        """
        super().__init__(n_splits)
        if not 0.0 < min_train_frac < 1.0:
            raise ConfigurationError(
                "min_train_frac must lie in (0, 1)", min_train_frac=min_train_frac
            )
        self.min_train_frac = min_train_frac
        self.embargo_frac = embargo_frac

    def split(
        self, x: Any, y: Any = None, groups: Any = None  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield expanding-window train/validation index pairs.

        Args:
            x: Feature matrix; only its length is used.
            y: Unused; present for interface compatibility.
            groups: Per-row timestamps defining the timeline.

        Yields:
            ``(train_index, validation_index)`` positional index arrays.
        """
        positions = _time_positions(len(x), groups)
        n_periods = int(positions.max()) + 1
        first_train_end = max(round(self.min_train_frac * n_periods), 1)
        remaining = n_periods - first_train_end
        step = max(remaining // self.n_splits, 1)
        embargo = max(round(self.embargo_frac * n_periods), 0)

        for fold in range(self.n_splits):
            train_end = first_train_end + fold * step
            val_end = min(train_end + step, n_periods)
            if train_end >= n_periods:
                break
            train_idx = np.flatnonzero(positions < train_end - embargo)
            val_idx = np.flatnonzero((positions >= train_end) & (positions < val_end))
            if not self._validate_fold(train_idx, val_idx, fold, "expanding"):
                continue
            yield train_idx, val_idx


class SlidingWindowSplit(TimeAwareSplitter):
    """Rolling walk-forward: a fixed-width training window slides forward.

    Matches a model deliberately retrained on recent history only, which is the
    right choice when the data-generating process drifts.

    Attributes:
        n_splits: Number of folds.
        window_frac: Training window width as a fraction of the timeline.
        embargo_frac: Fraction embargoed between train and validation.
    """

    def __init__(
        self, n_splits: int = 5, *, window_frac: float = 0.4, embargo_frac: float = 0.01
    ) -> None:
        """Initialise the splitter.

        Args:
            n_splits: Number of folds.
            window_frac: Training window width as a fraction of the timeline.
            embargo_frac: Fraction embargoed between train and validation.

        Raises:
            ConfigurationError: If ``window_frac`` is outside ``(0, 1)``.
        """
        super().__init__(n_splits)
        if not 0.0 < window_frac < 1.0:
            raise ConfigurationError("window_frac must lie in (0, 1)", window_frac=window_frac)
        self.window_frac = window_frac
        self.embargo_frac = embargo_frac

    def split(
        self, x: Any, y: Any = None, groups: Any = None  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield sliding-window train/validation index pairs.

        Args:
            x: Feature matrix; only its length is used.
            y: Unused; present for interface compatibility.
            groups: Per-row timestamps defining the timeline.

        Yields:
            ``(train_index, validation_index)`` positional index arrays.
        """
        positions = _time_positions(len(x), groups)
        n_periods = int(positions.max()) + 1
        window = max(round(self.window_frac * n_periods), 1)
        remaining = n_periods - window
        step = max(remaining // self.n_splits, 1)
        embargo = max(round(self.embargo_frac * n_periods), 0)

        for fold in range(self.n_splits):
            train_start = fold * step
            train_end = train_start + window
            val_end = min(train_end + step, n_periods)
            if train_end >= n_periods:
                break
            train_idx = np.flatnonzero(
                (positions >= train_start) & (positions < train_end - embargo)
            )
            val_idx = np.flatnonzero((positions >= train_end) & (positions < val_end))
            if not self._validate_fold(train_idx, val_idx, fold, "sliding"):
                continue
            yield train_idx, val_idx


def build_splitter(config: SplitConfig) -> TimeAwareSplitter:
    """Construct the splitter named by the configuration.

    Args:
        config: Cross-validation configuration.

    Returns:
        The configured splitter.

    Raises:
        ConfigurationError: If ``config.strategy`` names no known splitter.
    """
    strategy = config.strategy.strip().lower()
    if strategy == "purged_kfold":
        return PurgedKFold(
            config.n_splits, purge_frac=config.purge_frac, embargo_frac=config.embargo_frac
        )
    if strategy == "blocked":
        return BlockedTimeSeriesSplit(config.n_splits, embargo_frac=config.embargo_frac)
    if strategy == "expanding":
        return ExpandingWindowSplit(
            config.n_splits,
            min_train_frac=config.min_train_frac,
            embargo_frac=config.embargo_frac,
        )
    if strategy == "sliding":
        return SlidingWindowSplit(
            config.n_splits, window_frac=config.window_frac, embargo_frac=config.embargo_frac
        )
    raise ConfigurationError(
        "Unknown cross-validation strategy",
        strategy=config.strategy,
        known=["purged_kfold", "blocked", "expanding", "sliding"],
    )


def time_based_holdout(
    times: pd.Series, *, test_size_frac: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Split off the most recent slice of the timeline as a test set.

    The final holdout is always the *latest* period, never a random sample: the
    question a model owner has to answer is how the model behaves on data it
    could not have seen, and a random split silently answers a different one.

    Args:
        times: Per-row timestamps.
        test_size_frac: Share of distinct time points held out.

    Returns:
        ``(train_index, test_index)`` positional index arrays.

    Raises:
        ConfigurationError: If ``test_size_frac`` is outside ``(0, 1)`` or the
            timeline is too short to split.
    """
    if not 0.0 < test_size_frac < 1.0:
        raise ConfigurationError(
            "test_size_frac must lie in (0, 1)", test_size_frac=test_size_frac
        )

    positions = _time_positions(len(times), times)
    n_periods = int(positions.max()) + 1
    if n_periods < 2:
        raise ConfigurationError(
            "Need at least two distinct time points for a holdout", n_periods=n_periods
        )

    n_test_periods = max(round(test_size_frac * n_periods), 1)
    cutoff = n_periods - n_test_periods
    if cutoff < 1:
        cutoff = 1

    train_idx = np.flatnonzero(positions < cutoff)
    test_idx = np.flatnonzero(positions >= cutoff)

    _log.info(
        "splitter.holdout",
        n_periods=n_periods,
        cutoff_period=cutoff,
        n_train=len(train_idx),
        n_test=len(test_idx),
    )
    return train_idx, test_idx
