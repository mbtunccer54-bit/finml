"""Attribution stability.

An explanation that changes materially between runs is not evidence, and under
SR 11-7 it cannot support an adverse action notice. This module re-explains the
same obligor under several random seeds — varying the SHAP background sample —
and measures how much the attributions move.

The verdict logic itself lives in the domain
(:class:`~domain.services.AttributionStabilityAssessor`) because "is this
explanation trustworthy" is a business rule, not an implementation detail. This
module only supplies the measurements.

Temporal stability is measured too: attributions computed on successive time
windows should evolve gradually. A sharp discontinuity means either the
population moved or the model is keying on something unstable — both are
findings worth raising before a regulator does.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from domain.exceptions import ExplainerError
from domain.services import AttributionStabilityAssessor, StabilityVerdict
from infrastructure.logging import get_logger
from infrastructure.xai.shap_explainer import ShapExplainer

__all__ = ["StabilityAnalyzer", "TemporalStabilityReport"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TemporalStabilityReport:
    """Drift in global attributions across time windows.

    Attributes:
        window_labels: Label per time window.
        importance_by_window: Feature importance per window, indexed by feature.
        rank_correlation: Spearman correlation of feature ranks between
            consecutive windows.
        max_rank_shift: Largest change in a feature's rank between windows.
        unstable_features: Features whose rank moved more than the tolerance.
    """

    window_labels: tuple[str, ...]
    importance_by_window: pd.DataFrame
    rank_correlation: tuple[float, ...]
    max_rank_shift: int
    unstable_features: tuple[str, ...]

    @property
    def is_stable(self) -> bool:
        """Whether attributions held together across windows.

        Returns:
            ``True`` when every consecutive rank correlation is at least 0.7.
        """
        return all(c >= 0.7 for c in self.rank_correlation) if self.rank_correlation else True


class StabilityAnalyzer:
    """Measures how reproducible a model's explanations are.

    Attributes:
        model: The fitted model.
        background: Rows the SHAP background is sampled from.
        feature_names: Matrix column names.
        assessor: Domain service turning dispersion into a verdict.
    """

    def __init__(
        self,
        model: Any,
        background: pd.DataFrame,
        *,
        feature_names: list[str] | None = None,
        tolerance: float = 0.25,
        n_seeds: int = 5,
    ) -> None:
        """Initialise the analyser.

        Args:
            model: A fitted model exposing ``predict_proba``.
            background: Rows the SHAP background is sampled from.
            feature_names: Matrix column names; taken from the data when omitted.
            tolerance: Largest acceptable scaled dispersion.
            n_seeds: Number of seeds to re-explain under.

        Raises:
            ExplainerError: If the background set is too small to resample.
        """
        if len(background) < 10:
            raise ExplainerError(
                "Stability analysis needs at least 10 background rows",
                n_rows=len(background),
            )
        self.model = model
        self.feature_names = feature_names or list(background.columns)
        self.background = background[self.feature_names]
        self.n_seeds = max(n_seeds, 2)
        self.assessor = AttributionStabilityAssessor(tolerance=tolerance, min_runs=2)

    def assess_instance(self, row: pd.DataFrame) -> StabilityVerdict:
        """Re-explain one obligor under several seeds and score the dispersion.

        Args:
            row: A one-row feature matrix.

        Returns:
            The stability verdict.

        Raises:
            ExplainerError: If ``row`` is not exactly one row, or every seed failed.
        """
        if len(row) != 1:
            raise ExplainerError("assess_instance expects exactly one row", n_rows=len(row))

        attributions: dict[str, list[float]] = {name: [] for name in self.feature_names}
        successes = 0

        for seed in range(self.n_seeds):
            sample_size = min(len(self.background), max(20, len(self.background) // 2))
            sample = self.background.sample(n=sample_size, random_state=seed, replace=False)
            try:
                explainer = ShapExplainer(
                    self.model, sample, feature_names=self.feature_names, backend="auto"
                )
                values = explainer.explain(row).values[0]
            except ExplainerError as exc:
                _log.warning("xai.stability_seed_failed", seed=seed, reason=exc.message)
                continue
            for index, name in enumerate(self.feature_names):
                attributions[name].append(float(values[index]))
            successes += 1

        if successes < 2:
            raise ExplainerError(
                "Too few successful explanation runs for a stability verdict",
                n_successes=successes,
                n_seeds=self.n_seeds,
            )

        verdict = self.assessor.assess(attributions)
        _log.info(
            "xai.stability_assessed",
            n_runs=successes,
            is_stable=verdict.is_stable,
            max_cv=round(verdict.max_coefficient_of_variation, 4),
            n_unstable=len(verdict.unstable_features),
            unstable=list(verdict.unstable_features[:5]),
        )
        return verdict

    def assess_temporal(
        self,
        x: pd.DataFrame,
        times: pd.Series,
        *,
        n_windows: int = 4,
        max_rows_per_window: int = 200,
    ) -> TemporalStabilityReport:
        """Compare global attributions across consecutive time windows.

        Args:
            x: Feature matrix.
            times: Per-row timestamps.
            n_windows: Number of consecutive windows to split into.
            max_rows_per_window: Row cap per window, for tractability.

        Returns:
            The temporal stability report.

        Raises:
            ExplainerError: If fewer than two windows could be explained.
        """
        ordered = times.sort_values()
        boundaries = np.array_split(ordered.index.to_numpy(), max(n_windows, 2))

        importances: dict[str, pd.Series] = {}
        labels: list[str] = []

        for index, window_index in enumerate(boundaries):
            if len(window_index) < 10:
                continue
            window = x.loc[window_index]
            if len(window) > max_rows_per_window:
                window = window.sample(n=max_rows_per_window, random_state=42)
            label = f"w{index + 1}_{times.loc[window_index].min():%Y-%m}"
            try:
                explainer = ShapExplainer(
                    self.model, window, feature_names=self.feature_names, backend="auto"
                )
                ranking = explainer.explain(window).global_importance()
            except ExplainerError as exc:
                _log.warning("xai.temporal_window_failed", window=label, reason=exc.message)
                continue
            importances[label] = ranking.set_index("feature")["mean_abs_shap"]
            labels.append(label)

        if len(importances) < 2:
            raise ExplainerError(
                "Need at least two explainable windows for temporal stability",
                n_windows=len(importances),
            )

        frame = pd.DataFrame(importances).fillna(0.0)
        ranks = frame.rank(ascending=False)

        correlations: list[float] = [
            float(frame[left].corr(frame[right], method="spearman"))
            for left, right in pairwise(labels)
        ]

        shift = (ranks.max(axis=1) - ranks.min(axis=1)).astype(int)
        # A feature crossing a quarter of the ranking between windows is a real
        # change in what the model is keying on.
        threshold = max(int(len(frame) * 0.25), 3)
        unstable = tuple(shift[shift > threshold].sort_values(ascending=False).index.tolist())

        report = TemporalStabilityReport(
            window_labels=tuple(labels),
            importance_by_window=frame,
            rank_correlation=tuple(correlations),
            max_rank_shift=int(shift.max()),
            unstable_features=unstable,
        )
        _log.info(
            "xai.temporal_stability",
            n_windows=len(labels),
            rank_correlations=[round(c, 4) for c in correlations],
            max_rank_shift=report.max_rank_shift,
            is_stable=report.is_stable,
            unstable=list(unstable[:5]),
        )
        return report
