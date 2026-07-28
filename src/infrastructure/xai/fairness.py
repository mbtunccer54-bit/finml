"""Fairness diagnostics across protected or segment attributes.

Sector is the attribute measured by default. It is not a protected class in the
legal sense, but the arithmetic is identical and the business question is real:
is the model rejecting one sector at a systematically higher rate than its
realised default experience justifies?

Three metrics, because they can disagree and the disagreement is informative:

* **Demographic parity** — equal flag rates across groups. Violated whenever
  true risk genuinely differs by group, so a gap here is a prompt to look, not
  a finding on its own.
* **Equalised odds** — equal true-positive *and* false-positive rates. This is
  the one that matters: it asks whether the model is equally accurate per group,
  conditional on the actual outcome.
* **Calibration by group** — whether a predicted 0.2 means 0.2 in every group.
  A model can satisfy equalised odds and still be miscalibrated for a segment,
  which shows up directly in mispriced credit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from domain.exceptions import CalculationError
from infrastructure.logging import get_logger

__all__ = ["FairnessReport", "assess_fairness"]

_log = get_logger(__name__)

#: Positives a group needs before its TPR/FPR enters the equalised-odds spread.
_MIN_POSITIVES_FOR_ODDS = 10


@dataclass(frozen=True, slots=True)
class FairnessReport:
    """Group-level fairness metrics.

    Attributes:
        attribute: The attribute groups were formed on.
        per_group: One row per group with rates and counts.
        demographic_parity_gap: Largest minus smallest flag rate.
        equalised_odds_gap: Largest gap in TPR or FPR across groups.
        calibration_gap: Largest gap between mean predicted and observed rate.
        tolerance: Threshold the gaps were judged against.
        flagged_groups: Groups breaching the tolerance.
    """

    attribute: str
    per_group: pd.DataFrame
    demographic_parity_gap: float
    equalised_odds_gap: float
    calibration_gap: float
    tolerance: float
    flagged_groups: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_fair(self) -> bool:
        """Whether every gap sits within tolerance.

        Returns:
            ``True`` when no gap exceeds the tolerance.
        """
        return (
            self.demographic_parity_gap <= self.tolerance
            and self.equalised_odds_gap <= self.tolerance
            and self.calibration_gap <= self.tolerance
        )

    def summary(self) -> dict[str, float | bool | str]:
        """Render the headline numbers.

        Returns:
            A mapping suitable for logging or an API response.
        """
        return {
            "attribute": self.attribute,
            "demographic_parity_gap": self.demographic_parity_gap,
            "equalised_odds_gap": self.equalised_odds_gap,
            "calibration_gap": self.calibration_gap,
            "tolerance": self.tolerance,
            "is_fair": self.is_fair,
            "n_groups": len(self.per_group),
        }


def assess_fairness(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    *,
    attribute: str = "sector",
    threshold: float = 0.5,
    tolerance: float = 0.10,
    min_group_size: int = 30,
) -> FairnessReport:
    """Compute fairness metrics across groups.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.
        groups: Group label per row.
        attribute: Name of the grouping attribute.
        threshold: Decision cut-off used for the rate metrics.
        tolerance: Largest acceptable gap.
        min_group_size: Groups smaller than this are reported but excluded from
            the gap statistics, where they would only contribute sampling noise.

    Returns:
        The fairness report.

    Raises:
        CalculationError: If the inputs differ in length or no group is large
            enough to compare.
    """
    truth = np.asarray(y_true).ravel().astype(int)
    prob = np.asarray(y_prob, dtype=float).ravel()
    labels = pd.Series(groups).reset_index(drop=True).astype(str)

    if not (truth.size == prob.size == labels.size):
        raise CalculationError(
            "Fairness inputs differ in length",
            n_true=truth.size,
            n_prob=prob.size,
            n_groups=labels.size,
        )

    predicted = (prob >= threshold).astype(int)
    rows: list[dict[str, float | int | str | bool]] = []

    for group in sorted(labels.unique()):
        mask = (labels == group).to_numpy()
        n = int(mask.sum())
        group_truth, group_pred, group_prob = truth[mask], predicted[mask], prob[mask]

        positives = group_truth == 1
        negatives = group_truth == 0
        rows.append(
            {
                "group": group,
                "n": n,
                "flag_rate": float(group_pred.mean()) if n else 0.0,
                "observed_default_rate": float(group_truth.mean()) if n else 0.0,
                "mean_predicted_pd": float(group_prob.mean()) if n else 0.0,
                "true_positive_rate": float(group_pred[positives].mean())
                if positives.any()
                else float("nan"),
                "false_positive_rate": float(group_pred[negatives].mean())
                if negatives.any()
                else float("nan"),
                "n_positives": int(positives.sum()),
                "calibration_error": abs(float(group_prob.mean()) - float(group_truth.mean()))
                if n
                else 0.0,
                "sufficient_size": n >= min_group_size,
                # TPR is computed over positives only. A group with two
                # defaulters yields a TPR of 0.0, 0.5 or 1.0 and would dominate
                # the equalised-odds spread with pure sampling noise.
                "sufficient_positives": int(positives.sum()) >= _MIN_POSITIVES_FOR_ODDS,
            }
        )

    per_group = pd.DataFrame(rows)
    comparable = per_group[per_group["sufficient_size"]]
    odds_comparable = per_group[per_group["sufficient_size"] & per_group["sufficient_positives"]]

    if comparable.empty:
        raise CalculationError(
            "No group meets the minimum size for a fairness comparison",
            min_group_size=min_group_size,
            group_sizes=per_group.set_index("group")["n"].to_dict(),
        )

    def spread(column: str, frame: pd.DataFrame) -> float:
        """Range of a column across the supplied groups.

        Args:
            column: Column name.
            frame: Groups to compare.

        Returns:
            Max minus min, ignoring missing values; ``0.0`` if fewer than two
            groups carry a value.
        """
        values = frame[column].dropna()
        return float(values.max() - values.min()) if len(values) > 1 else 0.0

    parity_gap = spread("flag_rate", comparable)
    odds_gap = max(
        spread("true_positive_rate", odds_comparable),
        spread("false_positive_rate", odds_comparable),
    )
    calibration_gap = float(comparable["calibration_error"].max())

    flagged = tuple(comparable.loc[comparable["calibration_error"] > tolerance, "group"].tolist())

    report = FairnessReport(
        attribute=attribute,
        per_group=per_group.sort_values("n", ascending=False).reset_index(drop=True),
        demographic_parity_gap=parity_gap,
        equalised_odds_gap=odds_gap,
        calibration_gap=calibration_gap,
        tolerance=tolerance,
        flagged_groups=flagged,
    )

    log = _log.warning if not report.is_fair else _log.info
    log(
        "xai.fairness_assessed",
        attribute=attribute,
        demographic_parity_gap=round(parity_gap, 4),
        equalised_odds_gap=round(odds_gap, 4),
        calibration_gap=round(calibration_gap, 4),
        tolerance=tolerance,
        is_fair=report.is_fair,
        n_groups=len(per_group),
        n_comparable=len(comparable),
        n_odds_comparable=len(odds_comparable),
        flagged_groups=list(flagged),
    )
    return report
