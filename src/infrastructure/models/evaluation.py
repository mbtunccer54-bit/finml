"""Model evaluation and benchmarking.

Discrimination alone does not qualify a model for lending decisions. A model
can rank obligors perfectly and still report a 0.9 probability for a group that
defaults 30% of the time — useless for pricing, provisioning or IFRS 9 staging.
So every model is scored on four axes:

* **Discrimination** — PR-AUC first, ROC-AUC second. At an 8% base rate ROC-AUC
  is dominated by the majority class and flatters every model.
* **Calibration** — Brier score and expected calibration error.
* **Business cost** — the cost matrix evaluated at the cost-optimal threshold,
  not at an arbitrary 0.5.
* **Latency** — the median single-row cost of a decision.

Selection then applies the configured gates in that order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from domain.entities import ModelPerformance
from domain.exceptions import CalculationError
from domain.services import CostSensitiveThresholdOptimizer, ThresholdDecision
from domain.value_objects import CostMatrix, ModelId
from infrastructure.logging import get_logger

__all__ = [
    "ReliabilityCurve",
    "benchmark_table",
    "evaluate_model",
    "expected_calibration_error",
    "reliability_curve",
    "select_champion",
]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """Binned observed-versus-predicted frequencies for a reliability diagram.

    Attributes:
        mean_predicted: Mean predicted probability per bin.
        observed_frequency: Observed positive rate per bin.
        bin_counts: Rows per bin.
        expected_calibration_error: Count-weighted mean absolute gap.
    """

    mean_predicted: tuple[float, ...]
    observed_frequency: tuple[float, ...]
    bin_counts: tuple[int, ...]
    expected_calibration_error: float

    def as_frame(self) -> pd.DataFrame:
        """Render the curve for plotting.

        Returns:
            A frame with one row per populated bin.
        """
        return pd.DataFrame(
            {
                "mean_predicted": self.mean_predicted,
                "observed_frequency": self.observed_frequency,
                "count": self.bin_counts,
            }
        )


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10, strategy: str = "quantile"
) -> ReliabilityCurve:
    """Compute a reliability curve and the expected calibration error.

    Quantile bins are the default because a credit portfolio's predictions
    cluster near zero: uniform bins would leave the upper bins empty and make
    the ECE an average over three populated buckets.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.
        n_bins: Requested bin count.
        strategy: ``quantile`` for equal-mass bins, ``uniform`` for equal-width.

    Returns:
        The curve and its ECE.

    Raises:
        CalculationError: If the inputs are empty, differ in length, or
            ``strategy`` is unknown.
    """
    truth = np.asarray(y_true, dtype=float).ravel()
    prob = np.asarray(y_prob, dtype=float).ravel()
    if truth.size != prob.size:
        raise CalculationError(
            "Label and probability lengths differ", n_true=truth.size, n_prob=prob.size
        )
    if truth.size == 0:
        raise CalculationError("Cannot compute a reliability curve on an empty sample")

    if strategy == "quantile":
        edges = np.unique(np.quantile(prob, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise CalculationError(
            "Unknown binning strategy", strategy=strategy, known=["quantile", "uniform"]
        )

    # right=True with a lowered first edge keeps the minimum value inside bin 0.
    edges[0] = min(edges[0], prob.min()) - 1e-12
    indices = np.digitize(prob, edges[1:-1], right=True)

    means: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for b in range(len(edges) - 1):
        mask = indices == b
        n = int(mask.sum())
        if n == 0:
            continue
        means.append(float(prob[mask].mean()))
        observed.append(float(truth[mask].mean()))
        counts.append(n)

    total = sum(counts)
    ece = (
        sum(c * abs(m - o) for c, m, o in zip(counts, means, observed, strict=True)) / total
        if total
        else 0.0
    )
    return ReliabilityCurve(
        mean_predicted=tuple(means),
        observed_frequency=tuple(observed),
        bin_counts=tuple(counts),
        expected_calibration_error=float(ece),
    )


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10
) -> float:
    """Compute the expected calibration error.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.
        n_bins: Bin count.

    Returns:
        The count-weighted mean absolute gap between confidence and accuracy.
    """
    return reliability_curve(y_true, y_prob, n_bins=n_bins).expected_calibration_error


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    model_id: str,
    split: str,
    cost_matrix: CostMatrix | None = None,
    threshold: float | None = None,
    latency_ms: float = 0.0,
    n_bins: int = 10,
) -> tuple[ModelPerformance, ThresholdDecision]:
    """Score one model on one split.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.
        model_id: Model identifier.
        split: Split name, for example ``"validation"``.
        cost_matrix: Business costs; the platform default when omitted.
        threshold: Decision threshold. When omitted the cost-optimal threshold
            is searched for, which is the only defensible default under a 10:1
            cost asymmetry.
        latency_ms: Median single-row scoring latency.
        n_bins: Bins used for the calibration error.

    Returns:
        The performance record and the threshold decision behind its point metrics.

    Raises:
        CalculationError: If the split contains a single class.
    """
    truth = np.asarray(y_true).ravel().astype(int)
    prob = np.asarray(y_prob, dtype=float).ravel()
    costs = cost_matrix or CostMatrix()

    if np.unique(truth).size < 2:
        raise CalculationError(
            "Cannot evaluate a split containing a single class",
            split=split,
            model_id=model_id,
            classes=np.unique(truth).tolist(),
        )

    optimiser = CostSensitiveThresholdOptimizer(cost_matrix=costs)
    decision = (
        optimiser.optimise(truth.tolist(), prob.tolist())
        if threshold is None
        else optimiser.evaluate(truth.tolist(), prob.tolist(), threshold=threshold)
    )
    predicted = (prob >= decision.threshold).astype(int)

    performance = ModelPerformance(
        model_id=ModelId(model_id),
        split=split,
        roc_auc=float(roc_auc_score(truth, prob)),
        pr_auc=float(average_precision_score(truth, prob)),
        f1=float(f1_score(truth, predicted, zero_division=0)),
        precision=float(precision_score(truth, predicted, zero_division=0)),
        recall=float(recall_score(truth, predicted, zero_division=0)),
        brier_score=float(brier_score_loss(truth, prob)),
        expected_calibration_error=expected_calibration_error(truth, prob, n_bins=n_bins),
        business_cost=decision.expected_cost,
        inference_latency_ms=latency_ms,
        threshold=decision.threshold,
        n_samples=int(truth.size),
    )

    _log.info(
        "model.evaluated",
        model_id=model_id,
        split=split,
        pr_auc=round(performance.pr_auc, 4),
        roc_auc=round(performance.roc_auc, 4),
        ece=round(performance.expected_calibration_error, 4),
        brier=round(performance.brier_score, 4),
        business_cost=round(performance.business_cost, 2),
        threshold=round(performance.threshold, 4),
        recall=round(performance.recall, 4),
        latency_ms=round(latency_ms, 3),
    )
    return performance, decision


def confusion_at(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float) -> np.ndarray:
    """Compute the confusion matrix at a threshold.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.
        threshold: Decision cut-off.

    Returns:
        A ``2x2`` array ordered ``[[tn, fp], [fn, tp]]``.
    """
    predicted = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    return np.asarray(confusion_matrix(np.asarray(y_true).astype(int), predicted, labels=[0, 1]))


def benchmark_table(performances: list[ModelPerformance]) -> pd.DataFrame:
    """Assemble evaluation records into a comparison table.

    Args:
        performances: The records to tabulate.

    Returns:
        One row per model and split, sorted by split then descending PR-AUC.
    """
    if not performances:
        return pd.DataFrame()
    frame = pd.DataFrame([p.as_dict() for p in performances])
    return frame.sort_values(["split", "pr_auc"], ascending=[True, False]).reset_index(drop=True)


def select_champion(
    performances: list[ModelPerformance],
    *,
    split: str = "validation",
    max_calibration_error: float = 0.05,
    max_latency_ms: float = 50.0,
    min_pr_auc: float = 0.10,
) -> tuple[ModelId, list[str]]:
    """Choose the model to promote.

    The gates are applied in the order the credit policy states them: a model
    must discriminate, be calibrated, and be fast enough; among the survivors
    the cheapest in business cost wins. Ranking on PR-AUC alone would promote a
    model whose probabilities cannot be used for pricing.

    If no model clears every gate, the best PR-AUC is returned together with the
    reasons — refusing to select at all would leave the pipeline with no output,
    but the caller must be able to see that the gates were missed.

    Args:
        performances: Evaluation records across models.
        split: Split the decision is made on.
        max_calibration_error: Calibration gate.
        max_latency_ms: Latency gate.
        min_pr_auc: Discrimination gate.

    Returns:
        The champion's identifier and a list of human-readable notes.

    Raises:
        CalculationError: If no record exists for ``split``.
    """
    candidates = [p for p in performances if p.split == split]
    if not candidates:
        raise CalculationError(
            "No evaluation records for the selection split",
            split=split,
            available=sorted({p.split for p in performances}),
        )

    notes: list[str] = []
    eligible = [
        p
        for p in candidates
        if p.meets_sla(
            max_calibration_error=max_calibration_error,
            max_latency_ms=max_latency_ms,
            min_pr_auc=min_pr_auc,
        )
    ]

    if not eligible:
        fallback = max(candidates, key=lambda p: p.pr_auc)
        notes.append(
            f"No model cleared every gate (ECE<={max_calibration_error}, "
            f"latency<={max_latency_ms}ms, PR-AUC>={min_pr_auc}); "
            f"fell back to best PR-AUC: {fallback.model_id}"
        )
        for p in candidates:
            reasons = []
            if p.expected_calibration_error > max_calibration_error:
                reasons.append(f"ECE={p.expected_calibration_error:.4f}")
            if p.inference_latency_ms > max_latency_ms:
                reasons.append(f"latency={p.inference_latency_ms:.1f}ms")
            if p.pr_auc < min_pr_auc:
                reasons.append(f"PR-AUC={p.pr_auc:.4f}")
            if reasons:
                notes.append(f"  {p.model_id} failed: {', '.join(reasons)}")
        _log.warning(
            "model.selection_gates_missed",
            split=split,
            fallback=str(fallback.model_id),
            n_candidates=len(candidates),
            notes=notes,
        )
        return fallback.model_id, notes

    champion = min(eligible, key=lambda p: (p.business_cost, -p.pr_auc))
    notes.append(
        f"Selected {champion.model_id}: lowest business cost "
        f"({champion.business_cost:.1f}) among {len(eligible)} models clearing all gates"
    )
    _log.info(
        "model.champion_selected",
        model_id=str(champion.model_id),
        split=split,
        business_cost=round(champion.business_cost, 2),
        pr_auc=round(champion.pr_auc, 4),
        ece=round(champion.expected_calibration_error, 4),
        latency_ms=round(champion.inference_latency_ms, 3),
        n_eligible=len(eligible),
        n_candidates=len(candidates),
    )
    return champion.model_id, notes
