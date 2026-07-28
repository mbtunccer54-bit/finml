"""Input drift detection.

Two complementary tests run before every scoring batch:

* **Population Stability Index** — a symmetric, binned divergence between the
  reference and current distributions. It is the credit-industry convention and
  its thresholds (0.10 / 0.25) are what a model-risk committee expects to see.
* **Two-sample Kolmogorov-Smirnov** — a distribution-free test that catches
  shape changes the PSI's coarse bins can miss.

The PSI decides severity because its thresholds carry an agreed operational
meaning; the KS statistic is reported alongside as corroboration. Reporting both
matters because they fail differently: PSI is insensitive to a shift inside a
bin, KS is insensitive to tail mass changes that do not move the CDF much.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy import stats

from domain.entities import DriftReport, FeatureDrift
from domain.exceptions import DriftDetectedError
from domain.value_objects import DriftSeverity
from infrastructure.config.schemas import DriftConfig
from infrastructure.logging import get_logger

__all__ = ["DriftDetector", "PSIResult", "population_stability_index"]

_log = get_logger(__name__)

#: Floor applied to bin proportions so an empty bin does not make the PSI infinite.
_PROPORTION_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class PSIResult:
    """Population Stability Index for one feature.

    Attributes:
        psi: The index value.
        bin_edges: Quantile bin edges taken from the reference distribution.
        reference_proportions: Reference mass per bin.
        current_proportions: Current mass per bin.
    """

    psi: float
    bin_edges: tuple[float, ...]
    reference_proportions: tuple[float, ...]
    current_proportions: tuple[float, ...]

    @property
    def worst_bin(self) -> int:
        """Index of the bin contributing most to the divergence.

        Returns:
            The bin index, which localises *where* the population moved.
        """
        contributions = [
            (c - r) * np.log(max(c, _PROPORTION_FLOOR) / max(r, _PROPORTION_FLOOR))
            for r, c in zip(self.reference_proportions, self.current_proportions, strict=True)
        ]
        return int(np.argmax(contributions))


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, n_bins: int = 10
) -> PSIResult:
    """Compute the PSI between a reference and a current sample.

    Bins are quantiles of the *reference* distribution, so each reference bin
    carries roughly equal mass and the index is not dominated by a sparse tail.
    Duplicate edges (common on discrete or heavily-tied features) are collapsed
    rather than producing zero-width bins.

    Args:
        reference: Reference sample, typically the training distribution.
        current: Current sample being scored.
        n_bins: Requested quantile bin count.

    Returns:
        The PSI and the binning used to compute it.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if ref.size == 0 or cur.size == 0:
        return PSIResult(0.0, (), (), ())

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        # A constant reference feature cannot drift in a PSI sense; report the
        # binary case of "did the current sample leave the constant?".
        moved = float(np.mean(cur != ref[0]))
        return PSIResult(
            0.0 if moved == 0.0 else float("inf"), tuple(edges), (1.0,), (1.0 - moved,)
        )

    # Open the outer edges so values outside the reference range are counted
    # rather than dropped -- an out-of-range shift is precisely what we hunt.
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_prop = np.maximum(ref_counts / ref.size, _PROPORTION_FLOOR)
    cur_prop = np.maximum(cur_counts / cur.size, _PROPORTION_FLOOR)

    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return PSIResult(
        psi=psi,
        bin_edges=tuple(float(e) for e in edges),
        reference_proportions=tuple(float(p) for p in ref_prop),
        current_proportions=tuple(float(p) for p in cur_prop),
    )


class DriftDetector:
    """Compares a scoring batch against the training reference distribution.

    Attributes:
        config: Drift configuration supplying thresholds and the fail-safe policy.
    """

    def __init__(self, config: DriftConfig) -> None:
        """Initialise the detector.

        Args:
            config: Drift configuration.
        """
        self.config = config
        self._reference: pd.DataFrame | None = None

    def fit(self, reference: pd.DataFrame) -> DriftDetector:
        """Record the reference distribution.

        Args:
            reference: The training feature frame.

        Returns:
            This detector, for chaining.
        """
        self._reference = reference.copy()
        _log.info(
            "drift.reference_fitted",
            n_rows=len(reference),
            n_features=len(self._monitored_features(reference)),
        )
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether a reference distribution has been recorded.

        Returns:
            ``True`` once :meth:`fit` has been called.
        """
        return self._reference is not None

    def _monitored_features(self, frame: pd.DataFrame) -> list[str]:
        """Determine which columns to monitor.

        Args:
            frame: Frame whose columns are candidates.

        Returns:
            The numeric columns to monitor, restricted by configuration.
        """
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()

        if self.config.features:
            requested = set(self.config.features)
            selected = [c for c in numeric if c in requested]
            unknown = sorted(requested - set(numeric))
            if unknown:
                _log.warning("drift.unknown_features_requested", unknown=unknown)
        else:
            selected = numeric

        # A batch is typically one reporting period, within which every macro
        # variable takes a single value, while the reference pools many periods.
        # PSI between a point mass and a spread distribution is arbitrarily large
        # and carries no information about input quality -- macro movement is
        # what the scenario module is for, not what drift monitoring is for.
        excluded = set(self.config.exclude_features)
        return [c for c in selected if c not in excluded]

    def _severity(self, psi: float) -> DriftSeverity:
        """Bucket a PSI into a severity.

        Args:
            psi: The index value.

        Returns:
            The corresponding severity.
        """
        if psi >= self.config.psi_severe_threshold:
            return DriftSeverity.SEVERE
        if psi >= self.config.psi_moderate_threshold:
            return DriftSeverity.MODERATE
        return DriftSeverity.NONE

    def detect(self, current: pd.DataFrame) -> DriftReport:
        """Compare a batch against the reference distribution.

        Args:
            current: The scoring batch.

        Returns:
            The drift report. When the detector is disabled, unfitted, or the
            reference is too small to be informative, an empty report with
            ``NONE`` severity is returned rather than a spurious alarm.
        """
        now = datetime.now(UTC)

        if not self.config.enabled or self._reference is None:
            return DriftReport(computed_at=now, current_size=len(current))

        reference = self._reference
        if len(reference) < self.config.min_reference_size:
            _log.warning(
                "drift.reference_too_small",
                n_reference=len(reference),
                required=self.config.min_reference_size,
                detail="reporting NONE; the PSI would be dominated by sampling noise",
            )
            return DriftReport(
                computed_at=now, reference_size=len(reference), current_size=len(current)
            )

        # Symmetric guard on the batch side. A single-obligor scoring request
        # cannot express a population distribution, so its PSI against any
        # reference is large and meaningless -- refusing to score on that basis
        # would break interactive use for no safety benefit.
        if len(current) < self.config.min_batch_size:
            _log.info(
                "drift.batch_too_small",
                n_current=len(current),
                required=self.config.min_batch_size,
                detail="reporting NONE; a PSI on this few rows is not informative",
            )
            return DriftReport(
                computed_at=now, reference_size=len(reference), current_size=len(current)
            )

        features = [f for f in self._monitored_features(reference) if f in current.columns]
        results: list[FeatureDrift] = []

        for feature in features:
            ref_values = reference[feature].to_numpy(dtype=float)
            cur_values = current[feature].to_numpy(dtype=float)
            psi_result = population_stability_index(
                ref_values, cur_values, n_bins=self.config.n_bins
            )
            ks_stat, ks_p = self._ks_test(ref_values, cur_values)
            results.append(
                FeatureDrift(
                    feature=feature,
                    psi=psi_result.psi,
                    ks_statistic=ks_stat,
                    ks_p_value=ks_p,
                    severity=self._severity(psi_result.psi),
                )
            )

        report = DriftReport(
            computed_at=now,
            features=tuple(results),
            reference_size=len(reference),
            current_size=len(current),
        )
        self._log_report(report)
        return report

    @staticmethod
    def _ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
        """Run a two-sample Kolmogorov-Smirnov test.

        Args:
            reference: Reference sample.
            current: Current sample.

        Returns:
            The KS statistic and its p-value; ``(0.0, 1.0)`` when either sample
            is empty after removing non-finite values.
        """
        ref = reference[np.isfinite(reference)]
        cur = current[np.isfinite(current)]
        if ref.size == 0 or cur.size == 0:
            return 0.0, 1.0
        result = stats.ks_2samp(ref, cur)
        return float(result.statistic), float(result.pvalue)

    def _log_report(self, report: DriftReport) -> None:
        """Emit the drift report as a structured event.

        Args:
            report: The report to log.
        """
        drifted = report.drifted_features
        payload = {
            "severity": report.severity.name,
            "max_psi": round(report.max_psi, 4),
            "n_drifted": len(drifted),
            "n_monitored": len(report.features),
            "reference_size": report.reference_size,
            "current_size": report.current_size,
            "top_drifted": [
                {"feature": f.feature, "psi": round(f.psi, 4), "ks_p": round(f.ks_p_value, 4)}
                for f in drifted[:5]
            ],
        }
        if report.severity is DriftSeverity.SEVERE:
            # An alert is emitted on every severe report regardless of policy;
            # `on_severe` only decides whether scoring is allowed to proceed.
            _log.error("drift.alert", alert=True, policy=self.config.on_severe, **payload)
        elif report.severity is DriftSeverity.MODERATE:
            _log.warning("drift.detected", alert=True, **payload)
        else:
            _log.info("drift.clear", **payload)

    def enforce(self, report: DriftReport) -> DriftReport:
        """Apply the configured fail-safe policy to a report.

        Under ``block`` a severe report stops scoring: in a regulated lending
        decision, returning no score is safer than returning one produced on a
        population the model never saw. Under ``warn`` the alert has already
        been emitted and the severity travels on the response so the caller can
        decide.

        Args:
            report: The drift report.

        Returns:
            The same report, when scoring is permitted to continue.

        Raises:
            DriftDetectedError: If the policy is ``block`` and severity is
                ``SEVERE``.
        """
        if report.severity is not DriftSeverity.SEVERE:
            return report
        if self.config.on_severe != "block":
            _log.warning(
                "drift.proceeding_despite_severe",
                policy=self.config.on_severe,
                max_psi=round(report.max_psi, 4),
            )
            return report

        raise DriftDetectedError(
            "Severe input drift; refusing to score",
            max_psi=round(report.max_psi, 4),
            threshold=self.config.psi_severe_threshold,
            drifted_features=[f.feature for f in report.drifted_features[:10]],
            policy=self.config.on_severe,
        )
