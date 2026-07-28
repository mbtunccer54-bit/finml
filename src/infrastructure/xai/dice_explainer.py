"""Counterfactual explanations — actionable recourse.

Answers the question a relationship manager actually asks: *"this obligor's PD
is 0.6; which ratios must move, and by how little, to bring it to 0.2?"*
Attribution methods cannot answer that. SHAP says which features drove the
score, not what would change it.

Two tiers:

* **DiCE** (``dice-ml``) generates diverse counterfactuals — several genuinely
  different routes to the target, so the obligor is not handed one arbitrary
  prescription.
* **A greedy coordinate search** runs when DiCE is unavailable or fails. It is
  less diverse but deterministic, always respects the declared bounds, and
  always returns something actionable.

Only features the obligor can influence are varied. Shocking ``gdp_growth`` to
fix a PD is arithmetically valid and useless as advice, so macro variables are
excluded from the actionable set by default.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from domain.entities import FEATURE_CONTRACT, MACRO_VARIABLES, CounterfactualExample
from domain.exceptions import ExplainerError
from domain.value_objects import EntityId
from infrastructure.logging import get_logger

__all__ = ["CounterfactualGenerator"]

_log = get_logger(__name__)


class CounterfactualGenerator:
    """Finds minimal feature changes that reach a target PD.

    Attributes:
        model: The fitted model.
        training_data: Rows describing the feasible feature distribution.
        feature_names: Matrix column names.
        actionable_features: Features permitted to change.
        target_pd: PD the counterfactual should reach or beat.
    """

    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        *,
        feature_names: list[str] | None = None,
        actionable_features: list[str] | None = None,
        target_pd: float = 0.2,
        total_cfs: int = 3,
        method: str = "random",
        random_state: int = 42,
    ) -> None:
        """Initialise the generator.

        Args:
            model: A fitted model exposing ``predict_proba``.
            training_data: Rows describing the feasible feature distribution.
            feature_names: Matrix column names; taken from the data when omitted.
            actionable_features: Features permitted to change; defaults to every
                non-macro, non-categorical column.
            target_pd: PD the counterfactual should reach or beat.
            total_cfs: Counterfactuals to generate per query.
            method: DiCE generation method.
            random_state: Seed for reproducibility.

        Raises:
            ExplainerError: If the training data is empty or the target PD is
                outside ``(0, 1)``.
        """
        if training_data.empty:
            raise ExplainerError("Counterfactual search requires a non-empty sample")
        if not 0.0 < target_pd < 1.0:
            raise ExplainerError("target_pd must lie in (0, 1)", target_pd=target_pd)

        self.model = model
        self.feature_names = feature_names or list(training_data.columns)
        self.training_data = training_data[self.feature_names]
        self.target_pd = target_pd
        self.total_cfs = total_cfs
        self.method = method
        self.random_state = random_state
        self.actionable_features = actionable_features or self._default_actionable()
        self._dice: Any = None
        self._dice_failed = False

    def _default_actionable(self) -> list[str]:
        """Determine which features an obligor can realistically influence.

        Returns:
            Actionable feature names.
        """
        excluded = set(MACRO_VARIABLES)
        return [
            name
            for name in self.feature_names
            # One-hot columns carry '='; a sector is not a lever either.
            if "=" not in name and name not in excluded
        ]

    def _predict_pd(self, frame: pd.DataFrame) -> np.ndarray:
        """Score rows, returning the default probability.

        Args:
            frame: Feature matrix.

        Returns:
            One probability per row.
        """
        return np.asarray(self.model.predict_proba(frame[self.feature_names]))[:, 1]

    def generate(self, row: pd.DataFrame, *, entity_id: str) -> list[CounterfactualExample]:
        """Generate counterfactuals for one obligor.

        Args:
            row: A one-row feature matrix.
            entity_id: Obligor identifier recorded on the results.

        Returns:
            The counterfactuals found, best PD reduction first. Empty when the
            obligor is already below the target.

        Raises:
            ExplainerError: If ``row`` is not exactly one row.
        """
        if len(row) != 1:
            raise ExplainerError("generate expects exactly one row", n_rows=len(row))

        original_pd = float(self._predict_pd(row)[0])
        if original_pd <= self.target_pd:
            _log.info(
                "xai.counterfactual_not_needed",
                entity_id=entity_id,
                original_pd=round(original_pd, 4),
                target_pd=self.target_pd,
            )
            return []

        if not self._dice_failed:
            found = self._dice_generate(row, entity_id=entity_id, original_pd=original_pd)
            if found:
                return found

        return self._greedy_generate(row, entity_id=entity_id, original_pd=original_pd)

    def _dice_generate(
        self, row: pd.DataFrame, *, entity_id: str, original_pd: float
    ) -> list[CounterfactualExample]:
        """Generate counterfactuals with DiCE.

        Args:
            row: A one-row feature matrix.
            entity_id: Obligor identifier.
            original_pd: The obligor's current PD.

        Returns:
            The counterfactuals found, or an empty list on failure.
        """
        try:
            import dice_ml

            target_column = "_outcome"
            frame = self.training_data.copy()
            frame[target_column] = (self._predict_pd(self.training_data) >= self.target_pd).astype(
                int
            )

            data = dice_ml.Data(
                dataframe=frame,
                continuous_features=list(self.feature_names),
                outcome_name=target_column,
            )
            model = dice_ml.Model(model=self.model, backend="sklearn")
            engine = dice_ml.Dice(data, model, method=self.method)

            result = engine.generate_counterfactuals(
                row[self.feature_names],
                total_CFs=self.total_cfs,
                desired_class="opposite",
                features_to_vary=self.actionable_features,
                verbose=False,
            )
            candidates = result.cf_examples_list[0].final_cfs_df
            if candidates is None or candidates.empty:
                _log.info("xai.dice_no_solution", entity_id=entity_id)
                return []

            examples = [
                self._to_example(row, candidates.iloc[[i]], entity_id, original_pd)
                for i in range(len(candidates))
            ]
            valid = [e for e in examples if e.pd_reduction > 0]
            _log.info(
                "xai.counterfactuals_generated",
                entity_id=entity_id,
                backend="dice",
                n_found=len(valid),
                original_pd=round(original_pd, 4),
            )
            return sorted(valid, key=lambda e: (-e.pd_reduction, e.n_changes))
        except Exception as exc:
            # DiCE is sensitive to data shape; degrade rather than fail the request.
            self._dice_failed = True
            _log.warning(
                "xai.dice_unavailable",
                entity_id=entity_id,
                error_type=type(exc).__name__,
                reason=str(exc)[:200],
                fallback="greedy coordinate search",
            )
            return []

    def _to_example(
        self,
        original: pd.DataFrame,
        counterfactual: pd.DataFrame,
        entity_id: str,
        original_pd: float,
    ) -> CounterfactualExample:
        """Convert a DiCE row into a domain counterfactual.

        Args:
            original: The original one-row matrix.
            counterfactual: The proposed one-row matrix.
            entity_id: Obligor identifier.
            original_pd: The obligor's current PD.

        Returns:
            The domain counterfactual.
        """
        candidate = original.copy()
        changes: dict[str, tuple[float, float]] = {}
        for name in self.feature_names:
            if name not in counterfactual.columns:
                continue
            before = float(original.iloc[0][name])
            after = float(counterfactual.iloc[0][name])
            if abs(after - before) > 1e-9:
                changes[name] = (before, after)
                candidate.at[candidate.index[0], name] = after

        return CounterfactualExample(
            entity_id=EntityId(entity_id),
            original_pd=original_pd,
            counterfactual_pd=float(self._predict_pd(candidate)[0]),
            changes=changes,
        )

    def _greedy_generate(
        self, row: pd.DataFrame, *, entity_id: str, original_pd: float
    ) -> list[CounterfactualExample]:
        """Find a counterfactual by greedy coordinate search.

        At each step the single feature move that reduces PD most per unit of
        (normalised) change is applied, until the target is met or the change
        budget is spent. That yields a sparse, ordered recourse plan — "improve
        interest coverage first, then leverage" — which is what makes it usable
        advice rather than a vector.

        Args:
            row: A one-row feature matrix.
            entity_id: Obligor identifier.
            original_pd: The obligor's current PD.

        Returns:
            A single-element list with the best plan found, or empty if none
            reduced the PD.
        """
        candidate = row[self.feature_names].copy()
        changes: dict[str, tuple[float, float]] = {}
        current_pd = original_pd

        # Direction and step size come from the training distribution, so a
        # proposed move is always something the population actually exhibits.
        raw_quantiles = self.training_data[self.actionable_features].quantile([0.1, 0.5, 0.9])
        quantile_table: dict[str, dict[float, float]] = {
            str(column): {float(q): float(v) for q, v in values.items()}
            for column, values in raw_quantiles.to_dict().items()
        }
        max_changes = min(len(self.actionable_features), 5)

        for _ in range(max_changes):
            best_gain, best_feature, best_value = 0.0, "", 0.0

            for name in self.actionable_features:
                if name in changes:
                    continue
                current_value = float(candidate.iloc[0][name])
                for target_quantile in (0.1, 0.5, 0.9):
                    proposed = float(quantile_table[name][target_quantile])
                    if abs(proposed - current_value) < 1e-9:
                        continue
                    probe = candidate.copy()
                    probe.at[probe.index[0], name] = proposed
                    probe_pd = float(self._predict_pd(probe)[0])
                    gain = current_pd - probe_pd
                    if gain > best_gain:
                        best_gain, best_feature, best_value = gain, name, proposed

            if not best_feature or best_gain <= 1e-6:
                break

            before = float(candidate.iloc[0][best_feature])
            candidate.at[candidate.index[0], best_feature] = best_value
            changes[best_feature] = (before, best_value)
            current_pd -= best_gain

            if current_pd <= self.target_pd:
                break

        if not changes:
            _log.info("xai.counterfactual_none_found", entity_id=entity_id, backend="greedy")
            return []

        example = CounterfactualExample(
            entity_id=EntityId(entity_id),
            original_pd=original_pd,
            counterfactual_pd=float(self._predict_pd(candidate)[0]),
            changes=changes,
        )
        _log.info(
            "xai.counterfactuals_generated",
            entity_id=entity_id,
            backend="greedy",
            n_changes=example.n_changes,
            original_pd=round(original_pd, 4),
            counterfactual_pd=round(example.counterfactual_pd, 4),
            target_met=example.counterfactual_pd <= self.target_pd,
        )
        return [example]

    @staticmethod
    def to_frame(examples: list[CounterfactualExample]) -> pd.DataFrame:
        """Render counterfactuals as a readable recourse table.

        Args:
            examples: The counterfactuals.

        Returns:
            One row per required change, annotated with its business meaning.
        """
        rows: list[dict[str, Any]] = []
        for option, example in enumerate(examples, start=1):
            for feature, (before, after) in example.changes.items():
                try:
                    description = FEATURE_CONTRACT.get(feature).description
                except Exception:
                    description = ""
                rows.append(
                    {
                        "option": option,
                        "feature": feature,
                        "description": description,
                        "current": round(before, 4),
                        "required": round(after, 4),
                        "change": round(after - before, 4),
                        "pd_before": round(example.original_pd, 4),
                        "pd_after": round(example.counterfactual_pd, 4),
                        "n_changes": example.n_changes,
                    }
                )
        return pd.DataFrame(rows)
