"""LIME attributions.

LIME answers a different question from SHAP and is kept for exactly that reason.
SHAP attributes a prediction against a *global* baseline with game-theoretic
guarantees; LIME fits a sparse linear surrogate in a *local* neighbourhood. When
the two agree on a case, confidence in the reason code rises. When they disagree,
that is itself a finding — usually a sign the decision sits on a sharp boundary
where a single reason code is not defensible.

LIME is stochastic: it samples perturbations. The seed is therefore fixed by
default so an explanation attached to a credit file reproduces exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from domain.entities import FeatureContribution, ScoreExplanation
from domain.exceptions import ExplainerError
from domain.value_objects import EntityId, ModelId
from infrastructure.logging import get_logger

__all__ = ["LimeExplainer"]

_log = get_logger(__name__)


class LimeExplainer:
    """Local surrogate explanations for a fitted model.

    Attributes:
        model: The fitted model.
        training_data: Rows defining the perturbation distribution.
        feature_names: Matrix column names.
        num_features: Features retained in an explanation.
        num_samples: Perturbations drawn per explanation.
        random_state: Seed, fixed so explanations reproduce.
    """

    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        *,
        feature_names: list[str] | None = None,
        num_features: int = 10,
        num_samples: int = 2000,
        random_state: int = 42,
    ) -> None:
        """Initialise the explainer.

        Args:
            model: A fitted model exposing ``predict_proba``.
            training_data: Rows defining the perturbation distribution.
            feature_names: Matrix column names; taken from the data when omitted.
            num_features: Features retained in an explanation.
            num_samples: Perturbations drawn per explanation.
            random_state: Seed for reproducibility.

        Raises:
            ExplainerError: If the training data is empty.
        """
        if training_data.empty:
            raise ExplainerError("LIME requires a non-empty training sample")

        self.model = model
        self.feature_names = feature_names or list(training_data.columns)
        self.training_data = training_data[self.feature_names]
        self.num_features = num_features
        self.num_samples = num_samples
        self.random_state = random_state
        self._explainer: Any = None

    def _build(self) -> Any:
        """Construct the underlying LIME explainer.

        Returns:
            The constructed explainer.

        Raises:
            ExplainerError: If construction fails.
        """
        if self._explainer is not None:
            return self._explainer
        try:
            from lime.lime_tabular import LimeTabularExplainer

            self._explainer = LimeTabularExplainer(
                training_data=self.training_data.to_numpy(dtype=float),
                feature_names=list(self.feature_names),
                class_names=["no_default", "default"],
                mode="classification",
                discretize_continuous=True,
                random_state=self.random_state,
            )
        except Exception as exc:
            raise ExplainerError(
                "Could not construct the LIME explainer", reason=str(exc)[:300]
            ) from exc
        return self._explainer

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Score perturbed rows for LIME.

        Args:
            x: Perturbations as a plain array.

        Returns:
            An ``(n, 2)`` probability array.
        """
        frame = pd.DataFrame(x, columns=self.feature_names)
        return np.asarray(self.model.predict_proba(frame))

    def explain_instance(
        self,
        row: pd.DataFrame,
        *,
        entity_id: str,
        model_id: str | None = None,
    ) -> ScoreExplanation:
        """Explain a single obligor.

        Args:
            row: A one-row feature matrix.
            entity_id: Obligor identifier recorded on the explanation.
            model_id: Model identifier recorded on the explanation.

        Returns:
            The domain explanation.

        Raises:
            ExplainerError: If ``row`` is not exactly one row, or LIME fails.
        """
        if len(row) != 1:
            raise ExplainerError("explain_instance expects exactly one row", n_rows=len(row))

        explainer = self._build()
        values = row[self.feature_names].to_numpy(dtype=float)[0]

        try:
            explanation = explainer.explain_instance(
                data_row=values,
                predict_fn=self._predict,
                num_features=self.num_features,
                num_samples=self.num_samples,
                labels=(1,),
            )
            weights = dict(explanation.as_map()[1])
            intercept = float(explanation.intercept[1])
            local_r2 = float(getattr(explanation, "score", 0.0))
        except Exception as exc:
            raise ExplainerError(
                "LIME explanation failed", entity_id=entity_id, reason=str(exc)[:300]
            ) from exc

        contributions = tuple(
            FeatureContribution(
                feature=self.feature_names[index],
                value=float(values[index]),
                contribution=float(weight),
            )
            for index, weight in sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        # A low local R^2 means the linear surrogate does not describe this
        # neighbourhood well, which makes the reason codes unreliable.
        if local_r2 < 0.3:
            _log.warning(
                "xai.lime_poor_local_fit",
                entity_id=entity_id,
                local_r2=round(local_r2, 4),
                detail="surrogate explains little local variance; treat with caution",
            )

        _log.debug(
            "xai.lime_computed",
            entity_id=entity_id,
            n_features=len(contributions),
            local_r2=round(local_r2, 4),
        )
        return ScoreExplanation(
            entity_id=EntityId(entity_id),
            method="lime",
            base_value=intercept,
            contributions=contributions,
            model_id=ModelId(model_id) if model_id else None,
        )

    def explain_batch(
        self, rows: pd.DataFrame, entity_ids: list[str], *, model_id: str | None = None
    ) -> list[ScoreExplanation]:
        """Explain several obligors.

        A failure on one row is logged and skipped rather than aborting the
        batch — one unexplainable case should not cost the other explanations.

        Args:
            rows: Feature matrix.
            entity_ids: Obligor identifier per row.
            model_id: Model identifier recorded on the explanations.

        Returns:
            One explanation per successfully explained row.

        Raises:
            ExplainerError: If the row and identifier counts differ.
        """
        if len(rows) != len(entity_ids):
            raise ExplainerError(
                "Row and entity id counts differ", n_rows=len(rows), n_ids=len(entity_ids)
            )
        out: list[ScoreExplanation] = []
        for position, entity_id in enumerate(entity_ids):
            try:
                out.append(
                    self.explain_instance(
                        rows.iloc[[position]], entity_id=entity_id, model_id=model_id
                    )
                )
            except ExplainerError as exc:
                _log.warning("xai.lime_row_failed", entity_id=entity_id, reason=exc.message)
        return out
