"""Concrete adapters for the four base learners.

Each subclass exists to translate the platform's uniform vocabulary
(``random_state``, ``n_jobs``, positive-class weight, focal loss) into the
vendor's own spelling. CatBoost wants ``random_seed`` and ``thread_count``;
LightGBM and XGBoost accept a callable objective while CatBoost and the random
forest do not.

Where a library cannot honour a requested strategy the adapter degrades to the
nearest equivalent and logs that it did — a silent downgrade would leave the
benchmark comparing models trained under different objectives.
"""

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from domain.exceptions import ModelTrainingError
from infrastructure.config.schemas import ModelSpec
from infrastructure.logging import get_logger
from infrastructure.models.base import BaseModelAdapter, make_focal_objective

__all__ = [
    "MODEL_REGISTRY",
    "CatBoostAdapter",
    "LightGBMAdapter",
    "RandomForestAdapter",
    "XGBoostAdapter",
    "build_model",
]

_log = get_logger(__name__)


class CatBoostAdapter(BaseModelAdapter):
    """CatBoost gradient boosting.

    CatBoost has no callable-objective hook comparable to LightGBM's, so a
    ``focal`` request degrades to cost-aware class weighting.
    """

    @property
    def name(self) -> str:
        """Model name.

        Returns:
            ``"catboost"``.
        """
        return "catboost"

    def _build_estimator(self, positive_weight: float) -> CatBoostClassifier:
        """Construct the CatBoost estimator.

        Args:
            positive_weight: Weight to apply to the positive class.

        Returns:
            The unfitted estimator.
        """
        params = self._effective_params
        if self.imbalance_strategy == "focal":
            _log.warning(
                "model.focal_unsupported",
                model=self.name,
                fallback="class_weight",
                detail="CatBoost exposes no callable objective hook",
            )
            positive_weight = max(positive_weight, self._effective_cost_matrix.imbalance_ratio)

        return CatBoostClassifier(
            **params,
            random_seed=self.random_state,
            thread_count=self.n_jobs,
            scale_pos_weight=positive_weight,
        )


class XGBoostAdapter(BaseModelAdapter):
    """XGBoost gradient boosting."""

    @property
    def name(self) -> str:
        """Model name.

        Returns:
            ``"xgboost"``.
        """
        return "xgboost"

    def _build_estimator(self, positive_weight: float) -> XGBClassifier:
        """Construct the XGBoost estimator.

        Args:
            positive_weight: Weight to apply to the positive class.

        Returns:
            The unfitted estimator.
        """
        params = self._effective_params
        if self.imbalance_strategy == "focal":
            # A custom objective replaces the built-in one; leaving both set
            # would have XGBoost ignore the callable.
            params.pop("objective", None)
            params.pop("eval_metric", None)
            params["objective"] = make_focal_objective(self.focal_gamma, self.focal_alpha)
            positive_weight = 1.0

        return XGBClassifier(
            **params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            scale_pos_weight=positive_weight,
        )


class LightGBMAdapter(BaseModelAdapter):
    """LightGBM gradient boosting."""

    @property
    def name(self) -> str:
        """Model name.

        Returns:
            ``"lightgbm"``.
        """
        return "lightgbm"

    def _build_estimator(self, positive_weight: float) -> LGBMClassifier:
        """Construct the LightGBM estimator.

        Args:
            positive_weight: Weight to apply to the positive class.

        Returns:
            The unfitted estimator.
        """
        params = self._effective_params
        if self.imbalance_strategy == "focal":
            params.pop("objective", None)
            params["objective"] = make_focal_objective(self.focal_gamma, self.focal_alpha)
            positive_weight = 1.0

        return LGBMClassifier(
            **params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            scale_pos_weight=positive_weight,
        )


class RandomForestAdapter(BaseModelAdapter):
    """Random forest.

    Included as the interpretable, low-variance baseline. If a boosted model
    cannot beat it on PR-AUC, the extra complexity is not earning its keep.
    """

    @property
    def name(self) -> str:
        """Model name.

        Returns:
            ``"random_forest"``.
        """
        return "random_forest"

    def _build_estimator(self, positive_weight: float) -> RandomForestClassifier:
        """Construct the random forest estimator.

        Args:
            positive_weight: Weight to apply to the positive class.

        Returns:
            The unfitted estimator.
        """
        params = self._effective_params
        class_weight: str | dict[int, float] | None = None

        if self.imbalance_strategy == "class_weight":
            class_weight = {0: 1.0, 1: positive_weight}
        elif self.imbalance_strategy == "focal":
            _log.warning(
                "model.focal_unsupported",
                model=self.name,
                fallback="balanced_subsample",
                detail="a random forest has no gradient objective to replace",
            )
            class_weight = "balanced_subsample"

        return RandomForestClassifier(
            **params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=class_weight,
        )


#: Model name to adapter class.
MODEL_REGISTRY: dict[str, type[BaseModelAdapter]] = {
    "catboost": CatBoostAdapter,
    "xgboost": XGBoostAdapter,
    "lightgbm": LightGBMAdapter,
    "random_forest": RandomForestAdapter,
}


def build_model(
    name: str,
    spec: ModelSpec,
    *,
    random_state: int = 42,
    n_jobs: int = -1,
    imbalance_strategy: str = "class_weight",
    cost_matrix: Any = None,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    param_overrides: dict[str, Any] | None = None,
) -> BaseModelAdapter:
    """Construct a model adapter by name.

    Args:
        name: Model name; one of :data:`MODEL_REGISTRY`.
        spec: Configuration supplying the static hyperparameters.
        random_state: Seed for reproducibility.
        n_jobs: Worker count where supported.
        imbalance_strategy: Imbalance handling mode.
        cost_matrix: Business costs supplying the positive-class weight.
        focal_gamma: Focusing parameter for the focal objective.
        focal_alpha: Positive-class weight for the focal objective.
        param_overrides: Hyperparameters that win over ``spec.params``; this is
            how a tuning trial injects its sampled values.

    Returns:
        The configured, unfitted adapter.

    Raises:
        ModelTrainingError: If ``name`` matches no known adapter.
    """
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ModelTrainingError("Unknown model", model=name, known=sorted(MODEL_REGISTRY))

    params = {**dict(spec.params or {}), **(param_overrides or {})}
    return MODEL_REGISTRY[key](
        params=params,
        random_state=random_state,
        n_jobs=n_jobs,
        imbalance_strategy=imbalance_strategy,
        cost_matrix=cost_matrix,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
