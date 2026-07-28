"""SHAP attributions.

Two backends, chosen automatically:

* **TreeSHAP** for the tree ensembles — exact and fast enough to explain a whole
  portfolio.
* **KernelSHAP** for the stacked ensemble and calibrated wrappers, which are not
  single tree models. It is model-agnostic and far slower, so it runs against a
  summarised background set.

Most of the code here exists to normalise output shape. Depending on the model
and SHAP version, a binary classifier's values arrive as ``(n, features)``,
``(n, features, 2)``, or a two-element list of ``(n, features)`` arrays. Picking
the wrong one silently returns the attribution for *survival* rather than
*default* — every sign inverted, and nothing raises. :func:`normalise_shap_values`
resolves the shape explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from domain.entities import FeatureContribution, ScoreExplanation
from domain.exceptions import ExplainerError
from domain.value_objects import EntityId, ModelId
from infrastructure.logging import get_logger

__all__ = ["ShapExplainer", "ShapResult", "normalise_shap_values"]

_log = get_logger(__name__)


def normalise_shap_values(
    values: Any, *, n_rows: int, n_features: int, positive_class: int = 1
) -> np.ndarray:
    """Reduce any SHAP output to an ``(n_rows, n_features)`` array.

    Args:
        values: Raw output from a SHAP explainer.
        n_rows: Expected row count.
        n_features: Expected feature count.
        positive_class: Class index to extract for binary output.

    Returns:
        Attributions for the positive class.

    Raises:
        ExplainerError: If the output cannot be interpreted.
    """
    # A two-element list/tuple is the classic per-class layout.
    if isinstance(values, (list, tuple)):
        if len(values) == 2:
            values = values[positive_class]
        elif len(values) == 1:
            values = values[0]
        else:
            raise ExplainerError("Unexpected number of SHAP output classes", n_classes=len(values))

    array = np.asarray(values)

    if array.ndim == 3:
        # (n, features, classes) or (classes, n, features)
        if array.shape[:2] == (n_rows, n_features):
            return np.asarray(array[:, :, min(positive_class, array.shape[2] - 1)])
        if array.shape[1:] == (n_rows, n_features):
            return np.asarray(array[min(positive_class, array.shape[0] - 1), :, :])
        raise ExplainerError(
            "Cannot locate the class axis in the SHAP output",
            shape=tuple(array.shape),
            expected=(n_rows, n_features),
        )

    if array.ndim == 2:
        if array.shape == (n_rows, n_features):
            return array
        if array.shape == (n_features, n_rows):
            return array.T
        raise ExplainerError(
            "SHAP output shape does not match the input",
            shape=tuple(array.shape),
            expected=(n_rows, n_features),
        )

    if array.ndim == 1 and n_rows == 1 and array.shape[0] == n_features:
        return array.reshape(1, n_features)

    raise ExplainerError(
        "Uninterpretable SHAP output", shape=tuple(array.shape), expected=(n_rows, n_features)
    )


@dataclass(frozen=True, slots=True)
class ShapResult:
    """SHAP attributions for a set of rows.

    Attributes:
        values: ``(n_rows, n_features)`` attributions for the positive class.
        base_value: Explainer baseline, in the model's output space.
        feature_names: Column names matching the value columns.
        data: The feature values the attributions were computed on.
        backend: Explainer backend used.
    """

    values: np.ndarray
    base_value: float
    feature_names: list[str]
    data: pd.DataFrame
    backend: str

    def global_importance(self) -> pd.DataFrame:
        """Rank features by mean absolute attribution.

        Returns:
            A frame with ``feature``, ``mean_abs_shap``, ``mean_shap`` and
            ``direction``, most influential first.
        """
        mean_abs = np.abs(self.values).mean(axis=0)
        mean_signed = self.values.mean(axis=0)
        frame = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs,
                "mean_shap": mean_signed,
            }
        )
        frame["direction"] = np.where(frame["mean_shap"] > 0, "increases_risk", "decreases_risk")
        return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def for_row(self, index: int) -> dict[str, float]:
        """Return one row's attributions.

        Args:
            index: Positional row index.

        Returns:
            Feature name mapped to attribution.

        Raises:
            ExplainerError: If the index is out of range.
        """
        if not 0 <= index < self.values.shape[0]:
            raise ExplainerError("Row index out of range", index=index, n_rows=self.values.shape[0])
        return dict(zip(self.feature_names, self.values[index].tolist(), strict=True))


class ShapExplainer:
    """Computes SHAP attributions for a fitted model.

    Attributes:
        model: The fitted model or ``ModelBundle``-held estimator.
        background: Representative rows used as the explainer's reference.
        feature_names: Matrix column names.
        backend: ``auto``, ``tree`` or ``kernel``.
    """

    def __init__(
        self,
        model: Any,
        background: pd.DataFrame,
        *,
        feature_names: list[str] | None = None,
        backend: str = "auto",
        max_background: int = 100,
    ) -> None:
        """Initialise the explainer.

        Args:
            model: A fitted model exposing ``predict_proba``.
            background: Rows used as the reference distribution.
            feature_names: Matrix column names; taken from ``background`` when omitted.
            backend: ``auto``, ``tree`` or ``kernel``.
            max_background: Cap on background rows; KernelSHAP is quadratic in
                this, so it is summarised rather than used whole.

        Raises:
            ExplainerError: If the background set is empty.
        """
        if background.empty:
            raise ExplainerError("SHAP background set must not be empty")

        self.model = model
        self.feature_names = feature_names or list(background.columns)
        self.background = background[self.feature_names]
        self.backend = backend
        self.max_background = max_background
        self._explainer: Any = None
        self._resolved_backend = ""

    def _tree_target(self) -> Any:
        """Return the object TreeSHAP should be pointed at.

        The adapters wrap a vendor estimator; TreeSHAP needs the vendor object
        itself, not the wrapper.

        Returns:
            The underlying tree model, or ``None`` when there is not one.
        """
        candidate = getattr(self.model, "estimator_", None)
        if candidate is not None:
            return candidate
        base = getattr(self.model, "base_estimator", None)
        if base is not None:
            return getattr(base, "estimator_", base)
        return self.model

    def _supports_tree(self) -> bool:
        """Decide whether TreeSHAP is applicable.

        Returns:
            ``True`` when the model declares tree support and a vendor tree
            estimator can be reached.
        """
        if hasattr(self.model, "supports_tree_shap") and not self.model.supports_tree_shap:
            return False
        target = self._tree_target()
        tree_types = (
            "Booster",
            "CatBoost",
            "LGBM",
            "XGB",
            "RandomForest",
            "GradientBoosting",
            "DecisionTree",
            "ExtraTrees",
        )
        return any(t in type(target).__name__ for t in tree_types)

    def _build(self) -> Any:
        """Construct the SHAP explainer, falling back to KernelSHAP.

        Returns:
            The constructed explainer.

        Raises:
            ExplainerError: If no backend can be constructed.
        """
        if self._explainer is not None:
            return self._explainer

        import shap

        want_tree = self.backend == "tree" or (self.backend == "auto" and self._supports_tree())

        if want_tree:
            try:
                tree_explainer = shap.TreeExplainer(self._tree_target())
            except Exception as exc:
                if self.backend == "tree":
                    raise ExplainerError(
                        "TreeSHAP was requested but could not be built",
                        model=type(self.model).__name__,
                        reason=str(exc)[:300],
                    ) from exc
                _log.warning(
                    "xai.tree_shap_failed",
                    reason=str(exc)[:200],
                    fallback="kernel",
                )
            else:
                self._explainer = tree_explainer
                self._resolved_backend = "tree"
                _log.info("xai.shap_backend", backend="tree", model=type(self.model).__name__)
                return self._explainer

        try:
            summary = shap.kmeans(
                self.background.to_numpy(dtype=float),
                min(self.max_background, max(len(self.background) // 4, 1), 25),
            )
            self._explainer = shap.KernelExplainer(self._predict_positive, summary)
            self._resolved_backend = "kernel"
            _log.info(
                "xai.shap_backend",
                backend="kernel",
                model=type(self.model).__name__,
                n_background=len(self.background),
            )
        except Exception as exc:
            raise ExplainerError(
                "Could not construct any SHAP explainer",
                model=type(self.model).__name__,
                reason=str(exc)[:300],
            ) from exc
        return self._explainer

    def _predict_positive(self, x: Any) -> np.ndarray:
        """Score rows, returning the positive-class probability.

        Args:
            x: Rows as an array or dataframe.

        Returns:
            One probability per row.
        """
        frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x, columns=self.feature_names)
        return np.asarray(self.model.predict_proba(frame))[:, 1]

    def explain(self, x: pd.DataFrame) -> ShapResult:
        """Compute attributions for a set of rows.

        Args:
            x: Rows to explain.

        Returns:
            The attributions and their baseline.

        Raises:
            ExplainerError: If the computation fails.
        """
        explainer = self._build()
        frame = x[self.feature_names]

        try:
            if self._resolved_backend == "kernel":
                raw = explainer.shap_values(frame.to_numpy(dtype=float), silent=True)
            else:
                raw = explainer.shap_values(frame)
        except Exception as exc:
            raise ExplainerError(
                "SHAP value computation failed",
                backend=self._resolved_backend,
                n_rows=len(frame),
                reason=str(exc)[:300],
            ) from exc

        values = normalise_shap_values(raw, n_rows=len(frame), n_features=len(self.feature_names))
        base = self._extract_base_value(explainer)

        _log.debug(
            "xai.shap_computed",
            backend=self._resolved_backend,
            n_rows=len(frame),
            n_features=len(self.feature_names),
            base_value=round(base, 5),
        )
        return ShapResult(
            values=values,
            base_value=base,
            feature_names=list(self.feature_names),
            data=frame,
            backend=self._resolved_backend,
        )

    @staticmethod
    def _extract_base_value(explainer: Any) -> float:
        """Read the explainer's baseline as a scalar.

        Args:
            explainer: The SHAP explainer.

        Returns:
            The baseline for the positive class, or ``0.0`` when unavailable.
        """
        expected = getattr(explainer, "expected_value", None)
        if expected is None:
            return 0.0
        array = np.atleast_1d(np.asarray(expected, dtype=float))
        return float(array[-1]) if array.size > 1 else float(array[0])

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
            The domain explanation, ready to persist or return from the API.

        Raises:
            ExplainerError: If ``row`` is not exactly one row.
        """
        if len(row) != 1:
            raise ExplainerError("explain_instance expects exactly one row", n_rows=len(row))

        result = self.explain(row)
        contributions = tuple(
            FeatureContribution(
                feature=name,
                value=float(result.data.iloc[0][name]),
                contribution=float(result.values[0, i]),
            )
            for i, name in enumerate(result.feature_names)
        )
        return ScoreExplanation(
            entity_id=EntityId(entity_id),
            method=f"shap_{result.backend}",
            base_value=result.base_value,
            contributions=contributions,
            model_id=ModelId(model_id) if model_id else None,
        )

    def interaction_matrix(self, x: pd.DataFrame, *, top_k: int = 8) -> pd.DataFrame:
        """Compute pairwise SHAP interaction strengths for the top features.

        Interaction values are only available from TreeSHAP and cost
        ``O(features^2)``, so the matrix is restricted to the ``top_k`` features
        by main-effect importance. When they are unavailable, a correlation-of-
        attributions proxy is returned instead and logged as such — it answers a
        related but weaker question, and the caller should know which it got.

        Args:
            x: Rows to compute over.
            top_k: Number of features to retain.

        Returns:
            A square frame of interaction strengths.
        """
        result = self.explain(x)
        top_features = result.global_importance().head(top_k)["feature"].tolist()
        indices = [result.feature_names.index(f) for f in top_features]

        if self._resolved_backend == "tree":
            try:
                raw = self._explainer.shap_interaction_values(x[self.feature_names])
                array = np.asarray(raw)
                if array.ndim == 4:  # (n, f, f, classes)
                    array = array[:, :, :, -1]
                mean_abs = np.abs(array).mean(axis=0)
                sub = mean_abs[np.ix_(indices, indices)]
                _log.debug("xai.shap_interactions", backend="tree", top_k=len(indices))
                return pd.DataFrame(sub, index=top_features, columns=top_features)
            except Exception as exc:
                _log.warning(
                    "xai.shap_interactions_unavailable",
                    reason=str(exc)[:200],
                    fallback="attribution correlation proxy",
                )

        proxy = pd.DataFrame(result.values[:, indices], columns=top_features).corr().abs()
        return proxy.fillna(0.0)

    def save_summary_plot(
        self, x: pd.DataFrame, path: str | Path, *, max_display: int = 20
    ) -> Path | None:
        """Render and save a SHAP beeswarm summary plot.

        Args:
            x: Rows to plot.
            path: Destination file path.
            max_display: Features to show.

        Returns:
            The written path, or ``None`` when plotting was not possible.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        result = self.explain(x)
        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                result.values,
                result.data,
                feature_names=result.feature_names,
                max_display=max_display,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(destination, dpi=120, bbox_inches="tight")
            plt.close("all")
        except Exception as exc:
            plt.close("all")
            # A missing plot must never fail a training run.
            _log.warning("xai.summary_plot_failed", path=str(destination), reason=str(exc)[:200])
            return None
        _log.info("xai.summary_plot_saved", path=str(destination))
        return destination
