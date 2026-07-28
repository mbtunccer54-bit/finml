"""Feature matrix construction.

Every model, every explainer and the counterfactual generator operate on one
shared numeric matrix produced here. Pushing encoding into each model's own
pipeline would be more idiomatic sklearn, but it would mean SHAP, LIME and DiCE
each see a different feature space and their attributions would stop being
comparable — which defeats the purpose of running three of them.

The builder is fitted on the training split only. Fitting on the full frame
would leak the test distribution into the imputation and scaling statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from domain.entities import FEATURE_CONTRACT
from domain.exceptions import SchemaValidationError
from domain.value_objects import FeatureContract
from infrastructure.config.schemas import FeatureConfig
from infrastructure.logging import get_logger

__all__ = ["FeatureMatrixBuilder", "PreprocessingArtifacts"]

_log = get_logger(__name__)


@dataclass(slots=True)
class PreprocessingArtifacts:
    """Statistics learned from the training split.

    Attributes:
        feature_names: Output column names, in matrix order.
        numeric_names: Input numeric columns retained.
        categorical_names: Input categorical columns expanded.
        category_levels: Observed levels per categorical column.
        fill_values: Imputation value per numeric column.
        means: Column means used for scaling.
        scales: Column standard deviations used for scaling.
        dropped_correlated: Numeric columns dropped for redundancy.
    """

    feature_names: list[str] = field(default_factory=list)
    numeric_names: list[str] = field(default_factory=list)
    categorical_names: list[str] = field(default_factory=list)
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    fill_values: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    dropped_correlated: list[str] = field(default_factory=list)


class FeatureMatrixBuilder:
    """Turns a contract-shaped dataframe into a numeric model matrix.

    Attributes:
        config: Feature engineering configuration.
        contract: Contract describing the input feature space.
        artifacts: Statistics learned during :meth:`fit`.
    """

    def __init__(
        self,
        config: FeatureConfig,
        contract: FeatureContract = FEATURE_CONTRACT,
    ) -> None:
        """Initialise the builder.

        Args:
            config: Feature engineering configuration.
            contract: Contract describing the input feature space.
        """
        self.config = config
        self.contract = contract
        self.artifacts = PreprocessingArtifacts()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the builder has learned its statistics.

        Returns:
            ``True`` once :meth:`fit` has been called.
        """
        return self._fitted

    @property
    def feature_names(self) -> list[str]:
        """Output column names, in matrix order.

        Returns:
            The output feature names.

        Raises:
            SchemaValidationError: If the builder is not yet fitted.
        """
        self._require_fitted()
        return list(self.artifacts.feature_names)

    def _require_fitted(self) -> None:
        """Guard methods that need learned statistics.

        Raises:
            SchemaValidationError: If the builder is not yet fitted.
        """
        if not self._fitted:
            raise SchemaValidationError(
                "FeatureMatrixBuilder must be fitted before use",
                hint="Call fit() on the training split first",
            )

    def _selected_inputs(self) -> tuple[list[str], list[str]]:
        """Determine which contract columns take part.

        Returns:
            The numeric and categorical input column names.
        """
        from domain.entities import MACRO_VARIABLES

        nlp_columns = {
            "sentiment_compound",
            "sentiment_uncertainty",
            "sentiment_negative_prob",
            "topic_credit_risk_score",
            "topic_liquidity_risk_score",
            "topic_operational_risk_score",
            "news_volume",
        }

        numeric: list[str] = []
        for name in self.contract.numeric_feature_names:
            if not self.config.use_nlp_features and name in nlp_columns:
                continue
            if not self.config.use_macro_features and name in MACRO_VARIABLES:
                continue
            numeric.append(name)
        return numeric, list(self.contract.categorical_feature_names)

    def fit(self, df: pd.DataFrame) -> FeatureMatrixBuilder:
        """Learn imputation, encoding and scaling statistics.

        Args:
            df: The training split.

        Returns:
            This builder, for chaining.

        Raises:
            SchemaValidationError: If required columns are absent.
        """
        numeric, categorical = self._selected_inputs()
        missing = [c for c in (*numeric, *categorical) if c not in df.columns]
        if missing:
            raise SchemaValidationError(
                "Training frame is missing contract columns", missing=missing
            )

        fill_values: dict[str, float] = {}
        for name in numeric:
            spec = self.contract.get(name)
            if self.config.imputation == "median":
                value = float(df[name].median())
                fill_values[name] = value if np.isfinite(value) else spec.typical
            else:
                fill_values[name] = spec.typical

        kept_numeric = self._drop_correlated(df, numeric)

        levels: dict[str, list[str]] = {}
        for name in categorical:
            spec = self.contract.get(name)
            # Prefer the contract's declared levels so a level absent from the
            # training split still gets a (zero) column, keeping the matrix
            # width stable at inference time.
            observed = sorted(df[name].dropna().astype(str).unique().tolist())
            levels[name] = list(spec.allowed) if spec.allowed else observed

        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        if self.config.scale_numeric:
            for name in kept_numeric:
                filled = df[name].fillna(fill_values[name]).astype(float)
                mean = float(filled.mean())
                std = float(filled.std(ddof=0))
                means[name] = mean if np.isfinite(mean) else 0.0
                # A constant column would divide by zero; leave it unscaled.
                scales[name] = std if np.isfinite(std) and std > 1e-12 else 1.0

        feature_names = list(kept_numeric)
        for name in categorical:
            feature_names.extend(f"{name}={level}" for level in levels[name])

        self.artifacts = PreprocessingArtifacts(
            feature_names=feature_names,
            numeric_names=kept_numeric,
            categorical_names=categorical,
            category_levels=levels,
            fill_values=fill_values,
            means=means,
            scales=scales,
            dropped_correlated=[c for c in numeric if c not in kept_numeric],
        )
        self._fitted = True

        _log.info(
            "preprocessing.fitted",
            n_input_numeric=len(numeric),
            n_kept_numeric=len(kept_numeric),
            n_dropped_correlated=len(self.artifacts.dropped_correlated),
            dropped=self.artifacts.dropped_correlated,
            n_categorical=len(categorical),
            n_output_features=len(feature_names),
            scaled=self.config.scale_numeric,
            imputation=self.config.imputation,
        )
        return self

    def _drop_correlated(self, df: pd.DataFrame, numeric: list[str]) -> list[str]:
        """Drop one column from each near-duplicate pair.

        Args:
            df: The training split.
            numeric: Candidate numeric columns.

        Returns:
            The columns to keep, in input order.
        """
        threshold = self.config.drop_correlated_above
        if threshold >= 1.0 or len(numeric) < 2:
            return list(numeric)

        corr = df[numeric].corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        dropped = {col for col in upper.columns if (upper[col] >= threshold).any()}
        return [c for c in numeric if c not in dropped]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project a dataframe onto the learned feature space.

        Args:
            df: Frame to transform.

        Returns:
            A numeric dataframe with exactly :attr:`feature_names` as columns,
            in the learned order, with the input's index preserved.

        Raises:
            SchemaValidationError: If the builder is not fitted.
        """
        self._require_fitted()
        art = self.artifacts
        out = pd.DataFrame(index=df.index)

        for name in art.numeric_names:
            if name in df.columns:
                column = pd.to_numeric(df[name], errors="coerce")
            else:
                _log.warning("preprocessing.missing_column", column=name, action="imputed")
                column = pd.Series(np.nan, index=df.index)
            column = column.fillna(art.fill_values.get(name, 0.0)).astype(float)
            # Non-finite values survive fillna; they would poison a tree split.
            column = column.replace([np.inf, -np.inf], art.fill_values.get(name, 0.0))
            if self.config.scale_numeric:
                column = (column - art.means.get(name, 0.0)) / art.scales.get(name, 1.0)
            out[name] = column

        for name in art.categorical_names:
            raw = (
                df[name].astype(str)
                if name in df.columns
                else pd.Series("", index=df.index, dtype=str)
            )
            for level in art.category_levels[name]:
                out[f"{name}={level}"] = (raw == level).astype(float)

        return out[art.feature_names]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on a frame and transform it.

        Args:
            df: The training split.

        Returns:
            The transformed matrix.
        """
        return self.fit(df).transform(df)

    def inverse_feature(self, encoded_name: str) -> str:
        """Map an output column back to its source contract column.

        Used by the explainers so a one-hot column reports under the business
        feature an analyst recognises rather than ``sector=utilities``.

        Args:
            encoded_name: Output column name.

        Returns:
            The originating contract column name.
        """
        return encoded_name.split("=", 1)[0] if "=" in encoded_name else encoded_name
