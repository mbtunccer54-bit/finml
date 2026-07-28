"""Data contract tests.

The pandera schemas are compiled from :data:`domain.entities.FEATURE_CONTRACT`,
so these tests verify two things: that the compilation is faithful, and that the
contract actually rejects the malformed data it claims to.

A silently weakened contract is worse than none — it produces a green pipeline
on data nobody validated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domain.entities import DATE_COLUMN, ENTITY_ID_COLUMN, FEATURE_CONTRACT, TARGET_COLUMN
from domain.exceptions import SchemaValidationError

pytestmark = pytest.mark.data


class TestSchemaDerivation:
    """The schema must mirror the domain contract."""

    def test_training_schema_covers_every_contract_column(self) -> None:
        """Nothing declared in the domain is missing from the schema."""
        from infrastructure.data.validators import TRAINING_SCHEMA

        expected = {spec.name for spec in FEATURE_CONTRACT.fields}
        assert expected == set(TRAINING_SCHEMA.columns)

    def test_inference_schema_omits_the_label(self) -> None:
        """At scoring time the outcome is unknown by construction."""
        from infrastructure.data.validators import INFERENCE_SCHEMA

        assert TARGET_COLUMN not in INFERENCE_SCHEMA.columns

    def test_bounds_are_carried_onto_the_schema(self) -> None:
        """Declared bounds must become real checks, not documentation."""
        from infrastructure.data.validators import TRAINING_SCHEMA

        bounded = [s for s in FEATURE_CONTRACT.fields if s.ge is not None or s.le is not None]
        assert bounded
        for spec in bounded:
            assert TRAINING_SCHEMA.columns[spec.name].checks, spec.name

    def test_categorical_levels_are_enforced(self) -> None:
        """The permitted sector set is a schema check."""
        from infrastructure.data.validators import TRAINING_SCHEMA

        checks = TRAINING_SCHEMA.columns["sector"].checks
        assert any("allowed" in str(check.name) for check in checks)


class TestValidationBehaviour:
    """What the contract accepts and rejects."""

    def test_accepts_generated_data(self, panel: pd.DataFrame) -> None:
        """The synthetic generator must satisfy its own contract."""
        from infrastructure.data.validators import TRAINING_SCHEMA, validate_dataframe

        validated = validate_dataframe(panel, TRAINING_SCHEMA, context="test")
        assert len(validated) == len(panel)

    def test_rejects_out_of_bounds_values(self, panel: pd.DataFrame) -> None:
        """A ratio outside its plausible range is a data-quality failure."""
        from infrastructure.data.validators import TRAINING_SCHEMA, validate_dataframe

        corrupt = panel.copy()
        corrupt.loc[corrupt.index[0], "current_ratio"] = 1e9
        with pytest.raises(SchemaValidationError) as info:
            validate_dataframe(corrupt, TRAINING_SCHEMA, context="test")
        assert info.value.context["n_failure_cases"] >= 1

    def test_rejects_unknown_sector(self, panel: pd.DataFrame) -> None:
        """A sector outside the enumeration is rejected."""
        from infrastructure.data.validators import TRAINING_SCHEMA, validate_dataframe

        corrupt = panel.copy()
        corrupt.loc[corrupt.index[0], "sector"] = "crypto"
        with pytest.raises(SchemaValidationError):
            validate_dataframe(corrupt, TRAINING_SCHEMA, context="test")

    def test_rejects_nulls_in_non_nullable_columns(self, panel: pd.DataFrame) -> None:
        """Nulls are not permitted where the contract forbids them."""
        from infrastructure.data.validators import TRAINING_SCHEMA, validate_dataframe

        corrupt = panel.copy()
        corrupt.loc[corrupt.index[0], "debt_to_assets"] = np.nan
        with pytest.raises(SchemaValidationError):
            validate_dataframe(corrupt, TRAINING_SCHEMA, context="test")

    def test_rejects_duplicate_observations(self, panel: pd.DataFrame) -> None:
        """A duplicated (entity, date) row would be counted twice."""
        from infrastructure.data.validators import TRAINING_SCHEMA, validate_dataframe

        corrupt = pd.concat([panel, panel.head(1)], ignore_index=True)
        with pytest.raises(SchemaValidationError):
            validate_dataframe(corrupt, TRAINING_SCHEMA, context="test")

    def test_rejects_an_empty_frame(self) -> None:
        """An empty batch is a caller error worth surfacing."""
        from infrastructure.data.validators import TRAINING_SCHEMA, validate_dataframe

        with pytest.raises(SchemaValidationError):
            validate_dataframe(pd.DataFrame(), TRAINING_SCHEMA, context="test")

    def test_failure_report_names_the_offending_column(self, panel: pd.DataFrame) -> None:
        """A failure must localise the problem for whoever has to fix it."""
        from infrastructure.data.validators import TRAINING_SCHEMA, validate_dataframe

        corrupt = panel.copy()
        corrupt.loc[corrupt.index[0], "debt_to_assets"] = 99.0
        with pytest.raises(SchemaValidationError) as info:
            validate_dataframe(corrupt, TRAINING_SCHEMA, context="test")
        rendered = str(info.value.context["failures"])
        assert "debt_to_assets" in rendered

    def test_coercion_normalises_dtypes(self, panel: pd.DataFrame) -> None:
        """Numeric columns arriving as strings are coerced, not rejected."""
        from infrastructure.data.validators import TRAINING_SCHEMA, validate_dataframe

        stringly = panel.copy()
        stringly["current_ratio"] = stringly["current_ratio"].astype(str)
        validated = validate_dataframe(stringly, TRAINING_SCHEMA, context="test")
        assert validated["current_ratio"].dtype.kind == "f"


class TestContractCoercion:
    """Reshaping arbitrary input to the model's feature space."""

    def test_fills_missing_features_from_the_contract(self, panel: pd.DataFrame) -> None:
        """A partially observed obligor is still scoreable."""
        from infrastructure.data.validators import coerce_to_contract

        result = coerce_to_contract(panel.drop(columns=["current_ratio", "quick_ratio"]))
        assert list(result.columns) == list(FEATURE_CONTRACT.feature_names)
        assert result["current_ratio"].iloc[0] == FEATURE_CONTRACT.get("current_ratio").typical

    def test_enforces_canonical_column_order(self, panel: pd.DataFrame) -> None:
        """Tree models index features positionally once fitted."""
        from infrastructure.data.validators import coerce_to_contract

        shuffled = panel[list(reversed(panel.columns))]
        assert list(coerce_to_contract(shuffled).columns) == list(FEATURE_CONTRACT.feature_names)

    def test_can_refuse_to_impute(self, panel: pd.DataFrame) -> None:
        """Strict callers can require every feature to be present."""
        from infrastructure.data.validators import coerce_to_contract

        with pytest.raises(SchemaValidationError):
            coerce_to_contract(panel.drop(columns=["current_ratio"]), fill_missing=False)


class TestGeneratedDataQuality:
    """The synthetic generator is a test fixture, so it gets tested too."""

    def test_hits_the_configured_default_rate(self, panel: pd.DataFrame) -> None:
        """Prevalence is solved for, not left to chance."""
        assert panel[TARGET_COLUMN].mean() == pytest.approx(0.10, abs=0.04)

    def test_panel_is_balanced_across_periods(self, panel: pd.DataFrame) -> None:
        """Every obligor is observed in every period."""
        counts = panel.groupby(DATE_COLUMN)[ENTITY_ID_COLUMN].nunique()
        assert counts.nunique() == 1

    def test_label_depends_on_the_features(self, panel: pd.DataFrame) -> None:
        """A dataset with no signal would make every model test vacuous."""
        defaulters = panel[panel[TARGET_COLUMN] == 1]
        healthy = panel[panel[TARGET_COLUMN] == 0]
        assert defaulters["altman_z_score"].mean() < healthy["altman_z_score"].mean()
        assert defaulters["debt_to_assets"].mean() > healthy["debt_to_assets"].mean()
        assert defaulters["interest_coverage"].mean() < healthy["interest_coverage"].mean()

    def test_is_reproducible(self, data_config: object) -> None:
        """The same seed must produce byte-identical data."""
        from infrastructure.data.fetchers import SyntheticPanelFetcher

        first = SyntheticPanelFetcher(data_config).fetch().panel  # type: ignore[arg-type]
        second = SyntheticPanelFetcher(data_config).fetch().panel  # type: ignore[arg-type]
        pd.testing.assert_frame_equal(first, second)

    def test_macro_history_has_a_cycle(self, dataset: object) -> None:
        """A flat macro path would make the satellite model and scenarios vacuous."""
        macro = dataset.macro  # type: ignore[attr-defined]
        assert macro["gdp_growth"].min() < 0.0 < macro["gdp_growth"].max()
        assert macro["unemployment_rate"].std() > 0.0

    def test_rejects_degenerate_configuration(self) -> None:
        """An impossible panel shape is a configuration error."""
        from domain.exceptions import DataFetchError
        from infrastructure.config.schemas import DataConfig
        from infrastructure.data.fetchers import SyntheticPanelFetcher

        with pytest.raises(DataFetchError):
            SyntheticPanelFetcher(DataConfig(n_entities=1, n_periods=1))
        with pytest.raises(DataFetchError):
            SyntheticPanelFetcher(DataConfig(default_rate=0.0))
