"""Shared pytest fixtures.

The expensive fixtures are session-scoped: generating a panel and fitting four
models per test would make the suite unusable. They are also all deterministic,
so a failure reproduces.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from infrastructure.config.schemas import (
    DataConfig,
    DriftConfig,
    FeatureConfig,
    ModelSpec,
    SplitConfig,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session")
def data_config() -> DataConfig:
    """A small, fast, deterministic dataset configuration.

    Returns:
        The dataset configuration.
    """
    return DataConfig(
        n_entities=200, n_periods=8, default_rate=0.10, seed=1234, validate_schema=True
    )


@pytest.fixture(scope="session")
def panel(data_config: DataConfig) -> pd.DataFrame:
    """A synthetic credit panel.

    Args:
        data_config: Dataset configuration.

    Returns:
        The generated panel, sorted chronologically.
    """
    from infrastructure.data.fetchers import SyntheticPanelFetcher

    dataset = SyntheticPanelFetcher(data_config).fetch()
    return dataset.panel.sort_values(["as_of_date", "entity_id"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def dataset(data_config: DataConfig) -> Any:
    """The full synthetic dataset including macro history and news.

    Args:
        data_config: Dataset configuration.

    Returns:
        The generated :class:`~infrastructure.data.fetchers.PanelDataset`.
    """
    from infrastructure.data.fetchers import SyntheticPanelFetcher

    return SyntheticPanelFetcher(data_config).fetch()


@pytest.fixture(scope="session")
def feature_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, Any]:
    """A fitted feature matrix, labels and the builder that produced them.

    Args:
        panel: The synthetic panel.

    Returns:
        The matrix, the labels and the fitted builder.
    """
    from infrastructure.data.preprocessing import FeatureMatrixBuilder

    builder = FeatureMatrixBuilder(FeatureConfig()).fit(panel)
    return builder.transform(panel), panel["default_flag"].to_numpy(), builder


@pytest.fixture(scope="session")
def fitted_model(feature_matrix: tuple[pd.DataFrame, np.ndarray, Any]) -> Any:
    """A fitted LightGBM adapter.

    Args:
        feature_matrix: The matrix, labels and builder.

    Returns:
        The fitted adapter.
    """
    from infrastructure.models.trainers import build_model

    x, y, _builder = feature_matrix
    spec = ModelSpec(params={"n_estimators": 40, "num_leaves": 15, "verbosity": -1})
    return build_model("lightgbm", spec, n_jobs=1).fit(x, y)


@pytest.fixture
def split_config() -> SplitConfig:
    """A small cross-validation configuration.

    Returns:
        The split configuration.
    """
    return SplitConfig(strategy="purged_kfold", n_splits=3, purge_frac=0.02, embargo_frac=0.02)


@pytest.fixture
def drift_config() -> DriftConfig:
    """A drift configuration with the default thresholds.

    Returns:
        The drift configuration.
    """
    return DriftConfig(enabled=True, min_reference_size=20, min_batch_size=10)


@pytest.fixture
def tmp_store(tmp_path: Path) -> Iterator[Any]:
    """A feature store rooted in a temporary directory.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Yields:
        The feature store.
    """
    from infrastructure.data.feature_store import FeatureStore

    yield FeatureStore(tmp_path / "store")


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """A seeded random generator.

    Returns:
        The generator.
    """
    return np.random.default_rng(20240101)
