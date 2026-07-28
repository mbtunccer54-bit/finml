"""Architectural constraint tests.

Clean Architecture only holds if the dependency rule is enforced mechanically.
Stated in a README it decays within a sprint; asserted in CI it survives.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"


def _imported_roots(path: Path) -> set[str]:
    """Collect the root package of every import in a module.

    Args:
        path: Module path.

    Returns:
        The distinct root package names imported.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class TestDomainPurity:
    """The domain must depend on nothing but the standard library."""

    def test_no_forbidden_imports(self) -> None:
        """The domain layer imports no third-party or outer-layer package."""
        import sys

        sys.path.insert(0, str(_ROOT / "scripts"))
        from check_domain_purity import check_domain

        violations = check_domain(_SRC / "domain")
        assert not violations, "\n".join(violations)

    def test_domain_imports_cleanly_without_the_scientific_stack(self) -> None:
        """Every domain module is importable on its own."""
        import importlib

        for name in ("exceptions", "value_objects", "entities", "repositories", "services"):
            assert importlib.import_module(f"domain.{name}") is not None


class TestDependencyRule:
    """Dependencies point inwards only."""

    def test_domain_does_not_import_outer_layers(self) -> None:
        """Nothing in the domain may reference application or infrastructure."""
        for path in (_SRC / "domain").rglob("*.py"):
            roots = _imported_roots(path)
            assert not roots & {"application", "infrastructure", "presentation"}, path

    def test_application_does_not_import_presentation(self) -> None:
        """Use cases must not depend on the UI."""
        for path in (_SRC / "application").rglob("*.py"):
            assert "presentation" not in _imported_roots(path), path

    def test_infrastructure_does_not_import_presentation(self) -> None:
        """Adapters must not depend on the UI."""
        for path in (_SRC / "infrastructure").rglob("*.py"):
            assert "presentation" not in _imported_roots(path), path


class TestNoPrintStatements:
    """`print` is banned; every event goes through structlog."""

    def test_source_contains_no_print_calls(self) -> None:
        """No module under src/ calls print()."""
        offenders: list[str] = []
        for path in _SRC.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
        assert not offenders, "print() found at: " + ", ".join(offenders)


class TestNoSilentExceptionHandling:
    """No `except` block may be silently empty."""

    def test_no_bare_pass_handlers(self) -> None:
        """An except block containing only `pass` swallows the failure."""
        offenders: list[str] = []
        for path in _SRC.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                body = [s for s in node.body if not isinstance(s, ast.Expr)]
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
        assert not offenders, "silent except at: " + ", ".join(offenders)


class TestProtocolImplementations:
    """Adapters must satisfy the ports they claim to."""

    def test_synthetic_fetcher_satisfies_panel_source(self) -> None:
        """The fetcher is structurally a PanelSource."""
        from application.interfaces import PanelSource
        from infrastructure.config.schemas import DataConfig
        from infrastructure.data.fetchers import SyntheticPanelFetcher

        assert isinstance(SyntheticPanelFetcher(DataConfig()), PanelSource)

    def test_builder_satisfies_feature_builder(self) -> None:
        """The preprocessing builder is structurally a FeatureBuilder."""
        from application.interfaces import FeatureBuilder
        from infrastructure.config.schemas import FeatureConfig
        from infrastructure.data.preprocessing import FeatureMatrixBuilder

        assert isinstance(FeatureMatrixBuilder(FeatureConfig()), FeatureBuilder)

    def test_drift_detector_satisfies_drift_monitor(self) -> None:
        """The detector is structurally a DriftMonitor."""
        from application.interfaces import DriftMonitor
        from infrastructure.config.schemas import DriftConfig
        from infrastructure.data.drift_detector import DriftDetector

        assert isinstance(DriftDetector(DriftConfig()), DriftMonitor)

    def test_splitters_satisfy_split_strategy(self) -> None:
        """Every splitter is structurally a SplitStrategy."""
        from application.interfaces import SplitStrategy
        from infrastructure.config.schemas import SplitConfig
        from infrastructure.data.splitters import build_splitter

        for strategy in ("purged_kfold", "blocked", "expanding", "sliding"):
            assert isinstance(build_splitter(SplitConfig(strategy=strategy)), SplitStrategy)

    def test_sentiment_analyser_satisfies_sentiment_port(self) -> None:
        """The analyser is structurally a SentimentPort."""
        from application.interfaces import SentimentPort
        from infrastructure.config.schemas import NLPConfig
        from infrastructure.nlp.finbert import FinBERTSentimentAnalyzer

        assert isinstance(FinBERTSentimentAnalyzer(NLPConfig()), SentimentPort)


class TestConfiguration:
    """Hydra composition must stay valid."""

    def test_default_config_composes(self) -> None:
        """The shipped configuration tree composes without error."""
        from infrastructure.config.loader import load_config, to_object

        config = to_object(load_config())
        assert config.seed == 42
        assert config.data.split.strategy == "purged_kfold"
        assert config.model.cost.false_negative_cost == 10.0

    def test_overrides_apply(self) -> None:
        """CLI-style overrides reach the typed object."""
        from infrastructure.config.loader import load_config, to_object

        config = to_object(load_config(["seed=7", "data.n_entities=99"]))
        assert config.seed == 7
        assert config.data.n_entities == 99
        assert config.training.seed == 7  # interpolated from ${seed}

    def test_fast_model_profile_composes(self) -> None:
        """The CI profile is selectable and disables tuning."""
        from infrastructure.config.loader import load_config, to_object

        config = to_object(load_config(["models@model=fast"]))
        assert not config.model.tuning.enabled

    def test_unknown_override_is_rejected(self) -> None:
        """A typo in an override fails at compose time, not mid-training."""
        from domain.exceptions import ConfigurationError
        from infrastructure.config.loader import load_config

        with pytest.raises(ConfigurationError):
            load_config(["data.does_not_exist=1"])

    def test_scenarios_are_configured(self) -> None:
        """The CCAR scenario set is present and well-formed."""
        from infrastructure.config.loader import load_config, to_object

        config = to_object(load_config())
        names = {s.name for s in config.scenario.scenarios}
        assert {"baseline", "adverse", "severely_adverse"} <= names
