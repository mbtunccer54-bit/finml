"""FastAPI integration tests.

The API is the contract other systems hold, so these tests assert on the wire
format as much as the numbers: status codes, error bodies and the Prometheus
exposition.

The service loads its champion at startup, so the module-scoped client trains
once and reuses it.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Train a model, then start the API against it.

    Args:
        tmp_path_factory: Pytest temporary directory factory.

    Yields:
        A configured :class:`~fastapi.testclient.TestClient`.
    """
    from application.train_pipeline import TrainPipeline
    from infrastructure.config.loader import cached_config, load_config, to_object

    workspace = tmp_path_factory.mktemp("api")
    overrides = [
        "models@model=fast",
        "data.n_entities=200",
        "data.n_periods=8",
        "data.split.n_splits=3",
        "training.compute_shap_on_train=false",
        "tracking.enabled=false",
        f"paths.artifacts={workspace / 'artifacts'}",
        f"paths.reports={workspace / 'artifacts' / 'reports'}",
        f"paths.feature_store={workspace / 'store'}",
    ]
    TrainPipeline(to_object(load_config(overrides))).run()

    # The API composes its own configuration at startup, so point it at the
    # same artifact directory through the environment.
    os.environ["FINML_ARTIFACT_DIR"] = str(workspace / "artifacts")
    cached_config.cache_clear() if hasattr(cached_config, "cache_clear") else None

    from infrastructure.api import dependencies
    from infrastructure.api.fastapi_app import app

    dependencies.state.config = to_object(load_config(overrides))
    from application.inference_pipeline import InferencePipeline
    from application.scenario_simulator import ScenarioSimulator

    pipeline = InferencePipeline.from_champion(dependencies.state.config)
    dependencies.state.pipeline = pipeline
    dependencies.state.simulator = ScenarioSimulator(
        dependencies.state.config,
        scoring_fn=lambda x: pipeline.bundle.model.predict_proba(x)[:, 1],
    )
    dependencies.state.checks = {
        "config_loaded": True,
        "model_loaded": True,
        "scenarios_loaded": True,
    }

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def risky() -> dict[str, Any]:
    """A distressed obligor payload.

    Returns:
        The payload.
    """
    return {
        "entity_id": "DISTRESSED",
        "current_ratio": 0.35,
        "quick_ratio": 0.2,
        "working_capital_ratio": -0.4,
        "cash_flow_to_debt": -0.3,
        "debt_to_equity": 9.0,
        "debt_to_assets": 0.95,
        "interest_coverage": -3.0,
        "return_on_assets": -0.25,
        "return_on_equity": -0.9,
        "net_profit_margin": -0.3,
        "operating_margin": -0.2,
        "asset_turnover": 0.3,
        "revenue_growth": -0.4,
        "altman_z_score": -1.5,
        "sentiment_compound": -0.85,
        "sentiment_uncertainty": 0.3,
        "sentiment_negative_prob": 0.9,
        "topic_credit_risk_score": 0.95,
        "topic_liquidity_risk_score": 0.9,
        "topic_operational_risk_score": 0.6,
        "news_volume": 45.0,
        "sector": "consumer_discretionary",
    }


@pytest.fixture
def healthy() -> dict[str, Any]:
    """A strong obligor payload.

    Returns:
        The payload.
    """
    return {
        "entity_id": "SOLID",
        "current_ratio": 2.8,
        "quick_ratio": 2.0,
        "working_capital_ratio": 0.45,
        "cash_flow_to_debt": 1.2,
        "debt_to_equity": 0.3,
        "debt_to_assets": 0.15,
        "interest_coverage": 18.0,
        "return_on_assets": 0.16,
        "return_on_equity": 0.28,
        "net_profit_margin": 0.22,
        "operating_margin": 0.3,
        "asset_turnover": 1.3,
        "revenue_growth": 0.18,
        "altman_z_score": 6.5,
        "sentiment_compound": 0.7,
        "sentiment_uncertainty": 0.15,
        "sentiment_negative_prob": 0.05,
        "topic_credit_risk_score": 0.03,
        "topic_liquidity_risk_score": 0.02,
        "topic_operational_risk_score": 0.02,
        "news_volume": 6.0,
        "sector": "utilities",
    }


class TestHealth:
    """Readiness reporting."""

    def test_reports_ok_with_a_loaded_model(self, client: TestClient) -> None:
        """Health must name the model actually serving traffic."""
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"]
        assert body["model_id"]
        assert body["n_features"] > 0

    def test_reports_drift_configuration(self, client: TestClient) -> None:
        """Operators need to see the fail-safe policy in force."""
        body = client.get("/health").json()
        assert body["drift_monitoring"] is True
        assert body["drift_policy"] in ("block", "warn")


class TestPredict:
    """Scoring."""

    def test_scores_a_single_obligor(self, client: TestClient, risky: dict[str, Any]) -> None:
        """A single-entity request returns one well-formed prediction."""
        response = client.post("/predict", json={"entities": [risky]})
        assert response.status_code == 200
        body = response.json()
        assert body["n_scored"] == 1
        prediction = body["predictions"][0]
        assert 0.0 <= prediction["probability_of_default"] <= 1.0
        assert prediction["risk_band"]
        assert prediction["entity_id"] == "DISTRESSED"

    def test_ranks_a_distressed_obligor_above_a_healthy_one(
        self, client: TestClient, risky: dict[str, Any], healthy: dict[str, Any]
    ) -> None:
        """The model must order obvious cases correctly."""
        body = client.post("/predict", json={"entities": [risky, healthy]}).json()
        scores = {p["entity_id"]: p["probability_of_default"] for p in body["predictions"]}
        assert scores["DISTRESSED"] > scores["SOLID"]

    def test_credit_score_is_inverse_to_risk(
        self, client: TestClient, risky: dict[str, Any], healthy: dict[str, Any]
    ) -> None:
        """Higher PD must map to a lower scorecard number."""
        body = client.post("/predict", json={"entities": [risky, healthy]}).json()
        scores = {p["entity_id"]: p["credit_score"] for p in body["predictions"]}
        assert scores["DISTRESSED"] < scores["SOLID"]

    def test_batch_scoring(self, client: TestClient, healthy: dict[str, Any]) -> None:
        """A batch returns one prediction per entity."""
        entities = [{**healthy, "entity_id": f"E{i}"} for i in range(12)]
        body = client.post("/predict", json={"entities": entities}).json()
        assert body["n_scored"] == 12
        assert len({p["entity_id"] for p in body["predictions"]}) == 12

    def test_explanations_can_be_attached(self, client: TestClient, risky: dict[str, Any]) -> None:
        """`explain=true` returns drivers alongside the score."""
        body = client.post("/predict", json={"entities": [risky], "explain": True}).json()
        drivers = body["predictions"][0]["top_drivers"]
        assert drivers
        assert {"feature", "contribution", "direction"} <= set(drivers[0])

    def test_overlay_increases_risk(self, client: TestClient, healthy: dict[str, Any]) -> None:
        """A positive overlay is more conservative."""
        base = client.post("/predict", json={"entities": [healthy]}).json()
        overlaid = client.post(
            "/predict", json={"entities": [healthy], "overlay_log_odds": 2.0}
        ).json()
        assert (
            overlaid["predictions"][0]["probability_of_default"]
            > base["predictions"][0]["probability_of_default"]
        )

    def test_response_carries_model_provenance(
        self, client: TestClient, healthy: dict[str, Any]
    ) -> None:
        """Every score must be traceable to a model version."""
        body = client.post("/predict", json={"entities": [healthy]}).json()
        assert body["model_id"]
        assert body["model_version"]
        assert body["latency_ms"] >= 0.0

    def test_partial_payload_is_accepted(self, client: TestClient) -> None:
        """A partially observed obligor is scored on contract defaults."""
        response = client.post(
            "/predict", json={"entities": [{"entity_id": "PARTIAL", "current_ratio": 1.2}]}
        )
        assert response.status_code == 200


class TestValidation:
    """The API's input contract."""

    def test_rejects_out_of_bounds_values(self, client: TestClient) -> None:
        """Contract bounds are enforced at the edge."""
        response = client.post("/predict", json={"entities": [{"current_ratio": -5.0}]})
        assert response.status_code == 422

    def test_rejects_unknown_fields(self, client: TestClient) -> None:
        """A typo'd field name must fail loudly, not be silently ignored."""
        response = client.post("/predict", json={"entities": [{"not_a_feature": 1.0}]})
        assert response.status_code == 422

    def test_rejects_an_unknown_sector(self, client: TestClient) -> None:
        """The sector enumeration is closed."""
        response = client.post("/predict", json={"entities": [{"sector": "crypto"}]})
        assert response.status_code == 422

    def test_rejects_an_empty_batch(self, client: TestClient) -> None:
        """An empty request is a caller error."""
        assert client.post("/predict", json={"entities": []}).status_code == 422

    def test_rejects_an_oversized_batch(self, client: TestClient, healthy: dict[str, Any]) -> None:
        """The batch cap protects the worker from a memory blow-up."""
        entities = [healthy] * 1001
        assert client.post("/predict", json={"entities": entities}).status_code == 413


class TestExplain:
    """Explanation endpoints."""

    def test_shap_returns_ranked_contributions(
        self, client: TestClient, risky: dict[str, Any]
    ) -> None:
        """Contributions come back ranked by absolute influence."""
        body = client.post("/explain", json={"entity": risky, "method": "shap", "top_k": 5}).json()
        contributions = body["contributions"]
        assert 0 < len(contributions) <= 5
        magnitudes = [abs(c["contribution"]) for c in contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_lime_returns_contributions(self, client: TestClient, risky: dict[str, Any]) -> None:
        """LIME is available as an independent method."""
        body = client.post("/explain", json={"entity": risky, "method": "lime"}).json()
        assert body["method"] == "lime"
        assert body["contributions"]

    def test_dice_returns_actionable_recourse(
        self, client: TestClient, risky: dict[str, Any]
    ) -> None:
        """Counterfactuals must lower the PD and name what to change."""
        body = client.post(
            "/explain", json={"entity": risky, "method": "dice", "target_pd": 0.1}
        ).json()
        assert body["method"] == "dice"
        for option in body["counterfactuals"]:
            assert option["counterfactual_pd"] < option["original_pd"]
            assert option["changes"]
            assert all(c["feature"] for c in option["changes"])

    def test_rejects_an_unknown_method(self, client: TestClient, risky: dict[str, Any]) -> None:
        """The method enumeration is closed."""
        response = client.post("/explain", json={"entity": risky, "method": "magic"})
        assert response.status_code == 422


class TestScenario:
    """Stress testing."""

    def test_runs_every_configured_scenario(
        self, client: TestClient, risky: dict[str, Any], healthy: dict[str, Any]
    ) -> None:
        """Omitting a name runs the whole configured set."""
        body = client.post("/scenario", json={"entities": [risky, healthy]}).json()
        names = {o["scenario"] for o in body["outcomes"]}
        assert {"baseline", "adverse", "severely_adverse"} <= names

    def test_severity_ordering(self, client: TestClient, healthy: dict[str, Any]) -> None:
        """Stressed PD must rise with scenario severity."""
        body = client.post("/scenario", json={"entities": [healthy] * 5}).json()
        outcomes = {o["scenario"]: o["stressed_pd"] for o in body["outcomes"]}
        assert outcomes["baseline"] < outcomes["adverse"] < outcomes["severely_adverse"]

    def test_single_scenario_selection(self, client: TestClient, healthy: dict[str, Any]) -> None:
        """A named scenario runs alone."""
        body = client.post("/scenario", json={"entities": [healthy], "scenario": "adverse"}).json()
        assert len(body["outcomes"]) == 1
        assert body["outcomes"][0]["scenario"] == "adverse"

    def test_requires_entities(self, client: TestClient) -> None:
        """A scenario over nothing is a caller error."""
        assert client.post("/scenario", json={"entities": []}).status_code == 422

    def test_catalogue_exposes_the_transmission_matrix(self, client: TestClient) -> None:
        """The bridge is inspectable without running a portfolio."""
        body = client.get("/scenarios").json()
        assert body["scenarios"]
        impact = body["sector_impact"]
        assert (
            impact["severely_adverse"]["consumer_discretionary"]
            > impact["severely_adverse"]["utilities"]
        )


class TestObservability:
    """Metrics exposition."""

    def test_metrics_are_prometheus_formatted(
        self, client: TestClient, healthy: dict[str, Any]
    ) -> None:
        """The endpoint must be scrapeable and carry our counters."""
        client.post("/predict", json={"entities": [healthy]})
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "finml_predictions_total" in response.text
        assert "finml_request_latency_seconds" in response.text

    def test_openapi_documents_every_route(self, client: TestClient) -> None:
        """The published schema is the integration contract."""
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/predict", "/explain", "/scenario", "/health", "/metrics"} <= set(paths)
