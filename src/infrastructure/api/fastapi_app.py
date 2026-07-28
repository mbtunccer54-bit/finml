"""FastAPI inference service.

Endpoints:

* ``POST /predict``  — score one obligor or a batch, optionally with explanations
* ``POST /explain``  — SHAP, LIME or counterfactual explanation for one obligor
* ``POST /scenario`` — run macro stress scenarios over a portfolio
* ``GET  /health``   — model readiness and drift status
* ``GET  /metrics``  — Prometheus exposition

Two behaviours are worth calling out.

**Errors are structured.** A :class:`~domain.exceptions.FinMLError` becomes a
JSON body carrying ``error_type``, ``message`` and ``context``, with the status
code chosen by category — a domain violation is the caller's fault (422), an
infrastructure failure is ours (503). A caller can branch on the error without
parsing prose.

**Drift is surfaced, not hidden.** Under the ``block`` policy a severely drifted
batch returns 409 rather than a number nobody should act on. Under ``warn`` the
scores are returned with the drift status attached to the response.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from domain.exceptions import (
    DomainError,
    DriftDetectedError,
    FinMLError,
    InfrastructureError,
)
from domain.value_objects import Sector
from infrastructure.api.dependencies import (
    ServiceState,
    get_config,
    get_pipeline,
    get_simulator,
    get_state,
    state,
)
from infrastructure.api.schemas import (
    BatchPredictRequest,
    CounterfactualChange,
    CounterfactualResponse,
    DriftSummary,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureContributionModel,
    HealthResponse,
    PredictionItem,
    PredictResponse,
    ScenarioOutcome,
    ScenarioRequest,
    ScenarioResponse,
    ScenarioSectorImpact,
)
from infrastructure.config.schemas import RootConfig
from infrastructure.logging import bind_run_context, configure_logging, get_logger

__all__ = ["app", "create_app"]

_log = get_logger(__name__)

# --- Prometheus metrics -------------------------------------------------------
REQUESTS = Counter("finml_requests_total", "Requests handled", ["endpoint", "status"])
LATENCY = Histogram(
    "finml_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter("finml_predictions_total", "Obligors scored")
FLAGGED = Counter("finml_predictions_flagged_total", "Obligors above threshold")
DRIFT_PSI = Gauge("finml_drift_max_psi", "Largest PSI on the last scored batch")
MODEL_READY = Gauge("finml_model_ready", "1 when a champion model is loaded")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load configuration and the champion model at startup.

    Args:
        _app: The application; unused.

    Yields:
        Control for the lifetime of the application.
    """
    configure_logging()
    state.load()
    MODEL_READY.set(1 if state.is_ready else 0)
    _log.info(
        "api.startup",
        ready=state.is_ready,
        model_id=state.pipeline.bundle.model_id if state.is_ready else "",
        startup_error=state.startup_error,
    )
    yield
    _log.info("api.shutdown")


def create_app() -> FastAPI:
    """Construct the FastAPI application.

    Returns:
        The configured application.
    """
    application = FastAPI(
        title="FinML Credit Risk API",
        version="0.1.0",
        description=(
            "Probability-of-default scoring with explanations, drift monitoring "
            "and macro stress testing."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    _register_error_handlers(application)
    _register_routes(application)
    return application


def _register_error_handlers(application: FastAPI) -> None:
    """Attach structured error handlers.

    Args:
        application: The application to configure.
    """

    @application.exception_handler(DriftDetectedError)
    async def _drift_handler(_request: Request, exc: DriftDetectedError) -> JSONResponse:
        """Return 409 when a drifted batch is refused.

        Args:
            _request: The request; unused.
            exc: The raised error.

        Returns:
            A conflict response carrying the drift context.
        """
        _log.warning("api.drift_blocked", **exc.to_dict())
        REQUESTS.labels(endpoint="predict", status="drift_blocked").inc()
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=exc.to_dict())

    @application.exception_handler(DomainError)
    async def _domain_handler(_request: Request, exc: DomainError) -> JSONResponse:
        """Return 422 for a business-rule violation.

        Args:
            _request: The request; unused.
            exc: The raised error.

        Returns:
            An unprocessable-entity response.
        """
        _log.warning("api.domain_error", **exc.to_dict())
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=exc.to_dict())

    @application.exception_handler(InfrastructureError)
    async def _infra_handler(_request: Request, exc: InfrastructureError) -> JSONResponse:
        """Return 503 for an external dependency failure.

        Args:
            _request: The request; unused.
            exc: The raised error.

        Returns:
            A service-unavailable response.
        """
        _log.error("api.infrastructure_error", **exc.to_dict())
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=exc.to_dict())


def _entities_to_frame(entities: list[Any]) -> pd.DataFrame:
    """Convert request models into a raw feature frame.

    Args:
        entities: Validated request entities.

    Returns:
        One row per obligor, with unset optional fields left as nulls for the
        contract-default imputation to fill.
    """
    records = [e.model_dump(exclude_none=True) for e in entities]
    frame = pd.DataFrame(records)
    if "entity_id" not in frame.columns:
        frame["entity_id"] = [f"req_{i}" for i in range(len(frame))]
    frame["entity_id"] = frame["entity_id"].fillna(
        pd.Series([f"req_{i}" for i in range(len(frame))], index=frame.index)
    )
    return frame


def _timed(endpoint: str) -> Callable[[], float]:
    """Start a latency timer for an endpoint.

    Args:
        endpoint: Endpoint label.

    Returns:
        A callable returning elapsed milliseconds and recording the observation.
    """
    started = time.perf_counter()

    def stop() -> float:
        """Record and return the elapsed time.

        Returns:
            Elapsed milliseconds.
        """
        elapsed = time.perf_counter() - started
        LATENCY.labels(endpoint=endpoint).observe(elapsed)
        return elapsed * 1000.0

    return stop


def _register_routes(application: FastAPI) -> None:
    """Attach the API routes.

    Args:
        application: The application to configure.
    """

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(current: ServiceState = Depends(get_state)) -> HealthResponse:
        """Report model readiness and drift configuration.

        Args:
            current: The service state.

        Returns:
            The readiness report.
        """
        if not current.is_ready:
            MODEL_READY.set(0)
            return HealthResponse(
                status="degraded",
                model_loaded=False,
                version="0.1.0",
                checks=current.checks,
            )

        bundle = current.pipeline.bundle
        MODEL_READY.set(1)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_id=bundle.model_id,
            model_trained_at=bundle.trained_at.isoformat(),
            is_calibrated=bundle.is_calibrated,
            n_features=len(bundle.feature_names),
            drift_monitoring=current.pipeline.drift_detector.is_fitted,
            drift_policy=current.config.drift.on_severe if current.config else "",
            version="0.1.0",
            checks=current.checks,
        )

    @application.get("/metrics", tags=["ops"])
    async def metrics() -> Response:
        """Expose Prometheus metrics.

        Returns:
            The exposition payload.
        """
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.post(
        "/predict",
        response_model=PredictResponse,
        responses={409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["scoring"],
    )
    async def predict(
        request: BatchPredictRequest,
        pipeline: Any = Depends(get_pipeline),
        config: RootConfig = Depends(get_config),
    ) -> PredictResponse:
        """Score one obligor or a batch.

        Args:
            request: The scoring request.
            pipeline: The inference pipeline.
            config: The composed configuration.

        Returns:
            One prediction per obligor, with drift status attached.

        Raises:
            HTTPException: If the batch exceeds the configured maximum.
        """
        stop = _timed("predict")
        if len(request.entities) > config.api.max_batch_size:
            REQUESTS.labels(endpoint="predict", status="too_large").inc()
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error_type": "ValidationError",
                    "message": "Batch exceeds the configured maximum",
                    "context": {
                        "n_entities": len(request.entities),
                        "max_batch_size": config.api.max_batch_size,
                    },
                },
            )

        frame = _entities_to_frame(request.entities)
        bind_run_context(endpoint="predict", n_entities=len(frame))

        # DriftDetectedError propagates to the 409 handler by design.
        result = pipeline.score(
            frame,
            check_drift=request.check_drift,
            overlay_log_odds=request.overlay_log_odds,
        )

        explanations: dict[str, Any] = {}
        if request.explain:
            explanations = _explain_batch(pipeline, frame)

        items = [
            PredictionItem(
                entity_id=str(score.entity_id),
                probability_of_default=score.pd_value,
                risk_band=score.band,
                is_flagged=score.is_flagged,
                threshold=score.threshold,
                credit_score=round(score.probability_of_default.credit_score(), 2),
                top_drivers=explanations.get(str(score.entity_id), []),
            )
            for score in result.scores
        ]

        drift_summary = None
        if result.drift is not None:
            drift_summary = DriftSummary(
                severity=result.drift.severity.name,
                max_psi=round(result.drift.max_psi, 5),
                n_drifted=len(result.drift.drifted_features),
                drifted_features=[f.feature for f in result.drift.drifted_features[:10]],
                policy=config.drift.on_severe,
            )
            DRIFT_PSI.set(result.drift.max_psi)

        PREDICTIONS.inc(len(items))
        FLAGGED.inc(result.flagged_count)
        REQUESTS.labels(endpoint="predict", status="ok").inc()

        return PredictResponse(
            predictions=items,
            model_id=result.model_id,
            model_version=pipeline.bundle.trained_at.isoformat(),
            scored_at=result.scored_at,
            n_scored=len(items),
            n_flagged=result.flagged_count,
            drift=drift_summary,
            latency_ms=round(stop(), 3),
        )

    @application.post(
        "/explain",
        response_model=ExplainResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["explainability"],
    )
    async def explain(
        request: ExplainRequest, pipeline: Any = Depends(get_pipeline)
    ) -> ExplainResponse:
        """Explain a single obligor's score.

        Args:
            request: The explanation request.
            pipeline: The inference pipeline.

        Returns:
            The explanation, or recourse options for the ``dice`` method.
        """
        stop = _timed("explain")
        frame = _entities_to_frame([request.entity])
        entity_id = str(frame["entity_id"].iloc[0])
        bind_run_context(endpoint="explain", entity_id=entity_id, method=request.method)

        matrix = pipeline.build_matrix(frame)
        probability = float(pipeline.bundle.predict_pd(matrix)[0])

        if request.method == "dice":
            examples = pipeline.counterfactuals(
                frame, entity_id=entity_id, target_pd=request.target_pd
            )
            REQUESTS.labels(endpoint="explain", status="ok").inc()
            return ExplainResponse(
                entity_id=entity_id,
                method="dice",
                model_id=pipeline.bundle.model_id,
                probability_of_default=probability,
                counterfactuals=_to_counterfactual_models(examples),
                latency_ms=round(stop(), 3),
            )

        explanation = pipeline.explain(frame, entity_id=entity_id, method=request.method)
        REQUESTS.labels(endpoint="explain", status="ok").inc()
        return ExplainResponse(
            entity_id=entity_id,
            method=explanation.method,
            model_id=pipeline.bundle.model_id,
            probability_of_default=probability,
            base_value=explanation.base_value,
            contributions=[
                FeatureContributionModel(
                    feature=c.feature,
                    value=c.value,
                    contribution=c.contribution,
                    direction=c.direction,
                )
                for c in explanation.top_drivers(request.top_k)
            ],
            latency_ms=round(stop(), 3),
        )

    @application.post(
        "/scenario",
        response_model=ScenarioResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["stress-testing"],
    )
    async def scenario(
        request: ScenarioRequest,
        pipeline: Any = Depends(get_pipeline),
        simulator: Any = Depends(get_simulator),
    ) -> ScenarioResponse:
        """Run macro stress scenarios over a portfolio.

        Args:
            request: The scenario request.
            pipeline: The inference pipeline.
            simulator: The scenario simulator.

        Returns:
            One outcome per scenario, least severe first.

        Raises:
            HTTPException: If no obligors were supplied.
        """
        stop = _timed("scenario")
        if not request.entities:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error_type": "ValidationError",
                    "message": "Scenario simulation requires at least one entity",
                    "context": {},
                },
            )

        frame = _entities_to_frame(request.entities)
        bind_run_context(endpoint="scenario", n_entities=len(frame))

        result = pipeline.score(frame, check_drift=False)
        pds = pd.Series([s.pd_value for s in result.scores])
        sectors = (
            pd.Series([s.value for s in request.sectors])
            if request.sectors
            else frame.get("sector", pd.Series(["technology"] * len(frame)))
        )

        names = [request.scenario] if request.scenario else list(simulator.scenarios)
        outcomes: list[ScenarioOutcome] = []
        for name in names:
            outcome = simulator.run_scenario(name, pds, sectors.reset_index(drop=True))
            definition = simulator.get_scenario(name)
            outcomes.append(
                ScenarioOutcome(
                    scenario=name,
                    severity=definition.severity.value,
                    description=definition.description,
                    baseline_pd=outcome.baseline_pd,
                    stressed_pd=outcome.stressed_pd,
                    pd_uplift=outcome.pd_uplift,
                    relative_uplift=outcome.relative_uplift,
                    n_entities=outcome.n_entities,
                    by_sector=[
                        ScenarioSectorImpact(
                            sector=sector_name,
                            odds_multiplier=round(
                                simulator.transmission.odds_multiplier(
                                    definition, Sector.parse(sector_name)
                                ),
                                4,
                            ),
                            stressed_pd=value,
                        )
                        for sector_name, value in outcome.worst_sectors(10)
                    ],
                )
            )

        REQUESTS.labels(endpoint="scenario", status="ok").inc()
        return ScenarioResponse(
            outcomes=sorted(outcomes, key=lambda o: o.stressed_pd),
            model_id=pipeline.bundle.model_id,
            latency_ms=round(stop(), 3),
        )

    @application.get("/scenarios", tags=["stress-testing"])
    async def list_scenarios(simulator: Any = Depends(get_simulator)) -> dict[str, Any]:
        """List the configured scenarios and the sector impact matrix.

        Args:
            simulator: The scenario simulator.

        Returns:
            Scenario definitions and their per-sector odds multipliers.
        """
        return {
            "scenarios": [
                {
                    "name": name,
                    "severity": scenario.severity.value,
                    "description": scenario.description,
                    "horizon_quarters": scenario.horizon_quarters,
                    "probability_weight": scenario.probability_weight,
                    "shocks": dict(scenario.shocks),
                }
                for name, scenario in simulator.scenarios.items()
            ],
            "sector_impact": simulator.sector_impact_matrix().round(4).to_dict(),
        }


def _explain_batch(pipeline: Any, frame: pd.DataFrame) -> dict[str, list[Any]]:
    """Produce SHAP drivers for each row of a batch.

    Args:
        pipeline: The inference pipeline.
        frame: The raw request frame.

    Returns:
        Entity id mapped to its top contributions; empty on failure.
    """
    out: dict[str, list[Any]] = {}
    for position in range(len(frame)):
        row = frame.iloc[[position]]
        entity_id = str(row["entity_id"].iloc[0])
        try:
            explanation = pipeline.explain(row, entity_id=entity_id, method="shap")
        except FinMLError as exc:
            # An explanation failure must not cost the caller their scores.
            _log.warning("api.explanation_failed", entity_id=entity_id, reason=exc.message)
            continue
        out[entity_id] = [
            FeatureContributionModel(
                feature=c.feature,
                value=c.value,
                contribution=c.contribution,
                direction=c.direction,
            )
            for c in explanation.top_drivers(5)
        ]
    return out


def _to_counterfactual_models(examples: list[Any]) -> list[CounterfactualResponse]:
    """Convert domain counterfactuals into response models.

    Args:
        examples: Domain counterfactual examples.

    Returns:
        The response models.
    """
    from domain.entities import FEATURE_CONTRACT

    out: list[CounterfactualResponse] = []
    for option, example in enumerate(examples, start=1):
        changes: list[CounterfactualChange] = []
        for feature, (before, after) in example.changes.items():
            try:
                description = FEATURE_CONTRACT.get(feature).description
            except FinMLError:
                description = ""
            changes.append(
                CounterfactualChange(
                    feature=feature,
                    description=description,
                    current=round(before, 5),
                    required=round(after, 5),
                    change=round(after - before, 5),
                )
            )
        out.append(
            CounterfactualResponse(
                option=option,
                original_pd=round(example.original_pd, 5),
                counterfactual_pd=round(example.counterfactual_pd, 5),
                pd_reduction=round(example.pd_reduction, 5),
                n_changes=example.n_changes,
                changes=changes,
            )
        )
    return out


#: The ASGI application.
app = create_app()
