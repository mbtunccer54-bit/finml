"""Dependency wiring for the API.

The model is loaded once at startup and shared across requests. Loading it per
request would add hundreds of milliseconds and, worse, allow different requests
in the same deployment to be answered by different model versions mid-rollout.

Startup is deliberately tolerant: a missing champion leaves the service running
and reporting ``degraded`` on ``/health`` rather than crash-looping the
container. An orchestrator can then hold traffic off while an operator sees
exactly what is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException, status

from domain.exceptions import FinMLError, ModelNotFoundError
from infrastructure.config.loader import cached_config, to_object
from infrastructure.config.schemas import RootConfig
from infrastructure.logging import get_logger

__all__ = ["ServiceState", "get_config", "get_pipeline", "get_simulator", "get_state", "state"]

_log = get_logger(__name__)


@dataclass(slots=True)
class ServiceState:
    """Process-wide service state.

    Attributes:
        config: The composed configuration.
        pipeline: The loaded inference pipeline, when a champion is available.
        simulator: The scenario simulator, when a champion is available.
        startup_error: Why loading failed, if it did.
    """

    config: RootConfig | None = None
    pipeline: Any = None
    simulator: Any = None
    startup_error: str = ""
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Whether the service can score requests.

        Returns:
            ``True`` when a model is loaded.
        """
        return self.pipeline is not None

    def load(self, overrides: list[str] | None = None) -> ServiceState:
        """Compose configuration and load the champion model.

        Args:
            overrides: Hydra override strings.

        Returns:
            This state, populated as far as it could be.
        """
        from application.inference_pipeline import InferencePipeline
        from application.scenario_simulator import ScenarioSimulator

        self.checks = {"config_loaded": False, "model_loaded": False, "scenarios_loaded": False}

        try:
            self.config = to_object(cached_config(overrides))
            self.checks["config_loaded"] = True
        except FinMLError as exc:
            self.startup_error = f"configuration: {exc.message}"
            _log.error("api.config_load_failed", **exc.to_dict())
            return self

        try:
            self.pipeline = InferencePipeline.from_champion(self.config)
            self.checks["model_loaded"] = True
            _log.info(
                "api.model_loaded",
                model_id=self.pipeline.bundle.model_id,
                is_calibrated=self.pipeline.bundle.is_calibrated,
                n_features=len(self.pipeline.bundle.feature_names),
                threshold=round(self.pipeline.threshold, 4),
            )
        except ModelNotFoundError as exc:
            # Serve /health as degraded rather than crash-looping the container.
            self.startup_error = exc.message
            _log.error(
                "api.model_load_failed",
                **exc.to_dict(),
                detail="service will report degraded until a model is promoted",
            )
            return self

        try:
            self.simulator = ScenarioSimulator(
                self.config,
                scoring_fn=lambda x: self.pipeline.bundle.model.predict_proba(x)[:, 1],
            )
            self.checks["scenarios_loaded"] = True
            _log.info("api.scenarios_loaded", n_scenarios=len(self.simulator.scenarios))
        except FinMLError as exc:
            # Scenario simulation is an optional capability; scoring still works.
            _log.warning("api.scenario_load_failed", **exc.to_dict())

        return self


#: Process-wide state, populated by the application lifespan handler.
state = ServiceState()


def get_state() -> ServiceState:
    """Return the process-wide service state.

    Returns:
        The service state.
    """
    return state


def get_config(current: ServiceState = Depends(get_state)) -> RootConfig:
    """Return the composed configuration.

    Args:
        current: The service state.

    Returns:
        The configuration.

    Raises:
        HTTPException: If configuration was never loaded.
    """
    if current.config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": "ConfigurationError",
                "message": "Service configuration is not loaded",
                "context": {"startup_error": current.startup_error},
            },
        )
    return current.config


def get_pipeline(current: ServiceState = Depends(get_state)) -> Any:
    """Return the loaded inference pipeline.

    Args:
        current: The service state.

    Returns:
        The inference pipeline.

    Raises:
        HTTPException: If no model is loaded.
    """
    if current.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": "ModelNotFoundError",
                "message": "No model is loaded",
                "context": {
                    "startup_error": current.startup_error,
                    "hint": "Run: python -m application.train_pipeline",
                },
            },
        )
    return current.pipeline


def get_simulator(current: ServiceState = Depends(get_state)) -> Any:
    """Return the scenario simulator.

    Args:
        current: The service state.

    Returns:
        The scenario simulator.

    Raises:
        HTTPException: If scenario simulation is unavailable.
    """
    if current.simulator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": "ScenarioError",
                "message": "Scenario simulation is unavailable",
                "context": {"startup_error": current.startup_error},
            },
        )
    return current.simulator
