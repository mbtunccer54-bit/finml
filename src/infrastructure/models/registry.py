"""Experiment tracking and the model registry, backed by MLflow.

Under SR 11-7 a production score must be reconstructible: which model version,
fitted on which data, with which hyperparameters, scoring how well. That is a
lineage requirement, not a convenience, so tracking failures are logged loudly
but never abort a training run — losing the run is strictly worse than losing
its metadata.

The registry also always writes a local artifact alongside the MLflow record.
The API can then cold-start with no tracking server reachable, which is what
makes the container come up in an air-gapped environment.
"""

from __future__ import annotations

import json
import pickle  # nosec B403 - reads only artifacts this platform wrote itself
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from domain.entities import ModelPerformance
from domain.exceptions import ModelNotFoundError, RegistryError
from infrastructure.config.schemas import TrackingConfig
from infrastructure.logging import get_logger
from infrastructure.models.base import ModelBundle

__all__ = ["ExperimentTracker", "LocalModelStore"]

_log = get_logger(__name__)

#: File name of the promoted champion inside the artifact directory.
_CHAMPION_FILE = "champion.pkl"
_CHAMPION_META = "champion.json"


class LocalModelStore:
    """Filesystem persistence for model bundles.

    Attributes:
        base_dir: Directory holding the pickled bundles and champion pointer.
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Initialise the store.

        Args:
            base_dir: Directory for the artifacts.

        Raises:
            RegistryError: If the directory cannot be created.
        """
        self.base_dir = Path(base_dir)
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RegistryError(
                "Could not create the model artifact directory",
                base_dir=str(self.base_dir),
                reason=str(exc),
            ) from exc

    def _path_for(self, model_id: str) -> Path:
        """Build the artifact path for a model.

        Args:
            model_id: Model identifier.

        Returns:
            The file path.
        """
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id)
        return self.base_dir / f"{safe}.pkl"

    def save(self, bundle: ModelBundle, *, promote: bool = False) -> Path:
        """Persist a model bundle.

        Args:
            bundle: The bundle to save.
            promote: Also write it as the champion the API loads at startup.

        Returns:
            The path written.

        Raises:
            RegistryError: If the artifact cannot be written.
        """
        path = self._path_for(bundle.model_id)
        try:
            with path.open("wb") as handle:
                pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if promote:
                with (self.base_dir / _CHAMPION_FILE).open("wb") as handle:
                    pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
                (self.base_dir / _CHAMPION_META).write_text(
                    json.dumps(
                        {
                            "model_id": bundle.model_id,
                            "trained_at": bundle.trained_at.isoformat(),
                            "is_calibrated": bundle.is_calibrated,
                            "calibration_method": bundle.calibration_method,
                            "dataset_version": bundle.dataset_version,
                            "n_features": len(bundle.feature_names),
                            "metrics": bundle.metrics,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        except (OSError, pickle.PicklingError) as exc:
            raise RegistryError(
                "Could not write the model artifact",
                model_id=bundle.model_id,
                path=str(path),
                reason=str(exc),
            ) from exc

        _log.info(
            "registry.saved_local",
            model_id=bundle.model_id,
            path=str(path),
            promoted=promote,
        )
        return path

    def load(self, model_id: str) -> ModelBundle:
        """Load a model bundle by identifier.

        Args:
            model_id: Model identifier.

        Returns:
            The bundle.

        Raises:
            ModelNotFoundError: If the artifact is absent or unreadable.
        """
        return self._load_path(self._path_for(model_id), model_id=model_id)

    def load_champion(self) -> ModelBundle:
        """Load the promoted champion.

        Returns:
            The champion bundle.

        Raises:
            ModelNotFoundError: If no champion has been promoted.
        """
        return self._load_path(self.base_dir / _CHAMPION_FILE, model_id="<champion>")

    @staticmethod
    def _load_path(path: Path, *, model_id: str) -> ModelBundle:
        """Unpickle a bundle from a path.

        Args:
            path: Artifact path.
            model_id: Identifier used in error context.

        Returns:
            The bundle.

        Raises:
            ModelNotFoundError: If the artifact is absent or cannot be read.
        """
        if not path.is_file():
            raise ModelNotFoundError(
                "Model artifact not found",
                model_id=model_id,
                path=str(path),
                hint="Run the training pipeline first",
            )
        try:
            with path.open("rb") as handle:
                # nosec B301 - the artifact directory is written exclusively by
                # this platform's training pipeline and is never user-supplied.
                # Treat it as trusted only because nothing else can write there;
                # if that ever changes, move to a signed or non-pickle format.
                bundle = pickle.load(handle)  # nosec B301
        except (OSError, pickle.UnpicklingError, AttributeError, EOFError) as exc:
            raise ModelNotFoundError(
                "Model artifact could not be deserialised",
                model_id=model_id,
                path=str(path),
                reason=str(exc),
            ) from exc
        if not isinstance(bundle, ModelBundle):
            raise ModelNotFoundError(
                "Artifact is not a ModelBundle", model_id=model_id, got=type(bundle).__name__
            )
        return bundle

    def has_champion(self) -> bool:
        """Check whether a champion has been promoted.

        Returns:
            ``True`` when a champion artifact exists.
        """
        return (self.base_dir / _CHAMPION_FILE).is_file()

    def list_models(self) -> list[str]:
        """List every stored model identifier.

        Returns:
            The stored identifiers, excluding the champion pointer.
        """
        return sorted(p.stem for p in self.base_dir.glob("*.pkl") if p.name != _CHAMPION_FILE)


class ExperimentTracker:
    """MLflow wrapper that degrades to local-only operation.

    Attributes:
        config: Tracking configuration.
        store: Local artifact store, always written regardless of MLflow state.
    """

    def __init__(self, config: TrackingConfig, store: LocalModelStore) -> None:
        """Initialise the tracker.

        Args:
            config: Tracking configuration.
            store: Local artifact store.
        """
        self.config = config
        self.store = store
        self._active = False
        self._setup()

    def _setup(self) -> None:
        """Point MLflow at the configured tracking URI and experiment."""
        if not self.config.enabled:
            _log.info("tracking.disabled")
            return
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            self._active = True
            _log.info(
                "tracking.configured",
                tracking_uri=self.config.tracking_uri,
                experiment=self.config.experiment_name,
            )
        except Exception as exc:
            # A missing tracking server must not stop a model being trained.
            _log.warning(
                "tracking.setup_failed",
                tracking_uri=self.config.tracking_uri,
                error_type=type(exc).__name__,
                reason=str(exc)[:300],
                detail="continuing with local artifacts only",
            )
            self._active = False

    @property
    def is_active(self) -> bool:
        """Whether MLflow tracking is available.

        Returns:
            ``True`` when runs will be recorded to MLflow.
        """
        return self._active

    @contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[str | None]:
        """Open an MLflow run.

        Args:
            run_name: Name for the run.
            tags: Tags to set on the run.

        Yields:
            The MLflow run id, or ``None`` when tracking is unavailable.
        """
        if not self._active:
            yield None
            return
        try:
            with mlflow.start_run(run_name=run_name) as active:
                if tags:
                    mlflow.set_tags(tags)
                _log.info("tracking.run_started", run_id=active.info.run_id, run_name=run_name)
                yield active.info.run_id
        except Exception as exc:
            _log.warning(
                "tracking.run_failed",
                run_name=run_name,
                error_type=type(exc).__name__,
                reason=str(exc)[:300],
            )
            self._active = False
            yield None

    def log_params(self, params: dict[str, Any]) -> None:
        """Record hyperparameters on the active run.

        Args:
            params: Parameters to record.
        """
        if not self._active:
            return
        try:
            # MLflow rejects oversized values; stringify and truncate defensively.
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        except Exception as exc:
            _log.warning("tracking.log_params_failed", reason=str(exc)[:200])

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Record metrics on the active run.

        Args:
            metrics: Metrics to record.
            step: Optional step index.
        """
        if not self._active:
            return
        try:
            clean = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and pd.notna(v)
            }
            mlflow.log_metrics(clean, step=step)
        except Exception as exc:
            _log.warning("tracking.log_metrics_failed", reason=str(exc)[:200])

    def log_performance(self, performance: ModelPerformance) -> None:
        """Record one evaluation record as prefixed metrics.

        Args:
            performance: The record to log.
        """
        prefix = performance.split
        self.log_metrics(
            {
                f"{prefix}.roc_auc": performance.roc_auc,
                f"{prefix}.pr_auc": performance.pr_auc,
                f"{prefix}.f1": performance.f1,
                f"{prefix}.precision": performance.precision,
                f"{prefix}.recall": performance.recall,
                f"{prefix}.brier_score": performance.brier_score,
                f"{prefix}.ece": performance.expected_calibration_error,
                f"{prefix}.business_cost": performance.business_cost,
                f"{prefix}.latency_ms": performance.inference_latency_ms,
                f"{prefix}.threshold": performance.threshold,
            }
        )

    def log_artifact_file(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        """Upload a file to the active run.

        Args:
            path: Local file to upload.
            artifact_path: Destination directory within the run.
        """
        if not self._active or not self.config.log_artifacts:
            return
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as exc:
            _log.warning("tracking.log_artifact_failed", path=str(path), reason=str(exc)[:200])

    def log_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Upload a dataframe as a CSV artifact.

        Args:
            df: The frame to upload.
            name: Artifact file name, with or without the ``.csv`` suffix.
        """
        if not self._active or not self.config.log_artifacts or df.empty:
            return
        import tempfile

        filename = name if name.endswith(".csv") else f"{name}.csv"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / filename
                df.to_csv(path, index=False)
                mlflow.log_artifact(str(path))
        except (OSError, ValueError) as exc:
            _log.warning("tracking.log_dataframe_failed", name=name, reason=str(exc)[:200])

    def register_model(self, bundle: ModelBundle, *, promote: bool = True) -> str | None:
        """Persist a bundle locally and register it with MLflow.

        The local write happens first and unconditionally, so a registry outage
        cannot leave the pipeline with no artifact.

        Args:
            bundle: The bundle to register.
            promote: Mark the bundle as the champion locally.

        Returns:
            The MLflow model URI, or ``None`` when only the local copy was written.
        """
        self.store.save(bundle, promote=promote)

        if not self._active or not self.config.log_artifacts:
            return None
        try:
            # MLflow 3 defaults this flavour to skops, which refuses to round-trip
            # anything outside its allow-list -- and every bundle here carries
            # project-defined types (the model adapters, and the value objects on
            # their attributes). skops_trusted_types would mean enumerating those
            # by hand and re-editing the list on every field added, so the failure
            # would reappear silently at the next change. cloudpickle carries the
            # same trust assumption the local store already documents, and unlike
            # plain pickle it handles classes defined in this package.
            info = mlflow.sklearn.log_model(
                sk_model=bundle.model,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                registered_model_name=self.config.registered_model_name if promote else None,
            )
            _log.info(
                "registry.registered",
                model_id=bundle.model_id,
                model_uri=info.model_uri,
                registered_name=self.config.registered_model_name if promote else None,
            )
            return str(info.model_uri)
        except Exception as exc:
            _log.warning(
                "registry.mlflow_registration_failed",
                model_id=bundle.model_id,
                error_type=type(exc).__name__,
                reason=str(exc)[:300],
                detail="local artifact was written successfully",
            )
            return None

    def load_champion(self) -> ModelBundle:
        """Load the promoted champion from the local store.

        Returns:
            The champion bundle.

        Raises:
            ModelNotFoundError: If no champion has been promoted.
        """
        return self.store.load_champion()
