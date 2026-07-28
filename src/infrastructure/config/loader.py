"""Configuration loading outside a Hydra entrypoint.

``@hydra.main`` owns the process: it changes the working directory and installs
its own argument parsing. That is right for the training CLI but wrong for the
FastAPI worker and the Streamlit script, which are started by someone else.

:func:`load_config` composes the same configuration tree through the Compose
API, so every entrypoint reads identical settings regardless of how it was
launched.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from domain.exceptions import ConfigurationError
from infrastructure.config.schemas import RootConfig, register_configs

__all__ = ["config_dir", "load_config", "to_object"]

#: Environment variable that overrides the discovered configuration directory.
_CONFIG_DIR_ENV = "FINML_CONFIG_DIR"


def config_dir() -> Path:
    """Locate the ``configs/`` directory.

    Resolution order is the ``FINML_CONFIG_DIR`` environment variable, then a
    ``configs/`` directory beside the installed package, then one relative to
    the current working directory. The last case covers running from a source
    checkout without installing.

    Returns:
        Absolute path to the configuration directory.

    Raises:
        ConfigurationError: If no candidate directory exists.
    """
    override = os.environ.get(_CONFIG_DIR_ENV)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    # src/infrastructure/config/loader.py -> project root is three levels up.
    candidates.append(Path(__file__).resolve().parents[3] / "configs")
    candidates.append(Path.cwd() / "configs")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise ConfigurationError(
        "Could not locate the configs directory",
        searched=[str(c) for c in candidates],
        hint=f"Set {_CONFIG_DIR_ENV} to an absolute path",
    )


def load_config(
    overrides: list[str] | None = None,
    *,
    config_name: str = "config",
) -> DictConfig:
    """Compose the configuration tree.

    Args:
        overrides: Hydra override strings, for example ``["data.n_entities=100"]``.
        config_name: Root configuration file name, without the extension.

    Returns:
        The composed configuration.

    Raises:
        ConfigurationError: If composition fails, typically because an override
            names a key that does not exist.
    """
    register_configs()
    directory = config_dir()

    # An outer @hydra.main may already own the global state; clear it so this
    # call composes cleanly, and leave it clear afterwards for the same reason.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    try:
        with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
            return compose(config_name=config_name, overrides=overrides or [])
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            "Failed to compose configuration",
            config_dir=str(directory),
            config_name=config_name,
            overrides=overrides or [],
            reason=str(exc),
        ) from exc
    finally:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()


@lru_cache(maxsize=8)
def _cached_config(override_key: str) -> DictConfig:
    """Compose and memoise a configuration for a given override string.

    Args:
        override_key: Newline-joined override strings, used as the cache key.

    Returns:
        The composed configuration.
    """
    overrides = [line for line in override_key.split("\n") if line]
    return load_config(overrides)


def cached_config(overrides: list[str] | None = None) -> DictConfig:
    """Return a memoised configuration.

    Long-lived processes such as the API worker compose once and reuse the
    result; composition is not free and the settings do not change mid-process.

    Args:
        overrides: Hydra override strings.

    Returns:
        The composed configuration.
    """
    return _cached_config("\n".join(overrides or []))


def to_object(cfg: DictConfig) -> RootConfig:
    """Convert a composed configuration into the typed dataclass tree.

    Args:
        cfg: Composed configuration.

    Returns:
        The equivalent :class:`RootConfig`.

    Raises:
        ConfigurationError: If the configuration does not satisfy the schema.
    """
    try:
        resolved = OmegaConf.to_object(cfg)
    except Exception as exc:
        raise ConfigurationError(
            "Configuration does not satisfy the declared schema", reason=str(exc)
        ) from exc

    # When the root node carries the RootConfig schema, `to_object` already
    # instantiates it; a plain mapping only comes back if the schema was not
    # applied, which is worth surfacing rather than silently coercing.
    if isinstance(resolved, RootConfig):
        return resolved
    if isinstance(resolved, dict):
        return RootConfig(**resolved)  # type: ignore[arg-type]
    raise ConfigurationError(
        "Composed configuration is neither a RootConfig nor a mapping",
        got=type(resolved).__name__,
    )
