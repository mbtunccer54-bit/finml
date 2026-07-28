"""Macro stress testing and what-if simulation.

Two directions of travel, and they are genuinely different questions:

* **Top-down** — a macro scenario (CCAR/EBA style) is projected forward by a
  satellite model, then transmitted into obligor PDs through the two-stage
  bridge in :class:`~domain.services.MacroTransmissionService`.
* **Bottom-up** — an analyst overrides individual ratios on one obligor and sees
  the PD and its explanation move.

The transmission arithmetic deliberately lives in the domain, not here. It is a
business rule that a model-risk reviewer must be able to read without following
a pandas call chain, and it is the part a regulator will ask to see derived.

The satellite model is a VAR over the macro block, falling back to independent
AR(1) fits when the history is too short for a VAR — which it frequently is,
since macro series are quarterly and a decade is only forty observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from domain.entities import (
    MACRO_VARIABLES,
    MacroScenario,
    ScenarioResult,
)
from domain.exceptions import ScenarioError
from domain.services import (
    DEFAULT_SECTOR_SENSITIVITIES,
    MacroTransmissionService,
    SectorSensitivity,
)
from domain.value_objects import ProbabilityOfDefault, ScenarioSeverity, Sector
from infrastructure.config.schemas import RootConfig, ScenarioConfig
from infrastructure.logging import get_logger, log_duration

__all__ = ["MacroForecast", "SatelliteModel", "ScenarioSimulator", "WhatIfResult"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MacroForecast:
    """Projected macro path.

    Attributes:
        path: One row per projected quarter, one column per macro variable.
        model: Satellite model actually used.
        lag_order: Fitted lag order, where applicable.
    """

    path: pd.DataFrame
    model: str
    lag_order: int = 0

    @property
    def terminal(self) -> dict[str, float]:
        """Macro state at the end of the horizon.

        Returns:
            Macro variable mapped to its terminal value.
        """
        if self.path.empty:
            return {}
        return {c: float(self.path[c].iloc[-1]) for c in self.path.columns}


@dataclass(frozen=True, slots=True)
class WhatIfResult:
    """Outcome of a single-obligor what-if.

    Attributes:
        entity_id: Obligor examined.
        baseline_pd: PD before any change.
        adjusted_pd: PD after the overrides and scenario.
        scenario_name: Scenario applied, if any.
        overrides: Feature overrides applied.
        odds_multiplier: Macro odds multiplier applied.
        sector: Sector used for transmission.
    """

    entity_id: str
    baseline_pd: float
    adjusted_pd: float
    scenario_name: str = ""
    overrides: dict[str, float] = field(default_factory=dict)
    odds_multiplier: float = 1.0
    sector: str = ""

    @property
    def pd_change(self) -> float:
        """Absolute PD movement.

        Returns:
            ``adjusted_pd - baseline_pd``.
        """
        return self.adjusted_pd - self.baseline_pd


class SatelliteModel:
    """Projects the macro block forward.

    Attributes:
        config: Scenario configuration.
    """

    def __init__(self, config: ScenarioConfig) -> None:
        """Initialise the satellite model.

        Args:
            config: Scenario configuration.
        """
        self.config = config
        self._fitted: Any = None
        self._columns: list[str] = []
        self._history: pd.DataFrame = pd.DataFrame()
        self._model_used = ""

    def fit(self, macro: pd.DataFrame) -> SatelliteModel:
        """Fit the satellite model to macro history.

        Args:
            macro: Macro history including the date column.

        Returns:
            This model, fitted.

        Raises:
            ScenarioError: If the history lacks the macro columns or is too short.
        """
        columns = [c for c in MACRO_VARIABLES if c in macro.columns]
        if not columns:
            raise ScenarioError(
                "Macro history contains none of the expected variables",
                expected=list(MACRO_VARIABLES),
                got=list(macro.columns),
            )
        history = macro[columns].dropna().reset_index(drop=True)
        if len(history) < 4:
            raise ScenarioError(
                "Macro history is too short to fit a satellite model",
                n_observations=len(history),
                minimum=4,
            )

        self._columns = columns
        self._history = history

        # A VAR needs roughly (n_vars * lags) observations per equation; on a
        # short quarterly sample that is unattainable, so fall back rather than
        # fit something singular.
        required = len(columns) * self.config.var_max_lags + 2
        if self.config.satellite_model == "var" and len(history) >= required:
            try:
                from statsmodels.tsa.api import VAR

                max_lags = max(min(self.config.var_max_lags, len(history) // len(columns) - 1), 1)
                fitted = VAR(history.to_numpy(dtype=float)).fit(maxlags=max_lags, ic="aic")
            except Exception as exc:
                _log.warning(
                    "scenario.var_failed",
                    reason=str(exc)[:200],
                    fallback="independent AR(1)",
                )
            else:
                self._fitted = fitted
                self._model_used = "var"
                _log.info(
                    "scenario.satellite_fitted",
                    model="var",
                    lag_order=int(fitted.k_ar),
                    n_observations=len(history),
                    n_variables=len(columns),
                )
                return self

        self._fitted = self._fit_ar1(history)
        self._model_used = "ar1"
        _log.info(
            "scenario.satellite_fitted",
            model="ar1",
            n_observations=len(history),
            n_variables=len(columns),
            reason="insufficient history for a VAR"
            if self.config.satellite_model == "var"
            else "configured",
        )
        return self

    @staticmethod
    def _fit_ar1(history: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
        """Fit an independent AR(1) per macro variable.

        Args:
            history: Macro history.

        Returns:
            Variable mapped to ``(mean, persistence, residual_sd)``.
        """
        params: dict[str, tuple[float, float, float]] = {}
        for column in history.columns:
            series = history[column].to_numpy(dtype=float)
            mean = float(series.mean())
            centred = series - mean
            if len(centred) > 2 and np.var(centred[:-1]) > 1e-12:
                phi = float(
                    np.clip(
                        np.dot(centred[:-1], centred[1:]) / np.dot(centred[:-1], centred[:-1]),
                        -0.99,
                        0.99,
                    )
                )
            else:
                phi = 0.0
            residuals = centred[1:] - phi * centred[:-1]
            params[column] = (mean, phi, float(np.std(residuals)) if residuals.size else 0.0)
        return params

    def forecast(self, horizon: int | None = None) -> MacroForecast:
        """Project the macro block forward.

        Args:
            horizon: Quarters to project; the configured horizon when omitted.

        Returns:
            The projected path.

        Raises:
            ScenarioError: If the model has not been fitted.
        """
        if self._fitted is None:
            raise ScenarioError("Satellite model must be fitted before forecasting")

        steps = horizon if horizon is not None else self.config.forecast_horizon
        if steps < 1:
            raise ScenarioError("Forecast horizon must be positive", horizon=steps)

        if self._model_used == "var":
            history = self._history.to_numpy(dtype=float)
            projected = self._fitted.forecast(history[-self._fitted.k_ar :], steps=steps)
            path = pd.DataFrame(projected, columns=self._columns)
            return MacroForecast(path=path, model="var", lag_order=int(self._fitted.k_ar))

        rows: list[dict[str, float]] = []
        state = {c: float(self._history[c].iloc[-1]) for c in self._columns}
        for _ in range(steps):
            row: dict[str, float] = {}
            for column in self._columns:
                mean, phi, _sd = self._fitted[column]
                row[column] = mean + phi * (state[column] - mean)
            rows.append(row)
            state = row
        return MacroForecast(path=pd.DataFrame(rows), model="ar1")


class ScenarioSimulator:
    """Runs macro scenarios and single-obligor what-ifs.

    Attributes:
        config: The composed configuration.
        transmission: The macro-to-micro bridge.
        scenarios: Configured scenarios keyed by name.
    """

    def __init__(self, config: RootConfig, scoring_fn: Any | None = None) -> None:
        """Initialise the simulator.

        Args:
            config: The composed configuration.
            scoring_fn: Callable mapping a feature matrix to PDs. Required only
                for the what-if path, where the model is re-scored directly.
        """
        self.config = config
        self.scoring_fn = scoring_fn
        self.transmission = MacroTransmissionService(
            sensitivities=self._build_sensitivities(config.scenario),
            revenue_to_log_odds=config.scenario.revenue_to_log_odds,
            max_log_odds_shift=config.scenario.max_log_odds_shift,
        )
        self.scenarios = self._build_scenarios(config.scenario)
        self.satellite = SatelliteModel(config.scenario)

    @staticmethod
    def _build_sensitivities(config: ScenarioConfig) -> dict[Sector, SectorSensitivity]:
        """Merge configured overrides onto the shipped sensitivity table.

        Args:
            config: Scenario configuration.

        Returns:
            Sector mapped to its bridge parameters.
        """
        merged = dict(DEFAULT_SECTOR_SENSITIVITIES)
        for raw_sector, overrides in (config.sector_sensitivities or {}).items():
            try:
                sector = Sector.parse(str(raw_sector))
            except Exception:
                _log.warning("scenario.unknown_sector_override", sector=raw_sector)
                continue
            base = merged[sector]
            fields = {
                "gdp_revenue_beta": base.gdp_revenue_beta,
                "unemployment_revenue_beta": base.unemployment_revenue_beta,
                "inflation_revenue_beta": base.inflation_revenue_beta,
                "rate_log_odds_beta": base.rate_log_odds_beta,
                "spread_log_odds_beta": base.spread_log_odds_beta,
            }
            for key, value in dict(overrides).items():
                if key in fields:
                    fields[key] = float(value)
                else:
                    _log.warning(
                        "scenario.unknown_sensitivity_field",
                        sector=sector.value,
                        field=key,
                        known=sorted(fields),
                    )
            merged[sector] = SectorSensitivity(sector=sector, **fields)
        return merged

    @staticmethod
    def _build_scenarios(config: ScenarioConfig) -> dict[str, MacroScenario]:
        """Turn configured scenario specs into domain scenarios.

        Args:
            config: Scenario configuration.

        Returns:
            Scenario name mapped to the domain object.

        Raises:
            ScenarioError: If no scenario could be constructed.
        """
        out: dict[str, MacroScenario] = {}
        for spec in config.scenarios:
            try:
                out[spec.name] = MacroScenario(
                    name=spec.name,
                    severity=ScenarioSeverity(spec.severity),
                    shocks=dict(spec.shocks),
                    horizon_quarters=spec.horizon_quarters,
                    description=spec.description,
                    probability_weight=spec.probability_weight,
                )
            except (ScenarioError, ValueError) as exc:
                _log.warning("scenario.invalid_definition", name=spec.name, reason=str(exc)[:200])
        if not out:
            raise ScenarioError(
                "No valid scenarios are configured", hint="Check configs/scenario/ccar.yaml"
            )
        return out

    def get_scenario(self, name: str) -> MacroScenario:
        """Look up a configured scenario.

        Args:
            name: Scenario name.

        Returns:
            The scenario.

        Raises:
            ScenarioError: If no scenario carries that name.
        """
        if name not in self.scenarios:
            raise ScenarioError("Unknown scenario", name=name, known=sorted(self.scenarios))
        return self.scenarios[name]

    def project_macro(
        self, macro_history: pd.DataFrame, *, horizon: int | None = None
    ) -> MacroForecast:
        """Fit the satellite model and project the macro block.

        Args:
            macro_history: Macro history including the date column.
            horizon: Quarters to project.

        Returns:
            The projected path.
        """
        with log_duration("scenario.project_macro", n_history=len(macro_history)):
            return self.satellite.fit(macro_history).forecast(horizon)

    def run_scenario(
        self,
        scenario_name: str,
        pds: pd.Series,
        sectors: pd.Series,
    ) -> ScenarioResult:
        """Transmit a scenario across a scored portfolio.

        Args:
            scenario_name: Scenario to apply.
            pds: Baseline PD per obligor.
            sectors: Sector per obligor, aligned with ``pds``.

        Returns:
            Baseline and stressed PDs, overall and per sector.

        Raises:
            ScenarioError: If the inputs are misaligned or the scenario is unknown.
        """
        if len(pds) != len(sectors):
            raise ScenarioError(
                "PD and sector series differ in length", n_pds=len(pds), n_sectors=len(sectors)
            )
        scenario = self.get_scenario(scenario_name)

        grouped: dict[Sector, list[ProbabilityOfDefault]] = {}
        skipped = 0
        for probability, raw_sector in zip(pds.to_numpy(), sectors.astype(str), strict=True):
            try:
                sector = Sector.parse(raw_sector)
            except Exception:
                skipped += 1
                continue
            grouped.setdefault(sector, []).append(
                ProbabilityOfDefault(float(np.clip(probability, 0.0, 1.0)))
            )

        if skipped:
            _log.warning("scenario.unknown_sectors_skipped", n_skipped=skipped)

        result = self.transmission.transmit_portfolio(grouped, scenario)
        _log.info(
            "scenario.applied",
            scenario=scenario_name,
            severity=scenario.severity.value,
            n_entities=result.n_entities,
            baseline_pd=round(result.baseline_pd, 5),
            stressed_pd=round(result.stressed_pd, 5),
            pd_uplift=round(result.pd_uplift, 5),
            relative_uplift=round(result.relative_uplift, 4),
            worst_sectors=[(s, round(v, 5)) for s, v in result.worst_sectors(3)],
        )
        return result

    def run_all_scenarios(self, pds: pd.Series, sectors: pd.Series) -> pd.DataFrame:
        """Run every configured scenario over a portfolio.

        Args:
            pds: Baseline PD per obligor.
            sectors: Sector per obligor.

        Returns:
            One row per scenario, ordered by stressed PD.
        """
        rows: list[dict[str, Any]] = []
        for name, scenario in self.scenarios.items():
            try:
                result = self.run_scenario(name, pds, sectors)
            except ScenarioError as exc:
                _log.warning("scenario.run_failed", scenario=name, reason=exc.message)
                continue
            rows.append(
                {
                    "scenario": name,
                    "severity": scenario.severity.value,
                    "description": scenario.description,
                    "baseline_pd": result.baseline_pd,
                    "stressed_pd": result.stressed_pd,
                    "pd_uplift": result.pd_uplift,
                    "relative_uplift": result.relative_uplift,
                    "probability_weight": scenario.probability_weight,
                    "n_entities": result.n_entities,
                }
            )
        return pd.DataFrame(rows).sort_values("stressed_pd").reset_index(drop=True)

    def sector_impact_matrix(self) -> pd.DataFrame:
        """Tabulate the odds multiplier for every sector and scenario.

        This is the bridge laid bare: it shows exactly how much each scenario
        multiplies default odds in each sector, with no model in the way.

        Returns:
            Sectors as rows, scenarios as columns, odds multipliers as values.
        """
        data = {
            name: {
                sector.value: self.transmission.odds_multiplier(scenario, sector)
                for sector in Sector
            }
            for name, scenario in self.scenarios.items()
        }
        return pd.DataFrame(data)

    def what_if(
        self,
        row: pd.DataFrame,
        *,
        entity_id: str,
        sector: str,
        overrides: dict[str, float] | None = None,
        scenario_name: str | None = None,
    ) -> WhatIfResult:
        """Re-score one obligor under feature overrides and an optional scenario.

        The two channels compose: overrides are applied to the feature matrix and
        the model is genuinely re-scored, then the macro scenario multiplies the
        resulting default odds through the bridge.

        Args:
            row: A one-row model feature matrix.
            entity_id: Obligor identifier.
            sector: Obligor's sector, used for transmission.
            overrides: Feature values to replace.
            scenario_name: Scenario to apply on top.

        Returns:
            The baseline and adjusted PDs.

        Raises:
            ScenarioError: If no scoring function was supplied, or the row is not
                exactly one row.
        """
        if self.scoring_fn is None:
            raise ScenarioError(
                "what_if requires a scoring function",
                hint="Construct ScenarioSimulator(config, scoring_fn=...)",
            )
        if len(row) != 1:
            raise ScenarioError("what_if expects exactly one row", n_rows=len(row))

        baseline_pd = float(np.asarray(self.scoring_fn(row)).ravel()[0])

        adjusted = row.copy()
        applied: dict[str, float] = {}
        for name, value in (overrides or {}).items():
            if name not in adjusted.columns:
                _log.warning("scenario.unknown_override", feature=name, entity_id=entity_id)
                continue
            adjusted.at[adjusted.index[0], name] = float(value)
            applied[name] = float(value)

        micro_pd = float(np.asarray(self.scoring_fn(adjusted)).ravel()[0])

        multiplier = 1.0
        if scenario_name:
            scenario = self.get_scenario(scenario_name)
            parsed = Sector.parse(sector)
            multiplier = self.transmission.odds_multiplier(scenario, parsed)
            final_pd = (
                ProbabilityOfDefault(float(np.clip(micro_pd, 0.0, 1.0)))
                .shocked(multiplier=multiplier)
                .value
            )
        else:
            final_pd = micro_pd

        _log.info(
            "scenario.what_if",
            entity_id=entity_id,
            baseline_pd=round(baseline_pd, 5),
            micro_pd=round(micro_pd, 5),
            final_pd=round(final_pd, 5),
            n_overrides=len(applied),
            scenario=scenario_name or "",
            odds_multiplier=round(multiplier, 4),
        )
        return WhatIfResult(
            entity_id=entity_id,
            baseline_pd=baseline_pd,
            adjusted_pd=final_pd,
            scenario_name=scenario_name or "",
            overrides=applied,
            odds_multiplier=multiplier,
            sector=sector,
        )
