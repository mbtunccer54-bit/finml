"""Data sources.

Four adapters share one :class:`PanelFetcher` protocol:

* :class:`SyntheticPanelFetcher` — a reproducible quarterly panel with a genuine
  macro factor structure. It is the default so the platform trains, serves and
  tests with no network access and no API keys, which is what makes CI and a
  cold laptop behave identically.
* :class:`FREDFetcher`, :class:`WorldBankFetcher`, :class:`EDGARFetcher` — the
  real public sources.

Remote adapters retry transient failures with exponential backoff and raise
:class:`~domain.exceptions.DataFetchError` once the budget is spent. They never
fall back to synthetic data silently: a pipeline that thinks it read FRED must
have read FRED.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import httpx
import numpy as np
import pandas as pd

from domain.entities import (
    DATE_COLUMN,
    ENTITY_ID_COLUMN,
    FEATURE_CONTRACT,
    MACRO_VARIABLES,
    TARGET_COLUMN,
)
from domain.exceptions import DataFetchError
from domain.value_objects import Sector
from infrastructure.config.schemas import DataConfig
from infrastructure.logging import get_logger

__all__ = [
    "EDGARFetcher",
    "FREDFetcher",
    "PanelDataset",
    "PanelFetcher",
    "SyntheticPanelFetcher",
    "WorldBankFetcher",
    "build_fetcher",
]

_log = get_logger(__name__)

#: FRED series backing each macro variable.
FRED_SERIES: Final[dict[str, str]] = {
    "gdp_growth": "A191RL1Q225SBEA",
    "unemployment_rate": "UNRATE",
    "interest_rate": "FEDFUNDS",
    "inflation_rate": "CPIAUCSL",
    "credit_spread": "BAA10Y",
}


@dataclass(frozen=True, slots=True)
class PanelDataset:
    """The three tables every downstream stage consumes.

    Attributes:
        panel: One row per obligor-quarter, carrying features and the label.
        macro: One row per quarter of macro history, for the satellite models.
        news: Raw documents feeding the NLP stage; may be empty.
    """

    panel: pd.DataFrame
    macro: pd.DataFrame
    news: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate that the panel carries its identity columns.

        Raises:
            DataFetchError: If an identity column is missing.
        """
        required = {ENTITY_ID_COLUMN, DATE_COLUMN}
        missing = required - set(self.panel.columns)
        if missing:
            raise DataFetchError(
                "Panel is missing identity columns", missing=sorted(missing)
            )

    @property
    def n_entities(self) -> int:
        """Number of distinct obligors.

        Returns:
            The obligor count.
        """
        return int(self.panel[ENTITY_ID_COLUMN].nunique())

    @property
    def default_rate(self) -> float:
        """Observed positive-class prevalence.

        Returns:
            The share of defaulting rows, or ``0.0`` when unlabelled.
        """
        if TARGET_COLUMN not in self.panel.columns:
            return 0.0
        return float(self.panel[TARGET_COLUMN].mean())


@runtime_checkable
class PanelFetcher(Protocol):
    """A source of modelling data."""

    def fetch(self) -> PanelDataset:
        """Retrieve the dataset.

        Returns:
            The panel, macro history and news corpus.

        Raises:
            DataFetchError: If the source cannot be read.
        """
        ...


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
class SyntheticPanelFetcher:
    """Generates a reproducible credit panel with realistic structure.

    The generator is not noise with a label attached. It builds, in order:

    1. **A macro path** — AR(1) processes around plausible means, with an
       injected recession regime so the satellite VAR has something to fit.
    2. **Persistent firm quality** — a latent factor per obligor, so the panel
       has genuine cross-sectional heterogeneity rather than i.i.d. rows.
    3. **Ratios** — driven by quality, sector, and sector-specific macro
       sensitivity, which is what makes the macro-to-micro bridge meaningful.
    4. **Sentiment** — correlated with quality and with the macro cycle.
    5. **Labels** — Bernoulli draws from a log-odds index over the above, with
       the intercept solved so realised prevalence matches the configured rate.

    Because default risk genuinely depends on the features, a model that learns
    nothing scores near chance and one that overfits shows it on the test block.

    Attributes:
        config: Dataset configuration.
    """

    def __init__(self, config: DataConfig) -> None:
        """Initialise the generator.

        Args:
            config: Dataset configuration.

        Raises:
            DataFetchError: If the requested panel shape or prevalence is invalid.
        """
        if config.n_entities < 2 or config.n_periods < 2:
            raise DataFetchError(
                "Synthetic panel needs at least 2 entities and 2 periods",
                n_entities=config.n_entities,
                n_periods=config.n_periods,
            )
        if not 0.0 < config.default_rate < 1.0:
            raise DataFetchError(
                "default_rate must lie strictly inside (0, 1)",
                default_rate=config.default_rate,
            )
        self.config = config

    def fetch(self) -> PanelDataset:
        """Generate the dataset.

        Returns:
            The synthetic panel, macro history and news corpus.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        dates = pd.date_range(cfg.start_date, periods=cfg.n_periods, freq="QE")

        macro_df = self._generate_macro(rng, dates)
        sectors = rng.choice([s.value for s in Sector], size=cfg.n_entities)
        quality = rng.normal(0.0, 1.0, size=cfg.n_entities)

        frames = [
            self._generate_period(rng, macro_df.iloc[t], sectors, quality, dates[t])
            for t in range(cfg.n_periods)
        ]
        panel = pd.concat(frames, ignore_index=True)
        panel = self._assign_labels(rng, panel)
        panel = self._clip_to_contract(panel)
        news = self._generate_news(rng, panel)

        _log.info(
            "data.synthetic_generated",
            n_rows=len(panel),
            n_entities=cfg.n_entities,
            n_periods=cfg.n_periods,
            realised_default_rate=round(float(panel[TARGET_COLUMN].mean()), 4),
            target_default_rate=cfg.default_rate,
            n_news=len(news),
        )
        return PanelDataset(panel=panel, macro=macro_df, news=news)

    def _generate_macro(self, rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate the macro path.

        A recession is injected in the middle third of the sample: without a
        regime shift the VAR has no cycle to learn and every stress scenario
        extrapolates from a flat history.

        Args:
            rng: Seeded generator.
            dates: Quarterly index.

        Returns:
            One row per quarter, one column per macro variable.
        """
        n = len(dates)
        recession = np.zeros(n)
        start, end = int(n * 0.35), int(n * 0.55)
        if end > start:
            recession[start:end] = np.hanning(max(end - start, 3))[: end - start]

        def ar1(mean: float, phi: float, sigma: float, loading: float) -> np.ndarray:
            """Simulate an AR(1) path with a recession loading.

            Args:
                mean: Unconditional mean.
                phi: Persistence.
                sigma: Innovation standard deviation.
                loading: Sensitivity to the recession indicator.

            Returns:
                The simulated path.
            """
            out = np.empty(n)
            state = 0.0
            for i in range(n):
                state = phi * state + rng.normal(0.0, sigma)
                out[i] = mean + state + loading * recession[i]
            return out

        return pd.DataFrame(
            {
                DATE_COLUMN: dates,
                "gdp_growth": ar1(0.021, 0.55, 0.006, -0.055),
                "unemployment_rate": ar1(0.052, 0.80, 0.003, 0.038),
                "interest_rate": ar1(0.031, 0.88, 0.004, -0.012),
                "inflation_rate": ar1(0.026, 0.72, 0.005, 0.010),
                "credit_spread": ar1(0.018, 0.75, 0.002, 0.022),
            }
        )

    def _generate_period(
        self,
        rng: np.random.Generator,
        macro_row: pd.Series,
        sectors: np.ndarray,
        quality: np.ndarray,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Generate one quarter of obligor observations.

        Args:
            rng: Seeded generator.
            macro_row: Macro state for this quarter.
            sectors: Sector per obligor.
            quality: Latent quality per obligor.
            as_of: Observation date.

        Returns:
            One row per obligor.
        """
        n = len(quality)
        # Cyclical sectors load harder on the macro cycle, matching the bridge
        # sensitivities in domain/services.py.
        cyclical = np.isin(
            sectors,
            [
                Sector.CONSUMER_DISCRETIONARY.value,
                Sector.REAL_ESTATE.value,
                Sector.INDUSTRIALS.value,
                Sector.ENERGY.value,
                Sector.MATERIALS.value,
            ],
        ).astype(float)
        beta = 0.6 + 0.9 * cyclical

        cycle = (
            (float(macro_row["gdp_growth"]) - 0.021) * 12.0
            - (float(macro_row["unemployment_rate"]) - 0.052) * 8.0
            - (float(macro_row["credit_spread"]) - 0.018) * 10.0
        )
        # Health drives every ratio: high is a strong obligor in a good economy.
        health = quality + beta * cycle + rng.normal(0.0, 0.35, n)

        def draw(base: float, scale: float, noise: float, *, sign: float = 1.0) -> np.ndarray:
            """Draw a ratio from the health factor.

            Args:
                base: Value at neutral health.
                scale: Sensitivity to health.
                noise: Idiosyncratic standard deviation.
                sign: ``-1.0`` for ratios that fall as health rises.

            Returns:
                The drawn values.
            """
            return base + sign * scale * health + rng.normal(0.0, noise, n)

        data: dict[str, Any] = {
            ENTITY_ID_COLUMN: [f"ENT{i:05d}" for i in range(n)],
            DATE_COLUMN: as_of,
            "sector": sectors,
            # Liquidity
            "current_ratio": np.exp(draw(0.35, 0.22, 0.20)),
            "quick_ratio": np.exp(draw(0.02, 0.22, 0.20)),
            "working_capital_ratio": draw(0.18, 0.09, 0.07),
            "cash_flow_to_debt": draw(0.28, 0.14, 0.12),
            # Leverage (falls as health rises)
            "debt_to_equity": np.exp(draw(0.30, 0.28, 0.25, sign=-1.0)),
            "debt_to_assets": 1.0 / (1.0 + np.exp(-draw(-0.35, 0.40, 0.30, sign=-1.0))),
            "interest_coverage": draw(4.5, 2.4, 1.5),
            # Profitability
            "return_on_assets": draw(0.05, 0.035, 0.025),
            "return_on_equity": draw(0.12, 0.075, 0.055),
            "net_profit_margin": draw(0.07, 0.045, 0.035),
            "operating_margin": draw(0.11, 0.055, 0.040),
            # Efficiency
            "asset_turnover": np.exp(draw(-0.20, 0.10, 0.18)),
            "revenue_growth": draw(0.04, 0.045, 0.045),
        }

        # Altman Z is a deterministic composite of the ratios above plus noise,
        # so it is correlated with them the way the real score is.
        data["altman_z_score"] = (
            1.2 * data["working_capital_ratio"]
            + 1.4 * data["return_on_assets"]
            + 3.3 * data["operating_margin"]
            + 0.6 / (data["debt_to_assets"] + 0.1)
            + 1.0 * data["asset_turnover"]
            + rng.normal(0.0, 0.25, n)
        )

        # Sentiment tracks health with a lag and its own noise.
        sentiment_latent = 0.55 * health + rng.normal(0.0, 0.7, n)
        neg_prob = 1.0 / (1.0 + np.exp(1.2 * sentiment_latent))
        pos_prob = (1.0 - neg_prob) * rng.uniform(0.35, 0.75, n)
        data["sentiment_negative_prob"] = neg_prob
        data["sentiment_compound"] = pos_prob - neg_prob
        data["sentiment_uncertainty"] = np.clip(
            0.85 - 0.45 * np.abs(pos_prob - neg_prob) + rng.normal(0.0, 0.06, n), 0.0, 1.0
        )
        data["news_volume"] = rng.poisson(np.clip(9.0 - 2.2 * health, 1.0, 60.0), n).astype(float)

        # Risk-topic intensity rises as health deteriorates.
        for name, strength in (
            ("topic_credit_risk_score", 1.05),
            ("topic_liquidity_risk_score", 0.85),
            ("topic_operational_risk_score", 0.45),
        ):
            data[name] = np.clip(
                1.0 / (1.0 + np.exp(strength * health + rng.normal(0.0, 0.55, n))), 0.0, 1.0
            )

        for name in MACRO_VARIABLES:
            data[name] = float(macro_row[name])

        data["_health"] = health
        return pd.DataFrame(data)

    def _assign_labels(self, rng: np.random.Generator, panel: pd.DataFrame) -> pd.DataFrame:
        """Draw default outcomes and calibrate the base rate.

        The intercept is solved by bisection so realised prevalence matches the
        configured ``default_rate``. Fixing the intercept instead would make
        prevalence an accident of the coefficient values.

        Args:
            rng: Seeded generator.
            panel: Panel with the latent health column.

        Returns:
            The panel with ``default_flag`` added and the latent column dropped.
        """
        health = panel["_health"].to_numpy()
        score = (
            -1.15 * health
            - 0.55 * panel["interest_coverage"].to_numpy().clip(-10, 30) / 10.0
            + 0.85 * panel["debt_to_assets"].to_numpy()
            - 0.70 * panel["altman_z_score"].to_numpy() / 4.0
            + 0.60 * panel["sentiment_negative_prob"].to_numpy()
            + 0.45 * panel["topic_credit_risk_score"].to_numpy()
            + rng.normal(0.0, 0.45, len(panel))
        )

        target = self.config.default_rate
        low, high = -25.0, 25.0
        for _ in range(80):
            mid = 0.5 * (low + high)
            rate = float(np.mean(1.0 / (1.0 + np.exp(-(mid + score)))))
            if rate < target:
                low = mid
            else:
                high = mid
        intercept = 0.5 * (low + high)

        probability = 1.0 / (1.0 + np.exp(-(intercept + score)))
        out = panel.drop(columns=["_health"]).copy()
        out[TARGET_COLUMN] = (rng.uniform(0.0, 1.0, len(panel)) < probability).astype("int64")
        return out

    @staticmethod
    def _clip_to_contract(panel: pd.DataFrame) -> pd.DataFrame:
        """Clip every numeric column into its contractual bounds.

        Args:
            panel: The generated panel.

        Returns:
            The panel with values inside the contract's declared ranges.
        """
        out = panel.copy()
        for spec in FEATURE_CONTRACT.fields:
            if spec.name not in out.columns or spec.dtype not in ("float", "int"):
                continue
            if spec.ge is not None or spec.le is not None:
                out[spec.name] = out[spec.name].clip(lower=spec.ge, upper=spec.le)
        return out

    def _generate_news(self, rng: np.random.Generator, panel: pd.DataFrame) -> pd.DataFrame:
        """Generate a small headline corpus aligned with the panel's sentiment.

        Args:
            rng: Seeded generator.
            panel: The labelled panel.

        Returns:
            One row per sampled headline.
        """
        negative = [
            "{name} faces covenant breach as liquidity tightens",
            "Analysts downgrade {name} on weak cash flow",
            "{name} delays refinancing amid funding pressure",
            "Regulator opens investigation into {name}",
            "{name} warns on margins as demand softens",
        ]
        positive = [
            "{name} beats expectations on strong operating margin",
            "{name} refinances debt at improved terms",
            "Upgrade for {name} as leverage falls",
            "{name} reports record free cash flow",
            "{name} expands into a higher-growth segment",
        ]
        neutral = [
            "{name} confirms quarterly results date",
            "{name} appoints new chief financial officer",
            "{name} completes routine debt rollover",
        ]

        sample = panel.sample(n=min(len(panel), 600), random_state=int(self.config.seed))
        rows: list[dict[str, Any]] = []
        for record in sample.to_dict(orient="records"):
            compound = float(record["sentiment_compound"])
            pool = negative if compound < -0.15 else positive if compound > 0.15 else neutral
            headline = str(rng.choice(pool)).format(name=str(record[ENTITY_ID_COLUMN]))
            rows.append(
                {
                    ENTITY_ID_COLUMN: record[ENTITY_ID_COLUMN],
                    "published_at": record[DATE_COLUMN],
                    "headline": headline,
                    "body": headline + ". Full coverage is not reproduced in the demo corpus.",
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Remote sources
# ---------------------------------------------------------------------------
class _HttpFetcherBase:
    """Shared retry, caching and error handling for HTTP sources.

    Attributes:
        config: Dataset configuration supplying timeouts and the retry budget.
    """

    #: SEC requires a descriptive User-Agent and rejects requests without one.
    _USER_AGENT = "finml-platform/0.1 (compliance@example.com)"

    def __init__(self, config: DataConfig) -> None:
        """Initialise the fetcher.

        Args:
            config: Dataset configuration.
        """
        self.config = config
        self._cache_dir = Path(config.cache_dir)

    def _cache_path(self, key: str) -> Path:
        """Build the on-disk cache path for a request key.

        Args:
            key: Stable request identifier.

        Returns:
            Path the payload is cached at.
        """
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self._cache_dir / f"{safe}.json"

    def _get_json(self, url: str, params: dict[str, Any], *, cache_key: str) -> Any:
        """GET a JSON payload, with caching and bounded retries.

        The return type is deliberately ``Any``: FRED and EDGAR return a JSON
        object while the World Bank returns a two-element array, so narrowing
        this to ``dict`` would be a lie that also makes the World Bank adapter's
        shape check look like dead code.

        Args:
            url: Request URL.
            params: Query parameters.
            cache_key: Stable identifier for the on-disk cache.

        Returns:
            The decoded payload, either a mapping or a list.

        Raises:
            DataFetchError: If every attempt fails or the response is not JSON.
        """
        cached = self._cache_path(cache_key)
        if cached.is_file():
            try:
                payload: dict[str, Any] = json.loads(cached.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                # A corrupt cache entry must not be fatal; re-fetch instead.
                _log.warning("fetch.cache_unreadable", cache_key=cache_key, error=str(exc))
            else:
                _log.debug("fetch.cache_hit", cache_key=cache_key)
                return payload

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = httpx.get(
                    url,
                    params=params,
                    timeout=self.config.request_timeout_s,
                    headers={"User-Agent": self._USER_AGENT},
                    follow_redirects=True,
                )
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                backoff = min(2.0**attempt, 30.0)
                _log.warning(
                    "fetch.attempt_failed",
                    url=url,
                    attempt=attempt,
                    max_attempts=self.config.max_retries,
                    backoff_s=backoff,
                    error=str(exc),
                )
                if attempt < self.config.max_retries:
                    time.sleep(backoff)
                continue

            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(payload), encoding="utf-8")
            except OSError as exc:
                # Caching is an optimisation; failing to write must not fail the fetch.
                _log.warning("fetch.cache_write_failed", cache_key=cache_key, error=str(exc))
            return payload

        raise DataFetchError(
            "Exhausted retry budget",
            url=url,
            attempts=self.config.max_retries,
            reason=str(last_error),
        )


class FREDFetcher(_HttpFetcherBase):
    """Fetches macro series from the St. Louis Fed FRED API.

    Attributes:
        config: Dataset configuration; the API key is read from it or the
            ``FRED_API_KEY`` environment variable.
    """

    _BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_series(self, series_id: str) -> pd.Series:
        """Fetch one FRED series.

        Args:
            series_id: FRED series identifier.

        Returns:
            A date-indexed float series.

        Raises:
            DataFetchError: If no API key is configured or the payload is
                missing its observations.
        """
        import os

        api_key = self.config.fred_api_key or os.environ.get("FRED_API_KEY", "")
        if not api_key:
            raise DataFetchError(
                "FRED API key is not configured",
                hint="Set FRED_API_KEY or data.fred_api_key",
                series_id=series_id,
            )
        payload = self._get_json(
            self._BASE_URL,
            {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": self.config.start_date,
            },
            cache_key=f"fred_{series_id}_{self.config.start_date}",
        )
        observations = payload.get("observations")
        if not observations:
            raise DataFetchError("FRED returned no observations", series_id=series_id)

        frame = pd.DataFrame(observations)
        values = pd.to_numeric(frame["value"], errors="coerce")
        index = pd.to_datetime(frame["date"])
        return pd.Series(values.to_numpy(), index=index, name=series_id).dropna()

    def fetch_macro(self) -> pd.DataFrame:
        """Fetch every configured macro variable and align to quarter-end.

        Returns:
            One row per quarter, one column per macro variable.

        Raises:
            DataFetchError: If any series cannot be retrieved.
        """
        columns: dict[str, pd.Series] = {}
        for variable, series_id in FRED_SERIES.items():
            raw = self.fetch_series(series_id)
            quarterly = raw.resample("QE").last()
            # FRED publishes every one of these series in percent -- including
            # the annualised GDP growth rate -- and the contract stores fractions.
            columns[variable] = quarterly / 100.0

        macro = pd.DataFrame(columns).dropna()
        macro.index.name = DATE_COLUMN
        return macro.reset_index()

    def fetch(self) -> PanelDataset:
        """Fetch macro history and pair it with a synthetic obligor panel.

        FRED publishes no obligor-level financials, so the micro panel is
        generated and the *real* macro path is joined onto it. The result is
        explicitly a hybrid, and is logged as such.

        Returns:
            The hybrid dataset.
        """
        macro = self.fetch_macro()
        synthetic_cfg = DataConfig(**{**self.config.__dict__, "n_periods": max(len(macro), 4)})
        base = SyntheticPanelFetcher(synthetic_cfg).fetch()

        _log.warning(
            "data.hybrid_source",
            reason="FRED supplies macro only; obligor financials are synthetic",
            n_macro_rows=len(macro),
        )
        return PanelDataset(panel=base.panel, macro=macro, news=base.news)


class WorldBankFetcher(_HttpFetcherBase):
    """Fetches country indicators from the World Bank API.

    Attributes:
        config: Dataset configuration.
    """

    _BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

    _INDICATORS: Final[dict[str, str]] = {
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "unemployment_rate": "SL.UEM.TOTL.ZS",
        "inflation_rate": "FP.CPI.TOTL.ZG",
    }

    def fetch_indicator(self, indicator: str, country: str = "US") -> pd.Series:
        """Fetch one World Bank indicator.

        Args:
            indicator: World Bank indicator code.
            country: ISO country code.

        Returns:
            A year-indexed float series.

        Raises:
            DataFetchError: If the response is not in the documented shape.
        """
        payload = self._get_json(
            self._BASE_URL.format(country=country, indicator=indicator),
            {"format": "json", "per_page": "500"},
            cache_key=f"wb_{country}_{indicator}",
        )
        # The World Bank wraps results as [metadata, rows]; httpx returns that
        # as a list, so a dict here means an API-level error object.
        if isinstance(payload, dict):
            raise DataFetchError(
                "World Bank returned an error object", indicator=indicator, payload=payload
            )
        rows = payload[1] if len(payload) > 1 else []
        if not rows:
            raise DataFetchError("World Bank returned no rows", indicator=indicator)

        frame = pd.DataFrame(rows)[["date", "value"]].dropna()
        index = pd.to_datetime(frame["date"], format="%Y")
        return pd.Series(
            pd.to_numeric(frame["value"]).to_numpy() / 100.0, index=index, name=indicator
        ).sort_index()

    def fetch(self) -> PanelDataset:
        """Fetch indicators and pair them with a synthetic obligor panel.

        Returns:
            The hybrid dataset.
        """
        series = {
            name: self.fetch_indicator(code) for name, code in self._INDICATORS.items()
        }
        macro = pd.DataFrame(series).dropna()
        for missing in set(MACRO_VARIABLES) - set(macro.columns):
            macro[missing] = float(FEATURE_CONTRACT.get(missing).typical)
        macro.index.name = DATE_COLUMN
        macro = macro.reset_index()

        synthetic_cfg = DataConfig(**{**self.config.__dict__, "n_periods": max(len(macro), 4)})
        base = SyntheticPanelFetcher(synthetic_cfg).fetch()
        _log.warning(
            "data.hybrid_source",
            reason="World Bank supplies macro only; obligor financials are synthetic",
            n_macro_rows=len(macro),
        )
        return PanelDataset(panel=base.panel, macro=macro, news=base.news)


class EDGARFetcher(_HttpFetcherBase):
    """Fetches company facts from the SEC EDGAR submissions API.

    Attributes:
        config: Dataset configuration.
    """

    _FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

    def fetch_company_facts(self, cik: int) -> dict[str, Any]:
        """Fetch the XBRL company-facts payload for one filer.

        Args:
            cik: Central Index Key.

        Returns:
            The decoded payload.

        Raises:
            DataFetchError: If the filer cannot be retrieved.
        """
        payload = self._get_json(
            self._FACTS_URL.format(cik=cik), {}, cache_key=f"edgar_cik_{cik:010d}"
        )
        if not isinstance(payload, dict):
            raise DataFetchError(
                "EDGAR returned an unexpected payload shape",
                cik=cik,
                got=type(payload).__name__,
            )
        return payload

    def fetch(self) -> PanelDataset:
        """Not implemented as a standalone panel source.

        Turning raw XBRL facts into a comparable ratio panel requires a
        normalisation layer (taxonomy mapping, fiscal-period alignment,
        restatement handling) that is a project in its own right. Raising is
        honest; silently returning synthetic data would not be.

        Raises:
            DataFetchError: Always.
        """
        raise DataFetchError(
            "EDGAR panel construction is not implemented",
            hint=(
                "Use fetch_company_facts() for a single filer, or data.source=synthetic "
                "for an end-to-end run"
            ),
        )


def build_fetcher(config: DataConfig) -> PanelFetcher:
    """Construct the fetcher named by the configuration.

    Args:
        config: Dataset configuration.

    Returns:
        The matching fetcher.

    Raises:
        DataFetchError: If ``config.source`` names no known adapter.
    """
    registry: dict[str, type] = {
        "synthetic": SyntheticPanelFetcher,
        "fred": FREDFetcher,
        "worldbank": WorldBankFetcher,
        "edgar": EDGARFetcher,
    }
    source = config.source.strip().lower()
    if source not in registry:
        raise DataFetchError(
            "Unknown data source", source=config.source, known=sorted(registry)
        )
    fetcher: PanelFetcher = registry[source](config)
    _log.info("data.fetcher_selected", source=source, fetcher=type(fetcher).__name__)
    return fetcher
