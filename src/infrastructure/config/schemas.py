"""Hydra structured configuration schemas.

Every tunable in the platform is declared here as a dataclass and registered
with Hydra's :class:`~hydra.core.config_store.ConfigStore`. That gives three
things a plain YAML tree does not:

* **Validation at compose time** — a typo in ``configs/`` fails before a model
  is trained rather than three hours in.
* **Type coercion** — values arriving from the CLI are cast to the declared type.
* **A single readable inventory** of what can be configured.

Anything here can be overridden from the command line, for example::

    python -m application.train_pipeline model.ensemble.enabled=false seed=7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

__all__ = [
    "APIConfig",
    "CalibrationConfig",
    "DashboardConfig",
    "DataConfig",
    "DriftConfig",
    "EnsembleConfig",
    "FeatureConfig",
    "LoggingConfig",
    "ModelConfig",
    "ModelSpec",
    "NLPConfig",
    "PathsConfig",
    "RootConfig",
    "ScenarioConfig",
    "ScenarioSpec",
    "SplitConfig",
    "TrackingConfig",
    "TrainingConfig",
    "TuningConfig",
    "XAIConfig",
    "register_configs",
]


# ---------------------------------------------------------------------------
# Paths and cross-cutting concerns
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    """Filesystem locations used across the platform.

    Attributes:
        artifacts: Root for trained models and evaluation artifacts.
        feature_store: Root for the Parquet feature store.
        raw_data: Landing area for fetched raw data.
        reports: Destination for generated plots and benchmark tables.
    """

    artifacts: str = "artifacts"
    feature_store: str = "data/feature_store"
    raw_data: str = "data/raw"
    reports: str = "artifacts/reports"


@dataclass
class LoggingConfig:
    """Structured logging behaviour.

    Attributes:
        level: Root log level name.
        json_output: Emit JSON lines when true, human-readable console when false.
        add_timestamp: Include an ISO-8601 UTC timestamp on every event.
        include_caller: Include module, function and line number.
    """

    level: str = "INFO"
    json_output: bool = True
    add_timestamp: bool = True
    include_caller: bool = False


@dataclass
class TrackingConfig:
    """MLflow experiment tracking and model registry.

    Attributes:
        enabled: Disable to run pipelines without a tracking server.
        tracking_uri: MLflow tracking URI. ``sqlite:///`` needs no server and is
            the default; the older ``file:./mlruns`` backend was put into
            maintenance mode in MLflow 3 and now raises on connect.
        experiment_name: Experiment runs are grouped under.
        registered_model_name: Name used in the MLflow model registry.
        log_artifacts: Whether to upload plots and model binaries.
        dataset_version: Tag recorded on every run for lineage.
    """

    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "finml-credit-risk"
    registered_model_name: str = "finml-pd-model"
    log_artifacts: bool = True
    dataset_version: str = "v1"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class SplitConfig:
    """Time-series cross-validation strategy.

    ``TimeSeriesSplit`` is deliberately not offered: it leaves no gap between
    train and validation, so an obligor observed either side of the boundary
    leaks its outcome. The purged variants embargo that neighbourhood.

    Attributes:
        strategy: One of ``purged_kfold``, ``blocked``, ``expanding``, ``sliding``.
        n_splits: Number of folds.
        embargo_frac: Fraction of the sample embargoed after each validation
            block, guarding against serial correlation leaking backwards.
        purge_frac: Fraction purged between train and validation.
        test_size_frac: Share of the timeline held out as the final test set.
        min_train_frac: Smallest training share for the expanding strategy.
        window_frac: Training window width for the sliding strategy.
    """

    strategy: str = "purged_kfold"
    n_splits: int = 5
    embargo_frac: float = 0.01
    purge_frac: float = 0.01
    test_size_frac: float = 0.2
    min_train_frac: float = 0.3
    window_frac: float = 0.4


@dataclass
class DataConfig:
    """Dataset construction and sourcing.

    Attributes:
        source: ``synthetic`` generates a reproducible panel offline; ``fred``,
            ``worldbank`` and ``edgar`` hit the corresponding public APIs.
        n_entities: Obligor count for the synthetic generator.
        n_periods: Number of quarterly observations per obligor.
        start_date: First observation date.
        default_rate: Target positive-class prevalence for the generator.
        seed: Generator seed.
        fred_api_key: FRED API key; read from the environment when empty.
        cache_dir: Where fetched payloads are cached.
        request_timeout_s: Per-request timeout for remote fetchers.
        max_retries: Retry budget for transient fetch failures.
        validate_schema: Enforce the pandera contract after loading.
        split: Cross-validation strategy.
    """

    source: str = "synthetic"
    n_entities: int = 1200
    n_periods: int = 16
    start_date: str = "2019-01-01"
    default_rate: float = 0.08
    seed: int = 42
    fred_api_key: str = ""
    cache_dir: str = "data/raw/cache"
    request_timeout_s: float = 20.0
    max_retries: int = 3
    validate_schema: bool = True
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class FeatureConfig:
    """Feature engineering and imbalance handling.

    Attributes:
        use_nlp_features: Include the FinBERT and topic-model columns.
        use_macro_features: Include the macro columns.
        feature_store_enabled: Cache expensive features between runs.
        imputation: ``typical`` uses the contract's representative value;
            ``median`` computes it from the training split.
        imbalance_strategy: ``class_weight``, ``smote``, ``focal`` or ``none``.
        smote_k_neighbors: Neighbour count for SMOTE.
        smote_sampling_strategy: Desired minority-to-majority ratio after SMOTE.
        focal_gamma: Focusing parameter for the focal loss.
        focal_alpha: Positive-class weight for the focal loss.
        scale_numeric: Standardise numeric features; required by the linear
            meta-learner, harmless for the tree models.
        drop_correlated_above: Drop one of any feature pair above this absolute
            Pearson correlation; disabled when set to ``1.0``.
    """

    use_nlp_features: bool = True
    use_macro_features: bool = True
    feature_store_enabled: bool = True
    imputation: str = "typical"
    imbalance_strategy: str = "class_weight"
    smote_k_neighbors: int = 5
    smote_sampling_strategy: float = 0.3
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    scale_numeric: bool = True
    drop_correlated_above: float = 0.98


@dataclass
class NLPConfig:
    """Natural-language feature extraction.

    All three components degrade to deterministic lexicon or keyword fallbacks
    when the optional ``nlp`` extra is not installed, so the pipeline runs
    end-to-end on a machine without ``torch``.

    Attributes:
        enabled: Master switch for the NLP stage.
        finbert_model: Hugging Face model id for sentiment.
        ner_model: spaCy pipeline name for entity extraction.
        topic_model: Sentence-transformer backbone for BERTopic.
        batch_size: Inference batch size.
        max_length: Token truncation length.
        device: ``auto``, ``cpu`` or ``cuda``.
        lookback_days: News window terminating at each observation date.
        risk_topics: Risk categories mapped to their seed keywords.
        allow_fallback: Permit the lexicon fallback when transformers are absent.
    """

    enabled: bool = True
    finbert_model: str = "ProsusAI/finbert"
    ner_model: str = "en_core_web_sm"
    topic_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"
    lookback_days: int = 90
    risk_topics: dict[str, list[str]] = field(
        default_factory=lambda: {
            "credit": ["default", "downgrade", "covenant", "insolvency", "restructuring"],
            "liquidity": ["liquidity", "cash flow", "refinancing", "maturity", "funding"],
            "operational": ["fraud", "outage", "recall", "litigation", "cyber"],
        }
    )
    allow_fallback: bool = True


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """One base learner.

    Attributes:
        enabled: Include this learner in training and the ensemble.
        params: Static hyperparameters passed to the estimator constructor.
        search_space: Optuna search space, keyed by parameter name. Each value
            is a mapping such as
            ``{type: float, low: 0.01, high: 0.3, log: true}`` or
            ``{type: categorical, choices: [...]}``.
    """

    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningConfig:
    """Optuna hyperparameter optimisation.

    The study is genuinely multi-objective: PR-AUC is maximised while median
    single-row inference latency is minimised, and the Pareto front is resolved
    by the configured preference weights.

    Attributes:
        enabled: Run tuning; when false the static ``params`` are used as-is.
        n_trials: Trials per model.
        timeout_s: Wall-clock budget per model; ``0`` disables the limit.
        directions: Objective directions, in the order the objective returns them.
        pr_auc_weight: Preference weight on PR-AUC when collapsing the front.
        latency_weight: Preference weight on latency when collapsing the front.
        latency_budget_ms: Latency treated as the normalisation scale.
        n_startup_trials: Random trials before the sampler starts modelling.
        pruning_enabled: Enable the median pruner.
        seed: Sampler seed.
        n_jobs: Parallel trial workers.
    """

    enabled: bool = True
    n_trials: int = 25
    timeout_s: int = 0
    directions: list[str] = field(default_factory=lambda: ["maximize", "minimize"])
    pr_auc_weight: float = 0.85
    latency_weight: float = 0.15
    latency_budget_ms: float = 10.0
    n_startup_trials: int = 5
    pruning_enabled: bool = True
    seed: int = 42
    n_jobs: int = 1


@dataclass
class CalibrationConfig:
    """Probability calibration.

    Attributes:
        enabled: Fit calibrated variants of every model.
        methods: Calibration methods to fit, from ``sigmoid`` and ``isotonic``.
        cv_folds: Internal folds used by ``CalibratedClassifierCV``.
        n_bins: Bin count for the reliability diagram and the ECE.
        max_calibration_error: Deployment gate on expected calibration error.
    """

    enabled: bool = True
    methods: list[str] = field(default_factory=lambda: ["sigmoid", "isotonic"])
    cv_folds: int = 3
    n_bins: int = 10
    max_calibration_error: float = 0.05


@dataclass
class EnsembleConfig:
    """Stacked ensemble over the base learners.

    Attributes:
        enabled: Fit the stack.
        meta_learner: ``logistic_regression`` or ``lightgbm``.
        cv_folds: Folds used to generate out-of-fold meta-features.
        passthrough: Feed the original features to the meta-learner as well.
        calibrate: Also produce a calibrated variant of the stack.
    """

    enabled: bool = True
    meta_learner: str = "logistic_regression"
    cv_folds: int = 5
    passthrough: bool = False
    calibrate: bool = True


@dataclass
class CostConfig:
    """Business cost of misclassification.

    Attributes:
        false_negative_cost: Cost of missing a true defaulter.
        false_positive_cost: Cost of flagging a healthy obligor.
        true_positive_cost: Cost of a correctly flagged defaulter.
        true_negative_cost: Cost of a correctly cleared obligor.
    """

    false_negative_cost: float = 10.0
    false_positive_cost: float = 1.0
    true_positive_cost: float = 0.0
    true_negative_cost: float = 0.0


@dataclass
class ModelConfig:
    """The full modelling configuration.

    Attributes:
        catboost: CatBoost learner spec.
        xgboost: XGBoost learner spec.
        lightgbm: LightGBM learner spec.
        random_forest: Random forest learner spec.
        ensemble: Stacking configuration.
        calibration: Calibration configuration.
        tuning: Optuna configuration.
        cost: Business cost matrix.
        pd_floor: Regulatory PD floor applied to every reported score.
        pd_cap: Conservatism cap applied to every reported score.
        risk_band_thresholds: Four ascending PD cut-points for the rating bands.
        selection_min_pr_auc: Minimum PR-AUC for a model to be promotable.
        selection_max_latency_ms: Maximum median latency for a promotable model.
    """

    catboost: ModelSpec = field(default_factory=ModelSpec)
    xgboost: ModelSpec = field(default_factory=ModelSpec)
    lightgbm: ModelSpec = field(default_factory=ModelSpec)
    random_forest: ModelSpec = field(default_factory=ModelSpec)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    pd_floor: float = 0.0003
    pd_cap: float = 1.0
    risk_band_thresholds: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.15, 0.30])
    selection_min_pr_auc: float = 0.10
    selection_max_latency_ms: float = 50.0


@dataclass
class TrainingConfig:
    """Training run behaviour.

    Attributes:
        seed: Global random seed.
        n_jobs: Worker count for estimators that support it.
        early_stopping_rounds: Boosting rounds without improvement before stopping.
        verbose: Emit per-iteration training output.
        register_best: Register the selected champion in the model registry.
        save_artifacts: Persist models and plots to :attr:`PathsConfig.artifacts`.
        compute_shap_on_train: Produce a global SHAP summary during training.
        shap_sample_size: Rows sampled for the global SHAP computation.
    """

    seed: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 50
    verbose: bool = False
    register_best: bool = True
    save_artifacts: bool = True
    compute_shap_on_train: bool = True
    shap_sample_size: int = 500


# ---------------------------------------------------------------------------
# Monitoring, scenarios and explainability
# ---------------------------------------------------------------------------
@dataclass
class DriftConfig:
    """Input drift detection.

    Attributes:
        enabled: Run the drift check before scoring.
        n_bins: Quantile bins used for the PSI.
        psi_moderate_threshold: PSI above which drift is ``MODERATE``.
        psi_severe_threshold: PSI above which drift is ``SEVERE``.
        ks_alpha: Significance level for the Kolmogorov-Smirnov test.
        on_severe: ``block`` raises and refuses to score; ``warn`` logs, alerts
            and continues. Both paths always emit a structured alert.
        min_reference_size: Smallest reference window that yields a verdict.
        min_batch_size: Smallest scoring batch that yields a verdict. A PSI over
            a handful of rows is dominated by sampling noise, so an interactive
            single-obligor request must not be refused for "drift".
        features: Restrict monitoring to these features; empty means all numeric.
        exclude_features: Features never monitored. Macro variables belong here:
            they are constant within a single scoring batch but vary across the
            pooled training reference, so their PSI is enormous by construction
            and says nothing about input quality.
    """

    enabled: bool = True
    n_bins: int = 10
    psi_moderate_threshold: float = 0.10
    psi_severe_threshold: float = 0.25
    ks_alpha: float = 0.05
    on_severe: str = "block"
    min_reference_size: int = 50
    min_batch_size: int = 30
    features: list[str] = field(default_factory=list)
    exclude_features: list[str] = field(
        default_factory=lambda: [
            "gdp_growth",
            "unemployment_rate",
            "interest_rate",
            "inflation_rate",
            "credit_spread",
        ]
    )


@dataclass
class ScenarioSpec:
    """One macro stress scenario.

    Attributes:
        name: Scenario identifier.
        severity: ``baseline``, ``adverse`` or ``severely_adverse``.
        description: Narrative shown in the dashboard and reports.
        horizon_quarters: Projection horizon.
        probability_weight: Weight when aggregating scenarios.
        shocks: Absolute deltas keyed by macro variable name.
    """

    name: str = "baseline"
    severity: str = "baseline"
    description: str = ""
    horizon_quarters: int = 8
    probability_weight: float = 1.0
    shocks: dict[str, float] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    """Stress testing and the macro-to-micro bridge.

    Attributes:
        satellite_model: ``var`` fits a vector autoregression; ``ar`` fits
            independent univariate models as a fallback on short samples.
        var_max_lags: Maximum lag order considered by the VAR.
        forecast_horizon: Quarters projected by the satellite model.
        revenue_to_log_odds: Sensitivity of default log-odds to a relative
            revenue shock.
        max_log_odds_shift: Saturation cap on the transmitted shift.
        sector_sensitivities: Optional per-sector overrides of the shipped
            bridge parameters.
        scenarios: The configured scenarios.
    """

    satellite_model: str = "var"
    var_max_lags: int = 4
    forecast_horizon: int = 8
    revenue_to_log_odds: float = 8.0
    max_log_odds_shift: float = 3.0
    sector_sensitivities: dict[str, Any] = field(default_factory=dict)
    scenarios: list[ScenarioSpec] = field(default_factory=list)


@dataclass
class XAIConfig:
    """Explainability and model-risk diagnostics.

    Attributes:
        shap_backend: ``tree`` for TreeSHAP, ``kernel`` for the model-agnostic
            fallback, ``auto`` to pick per model.
        shap_sample_size: Rows used for global SHAP summaries.
        shap_interaction_top_k: Features retained for the interaction heatmap.
        lime_num_features: Features shown in a LIME explanation.
        lime_num_samples: Perturbations drawn per LIME explanation.
        dice_total_cfs: Counterfactuals generated per query.
        dice_method: DiCE generation method.
        dice_features_to_vary: Actionable features; empty means all numeric.
        dice_desired_pd: Target PD the counterfactual should reach.
        permutation_repeats: Shuffles per feature for permutation importance.
        stability_n_seeds: Seeds used for the attribution stability check.
        stability_tolerance: Largest acceptable scaled dispersion.
        fairness_attribute: Column used for the fairness breakdown.
        fairness_tolerance: Largest acceptable demographic-parity gap.
    """

    shap_backend: str = "auto"
    shap_sample_size: int = 500
    shap_interaction_top_k: int = 8
    lime_num_features: int = 10
    lime_num_samples: int = 2000
    dice_total_cfs: int = 3
    dice_method: str = "random"
    dice_features_to_vary: list[str] = field(default_factory=list)
    dice_desired_pd: float = 0.2
    permutation_repeats: int = 10
    stability_n_seeds: int = 5
    stability_tolerance: float = 0.25
    fairness_attribute: str = "sector"
    fairness_tolerance: float = 0.10


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------
@dataclass
class APIConfig:
    """FastAPI service behaviour.

    Attributes:
        host: Bind address.
        port: Bind port.
        title: OpenAPI title.
        version: API version string.
        cors_origins: Permitted CORS origins.
        max_batch_size: Largest accepted batch in one ``/predict`` call.
        enable_metrics: Expose the Prometheus ``/metrics`` endpoint.
        model_stage: Registry stage the service loads at startup.
        explain_default_method: Default explainer for ``/explain``.
        request_timeout_s: Server-side handler timeout.
    """

    # Loopback by default: the container and `make serve` pass `--host 0.0.0.0`
    # to uvicorn explicitly, so binding every interface stays a deliberate
    # deployment decision rather than something inherited from a default.
    host: str = "127.0.0.1"
    port: int = 8000
    title: str = "FinML Credit Risk API"
    version: str = "0.1.0"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_batch_size: int = 1000
    enable_metrics: bool = True
    model_stage: str = "Production"
    explain_default_method: str = "shap"
    request_timeout_s: float = 30.0


@dataclass
class DashboardConfig:
    """Streamlit dashboard behaviour.

    Attributes:
        api_url: Base URL of the FastAPI service.
        title: Page title.
        page_icon: Browser tab icon.
        layout: Streamlit layout mode.
        cache_ttl_s: Time-to-live for cached API responses.
        default_sector: Sector pre-selected on load; empty means all.
        max_entities_displayed: Row cap on the portfolio table.
        risk_matrix_bins: Grid resolution of the risk heatmap.
        show_raw_data: Expose the raw dataframe expander.
        theme_primary_color: Accent colour for the Plotly charts.
    """

    api_url: str = "http://localhost:8000"
    title: str = "FinML Credit Risk Platform"
    page_icon: str = "📊"
    layout: str = "wide"
    cache_ttl_s: int = 300
    default_sector: str = ""
    max_entities_displayed: int = 500
    risk_matrix_bins: int = 5
    show_raw_data: bool = False
    theme_primary_color: str = "#2E86AB"


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@dataclass
class RootConfig:
    """Top-level composed configuration.

    Attributes:
        seed: Global seed propagated to every stochastic component.
        run_name: Optional run label; a timestamp is generated when empty.
        paths: Filesystem locations.
        logging: Structured logging behaviour.
        tracking: MLflow configuration.
        data: Dataset construction.
        features: Feature engineering.
        nlp: Natural-language features.
        model: Modelling configuration.
        training: Training run behaviour.
        drift: Drift detection.
        scenario: Stress testing.
        xai: Explainability.
        api: FastAPI service.
        dashboard: Streamlit dashboard.
    """

    seed: int = 42
    run_name: str = ""
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    api: APIConfig = field(default_factory=APIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def register_configs() -> None:
    """Register every schema with Hydra's ``ConfigStore``.

    Registration is idempotent, so calling this from several entrypoints (the
    trainer, the API, the dashboard) is safe.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=RootConfig)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="features", name="base_features", node=FeatureConfig)
    cs.store(group="models", name="base_models", node=ModelConfig)
    cs.store(group="dashboard", name="base_dashboard", node=DashboardConfig)
    cs.store(group="scenario", name="base_scenario", node=ScenarioConfig)
    cs.store(group="drift", name="base_drift", node=DriftConfig)
    cs.store(group="xai", name="base_xai", node=XAIConfig)
    cs.store(group="nlp", name="base_nlp", node=NLPConfig)
    cs.store(group="api", name="base_api", node=APIConfig)
    cs.store(group="tracking", name="base_tracking", node=TrackingConfig)
