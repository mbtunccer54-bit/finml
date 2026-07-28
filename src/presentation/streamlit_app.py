"""Analyst dashboard.

Four tabs, matching how a credit review actually proceeds:

1. **Executive Summary** — portfolio shape: the risk matrix, sentiment trend and
   sectoral PD distribution.
2. **Model Benchmark** — which model was promoted and why, with the calibration
   curve and the cost-sensitive confusion matrix.
3. **Deep-Dive XAI** — global drivers, then a single obligor's SHAP waterfall,
   LIME comparison and counterfactual recourse.
4. **Scenario Simulation** — macro scenario plus manual ratio overrides, with
   the PD gauge and explanation updating live.

Scoring, explanation and stress testing all go through the FastAPI service, so
what an analyst sees is exactly what the API served. Portfolio-level visuals
that would mean thousands of round trips read the champion artifact directly;
the sidebar always shows which path is live.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# The dashboard is launched as a script, so the package root may not be on the
# path yet.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from domain.entities import FEATURE_CONTRACT, MACRO_VARIABLES  # noqa: E402
from domain.value_objects import RiskBand  # noqa: E402

API_URL = os.environ.get("FINML_API_URL", "http://localhost:8000")
ARTIFACTS = Path(os.environ.get("FINML_ARTIFACT_DIR", "artifacts"))
REPORTS = ARTIFACTS / "reports"
PRIMARY = "#2E86AB"

BAND_COLOURS = {
    RiskBand.VERY_LOW.value: "#1a9850",
    RiskBand.LOW.value: "#91cf60",
    RiskBand.MEDIUM.value: "#fee08b",
    RiskBand.HIGH.value: "#fc8d59",
    RiskBand.VERY_HIGH.value: "#d73027",
}

st.set_page_config(
    page_title="FinML Credit Risk Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# API and data access
# ---------------------------------------------------------------------------
def api_get(path: str, timeout: float = 30.0) -> tuple[dict[str, Any] | None, str]:
    """GET a JSON payload from the API.

    Args:
        path: Path beneath the API root.
        timeout: Request timeout in seconds.

    Returns:
        The payload and an empty string, or ``None`` and an error message.
    """
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json(), ""
    except httpx.HTTPStatusError as exc:
        return None, f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
    except httpx.HTTPError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def api_post(
    path: str, payload: dict[str, Any], timeout: float = 120.0
) -> tuple[dict[str, Any] | None, str]:
    """POST a JSON payload to the API.

    Args:
        path: Path beneath the API root.
        payload: Request body.
        timeout: Request timeout in seconds.

    Returns:
        The response body and an empty string, or ``None`` and an error message.
    """
    try:
        response = httpx.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json(), ""
    except httpx.HTTPStatusError as exc:
        # 409 carries the drift refusal, which is a result worth showing, not a
        # transport failure to hide.
        return None, f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"
    except httpx.HTTPError as exc:
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(ttl=300, show_spinner=False)
def load_health() -> tuple[dict[str, Any] | None, str]:
    """Fetch service health.

    Returns:
        The health payload and an error message.
    """
    return api_get("/health")


@st.cache_resource(show_spinner="Loading demo portfolio…")
def load_portfolio() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the demo portfolio, macro history and news corpus.

    Returns:
        The panel, macro history and news corpus. Empty frames on failure.
    """
    try:
        from infrastructure.config.loader import load_config, to_object
        from infrastructure.data.fetchers import build_fetcher

        config = to_object(load_config(["data.n_entities=400", "data.n_periods=12"]))
        dataset = build_fetcher(config.data).fetch()
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        st.warning(f"Could not build the demo portfolio: {exc}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    else:
        return dataset.panel, dataset.macro, dataset.news


@st.cache_resource(show_spinner="Loading champion model…")
def load_bundle() -> tuple[Any, Any]:
    """Load the champion bundle and its inference pipeline.

    Returns:
        The pipeline and bundle, or ``(None, None)`` if unavailable.
    """
    try:
        from application.inference_pipeline import InferencePipeline
        from infrastructure.config.loader import load_config, to_object

        config = to_object(load_config())
        pipeline = InferencePipeline.from_champion(config)
    except Exception:  # noqa: BLE001 - the dashboard degrades to API-only
        return None, None
    else:
        return pipeline, pipeline.bundle


@st.cache_data(ttl=300, show_spinner=False)
def load_benchmark() -> pd.DataFrame:
    """Read the benchmark table written by the training pipeline.

    Returns:
        The benchmark table, or an empty frame when absent.
    """
    path = REPORTS / "benchmark.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def entity_payload(row: pd.Series) -> dict[str, Any]:
    """Convert a panel row into an API entity payload.

    Args:
        row: One panel row.

    Returns:
        A payload carrying only contract features the API accepts.
    """
    payload: dict[str, Any] = {"entity_id": str(row.get("entity_id", "unknown"))}
    for spec in FEATURE_CONTRACT.fields:
        if not spec.is_feature or spec.name not in row.index:
            continue
        value = row[spec.name]
        if pd.isna(value):
            continue
        payload[spec.name] = str(value) if spec.allowed else float(value)
    return payload


@st.cache_data(ttl=300, show_spinner="Scoring portfolio…")
def score_portfolio(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """Score a portfolio through the API.

    Args:
        records: Entity payloads.

    Returns:
        The scores, the drift summary and an error message.
    """
    body, error = api_post("/predict", {"entities": records, "check_drift": True})
    if body is None:
        return pd.DataFrame(), None, error
    return pd.DataFrame(body["predictions"]), body.get("drift"), ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> dict[str, Any]:
    """Render the sidebar and return the active selections.

    Returns:
        The selected filters and connection status.
    """
    st.sidebar.title("📊 FinML Platform")
    st.sidebar.caption("Credit risk scoring, explainability and stress testing")

    health, error = load_health()
    if health and health.get("status") == "ok":
        st.sidebar.success(f"API online — `{health['model_id']}`")
        with st.sidebar.expander("Model detail"):
            st.write(
                {
                    "trained_at": health.get("model_trained_at", "")[:19],
                    "calibrated": health.get("is_calibrated"),
                    "features": health.get("n_features"),
                    "drift_monitoring": health.get("drift_monitoring"),
                    "drift_policy": health.get("drift_policy"),
                }
            )
    elif health:
        st.sidebar.warning("API online but no model is promoted")
        st.sidebar.code("make train", language="bash")
    else:
        st.sidebar.error("API unreachable")
        st.sidebar.caption(error[:200])
        st.sidebar.code(f"make serve   # expected at {API_URL}", language="bash")

    st.sidebar.divider()
    panel, _macro, _news = load_portfolio()
    sectors = sorted(panel["sector"].unique()) if not panel.empty else []
    selected = st.sidebar.multiselect("Sectors", sectors, default=sectors)
    max_rows = st.sidebar.slider("Max obligors scored", 50, 500, 200, step=50)

    st.sidebar.divider()
    st.sidebar.caption(
        "Scoring, explanations and scenarios are served by the API. "
        "Portfolio-wide SHAP reads the champion artifact directly."
    )
    return {
        "healthy": bool(health and health.get("status") == "ok"),
        "health": health,
        "sectors": selected,
        "max_rows": max_rows,
    }


# ---------------------------------------------------------------------------
# Tab 1 — Executive summary
# ---------------------------------------------------------------------------
def tab_executive(state: dict[str, Any]) -> None:
    """Render the executive summary tab.

    Args:
        state: Sidebar selections and connection status.
    """
    st.header("Executive Summary")
    panel, macro, _news = load_portfolio()
    if panel.empty:
        st.info("No portfolio available.")
        return

    latest = panel[panel["as_of_date"] == panel["as_of_date"].max()]
    if state["sectors"]:
        latest = latest[latest["sector"].isin(state["sectors"])]
    latest = latest.head(state["max_rows"])

    if not state["healthy"]:
        st.warning("API unavailable — scores cannot be displayed. Start it with `make serve`.")
        return

    scores, drift, error = score_portfolio([entity_payload(r) for _, r in latest.iterrows()])
    if scores.empty:
        st.error(f"Scoring failed: {error}")
        return

    merged = latest.reset_index(drop=True).join(
        scores.set_index("entity_id"), on="entity_id", rsuffix="_score"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Obligors", len(merged))
    c2.metric("Mean PD", f"{merged['probability_of_default'].mean():.2%}")
    c3.metric("Flagged", int(merged["is_flagged"].sum()))
    c4.metric(
        "High / very high",
        int(merged["risk_band"].isin([RiskBand.HIGH.value, RiskBand.VERY_HIGH.value]).sum()),
    )
    severity = (drift or {}).get("severity", "n/a")
    c5.metric("Drift", severity, delta=None if severity == "NONE" else "review")

    if severity not in ("NONE", "n/a"):
        st.warning(
            f"Input drift **{severity}** (max PSI {(drift or {}).get('max_psi', 0):.3f}). "
            f"Drifted: {', '.join((drift or {}).get('drifted_features', [])[:6])}"
        )

    st.subheader("Risk matrix")
    st.caption("Where exposure concentrates: sector against risk band.")
    matrix = (
        merged.pivot_table(index="sector", columns="risk_band", values="entity_id", aggfunc="count")
        .fillna(0)
        .reindex(columns=[b.value for b in RiskBand], fill_value=0)
    )
    heat = px.imshow(
        matrix,
        labels={"x": "Risk band", "y": "Sector", "color": "Obligors"},
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        text_auto=True,
    )
    heat.update_layout(height=420, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(heat, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Sectoral PD distribution")
        box = px.box(
            merged.sort_values("probability_of_default"),
            x="sector",
            y="probability_of_default",
            color="sector",
            points="outliers",
        )
        box.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="PD",
            xaxis_title="",
            margin={"l": 0, "r": 0, "t": 20, "b": 0},
        )
        box.update_xaxes(tickangle=-40)
        st.plotly_chart(box, use_container_width=True)

    with right:
        st.subheader("Sentiment trend")
        st.caption("Mean FinBERT net sentiment across the portfolio, by quarter.")
        trend = (
            panel.groupby("as_of_date")
            .agg(
                sentiment=("sentiment_compound", "mean"),
                uncertainty=("sentiment_uncertainty", "mean"),
                default_rate=("default_flag", "mean"),
            )
            .reset_index()
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=trend["as_of_date"],
                y=trend["sentiment"],
                name="Net sentiment",
                line={"color": PRIMARY, "width": 3},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=trend["as_of_date"],
                y=trend["default_rate"],
                name="Realised default rate",
                line={"color": "#d73027", "width": 2, "dash": "dot"},
                yaxis="y2",
            )
        )
        figure.update_layout(
            height=400,
            yaxis={"title": "Net sentiment"},
            yaxis2={"title": "Default rate", "overlaying": "y", "side": "right"},
            legend={"orientation": "h", "y": -0.2},
            margin={"l": 0, "r": 0, "t": 20, "b": 0},
        )
        st.plotly_chart(figure, use_container_width=True)

    if not macro.empty:
        st.subheader("Macro backdrop")
        long = macro.melt(
            id_vars="as_of_date",
            value_vars=list(MACRO_VARIABLES),
            var_name="variable",
            value_name="value",
        )
        line = px.line(long, x="as_of_date", y="value", color="variable", markers=True)
        line.update_layout(
            height=320,
            margin={"l": 0, "r": 0, "t": 20, "b": 0},
            legend={"orientation": "h", "y": -0.25},
        )
        st.plotly_chart(line, use_container_width=True)

    with st.expander("Scored portfolio"):
        st.dataframe(
            merged[
                [
                    "entity_id",
                    "sector",
                    "probability_of_default",
                    "risk_band",
                    "is_flagged",
                    "credit_score",
                ]
            ].sort_values("probability_of_default", ascending=False),
            use_container_width=True,
            height=340,
        )


# ---------------------------------------------------------------------------
# Tab 2 — Model benchmark
# ---------------------------------------------------------------------------
def tab_benchmark(state: dict[str, Any]) -> None:
    """Render the model benchmark tab.

    Args:
        state: Sidebar selections and connection status.
    """
    st.header("Model Benchmark")
    benchmark = load_benchmark()
    if benchmark.empty:
        st.info("No benchmark table found. Run `make train` to produce one.")
        return

    health = state.get("health") or {}
    champion = health.get("model_id", "")
    if champion:
        st.success(f"Promoted champion: **{champion}**")

    st.caption(
        "Selection order: PR-AUC over ROC-AUC at an 8% base rate, then calibration "
        "error, then business cost at the cost-optimal threshold, then latency."
    )

    split = st.radio("Split", sorted(benchmark["split"].unique()), horizontal=True)
    view = benchmark[benchmark["split"] == split].copy()

    st.dataframe(
        view[
            [
                "model_id",
                "pr_auc",
                "roc_auc",
                "f1",
                "precision",
                "recall",
                "brier_score",
                "expected_calibration_error",
                "business_cost",
                "inference_latency_ms",
                "threshold",
            ]
        ]
        .style.format(
            {
                "pr_auc": "{:.4f}",
                "roc_auc": "{:.4f}",
                "f1": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "brier_score": "{:.4f}",
                "expected_calibration_error": "{:.4f}",
                "business_cost": "{:.1f}",
                "inference_latency_ms": "{:.2f}",
                "threshold": "{:.4f}",
            }
        )
        .background_gradient(subset=["pr_auc"], cmap="Greens")
        .background_gradient(subset=["business_cost", "expected_calibration_error"], cmap="Reds"),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Discrimination vs cost")
        scatter = px.scatter(
            view,
            x="business_cost",
            y="pr_auc",
            text="model_id",
            size="inference_latency_ms",
            color="expected_calibration_error",
            color_continuous_scale="RdYlGn_r",
            labels={
                "business_cost": "Business cost (FN = 10x FP)",
                "pr_auc": "PR-AUC",
                "expected_calibration_error": "ECE",
            },
        )
        scatter.update_traces(textposition="top center")
        scatter.update_layout(height=430, margin={"l": 0, "r": 0, "t": 20, "b": 0})
        st.plotly_chart(scatter, use_container_width=True)
        st.caption("Upper-left is better. Bubble size is latency; colour is calibration error.")

    with right:
        st.subheader("Calibration error by model")
        bar = px.bar(
            view.sort_values("expected_calibration_error"),
            x="expected_calibration_error",
            y="model_id",
            orientation="h",
            color="expected_calibration_error",
            color_continuous_scale="RdYlGn_r",
        )
        bar.add_vline(
            x=0.05, line_dash="dash", line_color="black", annotation_text="deployment gate"
        )
        bar.update_layout(
            height=430, showlegend=False, yaxis_title="", margin={"l": 0, "r": 0, "t": 20, "b": 0}
        )
        st.plotly_chart(bar, use_container_width=True)

    st.subheader("Reliability and cost-sensitive confusion matrix")
    pipeline, _bundle = load_bundle()
    panel, _macro, _news = load_portfolio()
    if pipeline is None or panel.empty:
        st.info("Champion artifact unavailable — reliability diagram needs the local model.")
        return

    sample = panel[panel["as_of_date"] == panel["as_of_date"].max()].head(400)
    try:
        matrix = pipeline.build_matrix(sample)
        probabilities = pipeline.bundle.predict_pd(matrix)
        truth = sample["default_flag"].to_numpy()
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        st.warning(f"Could not score locally: {exc}")
        return

    from infrastructure.models.evaluation import confusion_at, reliability_curve

    curve = reliability_curve(truth, probabilities, n_bins=10)
    threshold = float(pipeline.threshold)

    c1, c2 = st.columns(2)
    with c1:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect",
                line={"dash": "dash", "color": "grey"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=list(curve.mean_predicted),
                y=list(curve.observed_frequency),
                mode="lines+markers",
                name=champion or "champion",
                line={"color": PRIMARY, "width": 3},
            )
        )
        figure.update_layout(
            height=400,
            xaxis_title="Mean predicted PD",
            yaxis_title="Observed default rate",
            title=f"ECE = {curve.expected_calibration_error:.4f}",
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Points below the diagonal mean the model over-predicts risk.")

    with c2:
        cm = confusion_at(truth, probabilities, threshold=threshold)
        costs = np.array([[0.0, 1.0], [10.0, 0.0]])
        weighted = cm * costs
        labels = [
            [f"TN {cm[0, 0]}<br>cost 0", f"FP {cm[0, 1]}<br>cost {weighted[0, 1]:.0f}"],
            [f"FN {cm[1, 0]}<br>cost {weighted[1, 0]:.0f}", f"TP {cm[1, 1]}<br>cost 0"],
        ]
        heat = go.Figure(
            go.Heatmap(
                z=weighted,
                x=["Predicted safe", "Predicted risky"],
                y=["Actually safe", "Actually default"],
                text=labels,
                texttemplate="%{text}",
                colorscale="Reds",
                showscale=False,
            )
        )
        heat.update_layout(
            height=400,
            title=f"Total business cost = {weighted.sum():.0f} @ threshold {threshold:.3f}",
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        st.plotly_chart(heat, use_container_width=True)
        st.caption("Cells are cost-weighted: a false negative counts ten times a false positive.")


# ---------------------------------------------------------------------------
# Tab 3 — Deep-dive XAI
# ---------------------------------------------------------------------------
def tab_xai(state: dict[str, Any]) -> None:
    """Render the explainability tab.

    Args:
        state: Sidebar selections and connection status.
    """
    st.header("Deep-Dive Explainability")
    panel, _macro, _news = load_portfolio()
    pipeline, bundle = load_bundle()
    if panel.empty:
        st.info("No portfolio available.")
        return

    latest = panel[panel["as_of_date"] == panel["as_of_date"].max()]
    if state["sectors"]:
        latest = latest[latest["sector"].isin(state["sectors"])]

    st.subheader("Global drivers")
    importance_path = REPORTS / "shap_global_importance.csv"
    if importance_path.is_file():
        importance = pd.read_csv(importance_path).head(20)
        bar = px.bar(
            importance.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_shap",
            color_continuous_scale="RdBu_r",
            labels={"mean_abs_shap": "Mean |SHAP|", "mean_shap": "Mean SHAP"},
        )
        bar.update_layout(height=560, yaxis_title="", margin={"l": 0, "r": 0, "t": 20, "b": 0})
        st.plotly_chart(bar, use_container_width=True)
        st.caption("Blue lowers modelled risk, red raises it.")
    else:
        st.info("Run `make train` to generate the global SHAP summary.")

    summary_png = REPORTS / "shap_summary.png"
    if summary_png.is_file():
        with st.expander("SHAP beeswarm"):
            st.image(str(summary_png), use_container_width=True)

    if pipeline is not None and bundle is not None and not latest.empty:
        with st.expander("Feature interaction heatmap"):
            st.caption("Pairwise interaction strength among the strongest features.")
            try:
                from infrastructure.xai.shap_explainer import ShapExplainer

                background = bundle.extra.get("drift_reference")
                sample = pipeline.build_matrix(latest.head(120))
                explainer = ShapExplainer(
                    bundle.model,
                    background if background is not None else sample,
                    feature_names=bundle.feature_names,
                )
                interactions = explainer.interaction_matrix(sample, top_k=8)
                figure = px.imshow(
                    interactions, color_continuous_scale="Viridis", aspect="auto", text_auto=".2f"
                )
                figure.update_layout(height=520, margin={"l": 0, "r": 0, "t": 20, "b": 0})
                st.plotly_chart(figure, use_container_width=True)
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                st.warning(f"Interaction values unavailable: {exc}")

    st.divider()
    st.subheader("Single obligor")
    if latest.empty or not state["healthy"]:
        st.info("Select a sector and ensure the API is running.")
        return

    choice = st.selectbox("Obligor", latest["entity_id"].tolist())
    row = latest[latest["entity_id"] == choice].iloc[0]
    payload = entity_payload(row)

    scored, error = api_post("/predict", {"entities": [payload], "check_drift": False})
    if scored is None:
        st.error(f"Scoring failed: {error}")
        return
    prediction = scored["predictions"][0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PD", f"{prediction['probability_of_default']:.2%}")
    m2.metric("Band", prediction["risk_band"])
    m3.metric("Credit score", f"{prediction['credit_score']:.0f}")
    m4.metric("Decision", "FLAGGED" if prediction["is_flagged"] else "PASS")

    shap_tab, lime_tab, dice_tab = st.tabs(["SHAP waterfall", "LIME", "Counterfactuals"])

    with shap_tab:
        body, error = api_post("/explain", {"entity": payload, "method": "shap", "top_k": 12})
        if body is None:
            st.error(error)
        else:
            render_waterfall(body, prediction["probability_of_default"])

    with lime_tab:
        body, error = api_post("/explain", {"entity": payload, "method": "lime", "top_k": 10})
        if body is None:
            st.error(error)
        else:
            frame = pd.DataFrame(body["contributions"])
            figure = px.bar(
                frame.sort_values("contribution"),
                x="contribution",
                y="feature",
                orientation="h",
                color="contribution",
                color_continuous_scale="RdBu_r",
            )
            figure.update_layout(
                height=440, yaxis_title="", margin={"l": 0, "r": 0, "t": 20, "b": 0}
            )
            st.plotly_chart(figure, use_container_width=True)
            st.caption(
                "LIME fits a local linear surrogate. Agreement with SHAP raises "
                "confidence in the reason codes; disagreement usually means the "
                "decision sits on a sharp boundary."
            )

    with dice_tab:
        target = st.slider("Target PD", 0.01, 0.50, 0.20, 0.01)
        body, error = api_post(
            "/explain", {"entity": payload, "method": "dice", "target_pd": target}
        )
        if body is None:
            st.error(error)
        elif not body["counterfactuals"]:
            st.success(f"PD is already at or below the {target:.0%} target — no recourse needed.")
        else:
            for option in body["counterfactuals"]:
                st.markdown(
                    f"**Option {option['option']}** — PD {option['original_pd']:.2%} → "
                    f"**{option['counterfactual_pd']:.2%}** "
                    f"({option['n_changes']} change(s))"
                )
                st.dataframe(pd.DataFrame(option["changes"]), use_container_width=True)


def render_waterfall(body: dict[str, Any], probability: float) -> None:
    """Render a SHAP waterfall for one obligor.

    Args:
        body: The ``/explain`` response.
        probability: The obligor's PD, shown in the title.
    """
    contributions = pd.DataFrame(body["contributions"])
    if contributions.empty:
        st.info("No contributions returned.")
        return

    contributions = contributions.reindex(
        contributions["contribution"].abs().sort_values(ascending=False).index
    )
    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            y=contributions["feature"].tolist()[::-1],
            x=contributions["contribution"].tolist()[::-1],
            measure=["relative"] * len(contributions),
            connector={"line": {"color": "rgba(120,120,120,0.4)"}},
            increasing={"marker": {"color": "#d73027"}},
            decreasing={"marker": {"color": "#1a9850"}},
        )
    )
    figure.update_layout(
        height=520,
        title=f"PD {probability:.2%} · baseline {body['base_value']:.3f} · {body['method']}",
        xaxis_title="Contribution to the model's log-odds output",
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption("Red pushes the obligor towards default; green pulls it away.")


# ---------------------------------------------------------------------------
# Tab 4 — Scenario simulation
# ---------------------------------------------------------------------------
def tab_scenario(state: dict[str, Any]) -> None:
    """Render the scenario simulation tab.

    Args:
        state: Sidebar selections and connection status.
    """
    st.header("Scenario Simulation")
    if not state["healthy"]:
        st.warning("API unavailable — start it with `make serve`.")
        return

    catalogue, error = api_get("/scenarios")
    if catalogue is None:
        st.error(f"Could not load scenarios: {error}")
        return

    names = [s["name"] for s in catalogue["scenarios"]]
    definitions = {s["name"]: s for s in catalogue["scenarios"]}

    panel, _macro, _news = load_portfolio()
    if panel.empty:
        st.info("No portfolio available.")
        return
    latest = panel[panel["as_of_date"] == panel["as_of_date"].max()]
    if state["sectors"]:
        latest = latest[latest["sector"].isin(state["sectors"])]
    latest = latest.head(state["max_rows"])

    st.subheader("Portfolio under stress")
    records = [entity_payload(r) for _, r in latest.iterrows()]
    outcome_body, error = api_post("/scenario", {"entities": records})
    if outcome_body is None:
        st.error(f"Scenario run failed: {error}")
    else:
        outcomes = pd.DataFrame(
            [
                {
                    "scenario": o["scenario"],
                    "severity": o["severity"],
                    "baseline_pd": o["baseline_pd"],
                    "stressed_pd": o["stressed_pd"],
                    "relative_uplift": o["relative_uplift"],
                }
                for o in outcome_body["outcomes"]
            ]
        )
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=outcomes["scenario"],
                y=outcomes["baseline_pd"],
                name="Baseline",
                marker_color="#91cf60",
            )
        )
        figure.add_trace(
            go.Bar(
                x=outcomes["scenario"],
                y=outcomes["stressed_pd"],
                name="Stressed",
                marker_color="#d73027",
            )
        )
        figure.update_layout(
            barmode="group",
            height=380,
            yaxis_title="Mean PD",
            margin={"l": 0, "r": 0, "t": 20, "b": 0},
            legend={"orientation": "h", "y": -0.2},
        )
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(
            outcomes.style.format(
                {"baseline_pd": "{:.4f}", "stressed_pd": "{:.4f}", "relative_uplift": "{:+.2%}"}
            ),
            use_container_width=True,
        )

    with st.expander("Sector transmission matrix (odds multipliers)"):
        st.caption(
            "The bridge laid bare: how much each scenario multiplies default "
            "odds in each sector, before any model is consulted."
        )
        impact = pd.DataFrame(catalogue["sector_impact"])
        figure = px.imshow(impact, color_continuous_scale="Reds", aspect="auto", text_auto=".2f")
        figure.update_layout(height=460, margin={"l": 0, "r": 0, "t": 20, "b": 0})
        st.plotly_chart(figure, use_container_width=True)

    st.divider()
    st.subheader("What-if: single obligor")

    left, right = st.columns([1, 2])
    with left:
        choice = st.selectbox("Obligor", latest["entity_id"].tolist(), key="whatif_entity")
        scenario_name = st.selectbox("Macro scenario", names, key="whatif_scenario")
        st.caption(definitions[scenario_name]["description"])
        shocks = definitions[scenario_name]["shocks"]
        if shocks:
            st.write({k: f"{v:+.3f}" for k, v in shocks.items()})

        row = latest[latest["entity_id"] == choice].iloc[0]
        st.markdown("**Adjust ratios**")
        overrides: dict[str, float] = {}
        adjustable = [
            "current_ratio",
            "quick_ratio",
            "debt_to_equity",
            "debt_to_assets",
            "interest_coverage",
            "return_on_assets",
            "operating_margin",
            "altman_z_score",
        ]
        for name in adjustable:
            spec = FEATURE_CONTRACT.get(name)
            current = float(row[name])
            low = float(spec.ge) if spec.ge is not None else current - 5.0
            high = float(spec.le) if spec.le is not None else current + 5.0
            # Contract bounds are wide by design; a slider spanning them would be
            # unusable, so centre a workable range on the obligor's own value.
            low = max(low, current - max(abs(current) * 2.0, 2.0))
            high = min(high, current + max(abs(current) * 2.0, 2.0))
            overrides[name] = st.slider(
                spec.description[:44],
                low,
                high,
                current,
                step=max((high - low) / 100.0, 1e-4),
                key=f"slider_{name}",
            )

    with right:
        payload = entity_payload(row)
        payload.update(overrides)

        baseline_body, _ = api_post(
            "/predict", {"entities": [entity_payload(row)], "check_drift": False}
        )
        adjusted_body, error = api_post("/predict", {"entities": [payload], "check_drift": False})
        if adjusted_body is None or baseline_body is None:
            st.error(f"Scoring failed: {error}")
            return

        baseline_pd = baseline_body["predictions"][0]["probability_of_default"]
        micro_pd = adjusted_body["predictions"][0]["probability_of_default"]

        multiplier = float(
            pd.DataFrame(catalogue["sector_impact"]).loc[str(row["sector"]), scenario_name]
        )
        odds = micro_pd / max(1.0 - micro_pd, 1e-9) * multiplier
        final_pd = odds / (1.0 + odds)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=final_pd * 100,
                delta={
                    "reference": baseline_pd * 100,
                    "suffix": " pp",
                    "increasing": {"color": "#d73027"},
                },
                number={"suffix": " %", "valueformat": ".2f"},
                title={"text": f"PD under <b>{scenario_name}</b>"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PRIMARY},
                    "steps": [
                        {"range": [0, 1], "color": BAND_COLOURS["very_low"]},
                        {"range": [1, 5], "color": BAND_COLOURS["low"]},
                        {"range": [5, 15], "color": BAND_COLOURS["medium"]},
                        {"range": [15, 30], "color": BAND_COLOURS["high"]},
                        {"range": [30, 100], "color": BAND_COLOURS["very_high"]},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "value": baseline_pd * 100,
                    },
                },
            )
        )
        gauge.update_layout(height=340, margin={"l": 20, "r": 20, "t": 60, "b": 10})
        st.plotly_chart(gauge, use_container_width=True)

        g1, g2, g3 = st.columns(3)
        g1.metric("Baseline PD", f"{baseline_pd:.2%}")
        g2.metric(
            "After ratio changes",
            f"{micro_pd:.2%}",
            delta=f"{micro_pd - baseline_pd:+.2%}",
            delta_color="inverse",
        )
        g3.metric(
            f"+ {scenario_name}",
            f"{final_pd:.2%}",
            delta=f"{final_pd - micro_pd:+.2%}",
            delta_color="inverse",
        )
        st.caption(
            f"Sector `{row['sector']}` transmits this scenario as a "
            f"**{multiplier:.2f}x** multiplier on default odds."
        )

        body, error = api_post("/explain", {"entity": payload, "method": "shap", "top_k": 12})
        if body is not None:
            render_waterfall(body, micro_pd)
        else:
            st.warning(f"Explanation unavailable: {error}")


# ---------------------------------------------------------------------------
def main() -> None:
    """Render the dashboard."""
    state = render_sidebar()
    st.title("FinML Credit Risk Platform")
    st.caption(
        "Probability-of-default scoring with leakage-aware validation, calibrated "
        "probabilities, explainability and CCAR-style stress testing."
    )
    executive, benchmark, xai, scenario = st.tabs(
        ["Executive Summary", "Model Benchmark", "Deep-Dive XAI", "Scenario Simulation"]
    )
    with executive:
        tab_executive(state)
    with benchmark:
        tab_benchmark(state)
    with xai:
        tab_xai(state)
    with scenario:
        tab_scenario(state)


main()
