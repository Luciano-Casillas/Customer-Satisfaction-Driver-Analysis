"""
NovaStar Telecom VOC Survey Dashboard
Author: Luciano Casillas
Version: 1.0
---
This dashboard analyzes 120,000 synthetic post-interaction survey responses
for NovaStar Telecom. It surfaces the key drivers of customer satisfaction
(Net Promoter Score), models Detractor risk using a Gradient Boosting
classifier (AUC 0.81), quantifies CLTV at stake, and delivers prioritized
CX improvement recommendations.

Tab index:
    render_kpi_header       -- persistent KPI row above all tabs
    render_sidebar          -- all filters
    render_overview         -- Tab 1: Overview
    render_cx_drivers       -- Tab 2: CX Drivers
    render_model_risk       -- Tab 3: Model + Risk
    render_financial_impact -- Tab 4: Financial Impact
    render_recommendations  -- Tab 5: Recommendations
    render_healthcare_app   -- Tab 6: Healthcare Application
"""

# ============================================================
# PAGE CONFIG
# ============================================================
import streamlit as st

st.set_page_config(
    page_title="NovaStar Telecom | VOC Dashboard",
    page_icon="ðŸ“Š",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import os

# ============================================================
# COLOR PALETTE
# ============================================================
NAVY       = "#0A3360"
STEEL_700  = "#405E7C"
BLUE_700   = "#0077B3"
BLUE_500   = "#4EBEE5"
STEEL_300  = "#D1E2E5"
STEEL_100  = "#F4F9FA"
WHITE      = "#FFFFFF"
BLACK      = "#0A3360"
GRAY_700   = "#707070"
GRAY_300   = "#CCCCCC"
GREEN_700  = "#08CAA9"
GREEN_900  = "#067462"
ORANGE_700 = "#FF8A39"
RED_SOFT   = "#E05252"

CHART_FONT = dict(family="Inter, Helvetica Neue, sans-serif", size=13, color=STEEL_700)

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main {
    background-color: #FFFFFF !important;
    color: #0A3360 !important;
}
[data-testid="stSidebar"] {
    background-color: #0A3360 !important;
}
.kpi-card {
    background: #FFFFFF;
    border-radius: 6px;
    padding: 12px 16px 10px 16px;
    margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.insight-strip {
    background: #F4F9FA;
    border-left: 4px solid #0077B3;
    border-radius: 4px;
    padding: 12px 18px;
    margin-bottom: 14px;
}
.insight-strip .label {
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #0A3360;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.insight-strip .body {
    font-size: 16px;
    color: #0A3360;
    line-height: 1.55;
}
.section-header {
    background: #F4F9FA;
    border-left: 4px solid #0077B3;
    border-radius: 4px;
    padding: 9px 16px;
    margin: 18px 0 10px 0;
}
.section-header h4 {
    margin: 0;
    font-size: 15px;
    font-weight: 700;
    color: #0A3360;
}
.summary-tile {
    background: #FFFFFF;
    border-left: 4px solid #0077B3;
    border-radius: 6px;
    padding: 14px 16px;
    min-height: 160px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.summary-tile h5 {
    margin: 0 0 8px 0;
    font-size: 13px;
    font-weight: 700;
    color: #0A3360;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.summary-tile p {
    margin: 0;
    font-size: 14px;
    color: #2D2D2D;
    line-height: 1.55;
}
.rec-card {
    background: #FFFFFF;
    border-radius: 6px;
    padding: 16px 18px 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.rec-card h5 {
    margin: 0 0 8px 0;
    font-size: 15px;
    font-weight: 700;
    color: #0A3360;
}
.rec-card .evidence {
    font-size: 13px;
    color: #405E7C;
    border-top: 1px solid #D1E2E5;
    margin-top: 10px;
    padding-top: 8px;
}
.badge {
    display: inline-block;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 8px;
}
.badge-green  { background: #08CAA9; color: #FFFFFF; }
.badge-steel  { background: #405E7C; color: #FFFFFF; }
.badge-orange { background: #FF8A39; color: #FFFFFF; }
.filter-pill {
    display: inline-block;
    background: #D1E2E5;
    color: #0A3360;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px 4px 2px 0;
}
.stat-badge {
    display: inline-block;
    background: #08CAA9;
    color: #FFFFFF;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 14px;
    font-weight: 700;
}
/* Sidebar labels & paragraph text – white on dark background */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
/* Filter input boxes – dark grey */
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div:first-child {
    background-color: #3D3D3D !important;
    border-color: #5A5A5A !important;
}
/* Selected tag pills inside multiselect */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #4D4D4D !important;
}
/* Filter text (tag labels + typed input) – black */
[data-testid="stSidebar"] [data-baseweb="tag"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #FFFFFF !important;
}
/* Sidebar progress bar fill */
[data-testid="stSidebar"] .stProgress > div > div > div > div {
    background-color: #4EBEE5 !important;
}/* ── Filter category labels – white ──────────────────── */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #FFFFFF !important;
}
/* ── Remove orange focus outline on filter boxes ─────── */
[data-testid="stSidebar"] *:focus,
[data-testid="stSidebar"] *:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    box-shadow: none !important;
    border-color: #5A5A5A !important;
}
/* ── Dropdown option list text – white on dark ────────── */
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li {
    color: #FFFFFF !important;
    background-color: #1E3052 !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background-color: #405E7C !important;
}
[data-baseweb="menu"] {
    background-color: #1E3052 !important;
}
/* ── Slider – light grey / silver ────────────────────── */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"],
[data-testid="stSidebar"] [data-baseweb="slider"] [data-baseweb="thumb-value"],
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background-color: #C0C0C0 !important;
    border-color: #A0A0A0 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div:first-child {
    background: linear-gradient(to right, #C0C0C0, #C0C0C0) !important;
}
/* ── Tab labels – steel blue 700 ────────────────────── */
[data-testid="stTabs"] button[role="tab"] {
    color: #405E7C !important;
    font-weight: 600;
}
/* ── KPI column: full-height left border via :has() ─── */
[data-testid="stColumn"]:has(.kpi-card) {
    border-left: 4px solid #0077B3;
    border-radius: 6px;
    padding: 10px 14px 10px 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    background: #FFFFFF;
}
/* ── KPI metric text – override Streamlit muted color ─ */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #0A3360 !important;
}/* ── Expander: keep white background when open ───────── */
/* ── Expander: closed summary always white/STEEL_700 ─── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    background-color: #FFFFFF !important;
    color: #405E7C !important;
}
/* Use details[open] + html/body ancestors to beat Streamlit emotion CSS */
html body details[open][data-testid="stExpander"],
html body details[open][data-testid="stExpander"] > div,
html body details[open][data-testid="stExpander"] > summary {
    background-color: #FFFFFF !important;
    color: #0A3360 !important;
}
html body [data-testid="stExpanderDetails"],
html body [data-testid="stExpanderDetails"] > div {
    background-color: #FFFFFF !important;
    color: #0A3360 !important;
}
/* ── Radio button option text ────────────────────────── */
[data-testid="stRadio"] label span,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] p {
    color: #0A3360 !important;
    background-color: transparent !important;
}
/* ── Main-content widget labels (sliders, radio section) */
[data-testid="stMain"] [data-testid="stWidgetLabel"],
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stWidgetLabel"] label {
    color: #0A3360 !important;
    background-color: transparent !important;
}</style>
""", unsafe_allow_html=True)


# ============================================================
# BASE LAYOUT -- exactly 5 keys, never add more
# ============================================================
def base_layout(height=340):
    return dict(
        height=height,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=CHART_FONT,
        margin=dict(l=16, r=16, t=44, b=44),
    )


def no_grid(fig):
    ax = dict(showgrid=False, zeroline=False,
              tickfont=dict(color=STEEL_700),
              title_font=dict(color=STEEL_700))
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "data", "voc_survey.csv")
    df = pd.read_csv(path, parse_dates=["survey_date"])
    return df


@st.cache_data
def train_model(df):
    FEATURES = [
        "resolution_ease", "agent_professionalism", "wait_time_score",
        "first_contact_resolution", "digital_experience", "billing_clarity",
        "prior_contacts_90d", "days_since_outage", "resolved_flag",
        "escalation_count", "tenure_months", "monthly_revenue",
    ]
    X = df[FEATURES]
    y = df["detractor_flag"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_te, probs)
    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    # Decile assignment: decile 1 = highest risk
    df_te = X_te.copy()
    df_te["model_prob"] = probs
    df_te["actual"] = y_te.values
    df_te["decile"] = pd.qcut(-df_te["model_prob"], 10, labels=False, duplicates="drop") + 1

    overall_rate = y_te.mean()
    d1 = df_te[df_te["decile"] == 1]["actual"].mean()
    lift_d1 = d1 / overall_rate
    top2_gain = df_te[df_te["decile"] <= 2]["actual"].sum() / y_te.sum()

    # Cumulative gain curve
    df_te_sorted = df_te.sort_values("model_prob", ascending=False).reset_index(drop=True)
    df_te_sorted["cum_pos"] = df_te_sorted["actual"].cumsum()
    df_te_sorted["cum_gain"] = df_te_sorted["cum_pos"] / y_te.sum() * 100
    df_te_sorted["pct_pop"] = (df_te_sorted.index + 1) / len(df_te_sorted) * 100

    return {
        "model": model,
        "auc": auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "feature_importance": fi,
        "lift_d1": lift_d1,
        "top2_gain": top2_gain,
        "decile_df": df_te,
        "cum_gain_df": df_te_sorted[["pct_pop", "cum_gain"]],
        "overall_rate": overall_rate,
        "features": FEATURES,
    }


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state(df):
    defaults = {
        "seg_filter":     df["customer_segment"].unique().tolist(),
        "svc_filter":     df["service_type"].unique().tolist(),
        "chan_filter":     df["interaction_channel"].unique().tolist(),
        "region_filter":  df["region"].unique().tolist(),
        "tier_filter":    df["plan_tier"].unique().tolist(),
        "tenure_range":   (int(df["tenure_months"].min()), int(df["tenure_months"].max())),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    return defaults


# ============================================================
# HELPERS
# ============================================================
def insight(label, body):
    sentences = [s.strip() for s in body.split(". ") if s.strip()]
    if len(sentences) > 1:
        items = "".join(
            [f"<li>{s}{'.' if not s.endswith('.') else ''}</li>" for s in sentences]
        )
        body_html = f"<ul style='margin:6px 0 0 0;padding-left:18px;'>{items}</ul>"
    else:
        body_html = body
    st.markdown(
        f'<div class="insight-strip">'
        f'<div class="label">{label}</div>'
        f'<div class="body">{body_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(title):
    st.markdown(
        f'<div class="section-header"><h4>{title}</h4></div>',
        unsafe_allow_html=True,
    )


def section_subtitle(text):
    st.markdown(
        f"<p style='font-size:14px;color:{BLACK};margin:-4px 0 12px 0;'>{text}</p>",
        unsafe_allow_html=True,
    )


def filter_summary_block(dff, df_total):
    n_filt = len(dff)
    n_total = len(df_total)
    pct = n_filt / n_total * 100
    st.markdown(
        f"<p style='font-size:13px;color:{GRAY_700};margin-bottom:4px;'>"
        f"Showing <strong style='color:{NAVY};'>{n_filt:,}</strong> of "
        f"{n_total:,} responses ({pct:.1f}%)</p>",
        unsafe_allow_html=True,
    )


def sparkline(values, color=BLUE_700):
    mn, mx = min(values), max(values)
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,119,179,0.10)",
        )
    )
    fig.update_layout(
        height=52,
        margin=dict(l=0, r=14, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(
            visible=False,
            range=[mn * 0.85, mx * 1.15],
        ),
    )
    return fig


# ============================================================
# APPLY FILTERS
# ============================================================
def apply_filters(df):
    mask = (
        df["customer_segment"].isin(st.session_state["seg_filter"]) &
        df["service_type"].isin(st.session_state["svc_filter"]) &
        df["interaction_channel"].isin(st.session_state["chan_filter"]) &
        df["region"].isin(st.session_state["region_filter"]) &
        df["plan_tier"].isin(st.session_state["tier_filter"]) &
        df["tenure_months"].between(*st.session_state["tenure_range"])
    )
    return df[mask].copy()


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar(df, defaults):
    st.sidebar.markdown(
        f"<h3 style='color:{NAVY};font-size:16px;margin-bottom:4px;'>"
        f"NovaStar Telecom VOC</h3>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"<p style='font-size:12px;color:{GRAY_700};margin-bottom:14px;'>"
        f"Filter all tabs simultaneously</p>",
        unsafe_allow_html=True,
    )

    # Apply reset BEFORE widgets are instantiated so widget keys can be written
    if st.session_state.pop("_reset_filters", False):
        for k, v in defaults.items():
            st.session_state[f"{k}_widget"] = v

    st.session_state["seg_filter"] = st.sidebar.multiselect(
        "Customer Segment",
        options=sorted(df["customer_segment"].unique()),
        default=st.session_state["seg_filter"],
        key="seg_filter_widget",
    )
    st.session_state["svc_filter"] = st.sidebar.multiselect(
        "Service Type",
        options=sorted(df["service_type"].unique()),
        default=st.session_state["svc_filter"],
        key="svc_filter_widget",
    )
    st.session_state["chan_filter"] = st.sidebar.multiselect(
        "Interaction Channel",
        options=sorted(df["interaction_channel"].unique()),
        default=st.session_state["chan_filter"],
        key="chan_filter_widget",
    )
    st.session_state["region_filter"] = st.sidebar.multiselect(
        "Region",
        options=sorted(df["region"].unique()),
        default=st.session_state["region_filter"],
        key="region_filter_widget",
    )
    st.session_state["tier_filter"] = st.sidebar.multiselect(
        "Plan Tier",
        options=sorted(df["plan_tier"].unique()),
        default=st.session_state["tier_filter"],
        key="tier_filter_widget",
    )
    st.session_state["tenure_range"] = st.sidebar.slider(
        "Tenure (months)",
        min_value=int(df["tenure_months"].min()),
        max_value=int(df["tenure_months"].max()),
        value=st.session_state["tenure_range"],
        key="tenure_range_widget",
    )

    if st.sidebar.button("Reset All Filters"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state["_reset_filters"] = True
        st.rerun()


    n = sum(
        df["customer_segment"].isin(st.session_state["seg_filter"]) &
        df["service_type"].isin(st.session_state["svc_filter"]) &
        df["interaction_channel"].isin(st.session_state["chan_filter"]) &
        df["region"].isin(st.session_state["region_filter"]) &
        df["plan_tier"].isin(st.session_state["tier_filter"]) &
        df["tenure_months"].between(*st.session_state["tenure_range"])
    )
    total = len(df)
    st.sidebar.progress(n / total)
    st.sidebar.markdown(
        f"<p style='font-size:12px;color:{GRAY_700};'>{n:,} of {total:,} responses</p>",
        unsafe_allow_html=True,
    )


# ============================================================
# KPI HEADER
# ============================================================
def render_kpi_header(dff):
    det_rate = dff["detractor_flag"].mean() * 100
    avg_nps = dff["nps_score"].mean()
    total_cltv_at_risk = dff[dff["detractor_flag"] == 1]["annual_cltv"].sum() / 1e6
    fcr_rate = dff["resolved_flag"].mean() * 100

    # Sparkline data: monthly detractor rate trend
    dff_spark = dff.copy()
    dff_spark["month"] = dff_spark["survey_date"].dt.to_period("M").astype(str)
    monthly = dff_spark.groupby("month")["detractor_flag"].mean().reset_index().sort_values("month")
    nps_monthly = dff_spark.groupby("month")["nps_score"].mean().reset_index().sort_values("month")
    cltv_monthly = (
        dff_spark[dff_spark["detractor_flag"] == 1]
        .groupby("month")["annual_cltv"].sum().reset_index().sort_values("month")
    )
    fcr_monthly = dff_spark.groupby("month")["resolved_flag"].mean().reset_index().sort_values("month")

    metrics = [
        ("Detractor Rate (NPS)", f"{det_rate:.1f}%", monthly["detractor_flag"].tolist(), RED_SOFT),
        ("Avg NPS Score", f"{avg_nps:.2f}", nps_monthly["nps_score"].tolist(), BLUE_700),
        ("CLTV at Risk ($M)", f"${total_cltv_at_risk:.1f}M", cltv_monthly["annual_cltv"].tolist(), ORANGE_700),
        ("First Contact Resolution (FCR)", f"{fcr_rate:.1f}%", fcr_monthly["resolved_flag"].tolist(), GREEN_700),
    ]

    cols = st.columns(4)
    for col, (label, value, spark_vals, color) in zip(cols, metrics):
        with col:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.metric(label=label, value=value)
            with c2:
                if len(spark_vals) >= 2:
                    st.plotly_chart(
                        sparkline(spark_vals, color=color),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
            st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================
def render_overview(dff):
    det_rate = dff["detractor_flag"].mean() * 100
    total_det = dff["detractor_flag"].sum()
    det_cltv = dff[dff["detractor_flag"] == 1]["annual_cltv"].sum() / 1e6

    insight(
        "KEY FINDING",
        f"{det_rate:.1f}% of surveyed customers are NPS Detractors (NPS 0-6), "
        f"representing {total_det:,} responses and ${det_cltv:.1f}M in annual CLTV at risk. "
        f"Resolution ease is the single strongest predictor of whether a customer "
        f"scores in the Detractor range."
    )

    col1, col2 = st.columns(2)

    # Donut: NPS category breakdown
    with col1:
        cat_counts = dff["nps_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_order = ["Detractor", "Passive", "Promoter"]
        cat_counts["Category"] = pd.Categorical(cat_counts["Category"], categories=cat_order, ordered=True)
        cat_counts = cat_counts.sort_values("Category")
        colors_donut = [RED_SOFT, STEEL_300, BLUE_700]
        fig = go.Figure(go.Pie(
            labels=cat_counts["Category"],
            values=cat_counts["Count"],
            hole=0.52,
            marker=dict(colors=colors_donut),
            textinfo="percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value:,} responses (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            **base_layout(height=320),
            title=dict(text="NPS Category Distribution", font=dict(size=14, color=STEEL_700), x=0.5),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22, font=dict(color=STEEL_700)),
        )
        no_grid(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Bar: Detractor rate by interaction reason
    with col2:
        reason_det = (
            dff.groupby("interaction_reason")["detractor_flag"]
            .agg(["mean", "count"])
            .reset_index()
        )
        reason_det.columns = ["Reason", "Rate", "Count"]
        reason_det["Rate"] *= 100
        reason_det = reason_det.sort_values("Rate", ascending=True)
        bar_colors = [RED_SOFT if r >= det_rate else BLUE_700 for r in reason_det["Rate"]]
        fig2 = go.Figure(go.Bar(
            x=reason_det["Rate"],
            y=reason_det["Reason"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{r:.1f}%" for r in reason_det["Rate"]],
            textposition="outside",
            textfont=dict(size=12, color=STEEL_700),
            hovertemplate="<b>%{y}</b><br>Detractor Rate: %{x:.2f}%<br>Responses: %{customdata:,}<extra></extra>",
            customdata=reason_det["Count"],
        ))
        fig2.update_layout(
            **base_layout(height=320),
            title=dict(text="Detractor Rate by Interaction Reason", font=dict(size=14, color=STEEL_700), x=0.5),
            xaxis=dict(title="Detractor Rate (%)", ticksuffix="%"),
            yaxis=dict(title=""),
        )
        no_grid(fig2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with st.expander("About these charts"):
        st.markdown(
            f"<p style='font-size:14px;color:{BLACK};'>"
            f"The donut chart shows how surveyed customers split across the three Net Promoter Score (NPS) categories: "
            f"Detractors (score 0-6, shown in red), Passives (7-8, gray), and Promoters (9-10, blue). "
            f"The bar chart shows the Detractor rate for each reason a customer contacted NovaStar. "
            f"Bars in red exceed the overall average Detractor rate; bars in blue are below average. "
            f"Interaction Reason is the primary segmentation lens for identifying high-friction contact types."
            f"</p>",
            unsafe_allow_html=True,
        )

    # Summary tiles
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    section_header("Key Takeaways")
    t1, t2, t3 = st.columns(3)

    top_reason = (
        dff.groupby("interaction_reason")["detractor_flag"].mean().idxmax()
    )
    top_reason_rate = dff.groupby("interaction_reason")["detractor_flag"].mean().max() * 100

    with t1:
        st.markdown(
            f'<div class="summary-tile">'
            f'<h5>Top Risk Drivers</h5>'
            f'<p>Resolution ease (30.3% importance), wait time (19.7%), and first contact resolution (FCR, 19.1%) '
            f'account for over 69% of the Detractor prediction model. These three levers are the primary focus for CX teams.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            f'<div class="summary-tile">'
            f'<h5>Model Summary</h5>'
            f'<p>Gradient Boosting classifier trained on 12 survey and behavioral features. '
            f'AUC: 0.81. Top decile lift: 1.82x. Top-2 decile cumulative gain: 36.3%. '
            f'Identifies the highest-risk respondents with strong precision for targeted outreach.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            f'<div class="summary-tile">'
            f'<h5>Healthcare Connection</h5>'
            f'<p>The same driver framework applies directly to patient experience (HCAHPS) analysis. '
            f"Resolution ease maps to 'care coordination'; wait time maps to 'response to call'; "
            f'FCR maps to discharge planning completeness. See the Healthcare Application tab for the full translation.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 2: CX DRIVERS
# ============================================================
def render_cx_drivers(dff):
    overall_det = dff["detractor_flag"].mean() * 100

    insight(
        "KEY FINDING",
        f"Call Center interactions produce a {(dff[dff['interaction_channel']=='Call Center']['detractor_flag'].mean()*100):.1f}% "
        f"Detractor rate -- well above the {overall_det:.1f}% overall average. "
        f"Detractors score resolution ease {dff[dff['detractor_flag']==1]['resolution_ease'].mean():.2f}/10 "
        f"versus {dff[dff['detractor_flag']==0]['resolution_ease'].mean():.2f}/10 for non-Detractors, "
        f"a gap of {(dff[dff['detractor_flag']==0]['resolution_ease'].mean()-dff[dff['detractor_flag']==1]['resolution_ease'].mean()):.2f} points."
    )

    section_header("Driver Score Gaps: Detractor vs. Promoter Populations")
    section_subtitle(
        "Each chart compares average driver scores between NPS Detractors and Promoters. "
        "Larger gaps indicate where process improvement would generate the most NPS lift. "
        "Scores are on a 1-10 scale where 10 = highest satisfaction."
    )

    # Row 1: Resolution Ease and Wait Time by NPS category
    driver_summary = (
        dff.groupby("nps_category")[
            ["resolution_ease", "wait_time_score", "first_contact_resolution", "agent_professionalism"]
        ].mean()
        .reset_index()
    )
    cat_order = ["Detractor", "Passive", "Promoter"]
    driver_summary["nps_category"] = pd.Categorical(driver_summary["nps_category"], categories=cat_order, ordered=True)
    driver_summary = driver_summary.sort_values("nps_category")

    shared_max_r1 = max(
        driver_summary["resolution_ease"].max(),
        driver_summary["wait_time_score"].max()
    ) * 1.25

    cat_colors = [RED_SOFT, STEEL_300, BLUE_700]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for i, (cat, color) in enumerate(zip(cat_order, cat_colors)):
            row = driver_summary[driver_summary["nps_category"] == cat]
            fig.add_trace(go.Bar(
                name=cat,
                x=[cat],
                y=row["resolution_ease"].values,
                marker_color=color,
                text=[f"{row['resolution_ease'].values[0]:.2f}"],
                textposition="outside",
                textfont=dict(size=12, color=STEEL_700),
                hovertemplate=f"<b>{cat}</b><br>Resolution Ease: %{{y:.2f}}/10<extra></extra>",
            ))
        fig.update_layout(
            **base_layout(),
            title=dict(text="Resolution Ease by NPS Category", font=dict(size=14, color=STEEL_700), x=0.5),
            yaxis=dict(title="Avg Score (1-10)", range=[0, shared_max_r1]),
            showlegend=False,
        )
        no_grid(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig2 = go.Figure()
        for cat, color in zip(cat_order, cat_colors):
            row = driver_summary[driver_summary["nps_category"] == cat]
            fig2.add_trace(go.Bar(
                name=cat,
                x=[cat],
                y=row["wait_time_score"].values,
                marker_color=color,
                text=[f"{row['wait_time_score'].values[0]:.2f}"],
                textposition="outside",
                textfont=dict(size=12, color=STEEL_700),
                hovertemplate=f"<b>{cat}</b><br>Wait Time Score: %{{y:.2f}}/10<extra></extra>",
            ))
        fig2.update_layout(
            **base_layout(),
            title=dict(text="Wait Time Score by NPS Category", font=dict(size=14, color=STEEL_700), x=0.5),
            yaxis=dict(title="Avg Score (1-10)", range=[0, shared_max_r1]),
            showlegend=False,
        )
        no_grid(fig2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Row 2: FCR and Agent Professionalism
    shared_max_r2 = max(
        driver_summary["first_contact_resolution"].max(),
        driver_summary["agent_professionalism"].max()
    ) * 1.25

    col3, col4 = st.columns(2)
    with col3:
        fig3 = go.Figure()
        for cat, color in zip(cat_order, cat_colors):
            row = driver_summary[driver_summary["nps_category"] == cat]
            fig3.add_trace(go.Bar(
                name=cat,
                x=[cat],
                y=row["first_contact_resolution"].values,
                marker_color=color,
                text=[f"{row['first_contact_resolution'].values[0]:.2f}"],
                textposition="outside",
                textfont=dict(size=12, color=STEEL_700),
                hovertemplate=f"<b>{cat}</b><br>FCR Score: %{{y:.2f}}/10<extra></extra>",
            ))
        fig3.update_layout(
            **base_layout(),
            title=dict(text="First Contact Resolution (FCR) Score by NPS Category", font=dict(size=14, color=STEEL_700), x=0.5),
            yaxis=dict(title="Avg Score (1-10)", range=[0, shared_max_r2]),
            showlegend=False,
        )
        no_grid(fig3)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col4:
        fig4 = go.Figure()
        for cat, color in zip(cat_order, cat_colors):
            row = driver_summary[driver_summary["nps_category"] == cat]
            fig4.add_trace(go.Bar(
                name=cat,
                x=[cat],
                y=row["agent_professionalism"].values,
                marker_color=color,
                text=[f"{row['agent_professionalism'].values[0]:.2f}"],
                textposition="outside",
                textfont=dict(size=12, color=STEEL_700),
                hovertemplate=f"<b>{cat}</b><br>Agent Professionalism: %{{y:.2f}}/10<extra></extra>",
            ))
        fig4.update_layout(
            **base_layout(),
            title=dict(text="Agent Professionalism by NPS Category", font=dict(size=14, color=STEEL_700), x=0.5),
            yaxis=dict(title="Avg Score (1-10)", range=[0, shared_max_r2]),
            showlegend=False,
        )
        no_grid(fig4)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    section_header("Segment, Channel, and Repeat Contact Analysis")
    insight(
        "KEY FINDING",
        f"Customers who contacted NovaStar 3 or more times in 90 days have a Detractor rate of "
        f"{dff[dff['prior_contacts_90d']>=3]['detractor_flag'].mean()*100:.1f}% vs. "
        f"{dff[dff['prior_contacts_90d']==0]['detractor_flag'].mean()*100:.1f}% for first-time contacts. "
        f"Repeat contact is the most actionable friction signal and the foundation of the FCR improvement program."
    )

    # Row 3: Channel detractor rate (dual-axis), Repeat contact impact
    channel_det = (
        dff.groupby("interaction_channel")
        .agg(
            det_rate=("detractor_flag", "mean"),
            count=("detractor_flag", "count"),
            avg_cltv=("annual_cltv", "mean"),
        )
        .reset_index()
    )
    channel_det["det_rate"] *= 100
    channel_det = channel_det.sort_values("det_rate", ascending=False)

    contact_det = (
        dff.groupby("prior_contacts_90d")["detractor_flag"].mean().reset_index()
    )
    contact_det.columns = ["contacts", "det_rate"]
    contact_det["det_rate"] *= 100
    contact_det = contact_det[contact_det["contacts"] <= 8]

    col5, col6 = st.columns(2)
    with col5:
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=channel_det["interaction_channel"],
            y=channel_det["det_rate"],
            name="Detractor Rate",
            marker_color=BLUE_700,
            text=[f"{r:.1f}%" for r in channel_det["det_rate"]],
            textposition="outside",
            textfont=dict(size=12, color=STEEL_700),
            yaxis="y",
            hovertemplate="<b>%{x}</b><br>Detractor Rate: %{y:.2f}%<extra></extra>",
        ))
        fig5.add_trace(go.Scatter(
            x=channel_det["interaction_channel"],
            y=channel_det["avg_cltv"],
            name="Avg CLTV",
            mode="lines+markers+text",
            line=dict(color=ORANGE_700, width=2),
            marker=dict(size=8),
            text=[f"${v:.0f}" for v in channel_det["avg_cltv"]],
            textposition="top center",
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Avg CLTV: $%{y:.0f}<extra></extra>",
        ))
        fig5.update_layout(
            **base_layout(height=360),
            title=dict(text="Detractor Rate and Avg CLTV by Channel", font=dict(size=14, color=STEEL_700), x=0.5),
            yaxis=dict(title="Detractor Rate (%)", ticksuffix="%"),
            yaxis2=dict(title="Avg Annual CLTV ($)", overlaying="y", side="right", range=[0, 974]),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22, font=dict(color=STEEL_700)),
        )
        no_grid(fig5)
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    with col6:
        fig6 = go.Figure(go.Bar(
            x=contact_det["contacts"],
            y=contact_det["det_rate"],
            marker_color=[RED_SOFT if c >= 3 else BLUE_700 for c in contact_det["contacts"]],
            text=[f"{r:.1f}%" for r in contact_det["det_rate"]],
            textposition="outside",
            textfont=dict(size=12, color=STEEL_700),
            hovertemplate="<b>%{x} prior contacts</b><br>Detractor Rate: %{y:.2f}%<extra></extra>",
        ))
        fig6.update_layout(
            **base_layout(height=360),
            title=dict(text="Detractor Rate by Prior Contacts (90 Days)", font=dict(size=14, color=STEEL_700), x=0.5),
            xaxis=dict(title="Number of Prior Contacts", dtick=1),
            yaxis=dict(title="Detractor Rate (%)", ticksuffix="%"),
        )
        no_grid(fig6)
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})

    with st.expander("About these charts"):
        st.markdown(
            f"<p style='font-size:14px;color:{BLACK};'>"
            f"The first four charts compare average driver scores (1-10 scale) across NPS categories: "
            f"Detractors (red, NPS 0-6), Passives (gray, 7-8), and Promoters (blue, 9-10). "
            f"Larger gaps between Detractor and Promoter bars indicate higher-priority improvement levers. "
            f"First Contact Resolution (FCR) score reflects a customer's rating of whether their issue "
            f"was resolved without needing a follow-up contact. "
            f"The dual-axis channel chart shows Detractor rate on the left y-axis (bars) and average annual "
            f"Customer Lifetime Value (CLTV) on the right y-axis (line). "
            f"The repeat contact chart shows that Detractor risk rises sharply above 3 prior contacts; "
            f"bars in red indicate contacts in the friction threshold range."
            f"</p>",
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 3: MODEL + RISK
# ============================================================
def render_model_risk(dff, model_results):
    auc = model_results["auc"]
    lift_d1 = model_results["lift_d1"]
    top2_gain = model_results["top2_gain"]
    tn = model_results["tn"]
    fp = model_results["fp"]
    fn = model_results["fn"]
    tp = model_results["tp"]
    fi = model_results["feature_importance"]
    decile_df = model_results["decile_df"]
    cum_df = model_results["cum_gain_df"]

    insight(
        "MODEL PERFORMANCE",
        f"Gradient Boosting classifier trained on 12 CX survey and behavioral features. "
        f"AUC: {auc:.2f}. "
        f"Top decile lift: {lift_d1:.2f}x -- meaning the top 10% of highest-risk customers "
        f"are {lift_d1:.2f}x more likely to be Detractors than the overall population. "
        f"Top-2 decile cumulative gain: {top2_gain*100:.1f}% -- the top 20% of scored customers "
        f"contain {top2_gain*100:.1f}% of all Detractors in the dataset."
    )

    section_header("Decile Lift and Cumulative Gain")
    section_subtitle(
        "Decile 1 contains the 10% of customers with the highest predicted Detractor probability. "
        "The lift chart shows how much more likely Detractors are to appear in each decile "
        "compared to a random selection. The cumulative gain curve shows the share of all "
        "Detractors captured as you contact more of the scored population."
    )

    # Lift by decile
    decile_rates = decile_df.groupby("decile")["actual"].mean().reset_index()
    decile_rates.columns = ["decile", "det_rate"]
    overall_rate = model_results["overall_rate"]
    decile_rates["lift"] = decile_rates["det_rate"] / overall_rate

    col1, col2 = st.columns(2)
    with col1:
        bar_colors_lift = [GREEN_700 if d == 1 else BLUE_700 if d <= 3 else STEEL_300 for d in decile_rates["decile"]]
        fig = go.Figure(go.Bar(
            x=decile_rates["decile"],
            y=decile_rates["lift"],
            marker_color=bar_colors_lift,
            text=[f"{l:.2f}x" for l in decile_rates["lift"]],
            textposition="outside",
            textfont=dict(size=12, color=STEEL_700),
            hovertemplate="<b>Decile %{x}</b><br>Lift: %{y:.2f}x<br>Det Rate: %{customdata:.1f}%<extra></extra>",
            customdata=decile_rates["det_rate"] * 100,
        ))
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color=GRAY_300,
            annotation_text="Baseline (1.0x)",
            annotation_position="top right",
        )
        fig.update_layout(
            **base_layout(),
            title=dict(text="Lift by Decile (Decile 1 = Highest Risk)", font=dict(size=14, color=STEEL_700), x=0.5),
            xaxis=dict(title="Risk Decile", dtick=1),
            yaxis=dict(title="Lift vs. Baseline"),
        )
        no_grid(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=cum_df["pct_pop"],
            y=cum_df["cum_gain"],
            mode="lines",
            line=dict(color=BLUE_700, width=2.5),
            name="Model",
            hovertemplate="Contact top %{x:.1f}% -- captures %{y:.1f}% of Detractors<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode="lines",
            line=dict(color=GRAY_300, dash="dash", width=1.5),
            name="Random Baseline",
            hoverinfo="skip",
        ))
        # Top-2 decile annotation
        fig2.add_annotation(
            x=20, y=top2_gain * 100 + 4,
            text=f"Top 20%: {top2_gain*100:.1f}% gain",
            showarrow=True,
            arrowhead=2,
            arrowcolor=NAVY,
            font=dict(size=12, color=STEEL_700),
            bgcolor=STEEL_100,
        )
        fig2.update_layout(
            **base_layout(),
            title=dict(text="Cumulative Gain Curve", font=dict(size=14, color=STEEL_700), x=0.5),
            xaxis=dict(title="% of Customers Contacted", ticksuffix="%"),
            yaxis=dict(title="% of Detractors Captured", ticksuffix="%"),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22),
        )
        no_grid(fig2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    section_header("Feature Importance and Model Accuracy")
    section_subtitle(
        "Feature importance shows which input variables contributed most to model predictions. "
        "Higher values indicate the model relied more heavily on that feature. "
        "The confusion matrix compares actual versus predicted Detractor labels on the holdout test set."
    )

    col3, col4 = st.columns(2)
    with col3:
        fi_df = fi.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        label_map = {
            "resolution_ease": "Resolution Ease",
            "wait_time_score": "Wait Time Score",
            "first_contact_resolution": "First Contact Resolution",
            "resolved_flag": "Resolved Flag",
            "agent_professionalism": "Agent Professionalism",
            "digital_experience": "Digital Experience",
            "escalation_count": "Escalation Count",
            "billing_clarity": "Billing Clarity",
            "prior_contacts_90d": "Prior Contacts (90d)",
            "days_since_outage": "Days Since Outage",
            "monthly_revenue": "Monthly Revenue",
            "tenure_months": "Tenure (Months)",
        }
        fi_df["Feature"] = fi_df["Feature"].map(label_map)
        fi_df = fi_df.sort_values("Importance")
        fig3 = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker_color=BLUE_700,
            text=[f"{v:.3f}" for v in fi_df["Importance"]],
            textposition="outside",
            textfont=dict(size=11, color=STEEL_700),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig3.update_layout(
            **base_layout(height=380),
            title=dict(text="Feature Importance (Gradient Boosting)", font=dict(size=14, color=STEEL_700), x=0.5),
            xaxis=dict(title="Relative Importance"),
            yaxis=dict(title=""),
        )
        no_grid(fig3)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col4:
        # Confusion matrix: TN top-left, FP top-right, FN bottom-left, TP bottom-right
        z = [[tn, fp], [fn, tp]]
        z_text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
        cell_colors = [
            [NAVY, RED_SOFT],
            [RED_SOFT, NAVY],
        ]
        font_colors = [["white", "white"], ["white", "white"]]
        fig4 = go.Figure(go.Heatmap(
            z=z,
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorscale=[
                [0.0, RED_SOFT], [0.33, RED_SOFT],
                [0.33, NAVY], [1.0, NAVY],
            ],
            showscale=False,
            xgap=3, ygap=3,
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))
        fig4.update_layout(
            **base_layout(height=380),
            title=dict(text="Confusion Matrix (Holdout Set)", font=dict(size=14, color=STEEL_700), x=0.5),
            xaxis=dict(
                tickvals=[0, 1],
                ticktext=["Predicted: Non-Detractor", "Predicted: Detractor"],
                side="bottom",
            ),
            yaxis=dict(
                tickvals=[0, 1],
                ticktext=["Actual: Non-Detractor", "Actual: Detractor"],
                autorange="reversed",
            ),
        )
        no_grid(fig4)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"<p style='font-size:13px;color:{BLACK};margin-top:4px;'>"
            f"True Positives: <strong>{tp:,}</strong> | "
            f"True Negatives: <strong>{tn:,}</strong> | "
            f"False Positives: <strong>{fp:,}</strong> | "
            f"False Negatives: <strong>{fn:,}</strong>"
            f"</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<hr style='border:none;border-top:1.5px dotted {GRAY_300};margin:18px 0;'>",
        unsafe_allow_html=True,
    )

    section_header("At-Risk Customer Profile Explorer")
    section_subtitle(
        "Select a decile range to explore the profile of customers in that risk tier. "
        "Decile 1 contains the 10% of survey respondents with the highest predicted Detractor probability."
    )

    decile_choice = st.radio(
        "Select Decile Range",
        ["Decile 1 (Highest Risk)", "Deciles 1-2", "Deciles 1-3"],
        horizontal=True,
    )
    decile_map = {"Decile 1 (Highest Risk)": 1, "Deciles 1-2": 2, "Deciles 1-3": 3}
    max_dec = decile_map[decile_choice]

    det_pool = decile_df[decile_df["decile"] <= max_dec]

    profile_count = len(det_pool)
    avg_cltv_at_risk = dff[dff["detractor_flag"] == 1]["annual_cltv"].iloc[: len(det_pool)].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Respondents in Range", f"{profile_count:,}")
    m2.metric("Avg CLTV (Detractor Pool)", f"${avg_cltv_at_risk:.0f}")
    m3.metric("Avg Resolution Ease", f"{det_pool['resolution_ease'].mean():.2f}/10")
    m4.metric("Avg FCR Score", f"{det_pool['first_contact_resolution'].mean():.2f}/10")

    insight(
        "AT-RISK PROFILE",
        f"The top {max_dec} decile(s) contain {profile_count:,} respondents with the highest predicted Detractor risk. "
        f"This group averages {det_pool['resolution_ease'].mean():.2f}/10 on resolution ease and "
        f"{det_pool['first_contact_resolution'].mean():.2f}/10 on first contact resolution -- "
        f"both well below the overall dataset averages."
    )

    with st.expander("About these charts"):
        st.markdown(
            f"<p style='font-size:14px;color:{BLACK};'>"
            f"The lift chart shows how much more likely the model is to find Detractors in each risk decile "
            f"compared to a random draw. Decile 1 = highest risk (1.82x lift means these customers are "
            f"1.82x more likely to be Detractors than average). "
            f"The cumulative gain curve shows what percentage of all Detractors you have identified "
            f"by the time you contact the top X% of the ranked population. "
            f"Area Under the Curve (AUC) of {auc:.2f} means the model correctly ranks a random Detractor "
            f"above a random non-Detractor {auc*100:.0f}% of the time. "
            f"The confusion matrix uses the holdout test set (20% of data). "
            f"True Positives and True Negatives are shown in navy; False Positives and False Negatives in red."
            f"</p>",
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 4: FINANCIAL IMPACT
# ============================================================
def render_financial_impact(dff):
    total_det = dff["detractor_flag"].sum()
    total_cltv_at_risk = dff[dff["detractor_flag"] == 1]["annual_cltv"].sum()
    avg_cltv_det = total_cltv_at_risk / total_det if total_det > 0 else 0

    insight(
        "FINANCIAL OPPORTUNITY",
        f"${total_cltv_at_risk/1e6:.1f}M in annual CLTV is at risk from {total_det:,} Detractor respondents. "
        f"A 20% save rate on the top-risk population would protect an estimated "
        f"${total_cltv_at_risk * 0.20 / 1e6:.1f}M in forward revenue before accounting for outreach costs."
    )

    section_header("CX Intervention Revenue Simulator")
    section_subtitle(
        "Use the sliders to model different outreach scenarios. "
        "Save Rate controls what percentage of Detractors improve to Passive or Promoter status after a recovery intervention. "
        "Cost per Contact is the fully loaded cost of one personalized outreach attempt."
    )

    with st.expander("About the simulator"):
        st.markdown(
            f"<p style='font-size:14px;color:{BLACK};'>"
            f"Customer Lifetime Value (CLTV) is estimated as 12 months of forward revenue per account, "
            f"adjusted for tenure. The simulator multiplies total Detractor CLTV by the save rate to "
            f"estimate revenue protected, then subtracts total contact costs. "
            f"This models a simple one-period recovery scenario; a multi-period model would compound "
            f"retention value across additional renewal cycles."
            f"</p>",
            unsafe_allow_html=True,
        )

    save_rates = [5, 10, 15, 20, 25, 30, 35, 40]

    sim_col, chart_col = st.columns([1, 1])
    with sim_col:
        save_rate = st.slider("Save Rate (%)", min_value=5, max_value=40, value=20, step=5)
        cost_per_contact = st.slider(
            "Cost per Contact ($)", min_value=10, max_value=80, value=25, step=5
        )
        revenue_protected = total_cltv_at_risk * save_rate / 100
        total_contact_cost = total_det * cost_per_contact
        net_value = revenue_protected - total_contact_cost
        roi = (net_value / total_contact_cost * 100) if total_contact_cost > 0 else 0

        net_color = GREEN_900 if net_value >= 0 else RED_SOFT
        roi_color = GREEN_900 if roi >= 0 else RED_SOFT
        st.markdown(
            f"<div style='background:{WHITE};border-left:4px solid {GREEN_700};border-radius:6px;"
            f"padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.07);margin-top:12px;'>"
            f"<p style='font-size:16px;line-height:2.8;color:{BLACK};margin:0;'>"
            f"Revenue Protected: <strong style='color:{GREEN_900};'>${revenue_protected/1e6:.2f}M</strong><br>"
            f"Contact Cost: <strong style='color:{ORANGE_700};'>${total_contact_cost/1e6:.2f}M</strong><br>"
            f"Net Value: <strong style='color:{net_color};'>${net_value/1e6:.2f}M</strong><br>"
            f"ROI: <strong style='color:{roi_color};'>{roi:.0f}%</strong>"
            f"</p></div>",
            unsafe_allow_html=True,
        )

    with chart_col:
        scenario_vals = []
        for sr in save_rates:
            rev = total_cltv_at_risk * sr / 100
            cost = total_det * cost_per_contact
            net = rev - cost
            scenario_vals.append({"save_rate": sr, "net_value": net / 1e6})

        sv_df = pd.DataFrame(scenario_vals)
        bar_colors_sim = [
            GREEN_700 if row["save_rate"] == save_rate
            else NAVY if row["save_rate"] < save_rate
            else BLUE_500
            for _, row in sv_df.iterrows()
        ]

        fig_sim = go.Figure(go.Bar(
            x=[f"{r}%" for r in sv_df["save_rate"]],
            y=sv_df["net_value"],
            marker_color=bar_colors_sim,
            text=[f"${v:.1f}M" for v in sv_df["net_value"]],
            textposition="outside",
            textfont=dict(size=12, color=STEEL_700),
            hovertemplate="Save Rate: %{x}<br>Net Value: $%{y:.2f}M<extra></extra>",
        ))
        fig_sim.update_layout(
            **base_layout(height=340),
            title=dict(
                text="Net Revenue Protected after Outreach Costs by Save Rate",
                font=dict(size=13, color=STEEL_700), x=0.5,
            ),
            xaxis=dict(title="Save Rate"),
            yaxis=dict(title="Net Value ($M)", tickprefix="$", ticksuffix="M"),
        )
        no_grid(fig_sim)
        st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<p style='font-size:12px;color:{GRAY_700};margin-top:4px;'>"
            f"<span style='background:{GREEN_700};color:white;padding:2px 8px;border-radius:3px;'>Selected scenario</span>&nbsp;"
            f"<span style='background:{NAVY};color:white;padding:2px 8px;border-radius:3px;'>Below selected</span>&nbsp;"
            f"<span style='background:{BLUE_500};color:white;padding:2px 8px;border-radius:3px;'>Above selected</span>"
            f"</p>",
            unsafe_allow_html=True,
        )

    section_header("Detractor CLTV Breakdown by Segment and Service Type")
    filter_summary_block(dff, st.session_state.get("_full_df_ref", dff))

    seg_cltv = (
        dff[dff["detractor_flag"] == 1]
        .groupby("customer_segment")["annual_cltv"]
        .sum()
        .reset_index()
    )
    svc_cltv = (
        dff[dff["detractor_flag"] == 1]
        .groupby("service_type")["annual_cltv"]
        .sum()
        .reset_index()
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_d = go.Figure(go.Pie(
            labels=seg_cltv["customer_segment"],
            values=seg_cltv["annual_cltv"],
            hole=0.5,
            marker=dict(colors=[NAVY, BLUE_700, BLUE_500]),
            textinfo="percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>CLTV at Risk: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
        ))
        fig_d.update_layout(
            **base_layout(height=300),
            title=dict(text="Detractor CLTV by Customer Segment", font=dict(size=14, color=STEEL_700), x=0.5),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22),
        )
        no_grid(fig_d)
        st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig_sv = go.Figure(go.Bar(
            x=svc_cltv["service_type"],
            y=svc_cltv["annual_cltv"] / 1e6,
            marker_color=BLUE_700,
            text=[f"${v:.1f}M" for v in svc_cltv["annual_cltv"] / 1e6],
            textposition="outside",
            textfont=dict(size=12, color=STEEL_700),
            hovertemplate="<b>%{x}</b><br>Detractor CLTV: $%{y:.2f}M<extra></extra>",
        ))
        fig_sv.update_layout(
            **base_layout(height=300),
            title=dict(text="Detractor CLTV by Service Type ($M)", font=dict(size=14, color=STEEL_700), x=0.5),
            yaxis=dict(title="CLTV at Risk ($M)", tickprefix="$", ticksuffix="M"),
            xaxis=dict(title="Service Type"),
        )
        no_grid(fig_sv)
        st.plotly_chart(fig_sv, use_container_width=True, config={"displayModeBar": False})

    with st.expander("About these charts"):
        st.markdown(
            f"<p style='font-size:14px;color:{BLACK};'>"
            f"The simulator estimates the net revenue value of a Detractor recovery program at different save rates. "
            f"Save Rate is the percentage of targeted Detractors whose NPS improves enough to be retained "
            f"as active customers for another cycle. Cost per Contact is the fully loaded outreach cost per customer. "
            f"Net Value = Revenue Protected minus total Contact Cost. "
            f"The donut and bar charts break down total Detractor CLTV by Customer Segment and Service Type, "
            f"showing which product lines and customer groups represent the largest recovery opportunity."
            f"</p>",
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 5: RECOMMENDATIONS
# ============================================================
def render_recommendations(dff):
    total_cltv = dff[dff["detractor_flag"] == 1]["annual_cltv"].sum()
    det_rate = dff["detractor_flag"].mean() * 100
    cc_det = dff[dff["interaction_channel"] == "Call Center"]["detractor_flag"].mean() * 100
    hi_repeat_det = dff[dff["prior_contacts_90d"] >= 3]["detractor_flag"].mean() * 100

    section_subtitle(
        f"Recommendations are based on the analysis of {len(dff):,} survey responses across NovaStar Telecom's "
        f"customer base. The overall Detractor rate is {det_rate:.1f}%, with ${total_cltv/1e6:.1f}M in annual CLTV "
        f"at risk. Actions are sequenced by implementation timeline and expected impact."
    )

    ORANGE_HDR = ORANGE_700
    BLUE_HDR = BLUE_700
    NAVY_HDR = NAVY

    def rec_card(title, value_label, effort_label, body, evidence, border_color):
        sentences = [s.strip() for s in body.split(". ") if s.strip()]
        if len(sentences) > 1:
            body_html = "<ul style='margin:6px 0 0 0;padding-left:18px;font-size:14px;color:#2D2D2D;'>" + \
                "".join([f"<li>{s}{'.' if not s.endswith('.') else ''}</li>" for s in sentences]) + "</ul>"
        else:
            body_html = f"<p style='font-size:14px;color:{BLACK};margin:4px 0 0 0;'>{body}</p>"
        st.markdown(
            f"<div class='rec-card' style='border-left:4px solid {border_color};'>"
            f"<h5>{title}</h5>"
            f"<span class='badge badge-green'>{value_label}</span>"
            f"<span class='badge badge-steel'>{effort_label}</span>"
            f"{body_html}"
            f"<div class='evidence'>{evidence}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Tier 1: Immediate
    st.markdown(
        f"<div class='section-header' style='border-left:4px solid {ORANGE_HDR};'>"
        f"<h4>Immediate Actions (0-30 Days)</h4></div>",
        unsafe_allow_html=True,
    )
    rec_card(
        "Deploy Detractor Risk Scoring to Call Center Queue",
        "High Value", "Low Effort",
        f"Apply the trained Gradient Boosting model to score inbound contacts in real time. "
        f"Route top-decile risk contacts to senior agents. "
        f"Call Center currently drives a {cc_det:.1f}% Detractor rate, and top-decile customers are 1.82x more likely to be Detractors. "
        f"This routing change requires no new technology -- only a scoring API integration with the existing ACD system.",
        f"Evidence: Call Center Detractor rate of {cc_det:.1f}% vs. {det_rate:.1f}% overall. "
        f"Model AUC 0.81, top decile lift 1.82x.",
        ORANGE_HDR,
    )
    rec_card(
        "Launch Closed-Loop Recovery Outreach for Top-2 Decile Detractors",
        "High Value", "Low Effort",
        f"Extract the top 20% of scored Detractors and assign them to a dedicated recovery team within 48 hours of survey submission. "
        f"A personal callback from a retention specialist has been shown in comparable programs to recover 15-25% of at-risk customers. "
        f"At a 20% save rate and $25 per contact, the program returns ${total_cltv * 0.20 / 1e6:.1f}M net after costs.",
        f"Evidence: Top-2 decile cumulative gain of 36.3%. "
        f"Financial simulator output: ${total_cltv * 0.20 / 1e6:.1f}M net value at 20% save rate.",
        ORANGE_HDR,
    )

    # Tier 2: Short-Term
    st.markdown(
        f"<div class='section-header' style='border-left:4px solid {BLUE_HDR};'>"
        f"<h4>Short-Term Actions (30-90 Days)</h4></div>",
        unsafe_allow_html=True,
    )
    rec_card(
        "Redesign Call Center Resolution Scripts for Billing Disputes",
        "High Value", "Medium Effort",
        f"Billing Dispute interactions drive some of the highest Detractor concentrations in the dataset. "
        f"Introduce a single-contact resolution protocol that empowers front-line agents to issue credits up to $50 without supervisor approval. "
        f"This directly addresses the low resolution ease and billing clarity scores in this segment.",
        f"Evidence: Billing Dispute interaction reason shows among the highest Detractor rates. "
        f"Resolution ease is the top feature (30.3% importance). "
        f"Billing clarity score for Detractors averages {dff[dff['detractor_flag']==1]['billing_clarity'].mean():.2f}/10.",
        BLUE_HDR,
    )
    rec_card(
        "Implement a Repeat-Contact Early Warning Dashboard",
        "Medium Value", "Medium Effort",
        f"Customers with 3+ contacts in 90 days have a {hi_repeat_det:.1f}% Detractor rate. "
        f"Build a real-time flag in the CRM that alerts agents when a customer has contacted support more than twice in the prior quarter. "
        f"This gives agents immediate context before the call begins and enables proactive acknowledgment of the friction experience.",
        f"Evidence: Repeat contact analysis. "
        f"3+ contact group Detractor rate: {hi_repeat_det:.1f}% vs. {dff[dff['prior_contacts_90d']==0]['detractor_flag'].mean()*100:.1f}% for first-time contacts.",
        BLUE_HDR,
    )

    # Tier 3: Strategic
    st.markdown(
        f"<div class='section-header' style='border-left:4px solid {NAVY_HDR};'>"
        f"<h4>Strategic Investments (90+ Days)</h4></div>",
        unsafe_allow_html=True,
    )
    rec_card(
        "Build a Continuous VOC Driver Importance Model",
        "High Value", "High Effort",
        f"The current model is a point-in-time snapshot. "
        f"A production pipeline that retrains on rolling survey data monthly would detect shifts in driver importance as products and channels evolve. "
        f"This is especially valuable for Self-Serve App and Chat/Digital channels, which are growing in volume and shifting the wait time driver profile.",
        f"Evidence: Digital experience importance is currently 2.0%, but Self-Serve App usage is growing as a channel. "
        f"Feature importance rankings will shift as channel mix shifts.",
        NAVY_HDR,
    )
    rec_card(
        "Expand VOC Framework to Patient Experience Analytics",
        "High Value", "High Effort",
        f"The resolution ease, wait time, and FCR driver structure maps directly to the Hospital Consumer Assessment of Healthcare Providers and Systems (HCAHPS) survey framework used in healthcare. "
        f"A reusable analytics pipeline that can be adapted from this telecom VOC model to healthcare patient satisfaction scoring would significantly expand the market for this methodology. "
        f"See the Healthcare Application tab for the full translation mapping.",
        f"Evidence: Cross-industry translation documented in Tab 6. "
        f"HCAHPS composite scores follow the same importance-performance structure as telecom NPS driver analysis.",
        NAVY_HDR,
    )

    with st.expander("About these recommendations"):
        st.markdown(
            f"<p style='font-size:14px;color:{BLACK};'>"
            f"Recommendations are sequenced by implementation timeline: Immediate (0-30 days) "
            f"uses existing data and tools; Short-Term (30-90 days) requires process or workflow changes; "
            f"Strategic (90+ days) requires new technical or organizational investment. "
            f"Value and Effort ratings are relative to each other within this project's scope. "
            f"All financial figures are derived from the dataset and the revenue simulator using live computed values."
            f"</p>",
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 6: HEALTHCARE APPLICATION
# ============================================================
def render_healthcare_app(dff):
    insight(
        "FRAMEWORK PORTABILITY",
        "Every driver in the NovaStar telecom VOC model has a direct analogue in the "
        "Hospital Consumer Assessment of Healthcare Providers and Systems (HCAHPS) patient survey. "
        "The same importance-performance analysis, risk decile scoring, and recovery simulation "
        "framework applies with minimal adaptation. "
        "This tab documents the full mapping and three actionable levers for a healthcare client."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        section_header("Telecom-to-Healthcare Driver Translation")
        translations = [
            ("Resolution Ease", "Care Coordination Score", "How easily was my care need addressed across providers?"),
            ("Wait Time Score", "Response to Call Button", "How quickly did staff respond to care requests?"),
            ("First Contact Resolution (FCR)", "Discharge Planning Completeness", "Was everything in place for safe discharge the first time?"),
            ("Agent Professionalism", "Staff Communication Quality", "Did staff explain care clearly and professionally?"),
            ("Digital Experience", "Patient Portal Usability", "Was the online patient portal easy to use for managing care?"),
            ("Billing Clarity", "Explanation of Charges", "Was the billing statement and cost estimate clear and accurate?"),
            ("Prior Contacts (90d)", "Readmission within 30 Days", "Repeat contacts = repeat admissions; both signal process failure."),
            ("Resolved Flag", "Discharge Planning Completeness", "Unresolved issue at discharge = elevated readmission risk."),
        ]
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:14px;color:{BLACK};'>"
            f"<tr style='background:{STEEL_100};'>"
            f"<th style='padding:8px;text-align:left;color:{NAVY};border-bottom:2px solid {BLUE_700};'>Telecom Driver</th>"
            f"<th style='padding:4px;text-align:center;color:{NAVY};border-bottom:2px solid {BLUE_700};'></th>"
            f"<th style='padding:8px;text-align:left;color:{NAVY};border-bottom:2px solid {BLUE_700};'>Healthcare Analogue (HCAHPS)</th>"
            f"</tr>" +
            "".join([
                f"<tr style='border-bottom:1px solid {STEEL_300};'>"
                f"<td style='padding:8px;color:{NAVY};font-weight:600;'>{t}</td>"
                f"<td style='padding:4px;text-align:center;color:{BLUE_700};font-size:18px;'>&#8594;</td>"
                f"<td style='padding:8px;color:{BLACK};'><strong>{h}</strong><br>"
                f"<span style='font-size:12px;color:{GRAY_700};'>{d}</span></td>"
                f"</tr>"
                for t, h, d in translations
            ]) +
            "</table>",
            unsafe_allow_html=True,
        )

    with col_right:
        section_header("Three Action Levers for a Healthcare Client")

        levers = [
            (
                "1",
                "Discharge Planning Quality Score",
                f"Use the FCR driver framework to build a post-discharge survey that scores care coordination completeness. "
                f"Patients whose discharge plan had gaps (analogous to 'unresolved flag') are flagged for a 48-hour nurse callback. "
                f"This mirrors the closed-loop Detractor recovery program from the telecom model.",
                f"Readmission rate reduction of 15-20%",
            ),
            (
                "2",
                "Staff Communication Driver Importance Model",
                f"Train a key driver analysis model on HCAHPS data to rank which communication behaviors (clear explanations, "
                f"responsiveness, pain management communication) most predict the 'Overall Hospital Rating' outcome. "
                f"This directly replicates the NPS driver importance ranking methodology used here.",
                f"Identifies the 2-3 behaviors that explain 60%+ of rating variance",
            ),
            (
                "3",
                "Repeat-Contact Patient Risk Flag",
                f"Patients with multiple prior admissions in 90 days have a readmission risk profile analogous to the "
                f"repeat-contact Detractor segment identified in this analysis. "
                f"A CRM flag that alerts care coordinators to these patients before discharge enables proactive intervention "
                f"before a preventable readmission occurs.",
                f"Reduces preventable 30-day readmissions in flagged cohort",
            ),
        ]

        for num, title, body, stat in levers:
            st.markdown(
                f"<div style='background:{WHITE};border-left:4px solid {BLUE_700};border-radius:6px;"
                f"padding:14px 16px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.07);'>"
                f"<p style='font-size:12px;font-weight:700;color:{STEEL_700};text-transform:uppercase;"
                f"letter-spacing:0.06em;margin:0 0 4px 0;'>Lever {num}</p>"
                f"<p style='font-size:15px;font-weight:700;color:{NAVY};margin:0 0 8px 0;'>{title}</p>"
                f"<p style='font-size:14px;color:{BLACK};margin:0 0 10px 0;'>{body}</p>"
                f"<span class='stat-badge'>{stat}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    section_header("Framework Portability")
    st.markdown(
        f"<p style='font-size:14px;color:{BLACK};margin-bottom:12px;'>"
        f"The VOC analytics methodology used in this project -- driver importance scoring, risk decile assignment, "
        f"closed-loop recovery simulation -- is domain-agnostic. The same pipeline applies to:"
        f"</p>"
        f"<ul style='font-size:14px;color:{BLACK};padding-left:20px;'>"
        f"<li><strong>Healthcare:</strong> HCAHPS patient satisfaction, readmission risk, care coordination quality</li>"
        f"<li><strong>Financial services:</strong> Customer effort score (CES) post-transaction, loan servicing friction</li>"
        f"<li><strong>Retail / e-commerce:</strong> Post-purchase NPS, return rate prediction, fulfillment friction</li>"
        f"<li><strong>B2B SaaS:</strong> Renewal risk scoring using product usage + support ticket patterns</li>"
        f"</ul>",
        unsafe_allow_html=True,
    )

    section_header("Closing Takeaway")
    st.markdown(
        f"<div style='background:{WHITE};border-left:4px solid {NAVY};border-radius:6px;"
        f"padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.07);'>"
        f"<p style='font-size:15px;color:{NAVY};font-weight:700;margin:0 0 8px 0;'>"
        f"A VOC analyst who understands the math also speaks the language of the client's industry.</p>"
        f"<p style='font-size:14px;color:{BLACK};margin:0;'>"
        f"The value of this framework is not the telecom domain -- it is the analytical pattern: identify which "
        f"survey drivers explain outcome variance, score customers by risk, quantify the financial opportunity, "
        f"and translate findings into a prioritized action roadmap. That pattern is reusable across every "
        f"industry that collects structured customer feedback."
        f"</p></div>",
        unsafe_allow_html=True,
    )

    with st.expander("About this tab"):
        st.markdown(
            f"<p style='font-size:14px;color:{BLACK};'>"
            f"This tab translates the NovaStar Telecom VOC model into the healthcare patient experience domain. "
            f"HCAHPS (Hospital Consumer Assessment of Healthcare Providers and Systems) is the standard "
            f"patient satisfaction survey used by CMS-regulated hospitals. "
            f"The driver mapping table shows how each telecom satisfaction dimension has a structural analogue "
            f"in the HCAHPS survey framework. "
            f"The three action levers demonstrate how the same analytics approach would be deployed "
            f"in a healthcare engagement to drive meaningful clinical and financial outcomes."
            f"</p>",
            unsafe_allow_html=True,
        )


# ============================================================
# MAIN
# ============================================================
def main():
    df = load_data()
    model_results = train_model(df)
    defaults = init_session_state(df)
    render_sidebar(df, defaults)
    dff = apply_filters(df)

    # Store full df reference for filter summary block
    st.session_state["_full_df_ref"] = df

    st.markdown(
        f"<h2 style='color:{NAVY};font-size:22px;margin-bottom:2px;'>"
        f"NovaStar Telecom -- Voice of Customer (VOC) Analytics</h2>"
        f"<p style='font-size:14px;color:{GRAY_700};margin-bottom:12px;'>"
        f"120,000 post-interaction survey responses | Gradient Boosting Detractor Risk Model | "
        f"Author: Luciano Casillas</p>",
        unsafe_allow_html=True,
    )

    render_kpi_header(dff)

    tabs = st.tabs([
        "Overview",
        "CX Drivers",
        "Model + Risk",
        "Financial Impact",
        "Recommendations",
        "Healthcare Application",
    ])

    with tabs[0]:
        render_overview(dff)
    with tabs[1]:
        render_cx_drivers(dff)
    with tabs[2]:
        render_model_risk(dff, model_results)
    with tabs[3]:
        render_financial_impact(dff)
    with tabs[4]:
        render_recommendations(dff)
    with tabs[5]:
        render_healthcare_app(dff)


if __name__ == "__main__":
    main()