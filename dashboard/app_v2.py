from __future__ import annotations

from pathlib import Path
import json
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Developer Pulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT ROOT
# ============================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "data" / "results").exists():
            return candidate

    raise FileNotFoundError(
        "Project root not found. Run Streamlit from the project root "
        "or from the dashboard folder."
    )


ROOT = find_project_root()

RESULTS = ROOT / "data" / "results"
FINAL = RESULTS / "final_synthesis"
EMOTIONS = RESULTS / "emotions"
STRESS_TOPICS = RESULTS / "stress_topics"
COMBINED = RESULTS / "combined"


# ============================================================
# HELPERS
# ============================================================

def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def fig_style(
    fig: go.Figure,
    *,
    height: int = 360,
    showlegend: bool = True,
) -> go.Figure:
    fig.update_layout(
        height=height,
        autosize=True,
        margin=dict(l=18, r=18, t=45, b=25),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family="Inter, Arial, sans-serif",
            color="#3f3f46",
            size=12,
        ),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#3f3f46"),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_color="#27272a",
            bordercolor="#d4d4d8",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#f1f1f4",
        zeroline=False,
        linecolor="#e4e4e7",
        tickfont=dict(color="#3f3f46"),
        title_font=dict(color="#3f3f46"),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f1f1f4",
        zeroline=False,
        linecolor="#e4e4e7",
        tickfont=dict(color="#3f3f46"),
        title_font=dict(color="#3f3f46"),
        automargin=True,
    )
    return fig


def card(title: str, value: str, note: str = "", accent: str = "#5b2aa8"):
    st.markdown(
        f"""
        <div class="metric-card" style="border-top:4px solid {accent};">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_note(title: str, description: str, takeaway: str = ""):
    takeaway_html = ""
    if takeaway:
        takeaway_html = (
            f'<div class="chart-takeaway"><strong>Main result:</strong> {takeaway}</div>'
        )

    st.markdown(
        f"""
        <div class="chart-note">
            <div class="chart-note-title">{title}</div>
            <div class="chart-note-text">{description}</div>
            {takeaway_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(title: str, body: str, kind: str = "info"):
    cls = {
        "info": "box-info",
        "warning": "box-warning",
        "success": "box-success",
    }.get(kind, "box-info")

    st.markdown(
        f"""
        <div class="{cls}">
            <strong>{title}</strong><br>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD FINAL RESULTS
# ============================================================

final_findings = load_json(FINAL / "final_findings.json")
sentiment_distribution = load_csv(FINAL / "sentiment_distribution.csv")
negative_vs_stress = load_csv(FINAL / "negative_sentiment_vs_stress.csv")
emotion_stress = load_csv(FINAL / "emotion_stress_association.csv")
topic_final = load_csv(FINAL / "topic_stress_final.csv")

emotion_prevalence = load_csv(
    EMOTIONS / "goemotions_analysis_ready_emotion_prevalence.csv"
)

unified = load_csv(
    COMBINED / "sentiment_emotion_stress_predictions.csv"
)

topic_assignments = load_csv(
    STRESS_TOPICS / "final_topic_assignments.csv"
)

topic_keywords = load_csv(
    STRESS_TOPICS / "final_topic_keywords.csv"
)

topic_count_eval = load_csv(
    STRESS_TOPICS / "topic_count_evaluation.csv"
)

topic_stat_summary = load_csv(
    STRESS_TOPICS / "topic_stress_statistical_summary.csv"
)


# ============================================================
# FALLBACK FINAL VALUES
# ============================================================

ANALYSIS_POSTS = int(final_findings.get("records", 2666))
COLLECTED_POSTS = 5406

STRESS_POSTS = int(
    final_findings
    .get("stress", {})
    .get("predicted_stress_posts", 248)
)

STRESS_RATE = float(
    final_findings
    .get("stress", {})
    .get("predicted_stress_rate", 0.0930232558)
)

AGREEMENT = float(
    final_findings
    .get("sentiment", {})
    .get("vader_transformer_agreement_rate", 0.3762190548)
)

AVG_EMOTIONS = float(
    final_findings
    .get("emotions", {})
    .get("average_emotions_per_post", 1.850337)
)


# ============================================================
# BUILD ROW-LEVEL VIEW FOR GLOBAL FILTERS
# ============================================================

explorer = None

if unified is not None:
    explorer = unified.copy()

    if (
        topic_assignments is not None
        and "record_id" in explorer.columns
        and "record_id" in topic_assignments.columns
    ):
        merge_cols = [
            c for c in [
                "record_id",
                "topic",
                "topic_name",
                "topic_weight",
            ]
            if c in topic_assignments.columns
        ]

        if len(merge_cols) > 1:
            explorer = explorer.merge(
                topic_assignments[merge_cols],
                on="record_id",
                how="left",
                validate="one_to_one",
            )


# ============================================================
# STYLE - INSPIRED BY THE REFERENCE IMAGE
# ============================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: light !important;
        --page: #eee7fb;
        --panel: #ffffff;
        --purple: #4b2491;
        --purple2: #7c3aed;
        --pink: #ff4f8b;
        --blue: #2563eb;
        --green: #10b981;
        --ink: #27272a;
        --muted: #71717a;
        --line: #d9d1eb;
    }

    html, body {
        color-scheme: light !important;
        background: var(--page) !important;
    }

    .stApp {
        color-scheme: light !important;
        background: var(--page);
        color: var(--ink) !important;
        -webkit-text-fill-color: initial;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header {
        background: rgba(238,231,251,0.95) !important;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    .top-header {
        background: #d9b7ff;
        border: 1px solid #cfb1ef;
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 0.9rem;
    }

    .brand-pill {
        display: inline-block;
        background: var(--purple);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        margin-bottom: 0.6rem;
    }

    .header-title {
        font-size: 2rem;
        font-weight: 800;
        color: #2d145f;
        margin-bottom: 0.25rem;
    }

    .header-subtitle {
        color: #5f4b7d;
        font-size: 0.95rem;
    }

    .metric-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 145px;
        box-shadow: 0 2px 8px rgba(72, 43, 117, 0.05);
    }

    .metric-title {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .metric-value {
        color: var(--ink);
        font-size: 2.2rem;
        line-height: 1.15;
        font-weight: 800;
        margin-top: 0.45rem;
    }

    .metric-note {
        color: #7c6f8f;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    .section-panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(72, 43, 117, 0.04);
    }

    .box-info {
        border-left: 4px solid var(--purple2);
        background: #f7f3ff;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        color: #4c3c65;
    }

    .box-warning {
        border-left: 4px solid #dc2626;
        background: #fff4f4;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        color: #7f1d1d;
    }

    .box-success {
        border-left: 4px solid var(--green);
        background: #eefcf6;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        color: #065f46;
    }

    div[data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.25rem;
        box-shadow: 0 2px 8px rgba(72, 43, 117, 0.04);
    }

    div[data-testid="stDataFrame"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stSelectbox"] > div,
    div[data-testid="stMultiSelect"] > div,
    div[data-testid="stTextInput"] > div {
        background: #ffffff;
        border-radius: 10px;
    }

    div[data-testid="stTabs"] button {
        font-weight: 650;
    }

    /* --------------------------------------------------------
       THEME SAFETY
       Keep the dashboard visually identical in browser/Streamlit
       light and dark modes by explicitly styling native widgets.
       -------------------------------------------------------- */

    /* General Streamlit text rendered directly on the page. */
    .stApp,
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp div[data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }

    /* Native headings/subheadings.
       Streamlit/OS dark mode can apply a light text fill even when `color` is
       set, so force both CSS color and WebKit text fill. */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        opacity: 1 !important;
    }

    /* Normal Markdown text outside our custom HTML cards. */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] em {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        opacity: 1 !important;
    }

    /* Captions and secondary text. */
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
        opacity: 1 !important;
    }

    /* Tabs: inactive and active labels stay readable. */
    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] button p {
        color: #4c3c65 !important;
        -webkit-text-fill-color: #4c3c65 !important;
        opacity: 1 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: var(--purple) !important;
        -webkit-text-fill-color: var(--purple) !important;
        opacity: 1 !important;
    }

    /* Selectbox / multiselect controls and their visible values. */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="tag"] span {
        color: var(--ink) !important;
    }

    /* Multiselect tags: keep contrast even in dark mode. */
    div[data-baseweb="tag"] {
        background-color: #eee7fb !important;
        color: #3b2565 !important;
    }

    /* Text inputs, including the search field. */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextInput"] input:focus {
        background-color: #ffffff !important;
        color: var(--ink) !important;
        caret-color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8b8494 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #8b8494 !important;
    }

    /* Disabled text areas used to inspect developer posts. */
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextArea"] textarea:disabled {
        background-color: #ffffff !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        opacity: 1 !important;
    }

    /* Expanders and their summaries. */
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        color: var(--ink);
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] details {
        color: var(--ink) !important;
    }

    /* Native info/warning messages, when used. */
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] div {
        color: var(--ink);
    }

    /* Dataframes/tables: keep surrounding UI and column controls legible. */
    div[data-testid="stDataFrame"] {
        color: var(--ink);
    }

    /* Widget labels such as Platform, Stress label, Topic and Search. */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] span {
        color: #4c3c65 !important;
        -webkit-text-fill-color: #4c3c65 !important;
        opacity: 1 !important;
    }

    .small-note {
        color: #756b80;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }

    /* --------------------------------------------------------
       SMALL CHART EXPLANATIONS
       Same visual language as the existing dashboard.
    -------------------------------------------------------- */

    .chart-note {
        background: #ffffff;
        border: 1px solid #d9d1eb;
        border-left: 4px solid #7c3aed;
        border-radius: 10px;
        padding: 0.72rem 0.85rem;
        margin: 0.35rem 0 0.65rem 0;
        color: #5f4b7d;
        font-size: 0.84rem;
        line-height: 1.5;
    }

    .chart-note-title {
        color: #3b2565;
        font-weight: 700;
        margin-bottom: 0.16rem;
    }

    .chart-note-text {
        color: #6d6080;
    }

    .chart-takeaway {
        color: #3b2565;
        margin-top: 0.28rem;
    }

    /* --------------------------------------------------------
       RESPONSIVENESS
       Preserve the original colors/cards. Only improve fitting.
    -------------------------------------------------------- */

    .block-container {
        width: min(100%, 1500px);
        padding-left: clamp(0.8rem, 3vw, 2.5rem);
        padding-right: clamp(0.8rem, 3vw, 2.5rem);
    }

    .metric-card {
        min-width: 0;
        width: 100%;
        box-sizing: border-box;
    }

    .metric-title,
    .metric-value,
    .metric-note,
    .box-info,
    .box-warning,
    .box-success {
        overflow-wrap: anywhere;
        word-break: normal;
    }

    div[data-testid="stColumn"] {
        min-width: 0 !important;
    }

    div[data-testid="stPlotlyChart"] {
        width: 100% !important;
        min-width: 0 !important;
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto;
    }

    div[data-testid="stTabs"] {
        width: 100%;
        max-width: 100%;
        overflow: hidden;
    }

    div[data-testid="stTabs"] div[role="tablist"] {
        overflow-x: auto;
        scrollbar-width: thin;
    }

    div[data-testid="stTabs"] button {
        white-space: nowrap;
    }

    div[data-testid="stSelectbox"],
    div[data-testid="stMultiSelect"],
    div[data-testid="stTextInput"] {
        min-width: 0 !important;
        max-width: 100% !important;
    }

    /* On medium screens, let Streamlit columns wrap to two per row. */
    @media (max-width: 1100px) {

        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.85rem !important;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            flex: 1 1 calc(50% - 0.85rem) !important;
            width: calc(50% - 0.85rem) !important;
            min-width: 290px !important;
        }

        .metric-value {
            font-size: 1.9rem;
        }
    }

    /* On small screens, stack every column. */
    @media (max-width: 720px) {

        .block-container {
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 0 !important;
        }

        .metric-card {
            min-height: 118px;
        }

        .metric-value {
            font-size: 1.8rem;
        }

        .chart-note {
            font-size: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="top-header">
        <div class="brand-pill">AI Developer Pulse</div>
        <div class="header-title">Developer discussions about AI tools</div>
        <div class="header-subtitle">
            Sentiment → Emotions → Stress → Integrated Analysis → Stress-associated Topics
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GLOBAL FILTERS
# ============================================================

filter_cols = st.columns([1.05, 1.05, 1.05, 1.2])

if explorer is not None:
    platform_options = sorted(
        explorer["platform"].dropna().astype(str).unique().tolist()
    ) if "platform" in explorer.columns else []

    stress_options = sorted(
        explorer["hybrid_stress_label"].dropna().astype(str).unique().tolist()
    ) if "hybrid_stress_label" in explorer.columns else []

    topic_options = sorted(
        explorer["topic_name"].dropna().astype(str).unique().tolist()
    ) if "topic_name" in explorer.columns else []

    with filter_cols[0]:
        selected_platform = st.multiselect(
            "Platform",
            platform_options,
            placeholder="All platforms",
        )

    with filter_cols[1]:
        selected_stress = st.multiselect(
            "Stress label",
            stress_options,
            placeholder="All labels",
        )

    with filter_cols[2]:
        selected_topic = st.multiselect(
            "Topic",
            topic_options,
            placeholder="All topics",
        )

    with filter_cols[3]:
        search_text = st.text_input(
            "Search",
            placeholder="ChatGPT, Copilot, GPU, billing...",
        )

    filtered = explorer.copy()

    if selected_platform and "platform" in filtered.columns:
        filtered = filtered[
            filtered["platform"].astype(str).isin(selected_platform)
        ]

    if selected_stress and "hybrid_stress_label" in filtered.columns:
        filtered = filtered[
            filtered["hybrid_stress_label"].astype(str).isin(selected_stress)
        ]

    if selected_topic and "topic_name" in filtered.columns:
        filtered = filtered[
            filtered["topic_name"].astype(str).isin(selected_topic)
        ]

    text_col = next(
        (
            c for c in [
                "text_clean_basic",
                "text",
                "full_text_raw",
            ]
            if c in filtered.columns
        ),
        None,
    )

    if search_text and text_col is not None:
        filtered = filtered[
            filtered[text_col]
            .fillna("")
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                regex=False,
            )
        ]
else:
    for col, label in zip(
        filter_cols,
        ["Platform", "Stress label", "Topic", "Search"],
    ):
        with col:
            st.text_input(
                label,
                disabled=True,
                placeholder="Row-level file unavailable",
            )
    filtered = None


# ============================================================
# OVERVIEW KPIs
# ============================================================

if filtered is not None and len(filtered) > 0:
    view_posts = len(filtered)

    if "hybrid_stress_label" in filtered.columns:
        view_stress = int(
            (filtered["hybrid_stress_label"].astype(str) == "Stress").sum()
        )
    else:
        view_stress = STRESS_POSTS

    view_stress_rate = view_stress / view_posts if view_posts else 0

    if (
        "vader_label" in filtered.columns
        and "transformer_label" in filtered.columns
    ):
        view_agreement = (
            filtered["vader_label"].astype(str)
            == filtered["transformer_label"].astype(str)
        ).mean()
    else:
        view_agreement = AGREEMENT
else:
    view_posts = ANALYSIS_POSTS
    view_stress = STRESS_POSTS
    view_stress_rate = STRESS_RATE
    view_agreement = AGREEMENT


k1, k2, k3, k4 = st.columns(4)

with k1:
    card(
        "Analysis corpus",
        f"{view_posts:,}",
        "posts in current view",
        "#4b2491",
    )

with k2:
    card(
        "Predicted Stress",
        f"{view_stress:,}",
        f"{pct(view_stress_rate, 2)} of current view",
        "#ff4f8b",
    )

with k3:
    card(
        "Sentiment agreement",
        pct(view_agreement, 1),
        "VADER ↔ Transformer",
        "#2563eb",
    )

with k4:
    card(
        "Average emotions",
        f"{AVG_EMOTIONS:.2f}",
        "GoEmotions labels per post",
        "#10b981",
    )


# ============================================================
# TABS
# ============================================================
# ============================================================
# TABS — ALIGNED WITH SUPERVISOR FEEDBACK
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Sentiment & Emotions",
        "Stress Detection",
        "Sentiment / Emotion / Stress",
        "Stress Topics",
        "Examples",
        "Insights & Conclusions",
    ]
)


# ============================================================
# 1 — OVERVIEW
# ============================================================

with tabs[0]:

    st.subheader("Global overview")
    st.caption(
        "A simple view of the full chain: data → models → analysis → interpretation → visualization."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Collected records", f"{COLLECTED_POSTS:,}", "raw records gathered", "#4b2491")
    with c2:
        card("Analysis-ready posts", f"{ANALYSIS_POSTS:,}", "final cleaned corpus", "#2563eb")
    with c3:
        card("Predicted Stress", f"{STRESS_POSTS:,}", f"{pct(STRESS_RATE, 2)} of analysis corpus", "#ff4f8b")
    with c4:
        card("Final topics", "7", "NMF discussion themes", "#10b981")

    st.markdown("#### Analysis chain")
    chain = pd.DataFrame(
        [
            ["Data preparation", "Cleaning and corpus preparation", "2,666 analysis-ready posts"],
            ["Sentiment", "VADER + Transformer", "Positive / Neutral / Negative"],
            ["Emotions", "GoEmotions multi-label", "Specific emotional signals"],
            ["Stress", "Dreaddit baseline + hybrid augmentation", "Stress / No stress"],
            ["Integrated analysis", "Sentiment × Emotions × Stress", "Cross-signal relationships"],
            ["Topic Modeling", "TF-IDF + NMF + association tests", "7 topics and Stress associations"],
        ],
        columns=["Stage", "Method", "Output"],
    )
    st.dataframe(chain, use_container_width=True, hide_index=True)

    left, center, right = st.columns([1.1, 1.1, 0.8])

    with left:
        st.markdown("#### Sentiment distribution")
        chart_note(
            "How to read this chart",
            "The same developer posts are classified by VADER and by the Transformer. "
            "Compare the two bars inside each sentiment class to see where the models agree or diverge.",
            "The Transformer is much more neutral, while VADER is much more positive."
        )
        if sentiment_distribution is not None:
            pivot = (
                sentiment_distribution
                .pivot(index="label", columns="method", values="rate")
                .reindex(["Negative", "Neutral", "Positive"])
            )
            fig = go.Figure()
            if "VADER" in pivot.columns:
                fig.add_bar(name="VADER", x=pivot.index, y=pivot["VADER"] * 100, marker_color="#ff4f8b")
            if "Transformer" in pivot.columns:
                fig.add_bar(name="Transformer", x=pivot.index, y=pivot["Transformer"] * 100, marker_color="#4b2491")
            fig.update_layout(barmode="group", yaxis_title="Share of posts (%)", xaxis_title="")
            fig_style(fig, height=335)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    with center:
        st.markdown("#### Predicted Stress by topic")
        chart_note(
            "How to read this chart",
            "Each bar is the share of posts classified as Stress inside one topic. "
            "The dotted line represents the overall 9.30% predicted-Stress rate.",
            "ChatGPT / OpenAI User Experience is above the overall rate; AI Labs / Industry News is below it."
        )
        if topic_final is not None:
            plot = topic_final.sort_values("predicted_stress_rate")
            colors = [
                "#ff4f8b" if bool(sig) and rr > 1 else "#10b981" if bool(sig) and rr < 1 else "#8b78b8"
                for sig, rr in zip(plot["significant_fdr_0_05"], plot["relative_risk"])
            ]
            fig = go.Figure(go.Bar(
                x=plot["predicted_stress_rate"] * 100,
                y=plot["topic_name"],
                orientation="h",
                marker_color=colors,
            ))
            fig.add_vline(x=STRESS_RATE * 100, line_dash="dot", line_color="#71717a")
            fig.update_layout(xaxis_title="Predicted Stress rate (%)", yaxis_title="")
            fig_style(fig, height=335, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    with right:
        st.markdown("#### Stress distribution")
        chart_note(
            "How to read this chart",
            "The donut shows the final hybrid model classification on all 2,666 analysis-ready posts.",
            "248 posts are classified as Stress (9.30%). This is a model prediction, not psychological-stress prevalence."
        )
        fig = go.Figure(go.Pie(
            labels=["No stress", "Predicted Stress"],
            values=[ANALYSIS_POSTS - STRESS_POSTS, STRESS_POSTS],
            hole=0.62,
            marker_colors=["#d9d1eb", "#ff4f8b"],
            textinfo="percent",
        ))
        fig_style(fig, height=335, showlegend=True)
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    info_box(
        "Main message",
        "The system analyzes several levels separately: general sentiment, specific emotions, "
        "predicted Stress, and finally the discussion topics associated with Stress. "
        "Negative sentiment is not used as a proxy for Stress.",
        "success",
    )


# ============================================================
# 2 — SENTIMENT & EMOTIONS
# ============================================================

with tabs[1]:

    st.subheader("Sentiment and emotions")
    st.caption("General tone first, then more specific emotional signals.")

    c1, c2 = st.columns([1.15, 0.85])

    with c1:
        st.markdown("#### VADER vs Transformer")
        chart_note(
            "Figure context",
            "VADER is lexicon-based while the Transformer uses contextual language. "
            "The chart compares their sentiment distributions on the same corpus.",
            "Their overall agreement is 37.6%, so the dashboard keeps both views instead of merging them."
        )
        if sentiment_distribution is not None:
            pivot = (
                sentiment_distribution
                .pivot(index="label", columns="method", values="rate")
                .reindex(["Negative", "Neutral", "Positive"])
            )
            fig = go.Figure()
            fig.add_bar(
                name="VADER", x=pivot.index, y=pivot["VADER"] * 100,
                marker_color="#ff4f8b",
                text=[f"{v:.1f}%" for v in pivot["VADER"] * 100], textposition="outside",
            )
            fig.add_bar(
                name="Transformer", x=pivot.index, y=pivot["Transformer"] * 100,
                marker_color="#4b2491",
                text=[f"{v:.1f}%" for v in pivot["Transformer"] * 100], textposition="outside",
            )
            fig.update_layout(barmode="group", yaxis_title="Share of posts (%)")
            fig_style(fig, height=390)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    with c2:
        st.markdown("#### Key result")
        card(
            "VADER ↔ Transformer agreement",
            pct(AGREEMENT, 1),
            "1,003 matching labels out of 2,666 posts",
            "#4b2491",
        )
        info_box(
            "Interpretation",
            "The methods capture sentiment differently. VADER is lexicon-based, while the Transformer "
            "uses contextual representations. Their outputs are therefore compared rather than merged "
            "into a single pseudo-ground-truth label.",
            "info",
        )

    st.markdown("#### Most frequent predicted emotions")
    chart_note(
        "Figure context",
        "GoEmotions is a multi-label model, so one post may receive several emotion labels. "
        "The bars show how often each emotion appears across the corpus.",
        "Neutral is the most common prediction, followed by approval, disapproval, curiosity and confusion."
    )
    if emotion_prevalence is not None:
        emotion_col = "emotion" if "emotion" in emotion_prevalence.columns else emotion_prevalence.columns[0]
        rate_col = next((c for c in emotion_prevalence.columns if "rate" in c.lower() or "percent" in c.lower() or "prevalence" in c.lower()), None)
        if rate_col is not None:
            e = emotion_prevalence.copy()
            e["_rate"] = e[rate_col].astype(float)
            if e["_rate"].max() > 1.5:
                e["_rate"] /= 100.0
            e = e.sort_values("_rate", ascending=False).head(10).sort_values("_rate")
            fig = go.Figure(go.Bar(
                x=e["_rate"] * 100,
                y=e[emotion_col],
                orientation="h",
                marker_color="#4b2491",
                text=[f"{v:.1f}%" for v in e["_rate"] * 100],
                textposition="outside",
            ))
            fig.update_layout(xaxis_title="Share of posts (%)", yaxis_title="")
            fig_style(fig, height=430, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    info_box(
        "Emotion layer",
        "GoEmotions is multi-label, so a post may express more than one emotion. "
        "The final predictions use per-emotion thresholds calibrated on the validation split.",
        "info",
    )


# ============================================================
# 3 — STRESS DETECTION
# ============================================================

with tabs[2]:

    st.subheader("Stress detection")
    st.caption("Model comparison and complementary validation on real developer-domain posts.")

    st.markdown("#### Baseline vs final hybrid model")
    comparison = pd.DataFrame(
        [
            ["Dreaddit-only baseline", 0.7259, 0.7249, 0.7224, 0.7615, 0.7414],
            ["Dreaddit + synthetic hybrid", 0.7371, 0.7359, 0.7303, 0.7778, 0.7533],
        ],
        columns=["Model", "Accuracy", "Macro F1", "Stress precision", "Stress recall", "Stress F1"],
    )
    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.3f}", "Macro F1": "{:.3f}", "Stress precision": "{:.3f}",
            "Stress recall": "{:.3f}", "Stress F1": "{:.3f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    info_box(
        "Synthetic data",
        "Synthetic developer examples were used only for training augmentation. "
        "The main conclusions remain based on real data: Dreaddit and the developer-domain validation set.",
        "warning",
    )

    st.markdown("#### Complementary validation on 600 developer posts")
    v1, v2, v3, v4, v5 = st.columns(5)
    with v1: card("Accuracy", "0.916", "596 evaluable posts", "#4b2491")
    with v2: card("Balanced accuracy", "0.834", "domain audit", "#2563eb")
    with v3: card("Stress precision", "0.111", "false positives remain", "#ff4f8b")
    with v4: card("Stress recall", "0.750", "6 of 8 Stress detected", "#10b981")
    with v5: card("MCC", "0.268", "imbalanced-domain metric", "#8b78b8")

    left, right = st.columns([0.9, 1.1])
    with left:
        chart_note(
            "Confusion matrix",
            "Rows are the real developer-domain reference labels and columns are the model predictions. "
            "Diagonal cells are correct classifications; off-diagonal cells are errors.",
            "The model detects 6 of 8 Stress cases but also produces 48 false positives, explaining high recall and low precision."
        )
        matrix = pd.DataFrame(
            [[540, 48], [2, 6]],
            index=["Reference No stress", "Reference Stress"],
            columns=["Predicted No stress", "Predicted Stress"],
        )
        fig = go.Figure(go.Heatmap(
            z=matrix.values, x=matrix.columns, y=matrix.index, text=matrix.values,
            texttemplate="%{text}",
            colorscale=[[0, "#f5efff"], [1, "#4b2491"]],
            showscale=False,
        ))
        fig_style(fig, height=335, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    with right:
        card(
            "Final predicted Stress",
            f"{STRESS_POSTS}",
            f"{pct(STRESS_RATE, 2)} of the 2,666 posts",
            "#ff4f8b",
        )
        info_box(
            "Interpretation",
            "The developer-domain audit shows useful recall but low precision. Technical frustration "
            "can create false positives. Therefore, 9.30% is reported only as a predicted-Stress rate, "
            "not as psychological-stress prevalence.",
            "warning",
        )


# ============================================================
# 4 — SENTIMENT / EMOTION / STRESS
# ============================================================

with tabs[3]:

    st.subheader("Relationship between Sentiment, Emotions and Stress")
    st.caption("The central analytical question: is negative sentiment the same as Stress?")

    r1, r2 = st.columns(2)
    with r1:
        card("VADER-negative → Stress", "14.3%", "81 of 565 negative posts", "#ff4f8b")
    with r2:
        card("Transformer-negative → Stress", "24.3%", "117 of 482 negative posts", "#4b2491")

    info_box(
        "Main finding",
        "Negative sentiment and Stress are not equivalent. A developer can write negatively because of "
        "bugs, dissatisfaction, criticism or technical frustration without expressing personal distress.",
        "success",
    )

    if emotion_stress is not None:
        st.markdown("#### Emotions enriched in predicted-Stress posts")
        chart_note(
            "Figure context",
            "For each emotion, the chart compares its frequency in posts classified as Stress with its frequency in No-stress posts.",
            "Annoyance, disappointment, confusion, realization and disapproval show the largest descriptive enrichments."
        )
        e = emotion_stress.sort_values("rate_difference", ascending=False).head(8).sort_values("rate_difference")
        fig = go.Figure()
        fig.add_bar(name="No stress", y=e["emotion"], x=e["no_stress_rate"] * 100, orientation="h", marker_color="#d9d1eb")
        fig.add_bar(name="Predicted Stress", y=e["emotion"], x=e["stress_rate"] * 100, orientation="h", marker_color="#ff4f8b")
        fig.update_layout(barmode="group", xaxis_title="Share of posts (%)", yaxis_title="")
        fig_style(fig, height=430)
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
        st.caption(
            "The largest descriptive enrichments are annoyance, disappointment, confusion, realization and disapproval. "
            "These are descriptive comparisons, not significance tests."
        )


# ============================================================
# 5 — STRESS TOPICS
# ============================================================

with tabs[4]:

    st.subheader("Topics most associated with predicted Stress")
    st.caption("Topics were learned from all 2,666 posts first; Stress association was tested afterwards.")

    if topic_final is not None:
        chart_note(
            "Figure context",
            "The bars show predicted-Stress rate for each final topic. "
            "The dotted line is the overall corpus rate. Colors highlight topics that remain significant after FDR correction.",
            "ChatGPT / OpenAI User Experience is significantly higher; AI Labs / Industry News is significantly lower."
        )
        plot = topic_final.sort_values("predicted_stress_rate")
        colors = [
            "#ff4f8b" if bool(sig) and rr > 1 else "#10b981" if bool(sig) and rr < 1 else "#8b78b8"
            for sig, rr in zip(plot["significant_fdr_0_05"], plot["relative_risk"])
        ]
        fig = go.Figure(go.Bar(
            x=plot["predicted_stress_rate"] * 100,
            y=plot["topic_name"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}%" for v in plot["predicted_stress_rate"] * 100],
            textposition="outside",
        ))
        fig.add_vline(
            x=STRESS_RATE * 100,
            line_dash="dot",
            line_color="#71717a",
            annotation_text=f"Overall {pct(STRESS_RATE, 2)}",
        )
        fig.update_layout(xaxis_title="Predicted Stress rate (%)", yaxis_title="")
        fig_style(fig, height=470, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

        stats_table = topic_final[[
            "topic_name", "posts", "predicted_stress_rate", "relative_risk",
            "odds_ratio", "fisher_p_fdr_bh", "significant_fdr_0_05",
        ]].copy()
        st.dataframe(
            stats_table.style.format({
                "predicted_stress_rate": "{:.2%}", "relative_risk": "{:.2f}×",
                "odds_ratio": "{:.2f}", "fisher_p_fdr_bh": "{:.4f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

    s1, s2 = st.columns(2)
    with s1:
        info_box(
            "Higher association",
            "ChatGPT / OpenAI User Experience: 14.99% predicted Stress, RR 1.77×, OR 1.91, FDR p=0.0013.",
            "warning",
        )
    with s2:
        info_box(
            "Lower association",
            "AI Labs / Industry News: 3.69% predicted Stress, RR 0.37×, OR 0.35, FDR p=0.0013.",
            "success",
        )

    info_box(
        "Overall result",
        "χ²(6)=28.53, p≈0.000075, Cramér’s V=0.103. The Topic × predicted-Stress association is statistically "
        "significant but small in magnitude. Association does not imply causality.",
        "info",
    )

    if topic_keywords is not None:
        with st.expander("Show topic keywords"):
            st.caption(
                "These are the highest-weight terms used to interpret each NMF topic."
            )
            for topic_row in topic_keywords.itertuples():
                st.markdown(
                    f"**T{getattr(topic_row, 'topic', '')} — "
                    f"{getattr(topic_row, 'topic_name', '')}**"
                )
                st.caption(str(getattr(topic_row, "top_terms", "")))


# ============================================================
# 6 — EXAMPLES
# ============================================================

with tabs[5]:

    st.subheader("Examples of developer publications")
    st.caption("Concrete examples help explain what the system predicts and why.")

    if filtered is None or filtered.empty:
        st.info("The row-level generated prediction file is not available in this environment.")
    else:
        st.write(f"**{len(filtered):,} posts match the global filters.**")

        text_col = next((c for c in ["text_clean_basic", "text", "full_text_raw"] if c in filtered.columns), None)
        display_cols = [
            c for c in [
                "platform", "vader_label", "transformer_label", "top_emotion",
                "hybrid_stress_label", "hybrid_stress_probability", "topic_name", text_col,
            ]
            if c is not None and c in filtered.columns
        ]
        st.dataframe(filtered[display_cols].head(50), hide_index=True, use_container_width=True)

        if text_col is not None and "record_id" in filtered.columns:
            st.markdown("#### Inspect one post")
            sample = filtered.head(100).copy()
            labels = {}
            for _, row in sample.iterrows():
                rid = str(row["record_id"])
                platform = str(row.get("platform", "Unknown"))
                preview = str(row.get(text_col, "")).replace("\n", " ")[:55]
                labels[f"{platform} · {rid} · {preview}…"] = rid

            chosen_label = st.selectbox("Select a post", list(labels.keys()))
            chosen_id = labels[chosen_label]
            row = sample[sample["record_id"].astype(str) == chosen_id].iloc[0]

            st.markdown(
                f"""
                **Platform:** {row.get('platform', '—')}  
                **VADER:** {row.get('vader_label', '—')}  
                **Transformer:** {row.get('transformer_label', '—')}  
                **Top emotion:** {row.get('top_emotion', '—')}  
                **Stress:** {row.get('hybrid_stress_label', '—')}  
                **Stress probability:** {float(row.get('hybrid_stress_probability', 0)):.3f}  
                **Topic:** {row.get('topic_name', '—')}
                """
            )
            st.text_area("Post text", value=str(row.get(text_col, "")), height=180, disabled=True)


# ============================================================
# 7 — INSIGHTS & CONCLUSIONS
# ============================================================

with tabs[6]:

    st.subheader("Insights and conclusions")
    st.caption("The final takeaways for someone who has not seen the notebooks.")

    insights = pd.DataFrame(
        [
            ["Sentiment depends on the modeling approach", "37.6% VADER ↔ Transformer agreement", "The two methods capture tone differently."],
            ["Negative sentiment is not Stress", "14.3% / 24.3% negative → Stress", "Stress needs its own modeling layer."],
            ["Predicted-Stress posts have a different emotion profile", "Annoyance, disappointment, confusion, realization and disapproval are enriched", "Specific emotions add information beyond polarity."],
            ["Stress modeling still faces domain shift", "Developer-domain precision 0.111; recall 0.750", "Technical frustration can create false positives."],
            ["Some topics are associated with predicted Stress", "Topic × Stress χ² significant; Cramér’s V=0.103", "The association exists but is small overall."],
            ["ChatGPT / OpenAI User Experience stands out", "14.99% Stress rate; RR 1.77×; FDR p=0.0013", "This topic has a significantly higher association with predicted Stress."],
        ],
        columns=["Insight", "Evidence", "Interpretation"],
    )
    st.dataframe(insights, hide_index=True, use_container_width=True)

    info_box(
        "Final conclusion",
        "The layered pipeline makes the analysis interpretable: sentiment describes general tone, emotions provide finer signals, "
        "Stress is modeled separately, and Topic Modeling adds context about where Stress-related language appears.",
        "success",
    )

    info_box(
        "Limits to keep visible",
        "Predicted Stress is not a diagnosis or prevalence estimate. Synthetic data supported training experiments only. "
        "The 600 developer posts remain the complementary real-domain validation. Topic associations are not causal claims.",
        "warning",
    )
