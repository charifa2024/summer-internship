from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Optional
from html import escape
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Developer Discourse Analysis | Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
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
SENTIMENT = RESULTS / "sentiment"
EMOTIONS = RESULTS / "emotions"
EDA = RESULTS / "eda"
STRESS_TOPICS = RESULTS / "stress_topics"
COMBINED = RESULTS / "combined"


# ============================================================
# BASIC HELPERS
# ============================================================

def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def html(markup: str) -> None:
    """
    Safe custom HTML renderer.

    st.html() is preferred. The fallback is dedented first so that
    Markdown cannot interpret indented HTML as a code block.
    """
    cleaned = dedent(markup).strip()

    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def find_matching_files(
    suffix: str,
    keywords: list[str],
    *,
    limit: int = 12,
) -> list[str]:
    """
    Discover likely notebooks/scripts in the repository without scanning .venv.
    This makes the dashboard presentation-friendly even when notebook names differ.
    """
    candidates: list[Path] = []

    root_files = list(ROOT.glob(f"*{suffix}"))
    candidates.extend(root_files)

    for folder_name in [
        "notebooks",
        "notebook",
        "analysis",
        "experiments",
        "scripts",
    ]:
        folder = ROOT / folder_name
        if folder.exists() and folder.is_dir():
            candidates.extend(folder.rglob(f"*{suffix}"))

    keywords_lower = [keyword.lower() for keyword in keywords]

    matched: list[Path] = []
    for path in candidates:
        relative = str(path.relative_to(ROOT))
        text = relative.lower()

        if any(keyword in text for keyword in keywords_lower):
            matched.append(path)

    # Keep unique relative paths.
    seen = set()
    output = []
    for path in sorted(matched):
        relative = str(path.relative_to(ROOT))
        if relative in seen:
            continue
        seen.add(relative)
        output.append(relative)

        if len(output) >= limit:
            break

    return output


# ============================================================
# LOAD FINAL RESULTS
# ============================================================

final_findings = load_json(FINAL / "final_findings.json")

sentiment_distribution = load_csv(
    FINAL / "sentiment_distribution.csv"
)

negative_vs_stress = load_csv(
    FINAL / "negative_sentiment_vs_stress.csv"
)

emotion_stress = load_csv(
    FINAL / "emotion_stress_association.csv"
)

topic_final = load_csv(
    FINAL / "topic_stress_final.csv"
)

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

# Critical-evaluation artifacts used directly in the final dashboard.
sentiment_method_eval = load_csv(
    SENTIMENT / "llm_assisted_method_evaluation.csv"
)

go_threshold_strategy = load_json(
    EMOTIONS / "goemotions_threshold_strategy_comparison.json"
)

final_source_summary = load_csv(
    EDA / "source_summary.csv"
)

dreaddit_candidate_comparison = load_csv(
    EMOTIONS / "dreaddit_stress_model_comparison.csv"
)

stress_real_eval = load_json(
    EMOTIONS / "stress_real_domain_evaluation.json"
)

stress_real_comparison = load_csv(
    EMOTIONS / "stress_real_domain_model_comparison.csv"
)

stress_real_predictions = load_csv(
    EMOTIONS / "stress_real_domain_predictions.csv"
)


# ============================================================
# VALIDATED PROJECT VALUES
# ============================================================

COLLECTED_POSTS = 5406
ANALYSIS_POSTS = int(final_findings.get("records", 2666))
REMOVED_POSTS = COLLECTED_POSTS - ANALYSIS_POSTS
RETENTION_RATE = ANALYSIS_POSTS / COLLECTED_POSTS

# Verified sequential preparation audit used in the final report and dashboard.
AI_RELEVANT_POSTS = 2726
UNIQUE_AI_RELEVANT_POSTS = 2694
LOW_RELEVANCE_REMOVED = 2680
RELEVANT_DUPLICATES_REMOVED = 32
SHORT_POSTS_REMOVED = 28
GLOBAL_DUPLICATE_ROWS_DETECTED = 37

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

# Raw collection provenance reconstructed from the frozen internship collection summary.
# These totals sum exactly to the 5,406 collected records; they are collection counts,
# not the post-cleaning composition.
RAW_SOURCE_COUNTS = pd.DataFrame(
    [
        ["Arctic Shift Reddit", 305],
        ["Hugging Face Reddit", 2000],
        ["divde/sentiment_posts", 141],
        ["Hacker News", 236],
        ["GitHub Issues", 1616],
        ["Stack Overflow", 1108],
    ],
    columns=["Raw source", "Collected records"],
)
RAW_SOURCE_COUNTS["Share of collection"] = (
    RAW_SOURCE_COUNTS["Collected records"] / COLLECTED_POSTS
)


# ============================================================
# BUILD ROW-LEVEL EXPLORER
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
            column
            for column in [
                "record_id",
                "topic",
                "topic_name",
                "topic_weight",
            ]
            if column in topic_assignments.columns
        ]

        if len(merge_cols) > 1:
            explorer = explorer.merge(
                topic_assignments[merge_cols],
                on="record_id",
                how="left",
                validate="one_to_one",
            )


# ============================================================
# PROFESSIONAL VISUAL SYSTEM
# ============================================================

html(
    """
    <style>
    :root {
        color-scheme: light !important;

        --bg: #f6f4f9;
        --surface: #ffffff;
        --surface-2: #fbfafc;

        --text: #18151d;
        --muted: #665f6e;
        --subtle: #8a8292;
        --border: #e4deea;

        --purple: #5c3198;
        --purple-2: #7849b8;
        --purple-soft: #efe8f8;

        --rose: #c84c6f;
        --rose-soft: #f9e8ed;

        --blue: #3567b8;
        --blue-soft: #eaf0fb;

        --green: #1f8765;
        --green-soft: #e9f5f0;

        --amber: #a66d1b;
        --amber-soft: #fbf0dc;

        --shadow: 0 8px 22px rgba(43, 31, 59, 0.045);
    }

    html,
    body {
        color-scheme: light !important;
        background: var(--bg) !important;
    }

    .stApp {
        background:
            linear-gradient(
                180deg,
                rgba(92, 49, 152, 0.035) 0,
                transparent 260px
            ),
            var(--bg);
        color: var(--text) !important;
    }

    /*
       IMPORTANT:
       Reserve real vertical space for Streamlit's native header.
       This prevents the toolbar/navigation from covering the dashboard.
    */
    header[data-testid="stHeader"] {
        height: 3.55rem !important;
        background: rgba(246, 244, 249, 0.97) !important;
        border-bottom: 1px solid rgba(228, 222, 234, 0.75);
        backdrop-filter: blur(10px);
    }

    div[data-testid="stToolbar"] {
        top: 0.45rem !important;
    }

    .block-container {
        width: min(100%, 1360px) !important;
        max-width: 1360px !important;

        /*
           Increased top/side/bottom spacing.
           The top padding is intentionally generous because the native
           Streamlit header is fixed.
        */
        padding-top: 5.15rem !important;
        padding-left: clamp(1.4rem, 4vw, 3.8rem) !important;
        padding-right: clamp(1.4rem, 4vw, 3.8rem) !important;
        padding-bottom: 5rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    /* --------------------------------------------------------
       TYPOGRAPHY
    -------------------------------------------------------- */

    /*
       Apply the dashboard font at the app level and let normal text inherit it.
       IMPORTANT: do NOT target every span/div. Streamlit renders navigation,
       expander and sidebar controls with Material Symbols ligatures; overriding
       their font turns icons such as "keyboard_arrow_right" into visible text.
    */
    .stApp {
        font-family:
            Inter,
            ui-sans-serif,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .stApp p,
    .stApp li,
    .stApp label,
    .stApp input,
    .stApp textarea,
    .stApp button {
        font-family: inherit;
    }

    /*
       Protect Streamlit's Material Symbols icons even if browser/theme CSS
       tries to inherit the dashboard font.
    */
    span[data-testid="stIconMaterial"],
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family:
            "Material Symbols Rounded",
            "Material Symbols Outlined" !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: inherit;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr;
        -webkit-font-feature-settings: "liga" !important;
        font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        letter-spacing: -0.024em;
        opacity: 1 !important;
    }

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] em {
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        opacity: 1 !important;
    }

    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
        opacity: 1 !important;
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #f0ebf5 !important;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2.4rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #332b3d !important;
        -webkit-text-fill-color: #332b3d !important;
    }

    .sidebar-brand {
        padding: 0.45rem 0 0.8rem;
    }

    .sidebar-kicker {
        color: var(--purple);
        font-size: 0.67rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 800;
    }

    .sidebar-title {
        color: var(--text);
        font-size: 1.25rem;
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-top: 0.2rem;
    }

    .sidebar-copy {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.45;
        margin-top: 0.42rem;
    }

    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: clamp(1.5rem, 2.7vw, 2.15rem);
        box-shadow: var(--shadow);
        margin-bottom: 1.35rem;
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        right: -90px;
        top: -95px;
        background: rgba(92, 49, 152, 0.06);
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: var(--purple);
        background: var(--purple-soft);
        border: 1px solid #dfd2ee;
        padding: 0.3rem 0.58rem;
        border-radius: 999px;
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.075em;
        text-transform: uppercase;
    }

    .hero-title {
        color: var(--text);
        font-size: clamp(2.15rem, 4vw, 3.65rem);
        line-height: 0.98;
        font-weight: 840;
        letter-spacing: -0.055em;
        max-width: 980px;
        margin-top: 0.85rem;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.65;
        max-width: 940px;
        margin-top: 0.85rem;
    }

    /* --------------------------------------------------------
       STAGE HEADER
    -------------------------------------------------------- */

    .stage-header {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr);
        gap: 0.85rem;
        align-items: start;
        margin: 1.55rem 0 1rem;
    }

    .stage-number {
        min-width: 48px;
        height: 48px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        background: var(--purple);
        color: white;
        font-size: 0.86rem;
        font-weight: 800;
        box-shadow: 0 5px 12px rgba(92, 49, 152, 0.12);
    }

    .stage-eyebrow {
        color: var(--purple);
        font-size: 0.66rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.085em;
    }

    .stage-title {
        color: var(--text);
        font-size: clamp(1.55rem, 2.4vw, 2rem);
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.12;
        margin-top: 0.12rem;
    }

    .stage-description {
        color: var(--muted);
        max-width: 970px;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 0.32rem;
    }

    /* --------------------------------------------------------
       KPI CARDS
    -------------------------------------------------------- */

    .metric-grid {
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(min(100%, 215px), 1fr));
        gap: 1rem;
        margin: 1rem 0 1.5rem;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        min-height: 122px;
        padding: 1.05rem 1.1rem;
        box-shadow: 0 5px 16px rgba(43, 31, 59, 0.035);
        position: relative;
        overflow: hidden;
    }

    .metric-card::after {
        content: "";
        position: absolute;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        right: -30px;
        bottom: -30px;
        background: var(--soft, var(--purple-soft));
    }

    .metric-line {
        width: 32px;
        height: 4px;
        border-radius: 999px;
        background: var(--accent, var(--purple));
        margin-bottom: 0.72rem;
    }

    .metric-label {
        color: var(--subtle);
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.078em;
        font-weight: 800;
    }

    .metric-value {
        color: var(--text);
        font-size: clamp(1.75rem, 2.8vw, 2.25rem);
        line-height: 1.05;
        font-weight: 840;
        letter-spacing: -0.045em;
        margin-top: 0.3rem;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.77rem;
        line-height: 1.42;
        margin-top: 0.34rem;
        padding-right: 0.65rem;
    }

    /* --------------------------------------------------------
       ANALYSIS / PIPELINE CARDS
    -------------------------------------------------------- */

    .flow-grid {
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(min(100%, 240px), 1fr));
        gap: 0.9rem;
        margin: 1rem 0 1.3rem;
    }

    .flow-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 1rem;
    }

    .flow-step {
        color: var(--purple);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .flow-title {
        color: var(--text);
        font-size: 0.98rem;
        font-weight: 770;
        line-height: 1.25;
        margin-top: 0.35rem;
    }

    .flow-body {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.48;
        margin-top: 0.35rem;
    }

    /* --------------------------------------------------------
       FIGURE LABELS
    -------------------------------------------------------- */

    .figure-label {
        margin: 1.4rem 0 0.6rem;
    }

    .figure-number {
        color: var(--purple);
        font-size: 0.64rem;
        text-transform: uppercase;
        letter-spacing: 0.085em;
        font-weight: 800;
    }

    .figure-title {
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 760;
        line-height: 1.25;
        margin-top: 0.14rem;
    }

    .figure-subtitle {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.48;
        margin-top: 0.2rem;
        max-width: 1000px;
    }

    /* --------------------------------------------------------
       INTERPRETATION / CONCLUSION BOXES
    -------------------------------------------------------- */

    .interpretation {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent, var(--purple));
        border-radius: 10px;
        padding: 0.82rem 0.95rem;
        margin: 0.72rem 0 1rem;
    }

    .interpretation-label {
        color: var(--accent, var(--purple));
        font-size: 0.64rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 820;
    }

    .interpretation-text {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
        margin-top: 0.3rem;
    }

    .interpretation-text strong {
        color: var(--text);
    }

    .conclusion-grid {
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(min(100%, 290px), 1fr));
        gap: 0.95rem;
        margin: 0.8rem 0 1.35rem;
    }

    .conclusion-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 1.05rem;
        box-shadow: 0 5px 16px rgba(43, 31, 59, 0.03);
    }

    .conclusion-chip {
        display: inline-block;
        color: var(--accent, var(--purple));
        background: var(--soft, var(--purple-soft));
        border-radius: 999px;
        padding: 0.25rem 0.52rem;
        font-size: 0.63rem;
        font-weight: 800;
        letter-spacing: 0.065em;
        text-transform: uppercase;
    }

    .conclusion-title {
        color: var(--text);
        font-size: 1rem;
        font-weight: 780;
        line-height: 1.25;
        margin-top: 0.55rem;
    }

    .conclusion-body {
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.5;
        margin-top: 0.35rem;
    }

    /* --------------------------------------------------------
       METHOD BOX
    -------------------------------------------------------- */

    .method-box {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0 1.15rem;
    }

    .method-title {
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 770;
    }

    .method-body {
        color: var(--muted);
        font-size: 0.79rem;
        line-height: 1.55;
        margin-top: 0.32rem;
    }

    /* --------------------------------------------------------
       PLOTLY / DATAFRAME / NATIVE WIDGETS
    -------------------------------------------------------- */

    div[data-testid="stPlotlyChart"] {
        width: 100% !important;
        min-width: 0 !important;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 0.35rem;
        box-shadow: 0 5px 16px rgba(43, 31, 59, 0.03);
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] {
        width: 100% !important;
        max-width: 100% !important;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 11px;
        overflow-x: auto !important;
    }

    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] span {
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
        opacity: 1 !important;
        font-weight: 650;
    }

    div[data-baseweb="select"] > div {
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }

    div[data-baseweb="tag"] {
        background: var(--purple-soft) !important;
    }

    div[data-baseweb="tag"] span {
        color: var(--purple) !important;
        -webkit-text-fill-color: var(--purple) !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextArea"] textarea:disabled {
        background: var(--surface) !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        opacity: 1 !important;
        caret-color: var(--text) !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #97909e !important;
        -webkit-text-fill-color: #97909e !important;
        opacity: 1 !important;
    }

    div[data-testid="stExpander"] {
        background: var(--surface);
        border-color: var(--border);
        border-radius: 11px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        min-height: 3.15rem;
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
        gap: 0.55rem;
    }

    div[data-testid="stExpander"] summary p {
        margin: 0 !important;
        line-height: 1.35 !important;
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p {
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] p {
        color: #3b3346 !important;
        -webkit-text-fill-color: #3b3346 !important;
    }

    /* --------------------------------------------------------
       RESPONSIVENESS
    -------------------------------------------------------- */

    div[data-testid="stColumn"] {
        min-width: 0 !important;
    }

    @media (max-width: 1100px) {
        .block-container {
            padding-top: 5.2rem !important;
            padding-left: 1.6rem !important;
            padding-right: 1.6rem !important;
        }

        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 1rem !important;
        }

        div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"] {
            flex: 1 1 calc(50% - 1rem) !important;
            width: calc(50% - 1rem) !important;
            min-width: 300px !important;
        }
    }

    @media (max-width: 720px) {
        .block-container {
            padding-top: 4.8rem !important;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
            padding-bottom: 3rem !important;
        }

        div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 0 !important;
        }

        .metric-grid,
        .flow-grid,
        .conclusion-grid {
            grid-template-columns: 1fr;
        }

        .hero {
            padding: 1.25rem;
        }

        .stage-header {
            grid-template-columns: 42px minmax(0, 1fr);
        }

        .stage-number {
            min-width: 42px;
            height: 42px;
        }
    }
    </style>
    """
)


# ============================================================
# COMPONENT HELPERS
# ============================================================

def hero() -> None:
    html(
        """
        <div class="hero">
            <div class="hero-kicker">Internship research dashboard</div>
            <div class="hero-title">Analysis of Developer-Oriented Discussions on AI Tools</div>
            <div class="hero-subtitle">
                Sentiment, Emotions, Stress-Related Language, and Topic Modeling in Public Technical Discussions.<br>
                <strong>Research question:</strong> How do developer-oriented public discussions about AI tools express
                sentiment, emotions, and stress-related language, and which discussion topics are associated
                with model-predicted Stress?
            </div>
        </div>
        """
    )


def stage_header(
    number: str,
    eyebrow: str,
    title: str,
    description: str,
) -> None:
    html(
        f"""
        <div class="stage-header">
            <div class="stage-number">{escape(number)}</div>
            <div>
                <div class="stage-eyebrow">{escape(eyebrow)}</div>
                <div class="stage-title">{escape(title)}</div>
                <div class="stage-description">{escape(description)}</div>
            </div>
        </div>
        """
    )


def metric_grid(items: list[dict]) -> None:
    cards = []

    for item in items:
        cards.append(
            f"""
            <div
                class="metric-card"
                style="
                    --accent:{item.get('accent', '#5c3198')};
                    --soft:{item.get('soft', '#efe8f8')};
                "
            >
                <div class="metric-line"></div>
                <div class="metric-label">{escape(str(item['label']))}</div>
                <div class="metric-value">{escape(str(item['value']))}</div>
                <div class="metric-note">{escape(str(item.get('note', '')))}</div>
            </div>
            """
        )

    html(
        f"""
        <div class="metric-grid">
            {''.join(cards)}
        </div>
        """
    )


def method_box(title: str, body: str) -> None:
    html(
        f"""
        <div class="method-box">
            <div class="method-title">{escape(title)}</div>
            <div class="method-body">{escape(body)}</div>
        </div>
        """
    )


def figure_label(
    number: str,
    title: str,
    subtitle: str,
) -> None:
    html(
        f"""
        <div class="figure-label">
            <div class="figure-number">Figure {escape(number)}</div>
            <div class="figure-title">{escape(title)}</div>
            <div class="figure-subtitle">{escape(subtitle)}</div>
        </div>
        """
    )


def interpretation(
    label: str,
    text: str,
    *,
    accent: str = "#5c3198",
) -> None:
    html(
        f"""
        <div class="interpretation" style="--accent:{accent};">
            <div class="interpretation-label">{escape(label)}</div>
            <div class="interpretation-text">{text}</div>
        </div>
        """
    )


def plot_style(
    fig: go.Figure,
    *,
    height: int = 390,
    showlegend: bool = True,
) -> go.Figure:
    fig.update_layout(
        height=height,
        autosize=True,
        margin=dict(
            l=48,
            r=58,
            t=48,
            b=52,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family="Inter, Arial, sans-serif",
            color="#4e4756",
            size=12,
        ),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            font=dict(color="#4e4756"),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#ded7e5",
            font_color="#18151d",
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#f0ecf3",
        zeroline=False,
        linecolor="#e8e2ed",
        tickfont=dict(color="#665f6e"),
        title_font=dict(color="#665f6e"),
        automargin=True,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f0ecf3",
        zeroline=False,
        linecolor="#e8e2ed",
        tickfont=dict(color="#665f6e"),
        title_font=dict(color="#665f6e"),
        automargin=True,
    )

    return fig


def technical_trace(
    *,
    stage_name: str,
    method: str,
    inputs: list[str],
    outputs: list[str],
    keywords: list[str],
) -> None:
    notebooks = find_matching_files(
        ".ipynb",
        keywords,
    )

    scripts = find_matching_files(
        ".py",
        keywords,
    )

    with st.expander(
        f"Technical traceability — {stage_name}"
    ):
        st.markdown(f"**Method / approach:** {method}")

        st.markdown("**Primary inputs**")
        for path in inputs:
            st.code(path, language=None)

        st.markdown("**Generated / frozen outputs**")
        for path in outputs:
            st.code(path, language=None)

        st.markdown("**Matching notebooks detected in the repository**")
        if notebooks:
            for notebook in notebooks:
                st.code(notebook, language=None)
        else:
            st.caption(
                "No notebook filename matching this stage was automatically detected. "
                "The result paths above remain the reliable traceability reference."
            )

        st.markdown("**Matching scripts detected in the repository**")
        if scripts:
            for script in scripts:
                st.code(script, language=None)
        else:
            st.caption(
                "No matching script filename was automatically detected."
            )


def conclusion_cards(cards: list[dict]) -> None:
    blocks = []

    for card in cards:
        blocks.append(
            f"""
            <div
                class="conclusion-card"
                style="
                    --accent:{card.get('accent', '#5c3198')};
                    --soft:{card.get('soft', '#efe8f8')};
                "
            >
                <div class="conclusion-chip">{escape(card['chip'])}</div>
                <div class="conclusion-title">{escape(card['title'])}</div>
                <div class="conclusion-body">{escape(card['body'])}</div>
            </div>
            """
        )

    html(
        f"""
        <div class="conclusion-grid">
            {''.join(blocks)}
        </div>
        """
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-kicker">AI Developer Pulse</div>
            <div class="sidebar-title">Presentation flow</div>
            <div class="sidebar-copy">
                Final audited flow · method → result → validation → critique → decision.
            </div>
        </div>
        """
    )

    pages = [
        "0 · Executive Overview",
        "1 · Data & Cleaning",
        "2 · Sentiment Analysis",
        "3 · Emotion Analysis",
        "4 · Stress & Critical Validation",
        "5 · Integrated Analysis",
        "6 · Topic Modeling & Statistics",
        "7 · Evidence Explorer",
        "8 · Final Conclusions",
    ]

    page = st.radio(
        "Navigate",
        pages,
        label_visibility="collapsed",
    )

    st.divider()

    st.caption(
        "Explorer filters affect the row-level current view and evidence page. "
        "Frozen research metrics remain unchanged."
    )

    selected_platform: list[str] = []
    selected_stress: list[str] = []
    selected_topic: list[str] = []
    search_text = ""

    if explorer is not None:
        if "platform" in explorer.columns:
            platform_options = sorted(
                explorer["platform"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_platform = st.multiselect(
                "Platform",
                platform_options,
            )

        if "hybrid_stress_label" in explorer.columns:
            stress_options = sorted(
                explorer["hybrid_stress_label"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_stress = st.multiselect(
                "Stress label",
                stress_options,
            )

        if "topic_name" in explorer.columns:
            topic_options = sorted(
                explorer["topic_name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_topic = st.multiselect(
                "Topic",
                topic_options,
            )

        search_text = st.text_input(
            "Search publications",
            placeholder="ChatGPT, Copilot, GPU...",
        )


# ============================================================
# FILTERED ROW-LEVEL VIEW
# ============================================================

filtered = None

if explorer is not None:
    filtered = explorer.copy()

    if selected_platform and "platform" in filtered.columns:
        filtered = filtered[
            filtered["platform"]
            .astype(str)
            .isin(selected_platform)
        ]

    if (
        selected_stress
        and "hybrid_stress_label" in filtered.columns
    ):
        filtered = filtered[
            filtered["hybrid_stress_label"]
            .astype(str)
            .isin(selected_stress)
        ]

    if selected_topic and "topic_name" in filtered.columns:
        filtered = filtered[
            filtered["topic_name"]
            .astype(str)
            .isin(selected_topic)
        ]

    text_col = next(
        (
            column
            for column in [
                "text_clean_basic",
                "text",
                "full_text_raw",
            ]
            if column in filtered.columns
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


# ============================================================
# HEADER
# ============================================================

hero()


# ============================================================
# PAGE 0 — EXECUTIVE OVERVIEW
# ============================================================

if page == "0 · Executive Overview":

    stage_header(
        "00",
        "Project overview",
        "Research question, analytical logic and headline findings",
        (
            "This opening page presents the complete project story before "
            "moving into the notebooks, models and detailed results."
        ),
    )

    metric_grid(
        [
            {
                "label": "Raw records",
                "value": f"{COLLECTED_POSTS:,}",
                "note": "initially collected",
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "label": "Analysis-ready posts",
                "value": f"{ANALYSIS_POSTS:,}",
                "note": f"{pct(RETENTION_RATE, 1)} retained after preparation",
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "label": "Model-predicted Stress",
                "value": f"{STRESS_POSTS:,}",
                "note": f"{pct(STRESS_RATE, 2)} of final corpus",
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "label": "Final topics",
                "value": "7",
                "note": "NMF discussion themes",
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
        ]
    )

    html(
        """
        <div class="flow-grid">
            <div class="flow-card">
                <div class="flow-step">Step 1</div>
                <div class="flow-title">Data preparation</div>
                <div class="flow-body">
                    Collect, clean, deduplicate and freeze one common
                    2,666-post scientific corpus.
                </div>
            </div>

            <div class="flow-card">
                <div class="flow-step">Step 2</div>
                <div class="flow-title">Sentiment</div>
                <div class="flow-body">
                    Compare VADER lexical polarity with a contextual
                    Transformer instead of forcing a single sentiment view.
                </div>
            </div>

            <div class="flow-card">
                <div class="flow-step">Step 3</div>
                <div class="flow-title">Emotions</div>
                <div class="flow-body">
                    Use multi-label GoEmotions to move from broad polarity
                    to specific affective signals.
                </div>
            </div>

            <div class="flow-card">
                <div class="flow-step">Step 4</div>
                <div class="flow-title">Stress</div>
                <div class="flow-body">
                    Model Stress separately with Dreaddit, hybrid augmentation
                    and complementary developer-domain validation.
                </div>
            </div>

            <div class="flow-card">
                <div class="flow-step">Step 5</div>
                <div class="flow-title">Integrated analysis</div>
                <div class="flow-body">
                    Join sentiment, emotions and Stress at post level to
                    test the central distinction: negative ≠ Stress.
                </div>
            </div>

            <div class="flow-card">
                <div class="flow-step">Step 6</div>
                <div class="flow-title">Topics & statistics</div>
                <div class="flow-body">
                    Learn NMF topics on all posts, then test Topic ×
                    model-predicted Stress association statistically.
                </div>
            </div>
        </div>
        """
    )

    interpretation(
        "How to read the results",
        (
            "This dashboard separates <strong>model outputs</strong> from established facts. "
            "Sentiment and emotion labels are computational predictions; Stress is reported as "
            "<strong>model-predicted Stress-related language</strong>, not a diagnosis or prevalence estimate. "
            "Every modeling page therefore ends with an explicit critical review of validation, "
            "limitations and the final analytical decision."
        ),
        accent="#a66d1b",
    )

    conclusion_cards(
        [
            {
                "chip": "Finding 1",
                "title": "Negative sentiment cannot replace Stress detection",
                "body": (
                    "Only 14.3% of VADER-negative posts and 24.3% of "
                    "Transformer-negative posts were classified as Stress."
                ),
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "chip": "Finding 2",
                "title": "Stress-classified posts show a distinct emotion profile",
                "body": (
                    "Annoyance, disappointment, confusion, realization and "
                    "disapproval are descriptively enriched in model-predicted Stress posts."
                ),
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "chip": "Finding 3",
                "title": "Discussion context matters, but the effect is small",
                "body": (
                    "Topic × model-predicted Stress association is statistically significant, "
                    "with Cramér’s V = 0.103."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
        ]
    )

    interpretation(
        "Research answer",
        (
            "The collected developer-oriented technical discussions exhibit <strong>distinct but connected "
            "sentiment, model-predicted emotion-related, and stress-related linguistic signals</strong>. "
            "Negative polarity alone is insufficient to identify model-predicted Stress. Predicted emotion "
            "labels and discussion context add descriptive information, while the Stress classifier remains "
            "limited by developer-domain false positives and topic associations must not be interpreted causally."
        ),
        accent="#1f8765",
    )


# ============================================================
# PAGE 1 — DATA & CLEANING
# ============================================================

elif page == "1 · Data & Cleaning":

    stage_header(
        "01",
        "Data preparation",
        "From 5,406 collected records to a frozen 2,666-post corpus",
        (
            "Every downstream model is applied to the same analysis-ready corpus so that "
            "sentiment, emotions, Stress and topic results can be compared consistently."
        ),
    )

    metric_grid(
        [
            {
                "label": "Collected",
                "value": f"{COLLECTED_POSTS:,}",
                "note": "raw records",
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "label": "Analysis-ready",
                "value": f"{ANALYSIS_POSTS:,}",
                "note": "frozen final corpus",
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "label": "Excluded during preparation",
                "value": f"{REMOVED_POSTS:,}",
                "note": "aggregate raw-to-final difference",
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "label": "Retention",
                "value": pct(RETENTION_RATE, 1),
                "note": "of initially collected records",
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
        ]
    )

    method_box(
        "Preparation approach",
        (
            "The preparation stage preserves the original text, creates cleaned text fields, applies strict "
            "AI-relevance rules, removes exact-text duplicates among the remaining relevant records, excludes "
            "posts containing fewer than five words, and freezes `text_clean_basic` for downstream NLP. "
            "The final technical corpus is stored in `data/processed/analysis_ready_posts.jsonl`."
        ),
    )

    figure_label(
        "1.1",
        "Raw collection provenance",
        (
            "Where the 5,406 collected records came from before relevance filtering, "
            "quality checks, deduplication and corpus freezing."
        ),
    )

    st.dataframe(
        RAW_SOURCE_COUNTS.style.format(
            {
                "Collected records": "{:,}",
                "Share of collection": "{:.1%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    interpretation(
        "Collection result",
        (
            "The raw corpus is deliberately multi-source, but it is <strong>not balanced by source</strong>. "
            "GitHub Issues and the Hugging Face Reddit corpus contribute most of the collected records. "
            "This improves language variety while also creating a sampling limitation that must be kept "
            "in mind when interpreting corpus-wide percentages."
        ),
        accent="#a66d1b",
    )

    if final_source_summary is not None:
        with st.expander("Final retained records by technical source"):
            source_view = final_source_summary.copy()
            if "percentage" in source_view.columns:
                source_view["percentage"] = source_view["percentage"].astype(float) / 100.0
                st.dataframe(
                    source_view.style.format({"records": "{:,}", "percentage": "{:.1%}"}),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.dataframe(source_view, hide_index=True, use_container_width=True)

    if explorer is not None and "platform" in explorer.columns:
        platform_counts = (
            explorer["platform"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )

        platform_counts.columns = [
            "platform",
            "posts",
        ]

        platform_counts = platform_counts.sort_values(
            "posts",
            ascending=True,
        )

        figure_label(
            "1.2",
            "Final corpus distribution by platform",
            (
                "Number of analysis-ready posts available from each developer-discussion "
                "source in the final row-level corpus."
            ),
        )

        fig = go.Figure(
            go.Bar(
                x=platform_counts["posts"],
                y=platform_counts["platform"],
                orientation="h",
                marker_color="#5c3198",
                text=platform_counts["posts"],
                textposition="outside",
                cliponaxis=False,
            )
        )

        fig.update_layout(
            xaxis_title="Analysis-ready posts",
            yaxis_title="",
        )

        plot_style(
            fig,
            height=max(
                330,
                min(
                    620,
                    65 * len(platform_counts),
                ),
            ),
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None,
            config={"displayModeBar": False},
        )

        interpretation(
            "Interpretation",
            (
                "The platform distribution documents the composition of the final corpus. "
                "All following analyses operate on these same <strong>2,666 records</strong>, "
                "which avoids comparing different data subsets across modeling stages."
            ),
        )

        if "source" in explorer.columns:
            source_counts = (
                explorer["source"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
            )
            top_source_name = str(source_counts.index[0])
            top_source_count = int(source_counts.iloc[0])
            top_source_share = top_source_count / len(explorer)

            figure_label(
                "1.C",
                "Critical review — corpus representativeness",
                "Why the final corpus is useful, and what it cannot represent perfectly.",
            )

            conclusion_cards(
                [
                    {
                        "chip": "Strength",
                        "title": "One frozen corpus across all models",
                        "body": (
                            "All downstream analyses use the same 2,666 record IDs, "
                            "which preserves row-level comparability."
                        ),
                        "accent": "#1f8765",
                        "soft": "#e9f5f0",
                    },
                    {
                        "chip": "Sampling limitation",
                        "title": "The corpus is source-concentrated",
                        "body": (
                            f"The largest single source contributes {top_source_count:,} posts "
                            f"({pct(top_source_share, 1)} of the final corpus)."
                        ),
                        "accent": "#a66d1b",
                        "soft": "#fbf0dc",
                    },
                    {
                        "chip": "Population limitation",
                        "title": "Not a random sample of all developers",
                        "body": (
                            "The data come from public developer-oriented and technical communities; "
                            "author profession and population representativeness cannot be guaranteed."
                        ),
                        "accent": "#c84c6f",
                        "soft": "#f9e8ed",
                    },
                    {
                        "chip": "Decision",
                        "title": "Interpret the corpus as public technical discourse",
                        "body": (
                            "Results describe the collected developer-oriented AI discussions, "
                            "not the psychological state of the global developer population."
                        ),
                        "accent": "#3567b8",
                        "soft": "#eaf0fb",
                    },
                ]
            )

    figure_label(
        "1.3",
        "Verified sequential data-preparation funnel",
        "Exact filtering sequence used to construct the frozen 2,666-post final technical corpus.",
    )

    cleaning_table = pd.DataFrame(
        [
            [
                "Raw collection",
                "Initial multi-source public technical collection",
                f"{COLLECTED_POSTS:,}",
                "—",
            ],
            [
                "Strict AI relevance",
                "Keep posts matching explicit AI-tool and terminology patterns",
                f"{AI_RELEVANT_POSTS:,}",
                f"{LOW_RELEVANCE_REMOVED:,}",
            ],
            [
                "Exact-text deduplication",
                "Remove exact duplicates among the remaining AI-relevant posts",
                f"{UNIQUE_AI_RELEVANT_POSTS:,}",
                f"{RELEVANT_DUPLICATES_REMOVED:,}",
            ],
            [
                "Minimum length ≥ 5 words",
                "Remove posts with insufficient textual content for downstream NLP",
                f"{ANALYSIS_POSTS:,}",
                f"{SHORT_POSTS_REMOVED:,}",
            ],
        ],
        columns=[
            "Sequential stage",
            "Purpose",
            "Remaining records",
            "Removed at this stage",
        ],
    )

    st.dataframe(
        cleaning_table,
        hide_index=True,
        use_container_width=True,
    )

    interpretation(
        "Verified cleaning audit",
        (
            "The sequential audit removes <strong>2,680 low-relevance records</strong>, then "
            "<strong>32 exact-text duplicates</strong> among the remaining relevant posts, and finally "
            "<strong>28 posts containing fewer than five words</strong>. This yields the frozen "
            "<strong>2,666-post final technical corpus</strong>. The broader diagnostic audit detected "
            "37 duplicate rows globally; five had already been removed by relevance filtering, which is why "
            "32 duplicates are removed at the sequential deduplication step. Missing dates are retained and "
            "are not used as an eligibility criterion."
        ),
        accent="#a66d1b",
    )

    technical_trace(
        stage_name="Data preparation",
        method=(
            "Cleaning, filtering, deduplication, text normalization and corpus freeze."
        ),
        inputs=[
            "Raw / collected developer-discussion datasets",
        ],
        outputs=[
            "data/processed/analysis_ready_posts.jsonl",
        ],
        keywords=[
            "clean",
            "preprocess",
            "prepare",
            "analysis_ready",
            "data",
        ],
    )


# ============================================================
# PAGE 2 — SENTIMENT
# ============================================================

elif page == "2 · Sentiment Analysis":

    stage_header(
        "02",
        "Sentiment analysis",
        "Compare lexical polarity and contextual sentiment",
        (
            "Two approaches are retained because they capture tone differently: "
            "VADER is lexicon-based, while the Transformer uses contextual language representations."
        ),
    )

    method_box(
        "Approach",
        (
            "VADER assigns sentiment from a sentiment lexicon and linguistic rules. "
            "The Transformer evaluates the sentence in context. Both produce Positive, "
            "Neutral or Negative labels on the same 2,666-post corpus."
        ),
    )

    metric_grid(
        [
            {
                "label": "VADER ↔ Transformer agreement",
                "value": pct(AGREEMENT, 1),
                "note": "1,003 matching labels out of 2,666",
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "label": "VADER macro F1",
                "value": "0.454",
                "note": "LLM-assisted reference sample",
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "label": "Transformer macro F1",
                "value": "0.510",
                "note": "LLM-assisted reference sample",
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
        ]
    )

    figure_label(
        "2.1",
        "Sentiment distribution by modeling approach",
        (
            "Final VADER and Transformer label distributions over the same "
            "2,666-post final technical corpus."
        ),
    )

    if sentiment_distribution is not None:
        pivot = (
            sentiment_distribution
            .pivot(
                index="label",
                columns="method",
                values="rate",
            )
            .reindex(
                ["Negative", "Neutral", "Positive"]
            )
        )

        fig = go.Figure()

        if "VADER" in pivot.columns:
            fig.add_bar(
                name="VADER",
                x=pivot.index,
                y=pivot["VADER"] * 100,
                marker_color="#c84c6f",
                text=[
                    f"{value:.1f}%"
                    for value in pivot["VADER"] * 100
                ],
                textposition="outside",
                cliponaxis=False,
            )

        if "Transformer" in pivot.columns:
            fig.add_bar(
                name="Transformer",
                x=pivot.index,
                y=pivot["Transformer"] * 100,
                marker_color="#5c3198",
                text=[
                    f"{value:.1f}%"
                    for value in pivot["Transformer"] * 100
                ],
                textposition="outside",
                cliponaxis=False,
            )

        fig.update_layout(
            barmode="group",
            xaxis_title="Sentiment class",
            yaxis_title="Share of posts (%)",
        )

        plot_style(
            fig,
            height=430,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None,
            config={"displayModeBar": False},
        )

    interpretation(
        "Result",
        (
            "VADER classifies <strong>63.8% of posts as Positive</strong>, whereas the "
            "Transformer classifies <strong>64.0% as Neutral</strong>. Their low 37.6% "
            "agreement confirms that model choice strongly changes the observed sentiment picture."
        ),
    )

    figure_label(
        "2.2",
        "Sentiment reference-sample evaluation",
        (
            "Performance on the available 150-post LLM-assisted sentiment reference sample; "
            "147 posts were evaluable."
        ),
    )

    sentiment_eval = pd.DataFrame(
        [
            [
                "VADER",
                0.4898,
                0.4535,
            ],
            [
                "Transformer",
                0.5102,
                0.5099,
            ],
        ],
        columns=[
            "Model",
            "Accuracy",
            "Macro F1",
        ],
    )

    st.dataframe(
        sentiment_eval.style.format(
            {
                "Accuracy": "{:.3f}",
                "Macro F1": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    interpretation(
        "Conclusion",
        (
            "The Transformer performs slightly better on the reference sample, but the "
            "reference labels are <strong>LLM-assisted rather than human gold-standard labels</strong>. "
            "The project therefore keeps both sentiment outputs for comparison instead of "
            "declaring either one a definitive ground truth."
        ),
        accent="#3567b8",
    )

    figure_label(
        "2.C",
        "Critical review — sentiment models",
        "Method, validation evidence, observed weakness and final decision.",
    )

    conclusion_cards(
        [
            {
                "chip": "Method",
                "title": "Two complementary sentiment views",
                "body": (
                    "VADER captures lexical/rule-based polarity; the Transformer captures "
                    "contextual sentiment. Both classify the same 2,666 posts."
                ),
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "chip": "Validation",
                "title": "147 evaluable reference posts",
                "body": (
                    "Macro F1 is 0.454 for VADER and 0.510 for the Transformer on the "
                    "LLM-assisted reference sample."
                ),
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "chip": "Critical finding",
                "title": "The observed sentiment picture is model-dependent",
                "body": (
                    "Only 37.6% of labels agree. VADER is predominantly Positive while the "
                    "Transformer is predominantly Neutral."
                ),
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "chip": "Decision",
                "title": "Do not declare one sentiment distribution as truth",
                "body": (
                    "Retain both outputs as comparative analytical lenses and state that the "
                    "reference sample is LLM-assisted, not a human gold standard."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
        ]
    )

    technical_trace(
        stage_name="Sentiment analysis",
        method=(
            "VADER lexical sentiment + contextual Transformer sentiment; "
            "three-class comparison and reference-sample evaluation."
        ),
        inputs=[
            "data/processed/analysis_ready_posts.jsonl",
        ],
        outputs=[
            "data/results/sentiment/sentiment_predictions.csv",
            "data/results/final_synthesis/sentiment_distribution.csv",
            "data/results/final_synthesis/negative_sentiment_vs_stress.csv",
        ],
        keywords=[
            "sentiment",
            "vader",
            "transformer",
        ],
    )


# ============================================================
# PAGE 3 — EMOTIONS
# ============================================================

elif page == "3 · Emotion Analysis":

    stage_header(
        "03",
        "Fine-grained emotion analysis",
        "Move beyond Positive / Neutral / Negative with GoEmotions",
        (
            "Emotion modeling captures specific affective signals such as annoyance, "
            "disapproval, curiosity and confusion. The task is multi-label, so a post may "
            "receive more than one emotion."
        ),
    )

    method_box(
        "Modeling approach",
        (
            "A CPU-friendly TF-IDF representation is combined with One-vs-Rest Logistic "
            "Regression across the 28 simplified GoEmotions labels. One decision threshold "
            "is calibrated per emotion using the validation split rather than forcing a "
            "single threshold across all labels."
        ),
    )

    metric_grid(
        [
            {
                "label": "GoEmotions train",
                "value": "43,410",
                "note": "multi-label training records",
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "label": "Validation",
                "value": "5,426",
                "note": "used for threshold calibration",
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "label": "Official test",
                "value": "5,427",
                "note": "kept for final evaluation",
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "label": "Emotion labels",
                "value": "28",
                "note": "multi-label prediction space",
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
        ]
    )

    figure_label(
        "3.1",
        "GoEmotions official-test performance",
        (
            "Final multi-label evaluation after per-emotion threshold calibration."
        ),
    )

    go_metrics = pd.DataFrame(
        [
            ["Subset accuracy", 0.298323],
            ["Micro precision", 0.449285],
            ["Micro recall", 0.620793],
            ["Micro F1", 0.521295],
            ["Macro F1", 0.443765],
            ["Samples F1", 0.532081],
            ["Hamming loss", 0.047487],
        ],
        columns=[
            "Metric",
            "Value",
        ],
    )

    st.dataframe(
        go_metrics.style.format(
            {
                "Value": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    interpretation(
        "Model interpretation",
        (
            "Micro F1 is <strong>0.521</strong> and Macro F1 is <strong>0.444</strong>. "
            "The lower Macro F1 indicates that rarer emotions remain harder than frequent labels."
        ),
        accent="#3567b8",
    )

    figure_label(
        "3.C",
        "Critical review — experimental decisions",
        "Show which emotion-modeling ideas were tested, compared and retained or rejected.",
    )

    if go_threshold_strategy:
        global_metrics = go_threshold_strategy.get("global_threshold_test_metrics", {})
        calibrated_metrics = go_threshold_strategy.get("per_label_threshold_test_metrics", {})
        threshold_compare = pd.DataFrame(
            [
                [
                    "Single global threshold (0.61)",
                    global_metrics.get("micro_f1"),
                    global_metrics.get("macro_f1"),
                    global_metrics.get("samples_f1"),
                    global_metrics.get("no_predicted_label"),
                ],
                [
                    "Per-emotion calibrated thresholds",
                    calibrated_metrics.get("micro_f1"),
                    calibrated_metrics.get("macro_f1"),
                    calibrated_metrics.get("samples_f1"),
                    calibrated_metrics.get("no_predicted_label"),
                ],
            ],
            columns=[
                "Threshold strategy",
                "Micro F1",
                "Macro F1",
                "Samples F1",
                "Posts with no predicted emotion",
            ],
        )
        st.dataframe(
            threshold_compare.style.format(
                {
                    "Micro F1": "{:.3f}",
                    "Macro F1": "{:.3f}",
                    "Samples F1": "{:.3f}",
                    "Posts with no predicted emotion": "{:.0f}",
                },
                na_rep="—",
            ),
            hide_index=True,
            use_container_width=True,
        )

    conclusion_cards(
        [
            {
                "chip": "Experiment 1",
                "title": "Per-label thresholds were retained",
                "body": (
                    "Calibrating one threshold per emotion improves Micro F1 from 0.493 to 0.521 "
                    "and reduces posts with no predicted label from 376 to 184."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "chip": "Experiment 2",
                "title": "Neutral-exclusivity rule was rejected",
                "body": (
                    "Forcing neutral to exclude every other emotion reduced multi-label performance, "
                    "so the final pipeline allows neutral to coexist with other predicted emotions."
                ),
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "chip": "Critical finding",
                "title": "Emotion performance is uneven across labels",
                "body": (
                    "Micro F1 remains above Macro F1, indicating that common emotions are modeled "
                    "more reliably than several rarer labels."
                ),
                "accent": "#a66d1b",
                "soft": "#fbf0dc",
            },
            {
                "chip": "Decision",
                "title": "Retain multi-label GoEmotions as an exploratory layer",
                "body": (
                    "Emotion outputs enrich polarity analysis, but they are reported as model-predicted "
                    "affective language rather than verified psychological states."
                ),
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
        ]
    )

    figure_label(
        "3.2",
        "Most frequent model-predicted emotion labels",
        (
            "Top GoEmotions labels applied to the final 2,666-post technical corpus."
        ),
    )

    if emotion_prevalence is not None:
        emotion_col = (
            "emotion"
            if "emotion" in emotion_prevalence.columns
            else emotion_prevalence.columns[0]
        )

        rate_col = next(
            (
                column
                for column in emotion_prevalence.columns
                if (
                    "rate" in column.lower()
                    or "percent" in column.lower()
                    or "prevalence" in column.lower()
                )
            ),
            None,
        )

        if rate_col is not None:
            emotion_plot = emotion_prevalence.copy()
            emotion_plot["_rate"] = emotion_plot[
                rate_col
            ].astype(float)

            if emotion_plot["_rate"].max() > 1.5:
                emotion_plot["_rate"] /= 100.0

            emotion_plot = (
                emotion_plot
                .sort_values(
                    "_rate",
                    ascending=False,
                )
                .head(10)
                .sort_values("_rate")
            )

            fig = go.Figure(
                go.Bar(
                    x=emotion_plot["_rate"] * 100,
                    y=emotion_plot[emotion_col],
                    orientation="h",
                    marker_color="#5c3198",
                    text=[
                        f"{value:.1f}%"
                        for value in emotion_plot["_rate"] * 100
                    ],
                    textposition="outside",
                    cliponaxis=False,
                )
            )

            fig.update_layout(
                xaxis_title="Share of posts (%)",
                yaxis_title="",
            )

            plot_style(
                fig,
                height=470,
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                theme=None,
                config={"displayModeBar": False},
            )

    metric_grid(
        [
            {
                "label": "Average emotions / post",
                "value": "1.85",
                "note": "multi-label cardinality",
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "label": "Multiple emotions",
                "value": "52.74%",
                "note": "1,406 posts",
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "label": "No emotion predicted",
                "value": "4.31%",
                "note": "115 posts",
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
        ]
    )

    interpretation(
        "Conclusion",
        (
            "The final technical corpus is not well represented by one emotion label per post. "
            "More than half of the posts receive multiple model-predicted emotion labels, which supports "
            "the use of a <strong>multi-label emotion layer</strong> between broad sentiment "
            "and Stress modeling."
        ),
        accent="#1f8765",
    )

    figure_label(
        "3.C",
        "Critical review — emotion model",
        "What the GoEmotions layer can support, and where caution is required.",
    )

    conclusion_cards(
        [
            {
                "chip": "Method",
                "title": "28-label multi-label classifier",
                "body": (
                    "TF-IDF + One-vs-Rest Logistic Regression is trained on GoEmotions, "
                    "with one validation-calibrated threshold per emotion."
                ),
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "chip": "Validation",
                "title": "Evaluated on the untouched GoEmotions test split",
                "body": (
                    "Per-label threshold calibration reaches Micro F1 = 0.521 and "
                    "Macro F1 = 0.444 on 5,427 official test records."
                ),
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "chip": "Critical finding",
                "title": "Rare emotions remain difficult",
                "body": (
                    "Macro F1 is lower than Micro F1, showing uneven label performance; "
                    "the model also transfers from general-domain GoEmotions to technical discussion."
                ),
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "chip": "Decision",
                "title": "Use emotions as fine-grained language signals",
                "body": (
                    "Emotion outputs are retained for exploratory comparison and enrichment, "
                    "not interpreted as verified psychological states of individual authors."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
        ]
    )

    technical_trace(
        stage_name="Emotion analysis",
        method=(
            "GoEmotions simplified 28-label task; TF-IDF + One-vs-Rest Logistic Regression; "
            "per-emotion threshold calibration on validation data."
        ),
        inputs=[
            "GoEmotions train / validation / test splits",
            "data/processed/analysis_ready_posts.jsonl",
        ],
        outputs=[
            "data/results/emotions/goemotions_predictions_analysis_ready.csv",
            "data/results/emotions/goemotions_analysis_ready_emotion_prevalence.csv",
        ],
        keywords=[
            "emotion",
            "goemotion",
            "goemotions",
        ],
    )


# ============================================================
# PAGE 4 — STRESS
# ============================================================

elif page == "4 · Stress & Critical Validation":

    stage_header(
        "04",
        "Stress detection & critical validation",
        "Model Stress separately, test domain transfer and expose model failure modes",
        (
            "Stress is not treated as negative sentiment. A dedicated supervised classifier "
            "is trained from Dreaddit, augmented experimentally with synthetic developer-style "
            "examples, then audited on real developer-oriented technical posts with LLM-assisted reference labels."
        ),
    )

    method_box(
        "Baseline and hybrid approach",
        (
            "The classifier uses TF-IDF text features and Logistic Regression. The baseline "
            "is trained from real Dreaddit examples. The final hybrid experiment combines "
            "2,838 Dreaddit training posts with 1,200 synthetic examples (600 Stress / "
            "600 No stress), for 4,038 training examples. Synthetic data are augmentation only."
        ),
    )

    figure_label(
        "4.M",
        "Stress baseline model selection",
        (
            "Logistic Regression, Calibrated Linear SVM and Complement Naive Bayes were compared "
            "with 5-fold cross-validation on the Dreaddit training split before the official test was used."
        ),
    )

    if dreaddit_candidate_comparison is not None:
        candidate_view = dreaddit_candidate_comparison.copy()
        candidate_view = candidate_view.rename(
            columns={
                "model": "Candidate model",
                "cv_accuracy_mean": "CV Accuracy",
                "cv_macro_f1_mean": "CV Macro F1",
                "cv_stress_precision_mean": "Stress Precision",
                "cv_stress_recall_mean": "Stress Recall",
                "cv_stress_f1_mean": "Stress F1",
            }
        )
        visible = [
            column
            for column in [
                "Candidate model",
                "CV Accuracy",
                "CV Macro F1",
                "Stress Precision",
                "Stress Recall",
                "Stress F1",
            ]
            if column in candidate_view.columns
        ]
        with st.expander("Candidate classifier cross-validation results", expanded=True):
            st.dataframe(
                candidate_view[visible].style.format(
                    {column: "{:.3f}" for column in visible if column != "Candidate model"}
                ),
                hide_index=True,
                use_container_width=True,
            )

    conclusion_cards(
        [
            {
                "chip": "Selection criterion",
                "title": "Logistic Regression had the highest CV Macro F1",
                "body": (
                    "CV Macro F1 was 0.755 for Logistic Regression, 0.753 for Complement Naive Bayes "
                    "and 0.745 for the calibrated Linear SVM."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "chip": "Critical nuance",
                "title": "The ranking changes with the evaluation metric",
                "body": (
                    "Complement Naive Bayes achieved higher Stress recall and Stress F1 in cross-validation, "
                    "while Logistic Regression led on the pre-declared Macro-F1 selection criterion."
                ),
                "accent": "#a66d1b",
                "soft": "#fbf0dc",
            },
            {
                "chip": "Leakage control",
                "title": "Model selection happened before official test evaluation",
                "body": (
                    "Candidate comparison used training-set cross-validation; the untouched Dreaddit test split "
                    "was used only after the baseline classifier was selected."
                ),
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "chip": "Decision",
                "title": "Retain Logistic Regression for the final baseline",
                "body": (
                    "It provides the best CV Macro-F1 balance, calibrated probabilities and an interpretable, "
                    "CPU-friendly TF-IDF pipeline suitable for the internship constraints."
                ),
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
        ]
    )

    figure_label(
        "4.1",
        "Dreaddit official-test model comparison",
        (
            "Baseline performance compared with the final hybrid training configuration."
        ),
    )

    comparison = pd.DataFrame(
        [
            [
                "Dreaddit-only baseline",
                0.7259,
                0.7249,
                0.7224,
                0.7615,
                0.7414,
            ],
            [
                "Dreaddit + synthetic hybrid",
                0.7371,
                0.7359,
                0.7303,
                0.7778,
                0.7533,
            ],
        ],
        columns=[
            "Model",
            "Accuracy",
            "Macro F1",
            "Stress precision",
            "Stress recall",
            "Stress F1",
        ],
    )

    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.3f}",
                "Macro F1": "{:.3f}",
                "Stress precision": "{:.3f}",
                "Stress recall": "{:.3f}",
                "Stress F1": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    interpretation(
        "Result",
        (
            "Hybrid augmentation improves the official Dreaddit test modestly: "
            "<strong>Stress F1 rises from 0.741 to 0.753</strong> and recall from "
            "<strong>0.762 to 0.778</strong>. These gains are modest and do not, by themselves, "
            "demonstrate robust transfer to developer-oriented technical language."
        ),
    )

    figure_label(
        "4.2",
        "Complementary developer-domain validation",
        (
            "The final Stress classifier is audited on a 600-post sample drawn from the real technical corpus. "
            "Reference labels are LLM-assisted; after 4 Unclear labels are excluded, 596 posts are evaluable."
        ),
    )

    metric_grid(
        [
            {
                "label": "Evaluable posts",
                "value": "596",
                "note": "588 No stress · 8 Stress",
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "label": "Balanced accuracy",
                "value": "0.834",
                "note": "important under strong imbalance",
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "label": "Stress precision",
                "value": "0.111",
                "note": "false positives remain",
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "label": "Stress recall",
                "value": "0.750",
                "note": "6 of 8 Stress cases detected",
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "label": "MCC",
                "value": "0.268",
                "note": "binary quality under imbalance",
                "accent": "#7849b8",
                "soft": "#efe8f8",
            },
        ]
    )

    left, right = st.columns(
        [0.95, 1.05],
        gap="large",
    )

    with left:
        figure_label(
            "4.3",
            "Developer-domain confusion matrix",
            (
                "Reference labels on rows and model predictions on columns."
            ),
        )

        matrix = pd.DataFrame(
            [
                [540, 48],
                [2, 6],
            ],
            index=[
                "Reference No stress",
                "Reference Stress",
            ],
            columns=[
                "Predicted No stress",
                "Predicted Stress",
            ],
        )

        fig = go.Figure(
            go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                text=matrix.values,
                texttemplate="%{text}",
                colorscale=[
                    [0, "#f8f4fa"],
                    [1, "#5c3198"],
                ],
                showscale=False,
            )
        )

        plot_style(
            fig,
            height=365,
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None,
            config={"displayModeBar": False},
        )

    with right:
        interpretation(
            "Domain-shift finding",
            (
                "The model detects <strong>6 of 8 Stress cases</strong>, but it also marks "
                "<strong>48 No-stress posts as Stress</strong>. Developer troubleshooting, "
                "technical frustration and dramatic technical language can resemble Stress "
                "patterns learned from another domain."
            ),
            accent="#c84c6f",
        )

        interpretation(
            "Scientific boundary",
            (
                "The developer-domain set is a <strong>complementary LLM-assisted validation/reference set</strong>, "
                "not a clinical or human gold-standard benchmark. The final corpus result "
                "must therefore be reported as <strong>model-predicted Stress</strong>, not a diagnosis."
            ),
            accent="#a66d1b",
        )

    figure_label(
        "4.4",
        "Critical comparison on the developer-domain reference set",
        (
            "The same 596 evaluable reference posts are used to compare the Dreaddit-only "
            "baseline with the hybrid model after synthetic augmentation."
        ),
    )

    if stress_real_comparison is not None:
        critical_cols = [
            column
            for column in [
                "model",
                "balanced_accuracy",
                "stress_precision",
                "stress_recall",
                "stress_f1",
                "false_positive_rate",
                "mcc",
            ]
            if column in stress_real_comparison.columns
        ]
        critical_table = stress_real_comparison[critical_cols].copy()
        st.dataframe(
            critical_table.style.format(
                {
                    "balanced_accuracy": "{:.3f}",
                    "stress_precision": "{:.3f}",
                    "stress_recall": "{:.3f}",
                    "stress_f1": "{:.3f}",
                    "false_positive_rate": "{:.3f}",
                    "mcc": "{:.3f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    conclusion_cards(
        [
            {
                "chip": "Improvement",
                "title": "Hybrid augmentation improves several developer-domain metrics",
                "body": (
                    "Relative to the Dreaddit-only baseline, recall improves from 0.625 to 0.750, "
                    "balanced accuracy from 0.764 to 0.834, and false-positive rate falls from 9.7% "
                    "to 8.2%. However, precision remains only 0.111 and the reference set contains "
                    "only eight positive Stress cases, so these gains do not demonstrate robust transfer."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "chip": "Main weakness",
                "title": "Stress precision remains very low",
                "body": (
                    "Only 6 of 54 hybrid Stress predictions match the reference Stress label; "
                    "48 are false positives, giving precision = 0.111."
                ),
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "chip": "Validation limitation",
                "title": "The positive reference class is extremely small",
                "body": (
                    "Only 8 of the 596 evaluable reference posts are labeled Stress. "
                    "Positive-class metrics are therefore unstable and must be interpreted cautiously."
                ),
                "accent": "#a66d1b",
                "soft": "#fbf0dc",
            },
            {
                "chip": "Final decision",
                "title": "Retain the model with a strict validity boundary",
                "body": (
                    "Use it to identify model-predicted Stress-related language and compare groups; "
                    "do not present its outputs as diagnoses or population prevalence."
                ),
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
        ]
    )

    figure_label(
        "4.5",
        "Qualitative error analysis",
        (
            "Inspect representative false positives and false negatives to understand "
            "which types of developer language challenge the final classifier."
        ),
    )

    if stress_real_predictions is not None:
        error_data = stress_real_predictions.copy()
        fp = error_data[
            (error_data["reference_stress_label"] == "No stress")
            & (error_data["hybrid_prediction"] == "Stress")
        ].copy()
        fn = error_data[
            (error_data["reference_stress_label"] == "Stress")
            & (error_data["hybrid_prediction"] == "No stress")
        ].copy()

        tab_fp, tab_fn = st.tabs(
            [
                f"False positives ({len(fp)})",
                f"False negatives ({len(fn)})",
            ]
        )

        error_columns = [
            column
            for column in [
                "platform",
                "community",
                "text_clean_basic",
                "annotation_rationale",
                "hybrid_stress_probability",
            ]
            if column in error_data.columns
        ]

        with tab_fp:
            st.caption(
                "Typical risk: technical frustration, bug reports or strong complaint language "
                "may resemble psychological Stress patterns."
            )
            st.dataframe(
                fp[error_columns].head(8),
                hide_index=True,
                use_container_width=True,
            )

        with tab_fn:
            st.caption(
                "These cases show Stress-reference language that the model failed to identify. "
                "Because only two false negatives exist, they should be inspected individually."
            )
            st.dataframe(
                fn[error_columns].head(8),
                hide_index=True,
                use_container_width=True,
            )

    figure_label(
        "4.6",
        "Final Stress classification on the technical corpus",
        (
            "Application of the final hybrid classifier to all 2,666 analysis-ready posts."
        ),
    )

    fig = go.Figure(
        go.Pie(
            labels=[
                "No stress",
                "Predicted Stress",
            ],
            values=[
                ANALYSIS_POSTS - STRESS_POSTS,
                STRESS_POSTS,
            ],
            hole=0.68,
            marker_colors=[
                "#e2dce7",
                "#c84c6f",
            ],
            textinfo="label+percent",
            sort=False,
        )
    )

    plot_style(
        fig,
        height=390,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={"displayModeBar": False},
    )

    interpretation(
        "Conclusion",
        (
            "<strong>248 of 2,666 posts (9.30%)</strong> are classified as Stress by the final "
            "model. Because of the observed domain shift and low developer-domain precision, "
            "9.30% is strictly a <strong>model-predicted Stress rate</strong> and must not be interpreted "
            "as psychological Stress prevalence among developers."
        ),
        accent="#c84c6f",
    )

    technical_trace(
        stage_name="Stress detection",
        method=(
            "TF-IDF + Logistic Regression; Dreaddit baseline; synthetic developer-style "
            "training augmentation; official-test comparison; complementary domain validation."
        ),
        inputs=[
            "Dreaddit human-labeled Stress / No-stress dataset",
            "1,200 synthetic training-augmentation examples",
            "600 developer-domain reference posts",
            "data/processed/analysis_ready_posts.jsonl",
        ],
        outputs=[
            "models/hybrid_dreaddit_synthetic_stress_classifier.joblib",
            "data/results/emotions/hybrid_stress_predictions_analysis_ready.csv",
            "data/results/emotions/hybrid_stress_analysis_ready_summary.json",
        ],
        keywords=[
            "stress",
            "dreaddit",
            "hybrid",
            "synthetic",
        ],
    )


# ============================================================
# PAGE 5 — INTEGRATED ANALYSIS
# ============================================================

elif page == "5 · Integrated Analysis":

    stage_header(
        "05",
        "Integrated analysis",
        "Join sentiment, emotions and Stress at post level",
        (
            "After each layer is frozen independently, the predictions are merged by record_id "
            "so the project can test cross-signal relationships on the same 2,666 posts."
        ),
    )

    method_box(
        "Integration approach",
        (
            "The final merge combines sentiment predictions, GoEmotions outputs and hybrid "
            "Stress predictions. The row-level integration was checked for 2,666 rows, unique "
            "record identifiers and no record-set differences before analysis."
        ),
    )

    metric_grid(
        [
            {
                "label": "Unified rows",
                "value": "2,666",
                "note": "same corpus across all signals",
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "label": "VADER-negative → Stress",
                "value": "14.3%",
                "note": "81 of 565 negative posts",
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "label": "Transformer-negative → Stress",
                "value": "24.3%",
                "note": "117 of 482 negative posts",
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
        ]
    )

    figure_label(
        "5.1",
        "Model-predicted Stress rate within sentiment classes",
        (
            "Stress classification is compared with the negative, neutral and positive "
            "outputs from each sentiment model."
        ),
    )

    integrated_table = pd.DataFrame(
        [
            [
                "VADER",
                "Negative",
                565,
                81,
                0.1434,
            ],
            [
                "VADER",
                "Neutral",
                400,
                17,
                0.0425,
            ],
            [
                "VADER",
                "Positive",
                1701,
                150,
                0.0882,
            ],
            [
                "Transformer",
                "Negative",
                482,
                117,
                0.2427,
            ],
            [
                "Transformer",
                "Neutral",
                1707,
                110,
                0.0644,
            ],
            [
                "Transformer",
                "Positive",
                477,
                21,
                0.0440,
            ],
        ],
        columns=[
            "Sentiment model",
            "Sentiment label",
            "Posts",
            "Predicted Stress posts",
            "Predicted Stress rate",
        ],
    )

    st.dataframe(
        integrated_table.style.format(
            {
                "Predicted Stress rate": "{:.2%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    interpretation(
        "Central result",
        (
            "Most negative posts are <strong>not</strong> classified as Stress. "
            "This empirically supports the project's central methodological distinction: "
            "<strong>negative sentiment ≠ Stress</strong>. Sentiment describes tone; "
            "Stress requires a dedicated modeling layer."
        ),
        accent="#1f8765",
    )

    if emotion_stress is not None:
        figure_label(
            "5.2",
            "Emotion prevalence in Stress vs No-stress classifications",
            (
                "Descriptive comparison of the emotions most enriched inside the "
                "model-predicted Stress subset."
            ),
        )

        emotion_plot = (
            emotion_stress
            .sort_values(
                "rate_difference",
                ascending=False,
            )
            .head(8)
            .sort_values("rate_difference")
        )

        fig = go.Figure()

        fig.add_bar(
            name="No stress",
            y=emotion_plot["emotion"],
            x=emotion_plot["no_stress_rate"] * 100,
            orientation="h",
            marker_color="#ded8e4",
        )

        fig.add_bar(
            name="Predicted Stress",
            y=emotion_plot["emotion"],
            x=emotion_plot["stress_rate"] * 100,
            orientation="h",
            marker_color="#c84c6f",
        )

        fig.update_layout(
            barmode="group",
            xaxis_title="Share of posts (%)",
            yaxis_title="",
        )

        plot_style(
            fig,
            height=475,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None,
            config={"displayModeBar": False},
        )

        interpretation(
            "Emotion pattern",
            (
                "The largest descriptive enrichments are <strong>annoyance, disappointment, "
                "confusion, realization and disapproval</strong>. These comparisons help explain "
                "what kind of affective language accompanies Stress predictions, but they are "
                "descriptive and are not presented as per-emotion significance tests."
            ),
            accent="#c84c6f",
        )

    figure_label(
        "5.C",
        "Critical review — integrated findings",
        "Interpret cross-signal relationships without turning model predictions into psychological ground truth.",
    )

    conclusion_cards(
        [
            {
                "chip": "Method",
                "title": "Predictions are merged one-to-one by record_id",
                "body": (
                    "Sentiment, multi-label emotions and hybrid Stress outputs refer to the same 2,666 posts, "
                    "so cross-signal comparisons do not mix different corpora."
                ),
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "chip": "Central result",
                "title": "Negative sentiment is not equivalent to predicted Stress",
                "body": (
                    "Only 14.3% of VADER-negative and 24.3% of Transformer-negative posts are classified "
                    "as Stress, supporting a separate Stress modeling layer."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "chip": "Critical limitation",
                "title": "Upstream model errors propagate into integrated comparisons",
                "body": (
                    "Sentiment, emotion and Stress values are predicted labels. Misclassification in any upstream "
                    "model can influence the observed cross-signal percentages."
                ),
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "chip": "Decision",
                "title": "Treat emotion enrichment as descriptive evidence",
                "body": (
                    "Annoyance, disappointment, confusion, realization and disapproval are more prevalent in "
                    "Stress-classified posts, but no causal or per-emotion significance claim is made here."
                ),
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
        ]
    )

    technical_trace(
        stage_name="Integrated Sentiment × Emotion × Stress analysis",
        method=(
            "One-to-one row-level merge by record_id followed by cross-tabulation and "
            "descriptive emotion enrichment."
        ),
        inputs=[
            "data/results/sentiment/sentiment_predictions.csv",
            "data/results/emotions/goemotions_predictions_analysis_ready.csv",
            "data/results/emotions/hybrid_stress_predictions_analysis_ready.csv",
        ],
        outputs=[
            "data/results/combined/sentiment_emotion_stress_predictions.csv",
            "data/results/final_synthesis/negative_sentiment_vs_stress.csv",
            "data/results/final_synthesis/emotion_stress_association.csv",
        ],
        keywords=[
            "unified",
            "combined",
            "integrated",
            "sentiment_emotion_stress",
        ],
    )


# ============================================================
# PAGE 6 — TOPICS & STATISTICS
# ============================================================

elif page == "6 · Topic Modeling & Statistics":

    stage_header(
        "06",
        "Topic modeling and statistical association",
        "Learn discussion themes independently, then test their association with predicted Stress",
        (
            "NMF topics are learned from all 2,666 posts before Stress association is tested. "
            "This prevents the topic structure from being learned only from a small, noisy "
            "model-predicted Stress subset."
        ),
    )

    method_box(
        "Topic-modeling approach",
        (
            "TF-IDF converts the cleaned corpus into a non-negative document-term matrix. "
            "NMF factorizes that matrix into document-topic and topic-term components. "
            "Candidate solutions from 4 to 10 topics were evaluated using reconstruction error, "
            "topic diversity, dominant-topic strength, margins, topic sizes and qualitative interpretability."
        ),
    )

    metric_grid(
        [
            {
                "label": "Selected topics",
                "value": "7",
                "note": "best balance of quality and interpretability",
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "label": "Smallest topic",
                "value": "87",
                "note": "posts",
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "label": "Largest topic",
                "value": "1,037",
                "note": "posts",
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "label": "Topic diversity",
                "value": "0.914",
                "note": "selected 7-topic solution",
                "accent": "#7849b8",
                "soft": "#efe8f8",
            },
            {
                "label": "Dominant-topic weight",
                "value": "0.709",
                "note": "mean assignment strength",
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "label": "Topic margin",
                "value": "0.501",
                "note": "mean top-1 vs top-2 separation",
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
        ]
    )

    if topic_count_eval is not None:
        with st.expander(
            "Candidate-topic evaluation table (4–10 topics)"
        ):
            st.dataframe(
                topic_count_eval,
                hide_index=True,
                use_container_width=True,
            )

    interpretation(
        "Model-selection decision",
        (
            "Seven topics were retained because they remain coherent and sufficiently sized. "
            "At eight or more topics, very small artifact-like clusters began to appear, "
            "reducing interpretability despite additional decomposition."
        ),
        accent="#3567b8",
    )

    figure_label(
        "6.C",
        "Critical review — topic model",
        "Why seven topics were retained and how far the statistical conclusions can go.",
    )

    conclusion_cards(
        [
            {
                "chip": "Method",
                "title": "Candidate NMF solutions from 4 to 10 topics",
                "body": (
                    "Every solution uses the same TF-IDF corpus and is compared using reconstruction, "
                    "topic diversity, separation, sizes and qualitative interpretability."
                ),
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "chip": "Selection evidence",
                "title": "Seven topics preserve usable cluster sizes",
                "body": (
                    "The 7-topic solution combines diversity = 0.914, mean dominant-topic weight = 0.709, "
                    "mean topic margin = 0.501 and a smallest topic of 87 posts; at 8–10 topics, the "
                    "smallest clusters collapse to 17, 15 and 14 posts."
                ),
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "chip": "Critical finding",
                "title": "Seven is an analytical choice, not a true natural number",
                "body": (
                    "NMF topic labels depend on vectorization, stopwords and analyst interpretation. "
                    "The selected solution is a defensible representation, not unique ground truth."
                ),
                "accent": "#a66d1b",
                "soft": "#fbf0dc",
            },
            {
                "chip": "Decision",
                "title": "Test Stress only after independent topic learning",
                "body": (
                    "Topics are learned on all 2,666 posts before Stress association is tested, "
                    "reducing the risk of constructing topics from the small model-predicted Stress subset."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
        ]
    )

    figure_label(
        "6.1",
        "Model-predicted Stress rate by final NMF topic",
        (
            "Each bar shows the percentage of posts classified as Stress inside that topic. "
            "The vertical reference line is the overall 9.30% model-predicted Stress rate."
        ),
    )

    if topic_final is not None:
        topic_plot = topic_final.copy()

        if "significant_fdr_0_05" in topic_plot.columns:
            topic_plot["_sig"] = as_bool(
                topic_plot["significant_fdr_0_05"]
            )
        else:
            topic_plot["_sig"] = False

        topic_plot = topic_plot.sort_values(
            "predicted_stress_rate",
            ascending=True,
        )

        colors = [
            "#c84c6f"
            if sig and rr > 1
            else "#1f8765"
            if sig and rr < 1
            else "#9686aa"
            for sig, rr in zip(
                topic_plot["_sig"],
                topic_plot["relative_risk"],
            )
        ]

        fig = go.Figure(
            go.Bar(
                x=topic_plot["predicted_stress_rate"] * 100,
                y=topic_plot["topic_name"],
                orientation="h",
                marker_color=colors,
                text=[
                    f"{value:.2f}%"
                    for value in (
                        topic_plot["predicted_stress_rate"] * 100
                    )
                ],
                textposition="outside",
                cliponaxis=False,
            )
        )

        fig.add_vline(
            x=STRESS_RATE * 100,
            line_dash="dot",
            line_color="#716977",
            annotation_text=f"Overall {pct(STRESS_RATE, 2)}",
        )

        fig.update_layout(
            xaxis_title="Model-predicted Stress rate (%)",
            yaxis_title="",
        )

        plot_style(
            fig,
            height=520,
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None,
            config={"displayModeBar": False},
        )

        interpretation(
            "Topic-level result",
            (
                "<strong>ChatGPT / OpenAI User Experience</strong> has the highest final "
                "model-predicted Stress rate at 14.99%, while <strong>AI Labs / Industry News</strong> "
                "is much lower at 3.69%. The statistical tests below determine whether these "
                "differences survive multiple-comparison correction."
            ),
        )

        figure_label(
            "6.2",
            "Per-topic association statistics",
            (
                "Relative risk, odds ratio, Fisher exact-test FDR p-value and final "
                "significance status."
            ),
        )

        stats_table = topic_final[
            [
                "topic_name",
                "posts",
                "predicted_stress_rate",
                "relative_risk",
                "odds_ratio",
                "fisher_p_fdr_bh",
                "significant_fdr_0_05",
            ]
        ].copy()

        st.dataframe(
            stats_table.style.format(
                {
                    "predicted_stress_rate": "{:.2%}",
                    "relative_risk": "{:.2f}×",
                    "odds_ratio": "{:.2f}",
                    "fisher_p_fdr_bh": "{:.4f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    conclusion_cards(
        [
            {
                "chip": "Higher association",
                "title": "ChatGPT / OpenAI User Experience",
                "body": (
                    "14.99% model-predicted Stress · RR 1.77× · OR 1.91 · "
                    "FDR p = 0.0013."
                ),
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "chip": "Lower association",
                "title": "AI Labs / Industry News",
                "body": (
                    "3.69% model-predicted Stress · RR 0.37× · OR 0.35 · "
                    "FDR p = 0.0013."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "chip": "Overall association",
                "title": "Statistically significant, small effect",
                "body": (
                    "χ²(6) = 28.53 · p ≈ 0.000075 · Cramér’s V = 0.103."
                ),
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
        ]
    )

    interpretation(
        "Statistical conclusion",
        (
            "Topic membership is associated with <strong>model-predicted Stress</strong>, "
            "but the overall relationship is small. After Benjamini-Hochberg FDR correction, "
            "the strongest robust topic signals are the higher ChatGPT/OpenAI User Experience "
            "association and the lower AI Labs / Industry News association. "
            "<strong>Association does not imply causality.</strong>"
        ),
        accent="#1f8765",
    )

    interpretation(
        "Upstream-uncertainty boundary",
        (
            "These tests use <strong>model-predicted Stress</strong> as the outcome. Because the Stress "
            "classifier shows developer-domain false positives, some classification uncertainty propagates "
            "into Topic × model-predicted Stress associations. Statistical significance therefore supports association with "
            "the model output, not a clinical Stress effect."
        ),
        accent="#a66d1b",
    )

    if topic_keywords is not None:
        with st.expander(
            "Final topic labels and highest-weight NMF terms"
        ):
            for topic_row in topic_keywords.itertuples():
                st.markdown(
                    f"**T{getattr(topic_row, 'topic', '')} — "
                    f"{getattr(topic_row, 'topic_name', '')}**"
                )
                st.caption(
                    str(
                        getattr(
                            topic_row,
                            "top_terms",
                            "",
                        )
                    )
                )

    technical_trace(
        stage_name="Topic modeling and statistical testing",
        method=(
            "TF-IDF + NMF topic modeling over all 2,666 posts; candidate-topic comparison; "
            "Topic × model-predicted Stress chi-square; per-topic Fisher exact tests; "
            "Benjamini-Hochberg FDR; relative risk and odds ratio."
        ),
        inputs=[
            "data/results/combined/sentiment_emotion_stress_predictions.csv",
            "data/processed/analysis_ready_posts.jsonl",
        ],
        outputs=[
            "data/results/stress_topics/final_topic_assignments.csv",
            "data/results/stress_topics/final_topic_keywords.csv",
            "data/results/stress_topics/topic_count_evaluation.csv",
            "data/results/stress_topics/topic_stress_statistical_summary.csv",
            "data/results/final_synthesis/topic_stress_final.csv",
        ],
        keywords=[
            "topic",
            "nmf",
            "stress_topic",
        ],
    )


# ============================================================
# PAGE 7 — EVIDENCE EXPLORER
# ============================================================

elif page == "7 · Evidence Explorer":

    stage_header(
        "07",
        "Qualitative evidence",
        "Inspect concrete developer publications and their model outputs",
        (
            "This page connects aggregate statistics back to actual posts. Use the sidebar "
            "filters to select a platform, Stress label, topic or search term."
        ),
    )

    if filtered is None or filtered.empty:
        st.info(
            "No row-level publications match the current selection."
        )

    else:
        current_count = len(filtered)

        if "hybrid_stress_label" in filtered.columns:
            current_stress = int(
                (
                    filtered["hybrid_stress_label"]
                    .astype(str)
                    == "Stress"
                ).sum()
            )
        else:
            current_stress = 0

        current_rate = (
            current_stress / current_count
            if current_count
            else 0
        )

        metric_grid(
            [
                {
                    "label": "Matching posts",
                    "value": f"{current_count:,}",
                    "note": "after current sidebar filters",
                    "accent": "#5c3198",
                    "soft": "#efe8f8",
                },
                {
                    "label": "Model-predicted Stress",
                    "value": f"{current_stress:,}",
                    "note": pct(current_rate, 2),
                    "accent": "#c84c6f",
                    "soft": "#f9e8ed",
                },
            ]
        )

        text_col = next(
            (
                column
                for column in [
                    "text_clean_basic",
                    "text",
                    "full_text_raw",
                ]
                if column in filtered.columns
            ),
            None,
        )

        if (
            text_col is not None
            and "record_id" in filtered.columns
        ):
            sample = filtered.head(200).copy()
            options = {}

            for _, row in sample.iterrows():
                rid = str(
                    row["record_id"]
                )

                platform = str(
                    row.get(
                        "platform",
                        "Unknown",
                    )
                )

                preview = (
                    str(
                        row.get(
                            text_col,
                            "",
                        )
                    )
                    .replace("\n", " ")
                    [:75]
                )

                options[
                    f"{platform} · {rid} · {preview}…"
                ] = rid

            chosen_display = st.selectbox(
                "Select a publication",
                list(options.keys()),
            )

            chosen_id = options[
                chosen_display
            ]

            row = sample[
                sample["record_id"]
                .astype(str)
                == chosen_id
            ].iloc[0]

            meta = pd.DataFrame(
                [
                    [
                        "Platform",
                        row.get(
                            "platform",
                            "—",
                        ),
                    ],
                    [
                        "VADER sentiment",
                        row.get(
                            "vader_label",
                            "—",
                        ),
                    ],
                    [
                        "Transformer sentiment",
                        row.get(
                            "transformer_label",
                            "—",
                        ),
                    ],
                    [
                        "Top emotion",
                        row.get(
                            "top_emotion",
                            "—",
                        ),
                    ],
                    [
                        "Stress class",
                        row.get(
                            "hybrid_stress_label",
                            "—",
                        ),
                    ],
                    [
                        "Stress probability",
                        f"{float(row.get('hybrid_stress_probability', 0)):.3f}",
                    ],
                    [
                        "Topic",
                        row.get(
                            "topic_name",
                            "—",
                        ),
                    ],
                ],
                columns=[
                    "Model output",
                    "Prediction",
                ],
            )

            st.dataframe(
                meta,
                hide_index=True,
                use_container_width=True,
            )

            st.text_area(
                "Publication text",
                value=str(
                    row.get(
                        text_col,
                        "",
                    )
                ),
                height=210,
                disabled=True,
            )

            interpretation(
                "Evidence-use rule",
                (
                    "Examples are used to understand model behavior and failure modes. "
                    "A single Stress prediction is <strong>not</strong> a clinical judgment. "
                    "Examples are especially useful for identifying technical-frustration false positives."
                ),
                accent="#a66d1b",
            )

        display_cols = [
            column
            for column in [
                "platform",
                "vader_label",
                "transformer_label",
                "top_emotion",
                "hybrid_stress_label",
                "hybrid_stress_probability",
                "topic_name",
            ]
            if column in filtered.columns
        ]

        with st.expander(
            "Show first 100 matching row-level predictions"
        ):
            st.dataframe(
                filtered[
                    display_cols
                ].head(100),
                hide_index=True,
                use_container_width=True,
            )


# ============================================================
# PAGE 8 — FINAL CONCLUSIONS
# ============================================================

elif page == "8 · Final Conclusions":

    stage_header(
        "08",
        "Final synthesis",
        "Answer the research question with evidence and visible limitations",
        (
            "The conclusion combines the strongest validated findings from every stage "
            "instead of repeating every metric."
        ),
    )

    conclusion_cards(
        [
            {
                "chip": "Sentiment",
                "title": "Sentiment depends strongly on the chosen model",
                "body": (
                    "VADER and the contextual Transformer agree on only 37.6% of posts. "
                    "The Transformer is much more neutral; VADER is much more positive."
                ),
                "accent": "#5c3198",
                "soft": "#efe8f8",
            },
            {
                "chip": "Core distinction",
                "title": "Negative sentiment is not equivalent to Stress",
                "body": (
                    "Only 14.3% of VADER-negative and 24.3% of Transformer-negative posts "
                    "are classified as Stress."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "chip": "Emotions",
                "title": "Fine-grained affect adds information beyond polarity",
                "body": (
                    "Annoyance, disappointment, confusion, realization and disapproval "
                    "are more prevalent inside the model-predicted Stress subset."
                ),
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "chip": "Stress model",
                "title": "Some discriminative signal remains, but target-domain precision is very limited",
                "body": (
                    "Developer-domain recall is 0.750, but Stress precision is only 0.111. "
                    "Only eight positive reference cases are evaluable, and technical frustration "
                    "produces substantial false positives."
                ),
                "accent": "#a66d1b",
                "soft": "#fbf0dc",
            },
            {
                "chip": "Topics",
                "title": "Discussion context is associated with model-predicted Stress",
                "body": (
                    "The Topic × model-predicted Stress chi-square is significant, but Cramér’s V = 0.103 "
                    "shows that the overall effect is small."
                ),
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "chip": "Strongest topic signal",
                "title": "ChatGPT / OpenAI User Experience stands out",
                "body": (
                    "This topic has a 14.99% model-predicted Stress rate and RR = 1.77× after "
                    "FDR correction; the result is associative, not causal."
                ),
                "accent": "#7849b8",
                "soft": "#efe8f8",
            },
        ]
    )

    interpretation(
        "Final answer to the research question",
        (
            "The collected developer-oriented technical discussions exhibit <strong>distinct but connected "
            "sentiment, model-predicted emotion-related, and stress-related linguistic signals</strong>. "
            "Sentiment alone is insufficient to identify model-predicted Stress: most Negative posts are "
            "not classified as Stress. Predicted emotion labels provide additional descriptive context, "
            "while model-predicted Stress rates vary across discussion topics. However, the Stress classifier "
            "has substantial developer-domain false positives and the overall Topic × model-predicted Stress "
            "association is statistically significant but small."
        ),
        accent="#1f8765",
    )

    figure_label(
        "8.C",
        "Scientific claim boundary",
        "The final project is strongest when it states clearly what the evidence supports and what it does not.",
    )

    conclusion_cards(
        [
            {
                "chip": "Can claim",
                "title": "The methods produce distinct analytical signals",
                "body": (
                    "Sentiment, fine-grained emotion predictions and Stress predictions capture "
                    "different patterns in the collected technical discussions."
                ),
                "accent": "#1f8765",
                "soft": "#e9f5f0",
            },
            {
                "chip": "Can claim",
                "title": "Negative sentiment is not equivalent to predicted Stress",
                "body": (
                    "The integrated row-level results directly show that most negative posts "
                    "are not classified as Stress."
                ),
                "accent": "#3567b8",
                "soft": "#eaf0fb",
            },
            {
                "chip": "Cannot claim",
                "title": "9.30% is not psychological Stress prevalence among developers",
                "body": (
                    "The Stress classifier has domain-shift false positives and the reference set "
                    "is LLM-assisted with only eight positive cases."
                ),
                "accent": "#c84c6f",
                "soft": "#f9e8ed",
            },
            {
                "chip": "Cannot claim",
                "title": "Topic associations are not causal effects",
                "body": (
                    "The Topic × model-predicted Stress association is significant but small "
                    "(Cramér’s V = 0.103), and does not show that a topic causes Stress."
                ),
                "accent": "#a66d1b",
                "soft": "#fbf0dc",
            },
        ]
    )

    interpretation(
        "What the project contributes",
        (
            "The principal contribution is the <strong>complete interpretable analytical chain</strong>: "
            "data preparation → sentiment comparison → multi-label emotions → dedicated Stress modeling "
            "→ developer-domain validation → integrated cross-signal analysis → independent topic modeling "
            "→ statistical association testing → qualitative evidence and dashboard visualization."
        ),
        accent="#5c3198",
    )

    interpretation(
        "Limits",
        (
            "Model-predicted Stress is not a diagnosis or prevalence estimate. The developer-domain reference "
            "contains only eight evaluable Stress cases and is not a perfect human clinical gold standard. "
            "Synthetic examples are training augmentation only. Emotion enrichments are descriptive unless "
            "separately tested. Topic associations do not establish causality."
        ),
        accent="#c84c6f",
    )

    interpretation(
        "Recommended continuation",
        (
            "The strongest next step is to build a larger manually annotated developer-Stress benchmark, "
            "then recalibrate or fine-tune the Stress model on developer-oriented technical language. A second extension would "
            "test the emotion enrichments statistically and evaluate more contextual Stress architectures "
            "once reliable domain labels are available."
        ),
        accent="#3567b8",
    )

    technical_trace(
        stage_name="Final analytical synthesis",
        method=(
            "Freeze the validated headline metrics and assemble a presentation-ready final synthesis."
        ),
        inputs=[
            "All frozen sentiment, emotion, Stress, integrated and topic-analysis results",
        ],
        outputs=[
            "data/results/final_synthesis/core_metrics.csv",
            "data/results/final_synthesis/emotion_stress_association.csv",
            "data/results/final_synthesis/final_findings.json",
            "data/results/final_synthesis/negative_sentiment_vs_stress.csv",
            "data/results/final_synthesis/sentiment_distribution.csv",
            "data/results/final_synthesis/topic_stress_final.csv",
        ],
        keywords=[
            "final",
            "synthesis",
        ],
    )
