
from __future__ import annotations

from pathlib import Path
import ast
import json
import logging

import pandas as pd
import plotly.express as px
import streamlit as st


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Developer Stress and Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
)


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "data" / "processed").exists():
            return candidate

    raise FileNotFoundError(
        "Project root not found. Run Streamlit from the project root "
        "or from the dashboard folder."
    )


PROJECT_ROOT = find_project_root()

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "dashboard.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
logger.info("Dashboard started")

TOPIC_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "topics"
    / "topic_assignments.csv"
)

EMOTION_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
    / "stress_emotion_predictions.csv"
)

SENTIMENT_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "sentiment"
    / "sentiment_predictions.csv"
)

VADER_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "sentiment"
    / "vader_predictions.csv"
)

ANALYSIS_READY_JSONL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analysis_ready_posts.jsonl"
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_list(value):
    if isinstance(value, list):
        return value

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        value = value.strip()

        if not value or value.lower() in {
            "nan",
            "none",
            "null",
            "[]",
        }:
            return []

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass

        return [value]

    return [value]


@st.cache_data(show_spinner=False)
def load_best_available_dataset():
    if TOPIC_CSV.exists():
        dataframe = pd.read_csv(TOPIC_CSV)
        source_name = "Topic assignments"
        source_path = TOPIC_CSV
    elif EMOTION_CSV.exists():
        dataframe = pd.read_csv(EMOTION_CSV)
        source_name = "Stress and emotion predictions"
        source_path = EMOTION_CSV
    elif SENTIMENT_CSV.exists():
        dataframe = pd.read_csv(SENTIMENT_CSV)
        source_name = "Combined sentiment predictions"
        source_path = SENTIMENT_CSV
    elif VADER_CSV.exists():
        dataframe = pd.read_csv(VADER_CSV)
        source_name = "VADER sentiment predictions"
        source_path = VADER_CSV
    elif ANALYSIS_READY_JSONL.exists():
        dataframe = pd.read_json(
            ANALYSIS_READY_JSONL,
            lines=True,
        )
        source_name = "Analysis-ready dataset"
        source_path = ANALYSIS_READY_JSONL
    else:
        raise FileNotFoundError(
            "No analysis output was found. Run the preprocessing, "
            "sentiment, stress, and topic notebooks first."
        )

    list_columns = [
        "ai_tools",
        "research_themes",
        "detected_emotions",
        "stress_causes",
        "stress_evidence",
    ]

    for column in list_columns:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].map(ensure_list)

    if "created_at_utc" in dataframe.columns:
        dataframe["created_at_utc"] = pd.to_datetime(
            dataframe["created_at_utc"],
            errors="coerce",
            utc=True,
        )

    if "transformer_label" in dataframe.columns:
        dataframe["dashboard_sentiment"] = dataframe[
            "transformer_label"
        ]
    elif "vader_label" in dataframe.columns:
        dataframe["dashboard_sentiment"] = dataframe[
            "vader_label"
        ]
    elif "selected_sentiment_label" in dataframe.columns:
        dataframe["dashboard_sentiment"] = dataframe[
            "selected_sentiment_label"
        ]
    else:
        dataframe["dashboard_sentiment"] = "Unavailable"

    return dataframe, source_name, str(source_path)


def available_values(dataframe, column):
    if column not in dataframe.columns:
        return []

    values = (
        dataframe[column]
        .dropna()
        .astype(str)
        .loc[lambda series: ~series.isin(["", "nan", "None"])]
        .unique()
        .tolist()
    )

    return sorted(values)


def filter_list_column(dataframe, column, selected_values):
    if not selected_values or column not in dataframe.columns:
        return dataframe

    selected_set = set(selected_values)

    mask = dataframe[column].map(
        lambda values: bool(
            selected_set.intersection(
                set(ensure_list(values))
            )
        )
    )

    return dataframe.loc[mask]


def normalized_distribution(dataframe, column):
    if column not in dataframe.columns or dataframe.empty:
        return pd.DataFrame(
            columns=[column, "records", "percentage"]
        )

    result = (
        dataframe[column]
        .fillna("Unavailable")
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="records")
    )

    result["percentage"] = (
        result["records"]
        / len(dataframe)
        * 100
    ).round(2)

    return result


def explode_distribution(dataframe, column, item_name):
    if column not in dataframe.columns or dataframe.empty:
        return pd.DataFrame(
            columns=[item_name, "records", "percentage"]
        )

    exploded = (
        dataframe[[column]]
        .explode(column)
        .reset_index(drop=True)
    )

    exploded[column] = (
        exploded[column]
        .replace("", pd.NA)
    )

    exploded = exploded.dropna(
        subset=[column]
    )

    if exploded.empty:
        return pd.DataFrame(
            columns=[item_name, "records", "percentage"]
        )

    result = (
        exploded[column]
        .astype(str)
        .value_counts()
        .rename_axis(item_name)
        .reset_index(name="records")
    )

    result["percentage"] = (
        result["records"]
        / len(dataframe)
        * 100
    ).round(2)

    return result


def render_empty_notice(message):
    st.info(message)


def normalize_boolean(value):
    """Safely normalize Boolean values loaded from CSV files."""
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def choose_text_column(dataframe):
    """Return the best available text column."""
    for column in [
        "full_text_raw",
        "text_clean_basic",
        "text_clean_lexical",
    ]:
        if column in dataframe.columns:
            return column

    return None


def choose_representative_row(dataframe, column, keywords):
    """Select a representative row, preferring high confidence."""
    if dataframe.empty or column not in dataframe.columns:
        return None

    labels = dataframe[column].fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=dataframe.index)

    for keyword in keywords:
        mask = mask | labels.str.contains(
            str(keyword).lower(),
            regex=False,
        )

    candidates = dataframe.loc[mask].copy()

    if candidates.empty:
        return None

    for confidence_column in [
        "topic_confidence",
        "transformer_confidence",
        "sentiment_confidence",
        "stress_intensity_score",
        "confidence",
    ]:
        if confidence_column in candidates.columns:
            candidates["_representative_score"] = pd.to_numeric(
                candidates[confidence_column],
                errors="coerce",
            ).fillna(-1)

            candidates = candidates.sort_values(
                "_representative_score",
                ascending=False,
            )
            break

    return candidates.iloc[0]


def render_representative_example(row, heading):
    """Render one representative record and its labels."""
    st.markdown(f"#### {heading}")

    if row is None:
        st.info(
            "No matching example is available "
            "for the current filters."
        )
        return

    text_column = choose_text_column(row.to_frame().T)

    if text_column is None:
        st.info("No text column is available.")
        return

    text_value = str(row.get(text_column, "")).strip()

    if len(text_value) > 1200:
        text_value = text_value[:1200].rstrip() + "…"

    st.write(text_value)

    metadata = []

    for column, label in [
        ("platform", "Platform"),
        ("dashboard_sentiment", "Sentiment"),
        ("stress_direction", "Stress"),
        ("primary_emotion", "Emotion"),
        ("topic_name", "Topic"),
    ]:
        if column not in row.index:
            continue

        value = row.get(column)

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value and value.lower() not in {
            "nan",
            "none",
            "unavailable",
        }:
            metadata.append(f"**{label}:** {value}")

    if metadata:
        st.caption(" | ".join(metadata))


def dominant_category(dataframe, column):
    """Return the most frequent usable value in a column."""
    if dataframe.empty or column not in dataframe.columns:
        return None

    values = dataframe[column].dropna().astype(str).str.strip()

    values = values.loc[
        ~values.str.lower().isin(
            [
                "",
                "nan",
                "none",
                "unavailable",
                "no clear emotion",
                "no clear stress",
            ]
        )
    ]

    if values.empty:
        return None

    return values.value_counts().index[0]


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
try:
    df, data_source_name, data_source_path = (
        load_best_available_dataset()
    )

    logger.info(
        "Loaded %s records from %s",
        len(df),
        data_source_path,
    )
except Exception as error:
    logger.exception("Dashboard data-loading error")
    st.error(str(error))
    st.stop()


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.title("Developer Stress, Emotion and Sentiment Dashboard")

st.caption(
    "Analysis of developer discussions about AI tools. "
    "This dashboard detects textual patterns and does not provide "
    "clinical diagnoses."
)

with st.expander("Data source and methodological note"):
    st.write(f"**Loaded source:** {data_source_name}")
    st.code(data_source_path)
    st.write(
        "Sentiment, stress, emotion, and topic labels are model- or "
        "rule-generated analytical outputs. LLM-assisted validation "
        "labels are provisional references, not independent human "
        "ground truth."
    )


# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
st.sidebar.header("Filters")

platform_options = available_values(df, "platform")
selected_platforms = st.sidebar.multiselect(
    "Platform",
    options=platform_options,
    default=platform_options,
)

sentiment_options = available_values(
    df,
    "dashboard_sentiment",
)
selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=sentiment_options,
    default=sentiment_options,
)

stress_options = available_values(
    df,
    "stress_direction",
)
selected_stress = st.sidebar.multiselect(
    "Stress direction",
    options=stress_options,
    default=stress_options,
)

topic_options = available_values(
    df,
    "topic_name",
)
selected_topics = st.sidebar.multiselect(
    "Topic",
    options=topic_options,
    default=topic_options,
)

tool_distribution_for_filters = explode_distribution(
    df,
    "ai_tools",
    "ai_tool",
)
tool_options = (
    tool_distribution_for_filters["ai_tool"].tolist()
    if not tool_distribution_for_filters.empty
    else []
)

selected_tools = st.sidebar.multiselect(
    "AI tool",
    options=tool_options,
)

filtered_df = df.copy()

if selected_platforms and "platform" in filtered_df.columns:
    filtered_df = filtered_df.loc[
        filtered_df["platform"].isin(
            selected_platforms
        )
    ]

if (
    selected_sentiments
    and "dashboard_sentiment" in filtered_df.columns
):
    filtered_df = filtered_df.loc[
        filtered_df["dashboard_sentiment"].isin(
            selected_sentiments
        )
    ]

if (
    selected_stress
    and "stress_direction" in filtered_df.columns
):
    filtered_df = filtered_df.loc[
        filtered_df["stress_direction"].isin(
            selected_stress
        )
    ]

if (
    selected_topics
    and "topic_name" in filtered_df.columns
):
    filtered_df = filtered_df.loc[
        filtered_df["topic_name"].isin(
            selected_topics
        )
    ]

filtered_df = filter_list_column(
    filtered_df,
    "ai_tools",
    selected_tools,
)

if (
    "created_at_utc" in filtered_df.columns
    and filtered_df["created_at_utc"].notna().any()
):
    min_date = filtered_df["created_at_utc"].min().date()
    max_date = filtered_df["created_at_utc"].max().date()

    selected_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if (
        isinstance(selected_dates, tuple)
        and len(selected_dates) == 2
    ):
        start_date, end_date = selected_dates

        filtered_df = filtered_df.loc[
            filtered_df["created_at_utc"].dt.date.between(
                start_date,
                end_date,
            )
        ]


if filtered_df.empty:
    st.warning(
        "No records match the selected filters. "
        "Change or reset one or more filters."
    )


# -------------------------------------------------------------------
# KPI section
# -------------------------------------------------------------------
total_records = len(filtered_df)

platform_count = (
    filtered_df["platform"].nunique()
    if "platform" in filtered_df.columns
    else 0
)

explicit_stress_records = (
    int(
        filtered_df["stress_signal"]
        .map(normalize_boolean)
        .sum()
    )
    if "stress_signal" in filtered_df.columns
    else 0
)

stress_rate = (
    explicit_stress_records
    / total_records
    * 100
    if total_records
    else 0
)

topic_count = (
    filtered_df["topic_name"].nunique()
    if "topic_name" in filtered_df.columns
    else 0
)

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

kpi_1.metric(
    "Filtered records",
    f"{total_records:,}",
)

kpi_2.metric(
    "Platforms",
    f"{platform_count:,}",
)

kpi_3.metric(
    "Explicit stress signals",
    f"{stress_rate:.1f}%",
)

kpi_4.metric(
    "Discovered topics",
    f"{topic_count:,}",
)


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
(
    overview_tab,
    sentiment_tab,
    stress_tab,
    topic_tab,
    time_tab,
    examples_tab,
    explorer_tab,
) = st.tabs(
    [
        "Overview",
        "Sentiment",
        "Stress & emotions",
        "Topics",
        "Time trends",
        "Examples & recommendations",
        "Data explorer",
    ]
)


# -------------------------------------------------------------------
# Overview tab
# -------------------------------------------------------------------
with overview_tab:
    st.subheader("Dataset overview")

    platform_distribution = normalized_distribution(
        filtered_df,
        "platform",
    )

    left, right = st.columns(2)

    with left:
        if platform_distribution.empty:
            render_empty_notice(
                "Platform information is unavailable."
            )
        else:
            figure = px.bar(
                platform_distribution,
                x="platform",
                y="records",
                title="Records by platform",
                labels={
                    "platform": "Platform",
                    "records": "Records",
                },
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    with right:
        tool_distribution = explode_distribution(
            filtered_df,
            "ai_tools",
            "ai_tool",
        ).head(12)

        if tool_distribution.empty:
            render_empty_notice(
                "AI-tool information is unavailable."
            )
        else:
            figure = px.bar(
                tool_distribution.sort_values("records"),
                x="records",
                y="ai_tool",
                orientation="h",
                title="Most-mentioned AI tools",
                labels={
                    "ai_tool": "AI tool",
                    "records": "Mentions",
                },
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    theme_distribution = explode_distribution(
        filtered_df,
        "research_themes",
        "research_theme",
    ).head(12)

    if not theme_distribution.empty:
        figure = px.bar(
            theme_distribution.sort_values("records"),
            x="records",
            y="research_theme",
            orientation="h",
            title="Predefined research themes",
            labels={
                "research_theme": "Research theme",
                "records": "Records",
            },
        )
        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# -------------------------------------------------------------------
# Sentiment tab
# -------------------------------------------------------------------
with sentiment_tab:
    st.subheader("General sentiment")

    sentiment_distribution = normalized_distribution(
        filtered_df,
        "dashboard_sentiment",
    )

    if sentiment_distribution.empty:
        render_empty_notice(
            "Sentiment predictions are unavailable."
        )
    else:
        left, right = st.columns(2)

        with left:
            figure = px.bar(
                sentiment_distribution,
                x="dashboard_sentiment",
                y="records",
                title="Sentiment distribution",
                labels={
                    "dashboard_sentiment": "Sentiment",
                    "records": "Records",
                },
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        with right:
            figure = px.pie(
                sentiment_distribution,
                names="dashboard_sentiment",
                values="records",
                title="Sentiment share",
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    if (
        "platform" in filtered_df.columns
        and "dashboard_sentiment" in filtered_df.columns
        and not filtered_df.empty
    ):
        sentiment_by_platform = (
            filtered_df.groupby(
                [
                    "platform",
                    "dashboard_sentiment",
                ]
            )
            .size()
            .reset_index(name="records")
        )

        figure = px.bar(
            sentiment_by_platform,
            x="platform",
            y="records",
            color="dashboard_sentiment",
            barmode="stack",
            title="Sentiment by platform",
            labels={
                "platform": "Platform",
                "records": "Records",
                "dashboard_sentiment": "Sentiment",
            },
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# -------------------------------------------------------------------
# Stress and emotions tab
# -------------------------------------------------------------------
with stress_tab:
    st.subheader("Stress and emotion signals")

    stress_distribution = normalized_distribution(
        filtered_df,
        "stress_direction",
    )

    emotion_distribution = normalized_distribution(
        filtered_df,
        "primary_emotion",
    )

    left, right = st.columns(2)

    with left:
        if stress_distribution.empty:
            render_empty_notice(
                "Stress-direction results are unavailable."
            )
        else:
            figure = px.bar(
                stress_distribution,
                x="stress_direction",
                y="records",
                title="Stress-direction distribution",
                labels={
                    "stress_direction": "Stress direction",
                    "records": "Records",
                },
            )
            figure.update_xaxes(
                tickangle=-25
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    with right:
        if emotion_distribution.empty:
            render_empty_notice(
                "Primary-emotion results are unavailable."
            )
        else:
            figure = px.bar(
                emotion_distribution.sort_values("records"),
                x="records",
                y="primary_emotion",
                orientation="h",
                title="Primary emotion signals",
                labels={
                    "primary_emotion": "Primary emotion",
                    "records": "Records",
                },
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    cause_distribution = explode_distribution(
        filtered_df,
        "stress_causes",
        "stress_cause",
    ).head(12)

    if not cause_distribution.empty:
        figure = px.bar(
            cause_distribution.sort_values("records"),
            x="records",
            y="stress_cause",
            orientation="h",
            title="Likely stress and emotion contexts",
            labels={
                "stress_cause": "Likely context",
                "records": "Records",
            },
        )
        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    if (
        "dashboard_sentiment" in filtered_df.columns
        and "stress_direction" in filtered_df.columns
        and not filtered_df.empty
    ):
        sentiment_stress = (
            filtered_df.groupby(
                [
                    "dashboard_sentiment",
                    "stress_direction",
                ]
            )
            .size()
            .reset_index(name="records")
        )

        figure = px.bar(
            sentiment_stress,
            x="dashboard_sentiment",
            y="records",
            color="stress_direction",
            barmode="stack",
            title="Sentiment compared with stress direction",
            labels={
                "dashboard_sentiment": "Sentiment",
                "records": "Records",
                "stress_direction": "Stress direction",
            },
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# -------------------------------------------------------------------
# Topics tab
# -------------------------------------------------------------------
with topic_tab:
    st.subheader("Discovered discussion topics")

    topic_distribution = normalized_distribution(
        filtered_df,
        "topic_name",
    )

    if topic_distribution.empty:
        render_empty_notice(
            "Topic-modeling outputs are unavailable. "
            "Run 07_topic_modeling.ipynb first."
        )
    else:
        figure = px.bar(
            topic_distribution.sort_values("records"),
            x="records",
            y="topic_name",
            orientation="h",
            title="Topic distribution",
            labels={
                "topic_name": "Topic",
                "records": "Records",
            },
        )
        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    if (
        "topic_name" in filtered_df.columns
        and "stress_direction" in filtered_df.columns
        and not filtered_df.empty
    ):
        stress_by_topic = (
            filtered_df.groupby(
                [
                    "topic_name",
                    "stress_direction",
                ]
            )
            .size()
            .reset_index(name="records")
        )

        figure = px.bar(
            stress_by_topic,
            x="topic_name",
            y="records",
            color="stress_direction",
            barmode="stack",
            title="Stress direction within topics",
            labels={
                "topic_name": "Topic",
                "records": "Records",
                "stress_direction": "Stress direction",
            },
        )

        figure.update_xaxes(
            tickangle=-30
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    if (
        "topic_name" in filtered_df.columns
        and "dashboard_sentiment" in filtered_df.columns
        and not filtered_df.empty
    ):
        sentiment_by_topic = (
            filtered_df.groupby(
                [
                    "topic_name",
                    "dashboard_sentiment",
                ]
            )
            .size()
            .reset_index(name="records")
        )

        figure = px.bar(
            sentiment_by_topic,
            x="topic_name",
            y="records",
            color="dashboard_sentiment",
            barmode="stack",
            title="Sentiment within topics",
            labels={
                "topic_name": "Topic",
                "records": "Records",
                "dashboard_sentiment": "Sentiment",
            },
        )

        figure.update_xaxes(
            tickangle=-30
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    topic_keyword_path = (
        PROJECT_ROOT
        / "data"
        / "results"
        / "topics"
        / "topic_keywords.csv"
    )

    if topic_keyword_path.exists():
        topic_keywords = pd.read_csv(
            topic_keyword_path
        )

        st.subheader("Topic keywords")

        keyword_topic_options = sorted(
            topic_keywords[
                "topic_id"
            ].unique()
        )

        selected_keyword_topic = st.selectbox(
            "Choose a topic ID",
            options=keyword_topic_options,
        )

        selected_keyword_table = (
            topic_keywords.loc[
                topic_keywords[
                    "topic_id"
                ].eq(
                    selected_keyword_topic
                )
            ]
            .sort_values("rank")
        )

        st.dataframe(
            selected_keyword_table,
            use_container_width=True,
            hide_index=True,
        )


# -------------------------------------------------------------------
# Time trends tab
# -------------------------------------------------------------------
with time_tab:
    st.subheader("Evolution over time")

    if (
        "created_at_utc" not in filtered_df.columns
        or filtered_df["created_at_utc"].notna().sum() == 0
    ):
        render_empty_notice(
            "No valid date information is available."
        )
    else:
        dated_df = filtered_df.dropna(
            subset=["created_at_utc"]
        ).copy()

        dated_df["month"] = (
            dated_df["created_at_utc"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly_records = (
            dated_df.groupby("month")
            .size()
            .reset_index(name="records")
        )

        figure = px.line(
            monthly_records,
            x="month",
            y="records",
            markers=True,
            title="Number of records over time",
            labels={
                "month": "Month",
                "records": "Records",
            },
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        if "stress_direction" in dated_df.columns:
            stress_over_time = (
                dated_df.groupby(
                    [
                        "month",
                        "stress_direction",
                    ]
                )
                .size()
                .reset_index(name="records")
            )

            figure = px.line(
                stress_over_time,
                x="month",
                y="records",
                color="stress_direction",
                markers=True,
                title="Stress-direction signals over time",
                labels={
                    "month": "Month",
                    "records": "Records",
                    "stress_direction": "Stress direction",
                },
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        if "topic_name" in dated_df.columns:
            topic_over_time = (
                dated_df.groupby(
                    [
                        "month",
                        "topic_name",
                    ]
                )
                .size()
                .reset_index(name="records")
            )

            figure = px.line(
                topic_over_time,
                x="month",
                y="records",
                color="topic_name",
                markers=True,
                title="Topics over time",
                labels={
                    "month": "Month",
                    "records": "Records",
                    "topic_name": "Topic",
                },
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )


# -------------------------------------------------------------------
# Representative examples and recommendations tab
# -------------------------------------------------------------------
with examples_tab:
    st.subheader("Representative examples")

    st.caption(
        "Examples are selected automatically from the currently "
        "filtered records. They illustrate model-generated patterns "
        "and are not independent human ground truth."
    )

    positive_example = choose_representative_row(
        filtered_df,
        "dashboard_sentiment",
        ["positive"],
    )
    negative_example = choose_representative_row(
        filtered_df,
        "dashboard_sentiment",
        ["negative"],
    )
    increased_stress_example = choose_representative_row(
        filtered_df,
        "stress_direction",
        ["increased", "increase", "distress"],
    )
    reduced_stress_example = choose_representative_row(
        filtered_df,
        "stress_direction",
        ["reduced", "reduce", "relief"],
    )

    left, right = st.columns(2)

    with left:
        render_representative_example(
            positive_example,
            "Positive sentiment example",
        )
        render_representative_example(
            increased_stress_example,
            "Increased-stress example",
        )

    with right:
        render_representative_example(
            negative_example,
            "Negative sentiment example",
        )
        render_representative_example(
            reduced_stress_example,
            "Reduced-stress example",
        )

    if "topic_name" in filtered_df.columns and not filtered_df.empty:
        st.subheader("Representative examples by main topic")

        main_topics = (
            filtered_df["topic_name"]
            .dropna()
            .astype(str)
            .loc[
                lambda values: ~values.str.lower().isin(
                    ["", "nan", "none", "unavailable"]
                )
            ]
            .value_counts()
            .head(3)
            .index
            .tolist()
        )

        for topic_name in main_topics:
            topic_example = choose_representative_row(
                filtered_df,
                "topic_name",
                [topic_name],
            )
            render_representative_example(
                topic_example,
                f"Topic: {topic_name}",
            )

    st.divider()
    st.subheader("Automatic synthesis of current filters")

    dominant_sentiment = dominant_category(
        filtered_df,
        "dashboard_sentiment",
    )
    dominant_emotion = dominant_category(
        filtered_df,
        "primary_emotion",
    )
    dominant_stress = dominant_category(
        filtered_df,
        "stress_direction",
    )
    dominant_topic = dominant_category(
        filtered_df,
        "topic_name",
    )

    tool_distribution = explode_distribution(
        filtered_df,
        "ai_tools",
        "ai_tool",
    )

    dominant_tool = (
        tool_distribution.iloc[0]["ai_tool"]
        if not tool_distribution.empty
        else None
    )

    findings = []

    if dominant_sentiment:
        findings.append(
            f"Dominant sentiment: **{dominant_sentiment}**."
        )
    if dominant_emotion:
        findings.append(
            "Most frequent explicit emotion signal: "
            f"**{dominant_emotion}**."
        )
    if dominant_stress:
        findings.append(
            f"Dominant stress direction: **{dominant_stress}**."
        )
    if dominant_topic:
        findings.append(
            f"Most frequent discovered topic: **{dominant_topic}**."
        )
    if dominant_tool:
        findings.append(
            f"Most-mentioned AI tool: **{dominant_tool}**."
        )

    if findings:
        for finding in findings:
            st.markdown(f"- {finding}")
    else:
        st.info(
            "There are not enough filtered records "
            "to generate a synthesis."
        )

    st.subheader("Cautious recommendations")

    recommendations = [
        (
            "Provide clear guidance on reviewing, testing, and "
            "verifying AI-generated code before use."
        ),
        (
            "Communicate the limits of AI assistants, especially "
            "hallucinations, privacy risks, and context loss."
        ),
        (
            "Support responsible adoption through training that "
            "keeps developers in control of final technical decisions."
        ),
    ]

    combined_signal = " ".join(
        str(value).lower()
        for value in [
            dominant_emotion,
            dominant_stress,
            dominant_topic,
        ]
        if value is not None
    )

    if any(
        term in combined_signal
        for term in ["job", "security", "replace", "fear"]
    ):
        recommendations.append(
            "Address job-security concerns directly and present AI "
            "as an assistive tool rather than an automatic replacement "
            "for developer expertise."
        )

    if any(
        term in combined_signal
        for term in [
            "stress",
            "anxiety",
            "burnout",
            "frustration",
        ]
    ):
        recommendations.append(
            "Introduce AI tools gradually and monitor whether they "
            "reduce repetitive work or create additional cognitive load."
        )

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    st.caption(
        "These recommendations are analytical implications from "
        "observed public-discussion patterns. They are not clinical, "
        "causal, or universally generalizable."
    )


# -------------------------------------------------------------------
# Data explorer tab
# -------------------------------------------------------------------
with explorer_tab:
    st.subheader("Filtered data explorer")

    display_columns = [
        column
        for column in [
            "record_id",
            "platform",
            "created_at_utc",
            "full_text_raw",
            "text_clean_basic",
            "dashboard_sentiment",
            "stress_direction",
            "stress_intensity",
            "primary_emotion",
            "stress_causes",
            "topic_id",
            "topic_name",
            "topic_confidence",
            "ai_tools",
            "research_themes",
        ]
        if column in filtered_df.columns
    ]

    search_query = st.text_input(
        "Search inside the text",
        value="",
    ).strip()

    explorer_df = filtered_df.copy()

    if search_query:
        searchable_column = (
            "full_text_raw"
            if "full_text_raw" in explorer_df.columns
            else "text_clean_basic"
        )

        if searchable_column in explorer_df.columns:
            explorer_df = explorer_df.loc[
                explorer_df[
                    searchable_column
                ]
                .fillna("")
                .astype(str)
                .str.contains(
                    search_query,
                    case=False,
                    regex=False,
                )
            ]

    max_rows = st.slider(
        "Maximum displayed rows",
        min_value=20,
        max_value=500,
        value=100,
        step=20,
    )

    st.dataframe(
        explorer_df[
            display_columns
        ].head(max_rows),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = explorer_df[
        display_columns
    ].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download filtered records as CSV",
        data=csv_bytes,
        file_name="filtered_dashboard_records.csv",
        mime="text/csv",
    )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.divider()

st.caption(
    "Interpret results as patterns in public developer discussions. "
    "Do not interpret the stress analysis as a medical or psychological "
    "diagnosis."
)
