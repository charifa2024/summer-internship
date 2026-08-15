from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENTIMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "sentiment"
    / "sentiment_predictions.csv"
)

EMOTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
    / "goemotions_predictions_analysis_ready.csv"
)

STRESS_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
    / "hybrid_stress_predictions_analysis_ready.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "combined"
    / "sentiment_emotion_stress_predictions.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "combined"
    / "sentiment_emotion_stress_summary.json"
)


def check_unique_ids(
    df,
    name,
):
    if "record_id" not in df.columns:
        raise ValueError(
            f"{name}: missing record_id"
        )

    missing = int(
        df["record_id"]
        .isna()
        .sum()
    )

    duplicates = int(
        df["record_id"]
        .duplicated()
        .sum()
    )

    print(
        f"{name}: "
        f"rows={len(df)}, "
        f"missing_ids={missing}, "
        f"duplicate_ids={duplicates}"
    )

    if missing > 0:
        raise ValueError(
            f"{name}: missing record_id values"
        )

    if duplicates > 0:
        raise ValueError(
            f"{name}: duplicate record_id values"
        )


def main():

    print("=" * 78)
    print(
        "UNIFIED SENTIMENT × EMOTIONS × STRESS DATASET"
    )
    print("=" * 78)

    # ========================================================
    # LOAD
    # ========================================================

    sentiment = pd.read_csv(
        SENTIMENT_FILE
    )

    emotions = pd.read_csv(
        EMOTION_FILE
    )

    stress = pd.read_csv(
        STRESS_FILE
    )

    print("\nInput integrity:")

    check_unique_ids(
        sentiment,
        "Sentiment",
    )

    check_unique_ids(
        emotions,
        "GoEmotions",
    )

    check_unique_ids(
        stress,
        "Hybrid Stress",
    )

    # ========================================================
    # VERIFY SAME RECORD SET
    # ========================================================

    sentiment_ids = set(
        sentiment["record_id"]
    )

    emotion_ids = set(
        emotions["record_id"]
    )

    stress_ids = set(
        stress["record_id"]
    )

    print(
        "\nRecord-ID differences:"
    )

    print(
        "Sentiment - Emotions:",
        len(
            sentiment_ids
            - emotion_ids
        ),
    )

    print(
        "Emotions - Sentiment:",
        len(
            emotion_ids
            - sentiment_ids
        ),
    )

    print(
        "Sentiment - Stress:",
        len(
            sentiment_ids
            - stress_ids
        ),
    )

    print(
        "Stress - Sentiment:",
        len(
            stress_ids
            - sentiment_ids
        ),
    )

    if not (
        sentiment_ids
        == emotion_ids
        == stress_ids
    ):
        raise ValueError(
            "The three analysis layers "
            "do not contain identical record_id sets."
        )

    # ========================================================
    # BASE / IDENTITY COLUMNS
    # ========================================================

    base_columns = [
        col
        for col in [
            "record_id",
            "source_id",
            "source",
            "platform",
            "community",
            "created_at_utc",
            "date_known",
            "is_recent_2025_plus",
            "full_text_raw",
            "text_clean_basic",
            "text_clean_lexical",
            "ai_tools",
            "research_themes",
            "word_count",
            "score",
            "num_comments",
        ]
        if col in sentiment.columns
    ]

    sentiment_columns = [
        "record_id",
        "vader_negative",
        "vader_neutral",
        "vader_positive",
        "vader_compound",
        "vader_label",
        "transformer_negative",
        "transformer_neutral",
        "transformer_positive",
        "transformer_label",
        "transformer_confidence",
        "methods_agree",
    ]

    missing_sentiment = [
        col
        for col in sentiment_columns
        if col not in sentiment.columns
    ]

    if missing_sentiment:
        raise ValueError(
            "Missing sentiment columns: "
            f"{missing_sentiment}"
        )

    # ========================================================
    # EMOTION COLUMNS
    # ========================================================

    emotion_core_columns = [
        "record_id",
        "predicted_emotions",
        "emotion_count",
        "top_emotion",
        "top_emotion_probability",
        "top_emotion_threshold",
        "top_emotion_passes_threshold",
        "neutral_with_other_emotions",
    ]

    emotion_prediction_columns = [
        col
        for col in emotions.columns
        if (
            col.startswith(
                "emotion_"
            )
            and (
                col.endswith(
                    "_pred"
                )
                or col.endswith(
                    "_prob"
                )
            )
        )
    ]

    emotion_columns = (
        emotion_core_columns
        + emotion_prediction_columns
    )

    missing_emotion = [
        col
        for col in emotion_core_columns
        if col not in emotions.columns
    ]

    if missing_emotion:
        raise ValueError(
            "Missing emotion columns: "
            f"{missing_emotion}"
        )

    # ========================================================
    # STRESS COLUMNS
    # ========================================================

    stress_columns = [
        "record_id",
        "hybrid_stress_prediction",
        "hybrid_stress_label",
        "hybrid_stress_probability",
        "hybrid_no_stress_probability",
    ]

    missing_stress = [
        col
        for col in stress_columns
        if col not in stress.columns
    ]

    if missing_stress:
        raise ValueError(
            "Missing Stress columns: "
            f"{missing_stress}"
        )

    # ========================================================
    # MERGE
    # ========================================================

    base = sentiment[
        base_columns
    ].copy()

    sentiment_layer = sentiment[
        sentiment_columns
    ].copy()

    emotion_layer = emotions[
        emotion_columns
    ].copy()

    stress_layer = stress[
        stress_columns
    ].copy()

    unified = (
        base
        .merge(
            sentiment_layer,
            on="record_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            emotion_layer,
            on="record_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            stress_layer,
            on="record_id",
            how="inner",
            validate="one_to_one",
        )
    )

    # ========================================================
    # FINAL INTEGRITY
    # ========================================================

    if len(unified) != 2666:
        raise ValueError(
            "Expected 2,666 unified records, "
            f"found {len(unified)}."
        )

    if unified[
        "record_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate record_id found "
            "after merge."
        )

    if set(
        unified["record_id"]
    ) != sentiment_ids:
        raise ValueError(
            "Unified record set changed "
            "during merge."
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # BASIC SUMMARY
    # ========================================================

    summary = {
        "records": int(
            len(unified)
        ),
        "columns": int(
            len(unified.columns)
        ),
        "unique_record_ids": int(
            unified[
                "record_id"
            ].nunique()
        ),
        "vader_distribution": (
            unified[
                "vader_label"
            ]
            .value_counts()
            .to_dict()
        ),
        "transformer_distribution": (
            unified[
                "transformer_label"
            ]
            .value_counts()
            .to_dict()
        ),
        "hybrid_stress_distribution": (
            unified[
                "hybrid_stress_label"
            ]
            .value_counts()
            .to_dict()
        ),
        "average_emotion_count": float(
            unified[
                "emotion_count"
            ].mean()
        ),
        "sentiment_methods_agreement_rate": float(
            unified[
                "methods_agree"
            ].fillna(False)
            .astype(bool)
            .mean()
        ),
        "interpretation_warning": (
            "Sentiment, emotion, and Stress "
            "fields are model outputs and must "
            "be interpreted according to their "
            "respective validation limitations."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL
    # ========================================================

    print("\n" + "=" * 78)
    print("FINAL INTEGRITY")
    print("=" * 78)

    print(
        "Unified records:",
        len(unified),
    )

    print(
        "Unique record IDs:",
        unified[
            "record_id"
        ].nunique(),
    )

    print(
        "Columns:",
        len(
            unified.columns
        ),
    )

    print(
        "\nSummary:"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"\nUnified dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
