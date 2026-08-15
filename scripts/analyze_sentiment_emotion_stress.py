from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "combined"
    / "sentiment_emotion_stress_predictions.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "combined"
)


def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main():

    print("=" * 78)
    print("SENTIMENT × EMOTIONS × STRESS ANALYSIS")
    print("=" * 78)

    df = pd.read_csv(
        INPUT_FILE
    )

    if len(df) != 2666:
        raise ValueError(
            f"Expected 2666 records, found {len(df)}"
        )

    # ========================================================
    # 1. VADER × TRANSFORMER
    # ========================================================

    vader_transformer_counts = pd.crosstab(
        df["vader_label"],
        df["transformer_label"],
    )

    vader_transformer_percent = pd.crosstab(
        df["vader_label"],
        df["transformer_label"],
        normalize="index",
    ) * 100

    vader_transformer_counts.to_csv(
        RESULTS_DIR
        / "vader_vs_transformer_counts.csv"
    )

    vader_transformer_percent.to_csv(
        RESULTS_DIR
        / "vader_vs_transformer_row_percent.csv"
    )

    # ========================================================
    # 2. VADER × STRESS
    # ========================================================

    vader_stress_counts = pd.crosstab(
        df["vader_label"],
        df["hybrid_stress_label"],
    )

    vader_stress_percent = pd.crosstab(
        df["vader_label"],
        df["hybrid_stress_label"],
        normalize="index",
    ) * 100

    vader_stress_counts.to_csv(
        RESULTS_DIR
        / "vader_vs_hybrid_stress_counts.csv"
    )

    vader_stress_percent.to_csv(
        RESULTS_DIR
        / "vader_vs_hybrid_stress_row_percent.csv"
    )

    # ========================================================
    # 3. TRANSFORMER × STRESS
    # ========================================================

    transformer_stress_counts = pd.crosstab(
        df["transformer_label"],
        df["hybrid_stress_label"],
    )

    transformer_stress_percent = pd.crosstab(
        df["transformer_label"],
        df["hybrid_stress_label"],
        normalize="index",
    ) * 100

    transformer_stress_counts.to_csv(
        RESULTS_DIR
        / "transformer_vs_hybrid_stress_counts.csv"
    )

    transformer_stress_percent.to_csv(
        RESULTS_DIR
        / "transformer_vs_hybrid_stress_row_percent.csv"
    )

    # ========================================================
    # 4. EMOTION PREVALENCE — STRESS VS NO STRESS
    # ========================================================

    emotion_pred_columns = [
        col
        for col in df.columns
        if (
            col.startswith("emotion_")
            and col.endswith("_pred")
        )
    ]

    stress_df = df[
        df["hybrid_stress_label"]
        == "Stress"
    ]

    no_stress_df = df[
        df["hybrid_stress_label"]
        == "No stress"
    ]

    emotion_rows = []

    for column in emotion_pred_columns:

        emotion = (
            column
            .replace("emotion_", "")
            .replace("_pred", "")
        )

        stress_count = int(
            stress_df[column].sum()
        )

        no_stress_count = int(
            no_stress_df[column].sum()
        )

        stress_rate = safe_rate(
            stress_count,
            len(stress_df),
        )

        no_stress_rate = safe_rate(
            no_stress_count,
            len(no_stress_df),
        )

        rate_difference = (
            stress_rate
            - no_stress_rate
        )

        rate_ratio = (
            stress_rate / no_stress_rate
            if no_stress_rate > 0
            else np.nan
        )

        emotion_rows.append(
            {
                "emotion": emotion,
                "stress_count": stress_count,
                "stress_rate": stress_rate,
                "no_stress_count": no_stress_count,
                "no_stress_rate": no_stress_rate,
                "rate_difference": rate_difference,
                "rate_ratio": rate_ratio,
            }
        )

    emotion_comparison = (
        pd.DataFrame(
            emotion_rows
        )
        .sort_values(
            "rate_difference",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    emotion_comparison.to_csv(
        RESULTS_DIR
        / "emotion_prevalence_stress_vs_no_stress.csv",
        index=False,
    )

    # ========================================================
    # 5. TOP EMOTION × STRESS
    # ========================================================

    top_emotion_stress_counts = pd.crosstab(
        df["top_emotion"],
        df["hybrid_stress_label"],
    )

    top_emotion_stress_percent = pd.crosstab(
        df["top_emotion"],
        df["hybrid_stress_label"],
        normalize="index",
    ) * 100

    top_emotion_stress_counts.to_csv(
        RESULTS_DIR
        / "top_emotion_vs_stress_counts.csv"
    )

    top_emotion_stress_percent.to_csv(
        RESULTS_DIR
        / "top_emotion_vs_stress_row_percent.csv"
    )

    # ========================================================
    # 6. NEGATIVE SENTIMENT IS NOT STRESS
    # ========================================================

    vader_negative = df[
        df["vader_label"]
        == "Negative"
    ]

    transformer_negative = df[
        df["transformer_label"]
        == "Negative"
    ]

    vader_negative_stress = int(
        (
            vader_negative[
                "hybrid_stress_label"
            ]
            == "Stress"
        ).sum()
    )

    transformer_negative_stress = int(
        (
            transformer_negative[
                "hybrid_stress_label"
            ]
            == "Stress"
        ).sum()
    )

    # ========================================================
    # 7. STRESS POSTS' SENTIMENT DISTRIBUTION
    # ========================================================

    stress_vader_distribution = (
        stress_df[
            "vader_label"
        ]
        .value_counts()
        .rename_axis("vader_label")
        .reset_index(
            name="count"
        )
    )

    stress_vader_distribution[
        "rate"
    ] = (
        stress_vader_distribution[
            "count"
        ]
        / len(stress_df)
    )

    stress_vader_distribution.to_csv(
        RESULTS_DIR
        / "stress_posts_vader_distribution.csv",
        index=False,
    )

    stress_transformer_distribution = (
        stress_df[
            "transformer_label"
        ]
        .value_counts()
        .rename_axis(
            "transformer_label"
        )
        .reset_index(
            name="count"
        )
    )

    stress_transformer_distribution[
        "rate"
    ] = (
        stress_transformer_distribution[
            "count"
        ]
        / len(stress_df)
    )

    stress_transformer_distribution.to_csv(
        RESULTS_DIR
        / "stress_posts_transformer_distribution.csv",
        index=False,
    )

    # ========================================================
    # 8. SUMMARY
    # ========================================================

    summary = {
        "records": int(
            len(df)
        ),
        "stress_records": int(
            len(stress_df)
        ),
        "no_stress_records": int(
            len(no_stress_df)
        ),
        "stress_rate": float(
            len(stress_df)
            / len(df)
        ),
        "sentiment_method_agreement_rate": float(
            df[
                "methods_agree"
            ]
            .fillna(False)
            .astype(bool)
            .mean()
        ),
        "vader_negative_records": int(
            len(vader_negative)
        ),
        "vader_negative_and_stress": (
            vader_negative_stress
        ),
        "vader_negative_stress_rate": float(
            safe_rate(
                vader_negative_stress,
                len(vader_negative),
            )
        ),
        "transformer_negative_records": int(
            len(transformer_negative)
        ),
        "transformer_negative_and_stress": (
            transformer_negative_stress
        ),
        "transformer_negative_stress_rate": float(
            safe_rate(
                transformer_negative_stress,
                len(transformer_negative),
            )
        ),
        "interpretation": (
            "Negative sentiment and predicted Stress "
            "are related analytical dimensions but "
            "must not be treated as equivalent."
        ),
    }

    (
        RESULTS_DIR
        / "sentiment_emotion_stress_analysis_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print("\nVADER × TRANSFORMER")
    print(vader_transformer_counts)

    print("\nVADER × HYBRID STRESS — ROW %")
    print(
        vader_stress_percent.round(2)
    )

    print(
        "\nTRANSFORMER × HYBRID STRESS — ROW %"
    )
    print(
        transformer_stress_percent.round(2)
    )

    print(
        "\nTOP EMOTIONS MORE COMMON IN "
        "PREDICTED STRESS POSTS"
    )

    print(
        emotion_comparison[
            [
                "emotion",
                "stress_rate",
                "no_stress_rate",
                "rate_difference",
                "rate_ratio",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print(
        "\nNEGATIVE SENTIMENT ≠ STRESS"
    )

    print(
        "VADER negative posts:",
        len(vader_negative),
    )

    print(
        "VADER negative + Stress:",
        vader_negative_stress,
        (
            f"({safe_rate(vader_negative_stress, len(vader_negative)):.2%})"
        ),
    )

    print(
        "\nTransformer negative posts:",
        len(transformer_negative),
    )

    print(
        "Transformer negative + Stress:",
        transformer_negative_stress,
        (
            f"({safe_rate(transformer_negative_stress, len(transformer_negative)):.2%})"
        ),
    )

    print(
        "\nStress posts — VADER:"
    )

    print(
        stress_vader_distribution.to_string(
            index=False
        )
    )

    print(
        "\nStress posts — Transformer:"
    )

    print(
        stress_transformer_distribution.to_string(
            index=False
        )
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


if __name__ == "__main__":
    main()
