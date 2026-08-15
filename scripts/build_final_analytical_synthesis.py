from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMBINED_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "combined"
)

EMOTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
)

TOPICS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "stress_topics"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "final_synthesis"
)


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def main():

    print("=" * 80)
    print("FINAL ANALYTICAL SYNTHESIS")
    print("=" * 80)

    # ========================================================
    # LOAD VALIDATED RESULTS
    # ========================================================

    unified_summary = load_json(
        COMBINED_DIR
        / "sentiment_emotion_stress_summary.json"
    )

    integrated_summary = load_json(
        COMBINED_DIR
        / "sentiment_emotion_stress_analysis_summary.json"
    )

    stress_summary = load_json(
        EMOTIONS_DIR
        / "hybrid_stress_analysis_ready_summary.json"
    )

    emotion_summary = load_json(
        EMOTIONS_DIR
        / "goemotions_analysis_ready_summary.json"
    )

    emotion_stress = pd.read_csv(
        COMBINED_DIR
        / "emotion_prevalence_stress_vs_no_stress.csv"
    )

    topic_summary = pd.read_csv(
        TOPICS_DIR
        / "final_topic_summary.csv"
    )

    topic_association = pd.read_csv(
        TOPICS_DIR
        / "topic_stress_association.csv"
    )

    topic_statistics = pd.read_csv(
        TOPICS_DIR
        / "topic_stress_statistical_summary.csv"
    )

    # ========================================================
    # BASIC INTEGRITY
    # ========================================================

    records = int(
        unified_summary["records"]
    )

    if records != 2666:
        raise ValueError(
            f"Expected 2666 records, found {records}."
        )

    if int(
        integrated_summary["records"]
    ) != records:
        raise ValueError(
            "Integrated analysis record count mismatch."
        )

    if int(
        stress_summary["records"]
    ) != records:
        raise ValueError(
            "Stress analysis record count mismatch."
        )

    # ========================================================
    # 1. CORE PROJECT METRICS
    # ========================================================

    vader = (
        unified_summary[
            "vader_distribution"
        ]
    )

    transformer = (
        unified_summary[
            "transformer_distribution"
        ]
    )

    stress_distribution = (
        unified_summary[
            "hybrid_stress_distribution"
        ]
    )

    core_metrics = pd.DataFrame(
        [
            {
                "metric": "Analysis-ready posts",
                "value": records,
                "unit": "posts",
            },
            {
                "metric": "VADER–Transformer agreement",
                "value": (
                    unified_summary[
                        "sentiment_methods_agreement_rate"
                    ]
                ),
                "unit": "proportion",
            },
            {
                "metric": "Predicted Stress posts",
                "value": (
                    stress_distribution[
                        "Stress"
                    ]
                ),
                "unit": "posts",
            },
            {
                "metric": "Predicted Stress rate",
                "value": (
                    stress_summary[
                        "stress_rate"
                    ]
                ),
                "unit": "proportion",
            },
            {
                "metric": "Average predicted emotions per post",
                "value": (
                    unified_summary[
                        "average_emotion_count"
                    ]
                ),
                "unit": "emotions/post",
            },
        ]
    )

    # ========================================================
    # 2. SENTIMENT DISTRIBUTION
    # ========================================================

    sentiment_rows = []

    for label, count in vader.items():
        sentiment_rows.append(
            {
                "method": "VADER",
                "label": label,
                "count": int(count),
                "rate": (
                    int(count)
                    / records
                ),
            }
        )

    for label, count in transformer.items():
        sentiment_rows.append(
            {
                "method": "Transformer",
                "label": label,
                "count": int(count),
                "rate": (
                    int(count)
                    / records
                ),
            }
        )

    sentiment_distribution = (
        pd.DataFrame(
            sentiment_rows
        )
    )

    # ========================================================
    # 3. NEGATIVE SENTIMENT ≠ STRESS
    # ========================================================

    negative_vs_stress = pd.DataFrame(
        [
            {
                "method": "VADER",
                "negative_posts": int(
                    integrated_summary[
                        "vader_negative_records"
                    ]
                ),
                "negative_and_stress": int(
                    integrated_summary[
                        "vader_negative_and_stress"
                    ]
                ),
                "stress_rate_within_negative": float(
                    integrated_summary[
                        "vader_negative_stress_rate"
                    ]
                ),
            },
            {
                "method": "Transformer",
                "negative_posts": int(
                    integrated_summary[
                        "transformer_negative_records"
                    ]
                ),
                "negative_and_stress": int(
                    integrated_summary[
                        "transformer_negative_and_stress"
                    ]
                ),
                "stress_rate_within_negative": float(
                    integrated_summary[
                        "transformer_negative_stress_rate"
                    ]
                ),
            },
        ]
    )

    # ========================================================
    # 4. EMOTIONS ASSOCIATED WITH PREDICTED STRESS
    # ========================================================

    emotion_final = (
        emotion_stress[
            [
                "emotion",
                "stress_count",
                "stress_rate",
                "no_stress_count",
                "no_stress_rate",
                "rate_difference",
                "rate_ratio",
            ]
        ]
        .sort_values(
            "rate_difference",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 5. TOPIC × STRESS
    # ========================================================

    topic_final = (
        topic_summary.merge(
            topic_association[
                [
                    "topic",
                    "relative_risk",
                    "odds_ratio",
                    "odds_ratio_ci95_lower",
                    "odds_ratio_ci95_upper",
                    "fisher_p_value",
                    "fisher_p_fdr_bh",
                    "significant_fdr_0_05",
                ]
            ],
            on="topic",
            how="left",
            validate="one_to_one",
        )
    )

    # ========================================================
    # 6. KEY FINDINGS
    # ========================================================

    significant_topics = (
        topic_final[
            topic_final[
                "significant_fdr_0_05"
            ]
            == True
        ]
        .copy()
    )

    top_emotions = (
        emotion_final
        .head(5)
    )

    chi_square = float(
        topic_statistics.iloc[0][
            "chi_square"
        ]
    )

    topic_p = float(
        topic_statistics.iloc[0][
            "p_value"
        ]
    )

    cramers_v = float(
        topic_statistics.iloc[0][
            "cramers_v"
        ]
    )

    findings = {
        "records": records,

        "sentiment": {
            "vader_transformer_agreement_rate": float(
                unified_summary[
                    "sentiment_methods_agreement_rate"
                ]
            ),
            "vader_distribution": vader,
            "transformer_distribution": transformer,
        },

        "stress": {
            "predicted_stress_posts": int(
                stress_distribution["Stress"]
            ),
            "predicted_no_stress_posts": int(
                stress_distribution["No stress"]
            ),
            "predicted_stress_rate": float(
                stress_summary[
                    "stress_rate"
                ]
            ),
            "average_stress_probability": float(
                stress_summary[
                    "average_stress_probability"
                ]
            ),
        },

        "negative_sentiment_is_not_stress": {
            "vader_negative_stress_rate": float(
                integrated_summary[
                    "vader_negative_stress_rate"
                ]
            ),
            "transformer_negative_stress_rate": float(
                integrated_summary[
                    "transformer_negative_stress_rate"
                ]
            ),
        },

        "top_emotions_enriched_in_predicted_stress": (
            top_emotions[
                [
                    "emotion",
                    "stress_rate",
                    "no_stress_rate",
                    "rate_difference",
                    "rate_ratio",
                ]
            ]
            .to_dict(
                orient="records"
            )
        ),

        "topic_stress_association": {
            "chi_square": chi_square,
            "p_value": topic_p,
            "cramers_v": cramers_v,
            "significant_topics": (
                significant_topics[
                    [
                        "topic",
                        "topic_name",
                        "predicted_stress_rate",
                        "relative_risk",
                        "odds_ratio",
                        "fisher_p_fdr_bh",
                    ]
                ]
                .to_dict(
                    orient="records"
                )
            ),
        },

        "interpretation_limits": [
            (
                "Predicted Stress is a model output, "
                "not a clinical diagnosis or an estimate "
                "of psychological-stress prevalence."
            ),
            (
                "Negative sentiment must not be treated "
                "as equivalent to Stress."
            ),
            (
                "Topic–Stress associations are "
                "associations and do not establish causality."
            ),
            (
                "Residual domain shift and false positives "
                "remain in the Stress classifier."
            ),
            (
                "GoEmotions predictions are transferred "
                "from a general-domain emotion dataset "
                "to developer discussions."
            ),
        ],
    }

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    core_metrics.to_csv(
        OUTPUT_DIR
        / "core_metrics.csv",
        index=False,
    )

    sentiment_distribution.to_csv(
        OUTPUT_DIR
        / "sentiment_distribution.csv",
        index=False,
    )

    negative_vs_stress.to_csv(
        OUTPUT_DIR
        / "negative_sentiment_vs_stress.csv",
        index=False,
    )

    emotion_final.to_csv(
        OUTPUT_DIR
        / "emotion_stress_association.csv",
        index=False,
    )

    topic_final.to_csv(
        OUTPUT_DIR
        / "topic_stress_final.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "final_findings.json"
    ).write_text(
        json.dumps(
            findings,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print(
        "\nCORE METRICS"
    )

    print(
        core_metrics.to_string(
            index=False
        )
    )

    print(
        "\nNEGATIVE SENTIMENT ≠ PREDICTED STRESS"
    )

    print(
        negative_vs_stress.to_string(
            index=False
        )
    )

    print(
        "\nTOP 10 EMOTIONS ENRICHED "
        "IN PREDICTED-STRESS POSTS"
    )

    print(
        emotion_final
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nTOPIC × PREDICTED-STRESS"
    )

    print(
        topic_final[
            [
                "topic",
                "topic_name",
                "posts",
                "predicted_stress_posts",
                "predicted_stress_rate",
                "relative_risk",
                "odds_ratio",
                "fisher_p_fdr_bh",
                "significant_fdr_0_05",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nOVERALL TOPIC ASSOCIATION"
    )

    print(
        f"Chi-square: {chi_square:.4f}"
    )

    print(
        f"p-value: {topic_p:.8f}"
    )

    print(
        f"Cramer's V: {cramers_v:.4f}"
    )

    print(
        f"\nFinal synthesis saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
