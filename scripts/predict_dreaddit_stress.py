from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analysis_ready_posts.jsonl"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "dreaddit_stress_classifier.joblib"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
    / "dreaddit_stress_predictions_analysis_ready.csv"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("DREADDIT MODEL → DEVELOPER POSTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load developer dataset
    # --------------------------------------------------------

    df = pd.read_json(
        INPUT_FILE,
        lines=True,
    )

    print(
        f"\nAnalysis-ready developer posts: {len(df)}"
    )

    text_column = "text_clean_basic"

    if text_column not in df.columns:
        raise ValueError(
            f"Missing required column: {text_column}"
        )

    missing_text = (
        df[text_column]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"Empty texts: {missing_text}"
    )

    if missing_text > 0:
        raise ValueError(
            "Some records contain empty text."
        )

    # --------------------------------------------------------
    # Load trained Dreaddit model
    # --------------------------------------------------------

    model_bundle = joblib.load(
        MODEL_FILE
    )

    model = model_bundle[
        "pipeline"
    ]

    print(
        "Loaded model:",
        model_bundle[
            "selected_model"
        ],
    )

    # --------------------------------------------------------
    # Predict stress
    # --------------------------------------------------------

    texts = df[
        text_column
    ].astype(str)

    predictions = model.predict(
        texts
    )

    probabilities = model.predict_proba(
        texts
    )

    stress_probabilities = (
        probabilities[:, 1]
    )

    no_stress_probabilities = (
        probabilities[:, 0]
    )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "record_id": df[
                "record_id"
            ],
            "platform": df[
                "platform"
            ],
            "source": df[
                "source"
            ],
            "text_clean_basic": df[
                text_column
            ],
            "dreaddit_stress_label": predictions,
            "dreaddit_stress_prediction": [
                (
                    "Stress"
                    if label == 1
                    else "No Stress"
                )
                for label in predictions
            ],
            "dreaddit_stress_probability": (
                stress_probabilities
            ),
            "dreaddit_no_stress_probability": (
                no_stress_probabilities
            ),
        }
    )

    # Preserve useful analytical metadata
    optional_columns = [
        "created_at_utc",
        "ai_tools",
        "research_themes",
        "sentiment_label",
        "word_count",
    ]

    for column in optional_columns:
        if column in df.columns:
            result[column] = df[
                column
            ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    counts = (
        result[
            "dreaddit_stress_prediction"
        ]
        .value_counts()
    )

    print("\nPREDICTION DISTRIBUTION")
    print("-" * 40)
    print(counts)

    stress_count = int(
        (
            result[
                "dreaddit_stress_prediction"
            ]
            == "Stress"
        ).sum()
    )

    stress_rate = (
        stress_count
        / len(result)
        * 100
    )

    print(
        f"\nStress predictions: "
        f"{stress_count}/{len(result)} "
        f"({stress_rate:.2f}%)"
    )

    print(
        "\nAverage stress probability:",
        round(
            result[
                "dreaddit_stress_probability"
            ].mean(),
            4,
        ),
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
