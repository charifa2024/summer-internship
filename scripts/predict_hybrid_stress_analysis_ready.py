from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "hybrid_dreaddit_synthetic_stress_classifier.joblib"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analysis_ready_posts.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
    / "hybrid_stress_predictions_analysis_ready.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
    / "hybrid_stress_analysis_ready_summary.json"
)


def main():

    print("=" * 76)
    print("FINAL HYBRID STRESS — ANALYSIS-READY DEVELOPER POSTS")
    print("=" * 76)

    # --------------------------------------------------------
    # Load final hybrid model
    # --------------------------------------------------------

    model_object = joblib.load(
        MODEL_FILE
    )

    if "pipeline" not in model_object:
        raise ValueError(
            "Saved hybrid model does not contain 'pipeline'."
        )

    model = model_object["pipeline"]

    print(
        "\nSelected model:",
        model_object.get(
            "selected_model",
            "unknown",
        ),
    )

    print(
        "Real training records:",
        model_object.get(
            "real_training_records",
            "unknown",
        ),
    )

    print(
        "Synthetic augmentation records:",
        model_object.get(
            "synthetic_training_records",
            "unknown",
        ),
    )

    print(
        "Total training records:",
        model_object.get(
            "total_training_records",
            "unknown",
        ),
    )

    # --------------------------------------------------------
    # Load analysis-ready developer posts
    # --------------------------------------------------------

    df = pd.read_json(
        INPUT_FILE,
        lines=True,
    )

    print(
        "\nAnalysis-ready records:",
        len(df),
    )

    if len(df) != 2666:
        print(
            "WARNING: expected 2,666 records, "
            f"but found {len(df)}."
        )

    required_columns = [
        "record_id",
        "text_clean_basic",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    texts = (
        df["text_clean_basic"]
        .fillna("")
        .astype(str)
    )

    empty_count = int(
        texts.str.strip()
        .eq("")
        .sum()
    )

    print(
        "Empty texts:",
        empty_count,
    )

    if empty_count:
        raise ValueError(
            "Empty text found in analysis-ready data."
        )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print(
        "\nPredicting final Hybrid Stress labels..."
    )

    predictions = model.predict(
        texts
    )

    probabilities = model.predict_proba(
        texts
    )

    no_stress_probability = (
        probabilities[:, 0]
    )

    stress_probability = (
        probabilities[:, 1]
    )

    # --------------------------------------------------------
    # Build minimal final output
    # --------------------------------------------------------

    keep_columns = [
        col
        for col in [
            "record_id",
            "platform",
            "source",
            "community",
            "created_at_utc",
            "text_clean_basic",
            "ai_tools",
            "research_themes",
            "word_count",
        ]
        if col in df.columns
    ]

    output = df[
        keep_columns
    ].copy()

    output[
        "hybrid_stress_prediction"
    ] = predictions

    output[
        "hybrid_stress_label"
    ] = [
        "Stress"
        if value == 1
        else "No stress"
        for value in predictions
    ]

    output[
        "hybrid_stress_probability"
    ] = stress_probability

    output[
        "hybrid_no_stress_probability"
    ] = no_stress_probability

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    stress_count = int(
        predictions.sum()
    )

    no_stress_count = int(
        len(predictions)
        - stress_count
    )

    summary = {
        "records": int(
            len(output)
        ),
        "model": (
            "Hybrid Dreaddit + Synthetic "
            "TF-IDF Logistic Regression"
        ),
        "classification_threshold": 0.50,
        "stress_count": stress_count,
        "no_stress_count": no_stress_count,
        "stress_rate": float(
            stress_count
            / len(output)
        ),
        "no_stress_rate": float(
            no_stress_count
            / len(output)
        ),
        "average_stress_probability": float(
            stress_probability.mean()
        ),
        "interpretation_warning": (
            "Predicted Stress is a model classification "
            "and must not be interpreted as clinical "
            "or psychological diagnosis."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Basic integrity checks
    # --------------------------------------------------------

    if output["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record_id values found."
        )

    if output["record_id"].isna().any():
        raise ValueError(
            "Missing record_id values found."
        )

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("FINAL HYBRID STRESS SUMMARY")
    print("=" * 76)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nLabel distribution:"
    )

    print(
        output[
            "hybrid_stress_label"
        ]
        .value_counts()
    )

    print(
        "\nTop 10 highest Stress probabilities:"
    )

    print(
        output[
            [
                "record_id",
                "platform",
                "hybrid_stress_label",
                "hybrid_stress_probability",
            ]
        ]
        .sort_values(
            "hybrid_stress_probability",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        f"\nPredictions saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
