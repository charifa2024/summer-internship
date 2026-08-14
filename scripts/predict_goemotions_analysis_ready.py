from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "goemotions_tfidf_ovr_logreg.joblib"
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
    / "goemotions_predictions_analysis_ready.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
    / "goemotions_analysis_ready_summary.json"
)

PREVALENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
    / "goemotions_analysis_ready_emotion_prevalence.csv"
)


def main():

    print("=" * 76)
    print("GOEMOTIONS — ANALYSIS-READY DEVELOPER POSTS")
    print("=" * 76)

    # --------------------------------------------------------
    # Load final calibrated model
    # --------------------------------------------------------

    obj = joblib.load(
        MODEL_FILE
    )

    vectorizer = obj[
        "vectorizer"
    ]

    classifier = obj[
        "classifier"
    ]

    label_names = list(
        obj["label_names"]
    )

    thresholds = np.asarray(
        obj["per_label_thresholds"],
        dtype=float,
    )

    if len(label_names) != len(thresholds):
        raise ValueError(
            "Number of emotion labels and thresholds do not match."
        )

    print(
        "\nPrediction strategy:",
        obj.get(
            "prediction_strategy",
            "unknown",
        ),
    )

    print(
        "Emotion labels:",
        len(label_names),
    )

    # --------------------------------------------------------
    # Load project data
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

    text_column = "text_clean_basic"

    if text_column not in df.columns:
        raise ValueError(
            f"Missing text column: {text_column}"
        )

    texts = (
        df[text_column]
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

    if empty_count > 0:
        raise ValueError(
            "Empty texts found in analysis-ready dataset."
        )

    # --------------------------------------------------------
    # Transform and predict
    # --------------------------------------------------------

    print(
        "\nVectorizing developer posts..."
    )

    X = vectorizer.transform(
        texts
    )

    print(
        "TF-IDF shape:",
        X.shape,
    )

    print(
        "\nPredicting emotion probabilities..."
    )

    probabilities = (
        classifier.predict_proba(
            X
        )
    )

    predictions = (
        probabilities
        >= thresholds[np.newaxis, :]
    ).astype(np.int8)

    # --------------------------------------------------------
    # Multi-label output
    # --------------------------------------------------------

    emotion_count = predictions.sum(
        axis=1
    )

    predicted_emotions = []

    for row in predictions:

        labels = [
            label_names[i]
            for i, value in enumerate(row)
            if value == 1
        ]

        predicted_emotions.append(
            "|".join(labels)
        )

    # Highest probability emotion is diagnostic.
    # It is NOT automatically treated as predicted
    # when it is below its calibrated threshold.
    top_indices = probabilities.argmax(
        axis=1
    )

    top_emotions = [
        label_names[i]
        for i in top_indices
    ]

    top_probabilities = (
        probabilities[
            np.arange(len(df)),
            top_indices,
        ]
    )

    top_thresholds = (
        thresholds[
            top_indices
        ]
    )

    top_passes_threshold = (
        top_probabilities
        >= top_thresholds
    )

    # --------------------------------------------------------
    # Neutral conflict diagnostic
    # --------------------------------------------------------

    neutral_index = label_names.index(
        "neutral"
    )

    neutral_predicted = (
        predictions[
            :,
            neutral_index
        ]
        == 1
    )

    non_neutral_count = (
        emotion_count
        - predictions[
            :,
            neutral_index
        ]
    )

    neutral_with_other_emotions = (
        neutral_predicted
        & (non_neutral_count > 0)
    )

    # --------------------------------------------------------
    # Build result table
    # --------------------------------------------------------

    output = df.copy()

    output[
        "predicted_emotions"
    ] = predicted_emotions

    output[
        "emotion_count"
    ] = emotion_count

    output[
        "top_emotion"
    ] = top_emotions

    output[
        "top_emotion_probability"
    ] = top_probabilities

    output[
        "top_emotion_threshold"
    ] = top_thresholds

    output[
        "top_emotion_passes_threshold"
    ] = top_passes_threshold

    output[
        "neutral_with_other_emotions"
    ] = neutral_with_other_emotions

    # Add binary prediction + probability
    # for every GoEmotions label.
    for i, emotion in enumerate(
        label_names
    ):

        output[
            f"emotion_{emotion}_pred"
        ] = predictions[
            :,
            i
        ]

        output[
            f"emotion_{emotion}_prob"
        ] = probabilities[
            :,
            i
        ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Emotion prevalence
    # --------------------------------------------------------

    prevalence_rows = []

    for i, emotion in enumerate(
        label_names
    ):

        count = int(
            predictions[
                :,
                i
            ].sum()
        )

        prevalence_rows.append(
            {
                "emotion": emotion,
                "predicted_count": count,
                "predicted_rate": (
                    count / len(df)
                ),
                "threshold": float(
                    thresholds[i]
                ),
                "average_probability": float(
                    probabilities[
                        :,
                        i
                    ].mean()
                ),
            }
        )

    prevalence = (
        pd.DataFrame(
            prevalence_rows
        )
        .sort_values(
            "predicted_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    prevalence.to_csv(
        PREVALENCE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    zero_emotion_count = int(
        (emotion_count == 0)
        .sum()
    )

    multi_emotion_count = int(
        (emotion_count > 1)
        .sum()
    )

    summary = {
        "records": int(
            len(df)
        ),
        "model": (
            "TF-IDF + One-vs-Rest "
            "Logistic Regression"
        ),
        "training_dataset": (
            "GoEmotions simplified"
        ),
        "prediction_strategy": (
            "Per-emotion validation-calibrated thresholds"
        ),
        "emotion_labels": int(
            len(label_names)
        ),
        "average_predicted_emotions_per_post": float(
            emotion_count.mean()
        ),
        "median_predicted_emotions_per_post": float(
            np.median(
                emotion_count
            )
        ),
        "records_with_zero_emotions": (
            zero_emotion_count
        ),
        "zero_emotion_rate": float(
            zero_emotion_count
            / len(df)
        ),
        "records_with_multiple_emotions": (
            multi_emotion_count
        ),
        "multiple_emotion_rate": float(
            multi_emotion_count
            / len(df)
        ),
        "neutral_with_other_emotions": int(
            neutral_with_other_emotions.sum()
        ),
        "neutral_conflict_rate": float(
            neutral_with_other_emotions.mean()
        ),
        "interpretation_warning": (
            "Emotion predictions are model outputs "
            "and must not be interpreted as "
            "psychological diagnoses."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("DEVELOPER EMOTION PREDICTION SUMMARY")
    print("=" * 76)

    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    print(
        "\nPredicted emotion prevalence:"
    )

    print(
        prevalence.to_string(
            index=False
        )
    )

    print(
        "\nTop-emotion distribution:"
    )

    print(
        pd.Series(
            top_emotions
        )
        .value_counts()
        .to_string()
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
