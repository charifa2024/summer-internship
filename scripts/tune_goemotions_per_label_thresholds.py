from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from datasets import load_from_disk

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "goemotions_tfidf_ovr_logreg.joblib"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "goemotions"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
)

GLOBAL_THRESHOLD = 0.61

# Avoid unstable threshold tuning
# for extremely rare validation labels.
MIN_VALIDATION_POSITIVES = 20


def encode_labels(
    label_lists,
    n_labels,
):
    y = np.zeros(
        (
            len(label_lists),
            n_labels,
        ),
        dtype=np.int8,
    )

    for row_index, labels in enumerate(
        label_lists
    ):
        for label_index in labels:
            y[
                row_index,
                label_index,
            ] = 1

    return y


def metrics(
    y_true,
    y_pred,
):
    return {
        "subset_accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "micro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "micro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "samples_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="samples",
                zero_division=0,
            )
        ),
        "hamming_loss": float(
            hamming_loss(
                y_true,
                y_pred,
            )
        ),
        "predicted_cardinality": float(
            y_pred.sum(
                axis=1
            ).mean()
        ),
        "no_predicted_label": int(
            (
                y_pred.sum(axis=1)
                == 0
            ).sum()
        ),
    }


def main():

    print("=" * 78)
    print(
        "GOEMOTIONS PER-EMOTION "
        "THRESHOLD CALIBRATION"
    )
    print("=" * 78)

    obj = joblib.load(
        MODEL_PATH
    )

    vectorizer = obj[
        "vectorizer"
    ]

    classifier = obj[
        "classifier"
    ]

    label_names = obj[
        "label_names"
    ]

    n_labels = len(
        label_names
    )

    dataset = load_from_disk(
        str(DATASET_PATH)
    )

    validation = dataset[
        "validation"
    ]

    test = dataset[
        "test"
    ]

    y_validation = encode_labels(
        validation["labels"],
        n_labels,
    )

    y_test = encode_labels(
        test["labels"],
        n_labels,
    )

    print(
        "\nVectorizing validation "
        "and test data..."
    )

    X_validation = (
        vectorizer.transform(
            validation["text"]
        )
    )

    X_test = (
        vectorizer.transform(
            test["text"]
        )
    )

    validation_probabilities = (
        classifier.predict_proba(
            X_validation
        )
    )

    test_probabilities = (
        classifier.predict_proba(
            X_test
        )
    )

    # ========================================================
    # GLOBAL 0.61 BASELINE
    # ========================================================

    global_predictions = (
        test_probabilities
        >= GLOBAL_THRESHOLD
    ).astype(int)

    global_metrics = metrics(
        y_test,
        global_predictions,
    )

    # ========================================================
    # PER-LABEL CALIBRATION
    # ========================================================

    candidate_thresholds = (
        np.arange(
            0.20,
            0.81,
            0.01,
        )
    )

    selected_thresholds = []
    calibration_rows = []

    for label_index, emotion in enumerate(
        label_names
    ):

        y_val_label = (
            y_validation[
                :,
                label_index,
            ]
        )

        probabilities = (
            validation_probabilities[
                :,
                label_index,
            ]
        )

        validation_support = int(
            y_val_label.sum()
        )

        # Very rare validation emotions:
        # retain global threshold.
        if (
            validation_support
            < MIN_VALIDATION_POSITIVES
        ):

            selected_threshold = (
                GLOBAL_THRESHOLD
            )

            prediction = (
                probabilities
                >= selected_threshold
            ).astype(int)

            validation_f1 = f1_score(
                y_val_label,
                prediction,
                zero_division=0,
            )

            reason = (
                "global_threshold_due_to_"
                "low_validation_support"
            )

        else:

            candidates = []

            for threshold in (
                candidate_thresholds
            ):

                prediction = (
                    probabilities
                    >= threshold
                ).astype(int)

                score = f1_score(
                    y_val_label,
                    prediction,
                    zero_division=0,
                )

                candidates.append(
                    (
                        float(threshold),
                        float(score),
                    )
                )

            # Select highest validation F1.
            # If tied, prefer threshold
            # closest to global 0.61.
            selected_threshold, (
                validation_f1
            ) = max(
                candidates,
                key=lambda x: (
                    x[1],
                    -abs(
                        x[0]
                        - GLOBAL_THRESHOLD
                    ),
                ),
            )

            reason = (
                "optimized_on_validation_f1"
            )

        selected_thresholds.append(
            selected_threshold
        )

        calibration_rows.append(
            {
                "emotion": emotion,
                "validation_support": (
                    validation_support
                ),
                "selected_threshold": (
                    selected_threshold
                ),
                "validation_f1": (
                    validation_f1
                ),
                "selection_reason": (
                    reason
                ),
            }
        )

    selected_thresholds = np.array(
        selected_thresholds
    )

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    calibrated_predictions = (
        test_probabilities
        >= selected_thresholds[
            np.newaxis,
            :
        ]
    ).astype(int)

    calibrated_metrics = metrics(
        y_test,
        calibrated_predictions,
    )

    # ========================================================
    # TEST PER-LABEL F1
    # ========================================================

    rows = []

    for label_index, emotion in enumerate(
        label_names
    ):

        global_f1 = f1_score(
            y_test[
                :,
                label_index,
            ],
            global_predictions[
                :,
                label_index,
            ],
            zero_division=0,
        )

        calibrated_f1 = f1_score(
            y_test[
                :,
                label_index,
            ],
            calibrated_predictions[
                :,
                label_index,
            ],
            zero_division=0,
        )

        rows.append(
            {
                "emotion": emotion,
                "global_threshold": (
                    GLOBAL_THRESHOLD
                ),
                "calibrated_threshold": (
                    selected_thresholds[
                        label_index
                    ]
                ),
                "global_test_f1": (
                    global_f1
                ),
                "calibrated_test_f1": (
                    calibrated_f1
                ),
                "f1_change": (
                    calibrated_f1
                    - global_f1
                ),
            }
        )

    comparison_per_label = (
        pd.DataFrame(rows)
    )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_df.to_csv(
        RESULTS_DIR
        / "goemotions_per_label_thresholds.csv",
        index=False,
    )

    comparison_per_label.to_csv(
        RESULTS_DIR
        / "goemotions_per_label_threshold_test_comparison.csv",
        index=False,
    )

    summary = {
        "global_threshold": (
            GLOBAL_THRESHOLD
        ),
        "minimum_validation_positives_for_calibration": (
            MIN_VALIDATION_POSITIVES
        ),
        "global_threshold_test_metrics": (
            global_metrics
        ),
        "per_label_threshold_test_metrics": (
            calibrated_metrics
        ),
    }

    (
        RESULTS_DIR
        / "goemotions_threshold_strategy_comparison.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\nSelected thresholds:"
    )

    print(
        calibration_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "GLOBAL 0.61 VS "
        "PER-EMOTION THRESHOLDS"
    )

    print(
        "=" * 78
    )

    comparison = pd.DataFrame(
        [
            {
                "strategy": (
                    "Global 0.61"
                ),
                **global_metrics,
            },
            {
                "strategy": (
                    "Per-emotion"
                ),
                **calibrated_metrics,
            },
        ]
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nPer-emotion F1 changes:"
    )

    print(
        comparison_per_label
        .sort_values(
            "f1_change",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
