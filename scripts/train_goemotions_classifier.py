from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from datasets import load_from_disk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    multilabel_confusion_matrix,
)
from sklearn.multiclass import OneVsRestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOEMOTIONS_DIR = (
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

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "goemotions_tfidf_ovr_logreg.joblib"
)


# ============================================================
# MULTI-LABEL ENCODING
# ============================================================

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


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
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
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
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
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("GOEMOTIONS MULTI-LABEL EMOTION CLASSIFIER")
    print("=" * 72)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_from_disk(
        str(GOEMOTIONS_DIR)
    )

    train = dataset["train"]
    validation = dataset["validation"]
    test = dataset["test"]

    label_names = (
        train.features[
            "labels"
        ]
        .feature
        .names
    )

    n_labels = len(label_names)

    print("\nDataset sizes:")
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))
    print("Emotion labels:", n_labels)

    # --------------------------------------------------------
    # Encode multi-label targets
    # --------------------------------------------------------

    y_train = encode_labels(
        train["labels"],
        n_labels,
    )

    y_validation = encode_labels(
        validation["labels"],
        n_labels,
    )

    y_test = encode_labels(
        test["labels"],
        n_labels,
    )

    # --------------------------------------------------------
    # Show class imbalance
    # --------------------------------------------------------

    train_counts = y_train.sum(
        axis=0
    )

    label_distribution = pd.DataFrame(
        {
            "emotion": label_names,
            "train_positive_count": (
                train_counts
            ),
            "train_positive_rate": (
                train_counts
                / len(train)
            ),
        }
    ).sort_values(
        "train_positive_count",
        ascending=False,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_distribution.to_csv(
        RESULTS_DIR
        / "goemotions_train_label_distribution.csv",
        index=False,
    )

    print(
        "\nTraining label distribution:"
    )

    print(
        label_distribution.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print("\nBuilding TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=12000,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(
        train["text"]
    )

    X_validation = vectorizer.transform(
        validation["text"]
    )

    X_test = vectorizer.transform(
        test["text"]
    )

    print(
        "TF-IDF train shape:",
        X_train.shape,
    )

    # --------------------------------------------------------
    # One-vs-Rest Logistic Regression
    # --------------------------------------------------------

    print(
        "\nTraining 28 "
        "One-vs-Rest Logistic Regression classifiers..."
    )

    classifier = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        n_jobs=1,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # VALIDATION — threshold selection
    # --------------------------------------------------------

    print(
        "\nSelecting threshold "
        "using validation split..."
    )

    validation_probabilities = (
        classifier.predict_proba(
            X_validation
        )
    )

    thresholds = np.arange(
        0.10,
        0.61,
        0.05,
    )

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            validation_probabilities
            >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        threshold_results.append(
            {
                "threshold": float(
                    round(
                        threshold,
                        2,
                    )
                ),
                **metrics,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    # Primary selection metric:
    # validation micro-F1.
    #
    # Macro-F1 acts as a secondary
    # criterion in case of ties.
    threshold_df = (
        threshold_df
        .sort_values(
            [
                "micro_f1",
                "macro_f1",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_threshold = float(
        threshold_df.iloc[0][
            "threshold"
        ]
    )

    threshold_df.to_csv(
        RESULTS_DIR
        / "goemotions_threshold_search.csv",
        index=False,
    )

    print(
        "\nValidation threshold results:"
    )

    print(
        threshold_df.to_string(
            index=False
        )
    )

    print(
        "\nSelected threshold:",
        best_threshold,
    )

    # --------------------------------------------------------
    # TEST — untouched until now
    # --------------------------------------------------------

    print(
        "\nEvaluating on untouched "
        "GoEmotions test split..."
    )

    test_probabilities = (
        classifier.predict_proba(
            X_test
        )
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    # --------------------------------------------------------
    # Per-emotion metrics
    # --------------------------------------------------------

    (
        per_label_precision,
        per_label_recall,
        per_label_f1,
        per_label_support,
    ) = precision_recall_fscore_support(
        y_test,
        test_predictions,
        average=None,
        zero_division=0,
    )

    per_label = pd.DataFrame(
        {
            "emotion": label_names,
            "precision": (
                per_label_precision
            ),
            "recall": (
                per_label_recall
            ),
            "f1": (
                per_label_f1
            ),
            "support": (
                per_label_support
            ),
        }
    )

    per_label.to_csv(
        RESULTS_DIR
        / "goemotions_test_per_label_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Per-label confusion counts
    # --------------------------------------------------------

    confusion = multilabel_confusion_matrix(
        y_test,
        test_predictions,
    )

    confusion_rows = []

    for i, emotion in enumerate(
        label_names
    ):

        tn, fp, fn, tp = (
            confusion[i]
            .ravel()
        )

        confusion_rows.append(
            {
                "emotion": emotion,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )

    pd.DataFrame(
        confusion_rows
    ).to_csv(
        RESULTS_DIR
        / "goemotions_test_confusion_counts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Additional diagnostics
    # --------------------------------------------------------

    true_cardinality = float(
        y_test.sum(axis=1).mean()
    )

    predicted_cardinality = float(
        test_predictions
        .sum(axis=1)
        .mean()
    )

    no_emotion_predictions = int(
        (
            test_predictions
            .sum(axis=1)
            == 0
        ).sum()
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    summary = {
        "model": (
            "TF-IDF + One-vs-Rest "
            "Logistic Regression"
        ),
        "training_dataset": (
            "GoEmotions simplified"
        ),
        "train_records": int(
            len(train)
        ),
        "validation_records": int(
            len(validation)
        ),
        "test_records": int(
            len(test)
        ),
        "emotion_labels": int(
            n_labels
        ),
        "selected_threshold": (
            best_threshold
        ),
        "threshold_selection": (
            "Maximum validation micro-F1"
        ),
        "class_weight": "balanced",
        **test_metrics,
        "true_label_cardinality": (
            true_cardinality
        ),
        "predicted_label_cardinality": (
            predicted_cardinality
        ),
        "test_records_with_no_predicted_label": (
            no_emotion_predictions
        ),
        "clinical_ground_truth": False,
    }

    (
        RESULTS_DIR
        / "goemotions_test_metrics.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "label_names": label_names,
            "threshold": (
                best_threshold
            ),
            "model_name": (
                "TF-IDF + One-vs-Rest "
                "Logistic Regression"
            ),
            "training_dataset": (
                "GoEmotions simplified"
            ),
            "random_state": 42,
        },
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("GOEMOTIONS TEST RESULTS")
    print("=" * 72)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nPer-emotion test metrics:"
    )

    print(
        per_label
        .sort_values(
            "f1",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nModel saved to:\n"
        f"{MODEL_FILE}"
    )


if __name__ == "__main__":
    main()
