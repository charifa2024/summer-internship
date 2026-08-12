from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_from_disk

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DREADDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dreaddit"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "dreaddit_stress_classifier.joblib"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "emotions"
)


# ============================================================
# TF-IDF PIPELINE
# ============================================================

def make_pipeline(classifier):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=12000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main() -> None:

    print("=" * 70)
    print("DREADDIT STRESS CLASSIFIER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load official Dreaddit splits
    # --------------------------------------------------------

    dataset = load_from_disk(
        str(DREADDIT_DIR)
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    x_train = train_dataset["text"]
    y_train = train_dataset["label"]

    x_test = test_dataset["text"]
    y_test = test_dataset["label"]

    print(f"\nTraining records: {len(x_train)}")
    print(f"Official test records: {len(x_test)}")

    print(
        "Training Stress:",
        sum(label == 1 for label in y_train),
    )
    print(
        "Training No Stress:",
        sum(label == 0 for label in y_train),
    )

    # --------------------------------------------------------
    # Candidate models
    # --------------------------------------------------------

    models = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
        ),

        "Calibrated Linear SVM": make_pipeline(
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
                cv=3,
                method="sigmoid",
                ensemble=False,
            )
        ),

        "Complement Naive Bayes": make_pipeline(
            ComplementNB(
                alpha=0.5
            )
        ),
    }

    # --------------------------------------------------------
    # Cross-validation ONLY on training set
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    comparison_rows = []

    print("\nRunning 5-fold cross-validation...\n")

    for name, model in models.items():

        print(f"Evaluating: {name}")

        scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "macro_f1": "f1_macro",
                "stress_precision": "precision",
                "stress_recall": "recall",
                "stress_f1": "f1",
            },
            n_jobs=1,
            error_score="raise",
        )

        comparison_rows.append(
            {
                "model": name,
                "cv_accuracy_mean": (
                    scores["test_accuracy"].mean()
                ),
                "cv_macro_f1_mean": (
                    scores["test_macro_f1"].mean()
                ),
                "cv_macro_f1_std": (
                    scores["test_macro_f1"].std()
                ),
                "cv_stress_precision_mean": (
                    scores[
                        "test_stress_precision"
                    ].mean()
                ),
                "cv_stress_recall_mean": (
                    scores[
                        "test_stress_recall"
                    ].mean()
                ),
                "cv_stress_f1_mean": (
                    scores[
                        "test_stress_f1"
                    ].mean()
                ),
            }
        )

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values(
            "cv_macro_f1_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    # --------------------------------------------------------
    # Select best model using TRAINING CV only
    # --------------------------------------------------------

    best_name = comparison.iloc[0]["model"]

    print(
        f"\nSelected model: {best_name}"
    )

    best_model = models[best_name]

    # --------------------------------------------------------
    # Train selected model on ALL official training data
    # --------------------------------------------------------

    print("\nTraining selected model on full training set...")

    best_model.fit(
        x_train,
        y_train,
    )

    # --------------------------------------------------------
    # FINAL evaluation on untouched official test split
    # --------------------------------------------------------

    print("\nEvaluating on official Dreaddit test set...")

    test_predictions = best_model.predict(
        x_test
    )

    test_probabilities = (
        best_model.predict_proba(
            x_test
        )[:, 1]
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    stress_precision = precision_score(
        y_test,
        test_predictions,
        pos_label=1,
        zero_division=0,
    )

    stress_recall = recall_score(
        y_test,
        test_predictions,
        pos_label=1,
        zero_division=0,
    )

    stress_f1 = f1_score(
        y_test,
        test_predictions,
        pos_label=1,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save CV model comparison
    # --------------------------------------------------------

    comparison.to_csv(
        RESULTS_DIR
        / "dreaddit_stress_model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save official test predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "text": x_test,
            "true_label": y_test,
            "predicted_label": test_predictions,
            "stress_probability": test_probabilities,
        }
    )

    prediction_df[
        "true_label_name"
    ] = prediction_df[
        "true_label"
    ].map(
        {
            0: "No Stress",
            1: "Stress",
        }
    )

    prediction_df[
        "predicted_label_name"
    ] = prediction_df[
        "predicted_label"
    ].map(
        {
            0: "No Stress",
            1: "Stress",
        }
    )

    prediction_df.to_csv(
        RESULTS_DIR
        / "dreaddit_stress_test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    ConfusionMatrixDisplay(
        matrix,
        display_labels=[
            "No Stress",
            "Stress",
        ],
    ).plot(
        values_format="d"
    )

    plt.title(
        f"Dreaddit Stress Classifier\n{best_name}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "dreaddit_stress_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Cross-validation model comparison figure
    # --------------------------------------------------------

    comparison.set_index(
        "model"
    )[
        [
            "cv_macro_f1_mean",
            "cv_stress_precision_mean",
            "cv_stress_recall_mean",
        ]
    ].plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.ylim(0, 1)
    plt.ylabel("Cross-validation score")

    plt.title(
        "Dreaddit Stress Model Comparison"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "dreaddit_stress_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    joblib.dump(
        {
            "pipeline": best_model,
            "selected_model": best_name,
            "text_column": "text",
            "training_records": len(x_train),
            "test_records": len(x_test),
            "label_mapping": {
                0: "No Stress",
                1: "Stress",
            },
            "training_dataset": "Dreaddit",
            "label_source": (
                "Dreaddit human annotations"
            ),
            "clinical_ground_truth": False,
            "random_state": 42,
        },
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Save summary metrics
    # --------------------------------------------------------

    summary = {
        "selected_model": best_name,
        "training_records": int(
            len(x_train)
        ),
        "test_records": int(
            len(x_test)
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "test_macro_f1": float(
            test_macro_f1
        ),
        "stress_precision": float(
            stress_precision
        ),
        "stress_recall": float(
            stress_recall
        ),
        "stress_f1": float(
            stress_f1
        ),
        "clinical_ground_truth": False,
    }

    (
        RESULTS_DIR
        / "dreaddit_stress_metrics.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"\nModel saved to:\n{MODEL_FILE}"
    )


if __name__ == "__main__":
    main()