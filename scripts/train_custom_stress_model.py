from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "llm_assisted_stress_labels.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "custom_stress_classifier.joblib"
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


def make_pipeline(classifier):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    max_features=12000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    labelled = pd.read_csv(
        ANNOTATION_FILE
    )

    x_train, x_test, y_train, y_test = (
        train_test_split(
            labelled["text_clean_basic"],
            labelled["human_stress_label"],
            test_size=0.25,
            stratify=labelled[
                "human_stress_label"
            ],
            random_state=42,
        )
    )

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
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
                cv=3,
            )
        ),
        "Complement Naive Bayes": make_pipeline(
            ComplementNB(alpha=0.5)
        ),
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    rows = []
    fitted = {}

    for name, model in models.items():
        cv_scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "macro_f1": "f1_macro",
            },
            error_score="raise",
        )

        model.fit(
            x_train,
            y_train,
        )

        predictions = model.predict(
            x_test
        )

        rows.append(
            {
                "model": name,
                "cv_accuracy": (
                    cv_scores[
                        "test_accuracy"
                    ].mean()
                ),
                "cv_macro_f1": (
                    cv_scores[
                        "test_macro_f1"
                    ].mean()
                ),
                "test_accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "test_macro_f1": f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "stress_precision": precision_score(
                    y_test,
                    predictions,
                    pos_label="Stress",
                    zero_division=0,
                ),
                "stress_recall": recall_score(
                    y_test,
                    predictions,
                    pos_label="Stress",
                    zero_division=0,
                ),
                "stress_f1": f1_score(
                    y_test,
                    predictions,
                    pos_label="Stress",
                    zero_division=0,
                ),
            }
        )

        fitted[name] = model

    comparison = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "cv_macro_f1",
                "stress_recall",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_name = comparison.iloc[0][
        "model"
    ]

    best_model = fitted[
        best_name
    ]

    test_predictions = best_model.predict(
        x_test
    )

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

    comparison.to_csv(
        RESULTS_DIR
        / "custom_stress_model_comparison.csv",
        index=False,
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[
            "No stress",
            "Stress",
        ],
    )

    ConfusionMatrixDisplay(
        matrix,
        display_labels=[
            "No stress",
            "Stress",
        ],
    ).plot(
        values_format="d"
    )

    plt.title(
        f"Custom Stress Classifier\n{best_name}"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR
        / "custom_stress_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    comparison.set_index(
        "model"
    )[
        [
            "cv_macro_f1",
            "test_macro_f1",
            "stress_recall",
        ]
    ].plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(
        "Custom Stress Model Comparison"
    )
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR
        / "custom_stress_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    best_model.fit(
        labelled["text_clean_basic"],
        labelled["human_stress_label"],
    )

    joblib.dump(
        {
            "pipeline": best_model,
            "selected_model": best_name,
            "text_column": "text_clean_basic",
            "training_records": len(labelled),
            "annotation_source": (
                "LLM-assisted manual review"
            ),
            "human_ground_truth": False,
        },
        MODEL_FILE,
    )

    summary = {
        "selected_model": best_name,
        "labelled_records": int(
            len(labelled)
        ),
        "test_records": int(
            len(y_test)
        ),
        "test_accuracy": float(
            accuracy_score(
                y_test,
                test_predictions,
            )
        ),
        "test_macro_f1": float(
            f1_score(
                y_test,
                test_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "stress_precision": float(
            precision_score(
                y_test,
                test_predictions,
                pos_label="Stress",
                zero_division=0,
            )
        ),
        "stress_recall": float(
            recall_score(
                y_test,
                test_predictions,
                pos_label="Stress",
                zero_division=0,
            )
        ),
        "stress_f1": float(
            f1_score(
                y_test,
                test_predictions,
                pos_label="Stress",
                zero_division=0,
            )
        ),
    }

    (
        RESULTS_DIR
        / "custom_stress_metrics.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
