from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_from_disk

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
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# PATHS
# ============================================================

DREADDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dreaddit"
)

SYNTHETIC_FILE = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "synthetic_stress_dataset_v1_completed.csv"
)

BASELINE_METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "emotions"
    / "dreaddit_stress_metrics.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "hybrid_dreaddit_synthetic_stress_classifier.joblib"
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
# MODEL
# ============================================================

def make_model():

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
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("HYBRID DREADDIT + SYNTHETIC STRESS MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Dreaddit
    # --------------------------------------------------------

    dreaddit = load_from_disk(
        str(DREADDIT_DIR)
    )

    dreaddit_train = dreaddit["train"]
    dreaddit_test = dreaddit["test"]

    real_train = pd.DataFrame(
        {
            "text": dreaddit_train["text"],
            "label": dreaddit_train["label"],
            "training_source": "Dreaddit",
        }
    )

    test_df = pd.DataFrame(
        {
            "text": dreaddit_test["text"],
            "label": dreaddit_test["label"],
        }
    )

    # --------------------------------------------------------
    # Load synthetic augmentation
    # --------------------------------------------------------

    synthetic = pd.read_csv(
        SYNTHETIC_FILE
    )

    required_columns = {
        "text",
        "stress_label",
    }

    missing_columns = (
        required_columns
        - set(synthetic.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing synthetic columns: "
            f"{missing_columns}"
        )

    synthetic_train = pd.DataFrame(
        {
            "text": synthetic["text"],
            "label": synthetic[
                "stress_label"
            ].map(
                {
                    "No stress": 0,
                    "Stress": 1,
                }
            ),
            "training_source": (
                "Synthetic developer augmentation"
            ),
        }
    )

    if synthetic_train[
        "label"
    ].isna().any():
        raise ValueError(
            "Unexpected synthetic stress label."
        )

    synthetic_train[
        "label"
    ] = synthetic_train[
        "label"
    ].astype(int)

    # --------------------------------------------------------
    # Combine REAL + SYNTHETIC training data
    # --------------------------------------------------------

    hybrid_train = pd.concat(
        [
            real_train,
            synthetic_train,
        ],
        ignore_index=True,
    )

    hybrid_train = (
        hybrid_train
        .sample(
            frac=1,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    print("\nTRAINING COMPOSITION")

    print(
        "\nDreaddit real training:",
        len(real_train),
    )

    print(
        "Synthetic augmentation:",
        len(synthetic_train),
    )

    print(
        "Hybrid total:",
        len(hybrid_train),
    )

    print(
        "\nHybrid label distribution:"
    )

    print(
        hybrid_train[
            "label"
        ].value_counts()
        .sort_index()
    )

    print(
        "\nTraining source distribution:"
    )

    print(
        hybrid_train[
            "training_source"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = make_model()

    print(
        "\nTraining hybrid Logistic Regression..."
    )

    model.fit(
        hybrid_train["text"],
        hybrid_train["label"],
    )

    # --------------------------------------------------------
    # Evaluate ONLY on untouched Dreaddit test set
    # --------------------------------------------------------

    print(
        "\nEvaluating on untouched "
        "Dreaddit official test set..."
    )

    predictions = model.predict(
        test_df["text"]
    )

    probabilities = (
        model.predict_proba(
            test_df["text"]
        )[:, 1]
    )

    accuracy = accuracy_score(
        test_df["label"],
        predictions,
    )

    macro_f1 = f1_score(
        test_df["label"],
        predictions,
        average="macro",
        zero_division=0,
    )

    stress_precision = precision_score(
        test_df["label"],
        predictions,
        pos_label=1,
        zero_division=0,
    )

    stress_recall = recall_score(
        test_df["label"],
        predictions,
        pos_label=1,
        zero_division=0,
    )

    stress_f1 = f1_score(
        test_df["label"],
        predictions,
        pos_label=1,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Output folders
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
    # Test prediction file
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "text": test_df["text"],
            "true_label": test_df["label"],
            "predicted_label": predictions,
            "stress_probability": probabilities,
        }
    )

    prediction_df.to_csv(
        RESULTS_DIR
        / "hybrid_stress_dreaddit_test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        test_df["label"],
        predictions,
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
        "Hybrid Dreaddit + Synthetic "
        "Stress Classifier"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "hybrid_stress_dreaddit_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        {
            "pipeline": model,
            "selected_model": (
                "Logistic Regression"
            ),
            "real_training_records": (
                len(real_train)
            ),
            "synthetic_training_records": (
                len(synthetic_train)
            ),
            "total_training_records": (
                len(hybrid_train)
            ),
            "training_sources": [
                "Dreaddit",
                (
                    "LLM-generated synthetic "
                    "developer augmentation"
                ),
            ],
            "label_mapping": {
                0: "No Stress",
                1: "Stress",
            },
            "synthetic_role": (
                "Experimental augmentation only"
            ),
            "clinical_ground_truth": False,
            "random_state": 42,
        },
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    summary = {
        "model": (
            "Hybrid Dreaddit + Synthetic"
        ),
        "classifier": (
            "Logistic Regression"
        ),
        "dreaddit_training_records": int(
            len(real_train)
        ),
        "synthetic_augmentation_records": int(
            len(synthetic_train)
        ),
        "total_training_records": int(
            len(hybrid_train)
        ),
        "dreaddit_test_records": int(
            len(test_df)
        ),
        "test_accuracy": float(
            accuracy
        ),
        "test_macro_f1": float(
            macro_f1
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
        "synthetic_role": (
            "Experimental augmentation only"
        ),
        "clinical_ground_truth": False,
    }

    (
        RESULTS_DIR
        / "hybrid_stress_metrics.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Compare against Dreaddit-only baseline
    # --------------------------------------------------------

    if BASELINE_METRICS_FILE.exists():

        baseline = json.loads(
            BASELINE_METRICS_FILE.read_text(
                encoding="utf-8"
            )
        )

        comparison = pd.DataFrame(
            [
                {
                    "model": (
                        "Dreaddit only"
                    ),
                    "accuracy": baseline[
                        "test_accuracy"
                    ],
                    "macro_f1": baseline[
                        "test_macro_f1"
                    ],
                    "stress_precision": baseline[
                        "stress_precision"
                    ],
                    "stress_recall": baseline[
                        "stress_recall"
                    ],
                    "stress_f1": baseline[
                        "stress_f1"
                    ],
                },
                {
                    "model": (
                        "Dreaddit + Synthetic"
                    ),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "stress_precision": (
                        stress_precision
                    ),
                    "stress_recall": (
                        stress_recall
                    ),
                    "stress_f1": stress_f1,
                },
            ]
        )

        comparison.to_csv(
            RESULTS_DIR
            / "dreaddit_vs_hybrid_comparison.csv",
            index=False,
        )

        print(
            "\nDREADDIT-ONLY VS HYBRID"
        )

        print(
            comparison.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID TEST RESULTS")
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
