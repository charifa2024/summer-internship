from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# PATHS
# ============================================================

BASELINE_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "dreaddit_stress_classifier.joblib"
)

HYBRID_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "hybrid_dreaddit_synthetic_stress_classifier.joblib"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "llm_assisted_stress_labels_v2_completed.csv"
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
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
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
        "stress_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "stress_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "stress_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity": float(
            specificity
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_stress": int(
            (y_pred == 1).sum()
        ),
        "predicted_no_stress": int(
            (y_pred == 0).sum()
        ),
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    output_file,
):

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No Stress",
            "Stress",
        ],
    )

    display.plot(
        values_format="d"
    )

    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("REAL DEVELOPER DOMAIN — STRESS MODEL EVALUATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Load validation/reference data
    # --------------------------------------------------------

    df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        "\nOriginal reference records:",
        len(df),
    )

    print(
        "\nOriginal label distribution:"
    )

    print(
        df[
            "reference_stress_label"
        ].value_counts(
            dropna=False
        )
    )

    # Keep only evaluable labels.
    evaluable = df[
        df[
            "reference_stress_label"
        ].isin(
            [
                "No stress",
                "Stress",
            ]
        )
    ].copy()

    excluded_unclear = (
        len(df)
        - len(evaluable)
    )

    evaluable[
        "true_label"
    ] = evaluable[
        "reference_stress_label"
    ].map(
        {
            "No stress": 0,
            "Stress": 1,
        }
    ).astype(int)

    text_column = "text_clean_basic"

    print(
        "\nEvaluable records:",
        len(evaluable),
    )

    print(
        "Excluded Unclear:",
        excluded_unclear,
    )

    print(
        "\nEvaluable reference distribution:"
    )

    print(
        evaluable[
            "reference_stress_label"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    baseline_object = joblib.load(
        BASELINE_MODEL_FILE
    )

    hybrid_object = joblib.load(
        HYBRID_MODEL_FILE
    )

    baseline_model = baseline_object[
        "pipeline"
    ]

    hybrid_model = hybrid_object[
        "pipeline"
    ]

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    texts = evaluable[
        text_column
    ].fillna("").astype(str)

    y_true = evaluable[
        "true_label"
    ]

    baseline_pred = baseline_model.predict(
        texts
    )

    hybrid_pred = hybrid_model.predict(
        texts
    )

    # Probabilities where supported.
    baseline_prob = (
        baseline_model.predict_proba(
            texts
        )[:, 1]
    )

    hybrid_prob = (
        hybrid_model.predict_proba(
            texts
        )[:, 1]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    baseline_metrics = calculate_metrics(
        y_true,
        baseline_pred,
    )

    hybrid_metrics = calculate_metrics(
        y_true,
        hybrid_pred,
    )

    baseline_metrics[
        "model"
    ] = "Dreaddit only"

    hybrid_metrics[
        "model"
    ] = "Dreaddit + Synthetic"

    # --------------------------------------------------------
    # Prediction-rate diagnostics
    # --------------------------------------------------------

    baseline_metrics[
        "predicted_stress_rate"
    ] = float(
        (baseline_pred == 1).mean()
    )

    hybrid_metrics[
        "predicted_stress_rate"
    ] = float(
        (hybrid_pred == 1).mean()
    )

    reference_stress_rate = float(
        (y_true == 1).mean()
    )

    # --------------------------------------------------------
    # Save detailed predictions
    # --------------------------------------------------------

    prediction_output = evaluable[
        [
            "annotation_id",
            "record_id",
            "platform",
            "source",
            "community",
            text_column,
            "reference_stress_label",
            "annotation_confidence",
            "annotation_rationale",
        ]
    ].copy()

    prediction_output[
        "baseline_prediction"
    ] = [
        "Stress" if x == 1
        else "No stress"
        for x in baseline_pred
    ]

    prediction_output[
        "baseline_stress_probability"
    ] = baseline_prob

    prediction_output[
        "hybrid_prediction"
    ] = [
        "Stress" if x == 1
        else "No stress"
        for x in hybrid_pred
    ]

    prediction_output[
        "hybrid_stress_probability"
    ] = hybrid_prob

    prediction_output[
        "models_disagree"
    ] = (
        baseline_pred
        != hybrid_pred
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_output.to_csv(
        RESULTS_DIR
        / "stress_real_domain_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save comparison table
    # --------------------------------------------------------

    comparison_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "stress_precision",
        "stress_recall",
        "stress_f1",
        "specificity",
        "false_positive_rate",
        "mcc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "predicted_stress",
        "predicted_no_stress",
        "predicted_stress_rate",
    ]

    comparison = pd.DataFrame(
        [
            baseline_metrics,
            hybrid_metrics,
        ]
    )[
        comparison_columns
    ]

    comparison.to_csv(
        RESULTS_DIR
        / "stress_real_domain_model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    save_confusion_matrix(
        y_true,
        baseline_pred,
        (
            "Dreaddit-only Model — "
            "Real Developer Reference Set"
        ),
        FIGURES_DIR
        / "dreaddit_only_real_domain_confusion_matrix.png",
    )

    save_confusion_matrix(
        y_true,
        hybrid_pred,
        (
            "Hybrid Model — "
            "Real Developer Reference Set"
        ),
        FIGURES_DIR
        / "hybrid_real_domain_confusion_matrix.png",
    )

    # --------------------------------------------------------
    # Save metadata / summary
    # --------------------------------------------------------

    summary = {
        "reference_dataset_records": int(
            len(df)
        ),
        "evaluable_records": int(
            len(evaluable)
        ),
        "excluded_unclear_records": int(
            excluded_unclear
        ),
        "reference_no_stress_records": int(
            (y_true == 0).sum()
        ),
        "reference_stress_records": int(
            (y_true == 1).sum()
        ),
        "reference_stress_rate": (
            reference_stress_rate
        ),
        "reference_annotation_source": (
            "LLM-assisted reference annotation"
        ),
        "human_ground_truth": False,
        "clinical_ground_truth": False,
        "warning": (
            "Only 8 Stress reference cases are "
            "available; positive-class metrics "
            "are therefore unstable and must "
            "be interpreted cautiously."
        ),
        "models": {
            "dreaddit_only": (
                baseline_metrics
            ),
            "dreaddit_plus_synthetic": (
                hybrid_metrics
            ),
        },
    }

    (
        RESULTS_DIR
        / "stress_real_domain_evaluation.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("REFERENCE DOMAIN CHARACTERISTICS")
    print("=" * 72)

    print(
        f"Reference Stress rate: "
        f"{reference_stress_rate:.2%}"
    )

    print(
        "IMPORTANT: only 8 positive "
        "Stress reference examples."
    )

    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)

    print(
        comparison.to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("CONFUSION MATRICES")
    print("=" * 72)

    for name, metrics in [
        (
            "Dreaddit only",
            baseline_metrics,
        ),
        (
            "Dreaddit + Synthetic",
            hybrid_metrics,
        ),
    ]:

        print(f"\n{name}")

        print(
            f"TN={metrics['true_negative']}  "
            f"FP={metrics['false_positive']}  "
            f"FN={metrics['false_negative']}  "
            f"TP={metrics['true_positive']}"
        )

        print(
            f"Predicted Stress rate: "
            f"{metrics['predicted_stress_rate']:.2%}"
        )

        print(
            f"False-positive rate: "
            f"{metrics['false_positive_rate']:.2%}"
        )

    print(
        "\nResults saved under:"
    )

    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
