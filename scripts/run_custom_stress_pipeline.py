
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train_custom_stress_model.py"
)

PREDICT_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "predict_custom_stress.py"
)

REQUIRED_INPUTS = [
    (
        PROJECT_ROOT
        / "data"
        / "annotations"
        / "llm_assisted_stress_labels.csv"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "analysis_ready_posts.jsonl"
    ),
]

EXPECTED_OUTPUTS = [
    (
        PROJECT_ROOT
        / "models"
        / "custom_stress_classifier.joblib"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "results"
        / "emotions"
        / "custom_stress_predictions.csv"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "results"
        / "emotions"
        / "custom_stress_model_comparison.csv"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "results"
        / "emotions"
        / "custom_stress_metrics.json"
    ),
    (
        PROJECT_ROOT
        / "figures"
        / "emotions"
        / "custom_stress_confusion_matrix.png"
    ),
    (
        PROJECT_ROOT
        / "figures"
        / "emotions"
        / "custom_stress_model_comparison.png"
    ),
]


def run_step(title: str, script: Path) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Running: {script}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise SystemExit(
            f"\nPipeline stopped because this step failed: {title}"
        )


def main() -> None:
    print("Custom Stress Model Pipeline")
    print(f"Project root: {PROJECT_ROOT}")

    missing_inputs = [
        path
        for path in REQUIRED_INPUTS
        if not path.exists()
    ]

    missing_scripts = [
        path
        for path in [
            TRAIN_SCRIPT,
            PREDICT_SCRIPT,
        ]
        if not path.exists()
    ]

    if missing_inputs or missing_scripts:
        print("\nMissing required files:")

        for path in [
            *missing_inputs,
            *missing_scripts,
        ]:
            print(f"- {path}")

        raise SystemExit(
            "\nCopy the missing files into the exact paths above, "
            "then run the pipeline again."
        )

    run_step(
        "STEP 1/2 — Train, test, compare, and save the model",
        TRAIN_SCRIPT,
    )

    run_step(
        "STEP 2/2 — Predict stress for all cleaned discussions",
        PREDICT_SCRIPT,
    )

    print("\n" + "=" * 70)
    print("OUTPUT VERIFICATION")
    print("=" * 70)

    missing_outputs = []

    for path in EXPECTED_OUTPUTS:
        if path.exists():
            size = path.stat().st_size
            print(f"[OK] {path.relative_to(PROJECT_ROOT)} ({size:,} bytes)")
        else:
            print(f"[MISSING] {path.relative_to(PROJECT_ROOT)}")
            missing_outputs.append(path)

    if missing_outputs:
        raise SystemExit(
            "\nThe pipeline finished, but some expected outputs "
            "were not generated."
        )

    metrics_path = (
        PROJECT_ROOT
        / "data"
        / "results"
        / "emotions"
        / "custom_stress_metrics.json"
    )

    metrics = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL SUMMARY")
    print("=" * 70)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nPipeline completed successfully.")
    print("\nOpen the dashboard with:")
    print("streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
