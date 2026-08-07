from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "custom_stress_classifier.joblib"
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
    / "custom_stress_predictions.csv"
)


artifact = joblib.load(
    MODEL_FILE
)

model = artifact[
    "pipeline"
]

df = pd.read_json(
    INPUT_FILE,
    lines=True,
)

texts = (
    df["text_clean_basic"]
    .fillna("")
    .astype(str)
)

labels = model.predict(
    texts
)

probabilities = model.predict_proba(
    texts
)

stress_index = list(
    model.classes_
).index("Stress")

result = pd.DataFrame(
    {
        "record_id": df["record_id"],
        "custom_stress_label": labels,
        "custom_stress_probability": (
            probabilities[
                :,
                stress_index,
            ]
        ),
        "custom_stress_confidence": (
            probabilities.max(
                axis=1
            )
        ),
        "custom_stress_model": (
            artifact[
                "selected_model"
            ]
        ),
    }
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"Saved {len(result):,} predictions "
    f"to {OUTPUT_FILE}"
)
