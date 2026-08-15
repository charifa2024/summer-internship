from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts"),
)

from evaluate_stress_topic_models import CUSTOM_STOPWORDS


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "combined"
    / "sentiment_emotion_stress_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "stress_topics"
)

ASSIGNMENTS_FILE = (
    OUTPUT_DIR
    / "final_topic_assignments.csv"
)

KEYWORDS_FILE = (
    OUTPUT_DIR
    / "final_topic_keywords.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_topic_summary.csv"
)

REPRESENTATIVES_FILE = (
    OUTPUT_DIR
    / "final_topic_representative_documents.csv"
)

DECISION_FILE = (
    OUTPUT_DIR
    / "final_topic_model_decision.json"
)


TOPIC_NAMES = {
    1: "General AI Development & Agent Workflows",
    2: "Gemini / Google Ecosystem",
    3: "Claude Coding & Development Tools",
    4: "VS Code / GitHub Copilot Technical Issues",
    5: "Local LLM Deployment & Inference Performance",
    6: "AI Labs / Industry News",
    7: "ChatGPT / OpenAI User Experience",
}


def main():

    print("=" * 80)
    print("FINAL 7-TOPIC NMF MODEL")
    print("=" * 80)

    df = pd.read_csv(
        INPUT_FILE
    )

    if len(df) != 2666:
        raise ValueError(
            f"Expected 2666 records, found {len(df)}."
        )

    texts = (
        df["text_clean_basic"]
        .fillna("")
        .astype(str)
    )

    stopwords = list(
        set(ENGLISH_STOP_WORDS)
        | CUSTOM_STOPWORDS
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.90,
        max_features=10000,
        sublinear_tf=True,
        token_pattern=(
            r"(?u)\b"
            r"(?=[A-Za-z0-9_-]*[A-Za-z])"
            r"[A-Za-z0-9_-]{2,}\b"
        ),
    )

    X = vectorizer.fit_transform(
        texts
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    model = NMF(
        n_components=7,
        init="nndsvda",
        random_state=42,
        max_iter=500,
        solver="cd",
    )

    W = model.fit_transform(
        X
    )

    df["topic"] = (
        W.argmax(axis=1)
        + 1
    )

    df["topic_name"] = (
        df["topic"]
        .map(TOPIC_NAMES)
    )

    df["topic_weight"] = (
        W.max(axis=1)
    )

    # ========================================================
    # Final assignments
    # ========================================================

    assignment_columns = [
        "record_id",
        "platform",
        "source",
        "text_clean_basic",
        "topic",
        "topic_name",
        "topic_weight",
        "hybrid_stress_label",
        "hybrid_stress_probability",
        "vader_label",
        "transformer_label",
        "top_emotion",
    ]

    assignments = df[
        [
            col
            for col in assignment_columns
            if col in df.columns
        ]
    ].copy()

    # ========================================================
    # Keywords
    # ========================================================

    keyword_rows = []

    for topic_index in range(7):

        weights = (
            model.components_[
                topic_index
            ]
        )

        top_indices = (
            weights
            .argsort()[-15:]
            [::-1]
        )

        top_terms = [
            feature_names[i]
            for i in top_indices
        ]

        keyword_rows.append(
            {
                "topic": topic_index + 1,
                "topic_name": TOPIC_NAMES[
                    topic_index + 1
                ],
                "top_terms": ", ".join(
                    top_terms
                ),
            }
        )

    keywords = pd.DataFrame(
        keyword_rows
    )

    # ========================================================
    # Topic summary
    # ========================================================

    summary_rows = []

    overall_stress_rate = (
        df[
            "hybrid_stress_label"
        ]
        .eq("Stress")
        .mean()
    )

    for topic in range(1, 8):

        topic_df = df[
            df["topic"]
            == topic
        ]

        stress_count = int(
            topic_df[
                "hybrid_stress_label"
            ]
            .eq("Stress")
            .sum()
        )

        stress_rate = (
            stress_count
            / len(topic_df)
        )

        summary_rows.append(
            {
                "topic": topic,
                "topic_name": (
                    TOPIC_NAMES[topic]
                ),
                "posts": int(
                    len(topic_df)
                ),
                "post_share": float(
                    len(topic_df)
                    / len(df)
                ),
                "predicted_stress_posts": (
                    stress_count
                ),
                "predicted_stress_rate": float(
                    stress_rate
                ),
                "overall_predicted_stress_rate": float(
                    overall_stress_rate
                ),
                "difference_from_overall_pp": float(
                    (
                        stress_rate
                        - overall_stress_rate
                    )
                    * 100
                ),
                "mean_stress_probability": float(
                    topic_df[
                        "hybrid_stress_probability"
                    ]
                    .mean()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # Representative documents
    # ========================================================

    representative_rows = []

    for topic in range(1, 8):

        topic_df = (
            df[
                df["topic"] == topic
            ]
            .sort_values(
                "topic_weight",
                ascending=False,
            )
            .head(5)
        )

        for rank, (_, row) in enumerate(
            topic_df.iterrows(),
            start=1,
        ):

            representative_rows.append(
                {
                    "topic": topic,
                    "topic_name": (
                        TOPIC_NAMES[topic]
                    ),
                    "rank": rank,
                    "record_id": row[
                        "record_id"
                    ],
                    "platform": row[
                        "platform"
                    ],
                    "topic_weight": row[
                        "topic_weight"
                    ],
                    "hybrid_stress_label": row[
                        "hybrid_stress_label"
                    ],
                    "text": row[
                        "text_clean_basic"
                    ],
                }
            )

    representatives = pd.DataFrame(
        representative_rows
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments.to_csv(
        ASSIGNMENTS_FILE,
        index=False,
    )

    keywords.to_csv(
        KEYWORDS_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    representatives.to_csv(
        REPRESENTATIVES_FILE,
        index=False,
    )

    decision = {
        "method": "TF-IDF + NMF",
        "records": 2666,
        "n_topics": 7,
        "selection_basis": (
            "Candidate models from 4 to 10 topics "
            "were compared using reconstruction error, "
            "topic diversity, topic separation, topic sizes, "
            "keywords, and qualitative inspection of "
            "representative documents."
        ),
        "stress_interpretation": (
            "Topics are learned independently from all "
            "2,666 posts. Predicted Stress is used only "
            "after topic assignment to test association."
        ),
        "causal_warning": (
            "Topic associations with predicted Stress "
            "must not be interpreted as causal effects "
            "or clinical psychological conclusions."
        ),
        "topic_names": TOPIC_NAMES,
    }

    DECISION_FILE.write_text(
        json.dumps(
            decision,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Terminal output
    # ========================================================

    print(
        "\nFinal topic keywords:"
    )

    print(
        keywords.to_string(
            index=False
        )
    )

    print(
        "\nFinal topic summary:"
    )

    display = summary.copy()

    for col in [
        "post_share",
        "predicted_stress_rate",
        "overall_predicted_stress_rate",
        "mean_stress_probability",
    ]:
        display[col] = (
            display[col]
            .round(4)
        )

    display[
        "difference_from_overall_pp"
    ] = (
        display[
            "difference_from_overall_pp"
        ]
        .round(2)
    )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\nFinal model:"
        " 7-topic NMF"
    )

    print(
        f"\nAssignments saved to:\n"
        f"{ASSIGNMENTS_FILE}"
    )


if __name__ == "__main__":
    main()
