from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

METRICS_FILE = (
    OUTPUT_DIR
    / "topic_count_evaluation.csv"
)

KEYWORDS_FILE = (
    OUTPUT_DIR
    / "candidate_topic_keywords.csv"
)


# ============================================================
# TOPIC-MODEL STOPWORDS
# ============================================================

CUSTOM_STOPWORDS = {
    # Conversational filler
    "like",
    "just",
    "use",
    "using",
    "used",
    "ve",
    "don",
    "does",
    "doesn",
    "know",
    "want",
    "need",
    "help",
    "way",
    "make",
    "better",
    "good",
    "really",
    "actually",
    "right",
    "think",
    "try",
    "tried",
    "trying",
    "new",
    "got",
    "let",
    "thanks",
    "did",
    "things",

    # Corpus-wide domain background
    "ai",
    "model",
    "models",
    "llm",
    "llms",

    # Broad action/background terms
    "work",
    "working",
    "time",
    "run",
    "running",
    "based",
    "example",
}


def extract_top_terms(
    model,
    feature_names,
    top_n=10,
):
    rows = []

    for topic_idx, topic_weights in enumerate(
        model.components_
    ):
        top_indices = (
            topic_weights
            .argsort()[-top_n:]
            [::-1]
        )

        top_terms = [
            feature_names[i]
            for i in top_indices
        ]

        rows.append(
            {
                "topic": topic_idx,
                "terms": top_terms,
            }
        )

    return rows


def main():

    print("=" * 80)
    print("NMF TOPIC-COUNT EVALUATION")
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

    if texts.str.strip().eq("").any():
        raise ValueError(
            "Empty text found in unified dataset."
        )

    print("\nRecords:", len(df))

    # ========================================================
    # TF-IDF
    #
    # token_pattern:
    # - token must contain at least one letter
    # - therefore standalone numbers are excluded
    # - names such as qwen3 remain valid
    # ========================================================

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

    print(
        "TF-IDF shape:",
        X.shape,
    )

    print(
        "Vocabulary size:",
        len(feature_names),
    )

    # ========================================================
    # CANDIDATE TOPIC COUNTS
    # ========================================================

    candidate_topics = range(
        4,
        11,
    )

    evaluation_rows = []
    keyword_rows = []

    for n_topics in candidate_topics:

        print(
            "\n" + "=" * 80
        )
        print(
            f"NMF WITH {n_topics} TOPICS"
        )
        print(
            "=" * 80
        )

        model = NMF(
            n_components=n_topics,
            init="nndsvda",
            random_state=42,
            max_iter=500,
            solver="cd",
        )

        W = model.fit_transform(
            X
        )

        # ----------------------------------------------------
        # Dominant topic per document
        # ----------------------------------------------------

        dominant_topics = (
            W.argmax(axis=1)
        )

        topic_counts = (
            pd.Series(
                dominant_topics
            )
            .value_counts()
            .reindex(
                range(n_topics),
                fill_value=0,
            )
        )

        topic_shares = (
            topic_counts
            / len(df)
        )

        # ----------------------------------------------------
        # Topic confidence / separation
        # ----------------------------------------------------

        row_sums = W.sum(
            axis=1
        )

        row_sums[
            row_sums == 0
        ] = 1.0

        normalized_W = (
            W
            / row_sums[:, None]
        )

        sorted_weights = np.sort(
            normalized_W,
            axis=1,
        )

        top1 = (
            sorted_weights[:, -1]
        )

        if n_topics >= 2:
            top2 = (
                sorted_weights[:, -2]
            )
        else:
            top2 = np.zeros(
                len(df)
            )

        margin = (
            top1
            - top2
        )

        # ----------------------------------------------------
        # Topic diversity
        # ----------------------------------------------------

        top_terms = extract_top_terms(
            model,
            feature_names,
            top_n=10,
        )

        all_top_words = []

        for row in top_terms:
            all_top_words.extend(
                row["terms"]
            )

        unique_top_words = len(
            set(all_top_words)
        )

        topic_diversity = (
            unique_top_words
            / len(all_top_words)
        )

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        evaluation_rows.append(
            {
                "n_topics": n_topics,
                "reconstruction_error": float(
                    model.reconstruction_err_
                ),
                "topic_diversity": float(
                    topic_diversity
                ),
                "mean_dominant_topic_weight": float(
                    top1.mean()
                ),
                "mean_topic_margin": float(
                    margin.mean()
                ),
                "smallest_topic_records": int(
                    topic_counts.min()
                ),
                "largest_topic_records": int(
                    topic_counts.max()
                ),
                "smallest_topic_share": float(
                    topic_shares.min()
                ),
                "largest_topic_share": float(
                    topic_shares.max()
                ),
            }
        )

        # ----------------------------------------------------
        # Print topic keywords
        # ----------------------------------------------------

        for row in top_terms:

            topic_number = (
                row["topic"]
                + 1
            )

            terms_string = ", ".join(
                row["terms"]
            )

            print(
                f"Topic {topic_number}: "
                f"{terms_string}"
            )

            keyword_rows.append(
                {
                    "n_topics": n_topics,
                    "topic": row["topic"],
                    "topic_number": topic_number,
                    "top_terms": terms_string,
                    "dominant_document_count": int(
                        topic_counts[
                            row["topic"]
                        ]
                    ),
                    "dominant_document_share": float(
                        topic_shares[
                            row["topic"]
                        ]
                    ),
                }
            )

    # ========================================================
    # SAVE
    # ========================================================

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    keywords = pd.DataFrame(
        keyword_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation.to_csv(
        METRICS_FILE,
        index=False,
    )

    keywords.to_csv(
        KEYWORDS_FILE,
        index=False,
    )

    # ========================================================
    # FINAL TERMINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 80
    )
    print(
        "TOPIC-COUNT COMPARISON"
    )
    print(
        "=" * 80
    )

    display = (
        evaluation.copy()
    )

    for column in [
        "reconstruction_error",
        "topic_diversity",
        "mean_dominant_topic_weight",
        "mean_topic_margin",
        "smallest_topic_share",
        "largest_topic_share",
    ]:
        display[column] = (
            display[column]
            .round(4)
        )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        f"\nMetrics saved to:\n"
        f"{METRICS_FILE}"
    )

    print(
        f"\nCandidate keywords saved to:\n"
        f"{KEYWORDS_FILE}"
    )

    print(
        "\nNo final topic count has been selected yet."
    )


if __name__ == "__main__":
    main()
