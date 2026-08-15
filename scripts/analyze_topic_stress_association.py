from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    chi2_contingency,
    fisher_exact,
)

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

ASSOCIATION_FILE = (
    OUTPUT_DIR
    / "topic_stress_association.csv"
)

CONTINGENCY_FILE = (
    OUTPUT_DIR
    / "topic_stress_contingency.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "topic_stress_statistical_summary.csv"
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


def benjamini_hochberg(p_values):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p_values)

    order = np.argsort(
        p_values
    )

    ranked = p_values[
        order
    ]

    adjusted_ranked = np.empty(
        n,
        dtype=float,
    )

    previous = 1.0

    for i in range(
        n - 1,
        -1,
        -1,
    ):
        rank = i + 1

        adjusted = (
            ranked[i]
            * n
            / rank
        )

        previous = min(
            previous,
            adjusted,
        )

        adjusted_ranked[i] = min(
            previous,
            1.0,
        )

    adjusted = np.empty(
        n,
        dtype=float,
    )

    adjusted[
        order
    ] = adjusted_ranked

    return adjusted


def main():

    print("=" * 80)
    print("TOPIC × PREDICTED-STRESS STATISTICAL ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(
        INPUT_FILE
    )

    if len(df) != 2666:
        raise ValueError(
            f"Expected 2666 rows, found {len(df)}."
        )

    # ========================================================
    # Reproduce frozen 7-topic model
    # ========================================================

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

    df["is_stress"] = (
        df["hybrid_stress_label"]
        .eq("Stress")
        .astype(int)
    )

    # ========================================================
    # Overall chi-square
    # ========================================================

    contingency = pd.crosstab(
        df["topic"],
        df["is_stress"],
    )

    contingency = contingency.reindex(
        index=range(1, 8),
        columns=[0, 1],
        fill_value=0,
    )

    contingency.columns = [
        "No stress",
        "Stress",
    ]

    chi2, p_value, dof, expected = (
        chi2_contingency(
            contingency
        )
    )

    n = contingency.values.sum()

    cramers_v = math.sqrt(
        chi2
        / (
            n
            * min(
                contingency.shape[0] - 1,
                contingency.shape[1] - 1,
            )
        )
    )

    baseline_rate = (
        df["is_stress"]
        .mean()
    )

    # ========================================================
    # Per-topic association
    # ========================================================

    rows = []

    for topic in range(1, 8):

        in_topic = (
            df["topic"]
            == topic
        )

        stress = (
            df["is_stress"]
            == 1
        )

        a = int(
            (
                in_topic
                & stress
            ).sum()
        )

        b = int(
            (
                in_topic
                & ~stress
            ).sum()
        )

        c = int(
            (
                ~in_topic
                & stress
            ).sum()
        )

        d = int(
            (
                ~in_topic
                & ~stress
            ).sum()
        )

        table = [
            [a, b],
            [c, d],
        ]

        odds_ratio, fisher_p = (
            fisher_exact(
                table,
                alternative="two-sided",
            )
        )

        topic_total = (
            a + b
        )

        outside_total = (
            c + d
        )

        topic_rate = (
            a
            / topic_total
        )

        outside_rate = (
            c
            / outside_total
        )

        relative_risk = (
            topic_rate
            / outside_rate
            if outside_rate > 0
            else np.nan
        )

        difference_pp = (
            topic_rate
            - baseline_rate
        ) * 100

        # Haldane correction for log-OR CI
        aa = a + 0.5
        bb = b + 0.5
        cc = c + 0.5
        dd = d + 0.5

        corrected_or = (
            aa * dd
        ) / (
            bb * cc
        )

        se_log_or = math.sqrt(
            1 / aa
            + 1 / bb
            + 1 / cc
            + 1 / dd
        )

        lower = math.exp(
            math.log(corrected_or)
            - 1.96 * se_log_or
        )

        upper = math.exp(
            math.log(corrected_or)
            + 1.96 * se_log_or
        )

        rows.append(
            {
                "topic": topic,
                "topic_name": (
                    TOPIC_NAMES[
                        topic
                    ]
                ),
                "posts": topic_total,
                "stress_posts": a,
                "no_stress_posts": b,
                "stress_rate": topic_rate,
                "baseline_stress_rate": baseline_rate,
                "difference_from_baseline_pp": difference_pp,
                "outside_topic_stress_rate": outside_rate,
                "relative_risk": relative_risk,
                "odds_ratio": odds_ratio,
                "odds_ratio_ci95_lower": lower,
                "odds_ratio_ci95_upper": upper,
                "fisher_p_value": fisher_p,
            }
        )

    results = pd.DataFrame(
        rows
    )

    results[
        "fisher_p_fdr_bh"
    ] = benjamini_hochberg(
        results[
            "fisher_p_value"
        ].values
    )

    results[
        "significant_fdr_0_05"
    ] = (
        results[
            "fisher_p_fdr_bh"
        ]
        < 0.05
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contingency.to_csv(
        CONTINGENCY_FILE
    )

    results.to_csv(
        ASSOCIATION_FILE,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "chi_square": chi2,
                "degrees_of_freedom": dof,
                "p_value": p_value,
                "cramers_v": cramers_v,
                "overall_predicted_stress_rate": baseline_rate,
                "records": len(df),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # Terminal report
    # ========================================================

    print(
        f"\nOverall predicted-Stress rate: "
        f"{baseline_rate:.2%}"
    )

    print("\nContingency table:")
    print(
        contingency.to_string()
    )

    print("\nOverall association:")
    print(
        f"Chi-square = {chi2:.4f}"
    )
    print(
        f"df = {dof}"
    )
    print(
        f"p = {p_value:.8f}"
    )
    print(
        f"Cramer's V = {cramers_v:.4f}"
    )

    print(
        "\nPer-topic association:"
    )

    display = results[
        [
            "topic",
            "topic_name",
            "posts",
            "stress_posts",
            "stress_rate",
            "difference_from_baseline_pp",
            "relative_risk",
            "odds_ratio",
            "fisher_p_value",
            "fisher_p_fdr_bh",
            "significant_fdr_0_05",
        ]
    ].copy()

    for col in [
        "stress_rate",
        "relative_risk",
        "odds_ratio",
        "fisher_p_value",
        "fisher_p_fdr_bh",
    ]:
        display[col] = (
            display[col]
            .round(4)
        )

    display[
        "difference_from_baseline_pp"
    ] = (
        display[
            "difference_from_baseline_pp"
        ]
        .round(2)
    )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:\n"
        f"{ASSOCIATION_FILE}"
    )


if __name__ == "__main__":
    main()
