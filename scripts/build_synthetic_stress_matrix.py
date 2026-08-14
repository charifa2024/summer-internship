from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "synthetic_stress_generation_matrix.csv"
)


STRESS_SCENARIOS = [
    "job_insecurity",
    "skill_obsolescence",
    "learning_pressure",
    "deadline_pressure",
    "debugging_overload",
    "ai_generated_code_problems",
    "productivity_pressure",
    "burnout_exhaustion",
    "loss_of_professional_confidence",
    "fear_of_ai_dependency",
    "academic_project_pressure",
    "rapid_ai_change_anxiety",
]


NO_STRESS_SCENARIOS = [
    "angry_ai_criticism",
    "technical_frustration",
    "software_bug",
    "hallucination_complaint",
    "model_reliability_criticism",
    "privacy_security_criticism",
    "technical_confusion",
    "general_ai_job_discussion",
    "negative_ai_news",
    "sarcasm",
    "model_comparison",
    "product_dissatisfaction",
]


PLATFORM_STYLES = [
    "Reddit",
    "GitHub Issue",
    "Stack Overflow",
    "Hacker News",
]


AI_CONTEXTS = [
    "ChatGPT",
    "Claude",
    "Gemini",
    "GitHub Copilot",
    "Llama",
    "DeepSeek",
    "AI agents",
    "local LLMs",
    "general generative AI",
]


LENGTH_STYLES = [
    "short",
    "medium",
    "long",
]


def balanced_values(values, total):
    repeats = total // len(values)
    remainder = total % len(values)

    result = values * repeats
    result += values[:remainder]

    return result


def build_label_rows(
    label,
    scenarios,
    start_id,
    seed,
):
    rng = np.random.default_rng(seed)

    total = 600

    # Exactly 50 examples per scenario
    scenario_values = []

    for scenario in scenarios:
        scenario_values.extend(
            [scenario] * 50
        )

    # Exact balanced distributions
    platform_values = balanced_values(
        PLATFORM_STYLES,
        total,
    )

    length_values = balanced_values(
        LENGTH_STYLES,
        total,
    )

    ai_values = balanced_values(
        AI_CONTEXTS,
        total,
    )

    # Shuffle each dimension independently
    rng.shuffle(scenario_values)
    rng.shuffle(platform_values)
    rng.shuffle(length_values)
    rng.shuffle(ai_values)

    rows = []

    for i in range(total):
        rows.append(
            {
                "synthetic_id": (
                    f"syn_{start_id + i:04d}"
                ),
                "text": "",
                "stress_label": label,
                "scenario": scenario_values[i],
                "platform_style": platform_values[i],
                "ai_context": ai_values[i],
                "length_style": length_values[i],
                "generation_source": (
                    "LLM-generated synthetic "
                    "training data"
                ),
                "generation_version": "v1",
            }
        )

    return rows


def main():

    rows = []

    rows.extend(
        build_label_rows(
            label="Stress",
            scenarios=STRESS_SCENARIOS,
            start_id=1,
            seed=42,
        )
    )

    rows.extend(
        build_label_rows(
            label="No stress",
            scenarios=NO_STRESS_SCENARIOS,
            start_id=601,
            seed=84,
        )
    )

    df = pd.DataFrame(rows)

    # Shuffle final row order reproducibly
    df = (
        df.sample(
            frac=1,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 70)
    print("SYNTHETIC STRESS GENERATION MATRIX")
    print("=" * 70)

    print("\nTotal records:", len(df))

    print("\nLabel distribution:")
    print(
        df["stress_label"]
        .value_counts()
    )

    print("\nPlatform distribution:")
    print(
        pd.crosstab(
            df["stress_label"],
            df["platform_style"],
        )
    )

    print("\nLength distribution:")
    print(
        pd.crosstab(
            df["stress_label"],
            df["length_style"],
        )
    )

    print("\nScenario counts:")
    print(
        df.groupby(
            [
                "stress_label",
                "scenario",
            ]
        )
        .size()
        .to_string()
    )

    print("\nAI-context distribution:")
    print(
        pd.crosstab(
            df["stress_label"],
            df["ai_context"],
        )
    )

    # Check that AI context is not locked to one length
    print("\nAI context x length:")
    print(
        pd.crosstab(
            df["ai_context"],
            df["length_style"],
        )
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()