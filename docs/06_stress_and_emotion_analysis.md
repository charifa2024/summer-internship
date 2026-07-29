# 06 — Stress and Emotion Analysis

## Objective

This phase extends the general sentiment analysis by identifying explicit
linguistic signals associated with stress and related emotions in developer
discussions about AI tools.

Sentiment classification and stress detection are treated as separate tasks.

- Sentiment describes general polarity: positive, neutral, or negative.
- Stress analysis searches for explicit signals of stress, anxiety,
  frustration, exhaustion, fear, distrust, or relief.

The method does not make clinical diagnoses.

## Notebook

`notebooks/06_stress_and_emotion_analysis.ipynb`

## Input Priority

1. `data/results/sentiment/sentiment_predictions.jsonl`
2. `data/results/sentiment/vader_predictions.jsonl`
3. `data/processed/analysis_ready_posts.jsonl`

## Method

The notebook uses transparent phrase and keyword rules with word boundaries.

The main emotion categories are:

- Stress or anxiety
- Frustration
- Burnout or exhaustion
- Job insecurity or fear
- Distrust or uncertainty
- Relief or reduced stress
- No clear emotion

The main stress directions are:

- Increased stress or distress
- Reduced stress or relief
- Mixed or ambiguous
- No clear stress signal

Negative sentiment alone is not treated as proof of stress. At least one
explicit emotion-related linguistic signal is required.

## Stress Intensity

The notebook estimates:

- None
- Low
- Moderate
- High

Intensity depends on the number of explicit negative-emotion matches and the
presence of linguistic intensifiers.

## Likely Causes

The method detects possible associations with:

- Job security and replacement
- Code quality and debugging
- Reliability and hallucinations
- Workload and productivity
- Privacy and security
- Cost and access
- Learning and skill pressure
- General AI uncertainty

Existing rule-based `research_themes` are used as supplementary evidence.

## Main Output Columns

- `stress_direction`
- `stress_signal`
- `stress_intensity`
- `stress_intensity_score`
- `primary_emotion`
- `detected_emotions`
- `stress_causes`
- `stress_evidence`
- `stress_detection_confidence`
- `negative_emotion_hit_count`
- `relief_hit_count`

## Main Outputs

`data/results/emotions/stress_emotion_predictions.jsonl`

`data/results/emotions/stress_emotion_predictions.csv`

Additional CSV outputs summarize:

- overall stress direction;
- primary emotions;
- intensity;
- platform differences;
- AI-tool differences;
- likely causes;
- sentiment versus stress;
- temporal evolution;
- representative review cases.

## Figures

Figures are saved in:

`figures/emotions/`

They include:

- stress-direction distribution;
- primary-emotion distribution;
- stress-intensity distribution;
- stress by platform;
- stress causes;
- stress by AI tool;
- stress over time.

## Interpretation

Safe interpretation:

> The rule-based method identified explicit textual signals associated with
> stress and related emotions.

Unsafe interpretation:

> The method diagnosed stressed developers.

The analysis concerns language in public posts, not the clinical state of an
individual.

## Limitations

- Rule-based detection may miss implicit emotions.
- Sarcasm and mixed sentiment remain difficult.
- Technical terms may cause false positives.
- Cause categories are approximate linguistic associations.
- The method is an explainable baseline, not ground truth.
- The dataset does not represent all developers.

## Completion

The phase is complete after:

- running all notebook cells;
- reviewing generated distributions;
- inspecting representative examples;
- confirming output files;
- documenting limitations.

The next phase is:

**07 — Topic Modeling**
