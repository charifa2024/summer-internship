# 05 — Sentiment Analysis

## Objective

Measure broad textual polarity and determine whether different sentiment approaches interpret the final developer-oriented technical corpus consistently.

## Methods

### VADER
Lexicon- and rule-based polarity model.

### Contextual Transformer
Context-sensitive sentiment classifier.

Both produce:
- Positive
- Neutral
- Negative

## Final corpus results

| Model | Negative | Neutral | Positive |
|---|---:|---:|---:|
| VADER | 565 (21.19%) | 400 (15.00%) | 1,701 (63.80%) |
| Transformer | 482 (18.08%) | 1,707 (64.03%) | 477 (17.89%) |

Agreement:

```text
1,003 / 2,666 = 37.62%
```

This low agreement is a major methodological result: the corpus-level sentiment picture is **strongly model-dependent**.

## LLM-assisted reference evaluation

Reference sample:
- 150 selected posts
- 147 evaluable

| Model | Accuracy | Macro F1 | Cohen's κ |
|---|---:|---:|---:|
| VADER | 0.490 | 0.454 | 0.230 |
| Transformer | 0.510 | 0.510 | 0.278 |

The reference annotations are **LLM-assisted**, not independently verified human ground truth. Performance is moderate, so neither method should be treated as a definitive measurement of sentiment.

## Main outputs

```text
data/results/sentiment/llm_assisted_method_evaluation.csv
data/results/final_synthesis/sentiment_distribution.csv
data/results/final_synthesis/negative_sentiment_vs_stress.csv
```

## Critical interpretation

VADER and the Transformer operationalize sentiment differently:
- VADER classifies **63.80%** of the corpus as Positive;
- the Transformer classifies **64.03%** as Neutral.

Correct interpretation:

> VADER and the Transformer produce substantially different sentiment distributions. Their 37.62% agreement and moderate LLM-assisted reference performance indicate that sentiment conclusions are model-dependent.

Avoid:

> Developers are mostly positive.

or:

> Developers are mostly neutral.

**Decision:** retain both methods, report their disagreement explicitly, and do not use either sentiment model as a proxy for Stress.
