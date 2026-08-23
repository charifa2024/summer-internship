# 05 — Sentiment Analysis

## Objective

Measure broad textual polarity and determine whether different sentiment approaches interpret developer-oriented technical language consistently.

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

This disagreement is a major methodological result.

## LLM-assisted reference evaluation

Reference sample:
- 150 selected posts
- 147 evaluable

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| VADER | 0.490 | 0.454 |
| Transformer | 0.510 | 0.510 |

The reference annotations are **LLM-assisted**, not independently verified human ground truth.

## Main outputs

```text
data/results/sentiment/llm_assisted_method_evaluation.csv
data/results/final_synthesis/sentiment_distribution.csv
data/results/final_synthesis/negative_sentiment_vs_stress.csv
```

## Critical interpretation

The models operationalize sentiment differently:
- VADER emphasizes lexical polarity;
- Transformer classification is more contextual.

No model is treated as absolute ground truth. Both are retained so downstream analysis can show how conclusions depend on sentiment methodology.
