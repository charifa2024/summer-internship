# 08 — Integrated Sentiment × Emotions × Stress Analysis

## Objective

Combine independently generated predictions at the post level and determine how the analytical signals relate to one another.

## Integration key

```text
record_id
```

Final unified dataset:

```text
data/results/combined/sentiment_emotion_stress_predictions.csv
```

Integrity checks:
- 2,666 rows
- unique record IDs
- no record-set differences between merged final branches

## Central result: Negative ≠ Stress

### VADER
- Negative posts: 565
- Predicted Stress among them: 81
- Rate: **14.34%**

### Transformer
- Negative posts: 482
- Predicted Stress among them: 117
- Rate: **24.27%**

Most negative posts are therefore **not** classified as Stress.

## Emotion enrichment in predicted-Stress posts

Examples:

| Emotion | Predicted-Stress posts | No-stress posts |
|---|---:|---:|
| Annoyance | 26.61% | 9.14% |
| Disappointment | 20.56% | 4.22% |
| Confusion | 23.79% | 11.25% |
| Realization | 18.15% | 7.20% |
| Disapproval | 29.84% | 19.44% |

These are **descriptive enrichments**. They should not be called causal effects or individually statistically significant unless tested separately.

## Main outputs

```text
data/results/combined/sentiment_emotion_stress_predictions.csv
data/results/final_synthesis/negative_sentiment_vs_stress.csv
data/results/final_synthesis/emotion_stress_association.csv
```

## Critical boundary

Integrated analysis combines predicted labels. Upstream classification errors can propagate into cross-signal comparisons, so results must be interpreted as model-based analytical evidence.
