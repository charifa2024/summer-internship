# 06 — Multi-Label Emotion Analysis

## Objective

Move beyond Positive/Neutral/Negative and identify more specific emotion-related language.

## Dataset

GoEmotions simplified:
- Training: 43,410
- Validation: 5,426
- Test: 5,427
- Labels: 28

## Final model

```text
TF-IDF (unigrams + bigrams)
        ↓
One-vs-Rest Logistic Regression
        ↓
28 independent emotion probabilities
        ↓
Per-emotion validation-calibrated thresholds
```

A post can receive multiple labels.

## Threshold experiments

A global threshold around **0.61** was first evaluated.

Final strategy:
- calibrate one threshold per emotion on validation data;
- keep the global threshold for emotions with fewer than 20 validation positives.

### Test comparison

| Metric | Global threshold | Final per-emotion thresholds |
|---|---:|---:|
| Subset Accuracy | 0.2720 | 0.2983 |
| Micro Precision | 0.4271 | 0.4493 |
| Micro Recall | 0.5830 | 0.6208 |
| Micro F1 | 0.4930 | **0.5213** |
| Macro F1 | 0.4365 | **0.4438** |
| Samples F1 | 0.4907 | **0.5321** |
| Hamming Loss | 0.0499 | **0.0475** |

A neutral-exclusivity post-processing rule was also tested and rejected because it reduced performance.

## Application to the 2,666 posts

- Average predicted emotions per post: **1.85**
- Median: **2**
- Multiple emotions: **1,406 posts (52.74%)**
- No predicted emotion: **115 posts (4.31%)**

Most frequent predicted emotions include:
- Neutral 57.16%
- Approval 31.06%
- Disapproval 20.41%
- Curiosity 16.80%
- Confusion 12.42%
- Annoyance 10.77%

Percentages do not sum to 100% because this is multi-label classification.

## Main outputs

```text
data/results/emotions/goemotions_predictions_analysis_ready.csv
data/results/emotions/goemotions_analysis_ready_emotion_prevalence.csv
data/results/emotions/goemotions_threshold_strategy_comparison.json
```

## Limitation

GoEmotions contains general Reddit language, while the target corpus is developer-oriented technical. Emotion predictions are therefore exploratory linguistic signals, not verified psychological states.
