# Final GoEmotions Model Decision

## Final model

```text
GoEmotions simplified — 28 labels
TF-IDF unigrams/bigrams
One-vs-Rest Logistic Regression
Per-emotion validation-calibrated thresholds
```

Dataset:
- Train: 43,410
- Validation: 5,426
- Test: 5,427

## Threshold decision

Global-threshold baseline:
- best validation region around 0.61

Final decision:
- one threshold per emotion using validation data;
- for emotions with fewer than 20 validation positives, retain the global threshold to avoid unstable calibration.

## Official test comparison

| Metric | Global | Final calibrated |
|---|---:|---:|
| Subset Accuracy | 0.2720 | 0.2983 |
| Micro Precision | 0.4271 | 0.4493 |
| Micro Recall | 0.5830 | 0.6208 |
| Micro F1 | 0.4930 | **0.5213** |
| Macro F1 | 0.4365 | **0.4438** |
| Samples F1 | 0.4907 | **0.5321** |
| Hamming Loss | 0.0499 | **0.0475** |

The final calibrated strategy is retained.

A neutral-exclusivity post-processing rule was tested and rejected because performance decreased.

## Final application

The model is applied to all **2,666** analysis-ready posts.

Main outputs:

```text
data/results/emotions/goemotions_predictions_analysis_ready.csv
data/results/emotions/goemotions_analysis_ready_emotion_prevalence.csv
data/results/emotions/goemotions_threshold_strategy_comparison.json
```

## Interpretation boundary

Emotion predictions are linguistic model outputs, not psychological diagnoses. General GoEmotions Reddit language differs from technical/developer-oriented language, so domain shift remains a limitation.
