# 07 — Stress Detection and Critical Validation

## Objective

Detect **Stress-related language as a separate task** rather than using negative sentiment as a proxy.

## Baseline training source

Dreaddit provides labelled Stress / No-stress Reddit text.

Candidate classifiers were compared using training cross-validation:
- Logistic Regression
- calibrated Linear SVM
- Complement Naive Bayes

The final baseline uses TF-IDF + Logistic Regression.

## Hybrid augmentation

Final training configuration:

```text
2,838 real Dreaddit training posts
+ 1,200 synthetic developer-style examples
  ├── 600 Stress
  └── 600 No stress
= 4,038 training examples
```

Synthetic examples are **training augmentation only**.

## Official Dreaddit test

| Metric | Dreaddit baseline | Hybrid |
|---|---:|---:|
| Accuracy | 0.7259 | 0.7371 |
| Macro F1 | 0.7249 | 0.7359 |
| Stress Precision | 0.7224 | 0.7303 |
| Stress Recall | 0.7615 | 0.7778 |
| Stress F1 | 0.7414 | **0.7533** |

The improvement is modest.

## Developer-domain reference evaluation

Reference set:
- 600 posts
- 588 No stress
- 8 Stress
- 4 Unclear
- 596 evaluable

The labels are LLM-assisted reference annotations and are not human/clinical ground truth.

Confusion matrix:

| | Predicted No stress | Predicted Stress |
|---|---:|---:|
| Reference No stress | 540 | 48 |
| Reference Stress | 2 | 6 |

Metrics:
- Accuracy: 0.9161
- Balanced Accuracy: 0.8342
- Macro F1: 0.5747
- Stress Precision: **0.1111**
- Stress Recall: **0.7500**
- Stress F1: 0.1935
- Specificity: 0.9184
- FPR: 0.0816
- MCC: 0.2679

## Error analysis

Main false-positive pattern:
- technical complaints;
- debugging frustration;
- strong product criticism;
- dramatic technical language without clear personal psychological distress.

This demonstrates domain shift.

## Threshold decision

The default 0.50 threshold is retained because raising it would reduce false positives but would also miss genuine positive reference examples.

## Final corpus prediction

```text
No stress: 2,418
Predicted Stress: 248
Prediction rate: 9.30%
```

Correct wording:

> 9.30% of posts were classified as Stress by the final model.

Incorrect wording:

> 9.30% of developers are stressed.

## Main outputs

```text
models/hybrid_dreaddit_synthetic_stress_classifier.joblib
data/results/emotions/dreaddit_stress_model_comparison.csv
data/results/emotions/hybrid_stress_predictions_analysis_ready.csv
data/results/emotions/hybrid_stress_analysis_ready_summary.json
data/results/emotions/stress_real_domain_evaluation.json
data/results/emotions/stress_real_domain_model_comparison.csv
data/results/emotions/stress_real_domain_predictions.csv
```
