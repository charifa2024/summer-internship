# 07 — Stress Detection and Critical Validation

## Objective

Detect **Stress-related language as a separate task** rather than using negative sentiment as a proxy.

## Baseline training source

Dreaddit provides labelled Stress / No-stress Reddit text.

Candidate classifiers were compared using training cross-validation:
- Logistic Regression
- calibrated Linear SVM
- Complement Naive Bayes

The pre-declared selection criterion was Macro-F1, so the retained baseline uses **TF-IDF + Logistic Regression**.

## Hybrid augmentation

Final training configuration:

```text
2,838 real Dreaddit training posts
+ 1,200 synthetic developer-style examples
  ├── 600 Stress
  └── 600 No stress
= 4,038 training examples
```

Synthetic examples are **training augmentation only**. They are not validation evidence and do not replace real developer-domain reference data.

## Official Dreaddit test

| Metric | Dreaddit baseline | Hybrid |
|---|---:|---:|
| Accuracy | 0.7259 | 0.7371 |
| Macro F1 | 0.7249 | 0.7359 |
| Stress Precision | 0.7224 | 0.7303 |
| Stress Recall | 0.7615 | 0.7778 |
| Stress F1 | 0.7414 | **0.7533** |

The gains are **modest**. They do not by themselves demonstrate robust transfer to developer-oriented technical language.

## Developer-domain reference evaluation

Reference set:
- 600 posts
- 588 No stress
- 8 Stress
- 4 Unclear
- 596 evaluable

The labels are **LLM-assisted reference annotations**, not independent human or clinical ground truth.

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

## Critical validity boundary

The model detects 6 of only 8 reference Stress cases but produces **48 false positives**. Precision is therefore only **0.1111**. Technical complaints, debugging frustration and dramatic product criticism can resemble distress patterns learned from another domain.

Because only **eight positive reference cases** are evaluable, positive-class performance estimates are also unstable.

The hybrid model improves several metrics relative to the Dreaddit-only baseline, but these gains do **not** establish robust target-domain transfer.

## Threshold decision

The default 0.50 threshold is retained because raising it would reduce false positives but would also miss genuine positive reference examples in the very small positive class.

## Final corpus prediction

```text
No stress: 2,418
Predicted Stress: 248
Model-predicted Stress rate: 9.30%
```

Correct wording:

> **248 of 2,666 posts (9.30%) were classified as Stress by the final model.**

This is an exploratory **model-prediction rate**, not an estimate of psychological Stress prevalence among developers.

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
