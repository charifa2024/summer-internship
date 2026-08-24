# Final Stress Model Decision

## Final classifier

```text
2,838 Dreaddit official training posts
+ 1,200 synthetic developer-style training examples
= 4,038 training records

TF-IDF unigrams/bigrams
+ Logistic Regression
+ threshold 0.50
```

Synthetic data are **training augmentation only**.

## Model-selection principle

Candidate baseline classifiers were compared using training cross-validation:
- Logistic Regression
- calibrated Linear SVM
- Complement Naive Bayes

The pre-declared selection criterion was Macro-F1, under which Logistic Regression ranked highest. The retained baseline was then evaluated on the untouched official Dreaddit test split.

## Dreaddit official test

Hybrid model:
- Accuracy: 0.7371
- Macro F1: 0.7359
- Stress Precision: 0.7303
- Stress Recall: 0.7778
- Stress F1: 0.7533

Synthetic augmentation provides a **modest observed improvement** over the Dreaddit-only baseline. This source-domain gain does not by itself demonstrate robust transfer to developer-oriented technical language.

## Developer-domain reference evaluation

600-post set:
- 588 No stress
- 8 Stress
- 4 Unclear
- 596 evaluable

Hybrid:
- Balanced Accuracy: 0.8342
- Stress Precision: **0.1111**
- Stress Recall: **0.7500**
- Stress F1: 0.1935
- MCC: 0.2679

Confusion matrix:
- TN=540
- FP=48
- FN=2
- TP=6

## Critical finding

The model detects 6 of only 8 reference Stress cases but produces 48 false positives. Technical frustration and strong complaint language frequently resemble general-domain distress patterns.

This is evidence of **domain shift**. Because only eight positive cases are available, positive-class metrics are unstable. The hybrid model improves several target-domain metrics relative to the Dreaddit-only baseline, but the result does **not** establish robust domain transfer.

## Final corpus

- 2,418 No stress
- 248 classified as Stress
- 9.30% **model-predicted Stress rate**

This is an exploratory model-prediction rate, not psychological Stress prevalence.

## Final role in the pipeline

```text
Sentiment
→ Emotions
→ Stress & critical validation
→ Integrated analysis
→ NMF topics
→ Topic × model-predicted Stress statistics
```
