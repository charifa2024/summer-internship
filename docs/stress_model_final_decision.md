# Final Stress Model Decision

## Final model

The retained stress classifier is the hybrid model:

- Dreaddit official training split: 2,838 real labelled posts
- Synthetic developer augmentation: 1,200 posts
- Total training records: 4,038
- TF-IDF unigram/bigram representation
- Logistic Regression
- Classification threshold: 0.50

Synthetic data are used only as experimental augmentation and do not
replace real Dreaddit training data.

## Dreaddit official test performance

The untouched Dreaddit test split contains 715 posts.

Hybrid model:

- Accuracy: 0.7371
- Macro F1: 0.7359
- Stress Precision: 0.7303
- Stress Recall: 0.7778
- Stress F1: 0.7533

The hybrid model improved all reported metrics relative to the
Dreaddit-only baseline.

## Developer-domain validation

The developer reference set originally contains 600 posts:

- 588 No stress
- 8 Stress
- 4 Unclear

The 4 Unclear records are excluded from metric calculation, leaving
596 evaluable records.

Hybrid performance:

- Accuracy: 0.9161
- Balanced Accuracy: 0.8342
- Macro F1: 0.5747
- Stress Precision: 0.1111
- Stress Recall: 0.7500
- Stress F1: 0.1935
- Specificity: 0.9184
- False-positive rate: 0.0816
- MCC: 0.2679

Confusion matrix:

- True Negative: 540
- False Positive: 48
- False Negative: 2
- True Positive: 6

## Error analysis

False positives are dominated by technical questions, criticism,
confusion, frustration, and general AI-risk discussions that contain
language resembling distress without expressing personal
psychological stress.

False negatives mainly involve more subtle career-related pressure,
such as interview nervousness, unemployment, or uncertainty about
future employability.

The remaining errors therefore indicate domain shift between general
Reddit stress language and AI/developer discussions.

## Threshold decision

The default 0.50 threshold is retained.

Several genuine Stress examples have probabilities only slightly above
0.50, so increasing the threshold would reduce false positives at the
cost of substantially reducing Stress Recall.

## Interpretation

The hybrid model is retained because synthetic augmentation improves
performance on both:

1. the untouched Dreaddit official test set; and
2. the developer-domain reference set.

However, developer-domain Stress prevalence must NOT be inferred
directly from model predictions.

The developer reference annotations are LLM-assisted exploratory
annotations and are neither human clinical ground truth nor diagnostic
labels.

Because the 600 developer posts have now been used for error analysis,
they are treated as a developer-domain validation/reference set rather
than an untouched final test set.

## Next stage

The Stress component is considered methodologically stable.

The next analysis stage is GoEmotions, followed by:

Sentiment -> Emotions -> Stress -> Stress-associated Topics.
