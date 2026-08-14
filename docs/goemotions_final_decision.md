# Final GoEmotions Model Decision

## Objective

The emotion-analysis component uses GoEmotions to classify developer
discussions into multiple emotion categories.

Unlike sentiment classification, emotion classification is multi-label:
a single post may express more than one emotion.

## Dataset

GoEmotions simplified dataset:

- Training records: 43,410
- Validation records: 5,426
- Test records: 5,427
- Emotion labels: 28

The labels are:

- admiration
- amusement
- anger
- annoyance
- approval
- caring
- confusion
- curiosity
- desire
- disappointment
- disapproval
- disgust
- embarrassment
- excitement
- fear
- gratitude
- grief
- joy
- love
- nervousness
- optimism
- pride
- realization
- relief
- remorse
- sadness
- surprise
- neutral

## Model

The retained model is:

TF-IDF
+
One-vs-Rest Logistic Regression

Configuration:

- TF-IDF unigrams and bigrams
- maximum 12,000 features
- balanced class weights
- 28 independent binary Logistic Regression classifiers

This lightweight approach was selected because it can be trained and
executed locally while preserving the multi-label structure of
GoEmotions.

## Threshold selection

A single global threshold was first evaluated.

The validation split identified approximately 0.61 as the strongest
global threshold according to micro-F1.

However, GoEmotions is highly imbalanced. Common emotions such as
neutral have thousands of examples, while emotions such as grief,
pride, relief, and nervousness are rare.

Therefore, a second calibration stage selected one probability
threshold per emotion using only the GoEmotions validation split.

For emotions with fewer than 20 validation positives, the global
threshold of 0.61 was retained to avoid unstable calibration.

The final prediction strategy is therefore:

Per-emotion validation-calibrated thresholds.

## Global threshold baseline

Using a global threshold of 0.61 on the GoEmotions test set:

- Subset Accuracy: 0.2720
- Micro Precision: 0.4271
- Micro Recall: 0.5830
- Micro F1: 0.4930
- Macro F1: 0.4365
- Samples F1: 0.4907
- Hamming Loss: 0.0499
- Predicted emotion cardinality: 1.592

## Final calibrated strategy

Using per-emotion thresholds:

- Subset Accuracy: 0.2983
- Micro Precision: 0.4493
- Micro Recall: 0.6208
- Micro F1: 0.5213
- Macro F1: 0.4438
- Samples F1: 0.5321
- Hamming Loss: 0.0475
- Predicted emotion cardinality: 1.611

The calibrated strategy improves the main multi-label metrics and is
therefore retained.

## Important limitations

Performance differs substantially across emotions.

Strong-performing emotions include:

- gratitude
- amusement
- love
- admiration
- fear

More difficult or rare emotions include:

- nervousness
- disappointment
- relief
- caring
- realization

The model still predicts more labels per post than are present on
average in the GoEmotions reference labels.

This must be considered when interpreting emotion prevalence on the
developer dataset.

GoEmotions contains general Reddit language, while the target dataset
contains AI and software-development discussions. Domain shift may
therefore remain when the classifier is applied to developer posts.

## Interpretation

Emotion predictions are used as an intermediate analytical layer:

Sentiment
->
Emotions
->
Stress
->
Stress-associated Topics

Emotion predictions must not be interpreted as psychological diagnoses.

## Final decision

The GoEmotions component is considered methodologically stable.

The final prediction strategy is:

TF-IDF
+
One-vs-Rest Logistic Regression
+
28 validation-calibrated emotion thresholds.

The next stage is to apply this model to the 2,666 analysis-ready
developer posts.
