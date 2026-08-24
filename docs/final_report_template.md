# Final Internship Report — Current Architecture

## Final project title

**Analysis of Developer-Oriented Discussions on AI Tools**

**Subtitle:** *Sentiment, Emotions, Stress-Related Language, and Topic Modeling in Public Technical Discussions*

## Front matter

- Cover
- Acknowledgements
- Résumé
- Abstract
- List of figures
- List of tables
- List of abbreviations
- Table of contents

## General Introduction

Include:
- context;
- problem;
- final research question;
- objectives;
- scientific boundaries;
- final analytical pipeline.

## Chapter 1 — Presentation of the Host Organization

Keep institutional presentation, internship context and organization information.

## Chapter 2 — Presentation of the Internship Project

- Project context and research problem
- Objectives
- Functional and technical requirements
- Methodological design
- Frozen corpus principle
- Layered analytical strategy
- Validation strategy
- Tools and technical environment
- Agile project management
- General architecture of the final solution

## Chapter 3 — Project Implementation, Results, and Critical Analysis

Follow the same architecture as the final dashboard.

### 3.1 Executive Overview
Pipeline, corpus counts and headline findings.

### 3.2 Data & Cleaning
Collection provenance, verified sequential preprocessing flow **5,406 → 2,726 → 2,694 → 2,666**, frozen corpus and representativeness.

### 3.3 Sentiment Analysis
VADER, Transformer, distributions, 37.62% agreement, LLM-assisted reference evaluation and cautious model-dependent interpretation.

### 3.4 Emotion Analysis
GoEmotions, TF-IDF + OVR LR, threshold calibration, official test metrics, final corpus emotion predictions and limitations.

### 3.5 Stress & Critical Validation
Dreaddit, candidate models, synthetic augmentation, source-domain test, developer-domain reference set, confusion matrix, error analysis, precision 0.1111, only eight positive reference cases and final 9.30% model-prediction rate.

### 3.6 Integrated Analysis
Sentiment × Emotions × Stress; Negative ≠ model-predicted Stress; descriptive emotion enrichment.

### 3.7 Topic Modeling & Statistics
NMF k=4…10, selection of 7 topics, final model-predicted Stress rates, χ², Cramér's V, Fisher, BH-FDR, RR and OR. Emphasize the statistically significant but small overall effect.

### 3.8 Evidence Explorer
Representative row-level cases, filters and model-error inspection.

### 3.9 Final Conclusions
What can be claimed, what cannot be claimed, limitations, scientific boundaries and future methodological work.

## General Conclusion and Perspectives

Synthesize the complete final analytical chain without unnecessarily repeating all numerical results. The principal methodological contribution is that sentiment, predicted emotion-related language, model-predicted Stress and discussion topics are distinct but complementary analytical layers.

## Final reporting boundaries

- No population-level claims about all developers.
- No diagnosis or Stress-prevalence claim.
- Sentiment is model-dependent.
- Synthetic augmentation provides modest gains but does not demonstrate robust transfer.
- Developer-domain Stress precision is very limited and only eight positive reference cases are available.
- Topic × model-predicted Stress is significant but small.
- Association does not imply causality.
