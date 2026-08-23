# Final Internship Report — Current Architecture

## Final project title

**NLP Analysis of Developer-Oriented Discussions on AI Tools: Sentiment, Emotions, Stress-Related Language, and Topic Modeling**

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
- research question;
- objectives;
- scientific boundaries;
- final pipeline.

## Chapter 1 — Host Organization

Keep institutional presentation, internship context and organization information.

## Chapter 2 — Project Context and Methodological Design

- Project context
- Problem statement
- Objectives
- Scope and constraints
- Requirements
- Technologies
- Final methodological architecture

## Chapter 3 — Implementation, Results and Critical Analysis

Follow the same architecture as the dashboard.

### 3.1 Executive Overview
Pipeline, corpus counts and headline findings.

### 3.2 Data & Cleaning
Collection provenance, preprocessing, frozen 2,666-post corpus and representativeness.

### 3.3 Sentiment Analysis
VADER, Transformer, distributions, agreement, LLM-assisted reference evaluation, critical decision.

### 3.4 Emotion Analysis
GoEmotions, TF-IDF + OVR LR, threshold calibration, test metrics, final corpus emotions, limitations.

### 3.5 Stress & Critical Validation
Dreaddit, candidate models, baseline/hybrid, developer-domain reference set, confusion matrix, error analysis, final 9.30% prediction rate.

### 3.6 Integrated Analysis
Sentiment × Emotions × Stress; Negative ≠ Stress; descriptive emotion enrichment.

### 3.7 Topic Modeling & Statistics
NMF k=4…10, selection of 7 topics, final topic rates, χ², Cramér's V, Fisher, BH-FDR, RR and OR.

### 3.8 Evidence Explorer / Qualitative Evidence
Representative row-level cases and model errors.

### 3.9 Dashboard
Live URL and selected screenshots:
- Executive Overview
- Stress & Critical Validation
- Topic Modeling & Statistics
- Evidence Explorer
- Final Conclusions

### 3.10 Final Synthesis, Limitations and Perspectives
What can be claimed, what cannot be claimed, future work.

## General Conclusion

Summarize the complete final analytical chain and the key methodological contribution: sentiment, emotions, Stress and topics represent distinct but complementary analytical layers.
