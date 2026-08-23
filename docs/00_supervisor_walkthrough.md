# Supervisor Walkthrough — Final NLP Project

**Project:** Summer Internship 2026  
**Purpose:** Explain the complete final workflow in plain language: what was done, why it was done, how it was evaluated, what was retained, and what the results mean.

## 1. Research question

> **How do developer-oriented public discussions about AI tools express sentiment, emotions, and stress-related language, and which discussion topics are associated with model-predicted Stress?**

The study analyses public technical discourse. It does not diagnose individuals and does not assume that every author is a verified professional developer.

## 2. Final pipeline

```text
1. Problem definition
        ↓
2. Data collection — 5,406 records
        ↓
3. Cleaning, relevance filtering and deduplication
        ↓
4. Frozen corpus — 2,666 posts
        ↓
5. Exploratory data analysis
        ↓
6. Sentiment — VADER + Transformer
        ↓
7. Emotions — GoEmotions multi-label classifier
        ↓
8. Stress — Dreaddit + hybrid augmentation + domain validation
        ↓
9. Integrated Sentiment × Emotions × Stress analysis
        ↓
10. NMF topic modeling on all 2,666 posts
        ↓
11. Topic × predicted-Stress statistical tests
        ↓
12. Evidence Explorer + final conclusions
        ↓
13. Streamlit dashboard + final report
```

## 3. Step-by-step summary

| Step | Objective | Method | Main output / result |
|---|---|---|---|
| Problem definition | Define scope and scientific boundaries | Research framing | `docs/01_problem_definition.md` |
| Collection | Build a multi-source public corpus | Public APIs + documented datasets | 5,406 records |
| Cleaning | Keep comparable AI-relevant records | Standardization, relevance, deduplication, minimum text length | 2,666 posts |
| EDA | Understand corpus composition | Descriptive statistics and charts | `data/results/eda/` |
| Sentiment | Measure broad polarity | VADER + Transformer | 37.62% model agreement |
| Emotions | Add specific affective signals | GoEmotions, TF-IDF, OVR Logistic Regression | Micro-F1 0.5213 |
| Stress | Detect Stress-related language separately | Dreaddit + hybrid augmentation | Final developer-domain recall 0.750, precision 0.111 |
| Integration | Compare signals on the same posts | One-to-one merge by `record_id` | Negative sentiment ≠ predicted Stress |
| Topics | Discover discussion context | TF-IDF + NMF, k=4…10 evaluated | 7 topics selected |
| Statistics | Test Topic × predicted-Stress association | χ², Fisher, BH-FDR, RR, OR, Cramér's V | Significant but small overall effect |
| Dashboard | Present final frozen results | Streamlit + Plotly | Public deployed application |

## 4. Headline findings

### Sentiment
- VADER: **63.80% Positive**
- Transformer: **64.03% Neutral**
- Agreement: **37.62%**
- On the LLM-assisted reference sample, Transformer Macro-F1 (**0.510**) was slightly higher than VADER (**0.454**).

**Decision:** retain both methods because their disagreement is itself an important methodological finding.

### Emotions
Final model:

```text
GoEmotions simplified (28 labels)
→ TF-IDF
→ One-vs-Rest Logistic Regression
→ per-emotion thresholds calibrated on validation data
```

Official test:
- Micro-F1: **0.5213**
- Macro-F1: **0.4438**

### Stress
Final training:

```text
2,838 Dreaddit training posts
+ 1,200 synthetic developer-style examples
= 4,038 training examples
```

Official Dreaddit test:
- Stress F1: **0.7533**
- Stress recall: **0.7778**

Developer-domain reference evaluation:
- 596 evaluable posts
- 588 No stress
- 8 Stress
- TP=6, FP=48, FN=2, TN=540
- Stress precision: **0.1111**
- Stress recall: **0.7500**
- MCC: **0.2679**

**Interpretation:** useful Stress-related signal remains, but technical frustration produces many false positives. Predictions must not be treated as diagnoses.

Final corpus:
- **248 / 2,666 = 9.30% model-predicted Stress**

### Integrated analysis
- VADER-negative → predicted Stress: **81/565 = 14.34%**
- Transformer-negative → predicted Stress: **117/482 = 24.27%**

**Central conclusion:** negative sentiment and Stress are not interchangeable.

### Topic modeling and statistics
NMF topics were learned **independently on all 2,666 posts**, then Stress predictions were analysed by topic.

Seven topics were retained after evaluating k=4…10 using reconstruction, diversity, dominant-topic strength, margin, cluster sizes and interpretability.

Overall association:
- χ²(6)=**28.53**
- p≈**0.000075**
- Cramér's V=**0.103** → small effect

FDR-significant topic signals:
- **ChatGPT / OpenAI User Experience:** 14.99%, RR=1.77, OR=1.91
- **AI Labs / Industry News:** 3.69%, RR=0.37, OR=0.35

The absolute lowest raw predicted-Stress rate is **VS Code / GitHub Copilot Technical Issues: 3.45%**, but its individual lower association does not survive FDR correction.

## 5. Final dashboard

Live:
https://summer-internship-dashboard.streamlit.app/

Pages:
1. Executive Overview
2. Data & Cleaning
3. Sentiment Analysis
4. Emotion Analysis
5. Stress & Critical Validation
6. Integrated Analysis
7. Topic Modeling & Statistics
8. Evidence Explorer
9. Final Conclusions

## 6. Scientific boundaries to defend

- Predicted Stress is not a clinical diagnosis.
- 9.30% is not Stress prevalence among developers.
- LLM-assisted annotations are reference annotations, not human clinical ground truth.
- Synthetic data are used only for training augmentation.
- Emotion enrichments are descriptive unless individually tested.
- Topic association does not imply causality.
- Sampling and source imbalance limit population-level generalization.
