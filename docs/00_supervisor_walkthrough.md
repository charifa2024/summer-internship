# Supervisor Walkthrough — Final NLP Project

**Project:** Summer Internship 2026  
**Purpose:** Explain the complete final workflow in plain language: what was done, why it was done, how it was evaluated, what was retained, and what the results mean.

## 1. Final research question

> **How do developer-oriented public discussions about AI tools express sentiment, emotions, and stress-related language, and which discussion topics are associated with model-predicted Stress?**

The study analyses public technical discourse. It does not diagnose individuals and does not assume that every author is a verified professional developer.

## 2. Final pipeline

```text
5,406 collected records
        ↓
Strict AI relevance: −2,680
        ↓
2,726 relevant posts
        ↓
Exact-text deduplication: −32
        ↓
2,694 unique relevant posts
        ↓
Minimum length ≥ 5 words: −28
        ↓
2,666 frozen analysis-ready posts
        ↓
Sentiment — VADER + Transformer
        ↓
Emotions — GoEmotions multi-label classifier
        ↓
Stress — Dreaddit + synthetic augmentation + domain validation
        ↓
Integrated Sentiment × Emotions × Stress analysis
        ↓
NMF topic modeling on all 2,666 posts
        ↓
Topic × model-predicted Stress statistics
        ↓
Evidence Explorer + final conclusions
        ↓
Streamlit dashboard + final report
```

**Audit note:** 37 duplicate rows were detected globally, but only 32 were removed at the sequential deduplication step because five had already been excluded by relevance filtering.

## 3. Step-by-step summary

| Step | Objective | Method | Main output / result |
|---|---|---|---|
| Problem definition | Define scope and scientific boundaries | Research framing | `docs/01_problem_definition.md` |
| Collection | Build a multi-source public corpus | Public APIs + documented datasets | 5,406 records |
| Cleaning | Keep comparable AI-relevant records | Strict relevance → deduplication → minimum length | 2,666 posts |
| EDA | Understand corpus composition | Descriptive statistics and charts | `data/results/eda/` |
| Sentiment | Measure broad polarity | VADER + Transformer | 37.62% agreement; model-dependent result |
| Emotions | Add specific affective signals | GoEmotions, TF-IDF, OVR Logistic Regression | Micro-F1 0.5213 |
| Stress | Detect Stress-related language separately | Dreaddit + synthetic augmentation | Recall 0.750, precision 0.111 on developer-domain reference |
| Integration | Compare signals on the same posts | One-to-one merge by `record_id` | Negative sentiment ≠ model-predicted Stress |
| Topics | Discover discussion context | TF-IDF + NMF, k=4…10 evaluated | 7 topics selected |
| Statistics | Test Topic × model-predicted Stress association | χ², Fisher, BH-FDR, RR, OR, Cramér's V | Significant but small overall effect |
| Dashboard | Present final frozen results | Streamlit + Plotly | Public deployed application |

## 4. Headline findings

### Sentiment
- VADER: **63.80% Positive**
- Transformer: **64.03% Neutral**
- Agreement: **37.62%**
- LLM-assisted reference Macro-F1: VADER **0.454**, Transformer **0.510**

**Decision:** retain both methods. Their disagreement is a methodological finding, and neither method is treated as ground truth.

### Emotions
Final model:

```text
GoEmotions simplified (28 labels)
→ TF-IDF
→ One-vs-Rest Logistic Regression
→ per-emotion validation-calibrated thresholds
```

Official test:
- Micro-F1: **0.5213**
- Macro-F1: **0.4438**

Interpret the outputs as **model-predicted emotion-related language**, not verified psychological states.

### Stress
Final training:

```text
2,838 Dreaddit training posts
+ 1,200 synthetic developer-style examples
= 4,038 training examples
```

Synthetic examples are training augmentation only.

Official Dreaddit test:
- Stress F1: **0.7533**
- Stress recall: **0.7778**

Developer-domain reference:
- 596 evaluable posts
- 588 No stress
- only 8 Stress
- TP=6, FP=48, FN=2, TN=540
- Stress precision: **0.1111**
- Stress recall: **0.7500**
- MCC: **0.2679**

**Interpretation:** the hybrid model improves several metrics relative to the Dreaddit-only baseline, but precision remains very low and the positive reference class is extremely small. The observed gains do **not** demonstrate robust target-domain transfer.

Final corpus:
- **248 / 2,666 = 9.30% model-predicted Stress**

This is an exploratory model-prediction rate, not psychological prevalence.

### Integrated analysis
- VADER-negative → model-predicted Stress: **81/565 = 14.34%**
- Transformer-negative → model-predicted Stress: **117/482 = 24.27%**

**Central conclusion:** negative sentiment and model-predicted Stress are not interchangeable.

### Topic modeling and statistics
NMF topics were learned **independently on all 2,666 posts**, then model-predicted Stress was analysed by topic.

Overall association:
- χ²(6)=**28.53**
- p≈**0.000075**
- Cramér's V=**0.103** → **small effect**

FDR-supported topic signals:
- **ChatGPT / OpenAI User Experience:** 14.99%, RR=1.77, OR=1.91
- **AI Labs / Industry News:** 3.69%, RR=0.37, OR=0.35

The lowest raw rate is **VS Code / GitHub Copilot Technical Issues: 3.45%**, but its individual lower association does not survive FDR correction.

## 5. Final dashboard

Live: https://summer-internship-dashboard.streamlit.app/

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

- Results describe **developer-oriented public technical discussions**, not a verified population of developers.
- Sentiment is model-dependent.
- Emotion labels are predictions, not verified emotional states.
- Model-predicted Stress is not a diagnosis.
- 9.30% is not Stress prevalence among developers.
- The developer-domain Stress reference contains only eight evaluable positive cases.
- LLM-assisted annotations are reference annotations, not independent human or clinical ground truth.
- Synthetic data are used only for training augmentation; the gain is modest and does not prove robust transfer.
- Emotion enrichments are descriptive unless individually tested.
- Topic × model-predicted Stress is statistically significant but the overall effect is small.
- Topic association does not imply causality.
- Sampling and source imbalance limit population-level generalization.
