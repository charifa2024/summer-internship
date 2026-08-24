# 10 — Dashboard and Final Reporting

## Objective

Present the frozen final analytical outputs in one transparent interface and align the report with the same methodological architecture and scientific boundaries.

## Live dashboard

https://summer-internship-dashboard.streamlit.app/

## Main file

```text
dashboard/app_final.py
```

## Local launch

```bash
streamlit run dashboard/app_final.py
```

## Final pages

### 0 — Executive Overview
Final research question, headline KPIs, final pipeline and main findings.

### 1 — Data & Cleaning
Raw-source provenance, final corpus composition, verified sequential cleaning funnel and representativeness limitations.

Verified funnel:

```text
5,406
→ 2,726 after strict AI relevance
→ 2,694 after exact-text deduplication
→ 2,666 after minimum-length rule
```

### 2 — Sentiment Analysis
VADER vs Transformer distributions, 37.62% agreement, LLM-assisted reference evaluation and model-dependent interpretation.

### 3 — Emotion Analysis
GoEmotions model, threshold comparison, official test metrics, final emotion prevalence and domain-shift limitation.

### 4 — Stress & Critical Validation
Candidate model selection, baseline vs hybrid comparison, developer-domain reference evaluation, confusion matrix, error analysis, precision 0.1111, only eight positive reference cases, and final 9.30% model-prediction rate.

### 5 — Integrated Analysis
Sentiment × Emotions × Stress relationships, including the central finding Negative ≠ model-predicted Stress.

### 6 — Topic Modeling & Statistics
Seven-topic NMF result, topic-count decision evidence, Topic × model-predicted Stress tests, small effect size and causal boundary.

### 7 — Evidence Explorer
Row-level filtering and representative qualitative evidence.

### 8 — Final Conclusions
Supported claims, non-claims, limitations and recommended next methodological steps.

## Final synthesis files

```text
data/results/final_synthesis/core_metrics.csv
data/results/final_synthesis/emotion_stress_association.csv
data/results/final_synthesis/final_findings.json
data/results/final_synthesis/negative_sentiment_vs_stress.csv
data/results/final_synthesis/sentiment_distribution.csv
data/results/final_synthesis/topic_stress_final.csv
```

## Reporting principle

The dashboard is a visualization and evidence layer. It **loads frozen final results** and does not retrain the complete research experiment when the app is opened.

## Report alignment

The final report follows the same sequence:

```text
Executive Overview
→ Data & Cleaning
→ Sentiment
→ Emotions
→ Stress & Critical Validation
→ Integrated Analysis
→ Topic Modeling & Statistics
→ Evidence Explorer / qualitative evidence
→ Final Conclusions
```

## Screenshot checklist for the report

Current report screenshots:
- Executive Overview
- Stress & Critical Validation
- Topic Modeling & Statistics
- Evidence Explorer
- Final Conclusions

The Data & Cleaning section uses the verified cleaning funnel directly in the report rather than relying on a dashboard screenshot.

## Scientific wording

Use:
- developer-oriented public/technical discussions
- final technical corpus
- model-predicted Stress
- Stress-related language
- model-predicted emotion labels / predicted emotion-related language
- LLM-assisted reference annotation
- descriptive emotion enrichment
- statistically significant but small association
- association, not causation

Avoid:
- “developers are mostly positive/neutral”
- Stress prevalence
- psychological diagnosis
- human gold standard for the current LLM-assisted reference sets
- successful/robust domain transfer
- causal claims about topics
