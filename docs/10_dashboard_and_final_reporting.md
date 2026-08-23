# 10 — Dashboard and Final Reporting

## Objective

Present the frozen final analytical outputs in one transparent interface and align the report with the same methodological architecture.

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
Research question, headline KPIs, final pipeline and main findings.

### 1 — Data & Cleaning
Raw-source provenance, final corpus composition, cleaning logic and representativeness limitations.

### 2 — Sentiment Analysis
VADER vs Transformer distributions, agreement, LLM-assisted reference evaluation and methodological decision.

### 3 — Emotion Analysis
GoEmotions model, threshold comparison, official test metrics, final emotion prevalence and domain-shift limitation.

### 4 — Stress & Critical Validation
Candidate model selection, baseline vs hybrid comparison, developer-domain reference evaluation, confusion matrix, error analysis and final 9.30% prediction rate.

### 5 — Integrated Analysis
Sentiment × Emotions × Stress relationships, including the central finding Negative ≠ Stress.

### 6 — Topic Modeling & Statistics
Seven-topic NMF result, topic-count decision evidence, Topic × predicted-Stress tests and causal boundary.

### 7 — Evidence Explorer
Row-level filtering and representative qualitative evidence.

### 8 — Final Conclusions
Claims, non-claims, limitations and recommended next methodological steps.

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

The dashboard is a visualization and evidence layer. It **loads frozen final results**; it is not intended to recompute the complete research experiment every time the app is opened.

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

Recommended screenshots:
- Executive Overview
- Stress & Critical Validation
- Topic Modeling & Statistics
- Evidence Explorer
- Final Conclusions

## Scientific wording

Use:
- model-predicted Stress
- Stress-related language
- LLM-assisted reference annotation
- descriptive emotion enrichment
- statistically significant but small association

Avoid:
- Stress prevalence
- psychological diagnosis
- human gold standard for the current LLM-assisted reference set
- causal claims about topics
