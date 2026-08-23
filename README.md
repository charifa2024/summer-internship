# NLP Analysis of Developer-Oriented Discussions on AI Tools

**Final internship project — Summer 2026**

This repository contains a reproducible NLP workflow for analysing public developer-oriented discussions about AI tools. The final study separates four analytical questions:

1. **Sentiment** — broad polarity using VADER and a contextual Transformer.
2. **Emotions** — multi-label emotion signals using GoEmotions.
3. **Stress-related language** — a dedicated Dreaddit-based classifier with developer-domain critical validation.
4. **Topics** — NMF discussion themes learned independently on the complete analysis-ready corpus, followed by statistical association with model-predicted Stress.

The project analyses **public technical discourse**. It does not diagnose individuals and does not claim that the corpus is representative of all software developers.

## Live links

- **Interactive dashboard:** https://summer-internship-dashboard.streamlit.app/
- **Repository:** https://github.com/charifa2024/summer-internship

## Final analytical pipeline

```text
5,406 collected public records
        ↓
Cleaning + relevance filtering + deduplication
        ↓
2,666 analysis-ready posts
        ↓
 ┌──────────────────┬──────────────────┬────────────────────┐
 │                  │                  │                    │
Sentiment          Emotions           Stress
VADER +            GoEmotions         Dreaddit baseline
Transformer        TF-IDF + OVR LR    + synthetic augmentation
 │                  │                  │
 └──────────────────┴──────────────────┴────────────────────┘
                         ↓
              Unified row-level dataset
                         ↓
          Sentiment × Emotions × Stress
                         ↓
            NMF topics on all 2,666 posts
                         ↓
          Topic × predicted-Stress statistics
                         ↓
          Streamlit dashboard + final report
```

## Core verified results

| Component | Final result |
|---|---|
| Raw collection | 5,406 records |
| Analysis-ready corpus | 2,666 posts |
| VADER | 63.80% Positive |
| Transformer sentiment | 64.03% Neutral |
| VADER–Transformer agreement | 37.62% |
| GoEmotions test Micro-F1 | 0.5213 |
| GoEmotions test Macro-F1 | 0.4438 |
| Hybrid Stress official-test F1 | 0.7533 |
| Developer-domain Stress precision | 0.1111 |
| Developer-domain Stress recall | 0.7500 |
| Model-predicted Stress on final corpus | 248 / 2,666 = 9.30% |
| Final NMF solution | 7 topics |
| Topic × predicted-Stress | χ²(6)=28.53, p≈0.000075 |
| Effect size | Cramér's V=0.103 (small) |

**Important:** 9.30% is a **model prediction rate**, not psychological-Stress prevalence.

## Final dashboard pages

1. Executive Overview
2. Data & Cleaning
3. Sentiment Analysis
4. Emotion Analysis
5. Stress & Critical Validation
6. Integrated Analysis
7. Topic Modeling & Statistics
8. Evidence Explorer
9. Final Conclusions

## Key project paths

```text
dashboard/app_final.py
data/processed/analysis_ready_posts.jsonl
data/results/final_synthesis/
data/results/sentiment/
data/results/emotions/
data/results/combined/
data/results/stress_topics/
docs/
```

## Run the dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app_final.py
```

## Documentation

Start with:

- `docs/00_supervisor_walkthrough.md`
- `docs/01_problem_definition.md`
- `docs/02_data_collection.md`
- `docs/03_data_cleaning_and_preprocessing.md`
- `docs/04_exploratory_data_analysis.md`
- `docs/05_sentiment_analysis.md`
- `docs/06_emotion_analysis.md`
- `docs/07_stress_critical_validation.md`
- `docs/08_integrated_analysis.md`
- `docs/09_topic_modeling_and_statistics.md`
- `docs/10_dashboard_and_final_reporting.md`

Final methodological decision documents are also stored in `docs/`.

## Scientific boundaries

- Public posts are not a random sample of all developers.
- Sentiment and emotion labels are computational predictions.
- The sentiment reference sample is LLM-assisted, not independent human ground truth.
- The developer-domain Stress reference is LLM-assisted and contains only eight evaluable Stress-positive examples.
- Synthetic Stress examples are training augmentation only.
- Emotion enrichments are descriptive unless individually tested.
- Topic association with predicted Stress does not imply causality.
