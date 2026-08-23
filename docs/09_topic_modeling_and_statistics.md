# 09 — Topic Modeling and Statistical Analysis

## Objective

Identify recurring discussion contexts independently of Stress prediction, then test whether topic membership is associated with model-predicted Stress.

## Why topics are learned on all posts

NMF is fitted on **all 2,666 posts**, not only on the 248 predicted-Stress posts.

This prevents Stress-classification errors from defining the topic structure.

## Method

Text representation:
- TF-IDF
- unigrams + bigrams
- document-frequency filtering
- domain stopwords
- product names retained when informative

Topic model:
- Non-negative Matrix Factorization (NMF)
- candidate k values: **4, 5, 6, 7, 8, 9, 10**

Selection criteria:
- reconstruction error;
- topic diversity;
- mean dominant-topic weight;
- topic margin;
- topic sizes;
- qualitative interpretability.

## Final decision

**7 topics** were retained.

Quality indicators:
- Topic diversity: **0.914**
- Mean dominant-topic weight: **≈0.709**
- Mean topic margin: **≈0.502**
- Smallest topic: **87 posts**
- Largest topic: **1,037 posts**

At k≥8, very small narrow clusters started to appear.

## Final topics

| Topic | Posts | Predicted Stress | Rate |
|---|---:|---:|---:|
| General AI Development & Agent Workflows | 1,037 | 91 | 8.78% |
| Gemini / Google Ecosystem | 349 | 32 | 9.17% |
| Claude Coding & Development Tools | 279 | 27 | 9.68% |
| VS Code / GitHub Copilot Technical Issues | 87 | 3 | **3.45%** |
| Local LLM Deployment & Inference Performance | 296 | 33 | 11.15% |
| AI Labs / Industry News | 271 | 10 | 3.69% |
| ChatGPT / OpenAI User Experience | 347 | 52 | **14.99%** |

Overall predicted-Stress rate: **9.30%**.

## Overall association test

Chi-square:

```text
χ²(6) = 28.53
p ≈ 0.000075
```

Effect size:

```text
Cramér's V = 0.103
```

Conclusion: statistically significant association, but **small overall effect**.

## Per-topic tests

For each topic:
- Fisher's Exact Test
- Relative Risk (RR)
- Odds Ratio (OR)
- Benjamini-Hochberg FDR correction

FDR-significant signals:

### ChatGPT / OpenAI User Experience
- 14.99%
- RR = 1.77
- OR = 1.91
- FDR p ≈ 0.0013

### AI Labs / Industry News
- 3.69%
- RR = 0.37
- OR = 0.35
- FDR p ≈ 0.0013

Note: **VS Code / GitHub Copilot has the lowest raw rate (3.45%)**, but AI Labs / Industry News is the robust statistically supported lower association after FDR correction.

## Main outputs

```text
data/results/stress_topics/final_topic_assignments.csv
data/results/stress_topics/final_topic_keywords.csv
data/results/stress_topics/topic_count_evaluation.csv
data/results/stress_topics/topic_stress_statistical_summary.csv
data/results/final_synthesis/topic_stress_final.csv
```

## Interpretation boundary

Topic association does not imply causality. Because the statistical outcome is model-predicted Stress, uncertainty in the Stress classifier propagates into Topic × Stress analysis.
