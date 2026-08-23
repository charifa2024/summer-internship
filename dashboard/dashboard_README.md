# Final Streamlit Dashboard

## Live application

https://summer-internship-dashboard.streamlit.app/

## Main file

```text
dashboard/app_final.py
```

## Local launch

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app_final.py
```

## Final dashboard architecture

1. **Executive Overview**
2. **Data & Cleaning**
3. **Sentiment Analysis**
4. **Emotion Analysis**
5. **Stress & Critical Validation**
6. **Integrated Analysis**
7. **Topic Modeling & Statistics**
8. **Evidence Explorer**
9. **Final Conclusions**

## Main analytical inputs

The dashboard primarily loads frozen result artifacts rather than rerunning the research experiments.

```text
data/results/final_synthesis/
data/results/sentiment/
data/results/emotions/
data/results/combined/
data/results/stress_topics/
data/processed/analysis_ready_posts.jsonl
```

Important examples:

```text
data/results/final_synthesis/final_findings.json
data/results/final_synthesis/sentiment_distribution.csv
data/results/final_synthesis/negative_sentiment_vs_stress.csv
data/results/final_synthesis/emotion_stress_association.csv
data/results/final_synthesis/topic_stress_final.csv

data/results/combined/sentiment_emotion_stress_predictions.csv

data/results/stress_topics/final_topic_assignments.csv
data/results/stress_topics/final_topic_keywords.csv
data/results/stress_topics/topic_count_evaluation.csv
data/results/stress_topics/topic_stress_statistical_summary.csv
```

## Interpretation rule

The dashboard reports **model-predicted Stress-related language**, not psychological diagnosis or prevalence. Topic statistics use predicted Stress as the outcome and therefore inherit uncertainty from the Stress classifier.
