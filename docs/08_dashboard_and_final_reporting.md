# 08 — Dashboard and Final Reporting

## Objective

This phase converts the analytical outputs into an interactive dashboard and
a structured final-report workflow.

## Dashboard

Main file:

`dashboard/app.py`

Launch command:

```bash
streamlit run dashboard/app.py
```

The dashboard automatically loads the most advanced available output file.

### Input Priority

1. `data/results/topics/topic_assignments.csv`
2. `data/results/emotions/stress_emotion_predictions.csv`
3. `data/results/sentiment/sentiment_predictions.csv`
4. `data/results/sentiment/vader_predictions.csv`
5. `data/processed/analysis_ready_posts.jsonl`

## Dashboard Components

### Overview

- Number of filtered records
- Number of platforms
- Share of explicit stress signals
- Number of discovered topics
- Records by platform
- Most-mentioned AI tools
- Predefined research themes

### Sentiment

- Positive, neutral, and negative distribution
- Sentiment share
- Sentiment by platform

### Stress and Emotions

- Increased stress or distress
- Reduced stress or relief
- Mixed or ambiguous
- No clear stress signal
- Primary emotions
- Likely stress and emotion contexts
- Sentiment compared with stress direction

### Topics

- Topic distribution
- Stress direction within topics
- Sentiment within topics
- Topic keywords

### Time Trends

- Number of records over time
- Stress direction over time
- Topics over time

### Data Explorer

- Interactive filters
- Text search
- Export of filtered records

## Methodological Wording

Safe wording:

> The dashboard visualizes textual patterns detected in public developer
> discussions about AI tools.

Unsafe wording:

> The dashboard diagnoses developer mental health.

## Final Report Structure

1. Executive summary
2. Project context and problem definition
3. Research questions
4. Data sources and collection
5. Data quality and preprocessing
6. Exploratory data analysis
7. Sentiment analysis
8. LLM-assisted provisional comparison
9. Stress and emotion analysis
10. Topic modeling
11. Dashboard
12. Main findings
13. Limitations
14. Recommendations
15. Conclusion
16. Appendices

## Main Limitations to Preserve

- Public posts do not represent all developers.
- Sentiment predictions are model outputs.
- LLM-assisted annotations are not independent human ground truth.
- Stress analysis detects language, not clinical states.
- Rule-based emotion detection can miss implicit expressions.
- Topic names require interpretation.
- Topic modeling identifies patterns rather than causal relationships.

## Completion Checklist

- Run all analytical notebooks successfully.
- Confirm the final CSV files exist.
- Launch and test the Streamlit dashboard.
- Review automatic topic names.
- Capture dashboard screenshots for the report.
- Complete the final report using actual generated results.
- Keep methodological limitations visible.
