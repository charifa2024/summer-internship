# Dashboard

## Main file

`dashboard/app.py`

## Installation

From the project root:

```bash
pip install -r dashboard/requirements.txt
```

## Launch

```bash
streamlit run dashboard/app.py
```

## Input priority

The dashboard automatically loads the most advanced available file:

1. `data/results/topics/topic_assignments.csv`
2. `data/results/emotions/stress_emotion_predictions.csv`
3. `data/results/sentiment/sentiment_predictions.csv`
4. `data/results/sentiment/vader_predictions.csv`
5. `data/processed/analysis_ready_posts.jsonl`

## Dashboard sections

- Overview
- Sentiment
- Stress and emotions
- Topics
- Time trends
- Filtered data explorer

## Filters

- Platform
- Sentiment
- Stress direction
- Topic
- AI tool
- Date range

## Methodological note

The dashboard visualizes model- and rule-generated analytical outputs.

It does not provide clinical diagnoses.

LLM-assisted validation labels are provisional references and must not be
presented as independent human ground truth.
