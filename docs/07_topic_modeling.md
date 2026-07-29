# 07 — Topic Modeling

## Objective

This phase identifies recurring subjects in developer discussions about AI
tools.

It complements:

- sentiment analysis, which identifies polarity;
- stress analysis, which identifies emotional signals;
- topic modeling, which identifies the discussion subject or context.

## Notebook

`notebooks/07_topic_modeling.ipynb`

## Preferred Input

`data/results/emotions/stress_emotion_predictions.jsonl`

Fallback inputs:

1. `data/results/sentiment/sentiment_predictions.jsonl`
2. `data/processed/analysis_ready_posts.jsonl`

## Text Field

`text_clean_lexical`

This field is used because it is prepared for lexical feature extraction.

## Method

### TF-IDF

The notebook uses TF-IDF with:

- unigrams;
- bigrams;
- English stopwords;
- domain-specific stopwords;
- document-frequency filtering;
- a maximum vocabulary size.

### NMF

Non-negative Matrix Factorization discovers latent topics from the TF-IDF
matrix.

The notebook tests several topic counts and compares:

- reconstruction error;
- topic diversity;
- a combined quality score.

The final selection is automatic unless a topic count is manually forced.

## Topic Interpretation

For every topic, the notebook extracts:

- top keywords;
- provisional topic name;
- representative documents;
- dominant platform;
- dominant sentiment;
- dominant stress direction;
- mean topic confidence.

Automatic topic names are provisional.

They should be replaced with human-readable names after reviewing keywords and
representative texts.

## Main Output Columns

- `topic_id`
- `topic_name`
- `topic_weight`
- `topic_confidence`
- `topic_low_confidence`

## Main Outputs

`data/results/topics/topic_assignments.jsonl`

`data/results/topics/topic_assignments.csv`

Additional outputs include:

- topic-count evaluation;
- topic keywords;
- provisional topic names;
- representative documents;
- topic summary;
- topics by platform;
- topics by AI tool;
- sentiment by topic;
- stress by topic;
- emotion by topic;
- topics over time;
- low-confidence assignments.

## Figures

Figures are saved in:

`figures/topics/`

They include:

- reconstruction error by topic count;
- topic diversity by topic count;
- combined topic-count score;
- topic distribution;
- topics by platform;
- topics by AI tool;
- sentiment by topic;
- stress by topic;
- topics over time.

## Interpretation

Safe wording:

> NMF discovered recurring lexical patterns that were interpreted as
> provisional topics.

Unsafe wording:

> The model objectively identified the only true topics.

## Limitations

- Topic-count selection is approximate.
- Results depend on TF-IDF settings.
- One dominant topic is assigned per document.
- Automatic names are provisional.
- Short technical texts may be ambiguous.
- Topic modeling identifies patterns, not causes.
- The sample does not represent all developers.

## Completion

The phase is complete after:

- running the notebook;
- reviewing keywords and representative texts;
- interpreting topic names;
- checking low-confidence assignments;
- confirming all exported files;
- documenting limitations.

The next phase is:

**08 — Dashboard and Final Reporting**
