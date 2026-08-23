# 03 — Data Cleaning and Preprocessing

## Objective

Transform the frozen 5,406-record collection into one stable, comparable analysis-ready corpus.

## Implementation

Main preprocessing script:

```text
src/preprocessing/prepare_analysis_dataset.py
```

Input:

```text
data/raw/combined_reddit_posts.jsonl
```

Final output:

```text
data/processed/analysis_ready_posts.jsonl
```

## Processing sequence

```text
Raw records
  ↓
schema harmonization
  ↓
text construction
  ↓
platform/date standardization
  ↓
strict AI relevance
  ↓
exact-text deduplication
  ↓
minimum text-length rule
  ↓
analysis eligibility
  ↓
frozen 2,666-post corpus
```

## Text fields

### `full_text_raw`
Preserves the combined original title/body evidence.

### `text_clean_basic`
Basic cleaned representation that preserves useful punctuation and negation. Used for contextual inspection and prediction tasks.

### `text_clean_lexical`
Lowercased lexical representation for TF-IDF, NMF and frequency analysis.

## Identifier strategy

A source-qualified ID is created so results can be merged safely:

```text
record_id = source + "::" + source_id
```

If a source identifier is unavailable, a stable fallback is generated.

## Relevance and eligibility

The final analysis-eligibility logic requires:
- strict AI relevance;
- no exact-text duplicate;
- usable text;
- at least five words.

The preprocessing audit contains:
- **5,406 raw records**
- **2,726 strict AI-relevant records**
- **37 exact-text duplicates detected**
- **2,666 final analysis-ready records**

Some exclusion indicators can overlap; they should not be summed as mutually exclusive causes.

## Why freeze the corpus?

Every final analytical branch uses the same record set:

```text
2,666 posts
├── Sentiment
├── Emotions
├── Stress
└── Topic modeling
```

This makes one-to-one integration by `record_id` possible and avoids comparing incompatible subsets.

## Decision

`data/processed/analysis_ready_posts.jsonl` is the official frozen corpus for downstream analysis.
