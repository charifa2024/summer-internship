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
5,406 raw collected records
        ↓
Schema harmonization + text construction
        ↓
Strict AI-relevance filtering
− 2,680
        ↓
2,726 AI-relevant posts
        ↓
Exact-text deduplication
− 32
        ↓
2,694 unique AI-relevant posts
        ↓
Minimum-length rule (≥ 5 words)
− 28
        ↓
2,666 final analysis-ready posts
```

Missing dates are retained and are **not** used as an eligibility criterion.

## Verified sequential audit

| Sequential stage | Remaining records | Removed at this stage |
|---|---:|---:|
| Raw collection | 5,406 | — |
| After strict AI-relevance filtering | 2,726 | 2,680 |
| After exact-text deduplication | 2,694 | 32 |
| After minimum-length rule (≥5 words) | 2,666 | 28 |
| **Total sequential exclusions** | — | **2,740** |

The broader raw diagnostic audit detects **37 exact-text duplicate rows**. These diagnostic indicators overlap: five duplicate rows were already removed by the preceding relevance filter, so only **32** are removed during the sequential deduplication step. The sequential counts above are therefore the correct exclusion funnel.

## Text fields

### `full_text_raw`
Preserves the combined original title/body evidence.

### `text_clean_basic`
Basic cleaned representation that preserves useful punctuation, case and negation for contextual inspection and prediction tasks.

### `text_clean_lexical`
Lowercased lexical representation for TF-IDF, NMF and frequency analysis.

## Identifier strategy

A source-qualified ID is created so results can be merged safely:

```text
record_id = source + "::" + source_id
```

If a source identifier is unavailable, a stable SHA-256-based fallback identifier is generated.

## Final eligibility rule

A record is analysis-ready when it:
- satisfies strict AI relevance;
- is not an exact-text duplicate among the remaining relevant records;
- contains at least five words.

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
