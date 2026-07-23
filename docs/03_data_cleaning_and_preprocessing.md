# 03 — Data Cleaning and Text Preprocessing

## 1. Objective

The purpose of this phase is to transform the frozen raw dataset into a clean,
standardized, and analysis-ready dataset.

The cleaning process is implemented in:

`src/preprocessing/prepare_analysis_dataset.py`

The raw dataset is never overwritten.

Input:

`data/raw/combined_reddit_posts.jsonl`

Main output:

`data/processed/analysis_ready_posts.jsonl`

---

## 2. Main Cleaning Principles

The preprocessing pipeline follows these principles:

1. Preserve the original text.
2. Create separate cleaned text columns.
3. Standardize identifiers across sources.
4. Standardize platform and date fields.
5. Detect exact text duplicates.
6. Apply strict AI relevance rules.
7. Preserve existing source labels separately from future model predictions.
8. Exclude records that are not suitable for analysis.
9. Keep a summary of all cleaning decisions.

---

## 3. Loading the Raw Dataset

The raw file is stored in JSON Lines format.

Each line represents one record collected from one of the project sources.

The script reads every valid JSON line and converts the complete collection
into a pandas DataFrame.

Invalid JSON lines raise an explicit error with the corresponding line number.

---

## 4. Standardization of Text Fields

Different platforms use different text columns.

Depending on the source, the content may be stored in:

- `title`
- `selftext`
- `text`

The pipeline creates the following standardized columns:

### `title_raw`

Contains the original title when available.

### `body_raw`

Contains the original body text from `selftext` or `text`.

### `full_text_raw`

Combines the title and body into one complete text field.

The original source columns remain available and are not overwritten.

---

## 5. Creation of Cleaned Text Versions

Two cleaned text versions are created for different NLP tasks.

### `text_clean_basic`

This version:

- decodes HTML entities;
- removes URLs;
- removes HTML tags;
- removes repeated whitespace;
- preserves uppercase letters;
- preserves punctuation;
- preserves negation.

This version is suitable for:

- VADER sentiment analysis;
- Transformer-based sentiment analysis;
- manual inspection.

### `text_clean_lexical`

This version:

- applies the basic cleaning;
- converts text to lowercase;
- removes most non-alphanumeric symbols;
- normalizes whitespace.

This version is suitable for:

- TF-IDF;
- topic modeling;
- NMF;
- K-Means;
- lexical frequency analysis.

---

## 6. Identifier Standardization

The original datasets may use different identifier fields:

- `id`
- `post_id`
- `sample_id`

The pipeline creates one common field:

`source_id`

When the original identifier is missing, a stable fallback identifier is
generated from the cleaned text using SHA-256 hashing.

A source-qualified identifier is then created:

`record_id = source + "::" + source_id`

Examples:

- `github_issues::123456`
- `stackoverflow::123456`
- `arctic_shift::abc123`

This avoids collisions between records coming from different platforms.

---

## 7. Platform Standardization

The script creates one common platform column:

`platform`

Possible platform values include:

- Reddit
- GitHub Issues
- Stack Overflow
- Hacker News
- Other social dataset
- Unknown

The platform is inferred from the original `source` and `subreddit` fields.

---

## 8. Date Standardization

The source datasets may contain dates in different fields:

- `created_iso`
- `created_at`
- `source_date`
- `created_utc`

The pipeline combines them into:

`created_at_utc`

Two additional indicators are created:

### `date_known`

Indicates whether a usable date is available.

### `is_recent_2025_plus`

Indicates whether the record date is on or after 1 January 2025.

Records with missing dates are not automatically removed. They can still be
used for general sentiment or topic analysis, but not for temporal analysis.

---

## 9. AI Relevance Detection

The original collection scripts used broad keyword matching.

A major limitation was the use of the substring `ai`, which could incorrectly
match unrelated words such as:

- `again`
- `email`
- `maintain`
- `training`

The cleaning script corrects this problem using regular expressions and word
boundaries.

The pipeline detects references to:

- ChatGPT
- OpenAI
- Claude and Anthropic
- Gemini
- GitHub Copilot
- Cursor
- DeepSeek
- LLMs
- Generative AI
- Artificial Intelligence
- GPT
- Llama
- Mistral
- Qwen
- Codeium
- Tabnine
- Aider
- AI coding assistants
- AI agents

The following columns are created:

### `ai_tools`

Contains the list of detected AI tools or expressions.

### `is_ai_relevant`

Indicates whether at least one valid AI-related pattern was found.

---

## 10. Research Theme Detection

The pipeline also detects transparent keyword-based theme signals.

The following themes are included:

- `stress_anxiety`
- `burnout`
- `job_security`
- `productivity`
- `trust_quality`
- `privacy_security`

The following columns are created:

### `research_themes`

Contains the list of detected research themes.

### `has_emotion_or_risk_signal`

Indicates whether at least one research theme was detected.

These theme indicators are descriptive rules. They do not replace the later
sentiment-analysis or topic-modeling methods.

---

## 11. Text Length Measures

The script creates:

### `text_length_chars`

Number of characters in the lightly cleaned text.

### `word_count`

Number of words in the lightly cleaned text.

These indicators are used to identify records that are too short for reliable
analysis.

---

## 12. Exact Text Duplicate Detection

The pipeline converts the lightly cleaned text to lowercase and creates a
SHA-256 text hash.

Records with the same normalized text hash are considered exact duplicates.

The following field is created:

`is_exact_text_duplicate`

The first occurrence is kept and later exact copies are excluded from the
analysis-ready dataset.

Duplicate detection is based on text, not only on original IDs, because
different platforms may contain the same or copied content.

---

## 13. Analysis Eligibility Rules

A record is considered eligible for analysis when all of the following
conditions are satisfied:

1. The record is AI-related.
2. The record is not an exact text duplicate.
3. The record contains at least five words.

The final decision is stored in:

`analysis_eligible`

The rule can be summarized as:

```text
analysis_eligible =
    is_ai_relevant
    AND not is_exact_text_duplicate
    AND word_count >= 5
```

---

## 14. Output Datasets

The preprocessing pipeline produces the following files.

### `analysis_ready_posts.jsonl`

Contains the records that satisfy the analysis eligibility rules.

This is the official dataset for:

- EDA;
- sentiment analysis;
- topic modeling;
- clustering;
- dashboard development.

### `excluded_low_relevance_posts.jsonl`

Contains records excluded because no valid AI-related pattern was detected.

This file is retained for traceability and manual verification.

### `data_quality_summary.csv`

Contains the main data-quality indicators, including:

- number of raw rows;
- number of raw columns;
- invalid JSON lines;
- missing original identifiers;
- exact text duplicates;
- strictly AI-related records;
- low-relevance records;
- analysis-ready records;
- missing dates;
- pre-2025 records;
- emotion or risk signals.

---

## 15. Expected Results

For the current frozen raw dataset, the expected results are approximately:

| Metric                             | Expected value |
| ---------------------------------- | -------------: |
| Raw records                        |          5,406 |
| Missing original IDs               |            141 |
| Exact text duplicate rows          |             37 |
| Strictly AI-relevant rows          |          2,726 |
| Low-relevance rows                 |          2,680 |
| Analysis-ready rows                |          2,666 |
| Missing-date rows                  |            141 |
| Pre-2025 rows                      |            378 |
| Emotion/risk rows in final dataset |            674 |

These values should remain stable as long as the raw dataset is unchanged.

---

## 16. Validation Checks

The cleaning phase is considered valid :

- the raw dataset remains unchanged;
- `record_id` contains no missing values;
- `record_id` is unique;
- `full_text_raw` is preserved;
- `text_clean_basic` exists;
- `text_clean_lexical` exists;
- all final records are AI-related;
- exact duplicates are excluded;
- all final records contain at least five words;
- the three output files are successfully created;
- a small sample is manually inspected.

---

## 17. Limitations

The cleaning process has several limitations:

- regex rules may still miss unusual spellings or new AI tools;
- keyword-based relevance does not guarantee that the full discussion is
  mainly about AI;
- exact duplicate detection does not identify paraphrases;
- records with missing dates cannot be used for timeline analysis;
- source imbalance remains after cleaning;
- short but meaningful records may be excluded by the five-word rule;
- research themes are based on transparent keyword rules rather than a trained
  classifier.

---

## 18. Reproducibility

The cleaning process can be executed from the project root with:

```bash
python src/preprocessing/prepare_analysis_dataset.py   --input data/raw/combined_reddit_posts.jsonl   --output-dir data/processed
```

Required package:

```text
pandas
```

The raw file must remain in:

`data/raw/combined_reddit_posts.jsonl`

---

## 19. Status

| Task                           | Status    |
| ------------------------------ | --------- |
| Preserve raw text              | Completed |
| Standardize text fields        | Completed |
| Create cleaned text columns    | Completed |
| Standardize identifiers        | Completed |
| Standardize platforms          | Completed |
| Standardize dates              | Completed |
| Correct AI relevance detection | Completed |
| Detect research themes         | Completed |
| Detect exact duplicates        | Completed |
| Filter unusable records        | Completed |
| Export processed datasets      | Completed |
| Validate final outputs         | Completed |

---

## 20. Conclusion

The raw multi-source dataset has been transformed into a standardized,
traceable, and analysis-ready dataset.

The official dataset for the next phases is:

`data/processed/analysis_ready_posts.jsonl`

The next phase is:

> **Exploratory Data Analysis**
