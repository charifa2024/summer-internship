# 04 — Exploratory Data Analysis

## Objective

Describe and validate the final 2,666-post corpus before interpreting model outputs.

## Input

```text
data/processed/analysis_ready_posts.jsonl
```

## Main checks

- record count;
- unique `record_id`;
- no empty analysis text;
- no remaining exact duplicate;
- minimum word count;
- platform/source composition;
- date coverage;
- text length;
- AI-tool mentions;
- predefined research-theme indicators.

## Main outputs

```text
data/results/eda/validation_summary.csv
data/results/eda/eda_overview.csv
data/results/eda/platform_summary.csv
data/results/eda/source_summary.csv
data/results/eda/top_communities.csv
data/results/eda/text_length_summary.csv
data/results/eda/ai_tool_counts.csv
data/results/eda/research_theme_counts.csv
data/results/eda/date_coverage.csv
data/results/eda/monthly_record_counts.csv
data/results/eda/eda_key_findings.csv
```

## Final role of EDA

EDA is descriptive. It does not establish psychological states and does not determine final NMF topics.

The main methodological conclusion is that the corpus is suitable for consistent downstream modeling but remains source-concentrated and not representative of all developers.
