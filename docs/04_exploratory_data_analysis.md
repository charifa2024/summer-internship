# 04 — Exploratory Data Analysis

## 1. Objective

Exploratory Data Analysis (EDA) is used to understand the official
analysis-ready dataset before applying sentiment analysis or topic modeling.

The EDA answers the following questions:

1. Is the processed dataset valid?
2. How are records distributed across platforms and sources?
3. How long are the texts?
4. Which AI tools are mentioned most often?
5. Which predefined research themes appear most often?
6. How does discussion volume evolve over time?
7. How do tools and themes differ across platforms?
8. Which limitations must be considered before modeling?

EDA describes the dataset. It does not yet assign sentiment labels or discover
final topics.

---

## 2. Implementation

The EDA is implemented in:

`notebooks/04_eda.ipynb`

The notebook automatically locates the project root and creates the folders
required for figures and result tables.

---

## 3. Input Dataset

The official input is:

`data/processed/analysis_ready_posts.jsonl`

The raw dataset is not used directly during EDA.

The input was created during the cleaning and text preprocessing phase and
contains standardized fields, cleaned text, AI-tool indicators, research-theme
indicators, dates, and text-length measures.

---

## 4. Main Fields

| Column | Description |
|---|---|
| `record_id` | Unique source-qualified identifier |
| `source` | Original data source |
| `platform` | Standardized platform |
| `community` | Subreddit, repository, or community |
| `created_at_utc` | Standardized UTC date |
| `date_known` | Indicates whether the date is available |
| `full_text_raw` | Preserved original text |
| `text_clean_basic` | Text prepared for sentiment analysis |
| `text_clean_lexical` | Text prepared for TF-IDF and topic modeling |
| `ai_tools` | Detected AI tools and technologies |
| `research_themes` | Detected predefined themes |
| `word_count` | Number of words |
| `text_length_chars` | Number of characters |
| `score` | Platform-specific score when available |
| `num_comments` | Comments or answers when available |

---

## 5. Libraries

The notebook uses:

- `pathlib`;
- `ast`;
- `pandas`;
- `numpy`;
- `matplotlib`;
- standard Python utilities.

No machine-learning model is used during EDA.

---

## 6. Output Folders

Figures are saved in:

`figures/eda/`

Result tables are saved in:

`data/results/eda/`

These folders are created automatically.

---

## 7. Dataset Loading

The processed JSONL file is loaded with pandas.

The notebook displays:

- the number of records;
- the number of columns;
- the first five rows.

If the file is missing, execution stops with an explicit error.

---

## 8. Column Validation

The notebook verifies that the following required fields exist:

- `record_id`;
- `source`;
- `platform`;
- `full_text_raw`;
- `text_clean_basic`;
- `text_clean_lexical`;
- `ai_tools`;
- `research_themes`;
- `word_count`.

Optional fields include:

- `community`;
- `created_at_utc`;
- `date_known`;
- `text_length_chars`;
- `score`;
- `num_comments`;
- `analysis_eligible`;
- `is_exact_text_duplicate`.

Execution stops when an essential field is missing.

---

## 9. List and Date Normalization

The notebook ensures that `ai_tools` and `research_themes` are stored as Python
lists.

It also converts `created_at_utc` into a valid UTC datetime field.

Invalid or missing dates become null values and are excluded only from
temporal analysis.

When `text_length_chars` is missing, it is recalculated from
`text_clean_basic`.

---

## 10. Final Data-Quality Validation

Before creating charts, the notebook checks:

- missing record IDs;
- duplicate record IDs;
- empty cleaned texts;
- minimum word count;
- missing dates;
- remaining exact duplicates;
- ineligible rows in the final dataset.

Expected conditions are:

- no missing `record_id`;
- no duplicate `record_id`;
- no empty analysis text;
- minimum word count of at least five;
- no exact duplicates;
- no ineligible records.

Missing dates are acceptable but reduce temporal coverage.

Output:

`data/results/eda/validation_summary.csv`

---

## 11. General Dataset Overview

The notebook calculates:

- total records;
- total columns;
- number of platforms;
- number of sources;
- dated records;
- missing dates;
- date coverage percentage;
- average word count;
- median word count;
- minimum and maximum word counts.

Output:

`data/results/eda/eda_overview.csv`

The mean and median should be interpreted together. A mean much higher than the
median usually indicates a right-skewed text-length distribution.

---

## 12. Platform Distribution

The notebook calculates counts and percentages by platform.

Output table:

`data/results/eda/platform_summary.csv`

Output figure:

`figures/eda/records_by_platform.png`

### Why It Matters

When one platform dominates the sample, global sentiment and topic results may
mainly reflect that platform.

Later results should therefore be reported globally and by platform.

---

## 13. Source Distribution

The notebook calculates counts and percentages by original collection source.

Output table:

`data/results/eda/source_summary.csv`

Output figure:

`figures/eda/records_by_source.png`

This distinguishes the original collection source from the standardized
platform.

---

## 14. Community Analysis

When available, the notebook identifies the twenty most represented
communities.

A community may be:

- a subreddit;
- a GitHub repository;
- Stack Overflow;
- Hacker News;
- another source-specific group.

Output:

`data/results/eda/top_communities.csv`

This analysis reveals whether a small number of communities dominate the
sample.

---

## 15. Text-Length Analysis

The notebook analyzes character count and word count using:

- count;
- mean;
- standard deviation;
- minimum;
- quartiles;
- median;
- maximum.

Output table:

`data/results/eda/text_length_summary.csv`

Output figure:

`figures/eda/text_length_distribution.png`

The histogram is clipped visually at the 99th percentile to remain readable.
No records are deleted.

---

## 16. Text Length by Platform

For every platform, the notebook calculates:

- record count;
- mean word count;
- median word count;
- minimum;
- maximum.

Output:

`data/results/eda/text_length_by_platform.csv`

Figure:

`figures/eda/text_length_by_platform.png`

This analysis is important because GitHub Issues, Stack Overflow questions,
Reddit posts, and Hacker News stories have different writing styles and
lengths.

---

## 17. AI-Tool Frequency Analysis

The `ai_tools` list is exploded so each detected tool can be counted.

The notebook calculates:

- number of documents mentioning each tool;
- percentage of final records mentioning each tool.

Output:

`data/results/eda/ai_tool_counts.csv`

Figure:

`figures/eda/ai_tool_frequencies.png`

One document may mention several tools, so percentages are not mutually
exclusive.

This section measures representation, not sentiment.

---

## 18. Research-Theme Frequency Analysis

The notebook counts the predefined themes:

- `stress_anxiety`;
- `burnout`;
- `job_security`;
- `productivity`;
- `trust_quality`;
- `privacy_security`.

Output:

`data/results/eda/research_theme_counts.csv`

Figure:

`figures/eda/research_theme_frequencies.png`

Theme detection is rule-based. It identifies the presence of a theme but not
whether the statement is positive or negative.

---

## 19. Date Coverage

The notebook calculates:

- known dates;
- missing dates;
- date coverage percentage;
- earliest date;
- latest date.

Output:

`data/results/eda/date_coverage.csv`

Records without dates remain usable for non-temporal analysis.

---

## 20. Temporal Analysis

Dated records are grouped by month.

Output:

`data/results/eda/monthly_record_counts.csv`

Figure:

`figures/eda/records_over_time.png`

A temporal peak may reflect:

- a real event;
- a product release;
- source coverage;
- the collection method;
- one imported dataset.

EDA identifies the pattern but does not prove its cause.

---

## 21. AI Tools by Platform

The notebook creates:

- tool counts by platform;
- tool percentages within each platform.

Outputs:

`data/results/eda/ai_tools_by_platform_counts.csv`

`data/results/eda/ai_tools_by_platform_percent.csv`

Figure:

`figures/eda/ai_tools_by_platform.png`

Counts show volume. Percentages support fairer comparison when platform sizes
differ.

---

## 22. Research Themes by Platform

The notebook compares predefined themes across platforms.

Outputs:

`data/results/eda/themes_by_platform_counts.csv`

`data/results/eda/themes_by_platform_percent.csv`

Figure:

`figures/eda/themes_by_platform.png`

The results can reveal whether certain platforms focus more on productivity,
job security, trust, stress, burnout, or privacy.

Conclusions must be based on the actual executed results.

---

## 23. Top AI Tools Over Time

The five most frequent AI tools are counted by month using only dated records.

Output:

`data/results/eda/top_tools_over_time.csv`

Figure:

`figures/eda/top_tools_over_time.png`

The chart is limited to five tools to remain readable.

---

## 24. Engagement Inspection

When available, `score` and `num_comments` are summarized separately by
platform.

Output:

`data/results/eda/engagement_by_platform.csv`

These measures are not directly comparable across platforms. A Reddit score,
Stack Overflow score, Hacker News points, and GitHub comments do not represent
the same behavior.

No global engagement score is created.

---

## 25. Representative Examples

The notebook displays:

- a random sample;
- shortest records;
- longest records;
- records mentioning several AI tools;
- records containing several themes.

Manual inspection helps identify:

- false AI matches;
- code-dominated texts;
- unclear records;
- unusual formatting;
- theme-detection limitations.

---

## 26. Automatically Generated Findings

The notebook creates a short summary containing:

- dataset size;
- dominant platform;
- dominant source;
- typical text length;
- date coverage;
- most represented AI tool;
- most frequent predefined theme.

Output:

`data/results/eda/eda_key_findings.csv`

These findings must be reviewed before being copied into the final report.

---

## 27. Figure Interpretation

Every important figure should be followed by:

### Observation

Describe what the chart shows.

Example:

> Reddit contains the largest number of analysis-ready records.

### Methodological Implication

Explain why the result matters.

Example:

> Global sentiment results may be influenced by Reddit, so platform-specific
> sentiment should also be reported.

Avoid generalizations such as:

> All developers think...

Use:

> In the collected multi-platform sample...

---

## 28. Generated Figures

The notebook may generate:

```text
figures/eda/
├── records_by_platform.png
├── records_by_source.png
├── text_length_distribution.png
├── text_length_by_platform.png
├── ai_tool_frequencies.png
├── research_theme_frequencies.png
├── records_over_time.png
├── ai_tools_by_platform.png
├── themes_by_platform.png
└── top_tools_over_time.png
```

---

## 29. Generated Result Tables

The notebook may generate:

```text
data/results/eda/
├── validation_summary.csv
├── eda_overview.csv
├── platform_summary.csv
├── source_summary.csv
├── top_communities.csv
├── text_length_summary.csv
├── text_length_by_platform.csv
├── ai_tool_counts.csv
├── research_theme_counts.csv
├── date_coverage.csv
├── monthly_record_counts.csv
├── ai_tools_by_platform_counts.csv
├── ai_tools_by_platform_percent.csv
├── themes_by_platform_counts.csv
├── themes_by_platform_percent.csv
├── top_tools_over_time.csv
├── engagement_by_platform.csv
└── eda_key_findings.csv
```

Some files depend on the availability of optional fields.

---

## 30. Reproducibility

From the project root, run:

```bash
jupyter notebook
```

Open:

`notebooks/04_eda.ipynb`

Then select:

`Kernel → Restart Kernel and Run All Cells`

The notebook should load the dataset, validate it, create figures, save tables,
display examples, and generate key findings.

---

## 31. Completion Conditions

The EDA phase is complete when:

- all cells run without errors;
- validation checks pass;
- figures are created;
- result tables are created;
- representative examples are reviewed;
- each important chart has an interpretation;
- limitations are documented;
- the notebook is saved with its outputs.

---

## 32. Limitations

The EDA has the following limitations:

- the sample does not represent all software developers;
- platform sizes are imbalanced;
- sources use different sampling methods;
- some records have missing dates;
- one document may mention multiple tools and themes;
- theme detection is rule-based;
- exact duplicate detection does not identify paraphrases;
- temporal peaks may reflect collection coverage;
- engagement measures are not equivalent across platforms;
- technical texts may not contain explicit personal sentiment;
- public discussions may contain sarcasm or ambiguous language.

---

## 33. Status

| Task | Status |
|---|---|
| Load processed dataset | To execute |
| Validate required columns | To execute |
| Validate data quality | To execute |
| Analyze platforms and sources | To execute |
| Analyze communities | To execute |
| Analyze text lengths | To execute |
| Analyze AI tools | To execute |
| Analyze research themes | To execute |
| Analyze date coverage and trends | To execute |
| Compare platforms | To execute |
| Inspect representative records | To execute |
| Generate figures and tables | To execute |
| Review findings | To execute |
| Write interpretations | To execute |

Change the status to **Completed** only after executing and reviewing the
notebook.

---

## 34. Conclusion

EDA transforms the analysis-ready corpus into interpretable information about:

- platform and source composition;
- text characteristics;
- AI-tool representation;
- predefined themes;
- temporal coverage;
- cross-platform differences;
- methodological limitations.

After successful execution and interpretation, the next phase is:

> **Sentiment Analysis**
