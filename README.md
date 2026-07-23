# AI Developer Sentiment Analysis

## Project Overview

This project analyzes how software developers perceive modern Artificial
Intelligence tools and AI coding assistants.

The study focuses on discussions related to tools and technologies such as:

- ChatGPT
- OpenAI
- GitHub Copilot
- Cursor
- Claude
- Gemini
- DeepSeek
- Large Language Models
- Generative AI
- AI agents

The objective is to build a complete and reproducible Data Science and Natural
Language Processing pipeline that can identify:

- positive, neutral, and negative sentiment;
- productivity-related perceptions;
- stress, anxiety, and burnout signals;
- job-security concerns;
- trust and reliability concerns;
- privacy and security concerns;
- major discussion topics;
- differences between platforms and over time.

---

## Current Project Status

| Phase | Status |
|---|---|
| Problem definition | Completed |
| Data collection | Completed |
| Data understanding | Completed |
| Data cleaning | Completed |
| Text preprocessing | Completed |
| Exploratory Data Analysis | Next |
| Sentiment analysis | Planned |
| Topic modeling | Planned |
| Dashboard | Planned |
| Final report | In progress |

The current official analysis-ready dataset contains approximately **2,666
records** selected from an initial multi-source dataset of **5,406 records**.

---

## Project Pipeline

```text
1. Problem Definition
        ↓
2. Data Collection
        ↓
3. Data Understanding
        ↓
4. Data Cleaning
        ↓
5. Text Preprocessing
        ↓
6. Exploratory Data Analysis
        ↓
7. Sentiment Analysis
        ↓
8. Topic Modeling
        ↓
9. Dashboard and Visualization
        ↓
10. Final Report
```

---

## Data Sources

The raw dataset combines several public sources.

| Source | Platform | Records |
|---|---|---:|
| Arctic Shift | Reddit | 305 |
| Hugging Face Reddit dataset | Reddit | 2,000 |
| Hugging Face `divde/sentiment_posts` | Public social dataset | 141 |
| Hacker News | Hacker News | 236 |
| GitHub Issues | GitHub | 1,616 |
| Stack Overflow | Stack Overflow | 1,108 |
| **Total** |  | **5,406** |

The official raw dataset is:

```text
data/raw/combined_reddit_posts.jsonl
```

The official processed dataset is:

```text
data/processed/analysis_ready_posts.jsonl
```

---

## Project Structure

```text
AI-Developer-Sentiment/
├── dashboard/
│   └── # Streamlit dashboard files will be added here
│
├── data/
│   ├── raw/
│   │   ├── combined_reddit_posts.jsonl
│   │   ├── data_sources_inventory.csv
│   │   ├── divde_sentiment_posts.jsonl
│   │   ├── github_posts.jsonl
│   │   ├── hn_posts.jsonl
│   │   ├── huggingface_reddit_posts.jsonl
│   │   ├── posts.jsonl
│   │   └── stackoverflow_posts.jsonl
│   │
│   └── processed/
│       ├── analysis_ready_posts.jsonl
│       ├── data_quality_summary.csv
│       └── excluded_low_relevance_posts.jsonl
│
├── docs/
│   ├── 01_problem_definition.md
│   ├── 02_data_collection.md
│   └── 03_data_cleaning_and_preprocessing.md
│
├── figures/
│   ├── ai_tool_mentions.png
│   ├── posts_by_platform.png
│   ├── posts_over_time.png
│   └── text_length_distribution.png
│
├── notebooks/
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_cleaning_and_preprocessing.ipynb
│   └── archive/
│       ├── 02_data_inspection.ipynb
│       └── 02_data_quality_and_cleaning_fixed.ipynb
│
├── presentation/
│   ├── Pipeline-Data-Science.pptx
│   └── Presentation_Pipeline_AI_Developer_Sentiment_Minimaliste.pptx
│
├── report/
│   ├── benchmark_art.pdf
│   └── pipeline_Documentation.pdf
│
├── src/
│   ├── collectors/
│   │   ├── github_collector.py
│   │   ├── hn_collector.py
│   │   ├── reddit_public_collector.py
│   │   └── stackoverflow_collector.py
│   │
│   └── preprocessing/
│       └── prepare_analysis_dataset.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Data Collection

The project uses reusable Python scripts rather than a collection notebook.

### Reddit

Reddit collection was initially attempted through public Reddit endpoints.
These requests returned HTTP 403 errors.

An official Reddit API access request was submitted, but no response was
received during the collection period.

Alternative collection methods were then tested:

- PullPush;
- Arctic Shift;
- documented Hugging Face datasets.

The final Reddit component combines Arctic Shift and public Hugging Face data.

### GitHub Issues

GitHub Issues were collected from several popular software repositories using
the GitHub REST API.

### Hacker News

Hacker News stories were collected through the public Firebase API.

### Stack Overflow

Stack Overflow questions were collected through the Stack Exchange API.

The collection scripts are stored in:

```text
src/collectors/
```

---

## Data Understanding

The notebook:

```text
notebooks/02_data_understanding.ipynb
```

inspects the raw dataset without modifying it.

It examines:

- number of rows and columns;
- available fields;
- missing values;
- platform distribution;
- missing identifiers;
- empty and very short texts;
- exact text duplicates;
- date coverage;
- text lengths;
- AI-tool mentions.

The notebook is used only for inspection and documentation.

---

## Data Cleaning and Text Preprocessing

The main cleaning script is:

```text
src/preprocessing/prepare_analysis_dataset.py
```

The script:

- preserves the original text;
- creates standardized text fields;
- removes URLs and HTML tags;
- creates lightly cleaned and lexical text versions;
- standardizes identifiers;
- standardizes platforms;
- standardizes date fields;
- detects AI-related records using word-boundary rules;
- detects research-theme signals;
- detects exact text duplicates;
- excludes records with fewer than five words;
- exports the analysis-ready dataset.

### Main Processed Columns

| Column | Description |
|---|---|
| `record_id` | Unique source-qualified identifier |
| `source` | Original data source |
| `platform` | Standardized platform |
| `created_at_utc` | Standardized date |
| `full_text_raw` | Preserved title and body |
| `text_clean_basic` | Light cleaning for sentiment models |
| `text_clean_lexical` | Lowercase lexical version for TF-IDF and topics |
| `ai_tools` | Detected AI tools |
| `is_ai_relevant` | AI relevance indicator |
| `research_themes` | Detected descriptive themes |
| `word_count` | Number of words |
| `analysis_eligible` | Final eligibility decision |

### Main Output Files

```text
data/processed/analysis_ready_posts.jsonl
data/processed/excluded_low_relevance_posts.jsonl
data/processed/data_quality_summary.csv
```

---

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate the environment

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Reproducing the Cleaning Process

Run the following command from the project root:

```bash
python src/preprocessing/prepare_analysis_dataset.py \
  --input data/raw/combined_reddit_posts.jsonl \
  --output-dir data/processed
```

Expected outputs:

```text
data/processed/analysis_ready_posts.jsonl
data/processed/excluded_low_relevance_posts.jsonl
data/processed/data_quality_summary.csv
```

---

## Running the Notebooks

Start Jupyter from the project root:

```bash
jupyter notebook
```

Then open:

```text
notebooks/02_data_understanding.ipynb
notebooks/03_data_cleaning_and_preprocessing.ipynb
```

The notebooks should be executed from top to bottom.

---

## Figures

Current figures include:

- number of posts by platform;
- text-length distribution;
- AI-tool mentions;
- number of posts over time.

The figures are stored in:

```text
figures/
```

These initial figures describe the dataset. More complete EDA figures will be
added during the next project phase.

---

## Documentation

Technical documentation is stored in:

```text
docs/
```

Current documents:

- `01_problem_definition.md`
- `02_data_collection.md`
- `03_data_cleaning_and_preprocessing.md`

Additional documentation will be created for:

- Exploratory Data Analysis;
- sentiment analysis;
- topic modeling;
- dashboard usage;
- testing and validation.

---

## Reports and Presentations

Current project reports are stored in:

```text
report/
```

Current presentations are stored in:

```text
presentation/
```

They include:

- pipeline documentation;
- state-of-the-art or benchmark material;
- project pipeline presentation.

---

## Next Steps

The next technical phase is **Exploratory Data Analysis**.

The planned work includes:

1. Validate the processed dataset.
2. Analyze records by platform and source.
3. Analyze text length.
4. Analyze AI-tool mentions.
5. Analyze predefined research themes.
6. Analyze temporal trends.
7. Select representative records.
8. Save figures and interpretation.
9. Document the EDA phase.
10. Continue with sentiment analysis.

---

## Known Limitations

- The dataset does not represent all software developers.
- The platforms use different sampling methods.
- Platform sizes are imbalanced.
- Some records have missing dates or metadata.
- Keyword rules may miss unusual AI-tool spellings.
- Exact duplicate detection does not identify paraphrases.
- Technical error messages may be difficult for generic sentiment models.
- Public posts may contain sarcasm or ambiguous language.
- Theme detection is currently rule-based and descriptive.

These limitations will be reported in the final analysis.

---

## Repository Cleanup Notes

The following files should not be committed to Git:

```text
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
```

Temporary files such as `_test.jsonl` should be removed after validation.

Older outputs such as `combined_posts_cleaned.jsonl` should remain archived or
be removed once `analysis_ready_posts.jsonl` is confirmed as the official
processed dataset.

---

## Author

**Charifa FAX**

Summer Internship Project — 2026
