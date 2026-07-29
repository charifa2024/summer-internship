# AI Developer Sentiment Analysis

## Project Overview

This project analyzes how software developers perceive modern artificial
intelligence tools and AI coding assistants in public online discussions.

The study focuses on tools and technologies such as ChatGPT, OpenAI, GitHub
Copilot, Cursor, Claude, Gemini, DeepSeek, large language models, generative AI,
and AI agents.

The project implements a complete Data Science and Natural Language Processing
pipeline to identify:

- positive, neutral, and negative sentiment;
- stress, anxiety, frustration, burnout, and relief signals;
- productivity-related perceptions;
- job-security concerns;
- trust, reliability, privacy, and security concerns;
- major discussion topics;
- differences across platforms, AI tools, and time.

The dashboard and analytical outputs describe patterns in public developer
discussions. They do not provide clinical, psychological, or causal diagnoses.

---

## Project Status

| Phase | Status |
|---|---|
| Problem definition | Completed |
| Benchmark and state of the art | Completed |
| Data collection | Completed |
| Data understanding | Completed |
| Data cleaning and preprocessing | Completed |
| Exploratory Data Analysis | Completed |
| VADER sentiment analysis | Completed |
| Transformer sentiment analysis | Completed |
| LLM-assisted reference evaluation | Completed |
| Stress and emotion analysis | Completed |
| Topic modeling | Completed |
| Automated tests | Completed — 4 tests passed |
| Streamlit dashboard | Completed and locally functional |
| Technical documentation | In progress |
| Public deployment | Pending |

The official analysis-ready dataset contains approximately **2,666 records**
selected from an initial multi-source dataset of **5,406 records**.

---

## Analytical Pipeline

```text
Public social data
        ↓
Data collection and source documentation
        ↓
Data understanding and quality audit
        ↓
Cleaning, normalization, relevance filtering, and deduplication
        ↓
Exploratory Data Analysis
        ↓
Sentiment analysis: VADER and Transformer
        ↓
LLM-assisted reference evaluation
        ↓
Stress and emotion signal analysis
        ↓
TF-IDF and NMF topic modeling
        ↓
Result tables and figures
        ↓
Interactive Streamlit and Plotly dashboard
        ↓
Interpretation, limitations, and recommendations
```

---

## Data Sources

The raw dataset combines several documented public sources.

| Source | Platform | Records |
|---|---|---:|
| Arctic Shift | Reddit | 305 |
| Hugging Face Reddit dataset | Reddit | 2,000 |
| Hugging Face `divde/sentiment_posts` | Public social dataset | 141 |
| Hacker News | Hacker News | 236 |
| GitHub Issues | GitHub | 1,616 |
| Stack Overflow | Stack Overflow | 1,108 |
| **Total** |  | **5,406** |

Official raw dataset:

```text
data/raw/combined_reddit_posts.jsonl
```

Official processed dataset:

```text
data/processed/analysis_ready_posts.jsonl
```

Data-source inventory:

```text
data/raw/data_sources_inventory.csv
```

The data consist of public posts and publicly available datasets. The project
does not attempt to identify individual users and reports aggregated patterns.

---

## Repository Structure

```text
AI-Developer-Sentiment/
├── dashboard/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
│       ├── eda/
│       ├── sentiment/
│       ├── emotions/
│       └── topics/
├── docs/
├── figures/
│   ├── eda/
│   ├── sentiment/
│   ├── emotions/
│   └── topics/
├── notebooks/
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_cleaning_and_preprocessing.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_sentiment_analysis_vader.ipynb
│   ├── 05_sentiment_analysis_transformer.ipynb
│   ├── 05_sentiment_llm_assisted_validation.ipynb
│   ├── 06_stress_and_emotion_analysis.ipynb
│   ├── 07_topic_modeling.ipynb
│   └── archive/
├── presentation/
├── report/
├── src/
│   ├── collectors/
│   └── preprocessing/
│       └── prepare_analysis_dataset.py
├── tests/
│   └── test_prepare_analysis_dataset.py
├── logs/
│   └── dashboard.log
├── README.md
├── requirements.txt
└── .gitignore
```

The `logs/` folder is created automatically when the dashboard starts.

---

## Installation

### 1. Create and activate a virtual environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Run from the project root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Main dependencies include Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn,
NLTK, Transformers, PyTorch, WordCloud, Plotly, Streamlit, Pytest, and Jupyter.

---

## Reproducing the Processed Dataset

Run from the project root:

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

The preprocessing script preserves original text, standardizes fields,
removes URLs and HTML markup, normalizes dates and identifiers, detects AI
relevance and predefined themes, identifies exact duplicates, excludes records
with fewer than five words, and generates a reproducible quality summary.

---

## Notebook Execution Order

Start Jupyter from the project root:

```bash
jupyter notebook
```

Run the notebooks in this order:

```text
1. notebooks/02_data_understanding.ipynb
2. notebooks/03_data_cleaning_and_preprocessing.ipynb
3. notebooks/04_eda.ipynb
4. notebooks/05_sentiment_analysis_vader.ipynb
5. notebooks/05_sentiment_analysis_transformer.ipynb
6. notebooks/05_sentiment_llm_assisted_validation.ipynb
7. notebooks/06_stress_and_emotion_analysis.ipynb
8. notebooks/07_topic_modeling.ipynb
```

Each notebook should be executed from top to bottom from the project root so
that relative file paths resolve correctly.

---

## Exploratory Data Analysis

The EDA notebook covers:

- records by source, platform, and community;
- meaningful word frequencies;
- overall and platform word clouds;
- text-length distributions;
- AI-tool mentions;
- predefined research themes;
- temporal discussion volume;
- cross-platform comparisons;
- representative examples;
- coverage and methodological limitations.

Main outputs are stored in:

```text
data/results/eda/
figures/eda/
```

Important outputs include:

```text
data/results/eda/word_frequency.csv
data/results/eda/word_frequency_by_platform.csv
data/results/eda/word_frequency_by_community.csv
figures/eda/word_frequency_top25.png
figures/eda/wordcloud_overall.png
```

---

## Sentiment Analysis

Two sentiment methods are compared.

### VADER

VADER is a lexicon-based baseline for social-media text. It produces positive,
neutral, negative, and compound scores.

### Transformer

A pretrained Transformer sentiment model produces contextual sentiment labels
and confidence values.

The comparison includes distributions, platform and AI-tool comparisons,
time trends, model agreement and disagreement, and evaluation against an
LLM-assisted reference sample.

The LLM-assisted labels are provisional analytical references, not independent
human-annotated ground truth.

Outputs are stored in:

```text
data/results/sentiment/
figures/sentiment/
```

---

## Stress and Emotion Analysis

An explainable rule-based baseline identifies textual signals associated with:

- increased stress or distress;
- reduced stress or relief;
- mixed or ambiguous signals;
- anxiety;
- frustration;
- burnout or exhaustion;
- job insecurity or fear;
- distrust or uncertainty.

Negative sentiment is not automatically treated as stress.

Outputs are stored in:

```text
data/results/emotions/
figures/emotions/
```

This component describes textual patterns and does not perform clinical
diagnosis.

---

## Topic Modeling

Topic modeling uses TF-IDF features and Non-negative Matrix Factorization.
Several candidate topic counts are compared using reconstruction error and
topic-diversity indicators. The outputs include topic keywords, representative
documents, and topic distributions by platform, AI tool, sentiment, stress,
and time.

Automatic topic names are provisional and should be interpreted with the
keywords and representative documents.

Outputs are stored in:

```text
data/results/topics/
figures/topics/
```

---

## Automated Tests

Run from the project root:

```bash
python -m pytest tests/test_prepare_analysis_dataset.py -v
```

The test suite contains four scenarios:

1. **Nominal:** a valid AI-related Reddit post is processed successfully.
2. **Boundary:** a valid post without an original ID or date is handled without
   crashing.
3. **Anomaly:** an exact normalized-text duplicate is detected and excluded.
4. **Error:** malformed JSON raises a clear error containing the line number.

Latest verified result:

```text
4 passed in 0.88s
```

---

## Running the Dashboard

From the project root:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard includes:

- filters by platform, sentiment, stress direction, topic, AI tool, and date;
- dataset indicators;
- sentiment distributions;
- stress and emotion signals;
- topics and topic keywords;
- trends over time;
- representative examples;
- an automatic synthesis of the selected data;
- cautious recommendations;
- a searchable data explorer;
- CSV export;
- missing-data and empty-filter warnings.

Dashboard activity and loading errors are written to:

```text
logs/dashboard.log
```

The dashboard currently runs locally. A public deployment URL has not yet been
documented.

---

## Methodological and Ethical Limitations

- The dataset does not represent all software developers.
- Platforms have different user populations and sampling methods.
- Platform sizes are imbalanced.
- Public posts may contain sarcasm, irony, slang, and ambiguous language.
- Some records have missing dates or metadata.
- Keyword rules may miss unusual AI-tool spellings.
- Exact duplicate detection does not identify paraphrases.
- Generic sentiment models may misinterpret technical error messages.
- Stress and emotion labels are descriptive signals, not clinical assessments.
- LLM-assisted validation is not independent expert annotation.
- Topic names are provisional interpretations of model factors.
- Temporal peaks may reflect collection coverage as well as genuine interest.
- Word clouds are descriptive and should not replace exact frequency tables.
- Observational social-media data do not establish causality.
- Findings should not be generalized to all developers or organizations.

---

## Demonstration Workflow

1. Activate the virtual environment.
2. Run the automated tests.
3. Launch the Streamlit dashboard.
4. Show the overview and filters.
5. Compare sentiment across platforms.
6. Inspect stress and emotion patterns.
7. Inspect topic keywords and representative examples.
8. Show trends over time.
9. Explain the main limitations and recommendations.

---

## Repository Cleanup

The following files should not be committed:

```text
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
```

Temporary files such as `_test.jsonl` should be deleted before submission.
Archived notebooks should remain inside `notebooks/archive/`.

The active validation workflow should use:

```text
notebooks/05_sentiment_llm_assisted_validation.ipynb
```

rather than an older manual-validation notebook.

---

## Author

**Charifa FAX**

Summer Internship Project — 2026
