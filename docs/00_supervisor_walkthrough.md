# Supervisor Walkthrough — AI Developer Sentiment Project

**Author:** Charifa FAX  
**Project:** Summer Internship 2026  
**Purpose:** Explain every step of the project in plain language, so a non-technical reader can follow what was done, why it was done, and what the results mean.

---

## 1. What This Project Is About (One Paragraph)

Developers talk about AI tools (ChatGPT, Copilot, Cursor, Claude, etc.) on public platforms like Reddit, GitHub, Stack Overflow, and Hacker News. This project **collects those public posts**, **cleans and organizes them**, **analyzes how developers feel**, **detects stress-related language**, **finds recurring discussion topics**, and **presents everything in an interactive dashboard**. The goal is to understand patterns in public developer discussions — not to diagnose individuals or claim that all developers think the same way.

---

## 2. The Big Picture (Pipeline Overview)

Think of the project as a factory line. Each step takes the output of the previous step and adds new information.

```text
STEP 1 — Define the problem
        ↓
STEP 2 — Collect public posts from 6 sources (5,406 records)
        ↓
STEP 3 — Understand and inspect the raw data
        ↓
STEP 4 — Clean, filter, and keep only AI-relevant posts (2,666 records)
        ↓
STEP 5 — Explore the data (charts, word counts, platform breakdown)
        ↓
STEP 6 — Sentiment analysis: is each post positive, neutral, or negative?
        ↓
STEP 7 — Validate sentiment methods (compare two models + LLM reference)
        ↓
STEP 8 — Stress and emotion analysis: does the text mention stress, anxiety, etc.?
        ↓
STEP 9 — Topic modeling: what are people actually talking about?
        ↓
STEP 10 — Dashboard: interactive tool to explore all results
        ↓
STEP 11 — Final report and presentation
```

Each step is documented below with four questions:

| Question | Meaning |
|---|---|
| **What?** | What happens in this step |
| **Why?** | Why it is necessary |
| **How?** | What tool or method was used |
| **Output?** | What file or result was produced |

---

## 3. Step-by-Step Explanation

### Step 1 — Problem Definition

| | |
|---|---|
| **What?** | Define the research question, objectives, and scope of the project. |
| **Why?** | Without a clear question, data collection and analysis have no direction. |
| **How?** | Written documentation based on literature review and project planning. |
| **Output?** | `docs/01_problem_definition.md` |

**Main research question:**

> How do software developers perceive modern AI tools and AI coding assistants?

**What we want to find out:**

- Are opinions mostly positive, negative, or neutral?
- Which AI tools are mentioned most?
- Do developers express stress, anxiety, or job-security concerns?
- What topics dominate the discussions?
- Do opinions differ between platforms (Reddit vs GitHub vs Stack Overflow)?

**What this project does NOT claim:**

- It does not represent every developer in the world.
- It does not prove that AI tools *cause* stress or happiness.
- It does not provide medical or psychological diagnoses.

---

### Step 2 — Data Collection

| | |
|---|---|
| **What?** | Gather public developer posts that mention AI tools from multiple online sources. |
| **Why?** | We need real-world text data to analyze. Manual reading of thousands of posts is impossible. |
| **How?** | Public APIs and documented datasets (Arctic Shift, Hugging Face, GitHub, Stack Overflow, Hacker News). |
| **Output?** | `data/raw/combined_reddit_posts.jsonl` — **5,406 records** |

**Where the data comes from:**

| Source | Platform | Records |
|---|---|---:|
| Arctic Shift | Reddit | 305 |
| Hugging Face Reddit dataset | Reddit | 2,000 |
| Hugging Face sentiment dataset | Mixed | 141 |
| Hacker News | Hacker News | 236 |
| GitHub Issues | GitHub | 1,616 |
| Stack Overflow | Stack Overflow | 1,108 |
| **Total** | | **5,406** |

**Important note about Reddit:** The official Reddit API was not available during the internship. Reddit data was collected through Arctic Shift (a public archive) and a Hugging Face dataset instead. This is documented and does not affect the rest of the pipeline.

**What each record contains (when available):** title, body text, date, platform, author, score, number of comments, and URL.

---

### Step 3 — Data Understanding

| | |
|---|---|
| **What?** | Inspect the raw dataset before cleaning — check size, missing values, duplicates, date coverage, text quality. |
| **Why?** | Cleaning rules should be based on evidence, not assumptions. We need to know what problems exist in the data first. |
| **How?** | Jupyter notebook with summary tables and visual checks. |
| **Output?** | `notebooks/02_data_understanding.ipynb` |

**What we checked:**

- How many records and columns exist?
- Are there missing dates, missing IDs, or empty texts?
- How are records distributed across platforms?
- Are there duplicate posts?
- How long are the texts?
- Are the posts actually about AI, or did keyword matching pull in irrelevant content?

**Key finding:** Many records matched broad keywords (like the letters "ai" inside words such as "email" or "again") but were not truly about artificial intelligence. This justified a strict filtering step in the next phase.

---

### Step 4 — Data Cleaning and Preprocessing

| | |
|---|---|
| **What?** | Transform the raw 5,406 records into a clean, standardized dataset ready for analysis. |
| **Why?** | Raw data from different sources has different formats, irrelevant content, and duplicates. Models need consistent, high-quality input. |
| **How?** | Automated Python script: `src/preprocessing/prepare_analysis_dataset.py` |
| **Output?** | `data/processed/analysis_ready_posts.jsonl` — **2,666 records** |

**What the cleaning script does (in plain language):**

1. **Keeps the original text** — nothing is destroyed; cleaned versions are stored separately.
2. **Standardizes fields** — every record gets the same column names regardless of source.
3. **Removes URLs and HTML** — so links and web formatting do not confuse the analysis.
4. **Detects AI relevance strictly** — uses word-boundary rules so "email" is not counted as "AI".
5. **Removes exact duplicates** — same text appearing twice is counted once.
6. **Removes very short posts** — fewer than 5 words are too short to analyze reliably.
7. **Tags research themes** — flags posts that mention stress, burnout, job security, productivity, trust, or privacy (using keyword rules).
8. **Creates a quality summary** — a report of how many records were kept and why others were excluded.

**What happened to the data:**

| Stage | Records |
|---|---:|
| Raw collected data | 5,406 |
| Strictly AI-related | 2,726 |
| After removing duplicates and short texts | **2,666** |
| Excluded (not AI-related) | 2,680 |

**How to reproduce this step:**

```bash
python src/preprocessing/prepare_analysis_dataset.py \
  --input data/raw/combined_reddit_posts.jsonl \
  --output-dir data/processed
```

---

### Step 5 — Exploratory Data Analysis (EDA)

| | |
|---|---|
| **What?** | Create charts and summary tables to understand the dataset before running any AI models. |
| **Why?** | You should know what your data looks like before interpreting model results. EDA reveals imbalances, patterns, and limitations. |
| **How?** | Jupyter notebook with Pandas, Matplotlib, and WordCloud. |
| **Output?** | `notebooks/04_eda.ipynb`, tables in `data/results/eda/`, figures in `figures/eda/` |

**Main questions EDA answers:**

- Which platform has the most posts?
- Which AI tools are mentioned most often?
- How long are the texts?
- Which predefined themes (stress, productivity, trust, etc.) appear most?
- How does discussion volume change over time?

**Key findings from the actual results:**

| Finding | Value |
|---|---|
| Analysis-ready records | 2,666 |
| Dominant platform | Reddit (76%) |
| Typical post length | 97 words (median) |
| Date coverage | 95% of records have a date |
| Most mentioned AI tool | "AI" (general term) |
| Most frequent theme | Trust and quality concerns |

**Why this matters for the supervisor:** Reddit dominates the sample. Any "overall" result mostly reflects Reddit users, not necessarily all developers. That is why results are also reported **by platform**.

---

### Step 6 — Sentiment Analysis

| | |
|---|---|
| **What?** | Classify each post as **positive**, **neutral**, or **negative** toward AI tools. |
| **Why?** | Sentiment is the core answer to "how do developers feel about AI?" |
| **How?** | Two different methods are used and compared (see below). |
| **Output?** | `data/results/sentiment/` — prediction files, comparison tables, and charts |

#### Method A — VADER (simple baseline)

- **What it is:** A dictionary-based tool designed for social media text.
- **How it works:** It looks up words in a sentiment lexicon (e.g., "great" = positive, "terrible" = negative) and computes a compound score.
- **Strength:** Fast, transparent, no GPU needed.
- **Weakness:** Does not understand context well (e.g., sarcasm, technical error messages).
- **Notebook:** `notebooks/05_sentiment_analysis_vader.ipynb`

#### Method B — Transformer (advanced model)

- **What it is:** A pretrained deep-learning model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) that reads full sentences and understands context.
- **How it works:** The model was trained on millions of social media posts. It outputs a label (positive/neutral/negative) and a confidence score.
- **Strength:** Better at context, negation, and informal language.
- **Weakness:** Slower, requires more computing resources; may misread technical jargon.
- **Notebook:** `notebooks/05_sentiment_analysis_transformer.ipynb`

**Key findings:**

| Finding | Value |
|---|---|
| Most frequent Transformer label | Neutral |
| Mean model confidence | 72.6% |
| VADER vs Transformer agreement | 37.6% |

**Why agreement is low:** The two methods use completely different approaches. Disagreement is expected and is exactly why we compare them rather than trusting one method alone.

---

### Step 7 — LLM-Assisted Validation

| | |
|---|---|
| **What?** | Compare VADER and Transformer against a third reference: labels produced with LLM-assisted semantic review. |
| **Why?** | We need a way to evaluate which sentiment method performs better, but we do not have a large human-annotated dataset. |
| **How?** | A sample of posts is labeled using LLM-assisted review, then both methods are scored against those labels. |
| **Output?** | `notebooks/05_sentiment_llm_assisted_validation.ipynb`, results in `data/results/sentiment/` |

**Critical limitation (must be stated clearly):**

> The LLM-assisted labels are **provisional reference annotations**, not independent expert human labels. They help compare methods but are not ground truth.

**Result:** The Transformer model performed slightly better (macro F1 = 0.51 vs VADER). The Transformer was therefore recommended as the primary sentiment method for downstream analysis.

---

### Step 8 — Stress and Emotion Analysis

| | |
|---|---|
| **What?** | Detect explicit language about stress, anxiety, frustration, burnout, job fear, distrust, or relief. |
| **Why?** | Sentiment alone is not enough. A post can be negative without mentioning stress ("this code is buggy"). Stress detection looks for **specific emotional language**. |
| **How?** | Transparent keyword and phrase rules (not a black-box model). |
| **Output?** | `notebooks/06_stress_and_emotion_analysis.ipynb`, results in `data/results/emotions/` |

**Important distinction:**

| Concept | What it measures | Example |
|---|---|---|
| **Sentiment** | General tone: positive, neutral, negative | "Copilot is amazing" → Positive |
| **Stress signal** | Explicit emotional language | "I'm burned out from debugging AI-generated code" → Stress detected |

**Stress categories detected:**

- Increased stress or distress
- Reduced stress or relief
- Mixed or ambiguous
- No clear stress signal

**Emotion categories detected:**

- Anxiety, frustration, burnout, job insecurity, distrust, relief, or no clear emotion

**Key findings:**

| Finding | Value |
|---|---|
| Posts with explicit stress or mixed signal | 412 (15.5%) |
| Posts with no clear stress signal | 84.4% |
| Negative sentiment alone ≠ stress | A post must contain explicit emotional language |

**What this does NOT do:** It does not diagnose anyone's mental health. It only detects **words and phrases** in public posts.

---

### Step 9 — Topic Modeling

| | |
|---|---|
| **What?** | Automatically discover the main subjects developers discuss about AI tools. |
| **Why?** | Sentiment tells you *how* people feel; topics tell you *what* they are talking about. |
| **How?** | TF-IDF (word importance scoring) + NMF (Non-negative Matrix Factorization). |
| **Output?** | `notebooks/07_topic_modeling.ipynb`, results in `data/results/topics/` |

**How to explain TF-IDF and NMF simply:**

- **TF-IDF:** Scores how important each word is in each document compared to the whole collection. Common words like "the" are ignored; distinctive words like "hallucination" or "copilot" are highlighted.
- **NMF:** Groups documents that share similar word patterns into topics. Each topic gets a set of keywords and a provisional name.

**Key findings:**

| Finding | Value |
|---|---|
| Number of topics selected | 10 |
| Largest topic | ~29% of documents |
| Mean topic confidence | 64.8% |
| Low-confidence assignments | 124 posts |

**Topic names are provisional.** They are generated automatically from keywords and must be reviewed with human judgment before being used in a final report.

---

### Step 10 — Interactive Dashboard

| | |
|---|---|
| **What?** | A web application that lets anyone explore the results with filters and charts. |
| **Why?** | Tables and CSV files are hard to present. A dashboard makes the project demonstrable in a meeting. |
| **How?** | Streamlit + Plotly (`dashboard/app.py`). |
| **Output?** | Running app at `http://localhost:8501` |

**How to launch:**

```bash
python -m streamlit run dashboard/app.py
```

**What the dashboard shows:**

- Overview: record counts, platform breakdown, AI tool mentions
- Sentiment: positive/neutral/negative distribution, by platform
- Stress and emotions: stress direction, primary emotions, likely causes
- Topics: topic distribution, keywords, sentiment within topics
- Time trends: how discussions evolve month by month
- Data explorer: search, filter, and export records as CSV

**Demo workflow for a supervisor meeting:**

1. Show the overview page (2,666 records, platform breakdown).
2. Filter by platform (e.g., Reddit only) and show how sentiment changes.
3. Open the stress section and explain that only 15.5% of posts contain explicit stress language.
4. Show topic keywords and representative example posts.
5. Use the time trend to show how discussion volume changes.
6. End with limitations (sample bias, model limitations, no clinical diagnosis).

---

### Step 11 — Automated Tests

| | |
|---|---|
| **What?** | Verify that the data cleaning script works correctly in normal, edge, and error cases. |
| **Why?** | Reproducibility — another person (or the supervisor) should be able to run the pipeline and get the same results. |
| **How?** | Pytest test suite. |
| **Output?** | `tests/test_prepare_analysis_dataset.py` — **4 tests, all passing** |

**Test scenarios:**

1. A valid AI-related post is processed correctly.
2. A post without an ID or date does not crash the pipeline.
3. An exact duplicate is detected and excluded.
4. Malformed JSON produces a clear error message.

**How to run:**

```bash
python -m pytest tests/test_prepare_analysis_dataset.py -v
```

---

## 4. Summary Table — All Steps at a Glance

| Step | Name | Input | Output | Records |
|---:|---|---|---|---:|
| 1 | Problem definition | Literature, planning | `docs/01_problem_definition.md` | — |
| 2 | Data collection | Public APIs and datasets | `data/raw/combined_reddit_posts.jsonl` | 5,406 |
| 3 | Data understanding | Raw dataset | `notebooks/02_data_understanding.ipynb` | 5,406 |
| 4 | Cleaning and preprocessing | Raw dataset | `data/processed/analysis_ready_posts.jsonl` | 2,666 |
| 5 | Exploratory data analysis | Clean dataset | `data/results/eda/`, `figures/eda/` | 2,666 |
| 6 | Sentiment analysis (VADER + Transformer) | Clean dataset | `data/results/sentiment/` | 2,666 |
| 7 | LLM-assisted validation | Sentiment outputs | Comparison metrics | Sample |
| 8 | Stress and emotion analysis | Sentiment + clean dataset | `data/results/emotions/` | 2,666 |
| 9 | Topic modeling | Emotion-enriched dataset | `data/results/topics/` | 2,666 |
| 10 | Dashboard | All result files | `dashboard/app.py` | 2,666 |
| 11 | Automated tests | Cleaning script | 4 passing tests | — |

---

## 5. Frequently Asked Questions (for Supervisors)

### "Does this tell us how ALL developers feel about AI?"

**No.** The data comes from public posts on specific platforms. Reddit alone represents 76% of the sample. The results describe patterns in **this collected sample**, not the entire developer population.

### "Can we say AI tools cause stress?"

**No.** The project finds **language about stress** in posts that also mention AI tools. That is an association in text, not proof of causation. A developer might be stressed for many reasons unrelated to AI.

### "Which sentiment method should we trust?"

The **Transformer model** performed slightly better in the LLM-assisted comparison and is used as the primary method. However, both methods disagree often (62% of the time), which shows that sentiment analysis of technical text is inherently difficult.

### "What does 'LLM-assisted validation' mean?"

An LLM (large language model) was used to generate reference labels for a sample of posts. This helps compare VADER and Transformer, but these labels are **not the same as expert human annotation**. They are a practical substitute given time constraints.

### "What is the difference between sentiment and stress?"

- **Sentiment** = overall tone (positive/neutral/negative). Example: "Copilot saves me time" → Positive.
- **Stress** = explicit emotional language (anxiety, burnout, fear). Example: "I'm exhausted from reviewing AI code" → Stress signal detected.
- A post can be negative without any stress signal, and vice versa.

### "Why did 5,406 records become 2,666?"

About half the raw data was not truly about AI (keyword false matches like "email" or "training"). Duplicates and very short posts were also removed. This filtering improves analysis quality.

### "Can someone else reproduce this work?"

Yes. The README contains installation instructions, the cleaning script is automated and tested, and every notebook can be re-run from the project root. All steps are documented in the `docs/` folder.

---

## 6. Where to Find More Detail

| Topic | Document |
|---|---|
| Problem definition | `docs/01_problem_definition.md` |
| Data collection | `docs/02_data_collection.md` |
| Cleaning and preprocessing | `docs/03_data_cleaning_and_preprocessing.md` |
| Exploratory data analysis | `docs/04_exploratory_data_analysis.md` |
| Stress and emotion analysis | `docs/06_stress_and_emotion_analysis.md` |
| Topic modeling | `docs/07_topic_modeling.md` |
| Dashboard and final report | `docs/08_dashboard_and_final_reporting.md` |
| Final report template | `docs/final_report_template.md` |
| Full technical README | `README.md` |

---

## 7. Current Project Status

| Phase | Status |
|---|---|
| Problem definition | Completed |
| Data collection | Completed |
| Data understanding | Completed |
| Cleaning and preprocessing | Completed |
| Exploratory data analysis | Completed |
| Sentiment analysis (VADER + Transformer) | Completed |
| LLM-assisted validation | Completed |
| Stress and emotion analysis | Completed |
| Topic modeling | Completed |
| Automated tests (4/4 passing) | Completed |
| Streamlit dashboard | Completed and functional locally |
| Supervisor documentation | Completed |
| Public deployment | Pending |
| Final internship report | In progress |

---


