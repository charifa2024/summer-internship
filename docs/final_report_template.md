# Final Internship Report

## Project Title

**Analysis of Developer Sentiment, Stress, Emotions and Discussion Topics
Related to AI Tools**

## 1. Executive Summary

Summarize:

- the objective;
- the data sources;
- the number of collected and analysis-ready records;
- the main methods;
- the most important findings;
- the main limitations.

Do not write final numerical findings until the notebooks have generated the
actual results.

## 2. Context and Problem Definition

Explain the growing use of AI tools by developers and the importance of
understanding:

- general sentiment;
- explicit stress and emotional signals;
- concerns and sources of relief;
- recurring discussion topics;
- differences across platforms and tools.

## 3. Research Questions

1. What is the overall sentiment toward AI tools?
2. Which explicit stress and emotional signals appear in developer posts?
3. What are the likely contexts associated with those signals?
4. Which recurring topics are discovered automatically?
5. How do sentiment, stress and topics vary by platform, tool and time?

## 4. Data Sources and Collection

Describe the collected sources and include the verified record counts from the
data-source inventory and collection documentation.

## 5. Data Quality and Preprocessing

Describe:

- JSONL loading;
- schema harmonization;
- missing values;
- duplicate detection;
- AI-relevance filtering;
- text cleaning;
- date normalization;
- platform normalization;
- `text_clean_basic`;
- `text_clean_lexical`.

## 6. Exploratory Data Analysis

Present:

- platform distribution;
- source distribution;
- text length;
- AI-tool mentions;
- research themes;
- temporal coverage.

Insert the actual figures generated in `figures/eda/`.

## 7. Sentiment Analysis

### 7.1 VADER

Explain the lexicon-based baseline.

### 7.2 Transformer

Explain the contextual Transformer method.

### 7.3 Provisional Method Comparison

State clearly:

> The reference annotations were generated through LLM-assisted semantic
> review and are not independently verified human ground truth.

Report the actual comparison metrics only after running the validation
notebook.

## 8. Stress and Emotion Analysis

Explain the distinction between sentiment and stress.

Report:

- stress direction;
- primary emotion;
- stress intensity;
- likely stress contexts;
- platform and tool differences;
- sentiment versus stress.

State clearly that this is textual-signal detection, not clinical diagnosis.

## 9. Topic Modeling

Explain:

- TF-IDF;
- NMF;
- candidate topic counts;
- reconstruction error;
- topic diversity;
- selected topic count;
- representative documents;
- topic interpretation.

Replace provisional topic names with readable names after reviewing keywords
and representative texts.

## 10. Dashboard

Describe the Streamlit dashboard and its filters.

Include screenshots of:

- the overview;
- stress and emotion analysis;
- topic analysis;
- time trends;
- data explorer.

## 11. Main Findings

Insert only findings supported by generated outputs.

Recommended structure:

- Finding 1: overall sentiment
- Finding 2: explicit stress or relief signals
- Finding 3: dominant emotion
- Finding 4: main likely stress context
- Finding 5: largest topic
- Finding 6: platform differences
- Finding 7: AI-tool differences
- Finding 8: temporal patterns

## 12. Limitations

Include all of the following:

- Public posts are not representative of all developers.
- Platform sampling is uneven.
- Sentiment labels are model predictions.
- LLM-assisted validation is not human ground truth.
- Rule-based stress analysis can miss implicit emotions.
- Sarcasm and mixed emotions remain difficult.
- Topic names require interpretation.
- One dominant topic may oversimplify a document.
- The analysis detects textual patterns, not medical conditions.
- Observed associations do not prove causation.

## 13. Recommendations

Possible recommendation categories:

- improve human validation;
- create a larger manually annotated sample;
- use multilabel emotion classification;
- compare rule-based and Transformer emotion models;
- monitor changes over time;
- expand data sources;
- improve multilingual coverage;
- deploy the dashboard internally.

## 14. Conclusion

Summarize the complete pipeline and explain how sentiment, stress analysis and
topic modeling provide complementary views.

## 15. Appendices

Include:

- project structure;
- data dictionary;
- model parameters;
- output file list;
- figures;
- sample records;
- installation and dashboard launch instructions.
