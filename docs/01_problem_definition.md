# 01 — Problem Definition

## Project Title

**Data-Driven Analysis of Developers' Perceptions and Emotional Trends Toward Artificial Intelligence on Social Platforms**

## 1. Project Context

Artificial intelligence tools are increasingly integrated into software development workflows. Developers now use tools such as ChatGPT, GitHub Copilot, Cursor, Claude, Gemini, DeepSeek, large language models, and AI agents to generate code, debug applications, search for technical information, and automate repetitive tasks.

These tools may improve productivity and learning, but they also generate concerns related to reliability, hallucinations, privacy, job security, stress, anxiety, and possible overdependence on automated systems.

Developers frequently discuss these experiences on platforms such as Reddit, GitHub Issues, Stack Overflow, Hacker News, and public discussion datasets. These discussions provide useful textual data for understanding how developers perceive modern AI tools.

## 2. Problem Statement

There is a large volume of online discussion about artificial intelligence in software development, but these discussions are distributed across different platforms and expressed in unstructured natural language.

It is therefore difficult to manually determine:

- whether developers express positive, negative, or neutral opinions;
- which AI tools are discussed most frequently;
- which concerns and benefits appear most often;
- how opinions differ between platforms;
- which themes dominate the discussions;
- whether emotional trends change over time.

A complete Natural Language Processing pipeline is needed to collect, clean, analyze, and visualize these discussions.

## 3. Main Research Question

> **How do software developers perceive modern AI tools and AI coding assistants?**

## 4. Secondary Questions

1. What sentiments are most frequently expressed by developers toward AI tools?
2. Which AI tools are mentioned most often?
3. What are the main benefits identified by developers?
4. What are the main concerns, including stress, anxiety, reliability, privacy, and job security?
5. What discussion topics appear most frequently?
6. How do sentiment and topics differ between platforms?
7. How do discussions evolve over time when date information is available?

## 5. General Objective

The general objective is to build a complete and reproducible Data Science and NLP prototype that analyzes developers' opinions, sentiments, emotional signals, and discussion topics related to modern artificial intelligence tools.

## 6. Specific Objectives

- define relevant platforms, communities, and search keywords;
- collect developer-generated discussions from documented public sources;
- standardize the data into a common structure;
- inspect missing values, duplicates, dates, and text quality;
- clean and preprocess textual data without destroying the original text;
- perform exploratory data analysis;
- compare at least two sentiment-analysis approaches;
- identify dominant themes using topic modeling or text clustering;
- create an interactive dashboard with filters and representative examples;
- document the methodology, limitations, tests, and reproducibility instructions.

## 7. Scope of the Project

### Included

The project includes discussions related to:

- ChatGPT, OpenAI, GitHub Copilot, Cursor, Claude, Gemini, and DeepSeek;
- large language models, generative AI, AI coding assistants, and AI agents;
- productivity, hallucinations, reliability, privacy, security, stress, anxiety, burnout, and job security.

Current sources:

- Reddit through Arctic Shift;
- public Hugging Face datasets;
- GitHub Issues;
- Stack Overflow;
- Hacker News.

### Excluded

The first prototype does not aim to:

- represent the opinion of every software developer;
- prove causal effects of AI tools;
- create a real-time monitoring system;
- train a large language model from scratch;
- collect private messages or restricted personal data;
- make psychological or medical diagnoses;
- make employment predictions about individual developers.

## 8. Unit of Analysis

The main unit of analysis is one developer-generated textual record, such as a Reddit post, GitHub issue, Stack Overflow question, Hacker News story, or record from a documented public dataset.

Each record may contain a title, body text, date, platform, author, score, number of comments, and original URL, depending on source availability.

## 9. Expected Outputs

1. A documented raw multi-source dataset.
2. A cleaned and analysis-ready dataset.
3. Data-quality and exploratory-analysis notebooks.
4. Sentiment-analysis results.
5. Topic-modeling or clustering results.
6. Visualizations and summary tables.
7. An interactive Streamlit dashboard.
8. Test scenarios and validation evidence.
9. Technical documentation.
10. A final internship report and presentation.

## 10. Success Criteria

The project will be considered successful when:

- dataset sources and transformations are documented;
- the cleaning pipeline is reproducible;
- the final dataset contains valid and relevant text;
- the EDA provides interpretable visualizations;
- at least two analysis methods are compared;
- the major sentiments and topics are interpreted;
- the dashboard runs and supports a simple demonstration;
- nominal, edge-case, and error scenarios are tested;
- limitations are clearly reported;
- another user can install and run the prototype using the documentation.

## 11. Main Constraints

- official Reddit API access was not obtained during the collection period;
- the sources use different schemas and sampling methods;
- some records have missing dates or metadata;
- platform sizes are imbalanced;
- generic sentiment models may misunderstand technical language, sarcasm, and error messages;
- the project must be completed within a limited internship deadline.

## 12. Current Status

| Item | Status |
|---|---|
| Problem definition | Completed |
| Research question | Completed |
| Objectives and scope | Completed |
| Data-source selection | Completed |
| Data collection | Completed |
| Data understanding | Next phase |
| Cleaning and preprocessing | To be reviewed and validated |
| EDA | Planned |
| Sentiment analysis | Planned |
| Topic modeling | Planned |
| Dashboard | Planned |
| Final report | In progress |

## 13. Decision for the Next Phase

The official raw dataset must now be frozen and inspected without being overwritten.

The next phase is **Data Understanding**, where the project will verify:

- the number of records and columns;
- available fields;
- platform distribution;
- missing values;
- duplicate records;
- date coverage;
- text lengths;
- language consistency;
- source-specific quality problems.

Only after this inspection will the final cleaning and preprocessing rules be confirmed.
