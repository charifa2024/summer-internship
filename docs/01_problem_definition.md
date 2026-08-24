# 01 — Problem Definition

## Final project title

**Analysis of Developer-Oriented Discussions on AI Tools**

**Subtitle:** *Sentiment, Emotions, Stress-Related Language, and Topic Modeling in Public Technical Discussions*

## Context

AI tools such as ChatGPT, GitHub Copilot, Claude, Gemini, Cursor, DeepSeek and local LLMs are increasingly discussed in software-development communities. Public technical discussions contain information about perceived usefulness, criticism, confusion, emotional reactions and pressure associated with AI-enabled workflows.

Because these discussions are distributed across platforms and written as unstructured text, a reproducible NLP workflow is needed to analyse them consistently.

## Main research question

> **How do developer-oriented public discussions about AI tools express sentiment, emotions, and stress-related language, and which discussion topics are associated with model-predicted Stress?**

## Secondary questions

1. How differently do lexical and contextual sentiment methods classify the corpus?
2. Which emotion-related language signals are predicted by a multi-label GoEmotions classifier?
3. Can a Dreaddit-trained Stress classifier transfer reliably to developer-oriented technical language?
4. How different is model-predicted Stress from negative sentiment?
5. Which recurring topics are discovered independently from Stress labels?
6. Is topic membership statistically associated with model-predicted Stress?

## General objective

Build a reproducible NLP prototype that moves from data collection to critically validated analytical results and an interactive dashboard.

## Specific objectives

- collect public developer-oriented AI discussions from documented sources;
- harmonize heterogeneous source schemas;
- preserve raw text and build stable analysis text fields;
- apply strict AI relevance filtering, exact-text deduplication and the minimum-length rule;
- freeze one common analysis-ready corpus;
- compare VADER and Transformer sentiment;
- train and evaluate a multi-label GoEmotions model;
- build a dedicated Stress classifier using Dreaddit;
- test synthetic developer-style augmentation;
- evaluate Stress transfer on an LLM-assisted developer-domain reference set;
- merge Sentiment × Emotions × Stress by `record_id`;
- learn NMF topics independently on all final posts;
- statistically test Topic × model-predicted Stress association;
- provide qualitative evidence and explicit model limitations in a Streamlit dashboard.

## Unit of analysis

One public textual record, such as a Reddit discussion, GitHub issue, Stack Overflow question, Hacker News post, or record from a documented public dataset.

## Scope and boundaries

### Included
- public English-language technical/developer-oriented discussions;
- AI tools, coding assistants, LLMs and generative-AI workflows;
- sentiment, multi-label emotions, Stress-related language and discussion topics.

### Not claimed
- representativeness of all software developers;
- verified professional identity for every author;
- causal effects of AI tools;
- psychological or medical diagnosis;
- individual employment or mental-health prediction.

## Success criteria

The final project is considered successful when:
- collection and preprocessing are reproducible;
- all downstream analyses use the same 2,666 record IDs;
- model choices are evaluated rather than assumed;
- final results include limitations and error analysis;
- topic associations are statistically tested with effect size and multiple-comparison correction;
- dashboard and report use frozen final outputs;
- another user can reproduce the workflow from the repository documentation.
