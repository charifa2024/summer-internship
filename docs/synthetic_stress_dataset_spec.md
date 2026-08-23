# Synthetic Developer Stress Dataset — Final Specification

## Purpose

Augment Dreaddit training with developer-style language to reduce domain mismatch and explicitly teach the distinction:

> **Negative sentiment ≠ Stress**

Synthetic examples are **training data only** and must never be used as evaluation ground truth.

## Dataset

- Total: 1,200
- Stress: 600
- No stress: 600

## Stress examples

Must include personal psychological pressure/distress such as:
- overwhelm;
- anxiety;
- exhaustion/burnout;
- deadline pressure;
- loss of confidence;
- personal job insecurity;
- skill-obsolescence anxiety;
- AI-related work pressure.

## Hard No-stress examples

Must intentionally include negative but non-Stress technical language:
- bug reports;
- AI criticism;
- hallucination complaints;
- technical frustration;
- privacy/security criticism;
- confusion;
- general job-loss discussion;
- negative AI news;
- sarcasm;
- product dissatisfaction.

## Platform styles

Examples should vary across:
- Reddit discussion
- GitHub Issue
- Stack Overflow question
- Hacker News discussion

## AI contexts

Examples may include:
- ChatGPT
- Claude
- Gemini
- GitHub Copilot
- Llama
- DeepSeek
- local LLMs
- AI agents
- general generative AI

## Quality requirements

- no copied real posts;
- avoid repetitive templates;
- avoid obvious label leakage;
- vary length and formality;
- include hard negatives;
- preserve realistic technical vocabulary.

## Final scientific use

The final comparison retained for reporting is:
1. Dreaddit-only baseline;
2. Dreaddit + synthetic hybrid.

The final hybrid model is evaluated separately on:
- untouched Dreaddit official test data;
- the developer-domain LLM-assisted reference set.

Synthetic examples do not contribute to final evaluation metrics.
