# Synthetic Developer Stress Dataset — Specification

## Purpose

This dataset is designed to augment stress-classification training data
for developer discussions about Artificial Intelligence.

The synthetic data must NOT be used as evaluation ground truth.

It is intended to help address:

- scarcity of explicit stress examples in real developer discussions;
- class imbalance;
- domain shift between Dreaddit and AI/developer discussions;
- confusion between negative sentiment and actual stress.

## Dataset Size

Total synthetic records: 1,200

- Stress: 600
- No stress: 600

The dataset must remain exactly balanced.

## Labels

Allowed labels:

- Stress
- No stress

No `Unclear` examples will be generated for training.

## Stress Definition

A Stress example must contain evidence that the speaker personally
experiences psychological pressure or distress.

Examples include:

- worry
- anxiety
- overwhelm
- exhaustion
- burnout
- personal fear
- inability to cope
- deadline pressure
- loss of confidence
- personal job insecurity
- skill-obsolescence anxiety
- distress caused by AI-related work

A negative opinion alone is NOT sufficient.

## No Stress Definition

A No stress example may be negative, angry, critical, or frustrated,
but must not contain clear evidence of personal psychological distress.

Examples include:

- technical bug reports
- criticism of AI tools
- model hallucination complaints
- anger
- annoyance
- sarcasm
- technical confusion
- general job-loss discussions
- privacy concerns
- AI news
- model comparisons

## Stress Scenarios

The 600 Stress examples should cover approximately:

1. Job insecurity / replacement fear
2. Skill obsolescence
3. Learning pressure
4. Deadline pressure
5. Debugging overload
6. AI-generated code problems
7. Productivity pressure
8. Burnout / exhaustion
9. Loss of professional confidence
10. Fear of depending too much on AI
11. Academic/project pressure
12. Rapid AI-change anxiety

The examples should be distributed across these scenarios rather than
concentrated in one category.

## Difficult No-Stress Scenarios

The 600 No stress examples must intentionally include difficult negative
cases such as:

1. Angry AI criticism
2. Technical frustration
3. Software bugs
4. Hallucination complaints
5. Model reliability criticism
6. Privacy/security criticism
7. Technical confusion
8. General AI-job discussion
9. Negative AI news
10. Sarcasm
11. Model comparison
12. Product dissatisfaction

These examples are important for teaching:

Negative sentiment != Stress

## Platform Styles

Synthetic posts should imitate different communication formats:

- Reddit discussion
- GitHub Issue
- Stack Overflow question
- Hacker News discussion

The texts should vary naturally in:

- length
- vocabulary
- punctuation
- formality
- technical detail

## AI Contexts

Examples should cover different AI contexts:

- ChatGPT
- Claude
- Gemini
- GitHub Copilot
- Llama
- DeepSeek
- AI agents
- local LLMs
- general generative AI

## Quality Requirements

The generated texts must:

- not repeat the same sentence structure excessively;
- not simply insert stress keywords into templates;
- contain realistic developer/technical language;
- vary from short posts to longer discussions;
- avoid obvious label leakage such as every Stress post containing
  the literal word "stress";
- avoid making every No stress example positive;
- contain hard negative examples;
- avoid copying real posts from the collected dataset.

## Metadata

Each generated record must contain:

- synthetic_id
- text
- stress_label
- scenario
- platform_style
- ai_context
- generation_source
- generation_version

Generation source:

LLM-generated synthetic training data

Generation version:

v1

## Scientific Use

Synthetic examples are TRAINING DATA ONLY.

They must not be used to calculate the final model accuracy as if they
were real ground truth.

Final evaluation must remain separate from synthetic training data.

The experiment will compare:

1. Old 79-label custom model
2. Dreaddit-only model
3. Synthetic-domain model
4. Dreaddit + synthetic hybrid model

The purpose is to investigate whether domain-specific synthetic
augmentation improves stress detection in developer discussions.
