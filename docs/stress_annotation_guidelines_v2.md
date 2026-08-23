# Stress Annotation Guidelines — Version 2

## Purpose

Provide an **LLM-assisted developer-domain reference set** for evaluating Stress-detection methods.

These labels are:
- exploratory reference annotations;
- not independent human gold-standard labels;
- not clinical diagnoses.

## Allowed labels

- `Stress`
- `No stress`
- `Unclear`

## Stress

Use `Stress` only when the author personally expresses evidence of psychological pressure or distress, such as:
- anxiety or worry;
- overwhelm;
- exhaustion/burnout;
- inability to cope;
- strong deadline/work pressure;
- personally experienced job insecurity;
- personal fear or hopelessness.

A negative tone alone is insufficient.

## No stress

Use `No stress` when there is no clear personal psychological-distress signal, including:
- technical complaints;
- bugs;
- AI criticism;
- anger/annoyance without distress;
- sarcasm;
- general AI-risk discussion;
- job-replacement discussion without personal distress;
- technical confusion.

## Unclear

Use when evidence is insufficient or ambiguous. Do not guess.

## Annotation independence

Annotators receive only the original post text.

Do not provide:
- VADER label;
- Transformer label;
- Dreaddit prediction/probability;
- hybrid prediction;
- previous rule-based or custom-model output.

## Final reference-set composition

- 600 annotated developer-domain posts
- 588 No stress
- 8 Stress
- 4 Unclear
- 596 evaluable for binary metrics

## Output fields

- annotation ID
- reference Stress label
- confidence
- short rationale
