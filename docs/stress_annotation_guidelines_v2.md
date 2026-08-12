# Stress Annotation Guidelines — Version 2

## Purpose

These annotations provide an LLM-assisted reference set for evaluating
stress-detection methods on developer discussions about AI.

They are exploratory reference annotations and are not human or clinical
ground truth.

## Allowed Labels

- Stress
- No stress
- Unclear

## Stress

Assign `Stress` only when the author personally expresses evidence of
psychological pressure or distress.

Examples include:

- worry or anxiety
- feeling overwhelmed
- fear affecting the author personally
- exhaustion or burnout
- strong work or deadline pressure
- inability to cope
- feeling useless or hopeless
- personal job insecurity
- persistent frustration accompanied by distress

A negative tone alone is not sufficient.

## No stress

Assign `No stress` when the text contains no clear evidence that the author
is personally experiencing psychological stress.

This includes:

- technical complaints
- bugs or software problems
- criticism of AI tools
- negative opinions
- anger or annoyance without distress
- sarcasm
- disagreement
- negative news
- discussion of another person's stress
- general discussion of job replacement
- technical confusion without clear distress

## Unclear

Assign `Unclear` when the text is ambiguous or there is insufficient evidence
to determine whether the author is personally experiencing stress.

Do not guess.

## Important Distinctions

Negative sentiment is not automatically stress.

Anger is not automatically stress.

Fear mentioned abstractly is not automatically stress.

Technical frustration is not automatically stress.

The stress signal must concern the author's own psychological state,
pressure, worry, anxiety, overwhelm, exhaustion, or distress.

## Annotation Independence

The annotator must receive only the original post text.

Do not provide:

- VADER sentiment
- Transformer sentiment
- Dreaddit prediction
- Dreaddit probability
- rule-based stress prediction
- previous custom-model prediction

This prevents model predictions from influencing the reference annotation.

## Confidence

Each annotation receives one confidence level:

- High
- Medium
- Low

## Rationale

Provide one short sentence explaining the decision.

Do not diagnose mental-health conditions.

## Output Fields

For each record return:

- annotation_id
- reference_stress_label
- annotation_confidence
- annotation_rationale
