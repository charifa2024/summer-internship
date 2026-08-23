# 02 — Data Collection

## Objective

Create a multi-source corpus of public developer-oriented discussions about AI tools and AI coding assistants.

## Sources

| Source | Platform | Method | Records |
|---|---|---|---:|
| Arctic Shift | Reddit | Public Reddit archive | 305 |
| Hugging Face Reddit dataset | Reddit | Public dataset | 2,000 |
| `divde/sentiment_posts` | Mixed public social data | Public dataset | 141 |
| Hacker News | Hacker News | Public API | 236 |
| GitHub Issues | GitHub | Public REST API | 1,616 |
| Stack Overflow | Stack Overflow | Stack Exchange API | 1,108 |
| **Total** |  |  | **5,406** |

## Main AI concepts

Collection and relevance logic targeted terms such as:
- ChatGPT / OpenAI
- GitHub Copilot
- Claude / Anthropic
- Gemini
- Cursor
- DeepSeek
- Llama / Mistral
- LLMs
- generative AI
- AI agents

## Main collected fields

Depending on source availability:
- source identifier
- platform/source
- community/repository
- title
- body text
- author
- creation date
- score
- comments/answers
- source URL

## Raw-data principle

The raw collection is preserved and is not overwritten by cleaning or modeling.

Primary raw dataset:

```text
data/raw/combined_reddit_posts.jsonl
```

## Sampling limitation

The collection is intentionally multi-source but not balanced. GitHub Issues and the Hugging Face Reddit corpus contribute much larger shares than some other sources. The corpus therefore describes the collected public technical discourse and must not be interpreted as a probability sample of all developers.

## Decision

Collection is frozen at **5,406 raw records**. All scientific filtering is performed in the preprocessing stage.
