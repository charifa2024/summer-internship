"""
Prepare an analysis-ready dataset for the AI Developer Sentiment project.

Key principles
--------------
1. Preserve the original text.
2. Create separate cleaned text for NLP instead of overwriting raw fields.
3. Use source-qualified IDs.
4. Detect exact text duplicates.
5. Use word-boundary AI relevance rules to avoid matching "ai" inside
   unrelated words such as "again", "email", and "maintain".
6. Preserve source-provided sentiment labels separately from future model output.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd


AI_PATTERNS = {
    "ChatGPT": r"\bchat\s*gpt\b|\bchatgpt\b",
    "OpenAI": r"\bopenai\b",
    "Claude/Anthropic": r"\bclaude\b|\banthropic\b",
    "Gemini": r"\bgemini\b",
    "GitHub Copilot": r"\bgithub\s+copilot\b|\bcopilot\b",
    "Cursor": r"\bcursor(?:\s+ai)?\b",
    "DeepSeek": r"\bdeepseek\b",
    "LLM": r"\bllms?\b|\blarge language models?\b",
    "Generative AI": r"\bgenerative\s+ai\b|\bgenai\b",
    "Artificial Intelligence": r"\bartificial intelligence\b",
    "AI": r"(?<![A-Za-z0-9_])ai(?![A-Za-z0-9_])",
    "GPT": r"\bgpt(?:-\d+(?:\.\d+)?)?\b",
    "Llama": r"\bllama(?:\s*\d+(?:\.\d+)?)?\b",
    "Mistral": r"\bmistral\b",
    "Qwen": r"\bqwen\b",
    "Codeium": r"\bcodeium\b",
    "Tabnine": r"\btabnine\b",
    "Aider": r"\baider\b",
    "AI coding assistant": (
        r"\bai[-\s]+(?:coding|code)\s+assistants?\b|"
        r"\bcoding\s+assistants?\b"
    ),
    "AI agent": r"\bai[-\s]+agents?\b|\bagentic\b",
}

THEME_PATTERNS = {
    "stress_anxiety": (
        r"\bstress(?:ed|ful)?\b|\banxi(?:ety|ous)\b|"
        r"\boverwhelm(?:ed|ing)?\b|\bworried?\b|\bworry\b|"
        r"\bfear(?:ed|ful|ing)?\b"
    ),
    "burnout": r"\bburn(?:ed|t)?\s*out\b|\bburnout\b",
    "job_security": (
        r"\bjob security\b|\blayoffs?\b|\breplac(?:e|ed|ement|ing)\b|"
        r"\bunemploy(?:ed|ment)\b|\bcareer\b"
    ),
    "productivity": (
        r"\bproductiv(?:e|ity)\b|\befficien(?:t|cy)\b|"
        r"\bsav(?:e|ed|ing)\s+time\b"
    ),
    "trust_quality": (
        r"\bhallucinat(?:e|ed|ion|ions|ing)\b|\btrust\b|"
        r"\breliable\b|\baccuracy\b|\bbugs?\b|\berrors?\b"
    ),
    "privacy_security": (
        r"\bprivacy\b|\bsecurity\b|\bdata leak\b|"
        r"\bconfidential\b|\bproprietary\b"
    ),
}

COMPILED_AI = {
    name: re.compile(pattern, flags=re.IGNORECASE)
    for name, pattern in AI_PATTERNS.items()
}
COMPILED_THEMES = {
    name: re.compile(pattern, flags=re.IGNORECASE)
    for name, pattern in THEME_PATTERNS.items()
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
LEXICAL_RE = re.compile(r"[^a-z0-9']+")


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc
    return pd.DataFrame(records)


def coalesce_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series("", index=df.index, dtype="object")
    for column in columns:
        if column not in df.columns:
            continue
        candidate = df[column].fillna("").astype(str).str.strip()
        result = result.mask(result.eq(""), candidate)
    return result


def basic_clean(text: str) -> str:
    """Remove markup and URLs while preserving case, punctuation, and negation."""
    text = html.unescape(str(text))
    text = URL_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def lexical_clean(text: str) -> str:
    """Lowercased lexical version for TF-IDF/topic modeling."""
    text = basic_clean(text).lower()
    text = LEXICAL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def platform_from_source(row: pd.Series) -> str:
    source = str(row.get("source") or "").lower()
    subreddit = str(row.get("subreddit") or "").lower()

    if source == "arctic_shift" or "uit-sentiment-dataset-reddit" in source:
        return "Reddit"
    if "github" in source or subreddit.startswith("github/"):
        return "GitHub Issues"
    if "stackoverflow" in source or subreddit == "stackoverflow":
        return "Stack Overflow"
    if "hackernews" in source or subreddit == "hackernews":
        return "Hacker News"
    if "divde" in source:
        return "Other social dataset"
    return "Unknown"


def parse_dates(df: pd.DataFrame) -> pd.Series:
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    for column in ["created_iso", "created_at", "source_date"]:
        if column in df.columns:
            candidate = pd.to_datetime(df[column], errors="coerce", utc=True)
            parsed = parsed.fillna(candidate)

    if "created_utc" in df.columns:
        candidate = pd.to_datetime(
            pd.to_numeric(df["created_utc"], errors="coerce"),
            unit="s",
            errors="coerce",
            utc=True,
        )
        parsed = parsed.fillna(candidate)

    return parsed


def extract_matches(text: str, patterns: dict[str, re.Pattern]) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def prepare_dataset(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_jsonl(raw_path)
    df["_original_order"] = range(len(df))

    # Canonical fields: preserve source-specific originals but add one shared schema.
    df["source"] = df.get("source", pd.Series("", index=df.index)).fillna("unknown")
    df["source_id"] = coalesce_text(df, ["id", "post_id", "sample_id"])
    df["title_raw"] = coalesce_text(df, ["title"])
    df["body_raw"] = coalesce_text(df, ["selftext", "text"])

    df["full_text_raw"] = (
        df["title_raw"].str.strip() + " " + df["body_raw"].str.strip()
    ).str.strip()
    df["text_clean_basic"] = df["full_text_raw"].map(basic_clean)
    df["text_clean_lexical"] = df["full_text_raw"].map(lexical_clean)

    # Generate a stable fallback ID for datasets that provide no identifier.
    text_hash = df["text_clean_basic"].str.lower().map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    )
    df["source_id"] = df["source_id"].mask(
        df["source_id"].eq(""),
        "text-" + text_hash,
    )
    df["record_id"] = df["source"].astype(str) + "::" + df["source_id"]

    df["platform"] = df.apply(platform_from_source, axis=1)
    df["community"] = coalesce_text(df, ["subreddit"])
    df["created_at_utc"] = parse_dates(df)
    df["date_known"] = df["created_at_utc"].notna()
    df["is_recent_2025_plus"] = (
        df["created_at_utc"].ge(pd.Timestamp("2025-01-01", tz="UTC"))
    )

    df["ai_tools"] = df["text_clean_basic"].map(
        lambda text: extract_matches(text, COMPILED_AI)
    )
    df["is_ai_relevant"] = df["ai_tools"].map(bool)
    df["research_themes"] = df["text_clean_basic"].map(
        lambda text: extract_matches(text, COMPILED_THEMES)
    )
    df["has_emotion_or_risk_signal"] = df["research_themes"].map(bool)

    df["text_length_chars"] = df["text_clean_basic"].str.len()
    df["word_count"] = df["text_clean_basic"].str.findall(r"\b\w+\b").str.len()

    # Exact-text dedupe after normalization. IDs alone are not enough across sources.
    df["text_hash"] = df["text_clean_basic"].str.lower().map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    df["is_exact_text_duplicate"] = df.duplicated("text_hash", keep="first")

    df["analysis_eligible"] = (
        df["is_ai_relevant"]
        & ~df["is_exact_text_duplicate"]
        & df["word_count"].ge(5)
    )

    ready = df.loc[df["analysis_eligible"]].copy()
    excluded = df.loc[~df["is_ai_relevant"]].copy()

    # Keep useful shared fields first. Source-specific fields remain available afterward.
    first_columns = [
        "record_id",
        "source_id",
        "source",
        "platform",
        "community",
        "created_at_utc",
        "date_known",
        "is_recent_2025_plus",
        "title_raw",
        "body_raw",
        "full_text_raw",
        "text_clean_basic",
        "text_clean_lexical",
        "ai_tools",
        "research_themes",
        "has_emotion_or_risk_signal",
        "text_length_chars",
        "word_count",
        "score",
        "num_comments",
        "author",
        "permalink",
        "url",
        "sentiment_label",
        "prediction",
        "confidence",
        "is_exact_text_duplicate",
        "analysis_eligible",
    ]
    ordered = [c for c in first_columns if c in ready.columns]
    remaining = [
        c for c in ready.columns
        if c not in ordered and not c.startswith("_")
    ]
    ready = ready[ordered + remaining]
    excluded = excluded[[c for c in ordered + remaining if c in excluded.columns]]

    summary_rows = [
        ("raw_rows", len(df)),
        ("raw_columns", len(df.columns)),
        ("invalid_json_lines", 0),
        ("missing_original_id_rows", int(
            coalesce_text(df, ["id", "post_id", "sample_id"]).eq("").sum()
        )),
        ("exact_text_duplicate_rows", int(df["is_exact_text_duplicate"].sum())),
        ("strict_ai_relevant_rows", int(df["is_ai_relevant"].sum())),
        ("low_relevance_rows", int((~df["is_ai_relevant"]).sum())),
        ("analysis_ready_rows", len(ready)),
        ("missing_date_rows", int((~df["date_known"]).sum())),
        ("pre_2025_rows", int(
            (df["date_known"] & ~df["is_recent_2025_plus"]).sum()
        )),
        ("emotion_or_risk_signal_rows_in_ready", int(
            ready["has_emotion_or_risk_signal"].sum()
        )),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    return df, ready, excluded, summary


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    output = df.copy()
    if "created_at_utc" in output.columns:
        output["created_at_utc"] = output["created_at_utc"].map(
            lambda value: value.isoformat() if pd.notna(value) else None
        )
    output.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/combined_reddit_posts.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full, ready, excluded, summary = prepare_dataset(args.input)
    write_jsonl(ready, args.output_dir / "analysis_ready_posts.jsonl")
    write_jsonl(
        excluded,
        args.output_dir / "excluded_low_relevance_posts.jsonl",
    )
    summary.to_csv(args.output_dir / "data_quality_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nAnalysis-ready rows: {len(ready)}")
    print("\nRows by platform:")
    print(ready["platform"].value_counts().to_string())


if __name__ == "__main__":
    main()
