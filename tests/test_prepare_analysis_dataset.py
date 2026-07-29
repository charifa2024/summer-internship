"""
Tests for src/preprocessing/prepare_analysis_dataset.py.

Run from the project root with:

    python -m pytest tests/test_prepare_analysis_dataset.py -v
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.preprocessing.prepare_analysis_dataset import (
    load_jsonl,
    prepare_dataset,
)


def write_jsonl(path, records):
    """Write test records to a temporary JSONL file."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def summary_value(summary, metric):
    """Read one value from the generated quality summary."""
    return int(
        summary.loc[
            summary["metric"].eq(metric),
            "value",
        ].iloc[0]
    )


def test_nominal_valid_reddit_post(tmp_path):
    """
    Nominal scenario:
    a normal Reddit post about ChatGPT should be cleaned and accepted.
    """
    input_path = tmp_path / "nominal.jsonl"

    write_jsonl(
        input_path,
        [
            {
                "source": "arctic_shift",
                "id": "reddit-1",
                "subreddit": "programming",
                "title": "ChatGPT helps with debugging",
                "selftext": (
                    "It saves time and removes repetitive work. "
                    "https://example.com"
                ),
                "created_utc": 1735689600,
            }
        ],
    )

    full, ready, excluded, summary = prepare_dataset(
        input_path
    )

    assert len(full) == 1
    assert len(ready) == 1
    assert len(excluded) == 0

    row = ready.iloc[0]

    assert row["record_id"] == (
        "arctic_shift::reddit-1"
    )
    assert row["platform"] == "Reddit"
    assert row["date_known"] == True
    assert "ChatGPT" in row["ai_tools"]
    assert "https://example.com" not in (
        row["text_clean_basic"]
    )
    assert row["analysis_eligible"] == True

    assert summary_value(
        summary,
        "analysis_ready_rows",
    ) == 1


def test_boundary_missing_id_and_date(tmp_path):
    """
    Boundary scenario:
    a valid AI-related post with no original ID and no date should not crash.

    The pipeline should generate a stable fallback ID and preserve the
    missing-date information.
    """
    input_path = tmp_path / "boundary.jsonl"

    write_jsonl(
        input_path,
        [
            {
                "source": "divde_sentiment_posts",
                "text": (
                    "Claude helps me write unit tests"
                ),
            }
        ],
    )

    full, ready, excluded, summary = prepare_dataset(
        input_path
    )

    assert len(full) == 1
    assert len(ready) == 1

    row = ready.iloc[0]

    assert row["source_id"].startswith("text-")
    assert row["record_id"].startswith(
        "divde_sentiment_posts::text-"
    )
    assert row["date_known"] == False
    assert pd.isna(row["created_at_utc"])
    assert row["platform"] == (
        "Other social dataset"
    )
    assert "Claude/Anthropic" in row["ai_tools"]

    assert summary_value(
        summary,
        "missing_original_id_rows",
    ) == 1

    assert summary_value(
        summary,
        "missing_date_rows",
    ) == 1


def test_anomaly_exact_duplicate_is_detected(tmp_path):
    """
    Anomaly scenario:
    two records containing the same normalized text should be detected.

    Only the first record should remain analysis-ready.
    """
    input_path = tmp_path / "duplicate.jsonl"

    duplicated_text = (
        "GitHub Copilot helps developers write "
        "reliable unit tests quickly"
    )

    write_jsonl(
        input_path,
        [
            {
                "source": "github",
                "id": "issue-1",
                "text": duplicated_text,
            },
            {
                "source": "github",
                "id": "issue-2",
                "text": duplicated_text,
            },
        ],
    )

    full, ready, excluded, summary = prepare_dataset(
        input_path
    )

    assert len(full) == 2
    assert full[
        "is_exact_text_duplicate"
    ].tolist() == [
        False,
        True,
    ]

    assert len(ready) == 1
    assert ready.iloc[0]["source_id"] == "issue-1"

    assert summary_value(
        summary,
        "exact_text_duplicate_rows",
    ) == 1

    assert summary_value(
        summary,
        "analysis_ready_rows",
    ) == 1


def test_error_invalid_json_reports_line_number(
    tmp_path,
):
    """
    Error scenario:
    malformed JSON should raise a clear ValueError with the line number.
    """
    input_path = tmp_path / "invalid.jsonl"

    input_path.write_text(
        '{"source": "arctic_shift", '
        '"id": "valid-1", '
        '"text": "ChatGPT is useful for testing"}\n'
        '{"source": "broken", invalid json}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid JSON on line 2",
    ):
        load_jsonl(input_path)
