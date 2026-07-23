"""Collect Reddit posts without official API credentials.

Tries public JSON endpoints first (www.reddit.com, then old.reddit.com).
Falls back to PullPush when Reddit blocks unauthenticated requests.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_USER_AGENT = (
    "AI-Developer-Sentiment/1.0 (internship research; contact: local)"
)

PRIORITY_KEYWORDS = [
    "chatgpt",
    "copilot",
    "cursor",
    "layoff",
    "burnout",
    "job security",
]

SUBREDDIT_ALIASES = {
    "ArtificialInteligence": "ArtificialIntelligence",
}


class PublicRedditCollector:
    """Collect and persist Reddit submissions for sentiment analysis."""

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        output_path: str | Path = "data/raw/posts.jsonl",
        request_delay: float = 2.0,
    ) -> None:
        self.user_agent = user_agent
        self.output_path = Path(output_path)
        self.request_delay = request_delay
        self.posts: list[dict[str, Any]] = []
        self.collection_method: str | None = None

    def collect(
        self,
        subreddits: list[str],
        keywords: list[str],
        limit_per_sub: int = 100,
    ) -> "PublicRedditCollector":
        """Fetch posts matching keywords from each subreddit."""
        normalized_subs = [SUBREDDIT_ALIASES.get(s, s) for s in subreddits]
        keywords_lower = [k.lower() for k in keywords]

        for backend in (
            self._collect_via_arctic_shift,
            # self._collect_via_pullpush,  # Disabled due to rate limiting
            # self._collect_via_public_json,  # Disabled due to 403
        ):
            self.posts = []
            try:
                backend(normalized_subs, keywords_lower, limit_per_sub)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                print(f"[{backend.__name__}] blocked or failed (HTTP {status})")
                continue
            except requests.RequestException as exc:
                print(f"[{backend.__name__}] request failed: {exc}")
                continue

            if self.posts:
                self.collection_method = backend.__name__
                existing = self._load_existing_ids()
                before = len(self.posts)
                self.posts = [p for p in self.posts if p["id"] not in existing]
                print(
                    f"[{backend.__name__}] collected {before} posts; "
                    f"{len(self.posts)} new after dedupe"
                )
                return self

        print("All collection backends failed or returned no matching posts.")
        return self

    def _get_with_retry(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int = 5,
    ) -> requests.Response:
        for attempt in range(max_retries):
            response = requests.get(
                url, params=params, headers=headers, timeout=30
            )
            if response.status_code == 429:
                wait = self.request_delay * (2**attempt)
                print(f"[retry] 429 rate limited; sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            return response
        return response

    def _collect_via_public_json(
        self,
        subreddits: list[str],
        keywords_lower: list[str],
        limit_per_sub: int,
    ) -> None:
        """Try www.reddit.com, then old.reddit.com public JSON listings."""
        bases = [
            "https://www.reddit.com/r/{subreddit}/new.json",
            "https://old.reddit.com/r/{subreddit}/new.json",
        ]

        headers = {"User-Agent": self.user_agent}
        matched: list[dict[str, Any]] = []

        for subreddit in subreddits:
            fetched = False
            for base in bases:
                url = base.format(subreddit=subreddit)
                params = {"limit": min(limit_per_sub, 100)}
                response = self._get_with_retry(
                    url, params=params, headers=headers
                )
                if response.status_code == 403:
                    host = "www.reddit.com" if "www" in base else "old.reddit.com"
                    print(f"[public_json] {host} returned 403 for r/{subreddit}")
                    continue
                response.raise_for_status()

                children = response.json().get("data", {}).get("children", [])
                for child in children:
                    post = self._normalize_public_json_post(
                        child.get("data", {}), subreddit
                    )
                    if self._matches_keywords(post, keywords_lower):
                        matched.append(post)
                fetched = True
                time.sleep(self.request_delay)
                break

            if not fetched:
                raise requests.HTTPError(
                    f"403 Client Error: blocked for r/{subreddit}",
                    response=response,
                )

        self.posts = self._dedupe_posts(matched)

    def _collect_via_pullpush(
        self,
        subreddits: list[str],
        keywords_lower: list[str],
        limit_per_sub: int,
    ) -> None:
        """Fallback using PullPush search API (Pushshift-style, no auth)."""
        print("[pullpush] Trying PullPush API fallback...")
        base_url = "https://api.pullpush.io/reddit/search/submission/"
        matched: list[dict[str, Any]] = []
        size = min(limit_per_sub, 100)
        search_terms = self._priority_search_terms(keywords_lower)

        for subreddit in subreddits:
            params = {
                "subreddit": subreddit,
                "size": size,
                "sort": "desc",
                "sort_type": "created_utc",
            }
            batch = self._fetch_pullpush_batch(base_url, params)
            matched.extend(
                post
                for post in batch
                if self._matches_keywords(post, keywords_lower)
            )

            for keyword in search_terms:
                params = {
                    "subreddit": subreddit,
                    "q": keyword,
                    "size": size,
                    "sort": "desc",
                    "sort_type": "created_utc",
                }
                batch = self._fetch_pullpush_batch(base_url, params)
                matched.extend(
                    post
                    for post in batch
                    if self._matches_keywords(post, keywords_lower)
                )

        self.posts = self._dedupe_posts(matched)

    def _fetch_pullpush_batch(
        self, base_url: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        response = self._get_with_retry(base_url, params=params)
        if response.status_code == 429:
            print("[pullpush] still rate limited after retries; skipping request")
            return []
        response.raise_for_status()

        subreddit = params.get("subreddit", "")
        posts = [
            self._normalize_pullpush_post(raw, subreddit)
            for raw in response.json().get("data", [])
        ]
        time.sleep(self.request_delay)
        return posts

    def _collect_via_arctic_shift(
        self,
        subreddits: list[str],
        keywords_lower: list[str],
        limit_per_sub: int,
    ) -> None:
        """Fallback using Arctic Shift API (no auth, higher throughput)."""
        print("[arctic_shift] Trying Arctic Shift API fallback...")
        base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
        matched: list[dict[str, Any]] = []
        size = min(limit_per_sub, 100)

        for subreddit in subreddits:
            print(f"[arctic_shift] Fetching from r/{subreddit}")
            try:
                params = {"subreddit": subreddit, "limit": size, "sort": "desc"}
                batch = self._fetch_arctic_shift_batch(base_url, params, subreddit)
                matched.extend(
                    post
                    for post in batch
                    if self._matches_keywords(post, keywords_lower)
                )
            except Exception as e:
                print(f"[arctic_shift] Skipping r/{subreddit} due to error: {e}")
                continue

        self.posts = self._dedupe_posts(matched)

    def _fetch_arctic_shift_batch(
        self, base_url: str, params: dict[str, Any], subreddit: str
    ) -> list[dict[str, Any]]:
        try:
            response = self._get_with_retry(base_url, params=params)
            if response.status_code == 429:
                print("[arctic_shift] rate limited after retries; skipping request")
                return []
            if response.status_code != 200:
                print(f"[arctic_shift] Request failed with status {response.status_code} for params {params}")
                return []
            data = response.json()
            posts = [
                self._normalize_arctic_shift_post(raw, subreddit)
                for raw in data.get("data", [])
            ]
            time.sleep(self.request_delay)
            return posts
        except Exception as e:
            print(f"[arctic_shift] Failed to fetch batch: {e}")
            return []

    @staticmethod
    def _priority_search_terms(keywords_lower: list[str]) -> list[str]:
        terms = [term for term in PRIORITY_KEYWORDS if term in keywords_lower]
        if terms:
            return terms
        return keywords_lower[:6]

    @staticmethod
    def _matches_keywords(post: dict[str, Any], keywords_lower: list[str]) -> bool:
        text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()
        return any(keyword in text for keyword in keywords_lower)

    @staticmethod
    def _dedupe_posts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for post in posts:
            post_id = post.get("id")
            if not post_id or post_id in seen:
                continue
            seen.add(post_id)
            unique.append(post)
        return unique

    @staticmethod
    def _normalize_public_json_post(
        raw: dict[str, Any], subreddit: str
    ) -> dict[str, Any]:
        created_utc = raw.get("created_utc")
        permalink = raw.get("permalink", "")
        if permalink and not permalink.startswith("http"):
            permalink = f"https://reddit.com{permalink}"
        return {
            "id": raw.get("id"),
            "subreddit": subreddit,
            "title": raw.get("title", ""),
            "selftext": raw.get("selftext", ""),
            "score": raw.get("score"),
            "num_comments": raw.get("num_comments"),
            "created_utc": created_utc,
            "created_iso": PublicRedditCollector._utc_to_iso(created_utc),
            "author": raw.get("author"),
            "permalink": permalink,
            "url": raw.get("url"),
            "source": "reddit_public_json",
        }

    @staticmethod
    def _normalize_pullpush_post(
        raw: dict[str, Any], subreddit: str
    ) -> dict[str, Any]:
        created_utc = raw.get("created_utc")
        permalink = raw.get("permalink") or raw.get("full_link") or ""
        if permalink and not permalink.startswith("http"):
            permalink = f"https://reddit.com{permalink}"

        return {
            "id": raw.get("id"),
            "subreddit": raw.get("subreddit") or subreddit,
            "title": raw.get("title", ""),
            "selftext": raw.get("selftext", ""),
            "score": raw.get("score"),
            "num_comments": raw.get("num_comments"),
            "created_utc": created_utc,
            "created_iso": PublicRedditCollector._utc_to_iso(created_utc),
            "author": raw.get("author"),
            "permalink": permalink,
            "url": raw.get("url"),
            "source": "pullpush",
        }

    @staticmethod
    def _normalize_arctic_shift_post(
        raw: dict[str, Any], subreddit: str
    ) -> dict[str, Any]:
        created_utc = raw.get("created_utc")
        permalink = raw.get("permalink") or ""
        if permalink and not permalink.startswith("http"):
            permalink = f"https://reddit.com{permalink}"
        # Remove duplicate https://reddit.com prefix if present
        if permalink.startswith("https://reddit.comhttps://reddit.com"):
            permalink = permalink.replace("https://reddit.comhttps://reddit.com", "https://reddit.com")
        return {
            "id": raw.get("id"),
            "subreddit": raw.get("subreddit") or subreddit,
            "title": raw.get("title", ""),
            "selftext": raw.get("selftext", ""),
            "score": raw.get("score"),
            "num_comments": raw.get("num_comments"),
            "created_utc": created_utc,
            "created_iso": PublicRedditCollector._utc_to_iso(created_utc),
            "author": raw.get("author"),
            "permalink": permalink,
            "url": raw.get("url"),
            "source": "arctic_shift",
        }

    @staticmethod
    def _utc_to_iso(timestamp: Any) -> str | None:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()

    def _load_existing_ids(self) -> set[str]:
        if not self.output_path.exists():
            return set()

        ids: set[str] = set()
        with self.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                post_id = record.get("id")
                if post_id:
                    ids.add(post_id)
        return ids

    def save_jsonl(self, path: str | Path | None = None) -> Path:
        """Append new posts to JSONL, skipping IDs already on disk."""
        target = Path(path) if path is not None else self.output_path
        target.parent.mkdir(parents=True, exist_ok=True)

        existing_ids = self._load_existing_ids() if target == self.output_path else set()
        new_posts = [p for p in self.posts if p["id"] not in existing_ids]

        with target.open("a", encoding="utf-8") as handle:
            for post in new_posts:
                handle.write(json.dumps(post, ensure_ascii=False) + "\n")

        print(f"Saved {len(new_posts)} new posts to {target}")
        return target

    def load_as_dataframe(self) -> pd.DataFrame:
        """Return collected posts as a DataFrame."""
        return pd.DataFrame(self.posts)

    def load_all_from_jsonl(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load every post stored in the JSONL file."""
        target = Path(path) if path is not None else self.output_path
        if not target.exists():
            return pd.DataFrame()

        records: list[dict[str, Any]] = []
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)


if __name__ == "__main__":
    collector = PublicRedditCollector(
        output_path=Path(__file__).resolve().parents[1] / "data/raw/posts.jsonl"
    )

    SUBREDDITS = [
        "ArtificialInteligence",
        "cscareerquestions",
        "programming",
        "MachineLearning",
        "webdev",
        "ExperiencedDevs",
    ]
    KEYWORDS = [
        "AI",
        "copilot",
        "chatgpt",
        "cursor",
        "agent",
        "burnout",
        "stress",
        "replaced",
        "layoff",
        "automation",
        "anxious",
        "job security",
    ]

    collector.collect(subreddits=SUBREDDITS, keywords=KEYWORDS, limit_per_sub=100)
    if collector.posts:
        collector.save_jsonl()
        df = collector.load_as_dataframe()
        print(f"Method: {collector.collection_method}")
        print(f"Shape: {df.shape}")
        print(df[["subreddit", "title", "created_iso", "score"]].head(10).to_string())
    else:
        print("No posts collected.")
