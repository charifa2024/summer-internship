
import requests

import json
import time
from datetime import datetime, timezone
from pathlib import Path

class GitHubCollector:
    def __init__(self, output_path="data/raw/github_posts.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.keywords = [
            "ai", "copilot", "chatgpt", "cursor", "agent", "burnout", "stress",
            "replaced", "layoff", "automation", "anxious", "job security"
        ]
        # Popular dev repos to check (expanded list!)
        self.repos = [
            "microsoft/vscode", "facebook/react", "vuejs/core", "angular/angular",
            "tensorflow/tensorflow", "pytorch/pytorch", "huggingface/transformers",
            "golang/go", "python/cpython", "rust-lang/rust", "nodejs/node",
            "docker/docker-ce", "kubernetes/kubernetes", "git/git",
            "vercel/next.js", "sveltejs/svelte", "tailwindlabs/tailwindcss"
        ]
        self.posts = []
        
    def _is_relevant(self, text):
        if not text:
            return False
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.keywords)
    
    def _search_issues(self, repo, per_page=100, pages=2):
        """Search issues in a repo"""
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {
            "state": "all",
            "per_page": per_page,
            "sort": "updated",
            "direction": "desc"
        }
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "AI-Developer-Sentiment/1.0 (internship research)"
        }
        
        for page in range(1, pages+1):
            params["page"] = page
            try:
                response = requests.get(url, params=params, headers=headers, timeout=15)
                if response.status_code == 403:
                    print(f"  Rate limited for {repo}, skipping...")
                    break
                response.raise_for_status()
                issues = response.json()
                if not issues:
                    break
                    
                for issue in issues:
                    # Skip pull requests (they have a "pull_request" key)
                    if "pull_request" in issue:
                        continue
                        
                    title = issue.get("title", "")
                    body = issue.get("body", "")
                    if self._is_relevant(title) or self._is_relevant(body):
                        normalized = {
                            "id": str(issue.get("id")),
                            "subreddit": f"github/{repo}",
                            "title": title,
                            "selftext": body,
                            "score": issue.get("comments"),
                            "num_comments": issue.get("comments"),
                            "created_utc": int(datetime.fromisoformat(issue.get("created_at").replace("Z", "+00:00")).timestamp()),
                            "created_iso": issue.get("created_at"),
                            "author": issue.get("user", {}).get("login"),
                            "permalink": issue.get("html_url"),
                            "url": issue.get("html_url"),
                            "source": "github_issues"
                        }
                        self.posts.append(normalized)
                time.sleep(1)  # Be nice to GitHub's API
            except Exception as e:
                print(f"  Error fetching issues for {repo} (page {page}): {e}")
                break
    
    def collect(self):
        print("Collecting GitHub issues from popular repos...")
        for idx, repo in enumerate(self.repos):
            print(f"Processing repo {idx+1}/{len(self.repos)}: {repo}")
            self._search_issues(repo, per_page=100, pages=3)
        print(f"Collected {len(self.posts)} relevant GitHub issues!")
        return self
    
    def save(self):
        existing_ids = set()
        if self.output_path.exists():
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    post = json.loads(line)
                    existing_ids.add(post.get("id"))
        
        new_posts = [p for p in self.posts if p["id"] not in existing_ids]
        with open(self.output_path, "a", encoding="utf-8") as f:
            for post in new_posts:
                f.write(json.dumps(post, ensure_ascii=False) + "\n")
        print(f"Saved {len(new_posts)} new posts to {self.output_path}")
        return self.output_path

if __name__ == "__main__":
    collector = GitHubCollector()
    # Override to collect more pages per repo
    for idx, repo in enumerate(collector.repos):
        print(f"Processing repo {idx+1}/{len(collector.repos)}: {repo}")
        collector._search_issues(repo, per_page=100, pages=5)  # 5 pages instead of 3!
    print(f"Collected {len(collector.posts)} relevant GitHub issues!")
    collector.save()
