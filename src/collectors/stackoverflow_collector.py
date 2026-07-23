
import requests
import json
import time
from datetime import datetime, timezone
from pathlib import Path

class StackOverflowCollector:
    def __init__(self, output_path="data/raw/stackoverflow_posts.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.keywords = [
            "ai", "copilot", "chatgpt", "cursor", "agent", "burnout", "stress",
            "replaced", "layoff", "automation", "anxious", "job security"
        ]
        self.posts = []
        
    def _is_relevant(self, text):
        if not text:
            return False
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.keywords)
    
    def collect(self, pages=10, pagesize=100):
        print("Collecting older Stack Overflow questions...")
        url = "https://api.stackexchange.com/2.3/questions"
        # Get posts from 2025 onwards, sorted by votes instead of creation
        from datetime import datetime
        import time
        two_years_ago = int(time.mktime(datetime(2025, 1, 1).timetuple()))
        params = {
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "pagesize": pagesize,
            "fromdate": two_years_ago,
            "filter": "withbody"  # Get question body too
        }
        headers = {
            "User-Agent": "AI-Developer-Sentiment/1.0 (internship research)"
        }
        
        for page in range(1, pages+1):
            params["page"] = page
            try:
                response = requests.get(url, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()
                questions = data.get("items", [])
                if not questions:
                    break
                    
                for q in questions:
                    title = q.get("title", "")
                    body = q.get("body", "")
                    if self._is_relevant(title) or self._is_relevant(body):
                        normalized = {
                            "id": str(q.get("question_id")),
                            "subreddit": "stackoverflow",
                            "title": title,
                            "selftext": body,
                            "score": q.get("score"),
                            "num_comments": q.get("answer_count"),
                            "created_utc": q.get("creation_date"),
                            "created_iso": datetime.fromtimestamp(q.get("creation_date"), tz=timezone.utc).isoformat(),
                            "author": q.get("owner", {}).get("display_name"),
                            "permalink": q.get("link"),
                            "url": q.get("link"),
                            "source": "stackoverflow"
                        }
                        self.posts.append(normalized)
                # Backoff if needed
                if "backoff" in data:
                    time.sleep(data["backoff"])
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"  Error fetching page {page}: {e}")
                break
                
        print(f"Collected {len(self.posts)} relevant Stack Overflow questions!")
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
    collector = StackOverflowCollector()
    collector.collect(pages=10, pagesize=100)
    collector.save()
