
import requests
import json
import time
from datetime import datetime, timezone
from pathlib import Path

class HackerNewsCollector:
    def __init__(self, output_path="data/raw/hn_posts.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.keywords = [
            "ai", "copilot", "chatgpt", "cursor", "agent", "burnout", "stress",
            "replaced", "layoff", "automation", "anxious", "job security",
            "software engineer", "developer"
        ]
        self.posts = []
        
    def _is_relevant(self, text):
        if not text:
            return False
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.keywords)
    
    def _get_item(self, item_id):
        url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching item {item_id}: {e}")
            return None
    
    def collect(self, num_stories=500):
        print(f"Collecting {num_stories} latest Hacker News stories...")
        
        # Get latest story IDs
        try:
            response = requests.get("https://hacker-news.firebaseio.com/v0/newstories.json", timeout=10)
            response.raise_for_status()
            latest_ids = response.json()[:num_stories]
        except Exception as e:
            print(f"Error fetching latest stories: {e}")
            return self
        
        # Fetch each story
        for idx, item_id in enumerate(latest_ids):
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(latest_ids)} stories...")
            item = self._get_item(item_id)
            time.sleep(0.1)  # Be nice to the API
            
            if not item or item.get("type") != "story":
                continue
                
            # Check if relevant
            title = item.get("title", "")
            text = item.get("text", "")
            if self._is_relevant(title) or self._is_relevant(text):
                # Normalize to match our Reddit post structure
                normalized = {
                    "id": str(item.get("id")),
                    "subreddit": "hackernews",  # Treat HN as a "subreddit"
                    "title": title,
                    "selftext": text,
                    "score": item.get("score"),
                    "num_comments": item.get("descendants", 0),
                    "created_utc": item.get("time"),
                    "created_iso": datetime.fromtimestamp(item.get("time"), tz=timezone.utc).isoformat() if item.get("time") else None,
                    "author": item.get("by"),
                    "permalink": f"https://news.ycombinator.com/item?id={item_id}",
                    "url": item.get("url"),
                    "source": "hackernews"
                }
                self.posts.append(normalized)
                
        print(f"Collected {len(self.posts)} relevant Hacker News stories!")
        return self
    
    def save(self):
        # Load existing HN posts to avoid duplicates
        existing_ids = set()
        if self.output_path.exists():
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    post = json.loads(line)
                    existing_ids.add(post.get("id"))
        
        # Save new posts
        new_posts = [p for p in self.posts if p["id"] not in existing_ids]
        with open(self.output_path, "a", encoding="utf-8") as f:
            for post in new_posts:
                f.write(json.dumps(post, ensure_ascii=False) + "\n")
        print(f"Saved {len(new_posts)} new posts to {self.output_path}")
        return self.output_path

if __name__ == "__main__":
    collector = HackerNewsCollector()
    collector.collect(num_stories=1000)  # Get 1000 latest stories
    collector.save()
