# 02 — Data Collection

## 1. Objective

The purpose of the data-collection phase was to create a recent
multi-platform corpus of developer discussions about artificial intelligence
tools and AI coding assistants.

## 2. Selected Keywords

The main keywords used during collection were:

- AI
- ChatGPT
- Copilot
- Cursor
- agent
- automation
- burnout
- stress
- anxiety
- layoff
- replaced
- job security
- software engineer
- developer

## 3. Data Sources

| Source | Platform | Collection method | Records |
|---|---|---|---:|
| Arctic Shift | Reddit | Public Reddit archive API | 305 |
| hnam25 Hugging Face dataset | Reddit | Public pre-existing dataset | 2,000 |
| divde Hugging Face dataset | Multiple | Public pre-existing dataset | 141 |
| Hacker News | Hacker News | Public Firebase API | 236 |
| GitHub Issues | GitHub | Public REST API | 1,616 |
| Stack Overflow | Stack Overflow | Stack Exchange API | 1,108 |
| **Total** | **All platforms** |  | **5,406** |

## 4. Reddit API Limitation

The original plan was to collect Reddit discussions through the official
Reddit API.

During the collection period:

- unauthenticated Reddit requests returned HTTP 403 errors;
- PullPush produced rate-limit problems;
- an official Reddit API request was submitted;
- no response was received.

To continue the project, Reddit publications were collected through Arctic
Shift and supplemented with documented public Hugging Face datasets.

The dataset was also expanded with GitHub Issues, Stack Overflow and Hacker
News discussions.

## 5. Collected Fields

The available fields vary by source but generally include:

- identifier;
- source or platform;
- subreddit, repository or community;
- title;
- body text;
- score;
- number of comments or answers;
- creation date;
- author;
- original URL.

## 6. Raw Dataset

All sources were merged into:

`data/raw/combined_reddit_posts.jsonl`

The file contains 5,406 records in JSON Lines format.

The raw file will remain unchanged during the rest of the project.

## 7. Collection Limitations

- The data does not represent all software developers.
- The platforms use different sampling methods.
- Some records have missing dates or metadata.
- GitHub Issues and Stack Overflow contain technical questions that may not
  always express a clear personal opinion.
- The existing sources have different numbers of records.
- Keyword filtering may include some irrelevant texts.

## 8. Collection Decision

Data collection is now frozen.

`combined_reddit_posts.jsonl` is considered the official raw dataset for the
project.

The next phase is Data Understanding.