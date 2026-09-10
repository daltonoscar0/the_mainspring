# the_mainspring

> *"Intelligent discontent is the mainspring of civilization."*, Eugene V. Debs

A self-hosted, privacy-first news aggregator and AI summarizer for progressive/left-wing media. Pulls RSS feeds from 14 outlets, scrapes full article text, and generates summaries using a locally-running BART model, **no API keys, no cloud services, no data leaving your machine.**

---

## Why Self-Host?

**Your reading habits are your business.** Commercial news aggregators and AI summary tools send your queries, reading history, and interests to third-party servers. This app runs entirely on your hardware:

- **No API keys.** The BART model runs locally, no OpenAI, no Anthropic, no Google.
- **No data egress.** Articles are fetched and stored in a local SQLite database. Nothing is uploaded anywhere.
- **No tracking.** No accounts, no analytics, no telemetry.
- **No subscriptions.** One-time model download (~1.6 GB), then it runs forever offline.
- **Complete data sovereignty.** Your digest, your summaries, your history, stored in a single `news.db` file you control entirely.
- **Audit everything.** The full source code is ~600 lines of Python and vanilla JS. Read it in an afternoon.

---

## Sources (14 Outlets)

| Outlet | Feed URL |
|---|---|
| Jacobin | `https://jacobin.com/feed` |
| The Intercept | `https://theintercept.com/feed/?rss` |
| Democracy Now | `https://www.democracynow.org/democracynow.rss` |
| Current Affairs | `https://www.currentaffairs.org/news/rss.xml` |
| In These Times | `https://inthesetimes.com/feed` |
| Truthout | `https://truthout.org/feed/` |
| Common Dreams | `https://www.commondreams.org/rss.xml` |
| Mother Jones | `https://www.motherjones.com/feed/` |
| The Nation | `https://www.thenation.com/feed/?post_type=article` |
| CounterPunch | `https://www.counterpunch.org/feed/` |
| People's World | `https://www.peoplesworld.org/feed/?post_type=article` |
| Labor Notes | `https://www.labornotes.org/rss.xml` |
| Portside | `https://portside.org/rss.xml` |
| Dissent Magazine | `https://dissentmagazine.org/feed/` |

Adding or removing outlets requires a one-line change in `scraper.py`.

---

## Features

- **AI Summaries**, Each article is summarized by `sshleifer/distilbart-cnn-12-6`, a distilled version of Facebook's BART model fine-tuned on CNN/DailyMail news.
- **Daily Digest**, After all articles are summarized, BART generates a single digest summary for the entire day's news.
- **Real-time Dashboard**, Dark-themed single-page web UI with date picker, per-source filtering, article counts, and live polling (summaries appear as they complete).
- **Scheduled Refresh**, APScheduler runs the full pipeline every 6 hours automatically.
- **Deduplication**, Articles are keyed by URL; re-runs never create duplicates.
- **Full-text Extraction**, BeautifulSoup strips nav/footer/ads/sidebars and extracts main article content before summarization.
- **Dev Mode**, Skip the model entirely and get instant fake summaries for UI development.

---

## Project Structure

```
the_mainspring/
├── main.py          # FastAPI app: lifespan hooks, pipeline orchestration, API endpoints
├── scraper.py       # RSS feed fetching (feedparser) + full-text scraping (BeautifulSoup)
├── summarizer.py    # BART model loading and inference; fake-summary shim for dev mode
├── models.py        # SQLAlchemy ORM models: Article, DailyDigest
├── database.py      # DB engine, session factory, init_db()
├── static/
│   └── index.html   # Single-file dashboard — HTML + CSS + vanilla JS, no build step
├── requirements.txt # Python dependencies
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- ~2 GB free disk space (BART model weights, downloaded once and cached)
- CPU is fine; a GPU makes summarization significantly faster

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/the_mainspring.git
cd the_mainspring
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** The `torch` package is large (~700 MB). If you only have a CPU, you can use the CPU-only wheel to save space:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 3. Run

```bash
python main.py
```

Or directly with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

**First-run timeline:**
- Model download: ~5-15 minutes depending on connection speed (one-time; cached by HuggingFace in `~/.cache/huggingface/`)
- Initial feed fetch + scrape: ~1-3 minutes
- Summarization of ~50-100 articles: ~20-60 minutes on CPU; ~2-5 minutes on GPU

Articles appear in the dashboard immediately with a "Summarizing…" spinner. Summaries populate in real time, the page polls every 10 seconds automatically and stops once all articles are done.

---

## Dev Mode

Skip the BART model entirely and get instant placeholder summaries:

```bash
python main.py --dev
# or
DEV_MODE=1 uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

In dev mode:
- The BART model is never loaded or downloaded.
- Each article gets a `[DEV] ...` placeholder summary drawn from the first 30 words of the article text.
- A red **DEV** badge appears in the dashboard header.
- `--reload` enables hot-reloading for frontend development.

---

## API Reference

The backend exposes a small REST API. All endpoints return JSON.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web dashboard (`static/index.html`) |
| `GET` | `/api/sources` | List of all configured source names |
| `GET` | `/api/articles?date=YYYY-MM-DD&source=…` | Articles for a given day, optionally filtered by source name |
| `GET` | `/api/digest?date=YYYY-MM-DD` | AI-generated daily digest summary for a given day |
| `POST` | `/api/refresh` | Manually trigger a full feed refresh (runs in background, returns immediately) |

**Example:**
```bash
# Get today's articles from The Intercept
curl "http://localhost:8000/api/articles?date=2026-04-05&source=The%20Intercept"

# Trigger a manual refresh
curl -X POST http://localhost:8000/api/refresh
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEV_MODE` | `""` | Set to `1`, `true`, or `yes` to skip BART and use fake summaries |
| `DB_PATH` | `news.db` | Path to the SQLite database file |

---

## How It Works

### Pipeline (runs on startup, then every 6 hours)

```
RSS Feeds (14 outlets)
       │
       ▼ feedparser
  Article list
       │
       ▼ SQLite upsert (deduplicate by URL)
  New article IDs
       │
       ▼ ThreadPoolExecutor (4 workers)
  Full-text scrape (requests + BeautifulSoup)
       │
       ▼ BART (sshleifer/distilbart-cnn-12-6)
  Per-article summaries
       │
       ▼ BART (second pass, today's articles only)
  Daily digest
```

### Summarization

1. For each new article, `scraper.scrape_article_text()` fetches the full HTML and strips boilerplate (nav, footer, ads, sidebars) using tag name and CSS class heuristics.
2. If scraping fails or returns less than 200 characters (paywalled or JS-rendered), the app falls back to the RSS `<description>` text.
3. Text is passed to BART's tokenizer (max 1024 tokens; excess is truncated).
4. Beam search (4 beams, no-repeat-ngram-size=3) produces a summary of 40-150 tokens.
5. After all articles for the day are done, their summaries are concatenated and BART produces a single 80-250-token daily digest.

### Database Schema

**`articles`**
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `title` | VARCHAR(512) | Article headline |
| `url` | VARCHAR(1024) | Unique; used for deduplication |
| `source` | VARCHAR(128) | Outlet name |
| `published_date` | DATETIME | UTC; from RSS feed |
| `full_text` | TEXT | Scraped article body |
| `summary` | TEXT | BART-generated summary |
| `summarizing` | BOOLEAN | True while summary is being generated |
| `created_at` | DATETIME | UTC; insertion time |

**`daily_digests`**
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `date` | VARCHAR(10) | `YYYY-MM-DD`; unique |
| `summary` | TEXT | BART digest of the day's articles |
| `article_count` | INTEGER | Number of articles summarized |
| `created_at` | DATETIME | UTC |

---

## Tips & Customization

- **Add or remove sources:** Edit the `FEEDS` dict in `scraper.py`.
- **Force re-summarization:** Delete `news.db` and restart. The model re-summarizes everything from scratch.
- **GPU acceleration:** In `summarizer.py`, move the model and input tensors to a CUDA device (e.g., `_model = _model.to("cuda")`).
- **Change the database location:** Set `DB_PATH=/path/to/your/news.db` before starting.
- **Change the refresh interval:** Edit the `hours=6` argument in the `scheduler.add_job(...)` call in `main.py`.
- **Bind to a specific interface:** Pass `--host 127.0.0.1` to restrict access to localhost only (recommended if running on a shared machine).

---

## Security Notes

Because this app is designed for local use:

- By default it binds to `0.0.0.0:8000` (all interfaces). If you run it on a machine accessible from a network, either bind to `127.0.0.1` or put it behind a reverse proxy (nginx/Caddy) with authentication.
- There is no authentication layer, anyone who can reach the port can trigger a feed refresh via `POST /api/refresh`.
- The SQLite database is stored in plaintext. Keep it on an encrypted volume if your reading history is sensitive.

---

## License

MIT
