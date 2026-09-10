"""
main.py — FastAPI application: pipeline orchestration, scheduler, and API endpoints.

This is the entry point for the application. It ties together every other module:

  scraper.py    — fetches RSS feeds and scrapes article text
  summarizer.py — runs BART locally to produce article and digest summaries
  models.py     — SQLAlchemy ORM definitions for Article and DailyDigest
  database.py   — SQLite engine, session factory, and schema initializer

Application lifecycle (on startup):
  1. Initialize the SQLite database schema.
  2. Load the BART model into memory (skipped in DEV_MODE).
  3. Run the pipeline once immediately in a background thread.
  4. Start APScheduler to repeat the pipeline every 6 hours.

On shutdown:
  APScheduler is stopped gracefully.

Run directly:
  python main.py [--dev] [--host HOST] [--port PORT]

Or via uvicorn:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import summarizer as summ
from database import SessionLocal, get_db, init_db
from models import Article, DailyDigest
from scraper import FEEDS, fetch_all_feeds, scrape_article_text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Configure root logger to include timestamp, level, and module name.
# All sub-module loggers (scraper, summarizer, etc.) inherit this config.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Read DEV_MODE at startup. When True, BART is never loaded and every article
# gets a fake placeholder summary prefixed with [DEV].
DEV_MODE = os.environ.get("DEV_MODE", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Pipeline, private helpers
# ---------------------------------------------------------------------------

def _upsert_article(db: Session, data: dict) -> Article | None:
    """Insert a new article row, or skip it if the URL already exists.

    Deduplication is by URL. This means re-running the pipeline after a
    restart will never produce duplicate articles or re-summarize existing ones.

    Returns the new Article object if inserted, or None if it was a duplicate.
    """
    # Check if this URL is already in the database.
    existing = db.query(Article).filter(Article.url == data["url"]).first()
    if existing:
        return None  # already processed; skip silently

    from scraper import strip_html

    # Store whatever body text the RSS feed provided, stripped of HTML tags.
    # This is the fallback in case full-page scraping fails later.
    raw = data.get("raw_text", "") or ""

    article = Article(
        title=data["title"],
        url=data["url"],
        source=data["source"],
        published_date=data.get("published_date"),
        full_text=strip_html(raw) if raw else None,
        summary=None,           # populated later by _summarize_article
        summarizing=False,      # set to True when a worker picks it up
        created_at=datetime.now(timezone.utc),
    )
    db.add(article)
    db.commit()
    db.refresh(article)  # reload to get the auto-assigned id
    return article


def _summarize_article(article_id: int):
    """Scrape full text and generate a BART summary for a single article.

    This function runs in a worker thread (ThreadPoolExecutor in run_pipeline).
    It opens its own database session rather than sharing one across threads.

    Flow:
      1. Mark the article as summarizing=True so the frontend shows a spinner.
      2. Attempt to scrape the full article page for better text.
      3. Fall back to the RSS body (stored in full_text) if scraping fails.
      4. Run BART on whatever text we have.
      5. Write the summary and clear the summarizing flag.

    On any error, the summarizing flag is reset to False so the article doesn't
    get stuck in a "forever summarizing" state.
    """
    db = SessionLocal()
    try:
        article = db.query(Article).filter(Article.id == article_id).first()
        if not article or article.summary:
            # Either the article was deleted, or it was already summarized
            # (can happen if the pipeline runs twice before the first finishes).
            return

        # Signal to the frontend that this article is being worked on.
        article.summarizing = True
        db.commit()

        # Attempt a fresh scrape of the article page for full-text content.
        # This yields much better summaries than the RSS excerpt.
        full_text = scrape_article_text(article.url)
        if not full_text or len(full_text) < 200:
            # Scrape failed or returned too little (paywall / JS-rendered).
            # Fall back to whatever text was saved from the RSS feed.
            full_text = article.full_text or ""

        # Persist the best text we found for future reference.
        if full_text:
            article.full_text = full_text

        if len(full_text.strip()) > 100:
            # Enough text to summarize, run BART.
            article.summary = summ.summarize(full_text)
        else:
            # Nothing usable. Leave summary=None and clear the flag so the
            # card stays in "Summarizing..." state until the next pipeline run
            # might find the article accessible.
            article.summarizing = False
            db.commit()
            return

        article.summarizing = False
        db.commit()
        logger.info(f"Summarized: {article.title[:60]}")

    except Exception as e:
        logger.error(f"Error summarizing article {article_id}: {e}")
        # Best-effort: clear the summarizing flag so the article doesn't
        # appear stuck in the UI. A fresh pipeline run will retry it.
        try:
            article = db.query(Article).filter(Article.id == article_id).first()
            if article:
                article.summarizing = False
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _build_daily_digest(date_str: str):
    """Build (or rebuild) the daily digest for a given calendar date.

    Collects all fully-summarized articles published on date_str, concatenates
    their titles and summaries, then runs BART one more time to produce a
    single overview paragraph for the day.

    If a digest for the date already exists (e.g., this is a second run after
    more articles arrived), the existing row is overwritten with fresh content.
    """
    db = SessionLocal()
    try:
        from datetime import date as date_type, timedelta

        # Build datetime bounds for the target day (midnight to midnight UTC).
        target_date = date_type.fromisoformat(date_str)
        day_start = datetime(target_date.year, target_date.month, target_date.day)
        day_end = day_start + timedelta(days=1)

        # Only include articles that are fully summarized (summary IS NOT NULL
        # and summarizing = False). Partially-done articles are excluded to
        # avoid an incomplete digest.
        articles = (
            db.query(Article)
            .filter(
                Article.published_date >= day_start,
                Article.published_date < day_end,
                Article.summary.isnot(None),
                Article.summarizing == False,
            )
            .all()
        )

        if not articles:
            # No summarized articles for this day yet, nothing to digest.
            return

        # Build the combined input for BART: "Title: summary Title: summary ..."
        combined = " ".join(
            f"{a.title}: {a.summary}" for a in articles if a.summary
        )

        # Run BART with a larger output window than individual articles since
        # this needs to cover an entire day's news.
        digest_summary = summ.summarize(combined, max_length=250, min_length=80)

        # Upsert: update if a digest row already exists, insert if not.
        existing = db.query(DailyDigest).filter(DailyDigest.date == date_str).first()
        if existing:
            existing.summary = digest_summary
            existing.article_count = len(articles)
        else:
            digest = DailyDigest(
                date=date_str,
                summary=digest_summary,
                article_count=len(articles),
                created_at=datetime.now(timezone.utc),
            )
            db.add(digest)

        db.commit()
        logger.info(f"Daily digest built for {date_str} ({len(articles)} articles).")

    except Exception as e:
        logger.error(f"Error building digest for {date_str}: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Pipeline, public entry point
# ---------------------------------------------------------------------------

def run_pipeline():
    """Run the full fetch → upsert → summarize → digest pipeline.

    Called on startup, every 6 hours by APScheduler, and on-demand via
    POST /api/refresh.

    Step-by-step:
      1. Fetch all RSS feeds and collect article metadata.
      2. Upsert each article into the DB (skip existing URLs).
      3. Summarize new articles using a 4-worker thread pool.
      4. Build today's daily digest from completed summaries.
    """
    logger.info("Starting feed refresh pipeline...")

    # Step 1 & 2: fetch feeds and insert new articles.
    db = SessionLocal()
    try:
        raw_articles = fetch_all_feeds()
        new_ids = []
        for data in raw_articles:
            article = _upsert_article(db, data)
            if article:
                new_ids.append(article.id)  # only track newly inserted rows

        logger.info(f"Inserted {len(new_ids)} new articles.")
    finally:
        db.close()

    # Step 3: summarize new articles in parallel using a thread pool.
    # 4 workers balances CPU utilization without overwhelming the model.
    # The pool blocks here until all summaries are complete before moving on.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_summarize_article, new_ids))

    # Step 4: (re-)build today's digest using all summarized articles so far.
    today = datetime.now(timezone.utc).date().isoformat()
    _build_daily_digest(today)

    logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager — replaces the deprecated @app.on_event hooks.

    Everything before `yield` runs on startup; everything after runs on shutdown.

    Startup sequence:
      1. init_db() — creates tables if they don't exist yet.
      2. summ.load_model() — loads BART into memory (or no-ops in DEV_MODE).
      3. Background thread — runs the initial pipeline without blocking startup.
      4. APScheduler — schedules the pipeline to repeat every 6 hours.

    The pipeline is run in a daemon thread so that if startup takes a long time
    (first-time model download + scraping + summarization), the server is still
    available to serve requests immediately.
    """
    # Initialize the database schema (safe to call on every startup).
    init_db()

    # Load the BART model into memory. This blocks startup briefly while the
    # model weights are read from disk (a few seconds). The one-time download
    # on first run may take 5-15 minutes depending on connection speed.
    summ.load_model()

    # Run the initial pipeline in a background thread so the server doesn't
    # wait minutes before accepting its first request.
    import threading
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()

    # Schedule the pipeline to run every 6 hours while the server is up.
    # APScheduler runs in its own background thread.
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_pipeline, "interval", hours=6, id="feed_refresh")
    scheduler.start()
    app.state.scheduler = scheduler  # store reference for potential future use

    # Hand control to FastAPI (server is now running and accepting requests).
    yield

    # Shutdown: stop the scheduler without waiting for a running job to finish.
    scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="the_mainspring", lifespan=lifespan)

# Serve everything in the static/ directory under /static (e.g., /static/index.html).
# The root route (GET /) serves index.html directly via FileResponse below.
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    """Serve the single-page dashboard at the root URL."""
    return FileResponse("static/index.html")


@app.get("/api/sources")
def get_sources():
    """Return the list of configured news outlet names.

    The frontend calls this on load to populate the source filter checkboxes.
    Returns them in the same order as FEEDS in scraper.py.
    """
    return {"sources": list(FEEDS.keys())}


@app.get("/api/articles")
def get_articles(
    date: str = Query(default=None, description="YYYY-MM-DD"),
    source: str = Query(default=None),
    db: Session = Depends(get_db),
):
    """Return all articles for a given date, optionally filtered by source.

    Query parameters:
      date   — Calendar date in YYYY-MM-DD format. Defaults to today (UTC).
      source — Exact outlet name (e.g., "Jacobin"). Omit to return all outlets.

    Date filtering uses an OR condition: articles are included if their
    published_date falls on the requested day, OR if published_date is NULL
    and their created_at (insertion time) falls on the requested day. This
    handles feeds that don't provide publication dates.

    Articles are ordered newest first (nulls last for missing published_date).
    """
    # Default to today if no date is provided.
    if not date:
        date = datetime.now(timezone.utc).date().isoformat()

    # Validate the date string before building the query.
    try:
        from datetime import date as date_type
        target = date_type.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Build midnight-to-midnight UTC boundaries for the requested day.
    from datetime import timedelta
    day_start = datetime(target.year, target.month, target.day)
    day_end = day_start + timedelta(days=1)

    # Include articles whose published_date is within the day, or whose
    # created_at is within the day when published_date is missing.
    from sqlalchemy import and_, or_
    date_filter = or_(
        and_(Article.published_date >= day_start, Article.published_date < day_end),
        and_(Article.published_date.is_(None), Article.created_at >= day_start, Article.created_at < day_end),
    )

    q = db.query(Article).filter(date_filter)
    if source:
        q = q.filter(Article.source == source)

    # Sort: newest published articles first; articles without a date go last,
    # ordered by insertion time within that group.
    articles = q.order_by(
        Article.published_date.desc().nullslast(),
        Article.created_at.desc(),
    ).all()

    return {
        "date": date,
        "articles": [
            {
                "id": a.id,
                "title": a.title,
                "url": a.url,
                "source": a.source,
                # Append "Z" to signal UTC to the frontend's Date parser.
                "published_date": (a.published_date.isoformat() + "Z") if a.published_date else None,
                "summary": a.summary,
                "summarizing": a.summarizing,
            }
            for a in articles
        ],
    }


@app.get("/api/digest")
def get_digest(
    date: str = Query(default=None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
):
    """Return the AI-generated daily digest for a given date.

    Query parameters:
      date — Calendar date in YYYY-MM-DD format. Defaults to today (UTC).

    Returns {"date": ..., "summary": null, "article_count": 0} if no digest
    exists yet for the requested date (e.g., summarization is still in progress).
    """
    if not date:
        date = datetime.now(timezone.utc).date().isoformat()

    digest = db.query(DailyDigest).filter(DailyDigest.date == date).first()
    if not digest:
        # Digest not built yet (pipeline still running or no articles for the day).
        return {"date": date, "summary": None, "article_count": 0}

    return {
        "date": date,
        "summary": digest.summary,
        "article_count": digest.article_count,
    }


@app.post("/api/refresh")
async def refresh_feeds(background_tasks: BackgroundTasks):
    """Manually trigger a full feed refresh pipeline.

    The pipeline runs in a FastAPI background task and this endpoint returns
    immediately (202-style). Check the server logs for progress.

    Useful for forcing an update without waiting for the 6-hour scheduler tick.
    """
    background_tasks.add_task(run_pipeline)
    return {"status": "Feed refresh started in background."}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="the_mainspring")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in dev mode: skip BART model load and use fake summaries.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    if args.dev:
        # Set the env var so summarizer.py picks it up at import time.
        os.environ["DEV_MODE"] = "1"

    # reload=True enables hot-reloading when running in dev mode, so code
    # changes take effect without restarting the server manually.
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.dev)
