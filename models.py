"""
models.py — SQLAlchemy ORM models: Article and DailyDigest.

Two tables live in the SQLite database:

  articles       — one row per news article; stores metadata, scraped full
                   text, and the BART-generated summary.
  daily_digests  — one row per calendar day; stores a digest summary produced
                   by running BART over all of that day's individual summaries.

Both tables are created automatically on first run via database.init_db().
"""

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from database import Base


class Article(Base):
    """Represents a single news article fetched from an RSS feed.

    Lifecycle of a row:
      1. Inserted by _upsert_article() in main.py with summary=None,
         summarizing=False. The RSS <description> body is stored as full_text.
      2. _summarize_article() picks it up, sets summarizing=True, scrapes the
         full article page for better text, runs BART, then writes the result
         to summary and clears summarizing back to False.
      3. The frontend polls /api/articles every 10 seconds and displays a
         spinner while summarizing=True or summary is still None.
    """

    __tablename__ = "articles"

    # Auto-incremented integer primary key.
    id = Column(Integer, primary_key=True, index=True)

    # Headline as it appears in the RSS feed entry.
    title = Column(String(512), nullable=False)

    # Canonical article URL. This is the deduplication key: if a URL already
    # exists in the DB, the article is skipped on subsequent pipeline runs so
    # the same article is never summarized twice.
    url = Column(String(1024), unique=True, nullable=False, index=True)

    # Human-readable name of the news outlet (e.g., "Jacobin", "The Intercept").
    # Must match a key in scraper.FEEDS so that source filtering in the UI works.
    source = Column(String(128), nullable=False, index=True)

    # Publication timestamp from the RSS feed, normalized to UTC.
    # Nullable because some feeds omit or malform this field.
    published_date = Column(DateTime, nullable=True, index=True)

    # Full article body text after HTML stripping. Initially populated from the
    # RSS <description>/<content> field at ingest time. May be overwritten with
    # better content scraped directly from the article URL in _summarize_article().
    full_text = Column(Text, nullable=True)

    # BART-generated summary (40-150 tokens). NULL until summarization completes.
    # The frontend treats NULL + summarizing=False as "still pending" (shows spinner).
    summary = Column(Text, nullable=True)

    # True while a background worker thread is actively summarizing this article.
    # Used by the frontend to show a spinner and by the poller to decide whether
    # to keep polling for updates.
    summarizing = Column(Boolean, default=False)

    # UTC timestamp of when this row was first inserted into the database.
    created_at = Column(DateTime, nullable=True)


class DailyDigest(Base):
    """An AI-generated digest summary covering an entire calendar day.

    After all articles for a given day are summarized, _build_daily_digest()
    in main.py concatenates their summaries and runs BART one more time to
    produce a short (80–250 token) overview of the day's news. If new articles
    arrive and get summarized later, the digest is rebuilt and overwritten.
    """

    __tablename__ = "daily_digests"

    # Auto-incremented integer primary key.
    id = Column(Integer, primary_key=True, index=True)

    # Calendar date in ISO format (YYYY-MM-DD). Unique, one digest per day.
    # Used as the lookup key in the /api/digest endpoint.
    date = Column(String(10), unique=True, nullable=False, index=True)

    # BART-generated digest of the day's news (80-250 tokens).
    summary = Column(Text, nullable=True)

    # Number of articles that were included when this digest was built.
    # Stored for informational display; also useful for deciding whether to
    # rebuild if more articles have been summarized since last build.
    article_count = Column(Integer, default=0)

    # UTC timestamp of when this digest row was first created.
    created_at = Column(DateTime, nullable=True)
