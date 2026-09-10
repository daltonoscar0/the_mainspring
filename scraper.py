"""
scraper.py — RSS feed fetching and full-text article extraction.

Two responsibilities:
  1. fetch_all_feeds() / fetch_feed() — download and parse RSS/Atom feeds
     using feedparser, returning lightweight article dicts with metadata and
     whatever body text the feed provides.
  2. scrape_article_text() — visit each article URL directly with requests +
     BeautifulSoup, strip navigation/footer/ads, and return clean body text
     suitable for BART summarization.

All network calls use a descriptive User-Agent so outlets can identify the bot,
and a 15-second timeout so a slow site doesn't stall the whole pipeline.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feed registry
# ---------------------------------------------------------------------------

# Add or remove outlets here. The key is the human-readable source name used
# throughout the UI and stored in the database; the value is the RSS/Atom URL.
FEEDS = {
    "Jacobin": "https://jacobin.com/feed",
    "The Intercept": "https://theintercept.com/feed/?rss",
    "Democracy Now": "https://www.democracynow.org/democracynow.rss",
    "Current Affairs": "https://www.currentaffairs.org/news/rss.xml",
    "In These Times": "https://inthesetimes.com/feed",
    "Truthout": "https://truthout.org/feed/",
    "Common Dreams": "https://www.commondreams.org/rss.xml",
    "Mother Jones": "https://www.motherjones.com/feed/",
    "The Nation": "https://www.thenation.com/feed/?post_type=article",
    "CounterPunch": "https://www.counterpunch.org/feed/",
    "People's World": "https://www.peoplesworld.org/feed/?post_type=article",
    "Labor Notes": "https://www.labornotes.org/rss.xml",
    "Portside": "https://portside.org/rss.xml",
    "Dissent Magazine": "https://dissentmagazine.org/feed/",
}

# ---------------------------------------------------------------------------
# HTTP configuration
# ---------------------------------------------------------------------------

# Identify the bot to outlet servers so they can reach out if needed.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TheMainspringBot/1.0; +https://github.com/the_mainspring)"
    )
}

# Maximum seconds to wait for any single HTTP response before giving up.
REQUEST_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_date(entry) -> Optional[datetime]:
    """Extract a UTC-normalized publication datetime from an RSS feed entry.

    feedparser exposes both pre-parsed time structs (published_parsed,
    updated_parsed) and raw strings (published, updated). We try the structs
    first since they're already parsed, then fall back to dateutil for the
    raw strings which may use any of dozens of date formats.

    Returns None if no usable date can be found.
    """
    # Try pre-parsed time struct first (9-element tuple from time.struct_time).
    # Only the first 6 elements (year, month, day, hour, minute, second) are used.
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass

    # Fall back to raw date strings (e.g., "Mon, 07 Apr 2025 10:30:00 +0000").
    for field in ("published", "updated"):
        val = getattr(entry, field, None)
        if val:
            try:
                dt = dateutil_parser.parse(val)
                # If no timezone in string, assume UTC.
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                # Normalize to UTC so all dates in the DB are comparable.
                return dt.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
            except Exception:
                pass

    return None


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def _extract_text(html: str) -> str:
    """Parse an article's HTML and return clean body text, stripped of boilerplate.

    Strategy:
      1. Remove known junk tags (nav, footer, script, ads, etc.) outright.
      2. Remove any element whose CSS class or id matches common boilerplate
         keywords (sidebar, newsletter, social, etc.).
      3. Try a list of CSS selectors known to wrap main article content.
      4. Fall back to the full page text if no container is found.

    The result is fed directly to BART, so quality here determines summary quality.
    Paywalled or heavily JS-rendered articles will return very little text; the
    caller checks length and falls back to the RSS body if needed.
    """
    soup = BeautifulSoup(html, "lxml")

    # Step 1, Remove entire tag families that are never article content.
    for tag in soup.find_all(
        ["nav", "footer", "script", "style", "aside", "header",
         "noscript", "figure", "figcaption", "iframe"]
    ):
        tag.decompose()

    # Step 2, Remove elements whose class or id name hints at boilerplate.
    # This catches things like <div class="sidebar">, <div id="newsletter-signup">, etc.
    junk_patterns = re.compile(
        r"(nav|footer|sidebar|menu|advert|banner|cookie|popup|modal|social|share|"
        r"related|subscribe|newsletter|promo|widget|comment)",
        re.IGNORECASE,
    )
    for el in soup.find_all(True):
        cls = " ".join(el.get("class", []))
        eid = el.get("id", "")
        if junk_patterns.search(cls) or junk_patterns.search(eid):
            el.decompose()

    # Step 3, Walk through selectors from most to least specific.
    # The first one that matches is treated as the article container.
    for selector in [
        "article",          # semantic HTML5 element
        "[role='main']",    # ARIA landmark
        "main",             # HTML5 main element
        ".post-content",    # common WordPress class
        ".entry-content",   # another common WordPress class
        ".article-body",    # used by several news sites
        ".content",         # generic
        "#content",         # generic id variant
        ".story-body",      # used by some outlets
    ]:
        container = soup.select_one(selector)
        if container:
            return container.get_text(separator=" ", strip=True)

    # Step 4, Last resort: return all remaining text from the page.
    return soup.get_text(separator=" ", strip=True)


def strip_html(html: str) -> str:
    """Strip all HTML tags and return plain text.

    Used to clean the RSS <description> or <content> field before storing it
    as a fallback full_text. Much simpler than _extract_text because RSS bodies
    are already short excerpts without page chrome.
    """
    return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)


# ---------------------------------------------------------------------------
# Public scraping interface
# ---------------------------------------------------------------------------

def scrape_article_text(url: str) -> str:
    """Fetch an article URL and return clean body text.

    Returns an empty string on any error (network failure, HTTP error, etc.)
    or if the extracted text is suspiciously short (< 200 characters), which
    typically indicates a paywall interstitial or a JS-rendered page that
    requests can't execute.

    The caller (_summarize_article in main.py) checks the length and falls
    back to the RSS body text if this returns empty.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        text = _extract_text(resp.text)
        # Reject suspiciously short results (paywall / JS-rendered pages).
        return text if len(text) > 200 else ""
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return ""


# ---------------------------------------------------------------------------
# RSS feed fetching
# ---------------------------------------------------------------------------

def fetch_feed(source_name: str, feed_url: str) -> list[dict]:
    """Fetch and parse one RSS/Atom feed, returning a list of article dicts.

    Each dict has the keys:
      title        (str)            — article headline
      url          (str)            — canonical link to the article
      source       (str)            — outlet name (same as source_name arg)
      published_date (datetime|None) — UTC publication time
      raw_text     (str)            — RSS body text (plain or HTML); used as a
                                     fallback if full-page scraping fails

    feedparser's bozo flag is True if the feed is malformed XML, but we still
    try to process it if there are any entries (many real-world feeds are
    technically invalid but parseable).
    """
    articles = []
    try:
        parsed = feedparser.parse(feed_url)

        # bozo=True means the feed has XML errors. Still attempt parsing if
        # entries exist; log a warning so the operator knows.
        if parsed.bozo and not parsed.entries:
            logger.warning(f"[{source_name}] Feed parse error: {parsed.bozo_exception}")
            return articles

        for entry in parsed.entries:
            try:
                # Every article needs a URL; skip entries without one.
                url = getattr(entry, "link", None)
                if not url:
                    continue

                title = getattr(entry, "title", "Untitled").strip()
                published = _parse_date(entry)

                # Extract whatever body text the feed provides. Prefer content
                # (full Atom body) over summary/description (usually a teaser).
                # This is stored as raw_text and used as a fallback if the live
                # scrape of the article page fails or returns too little text.
                raw_text = ""
                for field in ("content", "summary", "description"):
                    val = getattr(entry, field, None)
                    if val:
                        # The "content" field is a list of dicts in Atom feeds.
                        if isinstance(val, list):
                            raw_text = val[0].get("value", "")
                        else:
                            raw_text = val
                        break

                articles.append(
                    {
                        "title": title,
                        "url": url,
                        "source": source_name,
                        "published_date": published,
                        "raw_text": raw_text,  # RSS body; full scrape happens later
                    }
                )
            except Exception as e:
                # Skip malformed individual entries rather than aborting the feed.
                logger.warning(f"[{source_name}] Skipping entry: {e}")

    except Exception as e:
        logger.error(f"[{source_name}] Feed fetch failed: {e}")

    logger.info(f"[{source_name}] Fetched {len(articles)} entries.")
    return articles


def fetch_all_feeds() -> list[dict]:
    """Fetch every feed in the FEEDS registry and return all articles combined.

    Feeds are fetched sequentially. Each fetch_feed() call is independent, so
    a failure in one outlet does not affect the others.
    """
    all_articles = []
    for name, url in FEEDS.items():
        all_articles.extend(fetch_feed(name, url))
    return all_articles
