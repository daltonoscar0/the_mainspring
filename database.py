"""
database.py — SQLAlchemy engine, session factory, and schema initializer.

SQLite is used as the backing store. A single file (news.db by default) holds
everything: articles, summaries, and daily digests. No external database server
is required, which keeps the setup completely self-contained and local.

The DB_PATH environment variable lets you move the database file to a different
location (e.g., a mounted encrypted volume) without changing any code.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------

# Allow the caller to override the database path via an environment variable.
# Default: news.db in the current working directory (wherever you launch the app).
DB_PATH = os.environ.get("DB_PATH", "news.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# check_same_thread=False is required for SQLite when multiple threads share a
# connection. FastAPI's background tasks and the ThreadPoolExecutor in main.py
# both open their own sessions via SessionLocal, so this is safe, each thread
# gets its own Session rather than sharing one connection object.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

# autocommit=False: transactions must be committed explicitly (db.commit()).
# autoflush=False: objects are not flushed to the DB automatically on query;
#   we control exactly when data is written.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

# All ORM model classes (Article, DailyDigest in models.py) inherit from this
# Base so SQLAlchemy can map them to database tables.
Base = declarative_base()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def init_db():
    """Create all tables if they don't already exist.

    Called once at application startup (see main.py lifespan). Importing
    the model modules here ensures their Table definitions are registered
    on Base.metadata before create_all() runs.
    """
    from models import Article, DailyDigest  # noqa: F401, side-effect import
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a fresh database session per request.

    Usage in an endpoint:
        @app.get("/api/example")
        def example(db: Session = Depends(get_db)):
            rows = db.query(Article).all()

    The session is always closed in the finally block, even if an exception
    is raised during request handling, preventing connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
