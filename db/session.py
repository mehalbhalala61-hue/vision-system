# =============================================================================
# db/session.py — Database Engine + Session Factory
# =============================================================================
# Handles:
#   - Engine creation with PostgreSQL URL fix (postgres:// → postgresql://)
#   - SessionLocal factory for all DB operations
#   - get_db() FastAPI dependency — auto-closes session after each request
#
# Interview note:
#   "I used SQLAlchemy ORM with a centralized session factory and FastAPI
#    dependency injection — sessions are scoped per request and always closed,
#    preventing connection leaks."
# =============================================================================

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from db.base import Base  # noqa: F401 — needed so Alembic sees all models

load_dotenv()

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# DATABASE URL — Railway fix
# -----------------------------------------------------------------------------
# Railway injects DATABASE_URL as postgres:// but SQLAlchemy 1.4+ requires
# postgresql://  →  auto-fix with .replace() so deploy never crashes.
# -----------------------------------------------------------------------------
_raw_url = os.getenv("DATABASE_URL", "")

if not _raw_url:
    raise EnvironmentError(
        "DATABASE_URL is not set.\n"
        "  Local dev : add it to your .env file\n"
        "  Railway   : it is auto-injected after adding PostgreSQL service"
    )

DATABASE_URL: str = _raw_url.replace("postgres://", "postgresql://", 1)

# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    # Keep a small pool — Railway free tier has connection limits
    pool_size=5,
    max_overflow=10,
    # Recycle connections every 30 min — avoids stale connection errors
    pool_recycle=1800,
    # Test connection before using from pool
    pool_pre_ping=True,
    # Log SQL queries in development (set LOG_LEVEL=DEBUG in .env)
    echo=(os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG"),
)

# -----------------------------------------------------------------------------
# SESSION FACTORY
# -----------------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,   # Always explicit commit — prevents partial writes
    autoflush=False,    # Flush manually for predictable behaviour
    expire_on_commit=False,  # Keep objects accessible after commit
)

# -----------------------------------------------------------------------------
# FastAPI DEPENDENCY — one session per request, always closed
# -----------------------------------------------------------------------------
def get_db():
    """
    FastAPI dependency for database sessions.

    Usage in routes:
        from fastapi import Depends
        from db.session import get_db
        from sqlalchemy.orm import Session

        @router.post("/log")
        def log_meal(db: Session = Depends(get_db)):
            ...

    Guarantees the session is closed even if the endpoint raises an exception.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()   # Rollback on any unhandled exception
        raise
    finally:
        db.close()      # Always close — prevents connection leaks


# -----------------------------------------------------------------------------
# UTILITY — create all tables (dev / testing only)
# -----------------------------------------------------------------------------
def init_db() -> None:
    """
    Create all tables defined in ORM models.

    Use for local dev / unit tests only.
    In production, always use Alembic migrations:
        alembic upgrade head
    """
    # Import all models here so Base knows about them
    from db.models import meal_log  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")


# -----------------------------------------------------------------------------
# UTILITY — health check (used by /health endpoint)
# -----------------------------------------------------------------------------
def check_db_connection() -> bool:
    """
    Returns True if DB is reachable, False otherwise.
    Called by GET /health endpoint.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return False