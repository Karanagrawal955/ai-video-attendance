"""SQLAlchemy engine / session management."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _build_engine() -> Engine:
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if settings.database_url.startswith(("postgresql", "postgres")):
        kwargs.update(pool_size=10, max_overflow=20, pool_recycle=1800)
    return create_engine(settings.database_url, **kwargs)


engine: Engine = _build_engine()

SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, class_=Session
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: request-scoped session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_db() -> bool:
    """Cheap connectivity probe used by /health."""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - health check must not raise
        return False
