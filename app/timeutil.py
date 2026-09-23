"""Time helpers: UTC now + local calendar dates (midnight-rollover logic)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from .config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def local_tz() -> ZoneInfo:
    try:
        return ZoneInfo(settings.local_timezone)
    except Exception:  # noqa: BLE001 - unknown tz string in env
        return ZoneInfo("UTC")


def to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(local_tz())


def local_date(dt: datetime | None = None) -> date:
    """Local calendar date for a UTC timestamp (defaults to now)."""
    return to_local(dt or utcnow()).date()


def parse_date(value: str) -> date:
    return date.fromisoformat(value)  # raises ValueError -> 422 handled by Pydantic
