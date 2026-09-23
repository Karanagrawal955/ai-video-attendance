"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from ..db import get_db
from ..security import get_current_admin

__all__ = ["get_db", "get_current_admin"]
