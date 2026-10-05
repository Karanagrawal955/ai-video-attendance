#!/usr/bin/env python
"""Close stale 'ongoing' attendance sessions after N days.

Uses app.services.attendance._close_session(camera_id=None) so duration is
computed from entry_time to now. Designed to be run as a CLI or as the
Celery beat task jobs.close_stale_sessions.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal
from app.models import AttendanceSession
from app.services.attendance import _close_session
from app.timeutil import utcnow
from sqlalchemy import select


def close_stale(days: int = 1, dry_run: bool = False) -> int:
    cutoff = utcnow() - timedelta(days=days)
    closed = 0
    with SessionLocal() as db:
        stale = db.scalars(
            select(AttendanceSession).where(
                AttendanceSession.status == "ongoing",
                AttendanceSession.entry_time < cutoff,
            )
        ).all()
        print(f"[close_stale] cutoff={cutoff.isoformat()} ({days}d ago), found {len(stale)} ongoing stale session(s)")
        for s in stale:
            print(f"  - id={s.id} student={s.student_id} entry={s.entry_time} status={s.status}")
            if not dry_run:
                _close_session(db, s, utcnow(), camera_id=None)
                closed += 1
        if closed:
            db.commit()
        print(f"[close_stale] closed {closed} session(s)")
        return closed


def main() -> int:
    ap = argparse.ArgumentParser(description="Close stale ongoing sessions")
    ap.add_argument("--days", type=int, default=1, help="Close sessions older than N days (default 1)")
    ap.add_argument("--dry-run", action="store_true", help="List without closing")
    args = ap.parse_args()
    n = close_stale(days=args.days, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
