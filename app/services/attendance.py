"""Entry/exit attendance logic.

Rules (all times UTC in DB; ``date`` = local date of entry):

* **entry camera** and no open session today  -> open a session (entry_time).
* **entry camera** and session already open today -> no-op.
* **exit camera** and an open session exists  -> close it (exit_time,
  total_duration, status=completed).  A session opened before midnight that
  closes after midnight is still the *same* session - its ``date`` stays the
  entry date (midnight rollover).
* **stale session** (open from a previous day, student never exited) seen at
  an entry/both camera today -> auto-close it at the new sighting and open a
  fresh session for today.
* **both camera** behaves as exit-when-open, entry-otherwise.
* Multiple entry/exit cycles per day each create their own session.
* Total time present per day = sum of (exit - entry); ongoing sessions count
  from entry until *now*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import AttendanceSession, Camera, Student
from ..schemas import (
    AttendanceSessionOut,
    LiveSessionOut,
    SessionTotals,
    StudentAttendanceOut,
    StudentSummary,
    SummaryOut,
    SummaryRow,
    SummaryTotals,
)
from ..timeutil import local_date, to_local, utcnow

logger = logging.getLogger("app.attendance")


@dataclass
class RecogOutcome:
    action: str  # entry_created | exit_completed | stale_closed_and_entry | stale_closed | none
    session: AttendanceSession | None = None
    stale_session: AttendanceSession | None = None

    @property
    def changed(self) -> bool:
        return self.action != "none"


# --------------------------------------------------------------------- core
def _as_utc(value: datetime) -> datetime:
    """Normalize stored datetimes to UTC-aware.

    PostgreSQL ``timestamptz`` round-trips tz-aware; SQLite stores naive
    strings, so after a commit+reload we may get naive values back.  All times
    in the DB are UTC, so attaching UTC to naive values is correct.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def get_open_session(db: Session, student_id: int) -> AttendanceSession | None:
    row = db.scalars(
        select(AttendanceSession)
        .where(
            AttendanceSession.student_id == student_id,
            AttendanceSession.status == "ongoing",
        )
        .order_by(AttendanceSession.entry_time.desc())
        .limit(1)
    ).first()
    # Guard against a not-yet-flushed close in the same transaction: the SQL
    # WHERE saw the stale row, but the identity map already knows it's closed.
    if row is not None and (row.status != "ongoing" or row.exit_time is not None):
        return None
    return row


def _close_session(
    db: Session, session: AttendanceSession, exit_time: datetime, camera_id: int | None
) -> None:
    session.exit_time = exit_time
    if camera_id is not None:
        session.camera_out_id = camera_id
    seconds = (exit_time - _as_utc(session.entry_time)).total_seconds()
    session.total_duration = max(0, int(seconds))
    session.status = "completed"
    # Persist before any follow-up SELECT in the same transaction (autoflush
    # may be off and the same Session can query right after closing).
    db.flush()
    logger.info(
        "session closed",
        extra={
            "session_id": session.id,
            "student_id": session.student_id,
            "total_duration_s": session.total_duration,
        },
    )


def _open_session(
    db: Session, student_id: int, camera: Camera, entry_time: datetime
) -> AttendanceSession:
    session = AttendanceSession(
        student_id=student_id,
        camera_in_id=camera.id,
        entry_time=entry_time,
        date=local_date(entry_time),
        status="ongoing",
    )
    db.add(session)
    db.flush()
    logger.info(
        "session opened",
        extra={
            "session_id": session.id,
            "student_id": student_id,
            "camera_id": camera.id,
            "date": session.date.isoformat(),
        },
    )
    return session


def process_recognition(
    db: Session, *, student_id: int, camera: Camera, ts: datetime
) -> RecogOutcome:
    """Apply entry/exit rules for one (already deduplicated) sighting."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    today = local_date(ts)
    open_session = get_open_session(db, student_id)

    # ------------------------------------------------------------- exit cam
    if camera.type == "exit":
        if open_session is None:
            return RecogOutcome(action="none")
        _close_session(db, open_session, ts, camera.id)
        return RecogOutcome(action="exit_completed", session=open_session)

    # ----------------------------------------------------- entry / both cam
    if open_session is not None and open_session.date == today:
        if camera.type == "both":
            # Seen again at a both-camera with an open session => exiting.
            _close_session(db, open_session, ts, camera.id)
            return RecogOutcome(action="exit_completed", session=open_session)
        # Already inside today (entry cam) - nothing to do.
        return RecogOutcome(action="none", session=open_session)

    if open_session is not None and open_session.date < today:
        # Stale session from a previous day (student forgot to exit).
        _close_session(db, open_session, ts, camera.id)
        new_session = _open_session(db, student_id, camera, ts)
        return RecogOutcome(
            action="stale_closed_and_entry",
            session=new_session,
            stale_session=open_session,
        )

    new_session = _open_session(db, student_id, camera, ts)
    return RecogOutcome(action="entry_created", session=new_session)


# ---------------------------------------------------------------- queries
def _session_out(session: AttendanceSession, camera_names: dict[int, str]) -> AttendanceSessionOut:
    return AttendanceSessionOut(
        id=session.id,
        student_id=session.student_id,
        camera_in_id=session.camera_in_id,
        camera_out_id=session.camera_out_id,
        camera_in_name=camera_names.get(session.camera_in_id or -1),
        camera_out_name=camera_names.get(session.camera_out_id or -1),
        entry_time=_as_utc(session.entry_time),
        exit_time=_as_utc(session.exit_time) if session.exit_time else None,
        date=session.date,
        status=session.status,  # type: ignore[arg-type]
        total_duration=session.total_duration,
    )


def _camera_name_map(db: Session) -> dict[int, str]:
    from ..models import Camera as _Camera

    return {
        int(cid): (name or "")
        for cid, name in db.execute(select(_Camera.id, _Camera.name)).all()
    }


def _session_seconds(session: AttendanceSession, now: datetime) -> int:
    if session.status == "completed" and session.total_duration is not None:
        return session.total_duration
    end = _as_utc(session.exit_time or now)
    return max(0, int((end - _as_utc(session.entry_time)).total_seconds()))


def student_attendance(
    db: Session, student: Student, day: date
) -> StudentAttendanceOut:
    sessions = (
        db.scalars(
            select(AttendanceSession)
            .where(
                AttendanceSession.student_id == student.id,
                AttendanceSession.date == day,
            )
            .order_by(AttendanceSession.entry_time.asc())
        )
        .all()
    )
    camera_names = _camera_name_map(db)
    now = utcnow()
    totals = SessionTotals(
        entries=len(sessions),
        exits=sum(1 for s in sessions if s.status == "completed"),
        total_seconds=sum(_session_seconds(s, now) for s in sessions),
        first_entry=_as_utc(sessions[0].entry_time) if sessions else None,
        last_exit=_as_utc(max(
            (s.exit_time for s in sessions if s.exit_time is not None),
            default=None,
        )) if any(s.exit_time is not None for s in sessions) else None,
    )
    return StudentAttendanceOut(
        student=StudentSummary(
            id=student.id,
            name=student.name,
            registration_no=student.registration_no,
            section=student.section,
        ),
        date=day,
        sessions=[_session_out(s, camera_names) for s in sessions],
        totals=totals,
    )


def summary_for_date(
    db: Session, day: date, *, include_absent: bool = False
) -> SummaryOut:
    sessions = (
        db.scalars(
            select(AttendanceSession)
            .where(AttendanceSession.date == day)
            .order_by(AttendanceSession.entry_time.asc())
        )
        .all()
    )
    students = {
        int(s.id): s for s in db.scalars(select(Student)).all()
    }
    now = utcnow()

    by_student: dict[int, list[AttendanceSession]] = {}
    for s in sessions:
        by_student.setdefault(s.student_id, []).append(s)

    rows: list[SummaryRow] = []
    grand_seconds = 0
    ongoing_count = 0
    completed_count = 0

    student_ids = list(students.keys()) if include_absent else list(by_student.keys())
    for sid in sorted(student_ids):
        student = students.get(sid)
        if student is None:
            continue
        s_sessions = by_student.get(sid, [])
        total = sum(_session_seconds(s, now) for s in s_sessions)
        ongoing = any(s.status == "ongoing" for s in s_sessions)
        rows.append(
            SummaryRow(
                student_id=sid,
                name=student.name,
                registration_no=student.registration_no,
                section=student.section,
                entries=len(s_sessions),
                exits=sum(1 for s in s_sessions if s.status == "completed"),
                first_entry=(
                    _as_utc(min((s.entry_time for s in s_sessions), default=None))
                    if s_sessions
                    else None
                ),
                last_exit=_as_utc(max(
                    (s.exit_time for s in s_sessions if s.exit_time is not None),
                    default=None,
                )) if any(s.exit_time is not None for s in s_sessions) else None,
                total_seconds=total,
                ongoing=ongoing,
            )
        )
        grand_seconds += total
        ongoing_count += sum(1 for s in s_sessions if s.status == "ongoing")
        completed_count += sum(1 for s in s_sessions if s.status == "completed")

    return SummaryOut(
        date=day,
        timezone=str(to_local(utcnow()).tzinfo),
        students=rows,
        totals=SummaryTotals(
            present_students=sum(1 for r in rows if r.entries > 0),
            ongoing_sessions=ongoing_count,
            completed_sessions=completed_count,
            total_seconds=grand_seconds,
        ),
    )


def live_sessions(db: Session) -> list[LiveSessionOut]:
    sessions = (
        db.scalars(
            select(AttendanceSession)
            .where(AttendanceSession.status == "ongoing")
            .order_by(AttendanceSession.entry_time.desc())
        )
        .all()
    )
    camera_names = _camera_name_map(db)
    now = utcnow()
    out: list[LiveSessionOut] = []
    for s in sessions:
        student = db.get(Student, s.student_id)
        if student is None:
            continue
        out.append(
            LiveSessionOut(
                session_id=s.id,
                student=StudentSummary(
                    id=student.id,
                    name=student.name,
                    registration_no=student.registration_no,
                    section=student.section,
                ),
                entry_time=_as_utc(s.entry_time),
                date=s.date,
                elapsed_seconds=max(0, int((now - _as_utc(s.entry_time)).total_seconds())),
                camera_in_id=s.camera_in_id,
                camera_in_name=camera_names.get(s.camera_in_id or -1),
            )
        )
    return out
