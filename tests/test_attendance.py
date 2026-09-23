"""Entry/exit attendance rules: cycles, dedup-independent transitions, midnight."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.services.attendance import process_recognition


class FakeCamera:
    def __init__(self, cid: int = 1, ctype: str = "entry"):
        self.id = cid
        self.name = f"cam-{cid}"
        self.type = ctype
        self.location = "gate"


def _ts(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc)


def test_entry_creates_ongoing_session(db) -> None:
    from app.models import Student

    student = Student(name="Ada", registration_no="R1", embeddings=[], photo_paths=[])
    db.add(student)
    db.commit()

    outcome = process_recognition(
        db, student_id=student.id, camera=FakeCamera(1, "entry"), ts=_ts(23, 9)
    )
    assert outcome.action == "entry_created"
    assert outcome.session is not None
    assert outcome.session.status == "ongoing"
    assert outcome.session.date == date(2026, 9, 23)
    assert outcome.session.exit_time is None


def test_second_entry_same_day_is_noop(db) -> None:
    from app.models import Student

    student = Student(name="Ada", registration_no="R1", embeddings=[], photo_paths=[])
    db.add(student)
    db.commit()
    cam = FakeCamera(1, "entry")

    first = process_recognition(db, student_id=student.id, camera=cam, ts=_ts(23, 9))
    second = process_recognition(db, student_id=student.id, camera=cam, ts=_ts(23, 10))
    assert first.action == "entry_created"
    assert second.action == "none"
    db.commit()

    from sqlalchemy import func, select

    from app.models import AttendanceSession

    count = db.scalar(
        select(func.count()).select_from(AttendanceSession)
    )
    assert count == 1


def test_exit_closes_with_duration(db) -> None:
    from app.models import Student

    student = Student(name="Ada", registration_no="R1", embeddings=[], photo_paths=[])
    db.add(student)
    db.commit()

    process_recognition(
        db, student_id=student.id, camera=FakeCamera(1, "entry"), ts=_ts(23, 9)
    )
    outcome = process_recognition(
        db, student_id=student.id, camera=FakeCamera(2, "exit"), ts=_ts(23, 9, 50)
    )
    db.commit()

    assert outcome.action == "exit_completed"
    session = outcome.session
    assert session.status == "completed"
    assert session.total_duration == 50 * 60
    assert session.camera_out_id == 2


def test_multiple_cycles_create_multiple_sessions(db) -> None:
    from app.models import Student

    student = Student(name="Ada", registration_no="R1", embeddings=[], photo_paths=[])
    db.add(student)
    db.commit()
    entry = FakeCamera(1, "entry")
    exit_ = FakeCamera(2, "exit")

    process_recognition(db, student_id=student.id, camera=entry, ts=_ts(23, 8))
    process_recognition(db, student_id=student.id, camera=exit_, ts=_ts(23, 10))
    process_recognition(db, student_id=student.id, camera=entry, ts=_ts(23, 12))
    process_recognition(db, student_id=student.id, camera=exit_, ts=_ts(23, 13))
    db.commit()

    from sqlalchemy import select

    from app.models import AttendanceSession

    sessions = db.scalars(
        select(AttendanceSession).order_by(AttendanceSession.entry_time)
    ).all()
    assert len(sessions) == 2
    assert [s.status for s in sessions] == ["completed", "completed"]
    assert sessions[0].total_duration == 7200   # 08:00 -> 10:00 (2h)
    assert sessions[1].total_duration == 3600   # 12:00 -> 13:00 (1h)


def test_exit_without_entry_is_noop(db) -> None:
    from app.models import Student

    student = Student(name="Ada", registration_no="R1", embeddings=[], photo_paths=[])
    db.add(student)
    db.commit()

    outcome = process_recognition(
        db, student_id=student.id, camera=FakeCamera(2, "exit"), ts=_ts(23, 9)
    )
    assert outcome.action == "none"
    assert outcome.session is None


def test_midnight_rollover_same_session(db) -> None:
    """Session opened 23:50 day 23, exits 00:10 day 24 -> one session, date=23."""
    from app.models import Student

    student = Student(name="Ada", registration_no="R1", embeddings=[], photo_paths=[])
    db.add(student)
    db.commit()

    entry_ts = datetime(2026, 9, 23, 23, 50, tzinfo=timezone.utc)
    exit_ts = datetime(2026, 9, 24, 0, 10, tzinfo=timezone.utc)
    first = process_recognition(
        db, student_id=student.id, camera=FakeCamera(1, "entry"), ts=entry_ts
    )
    outcome = process_recognition(
        db, student_id=student.id, camera=FakeCamera(2, "exit"), ts=exit_ts
    )
    db.commit()

    assert first.session.date == date(2026, 9, 23)
    assert outcome.action == "exit_completed"
    assert outcome.session.id == first.session.id
    assert outcome.session.total_duration == 20 * 60
    assert outcome.session.date == date(2026, 9, 23)


def test_stale_session_auto_closed_on_next_day_entry(db) -> None:
    from app.models import Student

    student = Student(name="Ada", registration_no="R1", embeddings=[], photo_paths=[])
    db.add(student)
    db.commit()
    entry = FakeCamera(1, "entry")

    stale = process_recognition(
        db, student_id=student.id, camera=entry, ts=_ts(23, 9)
    )
    # Next day, still "open" (forgot to exit), student walks in again.
    outcome = process_recognition(
        db, student_id=student.id, camera=entry, ts=_ts(24, 9)
    )
    db.commit()

    assert outcome.action == "stale_closed_and_entry"
    assert outcome.stale_session is not None
    assert outcome.stale_session.id == stale.session.id
    assert outcome.stale_session.status == "completed"
    assert outcome.stale_session.exit_time == _ts(24, 9)
    assert outcome.session.date == date(2026, 9, 24)
    assert outcome.session.status == "ongoing"


def test_both_camera_closes_then_reopens(db) -> None:
    from app.models import Student

    student = Student(name="Ada", registration_no="R1", embeddings=[], photo_paths=[])
    db.add(student)
    db.commit()
    gate = FakeCamera(7, "both")

    opened = process_recognition(
        db, student_id=student.id, camera=gate, ts=_ts(23, 8)
    )
    assert opened.action == "entry_created"

    closed = process_recognition(
        db, student_id=student.id, camera=gate, ts=_ts(23, 9)
    )
    assert closed.action == "exit_completed"

    reopened = process_recognition(
        db, student_id=student.id, camera=gate, ts=_ts(23, 10)
    )
    assert reopened.action == "entry_created"
    db.commit()


def test_summary_totals_include_ongoing(db, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.models import Student
    from app.services import attendance as svc

    student = Student(name="Ada", registration_no="R1", embeddings=[], photo_paths=[])
    db.add(student)
    db.commit()

    entry_ts = _ts(23, 10)  # 10:00
    process_recognition(
        db, student_id=student.id, camera=FakeCamera(1, "entry"), ts=entry_ts
    )
    process_recognition(
        db, student_id=student.id, camera=FakeCamera(2, "exit"), ts=_ts(23, 10, 10)
    )
    # Second, still-ongoing session starting 11:30.
    process_recognition(
        db, student_id=student.id, camera=FakeCamera(1, "entry"), ts=_ts(23, 11, 30)
    )
    db.commit()

    # Freeze "now" at 11:40 so the ongoing session contributes 600s.
    frozen = _ts(23, 11, 40)
    monkeypatch.setattr(svc, "utcnow", lambda: frozen)

    summary = svc.summary_for_date(db, date(2026, 9, 23))
    assert len(summary.students) == 1
    row = summary.students[0]
    assert row.entries == 2
    assert row.exits == 1
    assert row.total_seconds == 600 + 600  # 10-min closed + 10-min ongoing
    assert row.ongoing is True
    assert summary.totals.present_students == 1
    assert summary.totals.ongoing_sessions == 1
    assert summary.totals.completed_sessions == 1
