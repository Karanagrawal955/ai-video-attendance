"""Attendance query endpoints: per-student, daily summary, live view, audit log."""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Camera, RecognitionLog, Student
from ..schemas import (
    LiveOut,
    RecognitionLogOut,
    StudentAttendanceOut,
    SummaryOut,
)
from ..services import attendance as svc
from ..timeutil import utcnow
from .deps import get_current_admin, get_db

router = APIRouter(prefix="/attendance", tags=["attendance"])


def _parse_day(value: str | None) -> date:
    if value is None:
        return svc.local_date()  # type: ignore[attr-defined]
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid date {value!r}, expected YYYY-MM-DD"
        ) from exc


@router.get("/summary", response_model=SummaryOut,
            summary="Daily per-student totals (entries/exits/time present)")
def attendance_summary(
    date_str: str | None = Query(default=None, alias="date",
                                 description="YYYY-MM-DD (defaults to today, local TZ)"),
    include_absent: bool = Query(default=False,
                                 description="Include students with zero sessions"),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> SummaryOut:
    day = _parse_day(date_str)
    return svc.summary_for_date(db, day, include_absent=include_absent)


@router.get("/live", response_model=LiveOut,
            summary="All ongoing sessions right now")
def attendance_live(
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> LiveOut:
    sessions = svc.live_sessions(db)
    return LiveOut(server_time=utcnow(), count=len(sessions), sessions=sessions)


@router.get("/logs", response_model=dict,
            summary="Raw recognition audit log (with GPU inference latency)")
def recognition_logs(
    student_id: int | None = Query(default=None),
    camera_id: int | None = Query(default=None),
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> dict:
    stmt = select(RecognitionLog)
    if student_id is not None:
        stmt = stmt.where(RecognitionLog.student_id == student_id)
    if camera_id is not None:
        stmt = stmt.where(RecognitionLog.camera_id == camera_id)
    if start is not None:
        stmt = stmt.where(RecognitionLog.timestamp >= start)
    if end is not None:
        stmt = stmt.where(RecognitionLog.timestamp <= end)

    rows = db.scalars(
        stmt.order_by(RecognitionLog.timestamp.desc()).offset(offset).limit(limit)
    ).all()

    student_names = {
        int(s.id): s.name
        for s in db.scalars(
            select(Student).where(Student.id.in_([r.student_id for r in rows if r.student_id]))
        ).all()
    } if rows else {}
    camera_names = {
        int(c.id): c.name
        for c in db.scalars(
            select(Camera).where(Camera.id.in_([r.camera_id for r in rows if r.camera_id]))
        ).all()
    } if rows else {}

    items = [
        RecognitionLogOut(
            id=r.id,
            student_id=r.student_id,
            student_name=student_names.get(r.student_id or -1),
            camera_id=r.camera_id,
            camera_name=camera_names.get(r.camera_id or -1),
            timestamp=r.timestamp,
            confidence_score=r.confidence_score,
            gpu_inference_time_ms=r.gpu_inference_time_ms,
        )
        for r in rows
    ]
    return {"items": items, "limit": limit, "offset": offset}


@router.get("/{student_id}", response_model=StudentAttendanceOut,
            summary="Attendance for one student on a given date")
def student_attendance(
    student_id: int,
    date_str: str | None = Query(default=None, alias="date",
                                 description="YYYY-MM-DD (defaults to today)"),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> StudentAttendanceOut:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"student {student_id} not found")
    return svc.student_attendance(db, student, _parse_day(date_str))
