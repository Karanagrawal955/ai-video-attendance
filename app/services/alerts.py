"""Alert creation + rule evaluation (Req 5)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import redis_client as rc
from ..config import settings
from ..models import Alert, AttendanceSession, Period, Student
from ..timeutil import to_local, utcnow

logger = logging.getLogger("app.alerts")


def create_alert(
    db: Session,
    r=None,
    *,
    type: str,
    severity: str = "medium",
    payload: dict[str, Any] | None = None,
    camera_id: int | None = None,
    student_id: int | None = None,
    period_id: int | None = None,
) -> Alert:
    payload = payload or {}
    alert = Alert(
        type=type,
        severity=severity,
        payload=payload,
        camera_id=camera_id,
        student_id=student_id,
        period_id=period_id,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    # publish to live feed so dashboard sees it
    try:
        from .events import publish as _publish

        # reuse generic event fan-out
        data = {
            "type": "alert",
            "alert_type": type,
            "severity": severity,
            "payload": payload,
            "camera_id": camera_id,
            "student_id": student_id,
            "period_id": period_id,
            "id": alert.id,
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
        }
        # best-effort publish
        _publish(r, data)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.debug("alert publish failed: %s", exc)
    logger.info("alert created", extra={"alert_id": alert.id, "type": type, "severity": severity})
    return alert


# ---------------------------------------------------------------- deviation checks
def _expected_students_for_period(db: Session, period: Period) -> list[Student]:
    q = select(Student)
    if period.section is not None:
        q = q.where(Student.section == period.section)
    return list(db.scalars(q).all())


def evaluate_period_deviations(db: Session, r=None, now: datetime | None = None) -> list[Alert]:
    """Check today's periods that have started; create absent/late/early alerts."""
    now = now or utcnow()
    local_now = to_local(now)
    today = local_now.date()
    weekday = local_now.weekday()
    bit = 1 << weekday
    periods = db.scalars(select(Period)).all()
    created: list[Alert] = []
    for p in periods:
        if not (p.days_bitmask & bit):
            continue
        # period window in local time
        start_local = datetime.combine(today, p.start_time).replace(tzinfo=local_now.tzinfo)
        end_local = datetime.combine(today, p.end_time).replace(tzinfo=local_now.tzinfo)
        # convert to UTC for comparison
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = end_local.astimezone(timezone.utc)
        # grace windows
        absent_grace = timedelta(minutes=15)
        late_grace = timedelta(minutes=5)
        # only evaluate if period has started + absent_grace
        if now < start_utc + absent_grace:
            continue
        expected = _expected_students_for_period(db, p)
        for stu in expected:
            sess = db.scalars(
                select(AttendanceSession).where(
                    AttendanceSession.student_id == stu.id,
                    AttendanceSession.period_id == p.id,
                    AttendanceSession.date == today,
                )
            ).all()
            if not sess:
                # absent: no session at all for this period today
                # dedup: don't spam same absent alert multiple times per day
                existing = db.scalars(
                    select(Alert).where(
                        Alert.type == "student_absent",
                        Alert.student_id == stu.id,
                        Alert.period_id == p.id,
                        Alert.created_at >= datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc),
                    )
                ).first()
                if existing is None:
                    created.append(
                        create_alert(
                            db,
                            r,
                            type="student_absent",
                            severity="high",
                            payload={"period": p.name, "date": today.isoformat(), "student": stu.registration_no},
                            student_id=stu.id,
                            period_id=p.id,
                        )
                    )
            else:
                # late: first entry after start+late_grace
                first_entry = min(s.entry_time for s in sess)
                # convert to aware if naive
                if first_entry.tzinfo is None:
                    first_entry = first_entry.replace(tzinfo=timezone.utc)
                if first_entry > start_utc + late_grace:
                    existing = db.scalars(
                        select(Alert).where(
                            Alert.type == "student_late",
                            Alert.student_id == stu.id,
                            Alert.period_id == p.id,
                            Alert.created_at >= datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc),
                        )
                    ).first()
                    if existing is None:
                        created.append(
                            create_alert(
                                db,
                                r,
                                type="student_late",
                                severity="medium",
                                payload={
                                    "period": p.name,
                                    "entry": first_entry.isoformat(),
                                    "start": start_utc.isoformat(),
                                },
                                student_id=stu.id,
                                period_id=p.id,
                            )
                        )
                # early exit: if session completed and exit before end -10min
                for s in sess:
                    if s.status == "completed" and s.exit_time is not None:
                        et = s.exit_time
                        if et.tzinfo is None:
                            et = et.replace(tzinfo=timezone.utc)
                        if et < end_utc - timedelta(minutes=10):
                            ex = db.scalars(
                                select(Alert).where(
                                    Alert.type == "early_exit",
                                    Alert.student_id == stu.id,
                                    Alert.period_id == p.id,
                                    Alert.created_at >= datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc),
                                )
                            ).first()
                            if ex is None:
                                created.append(
                                    create_alert(
                                        db,
                                        r,
                                        type="early_exit",
                                        severity="medium",
                                        payload={"period": p.name, "exit": et.isoformat(), "end": end_utc.isoformat()},
                                        student_id=stu.id,
                                        period_id=p.id,
                                    )
                                )
    return created
