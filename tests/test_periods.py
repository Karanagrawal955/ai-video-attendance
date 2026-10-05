"""Requirement 4: Period-Level Attendance."""

from __future__ import annotations

from datetime import datetime, time, timezone

import pytest


def test_period_crud(client, auth_header):
    # create
    r = client.post(
        "/periods",
        json={"name": "P1", "start_time": "09:00:00", "end_time": "10:00:00", "days_bitmask": 127},
        headers=auth_header,
    )
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    assert r.json()["name"] == "P1"

    # duplicate name 409
    dup = client.post(
        "/periods",
        json={"name": "P1", "start_time": "11:00:00", "end_time": "12:00:00"},
        headers=auth_header,
    )
    assert dup.status_code == 409

    # bad range 422
    bad = client.post(
        "/periods",
        json={"name": "Bad", "start_time": "10:00:00", "end_time": "09:00:00"},
        headers=auth_header,
    )
    assert bad.status_code == 422

    # list
    lst = client.get("/periods", headers=auth_header)
    assert lst.status_code == 200
    assert len(lst.json()) == 1

    # get one
    got = client.get(f"/periods/{pid}", headers=auth_header)
    assert got.status_code == 200
    assert got.json()["id"] == pid

    # update
    upd = client.put(f"/periods/{pid}", json={"name": "P1-updated"}, headers=auth_header)
    assert upd.status_code == 200
    assert upd.json()["name"] == "P1-updated"

    # delete
    d = client.delete(f"/periods/{pid}", headers=auth_header)
    assert d.status_code == 200
    assert client.get(f"/periods/{pid}", headers=auth_header).status_code == 404


def test_period_attendance_assignment(db, client, auth_header):
    """Entry at 09:30 should be tagged with period P1 (09-10), 10:30 -> P2."""
    from app.models import Camera, Student
    from app.services import attendance as svc

    # create periods via API
    p1 = client.post(
        "/periods",
        json={"name": "Morn", "start_time": "09:00:00", "end_time": "10:00:00", "days_bitmask": 127},
        headers=auth_header,
    ).json()
    p2 = client.post(
        "/periods",
        json={"name": "Mid", "start_time": "10:00:00", "end_time": "11:00:00", "days_bitmask": 127},
        headers=auth_header,
    ).json()

    from app.crypto import encrypt_embeddings

    # enroll student + camera directly in db
    stu = Student(name="PeriodStu", registration_no="PER100", section="A", embeddings=encrypt_embeddings([[1.0] * 512]), photo_paths=[])
    cam = Camera(name="GateP", type="entry", file_path="x.mp4", sampling_rate=5)
    db.add_all([stu, cam])
    db.commit()
    db.refresh(stu)
    db.refresh(cam)

    ts1 = datetime(2026, 9, 23, 9, 30, tzinfo=timezone.utc)
    out1 = svc.process_recognition(db, student_id=stu.id, camera=cam, ts=ts1)
    db.commit()
    assert out1.session is not None
    assert out1.session.period_id == p1["id"]

    # close so next entry creates new session
    svc._close_session(db, out1.session, datetime(2026, 9, 23, 9, 45, tzinfo=timezone.utc), cam.id)
    db.commit()

    ts2 = datetime(2026, 9, 23, 10, 30, tzinfo=timezone.utc)
    out2 = svc.process_recognition(db, student_id=stu.id, camera=cam, ts=ts2)
    db.commit()
    assert out2.session.period_id == p2["id"]

    # verify via API includes period fields
    r = client.get(f"/attendance/{stu.id}?date=2026-09-23", headers=auth_header)
    assert r.status_code == 200
    sessions = r.json()["sessions"]
    assert sessions[0]["period_id"] == p1["id"]
    assert sessions[0]["period_name"] == "Morn"
    assert sessions[1]["period_id"] == p2["id"]


def test_period_summary_and_per_student(db, client, auth_header):
    p1 = client.post(
        "/periods",
        json={"name": "S1", "start_time": "09:00:00", "end_time": "10:00:00", "days_bitmask": 127},
        headers=auth_header,
    ).json()
    p2 = client.post(
        "/periods",
        json={"name": "S2", "start_time": "10:00:00", "end_time": "11:00:00", "days_bitmask": 127},
        headers=auth_header,
    ).json()

    from app.models import Camera, Student
    from app.services import attendance as svc
    from app.crypto import encrypt_embeddings

    stu = Student(name="SumStu", registration_no="SUM001", embeddings=encrypt_embeddings([[1.0] * 512]), photo_paths=[])
    cam = Camera(name="GateS", type="entry", file_path="x.mp4", sampling_rate=5)
    db.add_all([stu, cam])
    db.commit()
    db.refresh(stu)
    db.refresh(cam)

    svc.process_recognition(db, student_id=stu.id, camera=cam, ts=datetime(2026, 9, 23, 9, 15, tzinfo=timezone.utc))
    db.commit()
    # close and create second period session
    from app.models import AttendanceSession

    sess1 = db.query(AttendanceSession).filter_by(student_id=stu.id).first()
    svc._close_session(db, sess1, datetime(2026, 9, 23, 9, 30, tzinfo=timezone.utc), cam.id)
    db.commit()
    svc.process_recognition(db, student_id=stu.id, camera=cam, ts=datetime(2026, 9, 23, 10, 15, tzinfo=timezone.utc))
    db.commit()

    # per-period summary should isolate one session each
    s1 = client.get(f"/attendance/period/{p1['id']}/summary?date=2026-09-23", headers=auth_header)
    assert s1.status_code == 200
    assert s1.json()["totals"]["completed_sessions"] == 1
    assert len(s1.json()["students"]) == 1

    s2 = client.get(f"/attendance/period/{p2['id']}/summary?date=2026-09-23", headers=auth_header)
    assert s2.json()["totals"]["ongoing_sessions"] == 1

    ps = client.get(f"/attendance/period/{p1['id']}/{stu.id}?date=2026-09-23", headers=auth_header)
    assert ps.status_code == 200
    assert len(ps.json()["sessions"]) == 1
    assert ps.json()["sessions"][0]["period_id"] == p1["id"]


def test_period_section_and_days_filter(db, client, auth_header):
    # Section-specific period: only students in that section get tagged
    p_sec = client.post(
        "/periods",
        json={"name": "SecA", "start_time": "09:00:00", "end_time": "10:00:00", "days_bitmask": 127, "section": "CSE-A"},
        headers=auth_header,
    ).json()
    p_all = client.post(
        "/periods",
        json={"name": "AllSec", "start_time": "10:00:00", "end_time": "11:00:00", "days_bitmask": 127, "section": None},
        headers=auth_header,
    ).json()

    from app.models import Camera, Student
    from app.services import attendance as svc
    from app.crypto import encrypt_embeddings

    stu_a = Student(name="A", registration_no="SEC_A", section="CSE-A", embeddings=encrypt_embeddings([[1.0] * 512]), photo_paths=[])
    stu_b = Student(name="B", registration_no="SEC_B", section="CSE-B", embeddings=encrypt_embeddings([[1.0] * 512]), photo_paths=[])
    cam = Camera(name="GateSec", type="entry", file_path="x.mp4", sampling_rate=5)
    db.add_all([stu_a, stu_b, cam])
    db.commit()
    for o in (stu_a, stu_b, cam):
        db.refresh(o)

    # 09:30 should match SecA only for stu_a; stu_b gets no period (since SecA != CSE-B and no other period covers 09:30)
    out_a = svc.process_recognition(db, student_id=stu_a.id, camera=cam, ts=datetime(2026, 9, 23, 9, 30, tzinfo=timezone.utc))
    db.commit()
    assert out_a.session.period_id == p_sec["id"]

    out_b = svc.process_recognition(db, student_id=stu_b.id, camera=cam, ts=datetime(2026, 9, 23, 9, 30, tzinfo=timezone.utc))
    db.commit()
    assert out_b.session.period_id is None  # no matching period for CSE-B at 09:30

    # Days bitmask: Monday only (1<<0 =1)
    p_mon = client.post(
        "/periods",
        json={"name": "MonOnly", "start_time": "14:00:00", "end_time": "15:00:00", "days_bitmask": 1},
        headers=auth_header,
    ).json()
    # 2026-09-23 is Wednesday (weekday 2) -> bit 4, not in 1
    stu_c = Student(name="C", registration_no="DAY001", embeddings=encrypt_embeddings([[1.0] * 512]), photo_paths=[])
    db.add(stu_c)
    db.commit()
    db.refresh(stu_c)
    out_c = svc.process_recognition(db, student_id=stu_c.id, camera=cam, ts=datetime(2026, 9, 23, 14, 30, tzinfo=timezone.utc))
    db.commit()
    assert out_c.session.period_id is None

    # Same time on Monday 2026-09-21 should match
    out_mon = svc.process_recognition(db, student_id=stu_c.id, camera=cam, ts=datetime(2026, 9, 21, 14, 30, tzinfo=timezone.utc))
    db.commit()
    assert out_mon.session.period_id == p_mon["id"]
