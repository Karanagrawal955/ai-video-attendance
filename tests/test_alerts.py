"""Requirement 5: Exception & Alert Management — alerts + quality + deviations."""

from __future__ import annotations

from datetime import datetime, time, timezone


def test_alert_crud_and_ack(client, auth_header, db):
    from app.models import Alert
    from app.services import alerts as svc

    # create via service (system-generated)
    a = svc.create_alert(db, None, type="unknown_face", severity="high", payload={"camera": "Gate"}, camera_id=1)
    assert a.id is not None
    assert a.acknowledged_at is None

    # list via API
    r = client.get("/alerts", headers=auth_header)
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["type"] == "unknown_face"

    # filter
    r2 = client.get("/alerts?type=unknown_face", headers=auth_header)
    assert len(r2.json()) == 1
    r3 = client.get("/alerts?type=low_confidence", headers=auth_header)
    assert len(r3.json()) == 0

    # ack
    ack = client.post(f"/alerts/{a.id}/ack", headers=auth_header)
    assert ack.status_code == 200
    assert ack.json()["acknowledged_at"] is not None

    # unacked filter
    assert len(client.get("/alerts?acknowledged=false", headers=auth_header).json()) == 0
    assert len(client.get("/alerts?acknowledged=true", headers=auth_header).json()) == 1

    # get one
    assert client.get(f"/alerts/{a.id}", headers=auth_header).status_code == 200
    assert client.get("/alerts/9999", headers=auth_header).status_code == 404


def test_unknown_and_low_confidence_alerts_via_pipeline(db, fake_redis, fake_embed):
    """_handle_faces should emit alerts for unknown and low-confidence matches."""
    from app.models import Alert, Camera, Student
    from app.pipeline.camera_task import _handle_faces
    from app.pipeline.frame_reader import FramePacket
    from app.matching import EmbeddingIndex
    import time

    # known student with high-confidence embedding
    vec_known = [0.0] * 512
    vec_known[0] = 1.0
    # unknown vector
    vec_unknown = [0.0] * 512
    vec_unknown[100] = 1.0
    # low-confidence: we need a vector that matches but with low score
    # Instead of relying on cosine, we directly trigger low_confidence path
    # by using a known embedding but mocking match.score in the pipeline.
    # Simpler: enroll known, then provide a probe that is close but not identical
    # Here we use vec_known for high confidence, and a second student for low.
    # For deterministic test, we will manually set match score via a patched index.

    from app.crypto import encrypt_embeddings

    s_known = Student(name="Known", registration_no="K001", embeddings=encrypt_embeddings([vec_known]), photo_paths=[])
    cam = Camera(name="Cam", type="entry", file_path="x.mp4", sampling_rate=1)
    db.add_all([s_known, cam])
    db.commit()
    for o in (s_known, cam):
        db.refresh(o)

    index = EmbeddingIndex()
    index.refresh()

    # unknown face -> should create unknown_face alert (LOG_UNKNOWN_FACES=true in tests)
    pkt = FramePacket(seq=1, ts=time.time(), image=None)  # type: ignore[arg-type]
    # Use small numpy zero image to avoid type error; FramePacket.image can be any np array
    import numpy as np

    pkt.image = np.zeros((10, 10, 3), dtype=np.uint8)  # type: ignore[attr-defined]
    faces_unknown = [{"embedding": vec_unknown, "bbox": [0, 0, 10, 10], "score": 0.9}]
    m, e = _handle_faces(db, fake_redis, cam, pkt, faces_unknown, index, {"total_ms": 5.0})
    # unknown generates an event and an alert
    alerts = db.query(Alert).filter_by(type="unknown_face").all()
    assert len(alerts) == 1
    assert alerts[0].severity == "high"

    # low confidence: mock a match with score 0.45
    # Patch index.match to return low score
    orig_match = index.match

    def low_match(emb):
        from app.matching import MatchResult

        return MatchResult(student_id=s_known.id, score=0.45)

    index.match = low_match  # type: ignore[method-assign]
    pkt2 = FramePacket(seq=2, ts=time.time() + 1, image=np.zeros((10, 10, 3), dtype=np.uint8))
    faces_low = [{"embedding": vec_known, "bbox": [0, 0, 10, 10], "score": 0.45}]
    # Need 2 frames for confirmation (but tests have CONFIRMATION_FRAMES=1, so one is enough)
    m2, e2 = _handle_faces(db, fake_redis, cam, pkt2, faces_low, index, {"total_ms": 5.0})
    low_alerts = db.query(Alert).filter_by(type="low_confidence").all()
    assert len(low_alerts) == 1
    assert low_alerts[0].severity == "low"


def test_deviation_absent_and_late(db, client, auth_header):
    """Absent / late alerts for period deviations."""
    from app.models import Camera, Period, Student
    from app.services import alerts as svc
    from datetime import timedelta

    # period 09-10 for section A
    p = client.post(
        "/periods",
        json={"name": "DevP", "start_time": "09:00:00", "end_time": "10:00:00", "days_bitmask": 127, "section": "A"},
        headers=auth_header,
    ).json()
    # student in that section, no attendance yet
    from app.crypto import encrypt_embeddings

    stu = Student(name="DevStu", registration_no="DEV002", section="A", embeddings=encrypt_embeddings([[1.0] * 512]), photo_paths=[])
    cam = Camera(name="GateD", type="entry", file_path="x.mp4", sampling_rate=5)
    db.add_all([stu, cam])
    db.commit()
    for o in (stu, cam):
        db.refresh(o)

    # Evaluate at 09:20 (past absent grace 15min) -> should create absent
    now = datetime(2026, 9, 23, 9, 20, tzinfo=timezone.utc)
    created = svc.evaluate_period_deviations(db, None, now=now)
    assert any(a.type == "student_absent" for a in created)

    # Now create a late entry at 09:10 (5 min after start + 5 grace -> late)
    # First clear alerts
    from app.models import Alert

    db.query(Alert).delete()
    db.commit()
    # Add late session at 09:10
    from app.services import attendance as att

    ts_late = datetime(2026, 9, 23, 9, 10, tzinfo=timezone.utc)
    att.process_recognition(db, student_id=stu.id, camera=cam, ts=ts_late)
    db.commit()
    # Evaluate again at 09:20 -> should be late, not absent
    db2_created = svc.evaluate_period_deviations(db, None, now=now)
    assert any(a.type == "student_late" for a in db2_created)
    assert not any(a.type == "student_absent" for a in db2_created)
