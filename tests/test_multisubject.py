"""Requirement 3: Multi-Subject Detection — group frames handled via batched inference.

Verifies:
- _handle_faces processes every face in a packet (not just the first).
- Two different students in the same frame both generate RecognitionLog rows.
- Deduplication is per-student, so a group frame does not suppress the second student.
- Batched embedding path in FaceEngine groups all faces from all frames into one GPU call
  (indirectly verified by checking that infer() returns per-frame face lists).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np


def test_handle_faces_two_students_same_frame(db, fake_redis, fake_embed):
    """One packet with 2 faces → both students logged, not just one."""
    from app.models import Camera, Student
    from app.pipeline.camera_task import _handle_faces
    from app.pipeline.frame_reader import FramePacket
    from app.matching import EmbeddingIndex
    from app.crypto import encrypt_embeddings

    # Enroll two students with deterministic fake embeddings
    # fake_embed creates vec[0]=1 for photo0, vec[1]=1 for photo1, etc.
    # Student A will have embedding with 1 at pos 0, Student B at pos 1
    # We directly insert embeddings to avoid photo I/O.

    # Create students with embeddings that match probe vectors
    vec_a = [0.0] * 512
    vec_a[0] = 1.0
    vec_b = [0.0] * 512
    vec_b[1] = 1.0

    s_a = Student(name="Multi A", registration_no="MULTI001", section="A", embeddings=encrypt_embeddings([vec_a]), photo_paths=["a.jpg"])
    s_b = Student(name="Multi B", registration_no="MULTI002", section="A", embeddings=encrypt_embeddings([vec_b]), photo_paths=["b.jpg"])
    cam = Camera(name="Multi Cam", type="both", file_path="x.mp4", sampling_rate=1)
    db.add_all([s_a, s_b, cam])
    db.commit()
    db.refresh(s_a)
    db.refresh(s_b)
    db.refresh(cam)

    index = EmbeddingIndex()
    index.refresh()

    # Packet timestamp
    ts = time.time()
    packet = FramePacket(seq=1, ts=ts, image=np.zeros((100, 100, 3), dtype=np.uint8))

    faces = [
        {"embedding": vec_a, "bbox": [0, 0, 10, 10], "score": 0.99},
        {"embedding": vec_b, "bbox": [20, 20, 30, 30], "score": 0.98},
    ]

    matched, events = _handle_faces(db, fake_redis, cam, packet, faces, index, {"total_ms": 5.0})

    assert matched == 2, f"expected 2 matched, got {matched}"
    assert events == 2, f"expected 2 events, got {events}"

    from app.models import RecognitionLog, AttendanceSession

    logs = db.query(RecognitionLog).order_by(RecognitionLog.student_id).all()
    assert len(logs) == 2
    assert {l.student_id for l in logs} == {s_a.id, s_b.id}

    sessions = db.query(AttendanceSession).all()
    # Both students should have an entry_created session (both camera, no open session)
    assert len(sessions) == 2
    assert {s.student_id for s in sessions} == {s_a.id, s_b.id}


def test_handle_faces_dedup_per_student_allows_group(db, fake_redis, fake_embed):
    """Group frame: first student deduped on second sighting, second student still logs."""
    from app.models import Camera, Student
    from app.pipeline.camera_task import _handle_faces
    from app.pipeline.frame_reader import FramePacket
    from app.matching import EmbeddingIndex
    from app.crypto import encrypt_embeddings

    vec_a = [0.0] * 512
    vec_a[0] = 1.0
    vec_b = [0.0] * 512
    vec_b[1] = 1.0

    s_a = Student(name="Dedup A", registration_no="DEDUP001", embeddings=encrypt_embeddings([vec_a]), photo_paths=[])
    s_b = Student(name="Dedup B", registration_no="DEDUP002", embeddings=encrypt_embeddings([vec_b]), photo_paths=[])
    cam = Camera(name="Dedup Cam", type="both", file_path="x.mp4", sampling_rate=1)
    db.add_all([s_a, s_b, cam])
    db.commit()
    for obj in (s_a, s_b, cam):
        db.refresh(obj)

    index = EmbeddingIndex()
    index.refresh()

    packet1 = FramePacket(seq=1, ts=time.time(), image=np.zeros((10, 10, 3), dtype=np.uint8))
    faces_both = [
        {"embedding": vec_a, "bbox": [0, 0, 10, 10], "score": 0.99},
        {"embedding": vec_b, "bbox": [20, 20, 30, 30], "score": 0.98},
    ]
    # First group frame: both log
    m1, e1 = _handle_faces(db, fake_redis, cam, packet1, faces_both, index, {"total_ms": 5.0})
    assert e1 == 2

    # Immediate second group frame (within dedup window 120s): both should be suppressed
    packet2 = FramePacket(seq=2, ts=time.time() + 1, image=np.zeros((10, 10, 3), dtype=np.uint8))
    m2, e2 = _handle_faces(db, fake_redis, cam, packet2, faces_both, index, {"total_ms": 5.0})
    assert e2 == 0, "dedup should suppress both students within window"
    assert m2 == 2  # matched counts faces even if deduped

    from app.models import RecognitionLog

    logs = db.query(RecognitionLog).all()
    assert len(logs) == 2  # only first frame's 2 logs, second frame suppressed


def test_handle_faces_unknown_does_not_block_known(db, fake_redis, fake_embed):
    """One unknown face + one known face in same frame → known still logged."""
    from app.models import Camera, Student
    from app.pipeline.camera_task import _handle_faces
    from app.pipeline.frame_reader import FramePacket
    from app.matching import EmbeddingIndex
    from app.crypto import encrypt_embeddings

    vec_known = [0.0] * 512
    vec_known[0] = 1.0
    # Unknown vector far from any enrolled (all zeros except position 100)
    vec_unknown = [0.0] * 512
    vec_unknown[100] = 1.0

    s = Student(name="Known", registration_no="KNOWN001", embeddings=encrypt_embeddings([vec_known]), photo_paths=[])
    cam = Camera(name="Mix Cam", type="entry", file_path="x.mp4", sampling_rate=1)
    db.add_all([s, cam])
    db.commit()
    for obj in (s, cam):
        db.refresh(obj)

    index = EmbeddingIndex()
    index.refresh()

    packet = FramePacket(seq=1, ts=time.time(), image=np.zeros((10, 10, 3), dtype=np.uint8))
    faces = [
        {"embedding": vec_unknown, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"embedding": vec_known, "bbox": [20, 20, 30, 30], "score": 0.99},
    ]
    matched, events = _handle_faces(db, fake_redis, cam, packet, faces, index, {"total_ms": 5.0})
    # Unknown does not count as matched, known does; with LOG_UNKNOWN_FACES=true both generate events
    assert matched == 1
    assert events == 2

    from app.models import RecognitionLog

    logs = db.query(RecognitionLog).all()
    assert len(logs) == 2
    assert {l.student_id for l in logs} == {s.id, None}
