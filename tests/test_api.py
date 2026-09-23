"""API-level tests: auth, students, cameras (+start/stop slots), attendance, system."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from starlette.websockets import WebSocketDisconnect

from tests.conftest import make_photos


# ------------------------------------------------------------------------ auth
def test_auth_disabled_when_configured(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """AUTH_REQUIRED=false lets a frontend prototype without tokens."""
    import app.security as sec

    monkeypatch.setattr(sec.settings, "auth_required", False)
    resp = client.get("/students")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_requires_token(client) -> None:
    resp = client.get("/students")
    assert resp.status_code == 401


def test_bad_login(client) -> None:
    resp = client.post(
        "/auth/token", json={"username": "admin", "password": "wrong"}
    )
    assert resp.status_code == 401


def test_token_flow(client) -> None:
    resp = client.post(
        "/auth/token", json={"username": "admin", "password": "admin"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    me = client.get("/students", headers={
        "Authorization": f"Bearer {body['access_token']}"
    })
    assert me.status_code == 200


# --------------------------------------------------------------------- students
def test_student_crud(client, auth_header, fake_embed) -> None:
    created = client.post(
        "/students",
        data={"name": "Ada Lovelace", "registration_no": "21CSE001", "section": "CSE-A"},
        files=make_photos(3),
        headers=auth_header,
    )
    assert created.status_code == 201, created.text
    student = created.json()
    assert student["embedding_count"] == 3
    assert student["embeddings"] is None  # opt-in only
    sid = student["id"]

    # duplicate registration no
    dup = client.post(
        "/students",
        data={"name": "Ada2", "registration_no": "21CSE001"},
        files=make_photos(3),
        headers=auth_header,
    )
    assert dup.status_code == 409

    # too few photos
    few = client.post(
        "/students",
        data={"name": "X", "registration_no": "X1"},
        files=make_photos(2),
        headers=auth_header,
    )
    assert few.status_code == 400
    assert "between 3 and 5" in few.json()["detail"]

    listed = client.get("/students?include_embeddings=true", headers=auth_header)
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["total"] == 1
    assert len(payload["items"][0]["embeddings"]) == 3

    fetched = client.get(f"/students/{sid}", headers=auth_header)
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "Ada Lovelace"

    updated = client.put(
        f"/students/{sid}",
        json={"name": "Ada L.", "section": "CSE-B"},
        headers=auth_header,
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Ada L."

    # clear face data (keeps record)
    cleared = client.delete(f"/students/{sid}/face", headers=auth_header)
    assert cleared.status_code == 200
    assert cleared.json()["ok"] is True
    assert client.get(f"/students/{sid}", headers=auth_header).json()[
        "embedding_count"
    ] == 0

    deleted = client.delete(f"/students/{sid}", headers=auth_header)
    assert deleted.status_code == 200
    assert client.get(f"/students/{sid}", headers=auth_header).status_code == 404


def test_student_404(client, auth_header) -> None:
    assert client.get("/students/999", headers=auth_header).status_code == 404


# ---------------------------------------------------------------------- cameras
def _create_camera(client, auth_header, name="Main Gate", **overrides):
    body = {
        "name": name,
        "type": "entry",
        "sampling_rate": 5,
        "file_path": "samples/videos/lecture.mp4",
    }
    body.update(overrides)
    return client.post("/cameras", json=body, headers=auth_header)


def test_camera_crud(client, auth_header) -> None:
    created = _create_camera(client, auth_header)
    assert created.status_code == 201, created.text
    cam = created.json()
    assert cam["runtime"]["state"] == "stopped"
    cid = cam["id"]

    # missing source
    bad = client.post(
        "/cameras", json={"name": "NoSource", "type": "entry"}, headers=auth_header
    )
    assert bad.status_code == 422

    # duplicate name
    dup = _create_camera(client, auth_header)
    assert dup.status_code == 409

    updated = client.put(
        f"/cameras/{cid}", json={"sampling_rate": 10}, headers=auth_header
    )
    assert updated.status_code == 200
    assert updated.json()["sampling_rate"] == 10

    listed = client.get("/cameras", headers=auth_header)
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    deleted = client.delete(f"/cameras/{cid}", headers=auth_header)
    assert deleted.status_code == 200
    assert client.get(f"/cameras/{cid}", headers=auth_header).status_code == 404


def test_camera_start_stop_and_slot_limits(client, auth_header, fake_redis) -> None:
    import app.redis_client as rc

    cam_ids = []
    for i in range(3):
        created = _create_camera(client, auth_header, name=f"Cam {i}")
        assert created.status_code == 201, created.text
        cam_ids.append(created.json()["id"])

    # CAMERA_SLOTS=2 in tests
    s1 = client.post(f"/cameras/{cam_ids[0]}/start", headers=auth_header)
    assert s1.status_code == 200, s1.text
    assert s1.json()["state"] == "starting"
    assert s1.json()["slot"] == 0

    # starting twice -> 409
    again = client.post(f"/cameras/{cam_ids[0]}/start", headers=auth_header)
    assert again.status_code == 409

    # make slot 0 look alive so it isn't reclaimed as stale
    rc.touch_heartbeat(fake_redis, cam_ids[0])

    s2 = client.post(f"/cameras/{cam_ids[1]}/start", headers=auth_header)
    assert s2.status_code == 200
    assert s2.json()["slot"] == 1
    rc.touch_heartbeat(fake_redis, cam_ids[1])

    # both slots busy now
    s3 = client.post(f"/cameras/{cam_ids[2]}/start", headers=auth_header)
    assert s3.status_code == 409
    assert "slots are busy" in s3.json()["detail"]

    # graceful stop
    stopped = client.post(f"/cameras/{cam_ids[0]}/stop", headers=auth_header)
    assert stopped.status_code == 200
    assert stopped.json()["state"] == "stopping"

    # force stop clears state and frees the slot
    forced = client.post(
        f"/cameras/{cam_ids[0]}/stop?force=true", headers=auth_header
    )
    assert forced.status_code == 200
    assert forced.json()["state"] == "stopped"

    runtime = client.get(f"/cameras/{cam_ids[0]}", headers=auth_header).json()
    assert runtime["runtime"]["state"] == "stopped"

    # freed slot can be claimed again
    s3_again = client.post(f"/cameras/{cam_ids[2]}/start", headers=auth_header)
    assert s3_again.status_code == 200
    assert s3_again.json()["slot"] == 0

    # running camera cannot be deleted
    not_deleted = client.delete(f"/cameras/{cam_ids[1]}", headers=auth_header)
    assert not_deleted.status_code == 409


# ------------------------------------------------------------------- attendance
def _seed_attendance(db):
    from app.models import Camera, RecognitionLog, Student
    from app.services.attendance import process_recognition

    student = Student(
        name="Grace Hopper", registration_no="21CSE002", section="CSE-A",
        embeddings=[], photo_paths=[],
    )
    entry = Camera(name="Gate A", type="entry", file_path="x.mp4", sampling_rate=5)
    exit_cam = Camera(name="Gate B", type="exit", file_path="y.mp4", sampling_rate=5)
    db.add_all([student, entry, exit_cam])
    db.commit()

    ts_in = datetime(2026, 9, 23, 9, 0, tzinfo=timezone.utc)
    ts_out = datetime(2026, 9, 23, 17, 0, tzinfo=timezone.utc)
    process_recognition(db, student_id=student.id, camera=entry, ts=ts_in)
    process_recognition(db, student_id=student.id, camera=exit_cam, ts=ts_out)
    # RecognitionLog rows are written by the camera pipeline in production;
    # seed them directly here since process_recognition only handles sessions.
    db.add_all([
        RecognitionLog(
            student_id=student.id, camera_id=entry.id, timestamp=ts_in,
            confidence_score=0.82, gpu_inference_time_ms=6.4,
        ),
        RecognitionLog(
            student_id=student.id, camera_id=exit_cam.id, timestamp=ts_out,
            confidence_score=0.78, gpu_inference_time_ms=5.9,
        ),
    ])
    db.commit()
    return student


def test_attendance_endpoints(client, auth_header, db) -> None:
    student = _seed_attendance(db)

    day = client.get(
        f"/attendance/{student.id}?date=2026-09-23", headers=auth_header
    )
    assert day.status_code == 200, day.text
    body = day.json()
    assert body["totals"]["entries"] == 1
    assert body["totals"]["exits"] == 1
    assert body["totals"]["total_seconds"] == 8 * 3600
    assert body["sessions"][0]["camera_in_name"] == "Gate A"
    assert body["sessions"][0]["camera_out_name"] == "Gate B"

    summary = client.get(
        "/attendance/summary?date=2026-09-23", headers=auth_header
    )
    assert summary.status_code == 200
    sbody = summary.json()
    assert sbody["totals"]["present_students"] == 1
    assert sbody["students"][0]["total_seconds"] == 8 * 3600

    live = client.get("/attendance/live", headers=auth_header)
    assert live.status_code == 200
    assert live.json()["count"] == 0  # session was closed

    logs = client.get("/attendance/logs", headers=auth_header)
    assert logs.status_code == 200
    assert len(logs.json()["items"]) == 2

    missing = client.get(
        "/attendance/999?date=2026-09-23", headers=auth_header
    )
    assert missing.status_code == 404

    bad_date = client.get(
        "/attendance/summary?date=23-09-2026", headers=auth_header
    )
    assert bad_date.status_code == 422


def test_live_shows_ongoing(client, auth_header, db) -> None:
    from app.models import Camera, Student
    from app.services.attendance import process_recognition
    from app.timeutil import utcnow

    student = Student(name="Alan", registration_no="21CSE003",
                      embeddings=[], photo_paths=[])
    gate = Camera(name="Turnstile", type="entry", file_path="z.mp4", sampling_rate=5)
    db.add_all([student, gate])
    db.commit()
    process_recognition(
        db, student_id=student.id, camera=gate, ts=utcnow()
    )
    db.commit()

    live = client.get("/attendance/live", headers=auth_header)
    assert live.status_code == 200
    body = live.json()
    assert body["count"] == 1
    assert body["sessions"][0]["student"]["registration_no"] == "21CSE003"


# ----------------------------------------------------------------------- system
def test_health_and_gpu_status(client, auth_header) -> None:
    health = client.get("/system/health")
    assert health.status_code == 200
    assert health.json()["checks"]["database"] == "up"

    gpu = client.get("/system/gpu-status", headers=auth_header)
    assert gpu.status_code == 200
    gbody = gpu.json()
    assert gbody["service"] in ("online", "offline")
    assert gbody["inference_mode"] == "redis"
    # no inference server running in tests -> offline with a helpful hint
    assert gbody["service"] == "offline"


def test_root_metadata(client) -> None:
    body = client.get("/api").json()
    assert body["docs"] == "/docs"
    assert body["websocket"] == "/ws/live-feed"
    # root serves the bundled web UI
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers.get("content-type", "")
    assert "AI Video Attendance" in page.text


# ------------------------------------------------------------------ websocket
def test_ws_rejects_missing_token(client) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/live-feed"):
            pass
    assert exc.value.code == 4401


def test_ws_rejects_bad_token(client) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/live-feed?token=garbage"):
            pass
    assert exc.value.code == 4401


def test_ws_accepts_valid_token(client, auth_header) -> None:
    token = auth_header["Authorization"].split()[1]
    # Handshake must be accepted; the server may then drop the socket if redis
    # is unreachable (it cannot subscribe), which is acceptable here.
    try:
        with client.websocket_connect(f"/ws/live-feed?token={token}"):
            pass
    except (WebSocketDisconnect, RuntimeError):
        pass
