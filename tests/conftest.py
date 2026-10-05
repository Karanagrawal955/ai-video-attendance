"""Pytest fixtures.

Environment is pinned BEFORE any ``app.*`` import so pydantic-settings picks it
up: file-backed SQLite (thread-safe across FastAPI's threadpool), temp data
dir, fixed admin credentials.

Redis is faked (fakeredis) and Celery's ``send_task`` is stubbed for every
test, so no external services are required to run the suite.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- env setup
_TMP_BASE = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "opencode" / "pytest"
_TMP_BASE.mkdir(parents=True, exist_ok=True)
_DB_PATH = _TMP_BASE / f"attendance_{os.getpid()}.db"
if _DB_PATH.exists():
    try:
        _DB_PATH.unlink()
    except PermissionError:
        # Windows holds the file after a previous run; fall back to a fresh name
        import time as _t
        _DB_PATH = _TMP_BASE / f"attendance_{os.getpid()}_{int(_t.time()*1000)}.db"
        if _DB_PATH.exists():
            try:
                _DB_PATH.unlink()
            except PermissionError:
                pass

os.environ.update(
    {
        "DATABASE_URL": f"sqlite:///{_DB_PATH.as_posix()}",
        "DATA_DIR": str(_TMP_BASE / "data"),
        "AUTH_REQUIRED": "true",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": "admin",
        "JWT_SECRET": "test-secret-test-secret-test-secret",
        "LOG_FORMAT": "text",
        "LOG_LEVEL": "WARNING",
        "LOCAL_TIMEZONE": "UTC",
        "CAMERA_SLOTS": "2",
        "FORCE_CPU": "true",
        "INFERENCE_MODE": "redis",
        "DEDUP_WINDOW_SECONDS": "120",
        "CONFIRMATION_FRAMES": "1",
        "CONFIRMATION_WINDOW_S": "5.0",
        "FACE_QUALITY_ENABLED": "false",
        "LOG_UNKNOWN_FACES": "true",
    }
)

import fakeredis  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, SessionLocal, engine  # noqa: E402


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def fake_redis(monkeypatch: pytest.MonkeyPatch):
    """Route every redis access in the app to an in-process fake."""
    import app.redis_client as rc

    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(rc, "get_redis", lambda: fake)
    monkeypatch.setattr(rc, "get_rpc_redis", lambda: fake)
    yield fake


@pytest.fixture(autouse=True)
def _clear_confirm_buffer():
    from app.pipeline import camera_task as _ct

    _ct._CONFIRM_BUFFER.clear()
    yield
    _ct._CONFIRM_BUFFER.clear()


@pytest.fixture(autouse=True)
def fake_celery(monkeypatch: pytest.MonkeyPatch):
    """Stub broker submission/revocation so start/stop work without a worker."""
    import app.api.cameras as cameras_mod

    class _Result:
        id = "test-task-id"

    monkeypatch.setattr(
        cameras_mod.celery_app,
        "send_task",
        lambda *args, **kwargs: _Result(),
    )
    monkeypatch.setattr(
        cameras_mod.celery_app.control,
        "revoke",
        lambda *args, **kwargs: None,
    )


@pytest.fixture()
def db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def client(db):
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_header(client) -> dict[str, str]:
    resp = client.post(
        "/auth/token", json={"username": "admin", "password": "admin"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture()
def fake_embed(monkeypatch: pytest.MonkeyPatch):
    """Stub the GPU embedding RPC: one deterministic vector per photo."""
    import app.services.enrollment as enrollment

    counter = {"n": 0}

    def _embed(photos: list[bytes]) -> list[dict]:
        out = []
        for _ in photos:
            vec = [0.0] * 512
            idx = counter["n"] % 512
            vec[idx] = 1.0
            out.append({"ok": True, "embedding": vec, "score": 0.99, "bbox": [0, 0, 1, 1]})
            counter["n"] += 1
        return out

    monkeypatch.setattr(enrollment, "_embed_photos", _embed)
    return _embed


@pytest.fixture(scope="session")
def test_video_path() -> Path:
    """Return path to the synthetic multi-student test video.

    - Looks for tests/assets/test_multi.mp4 (built by scripts/make_test_video.py).
    - If ffmpeg is not on PATH, prints a clear message but does NOT fail when
      OpenCV can still read the file (cv2 bundles its own ffmpeg dll).
    - Skips gracefully when the video is missing or unreadable.
    """
    import shutil
    import cv2

    video = Path(__file__).resolve().parent / "assets" / "test_multi.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(
            "[test_video_path] ffmpeg not found on PATH — "
            "using OpenCV fallback (cv2.VideoCapture bundles its own ffmpeg). "
            "Generate the video with: python scripts/make_test_video.py",
            file=sys.stderr if hasattr(sys, "stderr") else None,
        )
    if not video.exists():
        pytest.skip(
            f"test video missing at {video} — run: python scripts/make_test_video.py"
        )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        pytest.skip(f"cannot open {video} with cv2.VideoCapture — codec or file missing")
    cap.release()
    return video


def make_photos(n: int = 3) -> list[tuple[str, tuple[str, bytes, str]]]:
    """httpx `files` items are (field_name, (filename, bytes, content_type))."""
    return [
        (
            "photos",
            (f"photo_{i}.jpg", b"\xff\xd8\xff\xe0fakejpegbytes" + bytes([i]), "image/jpeg"),
        )
        for i in range(n)
    ]
