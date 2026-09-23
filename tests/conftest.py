"""Pytest fixtures.

Environment is pinned BEFORE any ``app.*`` import so pydantic-settings picks it
up: file-backed SQLite (thread-safe across FastAPI's threadpool), temp data
dir, fixed admin credentials.

Redis is faked (fakeredis) and Celery's ``send_task`` is stubbed for every
test, so no external services are required to run the suite.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- env setup
_TMP_BASE = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "opencode" / "pytest"
_TMP_BASE.mkdir(parents=True, exist_ok=True)
_DB_PATH = _TMP_BASE / f"attendance_{os.getpid()}.db"
if _DB_PATH.exists():
    _DB_PATH.unlink()

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

    def _embed(photos: list[bytes]) -> list[dict]:
        out = []
        for i, _ in enumerate(photos):
            vec = [0.0] * 512
            vec[i % 512] = 1.0
            out.append({"ok": True, "embedding": vec, "score": 0.99, "bbox": [0, 0, 1, 1]})
        return out

    monkeypatch.setattr(enrollment, "_embed_photos", _embed)
    return _embed


def make_photos(n: int = 3) -> list[tuple[str, tuple[str, bytes, str]]]:
    """httpx `files` items are (field_name, (filename, bytes, content_type))."""
    return [
        (
            "photos",
            (f"photo_{i}.jpg", b"\xff\xd8\xff\xe0fakejpegbytes" + bytes([i]), "image/jpeg"),
        )
        for i in range(n)
    ]
