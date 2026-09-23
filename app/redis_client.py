"""Redis helpers: sync clients, key naming, dedup, event fan-out.

All REST handlers are synchronous (FastAPI runs them in a threadpool) so they
use the sync client; only the WebSocket bridge uses ``redis.asyncio``.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

import redis as redis_lib

from .config import settings

# ------------------------------------------------------------------ clients
@lru_cache
def get_redis() -> redis_lib.Redis:
    """Shared short-timeout client for API-side operations."""
    return redis_lib.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=5,
    )


@lru_cache
def get_rpc_redis() -> redis_lib.Redis:
    """Client used for inference RPC blocking waits (longer socket timeout)."""
    return redis_lib.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=settings.inference_request_timeout_s + 5,
    )


def ping(url: str | None = None) -> bool:
    try:
        if url:
            client = redis_lib.Redis.from_url(
                url, decode_responses=True,
                socket_connect_timeout=2, socket_timeout=2,
            )
        else:
            client = get_redis()
        return bool(client.ping())
    except Exception:  # noqa: BLE001 - health probe
        return False


# ---------------------------------------------------------------- key names
def cam_state_key(camera_id: int) -> str:
    return f"cam:{camera_id}:state"


def cam_stop_key(camera_id: int) -> str:
    return f"cam:{camera_id}:stop"


def cam_heartbeat_key(camera_id: int) -> str:
    return f"cam:{camera_id}:hb"


def cam_task_key(camera_id: int) -> str:
    return f"cam:{camera_id}:task"


def cam_slot_key(slot: int) -> str:
    return f"slot:{slot}:camera"


def dedup_key(camera_id: int, part: str) -> str:
    return f"dedup:cam:{camera_id}:{part}"


# ------------------------------------------------------------- camera state
def get_camera_runtime(
    r: redis_lib.Redis | None = None, camera_id: int | None = None
) -> dict[str, Any]:
    """Read state/heartbeat/slot for a camera (or all cameras when id is None)."""
    r = r or get_redis()
    out: dict[str, Any] = {
        "state": "stopped",
        "running": False,
        "slot": None,
        "task_id": None,
        "last_heartbeat_at": None,
    }
    if camera_id is None:
        return out
    try:
        state = r.get(cam_state_key(camera_id))
        task_id = r.get(cam_task_key(camera_id))
        hb = r.get(cam_heartbeat_key(camera_id))
    except redis_lib.RedisError:
        return out
    if state:
        out["state"] = state
    if task_id:
        out["task_id"] = task_id
    if hb is not None:
        out["last_heartbeat_at"] = float(hb)
        out["running"] = (
            state == "running" and (time.time() - float(hb)) <= settings.heartbeat_ttl_s
        )
        # Task claims to run but heartbeats expired -> mark unhealthy.
        if state == "running" and not out["running"]:
            out["state"] = "unhealthy"
    out["slot"] = get_camera_slot(r, camera_id)
    return out


def get_camera_slot(r: redis_lib.Redis, camera_id: int) -> int | None:
    try:
        for i in range(settings.camera_slots):
            owner = r.get(cam_slot_key(i))
            if owner is not None and int(owner) == camera_id:
                return i
    except (redis_lib.RedisError, ValueError):
        return None
    return None


def heartbeat_ok(r: redis_lib.Redis, camera_id: int) -> bool:
    try:
        return bool(r.exists(cam_heartbeat_key(camera_id)))
    except redis_lib.RedisError:
        return False


def touch_heartbeat(r: redis_lib.Redis, camera_id: int) -> None:
    r.set(cam_heartbeat_key(camera_id), str(time.time()), ex=settings.heartbeat_ttl_s)


# ------------------------------------------------------------------- dedup
def dedup_pass(
    r: redis_lib.Redis, camera_id: int, part: str,
    window: int | None = None,
) -> bool:
    """Return True exactly once per (camera, subject) within the window."""
    window = window if window is not None else settings.dedup_window_seconds
    if window <= 0:
        return True
    try:
        return bool(
            r.set(dedup_key(camera_id, part), str(time.time()), nx=True, ex=window)
        )
    except redis_lib.RedisError:
        # Redis down must not lose attendance data - treat as "pass".
        return True


# ------------------------------------------------------------- event fanout
def publish_event(r: redis_lib.Redis | None, payload: dict[str, Any]) -> None:
    """Publish to the WebSocket channel and keep a small replay backlog."""
    r = r or get_redis()
    data = json.dumps(payload, default=str, separators=(",", ":"))
    pipe = r.pipeline()
    pipe.publish(settings.events_channel, data)
    pipe.lpush(settings.events_recent_key, data)
    pipe.ltrim(settings.events_recent_key, 0, settings.events_recent_limit - 1)
    pipe.execute()


# -------------------------------------------------------- embedding version
def students_version(r: redis_lib.Redis | None = None) -> str | None:
    r = r or get_redis()
    try:
        return r.get(settings.students_version_key)
    except redis_lib.RedisError:
        return None


def bump_students_version(r: redis_lib.Redis | None = None) -> None:
    r = r or get_redis()
    try:
        r.incr(settings.students_version_key)
    except redis_lib.RedisError:
        pass
