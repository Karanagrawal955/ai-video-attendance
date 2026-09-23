"""WebSocket event payloads published to ``events:live``."""

from __future__ import annotations

import logging
from typing import Any

import redis as redis_lib

from .. import redis_client
from ..models import Camera, Student
from ..timeutil import utcnow
from .attendance import RecogOutcome

logger = logging.getLogger("app.events")


def _session_block(outcome: RecogOutcome) -> dict[str, Any] | None:
    s = outcome.session
    if s is None:
        return None
    return {
        "id": s.id,
        "action": outcome.action,
        "status": s.status,
        "date": s.date.isoformat(),
        "entry_time": s.entry_time,
        "exit_time": s.exit_time,
        "total_duration": s.total_duration,
    }


def recognition_event(
    *,
    camera: Camera,
    student: Student,
    confidence: float,
    inference_ms: float | None,
    outcome: RecogOutcome,
) -> dict[str, Any]:
    return {
        "type": "recognition",
        "ts": utcnow().isoformat(),
        "camera": {
            "id": camera.id,
            "name": camera.name,
            "type": camera.type,
            "location": camera.location,
        },
        "student": {
            "id": student.id,
            "name": student.name,
            "registration_no": student.registration_no,
            "section": student.section,
        },
        "confidence": round(float(confidence), 4),
        "inference_ms": round(float(inference_ms), 2) if inference_ms else None,
        "session_action": outcome.action,
        "session": _session_block(outcome),
    }


def unknown_event(
    *, camera: Camera, confidence: float, inference_ms: float | None
) -> dict[str, Any]:
    return {
        "type": "unknown_face",
        "ts": utcnow().isoformat(),
        "camera": {
            "id": camera.id,
            "name": camera.name,
            "type": camera.type,
            "location": camera.location,
        },
        "confidence": round(float(confidence), 4),
        "inference_ms": round(float(inference_ms), 2) if inference_ms else None,
    }


def camera_state_event(
    *, camera_id: int, name: str, state: str, slot: int | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "camera_state",
        "ts": utcnow().isoformat(),
        "camera_id": camera_id,
        "camera_name": name,
        "state": state,
        "slot": slot,
        "detail": detail,
    }


def publish(r: redis_lib.Redis | None, payload: dict[str, Any]) -> None:
    """Best-effort publish - a broken dashboard must never kill the pipeline."""
    try:
        redis_client.publish_event(r, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event publish failed: %s", exc, extra={"type": payload.get("type")})
