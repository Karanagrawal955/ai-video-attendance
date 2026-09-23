"""System endpoints: health + GPU/inference telemetry."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text

from .. import redis_client as rc
from ..config import settings
from ..db import engine as db_engine
from ..schemas import GpuStatusOut, HealthCheck
from .deps import get_current_admin

router = APIRouter(prefix="/system", tags=["system"])

STALE_AFTER_S = 10.0


@router.get("/health", response_model=HealthCheck,
            summary="Liveness/readiness (503 when DB is down)")
def health(response: Response) -> HealthCheck:
    checks: dict[str, str] = {}
    try:
        with db_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "up"
    except Exception:  # noqa: BLE001
        checks["database"] = "down"
        response.status_code = 503

    checks["redis"] = "up" if rc.ping() else "down"

    gpu_age = _gpu_status_age()
    if gpu_age is None:
        checks["inference"] = "unknown"
    elif gpu_age <= STALE_AFTER_S:
        checks["inference"] = "online"
    else:
        checks["inference"] = f"stale ({int(gpu_age)}s)"

    status_value = "ok" if all(
        v == "up" for k, v in checks.items() if k in ("database", "redis")
    ) and checks["inference"] == "online" else "degraded"
    if checks["database"] == "down":
        status_value = "degraded"
    return HealthCheck(status=status_value, version=settings.version, checks=checks)


@router.get("/gpu-status", response_model=GpuStatusOut,
            summary="GPU utilization, VRAM, inference queue depth, throughput")
def gpu_status(_: str = Depends(get_current_admin)) -> GpuStatusOut:
    try:
        r = rc.get_redis()
        data = r.hgetall(settings.gpu_status_key)
        queue_len = r.llen(settings.inference_queue_key)
        running_cameras = len(
            [c for c in r.smembers(settings.running_cameras_key) if r.exists(f"cam:{c}:hb")]
        )
    except Exception:  # noqa: BLE001 - redis down
        return GpuStatusOut(
            service="offline",
            inference_mode=settings.inference_mode,
            detail="redis unavailable",
        )

    if not data:
        return GpuStatusOut(
            service="offline",
            inference_mode=settings.inference_mode,
            queue_length=max(0, queue_len),
            queue_key=settings.inference_queue_key,
            cameras_running=running_cameras,
            detail="inference service has not published status yet "
                   "(is the `inference` service running?)",
        )

    updated_at_epoch = float(data.get("updated_at", 0) or 0)
    age = time.time() - updated_at_epoch if updated_at_epoch else None
    service = "online" if age is not None and age <= STALE_AFTER_S else "offline"
    try:
        providers = json.loads(data.get("providers") or "[]")
    except (TypeError, ValueError):
        providers = []

    throughput = {
        key: _num(data.get(src))
        for key, src in (
            ("batches_total", "batches_total"),
            ("requests_total", "requests_total"),
            ("frames_total", "frames_total"),
            ("faces_total", "faces_total"),
            ("avg_batch_frames", "avg_batch_frames"),
            ("ema_total_ms", "ema_total_ms"),
            ("ema_det_ms", "ema_det_ms"),
            ("ema_emb_ms", "ema_emb_ms"),
            ("ema_fps", "ema_fps"),
            ("dropped_stale_total", "dropped_stale_total"),
        )
        if data.get(src) is not None
    }

    return GpuStatusOut(
        service=service,  # type: ignore[arg-type]
        updated_at=(
            datetime.fromtimestamp(updated_at_epoch, tz=timezone.utc)
            if updated_at_epoch
            else None
        ),
        provider=data.get("provider") or None,
        providers=providers,
        model=data.get("model") or None,
        batched_detection=data.get("batched_detection") == "1",
        device_name=data.get("device_name") or None,
        utilization_percent=_num(data.get("utilization_percent")),
        memory_used_mb=_num(data.get("memory_used_mb")),
        memory_total_mb=_num(data.get("memory_total_mb")),
        queue_length=max(0, queue_len),
        queue_key=settings.inference_queue_key,
        inference_mode=settings.inference_mode,
        throughput=throughput,
        cameras_running=running_cameras,
        detail=None
        if service == "online"
        else f"stale (last update {int(age or 0)}s ago)",
    )


def _num(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gpu_status_age() -> float | None:
    try:
        r = rc.get_redis()
        raw = r.hget(settings.gpu_status_key, "updated_at")
        if not raw:
            return None
        return time.time() - float(raw)
    except Exception:  # noqa: BLE001
        return None
