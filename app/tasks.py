"""Celery task definitions."""

from __future__ import annotations

import os

from .workers.celery_app import celery_app


@celery_app.task(name="camera.run", track_started=True)
def camera_run(camera_id: int, slot: int) -> dict:
    """Long-running per-camera pipeline (one Celery child per slot)."""
    from .pipeline.camera_task import run_camera_pipeline

    return run_camera_pipeline(camera_id, slot)


@celery_app.task(name="jobs.ping")
def jobs_ping() -> dict:
    """Trivial task to verify worker/broker health."""
    return {"pong": True, "pid": os.getpid()}


@celery_app.task(name="jobs.close_stale_sessions")
def jobs_close_stale_sessions(days: int = 1) -> dict:
    """Beat-registered task to close stale ongoing sessions."""
    from datetime import timedelta

    from sqlalchemy import select

    from .db import SessionLocal
    from .models import AttendanceSession
    from .services.attendance import _close_session
    from .timeutil import utcnow

    cutoff = utcnow() - timedelta(days=days)
    closed = 0
    with SessionLocal() as db:
        stale = db.scalars(
            select(AttendanceSession).where(
                AttendanceSession.status == "ongoing",
                AttendanceSession.entry_time < cutoff,
            )
        ).all()
        for s in stale:
            _close_session(db, s, utcnow(), camera_id=None)
            closed += 1
        if closed:
            db.commit()
    return {"closed": closed, "days": days}


@celery_app.task(name="jobs.check_period_deviations")
def jobs_check_period_deviations() -> dict:
    """Beat task to evaluate absent/late/early alerts for today's periods."""
    from . import redis_client as rc
    from .db import SessionLocal
    from .services import alerts as _alerts

    r = rc.get_redis()
    with SessionLocal() as db:
        created = _alerts.evaluate_period_deviations(db, r)
    return {"created": len(created), "types": [a.type for a in created]}


@celery_app.task(name="jobs.reap_dead_cameras")
def jobs_reap_dead_cameras() -> dict:
    """Beat-registered reaper for crashed camera pipelines (heartbeat expired)."""
    import time as _t

    from . import redis_client as rc
    from .config import settings

    r = rc.get_redis()
    reaped: list[int] = []
    try:
        # Use the running set as source of truth
        try:
            running = r.smembers(settings.running_cameras_key)
        except Exception:
            running = set()
        for member in list(running or []):
            try:
                cid = int(member)
            except ValueError:
                continue
            runtime = rc.get_camera_runtime(r, cid)
            # unhealthy means heartbeat expired while state still running
            if runtime.get("state") == "unhealthy":
                # force-clear like ?force=true
                try:
                    with r.pipeline() as pipe:
                        pipe.delete(rc.cam_state_key(cid))
                        pipe.delete(rc.cam_stop_key(cid))
                        pipe.delete(rc.cam_heartbeat_key(cid))
                        pipe.delete(rc.cam_task_key(cid))
                        pipe.srem(settings.running_cameras_key, cid)
                        pipe.execute()
                    slot = runtime.get("slot")
                    if slot is not None:
                        owner = r.get(rc.cam_slot_key(int(slot)))
                        if owner is not None and int(owner) == cid:
                            r.delete(rc.cam_slot_key(int(slot)))
                    reaped.append(cid)
                except Exception:
                    pass
        # also reclaim stale slot owners even if not in running set
        for slot in range(settings.camera_slots):
            try:
                owner = r.get(rc.cam_slot_key(slot))
                if owner is None:
                    continue
                ocid = int(owner)
                try:
                    hb = r.get(rc.cam_heartbeat_key(ocid))
                    stale = hb is None or (_t.time() - float(hb) > settings.heartbeat_ttl_s * 2)
                except Exception:
                    stale = True
                if stale and not rc.heartbeat_ok(r, ocid):
                    r.delete(rc.cam_slot_key(slot))
                    if ocid not in reaped:
                        reaped.append(ocid)
            except Exception:
                continue
    except Exception:
        pass
    return {"reaped": sorted(set(reaped))}
