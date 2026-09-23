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
