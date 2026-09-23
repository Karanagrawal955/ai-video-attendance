"""Celery application.

Queues
------
* ``control``       - short control tasks (and the default queue)
* ``jobs``          - miscellaneous background jobs
* ``camera_slot_N`` - ONE dedicated queue per camera slot.  The API claims a
  free slot when you start a camera and enqueues the long-running pipeline
  task onto that slot's queue; the worker pool runs ``CAMERA_SLOTS + 2``
  children with ``prefetch=1`` so a long camera task can never starve the
  control queue or another camera.
"""

from __future__ import annotations

from celery import Celery

from ..config import settings

celery_app = Celery(
    "attendance",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone="UTC",
    task_track_started=True,
    task_acks_late=False,
    worker_prefetch_multiplier=1,  # fair scheduling for long-running tasks
    broker_connection_retry_on_startup=True,
    result_expires=3600,
    task_default_queue="control",
    broker_transport_options={"visibility_timeout": 3600},
    task_routes={
        "camera.run": {"queue": "camera_slot_dynamic"},  # overridden per call
        "jobs.*": {"queue": "jobs"},
    },
)


def slot_queue(slot: int) -> str:
    return settings.slot_queue(slot)


def control_queue() -> str:
    return "control"


def jobs_queue() -> str:
    return "jobs"


def all_queues() -> list[str]:
    queues = [control_queue(), jobs_queue()]
    queues.extend(slot_queue(i) for i in range(settings.camera_slots))
    return queues
