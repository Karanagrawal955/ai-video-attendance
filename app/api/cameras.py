"""Camera registration + start/stop control.

Slot model: the worker exposes ``CAMERA_SLOTS`` dedicated Celery queues
(``camera_slot_0 .. N-1``).  Starting a camera claims a free slot (Redis
NX + stale-heartbeat reclaim), enqueues the long-running pipeline task onto
that queue, and records state keys.  Stopping sets a stop flag that the
pipeline checks every few hundred milliseconds; ``force=true`` additionally
revokes the Celery task and clears state immediately.
"""

from __future__ import annotations

import logging

import redis as redis_lib
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import redis_client as rc
from ..config import settings
from ..models import Camera
from ..schemas import (
    CameraCreate,
    CameraOut,
    CameraRuntime,
    CameraUpdate,
    OkResponse,
    StartStopResponse,
)
from ..services.events import camera_state_event, publish
from ..workers.celery_app import celery_app
from .deps import get_current_admin, get_db

logger = logging.getLogger("app.api.cameras")

router = APIRouter(prefix="/cameras", tags=["cameras"])


# --------------------------------------------------------------------- helpers
def _camera_out(camera: Camera) -> CameraOut:
    rt = rc.get_camera_runtime(camera_id=camera.id)
    runtime = CameraRuntime(
        state=rt["state"],  # type: ignore[arg-type]
        running=rt["running"],
        slot=rt["slot"],
        task_id=rt["task_id"],
        last_heartbeat_at=rt["last_heartbeat_at"],
    )
    return CameraOut(
        id=camera.id,
        name=camera.name,
        location=camera.location,
        rtsp_url=camera.rtsp_url,
        file_path=camera.file_path,
        type=camera.type,  # type: ignore[arg-type]
        sampling_rate=camera.sampling_rate,
        created_at=camera.created_at,
        updated_at=camera.updated_at,
        runtime=runtime,
    )


def _get_or_404(db: Session, camera_id: int) -> Camera:
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"camera {camera_id} not found")
    return camera


def _redis_or_503() -> redis_lib.Redis:
    try:
        r = rc.get_redis()
        r.ping()
        return r
    except redis_lib.RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"redis unavailable: {exc}",
        ) from exc


def _celery_or_503():
    return celery_app  # imported at module load; connection happens on send


def _acquire_slot(r: redis_lib.Redis, camera_id: int) -> int | None:
    for i in range(settings.camera_slots):
        key = rc.cam_slot_key(i)
        owner = r.get(key)
        if owner is not None and int(owner) == camera_id:
            return i
        if owner is not None:
            if rc.heartbeat_ok(r, int(owner)):
                continue  # live owner occupies this slot
            r.delete(key)  # stale owner (worker died) -> reclaim
        if r.set(key, camera_id, nx=True):
            return i
    return None


# ---------------------------------------------------------------------- routes
@router.post(
    "",
    response_model=CameraOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a camera (RTSP URL and/or local video file)",
)
def create_camera(
    body: CameraCreate,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> CameraOut:
    existing = db.scalars(select(Camera).where(Camera.name == body.name)).first()
    if existing is not None:
        raise HTTPException(
            status_code=409, detail=f"camera name {body.name!r} already exists"
        )
    camera = Camera(
        name=body.name.strip(),
        location=body.location,
        rtsp_url=body.rtsp_url,
        file_path=body.file_path,
        type=body.type,
        sampling_rate=body.sampling_rate,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return _camera_out(camera)


@router.get("", response_model=list[CameraOut], summary="List cameras + runtime state")
def list_cameras(
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> list[CameraOut]:
    cameras = db.scalars(select(Camera).order_by(Camera.id.asc())).all()
    return [_camera_out(c) for c in cameras]


@router.get("/{camera_id}", response_model=CameraOut, summary="Get one camera")
def get_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> CameraOut:
    return _camera_out(_get_or_404(db, camera_id))


@router.put("/{camera_id}", response_model=CameraOut,
            summary="Update camera config (stop it first to apply sampling_rate)")
def update_camera(
    camera_id: int,
    body: CameraUpdate,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> CameraOut:
    camera = _get_or_404(db, camera_id)
    fields = body.model_dump(exclude_unset=True)
    if "name" in fields and fields["name"] != camera.name:
        clash = db.scalars(
            select(Camera).where(Camera.name == fields["name"])
        ).first()
        if clash is not None:
            raise HTTPException(status_code=409, detail="camera name already exists")
    for key, value in fields.items():
        setattr(camera, key, value)
    db.commit()
    db.refresh(camera)
    return _camera_out(camera)


@router.delete("/{camera_id}", response_model=OkResponse,
               summary="Delete a camera (must be stopped first)")
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> OkResponse:
    camera = _get_or_404(db, camera_id)
    runtime = rc.get_camera_runtime(camera_id=camera_id)
    if runtime["state"] in ("starting", "running", "stopping", "unhealthy"):
        raise HTTPException(
            status_code=409,
            detail=f"camera is {runtime['state']} - stop it first "
            f"(POST /cameras/{camera_id}/stop)",
        )
    db.delete(camera)
    db.commit()
    # Clear any leftover runtime keys.
    try:
        r = rc.get_redis()
        with r.pipeline() as pipe:
            for key in (
                rc.cam_state_key(camera_id),
                rc.cam_stop_key(camera_id),
                rc.cam_heartbeat_key(camera_id),
                rc.cam_task_key(camera_id),
            ):
                pipe.delete(key)
            pipe.srem(settings.running_cameras_key, camera_id)
            pipe.execute()
    except redis_lib.RedisError:
        pass
    return OkResponse(detail="camera deleted")


@router.post("/{camera_id}/start", response_model=StartStopResponse,
             summary="Start the processing pipeline for a camera")
def start_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> StartStopResponse:
    camera = _get_or_404(db, camera_id)
    r = _redis_or_503()

    runtime = rc.get_camera_runtime(r=r, camera_id=camera_id)
    if runtime["state"] in ("starting", "running", "stopping", "unhealthy"):
        raise HTTPException(
            status_code=409,
            detail=f"camera already {runtime['state']} (slot={runtime['slot']}); "
                   "stop it first if you want to restart it",
        )

    slot = _acquire_slot(r, camera_id)
    if slot is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"all {settings.camera_slots} camera slots are busy - "
                   "stop another camera or raise CAMERA_SLOTS",
        )

    r.delete(rc.cam_stop_key(camera_id))
    r.set(rc.cam_state_key(camera_id), "starting", ex=86400)

    queue_name = settings.slot_queue(slot)
    try:
        result = celery_app.send_task(
            "camera.run", args=[camera_id, slot], queue=queue_name
        )
    except Exception as exc:  # noqa: BLE001 - broker down
        with r.pipeline() as pipe:
            pipe.delete(rc.cam_state_key(camera_id))
            pipe.delete(rc.cam_slot_key(slot))
            pipe.execute()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"could not enqueue camera task (broker down?): {exc}",
        ) from exc

    r.set(rc.cam_task_key(camera_id), result.id, ex=86400)
    r.sadd(settings.running_cameras_key, camera_id)
    publish(
        r,
        camera_state_event(
            camera_id=camera.id, name=camera.name, state="starting", slot=slot
        ),
    )
    logger.info("camera start requested", extra={"camera_id": camera_id, "slot": slot})
    return StartStopResponse(
        camera_id=camera_id, state="starting", slot=slot, task_id=result.id
    )


@router.post("/{camera_id}/stop", response_model=StartStopResponse,
             summary="Stop the processing pipeline (force=true revokes the task)")
def stop_camera(
    camera_id: int,
    force: bool = Query(default=False,
                        description="Immediately revoke the Celery task and clear state"),
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> StartStopResponse:
    camera = _get_or_404(db, camera_id)
    r = _redis_or_503()

    runtime = rc.get_camera_runtime(r=r, camera_id=camera_id)
    if runtime["state"] == "stopped" and not runtime["running"]:
        return StartStopResponse(
            camera_id=camera_id, state="stopped", detail="already stopped"
        )

    r.set(rc.cam_stop_key(camera_id), "1", ex=3600)
    r.set(rc.cam_state_key(camera_id), "stopping", ex=86400)

    if force:
        task_id = runtime["task_id"]
        if task_id:
            try:
                celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
            except Exception as exc:  # noqa: BLE001
                logger.warning("revoke failed: %s", exc)
        with r.pipeline() as pipe:
            for key in (
                rc.cam_state_key(camera_id),
                rc.cam_stop_key(camera_id),
                rc.cam_heartbeat_key(camera_id),
                rc.cam_task_key(camera_id),
            ):
                pipe.delete(key)
            pipe.srem(settings.running_cameras_key, camera_id)
            pipe.execute()
        slot = runtime["slot"]
        if slot is not None:
            owner = r.get(rc.cam_slot_key(slot))
            if owner is not None and int(owner) == camera_id:
                r.delete(rc.cam_slot_key(slot))
        state = "stopped"
        publish(
            r,
            camera_state_event(
                camera_id=camera.id, name=camera.name, state="stopped", slot=slot
            ),
        )
    else:
        state = "stopping"
        publish(
            r,
            camera_state_event(
                camera_id=camera.id,
                name=camera.name,
                state="stopping",
                slot=runtime["slot"],
            ),
        )

    logger.info(
        "camera stop requested",
        extra={"camera_id": camera_id, "force": force},
    )
    return StartStopResponse(
        camera_id=camera_id,
        state=state,
        slot=runtime["slot"],
        task_id=runtime["task_id"],
        detail=None if force else "stop flag set; poll GET /cameras/"
        f"{camera_id} until state=stopped",
    )
