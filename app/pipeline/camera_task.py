"""Per-camera pipeline body - executes inside a Celery worker slot.

Flow inside one Celery child process:

    FrameReader thread (CPU decode + sampling)
        -> bounded queue (drop-oldest)
    main loop (this task):
        collect up to BATCH_SIZE frames (BATCH_FLUSH_MS window)
        -> JPEG encode (CPU)
        -> Redis RPC to the shared GPU inference service (batched)
        -> cosine matching against the in-memory embedding index
        -> dedup gate -> attendance transition + RecognitionLog + WS event

The GPU model is NEVER loaded here in the default (redis) inference mode -
all cameras across all worker slots share the single model instance.
"""

from __future__ import annotations

import logging
import queue as queue_mod
import threading
import time
from datetime import datetime, timezone

import redis as redis_lib

from .. import redis_client as rc
from ..config import settings
from ..db import SessionLocal
from ..inference import client as inf_client
from ..matching import EmbeddingIndex
from ..models import Camera, RecognitionLog, Student
from ..pipeline.frame_reader import FramePacket, FrameReader
from ..services import attendance as attendance_service
from ..services.events import (
    camera_state_event,
    publish,
    recognition_event,
    unknown_event,
)

logger = logging.getLogger("app.pipeline.camera")


def _should_stop(r: redis_lib.Redis, camera_id: int) -> bool:
    try:
        return r.get(rc.cam_stop_key(camera_id)) == "1"
    except redis_lib.RedisError:
        return False  # redis outage: keep running; stop flag will arrive later


def _encode_jpeg(image) -> bytes | None:  # noqa: ANN001
    import cv2

    ok, buf = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality]
    )
    if not ok:
        return None
    return buf.tobytes()


def _collect_batch(
    out_q: "queue.Queue[FramePacket]",
    *,
    max_frames: int,
    flush_ms: int,
    should_stop,
) -> list[FramePacket]:
    try:
        first = out_q.get(timeout=0.25)
    except queue_mod.Empty:
        return []
    deadline = time.monotonic() + flush_ms / 1000.0
    batch = [first]
    while len(batch) < max_frames and time.monotonic() < deadline:
        if should_stop():
            break
        try:
            batch.append(out_q.get_nowait())
        except queue_mod.Empty:
            break
    return batch


def _handle_faces(
    db,
    r: redis_lib.Redis,
    camera: Camera,
    packet: FramePacket,
    faces: list[dict],
    index: EmbeddingIndex,
    timings: dict,
) -> tuple[int, int]:
    """Returns (matched, events) counts. Commits per recognized event."""
    inference_ms = timings.get("total_ms")
    ts = datetime.fromtimestamp(packet.ts, tz=timezone.utc)
    matched = 0
    events = 0

    for face in faces:
        embedding = face.get("embedding")
        if not embedding:
            continue
        match = index.match(embedding)

        if match.student_id is None:
            if settings.log_unknown_faces and rc.dedup_pass(
                r, camera.id, "unknown"
            ):
                db.add(
                    RecognitionLog(
                        student_id=None,
                        camera_id=camera.id,
                        timestamp=ts,
                        confidence_score=match.score,
                        gpu_inference_time_ms=inference_ms,
                    )
                )
                db.commit()
                publish(
                    r,
                    unknown_event(
                        camera=camera,
                        confidence=match.score,
                        inference_ms=inference_ms,
                    ),
                )
                events += 1
            continue

        matched += 1
        # Dedup window: one event per (camera, student) - controls DB growth
        # AND prevents repeated attendance transitions.
        if not rc.dedup_pass(r, camera.id, f"stu:{match.student_id}"):
            if settings.dedup_log_all:
                db.add(
                    RecognitionLog(
                        student_id=match.student_id,
                        camera_id=camera.id,
                        timestamp=ts,
                        confidence_score=match.score,
                        gpu_inference_time_ms=inference_ms,
                    )
                )
                db.commit()
            continue

        db.add(
            RecognitionLog(
                student_id=match.student_id,
                camera_id=camera.id,
                timestamp=ts,
                confidence_score=match.score,
                gpu_inference_time_ms=inference_ms,
            )
        )
        outcome = attendance_service.process_recognition(
            db, student_id=match.student_id, camera=camera, ts=ts
        )
        db.commit()
        events += 1

        if outcome.changed:
            student = db.get(Student, match.student_id)
            if student is not None:
                publish(
                    r,
                    recognition_event(
                        camera=camera,
                        student=student,
                        confidence=match.score,
                        inference_ms=inference_ms,
                        outcome=outcome,
                    ),
                )
    return matched, events


def run_camera_pipeline(camera_id: int, slot: int) -> dict:
    r = rc.get_redis()
    db = SessionLocal()
    stop_event = threading.Event()
    summary: dict = {"camera_id": camera_id, "slot": slot}
    camera: Camera | None = None
    reader: FrameReader | None = None

    camera = db.get(Camera, camera_id)
    if camera is None:
        logger.error("camera %s not found - nothing to run", camera_id)
        db.close()
        return {**summary, "error": "camera not found"}

    source = camera.rtsp_url or camera.file_path or ""
    out_q: queue_mod.Queue[FramePacket] = queue_mod.Queue(
        maxsize=settings.frame_queue_size
    )
    reader = FrameReader(
        camera_id=camera_id,
        source=source,
        sampling_rate=camera.sampling_rate,
        out_q=out_q,
        stop_event=stop_event,
        rtsp_transport=settings.rtsp_transport,
        open_timeout_ms=settings.rtsp_open_timeout_ms,
        read_timeout_ms=settings.rtsp_read_timeout_ms,
        reconnect_initial_delay_s=settings.reconnect_initial_delay_s,
        reconnect_max_delay_s=settings.reconnect_max_delay_s,
    )

    index = EmbeddingIndex()
    index.refresh(version=rc.students_version(r))

    # ------------------------------------------------- mark ourselves alive
    try:
        r.set(rc.cam_state_key(camera_id), "running", ex=86400)
        r.sadd(settings.running_cameras_key, camera_id)
        rc.touch_heartbeat(r, camera_id)
    except redis_lib.RedisError as exc:
        logger.warning("could not write running state: %s", exc)
    publish(
        r,
        camera_state_event(
            camera_id=camera.id, name=camera.name, state="running", slot=slot
        ),
    )
    reader.start()
    logger.info(
        "camera pipeline running",
        extra={
            "camera_id": camera.id,
            "slot": slot,
            "type": camera.type,
            "sampling_rate": camera.sampling_rate,
            "threshold": index.threshold,
        },
    )

    started = time.time()
    batches = frames_sent = faces_seen = matched_total = events = 0
    rpc_failures = 0
    ema_batch_fps = 0.0

    try:
        while not _should_stop(r, camera_id):
            try:
                rc.touch_heartbeat(r, camera_id)
            except redis_lib.RedisError:
                pass

            batch = _collect_batch(
                out_q,
                max_frames=settings.batch_size,
                flush_ms=settings.batch_flush_ms,
                should_stop=lambda: _should_stop(r, camera_id),
            )
            if not batch:
                continue

            jpegs: list[tuple[FramePacket, bytes]] = []
            for packet in batch:
                jpeg = _encode_jpeg(packet.image)
                if jpeg:
                    jpegs.append((packet, jpeg))
            if not jpegs:
                continue

            t_rpc = time.perf_counter()
            try:
                response = inf_client.infer_frames([j for _, j in jpegs])
                rpc_failures = 0
            except inf_client.InferenceError as exc:
                rpc_failures += 1
                backoff = min(2 ** min(rpc_failures, 5), 30)
                logger.error(
                    "inference RPC failed (%d consecutive): %s - backing off %ds",
                    rpc_failures,
                    exc,
                    backoff,
                    extra={"camera_id": camera_id},
                )
                stop_event.wait(backoff)
                continue

            rpc_ms = (time.perf_counter() - t_rpc) * 1000.0
            timings = response.get("timings", {})
            batch_faces = 0
            for (packet, _), faces in zip(jpegs, response["results"]):
                batch_faces += len(faces)
                m, ev = _handle_faces(
                    db, r, camera, packet, faces, index, timings
                )
                matched_total += m
                events += ev

            batches += 1
            frames_sent += len(jpegs)
            faces_seen += batch_faces
            fps = len(jpegs) / max(rpc_ms / 1000.0, 1e-6)
            ema_batch_fps += 0.1 * (fps - ema_batch_fps)
            index.maybe_refresh()

            logger.info(
                "camera batch processed",
                extra={
                    "camera_id": camera_id,
                    "frames": len(jpegs),
                    "faces": batch_faces,
                    "matched": matched_total,
                    "events": events,
                    "rpc_ms": round(rpc_ms, 1),
                    "gpu_det_ms": timings.get("det_ms"),
                    "gpu_emb_ms": timings.get("emb_ms"),
                    "gpu_total_ms": timings.get("total_ms"),
                    "batch_fps": round(fps, 2),
                    "ema_batch_fps": round(ema_batch_fps, 2),
                    "queue_depth": out_q.qsize(),
                },
            )
    finally:
        stop_event.set()
        if reader is not None:
            reader.join(timeout=3)
        # ------------------------------------------------ cleanup all keys
        try:
            with r.pipeline() as pipe:
                pipe.delete(rc.cam_state_key(camera_id))
                pipe.delete(rc.cam_stop_key(camera_id))
                pipe.delete(rc.cam_heartbeat_key(camera_id))
                pipe.delete(rc.cam_task_key(camera_id))
                pipe.srem(settings.running_cameras_key, camera_id)
                pipe.execute()
            owner = r.get(rc.cam_slot_key(slot))
            if owner is not None and int(owner) == camera_id:
                r.delete(rc.cam_slot_key(slot))
        except redis_lib.RedisError:
            pass
        db.close()

        summary.update(
            {
                "duration_s": round(time.time() - started, 1),
                "batches": batches,
                "frames": frames_sent,
                "faces": faces_seen,
                "matched": matched_total,
                "events": events,
            }
        )
        logger.info("camera pipeline stopped", extra=summary)
        publish(
            r,
            camera_state_event(
                camera_id=camera_id,
                name=camera.name if camera else f"#{camera_id}",
                state="stopped",
                slot=slot,
            ),
        )
    return summary
