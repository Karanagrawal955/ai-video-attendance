"""Shared GPU inference service.

Run with::

    python -m app.inference.server

Exactly ONE process loads InsightFace/ONNXRuntime-GPU.  Camera workers and
the API push JPEG frames onto a Redis list; this service coalesces requests
from *all* cameras into single batched GPU calls (N frames x M faces), runs
detection + embedding, and pushes per-request JSON responses back.

It also publishes a ``gpu:status`` hash every 2s (NVML utilization/VRAM,
queue length, throughput EMAs) consumed by ``GET /system/gpu-status``.
"""

from __future__ import annotations

import base64
import json
import logging
import signal
import threading
import time

import numpy as np

from ..config import settings
from ..logging_config import configure

configure(settings)
logger = logging.getLogger("app.inference.server")


# --------------------------------------------------------------------- stats
class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.batches = 0
        self.requests = 0
        self.frames = 0
        self.faces = 0
        self.dropped_stale = 0
        self.ema_batch_frames = 0.0
        self.ema_total_ms = 0.0
        self.ema_det_ms = 0.0
        self.ema_emb_ms = 0.0
        self.ema_fps = 0.0

    def record(
        self, requests: int, frames: int, faces: int, total_ms: float,
        det_ms: float, emb_ms: float,
    ) -> None:
        alpha = 0.1
        with self._lock:
            self.batches += 1
            self.requests += requests
            self.frames += frames
            self.faces += faces
            self.ema_batch_frames += alpha * (frames - self.ema_batch_frames)
            self.ema_total_ms += alpha * (total_ms - self.ema_total_ms)
            self.ema_det_ms += alpha * (det_ms - self.ema_det_ms)
            self.ema_emb_ms += alpha * (emb_ms - self.ema_emb_ms)
            fps = frames / max(total_ms / 1000.0, 1e-6)
            self.ema_fps += alpha * (fps - self.ema_fps)

    def mark_dropped(self, n: int) -> None:
        with self._lock:
            self.dropped_stale += n

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self.started_at, 1),
                "batches_total": self.batches,
                "requests_total": self.requests,
                "frames_total": self.frames,
                "faces_total": self.faces,
                "dropped_stale_total": self.dropped_stale,
                "avg_batch_frames": round(
                    self.frames / self.batches, 2
                ) if self.batches else 0.0,
                "ema_batch_frames": round(self.ema_batch_frames, 2),
                "ema_total_ms": round(self.ema_total_ms, 2),
                "ema_det_ms": round(self.ema_det_ms, 2),
                "ema_emb_ms": round(self.ema_emb_ms, 2),
                "ema_fps": round(self.ema_fps, 1),
            }


# ------------------------------------------------------------------- helpers
def _decode_frame(b64: str) -> np.ndarray | None:
    import cv2

    try:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:  # noqa: BLE001 - malformed input
        return None


def _shape_results(kind: str, frames_results: list) -> list:
    if kind != "embed":
        return frames_results
    shaped = []
    for faces in frames_results:
        if not faces:
            shaped.append({"ok": False, "reason": "no face detected"})
            continue
        top = max(faces, key=lambda f: f.get("score", 0.0))
        shaped.append(
            {
                "ok": True,
                "embedding": top["embedding"],
                "score": top["score"],
                "bbox": top["bbox"],
            }
        )
    return shaped


def _respond(r, request: dict, payload: dict) -> None:
    request_id = request.get("id")
    if not request_id:
        return
    key = f"infer:response:{request_id}"
    try:
        data = json.dumps(payload, separators=(",", ":"))
        pipe = r.pipeline()
        pipe.rpush(key, data)
        pipe.expire(key, settings.inference_response_ttl_s)
        pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to push response: %s", exc)


# ---------------------------------------------------------------- status loop
def _nvml_info() -> dict:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(settings.cuda_device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        return {
            "device_name": name,
            "utilization_percent": float(util.gpu),
            "memory_used_mb": round(mem.used / (1024 * 1024), 1),
            "memory_total_mb": round(mem.total / (1024 * 1024), 1),
        }
    except Exception as exc:  # noqa: BLE001 - NVML optional (CPU fallback)
        return {"nvml_error": str(exc)}


def _status_loop(r, engine, stats: Stats, stop: threading.Event) -> None:
    nvml_warned = False
    while not stop.wait(2.0):
        try:
            queue_len = r.llen(settings.inference_queue_key)
        except Exception:  # noqa: BLE001
            queue_len = -1
        payload: dict = {
            "updated_at": time.time(),
            "pid": str(r.client.pid if hasattr(r.client, "pid") else 0),
            "queue_len": queue_len,
        }
        status = engine.status()
        payload.update(
            {
                "model": status["model"],
                "provider": status["provider"] or "",
                "providers": json.dumps(status["providers"]),
                "batched_detection": "1" if status["batched_detection"] else "0",
                "force_cpu": "1" if status["force_cpu"] else "0",
            }
        )
        nv = _nvml_info()
        if "nvml_error" in nv and not nvml_warned:
            logger.warning("NVML unavailable: %s", nv["nvml_error"])
            nvml_warned = True
        payload.update({k: v for k, v in nv.items() if k != "nvml_error"})
        payload.update(stats.snapshot())
        try:
            r.hset(settings.gpu_status_key, mapping=payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gpu status publish failed: %s", exc)


# ------------------------------------------------------------------- main loop
def run_loop(r, engine, stats: Stats, stop: threading.Event) -> None:
    queue_key = settings.inference_queue_key

    while not stop.is_set():
        try:
            item = r.blpop(queue_key, timeout=1)
        except Exception as exc:  # noqa: BLE001 - redis restarts etc.
            logger.warning("redis error in inference loop: %s", exc)
            stop.wait(1.0)
            continue
        if item is None:
            continue

        # ---- gather a cross-camera batch within a short time window
        raws: list[str] = [item[1]]
        deadline = time.monotonic() + settings.inference_batch_wait_ms / 1000.0
        while (
            len(raws) < settings.max_batch_frames
            and time.monotonic() < deadline
        ):
            nxt = r.lpop(queue_key)
            if nxt:
                raws.append(nxt)
            else:
                time.sleep(0.002)

        # ---- parse + drop stale requests (their client already timed out)
        requests: list[dict] = []
        stale = 0
        for raw in raws:
            try:
                req = json.loads(raw)
            except (TypeError, ValueError):
                continue
            enqueued = float(req.get("enqueued_at", time.time()))
            if time.time() - enqueued > settings.inference_request_timeout_s:
                stale += 1
                continue
            requests.append(req)
        if stale:
            stats.mark_dropped(stale)
        if not requests:
            continue

        # ---- decode frames (CPU; this is the cheapest place to do it)
        images: list[np.ndarray] = []
        plan: list[tuple[int, int, dict]] = []  # (start, count, request)
        for req in requests:
            frames = req.get("frames") or []
            start = len(images)
            ok = True
            for b64 in frames:
                img = _decode_frame(b64)
                if img is None:
                    ok = False
                    break
                images.append(img)
            if not ok or not frames:
                del images[start:]
                _respond(r, req, {"ok": False, "error": "invalid frame data"})
                continue
            plan.append((start, len(frames), req))
        if not images or not plan:
            continue

        # ---- batched GPU inference (with one CPU-recovery retry)
        try:
            results, timings = engine.infer(images)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "inference failed (%s); attempting CPU recovery", exc,
                extra={"frames": len(images)},
            )
            try:
                engine.reset_to_cpu()
                results, timings = engine.infer(images)
                logger.warning("recovered: now running on CPU provider")
            except Exception as exc2:  # noqa: BLE001
                for _, _, req in plan:
                    _respond(
                        r, req, {"ok": False, "error": f"inference failed: {exc2}"}
                    )
                continue

        timing_dict = timings.as_dict()
        timing_dict["batch_frames"] = len(images)
        timing_dict["batch_requests"] = len(plan)

        for start, count, req in plan:
            kind = req.get("kind", "detect")
            _respond(
                r,
                req,
                {
                    "ok": True,
                    "kind": kind,
                    "results": _shape_results(kind, results[start : start + count]),
                    "timings": timing_dict,
                },
            )

        stats.record(
            len(plan), len(images), timings.faces, timings.total_ms,
            timings.det_ms, timings.emb_ms,
        )
        logger.info(
            "inference batch",
            extra={
                "requests": len(plan),
                "frames": len(images),
                "faces": timings.faces,
                "det_ms": round(timings.det_ms, 2),
                "emb_ms": round(timings.emb_ms, 2),
                "total_ms": round(timings.total_ms, 2),
                "batched_det": timings.batched_detection,
                "queue_len": max(0, r.llen(queue_key)),
            },
        )


# ------------------------------------------------------------------------ main
def main() -> None:
    import redis as redis_lib

    from .engine import FaceEngine

    logger.info(
        "starting shared inference service",
        extra={
            "queue": settings.inference_queue_key,
            "max_batch_frames": settings.max_batch_frames,
            "batch_wait_ms": settings.inference_batch_wait_ms,
            "device_index": settings.cuda_device_index,
        },
    )
    engine = FaceEngine()

    # Warmup: forces provider/CUDA initialisation.  If CUDA is broken on this
    # machine we drop to CPU here instead of failing every later request.
    try:
        warm = engine.warmup()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "warmup failed on %s (%s) - retrying on CPU",
            engine.status().get("provider"),
            exc,
        )
        engine.reset_to_cpu()
        warm = engine.warmup()
    logger.info(
        "inference engine ready",
        extra={**engine.status(), "warmup_ms": round(warm.total_ms, 2)},
    )

    r = redis_lib.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=None,  # blocking BLPOP
    )

    stats = Stats()
    stop = threading.Event()

    def _handle(signum, _frame):  # noqa: ANN001
        logger.info("signal %s received - shutting down", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    threading.Thread(
        target=_status_loop,
        args=(r, engine, stats, stop),
        daemon=True,
        name="gpu-status",
    ).start()

    try:
        run_loop(r, engine, stats, stop)
    finally:
        logger.info("inference service stopped", extra=stats.snapshot())


if __name__ == "__main__":
    main()
