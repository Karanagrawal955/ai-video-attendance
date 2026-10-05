"""In-process inference for CLI scripts (no Redis, no worker required).

``scripts/setup_student_database.py`` and ``scripts/test_cctv.py`` must work
on a machine where only the API/DB is running.  Rather than duplicating the
model, this module wraps the **same** ``FaceEngine`` the production service
uses (SCRFD detect -> alignment -> ArcFace 512-d embed) and exposes it
through the two functions ``app.inference.client`` provides, so enrollment,
the quality gate and recognition keep calling exactly the code they always
did.

Usage::

    from app.inference.local import LocalInference
    LocalInference().install()   # patches app.inference.client

Pass ``--rpc`` on the scripts to skip this and go through the shared Redis
inference service instead (production path).
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import cv2
import numpy as np

from ..config import settings
from .engine import FaceEngine

logger = logging.getLogger("app.inference.local")


def _decode(jpeg: bytes) -> np.ndarray | None:
    if not jpeg:
        return None
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


class LocalInference:
    """FaceEngine behind the Redis client's two entry points."""

    def __init__(self, engine: FaceEngine | None = None, *, warmup: bool = True):
        self.engine = engine or FaceEngine(settings)
        if warmup:
            try:
                warm = self.engine.warmup()
                logger.info("local inference ready (%.1f ms)", warm.total_ms)
            except Exception as exc:  # noqa: BLE001 - fall back like the server does
                logger.warning("warmup failed (%s) - retrying on CPU", exc)
                self.engine.reset_to_cpu()
                self.engine.warmup()

    # -- API compatible with app.inference.client ---------------------------
    def infer_frames(self, jpegs: list[bytes], timeout: float | None = None) -> dict:
        """Detect all faces + embeddings in a batch of JPEG frames."""
        del timeout  # nothing to wait for in-process
        images: list[np.ndarray] = []
        for jpeg in jpegs:
            img = _decode(jpeg)
            if img is None:
                images.append(np.zeros((480, 640, 3), dtype=np.uint8))
                logger.warning("unreadable frame passed to inference, padded")
            else:
                images.append(img)
        results, timings = self.engine.infer(images)
        try:
            timing_dict = {k: v for k, v in asdict(timings).items()}
        except Exception:  # noqa: BLE001
            timing_dict = {}
        return {"results": results, "timings": timing_dict}

    def embed_images(self, photos: list[bytes], timeout: float | None = None) -> list[dict]:
        """Best-face 512-d embedding per image (enrollment path)."""
        del timeout
        out: list[dict] = []
        for photo in photos:
            img = _decode(photo)
            if img is None:
                out.append({"ok": False, "reason": "cannot decode image"})
                continue
            results, _ = self.engine.infer([img])
            faces = results[0] if results else []
            if not faces:
                out.append({"ok": False, "reason": "no face detected"})
                continue
            best = faces[0]  # engine sorts by detection score
            out.append(
                {
                    "ok": True,
                    "embedding": [float(x) for x in best["embedding"]],
                    "score": float(best.get("score", 0.0)),
                    "bbox": best.get("bbox"),
                }
            )
        return out

    # -- wiring -------------------------------------------------------------
    def install(self) -> "LocalInference":
        """Patch ``app.inference.client`` so every caller uses this engine."""
        from . import client

        client.infer_frames = self.infer_frames  # type: ignore[assignment]
        client.embed_images = self.embed_images  # type: ignore[assignment]
        logger.info("in-process inference installed (INFERENCE_MODE=inprocess)")
        return self


def install(*, warmup: bool = True) -> LocalInference:
    """Convenience: build and install the local inference service."""
    return LocalInference(warmup=warmup).install()
