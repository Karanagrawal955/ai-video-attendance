"""Redis RPC client for the shared inference service.

Protocol (all JSON over Redis lists):

* request  -> LPUSH ``infer:requests``  ``{id, kind, frames: [b64jpeg...], enqueued_at}``
* response <- BLPOP ``infer:response:{id}``  ``{ok, kind, results, timings}``

One RPC may carry many frames; the server batches frames from *all* cameras
into single GPU calls.  ``kind`` is ``detect`` (every face + embedding) or
``embed`` (best face per frame - used for enrollment photos).
"""

from __future__ import annotations

import base64
import json
import time
import uuid

import redis as redis_lib

from .. import redis_client as rc
from ..config import settings


class InferenceError(Exception):
    """Inference service returned an error or is unreachable."""


class InferenceTimeout(InferenceError):
    pass


class InferenceServiceDown(InferenceError):
    pass


def _rpc(kind: str, jpegs: list[bytes], timeout: float | None = None) -> dict:
    if not jpegs:
        raise InferenceError("no frames to process")
    request_id = uuid.uuid4().hex
    payload = {
        "id": request_id,
        "kind": kind,
        "frames": [base64.b64encode(f).decode("ascii") for f in jpegs],
        "enqueued_at": time.time(),
    }
    response_key = f"infer:response:{request_id}"
    timeout = timeout or settings.inference_request_timeout_s
    try:
        r = rc.get_rpc_redis()
        r.lpush(settings.inference_queue_key, json.dumps(payload, separators=(",", ":")))
        item = r.blpop(response_key, timeout=timeout)
    except redis_lib.RedisError as exc:
        raise InferenceServiceDown(f"redis unavailable: {exc}") from exc
    except TimeoutError as exc:
        raise InferenceTimeout(f"no inference response within {timeout}s") from exc

    if item is None:
        raise InferenceTimeout(
            f"inference service did not respond within {timeout}s "
            "(is the `inference` service running?)"
        )
    try:
        response = json.loads(item[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise InferenceError("malformed inference response") from exc
    if not response.get("ok"):
        raise InferenceError(response.get("error", "inference failed"))
    return response


def infer_frames(jpegs: list[bytes], timeout: float | None = None) -> dict:
    """Detect all faces + embeddings in a batch of JPEG-encoded frames.

    Returns ``{"results": [[face, ...], ...], "timings": {...}}`` where each
    face is ``{"bbox": [x1,y1,x2,y2], "score": s, "embedding": [512 floats]}``.
    """
    resp = _rpc("detect", jpegs, timeout=timeout)
    return {"results": resp.get("results", []), "timings": resp.get("timings", {})}


def embed_images(photos: list[bytes], timeout: float | None = None) -> list[dict]:
    """Best-face 512-d embedding per image (used for student enrollment)."""
    resp = _rpc("embed", photos, timeout=timeout)
    return list(resp.get("results", []))
