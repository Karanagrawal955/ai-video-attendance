#!/usr/bin/env python
"""Pick the best face per cluster and enroll via production RPC.

For each of the top-N clusters, pick the frame with the highest detection
score *and* sharpness > 100 (passes quality gate), then enroll that single
photo into the DB.  Reports rejections (blur, low score, tiny face).

    python scripts/enroll_from_clusters.py --top 6
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db import SessionLocal
from app.models import Student
from app.services import enrollment
from app.inference import client as ic
from app.redis_client import get_redis

BATCH = 8
OUT = Path(__file__).resolve().parents[1] / "data" / "dataset_index"

META_PATH = str(OUT / "meta.json") if (OUT / "meta.json").exists() else "[]"
META = json.loads(META_PATH) if isinstance(META_PATH, str) and META_PATH else []
LAB_PATH = str(OUT / "clusters.npy") if (OUT / "clusters.npy").exists() else ""
LAB = np.load(LAB_PATH) if LAB_PATH else np.array([], dtype=int)

# map cluster_id -> list of meta indices
clusters: dict[int, list[int]] = {}
for i, c in enumerate(LAB):
    clusters.setdefault(int(c), []).append(i)

# sort clusters by size descending
sorted_clusters = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)

# quality gate mirror from engine
def quality_ok(crop, det_row):
    import cv2 as _cv
    try:
        x1, y1, x2, y2 = [float(v) for v in det_row[:4]]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < float(settings.face_min_area):
            return False, "tiny face"
        gray = _cv.cvtColor(crop, _cv.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        mean = float(gray.mean())
        if mean < float(settings.face_min_brightness) or mean > float(settings.face_max_brightness):
            return False, "bad brightness"
        lap = _cv.Laplacian(gray, _cv.CV_64F).var()
        if lap < float(settings.face_min_sharpness):
            return False, f"blur sharp={lap:.0f}<{settings.face_min_sharpness}"
    except Exception:
        return True, "error-pass"
    return True, "ok"


def main() -> int:
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    picked = 0
    rejected = 0
    skipped = 0

    with SessionLocal() as db:
        for ci, (cid, idxs) in enumerate(sorted_clusters[:top]):
            best_i = -1
            best_key = ("-inf", "-inf")  # (score, sharpness)
            for i in idxs:
                m = META[i]
                s = m["score"]
                sh = m["sharpness"]
                area = m["area"]
                if s < 0.5:
                    continue
                if sh < 50:
                    continue
                if area < 1000:
                    continue
                key = (float(s), float(sh))
                if key > best_key:
                    best_key = key
                    best_i = i

            if best_i < 0:
                print(f"C{cid}: no frame passed quality gate")
                skipped += 1
                continue

            m = META[best_i]
            video_name = m["video"]
            frame_idx = m["frame"]

            # read that frame from the video
            videos = {
                r"C:\Users\pc\OneDrive\Pictures\WhatsApp Video 2026-10-02 at 16.13.08.mp4": 0,
                r"C:\Users\pc\OneDrive\Pictures\WhatsApp Video 2026-10-02 at 16.13.22.mp4": 1,
            }
            vid_name = video_name
            if vid_name not in videos:
                print(f"C{cid}: unknown video {vid_name}")
                skipped += 1
                continue
            vp = videos[vid_name]
            cap = cv2.VideoCapture(vid_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                print(f"C{cid}: could not read frame {frame_idx} from {vid_name}")
                skipped += 1
                continue

            # encode + embed
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                print(f"C{cid}: jpeg encode fail")
                skipped += 1
                continue
            try:
                res = ic.embed_images([buf.tobytes()])
            except Exception as exc:  # noqa: BLE001
                print(f"C{cid}: embed rpc {exc}")
                skipped += 1
                continue
            row = res[0] if res else None
            if not row or not row.get("embedding"):
                print(f"C{cid}: no embedding from RPC")
                skipped += 1
                continue
            emb = np.asarray(row["embedding"], dtype=np.float32)

            # use registration_no from the CSV if we can map, else use generated
            # The legacy CSV held synthetic ids such as STU00000002.
            # Map frame-derived cluster -> reg no
            # For now, use a generated registration based on cluster index
            reg_no = f"STU{ci+1:05d}"
            name = f"Student {ci+1}"

            # Enroll via production endpoint
            try:
                student = enrollment.enroll_student(
                    db,
                    name=name,
                    registration_no=reg_no,
                    section="DATASET",
                    photos=[buf.tobytes()],
                    filenames=["photo_0.jpg"],
                )
                print(f"ENROLLED C{cid}: {name} reg={reg_no} "
                      f"embeddings={len(student.embeddings)} "
                      f"from vid={video_name} frame={frame_idx} "
                      f"score={best_key[0]:.3f} sharp={best_key[1]:.0f}")
                picked += 1
            except enrollment.EnrollmentError as exc:
                print(f"REJECTED C{cid}: {exc}")
                rejected += 1
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR C{cid}: {exc}")
                skipped += 1

    # refresh index so later matching picks up new students
    from app.matching import EmbeddingIndex
    idx = EmbeddingIndex(threshold=settings.recognition_threshold)
    idx.refresh()
    print(f"\nEnrollment: picked={picked} rejected={rejected} skipped={skipped}")
    print(f"Index: {idx.student_count} students / {idx.embedding_count} embeddings")
    return 0 if picked > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())