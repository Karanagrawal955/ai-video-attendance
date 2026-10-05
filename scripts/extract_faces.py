#!/usr/bin/env python
"""Dense face extraction from the dataset videos (stage 1 of evaluation).

Samples each video at N fps, runs every sampled frame through the PRODUCTION
GPU inference service in batches, applies the quality gate, and persists all
detections (bbox, det score, embedding, sharpness, brightness, face size) to a
``.npz`` + sidecar JSON for offline clustering/evaluation.

    python scripts/extract_faces.py --fps 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.inference import client as ic  # noqa: E402

DEFAULT_VIDEOS = [
    Path(r"C:\Users\pc\OneDrive\Pictures\WhatsApp Video 2026-10-02 at 16.13.08.mp4"),
    Path(r"C:\Users\pc\OneDrive\Pictures\WhatsApp Video 2026-10-02 at 16.13.22.mp4"),
]
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "dataset_index"
BATCH = 8


def face_quality(crop: np.ndarray, bbox) -> dict:
    """Mirror of the engine's quality gate + extra metrics, measured on crop."""
    import cv2 as _cv

    gray = _cv.cvtColor(crop, _cv.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    sharp = float(_cv.Laplacian(gray, _cv.CV_64F).var())
    bright = float(gray.mean())
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return {"sharpness": sharp, "brightness": bright, "area": area}


def extract(video: Path, fps_target: float) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps / fps_target))
    print(f"\n== {video.name}: {n} frames @ {fps:.2f}fps, sampling every {step} "
          f"(->{fps / step:.2f} fps)")

    frames: list[tuple[int, np.ndarray]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    print(f"  sampled {len(frames)} frames")

    all_emb: list[np.ndarray] = []
    all_meta: list[dict] = []
    t_start = time.perf_counter()
    for i in range(0, len(frames), BATCH):
        chunk = frames[i:i + BATCH]
        jpegs = []
        for _, fr in chunk:
            ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            jpegs.append(buf.tobytes() if ok else b"")
        try:
            res = ic.infer_frames(jpegs)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! batch {i}: {exc}")
            continue
        results = res.get("results") or []
        for (fidx, fr), faces in zip(chunk, results):
            for j, f in enumerate(faces or []):
                emb = f.get("embedding")
                bbox = f.get("bbox")
                score = float(f.get("score") or 0.0)
                if not emb or not bbox:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                h, w = fr.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                    continue
                crop = fr[cy1:cy2, cx1:cx2]
                q = face_quality(crop, bbox)
                all_emb.append(np.asarray(emb, dtype=np.float32))
                all_meta.append({
                    "video": video.name, "frame": fidx, "face": j,
                    "bbox": [x1, y1, x2, y2], "score": score, **q,
                })
        if (i // BATCH) % 10 == 0:
            print(f"  ... {min(i + BATCH, len(frames))}/{len(frames)} frames, "
                  f"{len(all_meta)} faces, {time.perf_counter() - t_start:.0f}s")
    elapsed = time.perf_counter() - t_start
    print(f"  {len(all_meta)} faces from {len(frames)} frames in {elapsed:.1f}s "
          f"({1000 * elapsed / max(len(frames), 1):.0f} ms/frame)")
    return {"emb": np.vstack(all_emb) if all_emb else np.empty((0, 512), np.float32),
            "meta": all_meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--videos", nargs="*", type=Path, default=None)
    args = ap.parse_args()
    videos = args.videos or DEFAULT_VIDEOS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_emb, merged_meta = [], []
    for v in videos:
        if not Path(v).exists():
            print(f"SKIP missing {v}")
            continue
        out = extract(Path(v), args.fps)
        if out["emb"].size:
            merged_emb.append(out["emb"])
            merged_meta.extend(out["meta"])
    if not merged_emb:
        print("no faces extracted")
        return 1
    emb = np.vstack(merged_emb)
    np.save(OUT_DIR / "embeddings.npy", emb)
    (OUT_DIR / "meta.json").write_text(json.dumps(merged_meta, indent=1))
    # quality summary
    scores = np.array([m["score"] for m in merged_meta])
    sharp = np.array([m["sharpness"] for m in merged_meta])
    area = np.array([m["area"] for m in merged_meta])
    print(f"\n== totals: {len(merged_meta)} face dets, emb {emb.shape} ==")
    print(f"  det score : p50={np.percentile(scores, 50):.3f} "
          f"p10={np.percentile(scores, 10):.3f} min={scores.min():.3f}")
    print(f"  sharpness : p50={np.percentile(sharp, 50):.1f} "
          f"p10={np.percentile(sharp, 10):.1f}")
    print(f"  bbox area : p50={np.percentile(area, 50):.0f} p10={np.percentile(area, 10):.0f}")
    rej = {
        "det_score<0.5": int((scores < 0.5).sum()),
        "sharpness<30": int((sharp < settings.face_min_sharpness).sum()),
        "area<min": int((area < settings.face_min_area).sum()),
    }
    print(f"  quality gate rejects: {rej}")
    print(f"  written -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
