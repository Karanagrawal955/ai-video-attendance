#!/usr/bin/env python
"""Measured resolution table: detect + embed faces on frames downscaled to
several resolutions, printing face width in pixels and detector confidence.
All numbers come from real FaceEngine runs (no scaling laws, no estimates).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["FORCE_CPU"] = "true"
os.environ["FACE_QUALITY_ENABLED"] = "true"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.inference.engine import FaceEngine  # noqa: E402

VIDEO = (
    sys.argv[1]
    if len(sys.argv) > 1
    else r"C:\Users\pc\OneDrive\Pictures\WhatsApp Video 2026-10-02 at 16.13.08.mp4"
)
SCALES = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15]
SAMPLE = 8


def main() -> None:
    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened(), VIDEO
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // SAMPLE)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()
    frames = frames[:SAMPLE]

    eng = FaceEngine(settings)
    eng.warmup()
    print(f"video={Path(VIDEO).name} frames_sampled={len(frames)} "
          f"native={frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"{'scale':>6} {'out_wxh':>12} {'faces':>6} {'width_px(min/med/max)':>26} "
          f"{'det(min)':>9} {'emb_ms':>8}")
    for sc in SCALES:
        imgs = []
        for f in frames:
            if sc == 1.0:
                imgs.append(f)
            else:
                imgs.append(cv2.resize(f, None, fx=sc, fy=sc,
                                       interpolation=cv2.INTER_AREA))
        res, tim = eng.infer(imgs)
        widths, dets, norms = [], [], []
        for faces in res:
            for fc in faces:
                x1, y1, x2, y2 = fc["bbox"]
                widths.append(x2 - x1)
                dets.append(fc["score"])
                norms.append(float(np.linalg.norm(fc["embedding"])))
        if widths:
            w = np.array(widths)
            d = np.array(dets)
            print(f"{sc:>6.2f} {imgs[0].shape[1]:>5}x{imgs[0].shape[0]:<6} "
                  f"{len(widths):>6} {w.min():>8.1f}/{np.median(w):>7.1f}/{w.max():>6.1f} "
                  f"{d.min():>9.3f} {tim.emb_ms:>8.1f}")
        else:
            print(f"{sc:>6.2f} {imgs[0].shape[1]:>5}x{imgs[0].shape[0]:<6} "
                  f"{0:>6} {'-':>26} {'-':>9} {tim.emb_ms:>8.1f}")


if __name__ == "__main__":
    main()
