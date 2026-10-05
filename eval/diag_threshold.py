#!/usr/bin/env python
"""STEP 1a-e: prove where the 0.152-vs-0.45 score regression comes from.

Runs ONE FaceEngine instance for every image source (enrollment photos AND
video frames), prints 512-d norms + cosine similarities, and verifies that
preprocessing (BGR input, alignment, model, L2 normalisation) is identical
across paths.  Privacy: only P1..P6 and last-3 registration digits are printed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///demo_real.db"
os.environ["FORCE_CPU"] = "true"
os.environ["FACE_QUALITY_ENABLED"] = "true"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.crypto import decrypt_embeddings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.inference.engine import FaceEngine  # noqa: E402
from app.models import Student  # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\pc\OneDrive\Pictures\WhatsApp Video 2026-10-02 at 16.13.08.mp4"
PHOTO_ROOT = ROOT / "data" / "students"
MAX_FRAMES = 60


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    print("=" * 72)
    print("STEP 1a-e  |  one FaceEngine, two image sources, same preprocessing")
    print("=" * 72)

    eng = FaceEngine(settings)
    eng.warmup()
    st = eng.status()
    print("\n[engine]")
    for k in ("model", "providers", "provider", "device", "force_cpu",
              "batched_detection", "service"):
        if k in st:
            print(f"  {k}: {st[k]}")
    print(f"  input color order: BGR (cv2.imread / cv2.VideoCapture)")
    print(f"  quality gate (face_quality_enabled): {settings.face_quality_enabled}")
    print(f"  embedding dim: {st.get('embedding_dim', 512)}  L2-normalised in _embed_faces: True")

    # ---------------------------------------------------------- gallery (DB)
    db = SessionLocal()
    students = db.query(Student).order_by(Student.id).all()
    print(f"\n[gallery] {len(students)} students loaded from {os.environ['DATABASE_URL']}")
    gallery: dict[int, list[np.ndarray]] = {}
    for i, s in enumerate(students, start=1):
        vecs = [np.asarray(v, dtype=np.float64) for v in s.embedding_list]
        gallery[s.id] = vecs
        print(f"  P{i} (...{s.registration_no[-3:]}) photos={len(vecs)} "
              f"norms={[round(float(np.linalg.norm(v)), 4) for v in vecs]}")

    # ------------------------------------- STEP 1b: enrollment photo re-embed
    print("\n[STEP 1b] enrollment photo through the SAME FaceEngine.infer")
    photo_vecs: dict[int, list[np.ndarray]] = {}
    for s in students:
        imgs = []
        import ast
        raw = s.photo_paths or "[]"
        rels = raw if isinstance(raw, list) else ast.literal_eval(raw)
        for rel in rels:
            img = cv2.imread(str(PHOTO_ROOT / rel))
            if img is not None:
                imgs.append(img)
        res, tim = eng.infer(imgs)
        vecs = []
        for faces in res:
            if faces:
                vecs.append(np.asarray(faces[0]["embedding"], dtype=np.float64))
        photo_vecs[s.id] = vecs
    s1 = students[0]
    g0 = gallery[s1.id][0]
    p0 = photo_vecs[s1.id][0]
    print(f"  P1 gallery vector : dim={g0.shape[0]} norm={np.linalg.norm(g0):.4f} "
          f"max|v|={np.abs(g0).max():.3f}")
    print(f"  P1 photo embedding: dim={p0.shape[0]} norm={np.linalg.norm(p0):.4f} "
          f"max|v|={np.abs(p0).max():.3f}")
    print(f"  cosine(gallery, freshly-embedded same photo) = {cos(g0, p0):.4f}  "
          f"(expect ~1.000: identical pixels -> identical vector)")

    # ------------------------------------------- same / different person stats
    same, diff = [], []
    for sid, pvecs in photo_vecs.items():
        for pv in pvecs:
            for gv in gallery[sid]:
                same.append(cos(gv, pv))
            for oid, ogv in gallery.items():
                if oid == sid:
                    continue
                for gv in ogv:
                    diff.append(cos(gv, pv))
    same_a, diff_a = np.array(same), np.array(diff)
    print("\n[calibration] gallery vs photo cosine")
    print(f"  SAME person      n={len(same_a):4d} min={same_a.min():.4f} "
          f"p05={np.percentile(same_a, 5):.4f} mean={same_a.mean():.4f} "
          f"max={same_a.max():.4f}")
    print(f"  DIFFERENT person n={len(diff_a):4d} min={diff_a.min():.4f} "
          f"mean={diff_a.mean():.4f} p95={np.percentile(diff_a, 95):.4f} "
          f"max={diff_a.max():.4f}")

    # ------------------------------------------------- video faces vs gallery
    print("\n[video] sampling frames, detecting + embedding with the same engine")
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("  ERROR: cannot open video")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // MAX_FRAMES) if total else 10
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
        if len(frames) >= MAX_FRAMES:
            break
    cap.release()
    print(f"  video frames total={total} sampled={len(frames)} step={step}")

    res, tim = eng.infer(frames)
    best_per_face, margins = [], []
    top_faces = []
    for fi, faces in enumerate(res):
        for f in faces:
            fv = np.asarray(f["embedding"], dtype=np.float64)
            scores = {sid: max(cos(fv, gv) for gv in gvecs)
                      for sid, gvecs in gallery.items()}
            ordered = sorted(scores.values(), reverse=True)
            b, sc = ordered[0], ordered[1] if len(ordered) > 1 else 0.0
            best_per_face.append(b)
            margins.append(b - sc)
            top_faces.append((b, f["score"], fi))
    if best_per_face:
        ba = np.array(best_per_face)
        ma = np.array(margins)
        top_faces.sort(reverse=True)
        print(f"  faces={len(ba)} det_score_min={min(t[1] for t in top_faces):.3f}")
        print("  best-gallery cosine per video face:")
        print(f"    min={ba.min():.4f} p25={np.percentile(ba, 25):.4f} "
              f"median={np.median(ba):.4f} p75={np.percentile(ba, 75):.4f} "
              f"max={ba.max():.4f}")
        print(f"    top-10: {[round(float(x), 4) for x in sorted(ba)[-10:]]}")
        print(f"  best-vs-second margin: median={np.median(ma):.4f} "
              f"p90={np.percentile(ma, 90):.4f} max={ma.max():.4f}")
        n_match = int((ba >= settings.recognition_threshold).sum())
        print(f"  faces scoring >= recognition_threshold "
              f"({settings.recognition_threshold}): {n_match} of {len(ba)}  "
              f"=> matches accepted: {n_match}")

    # -------------------------------------------------- STEP 1f suggestion
    print("\n[STEP 1f] single-threshold suggestion")
    lo, hi = float(diff_a.max()), float(same_a.min())
    print(f"  different-person max = {lo:.4f}")
    print(f"  same-person      min = {hi:.4f}")
    if hi > lo:
        mid = (lo + hi) / 2
        print(f"  separable: threshold in ({lo:.4f}, {hi:.4f}); midpoint={mid:.4f}")
    else:
        print("  NOT separable on photo-vs-photo; check video matches instead")
    print("=" * 72)


if __name__ == "__main__":
    main()
