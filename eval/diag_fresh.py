#!/usr/bin/env python
"""STEP 1 continued: fresh-vs-fresh cosines (is the engine sane?) and
fresh-vs-DB-gallery (is the stored gallery sane?)."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///demo_real2.db"
os.environ["FORCE_CPU"] = "true"
os.environ["FACE_QUALITY_ENABLED"] = "true"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.inference.engine import FaceEngine  # noqa: E402
from app.models import Student  # noqa: E402

PHOTO_ROOT = ROOT / "data" / "students"


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    eng = FaceEngine(settings)
    eng.warmup()

    db = SessionLocal()
    students = db.query(Student).order_by(Student.id).all()

    fresh: dict[int, list[np.ndarray]] = {}
    for s in students:
        raw = s.photo_paths or "[]"
        rels = raw if isinstance(raw, list) else ast.literal_eval(raw)
        imgs = [cv2.imread(str(PHOTO_ROOT / r)) for r in rels]
        imgs = [i for i in imgs if i is not None]
        res, _ = eng.infer(imgs)
        fresh[s.id] = [np.asarray(f[0]["embedding"], float) for f in res if f]
        print(f"P{s.id}: fresh embeds={len(fresh[s.id])}")

    # determinism: same image twice
    img = cv2.imread(str(PHOTO_ROOT / "1/photo_0.jpg"))
    r1, _ = eng.infer([img])
    r2, _ = eng.infer([img])
    e1 = np.asarray(r1[0][0]["embedding"], float)
    e2 = np.asarray(r2[0][0]["embedding"], float)
    print(f"\ndeterminism: cosine(run1, run2) on identical image = {cos(e1, e2):.6f}")

    # fresh-vs-fresh
    same, diff = [], []
    for sid, vecs in fresh.items():
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                same.append(cos(vecs[i], vecs[j]))
            for oid, ovecs in fresh.items():
                if oid == sid:
                    continue
                for ov in ovecs:
                    diff.append(cos(vecs[i], ov))
    same, diff = np.array(same), np.array(diff)
    print(f"\nFRESH-vs-FRESH (current engine, this machine):")
    print(f"  SAME person (2 different photos) n={len(same):3d} "
          f"min={same.min():.4f} mean={same.mean():.4f} max={same.max():.4f}")
    print(f"  DIFFERENT person               n={len(diff):3d} "
          f"min={diff.min():.4f} mean={diff.mean():.4f} "
          f"p95={np.percentile(diff, 95):.4f} max={diff.max():.4f}")

    # fresh-vs-DB gallery (decrypt stored)
    gsame, gdiff = [], []
    for sid, vecs in fresh.items():
        sv = next(s for s in students if s.id == sid)
        gal = [np.asarray(v, float) for v in sv.embedding_list]
        for fv in vecs:
            for gv in gal:
                gsame.append(cos(gv, fv))
            for o in students:
                if o.id == sid:
                    continue
                for gv in (np.asarray(v, float) for v in o.embedding_list):
                    gdiff.append(cos(gv, fv))
    gsame, gdiff = np.array(gsame), np.array(gdiff)
    print(f"\nDB-GALLERY-vs-FRESH (stored encrypted vs re-embedded):")
    print(f"  SAME person n={len(gsame):3d} min={gsame.min():.4f} "
          f"mean={gsame.mean():.4f} max={gsame.max():.4f}")
    print(f"  DIFFERENT  n={len(gdiff):3d} mean={gdiff.mean():.4f} "
          f"max={gdiff.max():.4f}")

    if same.min() > diff.max():
        print(f"\nFRESH path is separable: threshold in ({diff.max():.4f}, {same.min():.4f}) "
              f"midpoint={(same.min() + diff.max()) / 2:.4f}")
    else:
        print("\nFRESH path NOT separable")


if __name__ == "__main__":
    main()
