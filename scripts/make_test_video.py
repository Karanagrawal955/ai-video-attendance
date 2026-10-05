#!/usr/bin/env python
"""Build an MP4 slideshow from enrolled student photos for E2E tests.

- Discover enrolled photos from data/students/*/photo_*.jpg (or site.db).
- 2 seconds per face at 30 FPS, 0.5s cross-fade between faces.
- Includes a side-by-side composite (2 students at once) to test
  simultaneous multi-subject detection.
- Output: tests/assets/test_multi.mp4 via cv2.VideoWriter (mp4v).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sqlite3
import sys
import tempfile

import cv2
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
# Optional "site" database with the photo_paths used by older exports.
# Override with SITE_DB=<path>; never hard-code a machine-specific location.
SITE_DB = pathlib.Path(
    os.environ.get("SITE_DB")
    or pathlib.Path(tempfile.gettempdir()) / "opencode" / "site.db"
)
OUTPUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "assets" / "test_multi.mp4"

# video params
W, H = 640, 640
FPS = 30
SEC_PER_FACE = 2.0
CROSS_FADE_SEC = 0.5
FRAMES_PER_FACE = int(FPS * SEC_PER_FACE)  # 60
BLEND_FRAMES = int(FPS * CROSS_FADE_SEC)   # 15


def discover_photos() -> list[tuple[int, str, pathlib.Path]]:
    """Return list of (student_id, registration_no, photo_path) sorted."""
    photos: list[tuple[int, str, pathlib.Path]] = []
    # Try site.db photo_paths as primary (actual enrolled records)
    if SITE_DB.exists():
        try:
            con = sqlite3.connect(str(SITE_DB))
            cur = con.cursor()
            cur.execute("SELECT id, registration_no, photo_paths FROM students")
            for sid, reg, pp in cur.fetchall():
                import json
                try:
                    paths = json.loads(pp) if isinstance(pp, str) else (pp or [])
                except Exception:
                    paths = []
                for rel in paths:
                    p = DATA_DIR / rel
                    if p.exists():
                        photos.append((sid, reg, p))
                    else:
                        # try absolute
                        ap = pathlib.Path(rel)
                        if ap.exists():
                            photos.append((sid, reg, ap))
            con.close()
        except Exception as e:
            print(f"[warn] site.db read failed: {e}", file=sys.stderr)
    # Fallback: scan data/students/*
    if not photos:
        for p in DATA_DIR.rglob("photo_*.jpg"):
            # infer student id from parent dir name
            try:
                sid = int(p.parent.name)
            except ValueError:
                sid = 0
            reg = f"SID{sid}"
            photos.append((sid, reg, p))
    photos.sort(key=lambda x: (x[0], str(x[2])))
    return photos


def load_and_pad(img_path: pathlib.Path, size=(W, H)) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"cannot read {img_path}")
    # letterbox to size
    h, w = img.shape[:2]
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8) + 16  # dark background
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def make_side_by_side(path_a: pathlib.Path, path_b: pathlib.Path, size=(W, H)) -> np.ndarray:
    a = cv2.imread(str(path_a))
    b = cv2.imread(str(path_b))
    if a is None or b is None:
        raise RuntimeError(f"cannot read side-by-side sources {path_a} {path_b}")
    # Resize each to half width, same height
    th = size[1]
    half_w = size[0] // 2
    def fit_half(img):
        h, w = img.shape[:2]
        scale = min(half_w / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, half_w, 3), dtype=np.uint8) + 16
        y0 = (th - nh) // 2
        x0 = (half_w - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        return canvas
    left = fit_half(a)
    right = fit_half(b)
    return np.hstack([left, right])


def build_video(photos: list[tuple[int, str, pathlib.Path]], output: pathlib.Path, fps: int = FPS):
    # Pick one photo per student id for solo slides, plus one composite
    by_student: dict[int, list[pathlib.Path]] = {}
    for sid, reg, p in photos:
        by_student.setdefault(sid, []).append(p)
    # Order students deterministically
    sids = sorted(by_student.keys())
    if len(sids) < 2:
        print(f"[warn] only {len(sids)} student(s) with photos found — need >=2 for multi-subject test", file=sys.stderr)
    # Use first photo per student for solo
    solo_paths: list[pathlib.Path] = []
    for sid in sids[:3]:  # at most 3 solo slides
        solo_paths.append(by_student[sid][0])
    # Composite: side-by-side of first two students
    composite = None
    if len(sids) >= 2:
        composite = make_side_by_side(by_student[sids[0]][0], by_student[sids[1]][0])

    # Build image sequence (numpy arrays at W x H)
    images: list[np.ndarray] = []
    labels: list[str] = []
    for p in solo_paths:
        images.append(load_and_pad(p))
        labels.append(p.parent.name + "/" + p.name)
    if composite is not None:
        images.append(composite)
        labels.append(f"composite:{sids[0]}+{sids[1]}")

    if not images:
        print("[error] no images to encode", file=sys.stderr)
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, float(fps), (W, H))
    if not writer.isOpened():
        print("[error] VideoWriter could not be opened (mp4v)", file=sys.stderr)
        sys.exit(1)

    total = 0
    # Encode: for each image, write FRAMES_PER_FACE static frames,
    # but between images insert a cross-fade of BLEND_FRAMES where frames
    # are alpha-blended from current to next (without double-counting).
    # Strategy: for i < len-1, write (FRAMES_PER_FACE - BLEND_FRAMES) solid
    # frames then BLEND_FRAMES blended frames. Last image: full FRAMES_PER_FACE.
    for idx, img in enumerate(images):
        is_last = idx == len(images) - 1
        solid = FRAMES_PER_FACE if is_last else (FRAMES_PER_FACE - BLEND_FRAMES)
        for _ in range(solid):
            writer.write(img)
            total += 1
        if not is_last:
            nxt = images[idx + 1]
            for k in range(BLEND_FRAMES):
                alpha = (k + 1) / (BLEND_FRAMES + 1)  # fade progression
                blended = cv2.addWeighted(img, 1 - alpha, nxt, alpha, 0)
                writer.write(blended)
                total += 1
    writer.release()
    size = output.stat().st_size if output.exists() else 0
    dur = total / fps
    print(f"Wrote {output} ({total} frames, {dur:.1f}s @ {fps}fps, {size} bytes)")
    print(f"  source photos used ({len(photos)} total available, {len(images)} in sequence):")
    for lb, p in zip(labels, solo_paths + ([None] if composite is None else [])):
        if p:
            print(f"    - {lb} -> {p}")
    if composite is not None:
        print(f"    - {labels[-1]} (side-by-side)")
    return output


def main():
    ap = argparse.ArgumentParser(description="Build test_multi.mp4 from enrolled photos")
    ap.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()
    photos = discover_photos()
    print(f"Discovered {len(photos)} enrolled photo(s):")
    seen: dict[int, int] = {}
    for sid, reg, p in photos:
        seen[sid] = seen.get(sid, 0) + 1
    for sid, cnt in sorted(seen.items()):
        print(f"  student #{sid}: {cnt} photo(s)")
    if len(photos) == 0:
        print("[error] No enrolled photos found. Enroll students first (POST /students with 3-5 photos)", file=sys.stderr)
        sys.exit(2)
    if len(seen) < 2:
        print(f"[warn] Only {len(seen)} student(s) have photos — multi-subject segment still created where possible", file=sys.stderr)
    build_video(photos, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
