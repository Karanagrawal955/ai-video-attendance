"""FaceEngine micro-benchmark: GPU (CUDA) vs CPU numbers on YOUR machine.

    python scripts/benchmark.py                      # configured providers (CUDA if avail)
    python scripts/benchmark.py --cpu                # force CPU-only baseline
    python scripts/benchmark.py --frames 8 --iters 30
    python scripts/benchmark.py --image my_face.jpg  # use your own image

The benchmark loads buffalo_l, warms up, then measures:
  * end-to-end ``engine.infer`` latency for a 1-frame batch (per-stream cost)
  * end-to-end latency for an N-frame batch (cross-camera batching)
  * embedding-only throughput for varying face-batch sizes

Typical reference points (see README "Benchmarks" for measured numbers):
  GPU (CUDA, stock buffalo_l): ~15 ms/frame detection (static-batch det
      graph runs serial per frame), ~30 ms end-to-end, ~250-380 faces/s
      embedding when batched
  CPU-only (i7 laptop):        ~600 ms/frame detection, ~2+ s end-to-end
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def load_image(path: str | None) -> np.ndarray:
    import cv2

    if path:
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"could not read image: {path}")
        return img
    try:  # scikit-image ships with insightface deps and has a real face
        from skimage import data

        rgb = data.astronaut()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        print("WARNING: skimage unavailable, using random noise "
              "(no faces will be detected)", file=sys.stderr)
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def summarize(label: str, totals: list[float], det: list[float],
              emb: list[float], faces: int, frames_total: int,
              frames_per_call: int) -> None:
    if not totals:
        return
    # Throughput = frames one call processes / mean per-call time, i.e. what
    # you get running this workload back-to-back.  (frames_total /
    # mean(totals) would inflate by the iteration count.)
    mean_total = statistics.mean(totals)
    fps = frames_per_call / (mean_total / 1000.0)
    print(f"\n== {label} ==")
    print(f"  frames per call  : {frames_per_call} "
          f"({len(totals)} calls, {frames_total} frames total)")
    print(f"  faces detected   : {faces} "
          f"({faces / max(frames_total, 1):.1f}/frame)")
    print(f"  total  ms        : mean={statistics.mean(totals):7.2f}  "
          f"p50={pct(totals, 0.5):7.2f}  p95={pct(totals, 0.95):7.2f}")
    print(f"  detect ms        : mean={statistics.mean(det):7.2f}  "
          f"p50={pct(det, 0.5):7.2f}  p95={pct(det, 0.95):7.2f}")
    print(f"  embed  ms        : mean={statistics.mean(emb):7.2f}  "
          f"p50={pct(emb, 0.5):7.2f}  p95={pct(emb, 0.95):7.2f}")
    print(f"  throughput       : {fps:7.1f} frames/s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cpu", action="store_true",
                        help="force CPUExecutionProvider (baseline numbers)")
    parser.add_argument("--frames", type=int, default=8,
                        help="frames per batch (default 8)")
    parser.add_argument("--iters", type=int, default=20,
                        help="timed iterations (default 20)")
    parser.add_argument("--image", default=None,
                        help="path to an image with a face")
    parser.add_argument("--embed-batches", default="1,8,32,64",
                        help="face-batch sizes for the embedding microbench")
    args = parser.parse_args()

    from app.config import Settings
    from app.inference.engine import FaceEngine

    cfg = Settings(force_cpu=args.cpu) if args.cpu else Settings()
    print(f"loading model {cfg.face_model_name!r} "
          f"({'CPU forced' if args.cpu else 'auto provider'}) ...")
    t0 = time.perf_counter()
    engine = FaceEngine(cfg)
    print(f"loaded in {time.perf_counter() - t0:.1f}s | {engine.status()}")

    base = load_image(args.image)
    frames = []
    for i in range(args.frames):
        # vary frames slightly so the batch isn't a single cached shape
        shifted = np.roll(base, i * 7, axis=1)
        frames.append(np.ascontiguousarray(shifted))

    # warmup (JIT / cuDNN autotune / allocator)
    for _ in range(3):
        engine.infer([frames[0]])
        engine.infer(frames)
    print("warmup done\n")

    # ---- single-frame latency (what one camera pays per RPC)
    totals, dets, embs = [], [], []
    face_count = 0
    for _ in range(args.iters):
        _, t = engine.infer([frames[0]])
        totals.append(t.total_ms)
        dets.append(t.det_ms)
        embs.append(t.emb_ms)
        face_count += t.faces
    summarize(f"single frame (per-stream latency)", totals, dets, embs,
              face_count, args.iters, 1)

    # ---- N-frame batch (cross-camera GPU batching)
    totals, dets, embs = [], [], []
    face_count = 0
    for _ in range(args.iters):
        _, t = engine.infer(frames)
        totals.append(t.total_ms)
        dets.append(t.det_ms)
        embs.append(t.emb_ms)
        face_count += t.faces
    summarize(f"batch of {args.frames} frames", totals, dets, embs,
              face_count, args.iters * args.frames, args.frames)

    # ---- embedding-only throughput across face-batch sizes
    _, t = engine.infer([frames[0]])
    if t.faces == 0:
        print("\n(no faces detected in the sample image - "
              "pass --image with a clear face for meaningful numbers)")
        return 0

    print("\n== embedding microbench (get_feat, CPU align excluded) ==")
    print("  faces/batch |   mean ms |   faces/s")
    for size in [int(x) for x in args.embed_batches.split(",") if x.strip()]:
        samples = []
        for _ in range(10):
            crops = [frames[0][0:112, 0:112] for _ in range(size)]
            start = time.perf_counter()
            engine._embed_faces(crops)
            samples.append((time.perf_counter() - start) * 1000.0)
        mean = statistics.mean(samples)
        print(f"  {size:11d} | {mean:9.2f} | {size / (mean / 1000.0):9.1f}")

    status = engine.status()
    print(f"\nprovider={status['provider']} "
          f"batched_detection={status['batched_detection']} "
          f"threshold={status['recognition_threshold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
