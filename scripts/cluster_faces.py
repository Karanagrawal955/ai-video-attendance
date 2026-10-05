#!/usr/bin/env python
"""Cluster extracted face embeddings into identities (stage 2).

Agglomerative clustering on cosine distance; prints cluster sizes, quality and
per-video distribution so ground truth can be built from real evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "dataset_index"


def main() -> int:
    emb = np.load(OUT / "embeddings.npy").astype(np.float32)
    meta = json.loads((OUT / "meta.json").read_text())
    assert len(meta) == emb.shape[0]

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / np.maximum(norms, 1e-9)
    sim = emb_n @ emb_n.T
    dist = np.clip(1.0 - sim, 0, 1)

    # ---- agglomerative (average linkage) with a threshold sweep
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    print("threshold sweep (cosine distance cut):")
    best = None
    for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        lab = fcluster(Z, t=t, criterion="distance")
        sizes = np.bincount(lab)[1:]
        big = int((sizes >= 3).sum())
        print(f"  t={t:.2f}  clusters={lab.max():3d}  >=3 members: {big:3d}  "
              f"sizes(top8)={sorted(sizes, reverse=True)[:8]}")
        if best is None and 4 <= lab.max() <= 12:
            best = (t, lab)
    if best is None:
        best = (0.45, fcluster(Z, t=0.45, criterion="distance"))
    t, lab = best
    print(f"\nusing t={t:.2f} -> {lab.max()} clusters")

    # ---- per cluster report
    print("\ncluster report:")
    rows = []
    for c in range(1, lab.max() + 1):
        idx = np.where(lab == c)[0]
        if idx.size == 0:
            continue
        vids: dict[str, int] = {}
        for i in idx:
            vids[meta[i]["video"][-10:-4]] = vids.get(meta[i]["video"][-10:-4], 0) + 1
        scores = [meta[i]["score"] for i in idx]
        sharps = [meta[i]["sharpness"] for i in idx]
        frames = sorted({(meta[i]["video"][-10:-4], meta[i]["frame"]) for i in idx})
        rows.append((c, idx.size, float(np.mean(scores)), float(np.mean(sharps)), vids, frames))
    rows.sort(key=lambda r: -r[1])
    for c, n, sc, sh, vids, frames in rows:
        vshort = ",".join(f"{k}:{v}" for k, v in sorted(vids.items()))
        print(f"  C{c:<3} n={n:<4} det={sc:.3f} sharp={sh:6.1f}  videos=[{vshort}]  "
              f"frames={len(frames)}")

    np.save(OUT / "clusters.npy", lab)
    print(f"\nwritten {OUT / 'clusters.npy'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
