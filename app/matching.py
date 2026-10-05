"""Cosine-similarity face matching against enrolled students.

The index is a single float32 matrix of all stored reference embeddings
(L2-normalised at build time), so one probe is a single matvec.  A student
matches when ANY of their reference embeddings clears the threshold
(best-match over multiple photos = robustness to pose/lighting).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select

from .config import settings
from .db import SessionLocal
from .models import Student
from .crypto import decrypt_embeddings
from . import redis_client

logger = logging.getLogger("app.matching")

EMBEDDING_DIM = 512  # InsightFace buffalo_l / w600k_r50 output dimension


@dataclass
class MatchResult:
    student_id: int | None  # None => no student above threshold
    score: float  # best cosine similarity found (even if below threshold)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-9)


class EmbeddingIndex:
    """In-memory student embedding index with version-based cache invalidation."""

    def __init__(self, threshold: float | None = None, scan_top: int = 64,
                 margin: float | None = None):
        self.threshold = (
            threshold if threshold is not None else settings.recognition_threshold
        )
        self.margin = (
            margin if margin is not None else settings.recognition_margin
        )
        self.duplicate_sim = settings.duplicate_identity_sim
        self.scan_top = scan_top
        self.flat: np.ndarray | None = None  # (N, 512) float32, L2-normalised
        self.row_student_ids: np.ndarray | None = None  # (N,) int64
        self.student_count = 0
        self.embedding_count = 0
        self.loaded_at: float = 0.0
        self._version: str | None = None
        self._last_check: float = 0.0
        self._last_refresh: float = 0.0

    # ------------------------------------------------------------- building
    def refresh(self, version: str | None = None) -> None:
        """(Re)load every student embedding from the database."""
        rows: list[tuple[int, str]] = []
        with SessionLocal() as db:
            result = db.execute(select(Student.id, Student.embeddings))
            rows = [(int(r[0]), r[1] or "") for r in result]

        vectors: list[np.ndarray] = []
        sids: list[int] = []
        for sid, encrypted_embs in rows:
            if not encrypted_embs:
                continue
            try:
                embeddings = decrypt_embeddings(encrypted_embs)
            except Exception:
                continue
            for emb in embeddings:
                try:
                    arr = np.asarray(emb, dtype=np.float32).ravel()
                except (TypeError, ValueError):
                    continue
                if arr.size != EMBEDDING_DIM or not np.all(np.isfinite(arr)):
                    continue
                vectors.append(arr)
                sids.append(sid)

        if vectors:
            mat = _l2_normalize(np.vstack(vectors))
            self.flat = mat.astype(np.float32, copy=False)
            self.row_student_ids = np.asarray(sids, dtype=np.int64)
        else:
            self.flat = None
            self.row_student_ids = None

        self.student_count = len(rows)
        self.embedding_count = len(vectors)
        self.loaded_at = time.time()
        self._last_refresh = self.loaded_at
        if version is not None:
            self._version = version
        logger.info(
            "embedding index refreshed",
            extra={
                "students": self.student_count,
                "embeddings": self.embedding_count,
                "threshold": self.threshold,
            },
        )

    def maybe_refresh(self, check_interval: float = 5.0) -> None:
        """Cheap invalidation check (Redis version counter, periodic rebuild)."""
        now = time.time()
        if now - self._last_check < check_interval:
            return
        self._last_check = now
        version = redis_client.students_version()
        stale_periodic = (now - self._last_refresh) > 60.0
        if version is None:
            # Redis unavailable: fall back to a periodic full refresh.
            if stale_periodic:
                self.refresh()
            return
        if version != self._version or self.flat is None:
            self.refresh(version=version)

    # ------------------------------------------------------------- matching
    def match(self, embedding: np.ndarray | list[float]) -> MatchResult:
        """Best-match a single 512-d probe against every enrolled student.

        Acceptance rule (single threshold everywhere):
          1. best per-student score >= settings.recognition_threshold
          2. best - runner-up >= settings.recognition_margin, where the
             runner-up skips students whose gallery is a near-duplicate of the
             winner (same human enrolled twice -> would otherwise always
             produce a ~0 margin and reject genuine matches).
        """
        if self.flat is None or self.row_student_ids is None:
            return MatchResult(student_id=None, score=-1.0)
        probe = np.asarray(embedding, dtype=np.float32).ravel()
        if probe.size != EMBEDDING_DIM or not np.all(np.isfinite(probe)):
            return MatchResult(student_id=None, score=-1.0)
        probe = probe / max(float(np.linalg.norm(probe)), 1e-9)

        sims = self.flat @ probe  # (N,)
        if sims.size == 0:
            return MatchResult(student_id=None, score=-1.0)

        # per-student best score (any of their reference photos)
        sids = self.row_student_ids
        uniq, inverse = np.unique(sids, return_inverse=True)
        best_scores = np.full(uniq.size, -1.0, dtype=np.float64)
        np.maximum.at(best_scores, inverse, sims)

        b = int(np.argmax(best_scores))
        best_score = float(best_scores[b])
        if best_score < self.threshold:
            return MatchResult(student_id=None, score=best_score)

        best_sid = int(uniq[b])
        best_rows = self.flat[sids == best_sid]

        # runner-up over identities that are NOT duplicates of the winner
        second: float | None = None
        for k in range(uniq.size):
            if k == b:
                continue
            rows = self.flat[sids == uniq[k]]
            if best_rows.shape[0] and rows.shape[0]:
                dup = float((best_rows @ rows.T).max())
                if dup >= self.duplicate_sim:
                    continue  # same human, two student rows
            if second is None or best_scores[k] > second:
                second = float(best_scores[k])

        if second is not None and (best_score - second) < self.margin:
            logger.debug(
                "margin rule rejected match",
                extra={
                    "best": round(best_score, 4),
                    "second": round(second, 4),
                    "margin": self.margin,
                },
            )
            return MatchResult(student_id=None, score=best_score)
        return MatchResult(student_id=best_sid, score=best_score)


def cosine_similarity_batch(embeddings: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between all embeddings and probes.
    embeddings: (N, 512) - already L2-normalized
    probes: (M, 512) - already L2-normalized
    Returns: (M, N) similarity matrix
    """
    return probes @ embeddings.T  # (M, N)
