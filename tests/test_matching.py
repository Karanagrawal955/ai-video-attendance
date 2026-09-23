"""Tests for cosine-similarity matching against the embedding index."""

from __future__ import annotations

import numpy as np
import pytest

from app.matching import EMBEDDING_DIM, EmbeddingIndex, MatchResult


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _random_unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(EMBEDDING_DIM).astype(np.float32))


class _FakeStudent:
    def __init__(self, sid: int, embeddings: list[np.ndarray]):
        self.id = sid
        self.embeddings = [e.tolist() for e in embeddings]


@pytest.fixture()
def index(monkeypatch: pytest.MonkeyPatch) -> EmbeddingIndex:
    """Index preloaded without touching the database or redis."""
    students = [
        _FakeStudent(1, [_random_unit(10), _random_unit(11)]),
        _FakeStudent(2, [_random_unit(20)]),
        _FakeStudent(3, [_random_unit(30)]),
    ]

    def fake_refresh(self=None, version=None):  # noqa: ANN001
        vectors = np.vstack(
            [np.vstack(s.embeddings) for s in students]
        ).astype(np.float32)
        ids = np.concatenate(
            [np.full(len(s.embeddings), s.id, dtype=np.int64) for s in students]
        )
        idx = EmbeddingIndex.__new__(EmbeddingIndex)
        idx.threshold = 0.40
        idx.scan_top = 64
        idx.flat = _l2_rows(vectors)
        idx.row_student_ids = ids
        idx.student_count = len(students)
        idx.embedding_count = int(vectors.shape[0])
        idx.loaded_at = 0.0
        idx._version = "test"
        idx._last_check = 0.0
        idx._last_refresh = 0.0
        return idx

    built = fake_refresh()
    # Copy state onto a real instance so methods resolve normally.
    real = EmbeddingIndex(threshold=0.40)
    real.flat = built.flat
    real.row_student_ids = built.row_student_ids
    real.student_count = built.student_count
    real.embedding_count = built.embedding_count
    real.loaded_at = built.loaded_at
    real._version = built._version
    real._last_check = float("inf")  # never auto-refresh during tests
    real._last_refresh = built._last_refresh
    return real


def _l2_rows(mat: np.ndarray) -> np.ndarray:
    return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)


def test_exact_reference_match(index: EmbeddingIndex) -> None:
    probe = _random_unit(10)
    result = index.match(probe)
    assert result.student_id == 1
    assert result.score > 0.999


def test_best_match_across_multiple_references(index: EmbeddingIndex) -> None:
    probe = _random_unit(11)  # second reference photo of student 1
    result = index.match(probe)
    assert result.student_id == 1
    assert result.score > 0.999


def test_unknown_below_threshold(index: EmbeddingIndex) -> None:
    probe = _random_unit(999)  # never seen
    result = index.match(probe)
    assert result.student_id is None
    assert result.score < 0.40


def test_empty_index(index: EmbeddingIndex) -> None:
    index.flat = None
    index.row_student_ids = None
    result = index.match(_random_unit(1))
    assert result == MatchResult(student_id=None, score=-1.0)


def test_nan_embedding_rejected(index: EmbeddingIndex) -> None:
    bad = np.full(EMBEDDING_DIM, np.nan, dtype=np.float32)
    result = index.match(bad)
    assert result.student_id is None


def test_wrong_dimension_rejected(index: EmbeddingIndex) -> None:
    result = index.match(np.ones(256, dtype=np.float32))
    assert result.student_id is None
