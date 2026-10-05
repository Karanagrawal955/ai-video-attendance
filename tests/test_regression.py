"""STEP 1g — regression guard for the matching stack.

Covers the exact chain that broke in the 0/6 demo run:
  plaintext vector -> encrypt -> store in DB -> decrypt -> index build ->
  cosine search -> identity.

Any future change to crypto, normalisation, the threshold, or the metric that
makes enrolment and video scoring disagree will fail one of these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.crypto import decrypt_embeddings, encrypt_embeddings
from app.matching import EMBEDDING_DIM, EmbeddingIndex
from app.models import Student


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _add_student(db, sid: int, vecs: list[np.ndarray]) -> Student:
    s = Student(
        name=f"Regression-{sid}",
        registration_no=f"REG{sid:04d}",
        section="CSE-A",
        embeddings=encrypt_embeddings([v.tolist() for v in vecs]),
        photo_paths=[],
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# ---------------------------------------------------------------- crypto chain
def test_encrypt_store_decrypt_roundtrip_is_lossless(db) -> None:
    v = _unit(1)
    blob = encrypt_embeddings([v.tolist()])
    # stored value must NOT be the plaintext vector
    assert "0." not in blob and blob != str(v.tolist())
    back = np.asarray(decrypt_embeddings(blob)[0], dtype=np.float64)
    assert back.shape == (EMBEDDING_DIM,)
    assert np.max(np.abs(back - v.astype(np.float64))) < 1e-6
    # cosine of a vector with itself after the full roundtrip is exactly 1
    assert float(np.dot(back, back) / (np.linalg.norm(back) ** 2)) == pytest.approx(1.0)


def test_stored_column_is_ciphertext_not_json(db) -> None:
    v = _unit(2)
    s = _add_student(db, 9001, [v])
    assert isinstance(s.embeddings, str)
    assert not s.embeddings.lstrip().startswith("[")
    assert np.allclose(np.asarray(s.embedding_list[0]), v, atol=1e-6)


# ------------------------------------------------------------------ index path
@pytest.fixture()
def populated(db) -> dict[int, np.ndarray]:
    vectors = {i: _unit(100 + i) for i in (1, 2, 3)}
    for sid, vec in vectors.items():
        _add_student(db, sid, [vec, _unit(200 + sid)])  # 2 refs each
    return vectors


def test_index_finds_exact_identity_after_full_chain(populated, db) -> None:
    idx = EmbeddingIndex()
    idx.refresh()  # reads ONLY from the DB: decrypt -> normalise -> build
    assert idx.student_count >= 3
    rows = {s.registration_no: s.id for s in db.query(Student).all()}
    for sid, vec in populated.items():
        res = idx.match(vec)
        expected = rows[f"REG{sid:04d}"]
        assert res.student_id == expected, f"probe for student {sid} -> {res}"
        assert res.score > 0.999


def test_probe_scale_does_not_change_identity_or_score(populated, db) -> None:
    idx = EmbeddingIndex()
    idx.refresh()
    vec = populated[1]
    base = idx.match(vec)
    loud = idx.match(vec * 25.0)  # unnormalised probe
    assert loud.student_id == base.student_id
    assert loud.score == pytest.approx(base.score, abs=1e-5)


def test_unknown_probe_rejected_by_single_threshold(populated, db) -> None:
    idx = EmbeddingIndex()
    idx.refresh()
    assert idx.threshold == settings.recognition_threshold  # one threshold, not two
    res = idx.match(_unit(9999))
    assert res.student_id is None
    assert res.score < settings.recognition_threshold


def test_margin_rule_rejects_ambiguous_probe(db) -> None:
    """A probe sitting exactly between two enrolled identities must NOT match,
    even though both clear the raw threshold (best-vs-second margin rule)."""
    a, b = _unit(7), _unit(8)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    b = b - np.dot(b, a) * a  # orthogonalise so both scores are ~0.707
    b = b / np.linalg.norm(b)
    _add_student(db, 11, [a])
    _add_student(db, 12, [b])
    idx = EmbeddingIndex()
    idx.refresh()
    probe = (a + b) / np.linalg.norm(a + b)
    scores = sorted(
        (float(np.dot(probe, a)), float(np.dot(probe, b))), reverse=True
    )
    assert scores[0] >= settings.recognition_threshold  # threshold alone passes
    res = idx.match(probe)
    assert res.student_id is None  # ...but the margin rule refuses it
    assert res.score == pytest.approx(scores[0], abs=1e-5)


def test_duplicate_identity_does_not_block_genuine_match(db) -> None:
    """Same human enrolled twice (gallery sim >= duplicate_identity_sim) must
    not trigger the margin rule (measured cause of margin ~0.002)."""
    v = _unit(42)
    noisy = v + 1e-3 * np.random.default_rng(7).standard_normal(EMBEDDING_DIM).astype(np.float32)
    noisy = noisy / np.linalg.norm(noisy)
    s1 = _add_student(db, 21, [v])
    s2 = _add_student(db, 22, [noisy])  # duplicate identity
    idx = EmbeddingIndex()
    idx.refresh()
    res = idx.match(v)
    # genuine match survives: winner's duplicate is skipped as runner-up
    assert res.student_id in (s1.id, s2.id)
    assert res.score > 0.99


def test_threshold_and_margin_are_configured_not_hardcoded(db) -> None:
    idx = EmbeddingIndex()
    assert idx.threshold == settings.recognition_threshold
    assert idx.margin == settings.recognition_margin
    assert idx.duplicate_sim == settings.duplicate_identity_sim
    # the values must sit inside the measured score gap (see config comments)
    assert 0.1246 < settings.recognition_threshold < 0.5030
    assert settings.recognition_margin > 0
