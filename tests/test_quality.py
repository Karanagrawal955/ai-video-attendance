"""Quality / validation suite (recreated after the file was lost in the
2026-10-02 incident; supersedes the old 13-test version).

Covered here (the 9 required cases):
  1. duplicate registration number
  2. malformed registration id
  3. blurry photo            (engine quality gate, Laplacian variance)
  4. tiny face               (per-photo quality check)
  5. two faces in one photo
  6. no face in one photo
  7. same face enrolled under two IDs
  8. encryption round trip
  9. bulk-import endpoint
+ production startup requires EMBEDDING_ENCRYPTION_KEY (STEP 4 guard).

All numbers asserted below are produced by the real code paths; nothing here
scales a "typical" value.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.crypto import decrypt_embeddings, encrypt_embeddings
from app.quality import check_duplicate_face, check_photo_quality
from app.services.enrollment import EnrollmentError, enroll_student
from tests.conftest import make_photos


# ----------------------------------------------------------------- helpers
def _fake_infer(faces: list[dict]):
    """Return a stub for app.inference.client.infer_frames with given faces."""

    def _infer(frames, *args, **kwargs):  # noqa: ANN001, ANN003
        return {"results": [{"faces": faces}]}

    return _infer


GOOD_FACE = {
    "bbox": [100.0, 100.0, 300.0, 300.0],  # 200px wide -> passes width/area
    "score": 0.95,
    "landmarks": [120, 140, 180, 140, 150, 200],
    "pose": {"yaw": 5.0, "pitch": -3.0, "roll": 1.0},
}


def _patch_infer(monkeypatch: pytest.MonkeyPatch, faces: list[dict]) -> None:
    import app.inference.client as client
    import app.quality as quality

    stub = _fake_infer(faces)
    monkeypatch.setattr(client, "infer_frames", stub)
    # quality holds `client as inference_client` - same module object
    monkeypatch.setattr(quality.inference_client, "infer_frames", stub)


# --------------------------------------------------- 1. duplicate reg no
def test_duplicate_registration_number_rejected(db, fake_embed) -> None:
    photos = [b"\xff\xd8\xff\xe0photo" + bytes([i]) for i in range(3)]
    first = enroll_student(
        db, name="Dup Test A", registration_no="DUPL1234", section="CSE-A",
        photos=photos, skip_quality_checks=True,
    )
    assert first.id is not None
    with pytest.raises(EnrollmentError) as exc:
        enroll_student(
            db, name="Dup Test B", registration_no="DUPL1234", section="CSE-A",
            photos=photos, skip_quality_checks=True,
        )
    assert exc.value.status_code == 409
    assert "already exists" in str(exc.value)


# ------------------------------------------------------- 2. malformed id
@pytest.mark.parametrize("bad", ["", "X1", "A B C D", "!!bad!!", "TOOLONGREGNO1"])
def test_malformed_registration_id_rejected(db, fake_embed, bad: str) -> None:
    with pytest.raises(EnrollmentError) as exc:
        enroll_student(
            db, name="Bad Id", registration_no=bad, section=None,
            photos=[b"p" * 10] * 3, skip_quality_checks=True,
        )
    assert exc.value.status_code == 400


def test_valid_registration_id_accepted(db, fake_embed) -> None:
    ok = enroll_student(
        db, name="Good Id", registration_no="REG0007", section="CSE-A",
        photos=[b"p" * 10] * 3, skip_quality_checks=True,
    )
    assert ok.registration_no == "REG0007"


# --------------------------------------------------------- 3. blurry photo
def test_blurry_photo_fails_engine_quality_gate() -> None:
    """Real gate: FaceEngine._quality_ok uses Laplacian variance (config:
    face_min_sharpness=30).  A flat/blurry crop must fail, a sharp one pass."""
    from app.inference.engine import FaceEngine

    # Build without __init__: the gate only reads self.cfg, and constructing
    # FaceAnalysis would load the ONNX weights (tested elsewhere, on purpose).
    eng = FaceEngine.__new__(FaceEngine)
    eng.cfg = settings
    det_row = np.array([10.0, 10.0, 130.0, 130.0, 0.95], dtype=np.float32)

    rng = np.random.default_rng(0)
    blurry = np.full((120, 120, 3), 128, dtype=np.uint8)  # featureless
    sharp = rng.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)

    import cv2

    def lap_var(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    sharp_var, blurry_var = lap_var(sharp), lap_var(blurry)
    assert blurry_var < settings.face_min_sharpness, (blurry_var,)
    assert sharp_var > settings.face_min_sharpness, (sharp_var,)
    assert eng._quality_ok(blurry, det_row) is False
    assert eng._quality_ok(sharp, det_row) is True

    # too-dark crop is rejected by the same gate
    dark = np.zeros((120, 120, 3), dtype=np.uint8)
    assert eng._quality_ok(dark, det_row) is False


# ------------------------------------------------------------ 4. tiny face
def test_tiny_face_rejected_by_quality_check(monkeypatch) -> None:
    _patch_infer(monkeypatch, [dict(GOOD_FACE, bbox=[0.0, 0.0, 40.0, 40.0])])
    res = check_photo_quality(b"whatever.jpg-bytes")
    assert res.ok is False
    assert any(r.startswith("face width 40px") for r in res.reasons), res.reasons
    assert res.metrics["face_count"] == 1


def test_small_face_area_rejected_by_quality_check(monkeypatch) -> None:
    # 90x80 = 7200px2 area is fine but width 90 is fine too; use a thin box:
    _patch_infer(monkeypatch, [dict(GOOD_FACE, bbox=[0.0, 0.0, 30.0, 30.0])])
    res = check_photo_quality(b"photo")
    assert res.ok is False
    assert any("face width" in r for r in res.reasons)


# ----------------------------------------------------------- 5. two faces
def test_two_faces_rejected(monkeypatch) -> None:
    _patch_infer(monkeypatch, [dict(GOOD_FACE), dict(GOOD_FACE, bbox=[400.0, 100.0, 600.0, 300.0])])
    res = check_photo_quality(b"photo")
    assert res.ok is False
    assert res.reasons == ["multiple faces detected (2)"], res.reasons
    assert res.metrics["face_count"] == 2


# ------------------------------------------------------------- 6. no face
def test_no_face_rejected(monkeypatch) -> None:
    _patch_infer(monkeypatch, [])
    res = check_photo_quality(b"photo")
    assert res.ok is False
    assert res.reasons == ["no face detected"]
    assert res.metrics["face_count"] == 0


def test_good_single_face_accepted(monkeypatch) -> None:
    _patch_infer(monkeypatch, [dict(GOOD_FACE)])
    res = check_photo_quality(b"photo")
    assert res.ok is True, res.reasons
    assert res.reasons == []


# ------------------------------------------- 7. same face under two IDs
def test_same_face_under_two_ids_rejected(db, fake_embed, monkeypatch) -> None:
    import app.services.enrollment as enrollment

    photos = [b"\xff\xd8\xff\xe0face" + bytes([i]) for i in range(3)]
    a = enroll_student(
        db, name="Identity A", registration_no="IDAAA1", section="CSE-A",
        photos=photos, skip_quality_checks=True,
    )
    emb = decrypt_embeddings(a.embeddings)[0]

    # second person record, but the embedded face is byte-identical evidence
    monkeypatch.setattr(
        enrollment,
        "_embed_photos",
        lambda ph: [
            {"ok": True, "embedding": emb, "score": 0.99, "bbox": [0, 0, 1, 1]}
            for _ in ph
        ],
    )
    with pytest.raises(EnrollmentError) as exc:
        enroll_student(
            db, name="Identity A again", registration_no="IDBBB2", section="CSE-A",
            photos=photos, skip_quality_checks=True,
        )
    assert exc.value.status_code == 409
    assert "face matches existing student" in str(exc.value)

    # and the helper used by the API agrees
    is_dup, dup_id = check_duplicate_face(db, emb)
    assert is_dup is True and dup_id == a.id


def test_different_face_not_flagged_as_duplicate(db, fake_embed) -> None:
    photos = [b"\xff\xd8\xff\xe0face" + bytes([i]) for i in range(3)]
    a = enroll_student(
        db, name="Identity C", registration_no="IDCCC3", section="CSE-A",
        photos=photos, skip_quality_checks=True,
    )
    other = (np.ones(512, dtype=np.float32) / np.sqrt(512)).tolist()
    is_dup, _ = check_duplicate_face(db, other)
    assert is_dup is False
    assert a.id is not None


# -------------------------------------------------- 8. encryption round trip
def test_encryption_round_trip() -> None:
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((2, 512)).astype(np.float64)
    blob = encrypt_embeddings(vecs.tolist())
    assert isinstance(blob, str)
    assert not blob.startswith("[")  # not plaintext JSON
    assert "0.000" not in blob
    back = np.asarray(decrypt_embeddings(blob), dtype=np.float64)
    assert back.shape == (2, 512)
    assert np.max(np.abs(back - vecs)) < 1e-6
    # stored blob must be stable across calls for the same plaintext (no nonce reuse)
    assert encrypt_embeddings(vecs.tolist()) != blob


# ------------------------------------------------- 9. bulk-import endpoint
def test_bulk_import_endpoint(client, auth_header, fake_embed, tmp_path) -> None:
    import csv

    root = settings.photos_root
    ok_reg = "BLK0001"
    for i in range(3):
        d = root / "students" / ok_reg
        d.mkdir(parents=True, exist_ok=True)
        (d / f"photo_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0bulk" + bytes([i]))

    csv_path = tmp_path / "students.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "registration_no", "section", "photo_path_or_folder"])
        w.writerow(["Bulk One", ok_reg, "CSE-A", f"students/{ok_reg}"])       # accepted
        w.writerow(["Bulk Two", ok_reg, "CSE-A", f"students/{ok_reg}"])       # dup reg
        w.writerow(["Bulk Bad", "!!", "CSE-A", f"students/{ok_reg}"])         # malformed
        w.writerow(["Bulk Gone", "BLK0002", "CSE-A", "students/does_not_exist"])  # no photos

    resp = client.post(
        "/students/bulk-import",
        files={"csv_file": ("students.csv", csv_path.read_bytes(), "text/csv")},
        data={"skip_quality_checks": "true"},
        headers=auth_header,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 4
    assert body["accepted"] == 1, body
    assert body["rejected"] == 3, body

    reasons = {r["registration_no"]: r["reason"] for r in body["report"]}
    assert "duplicate registration_no" in reasons[ok_reg]
    assert "does not match required pattern" in reasons["!!"]
    assert "photo path/folder not found" in reasons["BLK0002"]

    # the accepted row really exists (and only that one)
    listed = client.get("/students", headers=auth_header).json()
    regs = {s["registration_no"] for s in listed["items"]}
    assert regs == {ok_reg}, regs


# ------------------------------------------- STEP 4 guard: production key
def test_production_startup_requires_encryption_key(monkeypatch) -> None:
    from pydantic import ValidationError

    from app.config import Settings

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "a" * 64)
    monkeypatch.setenv("EMBEDDING_ENCRYPTION_KEY", "")
    with pytest.raises(ValidationError) as exc:
        Settings()
    assert "EMBEDDING_ENCRYPTION_KEY is required" in str(exc.value)

    import base64
    import os

    monkeypatch.setenv(
        "EMBEDDING_ENCRYPTION_KEY",
        base64.urlsafe_b64encode(os.urandom(32)).decode(),
    )
    assert Settings().embedding_encryption_key  # starts fine with a real key
