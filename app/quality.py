"""Per-photo quality checks for enrollment and recognition."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import settings
from .inference import client as inference_client

logger = logging.getLogger("app.quality")


@dataclass
class QualityResult:
    """Result of quality checks on a single photo."""
    ok: bool
    reasons: list[str]
    metrics: dict[str, Any]
    face_bbox: list[float] | None = None
    face_landmarks: list[float] | None = None
    pose: dict[str, float] | None = None  # yaw, pitch, roll


def _analyze_face(photo_bytes: bytes) -> dict:
    """Call inference service to get face detection + landmarks + pose."""
    try:
        results = inference_client.infer_frames([photo_bytes])
        if not results or not results.get("results") or not results["results"][0]:
            return {"faces": []}
        return results["results"][0]
    except inference_client.InferenceError as exc:
        logger.warning("inference error during quality check: %s", exc)
        return {"faces": []}


def check_photo_quality(photo_bytes: bytes) -> QualityResult:
    """
    Run all quality checks on a single photo.
    Returns QualityResult with ok=True only if all checks pass.
    """
    reasons = []
    metrics = {}

    # Run inference to get face data
    face_data = _analyze_face(photo_bytes)
    faces = face_data.get("faces", [])

    if not faces:
        return QualityResult(
            ok=False,
            reasons=["no face detected"],
            metrics={"face_count": 0},
        )

    if len(faces) > 1:
        return QualityResult(
            ok=False,
            reasons=[f"multiple faces detected ({len(faces)})"],
            metrics={"face_count": len(faces)},
        )

    face = faces[0]
    bbox = face.get("bbox", [0, 0, 0, 0])
    landmarks = face.get("landmarks", [])
    pose = face.get("pose", {})
    det_score = face.get("score", 0.0)

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height

    metrics.update({
        "face_count": 1,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "detection_score": det_score,
        "yaw": pose.get("yaw", 0.0),
        "pitch": pose.get("pitch", 0.0),
        "roll": pose.get("roll", 0.0),
    })

    # Check detection score
    if det_score < settings.face_min_det_score:
        reasons.append(f"detection score {det_score:.2f} < {settings.face_min_det_score}")

    # Check face width
    if width < settings.face_min_width:
        reasons.append(f"face width {width:.0f}px < {settings.face_min_width}px")

    # Check face area
    if area < settings.face_min_area:
        reasons.append(f"face area {area:.0f}px² < {settings.face_min_area}px²")

    # Check pose
    yaw = abs(pose.get("yaw", 0.0))
    pitch = abs(pose.get("pitch", 0.0))
    roll = abs(pose.get("roll", 0.0))

    if yaw > settings.face_max_yaw:
        reasons.append(f"yaw {yaw:.1f}° > {settings.face_max_yaw}°")
    if pitch > settings.face_max_pitch:
        reasons.append(f"pitch {pitch:.1f}° > {settings.face_max_pitch}°")
    if roll > settings.face_max_roll:
        reasons.append(f"roll {roll:.1f}° > {settings.face_max_roll}°")

    # Check brightness (requires decoding image - skip for now, can be added)
    # Could use PIL/OpenCV to compute mean brightness if needed

    ok = len(reasons) == 0

    return QualityResult(
        ok=ok,
        reasons=reasons,
        metrics=metrics,
        face_bbox=bbox,
        face_landmarks=landmarks,
        pose=pose,
    )


def check_duplicate_face(db, new_embedding: list[float], exclude_student_id: int | None = None) -> tuple[bool, int | None]:
    """
    Check if a face embedding matches an existing student's embeddings.
    Returns (is_duplicate, matched_student_id).
    """
    from .models import Student
    from .crypto import decrypt_embeddings

    # Get all enrolled students with embeddings
    students = db.query(Student).filter(Student.embeddings.isnot(None)).all()
    if not students:
        return False, None

    # Collect all embeddings with student IDs
    all_embeddings = []
    student_ids = []
    for student in students:
        if student.id == exclude_student_id:
            continue
        try:
            embs = decrypt_embeddings(student.embeddings)
        except Exception:
            continue
        for emb in embs:
            all_embeddings.append(emb)
            student_ids.append(student.id)

    if not all_embeddings:
        return False, None

    # Compute cosine similarities
    import numpy as np
    new_emb = np.array(new_embedding, dtype=np.float32)
    all_embs = np.array(all_embeddings, dtype=np.float32)

    # Normalize - add small epsilon to avoid division by zero
    new_norm = np.linalg.norm(new_emb)
    if new_norm > 1e-8:
        new_emb = new_emb / new_norm

    all_norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    valid = all_norms.squeeze() > 1e-8
    if not np.any(valid):
        return False, None

    all_embs = all_embs[valid]
    all_embs = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True)
    # Ensure valid is always a 1D array for zip
    valid_array = np.atleast_1d(valid)
    student_ids = [sid for sid, v in zip(student_ids, valid_array) if v]

    sims = np.dot(all_embs, new_emb)
    # Ensure sims is at least 1D for indexing
    sims = np.atleast_1d(sims)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx].item())

    # Use recognition threshold for duplicate detection
    if best_sim >= settings.recognition_threshold:
        return True, student_ids[best_idx]

    return False, None