"""Student enrollment: batch-embed reference photos on the shared GPU service."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import redis_client
from ..config import settings
from ..inference import client as inference_client
from ..models import Student

logger = logging.getLogger("app.enrollment")

_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class EnrollmentError(Exception):
    """User-facing validation error (HTTP 400/409)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class InferenceUnavailable(Exception):
    """GPU/inference service unreachable (HTTP 503)."""


# Indirection so tests can monkeypatch without touching the RPC module.
def _embed_photos(photos: list[bytes]) -> list[dict]:
    try:
        return inference_client.embed_images(photos)
    except inference_client.InferenceError as exc:
        raise InferenceUnavailable(str(exc)) from exc


def _validate_photo_count(n: int) -> None:
    lo, hi = settings.min_reference_photos, settings.max_reference_photos
    if n < lo or n > hi:
        raise EnrollmentError(
            f"provide between {lo} and {hi} reference photos, got {n}"
        )


def _embed_or_raise(photos: list[bytes], *, context: str) -> list[list[float]]:
    _validate_photo_count(len(photos))
    results = _embed_photos(photos)
    embeddings: list[list[float]] = []
    for i, res in enumerate(results):
        if not res.get("ok"):
            reason = res.get("reason", "no face detected")
            raise EnrollmentError(
                f"{context} photo {i + 1}: {reason}. "
                "Use clear, front-facing photos with a single visible face."
            )
        embeddings.append([float(x) for x in res["embedding"]])
    return embeddings


def _save_photos(student_id: int, photos: list[bytes], filenames: list[str] | None) -> list[str]:
    directory = settings.photos_root / "students" / str(student_id)
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []
    for i, data in enumerate(photos):
        ext = ".jpg"
        if filenames and i < len(filenames):
            candidate = Path(filenames[i] or "").suffix.lower()
            if candidate in _ALLOWED_EXTS:
                ext = candidate
        abs_path = directory / f"photo_{i}{ext}"
        abs_path.write_bytes(data)
        rel_paths.append(
            (Path("students") / str(student_id) / f"photo_{i}{ext}").as_posix()
        )
    return rel_paths


def _remove_photo_dir(student_id: int) -> None:
    directory = settings.photos_root / "students" / str(student_id)
    shutil.rmtree(directory, ignore_errors=True)


def enroll_student(
    db: Session,
    *,
    name: str,
    registration_no: str,
    section: str | None,
    photos: list[bytes],
    filenames: list[str] | None = None,
) -> Student:
    name = name.strip()
    registration_no = registration_no.strip()
    if not name or not registration_no:
        raise EnrollmentError("name and registration_no are required")

    existing = db.scalars(
        select(Student).where(Student.registration_no == registration_no)
    ).first()
    if existing is not None:
        raise EnrollmentError(
            f"registration_no {registration_no!r} already exists", status_code=409
        )

    # One batched GPU call for all reference photos.
    embeddings = _embed_or_raise(photos, context="reference")

    student = Student(
        name=name,
        registration_no=registration_no,
        section=section.strip() if section else None,
        embeddings=[],
        photo_paths=[],
    )
    db.add(student)
    try:
        db.flush()
        student.embeddings = embeddings
        student.photo_paths = _save_photos(student.id, photos, filenames)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise EnrollmentError(
            f"registration_no {registration_no!r} already exists", status_code=409
        ) from exc
    except Exception:
        db.rollback()
        if student.id:
            _remove_photo_dir(student.id)
        raise

    redis_client.bump_students_version()
    logger.info(
        "student enrolled",
        extra={
            "student_id": student.id,
            "registration_no": registration_no,
            "photos": len(photos),
        },
    )
    db.refresh(student)
    return student


def add_photos(db: Session, student: Student, photos: list[bytes],
               filenames: list[str] | None = None) -> Student:
    """Append new reference photos to an existing student."""
    _validate_photo_count(len(photos))
    current = list(student.embeddings or [])
    if len(current) + len(photos) > settings.max_reference_photos:
        raise EnrollmentError(
            f"a student may have at most {settings.max_reference_photos} "
            f"reference photos (has {len(current)})"
        )
    new_embeddings = _embed_or_raise(photos, context="new")
    new_paths = _save_photos_append(student.id, photos, filenames, len(current))
    student.embeddings = current + new_embeddings
    student.photo_paths = list(student.photo_paths or []) + new_paths
    db.commit()
    redis_client.bump_students_version()
    db.refresh(student)
    return student


def _save_photos_append(
    student_id: int, photos: list[bytes], filenames: list[str] | None, start_index: int
) -> list[str]:
    directory = settings.photos_root / "students" / str(student_id)
    directory.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []
    for i, data in enumerate(photos):
        idx = start_index + i
        ext = ".jpg"
        if filenames and i < len(filenames):
            candidate = Path(filenames[i] or "").suffix.lower()
            if candidate in _ALLOWED_EXTS:
                ext = candidate
        (directory / f"photo_{idx}{ext}").write_bytes(data)
        rel_paths.append(
            (Path("students") / str(student_id) / f"photo_{idx}{ext}").as_posix()
        )
    return rel_paths


def replace_face(
    db: Session, student: Student, photos: list[bytes],
    filenames: list[str] | None = None,
) -> Student:
    """Replace all face data (embeddings + photos) for a student."""
    embeddings = _embed_or_raise(photos, context="reference")
    _remove_photo_dir(student.id)  # old files first? save new first then del old...
    # NOTE: _save_photos recreates the directory from scratch, so removing the
    # old one first is safe and keeps a single source of truth on disk.
    paths = _save_photos(student.id, photos, filenames)
    student.embeddings = embeddings
    student.photo_paths = paths
    db.commit()
    redis_client.bump_students_version()
    db.refresh(student)
    return student


def clear_face(db: Session, student: Student) -> Student:
    student.embeddings = []
    student.photo_paths = []
    _remove_photo_dir(student.id)
    db.commit()
    redis_client.bump_students_version()
    db.refresh(student)
    return student


def delete_student(db: Session, student: Student) -> None:
    student_id = student.id
    db.delete(student)  # cascades attendance sessions; logs keep SET NULL audit
    db.commit()
    _remove_photo_dir(student_id)
    redis_client.bump_students_version()
    logger.info("student deleted", extra={"student_id": student_id})
