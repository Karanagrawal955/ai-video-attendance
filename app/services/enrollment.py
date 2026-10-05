"""Student enrollment: batch-embed reference photos on the shared GPU service."""

from __future__ import annotations

import csv
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import redis_client
from ..config import settings
from ..crypto import encrypt_embeddings
from ..inference import client as inference_client
from ..models import Student
from ..quality import check_photo_quality, check_duplicate_face

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


def _validate_reg_no(reg_no: str) -> None:
    """Validate registration number format from config."""
    pattern = getattr(settings, "registration_no_pattern", r"^[A-Za-z0-9_-]{6,12}$")
    if not re.match(pattern, reg_no):
        raise EnrollmentError(
            f"registration_no {reg_no!r} does not match required pattern {pattern!r}",
            status_code=400,
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
    skip_quality_checks: bool = False,
) -> Student:
    name = name.strip()
    registration_no = registration_no.strip()
    if not name or not registration_no:
        raise EnrollmentError("name and registration_no are required")

    _validate_reg_no(registration_no)

    existing = db.scalars(
        select(Student).where(Student.registration_no == registration_no)
    ).first()
    if existing is not None:
        raise EnrollmentError(
            f"registration_no {registration_no!r} already exists", status_code=409
        )

    # Quality checks on each photo
    accepted_photos = []
    accepted_filenames = []
    for i, photo_bytes in enumerate(photos):
        fname = filenames[i] if filenames and i < len(filenames) else f"photo_{i}"
        if not skip_quality_checks and settings.face_quality_enabled:
            qr = check_photo_quality(photo_bytes)
            if not qr.ok:
                raise EnrollmentError(
                    f"photo {fname}: quality check failed: {', '.join(qr.reasons)}",
                    status_code=400,
                )
        accepted_photos.append(photo_bytes)
        accepted_filenames.append(fname)

    # Batch embed all accepted photos
    try:
        embeddings = _embed_or_raise(accepted_photos, context="reference")
    except InferenceUnavailable as exc:
        raise EnrollmentError(f"inference service unavailable: {exc}", status_code=503) from exc

    # Check for duplicate faces
    for new_emb in embeddings:
        is_dup, dup_id = check_duplicate_face(db, new_emb)
        if is_dup:
            dup_student = db.get(Student, dup_id)
            dup_reg = dup_student.registration_no if dup_student else "unknown"
            raise EnrollmentError(
                f"face matches existing student "
                f"(reg_no={dup_reg[:3]}***), cannot enroll duplicate face",
                status_code=409,
            )

    if not embeddings:
        raise EnrollmentError("no valid photos after quality checks", status_code=400)

    student = Student(
        name=name,
        registration_no=registration_no,
        section=section.strip() if section else None,
        embeddings="",  # encrypted string
        photo_paths=[],
    )
    db.add(student)
    try:
        db.flush()
        student.embeddings = encrypt_embeddings(embeddings)
        student.photo_paths = _save_photos(student.id, accepted_photos, accepted_filenames)
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
            "photos": len(accepted_photos),
        },
    )
    db.refresh(student)
    return student


@dataclass
class BulkImportResult:
    """Result of a single student import attempt."""
    row_number: int
    name: str
    registration_no: str
    section: str | None
    status: str  # "accepted" | "rejected"
    reason: str
    photo_count: int
    accepted_embeddings: int
    student_id: int | None = None


def bulk_enroll_from_csv(
    db: Session,
    csv_path: Path,
    *,
    data_dir: Path | None = None,
    skip_quality_checks: bool = False,
    dry_run: bool = False,
) -> tuple[list[BulkImportResult], int, int]:
    """
    Bulk enroll students from a CSV file.

    CSV columns: name, registration_no, section, photo_path_or_folder

    Returns: (results, accepted_count, rejected_count)
    """
    if data_dir is None:
        data_dir = settings.photos_root

    results: list[BulkImportResult] = []
    accepted = 0
    rejected = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            print(f"DEBUG: Processing row {row_num}")
            name = row.get("name", "").strip()
            reg_no = row.get("registration_no", "").strip()
            section = row.get("section", "").strip() or None
            photo_spec = row.get("photo_path_or_folder", "").strip()

            # Validate required fields
            if not name or not reg_no or not photo_spec:
                results.append(BulkImportResult(
                    row_number=row_num,
                    name=name,
                    registration_no=reg_no,
                    section=section,
                    status="rejected",
                    reason="missing required field(s): name, registration_no, photo_path_or_folder",
                    photo_count=0,
                    accepted_embeddings=0,
                ))
                rejected += 1
                continue

            # Validate registration number format
            try:
                _validate_reg_no(reg_no)
            except EnrollmentError as exc:
                results.append(BulkImportResult(
                    row_number=row_num,
                    name=name,
                    registration_no=reg_no,
                    section=section,
                    status="rejected",
                    reason=str(exc),
                    photo_count=0,
                    accepted_embeddings=0,
                ))
                rejected += 1
                continue

            # Check for duplicate registration number
            existing = db.scalars(
                select(Student).where(Student.registration_no == reg_no)
            ).first()
            if existing is not None:
                results.append(BulkImportResult(
                    row_number=row_num,
                    name=name,
                    registration_no=reg_no,
                    section=section,
                    status="rejected",
                    reason=f"duplicate registration_no: {existing.name} already enrolled",
                    photo_count=0,
                    accepted_embeddings=0,
                ))
                rejected += 1
                continue

            # Collect photos from path or folder
            photos_bytes = []
            photo_filenames = []
            photo_path = data_dir / photo_spec

            if photo_path.is_dir():
                for p in sorted(photo_path.iterdir()):
                    if p.suffix.lower() in _ALLOWED_EXTS:
                        try:
                            photos_bytes.append(p.read_bytes())
                            photo_filenames.append(p.name)
                        except Exception as e:
                            logger.warning("failed to read %s: %s", p, e)
            elif photo_path.is_file():
                try:
                    photos_bytes.append(photo_path.read_bytes())
                    photo_filenames.append(photo_path.name)
                except Exception as e:
                    logger.warning("failed to read %s: %s", photo_path, e)
            else:
                results.append(BulkImportResult(
                    row_number=row_num,
                    name=name,
                    registration_no=reg_no,
                    section=section,
                    status="rejected",
                    reason=f"photo path/folder not found: {photo_spec}",
                    photo_count=0,
                    accepted_embeddings=0,
                ))
                rejected += 1
                continue

            # Cap at max_reference_photos
            if len(photos_bytes) > settings.max_reference_photos:
                photos_bytes = photos_bytes[:settings.max_reference_photos]
                photo_filenames = photo_filenames[:settings.max_reference_photos]

            # Check minimum photos
            if len(photos_bytes) < settings.min_reference_photos:
                results.append(BulkImportResult(
                    row_number=row_num,
                    name=name,
                    registration_no=reg_no,
                    section=section,
                    status="rejected",
                    reason=(
                        f"too few photos: got {len(photos_bytes)}, "
                        f"need {settings.min_reference_photos}-{settings.max_reference_photos}"
                    ),
                    photo_count=len(photos_bytes),
                    accepted_embeddings=0,
                ))
                rejected += 1
                continue

            # Quality checks on all photos first
            quality_passed = True
            if not skip_quality_checks and settings.face_quality_enabled:
                for i, (photo_bytes, fname) in enumerate(zip(photos_bytes, photo_filenames)):
                    qr = check_photo_quality(photo_bytes)
                    if not qr.ok:
                        results.append(BulkImportResult(
                            row_number=row_num,
                            name=name,
                            registration_no=reg_no,
                            section=section,
                            status="rejected",
                            reason=f"photo {fname}: {', '.join(qr.reasons)}",
                            photo_count=len(photos_bytes),
                            accepted_embeddings=0,
                        ))
                        rejected += 1
                        quality_passed = False
                        break  # reject entire student if any photo fails

            if not quality_passed:
                continue

            # All photos passed quality checks - now embed all in batch
            print(f"DEBUG: Row {row_num} - calling _embed_or_raise")
            try:
                embs = _embed_or_raise(photos_bytes, context=f"reference batch")
                print(f"DEBUG: Row {row_num} - embedding success, got {len(embs)} embeddings")
            except (EnrollmentError, InferenceUnavailable) as exc:
                print(f"DEBUG: Row {row_num} - caught exception: {type(exc).__name__}: {exc}")
                results.append(BulkImportResult(
                    row_number=row_num,
                    name=name,
                    registration_no=reg_no,
                    section=section,
                    status="rejected",
                    reason=str(exc),
                    photo_count=len(photos_bytes),
                    accepted_embeddings=0,
                ))
                print(f"DEBUG: Row {row_num} - appended rejected result, results now: {len(results)}")
                rejected += 1
                continue

            # Check for duplicate faces across all embeddings
            for i, (new_emb, fname) in enumerate(zip(embs, photo_filenames)):
                is_dup, dup_id = check_duplicate_face(db, new_emb)
                if is_dup:
                    dup_student = db.get(Student, dup_id)
                    dup_reg = dup_student.registration_no if dup_student else "unknown"
                    results.append(BulkImportResult(
                        row_number=row_num,
                        name=name,
                        registration_no=reg_no,
                        section=section,
                        status="rejected",
                        reason=(
                            f"photo {fname}: face matches existing student "
                            f"(reg_no={dup_reg[:3]}***), cannot enroll duplicate face"
                        ),
                        photo_count=len(photos_bytes),
                        accepted_embeddings=0,
                    ))
                    rejected += 1
                    break
            else:
                # All photos passed - enroll student
                if not dry_run:
                    student = Student(
                        name=name,
                        registration_no=reg_no,
                        section=section,
                        embeddings=encrypt_embeddings(embs),
                        photo_paths=[],
                    )
                    db.add(student)
                    try:
                        db.flush()
                        student.photo_paths = _save_photos(student.id, photos_bytes, photo_filenames)
                        db.commit()
                        student_id = student.id
                    except Exception:
                        db.rollback()
                        if student.id:
                            _remove_photo_dir(student.id)
                        results.append(BulkImportResult(
                            row_number=row_num,
                            name=name,
                            registration_no=reg_no,
                            section=section,
                            status="rejected",
                            reason="database error during commit",
                            photo_count=len(photos_bytes),
                            accepted_embeddings=len(embs),
                        ))
                        rejected += 1
                        continue
                else:
                    student_id = None

                accepted += 1
                results.append(BulkImportResult(
                    row_number=row_num,
                    name=name,
                    registration_no=reg_no,
                    section=section,
                    status="accepted",
                    reason="",
                    photo_count=len(photos_bytes),
                    accepted_embeddings=len(embs),
                    student_id=student_id,
                ))
                if not dry_run:
                    redis_client.bump_students_version()

        print(f"DEBUG: Function complete. Results: {len(results)}")
        return results, accepted, rejected


def write_import_report(results: list[BulkImportResult], output_path: Path) -> None:
    """Write import report CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_number", "name", "registration_no", "section",
            "status", "reason", "photo_count", "accepted_embeddings", "student_id"
        ])
        for r in results:
            writer.writerow([
                r.row_number, r.name, r.registration_no, r.section or "",
                r.status, r.reason, r.photo_count, r.accepted_embeddings,
                r.student_id or ""
            ])


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
    # Decrypt old, append new, re-encrypt
    from ..crypto import decrypt_embeddings
    old_embs = decrypt_embeddings(student.embeddings) if student.embeddings else []
    all_embs = old_embs + new_embeddings
    student.embeddings = encrypt_embeddings(all_embs)
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
    _remove_photo_dir(student.id)
    paths = _save_photos(student.id, photos, filenames)
    student.embeddings = encrypt_embeddings(embeddings)
    student.photo_paths = paths
    db.commit()
    redis_client.bump_students_version()
    db.refresh(student)
    return student


def clear_face(db: Session, student: Student) -> Student:
    student.embeddings = encrypt_embeddings([])
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