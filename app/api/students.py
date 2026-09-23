"""Student enrollment & face-data management endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..models import Student
from ..schemas import OkResponse, StudentOut, StudentUpdate
from ..services import enrollment
from .deps import get_current_admin, get_db

logger = logging.getLogger("app.api.students")

router = APIRouter(prefix="/students", tags=["students"])

DbDep = Annotated[Session, Depends(get_db)]
AuthDep = Annotated[str, Depends(get_current_admin)]


def _student_out(student: Student, include_embeddings: bool = False) -> StudentOut:
    return StudentOut(
        id=student.id,
        name=student.name,
        registration_no=student.registration_no,
        section=student.section,
        photo_paths=list(student.photo_paths or []),
        embedding_count=len(student.embeddings or []),
        embeddings=[list(map(float, e)) for e in student.embeddings]
        if include_embeddings
        else None,
        created_at=student.created_at,
        updated_at=student.updated_at,
    )


def _get_or_404(db: Session, student_id: int) -> Student:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"student {student_id} not found")
    return student


def _map_enrollment_error(exc: enrollment.EnrollmentError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)


@router.post(
    "",
    response_model=StudentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a student with 3-5 reference photos",
)
def create_student(
    db: DbDep,
    _: AuthDep,
    name: Annotated[str, Form(min_length=1, max_length=200)],
    registration_no: Annotated[str, Form(min_length=1, max_length=64)],
    section: Annotated[str | None, Form(max_length=64)] = None,
    photos: list[UploadFile] = File(...),
) -> StudentOut:
    data, names = [], []
    for photo in photos:
        blob = photo.file.read()
        if not blob:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"photo {photo.filename or ''} is empty",
            )
        data.append(blob)
        names.append(photo.filename or "")
    try:
        student = enrollment.enroll_student(
            db,
            name=name,
            registration_no=registration_no,
            section=section,
            photos=data,
            filenames=names,
        )
    except enrollment.EnrollmentError as exc:
        raise _map_enrollment_error(exc) from exc
    except enrollment.InferenceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"inference service unavailable: {exc}",
        ) from exc
    return _student_out(student, include_embeddings=False)


@router.get("", response_model=dict, summary="List students (paginated)")
def list_students(
    db: DbDep,
    _: AuthDep,
    q: str | None = Query(default=None, description="Match name or registration no."),
    section: str | None = Query(default=None),
    include_embeddings: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    stmt = select(Student)
    count_stmt = select(func.count()).select_from(Student)
    if q:
        pattern = f"%{q.strip()}%"
        filt = or_(
            Student.name.ilike(pattern),
            Student.registration_no.ilike(pattern),
        )
        stmt = stmt.where(filt)
        count_stmt = count_stmt.where(filt)
    if section:
        stmt = stmt.where(Student.section == section)
        count_stmt = count_stmt.where(Student.section == section)

    total = int(db.scalar(count_stmt) or 0)
    rows = db.scalars(
        stmt.order_by(Student.created_at.desc()).offset(offset).limit(limit)
    ).all()
    return {
        "items": [
            _student_out(s, include_embeddings=include_embeddings) for s in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{student_id}", response_model=StudentOut, summary="Get one student")
def get_student(
    student_id: int,
    db: DbDep,
    _: AuthDep,
    include_embeddings: bool = Query(default=False),
) -> StudentOut:
    return _student_out(_get_or_404(db, student_id), include_embeddings)


@router.put("/{student_id}", response_model=StudentOut, summary="Update student details")
def update_student(
    student_id: int,
    body: StudentUpdate,
    db: DbDep,
    _: AuthDep,
) -> StudentOut:
    student = _get_or_404(db, student_id)
    if body.name is not None:
        student.name = body.name.strip()
    if body.section is not None:
        student.section = body.section.strip() or None
    if body.registration_no is not None:
        new_reg = body.registration_no.strip()
        if new_reg != student.registration_no:
            clash = db.scalars(
                select(Student).where(Student.registration_no == new_reg)
            ).first()
            if clash is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"registration_no {new_reg!r} already exists",
                )
        student.registration_no = new_reg
    db.commit()
    db.refresh(student)
    return _student_out(student)


@router.post(
    "/{student_id}/photos",
    response_model=StudentOut,
    summary="Add extra reference photos to an existing student",
)
def add_student_photos(
    student_id: int,
    db: DbDep,
    _: AuthDep,
    photos: list[UploadFile] = File(...),
) -> StudentOut:
    student = _get_or_404(db, student_id)
    data, names = [], []
    for photo in photos:
        blob = photo.file.read()
        if blob:
            data.append(blob)
            names.append(photo.filename or "")
    try:
        student = enrollment.add_photos(db, student, data, names)
    except enrollment.EnrollmentError as exc:
        raise _map_enrollment_error(exc) from exc
    except enrollment.InferenceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"inference service unavailable: {exc}",
        ) from exc
    return _student_out(student)


@router.put(
    "/{student_id}/face",
    response_model=StudentOut,
    summary="Replace all reference photos / face embeddings",
)
def replace_student_face(
    student_id: int,
    db: DbDep,
    _: AuthDep,
    photos: list[UploadFile] = File(...),
) -> StudentOut:
    student = _get_or_404(db, student_id)
    data, names = [], []
    for photo in photos:
        blob = photo.file.read()
        if blob:
            data.append(blob)
            names.append(photo.filename or "")
    try:
        student = enrollment.replace_face(db, student, data, names)
    except enrollment.EnrollmentError as exc:
        raise _map_enrollment_error(exc) from exc
    except enrollment.InferenceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"inference service unavailable: {exc}",
        ) from exc
    return _student_out(student)


@router.delete(
    "/{student_id}/face",
    response_model=OkResponse,
    summary="Delete a student's face data (keeps the record)",
)
def delete_student_face(student_id: int, db: DbDep, _: AuthDep) -> OkResponse:
    student = _get_or_404(db, student_id)
    enrollment.clear_face(db, student)
    return OkResponse(detail="face data cleared")


@router.delete(
    "/{student_id}",
    response_model=OkResponse,
    summary="Delete a student (cascades sessions, nulls audit logs)",
)
def delete_student(student_id: int, db: DbDep, _: AuthDep) -> OkResponse:
    student = _get_or_404(db, student_id)
    enrollment.delete_student(db, student)
    return OkResponse(detail="student deleted")
