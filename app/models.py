"""SQLAlchemy ORM models (PostgreSQL in production, SQLite in tests)."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Student(Base):
    __tablename__ = "students"
    __table_args__ = (
        CheckConstraint("length(trim(name)) > 0", name="ck_students_name_nonempty"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    registration_no: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    section: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # List of 512-dim L2-normalisable float vectors (one per reference photo).
    embeddings: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    # Reference photo paths, stored relative to settings.data_dir.
    photo_paths: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    sessions: Mapped[list["AttendanceSession"]] = relationship(
        back_populates="student",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @property
    def embedding_count(self) -> int:
        return len(self.embeddings or [])


class Camera(Base):
    __tablename__ = "cameras"
    __table_args__ = (
        CheckConstraint(
            "type IN ('entry', 'exit', 'both')", name="ck_cameras_type"
        ),
        CheckConstraint(
            "sampling_rate >= 1 AND sampling_rate <= 600",
            name="ck_cameras_sampling_rate",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    rtsp_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # entry -> open sessions, exit -> close sessions, both -> whichever applies
    type: Mapped[str] = mapped_column(String(16), nullable=False, default="entry")
    # Process every Nth decoded frame (CPU sampling before GPU batching).
    sampling_rate: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5, server_default="5"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    sessions_in: Mapped[list["AttendanceSession"]] = relationship(
        foreign_keys="AttendanceSession.camera_in_id",
        back_populates="camera_in",
    )
    sessions_out: Mapped[list["AttendanceSession"]] = relationship(
        foreign_keys="AttendanceSession.camera_out_id",
        back_populates="camera_out",
    )


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ongoing', 'completed')", name="ck_sessions_status"
        ),
        Index("ix_sessions_student_date", "student_id", "date"),
        Index("ix_sessions_date_status", "date", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    camera_in_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    camera_out_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    entry_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    exit_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Local (settings.local_timezone) calendar date of entry_time - all
    # queries for "attendance of day D" filter on this column.
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ongoing", server_default="ongoing"
    )
    # Seconds between entry and exit; while ongoing it is NULL and computed
    # on the fly by the summary endpoints.
    total_duration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    student: Mapped[Student] = relationship(back_populates="sessions")
    camera_in: Mapped[Camera | None] = relationship(
        foreign_keys=[camera_in_id], back_populates="sessions_in"
    )
    camera_out: Mapped[Camera | None] = relationship(
        foreign_keys=[camera_out_id], back_populates="sessions_out"
    )


class RecognitionLog(Base):
    """Raw audit log of every (deduplicated) recognition, including GPU latency."""

    __tablename__ = "recognition_logs"
    __table_args__ = (
        Index("ix_logs_student_ts", "student_id", "timestamp"),
        Index("ix_logs_camera_ts", "camera_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # NULL student_id => recognized as unknown (only written when
    # LOG_UNKNOWN_FACES=true).  SET NULL keeps the audit trail on delete.
    student_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("students.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    camera_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    # Cosine similarity in [-1, 1]; matches require >= RECOGNITION_THRESHOLD.
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # End-to-end batch inference latency reported by the GPU service.
    gpu_inference_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
