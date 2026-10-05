"""SQLAlchemy ORM models (PostgreSQL in production, SQLite in tests)."""

from __future__ import annotations

from datetime import date, datetime, time

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
    Time,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


from .crypto import decrypt_embeddings


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
    # Stored encrypted at rest.
    embeddings: Mapped[str] = mapped_column(
        String, nullable=False, default="", server_default=""
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
    def embedding_list(self) -> list[list[float]]:
        """Get decrypted embeddings as list of float lists."""
        if not self.embeddings:
            return []
        try:
            return decrypt_embeddings(self.embeddings)
        except Exception:
            return []

    @property
    def embedding_count(self) -> int:
        return len(self.embedding_list)


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


class Period(Base):
    """Timetable period definition for per-period attendance."""

    __tablename__ = "periods"
    __table_args__ = (
        CheckConstraint("length(trim(name)) > 0", name="ck_periods_name_nonempty"),
        CheckConstraint(
            "start_time < end_time", name="ck_periods_time_range"
        ),
        CheckConstraint(
            "days_bitmask >= 0 AND days_bitmask <= 127",
            name="ck_periods_days_range",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    # bitmask Mon=1<<0 ... Sun=1<<6 ; 127 = all days
    days_bitmask: Mapped[int] = mapped_column(Integer, nullable=False, default=127, server_default="127")
    # NULL = applies to all sections; otherwise exact section match
    section: Mapped[str | None] = mapped_column(String(64), nullable=True)
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
        back_populates="period", passive_deletes=True
    )


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ongoing', 'completed')", name="ck_sessions_status"
        ),
        Index("ix_sessions_student_date", "student_id", "date"),
        Index("ix_sessions_date_status", "date", "status"),
        Index("ix_sessions_period_id", "period_id"),
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
    # Per-period attendance: nullable so old rows stay valid
    period_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("periods.id", ondelete="SET NULL"), nullable=True, index=True
    )
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
    period: Mapped[Period | None] = relationship(back_populates="sessions")


class Alert(Base):
    """System alerts for deviations and anomalies (unknown faces, low confidence, etc.)."""

    __tablename__ = "alerts"
    __table_args__ = (
        CheckConstraint(
            "severity IN ('low', 'medium', 'high')", name="ck_alerts_severity"
        ),
        Index("ix_alerts_type", "type"),
        Index("ix_alerts_created_at", "created_at"),
        Index("ix_alerts_acknowledged", "acknowledged_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="medium")
    # Arbitrary JSON payload: e.g. {camera_id, student_id, period_id, confidence}
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, server_default="{}")
    camera_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    student_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("students.id", ondelete="SET NULL"), nullable=True)
    period_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("periods.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class User(Base):
    """Console account (multi-user auth).

    Roles
    -----
    ``super_admin``  the ONLY role allowed to set/change a username or
                     password, approve/reject accounts and resolve
                     forgotten-password requests.
    ``admin``        regular operator: reads its own profile, never edits
                     credentials (its own included).

    Statuses
    --------
    ``pending``   registered, waiting for super-admin review (cannot log in)
    ``active``    approved by the super admin (may log in)
    ``rejected``  refused by the super admin
    ``suspended`` approved once, later disabled by the super admin
    """

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('super_admin', 'admin')", name="ck_users_role"),
        CheckConstraint(
            "status IN ('pending', 'active', 'rejected', 'suspended')",
            name="ck_users_status",
        ),
        CheckConstraint(
            "security_level >= 0 AND security_level <= 5",
            name="ck_users_security_level",
        ),
        CheckConstraint("length(trim(username)) >= 3", name="ck_users_username_len"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(
        String(100), nullable=False, unique=True, index=True
    )
    # PBKDF2-SHA256 encoded string, see app.users.hash_password.
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    # 'env'    -> mirrors ADMIN_PASSWORD (rotation of .env keeps unlocking it)
    # 'manual' -> set by the super admin; .env no longer overrides it
    password_source: Mapped[str] = mapped_column(
        String(8), nullable=False, default="env", server_default="env"
    )
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="admin")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending"
    )
    # 0 = not reviewed yet; 1 (low) .. 5 (highest) assigned on approval.
    security_level: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    approved_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PasswordResetRequest(Base):
    """Forgotten-password request, routed to the super admin.

    Deliberately stores NO new password: only the super admin sets one, through
    ``POST /auth/admin/password-requests/{id}``.
    """

    __tablename__ = "password_reset_requests"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'resolved', 'rejected')",
            name="ck_reset_requests_status",
        ),
        Index("ix_reset_requests_username", "username"),
        Index("ix_reset_requests_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolved_by: Mapped[str | None] = mapped_column(String(100), nullable=True)


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
