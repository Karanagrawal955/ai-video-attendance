"""Pydantic request/response schemas (OpenAPI models shown in Swagger)."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CameraType = Literal["entry", "exit", "both"]
SessionStatus = Literal["ongoing", "completed"]


# --------------------------------------------------------------------- auth
class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    role: str | None = None  # super_admin | admin


class RegisterRequest(BaseModel):
    """Self-service sign-up: always creates a *pending* admin account."""

    username: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.-]{3,100}$",
        examples=["priya.n"],
    )
    password: str = Field(min_length=8, max_length=200)
    display_name: str | None = Field(default=None, max_length=200)
    # Only `admin` is self-registerable; super_admin is never self-assigned.
    requested_role: Literal["admin"] = "admin"


class RegisteredOut(BaseModel):
    id: int
    username: str
    status: str
    requested_role: str
    detail: str


class ForgotPasswordRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    reason: str | None = Field(default=None, max_length=500)


class ForgotPasswordOut(BaseModel):
    accepted: bool = True
    detail: str


class MeOut(BaseModel):
    id: int | None = None
    username: str
    role: str
    status: str = "active"
    display_name: str | None = None
    security_level: int = 0


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str | None = None
    role: str
    status: str
    security_level: int
    approved_at: datetime | None = None
    approved_by: str | None = None
    created_at: datetime
    updated_at: datetime


class UserUpdate(BaseModel):
    """Super-admin-only credential / role change.  All fields optional."""

    username: str | None = Field(
        default=None, min_length=3, max_length=100, pattern=r"^[A-Za-z0-9_.-]{3,100}$"
    )
    password: str | None = Field(default=None, min_length=8, max_length=200)
    role: Literal["super_admin", "admin"] | None = None
    security_level: int | None = Field(default=None, ge=1, le=5)
    status: Literal["pending", "active", "rejected", "suspended"] | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> "UserUpdate":
        if all(
            getattr(self, f) is None
            for f in ("username", "password", "role", "security_level", "status")
        ):
            raise ValueError("at least one field must be provided")
        return self


class UserApprove(BaseModel):
    security_level: int = Field(default=3, ge=1, le=5)


class PasswordRequestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    reason: str | None = None
    status: str
    created_at: datetime
    resolved_at: datetime | None = None
    resolved_by: str | None = None


class PasswordResolveRequest(BaseModel):
    """Super admin action on a forgotten-password request."""

    action: Literal["reset", "reject"]
    new_password: str | None = Field(default=None, min_length=8, max_length=200)

    @model_validator(mode="after")
    def _password_for_reset(self) -> "PasswordResolveRequest":
        if self.action == "reset" and not self.new_password:
            raise ValueError("new_password is required when action is 'reset'")
        return self


class UserListResponse(BaseModel):
    items: list[UserOut] = Field(default_factory=list)
    total: int = 0


class PasswordRequestListResponse(BaseModel):
    items: list[PasswordRequestOut] = Field(default_factory=list)
    total: int = 0


# ----------------------------------------------------------------- students
class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    registration_no: str
    section: str | None = None
    photo_paths: list[str] = Field(default_factory=list)
    embedding_count: int = 0
    # Only populated when ?include_embeddings=true
    embeddings: list[list[float]] | None = None
    created_at: datetime
    updated_at: datetime


class StudentSummary(BaseModel):
    """Compact student projection used inside attendance/event payloads."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    registration_no: str
    section: str | None = None


class StudentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    registration_no: str | None = Field(default=None, min_length=1, max_length=64)
    section: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _at_least_one(self) -> "StudentUpdate":
        if all(
            getattr(self, f) is None
            for f in ("name", "registration_no", "section")
        ):
            raise ValueError("at least one field must be provided")
        return self


# ------------------------------------------------------------------ cameras
class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    location: str | None = Field(default=None, max_length=200)
    rtsp_url: str | None = Field(default=None, max_length=500)
    file_path: str | None = Field(default=None, max_length=500)
    type: CameraType = "entry"
    sampling_rate: int = Field(
        default=5,
        ge=1,
        le=600,
        description="Process every Nth decoded frame (5 ≈ 6 FPS from a 30 FPS stream)",
    )

    @model_validator(mode="after")
    def _needs_source(self) -> "CameraCreate":
        if not self.rtsp_url and not self.file_path:
            raise ValueError("either rtsp_url or file_path is required")
        return self


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    location: str | None = Field(default=None, max_length=200)
    rtsp_url: str | None = Field(default=None, max_length=500)
    file_path: str | None = Field(default=None, max_length=500)
    type: CameraType | None = None
    sampling_rate: int | None = Field(default=None, ge=1, le=600)


class CameraRuntime(BaseModel):
    state: Literal["stopped", "starting", "running", "stopping", "unhealthy"] = "stopped"
    running: bool = False
    slot: int | None = None
    task_id: str | None = None
    last_heartbeat_at: datetime | None = None


class CameraOut(CameraCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
    runtime: CameraRuntime = Field(default_factory=CameraRuntime)


class StartStopResponse(BaseModel):
    camera_id: int
    state: str
    slot: int | None = None
    task_id: str | None = None
    detail: str | None = None


# --------------------------------------------------------------- attendance
class AttendanceSessionOut(BaseModel):
    id: int
    student_id: int
    camera_in_id: int | None = None
    camera_out_id: int | None = None
    camera_in_name: str | None = None
    camera_out_name: str | None = None
    entry_time: datetime
    exit_time: datetime | None = None
    date: date
    status: SessionStatus
    total_duration: int | None = None  # seconds; None while ongoing
    period_id: int | None = None
    period_name: str | None = None


class SessionTotals(BaseModel):
    entries: int = 0
    exits: int = 0
    total_seconds: int = 0
    first_entry: datetime | None = None
    last_exit: datetime | None = None


class StudentAttendanceOut(BaseModel):
    student: StudentSummary
    date: date
    sessions: list[AttendanceSessionOut] = Field(default_factory=list)
    totals: SessionTotals = Field(default_factory=SessionTotals)


class SummaryRow(BaseModel):
    student_id: int
    name: str
    registration_no: str
    section: str | None = None
    entries: int = 0
    exits: int = 0
    first_entry: datetime | None = None
    last_exit: datetime | None = None
    total_seconds: int = 0
    ongoing: bool = False


class SummaryTotals(BaseModel):
    present_students: int = 0
    ongoing_sessions: int = 0
    completed_sessions: int = 0
    total_seconds: int = 0


class SummaryOut(BaseModel):
    date: date
    timezone: str
    students: list[SummaryRow] = Field(default_factory=list)
    totals: SummaryTotals = Field(default_factory=SummaryTotals)


class LiveSessionOut(BaseModel):
    session_id: int
    student: StudentSummary
    entry_time: datetime
    date: date
    elapsed_seconds: int
    camera_in_id: int | None = None
    camera_in_name: str | None = None


class LiveOut(BaseModel):
    server_time: datetime
    count: int
    sessions: list[LiveSessionOut] = Field(default_factory=list)


class RecognitionLogOut(BaseModel):
    id: int
    student_id: int | None = None
    student_name: str | None = None
    camera_id: int | None = None
    camera_name: str | None = None
    timestamp: datetime
    confidence_score: float | None = None
    gpu_inference_time_ms: float | None = None


# ------------------------------------------------------------------ periods
class PeriodCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    start_time: time
    end_time: time
    days_bitmask: int = Field(default=127, ge=0, le=127, description="Bitmask Mon=1<<0 ... Sun=1<<6; 127=all days")
    section: str | None = Field(default=None, max_length=64, description="NULL = all sections")

    @model_validator(mode="after")
    def _check_range(self) -> "PeriodCreate":
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")
        return self


class PeriodUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    start_time: time | None = None
    end_time: time | None = None
    days_bitmask: int | None = Field(default=None, ge=0, le=127)
    section: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _at_least_one(self) -> "PeriodUpdate":
        if all(getattr(self, f) is None for f in ("name", "start_time", "end_time", "days_bitmask", "section")):
            raise ValueError("at least one field must be provided")
        return self

    @model_validator(mode="after")
    def _check_range_update(self) -> "PeriodUpdate":
        if self.start_time is not None and self.end_time is not None and self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")
        return self


class PeriodOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    start_time: time
    end_time: time
    days_bitmask: int
    section: str | None = None
    created_at: datetime
    updated_at: datetime


# ------------------------------------------------------------------ system
class HealthCheck(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    checks: dict[str, str] = Field(default_factory=dict)


class GpuStatusOut(BaseModel):
    service: Literal["online", "offline"] = "offline"
    updated_at: datetime | None = None
    provider: str | None = None
    providers: list[str] = Field(default_factory=list)
    model: str | None = None
    batched_detection: bool | None = None
    device_name: str | None = None
    utilization_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    queue_length: int = 0
    queue_key: str | None = None
    inference_mode: str = "redis"
    throughput: dict = Field(default_factory=dict)
    cameras_running: int = 0
    detail: str | None = None


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    type: str
    severity: str
    payload: dict = Field(default_factory=dict)
    camera_id: int | None = None
    student_id: int | None = None
    period_id: int | None = None
    created_at: datetime
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None


class OkResponse(BaseModel):
    ok: bool = True
    detail: str | None = None
