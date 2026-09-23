"""Pydantic request/response schemas (OpenAPI models shown in Swagger)."""

from __future__ import annotations

from datetime import date, datetime
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


class OkResponse(BaseModel):
    ok: bool = True
    detail: str | None = None
