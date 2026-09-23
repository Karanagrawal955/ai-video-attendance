"""Application configuration.

All settings come from environment variables (or a local `.env` file).
Field names are case-insensitive in env vars, e.g. ``RECOGNITION_THRESHOLD``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "AI Video Attendance API"
    environment: str = "development"
    log_level: str = "INFO"
    log_format: str = "json"  # json | text
    cors_origins: str = "*"
    data_dir: str = "./data"  # student reference photos live here
    version: str = "1.0.0"

    # -------------------------------------------------------------- database
    database_url: str = (
        "postgresql+psycopg://attendance:attendance@localhost:5432/attendance"
    )

    # ---------------------------------------------------------------- redis
    redis_url: str = "redis://localhost:6379/0"

    # ----------------------------------------------------------------- auth
    auth_required: bool = True
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720
    admin_username: str = "admin"
    admin_password: str = "admin"

    # ----------------------------------------------------------- face model
    face_model_name: str = "buffalo_l"
    insightface_root: str = "~/.insightface"
    cuda_device_index: int = 0
    force_cpu: bool = False  # skip CUDA entirely (for debugging)
    det_sizes: str = "640x640"  # comma separated WxH, e.g. "640x640" or "128x128,640x640"
    det_thresh: float = 0.5
    max_faces_per_frame: int = 20
    max_embedding_batch: int = 64  # faces per GPU call when batching embeddings
    enable_batched_detection: bool = True  # cross-frame batched SCRFD (falls back automatically)

    # ---------------------------------------------------------- recognition
    recognition_threshold: float = 0.40  # cosine similarity threshold
    dedup_window_seconds: int = 120  # same student + camera within window => one event
    dedup_log_all: bool = False  # still write RecognitionLog rows on dedup hits
    log_unknown_faces: bool = False
    min_reference_photos: int = 3
    max_reference_photos: int = 5

    # ------------------------------------------------- shared inference RPC
    inference_mode: str = "redis"  # redis (shared GPU service) | inprocess (worker-local)
    inference_queue_key: str = "infer:requests"
    inference_response_ttl_s: int = 30
    inference_request_timeout_s: float = 15.0
    max_batch_frames: int = 32  # max frames per engine call (across all cameras)
    inference_batch_wait_ms: int = 25  # max time to wait while filling a GPU batch

    # -------------------------------------------------------------- pipeline
    default_sampling_rate: int = 5  # process every Nth decoded frame
    frame_queue_size: int = 8  # bounded decode queue (drop-oldest)
    batch_size: int = 8  # frames per RPC sent by one camera worker
    batch_flush_ms: int = 120  # max time to fill a camera-side batch
    jpeg_quality: int = 85
    rtsp_transport: str = "tcp"
    rtsp_open_timeout_ms: int = 10_000
    rtsp_read_timeout_ms: int = 10_000
    reconnect_initial_delay_s: float = 2.0
    reconnect_max_delay_s: float = 30.0
    camera_slots: int = 8  # dedicated celery worker slots for camera streams
    heartbeat_ttl_s: int = 30  # pipeline heartbeat TTL (unhealthy if expired)

    # ---------------------------------------------------------------- time
    local_timezone: str = "UTC"

    # ---------------------------------------------------------------- seed
    seed_video_path: str = "./samples/videos/lecture.mp4"

    # ---------------------------------------------------------- redis keys
    events_channel: str = "events:live"
    events_recent_key: str = "events:recent"
    events_recent_limit: int = 200
    gpu_status_key: str = "gpu:status"
    students_version_key: str = "students:version"
    running_cameras_key: str = "cameras:running"

    # ------------------------------------------------------------ validators
    @field_validator("det_sizes")
    @classmethod
    def _validate_det_sizes(cls, v: str) -> str:
        for part in v.split(","):
            part = part.strip().lower()
            if not part:
                continue
            w, _, h = part.partition("x")
            if not (w.isdigit() and h.isdigit()):
                raise ValueError(f"det_sizes must look like '640x640', got {v!r}")
        return v

    # ------------------------------------------------------------- helpers
    @property
    def cors_origins_list(self) -> list[str]:
        value = self.cors_origins.strip()
        if value == "*":
            return ["*"]
        return [o.strip() for o in value.split(",") if o.strip()]

    @property
    def photos_root(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def det_size_list(self) -> list[tuple[int, int]]:
        sizes: list[tuple[int, int]] = []
        for part in self.det_sizes.split(","):
            part = part.strip().lower()
            if not part:
                continue
            w, _, h = part.partition("x")
            sizes.append((int(w), int(h)))
        return sizes or [(640, 640)]

    @property
    def slot_queue_prefix(self) -> str:
        return "camera_slot_"

    def slot_queue(self, slot: int) -> str:
        return f"{self.slot_queue_prefix}{slot}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
