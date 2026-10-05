"""Application configuration.

All settings come from environment variables (or a local `.env` file).
Field names are case-insensitive in env vars, e.g. ``RECOGNITION_THRESHOLD``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
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
    # Single threshold used EVERYWHERE (API, pipeline, demo, tests).  Validated
    # 2026-10-04 from printed cosine scores (eval/enroll_and_calibrate.py +
    # eval/diag_threshold.py, buffalo_l CPU, 6 enrolled students):
    #   different-person max = 0.1246 | same-person photo min = 0.9111
    #   genuine video-face min vs gallery = 0.5030 (median 0.9553)
    #   -> 0.40 sits inside (0.1246, 0.5030): 0.275 above the worst impostor,
    #      0.103 below the weakest genuine video match.
    recognition_threshold: float = 0.40  # cosine similarity threshold
    # Best-vs-second-best margin (duplicate identities excluded, see matching).
    # Runner-up over DIFFERENT identities never exceeded 0.1246 while genuine
    # bests are >= 0.5030 (gap 0.378); 0.10 is conservative inside that gap.
    recognition_margin: float = 0.10
    # Two students whose galleries are >= 0.90 similar are the same human
    # (measured: duplicated photo folders score 0.9998) -> excluded from the
    # runner-up so the margin rule does not reject genuine matches.
    duplicate_identity_sim: float = 0.90
    dedup_window_seconds: int = 120  # same student + camera within window => one event
    dedup_log_all: bool = False  # still write RecognitionLog rows on dedup hits
    log_unknown_faces: bool = True  # log unknown faces for alerting (audit)
    min_reference_photos: int = 3
    max_reference_photos: int = 5
    # quality gate for occlusion/blur/low-light (Req 5)
    face_quality_enabled: bool = True
    face_min_sharpness: float = 30.0  # Laplacian variance
    face_min_brightness: float = 30.0
    face_max_brightness: float = 230.0
    face_min_area: int = 800  # bbox area in pixels
    face_min_width: int = 80  # minimum face width in pixels
    face_max_yaw: float = 30.0  # max absolute yaw (degrees)
    face_max_pitch: float = 30.0  # max absolute pitch (degrees)
    face_max_roll: float = 30.0  # max absolute roll (degrees)
    face_min_det_score: float = 0.6  # minimum detection confidence

    # ---------------------------------------------------------- encryption
    # base64 url-safe 32-byte key (Fernet) used to encrypt embeddings at rest.
    # Generate: `python scripts/generate_key.py`
    # REQUIRED in production - startup fails hard without it (no JWT fallback).
    embedding_encryption_key: str | None = None

    # -------------------------------------------------------- registration
    registration_no_pattern: str = r"^[A-Za-z0-9_-]{6,12}$"  # alphanumeric + underscore + hyphen, 6-12 chars
    # K-of-N confirmation before state change (Req 5)
    confirmation_frames: int = 2
    confirmation_window_s: float = 5.0

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

    @field_validator("embedding_encryption_key")
    @classmethod
    def _validate_encryption_key(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:  # `EMBEDDING_ENCRYPTION_KEY=` in .env means "unset"
            return None
        import base64
        try:
            decoded = base64.urlsafe_b64decode(v + "=" * (-len(v) % 4))
            if len(decoded) != 32:
                raise ValueError("EMBEDDING_ENCRYPTION_KEY must decode to exactly 32 bytes")
        except Exception as e:
            raise ValueError(f"EMBEDDING_ENCRYPTION_KEY must be valid base64: {e}")
        return v

    @field_validator("jwt_secret")
    @classmethod
    def _validate_jwt_secret_production(cls, v: str) -> str:
        if v == "change-me-in-production" and os.getenv("ENVIRONMENT") == "production":
            raise ValueError("JWT_SECRET must be changed from default in production")
        return v

    @model_validator(mode="after")
    def _require_encryption_key_in_production(self) -> "Settings":
        """Hard-fail at startup when production has no encryption key.

        There is NO fallback (in particular NOT the JWT secret): silently
        deriving the at-rest key from a shared secret previously allowed a
        JWT leak to decrypt every stored face embedding.
        """
        if (
            not self.embedding_encryption_key
            and self.environment.strip().lower() in {"production", "prod"}
        ):
            raise ValueError(
                "EMBEDDING_ENCRYPTION_KEY is required when ENVIRONMENT=production. "
                "Generate one with `python scripts/generate_key.py` and put it in "
                ".env (base64 url-safe, decodes to exactly 32 bytes). "
                "Refusing to start: there is no fallback key."
            )
        return self

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
