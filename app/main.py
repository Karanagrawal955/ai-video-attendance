"""FastAPI application entrypoint.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Deliberately imports nothing GPU/CV2-related at module scope: the API process
never loads InsightFace (enrollment frames go through the shared inference
service over Redis), so it boots in milliseconds without CUDA.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import attendance, auth, cameras, students, system
from .api.ws import router as ws_router
from .config import settings
from .logging_config import configure as configure_logging

configure_logging(settings)
logger = logging.getLogger("app.main")

DESCRIPTION = """
AI-based video attendance backend (REST + WebSocket).

**Pipeline**: RTSP/CCTV or video file -> CPU frame reader (sampling) ->
Redis RPC -> shared GPU inference service (InsightFace `buffalo_l`, batched
SCRFD detection + batched ArcFace embeddings) -> cosine matching -> attendance
sessions + audit logs -> WebSocket live feed.

Obtain a JWT via `POST /auth/token` and pass it as
`Authorization: Bearer <token>` on every request (unless `AUTH_REQUIRED=false`).
"""


@asynccontextmanager
async def lifespan(_: FastAPI):
    photos_root = settings.photos_root
    (photos_root / "students").mkdir(parents=True, exist_ok=True)
    logger.info(
        "api starting",
        extra={
            "version": __version__,
            "auth_required": settings.auth_required,
            "inference_mode": settings.inference_mode,
            "data_dir": str(photos_root),
        },
    )
    yield
    logger.info("api shutting down")


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "auth", "description": "JWT token issuance"},
        {"name": "students", "description": "Enrollment & face data management"},
        {"name": "cameras", "description": "Camera registry + pipeline control"},
        {"name": "attendance", "description": "Sessions, summaries, live view, audit log"},
        {"name": "system", "description": "Health & GPU telemetry"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,  # we use Bearer headers, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(students.router)
app.include_router(cameras.router)
app.include_router(attendance.router)
app.include_router(system.router)
app.include_router(ws_router)


@app.get("/", tags=["system"], summary="Service metadata")
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "health": "/system/health",
        "gpu_status": "/system/gpu-status",
        "websocket": "/ws/live-feed",
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled error on %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


# Serve enrolled reference photos at /media/<relative_path>.
photos_root = settings.photos_root
photos_root.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(Path(photos_root))), name="media")
