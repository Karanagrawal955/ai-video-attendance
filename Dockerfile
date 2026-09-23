# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# AI Video Attendance backend - single image, three roles:
#   api        -> uvicorn app.main:app
#   worker     -> bash scripts/run_worker.sh  (Celery camera slots)
#   inference  -> python -m app.inference.server (the ONLY model owner)
#
# Base image is CUDA 12.8.1 + cuDNN 9 (Blackwell sm_120 / RTX 50-series
# support with driver >= 570).  Ubuntu 24.04 provides Python 3.12 (>= 3.11).
# Override with --build-arg CUDA_BASE=nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
# if your onnxruntime-gpu build needs a newer CUDA minor.
# ---------------------------------------------------------------------------
ARG CUDA_BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_BASE}

# onnxruntime-gpu 1.26.0 = newest build targeting CUDA 12 + cuDNN 9.
# (1.27+ targets CUDA 13 and requires driver >= 580 - see README.)
ARG ORT_GPU_VERSION=1.26.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    OPENCV_LOG_LEVEL=ERROR \
    INSIGHTFACE_ROOT=/models \
    DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        ffmpeg libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 \
        curl ca-certificates tini bash \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip

WORKDIR /app

# Dependencies layer (cached unless requirements change).
COPY requirements.txt ./
# NOTE: the uninstall is parenthesized so a FAILED requirements install still
# aborts the build instead of being swallowed by `|| true`.
RUN pip install -r requirements.txt \
    # insightface pulls the CPU `onnxruntime` and GUI `opencv-python`;
    # swap them for the GPU + headless wheels (official InsightFace recipe):
    && (pip uninstall -y onnxruntime opencv-python || true) \
    && pip install "onnxruntime-gpu==${ORT_GPU_VERSION}" "opencv-python-headless" \
    && python -c "import onnxruntime as ort; print('ORT', ort.__version__, ort.get_available_providers())"

COPY . .
RUN chmod +x scripts/*.sh 2>/dev/null || true \
    && mkdir -p /data/students /models /samples/videos /samples/faces

EXPOSE 8000

ENTRYPOINT ["tini", "--"]
# Default = API; compose overrides for worker/inference roles.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
