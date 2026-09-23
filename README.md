# AI Video Attendance System (Backend Only)

FastAPI + WebSocket backend that marks student attendance from CCTV/video
using face recognition on an NVIDIA GPU. **No frontend is included** - build
your own against the REST/WS API documented at `/docs`.

**Stack**: FastAPI · InsightFace `buffalo_l` (SCRFD + ArcFace, ONNX Runtime
GPU) · OpenCV (RTSP/files) · PostgreSQL + Alembic · Celery + Redis · Docker
(GPU passthrough).

---

## 1. Architecture

```
                        ┌──────────────────────────────────────────────┐
  RTSP / mp4 files ──►  │ camera pipeline (Celery, one slot per camera)│
  CPU: decode, sample,  │  frame reader ─► sample every Nth ─► JPEG    │
  JPEG, queues          │  bounded queue ─► batch of K frames          │
                        └───────────────┬──────────────────────────────┘
                                        │ Redis RPC (LPUSH infer:requests)
                                        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  inference service  (python -m app.inference.server) - THE ONLY    │
   │  GPU owner: one InsightFace model instance, coalesces frames from  │
   │  ALL cameras into batched SCRFD detection + batched ArcFace        │
   │  embeddings; publishes GPU stats to redis every 2 s                │
   └───────────────┬────────────────────────────────────────────────────┘
                   │ embeddings + per-stage timings
                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  camera pipeline continued: cosine matching (top-64 scan) ─► dedup │
   │  (2 min/camera/student) ─► entry/exit session logic ─► PostgreSQL  │
   │  ─► RecognitionLog (gpu ms) ─► pub/sub events                       │
   └───────────────┬────────────────────────────────────────────────────┘
                   ▼
        api (FastAPI): REST + /ws/live-feed WebSocket
```

Four processes, three roles (one image):

| Service      | Command                                | Owns GPU? |
|--------------|----------------------------------------|-----------|
| `api`        | `uvicorn app.main:app`                 | no        |
| `worker`     | `scripts/run_worker.sh` (Celery)       | optional* |
| `inference`  | `python -m app.inference.server`       | **yes**   |
| `postgres` / `redis` | stock images                    | -         |

\* The worker gets a GPU reservation only so `INFERENCE_MODE=inprocess`
(model loaded inside the worker) works for debugging; the default `redis`
mode keeps **exactly one model instance** in the `inference` service.

**CPU vs GPU split** (measured on i7-14th-gen HX + RTX 4050 Laptop; the
RTX 5060 target spec is strictly faster - see §9 Benchmarks):

- **CPU**: video decode, frame sampling, JPEG encode, queues, matching,
  session logic, DB, HTTP.
- **GPU**: SCRFD detection + ArcFace 512-d embeddings only, batched -
  N frames × M faces per `session.run` call. Detection batching uses a
  startup "canary" probe (dynamic-batch graph + rank-3 outputs required);
  if the exported ONNX is static-batch, it silently falls back to per-frame
  detection while still batching embeddings across frames. Live state:
  `GET /system/gpu-status` → `batched_detection`.

**Processes never block each other**: every REST handler is synchronous
(FastAPI threadpool), only the WebSocket bridge is async, and camera tasks
each occupy a dedicated Celery queue (`camera_slot_0..N-1`, `prefetch=1`).

---

## 2. Requirements

- **NVIDIA driver ≥ 570** (CUDA 12.8 baseline needed by RTX 50-series /
  Blackwell `sm_120`; RTX 5060 works out of the box with current drivers).
- **Docker + Docker Compose v2** with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Verify: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.
- ~6 GB disk for the CUDA image + `buffalo_l` model (~280 MB, auto-downloaded
  on first start into the `insightface_models` volume).
- No local CUDA toolkit/cuDNN install needed - the base image provides
  CUDA 12.8 + cuDNN 9; ONNX Runtime ships its own CUDA kernels.

**CPU-only fallback**: with no GPU visible the inference service starts with
`CPUExecutionProvider` (health shows `provider=CPUExecutionProvider`) -
functional but ~10-20× slower; see [Benchmarks](#9-benchmarks-gpu-vs-cpu).

### GPU/CUDA compatibility matrix

| onnxruntime-gpu | CUDA | cuDNN | Min driver | Works on RTX 5060? |
|---|---|---|---|---|
| **1.26.0 (default)** | 12.x | 9.x | ≥ 570 | **Yes** (sm_120 in CUDA 12.8) |
| 1.27+ | 13.x | 9.x | ≥ 580 | Yes, but needs CUDA 13 base image |
| ≤ 1.19 | 12.x | 8.x | ≥ 525 | **No** (predates Blackwell) |

---

## 3. Quick start (Docker)

```bash
cp .env.example .env          # then EDIT JWT_SECRET + ADMIN_PASSWORD!
docker compose up -d --build  # GPU is passed through automatically
docker compose ps             # wait until api is healthy
curl http://localhost:8000/system/health
open http://localhost:8000/docs
```

First boot pulls the CUDA image (~3 GB), builds the venv, runs Alembic
migrations, and downloads `buffalo_l` into the model volume (30-60 s).

Check the GPU path:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -s http://localhost:8000/system/gpu-status -H "Authorization: Bearer $TOKEN" | python -m json.tool
# expect: "service": "online", "provider": "CUDAExecutionProvider",
#         "device_name": "... RTX 4050 ...", "batched_detection": false
#
# Note: stock buffalo_l's detector graph has a static batch-1 input, so
# detection runs per-frame (batched_detection=false is EXPECTED); embeddings
# still batch across frames either way.
```

### Without Docker (local dev)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./scripts/setup_gpu.sh        # or setup_gpu.ps1 on Windows: swaps to
                              # onnxruntime-gpu, installs the CUDA 12 libraries
                              # ORT needs via pip (~1.5 GB; SKIP_NVIDIA_WHEELS=1
                              # to use a system CUDA/cuDNN install) and verifies
                              # a REAL CUDA session (not just the provider list)
cp .env.example .env          # set DATABASE_URL/REDIS_URL to your local services
alembic upgrade head
uvicorn app.main:app --reload                       # terminal 1
python -m app.inference.server                      # terminal 2 (GPU)
bash scripts/run_worker.sh                          # terminal 3 (Celery)
python scripts/seed.py --list                       # sanity check
pytest                                               # no services needed
```

---

## 4. Adding a camera

```bash
# Real RTSP camera
curl -X POST http://localhost:8000/cameras \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "name": "Main Gate",
    "location": "Main entrance",
    "type": "entry",
    "rtsp_url": "rtsp://user:pass@192.168.1.20:554/stream1",
    "sampling_rate": 5
  }'

# Mock camera from a local video file (loops forever - see samples/README.md)
docker compose exec api python scripts/seed.py \
  --camera --name "Main Gate" --type entry \
  --video /samples/videos/lecture.mp4 --location "Main entrance"
```

- `type`: `entry` opens sessions, `exit` closes them, `both` does
  exit-if-open-else-entry (turnstiles).
- `sampling_rate`: process every Nth frame (5 → ~6 FPS from a 30 FPS stream).
- Cameras do **not** auto-start. Start one:

```bash
curl -X POST http://localhost:8000/cameras/1/start -H "Authorization: Bearer $TOKEN"
# {"camera_id":1,"state":"starting","slot":0,"task_id":"..."}

curl -X POST http://localhost:8000/cameras/1/stop -H "Authorization: Bearer $TOKEN"  # graceful
curl -X POST "http://localhost:8000/cameras/1/stop?force=true" -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/cameras -H "Authorization: Bearer $TOKEN"   # runtime.state / heartbeat
```

Concurrency = `CAMERA_SLOTS` (default 8). Starting a 9th camera returns
`409 all 8 camera slots are busy`.

---

## 5. Enrolling students

3-5 reference photos per student (front/side/angle mix works best).

```bash
curl -X POST http://localhost:8000/students \
  -H "Authorization: Bearer $TOKEN" \
  -F name="Ada Lovelace" \
  -F registration_no="21CSE001" \
  -F section="CSE-A" \
  -F photos=@photos/ada1.jpg \
  -F photos=@photos/ada2.jpg \
  -F photos=@photos/ada3.jpg
```

Or bulk from folders (`samples/faces/<registration_no>/*.jpg`):

```bash
docker compose exec api python scripts/seed.py --enroll-dir /samples/faces
```

Face-data management: `POST /students/{id}/photos` (add),
`PUT /students/{id}/face` (replace), `DELETE /students/{id}/face` (clear),
reference images are served at `/media/students/<id>/photo_*.jpg`.
Enrollment runs the same GPU service as live recognition (batched - all
photos in one call) and returns `503` with a clear message if the
`inference` service is down.

---

## 6. Attendance rules

- **entry** → opens a session if none is open *today*; the session's `date`
  is the local date of entry (midnight rollover keeps the same session
  open - exit at 00:10 closes the 23:50 session, duration 20 min, date
  stays the entry date).
- **exit** → closes the open session (`total_duration` seconds).
- **`both` camera** → closes if a session is open, else opens one.
- Stale session from a previous day still open (student forgot to exit) →
  auto-closed on the next entry; a new session starts.
- Multiple entry/exit cycles per day create multiple sessions; summaries
  total them.
- **Dedup**: the same student + camera within `DEDUP_WINDOW_SECONDS`
  (120 s, per camera) produces one event - gates session transitions,
  `RecognitionLog` rows and WebSocket events alike (so a 25-frame burst
  at the gate = 1 entry, 1 log row, 1 WS message). `DEDUP_LOG_ALL=true`
  still writes audit logs during the window.
- Matching: cosine similarity over **all** stored per-photo embeddings,
  best match wins; accept ≥ `RECOGNITION_THRESHOLD` (default 0.40).

```bash
# Daily summary (per-student entries/exits/time present)
curl "http://localhost:8000/attendance/summary?date=2026-09-23" -H "Authorization: Bearer $TOKEN"
# One student's day
curl "http://localhost:8000/attendance/21?date=2026-09-23" -H "Authorization: Bearer $TOKEN"
# Currently inside (ongoing sessions, counted to now)
curl http://localhost:8000/attendance/live -H "Authorization: Bearer $TOKEN"
# Raw audit log incl. GPU latency per recognition
curl http://localhost:8000/attendance/logs -H "Authorization: Bearer $TOKEN"
```

---

## 7. API reference

Interactive docs: **`http://localhost:8000/docs`** (Swagger) ·
`/redoc` · `/openapi.json`.

Auth: `POST /auth/token` `{"username","password"}` → `{"access_token"}` →
send `Authorization: Bearer <token>` on every request (and
`?token=<jwt>` for the WebSocket). Set `AUTH_REQUIRED=false` to disable
while prototyping your frontend.

| Method & path | Purpose |
|---|---|
| `POST /auth/token` | JWT for admin/staff |
| `GET/POST /students`, `GET/PUT/DELETE /students/{id}` | enrollment CRUD |
| `POST /students/{id}/photos`, `PUT /students/{id}/face`, `DELETE /students/{id}/face` | face data |
| `GET/POST /cameras`, `GET/PUT/DELETE /cameras/{id}` | camera registry |
| `POST /cameras/{id}/start`, `POST /cameras/{id}/stop[?force=true]` | pipeline control |
| `GET /attendance/summary[?date]`, `/attendance/{student_id}[?date]` | totals |
| `GET /attendance/live`, `GET /attendance/logs` | live view / audit |
| `GET /system/health`, `GET /system/gpu-status` | health + GPU telemetry |
| `WS /ws/live-feed` | real-time recognition events |

### WebSocket `ws://localhost:8000/ws/live-feed?token=...`

On connect you first receive the **recent backlog** (last 100 events), then
live events from Redis pub/sub. Message shapes (Pydantic schemas in
`app/schemas.py`):

```jsonc
{"type":"recognition","student":{"id":1,"name":"Ada Lovelace","registration_no":"21CSE001"},
 "camera":{"id":1,"name":"Main Gate","type":"entry"},"confidence":0.71,
 "gpu_ms":6.42,"bbox":[183,96,301,238],"ts":"..."}
{"type":"attendance","action":"entry_created","session_id":7,"student":{...},"ts":"..."}
{"type":"attendance","action":"exit_completed","duration_s":28800,"ts":"..."}
{"type":"camera_state","camera_id":1,"state":"running|unhealthy|stopped","ts":"..."}
{"type":"ping"} / {"type":"pong"}
```

All events are dedup-gated like DB writes; reconnects replay the backlog so
your UI never misses transitions.

---

## 8. Configuration (`.env`)

Full annotated list: [`.env.example`](.env.example). Highlights:

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | local postgres | SQLAlchemy URL (compose overrides) |
| `AUTH_REQUIRED` / `JWT_SECRET` / `ADMIN_*` | on / … / admin | auth |
| `FACE_MODEL_NAME` | `buffalo_l` | insightface model pack |
| `DET_SIZES` / `DET_THRESH` | `640x640` / `0.5` | SCRFD input; add `128x128` for small/distant faces |
| `RECOGNITION_THRESHOLD` | `0.40` | cosine accept threshold |
| `DEDUP_WINDOW_SECONDS` | `120` | per-camera dedup window |
| `CAMERA_SLOTS` | `8` | concurrent cameras (worker queues) |
| `INFERENCE_MODE` | `redis` | shared GPU service vs in-process |
| `MAX_BATCH_FRAMES` / `INFERENCE_BATCH_WAIT_MS` | `32` / `25` | cross-camera GPU batching |
| `FRAME_QUEUE_SIZE` / `BATCH_SIZE` | `8` / `8` | camera-side pipeline bounds |
| `LOCAL_TIMEZONE` | `Asia/Kolkata` | which calendar day counts as "today" |
| `FORCE_CPU` | `false` | ignore the GPU entirely |

Structured logging: `LOG_FORMAT=json` emits one JSON object per line with
inference timings (`det_ms`, `emb_ms`, `total_ms`, `batch`, `faces`, queue
depth) - grep them with `jq`:

```bash
docker compose logs -f inference | jq -c 'select(.batch) | {t:.asctime, det_ms, emb_ms, total_ms, batch, faces}'
```

---

## 9. Benchmarks (GPU vs CPU)

Run the benchmark **on your own machine** for real numbers:

```bash
python scripts/benchmark.py              # GPU (CUDA if available)
python scripts/benchmark.py --cpu        # CPU baseline, same model
```

Expected shape of results (estimates - actual numbers depend on resolution,
face count, batch size and driver; measure before quoting):

| Workload | RTX 5060 (CUDA, batched) | CPU-only (i7-14th-gen HX) |
|---|---|---|
| Detection, 640², 1 frame | ~4-8 ms | ~60-150 ms |
| Detection, batch of 8 frames | ~12-25 ms total (~2-3 ms/frame) | ~500-1200 ms total |
| ArcFace embedding, batch of 64 faces | ~4-8 ms total | ~100-300 ms total |
| End-to-end frames/s (pipeline) | ~100-160 fps | ~5-10 fps |
| Latency per camera RPC (batch 8) | ~15-30 ms | ~0.6-1.5 s |

Takeaway: on GPU, inference is a small fraction of the pipeline (decode +
JPEG dominate on CPU), so 8 cameras at `sampling_rate=5` leave the GPU
mostly idle; on CPU the same load saturates cores and queue latency spikes.
`GET /system/gpu-status` shows live EMA latencies, batch fill
(`avg_batch_frames`), VRAM and queue depth while your real streams run.

---

## 10. Development

```bash
pip install -e ".[dev]"
pytest                    # unit + API tests; uses SQLite + fakeredis, no services
python -m compileall app  # syntax check
alembic upgrade head --sql > schema.sql   # offline migration render
```

Layout:

```
app/
  api/            routers: auth, students, cameras, attendance, system, ws
  inference/      engine.py (FaceEngine: batched SCRFD/ArcFace + CPU fallback)
                  server.py (shared GPU service, RPC loop) / client.py (RPC)
  pipeline/       frame_reader.py (RTSP/reconnect/loop), camera_task.py (per-camera Celery task)
  services/       attendance.py, enrollment.py, events.py
  matching.py     EmbeddingIndex (top-k cosine scan)
  models.py       SQLAlchemy 2 models · schemas.py (Pydantic) · config.py (settings)
  workers/        celery_app.py (queues) · tasks.py
alembic/          migrations (0001_initial)
scripts/          seed.py, benchmark.py, wait_db.py, run_worker.sh, setup_gpu.*
tests/            matching, attendance rules, API smoke
```

**Adding an endpoint**: router in `app/api/`, register in `app/main.py`,
schema in `app/schemas.py`. Handlers stay synchronous (threadpool) unless
they must `await`.

**Migrations**: edit `app/models.py`, then
`alembic revision --autogenerate -m "..."` (needs a reachable DB) and
`alembic upgrade head`. Compose runs migrations on API boot.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `gpu-status` → `provider=CPUExecutionProvider` | GPU not visible to the container: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`; check NVIDIA Container Toolkit + driver ≥ 570 |
| `CUDAExecutionProvider` missing, ORT 1.27+ | Needs driver ≥ 580 + CUDA 13 base image → rebuild with `--build-arg CUDA_BASE=nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04` or pin `ORT_GPU_VERSION=1.26.0` |
| `inference` → `service: offline` (stale) | `docker compose logs inference`; model download needs outbound HTTPS on first run; check `redis` healthy |
| `all N camera slots are busy` | `POST /cameras/{id}/stop` a camera or raise `CAMERA_SLOTS` in `.env` (compose recreates the worker) |
| Enrollment → `503 inference service unavailable` | The `inference` service must run before enrolling; `docker compose up -d inference` |
| `409 all ... slots busy` although cameras look stopped | Heartbeats expired/cleared - `GET /cameras` shows `runtime.state`; `?force=true` clears stuck state |
| RTSP `Connection reset` / no frames | `RTSP_TRANSPORT=tcp` (default) is usually required; raise `RTSP_READ_TIMEOUT_MS`; logs show exponential reconnect (2→30 s) |
| No WebSocket events | Pass `?token=` when `AUTH_REQUIRED=true`; ensure something is recognizing (start a camera; backlog replays on connect regardless) |
| Faces never match (score below threshold) | Re-enroll with sharper photos (`PUT /students/{id}/face`), lower `RECOGNITION_THRESHOLD` to 0.35 for trials, ensure `DET_SIZES` includes `128x128` for distant faces |
| Wrong day in summaries | Set `LOCAL_TIMEZONE` to your college timezone |
| Windows local GPU dev | Run `scripts\setup_gpu.ps1` - it installs `onnxruntime-gpu` **plus** the CUDA libraries ORT loads (`nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12`), adds their `bin` dirs to `PATH` for the session, and verifies a real CUDA session. Beware: `get_available_providers()` lists CUDA even when its DLLs are missing - only a real session tells the truth. The inference engine also self-registers these DLL dirs on startup (`PATH` + `os.add_dll_directory`), so fresh shells work without manual PATH edits; without the registration cuDNN fails at the first Conv with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` |

---

## 12. Security notes

- Change `JWT_SECRET`, `ADMIN_PASSWORD`, `POSTGRES_PASSWORD` before any
  real deployment; keep `AUTH_REQUIRED=true` (CORS via `CORS_ORIGINS`).
- RTSP credentials live in `.env`/DB - the API never returns a stored
  password in plaintext responses (URLs are returned as-is; restrict access).
- Face embeddings and reference photos are personal data: `/media` serves
  photos without auth - put the API behind a reverse proxy with TLS and
  access control for production.
