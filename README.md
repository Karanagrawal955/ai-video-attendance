# AI Video Attendance System

Backend for camera-based attendance: detect faces in a video/RTSP stream, match
them against an enrolled gallery, and record attendance periods.

> **Honesty note on accuracy.** The demo clip in this repository is *synthetic* —
> it is built from the enrolled photos by `scripts/make_test_video.py`. It shows
> **the flow**, not real-CCTV accuracy. Real-camera accuracy has **not been
> measured** yet; see [Known Limitations](#known-limitations) and
> [`eval/ROADMAP.md`](eval/ROADMAP.md).

---

## 1. Capabilities — honest status

| # | Capability | Status | Evidence |
|---|---|---|---|
| 1 | **Enrollment** — CSV bulk + API, per-photo quality gates (size/blur/pose/no-face/two-faces), duplicate registration and duplicate-identity rejection, encrypt-at-rest | **BUILT** | `tests/test_quality.py` (18 tests) |
| 2 | **Recognition & attendance** — detect → embed → match with a single threshold, margin rule, dedup window, period sessions, summaries | **BUILT** | `tests/test_regression.py`, `tests/test_attendance.py`, `tests/test_matching.py` |
| 3 | **Real-time multi-camera pipeline** — RTSP/file readers, bounded queues, per-camera Celery slots, shared GPU inference service with cross-camera batching | **PARTIAL** | Code complete and unit-tested; **never validated against a live RTSP camera** in this environment |
| 4 | **Deployment & operations** — one-command setup, Docker Compose (api/worker/redis/postgres/inference), JWT auth, encryption key hard-fail in production | **PARTIAL** | Setup + tests verified locally; the **GPU Compose stack was not run here** |
| 5 | **Operations workflow** — alerts/acknowledge, review queue, attendance disputes, RBAC, data retention | **ROADMAP** | Alerts are built; the rest is specified in `eval/ROADMAP.md` |

**Unmeasured:** recognition accuracy on real camera footage of enrolled people.
This is roadmap item #1 and is the single most important open question.

---

## 2. Architecture (text diagram)

```
                         ┌──────────────────────────────────────────┐
   RTSP / video file ──► │  frame_reader (bounded, drop-oldest)     │
                         └───────────────┬──────────────────────────┘
                                         │ sample every Nth frame
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │  camera_task  (Celery, one slot/camera)  │
                         │  JPEG-compress → batch → Redis RPC       │
                         └───────────────┬──────────────────────────┘
                                         │  infer:requests
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │  inference service  (ONE model owner)    │
                         │  SCRFD detect → quality gate → ArcFace   │
                         │  embed  (cross-camera batch, single GPU) │
                         └───────────────┬──────────────────────────┘
                                         │ 512-d L2-normalised vectors
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │  matching.EmbeddingIndex                 │
                         │  threshold 0.40 → margin 0.10 →          │
                         │  duplicate-identity skip (0.90)          │
                         └───────────────┬──────────────────────────┘
                                         │ match / unknown
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │  services.attendance                     │
                         │  dedup window → RecognitionLog →         │
                         │  AttendanceSession → period summary      │
                         └───────────────┬──────────────────────────┘
                                         ▼
                             REST API  ·  WebSocket  ·  reports
```

A much deeper version — module map, per-stage data flow, and the derivation of
every threshold — is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 3. Quick start (3 commands)

**Windows**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1
.venv\Scripts\python.exe scripts\run_demo.py --video C:\path\to\your-video.mp4
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

**Linux / macOS**

```bash
bash scripts/setup_dev.sh
.venv/bin/python scripts/run_demo.py --video /path/to/your-video.mp4
.venv/bin/python -m uvicorn app.main:app --reload
```

`setup_dev.ps1` / `setup_dev.sh` does all seven steps and is idempotent:
Python ≥3.11 check → venv → **pinned** install (`requirements.txt` +
`requirements-dev.txt`) → copy `.env.example` → `.env` + generate
`EMBEDDING_ENCRYPTION_KEY` → `alembic upgrade head` → download model weights →
run the test suite.

No Docker and no Redis are needed for that path — see
[Demo without Docker or Redis](#demo-without-docker-or-redis).

Prefer `make`? `make setup`, `make test`, `make demo`, `make run-api`,
`make enroll CSV=students.csv`, `make calibrate`.

---

## 4. Enrollment

### CSV + photo folder layout

`students.csv` (header required, order irrelevant):

```csv
name,registration_no,section,photo_path_or_folder
Aarav Sharma,STU100001,CSE-A,students/STU100001
Meera Iyer,STU100002,CSE-B,students/STU100002
```

`photo_path_or_folder` is **relative to `DATA_DIR`** (default `./data`):

```
data/
  students/
    STU100001/
      photo_0.jpg      # 3..5 photos, one face per photo
      photo_1.jpg
      photo_2.jpg
    STU100002/
      photo_0.jpg
      ...
```

Each row must resolve to **3–5 photos** (`MIN_REFERENCE_PHOTOS` /
`MAX_REFERENCE_PHOTOS`). Rules enforced on every photo:

* exactly one face (0 → `no face detected`, 2+ → `multiple faces detected`)
* face width ≥ 80 px and area ≥ 800 px²
* sharpness (Laplacian variance) ≥ 30, brightness within bounds
* detection score ≥ 0.60, yaw/pitch/roll within limits

Rejected rows are reported, never silently dropped.

```bash
# CLI
python scripts/bulk_enroll.py --csv students.csv --create-tables --report import_report.csv

# dry run (validate only)
python scripts/bulk_enroll.py --csv students.csv --dry-run

# HTTP
curl -F csv_file=@students.csv -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/students/bulk-import
```

### Single student

```bash
python scripts/seed.py --enroll data/students/STU100001 \
       --name "Aarav Sharma" --registration-no STU100001 --section CSE-A
```

### Duplicate protection

* a registration number already present → **409**
* a registration number that does not match `REGISTRATION_NO_PATTERN` → **400**
* a face that already belongs to another student (cosine ≥
  `DUPLICATE_IDENTITY_SIM`) → **409**

---

## 5. Demo

```bash
python scripts/run_demo.py --video /path/to/video.mp4
```

Runs the whole pipeline **in one process** — no Redis, no workers, CPU
inference, SQLite — and writes:

* `eval/demo_report.csv`, `eval/demo_report.html`
* `eval/demo_frames/*.jpg` — annotated frames

Both reports carry a `clip_label` column / banner stating exactly what the clip
is:

* synthetic clip → *"synthetic clip built from enrolled photos
  (scripts/make_test_video.py): shows the flow, not real-CCTV accuracy"*
* your own footage → *"real camera footage (`<file>`) - accuracy not yet
  calibrated"*

The attendance period window is set to the **clip length**, and the late/early
margins are 10 % of that window. Student identifiers are printed only as
`P1…Pn` plus the last 3 digits of the registration number.

**Gallery note:** the photo folders shipped for pipeline development contain
*placeholder* images, so `eval/enroll_and_calibrate.py --sample` enrols them
under synthetic ids (`SAMPLE001`…). Real registration numbers are never bound
to placeholder photos. For a real gallery, put real photos in
`data/students/<reg>/`, write `data/roster.local.csv`, and run without
`--sample` (see [`data/README.md`](data/README.md)).

---

## 6. Tests

```bash
python -m pytest                      # 69 tests (addopts already supplies -q)
python -m pytest --collect-only -q -o addopts= | tail -3
python -m pytest tests/test_quality.py   # the 9 required quality cases
```

> `-q` is already in `pyproject.toml` `addopts`. Passing a second one (`-q -q`
> = `-qq`) silently hides the `N passed` summary line — that is why the
> commands above use a single quiet level.

**In this archive:** `tests/test_pipeline_e2e.py` needs
`tests/assets/test_multi.mp4`, which is deliberately **not shipped** (the
archive must contain no media). That one test therefore reports
`68 passed, 1 skipped` here, and `69 passed` once you build the clip with
`scripts/make_test_video.py`. The skip is graceful — `tests/conftest.py`
checks for the file first. The floor is `> 51`, so both counts pass.

No external services are needed: Redis is faked, Celery's `send_task` is
stubbed, and the database is a per-run temp SQLite file.

`tests/test_quality.py` covers: duplicate registration number, malformed id,
blurry photo, tiny face, two faces, no face, the same face under two IDs,
encryption round trip, and the bulk-import endpoint.
`tests/test_regression.py` guards the full encrypt → store → decrypt → index →
cosine → identity chain that previously broke.

---

## 7. API overview

Base URL `http://localhost:8000`. Everything except `/auth/token`, `/health`
and `/system/*` requires `Authorization: Bearer <token>` when
`AUTH_REQUIRED=true`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/token` | exchange admin credentials for a JWT |
| GET/POST | `/students` | list / enroll students |
| GET/PUT/DELETE | `/students/{id}` | read / edit / delete |
| POST | `/students/{id}/photos` | add reference photos |
| PUT/DELETE | `/students/{id}/face` | replace / clear face data |
| POST | `/students/bulk-import` | CSV import with per-row report |
| GET/POST/PUT/DELETE | `/cameras[/{id}]` | camera registry |
| POST | `/cameras/{id}/start` · `/stop` | control a pipeline slot |
| GET/POST/PUT/DELETE | `/periods[/{id}]` | attendance periods |
| GET | `/attendance/summary` · `/live` · `/logs` | queries |
| GET | `/attendance/period/{period}/{student}` | one student in one period |
| POST | `/attendance/sessions/{id}/close` | close an open session |
| GET/POST | `/alerts`, `/alerts/{id}/ack`, `/alerts/{id}/resolve` | alerts |
| WS | `/ws` | live events |
| GET | `/health`, `/system/gpu-status` | liveness / GPU diagnostics |

Interactive docs: `http://localhost:8000/docs`.

---

## 8. Configuration (`.env`)

Copy `.env.example` → `.env`. Compose overrides `DATABASE_URL` and `REDIS_URL`
with in-network hostnames.

| Key | Default | Meaning |
|---|---|---|
| `APP_NAME` | AI Video Attendance API | service name |
| `ENVIRONMENT` | development | `production` enables the hard checks below |
| `LOG_LEVEL` / `LOG_FORMAT` | INFO / json | json = one grep-able line per event |
| `CORS_ORIGINS` | `*` | allowed browser origins |
| `DATA_DIR` | `./data` | reference photos live under this |
| `LOCAL_TIMEZONE` | Asia/Kolkata | which calendar day a timestamp belongs to |
| **`EMBEDDING_ENCRYPTION_KEY`** | *(empty)* | **REQUIRED in production** — base64 url-safe 32 bytes. Startup fails if missing; there is no fallback. `python scripts/generate_key.py` |
| `DATABASE_URL` | `sqlite:///./data/attendance.db` | local default; compose uses postgres |
| `REDIS_URL` | `redis://localhost:6379/0` | broker + RPC bus |
| `AUTH_REQUIRED` | true | require JWT on API routes |
| `JWT_SECRET` | change-me… | must be changed in production (startup check) |
| `JWT_ALGORITHM` / `JWT_EXPIRE_MINUTES` | HS256 / 720 | token settings |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | admin / admin | bootstrap credentials — change them |
| `FACE_MODEL_NAME` | buffalo_l | insightface model pack |
| `INSIGHTFACE_ROOT` | `~/.insightface` | weight cache (gitignored) |
| `CUDA_DEVICE_INDEX` / `FORCE_CPU` | 0 / false | GPU selection / debugging |
| `DET_SIZES` / `DET_THRESH` | 640x640 / 0.5 | SCRFD input size(s) and threshold |
| `ENABLE_BATCHED_DETECTION` | true | batch frames per `session.run` |
| `MAX_FACES_PER_FRAME` / `MAX_EMBEDDING_BATCH` | 20 / 64 | caps |
| **`RECOGNITION_THRESHOLD`** | **0.40** | the one threshold, everywhere |
| **`RECOGNITION_MARGIN`** | **0.10** | best-vs-second-best required gap |
| **`DUPLICATE_IDENTITY_SIM`** | **0.90** | gallery pairs at/above this = same human |
| `DEDUP_WINDOW_SECONDS` | 120 | same student+camera within this → one event |
| `DEDUP_LOG_ALL` / `LOG_UNKNOWN_FACES` | false / false | logging verbosity |
| `MIN_REFERENCE_PHOTOS` / `MAX_REFERENCE_PHOTOS` | 3 / 5 | enrollment photo count |
| `INFERENCE_MODE` | redis | `redis` = shared service, `inprocess` = debug |
| `INFERENCE_QUEUE_KEY` / `INFERENCE_RESPONSE_TTL_S` / `INFERENCE_REQUEST_TIMEOUT_S` | infer:requests / 30 / 15 | RPC tuning |
| `MAX_BATCH_FRAMES` / `INFERENCE_BATCH_WAIT_MS` | 32 / 25 | cross-camera GPU batching |
| `DEFAULT_SAMPLING_RATE` | 5 | process every Nth frame (~6 FPS from 30) |
| `FRAME_QUEUE_SIZE` / `BATCH_SIZE` / `BATCH_FLUSH_MS` | 8 / 8 / 120 | queueing |
| `JPEG_QUALITY` | 85 | compression for the RPC payload |
| `RTSP_TRANSPORT` | tcp | tcp = reliable, udp = lower latency |
| `RTSP_OPEN_TIMEOUT_MS` / `RTSP_READ_TIMEOUT_MS` | 10000 / 10000 | connection timeouts |
| `RECONNECT_INITIAL_DELAY_S` / `RECONNECT_MAX_DELAY_S` | 2 / 30 | backoff |
| `CAMERA_SLOTS` | 8 | dedicated Celery queues |
| `HEARTBEAT_TTL_S` | 30 | expired → `/cameras` reports unhealthy |
| `SEED_VIDEO_PATH` | `./samples/videos/lecture.mp4` | mock source for `seed.py` |
| `POSTGRES_PASSWORD` / `API_PORT` / `API_WORKERS` / `ORT_GPU_VERSION` | attendance / 8000 / 1 / 1.26.0 | compose interpolation only |

Key rotation: `python scripts/rotate_encryption_key.py --new-key <new> [--dry-run]`
— it re-encrypts every stored row in place and reports rows it cannot decrypt.

---

## 9. Privacy rules

These are enforced by tooling, not by good intentions:

1. **Never commit** `.env`, databases, reference photos, face crops, embeddings,
   videos, or model weights. `.gitignore` covers all of them — verify with
   `git check-ignore -v .env data/x.db data/students/1/photo_0.jpg a.mp4`.
2. **Real registration numbers and names live only in gitignored local files**
   (`data/roster.local.csv`), never in tracked files, logs, reports or archives.
3. Output identifies people as **`P1…Pn`** plus **last-3 registration digits**
   only — in demo stdout, CSV and HTML alike.
4. **Placeholder photos never carry a real registration number.** The sample
   gallery uses `SAMPLE###` ids.
5. The handoff archive must pass a case-insensitive PII scan with **0 matches**
   and a media scan with **0 files**.

---

## 10. Known limitations

* **Real-CCTV accuracy is unmeasured.** The only end-to-end accuracy number
  available is from the *synthetic* clip (built from the enrollment photos
  themselves), so it measures plumbing, not recognition in the wild.
* **Placeholder gallery.** The committed photo folders are development images,
  not students. Enroll real people before drawing any conclusion.
* **Live RTSP was not exercised here.** The reader, reconnect/backoff and slot
  scheduling are unit-tested; no physical camera was attached.
* **The GPU Compose stack was not run in this environment** (no NVIDIA GPU), so
  `Dockerfile` + `docker-compose.yml` are provided and lint-clean but unverified
  end-to-end.
* **Scale is untested beyond ~100 faces.** The index is a linear cosine scan;
  both videos tested so far produced 0 matches against a gallery that did not
  contain them (correct rejections: max cosine 0.1862 vs threshold 0.40).
* **Accuracy degrades with gallery size.** The measured different-person
  maximum (0.1246) came from a small pair count; with 50+ enrollees the maximum
  will rise and the threshold should be recalibrated on your own data — run
  `python eval/enroll_and_calibrate.py` and `python eval/diag_threshold.py <video>`.

---

## 11. Demo without Docker or Redis

Everything above runs on one machine with no external services:

```bash
# 1. setup (venv, deps, .env, key, migration, weights, tests)
bash scripts/setup_dev.sh                 # or scripts\setup_dev.ps1

# 2. run the whole pipeline in-process on a video you provide
python scripts/run_demo.py --video /path/to/video.mp4

# 3. or serve the API against local SQLite (no Redis, no workers)
python -m uvicorn app.main:app --reload
```

`scripts/run_demo.py` sets `DATABASE_URL=sqlite:///demo_real.db`,
`FORCE_CPU=true` and `FACE_QUALITY_ENABLED=true` itself, creates the section /
camera / period, processes every frame in-process, and tears down cleanly.

When you *do* want the full stack:

```bash
cp .env.example .env && python scripts/generate_key.py   # put it in .env
docker compose up -d          # postgres, redis, api, worker, inference
```

---

## 12. Repository layout

```
app/
  api/          REST + WebSocket routes
  inference/    engine (SCRFD+ArcFace), shared RPC client & server
  pipeline/     frame reader, camera Celery task
  services/     enrollment, attendance, alerts, events
  matching.py   EmbeddingIndex: threshold / margin / duplicate-identity rules
  crypto.py     encrypt embeddings at rest (Fernet, PBKDF2)
  config.py     every setting + the measured basis for each threshold
scripts/        setup_dev.*, download_models, bulk_enroll, run_demo, ...
tests/          69 tests (pytest, no external services)
alembic/        schema migrations
docs/           ARCHITECTURE.md
eval/           ROADMAP.md, calibration & diagnostic tools
data/           gitignored: photos, roster, databases (see data/README.md)
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
