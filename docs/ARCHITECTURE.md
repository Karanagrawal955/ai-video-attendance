# Architecture

## 1. Module map

| Module | Responsibility |
|---|---|
| `app/config.py` | every setting, read from env/`.env`, plus the **measured basis** for each threshold |
| `app/main.py` | FastAPI app, router mounting, startup checks |
| `app/api/*` | thin HTTP layer: validation, auth, status codes |
| `app/schemas.py` | request/response models |
| `app/models.py` | SQLAlchemy models (Student, Camera, Period, AttendanceSession, RecognitionLog, Alert) |
| `app/db.py`, `app/redis_client.py` | engines/Session factory, Redis pool |
| `app/security.py`, `app/crypto.py` | JWT issue/verify; **Fernet encryption of embeddings at rest** |
| `app/quality.py` | per-photo gate: face count, size, pose, detection score |
| `app/matching.py` | `EmbeddingIndex` — threshold, margin, duplicate-identity skip |
| `app/services/enrollment.py` | validation, quality, embedding, duplicate checks, CSV bulk import |
| `app/services/attendance.py` | sessions, dedup, period summaries |
| `app/services/alerts.py`, `app/services/events.py` | alert lifecycle, event emission |
| `app/inference/engine.py` | SCRFD detection → quality gate → ArcFace embedding (the only model owner) |
| `app/inference/client.py` | Redis RPC client (batching, timeouts) |
| `app/inference/server.py` | the shared GPU inference service |
| `app/pipeline/frame_reader.py` | RTSP/file decode, sampling, bounded drop-oldest queue |
| `app/pipeline/camera_task.py` | Celery per-camera loop: read → compress → batch → RPC → match → record |
| `app/workers/celery_app.py`, `app/tasks.py` | worker topology (camera slots + control queues) |
| `alembic/` | schema migrations |

## 2. Frame → record data flow

```
1. DECODE      frame_reader: VideoCapture(rtsp|file)
               → keep every DEFAULT_SAMPLING_RATE-th frame (30 FPS → ~6 FPS)
               → bounded queue (FRAME_QUEUE_SIZE=8), drop-oldest on overflow
               → JPEG encode (JPEG_QUALITY=85)

2. BATCH       camera_task groups up to BATCH_SIZE frames
               (waits at most BATCH_FLUSH_MS), then publishes an RPC request
               on INFERENCE_QUEUE_KEY with a correlation id + TTL.

3. INFER       the inference service merges requests from ALL cameras into one
               engine call (MAX_BATCH_FRAMES=32, INFERENCE_BATCH_WAIT_MS=25):
                 SCRFD detect (DET_SIZES, DET_THRESH, MAX_FACES_PER_FRAME)
                 → quality gate  (area, sharpness, brightness)   [app/quality.py]
                 → ArcFace embed → 512-d vector, L2-normalised
               reply: per-frame list of {bbox, score, embedding}.

4. MATCH       EmbeddingIndex.refresh() reads ONLY the DB: decrypt each stored
               blob → normalise → matrix.  For each video face:
                 a) cosine against every stored vector (linear scan)
                 b) best score >= RECOGNITION_THRESHOLD  ?  else → unknown
                 c) best - second >= RECOGNITION_MARGIN ?  else → ambiguous/unknown
                    (runner-ups at/above DUPLICATE_IDENTITY_SIM with the winner
                     are skipped — they are the same human, not a rival)
                 d) winner → student_id

5. DEDUP       same student + same camera within DEDUP_WINDOW_SECONDS
               → one event (DEDUP_LOG_ALL controls whether the suppressed row
               is still written to RecognitionLog)

6. RECORD      process_recognition():
                 open/extend an AttendanceSession for (student, period, day)
                 → RecognitionLog row (score, camera, timestamp)
                 → close the session when the exit is seen
               student_attendance() / period_summary() read those back into
               present / late / left-early / absent.
```

Failure paths: inference RPC timeout → `InferenceUnavailable` → HTTP 503; a
missing/broken stream triggers reconnect with `RECONNECT_INITIAL_DELAY_S` →
`RECONNECT_MAX_DELAY_S` backoff; a stale pipeline heartbeat makes
`/cameras` report `unhealthy`.

## 3. Why 0.40 / 0.10 / 0.90

All three are derived from **measured cosine scores on this codebase**, not from
a paper or a default. Reproduce with:

```bash
python eval/enroll_and_calibrate.py        # gallery photo distributions
python eval/diag_threshold.py <video>      # per-face scores against the gallery
```

### Measured inputs

| Distribution | n | min | mean | p95 | max |
|---|---|---|---|---|---|
| **Different people**, photo↔photo | 75–384 | −0.1297 | 0.0055 | 0.1200 | **0.1246** |
| **Same person**, photo↔photo | 45–90 | **0.9096** | 0.9570 | — | 0.9998 |
| **Genuine match on video** (synthetic clip) | 83 | **0.5030** | — | — | 0.9844 |
| Unknown people in a real video vs gallery | 96 faces | −0.0132 | 0.0825 | — | **0.1862** |
| Duplicate gallery folders (same human) | 3 pairs | — | — | — | **0.9998** |

### `RECOGNITION_THRESHOLD = 0.40`

Must sit strictly between the worst different-person score and the worst
genuine score:

```
different-person max  0.1246   ←——— 0.40 must be here ———→   genuine video min  0.5030
```

0.40 is inside that gap, above the *real-video* unknown maximum of 0.1862 with
~2.2× headroom, and well below every genuine score observed. Midpoint of the
photo gap would be 0.52; 0.40 was chosen slightly low to favour **recall** on
hard video frames while still sitting far above anything an impostor produced.

### `RECOGNITION_MARGIN = 0.10`

When the top score clears the threshold, the runner-up must trail by ≥ 0.10.
Measured: genuine bests on video are ≥ 0.5030 while any *different-person*
runner-up is ≤ 0.1246, so a genuine best/runner-up gap is far larger than 0.10.
A probe sitting exactly between two identities scores ~0.707 against both — it
clears the threshold but fails the margin, which is the desired refusal (see
`test_margin_rule_rejects_ambiguous_probe`).

### `DUPLICATE_IDENTITY_SIM = 0.90`

Two gallery entries whose vectors are ≥ 0.90 apart are the **same human**
(measured duplicate folders: 0.9998; different people never exceeded 0.1246).
0.90 therefore sits ~7.2× above the impostor maximum and ~0.10 below the
duplicate measurement. Such pairs are excluded from the runner-up slot so that
enrolling the same person twice can no longer collapse the margin and reject
their own genuine matches (the exact failure mode that produced the historical
0/6 demo run).

## 4. Storage & privacy

* Reference photos live under `DATA_DIR/students/<folder>/`.
* Embeddings are stored as a **single encrypted string** (`app/crypto.py`:
  PBKDF2-HMAC-SHA256, 100k iterations, fixed salt, Fernet). Plaintext vectors
  never hit the database.
* The key comes from `EMBEDDING_ENCRYPTION_KEY` only. In `ENVIRONMENT=production`
  a missing key **fails startup** (`app/config.py` model validator) — there is no
  JWT-secret fallback, so leaking a signing key can never decrypt the gallery.
  Non-production falls back to a fixed, public development constant and warns.
* All identifiers in output are `P1…Pn` plus last-3 registration digits.

## 5. Deployment topology

```
docker compose
├── postgres      state
├── redis         broker + inference RPC bus
├── api           uvicorn, runs migrations on boot
├── inference     THE shared GPU service (one model instance, one reservation)
└── worker        Celery: CAMERA_SLOTS camera queues + control/jobs queues
```

`INFERENCE_MODE=redis` (default) makes api/worker call the shared service;
`inprocess` loads the model inside the worker and is for debugging only.

**Without Docker or Redis** every component runs in one process:
`scripts/run_demo.py` sets `DATABASE_URL=sqlite:///…`, `FORCE_CPU=true` and
drives the same code paths.
